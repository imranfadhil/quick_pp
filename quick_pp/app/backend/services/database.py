import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import (
    APIRouter,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
)
from sqlalchemy import bindparam, select, text

from quick_pp.app.backend.utils.auth import (
    get_current_user_optional,
    is_demo_mode,
)
from quick_pp.app.backend.utils.db import get_db
from quick_pp.app.backend.utils.utils import sanitize_filename
from quick_pp.database import models as db_models
from quick_pp.database import objects as db_objects


router = APIRouter(prefix="/database", tags=["Database"])


# Simple in-memory cache for merged data (stores up to 50 wells, ~5 minutes TTL)
# Maps key "{project_id}_{well_name}_{tolerance}" -> (timestamp, pd.DataFrame)
_merged_data_cache: Dict[str, tuple[float, pd.DataFrame]] = {}
_cache_ttl = 300  # 5 minutes
_cache_max_entries = 50
MAX_DEMO_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB cap per file in demo mode
_ephemeral_wells: Dict[int, Dict[str, pd.DataFrame]] = {}
_demo_project_overrides: Dict[int, Dict[str, Any]] = {}
_demo_deleted_projects: set[int] = set()
_demo_well_renames: Dict[int, Dict[str, str]] = {}
_demo_deleted_wells: Dict[int, set[str]] = {}


def resolve_db_well_name(project_id: int, well_name: str) -> str:
    """Resolve the underlying database well name if the well was renamed in an ephemeral demo session."""
    if well_name in _demo_deleted_wells.get(project_id, set()):
        raise ValueError(f"Well '{well_name}' was deleted from project {project_id}")
    renames = _demo_well_renames.get(project_id, {})
    for orig_name, curr_name in renames.items():
        if curr_name == well_name:
            return orig_name
    return well_name


def reset_demo_project_state(project_id: Optional[int] = None) -> None:
    """Reset in-memory demo project/well renames and deletions."""
    if project_id is not None:
        _demo_project_overrides.pop(project_id, None)
        _demo_deleted_projects.discard(project_id)
        _demo_well_renames.pop(project_id, None)
        _demo_deleted_wells.pop(project_id, None)
    else:
        _demo_project_overrides.clear()
        _demo_deleted_projects.clear()
        _demo_well_renames.clear()
        _demo_deleted_wells.clear()


def invalidate_well_cache(
    project_id: Optional[int] = None, well_name: Optional[str] = None
) -> int:
    """Invalidate cached merged well data for a specific well, project, or globally."""
    if project_id is not None and well_name is not None:
        prefix = f"{project_id}_{well_name}_"
        keys_to_remove = [k for k in _merged_data_cache if k.startswith(prefix)]
    elif project_id is not None:
        prefix = f"{project_id}_"
        keys_to_remove = [k for k in _merged_data_cache if k.startswith(prefix)]
    else:
        count = len(_merged_data_cache)
        _merged_data_cache.clear()
        return count

    for key in keys_to_remove:
        _merged_data_cache.pop(key, None)
    return len(keys_to_remove)


@router.get("/health", summary="Database health check")
async def health_check():
    """Simple health endpoint that checks DB connectivity by running `SELECT 1`."""
    try:
        with get_db() as session:
            session.execute(text("SELECT 1")).all()
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}") from e


@router.post("/projects", summary="Create or get project")
async def create_project(payload: dict, request: Request):
    """Create a new project (or return existing by name).

    Payload: {"name": "Project A", "description": "...", "created_by_user_id": 1}
    """
    name = payload.get("name") if isinstance(payload, dict) else None
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    description = payload.get("description") if isinstance(payload, dict) else ""
    current_user = get_current_user_optional(request)
    created_by = (
        (payload.get("created_by_user_id") if isinstance(payload, dict) else None) or (int(current_user["user_id"]) if current_user else None)
    )

    try:
        with get_db() as session:
            proj = db_objects.Project(
                db_session=session,
                name=name,
                description=description,
                created_by_user_id=created_by,
            )
            proj.save()
            if created_by is not None:
                if not session.get(db_models.ProjectMember, (proj.project_id, int(created_by))):
                    session.add(
                        db_models.ProjectMember(
                            project_id=proj.project_id,
                            user_id=int(created_by),
                            role="admin",
                        )
                    )
                    session.flush()
            return {"project_id": proj.project_id, "name": proj.name}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create project: {e}"
        ) from e


@router.get("/projects", summary="List projects")
async def list_projects(request: Request):
    try:
        with get_db() as session:
            stmt = select(db_models.Project)
            raw_rows = session.execute(stmt).scalars().all()
            rows = []
            for r in raw_rows:
                if r.project_id in _demo_deleted_projects:
                    continue
                if r.project_id in _demo_project_overrides:
                    session.expunge(r)
                    ovr = _demo_project_overrides[r.project_id]
                    if "name" in ovr:
                        r.name = ovr["name"]
                    if "description" in ovr:
                        r.description = ovr["description"]
                rows.append(r)
            current_user = get_current_user_optional(request)
            if current_user and current_user.get("role") != "admin":
                uid = int(current_user["user_id"])
                pms = {
                    pm.project_id: pm.role
                    for pm in session.execute(
                        select(db_models.ProjectMember).where(
                            db_models.ProjectMember.user_id == uid
                        )
                    )
                    .scalars()
                    .all()
                }
                return [
                    {
                        "project_id": r.project_id,
                        "name": r.name,
                        "description": r.description,
                        "user_role": "admin" if r.created_by == uid else pms[r.project_id],
                    }
                    for r in rows
                    if r.created_by == uid or r.project_id in pms
                ]
            return [
                {
                    "project_id": r.project_id,
                    "name": r.name,
                    "description": r.description,
                    "user_role": "admin",
                }
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list projects: {e}"
        ) from e


@router.patch("/projects/{project_id}", summary="Rename or update a project")
async def update_project(project_id: int, payload: dict, request: Request):
    """Rename a project and/or update its description.

    Payload example: {"name": "Volve Field Phase 2", "description": "Updated description"}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    new_name = (payload.get("name") or "").strip()
    new_desc = payload.get("description")
    if not new_name and new_desc is None:
        raise HTTPException(
            status_code=400, detail="Project name or description is required"
        )

    try:
        with get_db() as session:
            if project_id in _demo_deleted_projects:
                raise HTTPException(status_code=404, detail="Project not found")
            orm_proj = session.get(db_models.Project, project_id)
            if not orm_proj:
                raise HTTPException(status_code=404, detail="Project not found")

            effective_name = new_name or _demo_project_overrides.get(
                project_id, {}
            ).get("name", orm_proj.name)
            effective_desc = (
                str(new_desc)
                if new_desc is not None
                else _demo_project_overrides.get(project_id, {}).get(
                    "description", orm_proj.description or ""
                )
            )

            if new_name:
                all_projs = session.execute(select(db_models.Project)).scalars().all()
                for p in all_projs:
                    if p.project_id == project_id or p.project_id in _demo_deleted_projects:
                        continue
                    p_name = _demo_project_overrides.get(p.project_id, {}).get(
                        "name", p.name
                    )
                    if p_name.strip().lower() == new_name.lower():
                        raise HTTPException(
                            status_code=400,
                            detail=f"A project named '{new_name}' already exists",
                        )

            if is_demo_mode():
                _demo_project_overrides[project_id] = {
                    "name": effective_name,
                    "description": effective_desc,
                }
            else:
                orm_proj.name = effective_name
                if new_desc is not None:
                    orm_proj.description = effective_desc
                session.add(orm_proj)
                session.flush()

            return {
                "project_id": project_id,
                "name": effective_name,
                "description": effective_desc,
                "updated": True,
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update project: {e}"
        ) from e


@router.delete("/projects/{project_id}", summary="Delete a project and all its wells")
async def delete_project(project_id: int, request: Request):
    """Delete a project and all associated wells, curves, and cache entries."""
    try:
        with get_db() as session:
            orm_proj = session.get(db_models.Project, project_id)
            if not orm_proj or project_id in _demo_deleted_projects:
                raise HTTPException(status_code=404, detail="Project not found")

            proj_name = _demo_project_overrides.get(project_id, {}).get(
                "name", orm_proj.name
            )
            if is_demo_mode():
                _demo_deleted_projects.add(project_id)
                _demo_project_overrides.pop(project_id, None)
                _demo_well_renames.pop(project_id, None)
                _demo_deleted_wells.pop(project_id, None)
            else:
                session.delete(orm_proj)
                session.flush()

            _ephemeral_wells.pop(project_id, None)
            invalidate_well_cache(project_id=project_id)

            return {
                "project_id": project_id,
                "name": proj_name,
                "deleted": True,
                "message": f"Project '{proj_name}' deleted",
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete project: {e}"
        ) from e


@router.patch(
    "/projects/{project_id}/wells/{well_name}",
    summary="Rename a well within a project",
)
async def rename_well(
    project_id: int, well_name: str, payload: dict, request: Request
):
    """Rename a well in the project and update cached data references.

    Payload example: {"name": "15_9-F-11B", "uwi": "15_9-F-11B"}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    new_name = (payload.get("name") or payload.get("new_name") or "").strip()
    if not new_name:
        raise HTTPException(status_code=400, detail="New well name is required")
    if new_name == well_name:
        return {
            "project_id": project_id,
            "old_name": well_name,
            "well_name": new_name,
            "name": new_name,
            "updated": False,
        }

    try:
        with get_db() as session:
            if project_id in _demo_deleted_projects:
                raise HTTPException(status_code=404, detail="Project not found")
            proj = db_objects.Project.load(session, project_id=project_id)
            deleted_set = _demo_deleted_wells.get(project_id, set())
            renames_map = _demo_well_renames.setdefault(project_id, {})

            # Gather current active well names to prevent collisions
            active_names = set()
            target_orm_well = None
            orig_db_name = None
            for w in proj._orm_project.wells:
                curr_w_name = renames_map.get(w.name, w.name)
                if w.name in deleted_set or curr_w_name in deleted_set:
                    continue
                active_names.add(curr_w_name)
                if curr_w_name == well_name or w.name == well_name:
                    target_orm_well = w
                    orig_db_name = w.name

            eph_proj = _ephemeral_wells.get(project_id, {})
            for eph_n in eph_proj:
                if eph_n not in deleted_set:
                    active_names.add(eph_n)

            if well_name not in active_names and target_orm_well is None and well_name not in eph_proj:
                raise HTTPException(
                    status_code=404,
                    detail=f"Well '{well_name}' not found in project {project_id}",
                )

            if any(
                n.lower() == new_name.lower() and n != well_name
                for n in active_names
            ):
                raise HTTPException(
                    status_code=400,
                    detail=f"A well named '{new_name}' already exists in this project",
                )

            # Update in-memory merged cache keys and WELL_NAME column
            old_prefix = f"{project_id}_{well_name}_"
            for key, (ts, cdf) in list(_merged_data_cache.items()):
                if key.startswith(old_prefix):
                    suffix = key[len(old_prefix) :]
                    updated_df = cdf.copy()
                    if "WELL_NAME" in updated_df.columns:
                        updated_df["WELL_NAME"] = new_name
                    _merged_data_cache.pop(key, None)
                    _merged_data_cache[f"{project_id}_{new_name}_{suffix}"] = (
                        time.time(),
                        updated_df,
                    )

            # Update ephemeral wells if present
            if well_name in eph_proj:
                eph_df = eph_proj.pop(well_name).copy()
                if "WELL_NAME" in eph_df.columns:
                    eph_df["WELL_NAME"] = new_name
                eph_proj[new_name] = eph_df

            if is_demo_mode():
                if orig_db_name:
                    renames_map[orig_db_name] = new_name
                deleted_set.discard(new_name)
            else:
                if target_orm_well is not None:
                    new_uwi = payload.get("uwi")
                    if new_uwi is not None:
                        target_orm_well.uwi = str(new_uwi).strip()
                    elif not target_orm_well.uwi or target_orm_well.uwi == target_orm_well.name:
                        target_orm_well.uwi = new_name
                    target_orm_well.name = new_name
                    session.add(target_orm_well)
                    session.flush()

            return {
                "project_id": project_id,
                "old_name": well_name,
                "well_name": new_name,
                "name": new_name,
                "updated": True,
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to rename well: {e}"
        ) from e


@router.delete(
    "/projects/{project_id}/wells/{well_name}",
    summary="Delete a well from a project",
)
async def delete_well(project_id: int, well_name: str, request: Request):
    """Delete a well and all its curve/ancillary data from a project."""
    try:
        with get_db() as session:
            if project_id in _demo_deleted_projects:
                raise HTTPException(status_code=404, detail="Project not found")
            proj = db_objects.Project.load(session, project_id=project_id)
            deleted_set = _demo_deleted_wells.setdefault(project_id, set())
            renames_map = _demo_well_renames.get(project_id, {})

            target_orm_well = None
            orig_db_name = None
            for w in proj._orm_project.wells:
                curr_w_name = renames_map.get(w.name, w.name)
                if curr_w_name == well_name or w.name == well_name:
                    target_orm_well = w
                    orig_db_name = w.name
                    break

            eph_proj = _ephemeral_wells.get(project_id, {})
            in_eph = well_name in eph_proj

            if target_orm_well is None and not in_eph:
                raise HTTPException(
                    status_code=404,
                    detail=f"Well '{well_name}' not found in project {project_id}",
                )

            if in_eph:
                eph_proj.pop(well_name, None)

            if is_demo_mode():
                deleted_set.add(well_name)
                if orig_db_name:
                    deleted_set.add(orig_db_name)
            else:
                if target_orm_well is not None:
                    session.delete(target_orm_well)
                    session.flush()

            invalidate_well_cache(project_id=project_id, well_name=well_name)
            if orig_db_name and orig_db_name != well_name:
                invalidate_well_cache(project_id=project_id, well_name=orig_db_name)

            return {
                "project_id": project_id,
                "well_name": well_name,
                "deleted": True,
                "message": f"Well '{well_name}' deleted",
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete well: {e}"
        ) from e


@router.get("/projects/{project_id}/wells", summary="List wells in project")
async def list_wells(project_id: int):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            wells = []
            for well in proj._orm_project.wells:
                wells.append(
                    {
                        "id": str(well.well_id),
                        "name": well.name,
                        "uwi": well.uwi,
                    }
                )
            existing_names = {w["name"] for w in wells}
            for eph_name in _ephemeral_wells.get(project_id, {}):
                if eph_name not in existing_names:
                    wells.append({"id": f"eph_{eph_name}", "name": eph_name, "uwi": eph_name})
            return {"project_id": proj.project_id, "wells": wells}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list wells: {e}") from e


@router.get(
    "/projects/{project_id}/summary",
    summary="Get lightweight project and per-well summary statistics",

)
def get_project_summary(project_id: int) -> Dict[str, Any]:
    """Return aggregated metadata and statistics for all wells in a project without loading full curve arrays."""
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if project_id in _demo_deleted_projects:
                raise ValueError(f"Project {project_id} not found")
            if project_id in _demo_project_overrides:
                session.expunge(proj._orm_project)
                ovr = _demo_project_overrides[project_id]
                if "name" in ovr:
                    proj._orm_project.name = ovr["name"]
                if "description" in ovr:
                    proj._orm_project.description = ovr["description"]
            deleted_w_set = _demo_deleted_wells.get(project_id, set())
            renamed_w_map = _demo_well_renames.get(project_id, {})
            orm_wells = []
            for w in proj._orm_project.wells:
                curr_name = renamed_w_map.get(w.name, w.name)
                if w.name in deleted_w_set or curr_name in deleted_w_set:
                    continue
                if curr_name != w.name:
                    session.expunge(w)
                    w.name = curr_name
                orm_wells.append(w)
            if not orm_wells:
                return {
                    "project_id": proj.project_id,
                    "name": proj.name,
                    "description": proj.description,
                    "well_count": 0,
                    "total_depth_sum": 0.0,
                    "all_zones": [],
                    "all_curves": [],
                    "wells": [],
                }

            well_ids = [w.well_id for w in orm_wells]

            # 1. Curves per well
            curves_stmt = text(
                "SELECT well_id, mnemonic FROM curves WHERE well_id IN :well_ids ORDER BY mnemonic"
            ).bindparams(bindparam("well_ids", expanding=True))
            curve_rows = session.execute(curves_stmt, {"well_ids": well_ids}).fetchall()
            curves_by_well: Dict[int, List[str]] = {wid: [] for wid in well_ids}
            all_curves_set = set()
            for wid, mnemonic in curve_rows:
                m_up = str(mnemonic).upper()
                curves_by_well.setdefault(wid, []).append(m_up)
                all_curves_set.add(m_up)

            # 2. Depth statistics per well
            depth_stmt = text("""
                SELECT
                    c.well_id,
                    MIN(cd.depth) AS min_depth,
                    MAX(cd.depth) AS max_depth,
                    COUNT(DISTINCT cd.depth) AS row_count
                FROM curve_data cd
                JOIN curves c ON cd.curve_id = c.curve_id
                WHERE c.well_id IN :well_ids
                GROUP BY c.well_id
            """).bindparams(bindparam("well_ids", expanding=True))
            depth_rows = session.execute(depth_stmt, {"well_ids": well_ids}).fetchall()
            depth_by_well = {
                r[0]: {
                    "min_depth": float(r[1]) if r[1] is not None else None,
                    "max_depth": float(r[2]) if r[2] is not None else None,
                    "row_count": int(r[3] or 0),
                }
                for r in depth_rows
            }

            # 3. Formation tops (zones) per well
            tops_stmt = text(
                "SELECT well_id, name, depth FROM formation_tops WHERE well_id IN :well_ids ORDER BY depth"
            ).bindparams(bindparam("well_ids", expanding=True))
            top_rows = session.execute(tops_stmt, {"well_ids": well_ids}).fetchall()
            zones_by_well: Dict[int, List[str]] = {wid: [] for wid in well_ids}
            tops_count_by_well: Dict[int, int] = {wid: 0 for wid in well_ids}
            all_zones_set = set()
            for wid, top_name, _ in top_rows:
                tops_count_by_well[wid] = tops_count_by_well.get(wid, 0) + 1
                if top_name and top_name not in zones_by_well.setdefault(wid, []):
                    zones_by_well[wid].append(top_name)
                    all_zones_set.add(top_name)

            # Also check if ZONES curve exists in curve_data (for LAS files with ZONES curve)
            zones_curve_stmt = text("""
                SELECT DISTINCT c.well_id, COALESCE(cd.value_text, CAST(cd.value_numeric AS TEXT)) AS zone_val
                FROM curve_data cd
                JOIN curves c ON cd.curve_id = c.curve_id
                WHERE c.well_id IN :well_ids AND UPPER(c.mnemonic) = 'ZONES'
            """).bindparams(bindparam("well_ids", expanding=True))
            for wid, zval in session.execute(
                zones_curve_stmt, {"well_ids": well_ids}
            ).fetchall():
                if zval is not None and str(zval).strip():
                    z_str = str(zval).strip()
                    if z_str not in zones_by_well.setdefault(wid, []):
                        zones_by_well[wid].append(z_str)
                        all_zones_set.add(z_str)

            # 4. Core samples & core curves per well
            core_stmt = text("""
                SELECT well_id, COUNT(*) AS sample_count
                FROM core_samples
                WHERE well_id IN :well_ids
                GROUP BY well_id
            """).bindparams(bindparam("well_ids", expanding=True))
            core_counts = {
                r[0]: int(r[1] or 0)
                for r in session.execute(core_stmt, {"well_ids": well_ids}).fetchall()
            }

            core_curve_stmt = text("""
                SELECT c.well_id, COUNT(DISTINCT cd.depth) AS core_pts
                FROM curve_data cd
                JOIN curves c ON cd.curve_id = c.curve_id
                WHERE c.well_id IN :well_ids
                  AND UPPER(c.mnemonic) IN ('CPORE', 'CPERM', 'KPORE', 'KPERM')
                  AND cd.value_numeric IS NOT NULL
                GROUP BY c.well_id
            """).bindparams(bindparam("well_ids", expanding=True))
            core_curve_counts = {
                r[0]: int(r[1] or 0)
                for r in session.execute(
                    core_curve_stmt, {"well_ids": well_ids}
                ).fetchall()
            }

            # 5. Fluid contacts, pressure tests, surveys counts
            contacts_stmt = text(
                "SELECT well_id, COUNT(*) FROM fluid_contacts WHERE well_id IN :well_ids GROUP BY well_id"
            ).bindparams(bindparam("well_ids", expanding=True))
            contact_counts = {
                r[0]: int(r[1] or 0)
                for r in session.execute(
                    contacts_stmt, {"well_ids": well_ids}
                ).fetchall()
            }

            pressure_stmt = text(
                "SELECT well_id, COUNT(*) FROM pressure_tests WHERE well_id IN :well_ids GROUP BY well_id"
            ).bindparams(bindparam("well_ids", expanding=True))
            pressure_counts = {
                r[0]: int(r[1] or 0)
                for r in session.execute(
                    pressure_stmt, {"well_ids": well_ids}
                ).fetchall()
            }

            survey_stmt = text(
                "SELECT well_id, COUNT(*) FROM well_surveys WHERE well_id IN :well_ids GROUP BY well_id"
            ).bindparams(bindparam("well_ids", expanding=True))
            survey_counts = {
                r[0]: int(r[1] or 0)
                for r in session.execute(survey_stmt, {"well_ids": well_ids}).fetchall()
            }

            wells_summary = []
            total_depth_sum = 0.0

            for w in orm_wells:
                wid = w.well_id
                d_info = depth_by_well.get(
                    wid, {"min_depth": None, "max_depth": None, "row_count": 0}
                )
                min_d = d_info["min_depth"]
                max_d = d_info["max_depth"]
                row_count = d_info["row_count"]
                td = (
                    round(max_d - min_d, 2)
                    if (min_d is not None and max_d is not None and max_d >= min_d)
                    else 0.0
                )
                total_depth_sum += td

                c_samples = core_counts.get(wid, 0)
                c_curve_pts = core_curve_counts.get(wid, 0)
                effective_core_pts = max(c_samples, c_curve_pts)
                core_pct = (
                    round(min(100.0, (effective_core_pts / row_count) * 100.0), 2)
                    if row_count > 0
                    else 0.0
                )

                wells_summary.append(
                    {
                        "id": str(wid),
                        "well_id": wid,
                        "name": w.name,
                        "uwi": w.uwi,
                        "depth_uom": w.depth_uom or "m",
                        "min_depth": min_d,
                        "max_depth": max_d,
                        "total_depth": td,
                        "row_count": row_count,
                        "depth_count": row_count,
                        "available_curves": curves_by_well.get(wid, []),
                        "curves": curves_by_well.get(wid, []),
                        "available_curves": curves_by_well.get(wid, []),
                        "zones": zones_by_well.get(wid, []),
                        "core_sample_count": effective_core_pts,
                        "core_availability_pct": core_pct,
                        "formation_top_count": tops_count_by_well.get(wid, 0),
                        "fluid_contact_count": contact_counts.get(wid, 0),
                        "pressure_test_count": pressure_counts.get(wid, 0),
                        "survey_count": survey_counts.get(wid, 0),
                    }
                )

            return {
                "project_id": proj.project_id,
                "name": proj.name,
                "description": proj.description,
                "well_count": len(wells_summary),
                "min_depth": min(
                    (w["min_depth"] for w in wells_summary if w.get("min_depth") is not None),
                    default=None,
                ),
                "max_depth": max(
                    (w["max_depth"] for w in wells_summary if w.get("max_depth") is not None),
                    default=None,
                ),
                "total_depth_sum": round(total_depth_sum, 2),
                "all_zones": sorted(all_zones_set),
                "all_curves": sorted(all_curves_set),
                "wells": wells_summary,
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get project summary: {e}"
        ) from e


@router.post("/projects/{project_id}/wells", summary="Create a new well in a project")
async def create_well(project_id: int, payload: dict):
    """Create a new well record in the project.

    Payload example: {"name": "Well-1", "uwi": "UWI-123", "depth_uom": "m"}
    """
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Well name is required")

    uwi = payload.get("uwi")
    depth_uom = payload.get("depth_uom")

    try:
        with get_db() as session:
            # check for existing well in this project by name
            existing = session.scalar(
                select(db_models.Well).filter_by(project_id=project_id, name=name)
            )
            if existing:
                return {"message": "Well already exists", "well_name": existing.name}

            well_obj = db_objects.Well(
                session,
                project_id=project_id,
                name=name,
                uwi=uwi or "",
                header_data={},
                depth_uom=depth_uom,
            )
            # persist
            well_obj.save()
            return {"well_name": well_obj.name, "well_id": well_obj.well_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create well: {e}"
        ) from e


@router.post("/projects/{project_id}/read_las", summary="Upload LAS files into project")
@router.post("/projects/{project_id}/upload", include_in_schema=False)
def project_read_las(
    project_id: int,
    files: List[UploadFile] = File(...),
    depth_uom: Optional[str] = Form("m"),
):
    """Upload LAS files and add them into the project in the database synchronously."""
    upload_dir = Path("uploads/las_db/")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        for f in files:
            # Prefix uploaded filename with a UUID to avoid collisions when
            # multiple uploads use the same original filename concurrently.
            safe_name = sanitize_filename(f.filename)
            if is_demo_mode():
                f.file.seek(0, 2)
                file_size = f.file.tell()
                f.file.seek(0)
                if file_size > MAX_DEMO_UPLOAD_BYTES:
                    f.file.close()
                    raise HTTPException(
                        status_code=413,
                        detail=f"File '{safe_name}' ({file_size / (1024 * 1024):.2f} MB) exceeds the 5.0 MB demo upload limit.",
                    )
            unique_name = f"{uuid4().hex}_{safe_name}"
            dest = upload_dir / unique_name
            try:
                with dest.open("wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
            finally:
                f.file.close()
            saved_paths.append(str(dest))

        invalidate_well_cache(project_id=project_id)

        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if is_demo_mode():
                import lasio

                proj_eph = _ephemeral_wells.setdefault(project_id, {})
                for p in saved_paths:
                    las = lasio.read(p)
                    ldf = las.df().reset_index()
                    ldf.columns = [str(c).upper() for c in ldf.columns]
                    if "DEPT" in ldf.columns and "DEPTH" not in ldf.columns:
                        ldf = ldf.rename(columns={"DEPT": "DEPTH"})
                    w_name = str(getattr(las.well, "WELL", None).value if hasattr(las.well, "WELL") else Path(p).stem.split("_", 1)[-1]).strip() or Path(p).stem
                    ldf["WELL_NAME"] = w_name
                    proj_eph[w_name] = ldf
                    _merged_data_cache[f"{project_id}_{w_name}_0.16"] = (time.time(), ldf.copy())
            else:
                proj.read_las(saved_paths, depth_uom=depth_uom)

        invalidate_well_cache(project_id=project_id)

        return {
            "message": "Files uploaded and processed successfully",
            "files": [Path(p).name for p in saved_paths],
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=(413 if getattr(e, "status_code", 500) == 413 else 500), detail=getattr(e, "detail", f"Failed to read LAS into project: {e}")
        ) from e


@router.put(
    "/projects/{project_id}/wells/{well_name}/data",
    summary="Save/overwrite well data from frontend",
)
async def save_well_data(project_id: int, well_name: str, payload: dict):
    """Save edited well data sent from the frontend.

    Expected payload:
    - Records format: { "data": [ {"DEPTH": 1234.5, "WELL_NAME": "Well-1", "DPHI": 12.3, ...}, ... ] }
    - Columnar format: { "data": {"DEPTH": [1234.5, ...], "DPHI": [12.3, ...]} }

    If `WELL_NAME` column is missing, it will be set to the path parameter `well_name`.
    The endpoint will convert the payload to a pandas DataFrame and call
    `Project.update_data` which performs upserts on curve data.
    """
    if not isinstance(payload, dict) or "data" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with a 'data' key"
        )

    data = payload.get("data")
    if not isinstance(data, (list, dict)):
        raise HTTPException(
            status_code=400,
            detail="'data' must be a list of row objects or a dict of column arrays",
        )

    try:
        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Accept case-insensitive depth and well name column names
        # Find depth column (DEPTH, depth, Depth, etc.)
        depth_col = next((c for c in df.columns if str(c).lower() == "depth"), None)
        if not depth_col:
            raise HTTPException(
                status_code=400,
                detail="Each row must include a 'DEPTH' (case-insensitive) field",
            )

        # Normalize to uppercase 'DEPTH' for internal processing
        if depth_col != "DEPTH":
            df = df.rename(columns={depth_col: "DEPTH"})

        # Ensure WELL_NAME present for Project.update_data; accept case-insensitive and normalize
        well_col = next((c for c in df.columns if str(c).lower() == "well_name"), None)
        if well_col and well_col != "WELL_NAME":
            df = df.rename(columns={well_col: "WELL_NAME"})
        if "WELL_NAME" not in df.columns:
            df["WELL_NAME"] = well_name

        # Cast DEPTH to numeric
        df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
        if df["DEPTH"].isna().any():
            # return which rows failed (up to first 5) to help debugging
            bad_idx = df[df["DEPTH"].isna()].index.tolist()[:5]
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Some DEPTH values could not be parsed as numbers. Failed row indices: {bad_idx}"
                ),
            )

        if is_demo_mode():
            for wn, wdf in df.groupby("WELL_NAME"):
                _ephemeral_wells.setdefault(project_id, {})[str(wn)] = wdf.copy()
                _merged_data_cache[f"{project_id}_{wn}_0.16"] = (time.time(), wdf.copy())
            return {
                "message": "Well data saved to ephemeral demo session",
                "rows": len(df),
                "wells_updated": int(df["WELL_NAME"].nunique()),
                "ephemeral": True,
            }

        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            # Use Project.update_data which groups by WELL_NAME and upserts per well
            proj.update_data(df, group_by="WELL_NAME")
            # Persist changes
            proj.save()

        # Invalidate cache for updated wells
        wells_updated = df["WELL_NAME"].unique()
        for wn in wells_updated:
            invalidate_well_cache(project_id=project_id, well_name=str(wn))

        return {
            "message": "Well data saved",
            "rows": len(df),
            "wells_updated": len(wells_updated),
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save well data: {e}"
        ) from e


@router.get(
    "/projects/{project_id}/wells/{well_name}/data",
    summary="Get top-to-bottom well data",

)
def get_well_data(
    project_id: int, well_name: str, include_ancillary: bool = False
) -> Any:
    """Return the full well data (curve logs) as a JSON array of rows.

    Each row is a dict mapping column names to values.
    Uses optimized data retrieval and faster JSON serialization.
    """
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Use optimized method if available
            try:
                df = proj.get_well_data_optimized(resolve_db_well_name(project_id, well_name))
            except AttributeError:
                df = proj.get_well_data(resolve_db_well_name(project_id, well_name))

            if df.empty:
                return []

            df = df.reset_index(drop=True)
            # Fast JSON serialization via Pydantic return annotation
            records = dataframe_to_records(df)

            if include_ancillary:
                well = proj.get_well(resolve_db_well_name(project_id, well_name))
                tops_df = well.get_formation_tops()
                contacts_df = well.get_fluid_contacts()
                pressure_df = well.get_pressure_tests()
                core_raw = well.get_core_data()
                tops = tops_df.to_dict(orient="records") if not tops_df.empty else []
                contacts = (
                    contacts_df.to_dict(orient="records")
                    if not contacts_df.empty
                    else []
                )
                pressure = (
                    pressure_df.to_dict(orient="records")
                    if not pressure_df.empty
                    else []
                )
                core_samples = []
                for name, s in core_raw.items():
                    measurements = []
                    if hasattr(s.get("measurements"), "to_dict"):
                        measurements = s.get("measurements").to_dict(orient="records")
                    core_samples.append(
                        {
                            "sample_name": name,
                            "depth": s.get("depth"),
                            "measurements": measurements,
                        }
                    )

                return {
                    "data": records,
                    "formation_tops": tops,
                    "fluid_contacts": contacts,
                    "pressure_tests": pressure,
                    "core_samples": core_samples,
                }

            return records
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get well data: {e}"
        ) from e


def _sanitize_for_json(obj):
    """Recursively sanitize an object to be JSON serializable.

    Converts NaN, Inf, -Inf to None and handles nested structures.
    """
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
        return obj
    elif isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(item) for item in obj]
    elif isinstance(obj, (np.floating, np.integer)):
        # Handle numpy types
        if isinstance(obj, np.floating) and (np.isnan(obj) or np.isinf(obj)):
            return None
        return obj.item() if hasattr(obj, "item") else obj
    return obj


def _series_to_py_list(series: pd.Series) -> list:
    """Convert a pandas Series to a native Python list with NaN/Inf mapped to None."""
    if pd.api.types.is_numeric_dtype(series):
        s = series.replace([np.inf, -np.inf], np.nan)
        return s.astype(object).where(pd.notna(s), None).tolist()
    elif pd.api.types.is_datetime64_any_dtype(series):
        s = series.dt.strftime("%Y-%m-%dT%H:%M:%S")
        return s.astype(object).where(pd.notna(series), None).tolist()
    else:
        return series.astype(object).where(pd.notna(series), None).tolist()


def dataframe_to_columnar(df: pd.DataFrame) -> Dict[str, Any]:
    """Convert a DataFrame into a compact columnar JSON structure."""
    if df is None or df.empty:
        return {"columns": [], "data": {}, "row_count": 0}
    cols = [str(c) for c in df.columns]
    data = {col: _series_to_py_list(df[col]) for col in cols}
    return {"columns": cols, "data": data, "row_count": len(df)}


def dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Convert a DataFrame into a list of row dicts with NaN/Inf sanitized to None."""
    if df is None or df.empty:
        return []
    cols = [str(c) for c in df.columns]
    col_lists = [_series_to_py_list(df[col]) for col in cols]
    return [dict(zip(cols, row)) for row in zip(*col_lists)]


def filter_merged_dataframe(
    df: pd.DataFrame,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Filter a merged well DataFrame by depth interval and/or zone names."""
    if df is None or df.empty:
        return pd.DataFrame()

    filtered = df
    if "DEPTH" in filtered.columns:
        if min_depth is not None:
            filtered = filtered[filtered["DEPTH"] >= float(min_depth)]
        if max_depth is not None:
            filtered = filtered[filtered["DEPTH"] <= float(max_depth)]

    if zones:
        valid_zones = [str(z).strip() for z in zones if str(z).strip()]
        if valid_zones and "ZONES" in filtered.columns:
            filtered = filtered[filtered["ZONES"].astype(str).isin(valid_zones)]

    return filtered.reset_index(drop=True)


def _get_merged_dataframe_sync(
    project_id: int,
    well_name: str,
    tolerance: Optional[float] = 0.16,
) -> pd.DataFrame:
    """Retrieve or compute the merged well DataFrame with in-memory TTL caching."""
    tol = float(tolerance) if tolerance is not None else 0.16
    cache_key = f"{project_id}_{well_name}_{tol}"
    now = time.time()

    cached_entry = _merged_data_cache.get(cache_key)
    if cached_entry is not None:
        cached_ts, cached_df = cached_entry
        if is_demo_mode() or (now - cached_ts < _cache_ttl):
            return cached_df.copy()
        else:
            _merged_data_cache.pop(cache_key, None)

    eph_df = _ephemeral_wells.get(project_id, {}).get(well_name)
    if eph_df is not None:
        _merged_data_cache[cache_key] = (now, eph_df.copy())
        return eph_df.copy()

    def _normalize_df_depth(dframe):
        """Optimize depth normalization with minimal copying."""
        if dframe is None or dframe.empty:
            return None
        if "depth" in dframe.columns:
            d = dframe
        else:
            d = dframe.copy()
            d.columns = [c.lower() for c in d.columns]
            if "depth" not in d.columns:
                candidate = next(
                    (c for c in d.columns if "depth" in c or c in ("md", "tvd")),
                    None,
                )
                if candidate:
                    d = d.rename(columns={candidate: "depth"})

        if "depth" not in d.columns:
            return None

        d["depth"] = pd.to_numeric(d["depth"], errors="coerce")
        d = d.dropna(subset=["depth"]) if not d.empty else d
        return d

    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            try:
                df = proj.get_well_data_optimized(resolve_db_well_name(project_id, well_name))
            except AttributeError:
                df = proj.get_well_data(resolve_db_well_name(project_id, well_name))
            if df.empty:
                return pd.DataFrame()

            # Normalize main dataframe depth column
            df = df.reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]
            if "depth" not in df.columns:
                candidate = next(
                    (c for c in df.columns if "depth" in c or c in ("md", "tvd")),
                    None,
                )
                if candidate:
                    df = df.rename(columns={candidate: "depth"})

            if "depth" not in df.columns:
                raise ValueError("No depth column found in well data")

            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
            df = df.dropna(subset=["depth"])
            df = df.sort_values("depth").reset_index(drop=True)

            well = proj.get_well(resolve_db_well_name(project_id, well_name))

            tops_df = _normalize_df_depth(well.get_formation_tops())
            contacts_df = _normalize_df_depth(well.get_fluid_contacts())
            pressure_df = _normalize_df_depth(well.get_pressure_tests())
            core_raw = well.get_core_data() or {}

            def _prefixed(dframe, prefix):
                if dframe is None or dframe.empty:
                    return None
                cols_to_rename = [c for c in dframe.columns if c != "depth"]
                rename_map = {c: f"{prefix}{c}" for c in cols_to_rename}
                return (
                    dframe.rename(columns=rename_map)
                    .sort_values("depth")
                    .reset_index(drop=True)
                )

            tops_p = _prefixed(tops_df, "top_")
            contacts_p = _prefixed(contacts_df, "contact_")
            pressure_p = _prefixed(pressure_df, "pressure_")

            merged = df

            if tops_p is not None and not tops_p.empty:
                merged = pd.merge_asof(
                    merged,
                    tops_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tol,
                    suffixes=("", "_top"),
                )
                if "top_name" in merged.columns:
                    merged["zones"] = merged["top_name"].ffill()

            if contacts_p is not None and not contacts_p.empty:
                merged = pd.merge_asof(
                    merged,
                    contacts_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tol,
                    suffixes=("", "_contact"),
                )

            if pressure_p is not None and not pressure_p.empty:
                merged = pd.merge_asof(
                    merged,
                    pressure_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tol,
                    suffixes=("", "_pressure"),
                )

            core_rows_by_depth: Dict[float, Dict[str, Any]] = {}
            if isinstance(core_raw, dict):
                for name, sd in core_raw.items():
                    try:
                        dval = sd.get("depth")
                        if dval is None:
                            continue
                        measurements = sd.get("measurements")
                        if measurements is None or measurements.empty:
                            continue

                        d_float = float(dval)
                        row_entry = core_rows_by_depth.setdefault(
                            d_float,
                            {"depth": d_float, "core_sample_name": name},
                        )
                        for prop in ("cpore", "cperm"):
                            mask = (measurements["property"] == prop) & measurements[
                                "value"
                            ].notna()
                            filtered = measurements[mask]
                            if not filtered.empty:
                                row_entry[prop] = float(filtered.iloc[0]["value"])
                    except Exception:
                        continue

            if core_rows_by_depth:
                core_df = (
                    pd.DataFrame(list(core_rows_by_depth.values()))
                    .sort_values("depth")
                    .reset_index(drop=True)
                )
                merged = pd.merge_asof(
                    merged,
                    core_df,
                    on="depth",
                    direction="nearest",
                    tolerance=tol,
                    suffixes=("", "_core"),
                )

            merged = merged.replace([np.inf, -np.inf], np.nan)
            merged.columns = [c.upper() for c in merged.columns]

            # Evict oldest entry if cache is full
            if len(_merged_data_cache) >= _cache_max_entries:
                oldest_key = min(
                    _merged_data_cache.keys(), key=lambda k: _merged_data_cache[k][0]
                )
                _merged_data_cache.pop(oldest_key, None)

            _merged_data_cache[cache_key] = (now, merged.copy())
            return merged
    except ValueError as e:
        raise ValueError(str(e)) from e
    except Exception as e:
        raise Exception(f"Failed to produce merged data: {e}") from e


def _compute_merged_data_sync(
    project_id: int,
    well_name: str,
    tolerance: Optional[float] = 0.16,
) -> list:
    """Compute merged well data synchronously (helper function)."""
    merged = _get_merged_dataframe_sync(project_id, well_name, tolerance=tolerance)
    return dataframe_to_records(merged)


@router.get(
    "/projects/{project_id}/wells/{well_name}/merged",
    summary="Get merged well data directly in columnar or records format",
)
def get_merged_well_data(
    project_id: int,
    well_name: str,
    tolerance: float = Query(0.16, description="Merge depth tolerance"),
    format: str = Query(
        "columnar", description="Output format: 'columnar' or 'records'"
    ),
    min_depth: Optional[float] = Query(None, description="Minimum depth filter"),
    max_depth: Optional[float] = Query(None, description="Maximum depth filter"),
    zones: Optional[str] = Query(
        None, description="Comma-separated zone names to filter"
    ),
) -> Any:
    """Return merged well curve and ancillary data directly with in-memory caching and optional depth/zone filtering."""
    try:
        merged = _get_merged_dataframe_sync(project_id, well_name, tolerance=tolerance)
        zone_list = [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        filtered = filter_merged_dataframe(
            merged, min_depth=min_depth, max_depth=max_depth, zones=zone_list
        )
        if format.lower() == "records":
            return dataframe_to_records(filtered)
        return dataframe_to_columnar(filtered)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get merged well data: {e}"
        ) from e


@router.post(
    "/projects/{project_id}/wells/{well_name}/merged/generate",
    summary="Compute merged data synchronously",
)
async def initiate_merged_data_generation(
    project_id: int,
    well_name: str,
    tolerance: Optional[float] = 0.16,
    async_task: bool = False,
) -> Dict[str, Any]:
    """Compute merged well data and return immediately."""
    try:
        result = _compute_merged_data_sync(project_id, well_name, tolerance)
        return {"task_id": "sync", "status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logging.getLogger(__name__).exception(
            "Failed to initiate merged data generation: %s", e
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate merged data generation: {e}",
        ) from e


@router.post("/cache/clear", summary="Clear merged data cache")
async def clear_merged_cache(
    project_id: Optional[int] = None, well_name: Optional[str] = None
):
    """Clear the merged data cache.

    If project_id and well_name are provided, clears only that specific cache entry.
    Otherwise, clears the entire cache.
    """
    cleared = invalidate_well_cache(project_id=project_id, well_name=well_name)
    if project_id is not None and well_name is not None:
        return {
            "message": f"Cleared cache for well {well_name}",
            "entries_cleared": cleared,
        }
    return {"message": "Cleared entire merged data cache", "entries_cleared": cleared}


@router.get("/cache/stats", summary="Get cache statistics")
async def get_cache_stats():
    """Get statistics about the merged data cache."""
    return {
        "cache_size": len(_merged_data_cache),
        "cache_ttl_seconds": _cache_ttl,
        "cached_wells": list({k.rsplit("_", 1)[0] for k in _merged_data_cache.keys()}),
    }


def _collect_project_handover_bundles(
    project_id: int,
    well_names_filter: Optional[List[str]] = None,
) -> tuple[str, str, List[Dict[str, Any]]]:
    """Gather project metadata and per-well handover dictionaries (including ephemeral/cached live edits)."""
    from quick_pp.handover import collect_well_handover_dict

    if project_id in _demo_deleted_projects:
        raise ValueError(f"Project {project_id} not found")

    allowed = (
        {str(w).strip() for w in well_names_filter if str(w).strip()}
        if well_names_filter
        else None
    )
    deleted_w_set = _demo_deleted_wells.get(project_id, set())
    renamed_w_map = _demo_well_renames.get(project_id, {})

    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)
        ovr = _demo_project_overrides.get(project_id, {})
        proj_name = ovr.get("name", proj.name)
        proj_desc = ovr.get("description", proj.description or "")

        wells_bundle: List[Dict[str, Any]] = []
        seen_names = set()

        for orm_well in proj._orm_project.wells:
            curr_name = renamed_w_map.get(orm_well.name, orm_well.name)
            if orm_well.name in deleted_w_set or curr_name in deleted_w_set:
                continue
            if allowed and curr_name not in allowed and orm_well.name not in allowed:
                continue

            seen_names.add(curr_name)
            well_obj = db_objects.Well(session, well_id=orm_well.well_id)
            # Check if there is an updated merged/ephemeral dataframe in memory
            live_df = None
            eph_entry = _ephemeral_wells.get(project_id, {}).get(curr_name)
            if isinstance(eph_entry, pd.DataFrame):
                live_df = eph_entry
            elif isinstance(eph_entry, dict) and isinstance(
                eph_entry.get("data"), pd.DataFrame
            ):
                live_df = eph_entry["data"]
            else:
                cached = _merged_data_cache.get(f"{project_id}_{curr_name}_0.16")
                if cached is not None:
                    live_df = cached[1]

            b_dict = collect_well_handover_dict(well_obj, override_df=live_df)
            b_dict["name"] = curr_name
            wells_bundle.append(b_dict)

        # Include any purely ephemeral uploaded wells in demo mode
        for eph_name, eph_val in _ephemeral_wells.get(project_id, {}).items():
            if eph_name in seen_names or eph_name in deleted_w_set:
                continue
            if allowed and eph_name not in allowed:
                continue
            eph_df = (
                eph_val
                if isinstance(eph_val, pd.DataFrame)
                else eph_val.get("data", pd.DataFrame())
                if isinstance(eph_val, dict)
                else pd.DataFrame()
            )
            wells_bundle.append(
                {
                    "name": eph_name,
                    "uwi": eph_name,
                    "depth_uom": "m",
                    "config": {},
                    "data": eph_df,
                    "formation_tops": pd.DataFrame(columns=["name", "depth"]),
                    "fluid_contacts": pd.DataFrame(columns=["name", "depth"]),
                    "well_surveys": pd.DataFrame(columns=["md", "inc", "azim"]),
                    "pressure_tests": pd.DataFrame(
                        columns=["depth", "pressure", "pressure_uom"]
                    ),
                    "core_data": {},
                    "scal_pc": [],
                }
            )

    return proj_name, proj_desc, wells_bundle


@router.get(
    "/projects/{project_id}/export",
    summary="Export Petrophysical Handover Package (.zip) for Techlog / Geolog / IP / Petrel",
)
def export_project_handover(
    project_id: int,
    well_name: Optional[str] = Query(
        None, description="Optional single well name to export"
    ),
    wells: Optional[str] = Query(
        None, description="Optional comma-separated list of well names to export"
    ),
    include_las: bool = Query(True, description="Include LAS 2.0 files"),
    include_excel: bool = Query(
        True, description="Include multi-tab Master Excel Workbook (.xlsx)"
    ),
):
    """Build and stream a standardized Petrophysical Handover Package (.zip)."""
    from quick_pp.handover import _sanitize_name, build_handover_archive_bytes

    try:
        filter_list: Optional[List[str]] = None
        if well_name:
            filter_list = [well_name]
        elif wells:
            filter_list = [w.strip() for w in wells.split(",") if w.strip()]

        proj_name, proj_desc, wells_bundle = _collect_project_handover_bundles(
            project_id=project_id, well_names_filter=filter_list
        )
        if not wells_bundle:
            raise ValueError("No matching wells found to export.")

        bundle_title = (
            f"{proj_name}_{well_name}"
            if well_name and len(wells_bundle) == 1
            else proj_name
        )
        zip_bytes = build_handover_archive_bytes(
            project_name=bundle_title,
            project_description=proj_desc,
            wells_bundle=wells_bundle,
            include_las=include_las,
            include_excel=include_excel,
        )
        filename = f"{_sanitize_name(bundle_title)}_Handover.zip"
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to export handover package: {e}"
        ) from e


@router.get(
    "/projects/{project_id}/wells/{well_name}/export_las",
    summary="Export evaluated well logs and CPI parameters as a LAS 2.0 file",
)
def export_well_las_endpoint(project_id: int, well_name: str):
    """Stream a single well's continuous curves and CPI parameters as a LAS 2.0 file."""
    from quick_pp.handover import _assign_zones_from_tops, _sanitize_name
    from quick_pp.las_handler import export_to_las_string

    try:
        _, _, wells_bundle = _collect_project_handover_bundles(
            project_id=project_id, well_names_filter=[well_name]
        )
        if not wells_bundle:
            raise ValueError(f"Well '{well_name}' not found in project {project_id}")

        w = wells_bundle[0]
        w_df: pd.DataFrame = w["data"]
        if w_df is None or w_df.empty:
            raise ValueError(f"Well '{well_name}' has no log data to export")

        tops_df: pd.DataFrame = w["formation_tops"]
        if not tops_df.empty and ("ZONES" not in w_df.columns or w_df["ZONES"].isna().all()):
            w_df = w_df.copy()
            w_df["ZONES"] = _assign_zones_from_tops(
                pd.to_numeric(w_df["DEPTH"], errors="coerce").to_numpy(dtype=float),
                tops_df,
            )

        las_content = export_to_las_string(
            well_data=w_df,
            well_name=w["name"],
            uwi=w["uwi"],
            depth_uom=w["depth_uom"],
            parameters=w["config"],
        )
        filename = f"{_sanitize_name(w['name'])}.las"
        return Response(
            content=las_content.encode("utf-8"),
            media_type="text/plain; charset=utf-8",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to export LAS file: {e}"
        ) from e
