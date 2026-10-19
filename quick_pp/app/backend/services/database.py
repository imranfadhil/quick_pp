import logging
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from celery.result import AsyncResult
from fastapi import APIRouter, File, Form, HTTPException, Query, UploadFile, status
from fastapi.responses import JSONResponse, ORJSONResponse
from sqlalchemy import bindparam, func, select, text

from quick_pp.app.backend.task_queue.celery_app import celery_app, is_broker_available
from quick_pp.app.backend.task_queue.tasks import process_las, process_merged_data
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


@router.get("/tasks/{task_id}", summary="Get Celery task status")
async def get_task_status(task_id: str):
    """Return Celery task state/result for a given task id."""
    try:
        res: AsyncResult = AsyncResult(task_id, app=celery_app)
        result = None
        error = None
        try:
            # If task is ready, retrieve result or exception
            if res.ready():
                if res.successful():
                    result = res.result
                else:
                    # Task failed; get the exception
                    error = str(res.info)
            elif res.failed():
                error = str(res.info)
        except Exception as exc:
            error = str(exc)

        return {
            "task_id": task_id,
            "state": res.state,
            "result": result,
            "error": error,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get task status: {e}"
        ) from e


@router.post("/projects", summary="Create or get project")
async def create_project(payload: dict):
    """Create a new project (or return existing by name).

    Payload: {"name": "Project A", "description": "...", "created_by_user_id": 1}
    """
    name = payload.get("name") if isinstance(payload, dict) else None
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    description = payload.get("description") if isinstance(payload, dict) else ""
    created_by = (
        payload.get("created_by_user_id") if isinstance(payload, dict) else None
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
            return {"project_id": proj.project_id, "name": proj.name}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create project: {e}"
        ) from e


@router.get("/projects", summary="List projects")
async def list_projects():
    try:
        with get_db() as session:
            stmt = select(db_models.Project)
            rows = session.execute(stmt).scalars().all()
            return [
                {
                    "project_id": r.project_id,
                    "name": r.name,
                    "description": r.description,
                }
                for r in rows
            ]
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list projects: {e}"
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
            return {"project_id": proj.project_id, "wells": wells}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list wells: {e}") from e


@router.get(
    "/projects/{project_id}/summary",
    summary="Get lightweight project and per-well summary statistics",
    response_class=ORJSONResponse,
)
def get_project_summary(project_id: int):
    """Return aggregated metadata and statistics for all wells in a project without loading full curve arrays."""
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            orm_wells = proj._orm_project.wells
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
async def project_read_las(
    project_id: int,
    files: List[UploadFile] = File(...),
    depth_uom: Optional[str] = Form("m"),
):
    """Upload LAS files and add them into the project in the database."""
    upload_dir = Path("uploads/las_db/")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        for f in files:
            # Prefix uploaded filename with a UUID to avoid collisions when
            # multiple uploads use the same original filename concurrently.
            safe_name = sanitize_filename(f.filename)
            unique_name = f"{uuid4().hex}_{safe_name}"
            dest = upload_dir / unique_name
            try:
                with dest.open("wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
            finally:
                f.file.close()
            saved_paths.append(str(dest))

        invalidate_well_cache(project_id=project_id)

        # Try to enqueue the Celery task if broker is available; otherwise fall back
        from quick_pp.app.backend.task_queue.celery_app import is_broker_available

        if is_broker_available():
            try:
                async_result = process_las.apply_async(
                    args=[saved_paths, project_id, depth_uom]
                )
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={
                        "message": "Files uploaded. Processing enqueued.",
                        "task_id": async_result.id,
                        "files": [Path(p).name for p in saved_paths],
                    },
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to enqueue Celery task, falling back to sync processing"
                )

        # Broker not available or enqueue failed -> do synchronous processing as fallback
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            proj.read_las(saved_paths, depth_uom=depth_uom)

        invalidate_well_cache(project_id=project_id)

        return {
            "message": "Files uploaded and processed (sync fallback)",
            "files": [Path(p).name for p in saved_paths],
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read LAS into project: {e}"
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
    response_class=ORJSONResponse,
)
def get_well_data(project_id: int, well_name: str, include_ancillary: bool = False):
    """Return the full well data (curve logs) as a JSON array of rows.

    Each row is a dict mapping column names to values.
    Uses optimized data retrieval and faster JSON serialization.
    """
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Use optimized method if available
            try:
                df = proj.get_well_data_optimized(well_name)
            except AttributeError:
                df = proj.get_well_data(well_name)

            if df.empty:
                return []

            df = df.reset_index(drop=True)
            # Use orjson for faster serialization (via ORJSONResponse)
            records = dataframe_to_records(df)

            if include_ancillary:
                well = proj.get_well(well_name)
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
        if now - cached_ts < _cache_ttl:
            return cached_df.copy()
        else:
            _merged_data_cache.pop(cache_key, None)

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
                df = proj.get_well_data_optimized(well_name)
            except AttributeError:
                df = proj.get_well_data(well_name)
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

            well = proj.get_well(well_name)

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
    """Compute merged well data synchronously (helper function).

    Args:
        project_id: Project ID
        well_name: Name of the well
        tolerance: Merge tolerance for depth-based merging

    Returns:
        List of merged data records
    """
    merged = _get_merged_dataframe_sync(project_id, well_name, tolerance=tolerance)
    return dataframe_to_records(merged)


@router.get(
    "/projects/{project_id}/wells/{well_name}/merged",
    summary="Get merged well data directly in columnar or records format",
    response_class=ORJSONResponse,
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
):
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
    summary="Initiate merged data generation (sync by default, optional async)",
    response_class=ORJSONResponse,
)
async def initiate_merged_data_generation(
    project_id: int,
    well_name: str,
    tolerance: Optional[float] = 0.16,
    async_task: bool = False,
):
    """Compute merged well data and return immediately (or enqueue a Celery task if async_task=True).

    Args:
        project_id: Project ID
        well_name: Name of the well
        tolerance: Merge tolerance for depth-based merging
        async_task: If True and Celery broker is available, enqueue background task

    Returns:
        JSON with task_id="sync" and result immediately, or task_id for async polling
    """
    try:
        if async_task and is_broker_available():
            logging.getLogger(__name__).info(
                "Celery broker available, enqueueing async merged data task"
            )
            task = process_merged_data.apply_async(
                args=(project_id, well_name),
                kwargs={"tolerance": float(tolerance)},
            )
            return ORJSONResponse(content={"task_id": task.id})

        result = _compute_merged_data_sync(project_id, well_name, tolerance)
        return ORJSONResponse(
            content={"task_id": "sync", "status": "success", "result": result}
        )
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


@router.get(
    "/projects/{project_id}/wells/{well_name}/merged/result/{task_id}",
    summary="Poll for merged data generation result",
)
async def get_merged_data_result(task_id: str):
    """Poll for the result of a merged data generation task.

    Args:
        project_id: Project ID (for validation)
        well_name: Well name (for validation)
        task_id: Task ID returned from initiate endpoint

    Returns:
        JSON with status and result (when ready)
    """
    try:
        # Handle synthetic "sync" task ID (synchronous fallback result)
        if task_id == "sync":
            return JSONResponse(
                content={
                    "status": "success",
                    "result": {},  # Result should have been included in initiate response
                }
            )

        from quick_pp.app.backend.task_queue.celery_app import celery_app as app

        task = app.AsyncResult(task_id)

        if task.state == "PENDING":
            return JSONResponse(content={"status": "pending"})
        elif task.state == "SUCCESS":
            result = task.result
            # Ensure result is JSON serializable
            if not isinstance(result, list):
                result = list(result) if hasattr(result, "__iter__") else []
            # Sanitize NaN/Inf values before JSON serialization
            result = _sanitize_for_json(result)
            return JSONResponse(content={"status": "success", "result": result})
        elif task.state == "FAILURE":
            error_msg = str(task.info) if task.info else "Unknown error"
            return JSONResponse(
                content={"status": "error", "error": error_msg},
                status_code=200,  # Still 200 to allow client to handle gracefully
            )
        else:
            # RETRY, REVOKED, etc.
            return JSONResponse(content={"status": "pending", "state": task.state})
    except Exception as e:
        logging.getLogger(__name__).exception("Failed to check task result: %s", e)
        return JSONResponse(
            content={"status": "error", "error": str(e)},
            status_code=500,
        )


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
