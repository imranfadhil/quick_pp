import logging
import os
import shutil
import time
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from celery.result import AsyncResult
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from fastapi.responses import JSONResponse, ORJSONResponse
from sqlalchemy import select, text

from quick_pp.app.backend.task_queue.celery_app import celery_app
from quick_pp.app.backend.task_queue.tasks import process_las, process_merged_data
from quick_pp.database import models as db_models
from quick_pp.database import objects as db_objects
from quick_pp.app.backend.utils.db import get_db
from quick_pp.app.backend.utils.utils import sanitize_filename


router = APIRouter(prefix="/database", tags=["Database"])


# Simple in-memory cache for merged data (stores up to 50 wells, ~5 minutes TTL)
_merged_data_cache = {}
_cache_ttl = 300  # 5 minutes


@router.get("/health", summary="Database health check")
async def health_check():
    """Simple health endpoint that checks DB connectivity by running `SELECT 1`."""
    try:
        with get_db() as session:
            session.execute(text("SELECT 1"))
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}") from e


@router.get("/tasks/{task_id}", summary="Get Celery task status")
async def get_task_status(task_id: str):
    """Return Celery task state/result for a given task id."""
    try:
        res: AsyncResult = AsyncResult(task_id, app=celery_app)
        result = None
        try:
            # Accessing res.result may raise if task failed; protect it
            if res.ready():
                result = res.result
        except Exception:
            result = None

        return {
            "task_id": task_id,
            "state": res.state,
            "result": result,
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

    Expected payload: { "data": [ {"DEPTH": 1234.5, "WELL_NAME": "Well-1", "DPHI": 12.3, ...}, ... ] }

    If `WELL_NAME` column is missing, it will be set to the path parameter `well_name`.
    The endpoint will convert the list to a pandas DataFrame and call
    `Project.update_data` which performs upserts on curve data.
    """
    if not isinstance(payload, dict) or "data" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with a 'data' key"
        )

    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(
            status_code=400, detail="'data' must be a list of row objects"
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
            keys_to_remove = [
                k
                for k in _merged_data_cache.keys()
                if k.startswith(f"{project_id}_{wn}_")
            ]
            for key in keys_to_remove:
                del _merged_data_cache[key]

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
            records = df.to_dict(orient="records")

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


@router.get(
    "/projects/{project_id}/wells/{well_name}/merged",
    summary="Get well data merged with ancillary datapoints by depth",
    response_class=ORJSONResponse,
)
def get_well_data_merged(
    project_id: int,
    well_name: str,
    tolerance: Optional[float] = 0.16,
    use_cache: bool = True,
):
    """Return well log rows with nearby ancillary datapoints merged by depth.

    Uses in-memory caching for performance (5 minute TTL).
    Set use_cache=false to bypass cache.
    """
    # Check cache first
    cache_key = f"{project_id}_{well_name}_{tolerance}"
    if use_cache and cache_key in _merged_data_cache:
        cached_data, cached_time = _merged_data_cache[cache_key]
        if time.time() - cached_time < _cache_ttl:
            return cached_data

    # Try to enqueue merged-data task; if broker unavailable or enqueue fails, compute synchronously
    try:
        from quick_pp.app.backend.task_queue.celery_app import is_broker_available

        if is_broker_available():
            try:
                task = process_merged_data.apply_async(
                    args=(project_id, well_name),
                    kwargs={
                        "tolerance": float(tolerance),
                        "use_cache": bool(use_cache),
                    },
                )
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED, content={"task_id": task.id}
                )
            except Exception:
                logging.getLogger(__name__).exception(
                    "Failed to enqueue merged-data task; will compute synchronously"
                )
        else:
            logging.getLogger(__name__).warning(
                "Celery broker unavailable; computing merged data synchronously"
            )
    except Exception:
        logging.getLogger(__name__).exception(
            "Error checking Celery broker; computing merged data synchronously"
        )

    # Optimized vectorized type conversion
    def _to_py_vectorized(series):
        """Convert pandas series to Python-native types efficiently."""
        if pd.api.types.is_numeric_dtype(series):
            # Replace inf/-inf with None, convert to list
            return (
                series.replace([np.inf, -np.inf], np.nan)
                .where(pd.notna(series), None)
                .tolist()
            )
        elif pd.api.types.is_datetime64_any_dtype(series):
            return (
                series.dt.strftime("%Y-%m-%dT%H:%M:%S")
                .where(pd.notna(series), None)
                .tolist()
            )
        else:
            return series.where(pd.notna(series), None).tolist()

    def _normalize_df_depth(dframe):
        """Optimize depth normalization with minimal copying."""
        if dframe is None or dframe.empty:
            return None
        # Avoid copy if already normalized
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

            # Use optimized method if available
            try:
                df = proj.get_well_data_optimized(well_name)
            except AttributeError:
                df = proj.get_well_data(well_name)
            if df.empty:
                result = []
                if use_cache:
                    _merged_data_cache[cache_key] = (result, time.time())
                return result

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

            # Fetch ancillary data in one object to reduce queries
            well = proj.get_well(well_name)

            # Batch fetch ancillary data
            tops_df = _normalize_df_depth(well.get_formation_tops())
            contacts_df = _normalize_df_depth(well.get_fluid_contacts())
            pressure_df = _normalize_df_depth(well.get_pressure_tests())
            core_raw = well.get_core_data() or {}

            # Optimized prefix function with minimal copying
            def _prefixed(dframe, prefix):
                if dframe is None or dframe.empty:
                    return None
                # Already normalized by _normalize_df_depth
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

            # Use df directly instead of copying
            merged = df

            # Batch merge operations for better performance
            if tops_p is not None and not tops_p.empty:
                merged = pd.merge_asof(
                    merged,
                    tops_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tolerance,
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
                    tolerance=tolerance,
                    suffixes=("", "_contact"),
                )

            if pressure_p is not None and not pressure_p.empty:
                merged = pd.merge_asof(
                    merged,
                    pressure_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tolerance,
                    suffixes=("", "_pressure"),
                )

            # Optimized core data processing with vectorization
            core_rows = []
            if isinstance(core_raw, dict):
                for name, sd in core_raw.items():
                    try:
                        dval = sd.get("depth")
                        if dval is None:
                            continue
                        measurements = sd.get("measurements")
                        if measurements is None or measurements.empty:
                            continue

                        # Vectorized filtering instead of row-by-row iteration
                        cpore_mask = (
                            measurements["property"] == "cpore"
                        ) & measurements["value"].notna()
                        cperm_mask = (
                            measurements["property"] == "cperm"
                        ) & measurements["value"].notna()

                        for prop, mask in [
                            ("cpore", cpore_mask),
                            ("cperm", cperm_mask),
                        ]:
                            filtered = measurements[mask]
                            if not filtered.empty:
                                core_rows.extend(
                                    [
                                        {
                                            "depth": float(dval),
                                            "core_sample_name": name,
                                            prop: row["value"],
                                        }
                                        for _, row in filtered.iterrows()
                                    ]
                                )
                    except Exception:
                        continue

            if core_rows:
                core_df = (
                    pd.DataFrame(core_rows).sort_values("depth").reset_index(drop=True)
                )
                merged = pd.merge_asof(
                    merged,
                    core_df,
                    on="depth",
                    direction="nearest",
                    tolerance=tolerance,
                    suffixes=("", "_core"),
                )

            # Clean up inf values and normalize column names
            merged = merged.replace([np.inf, -np.inf], np.nan)
            merged.columns = [c.upper() for c in merged.columns]

            # Optimized serialization using vectorized conversion
            records = []
            for col in merged.columns:
                if len(records) == 0:
                    records = [{} for _ in range(len(merged))]
                values = _to_py_vectorized(merged[col])
                for i, val in enumerate(values):
                    records[i][col] = val

            # Cache the result
            if use_cache:
                _merged_data_cache[cache_key] = (records, time.time())
                # Simple cache size management: keep only last 50 entries
                if len(_merged_data_cache) > 50:
                    oldest_key = min(
                        _merged_data_cache.keys(),
                        key=lambda k: _merged_data_cache[k][1],
                    )
                    del _merged_data_cache[oldest_key]

            return records
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to produce merged data: {e}"
        ) from e


@router.post("/cache/clear", summary="Clear merged data cache")
async def clear_merged_cache(
    project_id: Optional[int] = None, well_name: Optional[str] = None
):
    """Clear the merged data cache.

    If project_id and well_name are provided, clears only that specific cache entry.
    Otherwise, clears the entire cache.
    """
    global _merged_data_cache

    if project_id is not None and well_name is not None:
        # Clear specific entries for this well (all tolerance values)
        keys_to_remove = [
            k
            for k in _merged_data_cache.keys()
            if k.startswith(f"{project_id}_{well_name}_")
        ]
        for key in keys_to_remove:
            del _merged_data_cache[key]
        return {
            "message": f"Cleared cache for well {well_name}",
            "entries_cleared": len(keys_to_remove),
        }
    else:
        count = len(_merged_data_cache)
        _merged_data_cache.clear()
        return {"message": "Cleared entire merged data cache", "entries_cleared": count}


@router.get("/cache/stats", summary="Get cache statistics")
async def get_cache_stats():
    """Get statistics about the merged data cache."""
    return {
        "cache_size": len(_merged_data_cache),
        "cache_ttl_seconds": _cache_ttl,
        "cached_wells": list({k.rsplit("_", 1)[0] for k in _merged_data_cache.keys()}),
    }
