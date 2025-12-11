import json
import math
import os
import shutil
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy import select

from quick_pp.database import models as db_models
from quick_pp.database import objects as db_objects
from quick_pp.database.db_connector import DBConnector

router = APIRouter(prefix="/database", tags=["Database"])

# Module-level connector instance (initialized via endpoint)
connector: Optional[DBConnector] = None


@router.post("/init", summary="Initialize database connector")
async def init_db(payload: dict):
    """Initialize the DB connector. Payload optionally contains `db_url` and `setup`.

    Example: {"db_url": "sqlite:///./data/quick_pp.db", "setup": true}
    """
    global connector
    # Prefer explicit `db_url` from payload; fall back to the environment variable
    # `QPP_DATABASE_URL` so the service can initialize using container config.
    db_url = None
    if isinstance(payload, dict):
        db_url = payload.get("db_url")
        setup = bool(payload.get("setup")) if isinstance(payload, dict) else False
    if not db_url:
        db_url = os.environ.get("QPP_DATABASE_URL")
        setup = True

    try:
        # If the application startup already initialized the DB (per-worker), reuse it
        if DBConnector._engine is not None:
            connector = (
                DBConnector()
            )  # wrapper referencing existing class-level engine/session
            if setup:
                try:
                    connector.setup_db()
                except Exception as e:
                    raise HTTPException(
                        status_code=500, detail=f"Failed to setup DB: {e}"
                    ) from e
            return {
                "message": "DB connector already initialized",
                "db_url": str(connector._engine.url),
            }

        # Otherwise initialize a new connector instance for this process
        connector = DBConnector(db_url=db_url)
        if setup:
            connector.setup_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init DB: {e}") from e

    return {"message": "DB connector initialized", "db_url": str(connector._engine.url)}


@router.get("/health", summary="Database health check")
async def health_check():
    """Simple health endpoint that checks DB connectivity by running `SELECT 1`.

    Returns 200 + {'status': 'ok'} when the DB responds, otherwise raises 500.
    If the DB connector hasn't been initialized, returns 503.
    """
    global connector
    # Prefer already-initialized class-level engine if present
    engine = None
    if DBConnector._engine is not None:
        engine = DBConnector._engine
    elif connector is not None and getattr(connector, "_engine", None) is not None:
        engine = connector._engine

    if engine is None:
        raise HTTPException(status_code=503, detail="DB connector not initialized")

    try:
        with engine.connect() as conn:
            # Use exec_driver_sql for a lightweight check that works across drivers
            conn.exec_driver_sql("SELECT 1")
        return {"status": "ok"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}") from e


@router.post("/projects", summary="Create or get project")
async def create_project(payload: dict):
    """Create a new project (or return existing by name).

    Payload: {"name": "Project A", "description": "...", "created_by_user_id": 1}
    """
    global connector
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    name = payload.get("name") if isinstance(payload, dict) else None
    if not name:
        raise HTTPException(status_code=400, detail="Project name is required")

    description = payload.get("description") if isinstance(payload, dict) else ""
    created_by = (
        payload.get("created_by_user_id") if isinstance(payload, dict) else None
    )

    try:
        with connector.get_session() as session:
            proj = db_objects.Project(
                db_session=session,
                name=name,
                description=description,
                created_by_user_id=created_by,
            )
            proj.save()
            # session.commit() handled by connector.get_session
            return {"project_id": proj.project_id, "name": proj.name}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to create project: {e}"
        ) from e


@router.get("/projects", summary="List projects")
async def list_projects():
    global connector
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with connector.get_session() as session:
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
    global connector
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with connector.get_session() as session:
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
    global connector
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Invalid payload")

    name = payload.get("name")
    if not name:
        raise HTTPException(status_code=400, detail="Well name is required")

    uwi = payload.get("uwi")
    depth_uom = payload.get("depth_uom")

    try:
        with connector.get_session() as session:
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
    global connector
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    upload_dir = Path("uploads/las_db/")
    upload_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    try:
        for f in files:
            # Prefix uploaded filename with a UUID to avoid collisions when
            # multiple uploads use the same original filename concurrently.
            unique_name = f"{uuid4().hex}_{f.filename}"
            dest = upload_dir / unique_name
            try:
                with dest.open("wb") as buffer:
                    shutil.copyfileobj(f.file, buffer)
            finally:
                f.file.close()
            saved_paths.append(str(dest))

        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            # Pass depth unit of measurement to Project.read_las (defaults to 'm')
            proj.read_las(saved_paths, depth_uom=depth_uom)

        return {
            "message": "Files uploaded and processed",
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
    global connector
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

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

        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            # Use Project.update_data which groups by WELL_NAME and upserts per well
            proj.update_data(df, group_by="WELL_NAME")
            # Persist changes
            proj.save()

        return {"message": "Well data saved", "rows": len(df)}

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
def get_well_data(project_id: int, well_name: str, include_ancillary: bool = False):
    """Return the full well data (curve logs) as a JSON array of rows.

    Each row is a dict mapping column names to values.
    """
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            df = proj.get_well_data(well_name)
            if df.empty:
                return []

            df = df.reset_index(drop=True)
            records = json.loads(df.to_json(orient="records"))

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
)
def get_well_data_merged(
    project_id: int, well_name: str, tolerance: Optional[float] = 0.16
):
    """Return well log rows with nearby ancillary datapoints merged by depth."""
    if connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    def _to_py(val):
        try:
            if val is None:
                return None
            if hasattr(val, "item"):
                val = val.item()
            if hasattr(val, "isoformat") and not isinstance(val, str):
                try:
                    return val.isoformat()
                except Exception:
                    return None
            try:
                if isinstance(val, (float, int)):
                    if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                        return None
                    return val
            except Exception:
                pass
            try:
                if pd.isna(val):
                    return None
            except Exception:
                pass
            return val
        except Exception:
            return None

    def _normalize_df_depth(dframe):
        if dframe is None or dframe.empty:
            return None
        d = dframe.copy()
        d.columns = [c.lower() for c in d.columns]
        if "depth" not in d.columns:
            candidate = next(
                (c for c in d.columns if "depth" in c or c in ("md", "tvd")),
                None,
            )
            if candidate:
                d = d.rename(columns={candidate: "depth"})
        d["depth"] = pd.to_numeric(d["depth"], errors="coerce")
        d = d.dropna(subset=["depth"]) if not d.empty else d
        return d

    try:
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            df = proj.get_well_data(well_name)
            if df.empty:
                return []

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
            df = df.dropna(subset=["depth"]) if not df.empty else df
            df = df.sort_values("depth").reset_index(drop=True)

            well = proj.get_well(well_name)
            tops_df = _normalize_df_depth(well.get_formation_tops())
            contacts_df = _normalize_df_depth(well.get_fluid_contacts())
            pressure_df = _normalize_df_depth(well.get_pressure_tests())
            core_raw = well.get_core_data() or {}

            def _prefixed(dframe, prefix):
                if dframe is None or dframe.empty:
                    return None
                d2 = dframe.copy()
                d2.columns = [c.lower() for c in d2.columns]
                if "depth" not in d2.columns:
                    return None
                cols = [c for c in d2.columns if c != "depth"]
                rename = {c: f"{prefix}{c}" for c in cols}
                d2 = d2.rename(columns=rename)
                d2["depth"] = pd.to_numeric(d2["depth"], errors="coerce")
                d2 = d2.dropna(subset=["depth"]) if not d2.empty else d2
                return d2.sort_values("depth").reset_index(drop=True)

            tops_p = _prefixed(tops_df, "top_")
            contacts_p = _prefixed(contacts_df, "contact_")
            pressure_p = _prefixed(pressure_df, "pressure_")

            merged = df.copy()
            merged["depth"] = pd.to_numeric(merged["depth"], errors="coerce")
            merged = merged.dropna(subset=["depth"]) if not merged.empty else merged
            merged = merged.sort_values("depth").reset_index(drop=True)

            if tops_p is not None and not tops_p.empty:
                merged = pd.merge_asof(
                    merged,
                    tops_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tolerance,
                    suffixes=("", "_top"),
                )
                merged["ZONES"] = merged["top_name"].ffill()
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
                        for _, m in measurements.iterrows():
                            prop = m.get("property")
                            val = m.get("value")
                            if prop in ("cpore", "cperm") and pd.notna(val):
                                core_rows.append(
                                    {
                                        "depth": float(dval),
                                        "core_sample_name": name,
                                        prop: val,
                                    }
                                )
                    except Exception:
                        continue
            core_df = pd.DataFrame(core_rows) if core_rows else None
            if core_df is not None and not core_df.empty:
                core_df = core_df.sort_values("depth").reset_index(drop=True)
                merged = pd.merge_asof(
                    merged,
                    core_df,
                    on="depth",
                    direction="nearest",
                    tolerance=tolerance,
                    suffixes=("", "_core"),
                )

            merged = merged.replace([np.inf, -np.inf], pd.NA)
            merged.columns = [c.upper() for c in merged.columns]

            records = []
            for r in merged.to_dict(orient="records"):
                records.append({k: _to_py(v) for k, v in r.items()})
            return records
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to produce merged data: {e}"
        ) from e
