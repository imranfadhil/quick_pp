import os
from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pathlib import Path
from uuid import uuid4
import shutil

import pandas as pd
from quick_pp.database.db_connector import DBConnector
from quick_pp.database import objects as db_objects
from quick_pp.database import models as db_models
from sqlalchemy import select

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
                    )
            return {
                "message": "DB connector already initialized",
                "db_url": str(connector._engine.url),
            }

        # Otherwise initialize a new connector instance for this process
        connector = DBConnector(db_url=db_url)
        if setup:
            connector.setup_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init DB: {e}")

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
        raise HTTPException(status_code=500, detail=f"DB connection failed: {e}")


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
        raise HTTPException(status_code=500, detail=f"Failed to create project: {e}")


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
        raise HTTPException(status_code=500, detail=f"Failed to list projects: {e}")


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
            return {"project_id": proj.project_id, "wells": proj.get_well_names()}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list wells: {e}")


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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create well: {e}")


@router.post("/projects/{project_id}/read_las", summary="Upload LAS files into project")
async def project_read_las(project_id: int, files: List[UploadFile] = File(...)):
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
            proj.read_las(saved_paths)

        return {
            "message": "Files uploaded and processed",
            "files": [Path(p).name for p in saved_paths],
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to read LAS into project: {e}"
        )


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
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save well data: {e}")
