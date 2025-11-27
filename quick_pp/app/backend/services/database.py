from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List, Optional
from pathlib import Path
import shutil

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
    db_url = payload.get("db_url") if isinstance(payload, dict) else None
    setup = bool(payload.get("setup")) if isinstance(payload, dict) else False

    try:
        connector = DBConnector(db_url=db_url)
        if setup:
            connector.setup_db()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to init DB: {e}")

    return {"message": "DB connector initialized", "db_url": str(connector._engine.url)}


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
            dest = upload_dir / f.filename
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
