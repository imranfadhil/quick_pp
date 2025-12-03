from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json

from . import database as database_service
from quick_pp.database import objects as db_objects

from quick_pp.plotter import well_log as wl
from pydantic import BaseModel
from typing import Optional, List

router = APIRouter(prefix="/plotter", tags=["Plotter"])


@router.get(
    "/projects/{project_id}/wells/{well_name}/log", summary="Get well log Plotly JSON"
)
async def get_well_log(project_id: int, well_name: str):
    """Return a Plotly figure JSON for the given project well.

    This endpoint expects the DB connector to be initialized via
    `/database/init` beforehand.
    """
    # reference the connector on the `database` module so updates via /database/init
    # are visible here (importing the value directly would capture the initial None)
    db_connector = getattr(database_service, "connector", None)
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            df = proj.get_well_data_optimized(well_name)

            # If dataframe is empty, return 404
            if df.empty:
                raise HTTPException(
                    status_code=404, detail=f"No data for well {well_name}"
                )

            fig = wl.plotly_log(df, well_name=well_name)
            # plotly Figure to JSON string
            fig_json = fig.to_json()
            # Parse string to python object for JSONResponse
            parsed = json.loads(fig_json)
            return JSONResponse(content=parsed)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to build well log: {e}")
