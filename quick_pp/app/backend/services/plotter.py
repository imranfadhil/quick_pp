from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
import json
import numpy as np

from . import database as database_service
from quick_pp.database import objects as db_objects

from quick_pp.plotter import well_log as wl
from quick_pp.plotter.plotter import neutron_density_xplot as ndx
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


class NdRequest(BaseModel):
    dry_min1_point: List[float]
    dry_clay_point: List[float]
    fluid_point: Optional[List[float]] = [1.0, 1.0]
    wet_clay_point: Optional[List[float]] = None
    dry_silt_point: Optional[List[float]] = None


@router.post(
    "/projects/{project_id}/wells/{well_name}/ndx",
    summary="Neutron-density crossplot (Plotly JSON)",
)
async def get_neuden_xplot(project_id: int, well_name: str, body: NdRequest):
    """Build the NPHI-RHOB crossplot for a well and return Plotly JSON.

    The request body must include `dry_min1_point` and `dry_clay_point` as
    two-element arrays. Other points are optional.
    """
    db_connector = getattr(database_service, "connector", None)
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)

            # Find actual mnemonics for NPHI and RHOB using optimized search
            found_mnemonics = well.find_curve_mnemonics(["nphi", "rhob"])

            if len(found_mnemonics) < 2:
                raise HTTPException(
                    status_code=400,
                    detail="NPHI or RHOB columns not found in well data",
                )

            # Get only the curves we need for better performance
            needed_mnemonics = list(found_mnemonics.values())
            df = well.get_curve_data(needed_mnemonics)

            if df.empty:
                raise HTTPException(
                    status_code=400, detail="No valid NPHI/RHOB data in well"
                )

            # Get the actual column names
            nphi_col = found_mnemonics.get("nphi")
            rhob_col = found_mnemonics.get("rhob")

            if nphi_col not in df.columns or rhob_col not in df.columns:
                raise HTTPException(
                    status_code=400,
                    detail="NPHI or RHOB columns not found in retrieved data",
                )

            tmp = df[[nphi_col, rhob_col]].copy()
            tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna()
            if tmp.empty:
                raise HTTPException(
                    status_code=400, detail="No valid NPHI/RHOB data in well"
                )

            nphi = tmp[nphi_col].astype(float).values
            rhob = tmp[rhob_col].astype(float).values

            dry_min1 = tuple(body.dry_min1_point)
            dry_clay = tuple(body.dry_clay_point)
            fluid = tuple(body.fluid_point) if body.fluid_point else (1.0, 1.0)
            wet = tuple(body.wet_clay_point) if body.wet_clay_point else ()
            silt = tuple(body.dry_silt_point) if body.dry_silt_point else ()

            fig = ndx(
                nphi,
                rhob,
                dry_min1,
                dry_clay,
                fluid_point=fluid,
                wet_clay_point=wet,
                dry_silt_point=silt,
            )
            fig_json = fig.to_json()
            parsed = json.loads(fig_json)
            return JSONResponse(content=parsed)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to build ND crossplot: {e}"
        )
