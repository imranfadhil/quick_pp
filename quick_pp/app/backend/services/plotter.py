import json
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from quick_pp.database import objects as db_objects
from quick_pp.plotter import well_log as wl
from quick_pp.app.backend.task_queue.tasks import generate_well_plot
from quick_pp.app.backend.task_queue.celery_app import is_broker_available

from quick_pp.app.backend.utils.db import get_db


router = APIRouter(prefix="/plotter", tags=["Plotter"])


@router.get(
    "/projects/{project_id}/wells/{well_name}/log", summary="Get well log Plotly JSON"
)
async def get_well_log(
    project_id: int,
    well_name: str,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[str] = None,
):
    """Return a Plotly figure JSON for the given project well.

    Args:
        project_id: Project ID
        well_name: Name of the well
        min_depth: Optional minimum depth filter
        max_depth: Optional maximum depth filter

    Returns:
        Plotly figure JSON with optional depth filtering applied
    """
    try:
        # Try enqueueing a Celery task; if broker unavailable or enqueue fails, fall back to synchronous processing
        try:
            if is_broker_available():
                try:
                    task = generate_well_plot.apply_async(
                        args=(project_id, well_name),
                        kwargs={
                            "min_depth": min_depth,
                            "max_depth": max_depth,
                            "zones": zones,
                        },
                    )
                    # Wait for task to complete and return result directly
                    result = task.get(timeout=30)  # 30-second timeout
                    parsed = (
                        json.loads(json.dumps(result))
                        if isinstance(result, dict)
                        else result
                    )
                    return JSONResponse(content=parsed)
                except Exception:
                    logging.getLogger(__name__).exception(
                        "Failed to execute plot task asynchronously, falling back to sync"
                    )
            else:
                logging.getLogger(__name__).warning(
                    "Celery broker unavailable, generating plot synchronously"
                )
        except Exception:
            logging.getLogger(__name__).exception(
                "Unexpected error checking broker; proceeding synchronously"
            )

        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            df = proj.get_well_data_optimized(well_name)

            # Load ancillary data (formation tops and core data)
            try:
                ancillary = proj.get_well_ancillary_data(well_name)
            except Exception:
                ancillary = {}

            # If formation tops exist, mark their names in a 'ZONES' column
            tops_df = None
            if isinstance(ancillary, dict) and "formation_tops" in ancillary:
                tops_df = ancillary.get("formation_tops")
                if (
                    isinstance(tops_df, pd.DataFrame)
                    and not tops_df.empty
                    and not df.empty
                ):
                    # Ensure ZONES column exists
                    df["ZONES"] = pd.NA
                    # For each top, find the closest depth row in df and annotate
                    for _, top in tops_df.iterrows():
                        try:
                            top_depth = float(top.get("depth"))
                            top_name = str(top.get("name"))
                        except Exception:
                            continue
                        # find index of nearest depth
                        nearest_idx = (df["DEPTH"] - top_depth).abs().idxmin()
                        df.at[nearest_idx, "ZONES"] = top_name
                    df["ZONES"] = df["ZONES"].ffill()

            # If core data exists, extract CPORE and CPERM measurements and insert into df
            if (
                isinstance(ancillary, dict)
                and "core_data" in ancillary
                and not df.empty
            ):
                core_dict = ancillary.get("core_data") or {}
                # Ensure columns exist
                if "CPORE" not in df.columns:
                    df["CPORE"] = pd.NA
                if "CPERM" not in df.columns:
                    df["CPERM"] = pd.NA

                for _, sample in (
                    core_dict.items() if isinstance(core_dict, dict) else []
                ):
                    try:
                        sample_depth = sample.get("depth")
                        measurements = sample.get("measurements")
                        if measurements is None or measurements.empty:
                            continue
                        # measurements expected to have columns ['property','value']
                        for _, m in measurements.iterrows():
                            prop = str(m.get("property") or "").upper()
                            val = m.get("value")
                            if prop in ("CPORE", "CPERM") and pd.notna(val):
                                # find nearest depth and set value
                                try:
                                    nearest_idx = (
                                        (df["DEPTH"] - float(sample_depth))
                                        .abs()
                                        .idxmin()
                                    )
                                    df.at[nearest_idx, prop] = val
                                except Exception:
                                    continue
                    except Exception:
                        continue

            # If dataframe is empty, return 404
            if df.empty:
                raise HTTPException(
                    status_code=404, detail=f"No data for well {well_name}"
                )

            # Apply zone filtering if zones provided
            if zones:
                zone_list = [z.strip() for z in zones.split(",") if z.strip()]
                if "ZONES" in df.columns:
                    df = df[df["ZONES"].isin(zone_list)]
                    if df.empty:
                        raise HTTPException(
                            status_code=404,
                            detail=f"No data for well {well_name} in specified zones",
                        )
                else:
                    raise HTTPException(
                        status_code=404,
                        detail=f"No zone information available for well {well_name}",
                    )

            # Apply depth filtering if parameters provided
            if min_depth is not None or max_depth is not None:
                depth_col = None
                # Find depth column (try common naming conventions)
                for col in ["depth", "DEPTH", "Depth", "TVDSS", "tvdss", "TVD", "tvd"]:
                    if col in df.columns:
                        depth_col = col
                        break

                if depth_col is not None:
                    # Apply depth filter
                    if min_depth is not None:
                        df = df[df[depth_col] >= min_depth]
                    if max_depth is not None:
                        df = df[df[depth_col] <= max_depth]

                    # Check if any data remains after filtering
                    if df.empty:
                        raise HTTPException(
                            status_code=404,
                            detail=f"No data for well {well_name} in specified depth range",
                        )

            fig = wl.plotly_log(df, well_name=well_name)
            # plotly Figure to JSON string
            fig_json = fig.to_json()
            # Parse string to python object for JSONResponse
            parsed = json.loads(fig_json)
            return JSONResponse(content=parsed)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to build well log: {e}"
        ) from e
