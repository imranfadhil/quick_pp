import json
import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from quick_pp.app.backend.task_queue.tasks import generate_well_plot
from quick_pp.app.backend.task_queue.celery_app import is_broker_available
from quick_pp.app.backend.utils.db import get_db
from quick_pp.database import objects as db_objects
from quick_pp.plotter import well_log as wl


router = APIRouter(prefix="/plotter", tags=["Plotter"])


def _generate_well_plot_sync(
    project_id: int,
    well_name: str,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[str] = None,
) -> dict:
    """Generate well plot synchronously (helper function).

    Args:
        project_id: Project ID
        well_name: Name of the well
        min_depth: Optional minimum depth filter
        max_depth: Optional maximum depth filter
        zones: Optional comma-separated zone names

    Returns:
        Plotly figure JSON parsed as dict
    """
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
            if isinstance(tops_df, pd.DataFrame) and not tops_df.empty and not df.empty:
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
        if isinstance(ancillary, dict) and "core_data" in ancillary and not df.empty:
            core_dict = ancillary.get("core_data") or {}
            # Ensure columns exist
            if "CPORE" not in df.columns:
                df["CPORE"] = pd.NA
            if "CPERM" not in df.columns:
                df["CPERM"] = pd.NA

            for _, sample in core_dict.items() if isinstance(core_dict, dict) else []:
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
                                    (df["DEPTH"] - float(sample_depth)).abs().idxmin()
                                )
                                df.at[nearest_idx, prop] = val
                            except Exception:
                                continue
                except Exception:
                    continue

        # If dataframe is empty, raise error
        if df.empty:
            raise ValueError(f"No data for well {well_name}")

        # Apply zone filtering if zones provided
        if zones:
            zone_list = [z.strip() for z in zones.split(",") if z.strip()]
            if "ZONES" in df.columns:
                df = df[df["ZONES"].isin(zone_list)]
                if df.empty:
                    raise ValueError(f"No data for well {well_name} in specified zones")
            else:
                raise ValueError(f"No zone information available for well {well_name}")

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
                    raise ValueError(
                        f"No data for well {well_name} in specified depth range"
                    )

        fig = wl.plotly_log(df, well_name=well_name)
        # plotly Figure to JSON string
        fig_json = fig.to_json()
        # Parse string to python object
        parsed = json.loads(fig_json)
        return parsed


@router.post(
    "/projects/{project_id}/wells/{well_name}/log/generate",
    summary="Initiate async well log plot generation",
)
async def initiate_well_log_generation(
    project_id: int,
    well_name: str,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[str] = None,
):
    """Initiate an async plot generation task and return the task ID.

    If Celery broker is unavailable, falls back to synchronous generation
    and returns the result immediately.

    Args:
        project_id: Project ID
        well_name: Name of the well
        min_depth: Optional minimum depth filter
        max_depth: Optional maximum depth filter
        zones: Optional comma-separated zone names

    Returns:
        JSON with task_id for async polling, or result directly if sync fallback
    """
    try:
        if is_broker_available():
            logging.getLogger(__name__).info(
                "Celery broker available, enqueueing async plot task"
            )
            task = generate_well_plot.apply_async(
                args=(project_id, well_name),
                kwargs={
                    "min_depth": min_depth,
                    "max_depth": max_depth,
                    "zones": zones,
                },
            )
            return JSONResponse(content={"task_id": task.id})
        else:
            logging.getLogger(__name__).warning(
                "Celery broker unavailable, generating plot synchronously"
            )
            result = _generate_well_plot_sync(
                project_id, well_name, min_depth, max_depth, zones
            )
            # Return result directly with a synthetic task_id that will resolve immediately
            return JSONResponse(
                content={"task_id": "sync", "status": "success", "result": result}
            )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logging.getLogger(__name__).exception(
            "Failed to initiate plot generation: %s", e
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to initiate plot generation: {e}",
        ) from e


@router.get(
    "/projects/{project_id}/wells/{well_name}/log/result/{task_id}",
    summary="Poll for plot generation result",
)
async def get_plot_result(task_id: str):
    """Poll for the result of a plot generation task.

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
            parsed = (
                json.loads(json.dumps(result)) if isinstance(result, dict) else result
            )
            return JSONResponse(content={"status": "success", "result": parsed})
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
