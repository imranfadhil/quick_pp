import logging
import os

import json
import pandas as pd
import numpy as np

from quick_pp.app.backend.task_queue.celery_app import celery_app
from quick_pp.database import objects as db_objects
from quick_pp.database.db_connector import DBConnector
from quick_pp.plotter import well_log as wl


@celery_app.task(bind=True, name="quick_pp.tasks.process_las", acks_late=True)
def process_las(self, saved_paths, project_id, depth_uom="m"):
    """Process uploaded LAS files and add them into the project.

    This task runs in a worker process and creates its own DBConnector
    instance to avoid inheriting engines from the webserver process.
    """
    try:
        connector = DBConnector(db_url=os.environ.get("QPP_DATABASE_URL"))
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            result = proj.read_las(saved_paths, depth_uom=depth_uom)

        # If read_las returns a summary dict, inspect processed_files
        if isinstance(result, dict):
            processed = int(result.get("processed_files", 0))
            if processed == 0:
                # No files processed — treat this as a failure so the task is marked failed
                raise RuntimeError("No LAS files were successfully parsed")
            return {"status": "success", "summary": result}

        # Fallback: if read_las didn't return structured info, assume success
        return {
            "status": "success",
            "files": [os.path.basename(p) for p in saved_paths],
        }
    except Exception as exc:
        logging.getLogger(__name__).exception("Celery LAS processing failed: %s", exc)
        # Re-raise so Celery marks the task as failed and stores the traceback
        raise


@celery_app.task(bind=True, name="quick_pp.tasks.process_merged_data", acks_late=True)
def process_merged_data(
    self,
    project_id: int,
    well_name: str,
    tolerance: float = 0.16,
    use_cache: bool = True,
):
    """Produce merged well data (same as /database/.../merged) in a worker.

    Returns a list of dict records suitable for JSON serialization.
    """
    try:
        connector = DBConnector(db_url=os.environ.get("QPP_DATABASE_URL"))
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Get main dataframe
            try:
                df = proj.get_well_data_optimized(well_name)
            except AttributeError:
                df = proj.get_well_data(well_name)

            if df.empty:
                return []

            # normalize
            df = df.reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]
            if "depth" not in df.columns:
                candidate = next(
                    (c for c in df.columns if "depth" in c or c in ("md", "tvd")), None
                )
                if candidate:
                    df = df.rename(columns={candidate: "depth"})
            if "depth" not in df.columns:
                raise RuntimeError("No depth column found in well data")
            df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
            df = df.dropna(subset=["depth"]) if not df.empty else df
            df = df.sort_values("depth").reset_index(drop=True)

            # ancillary
            well = proj.get_well(well_name)

            def _normalize(dframe):
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

            tops = _normalize(well.get_formation_tops())
            contacts = _normalize(well.get_fluid_contacts())
            pressure = _normalize(well.get_pressure_tests())

            def _pref(dframe, prefix):
                if dframe is None or dframe.empty:
                    return None
                cols = [c for c in dframe.columns if c != "depth"]
                rename_map = {c: f"{prefix}{c}" for c in cols}
                return (
                    dframe.rename(columns=rename_map)
                    .sort_values("depth")
                    .reset_index(drop=True)
                )

            tops_p = _pref(tops, "top_")
            contacts_p = _pref(contacts, "contact_")
            pressure_p = _pref(pressure, "pressure_")

            merged = df
            if tops_p is not None and not tops_p.empty:
                merged = pd.merge_asof(
                    merged, tops_p, on="depth", direction="nearest", tolerance=tolerance
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
                )
            if pressure_p is not None and not pressure_p.empty:
                merged = pd.merge_asof(
                    merged,
                    pressure_p,
                    on="depth",
                    direction="nearest",
                    tolerance=tolerance,
                )

            merged = merged.replace([np.inf, -np.inf], np.nan)
            merged.columns = [c.upper() for c in merged.columns]

            # serialize
            records = []

            def _to_py(series):
                if pd.api.types.is_numeric_dtype(series):
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

            for col in merged.columns:
                if len(records) == 0:
                    records = [{} for _ in range(len(merged))]
                vals = _to_py(merged[col])
                for i, v in enumerate(vals):
                    records[i][col] = v

            return records
    except Exception as exc:
        logging.getLogger(__name__).exception("process_merged_data failed: %s", exc)
        raise


@celery_app.task(bind=True, name="quick_pp.tasks.generate_well_plot", acks_late=True)
def generate_well_plot(
    self,
    project_id: int,
    well_name: str,
    min_depth: float = None,
    max_depth: float = None,
    zones: str = None,
):
    """Generate Plotly figure JSON for a well in a worker and return parsed JSON object."""
    try:
        connector = DBConnector(db_url=os.environ.get("QPP_DATABASE_URL"))
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            df = proj.get_well_data_optimized(well_name)

            try:
                ancillary = proj.get_well_ancillary_data(well_name)
            except Exception:
                ancillary = {}

            # annotate tops and core similar to web endpoint
            tops_df = None
            if isinstance(ancillary, dict) and "formation_tops" in ancillary:
                tops_df = ancillary.get("formation_tops")
                if (
                    isinstance(tops_df, pd.DataFrame)
                    and not tops_df.empty
                    and not df.empty
                ):
                    df["ZONES"] = pd.NA
                    for _, top in tops_df.iterrows():
                        try:
                            top_depth = float(top.get("depth"))
                            top_name = str(top.get("name"))
                        except Exception:
                            continue
                        nearest_idx = (df["DEPTH"] - top_depth).abs().idxmin()
                        df.at[nearest_idx, "ZONES"] = top_name
                    df["ZONES"] = df["ZONES"].ffill()

            # apply filters
            if zones:
                zone_list = [z.strip() for z in zones.split(",") if z.strip()]
                if "ZONES" in df.columns:
                    df = df[df["ZONES"].isin(zone_list)]
            if min_depth is not None or max_depth is not None:
                depth_col = next(
                    (
                        c
                        for c in [
                            "depth",
                            "DEPTH",
                            "Depth",
                            "TVDSS",
                            "tvdss",
                            "TVD",
                            "tvd",
                        ]
                        if c in df.columns
                    ),
                    None,
                )
                if depth_col is not None:
                    if min_depth is not None:
                        df = df[df[depth_col] >= min_depth]
                    if max_depth is not None:
                        df = df[df[depth_col] <= max_depth]

            if df.empty:
                raise RuntimeError("No data for well")

            fig = wl.plotly_log(df, well_name=well_name)
            parsed = json.loads(fig.to_json())
            return parsed
    except Exception as exc:
        logging.getLogger(__name__).exception("generate_well_plot failed: %s", exc)
        raise
