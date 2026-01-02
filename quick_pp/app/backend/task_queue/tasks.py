import json
import logging
import os

import numpy as np
import pandas as pd

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

            # Core data processing with vectorization
            core_raw = well.get_core_data() or {}
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

            merged = merged.replace([np.inf, -np.inf], np.nan)
            merged.columns = [c.upper() for c in merged.columns]

            # Optimized vectorized type conversion
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

            records = []
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


@celery_app.task(bind=True, name="quick_pp.tasks.process_fzi_data", acks_late=True)
def process_fzi_data(self, project_id: int):
    """Compute FZI input arrays (PHIT, PERM, zones, depths, well_names, rock_flags) for a project.

    Returns a dict with keys: phit, perm, zones, well_names, depths, rock_flags
    """
    try:
        connector = DBConnector(db_url=os.environ.get("QPP_DATABASE_URL"))
        with connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            all_well_names = proj.get_well_names()
            if not all_well_names:
                return {
                    "phit": [],
                    "perm": [],
                    "zones": [],
                    "well_names": [],
                    "depths": [],
                    "rock_flags": [],
                }

            well_names = []
            depths_list = []
            cpore_list = []
            cperm_list = []
            zones_list = []
            rock_flags_list = []

            for well_name in all_well_names:
                try:
                    df = proj.get_well_data_optimized(well_name)
                except Exception:
                    try:
                        df = proj.get_well_data(well_name)
                    except Exception:
                        continue

                if df.empty:
                    # no data for this well
                    continue

                # Load ancillary (formation tops) to annotate ZONES
                try:
                    ancillary = proj.get_well_ancillary_data(well_name)
                except Exception:
                    ancillary = {}

                if isinstance(ancillary, dict) and "formation_tops" in ancillary:
                    tops_df = ancillary.get("formation_tops")
                    if (
                        isinstance(tops_df, pd.DataFrame)
                        and not tops_df.empty
                        and not df.empty
                    ):
                        df = df.copy()
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

                # Find PHIT and PERM columns
                phit_col = None
                perm_col = None
                rock_flag_col = None
                for col in df.columns:
                    if col.upper() in ["PHIT", "POROSITY", "POR"]:
                        phit_col = col
                    if col.upper() in ["PERM", "PERMEABILITY", "K"]:
                        perm_col = col
                    if col.upper() == "ROCK_FLAG":
                        rock_flag_col = col

                if not phit_col or not perm_col:
                    continue

                cpore = df[phit_col].dropna()
                cperm = df[perm_col].dropna()

                common_index = cpore.index.intersection(cperm.index)
                cpore = cpore.loc[common_index]
                cperm = cperm.loc[common_index]
                depths = df.loc[common_index, "DEPTH"]

                if "ZONES" in df.columns:
                    zones = df.loc[common_index, "ZONES"].fillna("Unknown").tolist()
                else:
                    zones = ["Unknown"] * len(cpore)

                if rock_flag_col:
                    rock_flags = df.loc[common_index, rock_flag_col]
                    rock_flags = rock_flags.where(pd.notna(rock_flags), None).tolist()
                else:
                    rock_flags = [None] * len(cpore)

                if len(cpore) == 0:
                    continue

                well_names.extend([well_name] * len(cpore))
                depths_list.extend(depths.tolist())
                cpore_list.append(cpore)
                cperm_list.append(cperm)
                zones_list.extend(zones)
                rock_flags_list.extend(rock_flags)

            phit_all = pd.concat(cpore_list).tolist() if cpore_list else []
            perm_all = pd.concat(cperm_list).tolist() if cperm_list else []

            # sanitize lists to built-in types
            phit_all = [
                None
                if (
                    isinstance(x, float)
                    and (pd.isna(x) or x == float("inf") or x == float("-inf"))
                )
                else (x.item() if hasattr(x, "item") else x)
                for x in phit_all
            ]
            perm_all = [
                None
                if (
                    isinstance(x, float)
                    and (pd.isna(x) or x == float("inf") or x == float("-inf"))
                )
                else (x.item() if hasattr(x, "item") else x)
                for x in perm_all
            ]
            depths_list = [
                None
                if (
                    isinstance(x, float)
                    and (pd.isna(x) or x == float("inf") or x == float("-inf"))
                )
                else (x.item() if hasattr(x, "item") else x)
                for x in depths_list
            ]

            # coerce rock flags to ints when possible
            rock_flags_list = [
                int(x)
                if (x is not None and not pd.isna(x) and str(x).strip() != "")
                else None
                for x in rock_flags_list
            ]

            return {
                "phit": phit_all,
                "perm": perm_all,
                "zones": zones_list,
                "well_names": well_names,
                "depths": depths_list,
                "rock_flags": rock_flags_list,
            }
    except Exception as exc:
        logging.getLogger(__name__).exception("process_fzi_data failed: %s", exc)
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

            # Process core data if it exists
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
