import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

import quick_pp.database.objects as db_objects
from quick_pp.app.backend.services.database import (
    _get_merged_dataframe_sync,
    invalidate_well_cache,
)
from quick_pp.app.backend.utils.db import get_db
from quick_pp.app.backend.utils.utils import sanitize_list


router = APIRouter(prefix="/database/projects/{project_id}", tags=["Rock Typing"])


def _compute_fzi_data_sync(
    project_id: int,
    dt_matrix: float = 47.6,
    dt_fluid: float = 189.0,
    data_source: str = "auto",
) -> dict:
    """Helper to compute FZI / rock typing data (including sonic & secondary porosity if available) synchronously.

    data_source:
        - 'core': strictly Core Data (CPORE & CPERM)
        - 'log': strictly Calculated Logs (PHIT & PERM)
        - 'auto': combined fallback across available columns
    """
    src_norm = str(data_source or "auto").strip().lower()
    if src_norm not in ("core", "log", "auto"):
        src_norm = "auto"

    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)
        all_well_names = proj.get_well_names()

    if not all_well_names:
        raise ValueError(f"No wells found for project {project_id}")

    well_names = []
    depths_list = []
    tvd_list = []
    cpore_list = []
    cperm_list = []
    phis_list = []
    phi_sec_list = []
    zones_list = []
    rock_flags_list = []
    sonic_wells = []
    tvd_wells = []
    no_tvd_wells = []
    any_non_empty_well = False
    available_cols = set()
    loaded_wells = []
    any_real_perm = False
    if src_norm == "core":
        poro_candidates = ("CPORE", "CORE_POR")
        perm_candidates = ("CPERM", "CORE_PERM")
    elif src_norm == "log":
        poro_candidates = ("PHIT", "PHIE", "POROSITY", "POR")
        perm_candidates = ("PERM", "KLOGH", "PERMEABILITY", "K", "PERM_TRANS")
    else:
        poro_candidates = ("PHIT", "PHIE", "CPORE", "POROSITY", "POR", "CORE_POR")
        perm_candidates = (
            "PERM",
            "PERM_TRANS",
            "KLOGH",
            "PERMEABILITY",
            "K",
            "CPERM",
            "CORE_PERM",
        )

    for well_name in all_well_names:
        df = _get_merged_dataframe_sync(project_id, well_name)
        if df.empty:
            continue
        any_non_empty_well = True
        available_cols.update(df.columns)
        if any(
            col in df.columns and df[col].notna().any() for col in perm_candidates
        ):
            any_real_perm = True
        loaded_wells.append((well_name, df))

    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)
        for well_name, df in loaded_wells:

            # Load ancillary data (formation tops)
            try:
                ancillary = proj.get_well_ancillary_data(well_name)
            except Exception:
                ancillary = {}

            tops_df = None
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

            phit_col = None
            perm_col = None
            rock_flag_col = None
            for col in poro_candidates:
                if col in df.columns and df[col].notna().any():
                    phit_col = col
                    break
            for col in perm_candidates:
                if col in df.columns and df[col].notna().any():
                    perm_col = col
                    break
            if "ROCK_FLAG" in df.columns:
                rock_flag_col = "ROCK_FLAG"

            # Fallback estimation for raw project logs only when data_source == 'auto'
            if src_norm == "auto":
                if not phit_col:
                    if "RHOB" in df.columns and df["RHOB"].notna().any():
                        rhob_s = pd.to_numeric(df["RHOB"], errors="coerce")
                        est_phit = ((2.65 - rhob_s) / (2.65 - 1.0)).clip(0.001, 0.45)
                        if "NPHI" in df.columns and df["NPHI"].notna().any():
                            nphi_s = pd.to_numeric(df["NPHI"], errors="coerce").clip(
                                0.001, 0.6
                            )
                            est_phit = ((est_phit + nphi_s) / 2.0).clip(0.001, 0.45)
                        df = df.copy()
                        df["_EST_PHIT"] = est_phit
                        phit_col = "_EST_PHIT"
                    elif "NPHI" in df.columns and df["NPHI"].notna().any():
                        df = df.copy()
                        df["_EST_PHIT"] = pd.to_numeric(df["NPHI"], errors="coerce").clip(
                            0.001, 0.45
                        )
                        phit_col = "_EST_PHIT"

                if phit_col and not perm_col and not any_real_perm:
                    phit_s = pd.to_numeric(df[phit_col], errors="coerce").clip(0.001, 0.45)
                    df = df.copy()
                    df["_EST_PERM"] = (10.0 ** (14.0 * phit_s - 1.5)).clip(0.001, 10000.0)
                    perm_col = "_EST_PERM"

            if not phit_col or not perm_col:
                continue

            cpore_s = pd.to_numeric(df[phit_col], errors="coerce")
            cpore_s = cpore_s.where(cpore_s > 0, np.nan)
            for fallback_pcol in poro_candidates:
                if fallback_pcol != phit_col and fallback_pcol in df.columns:
                    fp_s = pd.to_numeric(df[fallback_pcol], errors="coerce")
                    fp_s = fp_s.where(fp_s > 0, np.nan)
                    cpore_s = cpore_s.combine_first(fp_s)
            if cpore_s.notna().any() and float(cpore_s.median()) > 1.0:
                cpore_s = cpore_s / 100.0
            cpore = cpore_s.dropna()

            cperm_s = pd.to_numeric(df[perm_col], errors="coerce")
            cperm_s = cperm_s.where(cperm_s > 0, np.nan)
            for fallback_kcol in perm_candidates:
                if fallback_kcol != perm_col and fallback_kcol in df.columns:
                    fk_s = pd.to_numeric(df[fallback_kcol], errors="coerce")
                    fk_s = fk_s.where(fk_s > 0, np.nan)
                    cperm_s = cperm_s.combine_first(fk_s)
            cperm = cperm_s.dropna()

            common_index = cpore.index.intersection(cperm.index)
            cpore = cpore.loc[common_index]
            cperm = cperm.loc[common_index]
            depths = df.loc[common_index, "DEPTH"]

            if len(cpore) == 0:
                continue

            # Resolve sonic porosity (PHIS) and secondary/vuggy porosity (PHI_SEC = PHIT - PHIS) if available
            phis_s = pd.Series(np.nan, index=common_index, dtype=float)
            phi_sec_s = pd.Series(np.nan, index=common_index, dtype=float)
            well_has_sonic = False

            sec_col = next(
                (
                    c
                    for c in ("PHI_SEC", "PHIS_SEC", "SPI", "VUG_POR")
                    if c in df.columns and df[c].notna().any()
                ),
                None,
            )
            phis_col = next(
                (
                    c
                    for c in ("PHIS", "PHI_SONIC", "PHISONIC", "SPOR")
                    if c in df.columns and df[c].notna().any()
                ),
                None,
            )
            dt_col = next(
                (
                    c
                    for c in ("DT", "DTC", "AC", "DTCO", "DT4P")
                    if c in df.columns and df[c].notna().any()
                ),
                None,
            )

            if sec_col is not None:
                sec_raw = pd.to_numeric(df.loc[common_index, sec_col], errors="coerce")
                valid_sec = sec_raw.notna()
                if valid_sec.any():
                    phi_sec_s.loc[valid_sec] = np.clip(
                        sec_raw.loc[valid_sec], 0.0, cpore.loc[valid_sec]
                    )
                    phis_s.loc[valid_sec] = np.clip(
                        cpore.loc[valid_sec] - phi_sec_s.loc[valid_sec], 0.0, 0.99
                    )
                    well_has_sonic = True
            elif phis_col is not None:
                ps_raw = pd.to_numeric(df.loc[common_index, phis_col], errors="coerce")
                valid_ps = ps_raw.notna()
                if valid_ps.any():
                    phis_s.loc[valid_ps] = np.clip(ps_raw.loc[valid_ps], 0.0, 0.99)
                    phi_sec_s.loc[valid_ps] = np.maximum(
                        0.0, cpore.loc[valid_ps] - phis_s.loc[valid_ps]
                    )
                    well_has_sonic = True
            elif dt_col is not None:
                dt_raw = pd.to_numeric(df.loc[common_index, dt_col], errors="coerce")
                valid_dt = dt_raw.notna() & (dt_raw > 0)
                if valid_dt.any():
                    dt_vals = dt_raw.loc[valid_dt].copy()
                    # Convert metric us/m to imperial us/ft if median > 135
                    if float(dt_vals.median()) > 135.0:
                        dt_vals = dt_vals / 3.28084
                    denom = max(float(dt_fluid) - float(dt_matrix), 1.0)
                    ps_calc = ((dt_vals - float(dt_matrix)) / denom).clip(0.001, 0.60)
                    phis_s.loc[valid_dt] = ps_calc
                    phi_sec_s.loc[valid_dt] = np.maximum(
                        0.0, cpore.loc[valid_dt] - ps_calc
                    )
                    well_has_sonic = True

            if well_has_sonic:
                sonic_wells.append(well_name)

            # Resolve true TVD if available (never assume TVD == DEPTH/MD)
            tvd_col = next(
                (
                    c
                    for c in ("TVD", "TVDSS", "DEPT_TVD", "DEPTH_TVD")
                    if c in df.columns and pd.to_numeric(df[c], errors="coerce").notna().sum() >= 2
                ),
                None,
            )
            if tvd_col is not None:
                tvd_s = pd.to_numeric(df.loc[common_index, tvd_col], errors="coerce")
                if tvd_s.notna().any():
                    tvd_wells.append(well_name)
                else:
                    no_tvd_wells.append(well_name)
            else:
                tvd_s = pd.Series(np.nan, index=common_index, dtype=float)
                no_tvd_wells.append(well_name)

            if "ZONES" in df.columns:
                zones = df.loc[common_index, "ZONES"].fillna("Unknown").tolist()
            else:
                zones = ["Unknown"] * len(cpore)

            if rock_flag_col:
                rf_s = pd.to_numeric(
                    df.loc[common_index, rock_flag_col], errors="coerce"
                )
                rf_s = rf_s.where(rf_s >= 1, np.nan)
                if "ROCK_FLAG_ML" in df.columns:
                    ml_s = pd.to_numeric(
                        df.loc[common_index, "ROCK_FLAG_ML"], errors="coerce"
                    )
                    ml_s = ml_s.where(ml_s >= 1, np.nan)
                    rf_s = rf_s.combine_first(ml_s)
                rock_flags = rf_s.where(pd.notna(rf_s), None).tolist()
            elif "ROCK_FLAG_ML" in df.columns:
                rf_s = pd.to_numeric(
                    df.loc[common_index, "ROCK_FLAG_ML"], errors="coerce"
                )
                rf_s = rf_s.where(rf_s >= 1, np.nan)
                rock_flags = rf_s.where(pd.notna(rf_s), None).tolist()
            else:
                rock_flags = [None] * len(cpore)

            well_names.extend([well_name] * len(cpore))
            depths_list.extend(depths.tolist())
            tvd_list.append(tvd_s)
            cpore_list.append(cpore)
            cperm_list.append(cperm)
            phis_list.append(phis_s)
            phi_sec_list.append(phi_sec_s)
            zones_list.extend(zones)
            rock_flags_list.extend(rock_flags)

        if not any_non_empty_well:
            raise ValueError(f"No data for project {project_id}")

        if not cpore_list or not cperm_list:
            if src_norm == "core":
                raise ValueError(
                    "No Core Data (CPORE & CPERM) pairs found in the selected project wells. "
                    "Switch Input Data Source to 'Calculated Logs (PHIT & PERM)' or import RCA core data."
                )
            if src_norm == "log":
                raise ValueError(
                    "No Calculated Log (PHIT & PERM) pairs found in the selected project wells. "
                    "Compute and save Porosity (PHIT) and Permeability (PERM) first, or switch Input Data Source to 'Core Data (CPORE & CPERM)'."
                )
            raise ValueError(
                f"PHIT or PERM columns not found in project data. Available: {sorted(available_cols)}",
            )

        phit_all = pd.concat(cpore_list).tolist() if cpore_list else []
        perm_all = pd.concat(cperm_list).tolist() if cperm_list else []
        phis_all = pd.concat(phis_list).tolist() if phis_list else []
        phi_sec_all = pd.concat(phi_sec_list).tolist() if phi_sec_list else []
        tvd_all = pd.concat(tvd_list).tolist() if tvd_list else []

        phit_all = sanitize_list(phit_all)
        perm_all = sanitize_list(perm_all)
        phis_all = sanitize_list(phis_all)
        phi_sec_all = sanitize_list(phi_sec_all)
        tvd_all = sanitize_list(tvd_all)
        depths_list = sanitize_list(depths_list)
        rock_flags_list = [
            (
                int(x)
                if (not pd.isna(x) and x is not None and str(x).strip() != "")
                else None
            )
            if (x is not None)
            else None
            for x in rock_flags_list
        ]

        return {
            "data_source": src_norm,
            "poro_label": "CPORE" if src_norm == "core" else "PHIT",
            "perm_label": "CPERM" if src_norm == "core" else "PERM",
            "phit": phit_all,
            "perm": perm_all,
            "phis": phis_all,
            "phi_sec": phi_sec_all,
            "has_sonic": len(sonic_wells) > 0,
            "sonic_wells": sonic_wells,
            "tvd": tvd_all,
            "tvd_wells": tvd_wells,
            "no_tvd_wells": no_tvd_wells,
            "zones": zones_list,
            "well_names": well_names,
            "depths": depths_list,
            "rock_flags": rock_flags_list,
        }


@router.get("/fzi_data", summary="Get FZI data for plotting")
async def get_fzi_data(project_id: int, data_source: str = "auto") -> dict:
    """Return PHIT/CPORE and PERM/CPERM data for FZI plotting from all wells in the project."""
    try:
        return _compute_fzi_data_sync(project_id, data_source=data_source)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get FZI data: {e}"
        ) from e


@router.post(
    "/fzi_data/generate",
    summary="Compute FZI data synchronously",
)
async def initiate_fzi_data_generation(
    project_id: int, async_task: bool = False
) -> dict:
    """Compute FZI arrays synchronously."""
    try:
        result = _compute_fzi_data_sync(project_id)
        return {"task_id": "sync", "status": "success", "result": result}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to initiate FZI generation: {e}"
        ) from e


@router.post("/save_rock_flags", summary="Save rock flags for project wells")
async def save_rock_flags(project_id: int, payload: dict):
    """Save ROCK_FLAG column to wells based on FZI rock typing.

    Expected payload: { "rock_flag_pairs": [{"well_name": "...", "depth": 123.4, "rock_flag": 1}, ...], "cutoffs": "0.1,1.0,3.0" }

    Each pair contains the well name, depth, and assigned rock flag.
    """
    if not isinstance(payload, dict) or "rock_flag_pairs" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'rock_flag_pairs' array"
        )

    rock_flag_pairs = payload.get("rock_flag_pairs")
    if not isinstance(rock_flag_pairs, list):
        raise HTTPException(status_code=400, detail="'rock_flag_pairs' must be a list")

    cutoffs_str = payload.get("cutoffs", "")

    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Group by well
            well_data = {}
            for pair in rock_flag_pairs:
                if not isinstance(pair, dict) or not all(
                    k in pair for k in ["well_name", "depth", "rock_flag"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Each rock_flag_pair must be a dict with 'well_name', 'depth', and 'rock_flag' keys",
                    )

                well_name = pair["well_name"]
                depth = pair["depth"]
                rock_flag = pair["rock_flag"]

                if well_name not in well_data:
                    well_data[well_name] = []
                well_data[well_name].append({"DEPTH": depth, "ROCK_FLAG": rock_flag})

            # Save to each well
            for well_name, data in well_data.items():
                df = pd.DataFrame(data)
                well_obj = proj.get_well(well_name)
                well_obj.update_data(df)
                well_obj.save()
                invalidate_well_cache(project_id=project_id, well_name=well_name)

            proj.save()

        return {
            "message": f"Rock flags saved for {len(well_data)} wells",
            "cutoffs": cutoffs_str,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save rock flags: {e}"
        ) from e
