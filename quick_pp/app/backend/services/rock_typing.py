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


def _compute_fzi_data_sync(project_id: int) -> dict:
    """Helper to compute FZI data synchronously."""
    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)

        all_well_names = proj.get_well_names()
        if not all_well_names:
            raise ValueError(f"No wells found for project {project_id}")

        well_names = []
        depths_list = []
        cpore_list = []
        cperm_list = []
        zones_list = []
        rock_flags_list = []
        any_non_empty_well = False
        available_cols = set()

        for well_name in all_well_names:
            df = _get_merged_dataframe_sync(project_id, well_name)
            if df.empty:
                continue
            any_non_empty_well = True
            available_cols.update(df.columns)

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
            for col in ("PHIT", "PHIE", "CPORE", "POROSITY", "POR"):
                if col in df.columns and df[col].notna().any():
                    phit_col = col
                    break
            for col in ("PERM", "CPERM", "KLOGH", "PERMEABILITY", "K"):
                if col in df.columns and df[col].notna().any():
                    perm_col = col
                    break
            if "ROCK_FLAG" in df.columns:
                rock_flag_col = "ROCK_FLAG"

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

        if not any_non_empty_well:
            raise ValueError(f"No data for project {project_id}")

        if not cpore_list or not cperm_list:
            raise ValueError(
                f"PHIT or PERM columns not found in project data. Available: {sorted(available_cols)}",
            )

        phit_all = pd.concat(cpore_list).tolist() if cpore_list else []
        perm_all = pd.concat(cperm_list).tolist() if cperm_list else []

        phit_all = sanitize_list(phit_all)
        perm_all = sanitize_list(perm_all)
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
            "phit": phit_all,
            "perm": perm_all,
            "zones": zones_list,
            "well_names": well_names,
            "depths": depths_list,
            "rock_flags": rock_flags_list,
        }


@router.get("/fzi_data", summary="Get FZI data for plotting")
async def get_fzi_data(project_id: int) -> dict:
    """Return PHIT and PERM data for FZI plotting from all wells in the project."""
    try:
        return _compute_fzi_data_sync(project_id)
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
