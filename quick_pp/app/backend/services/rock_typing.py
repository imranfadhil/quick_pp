import pandas as pd
from fastapi import APIRouter, Body, File, HTTPException, Query, UploadFile

from quick_pp.app.backend.services.ancillary import _sanitize_list
from quick_pp.database import db_objects

from . import database

router = APIRouter(prefix="/database/projects/{project_id}", tags=["Rock Typing"])


@router.get("/fzi_data", summary="Get FZI data for plotting")
async def get_fzi_data(project_id: int):
    """Return PHIT and PERM data for FZI plotting from all wells in the project.

    Args:
        project_id: Project ID

    Returns:
        JSON with phit and perm arrays
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            all_well_names = proj.get_well_names()
            if not all_well_names:
                raise HTTPException(
                    status_code=404, detail=f"No wells found for project {project_id}"
                )
            well_names = []
            depths_list = []
            cpore_list = []
            cperm_list = []
            zones_list = []
            rock_flags_list = []
            for well_name in all_well_names:
                df = proj.get_well_data_optimized(well_name)

                if df.empty:
                    raise HTTPException(
                        status_code=404, detail=f"No data for project {project_id}"
                    )

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

                # Extract PHIT and PERM columns
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
                    continue  # skip this well if columns not found

                cpore = df[phit_col].dropna()
                cperm = df[perm_col].dropna()

                # Align the data
                common_index = cpore.index.intersection(cperm.index)
                cpore = cpore.loc[common_index]
                cperm = cperm.loc[common_index]
                depths = df.loc[common_index, "DEPTH"]

                # Extract zones if available
                zones = []
                if "ZONES" in df.columns:
                    zones = df.loc[common_index, "ZONES"].fillna("Unknown").tolist()
                else:
                    zones = ["Unknown"] * len(cpore)

                # Extract rock_flag if available
                rock_flags = []
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

            if not cpore_list or not cperm_list:
                raise HTTPException(
                    status_code=404,
                    detail=f"PHIT or PERM columns not found in project data. Available: {list(df.columns)}",
                )

            # Concatenate all data
            phit_all = pd.concat(cpore_list).tolist() if cpore_list else []
            perm_all = pd.concat(cperm_list).tolist() if cperm_list else []

            phit_all = _sanitize_list(phit_all)
            perm_all = _sanitize_list(perm_all)
            depths_list = _sanitize_list(depths_list)
            # rock flags may be numeric or None; coerce ints where possible
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

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        # Preserve explicit HTTPExceptions raised inside the handler (e.g., 404s)
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get FZI data: {e}"
        ) from e


@router.post("/save_rock_flags", summary="Save rock flags for project wells")
async def save_rock_flags(project_id: int, payload: dict):
    """Save ROCK_FLAG column to wells based on FZI rock typing.

    Expected payload: { "rock_flag_pairs": [{"well_name": "...", "depth": 123.4, "rock_flag": 1}, ...], "cutoffs": "0.1,1.0,3.0" }

    Each pair contains the well name, depth, and assigned rock flag.
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "rock_flag_pairs" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'rock_flag_pairs' array"
        )

    rock_flag_pairs = payload.get("rock_flag_pairs")
    if not isinstance(rock_flag_pairs, list):
        raise HTTPException(status_code=400, detail="'rock_flag_pairs' must be a list")

    cutoffs_str = payload.get("cutoffs", "")

    try:
        with database.connector.get_session() as session:
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
