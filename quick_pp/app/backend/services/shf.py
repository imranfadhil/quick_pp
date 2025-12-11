import math

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Query

from quick_pp.core_analysis import (
    auto_j_params,
    leverett_j,
    normalize_sw,
    sw_shf_leverett_j,
)
from quick_pp.database import db_objects
from quick_pp.rock_type import calc_fzi, rock_typing

from . import database, ancillary


router = APIRouter(
    prefix="/database/projects/{project_id}", tags=["Saturation Height Function"]
)


@router.get("/j_data", summary="Get J data for plotting")
async def get_j_data(project_id: int, cutoffs: str = Query("0.1,1.0,3.0")):
    """Return PC, SW, PERM, PHIT data for J plotting from all wells in the project.

    Args:
        project_id: Project ID
        cutoffs: Optional comma-separated cutoffs for FZI values to define rock flags, e.g., "0.1,1.0,3.0"

    Returns:
        JSON with pc, sw, perm, phit arrays
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
            pc_list = []
            sw_list = []
            perm_list = []
            phit_list = []
            zones_list = []
            rock_flags_list = []
            for well_name in all_well_names:
                try:
                    ancillary = proj.get_well_ancillary_data(well_name)
                except Exception:
                    continue

                if not isinstance(ancillary, dict) or "core_data" not in ancillary:
                    continue

                core_data = ancillary["core_data"]
                if not isinstance(core_data, dict):
                    continue

                for _, sample in core_data.items():
                    depth = sample.get("depth")
                    if depth is None:
                        continue

                    measurements = sample.get("measurements")
                    pc_data = sample.get("pc")

                    if (
                        measurements is None
                        or pc_data is None
                        or measurements.empty
                        or pc_data.empty
                    ):
                        continue

                    # Extract perm and phit from measurements (long format)
                    perm_row = measurements[
                        measurements["property"].str.upper() == "CPERM"
                    ]
                    phit_row = measurements[
                        measurements["property"].str.upper() == "CPORE"
                    ]
                    if perm_row.empty or phit_row.empty:
                        continue
                    perm = perm_row["value"].iloc[0]
                    phit = phit_row["value"].iloc[0]

                    # Find columns in pc_data
                    pc_col = None
                    sw_col = None
                    for col in pc_data.columns:
                        if col.upper() in ["PRESSURE", "PC", "PC_RES"]:
                            pc_col = col
                        if col.upper() in ["SATURATION", "SW", "SWN"]:
                            sw_col = col

                    if not pc_col or not sw_col:
                        continue

                    pc_vals = pc_data[pc_col].dropna()
                    sw_vals = pc_data[sw_col].dropna()

                    common_index = pc_vals.index.intersection(sw_vals.index)
                    pc_vals = pc_vals.loc[common_index]
                    sw_vals = sw_vals.loc[common_index]

                    for i in range(len(pc_vals)):
                        pc_list.append(pc_vals.iloc[i])
                        sw_list.append(sw_vals.iloc[i])
                        perm_list.append(perm)
                        phit_list.append(phit)
                        well_names.append(well_name)
                        depths_list.append(depth)
                        zones_list.append("Unknown")
                        rock_flags_list.append(None)

            if cutoffs:
                cutoff_values = [float(x.strip()) for x in cutoffs.split(",")]
                for i in range(len(pc_list)):
                    # Calculate FZI: FZI = sqrt(perm / phit) / (phit / (1 - phit))
                    if phit_list[i] > 0 and phit_list[i] < 1 and perm_list[i] > 0:
                        fzi = calc_fzi(phit_list[i], perm_list[i])
                        rock_flags_list[i] = rock_typing(fzi, cutoff_values).tolist()
                    else:
                        rock_flags_list[i] = None

            if not pc_list:
                raise HTTPException(
                    status_code=404,
                    detail="No RCA/SCAL data found with PC, SW, PERM, PHIT",
                )

            # Sanitize lists for JSON serialization
            pc_list = ancillary._sanitize_list(pc_list)
            sw_list = ancillary._sanitize_list(sw_list)
            perm_list = ancillary._sanitize_list(perm_list)
            phit_list = ancillary._sanitize_list(phit_list)
            rock_flags_list = ancillary._sanitize_list(rock_flags_list)
            return {
                "pc": pc_list,
                "sw": sw_list,
                "perm": perm_list,
                "phit": phit_list,
                "zones": zones_list,
                "well_names": well_names,
                "depths": depths_list,
                "rock_flags": rock_flags_list,
            }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get J data: {e}") from e


@router.post(
    "/compute_j_fits",
    summary="Compute J fits per rock flag",
    tags=["Saturation Height Function"],
)
async def compute_j_fits(payload: dict):
    """Compute fitted a and b parameters for J curves per ROCK_FLAG.

    Expected payload: { "data": { "pc": [...], "sw": [...], "perm": [...], "phit": [...], "rock_flags": [...] }, "ift": 30, "theta": 30 }

    Returns:
        JSON with fits per rock_flag
    """
    if not isinstance(payload, dict) or "data" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'data' object"
        )

    data = payload.get("data")
    ift = payload.get("ift", 30)
    theta = payload.get("theta", 30)

    try:
        pc = data.get("pc", [])
        sw = data.get("sw", [])
        perm = data.get("perm", [])
        phit = data.get("phit", [])
        rock_flags = data.get("rock_flags", [])

        if (
            len(pc) != len(sw)
            or len(pc) != len(perm)
            or len(pc) != len(phit)
            or len(pc) != len(rock_flags)
        ):
            raise HTTPException(
                status_code=400, detail="Data arrays must have the same length"
            )

        # Calculate J
        j_values = []
        for i in range(len(pc)):
            j = leverett_j(pc[i], ift, theta, perm[i], phit[i])
            j_values.append(j)

        # Normalize SW to SWN
        swn_values = normalize_sw(pd.Series(sw))

        # Prepare core_data
        core_data_list = []
        for i in range(len(pc)):
            if rock_flags[i] is not None:
                core_data_list.append(
                    {
                        "ROCK_FLAG": rock_flags[i],
                        "SWN": swn_values[i],
                        "J": j_values[i],
                        "Well": f"Well_{i}",
                        "Sample": f"Sample_{i}",
                    }
                )

        core_data = pd.DataFrame(core_data_list)
        j_params = auto_j_params(core_data)

        # Convert to dict
        fits = {}
        for param in j_params:
            rf = str(param["ROCK_FLAG"])
            fits[rf] = {"a": param["a"], "b": param["b"], "rmse": param["rmse"]}

        return fits

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to compute J fits: {e}"
        ) from e


@router.post("/compute_shf", summary="Compute SHF for project data")
async def compute_shf(payload: dict):
    """Compute SHF values using Leverett J function.

    Expected payload: { "data": { "pc": [...], "sw": [...], "perm": [...], "phit": [...], "depths": [...], "rock_flags": [...], "well_names": [...] }, "fits": {...}, "fwl": 1000, "ift": 30, "theta": 30, "gw": 1.05, "ghc": 0.8 }

    Returns:
        JSON with shf_data
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "data" not in payload or "fits" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'data' and 'fits' objects"
        )

    data = payload.get("data")
    fits = payload.get("fits")
    fwl = payload.get("fwl", 1000)
    ift = payload.get("ift", 30)
    theta = payload.get("theta", 30)
    gw = payload.get("gw", 1.05)
    ghc = payload.get("ghc", 0.8)

    try:
        perm = data.get("perm", [])
        phit = data.get("phit", [])
        depths = data.get("depths", [])
        rock_flags = data.get("rock_flags", [])
        well_names = data.get("well_names", [])

        shf_data = []
        for i in range(len(phit)):
            rf = (
                str(int(rock_flags[i]))
                if rock_flags and rock_flags[i] is not None
                else None
            )
            if rf and rf in fits.keys():
                a = fits[rf]["a"]
                b = fits[rf]["b"]
                shf = sw_shf_leverett_j(
                    perm[i], phit[i], depths[i], fwl, ift, theta, gw, ghc, a, b
                )
                if math.isfinite(shf):
                    shf_data.append(
                        {"well": well_names[i], "depth": depths[i], "shf": shf}
                    )
                else:
                    shf_data.append(
                        {"well": well_names[i], "depth": depths[i], "shf": None}
                    )

        return {"shf_data": shf_data}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to compute SHF: {e}"
        ) from e


@router.post(
    "/save_shf",
    summary="Save SHF for project wells",
    tags=["Saturation Height Function"],
)
async def save_shf(project_id: int, payload: dict):
    """Save SHF column to wells.

    Expected payload: { "shf_data": [{"well": "...", "depth": 123.4, "shf": 0.8}, ...] }

    Each item contains the well name, depth, and SHF value.
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "shf_data" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'shf_data' array"
        )

    shf_data = payload.get("shf_data")
    if not isinstance(shf_data, list):
        raise HTTPException(status_code=400, detail="'shf_data' must be a list")

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Group by well
            well_data = {}
            for item in shf_data:
                if not isinstance(item, dict) or not all(
                    k in item for k in ["well", "depth", "shf"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Each shf_data item must be a dict with 'well', 'depth', and 'shf' keys",
                    )

                well_name = item["well"]
                depth = item["depth"]
                shf = item["shf"]

                if well_name not in well_data:
                    well_data[well_name] = []
                well_data[well_name].append({"DEPTH": depth, "SHF": shf})

            # Save to each well
            for well_name, data in well_data.items():
                df = pd.DataFrame(data)
                well_obj = proj.get_well(well_name)
                well_obj.update_data(df)
                well_obj.save()

            proj.save()

        return {
            "message": f"SHF saved for {len(well_data)} wells",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save SHF: {e}") from e
