import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from quick_pp.core_analysis import fit_poroperm_curve
from quick_pp.database import objects as db_objects

from . import database


router = APIRouter(prefix="/database/projects/{project_id}", tags=["Perm Transform"])


@router.get("/poroperm_fits", summary="Get poro-perm fits per rock flag")
async def get_poroperm_fits(project_id: int):
    """Return fitted a and b parameters for poro-perm curves per ROCK_FLAG.

    Args:
        project_id: Project ID

    Returns:
        JSON with fits per rock_flag
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

            # Collect all data with rock_flags
            all_phit = []
            all_perm = []
            all_rock_flags = []

            for well_name in all_well_names:
                df = proj.get_well_data_optimized(well_name)

                if df.empty:
                    continue

                # Extract columns
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

                if not phit_col or not perm_col or not rock_flag_col:
                    continue

                cpore = df[phit_col].dropna()
                cperm = df[perm_col].dropna()
                crock = df[rock_flag_col].dropna()

                # Align
                common_index = cpore.index.intersection(cperm.index).intersection(
                    crock.index
                )
                cpore = cpore.loc[common_index]
                cperm = cperm.loc[common_index]
                crock = crock.loc[common_index]

                if len(cpore) == 0:
                    continue

                all_phit.extend(cpore.tolist())
                all_perm.extend(cperm.tolist())
                all_rock_flags.extend(crock.tolist())

            if not all_phit:
                raise HTTPException(
                    status_code=404, detail="No data with ROCK_FLAG found"
                )

            # Group by rock_flag
            from collections import defaultdict

            groups = defaultdict(list)
            for phit, perm, rf in zip(all_phit, all_perm, all_rock_flags, strict=True):
                groups[rf].append((phit, perm))

            # Fit for each group
            fits = {}
            for rf, data in groups.items():
                poro = [d[0] for d in data]
                perm = [d[1] for d in data]
                try:
                    a, b = fit_poroperm_curve(np.array(poro), np.array(perm))
                    fits[str(rf)] = {"a": a, "b": b}
                except Exception as e:
                    print(f"Failed to fit for ROCK_FLAG {rf}: {e}")
                    fits[str(rf)] = {"a": 1, "b": 1}

            return {"fits": fits}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get poroperm fits: {e}"
        ) from e


@router.post("/save_perm_trans", summary="Save perm transform for project wells")
async def save_perm_trans(project_id: int, payload: dict):
    """Save PERM_TRANS column to wells based on poro-perm fits per ROCK_FLAG.

    Expected payload: { "perm_trans_pairs": [{"well_name": "...", "depth": 123.4, "perm_trans": 1.23}, ...] }

    Each pair contains the well name, depth, and calculated perm_trans.
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "perm_trans_pairs" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'perm_trans_pairs' array"
        )

    perm_trans_pairs = payload.get("perm_trans_pairs")
    if not isinstance(perm_trans_pairs, list):
        raise HTTPException(status_code=400, detail="'perm_trans_pairs' must be a list")

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Group by well
            well_data = {}
            for pair in perm_trans_pairs:
                if not isinstance(pair, dict) or not all(
                    k in pair for k in ["well_name", "depth", "perm_trans"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Each perm_trans_pair must be a dict with 'well_name', 'depth', and 'perm_trans' keys",
                    )

                well_name = pair["well_name"]
                depth = pair["depth"]
                perm_trans = pair["perm_trans"]

                if well_name not in well_data:
                    well_data[well_name] = []
                well_data[well_name].append({"DEPTH": depth, "PERM_TRANS": perm_trans})

            # Save to each well
            for well_name, data in well_data.items():
                df = pd.DataFrame(data)
                well_obj = proj.get_well(well_name)
                well_obj.update_data(df)
                well_obj.save()

            proj.save()

        return {
            "message": f"Perm transforms saved for {len(well_data)} wells",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save perm transforms: {e}"
        ) from e
