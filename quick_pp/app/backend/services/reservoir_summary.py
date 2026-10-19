from typing import Any, Dict, List

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException

from quick_pp.app.backend.schemas.ressum import InputData, UncertaintyInputData
from quick_pp.ressum import calc_reservoir_summary
from quick_pp.uncertainty import calc_stochastic_reservoir_summary

router = APIRouter(prefix="/ressum", tags=["Reservoir Summary"])


def _clean_json_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    df_clean = df.replace([np.inf, -np.inf], np.nan)
    records = df_clean.to_dict(orient="records")
    out = []
    for r in records:
        out.append(
            {
                k: (None if isinstance(v, float) and np.isnan(v) else v)
                for k, v in r.items()
            }
        )
    return out


@router.post(
    "",
    summary="Calculate Reservoir Summary",
    description=(
        """
        Calculate reservoir summary statistics for respective zones based on input petrophysical data.
        This includes calculating average porosity, average water saturation, average permeability,
        and other relevant metrics for the specified zones.

        Input model: InputData (see quick_pp.app.backend.schemas.ressum.InputData)

        Request body must be a JSON object with the following fields:
        - data: list of objects, each with keys:
            - depth: float (required)
            - vclay: float (volume of clay, required)
            - phit: float (total porosity, required)
            - swt: float (water saturation, required)
            - perm: float (permeability, required)
            - zones: string or int (zone identifier, required)
        - cut_offs: dict with cutoff parameters for filtering, e.g.:
            - PHIT: float (optional, minimum porosity)
            - SWT: float (optional, maximum water saturation)
            - VSHALE: float (optional, minimum volume of shale)
            - ... (other cutoffs as needed)

        Example:
        {
            "data": [
                {"depth": 1000.0, "vclay": 0.25, "phit": 0.18, "swt": 0.35, "perm": 120.0, "zones": "A"},
                {"depth": 1001.0, "vclay": 0.22, "phit": 0.20, "swt": 0.30, "perm": 150.0, "zones": "A"},
                {"depth": 1020.0, "vclay": 0.30, "phit": 0.15, "swt": 0.40, "perm": 90.0, "zones": "B"}
            ],
            "cut_offs": {
                "VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9
            }
        }
        """
    ),
    operation_id="calculate_reservoir_summary",
    response_model=List[Dict],
)
async def calculate_reservoir_summary_(inputs: InputData):
    """
    Calculates reservoir summary statistics based on input petrophysical data.
    Args:
        inputs (InputData): Input data model containing petrophysical measurements and cutoff parameters.
    Returns:
        List[dict]: A list of dictionaries, each representing a row of the computed reservoir summary.
    Raises:
        HTTPException: If input data is invalid or calculation fails.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict["data"])
        ressum_df = calc_reservoir_summary(
            depth=input_df["depth"],
            vshale=input_df["vclay"],
            phit=input_df["phit"],
            swt=input_df["swt"],
            perm=input_df["perm"],
            zones=input_df["zones"],
            cutoffs=input_dict["cut_offs"],
        )
        ressum_df.dropna(inplace=True)
        return ressum_df.to_dict(orient="records")
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error calculating reservoir summary: {e}"
        ) from e


@router.post(
    "/uncertainty",
    summary="Calculate Stochastic Reservoir Summary & LHS Log Uncertainty (P10/P50/P90)",
    operation_id="calculate_stochastic_reservoir_summary",
)
async def calculate_stochastic_reservoir_summary_(inputs: UncertaintyInputData):
    """Run Latin Hypercube petrophysical & volumetric uncertainty (Carbonate or SandSiltClay + Waxman-Smits/Archie)
    and return P10/P50/P90 CPI curves (`SW_P10`, `SW_P50`, `SW_P90`, `PHIT_P10`, `PHIT_P50`, `PHIT_P90`, etc.),
    stochastic reservoir summary table, and zone-level P10/P50/P90 STOIIP volumes.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict["data"])
        if input_df.empty:
            raise ValueError("No log data provided for uncertainty calculation.")

        res = calc_stochastic_reservoir_summary(
            df=input_df,
            cutoffs=input_dict.get("cut_offs"),
            model=input_dict.get("model", "ssc"),
            saturation_method=input_dict.get("saturation_method", "waxman_smits"),
            n_samples=int(input_dict.get("n_samples", 100)),
            param_bounds=input_dict.get("param_bounds"),
            area_bound=tuple(input_dict.get("area_bound", [500.0, 2500.0, 1500.0, 400.0])),
            volume_factor_bound=tuple(input_dict.get("volume_factor_bound", [1.1, 1.4])),
            seed=input_dict.get("seed", 42),
        )

        cpi_df = res["cpi_df"]
        cpi_cols = [
            c
            for c in [
                "depth",
                "DEPTH",
                "SW_P10",
                "SW_P50",
                "SW_P90",
                "SWT_P10",
                "SWT_P50",
                "SWT_P90",
                "PHIT_P10",
                "PHIT_P50",
                "PHIT_P90",
                "PHIE_P10",
                "PHIE_P50",
                "PHIE_P90",
                "VCLAY_P10",
                "VCLAY_P50",
                "VCLAY_P90",
                "BVHC_P10",
                "BVHC_P50",
                "BVHC_P90",
            ]
            if c in cpi_df.columns
        ]
        cpi_export = cpi_df[cpi_cols].copy()
        if "depth" in cpi_export.columns and "DEPTH" not in cpi_export.columns:
            cpi_export = cpi_export.rename(columns={"depth": "DEPTH"})

        return {
            "summary": _clean_json_records(res["summary_df"]),
            "cpi_curves": _clean_json_records(cpi_export.round(4)),
            "volumetrics": res["volumetrics"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error calculating stochastic reservoir summary: {e}",
        ) from e

