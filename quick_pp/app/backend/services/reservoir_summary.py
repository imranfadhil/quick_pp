from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List, Dict

from quick_pp.app.backend.schemas.ressum import InputData
from quick_pp.ressum import calc_reservoir_summary

router = APIRouter(prefix="/ressum", tags=["Reservoir Summary"])


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
            - vcld: float (volume of clay, required)
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
                {"depth": 1000.0, "vcld": 0.25, "phit": 0.18, "swt": 0.35, "perm": 120.0, "zones": "A"},
                {"depth": 1001.0, "vcld": 0.22, "phit": 0.20, "swt": 0.30, "perm": 150.0, "zones": "A"},
                {"depth": 1020.0, "vcld": 0.30, "phit": 0.15, "swt": 0.40, "perm": 90.0, "zones": "B"}
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
            vshale=input_df["vcld"],
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
        )
