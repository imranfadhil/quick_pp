from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List, Dict

from api.schemas.ressum import InputData
from quick_pp.ressum import calc_reservoir_summary

router = APIRouter(prefix="/ressum", tags=["Reservoir Summary"])


@router.post(
    "",
    summary="Calculate Reservoir Summary",
    description="Calculate reservoir summary statistics based on input petrophysical data.",
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
        input_df = pd.DataFrame.from_records(input_dict['data'])
        ressum_df = calc_reservoir_summary(
            depth=input_df['depth'],
            vshale=input_df['vcld'],
            phit=input_df['phit'],
            swt=input_df['swt'],
            perm=input_df['perm'],
            zones=input_df['zones'],
            cutoffs=input_dict['cut_offs']
        )
        ressum_df.dropna(inplace=True)
        return ressum_df.to_dict(orient='records')
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error calculating reservoir summary: {e}")
