from fastapi import APIRouter, Body
import pandas as pd

from api.schemas.ressum import inputData, EXAMPLE

from quick_pp.ressum import calc_reservoir_summary

router = APIRouter(prefix="/ressum", tags=["Reservoir Summary"])


@router.post("", description="Estimate density porosity based on neutron porosity and bulk density.")
async def calculate_reservoir_summary_(inputs: inputData = Body(..., example=EXAMPLE)):

    input_dict = inputs.model_dump()

    # Get the data
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
