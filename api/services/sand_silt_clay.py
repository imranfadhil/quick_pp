from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List
import pandas as pd

from quick_pp.lithology import SandSiltClay

router = APIRouter(prefix="/lithology", tags=["lithology"])


class data(BaseModel):
    nphi: float
    rhob: float


class inputData(BaseModel):
    data: List[data]


EXAMPLE = {'data': [
    {'nphi': 0.3, 'rhob': 1.85},
    {'nphi': 0.35, 'rhob': 1.95},
    {'nphi': 0.34, 'rhob': 1.9},
]}


@router.post("")
async def estimate_ssc(inputs: inputData = Body(..., example=EXAMPLE)):

    input_dict = inputs.model_dump()
    input_df = pd.DataFrame.from_records(input_dict['data'])

    nphi = input_df['nphi']
    rhob = input_df['rhob']

    ssc_model = SandSiltClay()
    vsand, vsilt, vcld, vclb, _ = ssc_model.estimate_lithology(nphi, rhob, model='kuttan_modified')
    return_df = pd.DataFrame(
        {'VSAND': vsand, 'VSILT': vsilt, 'VCLW': vcld + vclb, 'VCLB': vclb, 'VCLD': vcld},
        index=input_df.index
    )
    return_dict = return_df.to_dict(orient='records')
    return return_dict
