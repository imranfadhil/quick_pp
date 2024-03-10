from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List
import pandas as pd

from quick_pp.lithology import gr_index

router = APIRouter(prefix="/vsh_gr", tags=["Lithology"])


class data(BaseModel):
    gr: float


class inputData(BaseModel):
    data: List[data]


EXAMPLE = {'data': [
    {'gr': 40},
    {'gr': 60},
    {'gr': 80},
    {'gr': 40},
    {'gr': 60},
    {'gr': 80},
]}


@router.post("")
async def estimate_vsh_gr(inputs: inputData = Body(..., example=EXAMPLE)):

    input_dict = inputs.model_dump()
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    vsh_gr = gr_index(input_df['gr'])
    return_dict = pd.DataFrame({'GR': vsh_gr.ravel()}).to_dict(orient='records')
    return return_dict
