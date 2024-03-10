from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List
import pandas as pd

from quick_pp.lithology import gr_index

router = APIRouter(prefix="/vsh_gr", tags=["lithology"])


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
    input_df = pd.DataFrame.from_records(input_dict['data'])

    gr = input_df['gr']
    vsh_gr = gr_index(gr)
    return_df = pd.DataFrame({'GR': vsh_gr.ravel()})
    return_dict = return_df.to_dict(orient='records')
    return return_dict
