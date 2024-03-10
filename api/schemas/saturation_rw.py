from pydantic import BaseModel
from typing import List


class data(BaseModel):
    temp_grad: float


class rw_inputData(BaseModel):
    data: List[data]
    water_salinity: float


RW_EXAMPLE = {
    'water_salinity': 30000.0,
    'data': [
        {
            "temp_grad": 134.46224999999998
        },
        {
            "temp_grad": 134.46625
        },
        {
            "temp_grad": 134.47
        },
        {
            "temp_grad": 134.47375
        },
        {
            "temp_grad": 134.47750000000002
        },
        {
            "temp_grad": 134.48149999999998
        },
        {
            "temp_grad": 134.48525
        },
        {
            "temp_grad": 134.48900000000003
        },
        {
            "temp_grad": 134.49275
        },
        {
            "temp_grad": 134.49649999999997
        }
    ]
}
