from pydantic import BaseModel
from typing import List


class data(BaseModel):
    tvdss: float


class temp_grad_inputData(BaseModel):
    data: List[data]
    meas_system: str = 'metric'


TEMP_GRAD_EXAMPLE = {
    'meas_system': 'metric',
    'data': [
        {'tvdss': 4098.49},
        {'tvdss': 4098.65},
        {'tvdss': 4098.8},
        {'tvdss': 4098.95},
        {'tvdss': 4099.1},
        {'tvdss': 4099.26},
        {'tvdss': 4099.41},
        {'tvdss': 4099.56},
        {'tvdss': 4099.71},
        {'tvdss': 4099.86}
    ]
}
