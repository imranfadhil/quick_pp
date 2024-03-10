from pydantic import BaseModel
from typing import List


class data(BaseModel):
    temp_grad: float
    rw: float


class b_inputData(BaseModel):
    data: List[data]


B_EXAMPLE = {
    'data': [
        {'temp_grad': 44.92, 'rw': 0.34},
        {'temp_grad': 46.08, 'rw': 0.34},
        {'temp_grad': 45.38, 'rw': 0.34},
        {'temp_grad': 46.49, 'rw': 0.33},
        {'temp_grad': 46.0, 'rw': 0.34},
        {'temp_grad': 45.31, 'rw': 0.34},
        {'temp_grad': 45.32, 'rw': 0.34},
        {'temp_grad': 46.18, 'rw': 0.34},
        {'temp_grad': 46.21, 'rw': 0.33},
        {'temp_grad': 46.61, 'rw': 0.33}
    ]
}
