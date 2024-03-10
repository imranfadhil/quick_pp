from pydantic import BaseModel
from typing import List


class data(BaseModel):
    rt: float
    rw: float
    phit: float


class swt_a_inputData(BaseModel):
    data: List[data]


SWT_A_EXAMPLE = {
    'data': [
        {'rt': 1.834, 'rw': 0.343, 'phit': 0.141},
        {'rt': 1.955, 'rw': 0.343, 'phit': 0.152},
        {'rt': 2.833, 'rw': 0.335, 'phit': 0.198},
        {'rt': 3.469, 'rw': 0.334, 'phit': 0.192},
        {'rt': 3.788, 'rw': 0.337, 'phit': 0.108},
        {'rt': 8.11, 'rw': 0.332, 'phit': 0.191},
        {'rt': 2.315, 'rw': 0.338, 'phit': 0.167},
        {'rt': 1.827, 'rw': 0.343, 'phit': 0.151},
        {'rt': 2.654, 'rw': 0.336, 'phit': 0.115},
        {'rt': 3.628, 'rw': 0.334, 'phit': 0.198}
    ]
}
