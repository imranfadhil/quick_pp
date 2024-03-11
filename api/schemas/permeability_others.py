from pydantic import BaseModel
from typing import List


class data(BaseModel):
    phit: float
    swirr: float


class inputData(BaseModel):
    data: List[data]


EXAMPLE = {
    'data': [
        {'phit': 0.328, 'swirr': 0.003},
        {'phit': 0.114, 'swirr': 0.016},
        {'phit': 0.111, 'swirr': 0.03},
        {'phit': 0.278, 'swirr': 0.006},
        {'phit': 0.245, 'swirr': 0.013},
        {'phit': 0.116, 'swirr': 0.018},
        {'phit': 0.131, 'swirr': 0.016},
        {'phit': 0.109, 'swirr': 0.042},
        {'phit': 0.2, 'swirr': 0.023},
        {'phit': 0.143, 'swirr': 0.01}
    ]
}
