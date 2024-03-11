from pydantic import BaseModel
from typing import List


class data(BaseModel):
    vclw: float
    vsilt: float
    phit: float


class inputData(BaseModel):
    data: List[data]


EXAMPLE = {
    'data': [
        {'vclw': 0.662, 'vsilt': 0.264, 'phit': 0.102},
        {'vclw': 0.618, 'vsilt': 0.22, 'phit': 0.147},
        {'vclw': 0.301, 'vsilt': 0.187, 'phit': 0.21},
        {'vclw': 0.684, 'vsilt': 0.226, 'phit': 0.098},
        {'vclw': 0.663, 'vsilt': 0.259, 'phit': 0.113},
        {'vclw': 0.553, 'vsilt': 0.228, 'phit': 0.258},
        {'vclw': 0.725, 'vsilt': 0.129, 'phit': 0.145},
        {'vclw': 0.806, 'vsilt': 0.103, 'phit': 0.108},
        {'vclw': 0.021, 'vsilt': 0.013, 'phit': 0.273},
        {'vclw': 0.052, 'vsilt': 0.032, 'phit': 0.448}
    ]
}
