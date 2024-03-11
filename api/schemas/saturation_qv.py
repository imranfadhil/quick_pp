from pydantic import BaseModel
from typing import List


class data(BaseModel):
    vcld: float
    phit: float


class inputData(BaseModel):
    data: List[data]
    rho_clay: float
    cec_clay: float


EXAMPLE = {
    'rho_clay': 2.65,
    'cec_clay': 0.062,
    'data': [
        {'vcld': 0.322, 'phit': 0.204},
        {'vcld': 0.608, 'phit': 0.192},
        {'vcld': 0.622, 'phit': 0.144},
        {'vcld': 0.768, 'phit': 0.102},
        {'vcld': 0.174, 'phit': 0.208},
        {'vcld': 0.018, 'phit': 0.271},
        {'vcld': 0.688, 'phit': 0.125},
        {'vcld': 0.493, 'phit': 0.248},
        {'vcld': 0.781, 'phit': 0.126},
        {'vcld': 0.731, 'phit': 0.115}
    ]
}
