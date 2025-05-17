from pydantic import BaseModel
from typing import List, Dict, Any


class Data(BaseModel):
    """Input record for permeability estimation (other models)."""

    phit: float
    swirr: float


class InputData(BaseModel):
    """Input data wrapper for permeability endpoints (other models)."""

    data: List[Data]

    class Config:
        orm_mode = True


EXAMPLE: Dict[str, Any] = {
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
