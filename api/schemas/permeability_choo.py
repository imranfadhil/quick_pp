from pydantic import BaseModel
from typing import List, Dict, Any


class Data(BaseModel):
    """Input record for Choo permeability estimation."""
    vcld: float
    vsilt: float
    phit: float


class InputData(BaseModel):
    """Input data wrapper for Choo permeability endpoint."""
    data: List[Data]

    class Config:
        orm_mode = True


EXAMPLE: Dict[str, Any] = {
    'data': [
        {'vcld': 0.662, 'vsilt': 0.264, 'phit': 0.102},
        {'vcld': 0.618, 'vsilt': 0.22, 'phit': 0.147},
        {'vcld': 0.301, 'vsilt': 0.187, 'phit': 0.21},
        {'vcld': 0.684, 'vsilt': 0.226, 'phit': 0.098},
        {'vcld': 0.663, 'vsilt': 0.259, 'phit': 0.113},
        {'vcld': 0.553, 'vsilt': 0.228, 'phit': 0.258},
        {'vcld': 0.725, 'vsilt': 0.129, 'phit': 0.145},
        {'vcld': 0.806, 'vsilt': 0.103, 'phit': 0.108},
        {'vcld': 0.021, 'vsilt': 0.013, 'phit': 0.273},
        {'vcld': 0.052, 'vsilt': 0.032, 'phit': 0.448}
    ]
}
