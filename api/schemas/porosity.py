from pydantic import BaseModel
from typing import List


class data(BaseModel):
    nphi: float
    rhob: float


class phit_inputData(BaseModel):
    dry_sand_point: tuple
    dry_silt_point: tuple
    dry_clay_point: tuple
    fluid_point: tuple
    wet_clay_point: tuple
    method: str
    silt_line_angle: float
    data: List[data]


PHIT_EXAMPLE = {
    'dry_sand_point': (-0.02, 2.65),
    'dry_silt_point': (0.1, 2.68),
    'dry_clay_point': (0.27, 2.7),
    'fluid_point': (1.0, 1.0),
    'wet_clay_point': (None, None),
    'method': 'kuttan_modified',
    'silt_line_angle': 117,
    'data': [
        {'nphi': 0.3, 'rhob': 1.85},
        {'nphi': 0.35, 'rhob': 1.95},
        {'nphi': 0.34, 'rhob': 1.9},
    ],
}
