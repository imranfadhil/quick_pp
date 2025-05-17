"""
Schemas for sand-silt-clay lithology input.
"""
from pydantic import BaseModel
from typing import List, Optional, Tuple


class SSCData(BaseModel):
    nphi: float
    rhob: float


class LithologySSCInput(BaseModel):
    dry_sand_point: Tuple[Optional[float], Optional[float]]
    dry_silt_point: Tuple[Optional[float], Optional[float]]
    dry_clay_point: Tuple[Optional[float], Optional[float]]
    fluid_point: Tuple[Optional[float], Optional[float]]
    wet_clay_point: Tuple[Optional[float], Optional[float]]
    method: str
    silt_line_angle: float
    data: List[SSCData]


EXAMPLE = {
    'dry_sand_point': (-0.02, 2.65),
    'dry_silt_point': (None, 2.68),
    'dry_clay_point': (None, 2.7),
    'fluid_point': (1.0, 1.0),
    'wet_clay_point': (None, None),
    'method': 'kuttan_modified',
    'silt_line_angle': 117,
    'data': [
        {'nphi': 0.365, 'rhob': 2.48},
        {'nphi': 0.17, 'rhob': 2.304},
        {'nphi': 0.31, 'rhob': 2.39},
        {'nphi': 0.45, 'rhob': 1.85},
        {'nphi': 0.244, 'rhob': 2.513},
        {'nphi': 0.207, 'rhob': 2.454},
        {'nphi': 0.323, 'rhob': 2.438},
        {'nphi': 0.097, 'rhob': 2.177},
        {'nphi': 0.353, 'rhob': 2.5},
        {'nphi': 0.208, 'rhob': 2.487},
    ],
}
