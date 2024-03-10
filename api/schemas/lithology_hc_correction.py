from pydantic import BaseModel
from typing import List


class data(BaseModel):
    gr: float
    nphi: float
    rhob: float


class hc_corr_inputData(BaseModel):
    dry_sand_point: tuple
    dry_silt_point: tuple
    dry_clay_point: tuple
    fluid_point: tuple
    wet_clay_point: tuple
    method: str
    silt_line_angle: float
    corr_angle: float
    data: List[data]


HC_CORR_EXAMPLE = {
    'dry_sand_point': (-0.02, 2.65),
    'dry_silt_point': (None, 2.68),
    'dry_clay_point': (0.33, 2.7),
    'fluid_point': (1.0, 1.0),
    'wet_clay_point': (None, None),
    'method': 'kuttan_modified',
    'silt_line_angle': 117,
    'corr_angle': 50,
    'data': [
        {'gr': 117.893, 'nphi': 0.285, 'rhob': 2.486},
        {'gr': 116.065, 'nphi': 0.245, 'rhob': 2.441},
        {'gr': 122.852, 'nphi': 0.306, 'rhob': 2.484},
        {'gr': 126.424, 'nphi': 0.357, 'rhob': 2.536},
        {'gr': 111.248, 'nphi': 0.295, 'rhob': 2.498},
        {'gr': 112.464, 'nphi': 0.278, 'rhob': 2.531},
        {'gr': 128.918, 'nphi': 0.388, 'rhob': 2.093},
        {'gr': 60.707, 'nphi': 0.067, 'rhob': 2.073},
        {'gr': 125.888, 'nphi': 0.327, 'rhob': 2.448},
        {'gr': 119.909, 'nphi': 0.313, 'rhob': 2.55}
    ],
}
