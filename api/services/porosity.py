from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List
import pandas as pd

from quick_pp.lithology import SandSiltClay
from quick_pp.porosity import neu_den_xplot_poro, density_porosity, rho_matrix

router = APIRouter(prefix="/phit", tags=["Porosity"])


class data(BaseModel):
    nphi: float
    rhob: float


class inputData(BaseModel):
    dry_sand_point: tuple
    dry_silt_point: tuple
    dry_clay_point: tuple
    fluid_point: tuple
    wet_clay_point: tuple
    method: str
    silt_line_angle: float
    data: List[data]


EXAMPLE = {
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


@router.post("/den")
async def estimate_phit_den(inputs: inputData = Body(..., example=EXAMPLE)):

    input_dict = inputs.model_dump()
    assert all([len(input_dict[k]) == 2 for k in input_dict.keys() if "_point" in k]), \
        "End points must be of 2 elements: neutron porosity and bulk density."

    DrySandPoint = input_dict['dry_sand_point']
    DrySiltPoint = input_dict['dry_silt_point']
    DryClayPoint = input_dict['dry_clay_point']
    FluidPoint = input_dict['fluid_point']
    WetClayPoint = input_dict['wet_clay_point']
    Method = input_dict['method']
    SiltLineAngle = input_dict['silt_line_angle']

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    nphi = input_df['nphi']
    rhob = input_df['rhob']

    ssc_model = SandSiltClay(
        dry_sand_point=DrySandPoint, dry_silt_point=DrySiltPoint, dry_clay_point=DryClayPoint,
        fluid_point=FluidPoint, wet_clay_point=WetClayPoint, silt_line_angle=SiltLineAngle
    )
    vsand, vsilt, vcld, vclb, _ = ssc_model.estimate_lithology(nphi, rhob, model=Method)
    df_ssc_model = pd.DataFrame(
        {'VSAND': vsand, 'VSILT': vsilt, 'VCLW': vcld + vclb, 'VCLD': vcld},
    )
    rho_ma = rho_matrix(df_ssc_model['VSAND'], df_ssc_model['VSILT'], df_ssc_model['VCLD'])

    phid = density_porosity(rhob, rho_ma, FluidPoint[1])
    return_dict = pd.DataFrame({'PHID': phid.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/neu_den")
async def estimate_phit_neu_den(inputs: inputData = Body(..., example=EXAMPLE)):

    input_dict = inputs.model_dump()
    assert all([len(input_dict[k]) == 2 for k in input_dict.keys() if "_point" in k]), \
        "End points must be of 2 elements: neutron porosity and bulk density."

    DrySandPoint = input_dict['dry_sand_point']
    DrySiltPoint = input_dict['dry_silt_point']
    DryClayPoint = input_dict['dry_clay_point']
    FluidPoint = input_dict['fluid_point']
    Method = input_dict['method']

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    nphi = input_df['nphi']
    rhob = input_df['rhob']

    phit = neu_den_xplot_poro(
        nphi, rhob, model=Method,
        dry_sand_point=DrySandPoint, dry_silt_point=DrySiltPoint, dry_clay_point=DryClayPoint,
        fluid_point=FluidPoint
    )

    return_dict = pd.DataFrame({'PHID': phit}).to_dict(orient='records')
    return return_dict
