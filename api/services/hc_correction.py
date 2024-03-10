from fastapi import APIRouter, Body
from pydantic import BaseModel
from typing import List
import pandas as pd

from quick_pp.qaqc import neu_den_xplot_hc_correction
from quick_pp.lithology import SandSiltClay

router = APIRouter(prefix="/hc_correction", tags=["Lithology"])


class data(BaseModel):
    gr: float
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
    corr_angle: float
    data: List[data]


EXAMPLE = {
    'dry_sand_point': (-0.02, 2.65),
    'dry_silt_point': (None, 2.68),
    'dry_clay_point': (0.33, 2.7),
    'fluid_point': (1.0, 1.0),
    'wet_clay_point': (None, None),
    'method': 'kuttan_modified',
    'silt_line_angle': 117,
    'corr_angle': 50,
    'data': [
        {'gr': 40, 'nphi': 0.3, 'rhob': 1.85},
        {'gr': 50, 'nphi': 0.35, 'rhob': 1.95},
        {'gr': 60, 'nphi': 0.34, 'rhob': 1.9},
    ],
}


@router.post("")
async def estimate_hc_correction(inputs: inputData = Body(..., example=EXAMPLE)):

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
    CorrAngle = input_dict['corr_angle']

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    nphihc, rhobhc = neu_den_xplot_hc_correction(input_df['nphi'], input_df['rhob'], gr=input_df['gr'],
                                                 dry_sand_point=DrySandPoint,
                                                 dry_clay_point=DryClayPoint,
                                                 corr_angle=CorrAngle)
    df_corr = pd.DataFrame({'NPHI': nphihc, 'RHOB': rhobhc}).astype(float)
    # neu_den_df = df_corr[['DEPTH', 'NPHI', 'RHOB']].dropna()

    ssc_model = SandSiltClay(
        dry_sand_point=DrySandPoint, dry_silt_point=DrySiltPoint, dry_clay_point=DryClayPoint,
        fluid_point=FluidPoint, wet_clay_point=WetClayPoint, silt_line_angle=SiltLineAngle
    )
    vsand, vsilt, vcld, vclb, _ = ssc_model.estimate_lithology(df_corr['NPHI'], df_corr['RHOB'], model=Method)
    return_df = pd.DataFrame(
        {'VSAND': vsand, 'VSILT': vsilt, 'VCLW': vcld + vclb, 'VCLB': vclb, 'VCLD': vcld},
        index=input_df.index
    )
    return_dict = return_df.to_dict(orient='records')
    return return_dict
