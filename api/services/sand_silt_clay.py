from fastapi import APIRouter, Body, Query
from pydantic import BaseModel
from typing import List
import pandas as pd

from quick_pp.lithology import SandSiltClay

router = APIRouter(prefix="/ssc", tags=["lithology"])


class data(BaseModel):
    nphi: float
    rhob: float


class inputData(BaseModel):
    data: List[data]


EXAMPLE = {'data': [
    {'nphi': 0.3, 'rhob': 1.85},
    {'nphi': 0.35, 'rhob': 1.95},
    {'nphi': 0.34, 'rhob': 1.9},
]}


@router.post("")
async def estimate_ssc(
    inputs: inputData = Body(..., example=EXAMPLE),
    DrySandPoint: tuple = Query(
        (-0.02, 2.65), max_length=2, description="(Neutron porosity of dry sand, Bulk density of dry sand)"),
    DrySiltPoint: tuple = Query(
        (None, 2.68), max_length=2, description="(Neutron porosity of dry silt, Bulk density of dry silt)"),
    DryClayPoint: tuple = Query(
        (None, 2.7), max_length=2, description="(Neutron porosity of dry clay, Bulk density of dry clay)"),
    FluidPoint: tuple = Query(
        (1.0, 1.0), max_length=2, description="(Neutron porosity of fluid, Bulk density of fluid)"),
    WetClayPoint: tuple = Query(
        (None, None), max_length=2, description="(Neutron porosity of wet clay, Bulk density of wet clay)"),
    Method: str = Query("kuttan_modified",
                        description="Choose the method to estimate lithology, either 'kuttan' or 'kuttan_modified'."),
    SiltLineAngle: float = 117
):
    # Change the tuples from str to float
    DrySandPoint = tuple(map(float, DrySandPoint))
    DrySiltPoint = tuple(map(lambda x: float(x) if x != '' else None, DrySiltPoint))
    DryClayPoint = tuple(map(lambda x: float(x) if x != '' else None, DryClayPoint))
    FluidPoint = tuple(map(lambda x: float(x) if x != '' else None, FluidPoint))
    WetClayPoint = tuple(map(lambda x: float(x) if x != '' else None, WetClayPoint))

    input_dict = inputs.model_dump()
    input_df = pd.DataFrame.from_records(input_dict['data'])

    nphi = input_df['nphi']
    rhob = input_df['rhob']

    ssc_model = SandSiltClay(
        dry_sand_point=DrySandPoint, dry_silt_point=DrySiltPoint, dry_clay_point=DryClayPoint,
        fluid_point=FluidPoint, wet_clay_point=WetClayPoint, silt_line_angle=SiltLineAngle
    )
    vsand, vsilt, vcld, vclb, _ = ssc_model.estimate_lithology(nphi, rhob, model=Method)
    return_df = pd.DataFrame(
        {'VSAND': vsand, 'VSILT': vsilt, 'VCLW': vcld + vclb, 'VCLB': vclb, 'VCLD': vcld},
        index=input_df.index
    )
    return_dict = return_df.to_dict(orient='records')
    return return_dict