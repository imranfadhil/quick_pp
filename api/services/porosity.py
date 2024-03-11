from fastapi import APIRouter, Body
import pandas as pd

from api.schemas.porosity import inputData as phit_inputData, EXAMPLE as PHIT_EXAMPLE

from quick_pp.lithology import SandSiltClay
from quick_pp.porosity import neu_den_xplot_poro, density_porosity, rho_matrix

router = APIRouter(prefix="/porosity", tags=["Porosity"])


@router.post("/den", description="Estimate density porosity based on neutron porosity and bulk density.")
async def estimate_phit_den(inputs: phit_inputData = Body(..., example=PHIT_EXAMPLE)):

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
    vsand, vsilt, vcld, vclb, _ = ssc_model.estimate_lithology(nphi, rhob, model=Method, normalize=False)
    df_ssc_model = pd.DataFrame(
        {'VSAND': vsand, 'VSILT': vsilt, 'VCLW': vcld + vclb, 'VCLD': vcld},
    )
    rho_ma = rho_matrix(df_ssc_model['VSAND'], df_ssc_model['VSILT'], df_ssc_model['VCLD'])

    phid = density_porosity(rhob, rho_ma, FluidPoint[1])
    return_dict = pd.DataFrame({'PHID': phid.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/neu_den", description="Estimate total porosity based on neutron porosity and bulk density.")
async def estimate_phit_neu_den(inputs: phit_inputData = Body(..., example=PHIT_EXAMPLE)):

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

    return_dict = pd.DataFrame({'PHIT': phit}).to_dict(orient='records')
    return return_dict
