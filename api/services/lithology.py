from fastapi import APIRouter, Body
import pandas as pd

from api.schemas.lithology_ssc import ssc_inputData, SSC_EXAMPLE
from api.schemas.lithology_vsh_gr import vsh_gr_inputData, VSH_GR_EXAMPLE
from api.schemas.lithology_hc_correction import hc_corr_inputData, HC_CORR_EXAMPLE

from quick_pp.lithology import SandSiltClay
from quick_pp.lithology import gr_index
from quick_pp.qaqc import neu_den_xplot_hc_correction

router = APIRouter(prefix="/lithology", tags=["Lithology"])


@router.post("/ssc")
async def estimate_ssc(inputs: ssc_inputData = Body(..., example=SSC_EXAMPLE)):

    input_dict = inputs.model_dump()
    assert all([len(input_dict[k]) == 2 for k in input_dict.keys() if "_point" in k]), \
        "End points must be of 2 elements: neutron porosity and bulk density."
    assert all([all(input_dict[k]) for k in ['dry_sand_point', 'fluid_point']]), \
        "'dry_sand_point' and 'fluid_point' points must not be None."

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
    return_dict = pd.DataFrame(
        {'VSAND': vsand, 'VSILT': vsilt, 'VCLW': vcld + vclb, 'VCLB': vclb, 'VCLD': vcld}, index=input_df.index
    ).to_dict(orient='records')
    return return_dict


@router.post("/vsh_gr")
async def estimate_vsh_gr(inputs: vsh_gr_inputData = Body(..., example=VSH_GR_EXAMPLE)):

    input_dict = inputs.model_dump()
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    vsh_gr = gr_index(input_df['gr'])
    return_dict = pd.DataFrame({'GR': vsh_gr.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/hc_corr")
async def estimate_hc_correction(inputs: hc_corr_inputData = Body(..., example=HC_CORR_EXAMPLE)):

    input_dict = inputs.model_dump()
    assert all([len(input_dict[k]) == 2 for k in input_dict.keys() if "_point" in k]), \
        "End points must be of 2 elements: neutron porosity and bulk density."
    assert all([all(input_dict[k]) for k in ['dry_sand_point', 'dry_clay_point', 'fluid_point']]), \
        "'dry_sand_point', 'dry_clay_point' and 'fluid_point' points must not be None."

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
