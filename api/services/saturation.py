from fastapi import APIRouter, Body
import pandas as pd

from api.schemas.saturation_ws import swt_ws_inputData, SWT_WS_EXAMPLE
from api.schemas.saturation_a import swt_a_inputData, SWT_A_EXAMPLE
from api.schemas.saturation_temp_grad import temp_grad_inputData, TEMP_GRAD_EXAMPLE
from api.schemas.saturation_rw import rw_inputData, RW_EXAMPLE
from api.schemas.saturation_b import b_inputData, B_EXAMPLE

from quick_pp.saturation import (
    waxman_smits_saturation, archie_saturation, estimate_rw_temperature_salinity, estimate_temperature_gradient,
    estimate_b_waxman_smits
)

router = APIRouter(prefix="/saturation", tags=["Saturation"])


@router.post("/temp_grad", description="Estimate formation temperature based on gradient of \
             25 degC/km or 15 degF/1000ft.")
async def estimate_temperature_gradient_(inputs: temp_grad_inputData = Body(..., example=TEMP_GRAD_EXAMPLE)):

    input_dict = inputs.model_dump()
    meas_system = input_dict['meas_system']
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    temp_grad = estimate_temperature_gradient(input_df['tvdss'], meas_system)

    return_dict = pd.DataFrame({'temp_grad': temp_grad.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/rw", description="Estimate formation water resistivity based on \
             temperature gradient (degC/meter) and water salinity (ppm).")
async def estimate_rw(inputs: rw_inputData = Body(..., example=RW_EXAMPLE)):

    input_dict = inputs.model_dump()
    water_salinity = input_dict['water_salinity']
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    rw = estimate_rw_temperature_salinity(input_df['temp_grad'], water_salinity)

    return_dict = pd.DataFrame({'RW': rw.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/b_waxman_smits", description="Estimating B (conductance parameter) for \
             Waxman-Smits model based on Juhasz 1981")
async def estimate_b_waxman_smits_(inputs: b_inputData = Body(..., example=B_EXAMPLE)):

    input_dict = inputs.model_dump()
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    b = estimate_b_waxman_smits(input_df['temp_grad'], input_df['rw'])

    return_dict = pd.DataFrame({'B': b.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/waxman_smits")
async def estimate_swt_waxman_smits(inputs: swt_ws_inputData = Body(..., example=SWT_WS_EXAMPLE)):

    input_dict = inputs.model_dump()
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    swt = waxman_smits_saturation(input_df['rt'], input_df['rw'], input_df['phit'],
                                  input_df['qv'], input_df['b'], input_df['m'])
    return_dict = pd.DataFrame({'SWT': swt.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/archie")
async def estimate_swt_archie(inputs: swt_a_inputData = Body(..., example=SWT_A_EXAMPLE)):

    input_dict = inputs.model_dump()
    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])

    swt = archie_saturation(input_df['rt'], input_df['rw'], input_df['phit'], 1, 2, 2)
    return_dict = pd.DataFrame({'SWT': swt.ravel()}).to_dict(orient='records')
    return return_dict
