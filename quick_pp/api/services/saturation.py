from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List, Dict, Any, Optional
import numpy as np

from quick_pp.api.schemas.saturation_waxman_smits import InputData as WaxmanSmitsInput
from quick_pp.api.schemas.saturation_archie import InputData as ArchieInput
from quick_pp.api.schemas.saturation_temp_grad import InputData as TempGradInput
from quick_pp.api.schemas.saturation_rw import InputData as RwInput
from quick_pp.api.schemas.saturation_b import InputData as BInput
from quick_pp.api.schemas.saturation_qv import InputData as QvInput

from quick_pp.saturation import (
    waxman_smits_saturation, archie_saturation, estimate_rw_temperature_salinity, estimate_temperature_gradient,
    estimate_b_waxman_smits, estimate_qv
)

router = APIRouter(prefix="/saturation", tags=["Saturation"])


def _safe_float(val):
    """
    Safely convert a value to float for API output.
    If the value is complex, returns the real part as float.
    If conversion fails, returns 0.0 to maintain API contract.
    """
    try:
        if isinstance(val, complex):
            return float(val.real)
        return float(val)
    except Exception:
        return 0.0  # fallback to 0.0 for API contract


def _parse_and_respond(
    inputs: Any,
    func,
    func_args: List[str],
    result_key: str,
    extra_args: Optional[Dict[str, Any]] = None,
    dtypes: Optional[Dict[str, Any]] = None
) -> List[Dict[str, float]]:
    """
    Helper to parse input, call calculation, and format response for batch endpoints.

    Args:
        inputs: Pydantic model instance containing the request data.
        func: The calculation function to call (e.g., estimate_b_waxman_smits).
        func_args: List of column names to extract from the input DataFrame as function arguments.
        result_key: The key to use in the output dictionary for the result.
        extra_args: Optional dictionary of extra arguments to pass to the calculation function.
        dtypes: Optional dictionary mapping column names to numpy dtypes for type enforcement.

    Returns:
        List[Dict[str, float]]: List of dictionaries with the result for each input record.

    Technical Details:
        - Converts input Pydantic model to dict and then to a pandas DataFrame.
        - Applies type conversions as specified in dtypes.
        - Extracts columns as numpy arrays for vectorized calculation.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict['data'])
        if dtypes:
            for col, dtype in dtypes.items():
                if col in input_df:
                    input_df[col] = input_df[col].astype(dtype)
        args = [np.asarray(input_df[col]) for col in func_args if col in input_df]
        if extra_args:
            args.extend(extra_args.values())
        result = func(*args)
        # Ensure result is always iterable
        if np.isscalar(result):
            result_list = [_safe_float(result)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(result).flatten()]
        return [{result_key: val} for val in result_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


@router.post(
    "/temp_grad",
    summary="Estimate Temperature Gradient",
    description="Estimate the temperature gradient based on depth data and specified measurement system "
    "(metric or imperial).",
    operation_id="estimate_temperature_gradient",
)
async def estimate_temperature_gradient_(inputs: TempGradInput) -> List[Dict[str, float]]:
    """
    Estimate the temperature gradient for each input record.

    Args:
        inputs (TempGradInput): Input data containing a list of records with 'tvdss' (true vertical depth subsea)
            and the measurement system ('meas_system').
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated temperature gradient under the key 'TEMP_GRAD'.
    Technical Details:
        - Converts input data to a pandas DataFrame and enforces float type for 'tvdss'.
        - Calls estimate_temperature_gradient with the TVDSS array and measurement system.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    input_dict = inputs.model_dump()
    meas_system = input_dict['meas_system']
    try:
        input_df = pd.DataFrame.from_records(input_dict['data'])
        input_df['tvdss'] = input_df['tvdss'].astype(np.float64)
        temp_grad = estimate_temperature_gradient(np.asarray(input_df['tvdss']), meas_system)
        if np.isscalar(temp_grad):
            result_list = [_safe_float(temp_grad)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(temp_grad).flatten()]
        return [{"TEMP_GRAD": val} for val in result_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


@router.post(
    "/rw",
    summary="Estimate Formation Water Resistivity (Rw)",
    description="Estimate formation water resistivity (Rw) based on temperature gradient and water salinity. "
    "The temperature gradient is provided in the input data, and the water salinity is a scalar value.",
    operation_id="estimate_formation_water_resistivity",
)
async def estimate_rw(inputs: RwInput) -> List[Dict[str, float]]:
    """
    Estimate formation water resistivity (Rw) for each input record.

    Args:
        inputs (RwInput): Input data containing water salinity and a list of temperature gradient records.
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated Rw values under the key 'RW'.
    Technical Details:
        - Converts input data to a pandas DataFrame and enforces float type for 'temp_grad'.
        - Calls estimate_rw_temperature_salinity with the temperature gradient array and water salinity.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    input_dict = inputs.model_dump()
    water_salinity = input_dict['water_salinity']
    try:
        input_df = pd.DataFrame.from_records(input_dict['data'])
        input_df['temp_grad'] = input_df['temp_grad'].astype(np.float64)
        rw = estimate_rw_temperature_salinity(np.asarray(input_df['temp_grad']), water_salinity)
        if np.isscalar(rw):
            result_list = [_safe_float(rw)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(rw).flatten()]
        return [{"RW": val} for val in result_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


@router.post(
    "/b_waxman_smits",
    summary="Estimate B Parameter using Waxman-Smits Model",
    description="Estimate the B parameter using the Waxman-Smits model based on temperature gradient "
    "and formation water resistivity.",
    operation_id="estimate_b_parameter_waxman_smits",
)
async def estimate_b_waxman_smits_(inputs: BInput) -> List[Dict[str, float]]:
    """
    Estimate the B parameter using the Waxman-Smits model for each input record.

    Args:
        inputs (BInput): Input data model containing a list of records with 'temp_grad' (temperature gradient)
            and 'rw' (formation water resistivity).
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated B parameter under the key 'B'.
    Technical Details:
        - Uses _parse_and_respond helper to handle input parsing, type enforcement, and output formatting.
        - Calls estimate_b_waxman_smits with temperature gradient and resistivity arrays.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    return _parse_and_respond(
        inputs,
        estimate_b_waxman_smits,
        ['temp_grad', 'rw'],
        'B',
        dtypes={"temp_grad": np.float64, "rw": np.float64}
    )


@router.post(
    "/estimate_qv",
    summary="Estimate Cation Exchange Capacity per Unit Pore Volume (Qv)",
    description="Estimate Qv based on volume of clay (vcld), total porosity (phit), and specific clay properties "
    "which are clay density (rho_clay) and cation exchange capacity (cec_clay).",
    operation_id="estimate_cation_exchange_capacity",
)
async def estimate_qv_(inputs: QvInput) -> List[Dict[str, float]]:
    """
    Estimate the cation exchange capacity per unit pore volume (Qv) for each input record.

    Args:
        inputs (QvInput): Input data containing well log measurements (vcld, phit) and
                          clay properties (rho_clay, cec_clay).
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated Qv values under the key 'QV'.
    Technical Details:
        - Converts input data to a pandas DataFrame and enforces float type for 'vcld' and 'phit'.
        - Calls estimate_qv with vcld, phit, rho_clay, and cec_clay.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    input_dict = inputs.model_dump()
    try:
        input_df = pd.DataFrame.from_records(input_dict['data'])
        input_df['vcld'] = input_df['vcld'].astype(np.float64)
        input_df['phit'] = input_df['phit'].astype(np.float64)
        qv = estimate_qv(
            np.asarray(input_df['vcld']), np.asarray(input_df['phit']),
            input_dict['rho_clay'], input_dict['cec_clay']
        )
        if np.isscalar(qv):
            result_list = [_safe_float(qv)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(qv).flatten()]
        return [{"QV": val} for val in result_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


@router.post(
    "/waxman_smits",
    summary="Estimate Water Saturation using Waxman-Smits Model",
    description="Estimate total water saturation (SWT) using the Waxman-Smits model based on input data. "
    "The model requires resistivity (rt), water resistivity (rw), total porosity (phit), "
    "cation exchange capacity per unit pore volume (qv), b parameter, and m parameter.",
    operation_id="estimate_waxman_smits_water_saturation",
)
async def estimate_swt_waxman_smits(inputs: WaxmanSmitsInput) -> List[Dict[str, float]]:
    """
    Estimate water saturation (SWT) using the Waxman-Smits model for each input record.

    Args:
        inputs (WaxmanSmitsInput): Input data for multiple samples, including rt, rw, phit, qv, b, and m.
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated water saturation under the key 'SWT'.
    Technical Details:
        - Converts input data to a pandas DataFrame and enforces correct types for all columns.
        - Passes m as a single integer (first value in the input) to waxman_smits_saturation.
        - Calls waxman_smits_saturation with all required arrays and m.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    input_dict = inputs.model_dump()
    try:
        input_df = pd.DataFrame.from_records(input_dict['data'])
        input_df['rt'] = input_df['rt'].astype(np.float64)
        input_df['rw'] = input_df['rw'].astype(np.float64)
        input_df['phit'] = input_df['phit'].astype(np.float64)
        input_df['qv'] = input_df['qv'].astype(np.float64)
        input_df['b'] = input_df['b'].astype(np.float64)
        input_df['m'] = input_df['m'].astype(np.int_)
        # Use the first value of m for all records if function expects a scalar
        m_value = int(input_df['m'].iloc[0])
        swt = waxman_smits_saturation(
            np.asarray(input_df['rt']), np.asarray(input_df['rw']), np.asarray(input_df['phit']),
            np.asarray(input_df['qv']), np.asarray(input_df['b']), m_value
        )
        if np.isscalar(swt):
            result_list = [_safe_float(swt)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(swt).flatten()]
        return [{"SWT": val} for val in result_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")


@router.post(
    "/archie",
    summary="Estimate Water Saturation using Archie Model",
    description="Estimate total water saturation (SWT) using the Archie model based on input data. "
    "The model requires resistivity (rt), water resistivity (rw), and total porosity (phit).",
    operation_id="estimate_archie_water_saturation",
)
async def estimate_swt_archie(inputs: ArchieInput) -> List[Dict[str, float]]:
    """
    Estimate water saturation (SWT) using Archie's equation for each input record.

    Args:
        inputs (ArchieInput): Input data for multiple samples, including rt, rw, and phit.
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated water saturation under the key 'SWT'.
    Technical Details:
        - Converts input data to a pandas DataFrame and enforces float type for all columns.
        - Calls archie_saturation with fixed parameters a=1, m=2, n=2 (typical for clean sandstone).
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    input_dict = inputs.model_dump()
    try:
        input_df = pd.DataFrame.from_records(input_dict['data'])
        input_df['rt'] = input_df['rt'].astype(np.float64)
        input_df['rw'] = input_df['rw'].astype(np.float64)
        input_df['phit'] = input_df['phit'].astype(np.float64)
        swt = archie_saturation(
            np.asarray(input_df['rt']), np.asarray(input_df['rw']), np.asarray(input_df['phit']), 1, 2, 2
        )
        if np.isscalar(swt):
            result_list = [_safe_float(swt)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(swt).flatten()]
        return [{"SWT": val} for val in result_list]
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing request: {e}")
