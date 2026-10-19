from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quick_pp.app.backend.schemas.saturation_archie import InputData as ArchieInput
from quick_pp.app.backend.schemas.saturation_b import InputData as BInput
from quick_pp.app.backend.schemas.saturation_qv import InputData as QvInput
from quick_pp.app.backend.schemas.saturation_qvn import InputData as QvnInput
from quick_pp.app.backend.schemas.saturation_rw import InputData as RwInput
from quick_pp.app.backend.schemas.saturation_temp_grad import InputData as TempGradInput
from quick_pp.app.backend.schemas.saturation_waxman_smits import (
    InputData as WaxmanSmitsInput,
)
from quick_pp.saturation import (
    archie_saturation,
    estimate_b_waxman_smits,
    estimate_qv,
    estimate_qvn,
    estimate_rw_temperature_salinity,
    estimate_temperature_gradient,
    optimize_fwl,
    predict_sw_shm,
    waxman_smits_saturation,
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
    dtypes: Optional[Dict[str, Any]] = None,
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
        input_df = pd.DataFrame.from_records(input_dict["data"])
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
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


@router.post(
    "/temp_grad",
    summary="Estimate Temperature Gradient",
    description=(
        """
        Estimate the temperature gradient based on depth data and specified measurement system (metric or imperial).

        Input model: TempGradInput (see quick_pp.app.backend.schemas.saturation_temp_grad.InputData).

        Request body must be a JSON object with the following fields:
        - meas_system: string (required, 'metric' or 'imperial')
        - data: list of objects, each with key 'tvdss' (float, required)

        Example:
        {
        'meas_system': 'metric',
        'data': [ {'tvdss': 1500.0}, {'tvdss': 2000.0} ]
        }
        """
    ),
    operation_id="estimate_temperature_gradient",
)
async def estimate_temperature_gradient_(
    inputs: TempGradInput,
) -> List[Dict[str, float]]:
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
    meas_system = input_dict["meas_system"]
    try:
        input_df = pd.DataFrame.from_records(input_dict["data"])
        input_df["tvdss"] = input_df["tvdss"].astype(np.float64)
        temp_grad = estimate_temperature_gradient(
            np.asarray(input_df["tvdss"]), meas_system
        )
        if np.isscalar(temp_grad):
            result_list = [_safe_float(temp_grad)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(temp_grad).flatten()]
        return [{"TEMP_GRAD": val} for val in result_list]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


@router.post(
    "/rw",
    summary="Estimate Formation Water Resistivity (Rw)",
    description=(
        """
        Estimate formation water resistivity (Rw) based on temperature gradient and water salinity.

        Input model: RwInput (see quick_pp.app.backend.schemas.saturation_rw.InputData).

        Request body must be a JSON object with the following fields:
        - water_salinity: float (required)
        - data: list of objects, each with key 'temp_grad' (float, required)

        Example:
        {
        'water_salinity': 35000.0,
        'data': [ {'temp_grad': 0.025}, {'temp_grad': 0.030} ]
        }
        """
    ),
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
    water_salinity = input_dict["water_salinity"]
    try:
        input_df = pd.DataFrame.from_records(input_dict["data"])
        input_df["temp_grad"] = input_df["temp_grad"].astype(np.float64)
        rw = estimate_rw_temperature_salinity(
            np.asarray(input_df["temp_grad"]), water_salinity
        )
        if np.isscalar(rw):
            result_list = [_safe_float(rw)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(rw).flatten()]
        return [{"RW": val} for val in result_list]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


@router.post(
    "/b_waxman_smits",
    summary="Estimate B Parameter using Waxman-Smits Model",
    description=(
        """
        Estimate the B parameter using the Waxman-Smits model based on temperature gradient and
        formation water resistivity.

        Input model: BInput (see quick_pp.app.backend.schemas.saturation_b.InputData).

        Request body must be a JSON object with the following fields:
        - data: list of objects, each with keys 'temp_grad' (float, required) and 'rw' (float, required)

        Example:
        {
        'data': [ {'temp_grad': 0.025, 'rw': 0.12},
                    {'temp_grad': 0.030, 'rw': 0.10} ]
        }
        """
    ),
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
        ["temp_grad", "rw"],
        "B",
        dtypes={"temp_grad": np.float64, "rw": np.float64},
    )


@router.post(
    "/estimate_qv",
    summary="Estimate Cation Exchange Capacity per Unit Pore Volume (Qv)",
    description=(
        """
        Estimate Qv based on volume of clay (vcld), total porosity (phit), and specific clay properties.

        Input model: QvInput (see quick_pp.app.backend.schemas.saturation_qv.InputData).

        Request body must be a JSON object with the following fields:
        - rho_clay: float (required)
        - cec_clay: float (required)
        - data: list of objects, each with keys 'vcld' (float, required) and 'phit' (float, required)

        Example (truncated): { 'rho_clay': 2.58, 'cec_clay': 0.9, 'data': [ {'vcld': 0.35, 'phit': 0.22}, ... ] }
        """
    ),
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
        input_df = pd.DataFrame.from_records(input_dict["data"])
        input_df["vcld"] = input_df["vcld"].astype(np.float64)
        input_df["phit"] = input_df["phit"].astype(np.float64)
        qv = estimate_qv(
            np.asarray(input_df["vcld"]),
            np.asarray(input_df["phit"]),
            input_dict["rho_clay"],
            input_dict["cec_clay"],
        )
        if np.isscalar(qv):
            result_list = [_safe_float(qv)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(qv).flatten()]
        return [{"QV": val} for val in result_list]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


@router.post(
    "/estimate_qvn",
    summary="Estimate Normalized Qv (Qvn) for Juhasz (1981) Model",
    description=(
        """
        Estimate normalized Qv (Qvn) for the Juhasz (1981) model based on volume of clay,
        total porosity, and total porosity of the clay.

        Input model: QvnInput (see quick_pp.app.backend.schemas.saturation_qvn.InputData).

        Request body must be a JSON object with the following fields:
        - data: list of objects, each with keys 'vclay' (float, required), 'phit' (float, required),
                and 'phit_clay' (float, required)

        Example:
        {
            'data': [
                {'vclay': 0.35, 'phit': 0.22, 'phit_clay': 0.15},
                {'vclay': 0.40, 'phit': 0.18, 'phit_clay': 0.15}
            ]
        }
        """
    ),
    operation_id="estimate_normalized_qv",
)
async def estimate_qvn_(inputs: QvnInput) -> List[Dict[str, float]]:
    """
    Estimate the normalized Qv (Qvn) for the Juhasz (1981) model for each input record.

    Args:
        inputs (QvnInput): Input data containing well log measurements (vclay, phit, phit_clay).
    Returns:
        List[Dict[str, float]]: List of dictionaries with the estimated Qvn values under the key 'QVN'.
    Technical Details:
        - Converts input data to a pandas DataFrame and enforces float type for all columns.
        - Calls estimate_qvn with vclay, phit, and phit_clay arrays.
        - Handles both scalar and array results, always returning a list of dicts.
        - Uses _safe_float to ensure all outputs are valid floats.
        - Raises HTTPException with status 400 on any error.
    """
    input_dict = inputs.model_dump()
    try:
        input_df = pd.DataFrame.from_records(input_dict["data"])
        input_df["vclay"] = input_df["vclay"].astype(np.float64)
        input_df["phit"] = input_df["phit"].astype(np.float64)
        input_df["phit_clay"] = input_df["phit_clay"].astype(np.float64)
        qvn = estimate_qvn(
            np.asarray(input_df["vclay"]),
            np.asarray(input_df["phit"]),
            np.asarray(input_df["phit_clay"]),
        )
        if np.isscalar(qvn):
            result_list = [_safe_float(qvn)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(qvn).flatten()]
        return [{"QVN": val} for val in result_list]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


@router.post(
    "/waxman_smits",
    summary="Estimate Water Saturation using Waxman-Smits Model",
    description=(
        """
        Estimate total water saturation (SWT) using the Waxman-Smits model based on input data.

        Input model: WaxmanSmitsInput (see quick_pp.app.backend.schemas.saturation_waxman_smits.InputData).

        Request body must be a JSON object with the following field:
        - data: list of objects, each with keys 'rt', 'rw', 'phit', 'qv', 'b' (all float, required),
                and 'm' (int, required)

        Example:
        {
            'data': [
                {'rt': 12.0, 'rw': 0.12, 'phit': 0.22, 'qv': 0.05, 'b': 0.8, 'm': 2},
                {'rt': 10.0, 'rw': 0.10, 'phit': 0.18, 'qv': 0.04, 'b': 0.7, 'm': 2}
            ]
        }
        """
    ),
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
        input_df = pd.DataFrame.from_records(input_dict["data"])
        input_df["rt"] = input_df["rt"].astype(np.float64)
        input_df["rw"] = input_df["rw"].astype(np.float64)
        input_df["phit"] = input_df["phit"].astype(np.float64)
        input_df["qv"] = input_df["qv"].astype(np.float64)
        input_df["b"] = input_df["b"].astype(np.float64)
        input_df["m"] = input_df["m"].astype(np.float64)
        swt = waxman_smits_saturation(
            np.asarray(input_df["rt"]),
            np.asarray(input_df["rw"]),
            np.asarray(input_df["phit"]),
            np.asarray(input_df["qv"]),
            np.asarray(input_df["b"]),
            np.asarray(input_df["m"]),
        )
        if np.isscalar(swt):
            result_list = [_safe_float(swt)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(swt).flatten()]
        return [{"SWT": val} for val in result_list]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


@router.post(
    "/archie",
    summary="Estimate Water Saturation using Archie Model",
    description=(
        """
        Estimate total water saturation (SWT) using the Archie model based on input data.

        Input model: ArchieInput (see quick_pp.app.backend.schemas.saturation_archie.InputData).

        Request body must be a JSON object with the following field:
        - data: list of objects, each with keys 'rt', 'rw', 'phit' (all float, required)

        Example (truncated): { 'data': [ {'rt': 12.0, 'rw': 0.12, 'phit': 0.22}, ... ] }
        """
    ),
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
        input_df = pd.DataFrame.from_records(input_dict["data"])
        input_df["rt"] = input_df["rt"].astype(np.float64)
        input_df["rw"] = input_df["rw"].astype(np.float64)
        input_df["phit"] = input_df["phit"].astype(np.float64)
        input_df["m"] = input_df["m"].astype(np.float64)
        swt = archie_saturation(
            np.asarray(input_df["rt"]),
            np.asarray(input_df["rw"]),
            np.asarray(input_df["phit"]),
            1,
            np.asarray(input_df["m"]),
            2,
        )
        if np.isscalar(swt):
            result_list = [_safe_float(swt)]
        else:
            result_list = [_safe_float(val) for val in np.asarray(swt).flatten()]
        return [{"SWT": val} for val in result_list]
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error processing request: {e}"
        ) from e


class OptimizeFWLInput(BaseModel):
    model: str = Field(
        "leverett_j",
        description="Saturation-height formulation ('leverett_j', 'skelt_harrison', or 'cuddy')",
    )
    shm_params: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = Field(
        default=None,
        description="Capillary SHM parameters (global dict or per-rock-type list/dict)",
    )
    fwl_bounds: Optional[Tuple[float, float]] = Field(
        default=None,
        description="Optional (fwl_min, fwl_max) search range in TVD/TVDSS",
    )
    ift: float = Field(30.0, gt=0.0, description="Reservoir interfacial tension (dynes/cm)")
    theta: float = Field(30.0, ge=0.0, le=180.0, description="Reservoir contact angle (deg)")
    gw: float = Field(1.05, gt=0.0, description="Formation water density (g/cc)")
    ghc: float = Field(0.80, gt=0.0, description="Hydrocarbon density (g/cc)")
    swirr: Optional[Union[float, str]] = Field(
        default=None, description="Irreducible water saturation fraction or column name"
    )
    depth_col: str = Field("DEPTH", description="Depth column name")
    sw_col: str = Field("SWT", description="Resistivity water saturation column name")
    phit_col: str = Field("PHIT", description="Total porosity column name")
    perm_col: str = Field("PERM", description="Permeability column name")
    rock_flag_col: str = Field("ROCK_FLAG", description="Rock flag column name")
    well_col: str = Field("WELL_NAME", description="Well name column name")
    vshale_col: str = Field("VSHALE", description="Shale volume column name")
    phit_cutoff: float = Field(0.03, ge=0.0, description="Minimum porosity cutoff")
    perm_cutoff: float = Field(0.01, ge=0.0, description="Minimum permeability cutoff (mD)")
    vshale_cutoff: float = Field(0.50, ge=0.0, le=1.0, description="Maximum shale volume cutoff")
    loss: str = Field("rmse", description="Objective loss ('rmse', 'mae', 'huber', or 'bvw')")
    pore_volume_weighted: bool = Field(
        False, description="Weight residuals by pore volume (PHIT)"
    )
    equal_well_weights: bool = Field(
        False, description="Weight each well equally in multi-well objective"
    )
    mdt_fwl: Optional[float] = Field(
        default=None, description="Prior FWL from MDT/RFT pressure gradients"
    )
    pressure_data: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Optional MDT/RFT pressure points (TVDSS, FORMATION_PRESSURE, FLUID_TYPE)",
    )
    mdt_tolerance: Optional[float] = Field(
        default=None, description="Maximum allowed deviation from mdt_fwl"
    )
    mdt_penalty_weight: float = Field(
        0.0, ge=0.0, description="Soft regularization weight toward mdt_fwl"
    )
    solve_per_well: bool = Field(
        True, description="Also solve individual per-well optimal FWLs"
    )
    optimize_shm_params: bool = Field(
        False, description="Jointly refine global SHM parameters alongside FWL"
    )
    unit: str = Field("imperial", description="Depth unit ('imperial' or 'metric')")
    data: List[Dict[str, Any]] = Field(..., description="Multi-well log records")


@router.post(
    "/optimize_fwl",
    summary="Optimize Multi-Well Free Water Level (FWL) & Reconcile Transition Zone",
    operation_id="optimize_multiwell_fwl",
)
async def optimize_fwl_endpoint(inputs: OptimizeFWLInput):
    """Solve for optimal Free Water Level (FWL) across wells by reconciling Sw_log, Sw_SHM, and MDT/RFT."""
    try:
        df = pd.DataFrame(inputs.data)
        pressure_df = (
            pd.DataFrame(inputs.pressure_data)
            if inputs.pressure_data is not None
            else None
        )
        res = optimize_fwl(
            df=df,
            model=inputs.model,
            shm_params=inputs.shm_params,
            fwl_bounds=inputs.fwl_bounds,
            ift=inputs.ift,
            theta=inputs.theta,
            gw=inputs.gw,
            ghc=inputs.ghc,
            swirr=inputs.swirr,
            depth_col=inputs.depth_col,
            sw_col=inputs.sw_col,
            phit_col=inputs.phit_col,
            perm_col=inputs.perm_col,
            rock_flag_col=inputs.rock_flag_col,
            well_col=inputs.well_col,
            vshale_col=inputs.vshale_col,
            phit_cutoff=inputs.phit_cutoff,
            perm_cutoff=inputs.perm_cutoff,
            vshale_cutoff=inputs.vshale_cutoff,
            loss=inputs.loss,
            pore_volume_weighted=inputs.pore_volume_weighted,
            equal_well_weights=inputs.equal_well_weights,
            mdt_fwl=inputs.mdt_fwl,
            pressure_df=pressure_df,
            mdt_tolerance=inputs.mdt_tolerance,
            mdt_penalty_weight=inputs.mdt_penalty_weight,
            solve_per_well=inputs.solve_per_well,
            optimize_shm_params=inputs.optimize_shm_params,
            unit=inputs.unit,
        )
        return {
            "fwl": res["fwl"],
            "owc": res["owc"],
            "ttz": res["ttz"],
            "capillary_entry_height": res["capillary_entry_height"],
            "transition_zone_thickness": res["transition_zone_thickness"],
            "rmse": res["rmse"],
            "mae": res["mae"],
            "r2": res["r2"],
            "bvw_rmse": res["bvw_rmse"],
            "mdt_fwl": res["mdt_fwl"],
            "mdt_details": res["mdt_details"],
            "fwl_reconciliation_diff": res["fwl_reconciliation_diff"],
            "shm_params": res["shm_params"],
            "well_metrics": res["well_metrics"]
            .replace({np.nan: None})
            .to_dict(orient="records"),
            "grid_profile": res["grid_profile"]
            .replace({np.nan: None})
            .to_dict(orient="records"),
            "data": res["df"].replace({np.nan: None}).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error optimizing FWL: {e}"
        ) from e

