"""Lithology API services.

Endpoints for lithology-related calculations:

- Estimate Sand / Silt / Clay (SSC)
- Estimate volume of shale (Vsh) from Gamma Ray (GR)
- Hydrocarbon correction for NPHI/RHOB crossplots

Input Pydantic models:

- `LithologySSCInput` — see `quick_pp.app.backend.schemas.lithology_ssc.LithologySSCInput`
- `LithologyVshGRInput` — see `quick_pp.app.backend.schemas.lithology_vsh_gr.LithologyVshGRInput`
- `LithologyHCCorrectionInput` — see `quick_pp.app.backend.schemas.lithology_hc_correction.LithologyHCCorrectionInput`
"""

from fastapi import APIRouter
import pandas as pd

from quick_pp.app.backend.schemas.lithology_ssc import LithologySSCInput
from quick_pp.app.backend.schemas.lithology_vsh_gr import LithologyVshGRInput
from quick_pp.app.backend.schemas.lithology_hc_correction import (
    LithologyHCCorrectionInput,
)

from quick_pp.lithology.sand_silt_clay import SandSiltClay
from quick_pp.lithology import gr_index
from quick_pp.qaqc import neu_den_xplot_hc_correction

router = APIRouter(prefix="/lithology", tags=["Lithology"])


def _validate_points(input_dict, required_points):
    """
    Validate that all required endpoint keys in input_dict are present and contain two elements (NPHI, RHOB),
    and that required points are not None or contain None values.
    Raises ValueError if validation fails.
    """
    for k in [key for key in input_dict if key.endswith("_point")]:
        if not (isinstance(input_dict[k], (list, tuple)) and len(input_dict[k]) == 2):
            raise ValueError(
                f"'{k}' must be a tuple/list of length 2 (NPHI, RHOB). Got: {input_dict[k]}"
            )
    for k in required_points:
        if not all(input_dict.get(k, ())):
            raise ValueError(f"'{k}' point must not be None or contain None values.")


def _to_dataframe(data, columns=None):
    """
    Convert a list of dicts to a pandas DataFrame and check for required columns.
    Raises ValueError if any required columns are missing.
    """
    df = pd.DataFrame.from_records(data)
    if columns:
        missing = set(columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in input data: {missing}")
    return df


@router.post(
    "/ssc",
    summary="Estimate Sand, Silt, and Clay (SSC) Volume Fractions",
    description=(
        """
        Estimate sand, silt, and clay (SSC) volume fractions based on a multi-endpoint lithology model.
        
        Input model: LithologySSCInput (see quick_pp.app.backend.schemas.lithology_ssc.LithologySSCInput).
        
        Request body must be a JSON object with the following fields:
        - dry_sand_point: [float, float] (required)
        - dry_silt_point: [float, float] or [null, null] (optional)
        - dry_clay_point: [float, float] or [null, null] (optional)
        - fluid_point: [float, float] (required)
        - wet_clay_point: [float, float] or [null, null] (optional)
        - method: string (optional)
        - silt_line_angle: float (required)
        - data: list of objects, each with keys 'nphi' (float) and 'rhob' (float) (required)
        
        Example:
        {
            'dry_sand_point': [0.1, 2.65],
            'dry_silt_point': [0.2, 2.5],
            'dry_clay_point': [0.3, 2.4],
            'fluid_point': [1.0, 1.0],
            'wet_clay_point': [0.5, 2.2],
            'method': 'default',
            'silt_line_angle': 119.0,
            'data': [{'nphi': 0.25, 'rhob': 2.45}, ... ]
        }
        """
    ),
    operation_id="estimate_sand_silt_clay_lithology",
)
async def estimate_ssc(inputs: LithologySSCInput):
    """
    Estimate sand, silt, and clay (SSC) volume fractions from well log data using a multi-endpoint lithology model.

    Parameters
    ----------
    inputs : LithologySSCInput
        Pydantic model containing:
            - dry_sand_point: Tuple[float, float]
            - dry_silt_point: Tuple[Optional[float], Optional[float]]
            - dry_clay_point: Tuple[Optional[float], Optional[float]]
            - fluid_point: Tuple[float, float]
            - wet_clay_point: Tuple[Optional[float], Optional[float]]
            - method: str
            - silt_line_angle: float
            - data: List[dict] with 'nphi' and 'rhob' keys

    Returns
    -------
    List[dict]
        List of dictionaries, each containing the estimated volume fractions:
            - 'VSAND': float (sand fraction)
            - 'VSILT': float (silt fraction)
            - 'VCLAY': float (clay fraction)

    Raises
    ------
    ValueError
        If endpoint validation fails or required data columns are missing.

    Technical Details
    ----------------
    - Validates endpoint coordinates for completeness and correct dimensionality.
    - Constructs a SandSiltClay model using the provided endpoints and silt line angle.
    - Estimates lithology fractions for each input data point using neutron porosity and bulk density logs.
    - Returns results as a list of dictionaries, preserving input order.
    """
    input_dict = inputs.model_dump()
    _validate_points(input_dict, required_points=["dry_sand_point", "fluid_point"])
    df = _to_dataframe(input_dict["data"], columns=["nphi", "rhob"])
    ssc_model = SandSiltClay(
        dry_sand_point=input_dict["dry_sand_point"],
        dry_silt_point=input_dict["dry_silt_point"],
        dry_clay_point=input_dict["dry_clay_point"],
        fluid_point=input_dict["fluid_point"],
        wet_clay_point=input_dict["wet_clay_point"],
        silt_line_angle=input_dict["silt_line_angle"],
    )
    vsand, vsilt, vcld, _ = ssc_model.estimate_lithology(df["nphi"], df["rhob"])
    return pd.DataFrame(
        {"VSAND": vsand, "VSILT": vsilt, "VCLAY": vcld}, index=df.index
    ).to_dict(orient="records")


@router.post(
    "/vsh_gr",
    summary="Estimate Volume of Shale (Vsh) from Gamma Ray Log Data",
    description=(
        """
        Estimate volume of shale (Vsh) using gamma ray log data. Requires gamma ray (GR) measurements.
        
        Input model: LithologyVshGRInput (see quick_pp.app.backend.schemas.lithology_vsh_gr.LithologyVshGRInput).
        
        Request body must be a JSON object with the following field:
        - data: list of objects, each with key 'gr' (float) (required)
        
        Example:
        {'data': [{'gr': 85.0}, {'gr': 110.0}, ... ]}
        """
    ),
    operation_id="estimate_vshale_gamma_ray",
)
async def estimate_vsh_gr(inputs: LithologyVshGRInput):
    """
    Estimate the volume of shale (Vsh) from gamma ray (GR) log data.

    Parameters
    ----------
    inputs : LithologyVshGRInput
        Pydantic model containing:
            - data: List[dict] with 'gr' key (gamma ray log values)

    Returns
    -------
    List[dict]
        List of dictionaries, each with a single key 'GR'.
        Each value represents the estimated volume of shale for the corresponding input record.

    Raises
    ------
    ValueError
        If required data columns are missing.

    Technical Details
    ----------------
    - Converts input data to a DataFrame and checks for the 'gr' column.
    - Applies the gamma ray index calculation using the gr_index function.
    - Returns the result as a list of dictionaries, preserving input order.
    """
    input_dict = inputs.model_dump()
    df = _to_dataframe(input_dict["data"], columns=["gr"])
    vsh_gr = gr_index(df["gr"])
    return pd.DataFrame({"GR": vsh_gr.ravel()}, index=df.index).to_dict(
        orient="records"
    )


@router.post(
    "/hc_corr",
    summary="Estimate Hydrocarbon Correction and Lithology Fractions",
    description=(
        """
        Estimate hydrocarbon correction and lithology fractions from well log data.
        
        Requires neutron porosity (NPHI) and bulk density (RHOB) measurements,
        along with reference points for dry sand, silt, clay, and fluid as well as correction angle.
        
        Input model: LithologyHCCorrectionInput
        (see quick_pp.app.backend.schemas.lithology_hc_correction.LithologyHCCorrectionInput).
        
        Request body must be a JSON object with the following fields:
        - dry_sand_point: [float, float] (required)
        - dry_silt_point: [float, float] or [null, null] (optional)
        - dry_clay_point: [float, float] or [null, null] (required)
        - fluid_point: [float, float] (required)
        - wet_clay_point: [float, float] or [null, null] (optional)
        - method: string (optional)
        - silt_line_angle: float (required)
        - corr_angle: float (required)
        - data: list of objects, each with keys 'nphi' (float) and 'rhob' (float) (required)
        
        Example:
        {
            'dry_sand_point': [0.1, 2.65],
            'dry_silt_point': [0.2, 2.5],
            'dry_clay_point': [0.3, 2.4],
            'fluid_point': [1.0, 1.0],
            'wet_clay_point': [0.5, 2.2],
            'method': 'default',
            'silt_line_angle': 35.0,
            'corr_angle': 12.0,
            'data': [{'nphi': 0.25, 'rhob': 2.45}, ... ]
        }
        """
    ),
    operation_id="estimate_hydro_carbon_correction",
)
async def estimate_hc_correction(inputs: LithologyHCCorrectionInput):
    """
    Estimate hydrocarbon correction and lithology fractions from input data.

    Parameters
    ----------
    inputs : LithologyHCCorrectionInput
        Pydantic model containing:
            - dry_sand_point: Tuple[float, float]
            - dry_silt_point: Tuple[Optional[float], Optional[float]]
            - dry_clay_point: Tuple[Optional[float], Optional[float]]
            - fluid_point: Tuple[float, float]
            - wet_clay_point: Tuple[Optional[float], Optional[float]]
            - method: str
            - silt_line_angle: float
            - corr_angle: float
            - data: List[dict] with 'nphi' and 'rhob' keys

    Returns
    -------
    List[dict]
        List of dictionaries, each containing the estimated volume fractions:
            - 'VSAND': float (sand fraction)
            - 'VSILT': float (silt fraction)
            - 'VCLAY': float (clay fraction)

    Raises
    ------
    ValueError
        If endpoint validation fails or required data columns are missing.

    Technical Details
    ----------------
    - Validates endpoint coordinates for completeness and correct dimensionality.
    - Applies hydrocarbon correction to neutron porosity and bulk density logs using neu_den_xplot_hc_correction,
      which rotates the crossplot by the specified correction angle.
    - Constructs a SandSiltClay model using the provided endpoints and silt line angle.
    - Estimates lithology fractions for each corrected data point.
    - Returns results as a list of dictionaries, preserving input order.
    """
    input_dict = inputs.model_dump()
    _validate_points(
        input_dict, required_points=["dry_sand_point", "dry_clay_point", "fluid_point"]
    )
    df = _to_dataframe(input_dict["data"], columns=["nphi", "rhob"])
    nphihc, rhobhc, _ = neu_den_xplot_hc_correction(
        df["nphi"],
        df["rhob"],
        dry_min1_point=input_dict["dry_sand_point"],
        dry_clay_point=input_dict["dry_clay_point"],
        corr_angle=input_dict["corr_angle"],
    )
    df_corr = pd.DataFrame({"NPHI": nphihc, "RHOB": rhobhc}).astype(float)
    ssc_model = SandSiltClay(
        dry_sand_point=input_dict["dry_sand_point"],
        dry_silt_point=input_dict["dry_silt_point"],
        dry_clay_point=input_dict["dry_clay_point"],
        fluid_point=input_dict["fluid_point"],
        wet_clay_point=input_dict["wet_clay_point"],
        silt_line_angle=input_dict["silt_line_angle"],
    )
    vsand, vsilt, vcld, _ = ssc_model.estimate_lithology(
        df_corr["NPHI"], df_corr["RHOB"]
    )
    return pd.DataFrame(
        {"VSAND": vsand, "VSILT": vsilt, "VCLAY": vcld}, index=df.index
    ).to_dict(orient="records")
