"""QAQC API services.

Endpoints for QAQC (Quality Assurance / Quality Control) calculations:

- Hydrocarbon correction for NPHI/RHOB crossplots

Input Pydantic models:

- `HCCorrectionInput` — Hydrocarbon correction for NPHI/RHOB data
"""

from typing import List, Tuple

import numpy as np
from fastapi import APIRouter
from pydantic import BaseModel, Field

from quick_pp.qaqc import neu_den_xplot_hc_correction

router = APIRouter(prefix="/qaqc", tags=["QAQC"])


class DataPoint(BaseModel):
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")


class HCCorrectionInput(BaseModel):
    dry_sand_point: Tuple[float, float] = Field(
        (-0.02, 2.65), description="Dry sand point (nphi, rhob)"
    )
    dry_clay_point: Tuple[float, float] = Field(
        (0.33, 2.7), description="Dry clay point (nphi, rhob)"
    )
    water_point: Tuple[float, float] = Field(
        (1.0, 1.0), description="Water point (nphi, rhob)"
    )
    corr_angle: float = Field(50, description="Correction angle (degrees)")
    buffer: float = Field(0.01, description="Buffer for HC flag detection")
    data: List[DataPoint] = Field(..., description="List of NPHI/RHOB data points")


@router.post("/hc_correction", tags=["HC Correction"])
async def apply_hc_correction(inputs: HCCorrectionInput):
    """
    Apply hydrocarbon correction to NPHI and RHOB data.

    Parameters
    ----------
    inputs : HCCorrectionInput
        Pydantic model containing:
            - dry_sand_point: Tuple[float, float] - Dry sand endpoint
            - dry_clay_point: Tuple[float, float] - Dry clay endpoint
            - water_point: Tuple[float, float] - Water/fluid endpoint
            - corr_angle: float - Correction angle in degrees
            - buffer: float - Buffer for HC detection
            - data: List[DataPoint] - List of NPHI/RHOB points

    Returns
    -------
    List[dict]
        List of dictionaries, each containing:
            - 'nphi': float - Corrected neutron porosity
            - 'rhob': float - Corrected bulk density

    Raises
    ------
    ValueError
        If required data columns are missing or input validation fails.

    Technical Details
    ----------------
    - Applies iterative hydrocarbon correction using neu_den_xplot_hc_correction
      which adjusts NPHI and RHOB along a specified correction angle to account for
      hydrocarbon effects on the neutron porosity log.
    - Returns corrected NPHI and RHOB values for each input data point.
    - Points with low NPHI (<0.45) and those beyond the mineral-clay line are candidates
      for correction.
    """
    input_dict = inputs.model_dump()

    # Extract data into arrays
    data_points = input_dict["data"]
    nphi_array = np.array([p["nphi"] for p in data_points])
    rhob_array = np.array([p["rhob"] for p in data_points])

    # Apply HC correction
    nphi_corrected, rhob_corrected, _ = neu_den_xplot_hc_correction(
        nphi_array,
        rhob_array,
        dry_min1_point=input_dict["dry_sand_point"],
        dry_clay_point=input_dict["dry_clay_point"],
        water_point=input_dict["water_point"],
        corr_angle=input_dict["corr_angle"],
        buffer=input_dict["buffer"],
    )

    # Return corrected data as list of dicts
    results = []
    for nphi, rhob in zip(nphi_corrected, rhob_corrected, strict=True):
        results.append({"nphi": float(nphi), "rhob": float(rhob)})

    return results
