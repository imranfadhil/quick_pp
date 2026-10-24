"""QAQC API services.

Endpoints for QAQC (Quality Assurance / Quality Control) calculations:

- Hydrocarbon correction for NPHI/RHOB crossplots

Input Pydantic models:

- `HCCorrectionInput` — Hydrocarbon correction for NPHI/RHOB data
"""

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

from quick_pp.qaqc import (
    apply_environmental_corrections,
    neu_den_xplot_hc_correction,
    normalize_by_zone,
    stratigraphic_normalization,
)

router = APIRouter(prefix="/qaqc", tags=["QAQC"])


class DataPoint(BaseModel):
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")
    vsh_gr: Optional[float] = Field(
        None, description="Calibrated Vshale from Gamma Ray (fraction)"
    )
    gr: Optional[float] = Field(None, description="Gamma Ray log (gAPI)")


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
    gr_clean: Optional[float] = Field(
        None, description="Clean sand GR endpoint (gAPI) for VSH_GR calculation"
    )
    gr_clay: Optional[float] = Field(
        None, description="Shale GR endpoint (gAPI) for VSH_GR calculation"
    )
    data: List[DataPoint] = Field(..., description="List of NPHI/RHOB data points")


class StratigraphicNormalizationInput(BaseModel):
    key_well: Optional[str] = Field(
        None,
        description="Reference key well name (uses field median per zone if None)",
    )
    zone_col: str = Field("ZONES", description="Stratigraphic zone column name")
    well_col: str = Field("WELL_NAME", description="Well identifier column name")
    quantiles: Tuple[float, float] = Field(
        (0.05, 0.95), description="Low and high quantiles for normalization"
    )
    logs: List[str] = Field(
        default_factory=lambda: ["GR", "NPHI"],
        description="Log curves to normalize",
    )
    method: str = Field(
        "quantile",
        description="Normalization method ('quantile' or 'histogram')",
    )
    data: List[dict] = Field(..., description="Multi-well log records")


class EnvironmentalCorrectionInput(BaseModel):
    bitsize_default: float = Field(8.5, description="Default bit size (inches)")
    mud_weight: float = Field(9.0, description="Drilling mud weight (ppg)")
    surface_temp: float = Field(75.0, description="Surface temperature (deg F)")
    temp_grad: float = Field(1.5, description="Geothermal gradient (deg F / 100 ft)")
    salinity_ppm: float = Field(0.0, description="Salinity (ppm NaCl)")
    rho_mudcake: float = Field(1.5, description="Mudcake bulk density (g/cm^3)")
    data: List[dict] = Field(..., description="Well log records")


@router.post("/hc_correction", tags=["HC Correction"])
async def apply_hc_correction(inputs: HCCorrectionInput):
    """
    Apply hydrocarbon correction to NPHI and RHOB data.
    """
    input_dict = inputs.model_dump()

    # Extract data into arrays
    data_points = input_dict["data"]
    nphi_array = np.array([p["nphi"] for p in data_points], dtype=float)
    rhob_array = np.array([p["rhob"] for p in data_points], dtype=float)

    vsh_gr_array: Optional[np.ndarray] = None
    if any(p.get("vsh_gr") is not None for p in data_points):
        vsh_gr_array = np.clip(
            np.array(
                [
                    float(p["vsh_gr"]) if p.get("vsh_gr") is not None else 0.0
                    for p in data_points
                ],
                dtype=float,
            ),
            0.0,
            1.0,
        )
    elif (
        input_dict.get("gr_clean") is not None
        and input_dict.get("gr_clay") is not None
        and abs(float(input_dict["gr_clay"]) - float(input_dict["gr_clean"])) > 1e-6
        and any(p.get("gr") is not None for p in data_points)
    ):
        gr_clean = float(input_dict["gr_clean"])
        gr_clay = float(input_dict["gr_clay"])
        gr_arr = np.array(
            [
                float(p["gr"]) if p.get("gr") is not None else gr_clean
                for p in data_points
            ],
            dtype=float,
        )
        vsh_gr_array = np.clip((gr_arr - gr_clean) / (gr_clay - gr_clean), 0.0, 1.0)

    # Apply HC correction
    nphi_corrected, rhob_corrected, hc_flag = neu_den_xplot_hc_correction(
        nphi_array,
        rhob_array,
        vsh_gr=vsh_gr_array,
        dry_min1_point=input_dict["dry_sand_point"],
        dry_clay_point=input_dict["dry_clay_point"],
        water_point=input_dict["water_point"],
        corr_angle=input_dict["corr_angle"],
        buffer=input_dict["buffer"],
    )

    # Return corrected data as list of dicts
    results = []
    for nphi, rhob, flag in zip(nphi_corrected, rhob_corrected, hc_flag, strict=True):
        results.append(
            {"nphi": float(nphi), "rhob": float(rhob), "hc_flag": float(flag)}
        )

    return results


@router.post("/normalize_by_zone", tags=["Normalization"])
@router.post("/stratigraphic_normalization", tags=["Normalization"])
async def apply_stratigraphic_normalization(inputs: StratigraphicNormalizationInput):
    """Apply stratigraphic (zone-by-zone) multi-well quantile or histogram normalization."""
    df = pd.DataFrame(inputs.data)
    norm_df, report_df = stratigraphic_normalization(
        df,
        key_well=inputs.key_well,
        zone_col=inputs.zone_col,
        quantiles=inputs.quantiles,
        logs=inputs.logs,
        well_col=inputs.well_col,
        method=inputs.method,
        return_report=True,
    )
    return {
        "data": norm_df.replace({np.nan: None}).to_dict(orient="records"),
        "report": report_df.replace({np.nan: None}).to_dict(orient="records"),
    }


@router.post("/environmental_corrections", tags=["Environmental Corrections"])
async def run_environmental_corrections(inputs: EnvironmentalCorrectionInput):
    """Apply borehole environmental and invasion corrections to well log data."""
    df = pd.DataFrame(inputs.data)
    corr_df = apply_environmental_corrections(
        df,
        bitsize_default=inputs.bitsize_default,
        mud_weight=inputs.mud_weight,
        surface_temp=inputs.surface_temp,
        temp_grad=inputs.temp_grad,
        salinity_ppm=inputs.salinity_ppm,
        rho_mudcake=inputs.rho_mudcake,
    )
    return {"data": corr_df.replace({np.nan: None}).to_dict(orient="records")}

