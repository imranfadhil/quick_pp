"""
Schemas for hydrocarbon correction lithology input.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class HCCorrectionData(BaseModel):
    gr: float = Field(..., description="Gamma Ray value (API units)")
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")


class LithologyHCCorrectionInput(BaseModel):
    dry_sand_point: Tuple[Optional[float], Optional[float]] = Field(
        (-0.02, 2.65), description="Dry sand point (nphi, rhob)")
    dry_silt_point: Tuple[Optional[float], Optional[float]] = Field(
        (0, 2.68), description="Dry silt point (nphi, rhob)")
    dry_clay_point: Tuple[Optional[float], Optional[float]] = Field(
        (0.33, 2.7), description="Dry clay point (nphi, rhob)")
    fluid_point: Tuple[Optional[float], Optional[float]] = Field(
        (1.0, 1.0), description="Fluid point (nphi, rhob)")
    wet_clay_point: Tuple[Optional[float], Optional[float]] = Field(
        (0, 0), description="Wet clay point (nphi, rhob)")
    method: str = Field('ssc', description="Method for lithology calculation")
    silt_line_angle: float = Field(117, description="Silt line angle (degrees)")
    corr_angle: float = Field(50, description="Correction angle (degrees)")
    data: List[HCCorrectionData] = Field([
        HCCorrectionData(gr=117.893, nphi=0.285, rhob=2.486),
        HCCorrectionData(gr=116.065, nphi=0.245, rhob=2.441),
        HCCorrectionData(gr=122.852, nphi=0.306, rhob=2.484),
        HCCorrectionData(gr=126.424, nphi=0.357, rhob=2.536),
        HCCorrectionData(gr=111.248, nphi=0.295, rhob=2.498),
        HCCorrectionData(gr=112.464, nphi=0.278, rhob=2.531),
        HCCorrectionData(gr=128.918, nphi=0.388, rhob=2.093),
        HCCorrectionData(gr=60.707, nphi=0.067, rhob=2.073),
        HCCorrectionData(gr=125.888, nphi=0.327, rhob=2.448),
        HCCorrectionData(gr=119.909, nphi=0.313, rhob=2.55),
    ], description="List of data points for lithology calculation")
