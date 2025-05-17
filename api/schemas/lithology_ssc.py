"""
Schemas for sand-silt-clay lithology input.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Tuple


class SSCData(BaseModel):
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")


class LithologySSCInput(BaseModel):
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
    data: List[SSCData] = Field([
        SSCData(nphi=0.365, rhob=2.48),
        SSCData(nphi=0.17, rhob=2.304),
        SSCData(nphi=0.31, rhob=2.39),
        SSCData(nphi=0.45, rhob=1.85),
        SSCData(nphi=0.244, rhob=2.513),
        SSCData(nphi=0.207, rhob=2.454),
        SSCData(nphi=0.323, rhob=2.438),
        SSCData(nphi=0.097, rhob=2.177),
        SSCData(nphi=0.353, rhob=2.5),
        SSCData(nphi=0.208, rhob=2.487),
    ], description="List of data points for lithology calculation")
