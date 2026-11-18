"""
Schemas for binary sand-shale lithology input.
"""
from typing import List, Optional, Tuple

from pydantic import BaseModel, Field


class SandShaleData(BaseModel):
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")


class LithologySandShaleInput(BaseModel):
    dry_sand_point: Tuple[Optional[float], Optional[float]] = Field(
        (-0.02, 2.65), description="Dry sand point (nphi, rhob)"
    )
    dry_clay_point: Tuple[Optional[float], Optional[float]] = Field(
        (0.33, 2.7), description="Dry clay point (nphi, rhob)"
    )
    fluid_point: Tuple[Optional[float], Optional[float]] = Field(
        (1.0, 1.0), description="Fluid point (nphi, rhob)"
    )
    wet_clay_point: Tuple[Optional[float], Optional[float]] = Field(
        (None, None), description="Wet clay point (nphi, rhob)"
    )
    method: str = Field("sand_shale", description="Method for lithology calculation")
    data: List[SandShaleData] = Field(
        default_factory=list,
        description="List of data points for lithology calculation",
    )
