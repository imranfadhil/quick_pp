from pydantic import BaseModel, Field
from typing import List, Tuple, Optional


class Data(BaseModel):
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")


class InputData(BaseModel):
    dry_sand_point: Tuple[float, float] = Field((-0.02, 2.65), description="Dry sand point (nphi, rhob)")
    dry_silt_point: Tuple[float, float] = Field((0.1, 2.68), description="Dry silt point (nphi, rhob)")
    dry_clay_point: Tuple[float, float] = Field((0.27, 2.7), description="Dry clay point (nphi, rhob)")
    fluid_point: Tuple[float, float] = Field((1.0, 1.0), description="Fluid point (nphi, rhob)")
    wet_clay_point: Optional[Tuple[float, float]] = Field((0, 0), description="Wet clay point (nphi, rhob)")
    method: str = Field('ssc', description="Method for lithology calculation")
    silt_line_angle: float = Field(117, description="Silt line angle (degrees)")
    data: List[Data] = Field([
        Data(nphi=0.3, rhob=1.85),
        Data(nphi=0.35, rhob=1.95),
        Data(nphi=0.34, rhob=1.9),
    ], description="List of data points for lithology calculation")
