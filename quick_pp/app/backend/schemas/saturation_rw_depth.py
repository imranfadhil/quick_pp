from typing import List

from pydantic import BaseModel, Field


class Data(BaseModel):
    """Depth sample for Rw estimation."""
    tvd: float = Field(..., description="True vertical depth (m)")


class InputData(BaseModel):
    """Rw-at-depth input: salinity, geothermal model and a list of depths."""
    water_salinity: float = Field(35000, description="Water salinity (ppm NaCl)")
    surface_temp: float = Field(25.0, description="Surface temperature (degC metric, degF imperial)")
    gradient: float = Field(30.0, description="Geothermal gradient (degC/km metric, degF/1000ft imperial)")
    meas_system: str = Field('metric', description="Measurement system: 'metric' or 'imperial'")
    data: List[Data] = Field([Data(tvd=1500.0), Data(tvd=2000.0)], description="Depths (m)")
