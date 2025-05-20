from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Temperature gradient input data for a single sample."""
    tvdss: float = Field(..., description="True vertical depth subsea (meters or feet)")


class InputData(BaseModel):
    """Temperature gradient input data for a batch of samples, with measurement system."""
    meas_system: str = Field('metric', description="Measurement system: 'metric' or 'imperial'")
    data: List[Data] = Field([
        Data(tvdss=4098.49),
        Data(tvdss=4098.65),
        Data(tvdss=4098.8),
        Data(tvdss=4098.95),
        Data(tvdss=4099.1),
        Data(tvdss=4099.26),
        Data(tvdss=4099.41),
        Data(tvdss=4099.56),
        Data(tvdss=4099.71),
        Data(tvdss=4099.86)
    ], description="List of true vertical depths.")
