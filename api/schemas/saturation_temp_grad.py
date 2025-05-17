from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Temperature gradient input data for a single sample."""
    tvdss: float = Field(..., description="True vertical depth subsea (meters or feet)")


class InputData(BaseModel):
    """Temperature gradient input data for a batch of samples, with measurement system."""
    data: List[Data] = Field(..., description="List of TVDSS input data")
    meas_system: str = Field('metric', description="Measurement system: 'metric' or 'imperial'")

    class Config:
        schema_extra = {
            "example": {
                'meas_system': 'metric',
                'data': [
                    {'tvdss': 4098.49},
                    {'tvdss': 4098.65},
                    {'tvdss': 4098.8},
                    {'tvdss': 4098.95},
                    {'tvdss': 4099.1},
                    {'tvdss': 4099.26},
                    {'tvdss': 4099.41},
                    {'tvdss': 4099.56},
                    {'tvdss': 4099.71},
                    {'tvdss': 4099.86}
                ]
            }
        }


# For backward compatibility
EXAMPLE = InputData.Config.schema_extra["example"]
