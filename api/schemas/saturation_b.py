from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Saturation B input data for a single sample."""
    temp_grad: float = Field(..., description="Temperature gradient (C/km or F/100ft)")
    rw: float = Field(..., description="Water resistivity (ohm-m)")


class InputData(BaseModel):
    """Saturation B input data for a batch of samples."""
    data: List[Data] = Field(..., description="List of input data for B calculation")

    class Config:
        schema_extra = {
            "example": {
                'data': [
                    {'temp_grad': 44.92, 'rw': 0.34},
                    {'temp_grad': 46.08, 'rw': 0.34},
                    {'temp_grad': 45.38, 'rw': 0.34},
                    {'temp_grad': 46.49, 'rw': 0.33},
                    {'temp_grad': 46.0, 'rw': 0.34},
                    {'temp_grad': 45.31, 'rw': 0.34},
                    {'temp_grad': 45.32, 'rw': 0.34},
                    {'temp_grad': 46.18, 'rw': 0.34},
                    {'temp_grad': 46.21, 'rw': 0.33},
                    {'temp_grad': 46.61, 'rw': 0.33}
                ]
            }
        }


# For backward compatibility
EXAMPLE = InputData.Config.schema_extra["example"]
