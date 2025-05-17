from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Rw input data for a single sample."""
    temp_grad: float = Field(..., description="Temperature gradient (C/km or F/100ft)")


class InputData(BaseModel):
    """Rw input data for a batch of samples, with water salinity."""
    data: List[Data] = Field(..., description="List of Rw input data")
    water_salinity: float = Field(..., description="Water salinity (ppm)")

    class Config:
        schema_extra = {
            "example": {
                'water_salinity': 30000.0,
                'data': [
                    {"temp_grad": 134.46224999999998},
                    {"temp_grad": 134.46625},
                    {"temp_grad": 134.47},
                    {"temp_grad": 134.47375},
                    {"temp_grad": 134.47750000000002},
                    {"temp_grad": 134.48149999999998},
                    {"temp_grad": 134.48525},
                    {"temp_grad": 134.48900000000003},
                    {"temp_grad": 134.49275},
                    {"temp_grad": 134.49649999999997}
                ]
            }
        }


# For backward compatibility
EXAMPLE = InputData.Config.schema_extra["example"]
