from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Rw input data for a single sample."""
    temp_grad: float = Field(..., description="Temperature gradient (C/km or F/100ft)")


class InputData(BaseModel):
    """Rw input data for a batch of samples, with water salinity."""
    water_salinity: float = Field(30000, description="Water salinity (ppm)")
    data: List[Data] = Field([
        Data(temp_grad=134.46224999999998),
        Data(temp_grad=134.46625),
        Data(temp_grad=134.47),
        Data(temp_grad=134.47375),
        Data(temp_grad=134.47750000000002),
        Data(temp_grad=134.48149999999998),
        Data(temp_grad=134.48525),
        Data(temp_grad=134.48900000000003),
        Data(temp_grad=134.49275),
        Data(temp_grad=134.49649999999997)
    ], description="List of temperature gradients (C/km or F/100ft)")
