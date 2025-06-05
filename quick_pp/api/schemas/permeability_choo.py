from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Input record for Choo permeability estimation."""
    vcld: float = Field(..., description="Volume of clay (fraction)")
    vsilt: float = Field(..., description="Volume of silt (fraction)")
    phit: float = Field(..., description="Total porosity (fraction)")


class InputData(BaseModel):
    """Input data wrapper for Choo permeability endpoint."""
    data: List[Data] = Field([
        Data(vcld=0.662, vsilt=0.264, phit=0.102),
        Data(vcld=0.618, vsilt=0.22, phit=0.147),
        Data(vcld=0.301, vsilt=0.187, phit=0.21),
        Data(vcld=0.684, vsilt=0.226, phit=0.098),
        Data(vcld=0.663, vsilt=0.259, phit=0.113),
        Data(vcld=0.553, vsilt=0.228, phit=0.258),
        Data(vcld=0.725, vsilt=0.129, phit=0.145),
        Data(vcld=0.806, vsilt=0.103, phit=0.108),
        Data(vcld=0.021, vsilt=0.013, phit=0.273),
        Data(vcld=0.052, vsilt=0.032, phit=0.448)
    ], description="List of input data for Choo permeability estimation.")
