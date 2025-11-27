from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Qv input data for a single sample."""
    vcld: float = Field(..., description="Volume of clay (fraction)")
    phit: float = Field(..., description="Total porosity (fraction)")


class InputData(BaseModel):
    """Qv input data for a batch of samples, with clay properties."""
    rho_clay: float = Field(2.65, description="Density of clay (g/cm^3)")
    cec_clay: float = Field(.062, description="Cation exchange capacity of clay (meq/g)")
    data: List[Data] = Field([
        Data(vcld=0.322, phit=0.204),
        Data(vcld=0.608, phit=0.192),
        Data(vcld=0.622, phit=0.144),
        Data(vcld=0.768, phit=0.102),
        Data(vcld=0.174, phit=0.208),
        Data(vcld=0.018, phit=0.271),
        Data(vcld=0.688, phit=0.125),
        Data(vcld=0.493, phit=0.248),
        Data(vcld=0.781, phit=0.126),
        Data(vcld=0.731, phit=0.115)
    ], description="List of input data for multiple samples.")
