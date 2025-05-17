from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Qv input data for a single sample."""
    vcld: float = Field(..., description="Volume of clay (fraction)")
    phit: float = Field(..., description="Total porosity (fraction)")


class InputData(BaseModel):
    """Qv input data for a batch of samples, with clay properties."""
    data: List[Data] = Field(..., description="List of Qv input data")
    rho_clay: float = Field(..., description="Clay density (g/cm^3)")
    cec_clay: float = Field(..., description="Cation exchange capacity of clay (meq/g)")

    class Config:
        schema_extra = {
            "example": {
                'rho_clay': 2.65,
                'cec_clay': 0.062,
                'data': [
                    {'vcld': 0.322, 'phit': 0.204},
                    {'vcld': 0.608, 'phit': 0.192},
                    {'vcld': 0.622, 'phit': 0.144},
                    {'vcld': 0.768, 'phit': 0.102},
                    {'vcld': 0.174, 'phit': 0.208},
                    {'vcld': 0.018, 'phit': 0.271},
                    {'vcld': 0.688, 'phit': 0.125},
                    {'vcld': 0.493, 'phit': 0.248},
                    {'vcld': 0.781, 'phit': 0.126},
                    {'vcld': 0.731, 'phit': 0.115}
                ]
            }
        }


# For backward compatibility
EXAMPLE = InputData.Config.schema_extra["example"]
