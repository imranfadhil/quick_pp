from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Input record for permeability estimation (other models)."""
    phit: float = Field(..., description="Total porosity (fraction)")
    swirr: float = Field(..., description="Irreducible water saturation (fraction)")


class InputData(BaseModel):
    """Input data wrapper for permeability endpoints (other models)."""
    data: List[Data] = Field([
        Data(phit=0.328, swirr=0.003),
        Data(phit=0.114, swirr=0.016),
        Data(phit=0.111, swirr=0.03),
        Data(phit=0.278, swirr=0.006),
        Data(phit=0.245, swirr=0.013),
        Data(phit=0.116, swirr=0.018),
        Data(phit=0.131, swirr=0.016),
        Data(phit=0.109, swirr=0.042),
        Data(phit=0.2, swirr=0.023),
        Data(phit=0.143, swirr=0.01)
    ], description="List of input data for permeability estimation using other models.")
