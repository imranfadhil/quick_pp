from typing import List

from pydantic import BaseModel, Field


class Data(BaseModel):
    """Qvn input data for a single sample."""

    vclay: float = Field(..., description="Volume of clay (fraction)")
    phit: float = Field(..., description="Total porosity (fraction)")
    phit_clay: float = Field(..., description="Total porosity of the clay (fraction)")


class InputData(BaseModel):
    """Qvn input data for a batch of samples."""

    data: List[Data] = Field(
        [
            Data(vclay=0.322, phit=0.204, phit_clay=0.15),
            Data(vclay=0.608, phit=0.192, phit_clay=0.15),
            Data(vclay=0.622, phit=0.144, phit_clay=0.15),
            Data(vclay=0.768, phit=0.102, phit_clay=0.15),
            Data(vclay=0.174, phit=0.208, phit_clay=0.15),
            Data(vclay=0.018, phit=0.271, phit_clay=0.15),
            Data(vclay=0.688, phit=0.125, phit_clay=0.15),
            Data(vclay=0.493, phit=0.248, phit_clay=0.15),
            Data(vclay=0.781, phit=0.126, phit_clay=0.15),
            Data(vclay=0.731, phit=0.115, phit_clay=0.15),
        ],
        description="List of input data for multiple samples.",
    )
