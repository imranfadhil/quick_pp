from typing import List

from pydantic import BaseModel, Field


class Data(BaseModel):
    """Shale porosity input data for a single sample."""

    nphi: float = Field(
        ..., description="Neutron porosity (hydrocarbon corrected) [fraction]"
    )
    phit: float = Field(..., description="Total porosity [fraction]")


class InputData(BaseModel):
    """Shale porosity input data for a batch of samples."""

    data: List[Data] = Field(
        [
            Data(nphi=0.35, phit=0.22),
            Data(nphi=0.40, phit=0.18),
            Data(nphi=0.38, phit=0.20),
            Data(nphi=0.45, phit=0.15),
            Data(nphi=0.30, phit=0.25),
        ],
        description="List of input data for multiple samples.",
    )
