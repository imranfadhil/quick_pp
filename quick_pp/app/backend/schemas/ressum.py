from pydantic import BaseModel, Field
from typing import List


class Data(BaseModel):
    """Schema for a single reservoir data point."""
    depth: float = Field(..., description="Depth of the sample")
    vcld: float = Field(..., description="Volume of clay")
    phit: float = Field(..., description="Total porosity")
    swt: float = Field(..., description="Water saturation")
    perm: float = Field(..., description="Permeability")
    zones: str = Field(..., description="Zone identifier")


class InputData(BaseModel):
    """Input schema for reservoir summary, including cut-offs and data list."""
    cut_offs: dict = Field(dict(VSHALE=0.3, PHIT=0.1, SWT=0.75), description="Cut-off values for reservoir properties")
    data: List[Data] = Field([
        Data(depth=3898.24, vcld=0.702, phit=0.095, swt=1.0, perm=0, zones='A'),
        Data(depth=3954.932, vcld=0.671, phit=0.108, swt=1.0, perm=0, zones='A'),
        Data(depth=3804.056, vcld=0.698, phit=0.121, swt=1.0, perm=0, zones='A'),
        Data(depth=3742.639, vcld=0.663, phit=0.149, swt=1.0, perm=0, zones='B'),
        Data(depth=3641.75, vcld=0.635, phit=0.121, swt=1.0, perm=0, zones='B'),
        Data(depth=3919.423, vcld=0.036, phit=0.483, swt=0.281, perm=0, zones='B'),
        Data(depth=4014.368, vcld=0.352, phit=0.178, swt=0.801, perm=0, zones='B'),
        Data(depth=3919.88, vcld=0.036, phit=0.483, swt=0.248, perm=0, zones='C'),
        Data(depth=3864.712, vcld=0.007, phit=0.255, swt=0.343, perm=0, zones='C'),
        Data(depth=4062.527, vcld=0.638, phit=0.188, swt=1.0, perm=0, zones='C')
    ], description="List of reservoir data points")
