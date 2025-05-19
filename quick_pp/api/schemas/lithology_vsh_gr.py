"""
Schemas for Vshale from GR log input.
"""

from pydantic import BaseModel, Field
from typing import List


class GRData(BaseModel):
    gr: float = Field(..., description="Gamma Ray value (API units)")


class LithologyVshGRInput(BaseModel):
    data: List[GRData] = Field([
        GRData(gr=121.54),
        GRData(gr=120.133),
        GRData(gr=117.059),
        GRData(gr=124.022),
        GRData(gr=125.076),
        GRData(gr=86.06),
        GRData(gr=130.02),
        GRData(gr=131.419),
        GRData(gr=103.284),
        GRData(gr=102.317),
    ], description="List of gamma ray values (API units)")
