from typing import List, Optional

from pydantic import BaseModel, Field


class MultiMineralData(BaseModel):
    gr: float = Field(..., description="Gamma Ray log (GAPI)")
    nphi: float = Field(..., description="Neutron porosity (fraction)")
    rhob: float = Field(..., description="Bulk density (g/cm^3)")
    pef: Optional[float] = Field(
        None, description="Photoelectric Factor log (barns/electron)"
    )
    dtc: Optional[float] = Field(None, description="Compressional slowness log (us/ft)")


class LithologyMultiMineralInput(BaseModel):
    minerals: List[str] = Field(
        ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"],
        description="List of minerals for the optimization",
    )
    porosity_method: str = Field(
        "density",
        description="Method for initial porosity estimation ('density', 'neutron_density', 'sonic')",
    )
    auto_scale: bool = Field(
        True,
        description="If True, automatically calculates scaling factors for robustness",
    )
    data: List[MultiMineralData] = Field(
        [
            MultiMineralData(gr=85.0, nphi=0.25, rhob=2.45, pef=2.5, dtc=65.0),
            MultiMineralData(gr=110.0, nphi=0.35, rhob=2.35, pef=3.0, dtc=80.0),
            MultiMineralData(gr=120.0, nphi=0.20, rhob=2.55, pef=2.0, dtc=55.0),
        ],
        description="List of data points for lithology calculation",
    )
