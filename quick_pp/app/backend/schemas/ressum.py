from typing import List

from pydantic import BaseModel, Field


class Data(BaseModel):
    """Schema for a single reservoir data point."""

    depth: float = Field(..., description="Depth of the sample")
    vclay: float = Field(..., description="Volume of clay")
    phit: float = Field(..., description="Total porosity")
    swt: float = Field(..., description="Water saturation")
    perm: float = Field(..., description="Permeability")
    zones: str = Field(..., description="Zone identifier")


class InputData(BaseModel):
    """Input schema for reservoir summary, including cut-offs and data list."""

    cut_offs: dict = Field(
        {"VSHALE": 0.3, "PHIT": 0.1, "SWT": 0.75},
        description="Cut-off values for reservoir properties",
    )
    data: List[Data] = Field(
        [
            Data(depth=3898.24, vclay=0.702, phit=0.095, swt=1.0, perm=0, zones="A"),
            Data(depth=3954.932, vclay=0.671, phit=0.108, swt=1.0, perm=0, zones="A"),
            Data(depth=3804.056, vclay=0.698, phit=0.121, swt=1.0, perm=0, zones="A"),
            Data(depth=3742.639, vclay=0.663, phit=0.149, swt=1.0, perm=0, zones="B"),
            Data(depth=3641.75, vclay=0.635, phit=0.121, swt=1.0, perm=0, zones="B"),
            Data(depth=3919.423, vclay=0.036, phit=0.483, swt=0.281, perm=0, zones="B"),
            Data(depth=4014.368, vclay=0.352, phit=0.178, swt=0.801, perm=0, zones="B"),
            Data(depth=3919.88, vclay=0.036, phit=0.483, swt=0.248, perm=0, zones="C"),
            Data(depth=3864.712, vclay=0.007, phit=0.255, swt=0.343, perm=0, zones="C"),
            Data(depth=4062.527, vclay=0.638, phit=0.188, swt=1.0, perm=0, zones="C"),
        ],
        description="List of reservoir data points",
    )


class UncertaintyData(BaseModel):
    """Schema for a single well log row for Latin Hypercube uncertainty propagation."""

    depth: float = Field(..., description="Depth of the sample")
    nphi: float | None = Field(None, description="Neutron porosity [v/v]")
    rhob: float | None = Field(None, description="Bulk density [g/cc]")
    rt: float | None = Field(None, description="True resistivity [ohm.m]")
    gr: float | None = Field(None, description="Gamma ray [GAPI]")
    vclay: float | None = Field(None, description="Volume of clay [v/v]")
    phit: float | None = Field(None, description="Total porosity [v/v]")
    swt: float | None = Field(None, description="Water saturation [v/v]")
    perm: float | None = Field(10.0, description="Permeability [mD]")
    zones: str = Field("ALL", description="Zone identifier")


class UncertaintyInputData(BaseModel):
    """Input schema for stochastic reservoir summary and log-level LHS uncertainty."""

    model: str = Field(
        "ssc",
        description="Lithology model: 'ssc' (SandSiltClay) or 'carbonate' (Carbonate)",
    )
    saturation_method: str = Field(
        "waxman_smits",
        description="Saturation equation: 'waxman_smits' or 'archie'",
    )
    n_samples: int = Field(100, ge=5, le=2000, description="Number of LHS realizations")
    param_bounds: dict = Field(
        default_factory=lambda: {
            "a": [0.8, 1.2, 1.0, 0.1],
            "m": [1.8, 2.2, 2.0, 0.1],
            "n": [1.8, 2.2, 2.0, 0.1],
            "rw": [0.02, 0.15, 0.08, 0.03],
            "rho_ma": [2.63, 2.75, 2.68, 0.03],
            "rho_sh": [2.55, 2.78, 2.68, 0.05],
        },
        description="Parameter uncertainty bounds for a, m, n, rw, rho_ma, rho_sh",
    )
    area_bound: list[float] = Field(
        default_factory=lambda: [500.0, 2500.0, 1500.0, 400.0],
        description="Area bound [min, max, mean, std] in acres",
    )
    volume_factor_bound: list[float] = Field(
        default_factory=lambda: [1.1, 1.4],
        description="Formation volume factor bound [min, max]",
    )
    cut_offs: dict = Field(
        default_factory=lambda: {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9},
        description="Cut-off values for reservoir properties",
    )
    seed: int | None = Field(42, description="Random seed for LHS reproducibility")
    data: List[UncertaintyData] = Field(..., description="List of well log rows")

