"""Pydantic schema for the continuous apparent fluid density (RHO_FL) engine."""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FluidDensityConfig(BaseModel):
    """Configuration for the 3-tiered continuous apparent fluid density solver.

    ``mode`` selects the engine:

    * ``static`` - a single scalar ``static_rho_fl`` (legacy behaviour).
    * ``zonal_coupled`` - continuous Batzle & Wang filtrate profile with a zonal
      constant residual hydrocarbon saturation (Tier 3).
    * ``iterative_sxo`` - the full 3-tier coupled solver (Rxo -> Sw^c -> zonal).
    """

    mode: Literal["static", "zonal_coupled", "iterative_sxo"] = Field(
        default="static",
        description="Method for estimating apparent fluid density (rho_fl).",
    )
    static_rho_fl: float = Field(
        default=1.0, ge=0.1, le=1.35, description="Static fluid density (g/cc)."
    )
    mud_type: Literal["WBM", "OBM", "SBM"] = Field(
        default="WBM", description="Mud system driving the filtrate density."
    )
    mud_salinity_ppm: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=350000.0,
        description="Mud-filtrate salinity (ppm NaCl). Overrides rmf_surface.",
    )
    rmf_surface: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Surface mud-filtrate resistivity (ohm.m) used to back-calculate salinity.",
    )
    rmf_meas_temp_c: float = Field(
        default=25.0,
        ge=0.0,
        le=200.0,
        description="Temperature of the rmf_surface measurement (degC).",
    )
    obm_filtrate_density: float = Field(
        default=0.82,
        ge=0.65,
        le=0.95,
        description="Surface base-oil density for OBM/SBM filtrate (g/cc).",
    )
    rxo_curve: Optional[str] = Field(
        default=None,
        description="Shallow resistivity curve mnemonic (e.g. MSFL, RXOZ). None disables Tier 1.",
    )
    rt_curve: Optional[str] = Field(
        default="RT",
        description="Deep resistivity curve mnemonic for the Sw^c fallback.",
    )
    flush_exponent: float = Field(
        default=0.20,
        ge=0.05,
        le=0.50,
        description="Poupon exponent c in Sxo = Sw^c.",
    )
    a: float = Field(default=1.0, gt=0.0, description="Archie tortuosity factor.")
    m: float = Field(default=2.0, gt=0.0, description="Cementation exponent.")
    n: float = Field(default=2.0, gt=0.0, description="Saturation exponent.")
    rw: Optional[float] = Field(
        default=None,
        gt=0.0,
        description="Formation water resistivity (ohm.m). None derives it from salinity/temperature.",
    )
    water_salinity_ppm: float = Field(
        default=30000.0,
        ge=0.0,
        le=350000.0,
        description="Formation water salinity (ppm) used to derive Rw when rw is None.",
    )
    oil_api: float = Field(default=35.0, ge=5.0, le=70.0)
    gas_gravity: float = Field(default=0.65, ge=0.55, le=1.50)
    gor_scf_stb: float = Field(
        default=500.0, ge=0.0, description="Solution GOR (scf/stb)."
    )
    default_shr_gas: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Tier 3 residual gas saturation."
    )
    default_shr_oil: float = Field(
        default=0.15, ge=0.0, le=1.0, description="Tier 3 residual oil saturation."
    )
    temp_grad: float = Field(
        default=30.0, ge=5.0, le=100.0, description="Temperature gradient (degC/km)."
    )
    surf_temp: float = Field(
        default=25.0, ge=0.0, le=60.0, description="Surface temperature (degC)."
    )
    pres_grad: float = Field(
        default=10.2, ge=5.0, le=25.0, description="Pressure gradient (MPa/km)."
    )
    surf_pres: float = Field(
        default=0.1013, ge=0.0, le=5.0, description="Absolute surface pressure (MPa)."
    )
    max_iter: int = Field(
        default=8, ge=1, le=50, description="Maximum fixed-point iterations."
    )
    tol: float = Field(
        default=1e-4, gt=0.0, description="Convergence tolerance on max PHID change."
    )
