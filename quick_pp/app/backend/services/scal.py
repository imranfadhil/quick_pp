"""Special Core Analysis (SCAL) & Relative Permeability API services.

Endpoints for two-phase relative permeability evaluation (Corey, Brooks-Corey, LET),
experimental SCAL curve fitting, Eclipse/Intersect SWOF table generation,
laboratory-to-reservoir capillary pressure conversion, and core plug corrections.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quick_pp.rock_physics.geomechanics import estimate_log_stress_profile
from quick_pp.scal import (
    brooks_corey_relperm,
    corey_relperm,
    correct_core_data,
    fit_grain_density_correlation,
    fit_grain_density_correlation_multivariate,
    fit_relperm,
    generate_swof_table,
    grain_density_from_correlation,
    grain_density_from_correlation_multivariate,
    let_relperm,
    pc_lab_to_reservoir,
)

router = APIRouter(prefix="/scal", tags=["SCAL"])


class RelPermEvalInput(BaseModel):
    model: str = Field(
        "let",
        description="Relative permeability formulation ('corey', 'brooks_corey', or 'let')",
    )
    sw: Optional[List[float]] = Field(
        default=None,
        description="Water saturation array (v/v). If omitted, generates 25 points from swirr to 1 - sor.",
    )
    swirr: float = Field(
        0.15, ge=0.0, lt=1.0, description="Irreducible water saturation"
    )
    sor: float = Field(
        0.20, ge=0.0, lt=1.0, description="Residual hydrocarbon saturation"
    )
    sorw: Optional[float] = Field(
        default=None, description="Optional alias for residual oil saturation (sor)"
    )
    krw_max: float = Field(
        0.30, gt=0.0, le=1.0, description="Endpoint water rel-perm at 1 - sor"
    )
    kro_max: float = Field(
        0.90, gt=0.0, le=1.0, description="Endpoint hydrocarbon rel-perm at swirr"
    )
    nw: float = Field(3.0, gt=0.0, description="Corey water exponent")
    no: float = Field(2.5, gt=0.0, description="Corey hydrocarbon exponent")
    lam: float = Field(
        2.0, gt=0.0, description="Brooks-Corey pore size distribution index"
    )
    lw: float = Field(2.0, gt=0.0, description="LET water lower parameter L_w")
    ew: float = Field(1.5, gt=0.0, description="LET water elevation parameter E_w")
    tw: float = Field(2.0, gt=0.0, description="LET water top parameter T_w")
    lo: float = Field(2.0, gt=0.0, description="LET hydrocarbon lower parameter L_o")
    eo: float = Field(
        1.5, gt=0.0, description="LET hydrocarbon elevation parameter E_o"
    )
    to: float = Field(2.0, gt=0.0, description="LET hydrocarbon top parameter T_o")


class RelPermFitInput(BaseModel):
    model: str = Field(
        "corey",
        description="Model to fit ('corey' or 'let')",
    )
    sw: List[float] = Field(..., description="Measured water saturation values (v/v)")
    krw: Optional[List[float]] = Field(
        default=None, description="Measured water relative permeability values"
    )
    kro: Optional[List[float]] = Field(
        default=None, description="Measured hydrocarbon relative permeability values"
    )
    swirr: Optional[float] = Field(
        default=None, description="Optional fixed irreducible water saturation"
    )
    sor: Optional[float] = Field(
        default=None, description="Optional fixed residual hydrocarbon saturation"
    )


class SWOFTableInput(BaseModel):
    n_points: int = Field(
        21, ge=5, le=200, description="Number of saturation table rows"
    )
    model: str = Field(
        "corey",
        description="Relative permeability model ('corey', 'brooks_corey', or 'let')",
    )
    swirr: float = Field(
        0.15, ge=0.0, lt=1.0, description="Irreducible water saturation"
    )
    sor: float = Field(0.20, ge=0.0, lt=1.0, description="Residual oil saturation")
    krw_max: float = Field(0.30, gt=0.0, le=1.0, description="Endpoint krw at 1 - sor")
    kro_max: float = Field(0.90, gt=0.0, le=1.0, description="Endpoint kro at swirr")
    nw: float = Field(3.0, gt=0.0, description="Corey water exponent")
    no: float = Field(2.5, gt=0.0, description="Corey oil exponent")
    lam: float = Field(2.0, gt=0.0, description="Brooks-Corey lambda exponent")
    lw: float = Field(2.0, gt=0.0, description="LET water L_w")
    ew: float = Field(1.5, gt=0.0, description="LET water E_w")
    tw: float = Field(2.0, gt=0.0, description="LET water T_w")
    lo: float = Field(2.0, gt=0.0, description="LET oil L_o")
    eo: float = Field(1.5, gt=0.0, description="LET oil E_o")
    to: float = Field(2.0, gt=0.0, description="LET oil T_o")
    pc_entry: float = Field(1.5, ge=0.0, description="Capillary entry pressure (psi)")
    pc_exponent: float = Field(
        1.5, gt=0.0, description="Capillary pressure pore-throat exponent"
    )
    pc_max: float = Field(
        50.0, gt=0.0, description="Maximum capillary pressure at swirr (psi)"
    )


class CoreCorrectionsInput(BaseModel):
    net_confining_stress: Optional[Union[float, str]] = Field(
        default=2500.0,
        description="Effective reservoir net confining stress (psi) or column name",
    )
    mean_pore_pressure: Optional[Union[float, str]] = Field(
        default=30.0,
        description="Mean flowing gas pressure (psi) for Klinkenberg correction or column name",
    )
    cbw: Optional[Union[bool, float, str]] = Field(
        default=None,
        description="Clay-bound water fraction, column name, or True to infer from PHIT - PHIE",
    )
    compaction_coeff: float = Field(
        4e-6, gt=0.0, description="Pore volume compaction coefficient (1/psi)"
    )
    perm_compaction_coeff: Optional[float] = Field(
        default=None,
        description="Permeability compaction coefficient (1/psi); defaults to 5 * compaction_coeff if omitted",
    )
    ambient_stress: float = Field(
        default=400.0, ge=0.0, description="Ambient laboratory confining stress (psi)"
    )
    compaction_method: str = Field(
        default="exponential",
        description="Overburden compaction model: 'exponential' or 'power_law'",
    )
    klinkenberg_method: str = Field(
        default="jones",
        description="Klinkenberg slippage correlation: 'jones' or 'heithaus'",
    )
    oven_dried: bool = Field(
        False, description="Whether core plugs were oven-dried prior to Dean-Stark"
    )
    replace_ori: bool = Field(
        False,
        description="Overwrite CPORE/CPERM in-place instead of preserving _AMB/_AIR columns",
    )
    data: List[Dict[str, Any]] = Field(..., description="Core/log records")


class StressProfileInput(BaseModel):
    md: List[Optional[float]] = Field(
        ..., description="Measured depth aligned with the log arrays"
    )
    tvd: List[Optional[float]] = Field(
        ..., description="Positive-down true vertical depth aligned with the log arrays"
    )
    rhob: List[Optional[float]] = Field(..., description="Bulk density [g/cm³]")
    dtc: Optional[List[Optional[float]]] = Field(
        default=None, description="Compressional sonic slowness"
    )
    dts: Optional[List[Optional[float]]] = Field(
        default=None, description="Shear sonic slowness; same units as DTC"
    )
    depth_unit: str = Field("m", description="'m' or 'ft', matching TVD")
    pore_pressure_gradient: float = Field(
        1.42, ge=0.0, description="Pore-pressure gradient [psi per selected depth unit]"
    )
    biot_coefficient: float = Field(1.0, ge=0.0, le=1.0)
    poisson_ratio: float = Field(
        0.25,
        ge=0.0,
        lt=0.5,
        description="Fallback Poisson's ratio when sonic data are invalid",
    )


class PcLabToReservoirInput(BaseModel):
    pc_lab: List[float] = Field(..., description="Laboratory capillary pressure (psi)")
    ift_lab: float = Field(
        72.0, gt=0.0, description="Laboratory interfacial tension (dynes/cm)"
    )
    theta_lab: float = Field(
        0.0, ge=0.0, le=180.0, description="Laboratory contact angle (deg)"
    )
    ift_res: float = Field(
        30.0, gt=0.0, description="Reservoir interfacial tension (dynes/cm)"
    )
    theta_res: float = Field(
        30.0, ge=0.0, le=180.0, description="Reservoir contact angle (deg)"
    )
    phi_amb: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="Optional ambient porosity (v/v) for Leverett-J stress correction",
    )
    phi_res: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="Optional reservoir porosity (v/v) for Leverett-J stress correction",
    )
    k_amb: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="Optional ambient permeability (mD) for Leverett-J stress correction",
    )
    k_res: Optional[Union[float, List[float]]] = Field(
        default=None,
        description="Optional reservoir permeability (mD) for Leverett-J stress correction",
    )


@router.post(
    "/relperm",
    summary="Evaluate Two-Phase Relative Permeability Curves (Corey, Brooks-Corey, LET)",
    operation_id="evaluate_scal_relperm",
)
async def evaluate_relperm_endpoint(inputs: RelPermEvalInput):
    """Compute water and hydrocarbon relative permeability curves using Corey, Brooks-Corey, or LET."""
    try:
        sor_val = inputs.sorw if inputs.sorw is not None else inputs.sor
        if inputs.sw is not None:
            sw_arr = np.asarray(inputs.sw, dtype=float)
        else:
            sw_arr = np.linspace(inputs.swirr, 1.0 - sor_val, 25)

        m = inputs.model.strip().lower()
        if m == "corey":
            res = corey_relperm(
                sw_arr,
                swirr=inputs.swirr,
                sor=sor_val,
                krw_max=inputs.krw_max,
                kro_max=inputs.kro_max,
                nw=inputs.nw,
                no=inputs.no,
            )
        elif m in ("brooks_corey", "brooks-corey", "bc"):
            res = brooks_corey_relperm(
                sw_arr,
                swirr=inputs.swirr,
                sor=sor_val,
                krw_max=inputs.krw_max,
                kro_max=inputs.kro_max,
                lam=inputs.lam,
            )
        elif m == "let":
            res = let_relperm(
                sw_arr,
                swirr=inputs.swirr,
                sor=sor_val,
                krw_max=inputs.krw_max,
                kro_max=inputs.kro_max,
                lw=inputs.lw,
                ew=inputs.ew,
                tw=inputs.tw,
                lo=inputs.lo,
                eo=inputs.eo,
                to=inputs.to,
            )
        else:
            raise ValueError(
                f"Unsupported model '{inputs.model}'. Choose 'corey', 'brooks_corey', or 'let'."
            )

        df_out = pd.DataFrame(
            {
                "sw": np.asarray(res["sw"], dtype=float),
                "swn": np.asarray(res["swn"], dtype=float),
                "krw": np.asarray(res["krw"], dtype=float),
                "kro": np.asarray(res["kro"], dtype=float),
                "fw": np.asarray(res["fw"], dtype=float),
            }
        )
        return {
            "model": m,
            "data": df_out.replace({np.nan: None}).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error evaluating relative permeability: {e}"
        ) from e


@router.post(
    "/fit_relperm",
    summary="Fit Corey or LET Relative Permeability Model to SCAL Data",
    operation_id="fit_scal_relperm",
)
async def fit_relperm_endpoint(inputs: RelPermFitInput):
    """Calibrate two-phase relative permeability parameters to experimental SCAL plug data."""
    try:
        res = fit_relperm(
            sw=inputs.sw,
            krw=inputs.krw,
            kro=inputs.kro,
            model=inputs.model,
            swirr=inputs.swirr,
            sor=inputs.sor,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error fitting relative permeability: {e}"
        ) from e


@router.post(
    "/swof",
    summary="Generate Eclipse / Intersect SWOF Saturation Table",
    operation_id="generate_scal_swof_table",
)
async def generate_swof_endpoint(inputs: SWOFTableInput):
    """Generate a two-phase water-oil relative permeability & capillary pressure SWOF table."""
    try:
        res = generate_swof_table(
            n_points=inputs.n_points,
            model=inputs.model,
            swirr=inputs.swirr,
            sor=inputs.sor,
            krw_max=inputs.krw_max,
            kro_max=inputs.kro_max,
            nw=inputs.nw,
            no=inputs.no,
            lam=inputs.lam,
            lw=inputs.lw,
            ew=inputs.ew,
            tw=inputs.tw,
            lo=inputs.lo,
            eo=inputs.eo,
            to=inputs.to,
            pc_entry=inputs.pc_entry,
            pc_exponent=inputs.pc_exponent,
            pc_max=inputs.pc_max,
        )
        return {
            "table": res["table"].replace({np.nan: None}).to_dict(orient="records"),
            "swof_keyword": res["swof_keyword"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error generating SWOF table: {e}"
        ) from e


@router.post(
    "/stress_profile",
    summary="Estimate a Log-Based Effective Confining Stress Profile",
    operation_id="estimate_scal_stress_profile",
)
async def estimate_stress_profile_endpoint(inputs: StressProfileInput):
    """Estimate vertical, pore, and equivalent effective confining stress from logs."""
    try:
        if not (len(inputs.md) == len(inputs.tvd) == len(inputs.rhob)):
            raise ValueError("md, tvd, and rhob must have matching lengths.")
        if (inputs.dtc is None) != (inputs.dts is None):
            raise ValueError("Provide both DTC and DTS for dynamic Poisson's ratio.")
        if inputs.dtc is not None and (
            len(inputs.dtc) != len(inputs.md) or len(inputs.dts or []) != len(inputs.md)
        ):
            raise ValueError(
                "DTC and DTS must match the lengths of the other log arrays."
            )

        def to_array(values):
            if values is None:
                return None
            return np.asarray(
                [np.nan if value is None else value for value in values], dtype=float
            )

        result = estimate_log_stress_profile(
            md=to_array(inputs.md),
            tvd=to_array(inputs.tvd),
            rhob=to_array(inputs.rhob),
            dtc=to_array(inputs.dtc),
            dts=to_array(inputs.dts),
            depth_unit=inputs.depth_unit.strip().lower(),
            pore_pressure_gradient=inputs.pore_pressure_gradient,
            biot_coefficient=inputs.biot_coefficient,
            poisson_ratio=inputs.poisson_ratio,
        )
        rows = [
            {
                "DEPTH": float(md) if np.isfinite(md) else None,
                "SV_EST": float(sv) if np.isfinite(sv) else None,
                "PP_EST": float(pp) if np.isfinite(pp) else None,
                "NU_DYN": float(nu) if np.isfinite(nu) else None,
                "NCS_EST": float(ncs) if np.isfinite(ncs) else None,
            }
            for md, sv, pp, nu, ncs in zip(
                result["md"],
                result["sv_psi"],
                result["pp_psi"],
                result["nu"],
                result["ncs_equivalent_psi"],
                strict=False,
            )
        ]
        return {"data": rows}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error estimating log-based stress profile: {e}"
        ) from e


@router.post(
    "/core_corrections",
    summary="Apply Overburden, Klinkenberg, and Clay-Bound Water Corrections to Core Data",
    operation_id="apply_scal_core_corrections",
)
async def apply_core_corrections_endpoint(inputs: CoreCorrectionsInput):
    """Correct ambient laboratory core plug porosity, permeability, and saturation to reservoir conditions."""
    try:
        df = pd.DataFrame(inputs.data)
        eff_perm_coeff = (
            inputs.perm_compaction_coeff
            if inputs.perm_compaction_coeff is not None
            else (
                float(inputs.compaction_coeff) * 5.0
                if inputs.compaction_coeff
                else 2.0e-5
            )
        )
        corr_df = correct_core_data(
            df,
            net_confining_stress=inputs.net_confining_stress,
            ambient_stress=inputs.ambient_stress,
            mean_pore_pressure=inputs.mean_pore_pressure,
            cbw=inputs.cbw,
            compaction_coeff=inputs.compaction_coeff,
            perm_compaction_coeff=eff_perm_coeff,
            compaction_method=inputs.compaction_method,
            klinkenberg_method=inputs.klinkenberg_method,
            oven_dried=inputs.oven_dried,
            replace_ori=inputs.replace_ori,
        )
        return {"data": corr_df.replace({np.nan: None}).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error applying core corrections: {e}"
        ) from e


class GrainDensityCorrelationInput(BaseModel):
    driver_col: str = Field(
        default="CGR",
        description=(
            "Driver log column to regress grain density against. Preference order falls back "
            "automatically: the given column, else CGR, SGR, GR, GAMMA."
        ),
    )
    grain_density_col: str = Field(
        default="CGD",
        description=(
            "Core grain density column (g/cc). Falls back automatically to GRDEN, RHOG, "
            "RHOMAA, CGD if the given column is absent."
        ),
    )
    rho_ma_clean: Optional[float] = Field(
        default=None,
        description="Clean-sand grain density anchor (g/cc). Defaults to 2.65 (quartz) if omitted.",
    )
    driver_clean: Optional[float] = Field(
        default=None,
        description=(
            "Driver reading of the clean/low-clay end member. If omitted, estimated from the "
            "clean_percentile of the driver data."
        ),
    )
    clean_percentile: float = Field(
        default=5.0,
        ge=0.0,
        le=50.0,
        description="Percentile of driver used to estimate driver_clean when not supplied.",
    )
    rho_ma_min: float = Field(
        default=2.55, gt=0.0, description="Lower physical clamp for predicted rho_ma (g/cc)."
    )
    rho_ma_max: float = Field(
        default=3.00, gt=0.0, description="Upper physical clamp for predicted rho_ma (g/cc)."
    )
    data: List[Dict[str, Any]] = Field(
        ..., description="Core/log records containing the driver and grain density columns."
    )


# Column fallback preference order (uppercase match).
# "DRIVER" is accepted last so callers that send a generic driver key still resolve.
_DRIVER_FALLBACKS = ("CGR", "SGR", "GR", "GAMMA", "DRIVER")
_CGD_FALLBACKS = ("CGD", "GRDEN", "RHOG", "RHOMAA", "GRAIN_DEN")


def _resolve_column(df: pd.DataFrame, preferred: str, fallbacks) -> Optional[str]:
    """Resolve a column name case-insensitively, trying `preferred` then `fallbacks`."""
    upper_map = {str(c).upper(): c for c in df.columns}
    for candidate in (preferred, *fallbacks):
        key = str(candidate).upper()
        if key in upper_map:
            return upper_map[key]
    return None


def _resolve_column_exact(df: pd.DataFrame, name: str) -> Optional[str]:
    """Resolve a column case-insensitively without silent substitution.

    Used by the multivariate fitter so a requested driver (e.g. ``NPHI``) is never replaced
    by an unrelated fallback such as ``CGR``.
    """
    key = str(name).upper()
    for col in df.columns:
        if str(col).upper() == key:
            return col
    return None


@router.post(
    "/grain_density_correlation",
    summary="Calibrate an Anchored Grain / Matrix Density Correlation from Core (project-scoped)",
    operation_id="fit_scal_grain_density_correlation",
)
async def fit_grain_density_correlation_endpoint(inputs: GrainDensityCorrelationInput):
    """Fit a field/project-scoped grain density correlation ``rho_ma = rho_ma_clean + k*(driver - driver_clean)``.

    Regresses measured core grain density (CGD) against a log driver (CGR = Th + K preferred,
    GR fallback). Returns the calibrated coefficients plus fit statistics, and echoes a
    predicted continuous rho_ma per input record for immediate QC against the core plugs.
    """
    try:
        df = pd.DataFrame(inputs.data)
        if df.empty:
            raise ValueError("No records supplied.")

        driver_col = _resolve_column(df, inputs.driver_col, _DRIVER_FALLBACKS)
        if driver_col is None:
            raise ValueError(
                f"No driver column found. Looked for '{inputs.driver_col}' and "
                f"fallbacks {_DRIVER_FALLBACKS}."
            )
        cgd_col = _resolve_column(df, inputs.grain_density_col, _CGD_FALLBACKS)
        if cgd_col is None:
            raise ValueError(
                f"No grain density column found. Looked for '{inputs.grain_density_col}' "
                f"and fallbacks {_CGD_FALLBACKS}."
            )

        driver = pd.to_numeric(df[driver_col], errors="coerce").to_numpy(dtype=float)
        cgd = pd.to_numeric(df[cgd_col], errors="coerce").to_numpy(dtype=float)

        fit = fit_grain_density_correlation(
            driver,
            cgd,
            rho_ma_clean=inputs.rho_ma_clean,
            driver_clean=inputs.driver_clean,
            clean_percentile=inputs.clean_percentile,
            rho_ma_min=inputs.rho_ma_min,
            rho_ma_max=inputs.rho_ma_max,
        )

        rho_ma_pred = grain_density_from_correlation(
            driver,
            rho_ma_clean=fit["rho_ma_clean"],
            k=fit["k"],
            driver_clean=fit["driver_clean"],
            rho_ma_min=fit["rho_ma_min"],
            rho_ma_max=fit["rho_ma_max"],
        )
        rho_ma_pred = np.asarray(rho_ma_pred, dtype=float)

        preview = [
            {
                "driver": None if not np.isfinite(d) else round(float(d), 4),
                "CGD": None if not np.isfinite(c) else round(float(c), 4),
                "RHO_MA_PRED": None if not np.isfinite(p) else round(float(p), 4),
            }
            for d, c, p in zip(driver, cgd, rho_ma_pred, strict=False)
        ]

        return {
            "fit": fit,
            "driver_col": driver_col,
            "grain_density_col": cgd_col,
            "preview": preview,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error fitting grain density correlation: {e}"
        ) from e


class GrainDensityMultivariateInput(BaseModel):
    """Multi-log ridge / Huber-robust grain density correlation request."""

    driver_cols: List[str] = Field(
        default_factory=lambda: ["CGR", "GR", "RHOB", "NPHI"],
        description=(
            "Driver log columns to regress grain density against jointly. Order is preserved "
            "in the returned coefficients. GR/CGR/SGR, RHOB, NPHI, M_LITH, N_LITH and AI are "
            "all valid drivers; use several because gamma alone is heteroscedastic."
        ),
    )
    grain_density_col: str = Field(
        default="CGD",
        description=(
            "Core grain density column (g/cc). Falls back automatically to GRDEN, RHOG, "
            "RHOMAA, GRAIN_DEN if the given column is absent."
        ),
    )
    rho_ma_clean: Optional[float] = Field(
        default=None,
        description="Clean-sand grain density anchor (g/cc). Defaults to 2.65 (quartz) if omitted.",
    )
    alpha: float = Field(
        1.0,
        ge=0.0,
        description=(
            "Ridge penalty on standardized driver coefficients. 0 = plain OLS; ~1 shrinks "
            "collinear gamma channels and stabilizes sparse core coverage."
        ),
    )
    robust: bool = Field(
        True,
        description=(
            "Huber IRLS reweighting. Recommended: down-weights radioactive / heavy-mineral "
            "outliers that drive heteroscedastic single-log fits."
        ),
    )
    huber_delta: float = Field(
        1.345,
        gt=0.0,
        description="Huber tuning constant in robust-residual units (1.345 = 95% Gaussian efficiency).",
    )
    rho_ma_min: float = Field(
        default=2.55,
        gt=0.0,
        description="Lower physical clamp for predicted rho_ma (g/cc).",
    )
    rho_ma_max: float = Field(
        default=3.00,
        gt=0.0,
        description="Upper physical clamp for predicted rho_ma (g/cc).",
    )
    data: List[Dict[str, Any]] = Field(
        ...,
        description="Core/log records containing the driver columns and grain density.",
    )


@router.post(
    "/grain_density_correlation_multivariate",
    summary="Calibrate a Multi-Log Ridge/Huber Grain Density Correlation from Core",
    operation_id="fit_scal_grain_density_correlation_multivariate",
)
async def fit_grain_density_correlation_multivariate_endpoint(
    inputs: GrainDensityMultivariateInput,
):
    """Fit a ridge / Huber-robust multi-driver grain density correlation.

    Regresses measured core grain density (CGD) jointly against several log drivers
    (GR/CGR/SGR, RHOB, NPHI, M-parameter, acoustic impedance, ...). Because total GR and
    CGR/SGR are heteroscedastic (radioactive accessory minerals add high-GR outliers), a
    single-log OLS slope is biased; ridge shrinkage plus Huber reweighting yields a stable
    multi-log trend. Returns standardized coefficients for application on logs, the VIF
    collinearity diagnostics, a Breusch-Pagan heteroscedasticity test, and a per-record
    predicted rho_ma preview.
    """
    try:
        df = pd.DataFrame(inputs.data)
        if df.empty:
            raise ValueError("No records supplied.")

        driver_cols = [c for c in (inputs.driver_cols or []) if c]
        if not driver_cols:
            raise ValueError("At least one driver column is required.")

        resolved_drivers: List[str] = []
        for preferred in driver_cols:
            resolved = _resolve_column_exact(df, preferred)
            if resolved is None:
                raise ValueError(
                    f"Driver column '{preferred}' was not found in the supplied records."
                )
            resolved_drivers.append(resolved)

        cgd_col = _resolve_column(df, inputs.grain_density_col, _CGD_FALLBACKS)
        if cgd_col is None:
            raise ValueError(
                f"No grain density column found. Looked for '{inputs.grain_density_col}' "
                f"and fallbacks {_CGD_FALLBACKS}."
            )

        driver_matrix = np.column_stack(
            [
                pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
                for col in resolved_drivers
            ]
        )
        cgd = pd.to_numeric(df[cgd_col], errors="coerce").to_numpy(dtype=float)

        fit = fit_grain_density_correlation_multivariate(
            driver_matrix,
            cgd,
            driver_names=resolved_drivers,
            rho_ma_clean=inputs.rho_ma_clean,
            alpha=inputs.alpha,
            robust=inputs.robust,
            huber_delta=inputs.huber_delta,
            rho_ma_min=inputs.rho_ma_min,
            rho_ma_max=inputs.rho_ma_max,
        )

        rho_ma_pred = np.asarray(
            grain_density_from_correlation_multivariate(
                driver_matrix, fit, driver_names=resolved_drivers
            ),
            dtype=float,
        )

        preview = [
            {
                **{
                    name: (None if not np.isfinite(v) else round(float(v), 4))
                    for name, v in zip(resolved_drivers, row, strict=False)
                },
                "CGD": None if not np.isfinite(c) else round(float(c), 4),
                "RHO_MA_PRED": None if not np.isfinite(p) else round(float(p), 4),
            }
            for row, c, p in zip(driver_matrix, cgd, rho_ma_pred, strict=False)
        ]

        return {
            "fit": fit,
            "driver_cols": resolved_drivers,
            "grain_density_col": cgd_col,
            "preview": preview,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Error fitting multivariate grain density correlation: {e}",
        ) from e


@router.post(
    "/pc_lab_to_reservoir",
    summary="Convert Laboratory Capillary Pressure to Reservoir Conditions",
    operation_id="convert_pc_lab_to_reservoir",
)
async def convert_pc_lab_to_reservoir_endpoint(inputs: PcLabToReservoirInput):
    """Scale laboratory capillary pressure (MICP / porous plate / centrifuge) to reservoir IFT and wettability."""
    try:
        pc_arr = np.asarray(inputs.pc_lab, dtype=float)
        pc_res = pc_lab_to_reservoir(
            pc_arr,
            ift_lab=inputs.ift_lab,
            theta_lab=inputs.theta_lab,
            ift_res=inputs.ift_res,
            theta_res=inputs.theta_res,
            phi_amb=inputs.phi_amb,
            phi_res=inputs.phi_res,
            k_amb=inputs.k_amb,
            k_res=inputs.k_res,
        )
        unit_res = float(
            np.asarray(
                pc_lab_to_reservoir(
                    1.0,
                    ift_lab=inputs.ift_lab,
                    theta_lab=inputs.theta_lab,
                    ift_res=inputs.ift_res,
                    theta_res=inputs.theta_res,
                    phi_amb=inputs.phi_amb
                    if isinstance(inputs.phi_amb, (int, float))
                    else None,
                    phi_res=inputs.phi_res
                    if isinstance(inputs.phi_res, (int, float))
                    else None,
                    k_amb=inputs.k_amb
                    if isinstance(inputs.k_amb, (int, float))
                    else None,
                    k_res=inputs.k_res
                    if isinstance(inputs.k_res, (int, float))
                    else None,
                )
            ).flat[0]
        )
        return {
            "pc_res": [float(x) for x in np.asarray(pc_res).flatten()],
            "scale_factor": round(unit_res, 6),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error converting capillary pressure: {e}"
        ) from e
