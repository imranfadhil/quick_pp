"""Source rock evaluation and Total Organic Carbon (TOC) estimation.

Provides the Passey et al. (1990) Delta Log R (ΔlogR) method for identifying
organic-rich source rocks and quantifying TOC (wt%) from wireline resistivity
and porosity logs (sonic DT, bulk density RHOB, or neutron porosity NPHI).
"""

from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

# Default Passey scaling coefficients
# Sonic: 50 us/ft per resistivity decade -> coeff = 1 / 50 = 0.02
DEFAULT_SONIC_COEFF = 0.02
# Density: 0.25 g/cm3 per decade -> coeff = -1 / 0.25 = -4.0 (or -2.5 in AAPG 1990)
DEFAULT_DENSITY_COEFF = -2.5
# Neutron: 0.25 v/v (or 25 p.u.) per decade -> coeff = +4.0 (for fraction 0-1)
DEFAULT_NEUTRON_COEFF = 4.0


def hood_ro_to_lom(ro: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert Vitrinite Reflectance (Ro %) to Level of Organic Metamorphism (LOM).

    Based on Hood et al. (1975) empirical maturity scale:
    Ro = 0.40 -> LOM ~ 6.0 (Immature)
    Ro = 0.60 -> LOM ~ 8.3 (Onset of oil window)
    Ro = 0.80 -> LOM ~ 9.6 (Peak oil generation)
    Ro = 1.00 -> LOM ~ 10.6 (Condensate window)
    Ro = 1.20 -> LOM ~ 11.4 (Wet gas)
    Ro = 2.00 -> LOM ~ 13.3 (Dry gas)

    Args:
        ro: Vitrinite reflectance in percent (e.g. 0.8 for 0.8% Ro).

    Returns:
        Level of Organic Metamorphism (LOM, typically 6 to 14).
    """
    ro_arr = np.asarray(ro, dtype=float)
    # Clip Ro to physically realistic range [0.2, 5.0]
    ro_safe = np.clip(ro_arr, 0.2, 5.0)
    # Natural logarithmic fit accurate across Hood's chart (R^2 > 0.99)
    lom_calc = 10.59 + 4.54 * np.log(ro_safe)
    lom_clamped = np.clip(lom_calc, 4.0, 16.0)

    if np.isscalar(ro):
        return float(lom_clamped)
    return lom_clamped


def hood_lom_to_ro(lom: Union[float, np.ndarray, pd.Series]) -> Union[float, np.ndarray, pd.Series]:
    """Convert Level of Organic Metamorphism (LOM) to Vitrinite Reflectance (Ro %).

    Args:
        lom: Hood et al. (1975) LOM value (typically 6 to 14).

    Returns:
        Vitrinite reflectance (Ro in %).
    """
    lom_arr = np.asarray(lom, dtype=float)
    lom_safe = np.clip(lom_arr, 4.0, 16.0)
    ro_calc = np.exp((lom_safe - 10.59) / 4.54)

    if np.isscalar(lom):
        return float(ro_calc)
    return ro_calc


def get_default_scaling_coeff(porosity_type: str = "sonic", poro_values: Optional[np.ndarray] = None) -> float:
    """Return standard Passey scaling coefficient for the selected porosity log type.

    Args:
        porosity_type: 'sonic' (DT), 'density' (RHOB), or 'neutron' (NPHI).
        poro_values: Optional sample of curve values to infer units (e.g. fraction vs percent).

    Returns:
        Float scaling coefficient.
    """
    ptype = porosity_type.lower()
    if "son" in ptype or "dt" in ptype:
        return DEFAULT_SONIC_COEFF
    elif "den" in ptype or "rho" in ptype:
        return DEFAULT_DENSITY_COEFF
    elif "neu" in ptype or "nphi" in ptype:
        if poro_values is not None:
            max_val = np.nanmax(poro_values)
            if max_val > 1.5:  # Log is in percent (0 - 100 p.u.)
                return 0.04
        return DEFAULT_NEUTRON_COEFF
    return DEFAULT_SONIC_COEFF


def passey_delta_log_r(
    resistivity: Union[float, np.ndarray, pd.Series],
    porosity_log: Union[float, np.ndarray, pd.Series],
    r_baseline: float,
    poro_baseline: float,
    porosity_type: str = "sonic",
    scaling_coeff: Optional[float] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Calculate Delta Log R (ΔlogR) from deep resistivity and a porosity log.

    Following Passey et al. (1990):
    ΔlogR = log10(R / R_baseline) + coeff * (Poro_log - Poro_baseline)

    Args:
        resistivity: Deep resistivity log (ohm.m).
        porosity_log: Porosity log (DT in us/ft, RHOB in g/cm3, or NPHI in v/v).
        r_baseline: Shale baseline resistivity in lean, water-saturated shale (ohm.m).
        poro_baseline: Shale baseline porosity value in lean, water-saturated shale.
        porosity_type: 'sonic', 'density', or 'neutron'.
        scaling_coeff: Optional custom scaling coefficient. If None, uses defaults.

    Returns:
        Delta Log R (dimensionless log10 separation).
    """
    if r_baseline <= 0:
        raise ValueError(f"r_baseline must be positive, got {r_baseline}")

    if scaling_coeff is None:
        p_sample = np.asarray(porosity_log) if not np.isscalar(porosity_log) else None
        scaling_coeff = get_default_scaling_coeff(porosity_type, p_sample)

    is_series = isinstance(resistivity, pd.Series) or isinstance(porosity_log, pd.Series)
    is_scalar = np.isscalar(resistivity) and np.isscalar(porosity_log)

    r_arr = np.asarray(resistivity, dtype=float)
    p_arr = np.asarray(porosity_log, dtype=float)

    # Resistivity must be strictly positive for log10
    with np.errstate(invalid="ignore", divide="ignore"):
        r_safe = np.where(r_arr > 0, r_arr, np.nan)
        res_ratio = r_safe / r_baseline
        log_res_term = np.log10(res_ratio)
        poro_term = scaling_coeff * (p_arr - poro_baseline)
        delta_log_r = log_res_term + poro_term

    if is_scalar:
        return float(delta_log_r)
    if is_series:
        idx = resistivity.index if isinstance(resistivity, pd.Series) else porosity_log.index
        return pd.Series(delta_log_r, index=idx, name="DELTA_LOG_R")
    return delta_log_r


def estimate_toc_passey(
    delta_log_r: Union[float, np.ndarray, pd.Series],
    lom: Optional[float] = None,
    ro: Optional[float] = None,
    toc_baseline: float = 0.8,
    clamp_negative: bool = True,
    max_toc: float = 50.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Estimate Total Organic Carbon (TOC in weight %) from Delta Log R.

    Passey et al. (1990) equation:
    TOC = (ΔlogR) * 10^(2.297 - 0.1688 * LOM) + TOC_baseline

    Args:
        delta_log_r: Calculated ΔlogR curve.
        lom: Level of Organic Metamorphism (typically 6 to 14). Default is 10.5 if neither lom nor ro is given.
        ro: Vitrinite reflectance (% Ro). If provided, converts to LOM via Hood et al. (1975).
        toc_baseline: Background non-source TOC in lean shale (wt%, default 0.8%).
        clamp_negative: If True, where ΔlogR <= 0, TOC is clamped to toc_baseline.
        max_toc: Upper bound clamp for TOC (default 50.0 wt%).

    Returns:
        Estimated Total Organic Carbon in wt%.
    """
    if lom is None:
        if ro is not None:
            lom = float(hood_ro_to_lom(ro))
        else:
            lom = 10.5  # Standard default: peak oil window

    is_series = isinstance(delta_log_r, pd.Series)
    is_scalar = np.isscalar(delta_log_r)

    dlogr_arr = np.asarray(delta_log_r, dtype=float)

    # Passey maturity exponent
    exponent = 2.297 - 0.1688 * float(lom)
    maturity_factor = 10.0 ** exponent

    with np.errstate(invalid="ignore"):
        calc_toc = dlogr_arr * maturity_factor + float(toc_baseline)
        if clamp_negative:
            # In non-source intervals where ΔlogR < 0, TOC does not fall below background
            calc_toc = np.where(dlogr_arr < 0, toc_baseline, calc_toc)
        calc_toc = np.clip(calc_toc, 0.0, max_toc)

    if is_scalar:
        return float(calc_toc)
    if is_series:
        return pd.Series(calc_toc, index=delta_log_r.index, name="TOC")
    return calc_toc


def auto_detect_shale_baseline(
    df: pd.DataFrame,
    vsh_col: Optional[str] = None,
    res_col: Optional[str] = None,
    poro_col: Optional[str] = None,
    vsh_threshold: float = 0.65,
) -> Dict[str, Any]:
    """Auto-detect candidate baseline values for lean shale from available curves.

    Looks for water-saturated, high-Vshale intervals where organic richness is lowest
    (low resistivity, typical shale compaction).

    Args:
        df: Well logs dataframe.
        vsh_col: Name of Vshale curve (or candidate like 'VSH_GR', 'VSHALE', 'VCLAY').
        res_col: Name of deep resistivity curve ('RD', 'RT', 'ILD', etc.).
        poro_col: Name of porosity curve ('DT', 'DTC', 'RHOB', 'NPHI').
        vsh_threshold: Minimum Vshale to consider an interval as shale (default 0.65).

    Returns:
        Dict with detected r_baseline, poro_baseline, and method summary.
    """
    cols_upper = {c.upper(): c for c in df.columns}

    # Resolve Vshale curve
    vsh_candidate = None
    if vsh_col and vsh_col.upper() in cols_upper:
        vsh_candidate = cols_upper[vsh_col.upper()]
    else:
        for c in ["VSHALE", "VSH_GR", "VSH_DN", "VCLAY", "GR"]:
            if c in cols_upper:
                vsh_candidate = cols_upper[c]
                break

    # Resolve Resistivity
    res_candidate = None
    if res_col and res_col.upper() in cols_upper:
        res_candidate = cols_upper[res_col.upper()]
    else:
        for c in ["RD", "RT", "ILD", "LLD", "AHT90", "AT90", "RES_DEEP"]:
            if c in cols_upper:
                res_candidate = cols_upper[c]
                break

    # Resolve Porosity
    poro_candidate = None
    if poro_col and poro_col.upper() in cols_upper:
        poro_candidate = cols_upper[poro_col.upper()]
    else:
        for c in ["DT", "DTC", "RHOB", "NPHI"]:
            if c in cols_upper:
                poro_candidate = cols_upper[c]
                break

    if not res_candidate or not poro_candidate:
        return {
            "r_baseline": 2.0,
            "poro_baseline": 80.0 if poro_candidate and "DT" in poro_candidate.upper() else 2.50,
            "res_curve": res_candidate,
            "poro_curve": poro_candidate,
            "status": "default_fallback",
        }

    res_series = pd.to_numeric(df[res_candidate], errors="coerce")
    poro_series = pd.to_numeric(df[poro_candidate], errors="coerce")

    # Filter for shale mask
    if vsh_candidate:
        vsh_vals = pd.to_numeric(df[vsh_candidate], errors="coerce")
        if "GR" in vsh_candidate.upper() and vsh_vals.max() > 1.5:
            # Raw GR instead of fraction
            shale_mask = vsh_vals >= np.nanpercentile(vsh_vals, 70)
        else:
            shale_mask = vsh_vals >= vsh_threshold
    else:
        # Without Vshale, pick the lower 30th percentile of resistivity as candidate shale
        shale_mask = res_series <= res_series.quantile(0.35)

    valid_mask = shale_mask & (res_series > 0.1) & res_series.notna() & poro_series.notna()
    if valid_mask.sum() < 5:
        # Fallback to general median
        valid_mask = (res_series > 0.1) & res_series.notna() & poro_series.notna()

    if valid_mask.sum() == 0:
        return {
            "r_baseline": 2.0,
            "poro_baseline": 80.0 if "DT" in poro_candidate.upper() else 2.50,
            "res_curve": res_candidate,
            "poro_curve": poro_candidate,
            "status": "no_valid_data",
        }

    # For lean shale, resistivity is typically low (1 - 5 ohm.m)
    r_base = float(res_series[valid_mask].median())
    poro_base = float(poro_series[valid_mask].median())

    return {
        "r_baseline": round(r_base, 3),
        "poro_baseline": round(poro_base, 3),
        "res_curve": res_candidate,
        "poro_curve": poro_candidate,
        "shale_samples": int(valid_mask.sum()),
        "status": "auto_detected",
    }


def calibrate_toc_to_core(
    delta_log_r: Union[np.ndarray, pd.Series],
    core_toc: Union[np.ndarray, pd.Series],
) -> Dict[str, Any]:
    """Perform least-squares linear calibration of Delta Log R to core TOC measurements.

    Fits the model: Core_TOC = slope * ΔlogR + intercept

    Args:
        delta_log_r: Log-derived ΔlogR values at core depth points.
        core_toc: Measured core TOC values (wt%).

    Returns:
        Dict with slope, intercept, R2, RMSE, sample_count, and implied LOM.
    """
    x = np.asarray(delta_log_r, dtype=float)
    y = np.asarray(core_toc, dtype=float)

    mask = np.isfinite(x) & np.isfinite(y) & (y >= 0)
    if mask.sum() < 3:
        return {
            "success": False,
            "error": "Insufficient valid core TOC and Delta Log R pairs (minimum 3 required)",
            "sample_count": int(mask.sum()),
        }

    x_valid = x[mask]
    y_valid = y[mask]

    slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
    y_pred = slope * x_valid + intercept

    ss_res = np.sum((y_valid - y_pred) ** 2)
    ss_tot = np.sum((y_valid - np.mean(y_valid)) ** 2)
    r2 = 1.0 - (ss_res / ss_tot) if ss_tot > 1e-9 else 0.0
    rmse = np.sqrt(np.mean((y_valid - y_pred) ** 2))

    # Derive implied LOM from slope: slope = 10^(2.297 - 0.1688 * LOM)
    implied_lom = None
    if slope > 0:
        implied_lom = (2.297 - np.log10(slope)) / 0.1688
        implied_lom = round(float(np.clip(implied_lom, 6.0, 14.0)), 2)

    return {
        "success": True,
        "slope": round(float(slope), 4),
        "intercept": round(float(intercept), 4),
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "sample_count": int(mask.sum()),
        "implied_lom": implied_lom,
    }
