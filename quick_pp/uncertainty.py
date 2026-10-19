"""Probabilistic uncertainty analysis and Latin Hypercube Monte Carlo engine for quick_pp.

This module promotes volumetric uncertainty and log-level petrophysical parameter uncertainty
into a core module using Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`).

Key capabilities:
1. `mc_volumetric_uncertainty`: Vectorized Latin Hypercube Monte Carlo simulation on
   zone/reservoir volumetric parameters (area, thickness, porosity, water saturation,
   volume factor, recovery factor) with P10/P50/P90 statistics and optional plotting.
2. `run_log_uncertainty`: Log-level Latin Hypercube petrophysical uncertainty propagation
   over uncertain parameters ($a, m, n, R_w, \\rho_{ma}, \\rho_{sh}$) across `Carbonate` or
   `SandSiltClay` lithology/porosity models and `waxman_smits_saturation` / `archie_saturation`,
   outputting P10 / P50 / P90 CPI log curves (`SW_P10`, `SW_P50`, `SW_P90`, `SWT_P10`,
   `SWT_P50`, `SWT_P90`, `PHIT_P10`, `PHIT_P50`, `PHIT_P90`, `VCLAY_P10`, `VCLAY_P50`,
   `VCLAY_P90`, `BVHC_P10`, `BVHC_P50`, `BVHC_P90`).
3. `calc_stochastic_reservoir_summary`: Combines log-level Latin Hypercube realizations
   with `calc_reservoir_summary` and volumetric Monte Carlo to output P10 / P50 / P90
   zone-level reservoir summary metrics and in-place volumes.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import qmc, truncnorm, uniform

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.lithology import gr_index
from quick_pp.porosity import (
    density_porosity,
    effective_porosity,
    estimate_shale_porosity,
    normalize_volumetric,
)
from quick_pp.saturation import (
    archie_saturation,
    estimate_b_waxman_smits,
    estimate_qvn,
    estimate_rw_temperature_salinity,
    estimate_temperature_gradient,
)
from quick_pp.utils import length_a_b, line_intersection, min_max_line, remove_outliers

DEFAULT_LOG_PARAM_BOUNDS: Dict[str, Tuple[float, ...]] = {
    "a": (0.8, 1.2, 1.0, 0.1),
    "m": (1.8, 2.2, 2.0, 0.1),
    "n": (1.8, 2.2, 2.0, 0.1),
    "rw": (0.02, 0.15, 0.08, 0.03),
    "rho_ma": (2.63, 2.75, 2.68, 0.03),
    "rho_sh": (2.55, 2.78, 2.68, 0.05),
}


def _map_qmc_column(u: np.ndarray, bound: Union[Tuple[float, ...], List[float], float, int]) -> np.ndarray:
    """Map a 1D unit-hypercube sample `u` in [0, 1] to a target distribution defined by `bound`.

    Supported `bound` formats:
      - float or int, or 1-tuple `(val,)`: deterministic constant `val`
      - 2-tuple `(min, max)`: uniform distribution on `[min, max]` (if `min == max`, constant)
      - 4-tuple `(min, max, mean, std)`: truncated normal distribution on `[min, max]`
    """
    if isinstance(bound, (int, float, np.number)):
        return np.full_like(u, float(bound), dtype=float)

    b = tuple(float(x) for x in bound)
    if len(b) == 1:
        return np.full_like(u, b[0], dtype=float)
    if len(b) == 2:
        low, high = b
        if high <= low:
            return np.full_like(u, low, dtype=float)
        return uniform.ppf(u, loc=low, scale=high - low)
    if len(b) == 4:
        low, high, mean, std = b
        if high <= low or std <= 0:
            return np.full_like(u, np.clip(mean, low, high), dtype=float)
        a_trans = (low - mean) / std
        b_trans = (high - mean) / std
        return truncnorm.ppf(u, a_trans, b_trans, loc=mean, scale=std)

    raise ValueError(
        f"Invalid parameter bound format {bound!r}. Expected scalar, (min, max), or (min, max, mean, std)."
    )


def sample_lhs_parameters(
    param_bounds: Dict[str, Union[Tuple[float, ...], List[float], float]],
    n_samples: int = 100,
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """Generate Latin Hypercube samples for a dictionary of parameter bounds.

    Args:
        param_bounds (dict): Mapping of parameter name to bound specification:
            - `(min, max, mean, std)` for truncated normal distribution
            - `(min, max)` for uniform distribution
            - `float` for fixed constant parameter
        n_samples (int, optional): Number of Latin Hypercube realizations. Defaults to 100.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        pd.DataFrame: DataFrame of shape `(n_samples, len(param_bounds))` with sampled parameters.
    """
    names = list(param_bounds.keys())
    d = len(names)
    if d == 0 or n_samples <= 0:
        return pd.DataFrame(columns=names)

    sampler = qmc.LatinHypercube(d=d, seed=seed)
    u_matrix = sampler.random(n=n_samples)
    # Clip away from exact 0 or 1 to avoid infinite ppf tails
    u_matrix = np.clip(u_matrix, 1e-9, 1.0 - 1e-9)

    samples = {}
    for idx, name in enumerate(names):
        samples[name] = _map_qmc_column(u_matrix[:, idx], param_bounds[name])

    return pd.DataFrame(samples)


def mc_volumetric_uncertainty(
    area_bound: Tuple[float, ...],
    thickness_bound: Tuple[float, ...],
    porosity_bound: Tuple[float, ...],
    water_saturation_bound: Tuple[float, ...],
    volume_factor_bound: Tuple[float, ...],
    recovery_factor_bound: Optional[Tuple[float, ...]] = None,
    n_try: int = 10000,
    percentile: Optional[List[float]] = None,
    seed: Optional[int] = 42,
    plot: bool = False,
) -> Dict[str, Any]:
    """Latin Hypercube Monte Carlo simulation for volumetric hydrocarbon in-place uncertainty.

    Uses `scipy.stats.qmc.LatinHypercube` on area, thickness, porosity, water saturation,
    formation volume factor, and optional recovery factor to estimate STOIIP / recoverable
    hydrocarbon volume distributions in MMbbl:
        $$V = 7758 \\times A \\times h \\times \\phi \\times (1 - S_w) / B_o \\times RF \\times 10^{-6}$$
    where $43560 \\times 0.1781 \\approx 7758.37$ bbl / acre-ft.

    Args:
        area_bound (tuple): `(min, max, mean, std)` or `(min, max)` for area in acres.
        thickness_bound (tuple): `(min, max, mean, std)` or `(min, max)` for net pay thickness in feet.
        porosity_bound (tuple): `(min, max, mean, std)` or `(min, max)` for porosity in fraction.
        water_saturation_bound (tuple): `(min, max, mean, std)` or `(min, max)` for water saturation in fraction.
        volume_factor_bound (tuple): `(min, max)` or `(min, max, mean, std)` for oil formation volume factor $B_o$.
        recovery_factor_bound (tuple, optional): `(min, max, mean, std)` or `(min, max)` for recovery factor.
        n_try (int, optional): Number of Latin Hypercube realizations. Defaults to 10000.
        percentile (list, optional): Percentiles to compute. Defaults to `[10, 50, 90]`.
        seed (int, optional): Random seed for LatinHypercube sampler. Defaults to 42.
        plot (bool, optional): Whether to display diagnostic RainCloud/histogram plots. Defaults to False.

    Returns:
        dict: Dictionary containing:
            - `volumes`: np.ndarray of simulated volumes (MMbbl)
            - `area`: np.ndarray of sampled areas (acres)
            - `thickness`: np.ndarray of sampled thicknesses (ft)
            - `porosity`: np.ndarray of sampled porosities (v/v)
            - `water_saturation`: np.ndarray of sampled water saturations (v/v)
            - `volume_factor`: np.ndarray of sampled formation volume factors
            - `recovery_factor`: np.ndarray of sampled recovery factors
            - `percentiles`: dict mapping `'P10'`, `'P50'`, `'P90'` (or custom percentiles) to float values
            - `mean`: float mean volume (MMbbl)
            - `std`: float standard deviation of volume (MMbbl)
    """
    percentile = percentile or [10, 50, 90]
    param_spec: Dict[str, Union[Tuple[float, ...], float]] = {
        "area": area_bound,
        "thickness": thickness_bound,
        "porosity": porosity_bound,
        "water_saturation": water_saturation_bound,
        "volume_factor": volume_factor_bound,
        "recovery_factor": recovery_factor_bound if recovery_factor_bound is not None else 1.0,
    }

    samples_df = sample_lhs_parameters(param_spec, n_samples=n_try, seed=seed)
    area = np.maximum(0.0, samples_df["area"].to_numpy(dtype=float))
    thickness = np.maximum(0.0, samples_df["thickness"].to_numpy(dtype=float))
    porosity = np.clip(samples_df["porosity"].to_numpy(dtype=float), 0.0, 1.0)
    water_saturation = np.clip(samples_df["water_saturation"].to_numpy(dtype=float), 0.0, 1.0)
    volume_factor = np.maximum(1e-6, samples_df["volume_factor"].to_numpy(dtype=float))
    recovery_factor = np.clip(samples_df["recovery_factor"].to_numpy(dtype=float), 0.0, 1.0)

    volumes = (
        (43560.0 * 0.1781)
        * area
        * thickness
        * porosity
        * (1.0 - water_saturation)
        / volume_factor
        * recovery_factor
        * 1e-6
    )

    pct_values = np.percentile(volumes, percentile)
    pct_dict = {f"P{int(p)}": float(val) for p, val in zip(percentile, pct_values, strict=True)}

    logger.info(
        f"LHS Monte Carlo volumetric simulation ({n_try} trials) - "
        f"Mean: {np.mean(volumes):.2f} MMbbl, Std: {np.std(volumes):.2f} MMbbl, "
        f"Percentiles: {pct_dict}"
    )

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.hist(volumes, density=True, bins="auto", alpha=0.7, color="steelblue", edgecolor="black")
        ax.set_title(f"LHS Volume Estimation ({n_try} realizations)")
        ax.set_xlabel("Volume (MM bbl)")
        for p, val in zip(percentile, pct_values, strict=True):
            ax.axvline(
                x=val,
                color="r",
                linestyle="dashed",
                label=f"P{int(p)}: {val:.2f} MM bbl",
            )
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        fig.tight_layout()
        plt.show()

    return {
        "volumes": volumes,
        "area": area,
        "thickness": thickness,
        "porosity": porosity,
        "water_saturation": water_saturation,
        "volume_factor": volume_factor,
        "recovery_factor": recovery_factor,
        "percentiles": pct_dict,
        "mean": float(np.mean(volumes)),
        "std": float(np.std(volumes)),
    }


def _vectorized_waxman_smits_saturation(
    rt: np.ndarray,
    rw: np.ndarray,
    phit: np.ndarray,
    Qv: Optional[np.ndarray] = None,
    B: Optional[np.ndarray] = None,
    a: Union[np.ndarray, float] = 1.0,
    m: Union[np.ndarray, float] = 2.0,
    n: Union[np.ndarray, float] = 2.0,
    n_iter: int = 40,
) -> np.ndarray:
    """Fast, vectorized Waxman-Smits saturation solver without tqdm overhead for 1D or 2D arrays.

    Solves the Waxman-Smits equation (with tortuosity factor `a`):
        $$S_w^n + R_w B Q_v S_w^{n-1} - \\frac{a R_w}{\\phi_t^m R_t} = 0$$
    """
    rt_arr = np.asarray(rt, dtype=float)
    rw_arr = np.asarray(rw, dtype=float)
    phit_arr = np.asarray(phit, dtype=float)
    a_arr = np.asarray(a, dtype=float)
    m_arr = np.asarray(m, dtype=float)
    n_arr = np.asarray(n, dtype=float)

    safe_rw = np.where((rw_arr > 1e-6) & np.isfinite(rw_arr), rw_arr, 0.08)
    safe_rt = np.where((rt_arr > 1e-6) & np.isfinite(rt_arr), rt_arr, 1.0)
    safe_phit = np.where((phit_arr > 1e-4) & np.isfinite(phit_arr), phit_arr, 1e-4)

    if B is None:
        B_arr = (1.0 - 0.83 * np.exp(-0.5 / safe_rw)) * 3.83
    else:
        B_arr = np.asarray(B, dtype=float)
        B_arr = np.where(np.isfinite(B_arr), B_arr, (1.0 - 0.83 * np.exp(-0.5 / safe_rw)) * 3.83)

    if Qv is None:
        Qv_arr = np.full_like(safe_phit, 0.3, dtype=float)
    else:
        Qv_arr = np.asarray(Qv, dtype=float)
        Qv_arr = np.where(np.isfinite(Qv_arr) & (Qv_arr >= 0.0), Qv_arr, 0.3)

    rhs = a_arr * (safe_phit ** (-m_arr)) * safe_rw / safe_rt
    swt = np.ones_like(rhs, dtype=float)
    swt_i = np.zeros_like(rhs, dtype=float)

    for _ in range(n_iter):
        fx = (swt ** n_arr) + safe_rw * B_arr * Qv_arr * (swt ** (n_arr - 1.0)) - rhs
        delta_sat = np.abs(swt - swt_i) * 0.5
        swt_i = swt
        swt = np.where(fx < 0.0, swt + delta_sat, swt - delta_sat)

    invalid_mask = ~((rt_arr > 0) & np.isfinite(rt_arr) & (phit_arr > 1e-3) & np.isfinite(phit_arr))
    swt = np.where(invalid_mask, 1.0, np.clip(swt, 0.0, 1.0))
    return swt


def _vectorized_carbonate_neu_den(
    nphi: np.ndarray,
    rhob: np.ndarray,
    vsh_gr: np.ndarray,
    dry_calc_point: Tuple[float, float],
    dry_dolo_point: Tuple[float, float],
    dry_clay_point: Tuple[float, float],
    fluid_point: Tuple[float, float],
    carbonate_model: str = "double",
    carbonate_type: str = "limestone",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized Carbonate lithology estimation for a single realization `(nphi, rhob)`."""
    vcld = np.clip(np.asarray(vsh_gr, dtype=float), 0.0, 1.0)
    if carbonate_model == "single":
        vcarb = 1.0 - vcld
        if carbonate_type == "dolostone":
            return vcld, np.zeros_like(vcld), vcarb
        return vcld, vcarb, np.zeros_like(vcld)

    A = dry_dolo_point
    C = dry_calc_point
    D = fluid_point
    rock_len = float(length_a_b(A, C))
    if rock_len <= 1e-9:
        rock_len = 1.0

    var_x, var_y = line_intersection((A, C), (D, (nphi, rhob)))
    proj_len = np.sqrt((var_x - A[0]) ** 2 + (var_y - A[1]) ** 2)
    vfrac = np.clip(proj_len / rock_len, 0.0, 1.0)
    # In Carbonate.lithology_fraction_neu_den(model='double'), vlitho1 is (1 - vfrac) and vlitho2 is vfrac
    vcalc_raw = np.where(np.isfinite(vfrac), 1.0 - vfrac, 0.5)
    vdolo_raw = np.where(np.isfinite(vfrac), vfrac, 0.5)
    vcalc = vcalc_raw * (1.0 - vcld)
    vdolo = vdolo_raw * (1.0 - vcld)
    return vcld, vcalc, vdolo


def _vectorized_ssc_neu_den(
    nphi: np.ndarray,
    rhob: np.ndarray,
    dry_sand_point: Tuple[float, float],
    dry_silt_point: Tuple[float, float],
    dry_clay_point: Tuple[float, float],
    fluid_point: Tuple[float, float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized SandSiltClay (modified Kuttan) lithology estimation for a single realization."""
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point

    rock_len = float(length_a_b(A, C))
    res_len = float(length_a_b(A, B))
    if rock_len <= 1e-9:
        rock_len = 1.0
    res_ratio = np.clip(res_len / rock_len, 1e-6, 1.0 - 1e-6)
    res_len_eff = rock_len * res_ratio
    non_res_len_eff = rock_len * (1.0 - res_ratio)

    var_x, var_y = line_intersection((A, C), (D, (nphi, rhob)))
    proj_len = np.sqrt((var_x - A[0]) ** 2 + (var_y - A[1]) ** 2)
    pt_ratio = proj_len / rock_len

    siltclayratio = 0.25
    in_res = pt_ratio <= res_ratio

    vsand = np.where(in_res, (-proj_len / res_len_eff) + 1.0, 0.0)
    vsilt = np.where(
        in_res,
        (siltclayratio / res_len_eff) * proj_len,
        (-siltclayratio / non_res_len_eff) * proj_len + (rock_len * siltclayratio / non_res_len_eff),
    )
    vcld = np.where(
        in_res,
        ((1.0 - siltclayratio) / res_len_eff) * proj_len,
        (siltclayratio / non_res_len_eff) * proj_len + (1.0 - (siltclayratio * rock_len / non_res_len_eff)),
    )

    vsand = np.clip(np.nan_to_num(vsand, nan=0.0), 0.0, 1.0)
    vsilt = np.clip(np.nan_to_num(vsilt, nan=0.0), 0.0, 1.0)
    vcld = np.clip(np.nan_to_num(vcld, nan=0.0), 0.0, 1.0)
    total = vsand + vsilt + vcld
    valid_tot = total > 1e-9
    vsand = np.where(valid_tot, vsand / total, 0.0)
    vsilt = np.where(valid_tot, vsilt / total, 0.0)
    vcld = np.where(valid_tot, vcld / total, 1.0)
    return vsand, vsilt, vcld


def run_log_uncertainty(
    df: pd.DataFrame,
    model: str = "ssc",
    saturation_method: str = "waxman_smits",
    n_samples: int = 100,
    param_bounds: Optional[Dict[str, Union[Tuple[float, ...], List[float], float]]] = None,
    carbonate_model: str = "double",
    carbonate_type: str = "limestone",
    temp_grad: Optional[Union[np.ndarray, float]] = None,
    water_salinity: Optional[float] = None,
    seed: Optional[int] = 42,
    return_realizations: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, Dict[str, np.ndarray]]]:
    """Run Latin Hypercube petrophysical uncertainty propagation across well log intervals.

    Extends the Latin Hypercube engine to log-level petrophysical parameter uncertainty
    ($a, m, n, R_w, \\rho_{ma}, \\rho_{sh}$), running `n_samples` (default 100) realizations of
    `Carbonate` or `SandSiltClay` lithology/porosity + `waxman_smits_saturation` (or `archie_saturation`)
    and computing P10 / P50 / P90 CPI log curves (`SW_P10`, `SW_P50`, `SW_P90`, `SWT_P10`, `SWT_P50`,
    `SWT_P90`, `PHIT_P10`, `PHIT_P50`, `PHIT_P90`, `VCLAY_P10`, `VCLAY_P50`, `VCLAY_P90`,
    `BVHC_P10`, `BVHC_P50`, `BVHC_P90`).

    Args:
        df (pd.DataFrame): Input well log DataFrame containing `NPHI`, `RHOB`, `RT` (or lowercase
            equivalents), and optionally `GR`, `DEPTH`, `TVD`/`TVDSS`, `PHIT`, `VCLAY`.
        model (str, optional): Lithology model to propagate: `'ssc'` (`SandSiltClay`) or
            `'carbonate'` / `'carb'` (`Carbonate`). Defaults to `'ssc'`.
        saturation_method (str, optional): Saturation equation: `'waxman_smits'` or `'archie'`.
            Defaults to `'waxman_smits'`.
        n_samples (int, optional): Number of Latin Hypercube realizations. Defaults to 100.
        param_bounds (dict, optional): Parameter uncertainty bounds for any subset of
            `('a', 'm', 'n', 'rw', 'rho_ma', 'rho_sh')`. Unspecified parameters use
            `DEFAULT_LOG_PARAM_BOUNDS`. Each entry can be `(min, max, mean, std)`, `(min, max)`, or scalar.
        carbonate_model (str, optional): `'single'` or `'double'` when `model='carbonate'`. Defaults to `'double'`.
        carbonate_type (str, optional): `'limestone'` or `'dolostone'` when `carbonate_model='single'`.
        temp_grad (np.ndarray or float, optional): Formation temperature [degC] for Juhasz `B` estimation.
        water_salinity (float, optional): If provided and `'rw'` is not explicitly overridden in `param_bounds`,
            scales baseline $R_w$ from temperature and salinity.
        seed (int, optional): Random seed for LatinHypercube sampler. Defaults to 42.
        return_realizations (bool, optional): If True, returns `(result_df, realizations_dict)`.

    Returns:
        pd.DataFrame or tuple[pd.DataFrame, dict]: Copy of `df` enriched with P10/P50/P90 CPI log curves
        (`SW_P10`, `SW_P50`, `SW_P90`, `SWT_P10`, `SWT_P50`, `SWT_P90`, `PHIT_P10`, `PHIT_P50`, `PHIT_P90`,
        `VCLAY_P10`, `VCLAY_P50`, `VCLAY_P90`, `BVHC_P10`, `BVHC_P50`, `BVHC_P90`), plus realizations if requested.
    """
    out_df = df.copy()
    n_rows = len(out_df)
    if n_rows == 0:
        return (out_df, {}) if return_realizations else out_df

    col_lookup = {c.upper(): c for c in out_df.columns}

    def _get_col(name: str, default_val: float = np.nan) -> np.ndarray:
        actual = col_lookup.get(name.upper())
        if actual is not None:
            return pd.to_numeric(out_df[actual], errors="coerce").to_numpy(dtype=float)
        return np.full(n_rows, default_val, dtype=float)

    nphi = _get_col("NPHI")
    rhob = _get_col("RHOB")
    rt = _get_col("RT")
    gr = _get_col("GR")
    depth = _get_col("DEPTH", default_val=1000.0)
    tvd = _get_col("TVDSS", default_val=np.nan)
    if np.all(np.isnan(tvd)):
        tvd = _get_col("TVD", default_val=np.nan)
    if np.all(np.isnan(tvd)):
        tvd = depth

    # Merge user parameter bounds with defaults
    merged_bounds: Dict[str, Union[Tuple[float, ...], List[float], float]] = dict(DEFAULT_LOG_PARAM_BOUNDS)
    if model.lower() in ("carbonate", "carb") and (param_bounds is None or "rho_ma" not in param_bounds):
        merged_bounds["rho_ma"] = (2.68, 2.85, 2.71, 0.04)
        merged_bounds["rho_sh"] = (2.65, 2.82, 2.78, 0.04)
    if param_bounds:
        for k, v in param_bounds.items():
            if v is not None:
                merged_bounds[k] = v

    lhs_df = sample_lhs_parameters(merged_bounds, n_samples=n_samples, seed=seed)

    # Precompute baseline shale volume from GR or existing VCLAY/VSHALE for Carbonate
    existing_vclay = _get_col("VCLAY")
    if np.all(np.isnan(existing_vclay)):
        existing_vclay = _get_col("VSHALE")
    if not np.all(np.isnan(gr)):
        vsh_gr_base = np.asarray(gr_index(pd.Series(gr)), dtype=float)
    elif not np.all(np.isnan(existing_vclay)):
        vsh_gr_base = np.clip(np.nan_to_num(existing_vclay, nan=0.1), 0.0, 1.0)
    else:
        vsh_gr_base = np.zeros(n_rows, dtype=float)

    # Precompute wet clay NPHI/RHOB for SandSiltClay if NPHI & RHOB are present
    has_nphi_rhob = not (np.all(np.isnan(nphi)) or np.all(np.isnan(rhob)))
    wet_clay_point = Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
    if has_nphi_rhob and not all(wet_clay_point):
        try:
            rhob_clean = remove_outliers(rhob)
            nphi_clean = remove_outliers(nphi)
            _, rhob_max_line = min_max_line(rhob_clean, 0.1)
            _, nphi_max_line = min_max_line(nphi_clean, 0.1)
            wet_clay_point = (float(np.nanmax(nphi_max_line)), float(np.nanmin(rhob_max_line)))
        except Exception:
            wet_clay_point = (0.45, 2.35)
    elif not all(wet_clay_point):
        wet_clay_point = (0.45, 2.35)

    silt_angle = float(Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"])
    drysilt_nphi = 1.0 - 1.68 * np.tan((silt_angle - 90.0) * np.pi / 180.0)
    fluid_pt = (1.0, 1.0)

    # Temperature profile for Waxman-Smits B parameter
    if temp_grad is not None:
        temp_arr = np.asarray(temp_grad, dtype=float)
    else:
        safe_tvd = np.where(np.isfinite(tvd) & (tvd > 0), tvd, 1500.0)
        temp_arr = np.asarray(estimate_temperature_gradient(safe_tvd, "metric"), dtype=float)

    existing_phit = _get_col("PHIT")
    phit_reals = np.zeros((n_samples, n_rows), dtype=float)
    phie_reals = np.zeros((n_samples, n_rows), dtype=float)
    vclay_reals = np.zeros((n_samples, n_rows), dtype=float)
    sw_reals = np.zeros((n_samples, n_rows), dtype=float)

    model_key = model.lower().strip()
    sat_key = saturation_method.lower().strip()

    for i in range(n_samples):
        row_p = lhs_df.iloc[i]
        a_i = float(row_p["a"])
        m_i = float(row_p["m"])
        n_i = float(row_p["n"])
        rw_i = float(row_p["rw"])
        rho_ma_i = float(row_p["rho_ma"])
        rho_sh_i = float(row_p["rho_sh"])

        if water_salinity is not None and (param_bounds is None or "rw" not in param_bounds):
            rw_base = estimate_rw_temperature_salinity(temp_arr, water_salinity)
            # Perturb around salinity-derived Rw using the relative LHS factor
            rw_mean_default = DEFAULT_LOG_PARAM_BOUNDS["rw"][2]
            rw_profile = np.clip(rw_base * (rw_i / rw_mean_default), 1e-4, 5.0)
        else:
            rw_profile = np.full(n_rows, rw_i, dtype=float)

        if has_nphi_rhob:
            if model_key in ("carbonate", "carb"):
                calc_pt = (Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"][0], rho_ma_i)
                dolo_delta = (
                    Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"][1]
                    - Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"][1]
                )
                dolo_pt = (
                    Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"][0],
                    rho_ma_i + dolo_delta,
                )
                clay_pt = (Config.CARB_NEU_DEN_ENDPOINTS["DRY_CLAY_POINT"][0], rho_sh_i)

                vcld_u, vcalc_u, vdolo_u = _vectorized_carbonate_neu_den(
                    nphi,
                    rhob,
                    vsh_gr_base,
                    dry_calc_point=calc_pt,
                    dry_dolo_point=dolo_pt,
                    dry_clay_point=clay_pt,
                    fluid_point=fluid_pt,
                    carbonate_model=carbonate_model,
                    carbonate_type=carbonate_type,
                )
                rho_ma_log = (
                    vcld_u * rho_sh_i
                    + vcalc_u * rho_ma_i
                    + vdolo_u * (rho_ma_i + dolo_delta)
                )
                phit_i = np.clip(density_porosity(rhob, rho_ma_log, rho_fluid=fluid_pt[1]), 1e-4, 0.60)
                norms = normalize_volumetric(phit_i, vclay=vcld_u, vcalc=vcalc_u, vdolo=vdolo_u)
                vclay_i = np.clip(norms["vclay"], 0.0, 1.0 - phit_i)
            else:
                # SandSiltClay (modified Kuttan)
                sand_pt = (Config.SSC_ENDPOINTS["DRY_SAND_POINT"][0], rho_ma_i)
                silt_pt = (drysilt_nphi, 0.5 * (rho_ma_i + rho_sh_i))
                denom_wc = fluid_pt[0] - wet_clay_point[0]
                m_slope = (
                    (fluid_pt[1] - wet_clay_point[1]) / denom_wc
                    if abs(denom_wc) > 1e-9
                    else -1.5
                )
                dryclay_nphi = ((rho_sh_i - fluid_pt[1]) / m_slope) + fluid_pt[0]
                clay_pt = (float(np.clip(dryclay_nphi, 0.15, 0.75)), rho_sh_i)

                vsand_u, vsilt_u, vcld_u = _vectorized_ssc_neu_den(
                    nphi,
                    rhob,
                    dry_sand_point=sand_pt,
                    dry_silt_point=silt_pt,
                    dry_clay_point=clay_pt,
                    fluid_point=fluid_pt,
                )
                rho_ma_log = (
                    vsand_u * rho_ma_i
                    + vsilt_u * silt_pt[1]
                    + vcld_u * rho_sh_i
                )
                phit_i = np.clip(density_porosity(rhob, rho_ma_log, rho_fluid=fluid_pt[1]), 1e-4, 0.60)
                norms = normalize_volumetric(phit_i, vsand=vsand_u, vsilt=vsilt_u, vclay=vcld_u)
                vclay_i = np.clip(norms["vclay"], 0.0, 1.0 - phit_i)

            phit_sh_i = estimate_shale_porosity(nphi, phit_i)
            phie_i = np.clip(effective_porosity(phit_i, phit_sh_i, vclay_i), 0.0, phit_i)
        else:
            # Fallback if NPHI/RHOB are missing: perturb existing PHIT via matrix/shale density sensitivity
            base_phi = np.where(np.isfinite(existing_phit), existing_phit, 0.15)
            vclay_i = vsh_gr_base
            rho_eff_base = (1.0 - vclay_i) * 2.68 + vclay_i * 2.68
            rho_eff_i = (1.0 - vclay_i) * rho_ma_i + vclay_i * rho_sh_i
            rhob_syn = (1.0 - base_phi) * rho_eff_base + base_phi * 1.0
            phit_i = np.clip(density_porosity(rhob_syn, rho_eff_i, rho_fluid=1.0), 1e-4, 0.60)
            phit_sh_i = np.full(n_rows, 0.15, dtype=float)
            phie_i = np.clip(phit_i * (1.0 - vclay_i), 0.0, phit_i)

        if sat_key == "archie":
            sw_i = archie_saturation(rt=rt, rw=rw_profile, phit=phit_i, a=a_i, m=m_i, n=n_i)
            invalid_mask = ~((rt > 0) & np.isfinite(rt) & (phit_i > 1e-3) & np.isfinite(phit_i))
            sw_i = np.where(invalid_mask, 1.0, np.clip(sw_i, 0.0, 1.0))
        else:
            B_i = estimate_b_waxman_smits(temp_arr, rw_profile)
            qvn_i = estimate_qvn(vclay=vclay_i, phit=phit_i, phit_clay=phit_sh_i)
            sw_i = _vectorized_waxman_smits_saturation(
                rt=rt,
                rw=rw_profile,
                phit=phit_i,
                Qv=qvn_i,
                B=B_i,
                a=a_i,
                m=m_i,
                n=n_i,
            )

        phit_reals[i] = phit_i
        phie_reals[i] = phie_i
        vclay_reals[i] = vclay_i
        sw_reals[i] = sw_i

    bvhc_reals = phit_reals * (1.0 - sw_reals)

    # Calculate P10, P50, P90 percentiles across realizations (axis=0)
    sw_p10, sw_p50, sw_p90 = np.nanpercentile(sw_reals, [10, 50, 90], axis=0)
    phit_p10, phit_p50, phit_p90 = np.nanpercentile(phit_reals, [10, 50, 90], axis=0)
    phie_p10, phie_p50, phie_p90 = np.nanpercentile(phie_reals, [10, 50, 90], axis=0)
    vclay_p10, vclay_p50, vclay_p90 = np.nanpercentile(vclay_reals, [10, 50, 90], axis=0)
    bvhc_p10, bvhc_p50, bvhc_p90 = np.nanpercentile(bvhc_reals, [10, 50, 90], axis=0)

    out_df["SW_P10"] = sw_p10
    out_df["SW_P50"] = sw_p50
    out_df["SW_P90"] = sw_p90

    out_df["SWT_P10"] = sw_p10
    out_df["SWT_P50"] = sw_p50
    out_df["SWT_P90"] = sw_p90

    out_df["PHIT_P10"] = phit_p10
    out_df["PHIT_P50"] = phit_p50
    out_df["PHIT_P90"] = phit_p90

    out_df["PHIE_P10"] = phie_p10
    out_df["PHIE_P50"] = phie_p50
    out_df["PHIE_P90"] = phie_p90

    out_df["VCLAY_P10"] = vclay_p10
    out_df["VCLAY_P50"] = vclay_p50
    out_df["VCLAY_P90"] = vclay_p90

    out_df["BVHC_P10"] = bvhc_p10
    out_df["BVHC_P50"] = bvhc_p50
    out_df["BVHC_P90"] = bvhc_p90

    # Populate primary curves if missing so plotly_log and calc_reservoir_summary work seamlessly
    if "SWT" not in out_df.columns or out_df["SWT"].isna().all():
        out_df["SWT"] = sw_p50
    if "SW" not in out_df.columns or out_df["SW"].isna().all():
        out_df["SW"] = sw_p50
    if "PHIT" not in out_df.columns or out_df["PHIT"].isna().all():
        out_df["PHIT"] = phit_p50
    if "PHIE" not in out_df.columns or out_df["PHIE"].isna().all():
        out_df["PHIE"] = phie_p50
    if "VCLAY" not in out_df.columns or out_df["VCLAY"].isna().all():
        out_df["VCLAY"] = vclay_p50

    if return_realizations:
        return out_df, {
            "parameters": lhs_df,
            "SW": sw_reals,
            "PHIT": phit_reals,
            "PHIE": phie_reals,
            "VCLAY": vclay_reals,
            "BVHC": bvhc_reals,
        }
    return out_df


def calc_stochastic_reservoir_summary(
    df: pd.DataFrame,
    cutoffs: Optional[Dict[str, float]] = None,
    model: str = "ssc",
    saturation_method: str = "waxman_smits",
    n_samples: int = 100,
    param_bounds: Optional[Dict[str, Union[Tuple[float, ...], List[float], float]]] = None,
    area_bound: Tuple[float, ...] = (500.0, 2500.0, 1500.0, 400.0),
    volume_factor_bound: Tuple[float, ...] = (1.1, 1.4),
    seed: Optional[int] = 42,
) -> Dict[str, Any]:
    """Run log-level LHS uncertainty and compute P10/P50/P90 CPI curves, zone summaries, and volumes.

    Args:
        df (pd.DataFrame): Input well log DataFrame.
        cutoffs (dict, optional): Cutoffs dict `{'VSHALE': 0.4, 'PHIT': 0.01, 'SWT': 0.9}`.
        model (str, optional): `'ssc'` or `'carbonate'`. Defaults to `'ssc'`.
        saturation_method (str, optional): `'waxman_smits'` or `'archie'`. Defaults to `'waxman_smits'`.
        n_samples (int, optional): Number of Latin Hypercube realizations. Defaults to 100.
        param_bounds (dict, optional): Parameter uncertainty bounds for `a, m, n, rw, rho_ma, rho_sh`.
        area_bound (tuple, optional): Area bound `(min, max, mean, std)` in acres for volumetric STOIIP.
        volume_factor_bound (tuple, optional): Formation volume factor `(min, max)` for volumetric STOIIP.
        seed (int, optional): Random seed. Defaults to 42.

    Returns:
        dict: Dictionary containing:
            - `cpi_df`: DataFrame with `DEPTH`, `SW_P10`, `SW_P50`, `SW_P90`, `PHIT_P10`, `PHIT_P50`, `PHIT_P90`, etc.
            - `summary_df`: Reservoir summary DataFrame enriched with P10/P50/P90 zone metrics and STOIIP (MMbbl)
            - `volumetrics`: Zone-by-zone volumetric P10/P50/P90 dictionary
    """
    from quick_pp.ressum import calc_reservoir_summary

    cutoffs = cutoffs or {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9}
    cpi_df, reals = run_log_uncertainty(
        df,
        model=model,
        saturation_method=saturation_method,
        n_samples=n_samples,
        param_bounds=param_bounds,
        seed=seed,
        return_realizations=True,
    )

    col_lookup = {c.upper(): c for c in cpi_df.columns}
    depth = pd.to_numeric(cpi_df[col_lookup.get("DEPTH", cpi_df.columns[0])], errors="coerce").to_numpy(dtype=float)
    perm_col = col_lookup.get("PERM")
    perm = (
        pd.to_numeric(cpi_df[perm_col], errors="coerce").fillna(10.0).to_numpy(dtype=float)
        if perm_col
        else np.full(len(cpi_df), 10.0, dtype=float)
    )
    zones_col = col_lookup.get("ZONES")
    zones = (
        cpi_df[zones_col].fillna("ALL").astype(str).to_numpy()
        if zones_col
        else np.full(len(cpi_df), "ALL", dtype=object)
    )

    # P50 baseline reservoir summary
    summary_p50 = calc_reservoir_summary(
        depth=depth,
        vshale=cpi_df["VCLAY_P50"].to_numpy(dtype=float),
        phit=cpi_df["PHIT_P50"].to_numpy(dtype=float),
        swt=cpi_df["SW_P50"].to_numpy(dtype=float),
        perm=perm,
        zones=zones,
        cutoffs=cutoffs,
    )
    # Optimistic (low Sw = SW_P10, high porosity = PHIT_P90) and Conservative (high Sw = SW_P90, low porosity = PHIT_P10)
    summary_opt = calc_reservoir_summary(
        depth=depth,
        vshale=cpi_df["VCLAY_P10"].to_numpy(dtype=float),
        phit=cpi_df["PHIT_P90"].to_numpy(dtype=float),
        swt=cpi_df["SW_P10"].to_numpy(dtype=float),
        perm=perm,
        zones=zones,
        cutoffs=cutoffs,
    )
    summary_con = calc_reservoir_summary(
        depth=depth,
        vshale=cpi_df["VCLAY_P90"].to_numpy(dtype=float),
        phit=cpi_df["PHIT_P10"].to_numpy(dtype=float),
        swt=cpi_df["SW_P90"].to_numpy(dtype=float),
        perm=perm,
        zones=zones,
        cutoffs=cutoffs,
    )

    merge_keys = ["zones", "flag"]
    summary_df = summary_p50.merge(
        summary_opt[merge_keys + ["net", "av_phit", "av_swt", "av_bvhc"]].rename(
            columns={
                "net": "net_p90",
                "av_phit": "av_phit_p90",
                "av_swt": "av_swt_p10",
                "av_bvhc": "av_bvhc_p90",
            }
        ),
        on=merge_keys,
        how="left",
    ).merge(
        summary_con[merge_keys + ["net", "av_phit", "av_swt", "av_bvhc"]].rename(
            columns={
                "net": "net_p10",
                "av_phit": "av_phit_p10",
                "av_swt": "av_swt_p90",
                "av_bvhc": "av_bvhc_p10",
            }
        ),
        on=merge_keys,
        how="left",
    )
    summary_df["av_swt_p50"] = summary_df["av_swt"]
    summary_df["av_phit_p50"] = summary_df["av_phit"]
    summary_df["av_bvhc_p50"] = summary_df["av_bvhc"]

    # Sample area and Bo across realizations to compute zone-level P10/P50/P90 STOIIP (MMbbl)
    vol_lhs = sample_lhs_parameters(
        {"area": area_bound, "bo": volume_factor_bound},
        n_samples=n_samples,
        seed=seed,
    )
    area_s = np.maximum(0.0, vol_lhs["area"].to_numpy(dtype=float))
    bo_s = np.maximum(1e-6, vol_lhs["bo"].to_numpy(dtype=float))

    # Compute per-sample thickness h in feet (assume depth step)
    diffs = pd.Series(depth).diff().abs().dropna()
    pos_diffs = diffs[diffs > 0]
    step = float(pos_diffs.median()) if len(pos_diffs) > 0 else 0.5
    h_arr = pd.Series(depth).diff().bfill().fillna(step).abs().to_numpy(dtype=float)

    volumetrics_by_zone: Dict[str, Dict[str, float]] = {}
    unique_zones = np.unique(zones)
    for z in unique_zones:
        z_mask = zones == z
        if not np.any(z_mask):
            continue
        vsh_z = reals["VCLAY"][:, z_mask]
        phi_z = reals["PHIT"][:, z_mask]
        sw_z = reals["SW"][:, z_mask]
        h_z = h_arr[z_mask][None, :]

        pay_mask = (
            (vsh_z < cutoffs.get("VSHALE", 0.4))
            & (phi_z > cutoffs.get("PHIT", 0.01))
            & (sw_z < cutoffs.get("SWT", 0.9))
        )
        hcpv_ft = np.nansum(np.where(pay_mask, phi_z * (1.0 - sw_z) * h_z, 0.0), axis=1)
        stoiip_reals = (43560.0 * 0.1781) * area_s * hcpv_ft / bo_s * 1e-6
        v_p10, v_p50, v_p90 = np.nanpercentile(stoiip_reals, [10, 50, 90])
        volumetrics_by_zone[str(z)] = {
            "stoiip_p10_mmbbl": round(float(v_p10), 3),
            "stoiip_p50_mmbbl": round(float(v_p50), 3),
            "stoiip_p90_mmbbl": round(float(v_p90), 3),
            "stoiip_mean_mmbbl": round(float(np.nanmean(stoiip_reals)), 3),
        }

    summary_df["stoiip_p10_mmbbl"] = summary_df["zones"].map(
        lambda z: volumetrics_by_zone.get(str(z), {}).get("stoiip_p10_mmbbl", 0.0)
    )
    summary_df["stoiip_p50_mmbbl"] = summary_df["zones"].map(
        lambda z: volumetrics_by_zone.get(str(z), {}).get("stoiip_p50_mmbbl", 0.0)
    )
    summary_df["stoiip_p90_mmbbl"] = summary_df["zones"].map(
        lambda z: volumetrics_by_zone.get(str(z), {}).get("stoiip_p90_mmbbl", 0.0)
    )
    summary_df = summary_df.round(3)

    return {
        "cpi_df": cpi_df,
        "summary_df": summary_df,
        "volumetrics": volumetrics_by_zone,
    }

