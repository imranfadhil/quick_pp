"""Formation Pressure Testing (FPT), Fluid Gradient Regression, Contact Estimation, and Gas Composition Analysis.

This module provides petrophysical fluid typing and formation pressure analysis workflows:
- Linear, Huber-IRLS, RANSAC, fixed-slope, and manual fluid gradient fitting (`FormationPressureAnalyzer`, `FluidGradientResult`)
- Free-Fluid Level / Fluid Contact (`GOC`, `OWC`/`FWL`, `GWC`) intersection with Monte Carlo P10/P50/P90 uncertainty (`FluidContactResult`)
- Excess pressure profile calculation relative to an aquifer baseline (`FormationPressureAnalyzer.compute_excess_pressure`)
- Unit-agnostic gradient/density conversion (`gradient_to_density`, `density_to_gradient`)
- Haworth (1985) mud-gas composition ratios (`gas_composition_analysis`) and physical fluid segregation (`fix_fluid_segregation`)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

from tqdm import tqdm

from quick_pp import logger
from quick_pp.utils import straight_line_func as func

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {"axes.labelsize": 10, "xtick.labelsize": 10, "legend.fontsize": "small"}
)


def fit_pressure_gradient(tvdss, formation_pressure):
    """Fit pressure gradient from formation pressure and true vertical depth subsea.

    Args:
        tvdss (array-like): True vertical depth subsea.
        formation_pressure (array-like): Formation pressure.

    Returns:
        tuple: Gradient and intercept of the best-fit line.
    """
    logger.debug(f"Fitting pressure gradient with {len(tvdss)} data points")
    popt, _ = curve_fit(func, tvdss, formation_pressure)
    gradient, intercept = round(popt[0], 3), round(popt[1], 3)
    logger.debug(f"Fitted pressure gradient: {gradient} psi/ft, intercept: {intercept}")
    return gradient, intercept


def fluid_contact_plot(
    tvdss,
    formation_pressure,
    m,
    c,
    fluid_type: str = "WATER",
    ylim: tuple = (5000, 3000),
    xlim: tuple = (1000, 5000),
    goc: float = 0,
    owc: float = 0,
    gwc: float = 0,
):
    """Generate fluid contact plot which is used to determine the fluid contact in a well.

    Args:
        tvdss (array-like): True vertical depth subsea.
        formation_pressure (array-like): Formation pressure.
        m (float): Gradient of the best-fit line.
        c (float): Intercept of the best-fit line.
        fluid_type (str, optional): Type of fluid. Defaults to 'WATER'.
        ylim (tuple, optional): Y-axis limits for the plot. Defaults to (5000, 3000).
        xlim (tuple, optional): X-axis limits for the plot. Defaults to (1000, 5000).
        goc (float, optional): Gas-oil contact. Defaults to 0.
        owc (float, optional): Oil-water contact. Defaults to 0.
        gwc (float, optional): Gas-water contact. Defaults to 0.

    Returns:
        None: The function generates a plot but does not return any object.
    """
    logger.debug(
        f"Creating fluid contact plot for {fluid_type} with gradient {m} psi/ft"
    )
    color_dict = {
        "OIL": ("green", "o"),
        "GAS": ("red", "x"),
        "WATER": ("blue", "^"),
        "HYDROSTATIC": ("brown", "s"),
    }
    marker = color_dict.get(fluid_type.upper(), ("black", "o"))
    label = f"{fluid_type} gradient: {round(m, 3)} psi/ft"
    sc = plt.scatter(
        formation_pressure, tvdss, label=label, c=marker[0], marker=marker[1]
    )

    line_color = sc.get_facecolors()[0]
    line_color[-1] = 0.5
    tvd_pts = np.linspace(ylim[0] - 30, ylim[1] + 30, 30)
    plt.plot(func(tvd_pts, m, c), tvd_pts, color=line_color)

    plt.ylim(ylim)
    plt.ylabel("TVDSS (ft)")
    plt.xlim(xlim)
    plt.xlabel("Formation Pressure (psi)")
    plt.legend()
    plt.title("Fluid Contact Plot")
    logger.debug("Fluid contact plot created successfully")


def gas_composition_analysis(c1, c2, c3, ic4, nc4, ic5, nc5):
    """Analyze hydrocarbon type based on gas composition (Haworth 1985).

    Hydrocarbon type flag (HC_TYPE_FLAG) is classified into 6 categories:
    0: Non Representative Sample
    1: Non Productive Dry Gas
    2: Potentially Productive Gas
    3: Potentially Productive High GOR Oil
    4: Potentially Productive Condensate
    5: Potentially Productive Oil

    Gas quality flag (GQ_FLAG) is classified into 2 categories:
    0: Not within the range of 0.8 to 1.2
    1: Within the range of 0.8 to 1.2

    Args:
        c1 (array-like): Concentration of methane in ppm.
        c2 (array-like): Concentration of ethane in ppm.
        c3 (array-like): Concentration of propane in ppm.
        ic4 (array-like): Concentration of isobutane in ppm.
        nc4 (array-like): Concentration of normal butane in ppm.
        ic5 (array-like): Concentration of isopentane in ppm.
        nc5 (array-like): Concentration of normal pentane in ppm.

    Returns:
        pd.DataFrame: Gas composition analysis.
    """
    logger.debug("Performing gas composition analysis using Haworth 1985 method")
    total = c1 + c2 + c3 + ic4 + nc4 + ic5 + nc5

    # Gas Quality Ratio and flag
    gqr = total / (c1 + 2 * c2 + 3 * c3 + 4 * (ic4 + nc4) + 5 * (ic5 + nc5))
    gq_flag = np.where((gqr > 0.8) & (gqr < 1.2), 1, 0)
    logger.debug(f"Gas quality ratio range: {np.min(gqr):.3f} - {np.max(gqr):.3f}")

    # Wetness ratio
    wh = (c2 + c3 + ic4 + nc4 + ic5 + nc5) / total * 100
    # Balance ration
    bal = (c1 + c2) / (c3 + ic4 + nc4 + ic5 + nc5)
    # Character ratio
    ch = (ic4 + nc4 + ic5 + nc5) / c3

    hc_type_dict = {
        0: "Non Representative Sample",
        1: "Non Productive Dry Gas",
        2: "Potentially Productive Gas",
        3: "Potentially Productive High GOR Oil",
        4: "Potentially Productive Condensate",
        5: "Potentially Productive Oil",
    }
    hc_type = np.zeros(1 if isinstance(c1, int) else len(c1))
    hc_type = np.where((wh < 5) & (bal > 100), 1, hc_type)
    hc_type = np.where((wh > 5) & (wh < 17.5) & (wh < bal) & (bal < 100), 2, hc_type)
    hc_type = np.where((wh > 5) & (wh < 17.5) & (wh > bal) & (ch > 0.5), 3, hc_type)
    hc_type = np.where((wh > 5) & (wh < 17.5) & (wh > bal) & (ch < 0.5), 4, hc_type)
    hc_type = np.where((wh > 17.5) & (wh < 40) & (wh > bal), 5, hc_type)

    result_df = pd.DataFrame(
        {
            "Gas Quality Ratio": gqr,
            "GQ_FLAG": gq_flag,
            "Wetness Ratio": wh,
            "Balance Ratio": bal,
            "Character Ratio": ch,
            "Hydrocarbon Type": [hc_type_dict[x] for x in hc_type],
            "HC_TYPE_FLAG": hc_type,
        }
    )

    logger.debug(
        f"Gas composition analysis completed. HC types found: {np.unique(hc_type)}"
    )
    return result_df


def fix_fluid_segregation(df: pd.DataFrame) -> pd.DataFrame:
    """Corrects fluid assignments in a DataFrame based on physical segregation.

    This function processes a petrophysical evaluation DataFrame on a per-well,
    per-hydrocarbon-interval basis. It enforces the rule that within a continuous
    hydrocarbon column, gas should be found above oil. If any oil is flagged at or
    above the deepest occurrence of gas within an interval, it is re-assigned as gas.

    The function modifies the DataFrame by creating 'VOIL' and 'VGAS' columns
    and adjusting their values. The original 'VHC' column is zeroed out where
    'VOIL' or 'VGAS' are populated.

    Args:
        df (pd.DataFrame): Input DataFrame containing well data, requiring at least
                           'WELL_NAME', 'DEPTH', 'VHC', 'OIL_FLAG', and 'GAS_FLAG' columns.

    Returns:
        pd.DataFrame: The DataFrame with corrected fluid volume assignments.
    """
    logger.info("Generating fluid volume fractions from OIL_FLAG and GAS_FLAG")
    df["VOIL"] = df["OIL_FLAG"] * df["VHC"]
    df["VGAS"] = df["GAS_FLAG"] * df["VHC"]

    for well_name, well_df in tqdm(
        df.groupby("WELL_NAME"), desc="Fixing fluid segregation"
    ):
        tqdm.write(f"Processing well {well_name}")
        # Fix fluid segregation issues bounded by continuous hydrocarbon intervals
        hc_mask = (well_df["VHC"] >= 1e-2).astype(int)
        # Identify continuous hydrocarbon intervals
        hc_groups = (hc_mask.diff() != 0).cumsum()

        for _, group_df in well_df.groupby(hc_groups):
            # Process only hydrocarbon-bearing intervals
            if hc_mask.loc[group_df.index].sum() > 0:
                # If both gas and oil are predicted in the same interval
                if (group_df["GAS_FLAG"] == 1).any() and (
                    group_df["OIL_FLAG"] == 1
                ).any():
                    # Find the deepest depth where gas is predicted
                    last_gas_depth = group_df[group_df["GAS_FLAG"] == 1]["DEPTH"].max()
                    # Identify indices of oil intervals above this deepest gas
                    oil_above_gas_indices = group_df[
                        (group_df["DEPTH"] <= last_gas_depth)
                        & (group_df["OIL_FLAG"] == 1)
                    ].index
                    # Re-assign oil volumes to gas for these intervals in the main dataframe
                    df.loc[oil_above_gas_indices, "VGAS"] = df.loc[
                        oil_above_gas_indices, "VHC"
                    ]
                    df.loc[oil_above_gas_indices, "VOIL"] = 0
                if (group_df["GAS_FLAG"] == 1).any() and (
                    group_df["OIL_FLAG"] == 0
                ).all():
                    df.loc[group_df.index, "VGAS"] = df.loc[group_df.index, "VHC"]
                    df.loc[group_df.index, "VHC"] = 0
                if (group_df["OIL_FLAG"] == 1).any() and (
                    group_df["GAS_FLAG"] == 0
                ).all():
                    df.loc[group_df.index, "VOIL"] = df.loc[group_df.index, "VHC"]
                    df.loc[group_df.index, "VHC"] = 0

    df["VHC"] = np.where((df["VOIL"] > 0) | (df["VGAS"] > 0), 0, df["VHC"])

    return df


# =============================================================================
# Formation Pressure Test (FPT) Fluid Gradient & Contact Analysis
# =============================================================================

FluidPhase = Literal["GAS", "OIL", "WATER"]
FLUID_PHASES: Tuple[str, ...] = ("GAS", "OIL", "WATER")
TEST_QUALITIES: Tuple[str, ...] = (
    "VALID",
    "TIGHT",
    "SUPERCHARGED",
    "LOST_SEAL",
    "EXCLUDED",
)

# Typical in-situ density bounds (g/cc) used for physical sanity checks
FLUID_DENSITY_BOUNDS: Dict[str, Tuple[float, float]] = {
    "GAS": (0.05, 0.45),
    "OIL": (0.50, 0.98),
    "WATER": (0.98, 1.25),
}
# Density break-points used for automatic phase suggestion from local gradients
_AUTO_GAS_OIL_DENSITY = 0.475
_AUTO_OIL_WATER_DENSITY = 0.98

PRESSURE_TO_PA: Dict[str, float] = {"psi": 6894.757293168, "bar": 1.0e5, "kPa": 1.0e3}
DEPTH_TO_M: Dict[str, float] = {"m": 1.0, "ft": 0.3048}
_G = 9.80665  # m/s^2

# Contact label for each (upper, lower) fluid pair
CONTACT_PAIRS: Tuple[Tuple[str, str, str], ...] = (
    ("GOC", "GAS", "OIL"),
    ("OWC", "OIL", "WATER"),
    ("GWC", "GAS", "WATER"),
)


class InvalidFluidGradientError(ValueError):
    """Raised when fluid gradients cannot yield a physically valid contact
    (parallel lines or inverted densities, i.e. lower fluid lighter than upper fluid)."""


def gradient_to_density(
    gradient: float, pressure_unit: str = "psi", depth_unit: str = "ft"
) -> float:
    """Convert a pressure gradient (`pressure_unit / depth_unit`) to in-situ fluid density (`g/cc`).

    Equivalent to: `psi/ft / 0.433527`, `psi/m / 1.42233`, `bar/m / 0.0980665`, `kPa/m / 9.80665`.

    Args:
        gradient (float): Pressure gradient value in `pressure_unit / depth_unit`.
        pressure_unit (str): Pressure unit (`'psi'`, `'bar'`, or `'kPa'`). Defaults to `'psi'`.
        depth_unit (str): Depth unit (`'ft'` or `'m'`). Defaults to `'ft'`.

    Returns:
        float: Equivalent in-situ fluid density in `g/cc`.
    """
    pa_per_m = float(gradient) * PRESSURE_TO_PA[pressure_unit] / DEPTH_TO_M[depth_unit]
    return pa_per_m / (_G * 1000.0)


def density_to_gradient(
    density_gcc: float, pressure_unit: str = "psi", depth_unit: str = "ft"
) -> float:
    """Convert an in-situ fluid density (`g/cc`) to a pressure gradient (`pressure_unit / depth_unit`).

    Args:
        density_gcc (float): In-situ fluid density in `g/cc`.
        pressure_unit (str): Target pressure unit (`'psi'`, `'bar'`, or `'kPa'`). Defaults to `'psi'`.
        depth_unit (str): Target depth unit (`'ft'` or `'m'`). Defaults to `'ft'`.

    Returns:
        float: Equivalent pressure gradient in `pressure_unit / depth_unit`.
    """
    pa_per_m = float(density_gcc) * 1000.0 * _G
    return pa_per_m * DEPTH_TO_M[depth_unit] / PRESSURE_TO_PA[pressure_unit]


def _norm_fluid(v: Any) -> str:
    s = str(v or "").strip().upper()
    if s in ("G", "GAS", "CONDENSATE"):
        return "GAS"
    if s in ("O", "OIL"):
        return "OIL"
    if s in ("W", "WTR", "WATER", "AQUIFER", "BRINE"):
        return "WATER"
    return "UNASSIGNED"


def _norm_quality(v: Any) -> str:
    s = str(v or "").strip().upper().replace(" ", "_").replace("-", "_")
    if not s or s in ("VALID", "GOOD", "OK", "NORMAL"):
        return "VALID"
    if s in ("TIGHT", "DRY"):
        return "TIGHT"
    if s in ("SUPERCHARGED", "SUPERCHARGE", "SC"):
        return "SUPERCHARGED"
    if s in ("LOST_SEAL", "LOSTSEAL", "NO_SEAL", "SEAL_FAILURE"):
        return "LOST_SEAL"
    if s in ("EXCLUDED", "EXCLUDE", "INVALID", "BAD"):
        return "EXCLUDED"
    return "VALID"


@dataclass
class FluidGradientResult:
    """Linear fluid gradient P = gradient * TVDSS + intercept for one fluid phase."""

    fluid_type: str
    gradient: float  # dP/dz (pressure_unit / depth_unit)
    intercept: float  # Pressure at TVDSS = 0
    density_gcc: float  # Equivalent in-situ fluid density (g/cc)
    r_squared: Optional[float]  # Goodness of fit (None for manual / fixed-slope lines)
    std_err_gradient: float  # Standard error of slope
    std_err_intercept: float  # Standard error of intercept
    cov_matrix: np.ndarray  # 2x2 covariance of (gradient, intercept)
    n_points: int  # Valid points used in regression
    tvdss_range: Tuple[float, float]
    method: str = "ols"  # ols | huber | ransac | fixed_gradient | manual
    pressure_unit: str = "psi"
    depth_unit: str = "ft"
    point_ids: List[Any] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def pressure_at(self, tvdss: Any) -> Any:
        """Evaluate fluid pressure `P = gradient * TVDSS + intercept` at one or more TVDSS depths.

        Args:
            tvdss (Any): Scalar or array-like True Vertical Depth Sub-Sea values.

        Returns:
            Any: Computed formation pressure in `self.pressure_unit`.
        """
        return self.gradient * np.asarray(tvdss, dtype=float) + self.intercept

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fluid gradient regression result into a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing gradient, intercept, density, R2, covariance, and warnings.
        """
        def _f(v: Any) -> Optional[float]:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        return {
            "fluid_type": self.fluid_type,
            "gradient": _f(self.gradient),
            "intercept": _f(self.intercept),
            "density_gcc": _f(self.density_gcc),
            "r_squared": _f(self.r_squared),
            "std_err_gradient": _f(self.std_err_gradient),
            "std_err_intercept": _f(self.std_err_intercept),
            "cov_matrix": [[_f(x) for x in row] for row in np.asarray(self.cov_matrix)],
            "n_points": int(self.n_points),
            "tvdss_range": [_f(self.tvdss_range[0]), _f(self.tvdss_range[1])],
            "method": self.method,
            "pressure_unit": self.pressure_unit,
            "depth_unit": self.depth_unit,
            "gradient_unit": f"{self.pressure_unit}/{self.depth_unit}",
            "point_ids": list(self.point_ids),
            "warnings": list(self.warnings),
        }


@dataclass
class FluidContactResult:
    """Free-fluid level / contact from the intersection of two fluid gradient lines."""

    contact_type: str  # GOC | OWC | GWC
    tvdss_p50: float  # Deterministic / median contact TVDSS
    tvdss_p10: float  # Shallow bound (10th percentile)
    tvdss_p90: float  # Deep bound (90th percentile)
    pressure_at_contact: float
    upper_fluid: FluidGradientResult
    lower_fluid: FluidGradientResult
    tvdss_deterministic: float = float("nan")
    std_tvdss: float = 0.0  # Delta-method standard deviation of contact depth
    n_valid_simulations: int = 0
    warnings: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the fluid contact result into a JSON-safe dictionary.

        Returns:
            Dict[str, Any]: Dictionary containing P10/P50/P90 contact TVDSS, contact pressure, and fluid densities.
        """
        def _f(v: Any) -> Optional[float]:
            try:
                fv = float(v)
            except (TypeError, ValueError):
                return None
            return fv if np.isfinite(fv) else None

        return {
            "contact_type": self.contact_type,
            "label": "OWC / FWL" if self.contact_type == "OWC" else self.contact_type,
            "tvdss_p10": _f(self.tvdss_p10),
            "tvdss_p50": _f(self.tvdss_p50),
            "tvdss_p90": _f(self.tvdss_p90),
            "tvdss_deterministic": _f(self.tvdss_deterministic),
            "std_tvdss": _f(self.std_tvdss),
            "pressure_at_contact": _f(self.pressure_at_contact),
            "upper_fluid": self.upper_fluid.fluid_type,
            "lower_fluid": self.lower_fluid.fluid_type,
            "upper_density_gcc": _f(self.upper_fluid.density_gcc),
            "lower_density_gcc": _f(self.lower_fluid.density_gcc),
            "n_valid_simulations": int(self.n_valid_simulations),
            "warnings": list(self.warnings),
        }


def _linear_fit_stats(
    z: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None
) -> Tuple[float, float, np.ndarray, Optional[float]]:
    """(Weighted) least-squares fit of p = m*z + c returning (m, c, cov, r2)."""
    n = len(z)
    if w is None:
        w = np.ones(n, dtype=float)
    X = np.column_stack([z, np.ones(n)])
    W = np.diag(w)
    XtWX = X.T @ W @ X
    beta = np.linalg.solve(XtWX, X.T @ W @ p)
    m, c = float(beta[0]), float(beta[1])
    resid = p - (m * z + c)
    if n > 2:
        dof = max(float(np.sum(w)) - 2.0, 1.0)
        s2 = float(np.sum(w * resid**2)) / dof
        cov = s2 * np.linalg.inv(XtWX)
    else:
        cov = np.zeros((2, 2), dtype=float)
    ss_tot = float(np.sum(w * (p - np.average(p, weights=w)) ** 2))
    r2 = 1.0 - float(np.sum(w * resid**2)) / ss_tot if ss_tot > 1e-12 else None
    return m, c, cov, r2


def _huber_fit(
    z: np.ndarray, p: np.ndarray, k: float = 1.345, max_iter: int = 50
) -> Tuple[float, float, np.ndarray, Optional[float]]:
    """Huber robust regression via iteratively re-weighted least squares (IRLS)."""
    w = np.ones(len(z), dtype=float)
    m, c, cov, r2 = _linear_fit_stats(z, p, w)
    for _ in range(max_iter):
        resid = p - (m * z + c)
        mad = float(np.median(np.abs(resid - np.median(resid))))
        scale = mad / 0.6745 if mad > 1e-12 else 1e-12
        u = np.abs(resid) / scale
        w_new = np.where(u <= k, 1.0, k / np.maximum(u, 1e-12))
        m_new, c_new, cov, r2 = _linear_fit_stats(z, p, w_new)
        converged = abs(m_new - m) < 1e-10 and abs(c_new - c) < 1e-8
        m, c, w = m_new, c_new, w_new
        if converged:
            break
    return m, c, cov, r2


def _ransac_inliers(
    z: np.ndarray,
    p: np.ndarray,
    threshold: Optional[float] = None,
    n_iter: int = 500,
    random_state: int = 42,
) -> np.ndarray:
    """RANSAC line inlier mask; threshold defaults to 2.5 x MAD of OLS residuals."""
    n = len(z)
    if n < 4:
        return np.ones(n, dtype=bool)
    if threshold is None or threshold <= 0:
        m0, c0, _, _ = _linear_fit_stats(z, p)
        r0 = p - (m0 * z + c0)
        mad = float(np.median(np.abs(r0 - np.median(r0))))
        threshold = max(2.5 * mad / 0.6745, 1e-6)
    rng = np.random.default_rng(random_state)
    best_mask = np.ones(n, dtype=bool)
    best_count, best_sse = -1, np.inf
    for _ in range(n_iter):
        i, j = rng.choice(n, size=2, replace=False)
        if abs(z[j] - z[i]) < 1e-9:
            continue
        m = (p[j] - p[i]) / (z[j] - z[i])
        c = p[i] - m * z[i]
        resid = np.abs(p - (m * z + c))
        mask = resid <= threshold
        cnt = int(mask.sum())
        sse = float(np.sum(resid[mask] ** 2))
        if cnt > best_count or (cnt == best_count and sse < best_sse):
            best_mask, best_count, best_sse = mask, cnt, sse
    if best_count < 2:
        return np.ones(n, dtype=bool)
    return best_mask


class FormationPressureAnalyzer:
    """Fluid gradient regression, contact estimation (P10/P50/P90) and excess pressure from FPT data.

    Units are agnostic: gradients are expressed in `pressure_unit / depth_unit` and in-situ densities
    are derived accordingly (supports psi+ft, psi+m, bar+m, kPa+m, ...). TVDSS is positive downwards.

    Example:
        >>> fpa = FormationPressureAnalyzer(df, pressure_unit="psi", depth_unit="ft")
        >>> fpa.auto_classify_points(min_mobility=0.5)
        >>> fpa.fit_gradient("OIL"); fpa.fit_gradient("WATER")
        >>> contacts = fpa.estimate_contacts()
    """

    def __init__(
        self,
        df: pd.DataFrame,
        pressure_unit: Literal["psi", "bar", "kPa"] = "psi",
        depth_unit: Literal["ft", "m"] = "ft",
        tvdss_col: str = "TVDSS",
        pressure_col: str = "PRESSURE",
        fluid_col: str = "FLUID_TYPE",
        quality_col: str = "TEST_QUALITY",
        mobility_col: str = "MOBILITY",
        id_col: Optional[str] = None,
    ):
        if pressure_unit not in PRESSURE_TO_PA:
            raise ValueError(f"Unsupported pressure unit '{pressure_unit}'")
        if depth_unit not in DEPTH_TO_M:
            raise ValueError(f"Unsupported depth unit '{depth_unit}'")
        for col in (tvdss_col, pressure_col):
            if col not in df.columns:
                raise ValueError(f"Column '{col}' not found in pressure data")
        self.pressure_unit = pressure_unit
        self.depth_unit = depth_unit
        self.tvdss_col = tvdss_col
        self.pressure_col = pressure_col
        self.fluid_col = fluid_col
        self.quality_col = quality_col
        self.mobility_col = mobility_col
        self.id_col = id_col

        self.df = df.copy()
        self.df[tvdss_col] = pd.to_numeric(self.df[tvdss_col], errors="coerce")
        self.df[pressure_col] = pd.to_numeric(self.df[pressure_col], errors="coerce")
        if fluid_col not in self.df.columns:
            self.df[fluid_col] = "UNASSIGNED"
        self.df[fluid_col] = self.df[fluid_col].map(_norm_fluid)
        if quality_col not in self.df.columns:
            self.df[quality_col] = "VALID"
        self.df[quality_col] = self.df[quality_col].map(_norm_quality)

        self.gradients: Dict[str, FluidGradientResult] = {}
        self.contacts: Dict[str, FluidContactResult] = {}
        self.warnings: List[str] = []

    # ------------------------------------------------------------------ helpers
    @property
    def gradient_unit(self) -> str:
        """Return the composite pressure gradient unit string (`'<pressure_unit>/<depth_unit>'`)."""
        return f"{self.pressure_unit}/{self.depth_unit}"

    def to_density(self, gradient: float) -> float:
        """Convert a gradient in `self.gradient_unit` to in-situ fluid density (`g/cc`).

        Args:
            gradient (float): Pressure gradient in `self.gradient_unit`.

        Returns:
            float: Fluid density in `g/cc`.
        """
        return gradient_to_density(gradient, self.pressure_unit, self.depth_unit)

    def from_density(self, density_gcc: float) -> float:
        """Convert an in-situ fluid density (`g/cc`) to a pressure gradient in `self.gradient_unit`.

        Args:
            density_gcc (float): Fluid density in `g/cc`.

        Returns:
            float: Pressure gradient in `self.gradient_unit`.
        """
        return density_to_gradient(density_gcc, self.pressure_unit, self.depth_unit)

    def _finite_mask(self) -> pd.Series:
        return self.df[self.tvdss_col].notna() & self.df[self.pressure_col].notna()

    def valid_mask(self, fluid_type: Optional[str] = None) -> pd.Series:
        """Return a boolean Series selecting rows with finite TVDSS/pressure and `TEST_QUALITY == 'VALID'`.

        Args:
            fluid_type (Optional[str]): Optional fluid phase (`'GAS'`, `'OIL'`, `'WATER'`) to filter by.

        Returns:
            pd.Series: Boolean mask indexed like `self.df`.
        """
        mask = self._finite_mask() & (self.df[self.quality_col] == "VALID")
        if fluid_type is not None:
            mask &= self.df[self.fluid_col] == fluid_type
        return mask

    def _ids(self, mask: pd.Series) -> List[Any]:
        if self.id_col and self.id_col in self.df.columns:
            return self.df.loc[mask, self.id_col].tolist()
        return self.df.index[mask].tolist()

    def _density_warnings(self, fluid_type: str, density: float) -> List[str]:
        lo, hi = FLUID_DENSITY_BOUNDS.get(fluid_type, (-np.inf, np.inf))
        if np.isfinite(density) and not (lo <= density <= hi):
            return [
                f"{fluid_type} density {density:.3f} g/cc outside typical range [{lo:.2f}, {hi:.2f}] g/cc"
            ]
        return []

    # ------------------------------------------------------- QC & classification
    def auto_classify_points(
        self,
        min_mobility: Optional[float] = None,
        ransac_residual_threshold: Optional[float] = None,
        assign_unassigned: bool = True,
        window: int = 2,
    ) -> pd.DataFrame:
        """Flag tight / supercharged points and suggest fluid phases for UNASSIGNED points.

        * Points with mobility below `min_mobility` are flagged TIGHT.
        * Unassigned VALID points get a phase from the density of a local (+/- `window` neighbour)
          gradient: < 0.475 g/cc GAS, < 0.98 g/cc OIL, otherwise WATER.
        * When `ransac_residual_threshold` is given (pressure units; <= 0 => automatic 2.5 x MAD),
          points per phase lying *above* the RANSAC trend by more than the threshold are flagged
          SUPERCHARGED.
        """
        df = self.df
        if min_mobility is not None and self.mobility_col in df.columns:
            mob = pd.to_numeric(df[self.mobility_col], errors="coerce")
            tight = (df[self.quality_col] == "VALID") & mob.notna() & (mob < float(min_mobility))
            df.loc[tight, self.quality_col] = "TIGHT"

        if assign_unassigned:
            cand = self.valid_mask() & (df[self.fluid_col] == "UNASSIGNED")
            valid_idx = df.index[self.valid_mask()]
            if cand.any() and len(valid_idx) >= 2:
                sub = df.loc[valid_idx].sort_values(self.tvdss_col)
                z_all = sub[self.tvdss_col].to_numpy(dtype=float)
                p_all = sub[self.pressure_col].to_numpy(dtype=float)
                order_idx = sub.index.to_list()
                for pos, idx in enumerate(order_idx):
                    if not cand.loc[idx]:
                        continue
                    lo, hi = max(0, pos - window), min(len(order_idx), pos + window + 1)
                    zz, pp = z_all[lo:hi], p_all[lo:hi]
                    if len(zz) < 2 or np.ptp(zz) < 1e-6:
                        continue
                    m, _, _, _ = _linear_fit_stats(zz, pp)
                    rho = self.to_density(m)
                    if not np.isfinite(rho) or rho <= 0 or rho > 1.6:
                        continue
                    if rho < _AUTO_GAS_OIL_DENSITY:
                        df.at[idx, self.fluid_col] = "GAS"
                    elif rho < _AUTO_OIL_WATER_DENSITY:
                        df.at[idx, self.fluid_col] = "OIL"
                    else:
                        df.at[idx, self.fluid_col] = "WATER"

        if ransac_residual_threshold is not None:
            thr = float(ransac_residual_threshold)
            for fl in FLUID_PHASES:
                m_fl = self.valid_mask(fl)
                if m_fl.sum() < 4:
                    continue
                idxs = df.index[m_fl]
                z = df.loc[idxs, self.tvdss_col].to_numpy(dtype=float)
                p = df.loc[idxs, self.pressure_col].to_numpy(dtype=float)
                inl = _ransac_inliers(z, p, threshold=thr if thr > 0 else None)
                if inl.sum() < 2:
                    continue
                m, c, _, _ = _linear_fit_stats(z[inl], p[inl])
                resid = p - (m * z + c)
                band = thr if thr > 0 else float(np.max(np.abs(resid[inl]))) + 1e-9
                sc = (~inl) & (resid > band)
                df.loc[idxs[sc], self.quality_col] = "SUPERCHARGED"
        return df

    # ------------------------------------------------------------ gradient fits
    def fit_gradient(
        self,
        fluid_type: FluidPhase,
        method: Literal["ols", "huber", "ransac"] = "ols",
        fixed_gradient: Optional[float] = None,
    ) -> FluidGradientResult:
        """Fit P = m * TVDSS + c for one phase using only VALID points of that phase.

        With `fixed_gradient`, only the intercept is estimated (works with a single point).
        """
        fluid_type = _norm_fluid(fluid_type)
        if fluid_type == "UNASSIGNED":
            raise ValueError("fluid_type must be GAS, OIL or WATER")
        mask = self.valid_mask(fluid_type)
        z = self.df.loc[mask, self.tvdss_col].to_numpy(dtype=float)
        p = self.df.loc[mask, self.pressure_col].to_numpy(dtype=float)
        ids = self._ids(mask)
        n = len(z)
        if n == 0:
            raise ValueError(f"No VALID {fluid_type} points to fit")

        if fixed_gradient is not None:
            m = float(fixed_gradient)
            cs = p - m * z
            c = float(np.mean(cs))
            var_c = float(np.var(cs, ddof=1) / n) if n > 1 else 0.0
            cov = np.array([[0.0, 0.0], [0.0, var_c]])
            r2 = None
            used_method = "fixed_gradient"
        else:
            if n < 2 or np.ptp(z) < 1e-6:
                raise ValueError(
                    f"{fluid_type} needs >= 2 VALID points at distinct depths (or a fixed gradient)"
                )
            used_method = method
            if method == "huber" and n >= 3:
                m, c, cov, r2 = _huber_fit(z, p)
            elif method == "ransac" and n >= 4:
                inl = _ransac_inliers(z, p)
                m, c, cov, r2 = _linear_fit_stats(z[inl], p[inl])
                ids = [i for i, keep in zip(ids, inl) if keep]
                n = int(inl.sum())
            else:
                m, c, cov, r2 = _linear_fit_stats(z, p)
                used_method = "ols" if method == "ols" else f"ols ({method} needs more points)"

        density = self.to_density(m)
        res = FluidGradientResult(
            fluid_type=fluid_type,
            gradient=m,
            intercept=c,
            density_gcc=density,
            r_squared=r2,
            std_err_gradient=float(np.sqrt(max(cov[0, 0], 0.0))),
            std_err_intercept=float(np.sqrt(max(cov[1, 1], 0.0))),
            cov_matrix=np.asarray(cov, dtype=float),
            n_points=int(n),
            tvdss_range=(float(np.min(z)), float(np.max(z))),
            method=used_method,
            pressure_unit=self.pressure_unit,
            depth_unit=self.depth_unit,
            point_ids=ids,
            warnings=self._density_warnings(fluid_type, density),
        )
        self.gradients[fluid_type] = res
        return res

    def set_manual_gradient(
        self,
        fluid_type: FluidPhase,
        gradient: float,
        anchor_tvdss: float,
        anchor_pressure: float,
        std_err_gradient: float = 0.0,
        std_err_intercept: float = 0.0,
    ) -> FluidGradientResult:
        """Define a synthetic / regional fluid line from one anchor point and a known gradient."""
        fluid_type = _norm_fluid(fluid_type)
        if fluid_type == "UNASSIGNED":
            raise ValueError("fluid_type must be GAS, OIL or WATER")
        m = float(gradient)
        z0, p0 = float(anchor_tvdss), float(anchor_pressure)
        c = p0 - m * z0
        var_m = float(std_err_gradient) ** 2
        var_c = var_m * z0**2 + float(std_err_intercept) ** 2
        cov = np.array([[var_m, -z0 * var_m], [-z0 * var_m, var_c]])
        density = self.to_density(m)
        res = FluidGradientResult(
            fluid_type=fluid_type,
            gradient=m,
            intercept=c,
            density_gcc=density,
            r_squared=None,
            std_err_gradient=float(std_err_gradient),
            std_err_intercept=float(np.sqrt(var_c)),
            cov_matrix=cov,
            n_points=0,
            tvdss_range=(z0, z0),
            method="manual",
            pressure_unit=self.pressure_unit,
            depth_unit=self.depth_unit,
            warnings=self._density_warnings(fluid_type, density),
        )
        self.gradients[fluid_type] = res
        return res

    def fit_all(
        self,
        method: Literal["ols", "huber", "ransac"] = "ols",
        overrides: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> Dict[str, FluidGradientResult]:
        """Fit every phase that has data, applying per-phase overrides.

        `overrides[phase]` may contain `gradient` (fixed slope) and optionally `anchor_tvdss` +
        `anchor_pressure` (manual line) and `std_err_gradient`. A phase with an anchored override
        does not need any measured points.
        """
        overrides = {(_norm_fluid(k)): v for k, v in (overrides or {}).items() if v}
        self.gradients = {}
        for fl in FLUID_PHASES:
            ov = overrides.get(fl) or {}
            grad = ov.get("gradient")
            try:
                if grad is not None and ov.get("anchor_tvdss") is not None and ov.get(
                    "anchor_pressure"
                ) is not None:
                    self.set_manual_gradient(
                        fl,
                        float(grad),
                        float(ov["anchor_tvdss"]),
                        float(ov["anchor_pressure"]),
                        std_err_gradient=float(ov.get("std_err_gradient") or 0.0),
                    )
                elif self.valid_mask(fl).any():
                    self.fit_gradient(
                        fl,  # type: ignore[arg-type]
                        method=method,
                        fixed_gradient=float(grad) if grad is not None else None,
                    )
            except ValueError as exc:
                self.warnings.append(str(exc))
        return self.gradients

    # ------------------------------------------------------------------ contacts
    def solve_contact(
        self, upper: FluidGradientResult, lower: FluidGradientResult
    ) -> Tuple[float, float]:
        """Deterministic intersection (z_contact, P_contact). Raises InvalidFluidGradientError."""
        dm = lower.gradient - upper.gradient
        if abs(dm) < 1e-5 * max(1.0, abs(lower.gradient)):
            raise InvalidFluidGradientError(
                f"{upper.fluid_type} and {lower.fluid_type} gradients are parallel; no contact"
            )
        if dm <= 0:
            raise InvalidFluidGradientError(
                f"Inverted densities: {lower.fluid_type} ({lower.density_gcc:.3f} g/cc) is not denser "
                f"than {upper.fluid_type} ({upper.density_gcc:.3f} g/cc)"
            )
        z = (upper.intercept - lower.intercept) / dm
        return float(z), float(upper.gradient * z + upper.intercept)

    def estimate_contacts(
        self, n_simulations: int = 2000, random_state: int = 42
    ) -> Dict[str, FluidContactResult]:
        """GOC (gas-oil), OWC / FWL (oil-water) or GWC (gas-water; only without oil) with
        P10 / P50 / P90 depth uncertainty from Monte Carlo sampling of the fit covariances."""
        self.contacts = {}
        rng = np.random.default_rng(random_state)
        have = set(self.gradients.keys())
        for label, up_fl, lo_fl in CONTACT_PAIRS:
            if up_fl not in have or lo_fl not in have:
                continue
            if label == "GWC" and "OIL" in have:
                continue
            up, lo = self.gradients[up_fl], self.gradients[lo_fl]
            try:
                z_det, p_det = self.solve_contact(up, lo)
            except InvalidFluidGradientError as exc:
                self.warnings.append(f"{label}: {exc}")
                continue

            # Delta-method standard deviation of z = (c1 - c2) / (m2 - m1)
            dm = lo.gradient - up.gradient
            grad_vec = np.array(
                [z_det / dm, 1.0 / dm, -z_det / dm, -1.0 / dm]
            )  # d z / d(m1, c1, m2, c2)
            big_cov = np.zeros((4, 4))
            big_cov[:2, :2] = up.cov_matrix
            big_cov[2:, 2:] = lo.cov_matrix
            std_z = float(np.sqrt(max(grad_vec @ big_cov @ grad_vec, 0.0)))

            p10 = p50 = p90 = z_det
            n_ok = 0
            if n_simulations > 0 and (np.any(up.cov_matrix) or np.any(lo.cov_matrix)):
                th1 = rng.multivariate_normal(
                    [up.gradient, up.intercept], up.cov_matrix, size=int(n_simulations),
                    check_valid="ignore",
                )
                th2 = rng.multivariate_normal(
                    [lo.gradient, lo.intercept], lo.cov_matrix, size=int(n_simulations),
                    check_valid="ignore",
                )
                dms = th2[:, 0] - th1[:, 0]
                ok = dms > 1e-9
                zs = (th1[ok, 1] - th2[ok, 1]) / dms[ok]
                zs = zs[np.isfinite(zs)]
                n_ok = int(len(zs))
                if n_ok >= 10:
                    p10, p50, p90 = (float(v) for v in np.percentile(zs, [10, 50, 90]))
            warnings: List[str] = []
            if n_simulations > 0 and 0 < n_ok < 0.9 * n_simulations:
                warnings.append(
                    f"{n_simulations - n_ok} of {n_simulations} simulations had non-intersecting / inverted gradients"
                )
            self.contacts[label] = FluidContactResult(
                contact_type=label,
                tvdss_p50=p50,
                tvdss_p10=p10,
                tvdss_p90=p90,
                pressure_at_contact=p_det,
                upper_fluid=up,
                lower_fluid=lo,
                tvdss_deterministic=z_det,
                std_tvdss=std_z,
                n_valid_simulations=n_ok,
                warnings=warnings,
            )

        if "GOC" in self.contacts and "OWC" in self.contacts:
            if self.contacts["GOC"].tvdss_p50 > self.contacts["OWC"].tvdss_p50:
                self.warnings.append(
                    "GOC is deeper than OWC: check fluid assignments / compartments"
                )
        return self.contacts

    # ---------------------------------------------------------- excess pressure
    def reference_line(
        self,
        ref_water_gradient: Optional[float] = None,
        datum_tvdss: Optional[float] = None,
        datum_pressure: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """Resolve (m_ref, z_datum, P_datum) for the hydrostatic reference used for excess pressure.

        Defaults to the fitted WATER line (else 1.0 g/cc) anchored at the shallowest valid point.
        """
        water = self.gradients.get("WATER")
        m_ref = (
            float(ref_water_gradient)
            if ref_water_gradient is not None
            else (water.gradient if water is not None else self.from_density(1.0))
        )
        if datum_tvdss is not None and datum_pressure is not None:
            return m_ref, float(datum_tvdss), float(datum_pressure)
        if water is not None:
            z0 = float(datum_tvdss) if datum_tvdss is not None else float(np.mean(water.tvdss_range))
            return m_ref, z0, float(water.pressure_at(z0))
        fin = self.valid_mask()
        if not fin.any():
            fin = self._finite_mask()
        if not fin.any():
            return m_ref, 0.0, 0.0
        sub = self.df.loc[fin].sort_values(self.tvdss_col)
        return (
            m_ref,
            float(sub[self.tvdss_col].iloc[0]),
            float(sub[self.pressure_col].iloc[0]),
        )

    def calculate_excess_pressure(
        self,
        ref_water_gradient: Optional[float] = None,
        datum_tvdss: Optional[float] = None,
        datum_pressure: Optional[float] = None,
    ) -> pd.Series:
        """P_excess(z) = P_measured(z) - [P_datum + m_ref * (z - z_datum)]."""
        m_ref, z0, p0 = self.reference_line(ref_water_gradient, datum_tvdss, datum_pressure)
        z = self.df[self.tvdss_col]
        return self.df[self.pressure_col] - (p0 + m_ref * (z - z0))

    def trend_lines(
        self, z_min: Optional[float] = None, z_max: Optional[float] = None
    ) -> Dict[str, List[Tuple[float, float]]]:
        """Pre-sampled (pressure, tvdss) end-points of each gradient line over [z_min, z_max]."""
        fin = self._finite_mask()
        zs = self.df.loc[fin, self.tvdss_col]
        lo = float(z_min) if z_min is not None else (float(zs.min()) if len(zs) else 0.0)
        hi = float(z_max) if z_max is not None else (float(zs.max()) if len(zs) else 1.0)
        out: Dict[str, List[Tuple[float, float]]] = {}
        for fl, g in self.gradients.items():
            out[fl] = [(float(g.pressure_at(lo)), lo), (float(g.pressure_at(hi)), hi)]
        return out
