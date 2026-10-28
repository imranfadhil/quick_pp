import hashlib
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from tqdm import tqdm

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.rock_type import estimate_pore_throat
from quick_pp.utils import (
    inv_power_law_func,
    power_law_func,
    r2_score,
    root_mean_squared_error,
)

GEO_ABBREVIATIONS = Config.CORE_GEO_ABBREVIATIONS
WORD_CATEGORIES = Config.CORE_WORD_CATEGORIES
SPECIAL_CASE_DESCRIPTIONS = Config.CORE_SPECIAL_CASE_DESCRIPTIONS


plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {"axes.labelsize": 10, "xtick.labelsize": 10, "legend.fontsize": "small"}
)


# Cross plots
def poroperm_xplot(
    poro, perm, a=None, b=None, core_group=None, label="", log_log=False
):
    """Generate porosity-permeability cross plot.

    Args:
        poro (float): Core porosity (frac).
        perm (float): Core permeability (mD).
        a (float, optional): a constant in perm=a*poro^b. Defaults to None.
        b (float, optional): b constant in perm=a*poro^b. Defaults to None.
        core_group (array-like, optional): Grouping for core samples to be used for coloring. Defaults to None.
        label (str, optional): Label for the data group. Defaults to ''.
        log_log (bool, optional): Whether to plot log-log or not. Defaults to False.
    """
    sc = plt.scatter(poro, perm, marker=".", c=core_group, cmap="Set1")
    if core_group is not None:
        for _, row in enumerate(zip(poro, perm, core_group, strict=False)):
            plt.annotate(row[2], (row[0], row[1]), fontsize=8, alpha=0.7)
    if a and b:
        line_color = sc.get_facecolors()[0]
        line_color[-1] = 0.5
        cpore = np.geomspace(0.01, 0.5, 30)
        plt.plot(
            cpore,
            power_law_func(cpore, a, b),
            color=line_color,
            label=label,
            linestyle="dashed",
        )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel("CPORE (frac)")
    plt.xlim(0.001, 0.5)
    plt.ylabel("CPERM (mD)")
    plt.ylim(0.001, 1e5)
    plt.yscale("log")
    if log_log:
        plt.xscale("log")


def bvw_xplot(bvw, pc, a=None, b=None, label=None, ylim=None, log_log=False):
    """Generate bulk volume water-capillary pressure cross plot.

    Args:
        bvw (float): Calculated bulk volume water (frac) from core.
        pc (float): Capillary pressure (psi) from core.
        a (float, optional): a constant in pc=a*bvw^b. Defaults to None.
        b (float, optional): b constant in pc=a*bvw^b. Defaults to None.
        label (str, optional): Label for the data group. Defaults to ''.
        ylim (tuple, optional): Range for the y axis in (min, max) format. Defaults to None.
        log_log (bool, optional): Whether to plot log-log or not. Defaults to False.
    """
    sc = plt.plot(bvw, pc, marker="s", label=label)
    if a is not None and b is not None:
        line_color = sc[0].get_color() + "66"  # Set opacity to 0
        cbvw = np.linspace(0.05, 0.35, 30)
        plt.scatter(cbvw, inv_power_law_func(cbvw, a, b), marker="x", color=line_color)
    plt.xlabel("BVW (frac)")
    plt.ylabel("Pc (psi)")
    plt.ylim(ylim) if ylim else plt.ylim(
        0.01, plt.gca().get_lines()[-1].get_ydata().max()
    )
    if label:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if log_log:
        plt.xscale("log")
        plt.yscale("log")


def pc_xplot(sw, pc, label=None, ylim=None):
    """Generate Pc-Sw cross plot.

    Args:
        sw (float): Core water saturation (frac).
        pc (float): Capillary pressure (psi) from core.
        label (str, optional): Label for the data group. Defaults to ''.
        ylim (tuple, optional): Range for the y axis in (min, max) format. Defaults to None.
    """
    plt.plot(sw, pc, marker=".", label=label)
    plt.xlabel("Sw (frac)")
    plt.xlim(0.01, 1)
    plt.ylabel("Pc (psi)")
    plt.ylim(ylim) if ylim else plt.ylim(
        0.01, plt.gca().get_lines()[-1].get_ydata().max()
    )
    if label:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")


def pc_xplot_plotly(sw, pc, label=None, ylim=None, fig=go.Figure()):
    """Generate J-Sw cross plot using Plotly.

    Args:
        sw (float): Core water saturation (frac).
        pc (float): Capillary pressure (psi).
        label (str, optional): Label for the data group. Defaults to ''.
        ylim (tuple, optional): Range for the y axis in (min, max) format. Defaults to None.
    """
    fig.add_trace(go.Scatter(x=sw, y=pc, name=label))
    fig.update_layout(
        xaxis_title="Sw (frac)",
        yaxis_title="Pc (psi)",
        xaxis_range=[0, 1],
        yaxis_range=[0, 50] if ylim is None else [ylim[0], ylim[1]],
        legend={"x": 1.04, "y": 1, "traceorder": "normal"},
        height=500,
        width=800,
    )
    return fig


def j_xplot(
    sw,
    j,
    a=None,
    b=None,
    core_group=None,
    label=None,
    log_log=False,
    ax=None,
    ylim=None,
):
    """Generate J-Sw cross plot.

    Args:
        sw (float): Core water saturation (frac).
        j (float): Calculated J value (unitless).
        a (float, optional): a constant in j=a*sw^b. Defaults to None.
        b (float, optional): b constant in j=a*sw^b. Defaults to None.
        core_group (array-like, optional): Grouping for core samples to be used for coloring. Defaults to None.
        label (str, optional): Label for the data group. Defaults to ''.
        ylim (tuple, optional): Range for the y axis in (min, max) format. Defaults to None.
        log_log (bool, optional): Whether to plot log-log or not. Defaults to False.
    """
    ax = ax or plt.gca()

    if core_group is not None:
        # Check if core_group contains non-numeric data
        if any(isinstance(item, str) for item in core_group):
            color_data, uniques = pd.factorize(core_group)
            scatter = ax.scatter(sw, j, marker=".", c=color_data, cmap="Set1")
            legend1 = ax.legend(
                handles=scatter.legend_elements()[0],
                labels=uniques.tolist(),
                title="Core Sample",
                bbox_to_anchor=(1.04, 0.9),
                loc="upper left",
            )
            ax.add_artist(legend1)
        else:
            scatter = ax.scatter(sw, j, marker=".", c=core_group, cmap="Set1")
            legend1 = ax.legend(
                *scatter.legend_elements(),
                title="Core Sample",
                bbox_to_anchor=(1.04, 0.9),
                loc="upper left",
            )
            ax.add_artist(legend1)
    else:
        ax.scatter(sw, j, marker=".")

    if a is not None and b is not None:
        csw = np.geomspace(0.01, 1.0, 20)
        ax.plot(csw, inv_power_law_func(csw, a, b), label=label, linestyle="dashed")
    ax.set_xlabel("Sw (frac)")
    ax.set_xlim(0.01, 1)
    ax.set_ylabel("J")
    ax.set_ylim(ylim) if ylim else ax.set_ylim(
        0.01, max(ax.get_lines()[-1].get_ydata())
    )
    if log_log:
        ax.set_yscale("log")
        ax.set_ylim(0.01, 100)
        ax.set_xscale("log")
    if label:
        ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return ax


# Best-fit functions
def _robust_fit_j_logspace(sw, j, dampen_outliers=True):
    """Robust log-space IRLS fit for Leverett-J curve J = a * Swn^(-b) with outlier and asymptote dampening.

    Returns:
        tuple: (a, b, weights, sw_clean, j_clean)
    """
    sw_arr = np.asarray(sw, dtype=float)
    j_arr = np.asarray(j, dtype=float)
    mask = np.isfinite(sw_arr) & np.isfinite(j_arr) & (sw_arr > 0) & (j_arr > 0)
    sw_arr = sw_arr[mask]
    j_arr = j_arr[mask]
    if len(sw_arr) < 2:
        return 0.2, 0.3, np.ones(len(sw_arr), dtype=float), sw_arr, j_arr

    x = -np.log(np.clip(sw_arr, 1e-6, 1.0))
    y = np.log(np.clip(j_arr, 1e-8, 1e8))

    if float(np.std(x)) < 1e-8:
        a_const = float(np.clip(np.exp(np.median(y)), 1e-4, 50.0))
        return round(a_const, 4), 0.5, np.ones(len(sw_arr), dtype=float), sw_arr, j_arr

    # Fast path for exact or near-noiseless synthetic power-law curves
    slope_ols, intercept_ols = np.polyfit(x, y, 1)
    resid_ols = y - (intercept_ols + slope_ols * x)
    if not dampen_outliers or len(sw_arr) <= 5 and float(np.max(np.abs(resid_ols))) < 0.05:
        a_ols = float(np.clip(np.exp(intercept_ols), 1e-4, 50.0))
        b_ols = float(np.clip(slope_ols, 0.02, 5.0))
        return round(a_ols, 4), round(b_ols, 4), np.ones(len(sw_arr), dtype=float), sw_arr, j_arr

    # Dampen capillary asymptote pile-ups (Swn <= 0.055 at Swirr and Swn >= 0.90 below entry pressure)
    interior_mask = (sw_arr > 0.055) & (sw_arr < 0.90)
    if int(np.sum(interior_mask)) >= 3 and float(np.std(x[interior_mask])) > 0.08:
        w_prior = np.where(
            (sw_arr >= 0.06) & (sw_arr <= 0.88),
            1.0,
            np.where((sw_arr <= 0.04) | (sw_arr >= 0.93), 0.05, 0.25),
        )
    else:
        interior_mask = np.ones(len(sw_arr), dtype=bool)
        w_prior = np.ones(len(sw_arr), dtype=float)

    x_int = x[interior_mask]
    y_int = y[interior_mask]

    # Pairwise Theil-Sen median slope across interior points for outlier-resistant initialization
    pair_slopes = []
    n_int = len(x_int)
    step = max(1, n_int // 40)
    for i in range(0, n_int, step):
        dx = x_int[i + 1 :: step] - x_int[i]
        dy = y_int[i + 1 :: step] - y_int[i]
        valid_dx = np.abs(dx) > 0.12
        if np.any(valid_dx):
            s_vals = dy[valid_dx] / dx[valid_dx]
            pos_s = s_vals[(s_vals > 0.02) & (s_vals < 4.0)]
            if len(pos_s) > 0:
                pair_slopes.extend(pos_s.tolist())

    if pair_slopes:
        b = float(np.clip(np.median(pair_slopes), 0.08, 2.8))
    else:
        b_init, _ = np.polyfit(x_int, y_int, 1)
        b = float(np.clip(b_init, 0.08, 2.8))
    alpha = float(np.median(y_int - b * x_int))

    # Iteratively Reweighted Least Squares (IRLS) with Tukey biweight + Huber tail dampening
    weights = w_prior.copy()
    for _ in range(8):
        resid = y - (alpha + b * x)
        r_ref = resid[interior_mask]
        med_r = float(np.median(r_ref))
        mad = float(np.median(np.abs(r_ref - med_r)))
        sigma = max(1.4826 * mad, 0.15)
        u = np.abs(resid - med_r) / (2.0 * sigma)
        w_out = np.where(u < 1.0, (1.0 - u**2) ** 2, 0.02 / np.maximum(u**2, 1.0))
        weights = w_prior * w_out
        w_sum = float(np.sum(weights))
        if w_sum <= 1e-12:
            weights = w_prior.copy()
            break
        x_bar = float(np.sum(weights * x) / w_sum)
        y_bar = float(np.sum(weights * y) / w_sum)
        denom = float(np.sum(weights * ((x - x_bar) ** 2)))
        if denom > 1e-8:
            b_new = float(np.sum(weights * (x - x_bar) * (y - y_bar)) / denom)
            b = float(np.clip(b_new, 0.05, 3.0))
        alpha = float(y_bar - b * x_bar)

    a_val = float(np.clip(np.exp(alpha), 1e-4, 50.0))
    b_val = float(np.clip(b, 0.05, 3.0))
    return round(a_val, 4), round(b_val, 4), weights, sw_arr, j_arr


def fit_j_curve(sw, j, dampen_outliers=True):
    """Estimate a and b constants of best fit given core water saturation and J value.

    Args:
        sw (float or array-like): Core water saturation (frac).
        j (float or array-like): Calculated J value (unitless).
        dampen_outliers (bool, optional): Whether to apply robust log-space IRLS outlier
            and capillary-asymptote dampening. Defaults to True.

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        a, b, _, _, _ = _robust_fit_j_logspace(sw, j, dampen_outliers=dampen_outliers)
        return a, b
    except Exception as e:
        logger.error(e)
        return 1, 1


def skelt_harrison_xplot(
    sw, pc, gw, ghc, a, b, c, d, core_group=None, label=None, ylim=None, ax=None
):
    """Generate Skelt-Harrison curve.

    Args:
        pc (float): Capillary pressure (psi).
        sw (float): Core water saturation (frac).
        h (float): Height above free water level (ft).
        a (float): a constant from the best-fit curve. Related to Swirr.
        b (float): b constant from the best-fit curve. Related to HAFWL.
        c (float): c constant from the best-fit curve. Related to PTSD.
        d (float): d constant from the best-fit curve. Related to entry pressure.
    """
    ax = ax or plt.gca()
    ax.scatter(sw, pc, marker=".", c=core_group, cmap="Set1")
    h = np.geomspace(0.01, 10000, 100)
    pci = h * (gw - ghc) * 0.433  # Convert g/cc to psi/ft
    ax.plot(skelt_harrison_func(h, a, b, c, d), pci, label=label)
    ax.set_ylabel("Pc (psi)")
    ax.set_ylim(ylim) if ylim else ax.set_ylim(
        0.01, ax.get_lines()[-1].get_ydata().max()
    )
    ax.set_xlabel("Sw (frac)")
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return ax


def skelt_harrison_func(h, a, b, c, d):
    """Calculate water saturation using the Skelt-Harrison model.

    This function models the relationship between water saturation and the height
    above the free water level using an exponential decay function.

    Args:
        h (float or np.array): Height above free water level (ft).
        a (float): A constant related to irreducible water saturation (Swirr).
                   It influences the maximum water saturation.
        b (float): A constant related to the height above free water level (HAFWL).
                   It controls the rate of saturation change with height.
        c (float): A constant related to the pore throat size distribution (PTSD).
                   It affects the shape of the saturation curve.
        d (float): A constant related to the entry pressure. It shifts the curve
                   along the height axis.

    Returns:
        float or np.array: The calculated water saturation (fraction).
    """
    h_arr = np.maximum(np.asarray(h, dtype=float) + d, 1e-6)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = 1 - a * np.exp(-((np.maximum(b, 1e-6) / h_arr) ** c))
    return float(res) if np.isscalar(h) else res


def fit_skelt_harrison_curve(sw, h):
    """Estimate a and b constants of best fit given core water saturation and capillary pressure values.

    Args:
        sw (float): Core water saturation (frac).
        h (float): Height above free water level (ft).

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        sw_arr = np.asarray(sw, dtype=float)
        h_arr = np.asarray(h, dtype=float)
        mask = (
            np.isfinite(sw_arr)
            & np.isfinite(h_arr)
            & (h_arr >= 0)
            & (sw_arr >= 0)
            & (sw_arr <= 1.0)
        )
        sw_arr = sw_arr[mask]
        h_arr = h_arr[mask]
        if len(sw_arr) < 4:
            a_est = round(
                float(np.clip(1.0 - np.nanmin(sw_arr) if len(sw_arr) else 0.85, 0.1, 0.98)),
                3,
            )
            b_est = round(
                float(np.clip(np.nanmedian(h_arr) if len(h_arr) else 25.0, 1.0, 500.0)),
                3,
            )
            return a_est, b_est, 1.5, 1.0
        a0 = float(np.clip(1.0 - np.min(sw_arr), 0.2, 0.98))
        pos_h = h_arr[h_arr > 0]
        b0 = float(np.clip(np.median(pos_h) if len(pos_h) else 25.0, 0.5, 2000.0))
        popt, _ = curve_fit(
            skelt_harrison_func,
            h_arr,
            sw_arr,
            p0=[a0, b0, 1.5, 1.5],
            bounds=([0.05, 0.01, 0.1, 0.0], [1.0, 10000.0, 8.0, 500.0]),
            maxfev=8000,
        )
        a = [round(c, 3) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        c = [round(c, 3) for c in popt][2]
        d = [round(c, 3) for c in popt][3]
        return a, b, c, d
    except Exception as e:
        logger.error(e)
        return 1, 1, 1, 1


def cuddy_func(pc, a, b):
    """Calculate Bulk Volume Water (BVW = PHIT * Sw) using Cuddy's power-law function.

    Args:
        pc (float or np.ndarray): Capillary pressure (psi) or height above FWL.
        a (float): Cuddy multiplier constant.
        b (float): Cuddy exponent constant (evaluated as a * pc ** (-|b|)).

    Returns:
        float or np.ndarray: Bulk volume water (BVW, fraction).
    """
    pc_arr = np.maximum(np.asarray(pc, dtype=float), 1e-6)
    b_abs = np.abs(b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        res = a * (pc_arr ** (-b_abs))
    return float(res) if np.isscalar(pc) else res


def fit_cuddy_curve(bvw, pc):
    """Estimate Cuddy constants (a, b) from core Bulk Volume Water (BVW = PHIT * Sw) and capillary pressure.

    Fits BVW = a * Pc^(-b) where b > 0.

    Args:
        bvw (float or array-like): Bulk volume water (PHIT * Sw, frac).
        pc (float or array-like): Capillary pressure (psi).

    Returns:
        tuple: (a, b) constants from the best-fit curve.
    """
    try:
        bvw_arr = np.asarray(bvw, dtype=float)
        pc_arr = np.asarray(pc, dtype=float)
        mask = np.isfinite(bvw_arr) & np.isfinite(pc_arr) & (bvw_arr > 0) & (pc_arr > 0)
        bvw_arr = bvw_arr[mask]
        pc_arr = pc_arr[mask]
        if len(bvw_arr) < 2:
            return 0.12, 0.35
        slope, intercept = np.polyfit(np.log(pc_arr), np.log(bvw_arr), 1)
        a0 = float(np.clip(np.exp(intercept), 1e-4, 2.0))
        b0 = float(np.clip(-slope, 0.02, 3.0))
        popt, _ = curve_fit(
            cuddy_func,
            pc_arr,
            bvw_arr,
            p0=[a0, b0],
            bounds=([1e-5, 0.01], [5.0, 5.0]),
            maxfev=5000,
        )
        return round(float(popt[0]), 4), round(float(popt[1]), 4)
    except Exception as e:
        logger.error(e)
        return 0.12, 0.35


def perm_transform(poro, a, b):
    """Transform porosity to permeability using a and b constants.

    Args:
        poro (float): Core porosity (frac).
        a (float): a constant from the best-fit curve.
        b (float): b constant from the best-fit curve.

    Returns:
        float: Permeability (mD).
    """
    return a * poro**b


def fit_poroperm_curve(poro, perm):
    """Estimate a and b constants of best fit given core porosity and permeability values.

    Args:
        poro (float): Core porosity (frac).
        perm (float): Core permeability (mD).

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        # Filter out non-positive values which are invalid for logarithmic fitting
        valid_data = (poro > 0) & (perm > 0)
        if not np.any(valid_data):
            logger.warning("No valid data points for poroperm curve fitting.")
            return 1, 1

        popt, _ = curve_fit(
            power_law_func, poro[valid_data], perm[valid_data], nan_policy="omit"
        )
        a = [round(c) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except RuntimeError as e:
        logger.error(f"Curve fit failed for poroperm: {e}")
        return 1, 1
    except Exception as e:
        logger.error(e)
        return 1, 1


from quick_pp.scal import (  # noqa: E402
    brooks_corey_relperm,
    corey_relperm,
    correct_core_data,
    dean_stark_cbw_correction,
    dean_stark_saturation_correction,
    estimate_cbw_from_qv,
    fit_relperm,
    generate_swof_table,
    humidity_to_total_porosity,
    klinkenberg_permeability_correction,
    let_relperm,
    normalize_water_saturation,
    overburden_permeability_correction,
    overburden_poroperm_correction,
    overburden_porosity_correction,
    pc_lab_to_reservoir,
)


def leverett_j(pc, ift, theta, perm, phit):
    """Estimate Leverett J.

    Args:
        pc (float or np.ndarray): Capillary pressure (psi).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        perm (float or np.ndarray): Permeability (mD).
        phit (float or np.ndarray): Total porosity (frac).

    Returns:
        float or np.ndarray: Leverett J value.
    """
    return (
        0.21665
        * pc
        / (ift * abs(np.cos(np.radians(theta))))
        * (np.maximum(perm, 0.0) / np.maximum(phit, 1e-6)) ** (0.5)
    )


def sw_skelt_harrison(depth, fwl, a, b, c, d, clip=True):
    """Estimate water saturation based on Skelt-Harrison.

    Args:
        depth (float or np.ndarray): True vertical depth.
        fwl (float): Free water level in true vertical depth.
        a (float): a constant from the best-fit curve.
        b (float): b constant from the best-fit curve.
        c (float): c constant from the best-fit curve.
        d (float): d constant from the best-fit curve.
        clip (bool, optional): Clip resulting Sw to [0, 1] and enforce Sw=1 at/below FWL.
            Defaults to True.

    Returns:
        float or np.ndarray: Water saturation.
    """
    is_scalar = np.isscalar(depth) and np.isscalar(fwl)
    h = np.maximum(fwl - depth, 0.0)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        sw = skelt_harrison_func(h, a, b, c, d)
    if clip:
        sw = np.where(fwl - depth <= 0, 1.0, np.clip(sw, 0.0, 1.0))
    return float(sw) if is_scalar else sw


def sw_lambda(phit, h, a, b, lamda):
    """Estimate water saturation based on Lambda model (Thomeer-type Hyperbolic).

    Args:
        phit (float): Total porosity in fraction.
        h (float): Height above free water level.
        a (float): A constant from the best-fit curve.
        b (float): B constant from the best-fit curve.
        lamda (float): Lambda constant from the best-fit curve, related to pore size distribution.

    Returns:
        float: Water saturation.
    """
    return (a / (h * phit**b)) ** (1 / lamda)


def sw_cuddy(phit, h, a, b):
    """Estimate water saturation based on Cuddy's.

    Args:
        phit (float): Total porosity (frac).
        h (float): Height above free water level.
        a (float): Fitting parameter.
        b (float): Fitting parameter (negative value).

    Returns:
        float: Water saturation.
    """
    return a / (h**b * phit)


def sw_shf_leverett_j(
    perm,
    phit,
    depth,
    fwl,
    ift,
    theta,
    gw,
    ghc,
    a,
    b,
    phie=None,
    swirr=0.0,
    unit="imperial",
    clip=True,
):
    """Estimate water saturation based on Leverett J function.

    Args:
        perm (float or np.ndarray): Permeability (mD).
        phit (float or np.ndarray): Total porosity (frac).
        depth (float or np.ndarray): True vertical depth (ft or m).
        fwl (float): Free water level in true vertical depth (ft or m).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        gw (float): Water density (g/cc).
        ghc (float): Hydrocarbon density (g/cc).
        a (float or np.ndarray): A constant from J function ($J = a \\cdot S_{wn}^{-b}$).
        b (float or np.ndarray): B constant from J function.
        phie (float or np.ndarray, optional): Effective porosity (frac), required for clay bound
            water conversion to total water saturation. Defaults to None.
        swirr (float or np.ndarray, optional): Irreducible water saturation fraction to un-normalize
            $S_{wn}$ ($S_w = S_{wirr} + (1 - S_{wirr}) S_{wn}$). Defaults to 0.0.
        unit (str, optional): Depth unit system, `'imperial'`/`'ft'` (0.433 psi/ft per g/cc) or
            `'metric'`/`'m'` (1.4206 psi/m per g/cc). Defaults to `'imperial'`.
        clip (bool, optional): Clip resulting saturation to [0.0, 1.0] and set 1.0 at/below FWL.
            Defaults to True.

    Returns:
        float or np.ndarray: Water saturation from saturation height function.
    """
    is_scalar = (
        np.isscalar(perm)
        and np.isscalar(phit)
        and np.isscalar(depth)
        and np.isscalar(fwl)
        and np.isscalar(a)
        and np.isscalar(b)
    )
    conv = 0.433 * 3.28084 if str(unit).strip().lower() in ("metric", "m") else 0.433
    h = np.maximum(fwl - depth, 0.0)
    pc = h * (gw - ghc) * conv
    j_val = leverett_j(pc, ift, theta, perm, phit)
    b_abs = np.abs(b)

    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shf = np.where(
            (h > 0) & (j_val > 0) & (a > 0) & (b_abs > 0),
            (a / j_val) ** (1.0 / b_abs),
            1.0,
        )

    if swirr is not None:
        shf = swirr + (1.0 - swirr) * shf

    if phie is not None:
        swb = np.clip(1.0 - (phie / np.maximum(phit, 1e-6)), 0.0, 1.0)
        shf = shf * (1.0 - swb) + swb

    if clip:
        shf = np.where(fwl - depth <= 0, 1.0, np.clip(shf, 0.0, 1.0))

    if is_scalar:
        return float(shf)
    if isinstance(depth, pd.Series):
        return pd.Series(shf, index=depth.index)
    return shf


def sw_shf_cuddy(phit, depth, fwl, gw, ghc, a, b, unit="imperial", clip=True):
    """Estimate water saturation based on Cuddy's saturation height function.

    Args:
        phit (float or np.ndarray): Porosity (frac).
        depth (float or np.ndarray): True vertical depth (ft or m).
        fwl (float): Free water level in true vertical depth (ft or m).
        gw (float): Water density (g/cc).
        ghc (float): Hydrocarbon density (g/cc).
        a (float): Fitting parameter.
        b (float): Fitting parameter.
        unit (str, optional): `'imperial'`/`'ft'` or `'metric'`/`'m'`. Defaults to `'imperial'`.
        clip (bool, optional): Clip resulting saturation to [0.0, 1.0]. Defaults to True.

    Returns:
        float or np.ndarray: Water saturation from saturation height function.
    """
    is_scalar = np.isscalar(phit) and np.isscalar(depth) and np.isscalar(fwl)
    conv = 0.433 * 3.28084 if str(unit).strip().lower() in ("metric", "m") else 0.433
    h = np.maximum(fwl - depth, 0.0)
    b_abs = np.abs(b)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shf = np.where(
            (h > 0) & (phit > 0),
            a / (((h * (gw - ghc) * conv) ** b_abs) * phit),
            1.0,
        )
    if clip:
        shf = np.where(fwl - depth <= 0, 1.0, np.clip(shf, 0.0, 1.0))
    return float(shf) if is_scalar else shf


def sw_shf_choo(perm, phit, phie, depth, fwl, ift, theta, gw, ghc, b0=0.4):
    """Estimate water saturation based on Choo's saturation height function.

    Args:
        perm (float or np.ndarray): Permeability (mD).
        phit (float or np.ndarray): Total porosity (frac).
        phie (float or np.ndarray): Effective porosity (frac).
        depth (float or np.ndarray): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        gw (float): Water density (g/cc).
        ghc (float): Hydrocarbon density (g/cc).
        b0 (float): Exponent constant. Defaults to 0.4.

    Returns:
        float or np.ndarray: Water saturation from saturation height function.
    """
    swb = np.clip(1.0 - (phie / np.maximum(phit, 1e-6)), 1e-6, 1.0 - 1e-6)
    h = np.maximum(fwl - depth, 0.0)
    pc = h * (gw - ghc) * 0.433
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        shf = 10 ** ((2 * b0 - 1) * np.log10(1 + swb**-1) + np.log10(1 + swb)) / (
            (
                0.2166
                * (pc / (ift * abs(np.cos(np.radians(theta)))))
                * (perm / phit) ** (0.5)
            )
            ** (b0 * np.log10(1 + swb**-1) / 3)
        )
    return np.where(h <= 0, 1.0, np.clip(shf, 0.0, 1.0))


def autocorrelate_core_depth(
    df,
    replace_ori=False,
    net_confining_stress=None,
    mean_pore_pressure=None,
    cbw=None,
    compaction_coeff=4e-6,
    oven_dried=False,
):
    """
    Automatically shift core depths to match log depths.

    This function aligns core porosity (CPORE) with log porosity (PHIT) for each
    well in the DataFrame. It uses a hybrid approach combining Dynamic Time Warping
    (DTW) for alignment and machine learning models for correction.

    The process is as follows:
    1.  **Segmentation**: Core data is clustered by depth proximity using DBSCAN to
        identify continuous segments.
    2.  **Interpolation & Smoothing**: Within each segment, sparse core porosity is
        interpolated to match log data resolution. Both core and log porosity
        signals are then smoothed to focus on trends.
    3.  **Weighted DTW Alignment**: A weighted Dynamic Time Warping (DTW) is used
        to find the optimal alignment path between the two porosity signals. The
        weighting is achieved by adding a second feature channel that identifies
        peaks and troughs, guiding the alignment.
    4.  **Filtered Model Training**: A RandomForestRegressor model is trained on the
        DTW alignment path. The training data is filtered to include only points
        where the proposed shift is within a plausible window, preventing the
        model from learning from extreme outliers.
    5.  **Correction & Post-processing**: The model predicts the corrected depths.
        This shift is clipped to the window size. A spacing correction ensures
        that core samples are not artificially compressed.
    6.  **Finalization**: Isotonic regression is applied across all corrected depths
        for the well to enforce strict monotonicity.

    Args:
        df (pd.DataFrame): DataFrame containing well log and core data. Must include
                           'WELL_NAME', 'DEPTH', 'PHIT', 'CPORE', 'CPERM', and 'CORE_ID'.
        replace_ori (bool, optional): If True, the original core data columns
                                      (CPORE, CPERM) will be overwritten with
                                      the corrected ones. If False, the original
                                      columns will be preserved with an '_ORI' suffix.
                                      Defaults to False.
        net_confining_stress (float, str, or np.ndarray, optional): Net confining stress (psi)
                                      or column name in `df` for overburden porosity correction
                                      prior to depth alignment. Defaults to None.
        mean_pore_pressure (float, str, or np.ndarray, optional): Mean flowing pore pressure (psi)
                                      or column name in `df` for Klinkenberg permeability correction.
                                      Defaults to None.
        cbw (bool, float, str, or np.ndarray, optional): Clay-bound water volume (frac), column
                                      name, or True to infer `PHIT - PHIE` to reconcile humidity-dried
                                      core porosity before aligning with `PHIT`. Defaults to None.
        compaction_coeff (float, optional): Pore compaction coefficient (1/psi). Defaults to 4e-6.
        oven_dried (bool, optional): Whether core plugs are oven-dried. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame with the original data merged with the depth-corrected core data.
            - A summary DataFrame detailing the depth shifts for each core sample.
    """
    from dtw import dtw
    from scipy.interpolate import interp1d
    from scipy.signal import find_peaks
    from sklearn.cluster import DBSCAN
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.preprocessing import minmax_scale

    required_cols = ["WELL_NAME", "DEPTH", "PHIT", "CPORE", "CPERM", "CORE_ID"]
    for col in required_cols:
        if col not in df.columns:
            raise AssertionError(f"Missing required column: {col}")

    df = df.copy()
    if any(x is not None for x in (net_confining_stress, mean_pore_pressure, cbw)):
        df = correct_core_data(
            df,
            net_confining_stress=net_confining_stress,
            mean_pore_pressure=mean_pore_pressure,
            cbw=cbw,
            compaction_coeff=compaction_coeff,
            oven_dried=oven_dried,
            replace_ori=replace_ori,
        )
    # Ensure porosity and permeability columns are numeric, coercing errors.
    df["CPERM"] = pd.to_numeric(df["CPERM"], errors="coerce")
    df["PHIT"] = pd.to_numeric(df["PHIT"], errors="coerce")
    all_wells_data = []
    shift_summaries = []

    for well, data in tqdm(df.groupby("WELL_NAME"), desc="Correlating core depths"):
        core_data = (
            data[["DEPTH", "CORE_ID", "CPORE", "CPERM"]]
            .dropna()
            .sort_values("DEPTH")
            .reset_index(drop=True)
        )

        if core_data.empty:
            all_wells_data.append(data)
            continue

        log_data = (
            data[["DEPTH", "PHIT"]].dropna().sort_values("DEPTH").reset_index(drop=True)
        )
        tqdm.write(
            f"Processing {well}: {len(core_data)} core data and {len(log_data)} log data"
        )

        # Cluster core data into segments based on depth proximity.
        depth_clusters = DBSCAN(eps=0.01, min_samples=3).fit(
            minmax_scale(core_data["DEPTH"].values.reshape(-1, 1))
        )
        core_data = core_data.assign(cluster=depth_clusters.labels_)

        all_alignments = []
        final_corrected_depths = np.zeros(len(core_data))

        # Process each core data segment (cluster) independently.
        for cluster_id in np.unique(core_data["cluster"]):
            cluster_core_data = core_data[core_data["cluster"] == cluster_id]
            core_indices_in_cluster = cluster_core_data.index

            # Create a window of log data around the current core segment
            min_depth, max_depth = (
                cluster_core_data["DEPTH"].min(),
                cluster_core_data["DEPTH"].max(),
            )
            window = 5  # Window size in depth units for log data subset.
            log_subset = log_data[
                (log_data["DEPTH"] >= min_depth - window)
                & (log_data["DEPTH"] <= max_depth + window)
            ].reset_index(drop=True)

            if log_subset.empty:
                # If no log data is found in the window, keep original depths.
                final_corrected_depths[core_indices_in_cluster] = cluster_core_data[
                    "DEPTH"
                ].values
                continue

            # Smooth the log data to focus on trends rather than high-frequency noise.
            log_phit_smooth = (
                log_subset["PHIT"]
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )

            # Interpolate sparse core data to create a dense signal for DTW.
            core_depths = cluster_core_data["DEPTH"].values
            core_poro = cluster_core_data["CPORE"].values

            # Use cubic interpolation if possible for a smoother curve.
            interp_kind = "cubic" if len(core_depths) >= 4 else "linear"
            interp_func = interp1d(
                core_depths,
                core_poro,
                kind=interp_kind,
                bounds_error=False,
                fill_value="extrapolate",
            )

            dense_core_depths = np.linspace(
                core_depths.min(), core_depths.max(), num=len(log_subset)
            )
            dense_core_poro = interp_func(dense_core_depths)

            # Smooth the interpolated core data to match the smoothing of log data.
            dense_core_poro_smooth = (
                pd.Series(dense_core_poro)
                .rolling(window=window, center=True, min_periods=1)
                .mean()
            )

            # Scale both signals to a [0, 1] range for DTW.
            cpore_vals = minmax_scale(dense_core_poro_smooth)
            phit_vals = minmax_scale(log_phit_smooth.values)

            # Identify peaks and troughs in both signals to act as anchor points.
            # This creates a second feature channel to guide the DTW alignment.
            peaks, _ = find_peaks(phit_vals, prominence=0.1, distance=2)
            troughs, _ = find_peaks(-phit_vals, prominence=0.1, distance=2)
            anchor_indices = np.concatenate([peaks, troughs])

            phit_anchor_channel = np.zeros_like(phit_vals)
            if len(anchor_indices) > 0:
                phit_anchor_channel[anchor_indices] = 1.0

            # Repeat for the core porosity signal.
            peaks, _ = find_peaks(cpore_vals, prominence=0.1, distance=2)
            troughs, _ = find_peaks(-cpore_vals, prominence=0.1, distance=2)
            anchor_indices = np.concatenate([peaks, troughs])
            cpore_anchor_channel = np.zeros_like(cpore_vals)
            if len(anchor_indices) > 0:
                cpore_anchor_channel[anchor_indices] = 1.0

            # Stack signals and anchor channels for 2D DTW.
            weight_factor = 5.0
            cpore_2d = np.vstack([cpore_vals, cpore_anchor_channel * weight_factor]).T
            phit_2d = np.vstack([phit_vals, phit_anchor_channel * weight_factor]).T

            # 5. Run DTW on the augmented 2D data.
            alignment = dtw(
                cpore_2d,
                phit_2d,
                keep_internals=True,
                step_pattern="symmetric2",
            )

            cluster_alignment = pd.DataFrame(
                {
                    "original_depth": dense_core_depths[
                        alignment.index1
                    ],  # Interpolated depths
                    "log_idx": log_subset.index[alignment.index2],
                }
            )
            cluster_alignment["aligned_depth"] = log_subset.loc[
                cluster_alignment["log_idx"], "DEPTH"
            ].values
            cluster_alignment["cluster"] = cluster_id
            all_alignments.append(cluster_alignment)

            # Filter the DTW path to create a reliable training set for the model,
            # excluding alignments that suggest an impossibly large shift.
            cluster_alignment["shift"] = (
                cluster_alignment["aligned_depth"] - cluster_alignment["original_depth"]
            )
            training_alignment = cluster_alignment[
                cluster_alignment["shift"].abs() <= window
            ]

            # If filtering removes all points, the alignment is unreliable; skip this cluster.
            if training_alignment.empty:
                tqdm.write(
                    f"Well {well}, Cluster {cluster_id}: No reliable alignment found within window. "
                    f"Skipping shift for this segment."
                )
                final_corrected_depths[core_indices_in_cluster] = cluster_core_data[
                    "DEPTH"
                ].values
                continue

            # Train a model on the filtered, reliable alignment points.
            X_fit = training_alignment["original_depth"].values.reshape(-1, 1)
            y_fit = training_alignment["aligned_depth"].values

            rf = RandomForestRegressor(random_state=42)
            rf.fit(X_fit, y_fit)

            # Predict the corrected depths for the original core points.
            original_depths = cluster_core_data["DEPTH"].values
            corrected_depths_for_cluster = rf.predict(original_depths.reshape(-1, 1))

            # Post-process the predicted depths for this cluster.
            # 1. Clip the shift to the maximum allowed window size.
            shift = corrected_depths_for_cluster - original_depths
            clipped_shift = np.clip(shift, -window, window)
            corrected_depths_for_cluster = original_depths + clipped_shift

            # 2. Ensure the spacing between points is not less than original spacing.
            if len(corrected_depths_for_cluster) > 1:
                original_diffs = cluster_core_data["DEPTH"].diff().to_numpy()
                for i in range(1, len(corrected_depths_for_cluster)):
                    current_position = (
                        corrected_depths_for_cluster[i] - original_depths[i]
                    )
                    min_spacing = (
                        original_diffs[i]
                        if current_position < 0
                        else -original_diffs[i]
                    )
                    current_spacing = (
                        corrected_depths_for_cluster[i]
                        - corrected_depths_for_cluster[i - 1]
                    )
                    if current_spacing < min_spacing:
                        corrected_depths_for_cluster[i] = (
                            corrected_depths_for_cluster[i - 1] + min_spacing
                        )  # Enforce minimum spacing

            final_corrected_depths[core_indices_in_cluster] = (
                corrected_depths_for_cluster
            )

        # Combine alignment results from all clusters for this well.
        alignment_df = (
            pd.concat(all_alignments, ignore_index=True)
            if all_alignments
            else pd.DataFrame()
        )

        # Enforce strict monotonicity across all corrected depths for the well.
        # This ensures no depth inversions occur between different clusters.
        ir = IsotonicRegression(out_of_bounds="clip")
        final_corrected_depths = ir.fit_transform(
            core_data["DEPTH"], final_corrected_depths
        )

        # Create shifted core data and summary
        shifted_core_data = core_data.copy()
        shifted_core_data["DEPTH_CORRECTED"] = final_corrected_depths
        shifted_core_data["DEPTH_SHIFT"] = (
            final_corrected_depths - shifted_core_data["DEPTH"]
        )

        summary = shifted_core_data[
            ["CORE_ID", "DEPTH", "DEPTH_CORRECTED", "DEPTH_SHIFT"]
        ].copy()
        summary.rename(columns={"DEPTH": "ORIGINAL_DEPTH"}, inplace=True)
        summary["WELL_NAME"] = well
        summary = summary[
            ["WELL_NAME", "CORE_ID", "ORIGINAL_DEPTH", "DEPTH_CORRECTED", "DEPTH_SHIFT"]
        ]
        shift_summaries.append(summary)
        tqdm.write(
            f"Well {well}: Applied segmented correction. Mean shift: {summary['DEPTH_SHIFT'].mean():.2f}m"
        )

        # Reconstruct the well's DataFrame with the newly corrected core depths.
        well_corrected_df = apply_autocorrelation_correction(
            data, shifted_core_data, replace_core_data=replace_ori
        )

        well_corrected_df.sort_values("DEPTH", inplace=True)

        all_wells_data.append(well_corrected_df.reset_index(drop=True))

        _plot_autocorrelation_results(
            well, log_data, core_data, shifted_core_data, alignment_df
        )

    # Combine data from all processed wells into the final return DataFrames.
    return_df = pd.concat(all_wells_data, ignore_index=True)

    # Plot xplot comparing the original and shifted data points together with R2 score and RMSE
    _plot_correction_xplot(return_df)

    final_summary_df = (
        pd.concat(shift_summaries, ignore_index=True)
        if shift_summaries
        else pd.DataFrame()
    )

    return return_df, final_summary_df


def apply_autocorrelation_correction(data, shifted_core_data, replace_core_data=False):
    """
    Applies the depth correction to the core data and merges it back into the main DataFrame.

    This function takes the original well data and the core data with corrected depths,
    then merges them. It can either replace the original core data columns or preserve
    them with an '_ORI' suffix.

    Args:
        data (pd.DataFrame): The original DataFrame for a single well.
        shifted_core_data (pd.DataFrame): A DataFrame containing the core data with
            'DEPTH_CORRECTED' and 'DEPTH_SHIFT' columns from the autocorrelation process.
        replace_core_data (bool, optional): If True, the original core data columns
            (CPORE, CPERM) will be overwritten. If False, they will be preserved with an
            '_ORI' suffix. Defaults to False.

    Returns:
        pd.DataFrame: The merged DataFrame with depth-corrected core data.
    """
    corrected_core_df = shifted_core_data.copy()
    corrected_core_df.rename(columns={"DEPTH": "ORIGINAL_DEPTH"}, inplace=True)
    corrected_core_df.rename(columns={"DEPTH_CORRECTED": "DEPTH"}, inplace=True)

    corrected_core_df["REMARKS"] = corrected_core_df.apply(
        lambda row: f"Corrected from {row['ORIGINAL_DEPTH']:.2f} (Shift: {row['DEPTH_SHIFT']:.2f})",
        axis=1,
    )

    corrected_core_df.drop(columns=["ORIGINAL_DEPTH"], inplace=True, errors="ignore")

    # Prepare for merge
    data_to_merge = data.copy()
    if replace_core_data:
        # Drop original core columns to be replaced by the corrected ones
        cols_to_drop = ["CPORE", "CPERM", "CORE_ID", "REMARKS"]
        data_to_merge.drop(columns=cols_to_drop, inplace=True, errors="ignore")
        suffixes = ("", "")  # No suffix needed as columns are dropped
    else:
        # Clean up any pre-existing _ORI columns before creating new ones
        data_to_merge = data_to_merge.drop(
            columns=[c for c in data_to_merge.columns if "_ORI" in c]
        )
        suffixes = ("_ORI", "")

    data_to_merge.sort_values("DEPTH", inplace=True)
    corrected_core_df.sort_values("DEPTH", inplace=True)

    core_cols_to_merge = ["DEPTH", "CPORE", "CPERM", "CORE_ID", "REMARKS"]
    # Use only columns that exist in corrected_core_df
    core_cols_to_merge = [
        col for col in core_cols_to_merge if col in corrected_core_df.columns
    ]

    well_corrected_df = pd.merge_asof(
        data_to_merge,
        corrected_core_df[core_cols_to_merge],
        on="DEPTH",
        direction="nearest",
        tolerance=1e-2,
        suffixes=suffixes,
    )

    # After merge, drop the temporary shift column
    well_corrected_df.drop(columns=["DEPTH_SHIFT"], inplace=True, errors="ignore")

    return well_corrected_df


def revert_autocorrelation_core_depth(df):
    """
    Reverts the core depth autocorrelation by restoring original depths and data.

    This function undoes the changes made by `autocorrelate_core_depth`. It
    looks for columns with the '_ORI' suffix, which are created during the
    correction process to store the original data. It restores the original
    depth, porosity, and permeability values and removes the columns added
    during correction.

    Args:
        df (pd.DataFrame): A DataFrame that has been processed by
                           `autocorrelate_core_depth`. It is expected to
                           contain columns like 'DEPTH_CORRECTED', 'CPORE_ORI', etc.

    Returns:
        pd.DataFrame: A DataFrame with the core data reverted to its original
                      state before depth correction.
    """
    df_reverted = df.copy()

    # Identify columns that were created during the original merge
    ori_cols = [col for col in df_reverted.columns if col.endswith("_ORI")]

    # Restore original values from the '_ORI' columns
    for col in ori_cols:
        original_col_name = col.replace("_ORI", "")
        # Use np.where to only fill in values where the original existed
        df_reverted[original_col_name] = np.where(
            df_reverted[col].notna(), df_reverted[col], df_reverted[original_col_name]
        )

    # Columns to drop: the backup '_ORI' columns and any correction artifacts
    cols_to_drop = ori_cols + ["REMARKS", "DEPTH_SHIFT", "cluster"]
    df_reverted.drop(columns=cols_to_drop, inplace=True, errors="ignore")

    return df_reverted


def _plot_correction_xplot(df):
    """
    Plots a crossplot comparing log porosity (PHIT) with original and corrected
    core porosity (CPORE), showing R-squared and RMSE for both.

    Args:
        df (pd.DataFrame): DataFrame containing 'PHIT', 'CPORE', 'CPORE_ORI', and 'CORE_ID'.
    """
    # Prepare data by dropping NaNs
    plot_df = df[["PHIT", "CPORE_ORI", "CPORE", "CORE_ID"]].dropna().copy()

    # Calculate metrics for original data
    r2_original = r2_score(plot_df["PHIT"], plot_df["CPORE_ORI"])
    rmse_original = root_mean_squared_error(plot_df["PHIT"], plot_df["CPORE_ORI"])

    # Calculate metrics for corrected data
    r2_corrected = r2_score(plot_df["PHIT"], plot_df["CPORE"])
    rmse_corrected = root_mean_squared_error(plot_df["PHIT"], plot_df["CPORE"])

    # Create the plot
    fig, ax = plt.subplots(figsize=(8, 8))

    # Scatter plot for original and corrected data
    ax.scatter(
        plot_df["PHIT"],
        plot_df["CPORE_ORI"],
        alpha=0.5,
        label=f"Original (R2={r2_original:.2f}, RMSE={rmse_original:.3f})",
        marker="x",
    )
    ax.scatter(
        plot_df["PHIT"],
        plot_df["CPORE"],
        alpha=0.5,
        label=f"Corrected (R2={r2_corrected:.2f}, RMSE={rmse_corrected:.3f})",
        marker="o",
    )

    # Add lines and annotations connecting original to corrected points
    for _, row in plot_df.iterrows():
        # Add a line from the original point to the corrected point
        ax.plot(
            [row["PHIT"], row["PHIT"]],
            [row["CPORE_ORI"], row["CPORE"]],
            color="gray",
            linestyle="--",
            linewidth=0.8,
            alpha=0.7,
        )
        # Annotate the CORE_ID near the corrected point
        ax.text(
            row["PHIT"] * 1.01, row["CPORE"], str(row["CORE_ID"]), fontsize=7, alpha=0.8
        )

    # Add 1:1 line for reference
    lims = [
        min(ax.get_xlim()[0], ax.get_ylim()[0]),
        max(ax.get_xlim()[1], ax.get_ylim()[1]),
    ]
    ax.plot(lims, lims, "k--", alpha=0.75, zorder=0, label="1:1 Line")

    # Formatting
    ax.set_xlabel("Log Porosity (PHIT)")
    ax.set_ylabel("Core Porosity (CPORE)")
    ax.set_title("Core vs. Log Porosity Before and After Depth Correction")
    ax.legend()
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.tight_layout()
    plt.show()


def _plot_autocorrelation_results(
    well_name, log_data, original_core, corrected_core, alignment_df
):
    """
    Visualize the results of the core depth correction using Plotly.

    This helper function generates a two-panel plot for quality control:
    1.  **Porosity Alignment**: Shows log porosity, original core porosity, and
        depth-corrected core porosity versus depth. Lines connect original to
        corrected points to visualize the shifts.
    2.  **DTW Alignment**: Plots the DTW path, showing how original core depths
        were mapped to log depths. A 1:1 line indicates no shift.

    Args:
        well_name (str): The name of the well being plotted.
        log_data (pd.DataFrame): DataFrame with log data ('DEPTH', 'PHIT').
        original_core (pd.DataFrame): DataFrame with original core data.
        corrected_core (pd.DataFrame): DataFrame with depth-corrected core data.
        alignment_df (pd.DataFrame): DataFrame containing the DTW alignment path.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Porosity Alignment", "DTW Alignment & Shift Clusters"),
        shared_yaxes=True,
    )

    # Panel 1: Porosity Alignment Plot
    fig.add_trace(
        go.Scatter(
            x=log_data["PHIT"],
            y=log_data["DEPTH"],
            name="Log PHIT",
            mode="lines",
            line={"color": "black", "width": 1},
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=original_core["CPORE"],
            y=original_core["DEPTH"],
            name="Original CPORE",
            mode="lines+markers",
            line={"color": "red"},
            marker={"symbol": "circle-open"},
            text=[f"Core ID: {c}" for c in original_core["CORE_ID"]],
            hoverinfo="x+y+text",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=corrected_core["CPORE"],
            y=corrected_core["DEPTH_CORRECTED"],
            name="Corrected CPORE",
            mode="lines+markers",
            line={"color": "blue"},
            text=[f"Core ID: {c}" for c in corrected_core["CORE_ID"]],
            hoverinfo="x+y+text",
        ),
        row=1,
        col=1,
    )

    # Add lines connecting original to corrected points to show the shift.
    x_lines, y_lines = [], []
    for _, row in corrected_core.iterrows():
        x_lines.extend([row["CPORE"], row["CPORE"], None])
        y_lines.extend([row["DEPTH"], row["DEPTH_CORRECTED"], None])
    fig.add_trace(
        go.Scatter(
            x=x_lines,
            y=y_lines,
            name="Depth Shift",
            mode="lines",
            line={"color": "gray", "dash": "dot", "width": 1},
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Panel 2: DTW Path and Cluster Visualization
    fig.add_trace(
        go.Scatter(
            x=alignment_df["original_depth"],
            y=alignment_df["aligned_depth"],
            name="DTW Path",
            mode="lines",
            line={"color": "lightgray"},
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=alignment_df["original_depth"],
            y=alignment_df["aligned_depth"],
            name="Clusters",
            mode="markers",
            marker={
                "color": alignment_df["cluster"],
                "colorscale": "Viridis",
                "showscale": False,
                "colorbar": {"title": "Cluster ID"},
                "cmin": 0,
                "cmid": alignment_df["cluster"].max() / 2
                if alignment_df["cluster"].max() > 0
                else 0.5,
            },
            text=[f"Cluster: {c}" for c in alignment_df["cluster"].values],
            hoverinfo="x+y+text",
        ),
        row=1,
        col=2,
    )

    # Add a 1:1 line to represent no depth shift.
    lim_min = min(
        alignment_df["original_depth"].min(), alignment_df["aligned_depth"].min()
    )
    lim_max = max(
        alignment_df["original_depth"].max(), alignment_df["aligned_depth"].max()
    )
    fig.add_shape(
        type="line",
        x0=lim_min,
        y0=lim_min,
        x1=lim_max,
        y1=lim_max,
        line={"color": "black", "width": 2, "dash": "dash"},
        row=1,
        col=2,
    )

    fig.update_layout(
        title_text=f"Core Depth Autocorrelation Results for Well: {well_name}",
        legend_title_text="Data",
        height=700,
        width=1200,
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_xaxes(title_text="Porosity (frac)", row=1, col=1)
    fig.update_yaxes(title_text="Depth", row=1, col=1)
    fig.update_xaxes(title_text="Original Core Depth", row=1, col=2)
    fig.update_yaxes(title_text="DTW Aligned Log Depth", row=1, col=2)

    fig.update_xaxes(fixedrange=True)

    fig.show()


def restructure_scal_data(df_wide):
    """
    Restructures a SCAL dataset from a wide format (multiple Pc/Sw columns per depth)
    to a long format (one Pc/Sw pair per row).

    Args:
        df_wide (pd.DataFrame): The input DataFrame in wide format.

    Returns:
        pd.DataFrame: The restructured DataFrame in long format.
    """
    # Identify columns to rename
    columns = df_wide.columns
    new_columns = {}
    for col in columns:
        if col.startswith("Pc") or col.startswith("Sw"):
            new_col = col.split("_")[0]
            new_col = f"{new_col[:2]}.{new_col[-1]}"
            new_columns[col] = new_col

    df_temp = df_wide.rename(columns=new_columns)

    # 2. Apply the wide_to_long function
    cols = [
        "Well",
        "Sample ID",
        "Depth_m",
        "K_mD",
        "PHI_frac",
        "Formation Tops",
        "Model",
    ]
    df_long = pd.wide_to_long(
        df_temp,
        # The 'stubnames' are the prefixes we want to turn into columns
        stubnames=["Pc", "Sw"],
        # 'i' is the column that identifies the groups/rows (Depth)
        i=cols,
        # 'j' is the new column that holds the suffix (the measurement index)
        j="Measurement_Index",
        # The separator used between the stubname and the suffix (e.g., Pc.1 uses '.')
        sep=".",
    ).reset_index()

    # Remove rows where all the newly created measurement columns are NaN (if any exist)
    df_long.dropna(subset=["Pc", "Sw"], how="all", inplace=True)

    # Sort by Depth and Measurement_Index for clean viewing (optional)
    df_long.sort_values(
        by=["Well", "Sample ID", "Depth_m", "Measurement_Index"], inplace=True
    )

    # Finalize columns for output
    df_final = df_long[cols + ["Pc", "Sw"]].copy()

    # Reset index and return
    df_final.reset_index(drop=True, inplace=True)

    return df_final


def string_to_int_hash(s):
    """Converts a string to a stable integer hash of at most 4 digits (0-9999).

    This function uses a SHA256 hash to ensure that the same string always
    produces the same integer, which is useful for creating reproducible
    color mappings or short identifiers.

    Args:
        s (str): The input string to hash.

    Returns:
        int or None: An integer between 0 and 9999, or None if the input is NaN.
    """
    if pd.isna(s):  # Handle NaN values
        return None
    # Use SHA256 for stability across runs
    hash_obj = hashlib.sha256(s.encode("utf-8"))
    # Convert first 4 bytes to a large integer
    large_int = int.from_bytes(hash_obj.digest()[:4], "big", signed=False)
    # Constrain the integer to a 4-digit number (0-9999)
    return large_int % 10000


def normalize_sw(sw):
    """Normalizes water saturation (Sw) values to a range between 0 and 1.

    This function scales the input water saturation values such that the minimum
    value becomes 0 and the maximum value becomes 1.

    Args:
    sw (np.ndarray or pd.Series): Array or Series of water saturation values.

    Returns:
    np.ndarray or pd.Series: Normalized water saturation values.
    """
    return (sw - sw.min()) / (1 - sw.min())


def auto_cluster_scal_data(core_data):
    """Automatically clusters core samples into rock types based on J-function parameters.

    This function first calculates the 'a' and 'b' parameters of the J-curve for
    each individual core sample. It then uses K-Means clustering on these parameters
    to group the samples into a specified number of clusters (rock types). If the
    number of clusters is not provided, it determines the optimal number using the
    Elbow method.

    Args:
        core_data (pd.DataFrame): DataFrame with SCAL data. Must include 'Well',
            'Sample', 'SWN', and 'J' columns.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'ROCK_FLAG' column containing
                      the cluster label for each sample.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    core_data = core_data.copy()

    # Calculate 'a' and 'b' for each sample
    sample_params = []
    for sample_id, data in core_data.groupby(["ROCK_FLAG", "Well", "Sample"]):
        clean_data = data.dropna(subset=["SWN", "J"]).query("SWN > 0")

        # Ensure there's still data left to fit
        if len(clean_data) < 2:
            continue  # Skip groups with insufficient data after filtering

        a, b = fit_j_curve(clean_data["SWN"], clean_data["J"])
        # Exclude samples where fitting might have failed (returned default values)
        if not (a == 1 and b == 1):
            sample_params.append(
                {
                    "ROCK_FLAG": sample_id[0],
                    "Well": sample_id[1],
                    "Sample": sample_id[2],
                    "CPORE": clean_data["CPORE"].iloc[0],
                    "CPERM": np.log10(clean_data["CPERM"].iloc[0].clip(1e-6)),
                    "a": a,
                    "b": b,
                }
            )

    if not sample_params:
        return core_data.assign(ROCK_FLAG=0)

    params_df = pd.DataFrame(sample_params)
    # Initialize an offset to ensure cluster labels are unique across all ROCK_FLAGs
    cluster_offset = 0
    for _, data in params_df.groupby("ROCK_FLAG"):
        X = data[["CPORE", "CPERM", "a", "b"]].values

        # Scale features for better clustering performance
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # 2. Automatically determine the optimal 'eps' for DBSCAN
        min_samples = 2
        if len(X_scaled) <= min_samples:
            # Not enough data to cluster, assign all to a single group
            params_df.loc[data.index, "CORE_CLUSTER"] = 1 + cluster_offset
            cluster_offset += 1
            continue

        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X_scaled)
        distances, _ = neighbors_fit.kneighbors(X_scaled)
        sorted_distances = np.sort(distances[:, min_samples - 1])
        # The point of maximum curvature (the "knee") is a good estimate for eps
        optimal_eps = sorted_distances[np.argmax(np.diff(sorted_distances, 2))] + 1e-6

        dbscan = DBSCAN(eps=optimal_eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(X_scaled)

        # Assign cluster labels, adding offset to ensure uniqueness across ROCK_FLAGs
        # Noise points (labeled -1 by DBSCAN) are assigned to cluster 0
        params_df.loc[data.index, "CORE_CLUSTER"] = np.where(
            clusters == -1, 0, clusters + 1 + cluster_offset
        )

        # Update the offset for the next ROCK_FLAG group
        if clusters.max() > -1:
            cluster_offset += clusters.max() + 1

    if "CORE_CLUSTER" in core_data.columns:
        core_data = core_data.drop(columns=["CORE_CLUSTER"])

    return pd.merge(
        core_data,
        params_df[["Well", "Sample", "CORE_CLUSTER"]],
        on=["Well", "Sample"],
        how="left",
    )


def auto_j_params(
    core_data,
    excluded_samples=None,
    cluster_by=None,
    ensure_non_crossing=True,
    dampen_outliers=True,
):
    """Automatically calculates J-function parameters (a, b) for each rock type.

    This function groups the data by 'ROCK_FLAG', fits a robust outlier-dampened J-curve
    for each group, and enforces non-crossing constraints across rock types over Swn in [0.01, 1.0].

    Args:
        core_data (pd.DataFrame): DataFrame containing 'ROCK_FLAG', 'SWN', and 'J' columns.
        excluded_samples (list, optional): A list of 'Sample' names to exclude from the calculation.
            Defaults to [].
        cluster_by (list, optional): A list of column names to create sub-groups
            within each 'ROCK_FLAG'. If provided, parameters will be calculated
            for each unique combination of 'ROCK_FLAG' and the cluster_by columns.
            Defaults to None.
        ensure_non_crossing (bool, optional): Enforce strict non-crossing ordering across
            rock-type curves over the saturation range Swn in [0.01, 1.0]. Defaults to True.
        dampen_outliers (bool, optional): Dampen extreme SCAL outliers and boundary asymptotes.
            Defaults to True.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              group identifiers (e.g., 'ROCK_FLAG') and the J-parameters ('a', 'b', 'rmse', 'r2', 'count').
    """
    from scipy.optimize import minimize

    excluded_samples = excluded_samples or []
    if "Sample" in core_data.columns:
        core_data = core_data[~core_data["Sample"].isin(excluded_samples)].copy()
    else:
        core_data = core_data.copy()

    grouping_cols = ["ROCK_FLAG"]
    if cluster_by:
        if not isinstance(cluster_by, list):
            cluster_by = [cluster_by]
        grouping_cols.extend(cluster_by)

    group_records = []
    for group_key, data in core_data.groupby(grouping_cols):
        clean_data = data.dropna(subset=["SWN", "J"]).query("SWN > 0 and J > 0")
        if len(clean_data) < 2:
            continue

        a_init, b_init, weights, sw_arr, j_arr = _robust_fit_j_logspace(
            clean_data["SWN"], clean_data["J"], dampen_outliers=dampen_outliers
        )
        if len(sw_arr) < 2:
            continue

        params = {
            col: (
                int(val) if isinstance(val, (np.integer, np.int32, np.int64)) else val
            )
            for col, val in zip(
                grouping_cols,
                group_key if isinstance(group_key, tuple) else [group_key],
                strict=True,
            )
        }
        group_records.append(
            {
                "id_params": params,
                "a": float(a_init),
                "b": float(b_init),
                "weights": weights,
                "sw": sw_arr,
                "j": j_arr,
                "x": -np.log(np.clip(sw_arr, 1e-6, 1.0)),
                "y": np.log(np.clip(j_arr, 1e-8, 1e8)),
                "count": int(len(clean_data)),
            }
        )

    # Joint non-crossing optimization across Rock Types in log-log space
    if ensure_non_crossing and not cluster_by and len(group_records) >= 2:
        group_records.sort(key=lambda r: r["id_params"].get("ROCK_FLAG", 0))
        k_groups = len(group_records)
        u_ref = float(-np.log(0.30))
        u_max = float(-np.log(0.01))

        mu_vals = np.array(
            [np.log(max(r["a"], 1e-4)) + r["b"] * u_ref for r in group_records],
            dtype=float,
        )
        rf_idx = np.arange(k_groups, dtype=float)
        cov_rf_mu = float(np.sum((rf_idx - np.mean(rf_idx)) * (mu_vals - np.mean(mu_vals))))

        # Order from lowest J curve to highest J curve following physical rock-type progression
        if cov_rf_mu >= 0.0:
            order = list(range(k_groups))
        else:
            order = list(range(k_groups - 1, -1, -1))

        sqrt_counts = np.array([np.sqrt(max(r["count"], 1)) for r in group_records], dtype=float)
        b_raw = np.array([r["b"] for r in group_records], dtype=float)
        b_shared = float(
            np.clip(np.sum(sqrt_counts * b_raw) / np.sum(sqrt_counts), 0.15, 2.2)
        )

        mu_span = max(float(np.max(mu_vals) - np.min(mu_vals)), 0.25 * (k_groups - 1))
        delta = float(np.clip(0.25 * mu_span / max(k_groups - 1, 1), 0.06, 0.18))

        # Initialize monotonic endpoints R_j (at Swn=1, u=0) and L_j (at Swn=0.01, u=u_max)
        mu_ordered = np.array([mu_vals[idx] for idx in order], dtype=float)
        mu_mono = np.zeros(k_groups, dtype=float)
        mu_mono[0] = float(np.min(mu_ordered)) if mu_ordered[0] > np.median(mu_ordered) else float(mu_ordered[0])
        for j_i in range(1, k_groups):
            mu_mono[j_i] = max(float(mu_ordered[j_i]), mu_mono[j_i - 1] + delta * 1.2)
        # Re-center monotonic mid-levels to match the empirical mean level
        mu_mono += float(np.mean(mu_ordered) - np.mean(mu_mono))

        r_init = mu_mono - b_shared * u_ref
        l_init = r_init + b_shared * u_max

        z0 = np.zeros(2 * k_groups, dtype=float)
        z0[0] = r_init[0]
        z0[1:k_groups] = np.maximum(np.diff(r_init), delta)
        z0[k_groups] = l_init[0]
        z0[k_groups + 1 : 2 * k_groups] = np.maximum(np.diff(l_init), delta)

        bounds = (
            [(float(np.log(1e-4)), float(np.log(50.0)))]
            + [(delta, 8.0)] * (k_groups - 1)
            + [(float(np.log(1e-4)), float(np.log(5000.0)))]
            + [(delta, 8.0)] * (k_groups - 1)
        )

        def _joint_loss(z_vec: np.ndarray) -> float:
            r_ends = np.cumsum(z_vec[:k_groups])
            l_ends = np.cumsum(z_vec[k_groups:])
            total = 0.0
            tau = 0.25
            for j_pos, rec_idx in enumerate(order):
                rec = group_records[rec_idx]
                alpha_k = r_ends[j_pos]
                b_k = (l_ends[j_pos] - r_ends[j_pos]) / u_max
                res_k = rec["y"] - (alpha_k + b_k * rec["x"])
                abs_r = np.abs(res_k)
                huber_k = np.where(
                    abs_r <= tau, 0.5 * (abs_r**2), tau * (abs_r - 0.5 * tau)
                )
                w_k = rec["weights"]
                w_sum = max(float(np.sum(w_k)), 1e-8)
                data_loss = float(np.sum(w_k * huber_k) / w_sum)
                slope_pen = (
                    0.18 * ((b_k - b_shared) ** 2)
                    + 0.04 * ((b_k - rec["b"]) ** 2)
                    + 30.0 * (max(0.0, 0.08 - b_k) ** 2)
                    + 30.0 * (max(0.0, b_k - 2.4) ** 2)
                )
                total += data_loss + slope_pen
            return total

        try:
            opt = minimize(
                _joint_loss,
                z0,
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 600, "ftol": 1e-9},
            )
            z_best = opt.x if opt.success or np.isfinite(opt.fun) else z0
        except Exception:
            z_best = z0

        r_final = np.cumsum(z_best[:k_groups])
        l_final = np.cumsum(z_best[k_groups:])
        for j_pos, rec_idx in enumerate(order):
            alpha_k = float(r_final[j_pos])
            b_k = float(np.clip((l_final[j_pos] - r_final[j_pos]) / u_max, 0.08, 2.5))
            group_records[rec_idx]["a"] = float(np.clip(np.exp(alpha_k), 1e-4, 50.0))
            group_records[rec_idx]["b"] = b_k

        # Guarantee strict non-crossing even after 4-decimal rounding
        for j_pos in range(k_groups):
            rec_idx = order[j_pos]
            a_r = max(round(group_records[rec_idx]["a"], 4), 1e-4)
            b_r = max(round(group_records[rec_idx]["b"], 4), 0.05)
            if j_pos > 0:
                prev_idx = order[j_pos - 1]
                a_prev = group_records[prev_idx]["a"]
                b_prev = group_records[prev_idx]["b"]
                if a_r <= a_prev:
                    a_r = round(a_prev * float(np.exp(delta)), 4)
                    if a_r <= a_prev:
                        a_r = round(a_prev + 0.005, 4)
                j_left_prev = a_prev * (0.01 ** (-b_prev))
                j_left_curr = a_r * (0.01 ** (-b_r))
                if j_left_curr <= j_left_prev * 1.02:
                    b_min_req = float(
                        np.log((j_left_prev * float(np.exp(delta))) / a_r) / u_max
                    )
                    b_r = max(round(b_min_req, 4), 0.05)
            group_records[rec_idx]["a"] = a_r
            group_records[rec_idx]["b"] = b_r

    j_params_list = []
    for rec in group_records:
        a = round(float(rec["a"]), 4)
        b = round(float(rec["b"]), 4)
        sw_arr = rec["sw"]
        j_arr = rec["j"]
        w_arr = rec["weights"]
        j_pred = inv_power_law_func(sw_arr, a, b)

        # Compute outlier-dampened RMSE and log-space / weighted R2
        w_norm = w_arr / max(float(np.sum(w_arr)), 1e-12)
        rmse = round(float(np.sqrt(np.sum(w_norm * ((j_arr - j_pred) ** 2)))), 4)
        y_log = rec["y"]
        y_pred_log = np.log(np.clip(j_pred, 1e-8, 1e8))
        y_mean_log = float(np.sum(w_norm * y_log))
        ss_res = float(np.sum(w_norm * ((y_log - y_pred_log) ** 2)))
        ss_tot = float(np.sum(w_norm * ((y_log - y_mean_log) ** 2)))
        r2 = round(max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0, 4)

        params = dict(rec["id_params"])
        params.update(
            {
                "a": a,
                "b": b,
                "rmse": rmse,
                "r2": r2,
                "count": rec["count"],
            }
        )
        j_params_list.append(params)

    j_params_list.sort(
        key=lambda x: (x.get("ROCK_FLAG", 0), x.get("a", 0), x.get("b", 0))
    )
    return j_params_list


def auto_skelt_harrison_params(core_data, excluded_samples=None, cluster_by=None):
    """Automatically calculates Skelt-Harrison parameters (a, b, c, d) for each rock type.

    Args:
        core_data (pd.DataFrame): DataFrame containing 'ROCK_FLAG', 'SW' (or 'SWN'), and 'H' columns.
        excluded_samples (list, optional): A list of 'Sample' names to exclude.
        cluster_by (list, optional): Additional columns to group by within each 'ROCK_FLAG'.

    Returns:
        list: List of dictionaries with group identifiers and ('a', 'b', 'c', 'd', 'rmse', 'r2', 'count').
    """
    excluded_samples = excluded_samples or []
    if "Sample" in core_data.columns:
        core_data = core_data[~core_data["Sample"].isin(excluded_samples)].copy()
    else:
        core_data = core_data.copy()

    sw_col = "SW" if "SW" in core_data.columns else "SWN"
    grouping_cols = ["ROCK_FLAG"]
    if cluster_by:
        if not isinstance(cluster_by, list):
            cluster_by = [cluster_by]
        grouping_cols.extend(cluster_by)

    params_list = []
    for group_key, data in core_data.groupby(grouping_cols):
        clean_data = data.dropna(subset=[sw_col, "H"]).query(f"{sw_col} >= 0 and H >= 0")
        if len(clean_data) < 2:
            continue

        a, b, c, d = fit_skelt_harrison_curve(clean_data[sw_col], clean_data["H"])
        sw_pred = skelt_harrison_func(clean_data["H"], a, b, c, d)
        rmse = round(root_mean_squared_error(clean_data[sw_col], sw_pred), 4)
        ss_res = float(np.sum((clean_data[sw_col] - sw_pred) ** 2))
        ss_tot = float(np.sum((clean_data[sw_col] - np.mean(clean_data[sw_col])) ** 2))
        r2 = round(max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0, 4)

        params = {
            col: (
                int(val) if isinstance(val, (np.integer, np.int32, np.int64)) else val
            )
            for col, val in zip(
                grouping_cols,
                group_key if isinstance(group_key, tuple) else [group_key],
                strict=True,
            )
        }
        params.update(
            {
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "c": round(float(c), 4),
                "d": round(float(d), 4),
                "rmse": rmse,
                "r2": r2,
                "count": int(len(clean_data)),
            }
        )
        params_list.append(params)

    params_list.sort(key=lambda x: (x.get("ROCK_FLAG", 0), x.get("b", 0)))
    return params_list


def auto_cuddy_params(core_data, excluded_samples=None, cluster_by=None):
    """Automatically calculates Cuddy BVW parameters (a, b) for each rock type or globally.

    Args:
        core_data (pd.DataFrame): DataFrame containing 'BVW' and 'PC' columns (and optionally 'ROCK_FLAG').
        excluded_samples (list, optional): A list of 'Sample' names to exclude.
        cluster_by (list, optional): Additional columns to group by.

    Returns:
        list: List of dictionaries with group identifiers and ('a', 'b', 'rmse', 'r2', 'count').
    """
    excluded_samples = excluded_samples or []
    if "Sample" in core_data.columns:
        core_data = core_data[~core_data["Sample"].isin(excluded_samples)].copy()
    else:
        core_data = core_data.copy()

    grouping_cols = ["ROCK_FLAG"] if "ROCK_FLAG" in core_data.columns else []
    if cluster_by:
        if not isinstance(cluster_by, list):
            cluster_by = [cluster_by]
        grouping_cols.extend(cluster_by)

    params_list = []
    groups = core_data.groupby(grouping_cols) if grouping_cols else [(0, core_data)]
    for group_key, data in groups:
        clean_data = data.dropna(subset=["BVW", "PC"]).query("BVW > 0 and PC > 0")
        if len(clean_data) < 2:
            continue

        a, b = fit_cuddy_curve(clean_data["BVW"], clean_data["PC"])
        bvw_pred = cuddy_func(clean_data["PC"], a, b)
        rmse = round(root_mean_squared_error(clean_data["BVW"], bvw_pred), 4)
        ss_res = float(np.sum((clean_data["BVW"] - bvw_pred) ** 2))
        ss_tot = float(np.sum((clean_data["BVW"] - np.mean(clean_data["BVW"])) ** 2))
        r2 = round(max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0, 4)

        params = (
            {
                col: (
                    int(val)
                    if isinstance(val, (np.integer, np.int32, np.int64))
                    else val
                )
                for col, val in zip(
                    grouping_cols,
                    group_key if isinstance(group_key, tuple) else [group_key],
                    strict=True,
                )
            }
            if grouping_cols
            else {"ROCK_FLAG": 0}
        )
        params.update(
            {
                "a": round(float(a), 4),
                "b": round(float(b), 4),
                "rmse": rmse,
                "r2": r2,
                "count": int(len(clean_data)),
            }
        )
        params_list.append(params)

    params_list.sort(key=lambda x: (x.get("ROCK_FLAG", 0), x.get("a", 0)))
    return params_list


def auto_group_core_description(
    core_data, geo_abbrv=None, word_cat=None, special_case_desc=None
):
    """Groups core descriptions into standardized categories using NLP and clustering.

    The function follows a multi-step process:
    1.  It first identifies and categorizes special descriptions (e.g., "NO PLUG POSSIBLE")
        based on the `special_case_desc` dictionary.
    2.  For the remaining descriptions, it expands common geological abbreviations
        (e.g., "ss" -> "sandstone") using the `geo_abbrv` dictionary.
    3.  It then uses TF-IDF to vectorize the cleaned text and DBSCAN to cluster
        similar descriptions together.
    4.  Finally, it generates a structured, descriptive name for each cluster
        (e.g., "sandstone_fine_grained") by selecting the most representative term
        from each category defined in `word_cat`.

    Args:
        core_data (pd.DataFrame): DataFrame containing a 'CORE_DESC' column with
                                  textual core descriptions.
        geo_abbrv (dict, optional): Dictionary mapping geological abbreviations to
            their full terms. Defaults to `Config.CORE_GEO_ABBREVIATIONS`.
        word_cat (dict, optional): Dictionary categorizing geological terms for
            structured naming. Defaults to `Config.CORE_WORD_CATEGORIES`.
        special_case_desc (dict, optional): Dictionary for special descriptions
            that bypass clustering. Defaults to `Config.CORE_SPECIAL_CASE_DESCRIPTIONS`.

    Returns:
        pd.DataFrame: The input DataFrame with an added 'CORE_DESC_GRP' column
                      containing the generated group names.
    """
    from sklearn.cluster import DBSCAN
    from sklearn.feature_extraction.text import TfidfVectorizer

    geo_abbrv = geo_abbrv or GEO_ABBREVIATIONS
    word_cat = word_cat or WORD_CATEGORIES
    special_case_desc = special_case_desc or SPECIAL_CASE_DESCRIPTIONS

    if "CORE_DESC" not in core_data.columns:
        raise ValueError("Input DataFrame must contain a 'CORE_DESC' column.")

    core_data = core_data.copy()
    descriptions = core_data["CORE_DESC"].fillna("").astype(str)
    core_data["CORE_DESC_GRP"] = "Miscellaneous"  # Default group

    # 1. Handle special cases first
    remaining_indices = core_data.index.to_series()
    for phrase, group_name in special_case_desc.items():
        is_special_case = descriptions.str.lower().str.contains(phrase, na=False)
        core_data.loc[is_special_case, "CORE_DESC_GRP"] = group_name
        # Exclude these from the clustering process
        remaining_indices = remaining_indices[~is_special_case]

    if remaining_indices.empty:
        return core_data  # All descriptions were special cases

    # Continue with clustering only on the remaining data
    clustering_data = core_data.loc[remaining_indices]
    descriptions_to_cluster = clustering_data["CORE_DESC"].fillna("").astype(str)

    # Keep only unique descriptions for clustering to improve performance
    unique_descriptions = descriptions_to_cluster.unique()

    def preprocess_text(text, abbreviations):
        # This function is now only used for the text that goes into clustering
        text = text.lower()
        # Expand abbreviations using word boundaries for safety
        for abbr, full in abbreviations.items():
            text = re.sub(rf"\b{abbr}\b", full, text)
        # Remove punctuation, numbers, and extra spaces
        text = re.sub(r"[^a-z\s]", "", text)
        return text

    processed_descriptions = [
        preprocess_text(desc, geo_abbrv) for desc in unique_descriptions
    ]

    # If after preprocessing there are no descriptions left with enough features, return
    if not processed_descriptions:
        return core_data

    # 2. Vectorize the text using TF-IDF
    vectorizer = TfidfVectorizer(stop_words="english", min_df=2)
    tfidf_matrix = vectorizer.fit_transform(processed_descriptions)

    # 3. Cluster the vectors using DBSCAN
    # eps is a crucial parameter; 0.5 is a reasonable starting point for normalized data
    dbscan = DBSCAN(eps=0.5, min_samples=2, metric="cosine")

    # Handle case where tfidf_matrix is empty
    if tfidf_matrix.shape[0] == 0:
        return core_data

    clusters = dbscan.fit_predict(tfidf_matrix)

    # 4. Generate representative names for each cluster
    cluster_names = {}
    feature_names = vectorizer.get_feature_names_out()
    for cluster_id in np.unique(clusters):
        if cluster_id == -1:
            cluster_names[cluster_id] = "Miscellaneous"
            continue

        # Find indices of descriptions in this cluster
        indices = np.where(clusters == cluster_id)[0]
        # Get the TF-IDF vectors for this cluster
        cluster_vectors = tfidf_matrix[indices]
        # Calculate the mean TF-IDF score for each word in the cluster
        mean_tfidf = cluster_vectors.mean(axis=0).A1

        # Create a dictionary of term -> score
        term_scores = {term: mean_tfidf[i] for i, term in enumerate(feature_names)}

        # Select the best term from each category based on score
        ordered_terms = []
        for category_name in word_cat:
            category_words = word_cat[category_name]

            # Find the word from this category with the highest score in the current cluster
            best_term = max(
                (term for term in category_words if term in term_scores),
                key=lambda term: term_scores[term],
                default=None,
            )
            if best_term:
                ordered_terms.append(best_term)

        cluster_names[cluster_id] = (
            "_".join(ordered_terms) if ordered_terms else "Undefined"
        )

    # 5. Create a mapping from unique description to its named group
    desc_to_group_map = {
        desc: cluster_names[cluster_id]
        for desc, cluster_id in zip(unique_descriptions, clusters, strict=True)
    }

    # 6. Assign the descriptive group names to the remaining part of the DataFrame
    core_data.loc[remaining_indices, "CORE_DESC_GRP"] = descriptions_to_cluster.map(
        desc_to_group_map
    )

    return core_data


def plot_ptsd_by_prt(core_data, ift, theta, no_of_rocks=5):
    """Plots Pore Throat Size Distribution (PTSD) for each Petrophysical Rock Type (PRT).

    This function calculates the pore throat radius from capillary pressure data and
    plots the derivative of water saturation with respect to the log of the radius
    (dSw/dLogR) against the log of the radius. This provides a visual representation
    of the pore throat size distribution for different rock types.

    Args:
        core_data (pd.DataFrame): DataFrame with core data, including 'Well', 'Sample',
            'PC_RES', 'SW', and 'ROCK_FLAG'.
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Contact angle (degrees).
        no_of_rocks (int, optional): The number of rock types to plot. Defaults to 5.

    Returns:
        pd.DataFrame: The input DataFrame with added columns 'R', 'LOG_R', and 'DSW'.
    """
    core_data = core_data.copy()

    temp_dfs = []
    for _, data in core_data.groupby(["Well", "Sample"]):
        # Sort by PC to ensure monotonic change in R and LOG_R
        data = data.sort_values("PC_RES", ascending=True).copy()
        data["R"] = estimate_pore_throat(data["PC_RES"], ift, theta)
        data["LOG_R"] = np.log10(data["R"])
        data["DSW"] = np.gradient(data["SWN"], data["LOG_R"])
        temp_dfs.append(data)

    if not temp_dfs:
        print("No data to plot.")
        return

    processed_df = pd.concat(temp_dfs)

    fig, axes = plt.subplots(4, 3, figsize=(15, 17))
    axes = axes.flatten()
    for i in range(no_of_rocks):
        rock = i + 1
        data = processed_df[processed_df.ROCK_FLAG == rock]
        if data.empty:
            continue  # Skip rock types with no data
        ax = axes[i]
        for sample, sample_data in data.groupby(["Well", "Sample"]):
            ax.plot(
                sample_data["LOG_R"],
                sample_data["DSW"],
                label=f"Sample {sample}",
                zorder=-1,
            )
            color = ax.lines[-1].get_color()
            ax.fill_between(
                sample_data["LOG_R"], sample_data["DSW"], color=color, alpha=0.9
            )

        ax.set_title(f"PRT {rock}")
        ax.set_xlabel("Log Pore Throat Radius (microns)")
        ax.set_xlim(-2, 2)
        ax.set_ylabel("dSw/dLogR")
        ax.legend(loc=2, prop={"size": 5})

    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.set_facecolor("aliceblue")
    plt.tight_layout()

    return processed_df


def plot_pc_by_prt(core_data, ymax=10):
    """Generates a grid of Pc-Sw plots, one for each Reservoir Rock Type (RRT).

    This function creates subplots to visualize the capillary pressure (Pc) versus
    water saturation (Sw) relationship for each rock type present in the dataset.

    Args:
        core_data (pd.DataFrame): DataFrame containing core data, including 'Well', 'Sample',
            'ROCK_FLAG', 'SW', and 'PC_RES'.
        ymax (int, optional): The maximum limit for the y-axis (Pc). Defaults to 10.
    """
    core_data = core_data.copy()

    # Get unique rock flags
    unique_rock_flags = sorted(core_data["ROCK_FLAG"].unique())

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(15, 20))
    axes = axes.flatten()

    # Plot Pc vs SW for each rock flag
    for i, rock in enumerate(unique_rock_flags):
        ax = axes[i]
        data = core_data[core_data["ROCK_FLAG"] == rock]
        if data.empty:
            continue
        for sample, sample_data in data.groupby(["Well", "Sample"]):
            sample_data = sample_data.sort_values("PC_RES").copy()
            ax.plot(
                sample_data["SW"],
                sample_data["PC_RES"],
                label=f"Sample {sample}",
                marker="o",
            )
        ax.set_ylabel("Pc (psia)")
        ax.set_xlabel("SW (frac)")
        ax.set_ylim(0, ymax)
        ax.set_xlim(0, 1)
        ax.set_title(f"RRT {int(rock)}")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(unique_rock_flags), len(axes)):
        fig.delaxes(axes[j])

    fig.set_facecolor("aliceblue")
    plt.tight_layout()
    plt.show()


def plot_j_by_prt(core_data, mapped_fzi_params, ymax=10, log_log=False):
    """Generates a grid of J-Sw plots, one for each Reservoir Rock Type (RRT).

    This function creates subplots to visualize the Leverett J-function (J) versus
    normalized water saturation (SWN) for each rock type. It also overlays the
    best-fit J-curve using the provided parameters.

    Args:
        core_data (pd.DataFrame): DataFrame containing core data, including 'ROCK_FLAG',
            'SWN', and 'J'.
        mapped_fzi_params (list[dict]): A list of dictionaries, where each dictionary
            contains the 'a' and 'b' parameters for the J-curve for a specific
            rock type, identified by a 'ROCK_FLAG' key.
        ymax (int, optional): The maximum limit for the y-axis (J). Defaults to 10.
        log_log (bool, optional): If True, both axes will be on a logarithmic scale. Defaults to False.
    """
    core_data = core_data.copy()
    # Get unique rock flags
    unique_rock_flags = sorted(core_data["ROCK_FLAG"].unique())

    # Create subplots
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(20, 25))
    axes = axes.flatten()

    # Plot j_xplot for each rock flag
    for i, rock in enumerate(unique_rock_flags):
        ax = axes[i]
        data = core_data[core_data["ROCK_FLAG"] == rock]
        params = next(item for item in mapped_fzi_params if item["ROCK_FLAG"] == rock)
        a, b = params["a"], params["b"]
        ax = j_xplot(
            data["SWN"],
            data["J"],
            a=a,
            b=b,
            label=f"a:{a}\nb:{b}",
            ax=ax,
            ylim=(0.01 if log_log else 0, ymax),
            log_log=log_log,
        )
        ax.set_title(f"RRT {rock}")
        ax.legend()
        ax.grid(True)

    # Hide any unused subplots
    for j in range(len(unique_rock_flags), len(axes)):
        fig.delaxes(axes[j])

    fig.set_facecolor("aliceblue")
    plt.tight_layout()
    plt.show()
