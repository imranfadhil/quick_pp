import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import truncnorm
from tqdm import tqdm

from quick_pp import logger
from quick_pp.uncertainty import (
    calc_stochastic_reservoir_summary,
    mc_volumetric_uncertainty,
    run_log_uncertainty,
)
from quick_pp.utils import minmax_scale

__all__ = [
    "arith_upscale",
    "block_logs",
    "calc_reservoir_summary",
    "calc_stochastic_reservoir_summary",
    "calculate_volume",
    "cardwell_parsons_bounds",
    "cutoffs_analysis",
    "expand_blocked_logs",
    "flag_interval",
    "geom_upscale",
    "harm_upscale",
    "majority_upscale",
    "mc_volumetric_method",
    "mc_volumetric_uncertainty",
    "plot_upscaled_logs",
    "pore_volume_upscale",
    "power_law_upscale",
    "run_log_uncertainty",
    "sensitivity_analysis",
    "upscale_logs",
    "upscale_permeability",
    "volumetric_method",
]


def _weighted_zone_metrics(group: pd.DataFrame, omega: float = 0.33) -> pd.Series:
    """Compute thickness-weighted and pore-volume-weighted petrophysical averages for a zone.

    Follows PRMS/SEC reserves audit and Static Geomodeling Handover standards:
      - Vshale & Porosity: thickness-weighted average (sum(x * h) / sum(h))
      - Water Saturation: pore-volume weighted average (sum(phit * swt * h) / sum(phit * h))
      - BVHC: thickness-weighted hydrocarbon pore volume (sum(phit * (1 - swt) * h) / sum(h))
      - Permeability upscaling proxies (Cardwell-Parsons, 1945):
          * Arithmetic mean (k_h parallel flow proxy, omega=1): sum(k * h) / sum(h)
          * Geometric mean (k_eff heterogeneous flow proxy, omega->0): exp(sum(h * ln(k)) / sum(h)) for k > 0
          * Harmonic mean (k_v vertical/series flow proxy, omega=-1): sum(h) / sum(h / k) for k > 0
          * Power-law mean (Cardwell-Parsons, omega in [-1, 1]): (sum(h * k^omega) / sum(h))^(1/omega) for k > 0
          * Anisotropy ratio (k_v / k_h): av_perm_hm / av_perm_am
    """
    h = group["h"].to_numpy(dtype=float)
    vsh = group["vshale"].to_numpy(dtype=float)
    phi = group["phit"].to_numpy(dtype=float)
    sw = group["swt"].to_numpy(dtype=float)
    bvhc = group["bvhc"].to_numpy(dtype=float)
    perm = group["perm"].to_numpy(dtype=float)

    def _thick_mean(vals: np.ndarray) -> float:
        mask = np.isfinite(vals) & np.isfinite(h) & (h > 0)
        if not np.any(mask):
            return np.nan
        return float(np.sum(vals[mask] * h[mask]) / np.sum(h[mask]))

    av_vshale = _thick_mean(vsh)
    av_phit = _thick_mean(phi)
    av_bvhc = _thick_mean(bvhc)

    # Pore-volume weighted water saturation: sum(phi * sw * h) / sum(phi * h)
    pv_mask = np.isfinite(phi) & np.isfinite(sw) & np.isfinite(h) & (h > 0)
    if np.any(pv_mask):
        pv = phi[pv_mask] * h[pv_mask]
        pv_sum = float(np.sum(pv))
        if pv_sum > 0:
            av_swt = float(np.sum(pv * sw[pv_mask]) / pv_sum)
        else:
            av_swt = _thick_mean(sw)
    else:
        av_swt = np.nan

    # Permeability upscaling proxies (thickness-weighted)
    perm_valid = np.isfinite(perm) & np.isfinite(h) & (h > 0)
    if np.any(perm_valid):
        av_perm_am = float(np.sum(perm[perm_valid] * h[perm_valid]) / np.sum(h[perm_valid]))
        perm_pos = perm_valid & (perm > 0)
        if np.any(perm_pos):
            h_pos = h[perm_pos]
            k_pos = perm[perm_pos]
            h_pos_sum = float(np.sum(h_pos))
            av_perm_gm = float(np.exp(np.sum(h_pos * np.log(k_pos)) / h_pos_sum))
            av_perm_hm = float(h_pos_sum / np.sum(h_pos / k_pos))
            omega_val = float(omega)
            if abs(omega_val) < 1e-6:
                av_perm_pw = av_perm_gm
            elif np.isclose(omega_val, 1.0, atol=1e-6):
                av_perm_pw = float(np.sum(k_pos * h_pos) / h_pos_sum)
            elif np.isclose(omega_val, -1.0, atol=1e-6):
                av_perm_pw = av_perm_hm
            else:
                av_perm_pw = float(np.power(np.sum(h_pos * np.power(k_pos, omega_val)) / h_pos_sum, 1.0 / omega_val))
        else:
            av_perm_gm = 0.0
            av_perm_hm = 0.0
            av_perm_pw = 0.0
        kv_kh_ratio = float(av_perm_hm / av_perm_am) if av_perm_am > 0 else np.nan
    else:
        av_perm_am = np.nan
        av_perm_gm = np.nan
        av_perm_hm = np.nan
        av_perm_pw = np.nan
        kv_kh_ratio = np.nan

    return pd.Series(
        {
            "av_vshale": av_vshale,
            "av_phit": av_phit,
            "av_swt": av_swt,
            "av_bvhc": av_bvhc,
            "av_perm_am": av_perm_am,
            "av_perm_gm": av_perm_gm,
            "av_perm_hm": av_perm_hm,
            "av_perm_pw": av_perm_pw,
            "kv_kh_ratio": kv_kh_ratio,
        }
    )


def calc_reservoir_summary(depth, vshale, phit, swt, perm, zones, cutoffs=None, h=None, omega: float = 0.33):
    """Calculate reservoir summary based on cutoffs on vshale, phit, and swt.

    This function aggregates petrophysical properties over specified zones and flags
    (all, rock, reservoir, pay) using PRMS/SEC-compliant weighting:
      - Porosity (`av_phit`, `av_phit_pay`) and Vshale (`av_vshale`): thickness-weighted
      - Water saturation (`av_swt`, `av_sw_pay`): pore-volume weighted
        (sum(phit * swt * h) / sum(phit * h))
      - Permeability upscaling proxies (Cardwell-Parsons):
        * Arithmetic (`av_perm_am`, `av_perm_arith_pay`) — k_h parallel flow proxy
        * Geometric (`av_perm_gm`, `av_perm_geom_pay`) — effective heterogeneous flow proxy
        * Harmonic (`av_perm_hm`, `av_perm_harm_pay`) — k_v vertical series flow proxy
        * Power-Law (`av_perm_pw`, `av_perm_power_pay`) — Cardwell-Parsons power-law proxy with exponent `omega`
        * Anisotropy (`kv_kh_ratio`) — vertical-to-horizontal permeability ratio (`k_harm / k_arith`)

    Args:
        depth (np.ndarray or float): Depth, either MD or TVD.
        vshale (np.ndarray or float): Volume of shale in fraction.
        phit (np.ndarray or float): Total porosity in fraction.
        swt (np.ndarray or float): Total water saturation in fraction.
        perm (np.ndarray or float): Permeability.
        zones (np.ndarray or str): Zone names.
        cutoffs (dict, optional): A dictionary with 'VSHALE', 'PHIT', and 'SWT' cutoffs.
                                  Defaults to `dict(VSHALE=0.4, PHIT=0.01, SWT=0.9)`.
        h (np.ndarray or float, optional): Sample interval thickness. If None, inferred from `depth`.
        omega (float, optional): Cardwell-Parsons power-law exponent in `[-1, 1]`. Defaults to `0.33`.

    Returns:
        pd.DataFrame: A DataFrame containing the reservoir summary in tabular format.
    """
    logger.debug(
        f"Starting reservoir summary calculation with {len(depth)} data points"
    )
    cutoffs = cutoffs or {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9}

    df = pd.DataFrame(
        {
            "depth": depth,
            "vshale": vshale,
            "phit": phit,
            "swt": swt,
            "perm": perm,
            "zones": zones,
        }
    )
    diffs = df["depth"].diff()
    step = diffs.mean()
    if pd.isna(step) or step <= 0:
        pos_diffs = diffs.abs().dropna()
        pos_diffs = pos_diffs[pos_diffs > 0]
        step = float(pos_diffs.median()) if len(pos_diffs) > 0 else 1.0

    if h is not None:
        df["h"] = np.asarray(h, dtype=float)
    elif (diffs.dropna() > 0).all() and not np.allclose(diffs.dropna(), step):
        df["h"] = diffs.bfill().fillna(step).abs()
    else:
        df["h"] = float(step)

    df["bvhc"] = df["phit"] * (1 - df["swt"])
    df["rock_flag"], df["reservoir_flag"], df["pay_flag"] = flag_interval(
        df["vshale"], df["phit"], df["swt"], cutoffs
    )
    df["all_flag"] = 1

    logger.debug(
        f"Data flags calculated - Rock: {df['rock_flag'].sum()}, "
        f"Reservoir: {df['reservoir_flag'].sum()}, Pay: {df['pay_flag'].sum()}"
    )

    avg_cols = [
        "av_vshale",
        "av_phit",
        "av_swt",
        "av_bvhc",
        "av_perm_am",
        "av_perm_gm",
        "av_perm_hm",
        "av_perm_pw",
        "kv_kh_ratio",
    ]

    ressum_df = pd.DataFrame()
    pay_zone_lookup = {}

    for flag in ["all", "rock", "reservoir", "pay"]:
        temp_df = pd.DataFrame()
        # Calculate net thickness from per-sample thickness h
        df["_net_h"] = np.where(df[f"{flag}_flag"] == 1, df["h"], 0.0)
        temp_df[["zones", "net"]] = (
            df.groupby(["zones"])[["_net_h"]]
            .agg(lambda x: np.nansum(x))
            .reset_index()
            .rename(columns={"_net_h": "net"})
        )

        # Average the properties using thickness / pore-volume weighting and merge
        flag_df = df[df[f"{flag}_flag"] == 1].copy()
        if not flag_df.empty:
            avg_records = []
            for zone_name, zgroup in flag_df.groupby("zones"):
                metrics = _weighted_zone_metrics(zgroup, omega=omega)
                metrics["zones"] = zone_name
                avg_records.append(metrics)
            avg_df = pd.DataFrame(avg_records)
        else:
            avg_df = pd.DataFrame(columns=["zones"] + avg_cols)

        if flag == "pay" and not avg_df.empty:
            for _, row in avg_df.iterrows():
                pay_zone_lookup[row["zones"]] = {
                    "av_phit_pay": row["av_phit"],
                    "av_sw_pay": row["av_swt"],
                    "av_perm_arith_pay": row["av_perm_am"],
                    "av_perm_geom_pay": row["av_perm_gm"],
                    "av_perm_harm_pay": row["av_perm_hm"],
                    "av_perm_power_pay": row["av_perm_pw"],
                }

        temp_df = temp_df.merge(avg_df, on=["zones"], how="left", validate="1:1")

        # Set the maximum depth as bottom depth
        bottom = (
            df.groupby(["zones"])[["depth"]]
            .agg(lambda x: np.nanmax(x))
            .reset_index()
            .rename(columns={"depth": "bottom"})
        )
        temp_df = temp_df.merge(
            bottom[["zones", "bottom"]], on=["zones"], how="left", validate="1:1"
        )

        # Set the minimum depth as top depth
        top = (
            df.groupby(["zones"])[["depth"]]
            .agg(lambda x: np.nanmin(x))
            .reset_index()
            .rename(columns={"depth": "top"})
        )
        temp_df = temp_df.merge(
            top[["zones", "top"]], on=["zones"], how="left", validate="1:1"
        )
        temp_df["flag"] = flag

        # Calculate gross thickness
        temp_df["gross"] = temp_df["bottom"] - temp_df["top"]

        # Calculate net-to-gross ratio
        temp_df["ntg"] = np.where(
            temp_df["gross"] > 0, temp_df["net"] / temp_df["gross"], np.nan
        )

        # Concat to ressum_df
        ressum_df = pd.concat([ressum_df, temp_df], ignore_index=True)

    # Attach zone-level pay averages (av_phit_pay, av_sw_pay, av_perm_*_pay)
    pay_cols = [
        "av_phit_pay",
        "av_sw_pay",
        "av_perm_arith_pay",
        "av_perm_geom_pay",
        "av_perm_harm_pay",
        "av_perm_power_pay",
    ]
    for col in pay_cols:
        ressum_df[col] = ressum_df["zones"].map(
            lambda z, c=col: pay_zone_lookup.get(z, {}).get(c, np.nan)
        )
        # For valid non-pay flag rows in zones without pay, default to 0.0 so row-wise dropna preserves them
        valid_non_pay = ressum_df["av_phit"].notna() & (ressum_df["flag"] != "pay") & ressum_df[col].isna()
        ressum_df.loc[valid_non_pay, col] = 0.0
    valid_non_pay_kvkh = ressum_df["av_phit"].notna() & ressum_df["kv_kh_ratio"].isna()
    ressum_df.loc[valid_non_pay_kvkh, "kv_kh_ratio"] = 0.0

    ressum_df = ressum_df.round(3)
    ressum_df = ressum_df.sort_values(by=["top"], ignore_index=True)

    # Sort the columns
    cols = [
        "zones",
        "flag",
        "top",
        "bottom",
        "gross",
        "net",
        "ntg",
        "av_vshale",
        "av_phit",
        "av_swt",
        "av_bvhc",
        "av_perm_am",
        "av_perm_gm",
        "av_perm_hm",
        "av_perm_pw",
        "kv_kh_ratio",
        "av_phit_pay",
        "av_sw_pay",
        "av_perm_arith_pay",
        "av_perm_geom_pay",
        "av_perm_harm_pay",
        "av_perm_power_pay",
    ]

    logger.debug(
        f"Reservoir summary calculation completed with {len(ressum_df)} summary rows"
    )
    return ressum_df[cols]


def flag_interval(vshale, phit, swt, cutoffs: dict):
    """Flag interval based on cutoffs.

    Args:
        vshale (np.ndarray or float): Vshale values.
        phit (np.ndarray or float): Total porosity values.
        swt (np.ndarray or float): Water saturation values.
        cutoffs (dict): A dictionary with 'VSHALE', 'PHIT', and 'SWT' cutoffs.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the rock, reservoir, and pay flags.
    """
    if len(cutoffs) != 3:
        logger.error(
            f"Invalid cutoffs format: expected 3 key-value pairs, got {len(cutoffs)}"
        )
        raise AssertionError(
            "cutoffs must be 3 key-value pairs: {VSHALE: x, PHIT: y, SWT: z}."
        )

    logger.debug(
        f"Flagging intervals with cutoffs: VSHALE={cutoffs.get('VSHALE')}, "
        f"PHIT={cutoffs.get('PHIT')}, SWT={cutoffs.get('SWT')}"
    )

    rock_flag = np.where(vshale < cutoffs["VSHALE"], 1, 0)
    reservoir_flag = np.where(rock_flag == 1, np.where(phit > cutoffs["PHIT"], 1, 0), 0)
    pay_flag = np.where(reservoir_flag == 1, np.where(swt < cutoffs["SWT"], 1, 0), 0)

    return rock_flag, reservoir_flag, pay_flag


def volumetric_method(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
    recovery_factor_bound: tuple = None,
    random_state=None,
):
    """Generate random samples for volumetric parameters based on specified distributions.

    This function uses truncated normal distributions for area, thickness, porosity,
    water saturation, and recovery factor, and a uniform distribution for the
    volume factor.

    Args:
        area_bound (tuple): (min, max, mean, std) for area in acres.
        thickness_bound (tuple): (min, max, mean, std) for thickness in feet.
        porosity_bound (tuple): (min, max, mean, std) for porosity in fraction.
        water_saturation_bound (tuple): (min, max, mean, std) for water saturation in fraction.
        volume_factor_bound (tuple): (min, max) for the volume factor (unitless).
        recovery_factor_bound (tuple, optional): (min, max, mean, std) for the recovery factor. Defaults to None.
        random_state (int, optional): A seed for the random number generator. Defaults to None.

    Returns:
        tuple: A tuple of the generated random samples for (area, thickness, porosity, sw, bo, rf).
    """
    logger.debug(
        f"Starting volumetric method calculation with random_state={random_state}"
    )

    random.seed(random_state)
    a_min_transformed = (area_bound[0] - area_bound[2]) / area_bound[3]
    a_max_transformed = (area_bound[1] - area_bound[2]) / area_bound[3]
    a = truncnorm.rvs(
        a_min_transformed,
        a_max_transformed,
        loc=area_bound[2],
        scale=area_bound[3],
        random_state=random_state,
    )

    h_min_transformed = (thickness_bound[0] - thickness_bound[2]) / thickness_bound[3]
    h_max_transformed = (thickness_bound[1] - thickness_bound[2]) / thickness_bound[3]
    h = truncnorm.rvs(
        h_min_transformed,
        h_max_transformed,
        loc=thickness_bound[2],
        scale=thickness_bound[3],
        random_state=random_state,
    )

    poro_min_transformed = (porosity_bound[0] - porosity_bound[2]) / porosity_bound[3]
    poro_max_transformed = (porosity_bound[1] - porosity_bound[2]) / porosity_bound[3]
    poro = truncnorm.rvs(
        poro_min_transformed,
        poro_max_transformed,
        loc=porosity_bound[2],
        scale=porosity_bound[3],
        random_state=random_state,
    )
    sw_min_transformed = (
        water_saturation_bound[0] - water_saturation_bound[2]
    ) / water_saturation_bound[3]
    sw_max_transformed = (
        water_saturation_bound[1] - water_saturation_bound[2]
    ) / water_saturation_bound[3]
    sw = truncnorm.rvs(
        sw_min_transformed,
        sw_max_transformed,
        loc=water_saturation_bound[2],
        scale=water_saturation_bound[3],
        random_state=random_state,
    )

    bo = random.uniform(volume_factor_bound[0], volume_factor_bound[1])

    if recovery_factor_bound is not None:
        rf_min_transformed = (
            recovery_factor_bound[0] - recovery_factor_bound[2]
        ) / recovery_factor_bound[3]
        rf_max_transformed = (
            recovery_factor_bound[1] - recovery_factor_bound[2]
        ) / recovery_factor_bound[3]
        rf = truncnorm.rvs(
            rf_min_transformed,
            rf_max_transformed,
            loc=recovery_factor_bound[2],
            scale=recovery_factor_bound[3],
            random_state=random_state,
        )
    else:
        rf = 1.0

    logger.debug(
        f"Volumetric parameters generated - Area: {a:.2f}, Thickness: {h:.2f}, "
        f"Porosity: {poro:.3f}, Sw: {sw:.3f}, Bo: {bo:.3f}, RF: {rf:.3f}"
    )
    return a, h, poro, sw, bo, rf


def mc_volumetric_method(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
    n_try=10000,
    percentile=None,
    seed: int = 42,
    plot: bool = True,
):
    """Monte Carlo simulation for volumetric method using Latin Hypercube Sampling.
    This function delegates to `mc_volumetric_uncertainty` in `quick_pp.uncertainty`
    to estimate the distribution of in-place hydrocarbon volumes and generates diagnostic plots.

    Args:
        area_bound (tuple): (min, max, mean, std) for area in acres.
        thickness_bound (tuple): (min, max, mean, std) for thickness in feet.
        porosity_bound (tuple): (min, max, mean, std) for porosity in fraction.
        water_saturation_bound (tuple): (min, max, mean, std) for water saturation in fraction.
        volume_factor_bound (tuple): (min, max) for the volume factor (unitless).
        n_try (int, optional): Number of trials. Defaults to 10000.
        percentile (list, optional): Percentiles to calculate and display. Defaults to [10, 50, 90].
        seed (int, optional): Random seed for LatinHypercube sampler. Defaults to 42.
        plot (bool, optional): Whether to render RainCloud and histogram plots. Defaults to True.

    Returns:
        tuple: A tuple containing arrays of `(volumes, area, thickness, porosity, water_saturation, volume_factor)`.
    """
    logger.info(f"Starting LHS Monte Carlo simulation with {n_try} trials")

    percentile = percentile or [10, 50, 90]
    res = mc_volumetric_uncertainty(
        area_bound=area_bound,
        thickness_bound=thickness_bound,
        porosity_bound=porosity_bound,
        water_saturation_bound=water_saturation_bound,
        volume_factor_bound=volume_factor_bound,
        n_try=n_try,
        percentile=percentile,
        seed=seed,
        plot=False,
    )

    volumes = res["volumes"]
    area = res["area"]
    thickness = res["thickness"]
    porosity = res["porosity"]
    water_saturation = res["water_saturation"]
    volume_factor = res["volume_factor"]

    if plot:
        import ptitprince as pt

        data = pd.DataFrame(
            {
                "Area": area,
                "Thickness": thickness,
                "Porosity": porosity,
                "Water Saturation": water_saturation,
                "Volume Factor": volume_factor,
                "Volume": volumes,
            }
        )
        fig, axs = plt.subplots(5, 1, figsize=(7, 11), constrained_layout=True)
        pt.RainCloud(
            data=data,
            y="Area",
            ax=axs[0],
            orient="h",
            bw=0.1,
            width_viol=0.5,
            alpha=0.6,
            dodge=True,
        )
        axs[0].set_ylabel("Area (acre)")
        pt.RainCloud(
            data=data,
            y="Thickness",
            ax=axs[1],
            orient="h",
            bw=0.1,
            width_viol=0.5,
            alpha=0.6,
            dodge=True,
        )
        axs[1].set_ylabel("Thickness (ft)")
        pt.RainCloud(
            data=data,
            y="Porosity",
            ax=axs[2],
            orient="h",
            bw=0.1,
            width_viol=0.5,
            alpha=0.6,
            dodge=True,
        )
        axs[2].set_ylabel("Porosity (frac)")
        pt.RainCloud(
            data=data,
            y="Water Saturation",
            ax=axs[3],
            orient="h",
            bw=0.1,
            width_viol=0.5,
            alpha=0.6,
            dodge=True,
        )
        axs[3].set_ylabel("Water Saturation (frac)")
        pt.RainCloud(
            data=data,
            y="Volume Factor",
            ax=axs[4],
            orient="h",
            bw=0.1,
            width_viol=0.5,
            alpha=0.6,
            dodge=True,
        )
        axs[4].set_ylabel("Volume Factor")
        fig.set_facecolor("aliceblue")
        fig.show()

        plt.figure(figsize=(12, 6))
        a = np.hstack(volumes.tolist())
        _ = plt.hist(a, density=True, bins="auto")
        plt.title(f"Volume Estimation ({n_try} samples)")
        plt.xlabel("Volume (MM bbl)")
        for i, pct in enumerate(np.percentile(volumes, percentile)):
            plt.axvline(
                x=pct,
                color="r",
                linestyle="dashed",
                label=f"P{percentile[i]}: {round(pct)} MM bbl",
            )
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    return volumes, area, thickness, porosity, water_saturation, volume_factor


def calculate_volume(
    x, area, thickness, porosity, water_saturation, volume_factor, recovery_factor=1
):
    """Calculate volume using the volumetric method.

    This function is a helper for sensitivity analysis, applying a perturbation `x`
    to the calculation.

    Args:
        x (float): A perturbation factor.
        area (float): Area in acres.
        thickness (float): Thickness in feet.
        porosity (float): Porosity in fraction.
        water_saturation (float): Water saturation in fraction.
        volume_factor (float): Volume factor.
        recovery_factor (float, optional): Recovery factor in fraction. Defaults to 1.

    Returns:
        float: Estimated volume in million barrels (MMbbl).
    """
    return (
        x
        * (43560 * 0.1781)
        * area
        * thickness
        * porosity
        * (1 - water_saturation)
        / (volume_factor)
        * recovery_factor
        * 1e-6
    )


def sensitivity_analysis(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
):
    """Sensitivity analysis for volumetric method.
    This function performs a Sobol sensitivity analysis to determine the influence
    of each input parameter on the final volume calculation and displays the
    results as tornado charts (both volume-based and sensitivity indices).

    Args:
        area_bound (tuple): (min, max) for area.
        thickness_bound (tuple): (min, max) for thickness.
        porosity_bound (tuple): (min, max) for porosity.
        water_saturation_bound (tuple): (min, max) for water saturation.
        volume_factor_bound (tuple): (min, max) for the volume factor.
    """
    from SALib.analyze.sobol import analyze
    from SALib.sample.saltelli import sample

    logger.info("Starting sensitivity analysis for volumetric method")

    # Define the model inputs
    problem = {
        "num_vars": 5,
        "names": ["area", "thickness", "porosity", "water_saturation", "volume_factor"],
        "bounds": [
            [area_bound[0], area_bound[1]],
            [thickness_bound[0], thickness_bound[1]],
            [porosity_bound[0], porosity_bound[1]],
            [water_saturation_bound[0], water_saturation_bound[1]],
            [volume_factor_bound[0], volume_factor_bound[1]],
        ],
    }

    logger.debug(
        f"Problem defined with {problem['num_vars']} variables and {2**10} samples"
    )

    # Generate samples
    param_values = sample(problem, 2**10, calc_second_order=False)

    # Evaluate model at each sample point
    x = np.linspace(-1, 1, 100)
    y = np.array([calculate_volume(x, *params) for params in param_values])

    # Perform Sobol analysis
    Si = analyze(problem, y[:, 0], calc_second_order=False)

    # Extract the sensitivity indices
    S1 = Si["S1"]
    S1_conf = Si["S1_conf"]
    names = problem["names"]

    logger.info(
        f"Sensitivity analysis completed. Top parameter: {names[np.argmax(S1)]} "
        f"(S1={S1[np.argmax(S1)]:.3f})"
    )

    # ===== VOLUME-BASED TORNADO CHART (PRIMARY - MORE INTUITIVE) =====
    # Calculate baseline volume at mean parameter values
    baseline_params = {
        name: (bound[0] + bound[1]) / 2
        for name, bound in zip(problem["names"], problem["bounds"], strict=True)
    }
    baseline_volume = calculate_volume(1, **baseline_params)

    # Calculate volume impacts for each parameter
    volume_impacts = []
    for param_name, (min_val, max_val) in zip(
        problem["names"], problem["bounds"], strict=True
    ):
        params_at_min = baseline_params.copy()
        params_at_min[param_name] = min_val
        vol_min = calculate_volume(1, **params_at_min)

        params_at_max = baseline_params.copy()
        params_at_max[param_name] = max_val
        vol_max = calculate_volume(1, **params_at_max)

        volume_impacts.append(
            {
                "name": param_name,
                "vol_min": vol_min,
                "vol_max": vol_max,
                "low_impact": vol_min - baseline_volume,
                "high_impact": vol_max - baseline_volume,
                "range": abs(vol_max - vol_min),
            }
        )

    # Sort by impact range for visualization
    volume_impacts_sorted = sorted(volume_impacts, key=lambda x: x["range"])

    # Create volume-based tornado chart
    plt.figure(figsize=(12, 7))
    y_pos = np.arange(len(volume_impacts_sorted))
    param_names_sorted = [imp["name"] for imp in volume_impacts_sorted]
    low_impacts = np.array([imp["low_impact"] for imp in volume_impacts_sorted])
    high_impacts = np.array([imp["high_impact"] for imp in volume_impacts_sorted])

    plt.barh(
        y_pos,
        low_impacts,
        height=0.6,
        align="center",
        alpha=0.8,
        color="steelblue",
        edgecolor="black",
        linewidth=1.5,
        label="At Minimum",
    )
    plt.barh(
        y_pos,
        high_impacts,
        height=0.6,
        align="center",
        alpha=0.8,
        color="steelblue",
        edgecolor="black",
        linewidth=1.5,
        label="At Maximum",
    )

    # Add baseline reference line
    plt.axvline(x=0, color="red", linestyle="--", linewidth=2.5, alpha=0.8)
    plt.text(
        0,
        len(volume_impacts_sorted) - 0.5,
        f"Baseline: {baseline_volume:.1f} MMbbl",
        ha="center",
        fontsize=10,
        fontweight="bold",
        color="red",
        bbox={"boxstyle": "round", "facecolor": "yellow", "alpha": 0.3},
    )

    # Add value labels on bars
    for i, (low, high) in enumerate(zip(low_impacts, high_impacts, strict=True)):
        if low < 0:
            plt.text(
                low - 5,
                i,
                f"{low:.0f}",
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=9,
            )
        else:
            plt.text(
                low + 5,
                i,
                f"{low:.0f}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

        if high > 0:
            plt.text(
                high + 5,
                i,
                f"{high:+.0f}",
                ha="left",
                va="center",
                fontweight="bold",
                fontsize=9,
            )
        else:
            plt.text(
                high - 5,
                i,
                f"{high:.0f}",
                ha="right",
                va="center",
                fontweight="bold",
                fontsize=9,
            )

    plt.yticks(y_pos, [p.replace("_", " ").title() for p in param_names_sorted])
    plt.xlabel("Volume Change from Baseline (MMbbl)", fontsize=12, fontweight="bold")
    plt.title(
        "Tornado Chart - STOIIP Volume Impacts\n(Practical View: Shows Actual Volume Ranges)",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, axis="x")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.show()

    # ===== SENSITIVITY INDEX TORNADO CHART (SECONDARY - TECHNICAL REFERENCE) =====
    # Sort by sensitivity index for comparison
    sorted_indices = np.argsort(S1)
    sorted_S1 = S1[sorted_indices]
    sorted_S1_conf = S1_conf[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]

    plt.figure(figsize=(10, 6))
    plt.barh(
        range(len(sorted_S1)),
        sorted_S1,
        xerr=sorted_S1_conf,
        align="center",
        alpha=0.7,
        color="darkorange",
        ecolor="black",
        capsize=5,
    )
    plt.yticks(
        range(len(sorted_S1)), [n.replace("_", " ").title() for n in sorted_names]
    )
    plt.xlabel("First-order Sensitivity Index (S1)", fontsize=12, fontweight="bold")
    plt.title(
        "Tornado Chart - Sensitivity Indices\n(Technical Reference: Normalized Importance)",
        fontsize=13,
        fontweight="bold",
    )
    plt.grid(True, alpha=0.3, axis="x")
    plt.tight_layout()
    plt.show()


def cutoffs_analysis(df, percentile=95):
    """Perform sensitivity analysis on Vshale, PHIT and SWT cutoffs against hydrocarbon pore volume.

    Args:
        df (pd.DataFrame): A DataFrame containing 'VSHALE', 'PHIT', and 'SWT' columns.
        percentile (int, optional): The percentile to use for determining the optimal cutoff point
                                    from the cumulative HCPV curve. Defaults to 95.

    Returns:
        tuple[float, float, float]: A tuple containing the optimal cutoffs for VSHALE, PHIT, and SWT.
    """
    logger.debug("Starting cutoffs sensitivity analysis")

    # Calculate distances from each point to line between endpoints
    def find_percentile_point(x, y, percentile):
        # Sort values by y
        sorted_indices = y.sort_values().index
        sorted_y = y.iloc[sorted_indices].round(3)
        sorted_x = x.iloc[sorted_indices].round(3)
        # Remove duplicates and find index closest to specified percentile
        unique_y = sorted_y.drop_duplicates()
        target_value = np.percentile(unique_y, percentile)
        idx = (np.abs(sorted_y - target_value)).argmin()
        # Return x value at percentile
        return sorted_x.iloc[idx]

    # Define parameter ranges
    vsh_range = np.linspace(1.0, 0.0, 100)
    phit_range = np.linspace(0.0, 1.0, 100)
    swt_range = np.linspace(1.0, 0.0, 100)

    # Calculate total hydrocarbon pore volume
    df["HCPV"] = df["PHIT"] * (1 - df["SWT"])

    # Sensitivity analysis for VSHALE
    results = []
    phi = 0
    sw = 1
    for vsh in tqdm(vsh_range, desc="Analyzing cutoffs"):
        cutoffs = {"VSHALE": vsh, "PHIT": phi, "SWT": sw}
        rock_flag, _, _ = flag_interval(df["VSHALE"], df["PHIT"], df["SWT"], cutoffs)
        # Calculate total hydrocarbon pore volume
        hcpv = np.sum(df.loc[rock_flag == 1, "HCPV"])
        results.append(
            {
                "flag": "rock",
                "vshale_cut": vsh,
                "phit_cut": phi,
                "swt_cut": sw,
                "HCPV": hcpv,
            }
        )
    temp_df = pd.DataFrame(results).reset_index(drop=True)
    temp_df["HCPV"] = minmax_scale(temp_df["HCPV"])
    optimal_vsh_cut = find_percentile_point(
        temp_df["vshale_cut"], temp_df["HCPV"], percentile
    )

    # Sensitivity analysis for PHIT
    vsh = optimal_vsh_cut
    sw = 1
    for phi in phit_range:
        cutoffs = {"VSHALE": vsh, "PHIT": phi, "SWT": sw}
        _, res_flag, _ = flag_interval(df["VSHALE"], df["PHIT"], df["SWT"], cutoffs)
        # Calculate total hydrocarbon pore volume
        hcpv = np.sum(df.loc[res_flag == 1, "HCPV"])
        results.append(
            {
                "flag": "res",
                "vshale_cut": vsh,
                "phit_cut": phi,
                "swt_cut": sw,
                "HCPV": hcpv,
            }
        )
    temp_df = pd.DataFrame(results)
    temp_df = temp_df[temp_df["flag"] == "res"].reset_index(drop=True)
    temp_df["HCPV"] = minmax_scale(temp_df["HCPV"])
    optimal_phit_cut = find_percentile_point(
        temp_df["phit_cut"], temp_df["HCPV"], percentile
    )

    # Sensitivity analysis for SWT
    vsh = optimal_vsh_cut
    phi = optimal_phit_cut
    for sw in swt_range:
        cutoffs = {"VSHALE": vsh, "PHIT": phi, "SWT": sw}
        _, _, pay_flag = flag_interval(df["VSHALE"], df["PHIT"], df["SWT"], cutoffs)
        # Calculate total hydrocarbon pore volume
        hcpv = np.sum(df.loc[pay_flag == 1, "HCPV"])
        results.append(
            {
                "flag": "pay",
                "vshale_cut": vsh,
                "phit_cut": phi,
                "swt_cut": sw,
                "HCPV": hcpv,
            }
        )

    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    temp_df = results_df[results_df["flag"] == "pay"].reset_index(drop=True)
    temp_df["HCPV"] = minmax_scale(temp_df["HCPV"])
    optimal_sw_cut = find_percentile_point(
        temp_df["swt_cut"], temp_df["HCPV"], percentile
    )

    logger.info(
        f"Optimal cutoffs found - Vshale: {optimal_vsh_cut:.3f}, "
        f"PHIT: {optimal_phit_cut:.3f}, SWT: {optimal_sw_cut:.3f}"
    )

    # Plot sensitivity crossplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, (flag, data) in zip(axes, results_df.groupby("flag"), strict=True):
        param = (
            "vshale_cut"
            if flag == "rock"
            else "phit_cut"
            if flag == "res"
            else "swt_cut"
        )
        data["HCPV"] = minmax_scale(data["HCPV"])
        ax.plot(data[param], data["HCPV"], "b-", label="HCPV")
        cut_off = (
            optimal_vsh_cut
            if flag == "rock"
            else optimal_phit_cut
            if flag == "res"
            else optimal_sw_cut
        )
        ax.axvline(
            x=cut_off,
            color="r",
            linestyle="--",
            label=f"Optimal Cutoff for {param}={round(cut_off, 3)}",
        )
        ax.set_xlabel(param)
        ax.set_ylabel("HCPV")
        ax.set_title("HCPV vs " + param + " with Optimal Cutoff Point")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()

    return optimal_vsh_cut, optimal_phit_cut, optimal_sw_cut


from quick_pp.upscaling import (  # noqa: E402
    arith_upscale,
    block_logs,
    cardwell_parsons_bounds,
    expand_blocked_logs,
    geom_upscale,
    harm_upscale,
    majority_upscale,
    plot_upscaled_logs,
    pore_volume_upscale,
    power_law_upscale,
    upscale_logs,
    upscale_permeability,
)

