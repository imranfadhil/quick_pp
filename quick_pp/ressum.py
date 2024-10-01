import numpy as np
import pandas as pd
from scipy.stats import gmean
import random
from scipy.stats import truncnorm


def calc_reservoir_summary(depth, vshale, phit, swt, perm, zones,
                           cutoffs=dict(VSHALE=0.4, PHIT=0.01, SWT=0.9), uom: str = 'ft'):
    """Calculate reservoir summary based on cutoffs on vshale, phit, and swt.

    Args:
        depth (float): Depth either in MD or TVD.
        vshale (float): Volume of shale in fraction.
        phit (float): Total porosity in fraction.
        swt (float): Total water saturation in fraction
        zones (str): Zone names.
        cutoffs (dict, optional): {VSHALE: x, PHIT: y, SWT: z}. Defaults to dict(VSHALE=0.4, PHIT=0.01, SWT=0.9).
        uom (str, optional): Unit of measurement for depth. Defaults to 'ft'.

    Returns:
        pd.Dataframe: Reservoir summary in tabular format.
    """
    step = 0.1524 if uom == 'm' else 0.5
    df = pd.DataFrame({'depth': depth, 'vshale': vshale, 'phit': phit, 'swt': swt, 'perm': perm, 'zones': zones})
    df['rock_flag'], df['reservoir_flag'], df['pay_flag'] = flag_interval(df['vshale'], df['phit'], df['swt'], cutoffs)

    ressum_df = pd.DataFrame()
    for flag in ['rock', 'reservoir', 'pay']:
        temp_df = pd.DataFrame()
        # Calculate net thickness
        temp_df[["zones", "net"]] = df.groupby(["zones"])[[f"{flag}_flag"]].agg(
            lambda x: np.nansum(x) * step).reset_index()

        # Average the properties and merge
        flag_df = df[df[f"{flag}_flag"] == 1].copy()
        avg_mddf = flag_df.groupby(["zones"]).agg({
            "vshale": lambda x: np.nanmean(x),
            "phit": lambda x: np.nanmean(x),
            "swt": lambda x: np.nanmean(x),
            "perm": lambda x: gmean(x, nan_policy='omit')
        }).reset_index()
        temp_df = temp_df.merge(avg_mddf, on=["zones"], how="left", validate="1:1")

        # Calculate gross thickness and merge
        gross = df.groupby(["zones"])[["depth"]].agg(lambda x: np.nanmax(x) - np.nanmin(x)).reset_index().rename(
            columns={"depth": "gross"})
        temp_df = temp_df.merge(gross[["zones", 'gross']], on=["zones"], how="left", validate="1:1")

        # Set the minimum depth as top depth and merge
        top = df.groupby(["zones"])[["depth"]].agg(lambda x: np.nanmin(x)).reset_index().rename(
            columns={"depth": "top"})
        temp_df = temp_df.merge(top[["zones", 'top']], on=["zones"], how="left", validate="1:1")
        temp_df['flag'] = flag

        # Concat to ressum_df
        ressum_df = pd.concat([ressum_df, temp_df], ignore_index=True)

    ressum_df = ressum_df.round(3)
    ressum_df = ressum_df.sort_values(by=['top'], ignore_index=True)

    # Sort the columns
    cols = ['zones', 'flag', 'top', 'gross', 'net', 'vshale', 'phit', 'swt', 'perm']

    return ressum_df[cols]


def flag_interval(vshale, phit, swt, cutoffs: dict):
    """Flag interval based on cutoffs.

    Args:
        vshale (float): Vshale.
        phit (float): Total porosity.
        swt (float): Water saturation.
        cutoffs (list, optional): List of cutoffs. Defaults to [].

    Returns:
        float: Flagged interval.
    """
    assert len(cutoffs) == 3, 'cutoffs must be 3 key-value pairs: {VSHALE: x, PHIT: y, SWT: z}.'
    rock_flag = np.where(vshale < cutoffs['VSHALE'], 1, 0)
    reservoir_flag = np.where(rock_flag == 1, np.where(phit > cutoffs['PHIT'], 1, 0), 0)
    pay_flag = np.where(reservoir_flag == 1, np.where(swt < cutoffs['SWT'], 1, 0), 0)

    return rock_flag, reservoir_flag, pay_flag


def volumetric_method(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
    recovery_factor_bound: tuple,
    random_state=123
):
    """Calculate reserves using the volumetric method.

    Args:
        area_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for area.
        thickness_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for thickness.
        porosity_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for porosity.
        water_saturation_bound (tuple): (min, max, mode) - Triangular distribution parameters for water saturation.
        volume_factor_bound (tuple): (min, max) - Uniform distribution parameters for volume factor.
        recovery_factor_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for RF.
        random_state (int, optional): Random seed. Defaults to 123.

    Returns:
        _type_: _description_
    """
    random.seed(random_state)
    a = truncnorm.rvs(area_bound[0], area_bound[1], loc=area_bound[2], scale=area_bound[3],
                      random_state=random_state)
    h = truncnorm.rvs(thickness_bound[0], thickness_bound[1], loc=thickness_bound[2], scale=thickness_bound[3],
                      random_state=random_state)
    poro = truncnorm.rvs(porosity_bound[0], porosity_bound[1], loc=porosity_bound[2], scale=porosity_bound[3],
                         random_state=random_state)
    sw = random.triangular(water_saturation_bound[0], water_saturation_bound[1], mode=water_saturation_bound[2])
    bo = random.uniform(volume_factor_bound[0], volume_factor_bound[1])
    rf = truncnorm.rvs(
        recovery_factor_bound[0],
        recovery_factor_bound[1],
        loc=recovery_factor_bound[2],
        scale=recovery_factor_bound[3],
        random_state=random_state)
    return a * h * poro * (1 - sw) / bo * rf


def mc_volumetric_method(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
    recovery_factor_bound: tuple,
    n_try=10000, random_state=123
):
    """Monte Carlo simulation for volumetric method.

    Args:
        area_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for area.
        thickness_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for thickness.
        porosity_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for porosity.
        water_saturation_bound (tuple): (min, max, mode) - Triangular distribution parameters for water saturation.
        volume_factor_bound (tuple): (min, max) - Uniform distribution parameters for volume factor.
        recovery_factor_bound (tuple): (min, max, mean, std) - Truncated normal distribution parameters for RF.
        n_try (int, optional): Number of trials. Defaults to 10000.
        random_state (int, optional): Random seed. Defaults to 123.

    Returns:
        _type_: _description_
    """
    reserves = np.empty(0)
    for i in range(n_try):
        result = volumetric_method(
            area_bound=area_bound,
            thickness_bound=thickness_bound,
            porosity_bound=porosity_bound,
            water_saturation_bound=water_saturation_bound,
            volume_factor_bound=volume_factor_bound,
            recovery_factor_bound=recovery_factor_bound,
            random_state=random_state
        )
        reserves = np.append(reserves, result)
    return reserves
