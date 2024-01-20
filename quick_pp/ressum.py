import numpy as np
import pandas as pd
from scipy.stats import gmean


def calc_reservoir_summary(depth, vshale, phit, swt, perm, zones, cutoffs: list = [], uom: str = 'ft'):
    """_summary_

    Args:
        depth (_type_): _description_
        vshale (_type_): _description_
        phit (_type_): _description_
        swt (_type_): _description_
        zones (_type_): _description_
        cutoffs (list, optional): _description_. Defaults to [].
        uom (str, optional): _description_. Defaults to 'ft'.

    Returns:
        _type_: _description_
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


def flag_interval(vshale, phit, swt, cutoffs: list = []):
    """Flag interval based on cutoffs.

    Args:
        vshale (float or array_like): Vshale.
        phit (float or array_like): Total porosity.
        swt (float or array_like): Water saturation.
        cutoffs (list, optional): List of cutoffs. Defaults to [].

    Returns:
        float or array_like: Flagged interval.
    """
    assert len(cutoffs) == 3, 'cutoffs must be a list of 3 values: [vshale, phit, swt].'
    rock_flag = np.where(vshale < cutoffs[0], 1, 0)
    reservoir_flag = np.where(rock_flag == 1, np.where(phit > cutoffs[1], 1, 0), 0)
    pay_flag = np.where(reservoir_flag == 1, np.where(swt < cutoffs[2], 1, 0), 0)

    return rock_flag, reservoir_flag, pay_flag
