import numpy as np
import pandas as pd

from quick_pp.rock_type import estimate_vsh_gr, rock_typing, find_cutoffs


def rhob_integral(rhob):
    """Calculate the integral of the bulk density log.

    Args:
        rhob (pd.Series): Bulk density log

    Returns:
        numpy.ndarray: Integral of the bulk density log in g/cm³
    """
    return np.cumsum(rhob.clip(2, 3).diff(15).cumsum())


def density_porosity(rhob):
    """Calculate the density porosity from the bulk density log.

    Args:
        rhob (pd.Series): Bulk density log

    Returns:
        pd.Series: Density porosity
    """
    rhob_min, rhob_max = 1.95, 2.95
    nphi_min_scale, nphi_max_scale = 0.45, -0.15
    return nphi_min_scale + (
        rhob - rhob_min) * (nphi_max_scale - nphi_min_scale) / (rhob_max - rhob_min)


def gas_xover(rhob, nphi):
    """Calculate the gas crossover from the bulk density and neutron porosity logs.
    Args:
        rhob (pd.Series): Bulk density log
        nphi (pd.Series): Neutron porosity log

    Returns:
        int: Gas crossover
    """
    return (nphi < density_porosity(rhob)).astype(float)


def log_perm(perm):
    """Calculate the log permeability from the permeability.

    Args:
        perm (pd.Series): Permeability

    Returns:
        pd.Series: Log permeability
    """
    return np.log10(perm.clip(lower=1e-3))


def rock_flag_gr(gr):
    vsh_gr = estimate_vsh_gr(gr)
    cutoffs = find_cutoffs(vsh_gr, 4)
    return rock_typing(vsh_gr, cut_offs=cutoffs, higher_is_better=False)


def coal_flagging(nphi, rhob, rhob_threshold=2.0, nphi_threshold=0.3, window_size=21, trend_factor=0.5):
    """Flag coal intervals based on high NPHI and low RHOB, considering log trends.

    Coal is typically characterized by very low bulk density and high
    apparent neutron porosity. This function combines absolute thresholds with
    a trend-based approach, flagging points where values deviate significantly
    from their rolling average.

    Args:
        nphi (pd.Series): Neutron porosity log (fraction).
        rhob (pd.Series): Bulk density log (g/cm³).
        rhob_threshold (float, optional): RHOB threshold for coal. Defaults to 1.95.
        nphi_threshold (float, optional): NPHI threshold for coal. Defaults to 0.45.
        window_size (int, optional): The size of the rolling window to calculate trends. Defaults to 21.
        trend_factor (float, optional): A factor to control sensitivity to trend deviation. Defaults to 0.5.

    Returns:
        pd.Series: A series of booleans, True where coal is flagged.
    """
    # Calculate rolling averages to establish local trends
    rhob_trend = rhob.rolling(window=window_size, center=True, min_periods=1).mean()
    nphi_trend = nphi.rolling(window=window_size, center=True, min_periods=1).mean()

    # Flag where RHOB is significantly below its trend and NPHI is significantly above its trend
    trend_condition = (rhob < rhob_trend * (1 - trend_factor)) & (nphi > nphi_trend * (1 + trend_factor))
    threshold_condition = (rhob < rhob_threshold) & (nphi > nphi_threshold)

    return (trend_condition & threshold_condition).astype(float)


def generate_fe_features(df):
    """Generate feature engineered features from the raw features.

    Args:
        df (pd.DataFrame): DataFrame containing the raw features.

    Returns:
        pd.DataFrame: DataFrame containing the engineered features.
    """
    df = df.copy()

    # Well based features
    for well_name, well_df in df.groupby('WELL_NAME'):
        well_df = well_df.sort_values('DEPTH')
        rhob_mask = well_df['RHOB'].notna()
        rhob_int_values = rhob_integral(well_df.loc[rhob_mask, 'RHOB'])
        df.loc[well_df.index, 'RHOB_INT'] = pd.Series(rhob_int_values, index=well_df.index)

        gr_mask = well_df['GR'].notna()
        rock_flag_values = rock_flag_gr(well_df.loc[gr_mask, 'GR'])
        df.loc[well_df.index, 'ROCK_FLAG'] = pd.Series(rock_flag_values, index=well_df.index)

    # Point based features
    df['DPHI'] = density_porosity(df['RHOB'])
    df['GAS_XOVER'] = gas_xover(df['RHOB'], df['DPHI'])
    if 'PERM' in df.columns and 'LOG_PERM' not in df.columns:
        df['LOG_PERM'] = log_perm(df['PERM'])
    return df
