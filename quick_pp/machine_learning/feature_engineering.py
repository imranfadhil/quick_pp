import numpy as np
import pandas as pd

from quick_pp.rock_type import estimate_vsh_gr, rock_typing, find_cutoffs
from tqdm import tqdm


def rhob_integral(rhob, step=15):
    """Calculate the integral of the bulk density log.

    Args:
        rhob (pd.Series): Bulk density log

    Returns:
        numpy.ndarray: Integral of the bulk density log in g/cm³
    """
    return np.cumsum(rhob.clip(2, 3).diff(step).cumsum())


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
    # get the IQRs as the min and max value
    q1 = gr.quantile(0.25)
    q3 = gr.quantile(0.75)
    iqr = q3 - q1
    min_gr = q1 - 1.5 * iqr
    max_gr = q3 + 1.5 * iqr

    vsh_gr = estimate_vsh_gr(gr, min_gr, max_gr)
    cutoffs = find_cutoffs(vsh_gr, 4)
    return rock_typing(vsh_gr, cut_offs=cutoffs, higher_is_better=False)


def coal_flagging(nphi, rhob, rhob_threshold=2.0, nphi_threshold=0.3, window_size=21, trend_factor=0.1):
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


def tight_streak_flagging(rhob, rhob_threshold=2.3, window_size=15, trend_factor=0.03):
    """Flag tight streak intervals based on high RHOB, high RT, and low NPHI.

    Tight streaks (e.g., carbonate cemented layers) are characterized by high
    bulk density, high resistivity, and low porosity. This function flags
    these zones by identifying points where log values deviate significantly
    from their local trend, in addition to crossing absolute thresholds.

    Args:
        rhob (pd.Series): Bulk density log (g/cm³).
        rhob_threshold (float, optional): RHOB threshold for tight streak. Defaults to 2.5.
        window_size (int, optional): The size of the rolling window for trend calculation. Defaults to 21.
        trend_factor (float, optional): A factor to control sensitivity to trend deviation.
                          Defaults to 0.05 (5%).

    Returns:
        pd.Series: A series of floats (0.0 or 1.0), 1.0 where a tight streak is flagged.
    """
    # Calculate rolling averages to establish local trends
    rhob_trend = rhob.rolling(window=window_size, center=True, min_periods=1).mean()

    # Flag where RHOB are significantly above trend
    trend_condition = (rhob > rhob_trend * (1 + trend_factor)) 

    # Flag where values cross absolute thresholds
    threshold_condition = (rhob > rhob_threshold)

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
    for well_name, well_df in tqdm(df.groupby('WELL_NAME'), desc="Generating well-based features"):
        tqdm.write(f'Processing well {well_name}')
        well_df = well_df.sort_values('DEPTH').copy()

        df.loc[well_df.index, 'TIGHT_FLAG'] = tight_streak_flagging(well_df['RHOB'], well_df['RT'])

        df.loc[well_df.index, 'COAL_FLAG'] = coal_flagging(well_df['NPHI'], well_df['RHOB'])

        rhob_mask = well_df['RHOB'].notna()
        step = np.ceil(well_df['DEPTH'].diff().mean())
        rhob_int_values = rhob_integral(well_df.loc[rhob_mask, 'RHOB'], step=step)
        df.loc[well_df.index, 'RHOB_INT'] = pd.Series(rhob_int_values, index=well_df.index)

        gr_mask = well_df['GR'].notna()
        rock_flag_values = rock_flag_gr(well_df.loc[gr_mask, 'GR'])
        df.loc[well_df.index, 'ROCK_FLAG'] = pd.Series(rock_flag_values, index=well_df.index)

    # Point based features
    df['DPHI'] = density_porosity(df['RHOB'])
    df['GAS_XOVER'] = gas_xover(df['RHOB'], df['NPHI'])
    if 'PERM' in df.columns and 'LOG_PERM' not in df.columns:
        df['LOG_PERM'] = log_perm(df['PERM'])
    return df
