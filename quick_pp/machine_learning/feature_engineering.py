import numpy as np
import pandas as pd


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
        mask = well_df['RHOB'].notna()
        rhob_int_values = rhob_integral(well_df.loc[mask, 'RHOB'])
        rhob_int_series = pd.Series(rhob_int_values, index=well_df.index)
        df.loc[well_df.index, 'RHOB_INT'] = rhob_int_series

    # Point based features
    df['DPHI'] = density_porosity(df['RHOB'])
    df['GAS_XOVER'] = gas_xover(df['RHOB'], df['DPHI'])
    if 'PERM' in df.columns and 'LOG_PERM' not in df.columns:
        df['LOG_PERM'] = log_perm(df['PERM'])
    return df
