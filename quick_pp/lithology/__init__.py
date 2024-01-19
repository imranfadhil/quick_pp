from .sand_silt_clay import sand_silt_clay_model
from .multi_mineral import multi_mineral_model
from .sand_shale import sand_shale_model
from .shale import *

import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_lithology(df: pd.DataFrame, input_features: list):
    """Normalize lithology curves for better visualization

    Args:
        df (pd.DataFrame): _description_

    Returns:
        pd.DataFrame: _description_
    """
    scaler = MinMaxScaler()
    transposed = df[input_features].T
    return_df = pd.DataFrame(scaler.fit_transform(transposed).T, columns=input_features)
    df[input_features] = return_df[input_features]

    return df