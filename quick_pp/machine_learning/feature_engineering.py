import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

from quick_pp.rock_type import estimate_vsh_gr, rock_typing, find_cutoffs
from tqdm import tqdm


def rhob_integral(rhob, step=15):
    """Calculate a pseudo-integral of the bulk density log.

    Args:
        rhob (pd.Series): Bulk density log [g/cm³].
        step (int, optional): The step for calculating the difference before integration. Defaults to 15.

    Returns:
        np.ndarray: An array representing the cumulative sum of the stepped difference of RHOB.
    """
    return np.cumsum(rhob.clip(2, 3).diff(step).cumsum())


def density_porosity(rhob):
    """Calculate the density porosity from the bulk density log.

    Args:
        rhob (pd.Series): Bulk density log [g/cm³].

    Returns:
        pd.Series: Density porosity, scaled to a typical neutron porosity range.
    """
    rhob_min, rhob_max = 1.95, 2.95
    nphi_min_scale, nphi_max_scale = 0.45, -0.15
    return nphi_min_scale + (rhob - rhob_min) * (nphi_max_scale - nphi_min_scale) / (
        rhob_max - rhob_min
    )


def gas_xover(rhob, nphi):
    """Calculate the gas crossover from the bulk density and neutron porosity logs.

    Args:
        rhob (pd.Series): Bulk density log [g/cm³].
        nphi (pd.Series): Neutron porosity log [v/v].

    Returns:
        pd.Series: A float series (0.0 or 1.0) indicating gas crossover.
    """
    return (nphi < density_porosity(rhob)).astype(float)


def log_perm(perm):
    """Calculate the log permeability from the permeability.

    Args:
        perm (pd.Series): Permeability [mD].

    Returns:
        pd.Series: Log10 of permeability, clipped at a minimum of 1e-3 mD.
    """
    return np.log10(perm.clip(lower=1e-3))


def rock_flag_gr(gr):
    """Generate rock type flags based on the gamma-ray log.

    This function estimates shale volume from GR, finds optimal cutoffs using
    KMeans clustering, and then assigns rock type flags.

    Args:
        gr (pd.Series): Gamma-ray log [API].

    Returns:
        np.ndarray: An array of integer rock type flags.
    """
    vsh_gr = estimate_vsh_gr(gr)
    cutoffs = find_cutoffs(vsh_gr, 5)
    return rock_typing(vsh_gr, cut_offs=cutoffs, higher_is_better=False)


def coal_flagging(
    nphi, rhob, rhob_threshold=2.0, nphi_threshold=0.3, window_size=21, trend_factor=0.1
):
    """Flag coal intervals based on high NPHI and low RHOB, considering log trends.

    Coal is typically characterized by very low bulk density and high
    apparent neutron porosity. This function combines absolute thresholds with
    a trend-based approach, flagging points where values deviate significantly
    from their rolling average.

    Args:
        nphi (pd.Series): Neutron porosity log [v/v].
        rhob (pd.Series): Bulk density log [g/cm³].
        rhob_threshold (float, optional): Absolute RHOB threshold for coal. Defaults to 2.0.
        nphi_threshold (float, optional): Absolute NPHI threshold for coal. Defaults to 0.3.
        window_size (int, optional): The size of the rolling window to calculate trends. Defaults to 21.
        trend_factor (float, optional): A factor to control sensitivity to trend deviation. Defaults to 0.1.

    Returns:
        pd.Series: A series of floats (0.0 or 1.0), where 1.0 indicates coal.
    """
    # Calculate rolling averages to establish local trends
    rhob_trend = rhob.rolling(window=window_size, center=True, min_periods=1).mean()
    nphi_trend = nphi.rolling(window=window_size, center=True, min_periods=1).mean()

    # Flag where RHOB is significantly below its trend and NPHI is significantly above its trend
    trend_condition = (rhob < rhob_trend * (1 - trend_factor)) & (
        nphi > nphi_trend * (1 + trend_factor)
    )
    threshold_condition = (rhob < rhob_threshold) & (nphi > nphi_threshold)

    return (trend_condition & threshold_condition).astype(float)


def tight_streak_flagging(rhob, rhob_threshold=2.3, window_size=15, trend_factor=0.03):
    """Flag tight streak intervals based on high RHOB.

    Tight streaks (e.g., carbonate-cemented layers) are characterized by high
    bulk density. This function flags these zones by identifying points where
    RHOB deviates significantly from its local trend and also crosses an
    absolute threshold.

    Args:
        rhob (pd.Series): Bulk density log [g/cm³].
        rhob_threshold (float, optional): Absolute RHOB threshold for a tight streak. Defaults to 2.3.
        window_size (int, optional): The size of the rolling window for trend calculation. Defaults to 21.
        trend_factor (float, optional): A factor to control sensitivity to trend deviation. Defaults to 0.03.

    Returns:
        pd.Series: A series of floats (0.0 or 1.0), 1.0 where a tight streak is flagged.
    """
    # Calculate rolling averages to establish local trends
    rhob_trend = rhob.rolling(window=window_size, center=True, min_periods=1).mean()

    # Flag where RHOB are significantly above trend
    trend_condition = rhob > rhob_trend * (1 + trend_factor)

    # Flag where values cross absolute thresholds
    threshold_condition = rhob > rhob_threshold

    return (trend_condition & threshold_condition).astype(float)


def generate_fe_features(df):
    """Generate a suite of engineered features from raw well log data.

    Args:
        df (pd.DataFrame): DataFrame containing raw log data, including 'WELL_NAME' and 'DEPTH'.

    Returns:
        pd.DataFrame: The input DataFrame with added feature-engineered columns.
    """
    df = df.copy()

    # Well based features
    for well_name, well_df in tqdm(
        df.groupby("WELL_NAME"), desc="Generating well-based features"
    ):
        tqdm.write(f"Processing well {well_name}")
        well_df = well_df.sort_values("DEPTH").copy()

        df.loc[well_df.index, "TIGHT_FLAG"] = tight_streak_flagging(
            well_df["RHOB"], well_df["RT"]
        )

        df.loc[well_df.index, "COAL_FLAG"] = coal_flagging(
            well_df["NPHI"], well_df["RHOB"]
        )

        rhob_mask = well_df["RHOB"].notna()
        step = np.ceil(well_df["DEPTH"].diff().mean())
        rhob_int_values = rhob_integral(well_df.loc[rhob_mask, "RHOB"], step=step)
        df.loc[well_df.index, "RHOB_INT"] = pd.Series(
            rhob_int_values, index=well_df.index
        )

        gr_mask = well_df["GR"].notna()
        rock_flag_values = rock_flag_gr(well_df.loc[gr_mask, "GR"])
        df.loc[well_df.index, "ROCK_FLAG"] = pd.Series(
            rock_flag_values, index=well_df.index
        )

    # Point based features
    df["DPHI"] = density_porosity(df["RHOB"])
    df["GAS_XOVER"] = gas_xover(df["RHOB"], df["NPHI"])
    if "PERM" in df.columns and "LOG_PERM" not in df.columns:
        df["LOG_PERM"] = log_perm(df["PERM"])
    return df


def perform_rrt_smote(core_data, input_features: list, plot_distribution: bool = True):
    """Address class imbalance for RRT using SMOTE.

    This function uses the Synthetic Minority Over-sampling Technique (SMOTE) to
    address class imbalance in the Reservoir Rock Type (RRT) data. It is assumed
    that the 'ROCK_FLAG' column is the target variable and other numeric columns
    are features.

    Args:
        core_data (pd.DataFrame): DataFrame containing features and the 'ROCK_FLAG' target column.
        input_features (list): A list of column names to be used as features for SMOTE.
        plot_distribution (bool, optional): If True, plots the class distribution before and after SMOTE. Defaults to False.

    Returns:
        pd.DataFrame: A new DataFrame with a balanced class distribution for 'ROCK_FLAG'.
    """
    if "ROCK_FLAG" not in core_data.columns:
        raise ValueError(
            "Input DataFrame must contain a 'ROCK_FLAG' column as the target variable."
        )

    # Drop non-feature columns and rows with missing target values
    data = core_data.dropna(subset=["ROCK_FLAG"])
    y = data["ROCK_FLAG"]
    X = data.drop(columns=["ROCK_FLAG", "WELL_NAME", "DEPTH"], errors="ignore")

    # Select only numeric features for SMOTE
    X_numeric = X[input_features]

    # SMOTE cannot handle NaN values in features, so we drop them.
    # This might reduce the dataset size. Consider imputation as an alternative.
    valid_indices = X_numeric.dropna().index
    X_res = X_numeric.loc[valid_indices]
    y_res = y.loc[valid_indices]

    # Determine the number of samples in the smallest class
    min_class_count = y_res.value_counts().min()

    # k_neighbors must be less than the number of samples in the smallest class.
    # The default for k_neighbors is 5.
    if min_class_count <= 1:
        print("Warning: Cannot apply SMOTE. One of the classes has 1 or fewer samples.")
        return core_data.loc[valid_indices]

    k_neighbors = min(5, min_class_count - 1)

    smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
    X_smote, y_smote = smote.fit_resample(X_res, y_res)

    if plot_distribution:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Before SMOTE
        sns.countplot(x=y_res, ax=axes[0], palette="viridis")
        axes[0].set_title("Class Distribution Before SMOTE")
        axes[0].set_xlabel("ROCK_FLAG")
        axes[0].set_ylabel("Count")

        # After SMOTE
        sns.countplot(x=y_smote, ax=axes[1], palette="viridis")
        axes[1].set_title("Class Distribution After SMOTE")
        axes[1].set_xlabel("ROCK_FLAG")
        axes[1].set_ylabel("Count")

        fig.suptitle("ROCK_FLAG Distribution Before and After SMOTE", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    return pd.concat(
        [
            pd.DataFrame(X_smote, columns=X_res.columns),
            pd.DataFrame(y_smote, columns=["ROCK_FLAG"]),
        ],
        axis=1,
    )
