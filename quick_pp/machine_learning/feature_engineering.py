import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import seaborn as sns
except ImportError:
    sns = None  # type: ignore[assignment]

try:
    from imblearn.over_sampling import SMOTE
except ImportError:
    SMOTE = None  # type: ignore[assignment,misc]

from quick_pp.rock_type import estimate_vsh_gr, find_cutoffs, rock_typing


def rhob_integral(rhob, step=15):
    """Calculate a pseudo-integral of the bulk density log.

    Args:
        rhob (pd.Series): Bulk density log [g/cm³].
        step (int, optional): The step for calculating the difference before integration. Defaults to 15.

    Returns:
        np.ndarray: An array representing the cumulative sum of the stepped difference of RHOB.
    """
    return np.cumsum(rhob.clip(2, 3).diff(int(step)).fillna(0.0).cumsum())


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


def rock_flag_gr(gr, no_of_rocks=5):
    """Generate rock type flags based on the gamma-ray log.

    This function estimates shale volume from GR, finds optimal cutoffs using
    KMeans clustering, and then assigns rock type flags.

    Args:
        gr (pd.Series): Gamma-ray log [API].
        no_of_rocks (int, optional): The number of rock types to identify. Defaults to 5.

    Returns:
        np.ndarray: An array of integer rock type flags.
    """
    vsh_gr = estimate_vsh_gr(gr)
    cutoffs = find_cutoffs(vsh_gr, no_of_rocks)
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


def tight_streak_flagging(
    rhob, rt=None, rhob_threshold=2.3, window_size=15, trend_factor=0.03
):
    """Flag tight streak intervals based on high RHOB (and optional resistivity).

    Tight streaks (e.g., carbonate-cemented layers) are characterized by high
    bulk density and low porosity. This function flags these zones by identifying
    points where RHOB deviates significantly from its local trend and crosses an
    absolute threshold.

    Args:
        rhob (pd.Series): Bulk density log [g/cm³].
        rt (pd.Series or float, optional): True resistivity log [ohm.m] or rhob_threshold
                                           for backward compatibility. Defaults to None.
        rhob_threshold (float, optional): Absolute RHOB threshold for a tight streak. Defaults to 2.3.
        window_size (int, optional): The size of the rolling window for trend calculation. Defaults to 15.
        trend_factor (float, optional): A factor to control sensitivity to trend deviation. Defaults to 0.03.

    Returns:
        pd.Series: A series of floats (0.0 or 1.0), 1.0 where a tight streak is flagged.
    """
    if isinstance(rt, (int, float, np.floating)):
        rhob_threshold = float(rt)
        rt = None

    rhob_series = pd.Series(rhob)
    # Calculate rolling averages to establish local trends
    rhob_trend = rhob_series.rolling(
        window=window_size, center=True, min_periods=1
    ).mean()

    # Flag where RHOB are significantly above trend
    trend_condition = rhob_series > rhob_trend * (1 + trend_factor)

    # Flag where values cross absolute thresholds
    threshold_condition = rhob_series > rhob_threshold

    return (trend_condition & threshold_condition).astype(float)


def apparent_matrix_density(
    rhob, nphi, rho_fluid: float = 1.0, min_rho: float = 2.0, max_rho: float = 3.2
):
    """Calculate apparent matrix density (rho_ma,app) from bulk density and neutron porosity.

    Physics relation:
        rho_b = (1 - phi_app) * rho_ma,app + phi_app * rho_fluid
        => rho_ma,app = (rho_b - phi_app * rho_fluid) / (1 - phi_app)

    Helps distinguish quartz (~2.65 g/cc), calcite (~2.71 g/cc), dolomite (~2.87 g/cc),
    and heavy minerals/gas effects before ML regression.

    Args:
        rhob (pd.Series or np.ndarray): Bulk density log [g/cm³].
        nphi (pd.Series or np.ndarray): Neutron porosity log [v/v].
        rho_fluid (float, optional): Pore fluid density [g/cm³]. Defaults to 1.0.
        min_rho (float, optional): Lower physical clip bound [g/cm³]. Defaults to 2.0.
        max_rho (float, optional): Upper physical clip bound [g/cm³]. Defaults to 3.2.

    Returns:
        pd.Series or np.ndarray: Apparent matrix density [g/cm³].
    """
    phi_app = np.clip(nphi, 0.0, 0.60)
    denom = np.clip(1.0 - phi_app, 0.40, None)
    rhoma_app = (rhob - phi_app * rho_fluid) / denom
    return np.clip(rhoma_app, min_rho, max_rho)


def _estimate_dtc_from_rhob_gardner(rhob):
    """Estimate synthetic compressional slowness [us/ft] from bulk density using Gardner's relation.

    Gardner's equation:
        rho_b = 0.23 * Vp_ft_s ** 0.25  =>  Vp_ft_s = (rho_b / 0.23) ** 4
        DTC [us/ft] = 1e6 / Vp_ft_s
    """
    rhob_safe = np.clip(rhob, 1.5, 3.1)
    vp_ft_s = (rhob_safe / 0.23) ** 4
    dtc_syn = 1e6 / np.clip(vp_ft_s, 3500.0, 25000.0)
    return np.clip(dtc_syn, 40.0, 220.0)


def acoustic_impedance_feature(rhob, dtc=None):
    """Calculate Acoustic Impedance (AI = rho_b * Vp) in [(g/cm³)*(m/s)].

    Uses measured sonic slowness (DTC/DT in us/ft) when available, and falls back
    to Gardner-derived synthetic slowness where sonic data is absent or NaN.

    Args:
        rhob (pd.Series or np.ndarray): Bulk density log [g/cm³].
        dtc (pd.Series or np.ndarray, optional): Compressional slowness log [us/ft]. Defaults to None.

    Returns:
        pd.Series or np.ndarray: Acoustic impedance [(g/cm³)*(m/s)].
    """
    from quick_pp.rock_physics.geophysics import calculate_acoustic_impedance

    dtc_fallback = _estimate_dtc_from_rhob_gardner(rhob)
    if dtc is None:
        dtc_eff = dtc_fallback
    else:
        dtc_series = pd.Series(dtc, index=getattr(rhob, "index", None))
        dtc_eff = np.where(
            dtc_series.notna() & (dtc_series > 20.0), dtc_series, dtc_fallback
        )
        if isinstance(rhob, pd.Series):
            dtc_eff = pd.Series(dtc_eff, index=rhob.index)

    return calculate_acoustic_impedance(rhob, dtc_eff)


def mn_lithology_parameters(
    rhob,
    nphi,
    dtc=None,
    dt_fluid: float = 189.0,
    rho_fluid: float = 1.0,
    nphi_fluid: float = 1.0,
):
    """Calculate Burke's M-N lithology crossplot parameters.

    The M and N parameters remove primary porosity dependence by taking the slopes
    of the sonic-density and neutron-density lines relative to the fluid point:
        M = ((dt_fluid - dtc) / (rhob - rho_fluid)) * 0.01
        N = (nphi_fluid - nphi) / (rhob - rho_fluid)

    Args:
        rhob (pd.Series or np.ndarray): Bulk density log [g/cm³].
        nphi (pd.Series or np.ndarray): Neutron porosity log [v/v].
        dtc (pd.Series or np.ndarray, optional): Compressional slowness [us/ft]. Defaults to None.
        dt_fluid (float, optional): Fluid slowness [us/ft]. Defaults to 189.0.
        rho_fluid (float, optional): Fluid density [g/cm³]. Defaults to 1.0.
        nphi_fluid (float, optional): Fluid neutron porosity [v/v]. Defaults to 1.0.

    Returns:
        tuple: (M_LITH, N_LITH) lithology parameters.
    """
    dtc_fallback = _estimate_dtc_from_rhob_gardner(rhob)
    if dtc is None:
        dtc_eff = dtc_fallback
    else:
        dtc_series = pd.Series(dtc, index=getattr(rhob, "index", None))
        dtc_eff = np.where(
            dtc_series.notna() & (dtc_series > 20.0), dtc_series, dtc_fallback
        )
        if isinstance(rhob, pd.Series):
            dtc_eff = pd.Series(dtc_eff, index=rhob.index)

    denom = rhob - rho_fluid
    # Avoid division by zero where rhob approaches rho_fluid
    safe_denom = np.where(np.abs(denom) < 0.1, np.where(denom >= 0, 0.1, -0.1), denom)

    m_lith = ((dt_fluid - dtc_eff) / safe_denom) * 0.01
    n_lith = (nphi_fluid - nphi) / safe_denom

    m_lith = np.clip(m_lith, -0.5, 2.0)
    n_lith = np.clip(n_lith, -0.5, 2.0)

    if isinstance(rhob, pd.Series):
        m_lith = pd.Series(m_lith, index=rhob.index)
        n_lith = pd.Series(n_lith, index=rhob.index)

    return m_lith, n_lith


def velocity_deviation_log(
    rhob=None,
    nphi=None,
    dtc=None,
    phit=None,
    dt_matrix: float = 47.6,
    dt_fluid: float = 189.0,
    rho_matrix: float = 2.68,
    rho_fluid: float = 1.0,
):
    """Calculate the Velocity Deviation Log (VDL = Vp_measured - Vp_Wyllie) in [m/s],
    based on Anselmetti & Eberli (1999).

    Compares measured sonic velocity (Vp = 304800 / DTC) against the synthetic velocity
    predicted from total porosity (or neutron-density porosity) via Wyllie's time-average
    equation:
        DT_Wyllie = (1 - phi_T) * dt_matrix + phi_T * dt_fluid
        Vp_Wyllie = 304800.0 / DT_Wyllie
        VDL = Vp_measured - Vp_Wyllie  [m/s]

    Petrophysical interpretation:
        - VDL > 0 (positive anomaly): Stiff frame-forming pores (intraparticle, moldic, or separate vugs).
        - VDL ~ 0 (zero deviation): Interparticle / intercrystalline matrix porosity following Wyllie.
        - VDL < 0 (negative anomaly): Fractures, microcracks, unconsolidated frame, or gas effect.

    Args:
        rhob (pd.Series or np.ndarray, optional): Bulk density log [g/cm³].
        nphi (pd.Series or np.ndarray, optional): Neutron porosity log [v/v].
        dtc (pd.Series or np.ndarray, optional): Compressional sonic slowness [us/ft or us/m].
        phit (pd.Series or np.ndarray, optional): Total porosity [v/v]. If None, estimated from RHOB/NPHI.
        dt_matrix (float, optional): Matrix sonic transit time [us/ft]. Defaults to 47.6 (carbonate/mixed).
        dt_fluid (float, optional): Fluid sonic transit time [us/ft]. Defaults to 189.0.
        rho_matrix (float, optional): Fallback matrix density [g/cm³] when `phit` is None. Defaults to 2.68.
        rho_fluid (float, optional): Fallback fluid density [g/cm³] when `phit` is None. Defaults to 1.0.

    Returns:
        pd.Series or np.ndarray: Velocity Deviation Log (VDL) in [m/s].
    """
    ref_index = None
    for candidate in (phit, rhob, nphi, dtc):
        if isinstance(candidate, pd.Series):
            ref_index = candidate.index
            break

    # 1. Resolve total porosity phi_t in [0.001, 0.55]
    phi_est = None
    if rhob is not None:
        rhob_arr = np.asarray(rhob, dtype=float)
        phi_d = (float(rho_matrix) - rhob_arr) / max(
            float(rho_matrix) - float(rho_fluid), 0.1
        )
        if nphi is not None:
            nphi_arr = np.asarray(nphi, dtype=float)
            both_nd = np.isfinite(phi_d) & np.isfinite(nphi_arr)
            phi_est = np.where(both_nd, 0.5 * (phi_d + nphi_arr), phi_d)
        else:
            phi_est = phi_d
    elif nphi is not None:
        phi_est = np.asarray(nphi, dtype=float)

    if phit is not None:
        phit_arr = np.asarray(phit, dtype=float)
        if phi_est is not None:
            phi_t = np.where(np.isfinite(phit_arr), phit_arr, phi_est)
        else:
            phi_t = phit_arr
    elif phi_est is not None:
        phi_t = phi_est
    else:
        raise ValueError(
            "At least one of `phit`, `rhob`, or `nphi` must be provided to calculate VDL."
        )

    phi_t_clipped = np.clip(phi_t, 0.001, 0.55)

    # 2. Synthetic Wyllie transit time [us/ft] and velocity [m/s]
    dt_wyllie = (1.0 - phi_t_clipped) * float(dt_matrix) + phi_t_clipped * float(
        dt_fluid
    )
    vp_wyllie = 304800.0 / np.clip(dt_wyllie, 35.0, 220.0)

    # 3. Measured sonic transit time [us/ft] (with Gardner fallback where DTC is missing)
    dtc_fallback = (
        _estimate_dtc_from_rhob_gardner(np.asarray(rhob, dtype=float))
        if rhob is not None
        else dt_wyllie
    )
    if dtc is None:
        dtc_eff = dtc_fallback
    else:
        dtc_arr = np.asarray(dtc, dtype=float)
        valid_dt = np.isfinite(dtc_arr) & (dtc_arr > 20.0)
        if valid_dt.any():
            med_dt = float(np.nanmedian(dtc_arr[valid_dt]))
            if med_dt > 135.0:
                dtc_arr = np.where(valid_dt, dtc_arr / 3.28084, np.nan)
        dtc_eff = np.where(valid_dt, dtc_arr, dtc_fallback)

    vp_meas = 304800.0 / np.clip(dtc_eff, 35.0, 220.0)
    vdl = np.clip(vp_meas - vp_wyllie, -3000.0, 3000.0)

    if ref_index is not None:
        return pd.Series(vdl, index=ref_index, name="VDL")
    return vdl


def generate_fe_features(df):
    """Generate a suite of engineered and physics-derived features from raw well log data.

    Includes:
        - Well-based context features: TIGHT_FLAG, COAL_FLAG, RHOB_INT, ROCK_FLAG
        - Point-based petrophysical features: DPHI, GAS_XOVER, LOG_PERM
        - Domain physics features:
            * RHOMA_APP: Apparent matrix density (g/cm³)
            * AI: Acoustic impedance ((g/cm³)*(m/s))
            * M_LITH, N_LITH: Burke's M-N lithology parameters

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
            well_df["RHOB"], well_df.get("RT")
        )

        df.loc[well_df.index, "COAL_FLAG"] = coal_flagging(
            well_df["NPHI"], well_df["RHOB"]
        )

        rhob_mask = well_df["RHOB"].notna()
        depth_diff_mean = well_df["DEPTH"].diff().mean()
        step = (
            int(np.ceil(depth_diff_mean))
            if pd.notna(depth_diff_mean) and depth_diff_mean > 0
            else 1
        )
        rhob_int_values = rhob_integral(well_df.loc[rhob_mask, "RHOB"], step=step)
        df.loc[well_df.index, "RHOB_INT"] = pd.Series(
            rhob_int_values, index=well_df.loc[rhob_mask].index
        )

        gr_mask = well_df["GR"].notna()
        if gr_mask.sum() >= 5:
            rock_flag_values = rock_flag_gr(well_df.loc[gr_mask, "GR"])
            df.loc[well_df.loc[gr_mask].index, "ROCK_FLAG"] = pd.Series(
                rock_flag_values, index=well_df.loc[gr_mask].index
            )
        else:
            df.loc[well_df.index, "ROCK_FLAG"] = 1

    # Point based features
    df["DPHI"] = density_porosity(df["RHOB"])
    df["GAS_XOVER"] = gas_xover(df["RHOB"], df["NPHI"])

    # Physics-derived domain features
    sonic_col = (
        df["DTC"]
        if "DTC" in df.columns
        else (df["DT"] if "DT" in df.columns else None)
    )
    df["RHOMA_APP"] = apparent_matrix_density(df["RHOB"], df["NPHI"])
    df["AI"] = acoustic_impedance_feature(df["RHOB"], sonic_col)
    m_lith, n_lith = mn_lithology_parameters(df["RHOB"], df["NPHI"], sonic_col)
    df["M_LITH"] = m_lith
    df["N_LITH"] = n_lith
    df["VDL"] = velocity_deviation_log(
        df.get("RHOB"), df.get("NPHI"), sonic_col, phit=df.get("PHIT")
    )

    if "PERM" in df.columns and "LOG_PERM" not in df.columns:
        df["LOG_PERM"] = log_perm(df["PERM"])
    elif "CORE_PERM" in df.columns and "LOG_PERM" not in df.columns:
        df["LOG_PERM"] = log_perm(df["CORE_PERM"])
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
    if SMOTE is None:
        raise ImportError(
            "imbalanced-learn is required for perform_rrt_smote. Install with: uv pip install 'quick-pp[ml]'"
        )
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
        if sns is None:
            raise ImportError(
                "seaborn is required to plot SMOTE distribution. Install with: uv pip install 'quick-pp[ml]'"
            )
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
