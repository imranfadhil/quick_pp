from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import re
import pandas as pd
import hashlib
from sklearn.metrics import root_mean_squared_error

from quick_pp.rock_type import estimate_pore_throat
from quick_pp.utils import power_law_func, inv_power_law_func
from quick_pp.config import Config
from quick_pp import logger

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
        for i, row in enumerate(zip(poro, perm, core_group)):
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
        legend=dict(x=1.04, y=1, traceorder="normal"),
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
def fit_j_curve(sw, j):
    """Estimate a and b constants of best fit given core water saturation and J value.

    Args:
        sw (float): Core water saturation (frac).
        j (float): Calculated J value (unitless).

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        popt, _ = curve_fit(inv_power_law_func, sw, j, p0=[0.01, 1])
        a = [round(c, 3) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
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
    return 1 - a * np.exp(-((b / (h + d)) ** c))


def fit_skelt_harrison_curve(sw, h):
    """Estimate a and b constants of best fit given core water saturation and capillary pressure values.

    Args:
        sw (float): Core water saturation (frac).
        h (float): Height above free water level (ft).

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        popt, _ = curve_fit(skelt_harrison_func, h, sw, p0=[0.9, 100, 1.5, 1.5])
        a = [round(c, 3) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        c = [round(c, 3) for c in popt][2]
        d = [round(c, 3) for c in popt][3]
        return a, b, c, d
    except Exception as e:
        logger.error(e)
        return 1, 1, 1, 1


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


def leverett_j(pc, ift, theta, perm, phit):
    """Estimate Leverett J.

    Args:
        pc (float): Capillary pressure (psi).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).

    Returns:
        float: Leverett J value.
    """
    return (
        0.21665 * pc / (ift * abs(np.cos(np.radians(theta)))) * (perm / phit) ** (0.5)
    )


def sw_skelt_harrison(depth, fwl, a, b, c, d):
    """Estimate water saturation based on Skelt-Harrison.

    Args:
        depth (float): True vertical depth.
        fwl (float): Free water level in true vertical depth.
        a (float): a constant from the best-fit curve.
        b (float): b constant from the best-fit curve.
        c (float): c constant from the best-fit curve.
        d (float): d constant from the best-fit curve.

    Returns:
        float: Water saturation.
    """
    h = (fwl - depth).clip(0)
    return skelt_harrison_func(h, a, b, c, d)


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
        sw (float): Water saturation (frac).
        phit (float): Total porosity (frac).
        h (float): True vertical depth.
        a (float): Fitting parameter.
        b (float): Fitting parameter (negative value).

    Returns:
        float: Water saturation.
    """
    return a / (h**b * phit)


def sw_shf_leverett_j(perm, phit, depth, fwl, ift, theta, gw, ghc, a, b, phie=None):
    """Estimate water saturation based on Leverett J function.

    Args:
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        gw (float): Water density (g/cc).
        ghc (float): Hydrocarbon density (g/cc).
        a (float): A constant from J function.
        b (float): B constant from J function.
        phie (float): Effective porosity (frac), required for clay bound water calculation. Defaults to None.

    Returns:
        float: Water saturation from saturation height function.
    """
    h = (fwl - depth).clip(0)
    pc = h * (gw - ghc) * 0.433  # Convert g/cc to psi/ft
    shf = (a / leverett_j(pc, ift, theta, perm, phit)) ** (1 / b)
    return shf if not phie else shf * (1 - (phie / phit)) + (phie / phit)


def sw_shf_cuddy(phit, depth, fwl, gw, ghc, a, b):
    """Estimate water saturation based on Cuddy's saturation height function.

    Args:
        phit (float): Porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        gw (float): Water density (g/cc).
        ghc (float): Hydrocarbon density (g/cc).
        a (float): Fitting parameter.
        b (float): Fitting parameter (negative value).

    Returns:
        float: Water saturation from saturation height function.
    """
    h = (fwl - depth).clip(0)
    shf = a / ((h * (gw - ghc) * 0.433) ** b * phit)
    return shf


def sw_shf_choo(perm, phit, phie, depth, fwl, ift, theta, gw, ghc, b0=0.4):
    """Estimate water saturation based on Choo's saturation height function.

    Args:
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).
        phie (float): Effective porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        gw (float): Water density (g/cc).
        ghc (float): Hydrocarbon density (g/cc).
        b0 (float): _description_. Defaults to 0.4.

    Returns:
        float: Water saturation from saturation height function.
    """
    swb = 1 - (phie / phit)
    h = (fwl - depth).clip(0)
    pc = h * (gw - ghc) * 0.433
    shf = 10 ** ((2 * b0 - 1) * np.log10(1 + swb**-1) + np.log10(1 + swb)) / (
        (
            0.2166
            * (pc / (ift * abs(np.cos(np.radians(theta)))))
            * (perm / phit) ** (0.5)
        )
        ** (b0 * np.log10(1 + swb**-1) / 3)
    )
    return shf


def autocorrelate_core_depth(df):
    """Automatically shift core depths to match log depths using Dynamic Time Warping (DTW).

    This function aligns core porosity (CPORE) with log porosity (PHIT) to correct for
    depth discrepancies between core and wireline log measurements. It processes the data
    on a per-well basis.

    Args:
        df (pd.DataFrame): DataFrame containing well log and core data. Must include
                           'WELL_NAME', 'DEPTH', 'PHIT', 'CPORE', 'CPERM', and 'CORE_ID'.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame with the original data merged with the depth-corrected core
              data. New columns include 'DEPTH_CORRECTED', 'CORE_ID_SHIFTED',
              'CPORE_SHIFTED', and 'CPERM_SHIFTED'.
            - A summary DataFrame detailing the depth shifts for each core sample,
              including 'WELL_NAME', 'CORE_ID', 'ORIGINAL_DEPTH', 'DEPTH_CORRECTED',
              and 'DEPTH_SHIFT'.
    """
    from dtw import dtw
    from sklearn.preprocessing import minmax_scale

    required_cols = ["WELL_NAME", "DEPTH", "PHIT", "CPORE", "CPERM", "CORE_ID"]
    for col in required_cols:
        if col not in df.columns:
            raise AssertionError(f"Missing required column: {col}")

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

        # Create a window of log data around core points to speed up DTW
        core_indices_in_log = np.searchsorted(
            log_data.DEPTH, core_data.DEPTH, side="left"
        )
        window = 50  # Increased window size to allow for larger shifts
        window_indices = core_indices_in_log[:, None] + np.arange(-window, window + 1)
        window_indices = np.clip(window_indices, 0, len(log_data) - 1)
        unique_indices = np.unique(window_indices.flatten())
        log_data_subset = log_data.iloc[unique_indices].reset_index(drop=True)

        # Extract numpy arrays for DTW
        phit_vals = minmax_scale(log_data_subset["PHIT"].values)
        cpore_vals = minmax_scale(core_data["CPORE"].values)

        alignment = dtw(
            cpore_vals, phit_vals, keep_internals=True, step_pattern="symmetric2"
        )

        # Map alignment indices back to original data
        core_indices = alignment.index1
        log_subset_indices = alignment.index2

        original_depths = core_data.loc[core_indices, "DEPTH"].values
        corrected_depths = log_data_subset.loc[log_subset_indices, "DEPTH"].values

        # Calculate the shift for each point in the DTW path
        depth_shifts = corrected_depths - original_depths

        # Determine the single best shift for the entire core sequence (using the median for robustness)
        if len(depth_shifts) > 0:
            best_shift = round(np.median(depth_shifts), 4)
        else:
            best_shift = 0.0

        tqdm.write(f"Determined best shift for {well}: {best_shift:.2f}")

        # Apply the consistent shift to all original core data for this well
        shifted_core_data = core_data.copy()
        shifted_core_data["DEPTH_CORRECTED"] = shifted_core_data["DEPTH"] + best_shift
        shifted_core_data["DEPTH_SHIFT"] = best_shift

        # Create summary for this well
        summary = shifted_core_data[
            ["CORE_ID", "DEPTH", "DEPTH_CORRECTED", "DEPTH_SHIFT"]
        ].copy()
        summary.rename(columns={"DEPTH": "ORIGINAL_DEPTH"}, inplace=True)
        summary["WELL_NAME"] = well
        summary = summary[
            ["WELL_NAME", "CORE_ID", "ORIGINAL_DEPTH", "DEPTH_CORRECTED", "DEPTH_SHIFT"]
        ]
        shift_summaries.append(summary)

        # Prepare shifted data for merging
        final_shifted = shifted_core_data.rename(
            columns={
                "CPORE": "CPORE_SHIFTED",
                "CPERM": "CPERM_SHIFTED",
                "CORE_ID": "CORE_ID_SHIFTED",
            }
        )

        # Prepare the log data: keep only necessary columns and set DEPTH as index
        log_df_to_merge = data.set_index("DEPTH")

        # Prepare the shifted core data: keep shifted values and set corrected depth as index
        core_df_to_merge = final_shifted[
            ["DEPTH_CORRECTED", "CPORE_SHIFTED", "CPERM_SHIFTED", "CORE_ID_SHIFTED"]
        ].set_index("DEPTH_CORRECTED")

        # Merge the two dataframes on their indices (which are the depths)
        well_corrected_df = pd.merge(
            log_df_to_merge,
            core_df_to_merge,
            left_index=True,
            right_index=True,
            how="outer",
        ).reset_index()

        # The merged 'index' column is the unified depth column, so rename it
        well_corrected_df.rename(columns={"index": "DEPTH"}, inplace=True)

        # Sort by the final unified depth
        well_corrected_df.sort_values("DEPTH", inplace=True)

        # Add well name back if lost during merge
        well_corrected_df["WELL_NAME"] = well
        all_wells_data.append(well_corrected_df)

    # Combine all processed well data and summaries
    return_df = pd.concat(all_wells_data, ignore_index=True)
    final_summary_df = (
        pd.concat(shift_summaries, ignore_index=True)
        if shift_summaries
        else pd.DataFrame()
    )

    return return_df, final_summary_df


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
        if col.startswith("Pc_") or col.startswith("Sw_"):
            parts = col.split("_")
            if len(parts) == 2 and parts[1].isdigit():
                new_columns[col] = f"{parts[0]}.{parts[1]}"

    df_temp = df_wide.rename(columns=new_columns)

    # 2. Apply the wide_to_long function
    cols = ["Well", "Sample ID", "Depth_m", "K_mD", "PHI_frac"]
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


def auto_j_params(core_data, excluded_samples=[], cluster_by=None):
    """Automatically calculates J-function parameters (a, b) for each rock type.

    This function groups the data by 'ROCK_FLAG', fits a J-curve for each group,
    and returns the best-fit parameters 'a' and 'b' along with the RMSE. It can
    also create sub-groups within each rock type for more granular parameterization.

    Args:
        core_data (pd.DataFrame): DataFrame containing 'ROCK_FLAG', 'SWN', and 'J' columns.
        excluded_samples (list, optional): A list of 'Sample' names to exclude from the calculation.
            Defaults to [].
        cluster_by (list, optional): A list of column names to create sub-groups
            within each 'ROCK_FLAG'. If provided, parameters will be calculated
            for each unique combination of 'ROCK_FLAG' and the cluster_by columns.
            Defaults to None.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              group identifiers (e.g., 'ROCK_FLAG') and the J-parameters ('a', 'b', 'rmse').
    """
    core_data = core_data[~core_data["Sample"].isin(excluded_samples)].copy()

    grouping_cols = ["ROCK_FLAG"]
    if cluster_by:
        if not isinstance(cluster_by, list):
            cluster_by = [cluster_by]
        grouping_cols.extend(cluster_by)

    j_params_list = []
    for group_key, data in core_data.groupby(grouping_cols):
        clean_data = data.dropna(subset=["SWN", "J"]).query("SWN > 0")

        # Ensure there's still data left to fit
        if len(clean_data) < 2:
            continue  # Skip groups with insufficient data after filtering

        a, b = fit_j_curve(clean_data["SWN"], clean_data["J"])
        rmse = round(
            root_mean_squared_error(clean_data["J"], a * clean_data["SWN"] ** b),
            4,
        )

        # Create a dictionary for the group identifiers
        params = {
            # Cast to standard python types to ensure JSON serializability
            col: (
                int(val) if isinstance(val, (np.integer, np.int32, np.int64)) else val
            )
            for col, val in zip(
                grouping_cols,
                group_key if isinstance(group_key, tuple) else [group_key],
            )
        }

        # Add the calculated J-parameters
        params.update({"a": round(a, 4), "b": round(b, 4), "rmse": rmse})
        j_params_list.append(params)

    # Sort the list to ensure the best rock quality for each ROCK_FLAG is first.
    # Best quality = lowest 'a' (entry pressure) and most negative 'b' (pore distribution).
    # The sorting is done by ROCK_FLAG, then 'a', then 'b'.
    j_params_list.sort(
        key=lambda x: (x.get("ROCK_FLAG", 0), x.get("a", 0), x.get("b", 0))
    )
    return j_params_list


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
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import DBSCAN

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
        for desc, cluster_id in zip(unique_descriptions, clusters)
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
    for sample, data in core_data.groupby(["Well", "Sample"]):
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
