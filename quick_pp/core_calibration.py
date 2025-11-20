from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
import pandas as pd

from quick_pp.utils import power_law_func, inv_power_law_func
from quick_pp import logger


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
    """Generate J-Sw cross plot.

    Args:
        sw (float): Core water saturation (frac).
        j (float): Calculated J value (unitless).
        a (float, optional): a constant in j=a*sw^b. Defaults to None.
        b (float, optional): b constant in j=a*sw^b. Defaults to None.
        label (str, optional): Label for the data group. Defaults to ''.
        ylim (tuple, optional): Range for the y axis in (min, max) format. Defaults to None.
        log_log (bool, optional): Whether to plot log-log or not. Defaults to False.
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
        popt, _ = curve_fit(power_law_func, poro, perm, nan_policy="omit")
        a = [round(c) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
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

        # --- SOLUTION: Use index-based merging to avoid column conflicts ---

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
