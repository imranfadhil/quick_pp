from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from tqdm import tqdm
from dtw import dtw
import pandas as pd

from quick_pp.utils import power_law_func, inv_power_law_func
from quick_pp import logger


plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


# Cross plots
def poroperm_xplot(poro, perm, a=None, b=None, core_group=None, label='', log_log=False):
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
    sc = plt.scatter(poro, perm, marker='.', c=core_group, cmap='Set1')
    if core_group is not None:
        for i, row in enumerate(zip(poro, perm, core_group)):
            plt.annotate(row[2], (row[0], row[1]), fontsize=8, alpha=0.7)
    if a and b:
        line_color = sc.get_facecolors()[0]
        line_color[-1] = 0.5
        cpore = np.geomspace(0.01, 0.5, 30)
        plt.plot(cpore, power_law_func(cpore, a, b), color=line_color, label=label, linestyle='dashed')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.xlabel('CPORE (frac)')
    plt.xlim(0.001, 0.5)
    plt.ylabel('CPERM (mD)')
    plt.ylim(.001, 1e5)
    plt.yscale('log')
    if log_log:
        plt.xscale('log')


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
    sc = plt.plot(bvw, pc, marker='s', label=label)
    if a is not None and b is not None:
        line_color = sc[0].get_color() + '66'  # Set opacity to 0
        cbvw = np.linspace(0.05, 0.35, 30)
        plt.scatter(cbvw, inv_power_law_func(cbvw, a, b), marker='x', color=line_color)
    plt.xlabel('BVW (frac)')
    plt.ylabel('Pc (psi)')
    plt.ylim(ylim) if ylim else plt.ylim(0.01, plt.gca().get_lines()[-1].get_ydata().max())
    if label:
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if log_log:
        plt.xscale('log')
        plt.yscale('log')


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
    plt.plot(sw, pc, marker='.', label=label)
    plt.xlabel('Sw (frac)')
    plt.xlim(0.01, 1)
    plt.ylabel('Pc (psi)')
    plt.ylim(ylim) if ylim else plt.ylim(0.01, plt.gca().get_lines()[-1].get_ydata().max())
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
        xaxis_title='Sw (frac)',
        yaxis_title='Pc (psi)',
        xaxis_range=[0, 1],
        yaxis_range=[0, 50] if ylim is None else [ylim[0], ylim[1]],
        legend=dict(x=1.04, y=1, traceorder='normal'),
        height=500,
        width=800
    )
    return fig


def j_xplot(sw, j, a=None, b=None, core_group=None, label=None, log_log=False, ax=None, ylim=None):
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
    ax = ax or plt.gca()
    scatter = ax.scatter(sw, j, marker='.', c=core_group, cmap='Set1')
    if core_group is not None:
        legend1 = ax.legend(*scatter.legend_elements(), title="Core Sample")
        ax.add_artist(legend1)
    if a is not None and b is not None:
        csw = np.geomspace(0.01, 1.0, 20)
        ax.plot(csw, inv_power_law_func(csw, a, b), label=label, linestyle='dashed')
    ax.set_xlabel('Sw (frac)')
    ax.set_xlim(0.01, 1)
    ax.set_ylabel('J')
    ax.set_ylim(ylim) if ylim else ax.set_ylim(0.01, max(ax.get_lines()[-1].get_ydata()))
    if log_log:
        ax.set_yscale('log')
        ax.set_ylim(0.01, 100)
        ax.set_xscale('log')
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
        popt, _ = curve_fit(inv_power_law_func, sw, j, p0=[.01, 1])
        a = [round(c, 3) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except Exception as e:
        logger.error(e)
        return 1, 1


def skelt_harrison_xplot(sw, pc, gw, ghc, a, b, c, d, core_group=None, label=None, ylim=None, ax=None):
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
    ax.scatter(sw, pc, marker='.', c=core_group, cmap='Set1')
    h = np.geomspace(.01, 10000, 100)
    pci = h * (gw - ghc) * .433  # Convert g/cc to psi/ft
    ax.plot(skelt_harrison_func(h, a, b, c, d), pci, label=label)
    ax.set_ylabel('Pc (psi)')
    ax.set_ylim(ylim) if ylim else ax.set_ylim(0.01, ax.get_lines()[-1].get_ydata().max())
    ax.set_xlabel('Sw (frac)')
    ax.set_xlim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    return ax


def skelt_harrison_func(h, a, b, c, d):
    return 1 - a * np.exp(-(b / (h + d))**c)


def fit_skelt_harrison_curve(sw, h):
    """Estimate a and b constants of best fit given core water saturation and capillary pressure values.

    Args:
        sw (float): Core water saturation (frac).
        h (float): Height above free water level (ft).

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        popt, _ = curve_fit(skelt_harrison_func, h, sw, p0=[.9, 100, 1.5, 1.5])
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
        popt, _ = curve_fit(power_law_func, poro, perm, nan_policy='omit')
        a = [round(c) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except Exception as e:
        logger.error(e)
        return 1, 1


def leverett_j(pc, ift, theta, perm, phit):
    """ Estimate Leverett J.

    Args:
        pc (float): Capillary pressure (psi).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).

    Returns:
        float: Leverett J value.
    """
    return 0.21665 * pc / (ift * abs(np.cos(np.radians(theta)))) * (perm / phit)**(0.5)


def pseudo_leverett_j():
    """TODO: Generate Pseudo-Leverett J based.

    Args:
        pc (float): Capillary pressure.
        ift (float): Interfacial tension.
        perm (float): Permeability.
        phit (float): Total porosity.

    Returns:
        float: Pseudo-Leverett J function.
    """
    pass


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
    h = fwl - depth
    return skelt_harrison_func(h, a, b, c, d)


def sw_cuddy(phit, h, a, b):
    """Estimate water saturation based on Cuddy's.

    Args:
        sw (float): Water saturation (frac).
        phit (float): Total porosity (frac).
        h (float): True vertical depth.
        a (float): Cementation exponent.
        b (float): Saturation exponent.

    Returns:
        float: Water saturation.
    """
    return a / phit * h**b


def sw_shf_leverett_j(perm, phit, depth, fwl, ift, theta, gw, ghc, a, b, phie=None):
    """Estimate water saturation based on Leverett J function.

    Args:
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        ift (float): Interfacial tension (dynes/cm).
        theta (float): Wetting angle (degree).
        gw (float): Gas density (g/cc).
        ghc (float): Gas height (g/cc).
        a (float): A constant from J function.
        b (float): B constant from J function.
        phie (float): Effective porosity (frac), required for clay bound water calculation. Defaults to None.

    Returns:
        float: Water saturation from saturation height function.
    """
    h = fwl - depth
    pc = h * (gw - ghc) * .433  # Convert g/cc to psi/ft
    shf = (a / leverett_j(pc, ift, theta, perm, phit))**(1 / b)
    return shf if not phie else shf * (1 - (phie / phit)) + (phie / phit)


def sw_shf_cuddy(phit, depth, fwl, gw, ghc, a, b):
    """Estimate water saturation based on Cuddy's saturation height function.

    Args:
        phit (float): Porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        gw (float): Gas density (g/cc).
        ghc (float): Gas height (g/cc).
        a (float): Cementation exponent.
        b (float): Saturation exponent.

    Returns:
        float: Water saturation from saturation height function.
    """
    h = fwl - depth
    shf = (h * (gw - ghc) * .433 / a)**(1 / b) / phit
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
        gw (float): Gas density (g/cc).
        ghc (float): Gas height (g/cc).
        b0 (float): _description_. Defaults to 0.4.

    Returns:
        float: Water saturation from saturation height function.
    """
    swb = 1 - (phie / phit)
    h = fwl - depth
    pc = h * (gw - ghc) * .433
    shf = 10**((2 * b0 - 1) * np.log10(1 + swb**-1) + np.log10(1 + swb)) / (
        (0.2166 * (pc / (ift * abs(np.cos(np.radians(theta))))) * (
            perm / phit)**(0.5))**(b0 * np.log10(1 + swb**-1) / 3))
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
    required_cols = ['WELL_NAME', 'DEPTH', 'PHIT', 'CPORE', 'CPERM', 'CORE_ID']
    for col in required_cols:
        if col not in df.columns:
            raise AssertionError(f"Missing required column: {col}")

    shift_summaries = []
    return_df = pd.DataFrame()
    for well, data in tqdm(df.groupby('WELL_NAME'), desc='Correlating core depths'):
        core_data = data[['DEPTH', 'CORE_ID', 'CPORE', 'CPERM']].dropna().sort_values('DEPTH').reset_index(drop=True)

        if core_data.empty:
            return_df = pd.concat([return_df, data])
            continue

        log_data = data[['DEPTH', 'PHIT']].dropna().sort_values('DEPTH').reset_index(drop=True)
        tqdm.write(f'Processing {well}: {len(core_data)} core data and {len(log_data)} log data')

        # Create a window of log data around core points to speed up DTW
        core_indices_in_log = np.searchsorted(log_data.DEPTH, core_data.DEPTH, side='left')
        window = 1
        window_indices = core_indices_in_log[:, None] + np.arange(-window, window + 1)
        window_indices = np.clip(window_indices, 0, len(log_data) - 1)
        unique_indices = np.unique(window_indices.flatten())
        log_data_subset = log_data.iloc[unique_indices].reset_index(drop=True)

        # Extract the numpy arrays for the DTW algorithm
        phit_vals = log_data_subset['PHIT'].values
        cpore_vals = core_data['CPORE'].values

        alignment = dtw(cpore_vals, phit_vals,
                        distance_only=False,
                        keep_internals=True,
                        step_pattern="symmetric2")

        # The alignment object contains the mapping between indices
        core_indices = alignment.index1  # Indices for the `core_data` DataFrame
        log_indices = alignment.index2   # Indices for the `log_data_subset` DataFrame

        # Create a detailed map from the alignment
        correction_map_df = pd.DataFrame({
            'ORIGINAL_DEPTH': core_data.loc[core_indices, 'DEPTH'].values,
            'CORE_ID_SHIFTED': core_data.loc[core_indices, 'CORE_ID'].values,
            'CPORE_SHIFTED': core_data.loc[core_indices, 'CPORE'].values,
            'CPERM_SHIFTED': core_data.loc[core_indices, 'CPERM'].values,
            'MATCHED_PHIT': log_data_subset.loc[log_indices, 'PHIT'].values,
            'DEPTH_CORRECTED': log_data_subset.loc[log_indices, 'DEPTH'].values
        })

        # Find the best match for each original core depth by minimizing porosity difference
        correction_map_df['PORO_DIFF'] = (correction_map_df['CPORE_SHIFTED'] - correction_map_df['MATCHED_PHIT']).abs()
        correction_map_df['DEPTH_SHIFT'] = round(
            correction_map_df.DEPTH_CORRECTED - correction_map_df.ORIGINAL_DEPTH, 3)
        best_matches = correction_map_df.sort_values('PORO_DIFF').drop_duplicates('ORIGINAL_DEPTH')

        if not best_matches.empty:
            summary = best_matches[['CORE_ID_SHIFTED', 'ORIGINAL_DEPTH', 'DEPTH_CORRECTED', 'DEPTH_SHIFT']].copy()
            summary.rename(columns={'CORE_ID_SHIFTED': 'CORE_ID'}, inplace=True)
            summary['WELL_NAME'] = well
            # Reorder columns for clarity
            summary = summary[['WELL_NAME', 'CORE_ID', 'ORIGINAL_DEPTH', 'DEPTH_CORRECTED', 'DEPTH_SHIFT']]
            shift_summaries.append(summary)

        # Merge the corrected core data back into the original well data
        df_corrected = pd.merge(
            data,
            best_matches[['DEPTH_CORRECTED', 'CORE_ID_SHIFTED', 'CPORE_SHIFTED', 'CPERM_SHIFTED']],
            left_on='DEPTH',
            right_on='DEPTH_CORRECTED',
            how='left'
        )

        # Remove original depths if being corrected
        ori_core_ids = df_corrected['CORE_ID'].dropna().unique()
        shifted_core_ids = df_corrected['CORE_ID_SHIFTED'].dropna().unique()

        mask_shifted = df_corrected['CORE_ID'].isin(shifted_core_ids)
        mask_ori = df_corrected['CORE_ID_SHIFTED'].isin(ori_core_ids)

        df_corrected.loc[mask_shifted, ['CORE_ID', 'CPORE', 'CPERM']] = np.nan
        df_corrected.loc[mask_ori, ['CORE_ID', 'CPORE', 'CPERM']] = df_corrected.loc[
            mask_ori, ['CORE_ID_SHIFTED', 'CPORE_SHIFTED', 'CPERM_SHIFTED']].values

        return_df = pd.concat([return_df, df_corrected])

    # After the loop, combine all summaries into a single DataFrame
    final_summary_df = pd.DataFrame()
    if shift_summaries:
        final_summary_df = pd.concat(shift_summaries, ignore_index=True)

    return return_df, final_summary_df
