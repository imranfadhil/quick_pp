from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

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
