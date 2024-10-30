from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

from quick_pp.utils import power_law_func as func

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


# Cross plots
def poroperm_xplot(poro, perm, a=None, b=None, label='', log_log=False):
    """Generate porosity-permeability cross plot.

    Args:
        poro (float): Core porosity (frac).
        perm (float): Core permeability (mD).
        a (float, optional): a constant in perm=a*poro^b. Defaults to None.
        b (float, optional): b constant in perm=a*poro^b. Defaults to None.
        label (str, optional): Label for the data group. Defaults to ''.
        log_log (bool, optional): Whether to plot log-log or not. Defaults to False.
    """
    sc = plt.scatter(poro, perm, marker='o', label=label)
    if a and b:
        line_color = sc.get_facecolors()[0]
        line_color[-1] = 0.5
        cpore = np.geomspace(0.05, 0.5, 30)
        plt.plot(cpore, func(cpore, a, b), color=line_color, linestyle='dashed')
    plt.xlabel('CPORE (frac)')
    plt.xlim(0, 0.5)
    plt.ylabel('CPERM (mD)')
    plt.yscale('log')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if log_log:
        plt.xscale('log')


def bvw_xplot(bvw, pc, a=None, b=None, label='', ylim=None, log_log=False):
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
    if a and b:
        line_color = sc[0].get_color() + '66'  # Set opacity to 0
        cbvw = np.linspace(0.05, 0.35, 30)
        plt.scatter(cbvw, func(cbvw, a, b), marker='x', color=line_color)
    plt.xlabel('BVW (frac)')
    plt.ylabel('Pc (psi)')
    plt.ylim(ylim) if ylim else plt.ylim(0.01, plt.gca().get_lines()[-1].get_ydata().max())
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if log_log:
        plt.xscale('log')
        plt.yscale('log')


def j_xplot(sw, j, a=None, b=None, label='', ylim=None, log_log=False):
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
    sc = plt.plot(sw, j, marker='s', label=label)
    if a and b:
        line_color = sc[0].get_color() + '80'  # Set opacity to 0.5
        csw = np.geomspace(0.1, 1.0, 30)
        plt.plot(csw, func(csw, a, b), color=line_color, linestyle='dashed')
    plt.xlabel('Sw (frac)')
    plt.xlim(0.01, 1)
    plt.ylabel('J')
    plt.ylim(ylim) if ylim else plt.ylim(0.01, plt.gca().get_lines()[-1].get_ydata().max())
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if log_log:
        plt.xscale('log')
        plt.yscale('log')


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
        popt, _ = curve_fit(func, sw, j)
        a = [round(c, 3) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except Exception as e:
        print(e)
        return 1, 1


def fit_poroperm_curve(poro, perm):
    """Estimate a and b constants of best fit given core porosity and permeability values.

    Args:
        poro (float): Core porosity (frac).
        perm (float): Core permeability (mD).

    Returns:
        tuple: a and b constants from the best-fit curve.
    """
    try:
        popt, _ = curve_fit(func, poro, perm)
        a = [round(c) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except Exception as e:
        print(e)
        return 1, 1


def leverett_j(pc, ift, perm, phit):
    """ Estimate Leverett J.

    Args:
        pc (float): Capillary pressure.
        ift (float): Interfacial tension.
        perm (float): Permeability.
        phit (float): Total porosity.

    Returns:
        float: Leverett J value.
    """
    return 0.216 * (pc / ift) * (perm / phit)**(0.5)


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


def sw_skelt_harrison():
    """TODO: Estimate water saturation based on Skelt-Harrison."""
    pass


def sw_cuddy(phit, h, a, b):
    """Estimate water saturation based on Cuddy's.

    Args:
        sw (float): Water saturation.
        phit (float): Total porosity.
        h (float): True vertical depth.
        a (float): Cementation exponent.
        b (float): Saturation exponent.

    Returns:
        float: Water saturation.
    """
    return a / phit * h**b


def sw_shf_leverett_j(perm, phit, depth, fwl, ift, gw, ghc, a, b, phie=None):
    """Estimate water saturation based on Leverett J function.

    Args:
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        ift (float): Interfacial tension (dynes/cm).
        gw (float): Gas density (psi/ft).
        ghc (float): Gas height (psi/ft).
        a (float): A constant from J function.
        b (float): B constant from J function.
        phie (float): Effective porosity (frac), required for clay bound water calculation. Defaults to None.

    Returns:
        float: Water saturation from saturation height function.
    """
    h = fwl - depth
    pc = h * (gw - ghc)
    shf = ((0.216 * (pc / ift) * (perm / phit)**(0.5)) / a)**(1 / b)
    # return np.where(shf > 1, 1, shf)
    return shf if not phie else shf * (1 - (phie / phit)) + (phie / phit)


def sw_shf_cuddy(phit, depth, fwl, gw, ghc, a, b):
    """Estimate water saturation based on Cuddy's saturation height function.

    Args:
        phit (float): Porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        a (float): Cementation exponent.
        b (float): Saturation exponent.

    Returns:
        float: Water saturation from saturation height function.
    """
    h = fwl - depth
    shf = (h * (gw - ghc) / a)**(1 / b) / phit
    # return np.where(shf > 1, 1, shf)
    return shf


def sw_shf_choo(perm, phit, phie, depth, fwl, ift, gw, ghc, b0=0.4):
    """Estimate water saturation based on Choo's saturation height function.

    Args:
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).
        phie (float): Effective porosity (frac).
        depth (float): True vertical depth (ft).
        fwl (float): Free water level in true vertical depth (ft).
        ift (float): Interfacial tension (dynes/cm).
        gw (float): Gas density (psi/ft).
        ghc (float): Gas height (psi/ft).
        b0 (float): _description_. Defaults to 0.4.

    Returns:
        float: Water saturation from saturation height function.
    """
    swb = 1 - (phie / phit)
    h = fwl - depth
    pc = h * (gw - ghc)
    shf = 10**((2 * b0 - 1) * np.log10(1 + swb**-1) + np.log10(1 + swb)) / (
        (0.2166 * (pc / ift) * (perm / phit)**(0.5))**(b0 * np.log10(1 + swb**-1) / 3))
    # return np.where(shf > 1, 1, shf)
    return shf
