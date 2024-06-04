from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


def leverett_j_function(pc, ift, perm, phit):
    """Leverett J function.

    Args:
        pc (float): Capillary pressure.
        ift (float): Interfacial tension.
        perm (float): Permeability.
        phit (float): Total porosity.

    Returns:
        float: Leverett J function.
    """
    return 0.216 * (pc / ift) * (perm / phit)**(0.5)


def pseudo_leverett_j_function(pc, ift, perm, phit):
    """Pseudo-Leverett J function.

    Args:
        pc (float): Capillary pressure.
        ift (float): Interfacial tension.
        perm (float): Permeability.
        phit (float): Total porosity.

    Returns:
        float: Pseudo-Leverett J function.
    """
    pass


def sw_leverett_j(pc, ift, perm, phit, a, b):
    """Saturation J function.

    Args:
        pc (float): Capillary pressure.
        ift (float): Interfacial tension.
        perm (float): Permeability.
        phit (float): Total porosity.

    Returns:
        float: Saturation J function.
    """
    return (a / (0.216 * (pc / ift) * (perm / phit)**(0.5)))**(1 / b)


def sw_skelt_harrison():
    pass


def sw_cuddy(phit, h, a, b):
    """Cuddy's saturation model.

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


def func(x, a, b):
    return a * x**b


def fit_j_curve(sw, j):
    """_summary_

    Args:
        sw (_type_): _description_
        j (_type_): _description_
    """
    try:
        popt, _ = curve_fit(func, sw, j)
        a = [round(c, 3) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except Exception as e:
        print(e)
        return 1, 1


def j_xplot(sw, j, a=None, b=None, label='', log_log=False, ylim=None):

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


def fit_poroperm_curve(poro, perm):
    """_summary_

    Args:
        poro (_type_): _description_
        perm (_type_): _description_
    """
    try:
        popt, _ = curve_fit(func, poro, perm)
        a = [round(c) for c in popt][0]
        b = [round(c, 3) for c in popt][1]
        return a, b
    except Exception as e:
        print(e)
        return 1, 1


def poroperm_xplot(poro, perm, a=None, b=None, label='', log_log=False):

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


def estimate_hafwl(sw, poro, perm, ift, gw, ghc, a, b):
    """Estimate the height (feet) above free water level.

    Args:
        sw (float): Water saturation (frac).
        poro (float): Porosity (frac).
        perm (float): Permeability (mD).
        ift (float): Interfacial tension (dynes/cm).
        gw (float): Gas density (psi/ft).
        ghc (float): Gas height (psi/ft).
        a (float): A constant from J function.
        b (float): B constant from J function.

    Returns:
        float: Height above free water level.
    """
    # Calculate the saturation J function.
    j = a * sw**b
    return (j / 0.216) * (ift / (gw - ghc)) * (perm / poro)**0.5


def sw_shf_leverett_j(perm, poro, depth, fwl, ift, gw, ghc, a, b):
    """Saturation height function.

    Args:
        perm (float): Permeability (mD).
        poro (float): Porosity (frac).
        depth (_type_): _description_
        fwl (_type_): _description_
        ift (float): Interfacial tension (dynes/cm).
        gw (float): Gas density (psi/ft).
        ghc (float): Gas height (psi/ft).
        a (float): A constant from J function.
        b (float): B constant from J function.

    Returns:
        _type_: _description_
    """    """TODO: Saturation height function.
    """
    h = fwl - depth
    pc = h * (gw - ghc)
    shf = ((0.216 * (pc / ift) * (perm / poro)**(0.5)) / a)**(1 / b)
    # return np.where(shf > 1, 1, shf)
    return shf


def sw_shf_cuddy(poro, depth, fwl, gw, ghc, a, b):
    """Saturation height function.

    Args:
        poro (float): Porosity (frac).
        depth (_type_): _description_
        fwl (_type_): _description_
        a (float): Cementation exponent.
        b (float): Saturation exponent.

    Returns:
        _type_: _description_
    """
    h = fwl - depth
    shf = (h * (gw - ghc) / a)**(1 / b) / poro
    # return np.where(shf > 1, 1, shf)
    return shf


def sw_shf_choo(perm, phit, phie, depth, fwl, ift, gw, ghc, b0=0.4):
    """Saturation height function.

    Args:
        perm (float): Permeability (mD).
        phit (float): Total porosity (frac).
        phie (float): Effective porosity (frac).
        depth (float): _description_
        fwl (float): _description_
        ift (float): Interfacial tension (dynes/cm).
        gw (float): Gas density (psi/ft).
        ghc (float): Gas height (psi/ft).
        b0 (float): _description_. Defaults to 0.4.

    Returns:
        _type_: _description_
    """
    swb = 1 - (phie / phit)
    h = fwl - depth
    pc = h * (gw - ghc)
    shf = 10**((2 * b0 - 1) * np.log10(1 + swb**-1) + np.log10(1 + swb)) / (
        (0.2166 * (pc / ift) * (perm / phit)**(0.5))**(b0 * np.log10(1 + swb**-1) / 3))
    # return np.where(shf > 1, 1, shf)
    return shf
