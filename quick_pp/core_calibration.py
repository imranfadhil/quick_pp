from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


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


def j_xplot(sw, j, a, b, label):

    sc = plt.scatter(sw, j, marker='o', label=label)
    line_color = sc.get_facecolors()[0]
    line_color[-1] = 0.5
    csw = np.linspace(0.15, 1.0, 30)
    plt.scatter(csw, func(csw, a, b), marker='x', color=line_color)
    plt.xlabel('Sw (frac)')
    plt.ylabel('J')
    plt.legend()


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


def poroperm_xplot(poro, perm, a, b, label):

    sc = plt.scatter(poro, perm, marker='o', label=label)
    line_color = sc.get_facecolors()[0]
    line_color[-1] = 0.5
    cpore = np.linspace(0.05, 0.5, 30)
    plt.scatter(cpore, func(cpore, a, b), marker='x', color=line_color)
    plt.xlabel('CPORE (frac)')
    plt.ylabel('CPERM (mD)')
    plt.yscale('log')
    plt.legend()


def bvw_xplot(bvw, pc, a, b, label):

    sc = plt.scatter(bvw, pc, marker='o', label=label)
    line_color = sc.get_facecolors()[0]
    line_color[-1] = 0.5
    cbvw = np.linspace(0.05, 0.35, 30)
    plt.scatter(cbvw, func(cbvw, a, b), marker='x', color=line_color)
    plt.xlabel('BVW (frac)')
    plt.ylabel('Pc (psi)')
    plt.legend()


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
    return np.where(shf > 1, 1, shf)


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
    return np.where(shf > 1, 1, shf)