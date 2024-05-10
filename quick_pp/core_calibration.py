from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np


def saturation_height_function():
    """TODO: Saturation height function.
    """
    pass


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


def j_func(x, a, b):
    return a * x**-b


def fit_j_curve(sw, j):
    """_summary_

    Args:
        sw (_type_): _description_
        j (_type_): _description_
    """
    popt, _ = curve_fit(j_func, sw, j)
    a = [round(c, 3) for c in popt][0]
    b = [round(c, 3) for c in popt][1]
    return a, b


def j_xplot(sw, j, a, b, label):

    sc = plt.scatter(sw, j, marker='o', label=label)
    csw = np.linspace(0.15, 1.0, 30)
    plt.scatter(csw, j_func(csw, a, b), marker='x', color=sc.get_facecolors()[0])
    plt.xlabel('Sw (frac)')
    plt.ylabel('J')
    plt.legend()


def poroperm_func(x, a, b):
    return a * x**b


def fit_poroperm_curve(poro, perm):
    """_summary_

    Args:
        poro (_type_): _description_
        perm (_type_): _description_
    """
    popt, _ = curve_fit(poroperm_func, poro, perm)
    a = [round(c) for c in popt][0]
    b = [round(c, 3) for c in popt][1]
    return a, b


def poroperm_xplot(poro, perm, a, b, label):

    sc = plt.scatter(poro, perm, marker='o', label=label)
    cpore = np.linspace(0.05, 0.5, 30)
    sc = plt.scatter(cpore, poroperm_func(cpore, a, b), marker='x', color=sc.get_facecolors()[0])
    plt.xlabel('CPORE (frac)')
    plt.ylabel('CPERM (mD)')
    plt.yscale('log')
    plt.legend()
