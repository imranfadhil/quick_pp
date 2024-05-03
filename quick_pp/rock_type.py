import numpy as np


def fzi(k, phit):
    """FZI from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: FZI
    """
    return (0.0314 * (k / phit)**0.5) / (phit / (1 - phit))


def rqi(k, phit):
    """RQI from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: RQI
    """
    return (0.0314 * (k / phit)**0.5)


def rock_flag(vclw):
    """Rock type classification based on clay bound water volume.

    Args:
        vclw (float): Volume of clay bound water in fraction.

    Returns:
        str: Rock flag classification.
    """
    std = np.nanstd(vclw)
    standard_q = [0.05, 0.15, 0.5]
    proportion = [pct - std for pct in standard_q]
    proportion = standard_q if any([p < 0.15 for p in proportion]) else proportion
    q_list = np.nanquantile(vclw, proportion)
    rock_flag = np.where(vclw < q_list[0], 1,
                         np.where(vclw < q_list[1], 2,
                                  np.where(vclw < q_list[2], 3, 4)))
    return rock_flag
