import numpy as np
import matplotlib.pyplot as plt

from quick_pp.utils import min_max_line
from quick_pp.lithology import shale_volume_steiber

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


def calc_rqi(k, phit):
    """Calculate RQI (Rock Quality Index) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: RQI
    """
    return 0.0314 * (k / phit)**0.5


def calc_fzi(k, phit):
    """Calculate FZI (Flow Zone Indicator) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: FZI
    """
    return calc_rqi(k, phit) / (phit / (1 - phit))


def plot_fzi(cpore, cperm, title='Flow Zone Indicator (FZI)'):
    """Plot the FZI cross plot.

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction
    """
    # Plot the FZI cross plot
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='s')
    phit_points = np.geomspace(0.01, 1, 20)
    for fzi in np.arange(0.5, 5):
        perm_points = phit_points * ((phit_points * fzi) / (.0314 * (1 - phit_points)))**2
        plt.plot(phit_points, perm_points, linestyle='dashed', label=f'FZI={round(fzi, 3)}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(0, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(0.01, 10000)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')


def plot_ward(cpore, cperm, title="Ward's Plot"):
    """Plot Ward's plot to identify possible flow units.

    Args:
        cpore (float): Core porosity in fraction.
        cperm (float): Core permeability in mD.
        title (str, optional): Title of the plot. Defaults to "Ward's Plot".
    """
    fzi = calc_fzi(cperm, cpore)
    fzi = fzi[~np.isnan(fzi)]
    sorted_fzi = np.sort(fzi)
    log_fzi = np.log10(sorted_fzi**.5)
    p_log_fzi = (np.arange(1, len(log_fzi) + 1) - .5) / len(log_fzi)
    t = (-2 * np.log(p_log_fzi))**0.5
    zi = abs(t - ((2.30753 + .27061 * t) / (1 + 0.99229 * t + 0.04481 * t**2)))

    # Generate Ward's plot
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.scatter(log_fzi, zi, marker='s')

    plt.xlabel('log(fzi)')
    plt.ylabel('zi')


def plot_lorenz(cpore, cperm, title="Lorenz's Plot"):
    fzi = calc_fzi(cperm, cpore)
    fzi = fzi[~np.isnan(fzi)]
    sorted_fzi = np.sort(fzi)
    log_fzi = np.log10(sorted_fzi)
    cdf_log_fzi = np.cumsum(sorted_fzi) / np.sum(sorted_fzi)
    # Generate Ward's plot
    plt.figure(figsize=(10, 4))
    plt.title(title)
    plt.scatter(log_fzi, cdf_log_fzi, marker='s')

    plt.xlabel('log(fzi)')
    plt.ylabel('CDF FZI')


def plot_rfn(cpore, cperm, title='Lucia RFN'):
    """Plot the Rock Fabric Number (RFN) lines on porosity and permeability cross plot. The permeability (mD) is
    calculated based on Lucia-Jenkins, 2003 -
    > k = 10**(9.7892 - 12.0838 * log(RFN) + (8.6711 - 8.2965 * log(RFN)) * log(phi))

    Args:
        cpore (float): Critical porosity in v/v
        cperm (float): Critical permeability in mD
    """
    # Plot the RFN cross plot
    plt.figure(figsize=(5, 4))
    plt.title(title)
    plt.scatter(cpore, cperm, marker='s')
    pore_points = np.linspace(0, .6, 20)
    for rfn in np.arange(.5, 4.5, .5):
        perm_points = 10**(9.7892 - 12.0838 * np.log10(rfn) + (8.6711 - 8.2965 * np.log10(rfn)) * np.log10(pore_points))
        plt.plot(pore_points, perm_points, linestyle='dashed', label=f'RFN={rfn}')

    plt.xlabel('Porosity (frac)')
    plt.xlim(0, .5)
    plt.ylabel('Permeability (mD)')
    plt.ylim(0.01, 10000)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale('log')


def determine_rfn(phig, swirr):
    """Determine the Rock Fabric Number (RFN) based on Lucia-Jenkins, 2003.

    Args:
        phig (float): Inter-grain porosity in v/v
        swirr (float): Irreducible Water Saturation in v/v

    Returns:
        float: Rock Fabric Number (RFN)
    """
    return 10**((7.163 + 1.883 * np.log10(phig) + np.log10(swirr)) / (3.063 + 0.610 * np.log10(phig)))


def estimate_vsh_gr(gr, min_gr=None, max_gr=None, alpha=0.1):
    """Estimate volume of shale from gamma ray. If min_gr and max_gr are not provided,
    it will be automatically estimated.

    Args:
        gr (float): Gamma ray from well log.
        min_gr (float, optional): Minimum gamma ray value. Defaults to None.
        max_gr (float, optional): Maximum gamma ray value. Defaults to None.
        alpha (float, optional): Alpha value for min-max normalization. Defaults to 0.1.

    Returns:
        float: VSH_GR.
    """
    # Remove high outliers and forward fill missing values
    gr = np.where(gr <= np.nanmean(gr) + 1.5 * np.nanstd(gr), gr, np.nan)
    mask = np.isnan(gr)
    idx = np.where(~mask, np.arange(len(mask)), 0)
    np.maximum.accumulate(idx, axis=0, out=idx)
    gr = gr[idx]

    # Normalize gamma ray
    if not max_gr or (not min_gr and min_gr != 0):
        _, max_gr = min_max_line(gr, alpha)
        gri = np.where(gr < max_gr, gr / max_gr, 1)
    else:
        gri = (gr - min_gr) / (max_gr - min_gr)
        gri = np.where(gri < 1, gri, 1)
    return shale_volume_steiber(gri).flatten()


def estimate_vsh_dn(phin, phid, phin_sh=0.35, phid_sh=0.05):
    """Estimate volume of shale from neutron porosity and density porosity.

    Args:
        phin (float): Neutron porosity in fraction.
        phid (float): Density porosity in fraction.
        phin_sh (float, optional): Neutron porosity for shale. Defaults to 0.35.
        phid_sh (float, optional): Density porosity for shale. Defaults to 0.05.

    Returns:
        float: Volume of shale.
    """
    return (phin - phid) / (phin_sh - phid_sh)


def rock_typing(curve, cut_offs=[.1, .3, .4], higher_is_better=True):
    """Rock typing based on cutoffs.

    Args:
        curve (float): Curve to be used for rock typing.
        cut_offs (list, optional): 3 cutoffs to group the curve into 4 rock types. Defaults to [.1, .3, .4].
        higher_is_better (bool, optional): Whether higher value of curve is better quality or not. Defaults to True.

    Returns:
        float: Rock type.
    """
    rock_type = [4, 3, 2, 1] if higher_is_better else [1, 2, 3, 4]
    return np.where(np.isnan(curve), 4, np.where(
        curve < cut_offs[0], rock_type[0],
                    np.where(curve < cut_offs[1], rock_type[1],
                             np.where(curve < cut_offs[2], rock_type[2], rock_type[3]))))
