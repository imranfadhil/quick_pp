import numpy as np
from quick_pp.utils import min_max_line
from quick_pp.lithology import shale_volume_steiber


def fzi(k, phit):
    """Calculate FZI (Flow Zone Indicator) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: FZI
    """
    return (0.0314 * (k / phit)**0.5) / (phit / (1 - phit))


def rqi(k, phit):
    """Calculate RQI (Rock Quality Index) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        k (float): Permeability in mD
        phit (float): Total porosity in fraction

    Returns:
        float: RQI
    """
    return (0.0314 * (k / phit)**0.5)


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
