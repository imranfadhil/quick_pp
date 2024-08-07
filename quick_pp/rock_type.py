import numpy as np
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler
from quick_pp.utils import min_max_line


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


def estimate_vsh_gr(gr, alpha=0.1):
    """Estimate volume of shale from gamma ray.

    Args:
        gr (float): Gamma ray from well log.

    Returns:
        float: VSH_GR.
    """
    gr = np.where(np.isnan(gr), np.nanmedian(gr), gr)
    dtr_gr = detrend(gr, axis=0) + np.nanmean(gr)
    _, max_gr = min_max_line(dtr_gr, alpha)
    gri = dtr_gr / max_gr
    # Apply min-max scaling
    gri = MinMaxScaler().fit_transform(gri.reshape(-1, 1)).flatten()
    # Apply Steiber equation
    return gri / (3 - 2 * gri)


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
