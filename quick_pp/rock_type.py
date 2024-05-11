import numpy as np
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler


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


def vsh_gr(gr):
    """Estimate volume of shale from gamma ray.

    Args:
        gr (float): Gamma ray from well log.

    Returns:
        float: VSH_GR.
    """
    dtr_gr = detrend(gr, axis=0) + np.nanmean(gr)
    scaler = MinMaxScaler()
    gri = scaler.fit_transform(np.reshape(dtr_gr, (-1, 1)))
    # Apply Steiber equation
    return gri / (3 - 2 * gri)


def rock_typing(curve, cut_offs=[.33, .45, .7], higher_is_better=True):
    rock_type = [1, 2, 3, 4] if higher_is_better else [4, 3, 2, 1]
    return np.where(curve < cut_offs[0], rock_type[0],
                    np.where(curve < cut_offs[1], rock_type[1],
                             np.where(curve < cut_offs[2], rock_type[2], rock_type[3])))
