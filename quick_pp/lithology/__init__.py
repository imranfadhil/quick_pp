import numpy as np
import statistics
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler

from quick_pp import logger


def shale_volume_larinov_tertiary(igr):
    """
    Computes shale volume from gamma ray using Larinov's method for tertiary rocks.

    Parameters
    ----------
    igr : float
        Interval gamma ray [API].

    Returns
    -------
    vshale : float
        Shale volume [fraction].

    """
    logger.debug("Calculating shale volume using Larinov's method for tertiary rocks")
    vshale = 0.083 * (2**(3.7 * igr) - 1)
    logger.debug(f"Tertiary shale volume range: {vshale.min():.3f} - {vshale.max():.3f}")
    return vshale


def shale_volume_larinov_older(igr):
    """
    Computes shale volume from gamma ray using Larinov's method for older rocks.

    Parameters
    ----------
    igr : float
        Interval gamma ray [API].

    Returns
    -------
    vshale : float
        Shale volume [fraction].

    """
    logger.debug("Calculating shale volume using Larinov's method for older rocks")
    vshale = 0.33 * (2**(2 * igr) - 1)
    logger.debug(f"Older rocks shale volume range: {vshale.min():.3f} - {vshale.max():.3f}")
    return vshale


def shale_volume_steiber(igr):
    """
    Computes shale volume from gamma ray using Steiber's method.

    Parameters
    ----------
    igr : float
        Interval gamma ray [API].

    Returns
    -------
    vshale : float
        Shale volume [fraction].

    """
    logger.debug("Calculating shale volume using Steiber's method")
    vshale = igr / (3 - 2 * igr)
    logger.debug(f"Steiber shale volume range: {vshale.min():.3f} - {vshale.max():.3f}")
    return vshale


def gr_index(gr):
    """
    Computes gamma ray index from gamma ray.

    Parameters
    ----------
    gr : float
        Gamma ray [API].

    Returns
    -------
    gr_index : float
        Gamma ray index [API].

    """
    logger.debug("Computing gamma ray index with detrending and normalization")
    gr = np.where(np.isnan(gr), np.min(gr), gr)
    dtr_gr = detrend(gr, axis=0) + statistics.mean(gr)
    scaler = MinMaxScaler()
    igr = scaler.fit_transform(dtr_gr.reshape(-1, 1))
    logger.debug(f"Gamma ray index range: {igr.min():.3f} - {igr.max():.3f}")
    return igr
