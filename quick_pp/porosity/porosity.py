import numpy as np

from ..config import Config
from ..utils import length_a_b, line_intersection


def normalize_volumetric(vsand, vsilt, vclay, phit):
    # Normalize the volumetrics
    vmatrix = 1 - phit
    vsand = vsand * vmatrix
    vsilt = vsilt * vmatrix
    vclay = vclay * vmatrix

    return vsand, vsilt, vclay


def effective_porosity(phit, phi_shale, vshale):
    """
    Computes effective porosity from total porosity, total porosity of shale and shale volume.

    Parameters
    ----------
    phit : float
        Total porosity [fraction].
    phi_vsh : float
        Total porosity of shale [fraction].
    vshale : float
        Shale volume [fraction].

    Returns
    -------
    porosity : float
        Effective porosity [fraction].

    """
    return phit - (vshale * phi_shale)


def clay_porosity(rho_clw: np.array, rho_dry_clay: float = 2.72, rho_fluid: float = 1.0):
    rho_dry_clay = rho_dry_clay or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"][1]
    return (rho_dry_clay - rho_clw) / (rho_dry_clay - rho_fluid)


def shale_porosity(vshale, phi_shale):
    """
    Computes shale porosity from shale volume and total porosity of shale.

    Parameters
    ----------
    vshale : float
        Shale volume [fraction].
    phi_vsh : float
        Total porosity of shale [fraction].

    Returns
    -------
    porosity : float
        Shale porosity [fraction].

    """
    return vshale * phi_shale


def rho_matrix(vsand, vsilt, vclay, rho_sand: float = None, rho_silt: float = None, rho_clay: float = None):
    """Estimate average matrix density based on dry sand, dry silt and dry clay volume and density.

    Args:
        vsand (float or array): Volume of dry sand.
        vsilt (float or array): Volume of dry silt.
        vclay (float or array): Volume of dry clay.
        rho_sand (float, optional): _description_. Defaults to None.
        rho_silt (float, optional): _description_. Defaults to None.
        rho_clay (float, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    ssc_endpoints = Config.SSC_ENDPOINTS
    rho_sand = rho_sand or ssc_endpoints['DRY_SAND_POINT'][1]
    rho_silt = rho_silt or ssc_endpoints['DRY_SILT_POINT'][1]
    rho_clay = rho_clay or ssc_endpoints['DRY_CLAY_POINT'][1]
    return vsand * rho_sand + vsilt * rho_silt + vclay * rho_clay


def density_porosity(rhob, rho_matrix, rho_fluid: float = 1.0):
    """Computes density porosity from bulk, matrix and fluid densities

    Args:
        rhob (_type_): _description_
        rho_matrix (_type_): _description_
        rho_fluid (float, optional): _description_. Defaults to 1.0.

    Returns:
        _type_: Density porosity [fraction]
    """
    return (rho_matrix - rhob) / (rho_matrix - rho_fluid)


def sonic_porosity_wyllie(dt, dt_matrix, dt_fluid):
    """
    Computes sonic porosity based on Wyllie's equation from interval, matrix, and fluid transit time.

    Parameters
    ----------
    dt : float
        Interval transit time [us/ft].
    dt_matrix : float
        Matrix transit time [us/ft].
    dt_fluid : float
        Fluid transit time [us/ft].

    Returns
    -------
    porosity : float
        Sonic porosity [fraction].

    """
    return (dt - dt_matrix) / (dt_fluid - dt_matrix)


def sonic_porosity_hunt_raymer(dt, dt_matrix, c):
    """
    Computes sonic porosity based on Hunt-Raymer's equation from interval, matrix and transit time.

    Parameters
    ----------
    dt : float
        Interval transit time [us/ft].
    dt_matrix : float
        Matrix transit time [us/ft].
    c : float
        constant (0.62 to 0.7).

    Returns
    -------
    porosity : float
        Sonic porosity [fraction].

    """
    return c * (dt - dt_matrix) / dt_matrix


def neu_den_xplot_poro_pt(
        nphi: float, rhob: float, model: str = 'ssc', reservoir: bool = False,
        dry_sand_point: tuple = None,
        dry_silt_point: tuple = None,
        dry_clay_point: tuple = None,
        fluid_point: tuple = (1.0, 1.0)):
    assert model in ['ssc', 'ss'], f"'{model}' model is not available."
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point

    phit = []
    if model == 'ssc':
        poro_line = length_a_b(D, B)
        if reservoir:
            m = (A[1] - B[1]) / (A[0] - B[0])
        else:
            m = (C[1] - B[1]) / (C[0] - B[0])

        c = rhob - m * nphi
        iso_poro_pt = line_intersection(((0, c), (nphi, rhob)), (D, B))
        iso_poro_line = length_a_b(iso_poro_pt, B)
        phit = iso_poro_line / poro_line
        return phit

    else:
        poro_line = length_a_b(D, A)
        m = (A[1] - C[1]) / (A[0] - C[0])
        c = rhob - m * nphi
        iso_poro_pt = line_intersection(((0, c), (nphi, rhob)), (D, A))
        iso_poro_line = length_a_b(iso_poro_pt, A)
        phit = iso_poro_line / poro_line

    return phit


def neu_den_xplot_poro(nphi, rhob, model: str = 'ssc', reservoir=True,
                       dry_sand_point: tuple = None,
                       dry_silt_point: tuple = None,
                       dry_clay_point: tuple = None,
                       fluid_point: tuple = (1.0, 1.0)):
    """_summary_

    Args:
        nphi (_type_): _description_
        rhob (_type_): _description_
        model (str, optional): _description_. Defaults to 'ssc'.
        reservoir (bool, optional): _description_. Defaults to True.
        dry_sand_point (tuple, optional): _description_. Defaults to None.
        dry_silt_point (tuple, optional): _description_. Defaults to None.
        dry_clay_point (tuple, optional): _description_. Defaults to None.
        fluid_point (tuple, optional): _description_. Defaults to (1.0, 1.0).

    Returns:
        _type_: _description_
    """
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))

    phit = []
    for i, point in enumerate(E):
        if model == 'ssc':
            phit.append(neu_den_xplot_poro_pt(point[0], point[1], 'ssc', reservoir, A, B, C, D))
        else:
            phit.append(neu_den_xplot_poro_pt(point[0], point[1], 'ss', reservoir, A, (0, 0), C, D))

    return phit


def neu_den_poro(nphi, rhob, rho_ma=2.65, method='simplified'):
    """_summary_

    Args:
        nphi (_type_): _description_
        rhob (_type_): _description_
        rho_ma (float or array, optional): _description_. Defaults to 2.65.

    Returns:
        _type_: _description_
    """
    assert method in ['simple', 'emperical', 'gas'], 'Please select either simple, emperical or gas method.'
    phid = density_porosity(rhob, rho_ma, 1.0)
    if method == 'simple':
        return (nphi + phid) / 2
    elif method == 'emperical':
        return (1 / 3) * nphi + (2 / 3) * phid
    elif method == 'gas':
        return ((nphi**2 + phid**2) / 2)**0.5
