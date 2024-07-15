import numpy as np

from quick_pp.config import Config
from quick_pp.utils import length_a_b, line_intersection


def normalize_volumetric(vsand, vsilt, vclay, phit):
    """Normalize lithology (vsand, vsilt and vclay) given total porosity.

    Args:
        vsand (float): Volume of sand in fraction (v/v).
        vsilt (float): Volume of silt in fraction (v/v).
        vclay (float): Volume of clay in fraction (v/v).
        phit (float): Total porosity in fraction (v/v).

    Returns:
        float: Normalized vsand, vsilt and vclay.
    """
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
    """Calculate clay porosity given bulk density of wet clay line.

    Args:
        rho_clw (float): Bulk density of wet clay line.
        rho_dry_clay (float, optional): Bulk density of dry clay. Defaults to 2.72.
        rho_fluid (float, optional): Bulk density of fluid. Defaults to 1.0.

    Returns:
        float: Clay porosity.
    """
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


def rho_matrix(vsand=0, vsilt=0, vclay=0, vcalc=0, vdolo=0, vheavy=0,
               rho_sand: float = None, rho_silt: float = None, rho_clay: float = None,
               rho_calc: float = None, rho_dolo: float = None, rho_heavy: float = 0):
    """Estimate average matrix density based on dry sand, dry silt and dry clay volume and density.

    Args:
        vsand (float): Volume of sand.
        vsilt (float): Volume of silt.
        vclay (float): Volume of clay.
        vcalc (float): Volume of calcite.
        vdolo (float): Volume of dolomite.
        vheavy (float): Volume of heavy minerals.
        rho_sand (float, optional): Density of sand in g/cc. Defaults to None.
        rho_silt (float, optional): Density of silt in g/cc. Defaults to None.
        rho_clay (float, optional): Density of clay in g/cc. Defaults to None.
        rho_calc (float, optional): Density of calcite in g/cc. Defaults to None.
        rho_dolo (float, optional): Density of dolomite in g/cc. Defaults to None.
        rho_heavy (float, optional): Density of heavy minerals in g/cc. Defaults to 0.

    Returns:
        float: Matrix density in g/cc.
    """
    minerals_log_value = Config.MINERALS_LOG_VALUE
    rho_sand = rho_sand or minerals_log_value['RHOB_QUARTZ']
    rho_silt = rho_silt or minerals_log_value['RHOB_SILT']
    rho_clay = rho_clay or minerals_log_value['RHOB_SH']
    rho_calc = rho_calc or minerals_log_value['RHOB_CALCITE']
    rho_dolo = rho_dolo or minerals_log_value['RHOB_DOLOMITE']
    return vsand * rho_sand + vsilt * rho_silt + vclay * rho_clay + \
        vcalc * rho_calc + vdolo * rho_dolo + vheavy * rho_heavy


def density_porosity(rhob, rho_matrix, rho_fluid: float = 1.0):
    """Computes density porosity from bulk, matrix and fluid densities

    Args:
        rhob (float): Bulk density log in g/cc.
        rho_matrix (float): Matrix density in g/cc.
        rho_fluid (float, optional): Density of fluid in g/cc. Defaults to 1.0 g/cc.

    Returns:
        float: Density porosity [fraction]
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
    """Calculate porosity given a pair of neutron porosity and bulk density data point.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        model (str, optional): Lithology model, either 'ssc' (Sand Silt Clay) or 'ss' (Sand Shale). Defaults to 'ssc'.
        reservoir (bool, optional): Either in reservoir or non-reservoir section. Defaults to False.
        dry_sand_point (tuple): Neutron porosity and bulk density of dry sand point.
        dry_silt_point (tuple): Neutron porosity and bulk density of dry silt point.
        dry_clay_point (tuple): Neutron porosity and bulk density of dry clay point.
        fluid_point (tuple): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).

    Returns:
        float: Total porosity.
    """
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
    """Calculate porosity given neutron porosity and bulk density logs.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        model (str, optional): Lithology model, either 'ssc' (Sand Silt Clay) or 'ss' (Sand Shale). Defaults to 'ssc'.
        reservoir (bool, optional): Either in reservoir or non-reservoir section. Defaults to False.
        dry_sand_point (tuple): Neutron porosity and bulk density of dry sand point. Defaults to None.
        dry_silt_point (tuple): Neutron porosity and bulk density of dry silt point. Defaults to None.
        dry_clay_point (tuple): Neutron porosity and bulk density of dry clay point. Defaults to None.
        fluid_point (tuple): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).

    Returns:
        float: Total porosity.
    """
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))

    phit = np.empty(0)
    for i, point in enumerate(E):
        if model == 'ssc':
            phit = np.append(phit, neu_den_xplot_poro_pt(point[0], point[1], 'ssc', reservoir, A, B, C, D))
        else:
            phit = np.append(phit, neu_den_xplot_poro_pt(point[0], point[1], 'ss', reservoir, A, (0, 0), C, D))

    return phit


def neu_den_poro(nphi, rhob, rho_ma=2.65, method='simplified'):
    """Calculate porosity based 'simple', 'emperical' or 'gas' method, given neutron porosity and bulk density logs.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        rho_ma (float, optional): Matrix bulk density. Defaults to 2.65.

    Returns:
        float: Porosity.
    """
    assert method in ['simple', 'emperical', 'gas'], 'Please select either simple, emperical or gas method.'
    phid = density_porosity(rhob, rho_ma, 1.0)
    if method == 'simple':
        return (nphi + phid) / 2
    elif method == 'emperical':
        return (1 / 3) * nphi + (2 / 3) * phid
    elif method == 'gas':
        return ((nphi**2 + phid**2) / 2)**0.5
