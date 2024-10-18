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
    """Estimate average matrix density based on dry sand, dry silt, dry clay, dry calcite and
    dry dolomite volume and density of each.

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


def dt_matrix(vsand=0, vsilt=0, vclay=0, vcalc=0, vdolo=0, vheavy=0,
              dt_sand: float = None, dt_silt: float = None, dt_clay: float = None,
              dt_calc: float = None, dt_dolo: float = None, dt_heavy: float = 0):
    """Estimate average matrix sonic transit time based on dry sand, dry silt dry calcite and
    dry dolomite volume and transit time of each.

    Args:
        vsand (float): Volume of sand.
        vsilt (float): Volume of silt.
        vclay (float): Volume of clay.
        vcalc (float): Volume of calcite.
        vdolo (float): Volume of dolomite.
        vheavy (float): Volume of heavy minerals.
        dt_sand (float, optional): Sonic transit time of sand in us/ft. Defaults to None.
        dt_silt (float, optional): Sonic transit time of silt in us/ft. Defaults to None.
        dt_clay (float, optional): Sonic transit time of clay in us/ft. Defaults to None.
        dt_calc (float, optional): Sonic transit time of calcite in us/ft. Defaults to None.
        dt_dolo (float, optional): Sonic transit time of dolomite in us/ft. Defaults to None.
        dt_heavy (float, optional): Sonic transit time of heavy minerals in us/ft. Defaults to 0.

    Returns:
        float: Matrix sonic transit time in us/ft.
    """
    minerals_log_value = Config.MINERALS_LOG_VALUE
    dt_sand = dt_sand or minerals_log_value['DTC_QUARTZ']
    dt_silt = dt_silt or minerals_log_value['DTC_SILT']
    dt_clay = dt_clay or minerals_log_value['DTC_SH']
    dt_calc = dt_calc or minerals_log_value['DTC_CALCITE']
    dt_dolo = dt_dolo or minerals_log_value['DTC_DOLOMITE']
    return vsand * dt_sand + vsilt * dt_silt + vclay * dt_clay + vcalc * dt_calc + vdolo * dt_dolo + vheavy * dt_heavy


def sonic_porosity_wyllie(dt, dt_matrix, dt_fluid):
    """
    Computes sonic porosity based on Wyllie's equation from interval, matrix, and fluid transit time.

    Parameters
    ----------
    dt : float
        Interval transit time [us/ft].
    dt_matrix : float
        Matrix transit time [us/ft]. Sandstone: 51-55, Limestone: 43-48, Dolomite: 43-39, Shale: 60-170.
    dt_fluid : float
        Fluid transit time [us/ft]. Water: 190, Oil: 240, Gas: 630.

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
        Matrix transit time [us/ft]. Sandstone: 51-55, Limestone: 43-48, Dolomite: 43-39, Shale: 60-170.
    c : float
        constant (0.62 to 0.7).

    Returns
    -------
    porosity : float
        Sonic porosity [fraction].

    """
    return c * (dt - dt_matrix) / dt_matrix


def neu_den_xplot_poro_pt(
        nphi: float, rhob: float, model: str = 'ssc',
        dry_min1_point: tuple = None,
        dry_silt_point: tuple = None,
        dry_clay_point: tuple = None,
        fluid_point: tuple = (1.0, 1.0)):
    """Calculate porosity given a pair of neutron porosity and bulk density data point.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        model (str, optional): Lithology model, either 'ssc' (Sand Silt Clay) or 'ss' (Sand Shale). Defaults to 'ssc'.
        reservoir (bool, optional): Either in reservoir or non-reservoir section. Defaults to False.
        dry_min1_point (tuple): Neutron porosity and bulk density of mineral 1 point.
        dry_silt_point (tuple): Neutron porosity and bulk density of dry silt point.
        dry_clay_point (tuple): Neutron porosity and bulk density of dry clay point.
        fluid_point (tuple): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).

    Returns:
        float: Total porosity.
    """
    assert model in ['ssc', 'ss', 'carb'], "Please specify either 'ssc', 'ss' or 'carb' model."

    A = dry_min1_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point

    phit = []
    if model == 'ssc':
        # Check if the point is in the reservoir or non-reservoir section
        thold_pt = line_intersection((A, C), (D, B))
        thold_line = length_a_b(thold_pt, A)
        proj_pt = line_intersection((A, C), (D, (nphi, rhob)))
        proj_line = length_a_b(proj_pt, A)
        if proj_line < thold_line:
            m = (A[1] - B[1]) / (A[0] - B[0])
        else:
            m = (C[1] - B[1]) / (C[0] - B[0])

        c = rhob - m * nphi
        iso_poro_pt = line_intersection(((0, c), (nphi, rhob)), (D, B))
        iso_poro_line = length_a_b(iso_poro_pt, B)
        poro_line = length_a_b(D, B)
        phit = iso_poro_line / poro_line
    else:
        m = (A[1] - C[1]) / (A[0] - C[0])
        c = rhob - m * nphi
        iso_poro_pt = line_intersection(((0, c), (nphi, rhob)), (D, A))
        iso_poro_line = length_a_b(iso_poro_pt, A)
        poro_line = length_a_b(D, A)

    phit = iso_poro_line / poro_line
    return phit


def neu_den_xplot_poro(nphi, rhob, model: str = 'ssc',
                       dry_min1_point: tuple = None,
                       dry_silt_point: tuple = None,
                       dry_clay_point: tuple = None,
                       fluid_point: tuple = (1.0, 1.0)):
    """Calculate porosity given neutron porosity and bulk density logs.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        model (str, optional): Lithology model, either 'ssc' (Sand Silt Clay), 'ss' (Sand Shale) or 'carb' (Carbonate).
                               Defaults to 'ssc'.
        reservoir (bool, optional): Either in reservoir or non-reservoir section. Defaults to False.
        dry_min1_point (tuple): Neutron porosity and bulk density of dry min1 point. Defaults to None.
        dry_silt_point (tuple): Neutron porosity and bulk density of dry silt point. Defaults to None.
        dry_clay_point (tuple): Neutron porosity and bulk density of dry clay point. Defaults to None.
        fluid_point (tuple): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).

    Returns:
        float: Total porosity.
    """
    assert model in ['ssc', 'ss', 'carb'], "Please specify either 'ssc', 'ss' or 'carb' model."

    A = dry_min1_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))

    phit = np.empty(0)
    for i, point in enumerate(E):
        if model == 'ssc':
            phit = np.append(phit, neu_den_xplot_poro_pt(point[0], point[1], 'ssc', A, B, C, D))
        else:
            phit = np.append(phit, neu_den_xplot_poro_pt(point[0], point[1], 'ss', A, (0, 0), C, D))

    return phit


def porosity_correction_averaging(nphi, rhob, rho_ma=2.65, method='weighted'):
    """Correct porosity using averaging method.
    Weighted average: (2 * dphi + nphi) / 3
    Arithmetic average: (dphi + nphi) / 2
    Gaymard average: sqrt((dphi**2 + nphi**2) / 2)
    Gas average: sqrt((dphi**2 + nphi**2) / 2)

    Args:
        nphi (float): Neutron porosity.
        dphi (float): Density porosity.
        method (str, optional): Averaging method selection from 'weighted', 'arithmetic' or 'gaymard'.
         Defaults to 'weighted'.

    Returns:
        float: Corrected porosity.
    """
    assert method in ['weighted', 'arithmetic', 'gaymard', 'gas'], "method must be either \
        'weighted', 'arithmetic', 'gaymard' or 'gas"
    dphi = density_porosity(rhob, rho_ma, 1.0)
    if method == 'weighted':
        return (2 * dphi + nphi) / 3
    elif method == 'arithmetic':
        return (dphi + nphi) / 2
    elif method == 'gaymard':
        return np.sqrt((dphi**2 + nphi**2) / 2)
    elif method == 'gas':
        return ((nphi**2 + dphi**2) / 2)**0.5


def porosity_trend(tvdss, unit='ft'):
    """Calculate porosity trend based on TVDSS (Schmoker, 1982)

    Args:
        tvdss (float): True Vertical Depth Subsea.

    Returns:
        float: Porosity trend.
    """
    assert unit in ['ft', 'm'], 'Please specify either ft or m as unit.'
    if unit == 'ft':
        return 41.73 * np.exp(-tvdss / 8197)
    else:
        return 41.73 * np.exp(-tvdss / 2498)
