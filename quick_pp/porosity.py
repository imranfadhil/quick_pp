import numpy as np

from quick_pp.config import Config
from quick_pp.utils import length_a_b, line_intersection
from quick_pp import logger


def normalize_volumetric(phit, **volumetrics):
    """Normalize lithology given total porosity.

    Args:
        phit (float): Total porosity in fraction (v/v).
        **volumetrics: Keyword arguments representing volumetric fractions (v/v).

    Returns:
        dict: Normalized volumetric fractions.
    """
    logger.debug("Normalizing volumetrics with total porosity")
    # Normalize the volumetrics
    vmatrix = 1 - phit
    normalized_volumetrics = {key: value * vmatrix for key, value in volumetrics.items()}
    logger.debug(f"Normalized volumetrics: {list(normalized_volumetrics.keys())}")
    return normalized_volumetrics


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
    logger.debug(f"Calculating effective porosity with shale porosity: {phi_shale:.3f}")
    phie = phit - (vshale * phi_shale)
    logger.debug(f"Effective porosity range: {phie.min():.3f} - {phie.max():.3f}")
    return phie


def clay_porosity(rho_clw: np.ndarray, rho_dry_clay: float = 2.72, rho_fluid: float = 1.0):
    """Calculate clay porosity given bulk density of wet clay line.

    Args:
        rho_clw (float): Bulk density of wet clay line.
        rho_dry_clay (float, optional): Bulk density of dry clay. Defaults to 2.72.
        rho_fluid (float, optional): Bulk density of fluid. Defaults to 1.0.

    Returns:
        float: Clay porosity.
    """
    logger.debug(f"Calculating clay porosity with dry clay density: {rho_dry_clay} g/cm³")
    phi_clay = (rho_dry_clay - rho_clw) / (rho_dry_clay - rho_fluid)
    logger.debug(f"Clay porosity range: {phi_clay.min():.3f} - {phi_clay.max():.3f}")
    return phi_clay


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
    logger.debug(f"Calculating shale porosity with shale porosity: {phi_shale:.3f}")
    phi_sh = vshale * phi_shale
    logger.debug(f"Shale porosity range: {phi_sh.min():.3f} - {phi_sh.max():.3f}")
    return phi_sh


def rho_matrix(vsand=0, vsilt=0, vclay=0, vcalc=0, vdolo=0, vheavy=0,
               rho_sand: float = 0, rho_silt: float = 0, rho_clay: float = 0,
               rho_calc: float = 0, rho_dolo: float = 0, rho_heavy: float = 0):
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
    logger.debug("Calculating matrix density from mineral volumes")
    minerals_log_value = Config.MINERALS_LOG_VALUE
    rho_sand = rho_sand or minerals_log_value['RHOB_QUARTZ']
    rho_silt = rho_silt or minerals_log_value['RHOB_SILT']
    rho_clay = rho_clay or minerals_log_value['RHOB_SHALE']
    rho_calc = rho_calc or minerals_log_value['RHOB_CALCITE']
    rho_dolo = rho_dolo or minerals_log_value['RHOB_DOLOMITE']

    rho_matrix = vsand * rho_sand + vsilt * rho_silt + vclay * rho_clay + \
        vcalc * rho_calc + vdolo * rho_dolo + vheavy * rho_heavy

    logger.debug(f"Matrix density range: {rho_matrix.min():.3f} - {rho_matrix.max():.3f} g/cm³")
    return rho_matrix


def density_porosity(rhob, rho_matrix, rho_fluid: float = 1.0):
    """Computes density porosity from bulk, matrix and fluid densities

    Args:
        rhob (float): Bulk density log in g/cc.
        rho_matrix (float): Matrix density in g/cc.
        rho_fluid (float, optional): Density of fluid in g/cc. Defaults to 1.0 g/cc.

    Returns:
        float: Density porosity [fraction]
    """
    logger.debug("Calculating density porosity with fluid density")
    phi_d = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    logger.debug(f"Density porosity range: {phi_d.min():.3f} - {phi_d.max():.3f}")
    return phi_d


def dt_matrix(vsand=0, vsilt=0, vclay=0, vcalc=0, vdolo=0, vheavy=0,
              dt_sand: float = 0, dt_silt: float = 0, dt_clay: float = 0,
              dt_calc: float = 0, dt_dolo: float = 0, dt_heavy: float = 0):
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
    logger.debug("Calculating matrix sonic transit time from mineral volumes")
    minerals_log_value = Config.MINERALS_LOG_VALUE
    dt_sand = dt_sand or minerals_log_value['DTC_QUARTZ']
    dt_silt = dt_silt or minerals_log_value['DTC_SILT']
    dt_clay = dt_clay or minerals_log_value['DTC_SHALE']
    dt_calc = dt_calc or minerals_log_value['DTC_CALCITE']
    dt_dolo = dt_dolo or minerals_log_value['DTC_DOLOMITE']

    dt_matrix = (vsand * dt_sand + vsilt * dt_silt + vclay * dt_clay +
                 vcalc * dt_calc + vdolo * dt_dolo + vheavy * dt_heavy)
    logger.debug(f"Matrix sonic transit time range: {dt_matrix.min():.1f} - {dt_matrix.max():.1f} us/ft")
    return dt_matrix


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
    logger.debug(f"Calculating Wyllie sonic porosity with fluid transit time: {dt_fluid} us/ft")
    phi_s = (dt - dt_matrix) / (dt_fluid - dt_matrix)
    logger.debug(f"Wyllie sonic porosity range: {phi_s.min():.3f} - {phi_s.max():.3f}")
    return phi_s


def sonic_porosity_hunt_raymer(dt, dt_matrix, dt_fluid):
    """
    Computes sonic porosity based on Hunt-Raymer's equation from interval, matrix and transit time.

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
    logger.debug(f"Calculating Hunt-Raymer sonic porosity with fluid transit time: {dt_fluid} us/ft")
    c = (dt_matrix / (2 * dt_fluid)) - 1
    phi_s = - c - (c**2 + (dt_matrix / dt) - 1)**0.5
    logger.debug(f"Hunt-Raymer sonic porosity range: {phi_s.min():.3f} - {phi_s.max():.3f}")
    return phi_s


def neu_den_xplot_poro_pt(
        nphi: float, rhob: float, model: str = 'ssc',
        dry_min1_point: tuple = (),
        dry_silt_point: tuple = (),
        dry_clay_point: tuple = (),
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
    logger.debug(f"Calculating neutron-density crossplot porosity with model: {model}")
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
    logger.debug(f"Crossplot porosity: {phit:.3f}")
    return phit


def neu_den_xplot_poro(nphi, rhob, model: str = 'ssc',
                       dry_min1_point: tuple = (),
                       dry_silt_point: tuple = (),
                       dry_clay_point: tuple = (),
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
    logger.debug(f"Calculating neutron-density crossplot porosity for {len(nphi)} points with model: {model}")
    assert model in ['ssc', 'ss', 'carb'], ("Please specify either 'ssc', 'ss' or 'carb' model.")

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

    logger.debug(f"Crossplot porosity range: {phit.min():.3f} - {phit.max():.3f}")
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
    logger.debug(f"Correcting porosity using {method} averaging method")
    assert method in ['weighted', 'arithmetic', 'gaymard', 'gas'], "method must be either \
        'weighted', 'arithmetic', 'gaymard' or 'gas"
    dphi = density_porosity(rhob, rho_ma, 1.0)
    if method == 'weighted':
        phi_corr = (2 * dphi + nphi) / 3
    elif method == 'arithmetic':
        phi_corr = (dphi + nphi) / 2
    elif method == 'gaymard':
        phi_corr = np.sqrt((dphi**2 + nphi**2) / 2)
    elif method == 'gas':
        phi_corr = ((nphi**2 + dphi**2) / 2)**0.5

    logger.debug(f"Corrected porosity range: {phi_corr.min():.3f} - {phi_corr.max():.3f}")
    return phi_corr


def porosity_trend(tvdss, unit='ft'):
    """Calculate porosity trend based on TVDSS (Schmoker, 1982)

    Args:
        tvdss (float): True Vertical Depth Subsea.

    Returns:
        float: Porosity trend.
    """
    logger.debug(f"Calculating porosity trend with unit: {unit}")
    assert unit in ['ft', 'm'], 'Please specify either ft or m as unit.'
    if unit == 'ft':
        phi_trend = 41.73 * np.exp(-tvdss / 8197)
    else:
        phi_trend = 41.73 * np.exp(-tvdss / 2498)

    logger.debug(f"Porosity trend range: {phi_trend.min():.3f} - {phi_trend.max():.3f}")
    return phi_trend
