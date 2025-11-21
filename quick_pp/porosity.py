import numpy as np

from quick_pp.config import Config
from quick_pp.utils import length_a_b, line_intersection
from quick_pp import logger


def normalize_volumetric(phit, **volumetrics):
    """Normalize lithology given total porosity.

    Args:
        phit (np.ndarray or float): Total porosity in fraction (v/v).
        **volumetrics: Keyword arguments representing volumetric fractions (v/v) relative to matrix.

    Returns:
        dict: Normalized volumetric fractions relative to bulk volume.
    """
    logger.debug("Normalizing volumetrics with total porosity")
    # Normalize the volumetrics
    vmatrix = 1 - phit
    normalized_volumetrics = {
        key: value * vmatrix for key, value in volumetrics.items()
    }
    logger.debug(f"Normalized volumetrics: {list(normalized_volumetrics.keys())}")
    return normalized_volumetrics


def unnormalize_volumetric(phit, **normalized_volumetrics):
    """Unnormalize lithology given total porosity.

    This is the reverse of normalize_volumetric. It calculates the matrix-relative
    volumetrics from bulk-relative (normalized) volumetrics.

    Args:
        phit (np.ndarray or float): Total porosity in fraction (v/v).
        **normalized_volumetrics: Keyword arguments representing bulk-relative (normalized)
                                  volumetric fractions of the bulk volume (v/v).

    Returns:
        dict: Unnormalized volumetric fractions relative to the matrix volume.
    """
    logger.debug("Unnormalizing volumetrics with total porosity")
    # Unnormalize the volumetrics
    vmatrix = 1 - phit
    # Use np.divide for safe division by zero, returning np.nan
    unnormalized_volumetrics = {
        key: np.divide(
            value, vmatrix, out=np.full_like(value, np.nan), where=vmatrix != 0
        )
        for key, value in normalized_volumetrics.items()
    }
    logger.debug(f"Unnormalized volumetrics: {list(unnormalized_volumetrics.keys())}")
    return unnormalized_volumetrics


def get_volumetric_dict(df):
    """Given dataframe, return a dictionary of the key and values of lithology metrics in the data.

    Args:
        df (pd.DataFrame): Dataframe with well log data.

    Returns:
        dict: Dictionary of the key and values of lithology metrics in the data.
    """
    volumetric_dict = {}
    for vol_log in Config.MINERALS_LOG_VALUE.keys():
        vol_log = Config.MINERALS_NAME_MAPPING.get(vol_log, vol_log)
        if vol_log in df.columns:
            volumetric_dict[vol_log.lower()] = df[vol_log].values
    return volumetric_dict


def effective_porosity(phit, phi_shale, vshale):
    """
    Computes effective porosity from total porosity, total porosity of shale and shale volume.

    Parameters
    ----------
    phit : np.ndarray or float
        Total porosity [fraction].
    phi_shale : np.ndarray or float
        Total porosity of shale [fraction].
    vshale : np.ndarray or float
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


def estimate_shale_porosity_trend(
    rho_clw: np.ndarray, rho_dry_clay: float = 2.72, rho_fluid: float = 1.0
):
    """Calculate clay porosity given bulk density of wet clay line.

    Args:
        rho_clw (np.ndarray or float): Bulk density of wet clay line.
        rho_dry_clay (float, optional): Bulk density of dry clay. Defaults to 2.72.
        rho_fluid (float, optional): Bulk density of fluid. Defaults to 1.0.

    Returns:
        np.ndarray or float: Clay porosity.
    """
    logger.debug(
        f"Calculating clay porosity with dry clay density: {rho_dry_clay} g/cm³"
    )
    phi_clay = (rho_dry_clay - rho_clw) / (rho_dry_clay - rho_fluid)
    logger.debug(f"Clay porosity range: {phi_clay.min():.3f} - {phi_clay.max():.3f}")
    return phi_clay


def estimate_shale_porosity(nphi, phit):
    """
    Computes shale porosity from neutron porosity and total porosity.

    Args:
        nphi (np.ndarray or float): Neutron porosity (hydrocarbon corrected) [fraction].
        phit (np.ndarray or float): Total porosity [fraction].

    Returns:
        np.ndarray or float: Shale porosity [fraction].

    """
    phit_sh = nphi - phit
    return np.where(phit_sh > 0, phit_sh, phit).clip(1e-2, 1.0)


def rho_matrix(**volumetrics):
    """Estimate average matrix density based on mineral volumes and their densities from Config.

    Args:
        **volumetrics: Keyword arguments where keys are mineral volume names
                       (e.g., vsand, vclay) and values are their volume fractions.


    Returns:
        np.ndarray or float: Matrix density in g/cc.
    """
    logger.debug("Calculating matrix density from mineral volumes")
    rho_ma = 0.0
    mineral_properties = Config.MINERALS_LOG_VALUE
    # Create a reverse mapping from volume name (e.g., 'VSAND') to mineral name (e.g., 'QUARTZ')
    name_to_mineral_map = {v: k for k, v in Config.MINERALS_NAME_MAPPING.items()}

    for vol_name, vol_value in volumetrics.items():
        # Find the mineral name (e.g., 'QUARTZ') from the volume name (e.g., 'VSAND')
        mineral_name = name_to_mineral_map.get(vol_name.upper())

        if mineral_name and mineral_name in mineral_properties:
            rho_mineral = mineral_properties[mineral_name].get("RHOB")
            if rho_mineral is not None:
                rho_ma += vol_value * rho_mineral
        elif (
            vol_name.upper() == "VSILT"
        ):  # Handle silt separately if not in main config
            rho_ma += vol_value * 2.68

    logger.debug(
        f"Matrix density range: {np.nanmin(rho_ma):.3f} - {np.nanmax(rho_ma):.3f} g/cm³"
    )
    return rho_ma


def density_porosity(rhob, rho_matrix, rho_fluid: float = 1.0):
    """Computes density porosity from bulk, matrix and fluid densities

    Args:
        rhob (np.ndarray or float): Bulk density log in g/cc.
        rho_matrix (np.ndarray or float): Matrix density in g/cc.
        rho_fluid (float, optional): Density of fluid in g/cc. Defaults to 1.0 g/cc.

    Returns:
        np.ndarray or float: Density porosity [fraction]
    """
    logger.debug("Calculating density porosity with fluid density")
    phi_d = (rho_matrix - rhob) / (rho_matrix - rho_fluid)
    logger.debug(f"Density porosity range: {phi_d.min():.3f} - {phi_d.max():.3f}")
    return phi_d


def dt_matrix(
    vsand=0,
    vclay=0,
    vcalc=0,
    vdolo=0,
    vheavy=0,
    dt_sand: float = 0,
    dt_silt: float = 0,
    dt_clay: float = 0,
    dt_calc: float = 0,
    dt_dolo: float = 0,
    dt_heavy: float = 0,
):
    """Estimate average matrix sonic transit time based on dry sand, dry silt dry calcite and
    dry dolomite volume and transit time of each.

    Args:
        vsand (np.ndarray or float, optional): Volume of sand. Defaults to 0.
        vclay (np.ndarray or float, optional): Volume of clay. Defaults to 0.
        vcalc (np.ndarray or float, optional): Volume of calcite. Defaults to 0.
        vdolo (np.ndarray or float, optional): Volume of dolomite. Defaults to 0.
        vheavy (np.ndarray or float, optional): Volume of heavy minerals. Defaults to 0.
        dt_sand (float, optional): Sonic transit time of sand in us/ft. Defaults to None.
        dt_silt (float, optional): Sonic transit time of silt in us/ft. Defaults to None.
        dt_clay (float, optional): Sonic transit time of clay in us/ft. Defaults to None.
        dt_calc (float, optional): Sonic transit time of calcite in us/ft. Defaults to None.
        dt_dolo (float, optional): Sonic transit time of dolomite in us/ft. Defaults to None.
        dt_heavy (float, optional): Sonic transit time of heavy minerals in us/ft. Defaults to 0.0.

    Returns:
        np.ndarray or float: Matrix sonic transit time in us/ft.
    """
    logger.debug("Calculating matrix sonic transit time from mineral volumes")
    minerals_log_value = Config.MINERALS_LOG_VALUE
    dt_sand = dt_sand or minerals_log_value["QUARTZ"]["DTC"]
    dt_clay = dt_clay or minerals_log_value["SHALE"]["DTC"]
    dt_calc = dt_calc or minerals_log_value["CALCITE"]["DTC"]
    dt_dolo = dt_dolo or minerals_log_value["DOLOMITE"]["DTC"]

    dt_matrix = (
        vsand * dt_sand
        + vclay * dt_clay
        + vcalc * dt_calc
        + vdolo * dt_dolo
        + vheavy * dt_heavy
    )
    logger.debug(
        f"Matrix sonic transit time range: {dt_matrix.min():.1f} - {dt_matrix.max():.1f} us/ft"
    )
    return dt_matrix


def sonic_porosity_wyllie(dt, dt_matrix, dt_fluid):
    """
    Computes sonic porosity based on Wyllie's equation from interval, matrix, and fluid transit time.

    Parameters
    ----------
    dt : np.ndarray or float
        Interval transit time [us/ft].
    dt_matrix : np.ndarray or float
        Matrix transit time [us/ft]. Sandstone: 51-55, Limestone: 43-48, Dolomite: 43-39, Shale: 60-170.
    dt_fluid : np.ndarray or float
        Fluid transit time [us/ft]. Water: 190, Oil: 240, Gas: 630.

    Returns
    -------
    porosity : np.ndarray or float
        Sonic porosity [fraction].

    """
    logger.debug(
        f"Calculating Wyllie sonic porosity with fluid transit time: {dt_fluid} us/ft"
    )
    phi_s = (dt - dt_matrix) / (dt_fluid - dt_matrix)
    logger.debug(f"Wyllie sonic porosity range: {phi_s.min():.3f} - {phi_s.max():.3f}")
    return phi_s


def sonic_porosity_hunt_raymer(dt, dt_matrix, dt_fluid):
    """
    Computes sonic porosity based on Hunt-Raymer's equation from interval, matrix and transit time.

    Parameters
    ----------
    dt : np.ndarray or float
        Interval transit time [us/ft].
    dt_matrix : np.ndarray or float
        Matrix transit time [us/ft]. Sandstone: 51-55, Limestone: 43-48, Dolomite: 43-39, Shale: 60-170.
    dt_fluid : np.ndarray or float
        Fluid transit time [us/ft]. Water: 190, Oil: 240, Gas: 630.

    Returns
    -------
    porosity : np.ndarray or float
        Sonic porosity [fraction].

    """
    logger.debug(
        f"Calculating Hunt-Raymer sonic porosity with fluid transit time: {dt_fluid} us/ft"
    )
    c = (dt_matrix / (2 * dt_fluid)) - 1
    phi_s = -c - (c**2 + (dt_matrix / dt) - 1) ** 0.5
    logger.debug(
        f"Hunt-Raymer sonic porosity range: {phi_s.min():.3f} - {phi_s.max():.3f}"
    )
    return phi_s


def neu_den_xplot_poro_pt(
    nphi: float,
    rhob: float,
    model: str = "ssc",
    dry_min1_point: tuple = (),
    dry_silt_point: tuple = (),
    dry_clay_point: tuple = (),
    fluid_point: tuple = (1.0, 1.0),
):
    """Calculate porosity given a pair of neutron porosity and bulk density data point.
    This function is designed to process a single data point.

    Args:
        nphi (float): Neutron porosity log value.
        rhob (float): Bulk density log value.
        model (str, optional): Lithology model, either 'ssc' (Sand Silt Clay) or 'ss' (Sand Shale). Defaults to 'ssc'.
        dry_min1_point (tuple, optional): Neutron porosity and bulk density of mineral 1 point. Defaults to ().
        dry_silt_point (tuple, optional): Neutron porosity and bulk density of dry silt point. Defaults to ().
        dry_clay_point (tuple, optional): Neutron porosity and bulk density of dry clay point. Defaults to ().
        fluid_point (tuple, optional): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).

    Returns:
        float: Total porosity for the given data point.
    """
    logger.debug(f"Calculating neutron-density crossplot porosity with model: {model}")
    assert model in ["ssc", "ss", "carb"], (
        "Please specify either 'ssc', 'ss' or 'carb' model."
    )

    A = dry_min1_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point

    phit = []
    if model == "ssc":
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


def neu_den_xplot_poro(
    nphi,
    rhob,
    model: str = "ssc",
    dry_min1_point: tuple = (),
    dry_silt_point: tuple = (),
    dry_clay_point: tuple = (),
    fluid_point: tuple = (1.0, 1.0),
):
    """Calculate porosity given neutron porosity and bulk density logs.

    This function processes arrays of log data.

    Args:
        nphi (np.ndarray or float): Neutron porosity log.
        rhob (np.ndarray or float): Bulk density log.
        model (str, optional): Lithology model, either 'ssc' (Sand Silt Clay), 'ss' (Sand Shale) or 'carb' (Carbonate).
                               Defaults to 'ssc'.
        dry_min1_point (tuple, optional): Neutron porosity and bulk density of dry min1 point. Defaults to ().
        dry_silt_point (tuple, optional): Neutron porosity and bulk density of dry silt point. Defaults to ().
        dry_clay_point (tuple, optional): Neutron porosity and bulk density of dry clay point. Defaults to ().
        fluid_point (tuple, optional): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).

    Returns:
        np.ndarray or float: Total porosity log.
    """
    logger.debug(
        f"Calculating neutron-density crossplot porosity for {len(nphi)} points with model: {model}"
    )
    assert model in ["ssc", "ss", "carb"], (
        "Please specify either 'ssc', 'ss' or 'carb' model."
    )

    A = dry_min1_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))

    phit = np.empty(0)
    for i, point in enumerate(E):
        if model == "ssc":
            phit = np.append(
                phit, neu_den_xplot_poro_pt(point[0], point[1], "ssc", A, B, C, D)
            )
        else:
            phit = np.append(
                phit, neu_den_xplot_poro_pt(point[0], point[1], "ss", A, (0, 0), C, D)
            )

    logger.debug(f"Crossplot porosity range: {phit.min():.3f} - {phit.max():.3f}")
    return phit


def porosity_correction_averaging(nphi, rhob, rho_ma=2.65, method="weighted"):
    """Correct porosity using averaging method.
    Weighted average: (2 * dphi + nphi) / 3
    Arithmetic average: (dphi + nphi) / 2
    Gaymard average: sqrt((dphi**2 + nphi**2) / 2)
    Gas average: sqrt((dphi**2 + nphi**2) / 2)

    Args:
        nphi (np.ndarray or float): Neutron porosity.
        rhob (np.ndarray or float): Bulk density log.
        rho_ma (float, optional): Matrix density. Defaults to 2.65.
        method (str, optional): Averaging method selection from 'weighted', 'arithmetic' or 'gaymard'.
         Defaults to 'weighted'.

    Returns:
        np.ndarray or float: Corrected porosity.
    """
    logger.debug(f"Correcting porosity using {method} averaging method")
    assert method in ["weighted", "arithmetic", "gaymard", "gas"], (
        "method must be either \
        'weighted', 'arithmetic', 'gaymard' or 'gas"
    )
    dphi = density_porosity(rhob, rho_ma, 1.0)
    if method == "weighted":
        phi_corr = (2 * dphi + nphi) / 3
    elif method == "arithmetic":
        phi_corr = (dphi + nphi) / 2
    elif method == "gaymard":
        phi_corr = np.sqrt((dphi**2 + nphi**2) / 2)
    elif method == "gas":
        phi_corr = ((nphi**2 + dphi**2) / 2) ** 0.5

    logger.debug(
        f"Corrected porosity range: {phi_corr.min():.3f} - {phi_corr.max():.3f}"
    )
    return phi_corr


def porosity_trend(tvdss, unit="ft"):
    """Calculate porosity trend based on TVDSS (Schmoker, 1982)

    Args:
        tvdss (np.ndarray or float): True Vertical Depth Subsea.
        unit (str, optional): Unit of depth, either 'ft' or 'm'. Defaults to 'ft'.

    Returns:
        np.ndarray or float: Porosity trend.
    """
    logger.debug(f"Calculating porosity trend with unit: {unit}")
    assert unit in ["ft", "m"], "Please specify either ft or m as unit."
    if unit == "ft":
        phi_trend = 41.73 * np.exp(-tvdss / 8197)
    else:
        phi_trend = 41.73 * np.exp(-tvdss / 2498)

    logger.debug(f"Porosity trend range: {phi_trend.min():.3f} - {phi_trend.max():.3f}")
    return phi_trend


def nmr_porosity(t2_dist, t2_bins, t2_cutoff=33):
    """Calculate NMR porosity from T2 distribution.

    Total porosity is the sum of all T2 distribution amplitudes.
    Effective porosity is the sum of T2 distribution amplitudes above the T2 cutoff.

    Args:
        t2_dist (array-like): T2 distribution amplitudes
        t2_bins (array-like): T2 time bins corresponding to distribution
        t2_cutoff (float, optional): T2 cutoff time in ms. Defaults to 33ms.

    Returns:
        tuple: Total porosity and effective porosity (phit, phie)
    """
    logger.debug(f"Calculating NMR porosity with T2 cutoff: {t2_cutoff}ms")

    # Total porosity is sum of all T2 amplitudes
    phit = np.sum(t2_dist)

    # Effective porosity is sum of amplitudes above T2 cutoff
    phie = np.sum(t2_dist[t2_bins >= t2_cutoff])

    logger.debug(f"NMR total porosity range: {phit.min():.3f} - {phit.max():.3f}")
    logger.debug(f"NMR effective porosity range: {phie.min():.3f} - {phie.max():.3f}")

    return phit, phie
