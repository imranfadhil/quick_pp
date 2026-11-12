"""Porosity and multi-mineral volumetric estimation models.

Provides density, neutron-density, sonic, and multi-mineral solver routines,
shale porosity trend estimation, hydrocarbon density corrections, Thomas-Stieber
thin-bed sand porosity partitioning, and the continuous apparent fluid density
(``RHO_FL``) iterative solver.
"""

import numpy as np

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.utils import length_a_b, line_intersection


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
    logger.debug(
        f"Calculating effective porosity with mean shale porosity: {float(np.nanmean(phi_shale)):.3f}"
    )
    phie = phit - (vshale * phi_shale)
    logger.debug(
        f"Effective porosity range: {np.nanmin(phie):.3f} - {np.nanmax(phie):.3f}"
    )
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
    logger.debug(
        f"Density porosity range: {np.nanmin(phi_d):.3f} - {np.nanmax(phi_d):.3f}"
    )
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

    nphi_arr = np.asarray(nphi, dtype=float)
    rhob_arr = np.asarray(rhob, dtype=float)
    if nphi_arr.shape != rhob_arr.shape:
        raise ValueError("nphi and rhob must have the same shape")

    A = dry_min1_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point

    if model == "ssc":
        thold_pt = line_intersection((A, C), (D, B))
        thold_line = length_a_b(thold_pt, A)
        proj_pt = line_intersection((A, C), (D, (nphi_arr, rhob_arr)))
        proj_line = length_a_b(proj_pt, A)
        m_res = (A[1] - B[1]) / (A[0] - B[0])
        m_nonres = (C[1] - B[1]) / (C[0] - B[0])
        m = np.where(proj_line < thold_line, m_res, m_nonres)
        c = rhob_arr - m * nphi_arr
        iso_poro_pt = line_intersection(
            ((np.zeros_like(c), c), (nphi_arr, rhob_arr)), (D, B)
        )
        iso_poro_line = length_a_b(iso_poro_pt, B)
        poro_line = length_a_b(D, B)
        phit = iso_poro_line / poro_line
    else:
        m = (A[1] - C[1]) / (A[0] - C[0])
        c = rhob_arr - m * nphi_arr
        iso_poro_pt = line_intersection(
            ((np.zeros_like(c), c), (nphi_arr, rhob_arr)), (D, A)
        )
        iso_poro_line = length_a_b(iso_poro_pt, A)
        poro_line = length_a_b(D, A)
        phit = iso_poro_line / poro_line

    logger.debug(
        f"Crossplot porosity range: {np.nanmin(phit):.3f} - {np.nanmax(phit):.3f}"
    )
    return phit


def porosity_correction_averaging(
    nphi, rhob, rho_ma: float | np.ndarray = 2.65, method="weighted", rho_fluid=1.0
):
    """Correct porosity using averaging method.
    Weighted average: (2 * dphi + nphi) / 3
    Arithmetic average: (dphi + nphi) / 2
    Gaymard average: sqrt((dphi**2 + nphi**2) / 2)
    Gas average: sqrt((dphi**2 + nphi**2) / 2)

    Args:
        nphi (np.ndarray or float): Neutron porosity.
        rhob (np.ndarray or float): Bulk density log.
        rho_ma (np.ndarray or float, optional): Matrix density. Defaults to 2.65.
        method (str, optional): Averaging method selection from 'weighted', 'arithmetic' or 'gaymard'.
         Defaults to 'weighted'.
        rho_fluid (float, optional): Fluid density in g/cc. Defaults to 1.0.

    Returns:
        np.ndarray or float: Corrected porosity.
    """
    logger.debug(f"Correcting porosity using {method} averaging method")
    assert method in ["weighted", "arithmetic", "gaymard", "gas"], (
        "method must be either \
        'weighted', 'arithmetic', 'gaymard' or 'gas"
    )
    dphi = density_porosity(rhob, rho_ma, rho_fluid)
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


def calc_secondary_porosity(phit, phis):
    """Calculate secondary (vuggy/fracture) porosity and interparticle porosity from total and sonic porosity.

    Based on Lucia (1995, 1999) carbonate rock fabric petrophysics:
        phi_sec = max(0, phi_T - phi_sonic)
        phi_ip = clip(phi_T - phi_sec, 0.001, phi_T)

    Where `phis` is NaN, `phi_sec` is NaN and `phi_ip` falls back to `phit`.

    Args:
        phit (np.ndarray or float): Total porosity in fraction (v/v).
        phis (np.ndarray or float): Sonic (matrix/interparticle) porosity in fraction (v/v).

    Returns:
        tuple: (phi_sec, phi_ip) as numpy arrays.
    """
    logger.debug(
        "Calculating secondary and interparticle porosity from total and sonic porosity"
    )
    phit_arr = np.asarray(phit, dtype=float)
    phis_arr = np.asarray(phis, dtype=float)
    both_valid = np.isfinite(phit_arr) & np.isfinite(phis_arr) & (phit_arr > 0)
    phi_sec = np.full_like(phit_arr, np.nan, dtype=float)
    phi_ip = np.where(np.isfinite(phit_arr), np.clip(phit_arr, 0.001, 0.99), np.nan)
    if both_valid.any():
        sec_vals = np.maximum(
            0.0, phit_arr[both_valid] - np.clip(phis_arr[both_valid], 0.0, 0.99)
        )
        phi_sec[both_valid] = sec_vals
        phi_ip[both_valid] = np.clip(
            phit_arr[both_valid] - sec_vals, 0.001, phit_arr[both_valid]
        )
    return phi_sec, phi_ip


secondary_porosity = calc_secondary_porosity


# ---------------------------------------------------------------------------
# Continuous apparent fluid density (RHO_FL) & iterative density porosity
# ---------------------------------------------------------------------------

# Tier routing codes returned in RHO_FL_TIER
RHO_FL_TIER_RXO = 1
RHO_FL_TIER_SW_FALLBACK = 2
RHO_FL_TIER_ZONAL = 3

# Default residual hydrocarbon saturations used by the Tier 3 zonal fallback
DEFAULT_SHR_GAS = 0.25
DEFAULT_SHR_OIL = 0.15

# Default filtrate densities used when no mud-filtrate property is supplied
DEFAULT_WBM_FILTRATE_DENSITY = 1.00
DEFAULT_OBM_FILTRATE_DENSITY = 0.80

# Worst-case WBM filtrate salinities (ppm) when neither Rmf nor salinity is known
DEFAULT_WBM_FRESH_SALINITY_PPM = 5_000.0
DEFAULT_WBM_SALINE_SALINITY_PPM = 150_000.0


def salinity_from_rmf(rmf: float, temp_c: float = 25.0) -> float:
    """Back-calculate NaCl-equivalent salinity (ppm) from a mud-filtrate resistivity.

    The measured ``Rmf`` is first corrected to 75 degF with the Arps/Bateman-Konen
    temperature relation, then inverted through the Bateman-Konen salinity
    correlation.

    Args:
        rmf (float): Mud-filtrate resistivity in ohm.m at ``temp_c``.
        temp_c (float, optional): Temperature of the ``rmf`` measurement in degrees
            Celsius. Defaults to 25.0.

    Returns:
        float: NaCl-equivalent salinity in ppm, clipped to [0, 350000].
    """
    if not np.isfinite(rmf) or rmf <= 0:
        raise ValueError("rmf must be a positive finite value.")
    temp_f = float(temp_c) * 9.0 / 5.0 + 32.0
    rmf_75 = float(rmf) * (temp_f + 6.77) / (75.0 + 6.77)
    if rmf_75 <= 0.0123:
        return 350_000.0
    exponent = (3.562 - np.log10(rmf_75 - 0.0123)) / 0.955
    salinity = float(10.0**exponent)
    return float(np.clip(salinity, 0.0, 350_000.0))


def _as_float_array(value, n: int, default: float) -> np.ndarray:
    """Broadcast a scalar/array/None into a float array of length ``n``."""
    if value is None:
        return np.full(n, float(default), dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.ndim == 0:
        return np.full(n, float(arr), dtype=float)
    return np.broadcast_to(arr, (n,)).astype(float)


def _normalize_fluid_zone(fluid_zone, n: int) -> np.ndarray:
    """Normalize a fluid-zone array into uppercase strings ('GAS'/'OIL'/'WATER'/'')."""
    if fluid_zone is None:
        return np.full(n, "", dtype=object)
    arr = np.asarray(fluid_zone).astype(str)
    if arr.ndim == 0:
        arr = np.full(n, str(arr), dtype=object)
    else:
        arr = np.broadcast_to(arr, (n,)).astype(object)
    return np.char.upper(arr.astype(str))


def solve_continuous_rho_fluid_and_phid(
    rhob: np.ndarray,
    rhoma: np.ndarray | float,
    tvd: np.ndarray,
    mud_type: str = "WBM",
    rho_mf: np.ndarray | float | None = None,
    rho_w: np.ndarray | float | None = 1.02,
    rho_h_app: np.ndarray | float | None = 0.75,
    rxo: np.ndarray | None = None,
    rt: np.ndarray | None = None,
    rmf_downhole: np.ndarray | None = None,
    rw_downhole: np.ndarray | None = None,
    badhole_flag: np.ndarray | None = None,
    fluid_zone: np.ndarray | None = None,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    flush_exponent: float = 0.2,
    default_shr_gas: float = DEFAULT_SHR_GAS,
    default_shr_oil: float = DEFAULT_SHR_OIL,
    max_iter: int = 8,
    tol: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Vectorized fixed-point solver for continuous apparent fluid density and PHID.

    At every depth the flushed-zone apparent fluid density is
    ``rho_fl = Sxo * rho_mf + (1 - Sxo) * rho_h_app`` and density porosity is
    ``phid = (rhoma - rhob) / (rhoma - rho_fl)``. Because ``Sxo`` depends on
    ``phid``, the pair is solved by fixed-point iteration.

    The solver auto-degrades per depth sample:

    * Tier 1 - ``Rxo`` shallow resistivity with a valid ``Rmf`` (WBM only).
    * Tier 2 - ``Sxo = Sw ** flush_exponent`` from deep ``Rt`` (Poupon coupling).
    * Tier 3 - zonal constant residual hydrocarbon saturation.

    OBM/SBM filtrate miscibly displaces mobile hydrocarbons, so the flushed zone
    is modelled as capillary-bound brine plus OBM filtrate/residual hydrocarbon.

    Args:
        rhob: Bulk density log in g/cc.
        rhoma: Matrix density in g/cc (scalar or per-sample).
        tvd: True vertical depth in m (used only for length/broadcasting).
        mud_type: 'WBM', 'OBM', or 'SBM'.
        rho_mf: Mud-filtrate density in g/cc (scalar or per-sample).
        rho_w: Formation water density in g/cc.
        rho_h_app: Apparent (electron-density) hydrocarbon density in g/cc.
        rxo: Shallow/invaded-zone resistivity in ohm.m.
        rt: Deep resistivity in ohm.m.
        rmf_downhole: Downhole mud-filtrate resistivity in ohm.m.
        rw_downhole: Downhole formation water resistivity in ohm.m.
        badhole_flag: Boolean/0-1 flag marking unreliable intervals.
        fluid_zone: Per-sample fluid zone labels ('GAS', 'OIL', 'WATER').
        a: Archie tortuosity factor.
        m: Cementation exponent.
        n: Saturation exponent.
        flush_exponent: Poupon exponent ``c`` in ``Sxo = Sw ** c``.
        default_shr_gas: Tier 3 residual gas saturation.
        default_shr_oil: Tier 3 residual oil saturation.
        max_iter: Maximum fixed-point iterations.
        tol: Convergence tolerance on the maximum PHID change.

    Returns:
        dict[str, np.ndarray]: ``PHID``, ``PHID_RAW_100`` (rho_fl = 1.0),
        ``RHO_FL``, ``RHO_MF``, ``SXO_EST``, and ``RHO_FL_TIER``.
    """
    rhob_arr = np.asarray(rhob, dtype=float).ravel()
    n_samples = rhob_arr.size
    if n_samples == 0:
        empty = np.array([], dtype=float)
        return {
            "PHID": empty,
            "PHID_RAW_100": empty,
            "RHO_FL": empty,
            "RHO_MF": empty,
            "SXO_EST": empty,
            "RHO_FL_TIER": np.array([], dtype=int),
        }
    rhoma_arr = _as_float_array(rhoma, n_samples, 2.65)
    tvd_arr = _as_float_array(tvd, n_samples, 0.0)
    del tvd_arr  # depth is carried by the caller; kept for API symmetry

    mud = str(mud_type).upper()
    is_obm = mud in ("OBM", "SBM")

    default_mf = DEFAULT_OBM_FILTRATE_DENSITY if is_obm else DEFAULT_WBM_FILTRATE_DENSITY
    rho_mf_arr = _as_float_array(rho_mf, n_samples, default_mf)
    rho_w_arr = _as_float_array(rho_w, n_samples, 1.02)
    rho_h_arr = _as_float_array(rho_h_app, n_samples, 0.75)

    zone = _normalize_fluid_zone(fluid_zone, n_samples)
    is_water_zone = zone == "WATER"
    # In known water zones the hydrocarbon term collapses to formation water.
    rho_h_arr = np.where(is_water_zone, rho_w_arr, rho_h_arr)

    badhole = (
        np.asarray(badhole_flag, dtype=bool).ravel()
        if badhole_flag is not None
        else np.zeros(n_samples, dtype=bool)
    )
    if badhole.size != n_samples:
        badhole = np.broadcast_to(badhole, (n_samples,)).astype(bool)

    def _finite_positive(arr: np.ndarray | None) -> np.ndarray:
        if arr is None:
            return np.zeros(n_samples, dtype=bool)
        vals = np.asarray(arr, dtype=float).ravel()
        if vals.size != n_samples:
            vals = np.broadcast_to(vals, (n_samples,)).astype(float)
        return np.isfinite(vals) & (vals > 0)

    rxo_ok = _finite_positive(rxo)
    rt_ok = _finite_positive(rt)
    rmf_ok = _finite_positive(rmf_downhole)
    rw_ok = _finite_positive(rw_downhole)

    has_rxo = rxo_ok & rmf_ok & (~badhole)
    has_rt = rt_ok & rw_ok

    tier_used = np.full(n_samples, RHO_FL_TIER_ZONAL, dtype=int)
    tier_used[has_rt] = RHO_FL_TIER_SW_FALLBACK
    if not is_obm:
        tier_used[has_rxo] = RHO_FL_TIER_RXO

    rxo_arr = _as_float_array(rxo, n_samples, np.nan)
    rt_arr = _as_float_array(rt, n_samples, np.nan)
    rmf_arr = _as_float_array(rmf_downhole, n_samples, np.nan)
    rw_arr = _as_float_array(rw_downhole, n_samples, np.nan)

    # Tier 3 zonal residual hydrocarbon saturation
    shr_zonal = np.where(
        zone == "GAS",
        float(default_shr_gas),
        np.where(zone == "OIL", float(default_shr_oil), 0.0),
    )
    sxo_t3 = np.clip(1.0 - shr_zonal, 0.0, 1.0)

    # Initialization (k = 0)
    rho_fl = rho_mf_arr.copy()
    phid_raw = np.clip(
        (rhoma_arr - rhob_arr) / np.maximum(rhoma_arr - 1.0, 0.1), 0.0, 0.60
    )
    phid = np.clip(
        (rhoma_arr - rhob_arr) / np.maximum(rhoma_arr - rho_fl, 0.1), 0.001, 0.60
    )

    sxo = np.ones(n_samples, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        for _ in range(max(1, int(max_iter))):
            phid_prev = phid.copy()
            phi_m = np.maximum(phid, 0.01) ** m

            sw = np.where(
                has_rt,
                np.clip(
                    ((a * rw_arr) / (phi_m * np.maximum(rt_arr, 0.01))) ** (1.0 / n),
                    0.02,
                    1.0,
                ),
                1.0,
            )

            if is_obm:
                # OBM/SBM: capillary-bound brine + OBM filtrate / residual HC.
                sxo = np.where(
                    has_rt,
                    sw,
                    np.where(is_water_zone, 0.15, 0.10),
                )
                rho_fl = sxo * rho_w_arr + (1.0 - sxo) * rho_mf_arr
            else:
                sxo_t2 = np.clip(sw**flush_exponent, sw, 1.0)
                # The spec clips Tier 1 Sxo to [Sw, 1]. When Rt is unavailable Sw is
                # undefined, so fall back to the zonal residual saturation as the
                # lower bound instead of forcing Sxo = 1.
                sxo_lower = np.where(has_rt, sw, sxo_t3)
                sxo_t1 = np.where(
                    has_rxo,
                    np.clip(
                        ((a * rmf_arr) / (phi_m * np.maximum(rxo_arr, 0.01)))
                        ** (1.0 / n),
                        sxo_lower,
                        1.0,
                    ),
                    sxo_t2,
                )
                sxo = np.where(
                    tier_used == RHO_FL_TIER_RXO,
                    sxo_t1,
                    np.where(tier_used == RHO_FL_TIER_SW_FALLBACK, sxo_t2, sxo_t3),
                )
                sxo = np.where(is_water_zone, 1.0, sxo)
                rho_fl = sxo * rho_mf_arr + (1.0 - sxo) * rho_h_arr

            phid = np.clip(
                (rhoma_arr - rhob_arr) / np.maximum(rhoma_arr - rho_fl, 0.1),
                0.0,
                0.60,
            )
            if np.nanmax(np.abs(phid - phid_prev)) < tol:
                break

    return {
        "PHID": phid,
        "PHID_RAW_100": phid_raw,
        "RHO_FL": rho_fl,
        "RHO_MF": rho_mf_arr,
        "SXO_EST": sxo,
        "RHO_FL_TIER": tier_used,
    }


def estimate_continuous_rho_fluid(
    rhob: np.ndarray,
    rhoma: np.ndarray | float,
    tvd: np.ndarray,
    temp_c: np.ndarray | float,
    pressure_mpa: np.ndarray | float,
    mud_type: str = "WBM",
    salinity_ppm: np.ndarray | float | None = None,
    rmf: float | None = None,
    rmf_temp_c: float = 25.0,
    obm_base_density: float = 0.82,
    rho_w: np.ndarray | float | None = None,
    rho_h_app: np.ndarray | float | None = 0.75,
    rxo: np.ndarray | None = None,
    rt: np.ndarray | None = None,
    rmf_downhole: np.ndarray | None = None,
    rw_downhole: np.ndarray | None = None,
    badhole_flag: np.ndarray | None = None,
    fluid_zone: np.ndarray | None = None,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    flush_exponent: float = 0.2,
    default_shr_gas: float = DEFAULT_SHR_GAS,
    default_shr_oil: float = DEFAULT_SHR_OIL,
    max_iter: int = 8,
    tol: float = 1e-4,
) -> dict[str, np.ndarray]:
    """Build continuous downhole filtrate/water densities then solve for RHO_FL.

    This is the physics-facing wrapper around
    :func:`solve_continuous_rho_fluid_and_phid`. It derives the continuous
    mud-filtrate density profile ``rho_mf(z)`` from the mud system:

    * WBM with a known salinity -> Batzle & Wang brine density.
    * WBM with only a surface ``Rmf`` -> salinity back-calculated via
      :func:`salinity_from_rmf`, then Batzle & Wang brine density.
    * WBM with neither -> fresh (5,000 ppm) or saline (150,000 ppm) fallback.
    * OBM/SBM -> base-oil thermal expansion/compressibility profile.

    Args:
        rhob: Bulk density log in g/cc.
        rhoma: Matrix density in g/cc.
        tvd: True vertical depth in m.
        temp_c: Formation temperature in degrees Celsius.
        pressure_mpa: Absolute formation pressure in MPa.
        mud_type: 'WBM', 'OBM', or 'SBM'.
        salinity_ppm: Mud-filtrate salinity in ppm (WBM).
        rmf: Surface mud-filtrate resistivity in ohm.m (WBM fallback).
        rmf_temp_c: Temperature of the ``rmf`` measurement in degrees Celsius.
        obm_base_density: Surface base-oil density in g/cc (OBM/SBM).
        rho_w: Formation water density in g/cc; defaults to the filtrate profile.
        rho_h_app: Apparent hydrocarbon density in g/cc.
        rxo: Shallow/invaded-zone resistivity in ohm.m.
        rt: Deep resistivity in ohm.m.
        rmf_downhole: Downhole mud-filtrate resistivity in ohm.m.
        rw_downhole: Downhole formation water resistivity in ohm.m.
        badhole_flag: Boolean/0-1 flag marking unreliable intervals.
        fluid_zone: Per-sample fluid zone labels.
        a: Archie tortuosity factor.
        m: Cementation exponent.
        n: Saturation exponent.
        flush_exponent: Poupon exponent for the Tier 2 fallback.
        default_shr_gas: Tier 3 residual gas saturation.
        default_shr_oil: Tier 3 residual oil saturation.
        max_iter: Maximum fixed-point iterations.
        tol: Convergence tolerance on the maximum PHID change.

    Returns:
        dict[str, np.ndarray]: Solver outputs plus ``RHO_MF`` and ``SALINITY_PPM``.
    """
    from quick_pp.fluid_type import brine_density, obm_filtrate_density

    rhob_arr = np.asarray(rhob, dtype=float).ravel()
    n_samples = rhob_arr.size
    temp_arr = _as_float_array(temp_c, n_samples, 25.0)
    pres_arr = _as_float_array(pressure_mpa, n_samples, 0.1013)
    mud = str(mud_type).upper()

    if mud in ("OBM", "SBM"):
        rho_mf_arr = obm_filtrate_density(temp_arr, pres_arr, obm_base_density)
        salinity_arr = np.zeros(n_samples, dtype=float)
    else:
        if salinity_ppm is not None:
            salinity_arr = _as_float_array(salinity_ppm, n_samples, 0.0)
        elif rmf is not None and np.isfinite(rmf) and rmf > 0:
            salinity_arr = np.full(
                n_samples, salinity_from_rmf(rmf, rmf_temp_c), dtype=float
            )
        else:
            salinity_arr = np.full(
                n_samples, DEFAULT_WBM_FRESH_SALINITY_PPM, dtype=float
            )
        rho_mf_arr = brine_density(temp_arr, pres_arr, salinity_arr)

    if rho_w is None:
        rho_w_arr = rho_mf_arr
    else:
        rho_w_arr = _as_float_array(rho_w, n_samples, 1.02)

    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob_arr,
        rhoma=rhoma,
        tvd=tvd,
        mud_type=mud,
        rho_mf=rho_mf_arr,
        rho_w=rho_w_arr,
        rho_h_app=rho_h_app,
        rxo=rxo,
        rt=rt,
        rmf_downhole=rmf_downhole,
        rw_downhole=rw_downhole,
        badhole_flag=badhole_flag,
        fluid_zone=fluid_zone,
        a=a,
        m=m,
        n=n,
        flush_exponent=flush_exponent,
        default_shr_gas=default_shr_gas,
        default_shr_oil=default_shr_oil,
        max_iter=max_iter,
        tol=tol,
    )
    result["RHO_MF"] = rho_mf_arr
    result["SALINITY_PPM"] = salinity_arr
    return result
