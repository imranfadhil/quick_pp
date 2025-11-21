import numpy as np
import math
import matplotlib.pyplot as plt
import collections.abc

from quick_pp.saturation import estimate_rt_water_trend
from quick_pp.rock_type import estimate_vsh_gr
from quick_pp import logger


def extrapolate_rhob(rhob, tvd, a, b):
    """Extrapolate density log to the surface using a power law model.
    TODO: Revisit

    Args:
        rhob (np.ndarray or float): Bulk density log [g/cm³].
        tvd (np.ndarray or float): True vertical depth [m].
        a (float): The scaling factor for the power law model.
        b (float): The exponent for the power law model.

    Returns:
        np.ndarray: The extrapolated density log, concatenated with the original [g/cm³].
    """
    logger.debug(f"Extrapolating density log with parameters: a={a}, b={b}")
    rhob_extrapolated = 1.8 + a * tvd**b
    logger.debug(
        f"Extrapolated density range: {np.min(rhob_extrapolated):.3f} - {np.max(rhob_extrapolated):.3f} g/cm³"
    )
    return np.append(rhob_extrapolated, rhob)


def estimate_gardner_density(vp, alpha=0.31, beta=0.25):
    """
    Estimate density from vp using Gardner's relation.

    Args:
        vp (np.ndarray or float): P-wave velocity [m/s].
        alpha (float, optional): Gardner's coefficient. Defaults to 0.31.
        beta (float, optional): Gardner's exponent. Defaults to 0.25.

    Returns:
        np.ndarray or float: Estimated density [g/cm³].
    """
    logger.debug(
        f"Estimating density using Gardner's relation with alpha={alpha}, beta={beta}"
    )
    density = alpha * vp**beta
    logger.debug(
        f"Estimated density range: {np.min(density):.3f} - {np.max(density):.3f} g/cm³"
    )
    return density


def estimate_compressional_velocity(density, alpha=0.31, beta=0.25):
    """Estimate compressional wave velocity from density using the empirical relation by Gardner et al. 1974.

    Args:
        density (np.ndarray or float): Bulk density [g/cm³].
        alpha (float, optional): Gardner's coefficient. Defaults to 0.31.
        beta (float, optional): Gardner's exponent. Defaults to 0.25.

    Returns:
        np.ndarray or float: Estimated P-wave velocity [m/s].
    """
    logger.debug(
        f"Estimating compressional velocity using Gardner's relation with alpha={alpha}, beta={beta}"
    )
    vp = (density / alpha) ** (1 / beta)
    logger.debug(
        f"Estimated P-wave velocity range: {np.min(vp):.1f} - {np.max(vp):.1f} m/s"
    )
    return vp


def estimate_shear_velocity(vp):
    """Estimate shear wave velocity from P-wave velocity using the empirical relation by Castagna et al. 1985.

    Args:
        vp (np.ndarray or float): P-wave velocity [m/s].

    Returns:
        np.ndarray or float: Estimated S-wave velocity [m/s].
    """
    logger.debug("Estimating shear wave velocity using Castagna et al. 1985 relation")
    vs = 0.8043 * vp - 855.9
    logger.debug(
        f"Estimated S-wave velocity range: {np.min(vs):.1f} - {np.max(vs):.1f} m/s"
    )
    return vs


def estimate_hydrostatic_pressure(tvd, rhob_water=1.0, g=9.81):
    """Estimate hydrostatic pressure from bulk density and true vertical.

    Args:
        tvd (np.ndarray or float): True vertical depth [m].
        rhob_water (float, optional): Bulk density of water [g/cm³]. Defaults to 1.0.
        g (float, optional): Acceleration due to gravity [m/s²]. Defaults to 9.81.

    Returns:
        np.ndarray or float: Estimated hydrostatic pressure [MPa].
    """
    logger.debug(
        f"Calculating hydrostatic pressure with water density={rhob_water} g/cm³, g={g} m/s²"
    )
    pressure = 1000 * rhob_water * g * tvd
    logger.debug(
        f"Hydrostatic pressure range: {np.min(pressure):.2f} - {np.max(pressure):.2f} MPa"
    )
    return pressure


def estimate_overburden_pressure(rhob, tvd, rhob_water=1.0, water_depth=0, g=9.81):
    """Estimate overburden pressure from bulk density and true vertical depth.
    TODO: Revisit

    Args:
        rhob (np.ndarray or float): Bulk density log [g/cm³].
        tvd (np.ndarray or float): True vertical depth [m].
        rhob_water (float, optional): Bulk density of water [g/cm³]. Defaults to 1.0.
        water_depth (float, optional): The depth of the water column [m]. Defaults to 0.
        g (float, optional): Acceleration due to gravity [m/s²]. Defaults to 9.81.

    Returns:
        np.ndarray or float: Estimated overburden pressure [MPa].
    """
    logger.debug(f"Calculating overburden pressure with water depth={water_depth} m")
    pressure = 1000 * (rhob_water * g * water_depth + np.cumsum(rhob) * g * tvd)
    logger.debug(
        f"Overburden pressure range: {np.min(pressure):.2f} - {np.max(pressure):.2f} MPa"
    )
    return pressure


def estimate_normal_compaction_trend_dt(dtc, vshale, depth, vsh_threshold=0.6):
    """Estimate normal compaction trend from sonic transit time accounting shale intervals only.

    This function identifies shale intervals based on a Vshale threshold, then fits a
    logarithmic trendline to the sonic transit time (dtc) in those intervals against depth.
    This trendline represents the normal compaction trend.

    Args:
        dtc (np.ndarray): Compressional sonic transit time [us/ft].
        vshale (np.ndarray): Volume of shale [v/v].
        depth (np.ndarray): Depth array, typically TVD [m].
        vsh_threshold (float, optional): Vshale cutoff to identify shales [v/v]. Defaults to 0.6.

    Returns:
        np.ndarray: The normal compaction trend for dtc over the entire depth range [us/ft].
    """
    # Ensure inputs are numpy arrays and handle NaNs
    dtc = np.asarray(dtc)
    vshale = np.asarray(vshale)
    depth = np.asarray(depth)

    # Create a mask for valid, finite data points
    valid_data_mask = (
        np.isfinite(dtc) & np.isfinite(vshale) & np.isfinite(depth) & (depth > 0)
    )

    # Identify shale intervals
    shale_mask = (vshale >= vsh_threshold) & valid_data_mask
    dtc_shale = dtc[shale_mask]
    depth_shale = depth[shale_mask]

    if len(depth_shale) < 2:
        logger.warning(
            "Not enough valid shale data points (< 2) to estimate normal compaction trend. "
            "Returning original dtc."
        )
        return dtc, None

    # Estimate trendline on the filtered data
    # A logarithmic fit is common for compaction trends: dt = a - b*log(depth)
    try:
        params = np.polyfit(np.log(depth_shale), dtc_shale, 1)
        dtc_nct = params[0] * np.log(depth) + params[1]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(1 / dtc_shale, depth_shale, alpha=0.5, label="Shale Data Points")
        # Use the calculated trend for the full depth range for plotting
        valid_depth_mask = depth > 0
        ax.plot(
            1 / dtc_nct[valid_depth_mask],
            depth[valid_depth_mask],
            "r-",
            label=f"NCT Fit (y = {params[0]:.2f}*log(x) + {params[1]:.2f})",
        )
        ax.set_xlabel("Compressional Sonic (dtc) [us/ft]")
        ax.set_ylabel("Depth [m]")
        ax.set_title("Normal Compaction Trend (NCT) on Shales")
        ax.invert_yaxis()
        ax.set_xscale("log")
        ax.set_ylim(top=np.min(depth_shale) - 100)
        ax.grid(True)
        ax.legend()
        fig.tight_layout()

        return dtc_nct
    except np.linalg.LinAlgError as e:
        logger.error(
            f"Failed to fit normal compaction trend due to a linear algebra error: {e}"
        )
        return dtc, None


def estimate_pore_pressure_dt(s, p_hyd, dtc, dtc_shale, x=3.0):
    """Estimate pore pressure from sonic transit time and hydrostatic pressure, based on Eaton's method.

    Args:
        s (np.ndarray or float): Overburden stress [MPa].
        p_hyd (np.ndarray or float): Hydrostatic pressure [MPa].
        dtc (np.ndarray or float): Compressional sonic transit time [us/ft].
        dtc_shale (np.ndarray or float): Normal compaction trend for sonic transit time [us/ft].
        x (float, optional): Exponent for the empirical relation. Defaults to 3.0.

    Returns:
        np.ndarray or float: Estimated pore pressure [MPa].
    """
    logger.debug(f"Estimating pore pressure from sonic with exponent x={x}")
    pp = s - (s - p_hyd) * (dtc_shale / dtc) ** x
    logger.debug(f"Pore pressure range: {np.min(pp):.2f} - {np.max(pp):.2f} MPa")
    return pp


def estimate_pore_pressure_res(s, p_hyd, res, res_shale=None, x=1.2):
    """Estimate pore pressure from resistivity and hydrostatic pressure, based on Eaton's method.

    Args:
        s (np.ndarray or float): Overburden stress [MPa].
        p_hyd (np.ndarray or float): Hydrostatic pressure [MPa].
        res (np.ndarray or float): Formation resistivity [ohm.m].
        res_shale (np.ndarray or float, optional): Normal compaction trend for resistivity [ohm.m]. Defaults to None.
        x (float, optional): Exponent for the empirical relation. Defaults to 1.2.

    Returns:
        np.ndarray or float: Estimated pore pressure [MPa].
    """
    if res_shale is None:
        logger.debug("Shale resistivity not provided, estimating from water trend")
        res_shale = estimate_rt_water_trend(res)

    logger.debug(f"Estimating pore pressure from resistivity with exponent x={x}")
    pp = s - (s - p_hyd) * (res_shale / res) ** x
    logger.debug(f"Pore pressure range: {np.min(pp):.2f} - {np.max(pp):.2f} MPa")
    return pp


def estimate_fracture_pressure(pp, tvd):
    """Estimate fracture pressure from pore pressure and true vertical depth.
    TODO: Revisit

    Args:
        pp (np.ndarray or float): Pore pressure [MPa].
        tvd (np.ndarray or float): True vertical depth [m].

    Returns:
        tuple[np.ndarray, np.ndarray]: Two estimations of fracture pressure [MPa].
    """
    logger.debug("Estimating fracture pressure using empirical relations")
    pf1 = 1 / 3 * (1 + 2 * pp / tvd)
    pf2 = 1 / 2 * (1 + 2 * pp / tvd)
    logger.debug(
        f"Fracture pressure estimates: {np.min(pf1):.2f} - {np.max(pf1):.2f} MPa (method 1), "
        f"{np.min(pf2):.2f} - {np.max(pf2):.2f} MPa (method 2)"
    )
    return pf1, pf2


def estimate_ucs(dtc):
    """Estimate Unconfined Compressive Strength (UCS) from dtc using the empirical relation by McNally 1987.

    Args:
        dtc (np.ndarray or float): Compressional sonic transit time [us/ft].

    Returns:
        np.ndarray or float: Estimated UCS [MPa].
    """
    logger.debug("Estimating UCS using McNally 1987 relation")
    ucs = 1200 * math.exp(-0.036 * dtc)
    logger.debug(f"UCS range: {np.min(ucs):.1f} - {np.max(ucs):.1f} MPa")
    return ucs


def estimate_friction_angle(gr):
    """Estimate internal friction angle using GR.

    Args:
        gr (np.ndarray or float): Gamma ray log [API].

    Returns:
        np.ndarray or float: Internal friction angle in degrees, bounded between 15 and 40.
    """
    return (40 - 0.1875 * (gr - 13.33)).clip(15, 40)


def estimate_cohesion(ucs, friction_angle):
    """Estimate cohesion from UCS and internal friction angle.

    Args:
        ucs (np.ndarray or float): Unconfined compressive strength [MPa].
        friction_angle (np.ndarray or float): Internal friction angle [degrees].

    Returns:
        np.ndarray or float: Estimated cohesion [MPa].
    """
    return ucs * (1 - np.sin(friction_angle)) / (2 * np.cos(friction_angle))


def estimate_poisson_ratio(vp, vs):
    """Estimate dynamic Poisson's ratio from P-wave and S-wave velocities. Usually is assumed the same as static,
     especially if the lab radial strain measurement uncertainty).

    Args:
        vp (np.ndarray or float): P-wave velocity [m/s].
        vs (np.ndarray or float): S-wave velocity [m/s].

    Returns:
        np.ndarray or float: Estimated dynamic Poisson's ratio (unitless).
    """
    logger.debug("Calculating dynamic Poisson's ratio from P and S wave velocities")
    vp_vs = vp / vs
    poisson_ratio = (0.5 * vp_vs**2 - 1) / (vp_vs**2 - 1)
    logger.debug(
        f"Poisson's ratio range: {np.min(poisson_ratio):.3f} - {np.max(poisson_ratio):.3f}"
    )
    return poisson_ratio


def estimate_shear_modulus(rhob, vs):
    """Estimate dynamic shear modulus from density, S-wave velocity and constant a.
     Shear modulus is the shear stiffness of a material, which is the ratio of shear stress to shear strain.
     Also known as the modulus of rigidity.

    Args:
        rhob (np.ndarray or float): Bulk density [g/cm³].
        vs (np.ndarray or float): S-wave velocity [m/s].

    Returns:
        np.ndarray or float: Estimated dynamic shear modulus [Pa].
    """
    logger.debug("Calculating dynamic shear modulus from density and S-wave velocity")
    shear_modulus = rhob * vs**2
    logger.debug(
        f"Shear modulus range: {np.min(shear_modulus):.2e} - {np.max(shear_modulus):.2e} Pa"
    )
    return shear_modulus


def estimate_bulk_modulus(rhob, vp, vs):
    """Estimate dynamic Bulk modulus from density, P-wave and S-wave velocities.
     Bulk modulus is the measure of resistance of a material to bulk compression.
     It is the reciprocal of compressibility.

    Args:
        rhob (np.ndarray or float): Bulk density [g/cm³].
        vp (np.ndarray or float): P-wave velocity [m/s].
        vs (np.ndarray or float): S-wave velocity [m/s].

    Returns:
        np.ndarray or float: Estimated dynamic Bulk modulus [Pa].
    """
    logger.debug("Calculating bulk modulus from density, P-wave and S-wave velocities")
    shear_modulus = estimate_shear_modulus(rhob, vs)
    bulk_modulus = rhob * vp**2 - 4 / 3 * shear_modulus
    logger.debug(
        f"Bulk modulus range: {np.min(bulk_modulus):.2e} - {np.max(bulk_modulus):.2e} Pa"
    )
    return bulk_modulus


def estimate_young_modulus(rhob, vp, vs):
    """Estimate dynamic Young's modulus from density, P-wave and S-wave velocities.
     Young's modulus is the measure of the resistance of a material to stress.
     It quantifies the relationship between stress and strain in a material.

    Args:
        rhob (np.ndarray or float): Bulk density [g/cm³].
        vp (np.ndarray or float): P-wave velocity [m/s].
        vs (np.ndarray or float): S-wave velocity [m/s].

    Returns:
        np.ndarray or float: Estimated dynamic Young's modulus [Pa].
    """
    logger.debug("Calculating dynamic Young's modulus from density and wave velocities")
    shear_modulus = estimate_shear_modulus(rhob, vs)
    bulk_modulus = estimate_bulk_modulus(rhob, vp, vs)
    young_modulus = (
        shear_modulus
        * (3 * bulk_modulus + shear_modulus)
        / (bulk_modulus + shear_modulus)
    )
    logger.debug(
        f"Young's modulus range: {np.min(young_modulus):.2e} - {np.max(young_modulus):.2e} Pa"
    )
    return young_modulus


def estimate_shmin(vp, vs, rhob, gr, tvd, biot_coef=1):
    """Estimate minimum horizontal stress (Shmin) using an empirical formula.

    This function calculates Shmin based on Poisson's ratio, overburden stress,
    pore pressure, and Biot's coefficient. It internally estimates several
    required parameters if they are not provided.

    Args:
        vp (np.ndarray): P-wave velocity [m/s].
        vs (np.ndarray): S-wave velocity [m/s].
        rhob (np.ndarray): Bulk density [g/cm³].
        gr (np.ndarray): Gamma ray log [API].
        tvd (np.ndarray): True vertical depth [m].
        biot_coef (float, optional): Biot's coefficient, representing the ratio of fluid
                                     pressure to rock stress. Defaults to 1.

    Returns:
        np.ndarray: Estimated minimum horizontal stress (Shmin) [MPa].
    """
    hydrostatic_pres = estimate_hydrostatic_pressure(tvd)
    overburden_pres = estimate_overburden_pressure(rhob, tvd)
    vshale = estimate_vsh_gr(gr)
    dtc_shale = estimate_normal_compaction_trend_dt(1 / vp, vshale, tvd)
    pore_pres = estimate_pore_pressure_dt(
        overburden_pres, hydrostatic_pres, 1 / vp, dtc_shale
    )
    poisson_ratio = estimate_poisson_ratio(vp, vs)
    k = poisson_ratio / (1 - poisson_ratio)
    return k * (overburden_pres - pore_pres * biot_coef)


def estimate_mohrs_circle(sigma1, sigma3):
    """Estimate Mohr's circle parameters from principal stresses.

    Args:
        sigma1 (np.ndarray or float): Major principal stress.
        sigma3 (np.ndarray or float): Minor principal stress.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the shear and normal stresses defining the circle.
    """
    center = (sigma1 + sigma3) / 2
    radius = (sigma1 - sigma3) / 2
    normal_stress = np.linspace(sigma3, sigma1, 100)
    shear_stress = np.sqrt(radius**2 - (normal_stress - center) ** 2)

    return shear_stress, normal_stress


def estimate_mohrs_coulomb_failure(sigma1, sigma3, return_tangent_points=False):
    """
    Determines the Mohr-Coulomb failure envelope from multiple Mohr's circles.

    This function calculates the best-fit straight line (failure envelope) that is
    tangent to a series of Mohr's circles defined by pairs of major (sigma1) and
    minor (sigma3) principal stresses. It uses linear regression to find the
    cohesion (c) and the angle of internal friction (phi).

    Args:
        sigma1 (np.ndarray or float): Major principal stresses.
        sigma3 (np.ndarray or float): Minor principal stresses.
        return_tangent_points (bool, optional): If True, also returns the coordinates
                                                of the tangent points. Defaults to False.

    Returns:
        tuple: A tuple containing cohesion, friction angle, and optionally,
               the tangent points (normal stress, shear stress, beta angle).
    """
    sigma1 = np.asarray(sigma1)
    sigma3 = np.asarray(sigma3)

    if sigma1.shape != sigma3.shape or sigma1.ndim != 1:
        raise ValueError("sigma1 and sigma3 must be 1D arrays of the same shape.")

    centers = (sigma1 + sigma3) / 2
    radii = (sigma1 - sigma3) / 2

    # Fit a line to the centers and radii of the circles: Radius=sin(ϕ)⋅Center+c⋅cos(ϕ)
    sin_phi, c_cos_phi = np.polyfit(centers, radii, 1)

    # Calculate cohesion (c) and friction angle (phi)
    phi = np.arcsin(sin_phi)
    friction_angle = math.degrees(phi)
    cohesion = c_cos_phi / np.cos(phi)

    if return_tangent_points:
        # Calculate tangent points for plotting
        tangent_normal_stress = centers - radii * np.sin(phi)
        tangent_shear_stress = radii * np.cos(phi)
        # Calculate angle from horizontal east line of the individual circles to the tangent points
        beta2_rad = np.arctan2(tangent_shear_stress, tangent_normal_stress - centers)[0]

        return (
            cohesion,
            friction_angle,
            (tangent_normal_stress, tangent_shear_stress, beta2_rad),
        )

    return cohesion, friction_angle


def plot_mohrs_circle(sigma1, sigma3, ax=None, **kwargs):
    """Plot Mohr's semi-circle on a shear vs. normal stress plot.
    Can plot single or multiple circles if inputs are iterables.

    Args:
        sigma1 (np.ndarray or float): Major principal stress(es).
        sigma3 (np.ndarray or float): Minor principal stress(es).
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on.
                                              If None, a new figure and axes will be created.
                                              Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `ax.plot()`.

    Returns:
        matplotlib.axes.Axes: The axes object with the Mohr's circle plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    # Ensure sigma1 and sigma3 are iterable
    if not isinstance(sigma1, collections.abc.Iterable):
        sigma1 = [sigma1]
    if not isinstance(sigma3, collections.abc.Iterable):
        sigma3 = [sigma3]

    if len(sigma1) != len(sigma3):
        raise ValueError("sigma1 and sigma3 must have the same length.")

    cohesion, friction_angle, (tangent_x, tangent_y, beta2_rad) = (
        estimate_mohrs_coulomb_failure(sigma1, sigma3, return_tangent_points=True)
    )
    x = np.linspace(0, np.max(sigma1), 100)
    failure_line = cohesion + np.tan(np.radians(friction_angle)) * x
    beta2 = beta2_rad * 180 / np.pi / 2
    ax.plot(
        x,
        failure_line,
        linestyle="--",
        color="magenta",
        label=f"Cohesion: {cohesion:.2f}\n"
        f"Friction Angle: {friction_angle:.2f}°\n"
        f"Beta Angle: {beta2:.2f}°",
    )

    centers = (np.asarray(sigma1) + np.asarray(sigma3)) / 2
    radii = (np.asarray(sigma1) - np.asarray(sigma3)) / 2
    for s1, s3 in zip(sigma1, sigma3):
        shear_stress, normal_stress = estimate_mohrs_circle(s1, s3)
        (line,) = ax.plot(normal_stress, shear_stress, **kwargs)
        # Plot lines from center to tangent points
        center = (s1 + s3) / 2
        radius = (s1 - s3) / 2
        # Find the index of the current circle's tangent point
        idx = np.where(np.isclose(centers, center) & np.isclose(radii, radius))[0][0]
        ax.plot(
            [center, tangent_x[idx]],
            [0, tangent_y[idx]],
            color=line.get_color(),
            linestyle=":",
        )
        # Plot the tangent and center points
        ax.plot(center, 0, "o", markersize=3, color=line.get_color())
        ax.plot(
            tangent_x[idx], tangent_y[idx], "o", markersize=5, color=line.get_color()
        )

    ax.set_xlabel("Normal Stress (σ)")
    ax.set_ylabel("Shear Stress (τ)")
    ax.set_title("Mohr's Circle")
    ax.grid(True)
    ax.axis("equal")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    return ax


def estimate_rock_strength(sigma1, sigma3):
    """Estimate rock strength parameters from triaxial test data (sigma1 vs sigma3).

    This function performs a linear regression on the major (sigma1) and minor (sigma3)
    principal stresses to determine the unconfined compressive strength (UCS) and
    the parameter 'n'. From these, it calculates the cohesion and the internal
    friction angle based on the Mohr-Coulomb failure criterion. It also generates
    a plot of sigma1 vs sigma3 with the best-fit line and annotations.

    Args:
        sigma1 (np.ndarray or float): Major principal stresses.
        sigma3 (np.ndarray or float): Minor principal stresses.

    Returns:
        tuple[float, float, float]: A tuple containing the cohesion, internal friction angle,
                                    and unconfined compressive strength (UCS).
    """
    n, ucs = np.polyfit(sigma3, sigma1, 1)

    sin_phi = (n - 1) / (n + 1)
    phi = np.arcsin(sin_phi)
    cohesion = ucs * (1 - np.sin(phi)) / (2 * np.cos(phi))
    internal_friction_angle = math.degrees(phi)

    _, ax = plt.subplots()
    ax.plot(sigma3, sigma1, "o")
    x = np.linspace(0, np.max(sigma3), 100)
    ax.plot(
        x,
        n * np.asarray(x) + ucs,
        "r-",
        label=f"UCS: {ucs:.2f}\n"
        f"Cohesion: {cohesion:.2f}\n"
        f"Friction Angle: {internal_friction_angle:.2f}°",
    )
    ax.set_xlabel("Sigma 3 (Minor Principal Stress)")
    ax.set_ylabel("Sigma 1 (Major Principal Stress)")
    ax.set_title("Rock Strength Analysis")
    ax.set_xlim(left=-10)
    ax.set_ylim(bottom=0)
    ax.legend()
    ax.grid(True)

    return cohesion, internal_friction_angle, ucs
