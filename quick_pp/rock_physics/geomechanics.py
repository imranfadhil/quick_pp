import numpy as np
import math

from quick_pp.saturation import estimate_rt_water_trend
from quick_pp import logger


def extrapolate_rhob(rhob, tvd, a, b):
    """Extrapolate density log to the surface using a power law model.
    TODO: Revisit

    Args:
        rhob (float): Bulk density in g/cm^3.
        tvd (float): True vertical depth in m.
        a (float): _description_
        b (float): _description_

    Returns:
        float: Extrapolated density in g/cm^3.
    """
    logger.debug(f"Extrapolating density log with parameters: a={a}, b={b}")
    rhob_extrapolated = 1.8 + a * tvd**b
    logger.debug(f"Extrapolated density range: {np.min(rhob_extrapolated):.3f} - {np.max(rhob_extrapolated):.3f} g/cm³")
    return np.append(rhob_extrapolated, rhob)


def estimate_gardner_density(vp, alpha=.31, beta=.25):
    """
    Estimate density from vp using Gardner's relation.

    Args:
        vp (float): P-wave velocity in m/s.
        alpha (float): Gardner's coefficient alpha. 1.75 for shale and 1.66 for sandstone.
        beta (float): Gardner's coefficient beta. 0.265 for shale and 0.261 for sandstone.

    Returns:
        float: Estimated density in g/cm^3.
    """
    logger.debug(f"Estimating density using Gardner's relation with alpha={alpha}, beta={beta}")
    density = alpha * vp**beta
    logger.debug(f"Estimated density range: {np.min(density):.3f} - {np.max(density):.3f} g/cm³")
    return density


def estimate_compressional_velocity(density, alpha=.31, beta=.25):
    """Estimate compressional wave velocity from density using the empirical relation by Gardner et al. 1974.

    Args:
        density (float): Density in g/cm^3.

    Returns:
        float: Estimated P-wave velocity in m/s.
    """
    logger.debug(f"Estimating compressional velocity using Gardner's relation with alpha={alpha}, beta={beta}")
    vp = (density / alpha)**(1 / beta)
    logger.debug(f"Estimated P-wave velocity range: {np.min(vp):.1f} - {np.max(vp):.1f} m/s")
    return vp


def estimate_shear_velocity(vp):
    """Estimate shear wave velocity from P-wave velocity using the empirical relation by Castagna et al. 1985.

    Args:
        vp (float): P-wave velocity in m/s.

    Returns:
        float: Estimated S-wave velocity in m/s.
    """
    logger.debug("Estimating shear wave velocity using Castagna et al. 1985 relation")
    vs = 0.8043 * vp - 855.9
    logger.debug(f"Estimated S-wave velocity range: {np.min(vs):.1f} - {np.max(vs):.1f} m/s")
    return vs


def estimate_hydrostatic_pressure(tvd, rhob_water=1.0, g=9.81):
    """Estimate hydrostatic pressure from bulk density and true vertical.

    Args:
        tvd (float): Depth in m.
        rhob_water (float, optional): Bulk density of water in g/cm^3. Defaults to 1.0.
        g (float, optional): Acceleration due to gravity in m/s^2. Defaults to 9.81.

    Returns:
        float: Estimated hydrostatic pressure in MPa.
    """
    logger.debug(f"Calculating hydrostatic pressure with water density={rhob_water} g/cm³, g={g} m/s²")
    pressure = 1000 * rhob_water * g * tvd
    logger.debug(f"Hydrostatic pressure range: {np.min(pressure):.2f} - {np.max(pressure):.2f} MPa")
    return pressure


def estimate_overburden_pressure(rhob, tvd, rhob_water=1.0, depth_water=0, g=9.81):
    """Estimate overburden pressure from bulk density and true vertical depth.
    TODO: Revisit

    Args:
        rhob (float): Bulk density in g/cm^3.
        tvd (float): Depth in m.
        g (float, optional): Acceleration due to gravity in m/s^2. Defaults to 9.81.

    Returns:
        float: Estimated overburden pressure in MPa.
    """
    logger.debug(f"Calculating overburden pressure with water depth={depth_water} m")
    pressure = 1000 * (rhob_water * g * depth_water + np.cumsum(rhob) * g * tvd)
    logger.debug(f"Overburden pressure range: {np.min(pressure):.2f} - {np.max(pressure):.2f} MPa")
    return pressure


def estimate_pore_pressure_dt(s, p_hyd, dtc, dtc_shale, x=3.0):
    """Estimate pore pressure from sonic transit time and hydrostatic pressure, based on Eaton's method.

    Args:
        s (float): Overburden stress in MPa.
        p_hyd (float): Hydrostatic pressure in MPa.
        dtc (float): Compressional sonic transit time in us/ft.
        dtc_shale (float): Compressional sonic transit time in us/ft for shale. Represents the normal compaction trend.
        x (float, optional): Exponent for the empirical relation. Defaults to 3.0.

    Returns:
        float: Estimated pore pressure in MPa.
    """
    logger.debug(f"Estimating pore pressure from sonic with exponent x={x}")
    pp = s - (s - p_hyd) * (dtc_shale / dtc)**x
    logger.debug(f"Pore pressure range: {np.min(pp):.2f} - {np.max(pp):.2f} MPa")
    return pp


def estimate_pore_pressure_res(s, p_hyd, res, res_shale=None, x=1.2):
    """Estimate pore pressure from resistivity and hydrostatic pressure, based on Eaton's method.

    Args:
        s (float): Overburden stress in MPa.
        p_hyd (float): Hydrostatic pressure in MPa.
        res (float): Resistivity in ohm.m.
        res_shale (float): Resistivity in ohm.m for shale. Represents the normal compaction trend. Defaults to None.
        x (float, optional): Exponent for the empirical relation. Defaults to 1.2.

    Returns:
        float: Estimated pore pressure in MPa.
    """
    if res_shale is None:
        logger.debug("Shale resistivity not provided, estimating from water trend")
        res_shale = estimate_rt_water_trend(res)

    logger.debug(f"Estimating pore pressure from resistivity with exponent x={x}")
    pp = s - (s - p_hyd) * (res_shale / res)**x
    logger.debug(f"Pore pressure range: {np.min(pp):.2f} - {np.max(pp):.2f} MPa")
    return pp


def estimate_fracture_pressure(pp, tvd):
    """Estimate fracture pressure from pore pressure and true vertical depth.
    TODO: Revisit

    Args:
        pp (float): Pore pressure in MPa.
        tvd (float): True vertical depth in m.

    Returns:
        float: Estimated fracture pressure in MPa.
    """
    logger.debug("Estimating fracture pressure using empirical relations")
    pf1 = 1 / 3 * (1 + 2 * pp / tvd)
    pf2 = 1 / 2 * (1 + 2 * pp / tvd)
    logger.debug(f"Fracture pressure estimates: {np.min(pf1):.2f} - {np.max(pf1):.2f} MPa (method 1), "
                 f"{np.min(pf2):.2f} - {np.max(pf2):.2f} MPa (method 2)")
    return pf1, pf2


def estimate_ucs(dtc):
    """Estimate Unconfined Compressive Strength (UCS) from dtc using the empirical relation by McNally 1987.

    Args:
        dtc (float): Compressional sonic transit time in us/ft.

    Returns:
        float: Estimated UCS in MPa.
    """
    logger.debug("Estimating UCS using McNally 1987 relation")
    ucs = 1200 * math.exp(-.036 * dtc)
    logger.debug(f"UCS range: {np.min(ucs):.1f} - {np.max(ucs):.1f} MPa")
    return ucs


def estimate_poisson_ratio(vp, vs):
    """Estimate Poisson's ratio from P-wave and S-wave velocities.

    Args:
        vp (float): P-wave velocity in m/s.
        vs (float): S-wave velocity in m/s.

    Returns:
        float: Estimated Poisson's ratio.
    """
    logger.debug("Calculating Poisson's ratio from P and S wave velocities")
    vp_vs = vp / vs
    poisson_ratio = (.5 * vp_vs**2 - 1) / (vp_vs**2 - 1)
    logger.debug(f"Poisson's ratio range: {np.min(poisson_ratio):.3f} - {np.max(poisson_ratio):.3f}")
    return poisson_ratio


def estimate_shear_modulus(rhob, vs):
    """Estimate shear modulus from density, S-wave velocity and constant a.
     Shear modulus is the shear stiffness of a material, which is the ratio of shear stress to shear strain.
     Also known as the modulus of rigidity.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vs (float): S-wave velocity in m/s.
    """
    logger.debug("Calculating shear modulus from density and S-wave velocity")
    shear_modulus = rhob * vs**2
    logger.debug(f"Shear modulus range: {np.min(shear_modulus):.2e} - {np.max(shear_modulus):.2e} Pa")
    return shear_modulus


def estimate_bulk_modulus(rhob, vp, vs):
    """Estimate Bulk modulus from density, P-wave and S-wave velocities.
     Bulk modulus is the measure of resistance of a material to bulk compression.
     It is the reciprocal of compressibility.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vp (float): P-wave velocity in m/s.
        vs (float): S-wave velocity in m/s.

    Returns:
        float: Estimated Bulk modulus in Pa.
    """
    logger.debug("Calculating bulk modulus from density, P-wave and S-wave velocities")
    shear_modulus = estimate_shear_modulus(rhob, vs)
    bulk_modulus = rhob * vp**2 - 4 / 3 * shear_modulus
    logger.debug(f"Bulk modulus range: {np.min(bulk_modulus):.2e} - {np.max(bulk_modulus):.2e} Pa")
    return bulk_modulus


def estimate_young_modulus(rhob, vp, vs):
    """Estimate Young's modulus from density, P-wave and S-wave velocities.
     Young's modulus is the measure of the resistance of a material to stress.
     It quantifies the relationship between stress and strain in a material.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vp (float): P-wave velocity in m/s.
        vs (float): S-wave velocity in m/s.

    Returns:
        float: Estimated Young's modulus in Pa.
    """
    logger.debug("Calculating Young's modulus from density and wave velocities")
    shear_modulus = estimate_shear_modulus(rhob, vs)
    bulk_modulus = estimate_bulk_modulus(rhob, vp, vs)
    young_modulus = shear_modulus * (3 * bulk_modulus + shear_modulus) / (bulk_modulus + shear_modulus)
    logger.debug(f"Young's modulus range: {np.min(young_modulus):.2e} - {np.max(young_modulus):.2e} Pa")
    return young_modulus


def estimate_mohrs_circle(sigma1, sigma3):
    """Estimate Mohr's circle parameters from principal stresses.

    Args:
        sigma1 (float): Major principal stress.
        sigma3 (float): Minor principal stress.

    Returns:
        tuple: (center, radius) of Mohr's circle.
    """
    center = (sigma1 + sigma3) / 2
    radius = (sigma1 - sigma3) / 2
    normal_stress = np.linspace(sigma3, sigma1, 100)
    shear_stress = np.sqrt(radius**2 - (normal_stress - center)**2)

    return shear_stress, normal_stress


def estimate_mohrs_coulomb_failure(sigma1, sigma3, return_tangent_points=False):
    """
    Determines the Mohr-Coulomb failure envelope from multiple Mohr's circles.

    This function calculates the best-fit straight line (failure envelope) that is
    tangent to a series of Mohr's circles defined by pairs of major (sigma1) and
    minor (sigma3) principal stresses. It uses linear regression to find the
    cohesion (c) and the angle of internal friction (phi).

    Args:
        sigma1 (array-like): Major principal stresses.
        sigma3 (array-like): Minor principal stresses.
        return_tangent_points (bool, optional): If True, also returns the coordinates
                                                of the tangent points. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - cohesion (float): The y-intercept of the failure envelope (c).
            - friction_angle (float): The angle of internal friction in degrees (phi).
            - (Optional) tangent_points (tuple): A tuple of (normal_stresses, shear_stresses)
                                                  at the tangent points.
    """
    sigma1 = np.asarray(sigma1)
    sigma3 = np.asarray(sigma3)

    if sigma1.shape != sigma3.shape or sigma1.ndim != 1:
        raise ValueError("sigma1 and sigma3 must be 1D arrays of the same shape.")

    centers = (sigma1 + sigma3) / 2
    radii = (sigma1 - sigma3) / 2

    # Fit a line to the centers and radii of the circles
    # The slope of this line is sin(phi) and intercept is c*cos(phi)
    sin_phi, c_cos_phi = np.polyfit(centers, radii, 1)

    # Calculate cohesion (c) and friction angle (phi)
    friction_angle_rad = np.arcsin(sin_phi)
    cohesion = c_cos_phi / np.cos(friction_angle_rad)

    # Calculate slope (tan(phi)) and intercept (c) of the failure envelope
    slope = np.tan(friction_angle_rad)
    intercept = cohesion

    if return_tangent_points:
        # Calculate tangent points for plotting
        tangent_normal_stress = centers - radii * np.sin(friction_angle_rad)
        tangent_shear_stress = radii * np.cos(friction_angle_rad)
        # Calculate angle from horizontal east line of the individual circles to the tangent points
        angles = np.arctan2(tangent_shear_stress, tangent_normal_stress - centers)

        return intercept, slope, (tangent_normal_stress, tangent_shear_stress, angles)

    return intercept, slope


def plot_mohrs_circle(sigma1, sigma3, ax=None, **kwargs):
    """Plot Mohr's semi-circle on a shear vs. normal stress plot.
    Can plot single or multiple circles if inputs are iterables.

    Args:
        sigma1 (float or array-like): Major principal stress(es).
        sigma3 (float or array-like): Minor principal stress(es).
        ax (matplotlib.axes.Axes, optional): Matplotlib axes object to plot on.
                                              If None, a new figure and axes will be created.
                                              Defaults to None.
        **kwargs: Additional keyword arguments to be passed to `ax.plot()`.

    Returns:
        matplotlib.axes.Axes: The matplotlib axes object with the plot.
    """
    import matplotlib.pyplot as plt
    import collections.abc

    if ax is None:
        fig, ax = plt.subplots()

    # Ensure sigma1 and sigma3 are iterable
    if not isinstance(sigma1, collections.abc.Iterable):
        sigma1 = [sigma1]
    if not isinstance(sigma3, collections.abc.Iterable):
        sigma3 = [sigma3]

    if len(sigma1) != len(sigma3):
        raise ValueError("sigma1 and sigma3 must have the same length.")

    for s1, s3 in zip(sigma1, sigma3):
        shear_stress, normal_stress = estimate_mohrs_circle(s1, s3)
        ax.plot(normal_stress, shear_stress, **kwargs)

    cohesion, tan_friction_angle, (tangent_x, tangent_y, angles) = estimate_mohrs_coulomb_failure(
        sigma1, sigma3, return_tangent_points=True)
    x = np.linspace(0, np.max(sigma1), 100)
    failure_line = cohesion + tan_friction_angle * x
    beta_angle = np.unique(angles * 180 / np.pi)[0] / 2
    ax.plot(x, failure_line,
            label=f'Cohesion: {cohesion:.2f}\n'
            f'Friction Angle: {np.degrees(np.arctan(tan_friction_angle)):.2f}°\n'
            f'Beta Angles: {beta_angle:.2f}°')

    # Plot the tangent points
    ax.plot(tangent_x, tangent_y, 'ro', label='Tangent Points')

    ax.set_xlabel("Normal Stress (σ)")
    ax.set_ylabel("Shear Stress (τ)")
    ax.set_title("Mohr's Circle")
    ax.grid(True)
    ax.axis('equal')
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.legend()

    return ax
