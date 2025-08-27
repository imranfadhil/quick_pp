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
    """Estimate pore pressure from sonic transit time and hydrostatic pressure.

    Args:
        s (float): Overburden stress in MPa.
        p_hyd (float): Hydrostatic pressure in MPa.
        dtc (float): Compressional sonic transit time in us/ft.
        dtc_shale (float): Compressional sonic transit time in us/ft for shale.
        x (float, optional): Exponent for the empirical relation. Defaults to 3.0.

    Returns:
        float: Estimated pore pressure in MPa.
    """
    logger.debug(f"Estimating pore pressure from sonic with exponent x={x}")
    pp = s - (s - p_hyd) * (dtc_shale / dtc)**x
    logger.debug(f"Pore pressure range: {np.min(pp):.2f} - {np.max(pp):.2f} MPa")
    return pp


def estimate_pore_pressure_res(s, p_hyd, res, res_shale=None, x=1.2):
    """Estimate pore pressure from resistivity and hydrostatic pressure.

    Args:
        s (float): Overburden stress in MPa.
        p_hyd (float): Hydrostatic pressure in MPa.
        res (float): Resistivity in ohm.m.
        res_shale (float): Resistivity in ohm.m for shale.
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
