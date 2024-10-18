import numpy as np
import math

from quick_pp.saturation import estimate_rt_water_trend


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
    rhob_extrapolated = 1.8 + a * tvd**b
    return np.append(rhob_extrapolated, rhob)


def estimate_gardner_density(vp, alpha=.23, beta=.25):
    """
    Estimate density from vp using Gardner's relation.

    Args:
        vp (float): P-wave velocity in ft/s.
        alpha (float): Gardner's coefficient alpha. 1.75 for shale and 1.66 for sandstone.
        beta (float): Gardner's coefficient beta. 0.265 for shale and 0.261 for sandstone.

    Returns:
        float: Estimated density in g/cm^3.
    """
    return alpha * vp**beta


def estimate_shear_wave_velocity(vp):
    """Estimate shear wave velocity from P-wave velocity using the empirical relation by Castagna et al. 1985.

    Args:
        vp (float): P-wave velocity in m/s.

    Returns:
        float: Estimated S-wave velocity in m/s.
    """
    return 0.8043 * vp - 855.9


def estimate_hydrostatic_pressure(tvd, rhob_water=1.0, g=9.81):
    """Estimate hydrostatic pressure from bulk density and true vertical.

    Args:
        tvd (float): Depth in m.
        rhob_water (float, optional): Bulk density of water in g/cm^3. Defaults to 1.0.
        g (float, optional): Acceleration due to gravity in m/s^2. Defaults to 9.81.

    Returns:
        float: Estimated hydrostatic pressure in MPa.
    """
    return 1000 * rhob_water * g * tvd


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
    return 1000 * (rhob_water * g * depth_water + np.cumsum(rhob) * g * tvd)


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
    return s - (s - p_hyd) * (dtc_shale / dtc)**x


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
    res_shale = res_shale or estimate_rt_water_trend(res)
    return s - (s - p_hyd) * (res_shale / res)**x


def estimate_fracture_pressure(pp, tvd):
    """Estimate fracture pressure from pore pressure and true vertical depth.
    TODO: Revisit

    Args:
        pp (float): Pore pressure in MPa.
        tvd (float): True vertical depth in m.

    Returns:
        float: Estimated fracture pressure in MPa.
    """
    return 1 / 3 * (1 + 2 * pp / tvd), 1 / 2 * (1 + 2 * pp / tvd)


def estimate_ucs(dtc):
    """Estimate Unconfined Compressive Strength (UCS) from dtc using the empirical relation by McNally 1987.

    Args:
        dtc (float): Compressional sonic transit time in us/ft.

    Returns:
        float: Estimated UCS in MPa.
    """
    return 1200 * math.exp(-.036 * dtc)


def estimate_poisson_ratio(vp, vs):
    """Estimate Poisson's ratio from P-wave and S-wave velocities.

    Args:
        vp (float): P-wave velocity in m/s.
        vs (float): S-wave velocity in m/s.

    Returns:
        float: Estimated Poisson's ratio.
    """
    return (vp**2 - 2 * vs**2) / (2 * (vp**2 - vs**2))


def estimate_shear_modulus(rhob, vs, a):
    """Estimate shear modulus from density, S-wave velocity and constant a.
     Shear modulus is the shear stiffness of a material, which is the ratio of shear stress to shear strain.
     Also known as the modulus of rigidity.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vs (float): S-wave velocity in m/s.
        a (float): _description_
    """
    return a * rhob / vs**2


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
    return rhob * 1000 * vp**2 * (3 * vp**2 - 4 * vs**2) / (vp**2 - vs**2)


def estimate_bulk_modulus(rhob, vp, vs, a):
    """Estimate Bulk modulus from density, P-wave and S-wave velocities.
     Bulk modulus is the measure of resistance of a material to bulk compression.
     It is the reciprocal of compressibility.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vp (float): P-wave velocity in m/s.
        vs (float): S-wave velocity in m/s.
        a (float): _description_

    Returns:
        float: Estimated Bulk modulus in Pa.
    """
    return a * rhob * (1 / vp**2 - 4 / (3 * vs**2))
