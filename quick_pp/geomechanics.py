import numpy as np
import math

from quick_pp.saturation import estimate_rt_water_trend


def estimate_hydrostatic_pressure(depth, rhob_water=1.0, g=9.81):
    """Estimate hydrostatic pressure from bulk density and depth.

    Args:
        depth (float): Depth in m.
        rhob_water (float, optional): Bulk density of water in g/cm^3. Defaults to 1.0.
        g (float, optional): Acceleration due to gravity in m/s^2. Defaults to 9.81.

    Returns:
        float: Estimated hydrostatic pressure in MPa.
    """
    return rhob_water * g * depth


def estimate_overburden_pressure(rhob, depth, rhob_water=1.0, depth_water=0, g=9.81):
    # TODO: revisit
    """Estimate overburden pressure from bulk density and depth.

    Args:
        rhob (float): Bulk density in g/cm^3.
        depth (float): Depth in m.
        g (float, optional): Acceleration due to gravity in m/s^2. Defaults to 9.81.

    Returns:
        float: Estimated overburden pressure in MPa.
    """
    return rhob_water * g * depth_water + np.cumsum(rhob)[-1] * g * depth


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


def estimate_fracture_pressure(pp, tvdss):
    # TODO: Revisit
    return 1 / 3 * (1 + 2 * pp / tvdss), 1 / 2 * (1 + 2 * pp / tvdss)


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
    """Estimate shear modulus from density, S-wave velocity and a constant a.

    Args:
        rhob (_type_): _description_
        vs (_type_): _description_
        a (_type_): _description_
    """
    return a * rhob / vs**2


def estimate_young_modulus(rhob, vp, vs):
    """Estimate Young's modulus from density, P-wave and S-wave velocities.
     Young's modulus measures the resistance of a material to stress where it quantifies the relationship between
     stress and strain in a material.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vp (_type_): P-wave velocity in m/s.
        vs (_type_): S-wave velocity in m/s.

    Returns:
        float: Estimated Young's modulus in Pa.
    """
    return rhob * 1000 * vp**2 * (3 * vp**2 - 4 * vs**2) / (vp**2 - vs**2)


def estimate_bulk_modulus(rhob, vp, vs, a):
    """Estimate Bulk modulus which is the reciprocal of compressibility.

    Args:
        rhob (float): Bulk density in g/cm^3.
        vp (_type_): P-wave velocity in m/s.
        vs (_type_): S-wave velocity in m/s.
        a (_type_): _description_

    Returns:
        float: Estimated Bulk modulus in Pa.
    """
    return a * rhob * (1 / vp**2 - 4 / (3 * vs**2))


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


def extrapolate_rhob(rhob, tvdss, a, b):
    # TODO: Revisit
    rhob_extrapolated = 0 + a * tvdss**b
    return np.append(rhob_extrapolated, rhob)
