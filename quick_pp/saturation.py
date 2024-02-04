import numpy as np


def archie_saturation(rt, rw, phit, a=1, m=2, n=2):
    """Archie's saturation model.

    Args:
        sw (float): Water saturation.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.

    Returns:
        float: Water saturation.

    """
    swt = ((a / (phit ** m)) * (rw / rt)) ** (1 / n)
    return np.where(swt < 1, swt, 1)


def waxman_smits_saturation(rt, rw, phit, Qv=None, B=None, m=2, n=2):
    """Waxman-Smits saturation model for dispersed clay mineral.
    ref: Ausburn, Brian E., and Robert Freedman. "The Waxman-smits Equation For Shaly Sands:
    I. Simple Methods Of Solution; Ii. Error Analysis." The Log Analyst 26 (1985)

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.
        B (float): Conductance parameter.
        Qv (float): Cation exchange capacity per unit total pore volume (meq/cm3).
        m (float): Saturation factor.
        n (float): Cementartion factor.

    Returns:
        float: Water saturation.
    """
    # Estimate B at 25 degC if not provided
    if B is None:
        B = (1 - 0.83 * np.exp(-0.5 / rw)) * 3.83

    if Qv is None:
        Qv = 0.3

    # Initial guess
    swt = 1
    swt_i = 0
    for i in range(50):
        fx = swt**n + rw * B * Qv * swt**(n - 1) - (phit**-m * rw / rt)  # Ausburn, 1985
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

    return np.where(swt < 1, swt, 1)


def dual_water_saturation(rt, rw, phit, a, m, n, swb, rwb):
    """Dual water saturation model, extension from Waxman-Smits.

    Args:
        sw (float): Water saturation.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.
        swb (float): Bound water saturation.
        rwb (float): Bound water resistivity.

    Returns:
        float: Water saturation.

    """
    # TODO: Estimate swb and rwb if not provided
    # Initial guess
    swt = 1
    swt_i = 0
    for i in range(50):
        fx = phit**m * swt**n / a * (1 / rw * (swb / swt) * (1 / rwb - 1 / rw)) - 1 / rt
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

    return np.where(swt < 1, swt, 1)


def indonesian_saturation(rt, rw, phie, vsh, rsh, a, m, n):
    """Indonesian saturation model may work well with fresh formation water. Based on Poupon-Leveaux 1971.

    Args:
        sw (float): Water saturation.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.

    Returns:
        float: Water saturation.

    """
    swt = ((1 / rt)**(1 / 2) / ((vsh**(1 - 0.5 * vsh) / rsh**(1 / 2)) + (phie**m / (a * rw))**(1 / 2)))**(2 / n)
    return np.where(swt < 1, swt, 1)


def simandoux_saturation(rt, rw, phit, vsh, rsh, a, m):
    """Simandoux's saturation model.

    Args:
        sw (float): Water saturation.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.

    Returns:
        float: Water saturation.

    """
    shale_factor = vsh / rsh
    swt = (a * rw / (2 * phit**m)) * ((shale_factor**2 + (4 * phit**m / (a * rw * rt)))**(1 / 2) - shale_factor)
    return np.where(swt < 1, swt, 1)


def modified_simandoux_saturation(sw, a, m, n):
    """TODO: Modified Simandoux's saturation model.
    """
    pass


def saturation_height_function():
    """TODO: Saturation height function.
    """
    pass


def estimate_temperature_gradient(tvd, unit='metric'):
    """Estimate formation temperature based on gradient of 25 degC/km or 15 degF/1000ft.

    Args:
        tvd (float or array): True vertical depth in m.

    Returns:
        float or array: Formation temperature in degC.
    """
    assert unit in ['metric', 'imperial'], "Please choose from 'metric' or 'imperial' units."
    return 32 + 25 * tvd / 1000 if unit == 'metric' else 90 + 15 * tvd / 1000


def estimate_b_waxman_smits(T, rw):
    """Estimating B (conductance parameter) for Waxman-Smits model based on Juhasz 1981.

    Args:
        T (float or array): Temperature in degC.
        rw (float or array): Water resistivity in ohm.m.

    Returns:
        float: B parameter.
    """
    return (-1.28 + 0.225 * T - 0.0004059 * T**2) / (1 + (0.045 * T - 0.27) * rw**1.23)


def estimate_rw_temperature_salinity(temperature_gradient, water_salinity):
    """Estimate formation water resistivity based on temperature gradient and water salinity.

    Args:
        temperature_gradient (float): Temperature gradient in degC/meter
        water_salinity (float): Water salinity in ppm

    Returns:
        float: Formation water resistivity.
    """
    return (400000 / temperature_gradient / water_salinity)**.88


def estimate_rw_archie(phit, rt, a=1, m=2):
    """Estimate water saturation based on Archie's equation.

    Args:
        phit (float): Total porosity.
        rt (float): True resistivity.

    Returns:
        float: Formation water resistivity.
    """
    return phit**m * rt / a


def estimate_qv(vcld, phit, rho_clay=2.65, cec_clay=.062):
    """Estimate Qv, cation exchange capacity per unit total pore volume (meq/cm3).

    Args:
        vcld (float): Volume of dry clay in fraction
        phit (float): Total porosity in fraction
        rho_clay (float): Bulk density of clay in g/cc
        cec_clay (float): Cation exchange capacity of clay in meq/g

    Returns:
        float: Qv in meq/cm3.
    """
    return vcld * rho_clay * cec_clay / phit


def estimate_qv_hill(vclb, phit, water_salinity=10000):
    """Estimating Qv based on Hill et. al, 1979.

    Args:
        vclb (float or array): _description_
        phit (float or array): _description_
        water_salinity (float): Water salinity in ppm or meq/cc.

    Returns:
        float or array: _description_ in meq/cc
    """
    return (vclb / phit) / (0.084 * water_salinity**-0.5 + 0.22)


def estimate_qv_lavers(phit, a=3.05e-4, b=3.49):
    """Based on Lavers, 1975.

    Args:
        phit (float or array): Total porosity
        a (float): _description_
        b (float): _description_

    Returns:
        float or array: _description_
    """
    return a * phit**-b


def estimate_m_archie(rt, rw, phit):
    """Estimate apparent m (saturation factor) based on Archie's equation in water bearing zone.

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.

    Returns:
        float: Apparent m parameter.
    """
    return np.log(rw / rt) / np.log(phit)


def estimate_m_indonesian(rt, rw, phie, vsh, rsh):
    """Estimate apparent m based on Indonesian model in water bearing zone (shaly ~ vshale < 25%).

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phie (float): Effective porosity.
        vsh (float): Volume of shale.
        rsh (float): Resistivity of shale

    Returns:
        float: Apparent m parameter.
    """
    m = (2 / np.log(phie)) * np.log(rw**0.5 * ((1 / rt)**0.5 - (vsh**(1 - 0.5 * vsh) / rsh**0.5)))
    return m
