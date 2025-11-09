import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

from quick_pp.utils import min_max_line, remove_outliers
from quick_pp import logger

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 8,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


def archie_saturation(rt, rw, phit, a=1, m=2, n=2):
    """Estimate water saturation based on Archie's model for clean sand.

    Args:
        rt (np.ndarray or float): True resistivity or deep resistivity log (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        a (float, optional): Tortuosity factor. Defaults to 1.
        m (float, optional): Cementation exponent. Defaults to 2.
        n (float, optional): Saturation exponent. Defaults to 2.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Archie, G.E. (1942). The Electrical Resistivity Log as an Aid in
        Determining Some Reservoir Characteristics. Transactions of the AIME,
        146(1), 54-62.

    """
    logger.debug(f"Calculating Archie saturation with a={a}, m={m}, n={n}")
    sw = ((a / (phit ** m)) * (rw / rt)) ** (1 / n)
    logger.debug(f"Archie saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def waxman_smits_saturation(rt, rw, phit, Qv=None, B=None, m=2, n=2):
    """Estimate water saturation using the Waxman-Smits model for shaly sands.

    This function iteratively solves the Waxman-Smits equation, which accounts
    for the conductivity of clay minerals in the formation.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        Qv (np.ndarray or float, optional): Cation exchange capacity per unit pore volume (meq/cm³).
                                            Defaults to 0.3 if not provided.
        B (np.ndarray or float, optional): Equivalent conductance of clay exchange cations (S·m²/meq).
                                           If not provided, it is estimated based on Rw at 25°C.
        m (float, optional): Cementation exponent. Defaults to 2.
        n (float, optional): Saturation exponent. Defaults to 2.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Waxman, M.H. and Smits, L.J.M. (1968). Electrical Conductivities in Oil-Bearing Shaly Sands.
        Society of Petroleum Engineers Journal, 8(2), 107-122.

        Ausburn, B.E. and Freedman, R. (1985). The Waxman-Smits Equation For Shaly Sands:
        I. Simple Methods Of Solution, II. Error Analysis. The Log Analyst, 26.
    """
    logger.debug("Calculating Waxman-Smits saturation")
    # Estimate B at 25 degC if not provided
    if B is None:
        B = (1 - 0.83 * np.exp(-0.5 / rw)) * 3.83
        logger.debug(f"Estimated average B parameter: {B.mean():.3f}")

    if Qv is None:
        Qv = 0.3
        logger.debug(f"Using default Qv: {Qv}")

    # Initial guess
    swt = 1
    swt_i = 0
    logger.debug("Starting iterative solution for Waxman-Smits")

    # Use tqdm for progress bar during iterations
    for i in tqdm(range(50), desc="Waxman-Smits iteration"):
        fx = swt**n + rw * B * Qv * swt**(n - 1) - (phit**-m * rw / rt)  # Ausburn, 1985
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

    logger.debug(f"Waxman-Smits saturation range: {min(swt)} - {max(swt)}")
    # Post process only valid data
    invalid_mask = ~((rt > 0) & np.isfinite(rt) & (phit > 0) & np.isfinite(phit))
    swt[invalid_mask] = 1.0
    return swt


def normalized_waxman_smits_saturation(rt, rw, phit, vshale, phit_shale, rt_shale, m=2, n=2):
    """Estimate water saturation using the normalized Waxman-Smits model (Juhasz, 1981).

    This is a variation of the Waxman-Smits model that uses normalized Qv (Qvn)
    and derives the B parameter from shale properties.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        vshale (np.ndarray or float): Volume of shale (fraction).
        phit_shale (np.ndarray or float): Porosity of the shale (fraction).
        rt_shale (np.ndarray or float): Resistivity of the shale (ohm.m).
        m (float, optional): Cementation exponent. Defaults to 2.
        n (float, optional): Saturation exponent. Defaults to 2.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Juhasz, I. (1981). Normalized Qv—The Key to Shaly Sand Evaluation Using the Waxman-Smits Equation
        in the Absence of Core Data. SPWLA 22nd Annual Logging Symposium.
    """
    logger.debug("Calculating Normalized Waxman-Smits saturation")
    Qvn = estimate_qvn(vclay=vshale, phit=phit, phit_clay=phit_shale)
    B = (phit_shale**-m / rt_shale) - (1 / rw)

    # Initial guess
    swt = 1
    swt_i = 0
    logger.debug("Starting iterative solution for Waxman-Smits")

    # Use tqdm for progress bar during iterations
    for i in tqdm(range(50), desc="Normalized Waxman-Smits iteration"):
        fx = swt**n / rw + Qvn * B * swt**(n - 1) - (phit**-m / rt)  # Juhasz, 1981
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

    logger.debug(f"Normalized Waxman-Smits saturation range: {min(swt)} - {max(swt)}")
    # Post process only valid data
    invalid_mask = ~((rt > 0) & np.isfinite(rt) & (phit > 0) & np.isfinite(phit))
    swt[invalid_mask] = 1.0
    return swt


def dual_water_saturation(rt, rw, phit, a, m, n, swb, rwb):
    """Estimate water saturation using the Dual Water model.

    The Dual Water model is an extension of the Waxman-Smits model that considers
    two types of water in the pore space: bound water and free water.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        rw (np.ndarray or float): Formation (free) water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        a (float): Tortuosity factor.
        m (float): Cementation exponent.
        n (float): Saturation exponent.
        swb (np.ndarray or float): Bound water saturation (fraction of total porosity).
        rwb (np.ndarray or float): Bound water resistivity (ohm.m).

    Returns:
        np.ndarray or float: Total water saturation (fraction).

    References:
        Clavier, C., Coates, G., and Dumanoir, J. (1984). The Theoretical and Experimental
        Bases for the "Dual Water" Model for the Interpretation of Shaly Sands.
        Society of Petroleum Engineers Journal, 24(2), 153-168.
    """
    logger.debug("Calculating dual water saturation")
    # Initial guess
    swt = 1
    swt_i = 0
    logger.debug("Starting iterative solution for dual water model")

    # Use tqdm for progress bar during iterations
    for i in tqdm(range(50), desc="Dual water iteration"):
        fx = phit**m * swt**n / a * (1 / rw * (swb / swt) * (1 / rwb - 1 / rw)) - 1 / rt
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

    logger.debug(f"Dual water saturation range: {swt.min():.3f} - {swt.max():.3f}")
    # Post process only valid data
    invalid_mask = ~((rt > 0) & np.isfinite(rt) & (phit > 0) & np.isfinite(phit))
    swt[invalid_mask] = 1.0
    return swt


def indonesian_saturation(rt, rw, phie, vsh, rsh, a, m, n):
    """Estimate water saturation using the Indonesian model (Poupon-Leveaux, 1971).

    This model is often used for shaly sands, particularly in formations with
    fresh water.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phie (np.ndarray or float): Effective porosity (fraction).
        vsh (np.ndarray or float): Volume of shale (fraction).
        rsh (np.ndarray or float): Resistivity of shale (ohm.m).
        a (float): Tortuosity factor.
        m (float): Cementation exponent.
        n (float): Saturation exponent.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Poupon, A., and Leveaux, J. (1971). Evaluation of Water Saturation in Shaly Formations.
        The Log Analyst, 12(4).
    """
    logger.debug("Calculating Indonesian saturation")
    sw = ((1 / rt)**(1 / 2) / ((vsh**(1 - 0.5 * vsh) / rsh**(1 / 2)) + (phie**m / (a * rw))**(1 / 2)))**(2 / n)
    logger.debug(f"Indonesian saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def simandoux_saturation(rt, rw, phit, vsh, rsh, a, m):
    """Estimate water saturation using the Simandoux model (1963).

    This is a classic shaly sand model that adds a shale conductivity term
    to the clean sand Archie equation.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        vsh (np.ndarray or float): Volume of shale (fraction).
        rsh (np.ndarray or float): Resistivity of shale (ohm.m).
        a (float): Tortuosity factor.
        m (float): Cementation exponent.

    Returns:
        np.ndarray or float: Water saturation (fraction).
    """
    logger.debug("Calculating Simandoux saturation")
    shale_factor = vsh / rsh
    sw = (a * rw / (2 * phit**m)) * ((shale_factor**2 + (4 * phit**m / (a * rw * rt)))**(1 / 2) - shale_factor)
    logger.debug(f"Simandoux saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def connectivity_saturation(rt, rw, phit, mu=2, chi_w=0):
    """Estimate water saturation using the Connectivity Equation (Montaron, 2009).

    This model relates water saturation to porosity and a water connectivity
    index, offering an alternative to traditional shaly sand models.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        mu (float, optional): Conductivity exponent, typically between 1.6 and 2.0. Defaults to 2.
        chi_w (float, optional): Water connectivity correction index, typically between -0.02 and 0.02.
                                 Defaults to 0.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Montaron, B. (2009). The Connectivity Equation: A New Formation Evaluation Concept.
        SPWLA 50th Annual Logging Symposium.
    """
    logger.debug("Calculating connectivity saturation")

    rw_prime = rw * (1 + chi_w) ** mu
    sw = ((rw_prime / rt) ** (1 / mu) + chi_w) / phit
    logger.debug(f"Connectivity saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def estimate_swb(phit, vsh, nphi_sh):
    """Estimate bound water saturation based on dual water model.
    Args:
        phit (float): Total porosity in fraction.
        vsh (float): Volume of shale in fraction.
        nphi_sh (float): Neutron porosity reading in a nearby 100% shale interval in fraction.

    Returns:
        float: Bound water saturation.
    """
    swb = vsh * nphi_sh / phit
    return swb


def estimate_chi_w(s_cw, phit, sigma_cw, sigma_w, mu=2):
    """Estimate water connectivity correction index based on connectivity model Montaron 2009
    Args:
        s_cw (float): Water saturation, ranges from 0 to 1.
        phit (float): Total porosity.
        sigma_cw (float): Conductivity of clay water, ranges from 0 to 1.
        sigma_w (float): Water conductivity, ranges from 0 to 1.
        mu (float): Conductivity exponent, ranges from 1.6 to 2. Defaults to 2.

    Returns:
        float: Water connectivity correction index.

    """
    logger.debug("Calculating water connectivity correction index")
    chi_w = -s_cw * phit * ((sigma_cw / sigma_w)**(1 / mu) - 1)
    logger.debug(f"Water connectivity correction index range: {chi_w.min():.3f} - {chi_w.max():.3f}")
    return chi_w


def estimate_temperature_gradient(tvd, unit='metric'):
    """Estimate formation temperature based on gradient of 25 degC/km or 15 degF/1000ft.

    Args:
        tvd (float): True vertical depth in m.
        unit (str, optional): The unit system for depth and gradient, either 'metric' or 'imperial'.
                              Defaults to 'metric'.

    Returns:
        np.ndarray or float: Formation temperature in degrees Celsius.
    """
    logger.debug(f"Estimating temperature gradient with unit: {unit}")
    assert unit in ['metric', 'imperial'], "Please choose from 'metric' or 'imperial' units."
    temp = 32 + 25 * tvd / 1000 if unit == 'metric' else 90 + 15 * tvd / 1000
    logger.debug(f"Temperature range: {temp.min():.1f} - {temp.max():.1f} °C")
    return temp


def estimate_b_waxman_smits(T, rw):
    """Estimate the B parameter (equivalent conductance) for the Waxman-Smits model.

    This function uses the empirical relationship proposed by Juhasz (1981) which
    relates B to formation temperature and water resistivity.

    Args:
        T (np.ndarray or float): Formation temperature in degrees Celsius.
        rw (np.ndarray or float): Formation water resistivity (ohm.m).

    Returns:
        np.ndarray or float: The B parameter for the Waxman-Smits equation.

    References:
        Juhasz, I. (1981). Normalized Qv—The Key to Shaly Sand Evaluation Using the Waxman-Smits Equation
        in the Absence of Core Data. SPWLA 22nd Annual Logging Symposium.
    """
    logger.debug("Estimating B parameter for Waxman-Smits (Juhasz 1981)")
    B = (-1.28 + 0.225 * T - 0.0004059 * T**2) / (1 + (0.045 * T - 0.27) * rw**1.23)
    logger.debug(f"B parameter range: {B.min():.3f} - {B.max():.3f}")
    return B


def estimate_rw_temperature_salinity(temperature_gradient, water_salinity):
    """Estimate formation water resistivity (Rw) from temperature and salinity.

    This function uses a common empirical formula to approximate Rw.

    Args:
        temperature_gradient (np.ndarray or float): Formation temperature in degrees Celsius.
        water_salinity (np.ndarray or float): Water salinity in parts per million (ppm).

    Returns:
        np.ndarray or float: Estimated formation water resistivity (ohm.m).
    """
    logger.debug("Estimating Rw from temperature and salinity")
    rw = (400000 / water_salinity)**.88 / temperature_gradient
    logger.debug(f"Formation water resistivity range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_surface(temperature_gradient, rw_surface, temp_surface=20):
    """Estimate downhole formation water resistivity (Rw) from surface measurements.

    This uses Arps' formula to adjust resistivity for temperature.

    Args:
        temperature_gradient (np.ndarray or float): Formation temperature in degrees Celsius.
        rw_surface (float): Water resistivity measured at surface temperature (ohm.m).
        temp_surface (float, optional): Surface temperature in degrees Celsius. Defaults to 20.

    Returns:
        np.ndarray or float: Estimated formation water resistivity at formation temperature (ohm.m).
    """
    logger.debug("Estimating Rw from surface temperature and resistivity using Arps' formula")
    rw = rw_surface * (temp_surface + 21.5) / (temperature_gradient + 21.5)
    logger.debug(f"Formation water resistivity range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_archie(phit, rt, a=1, m=2):
    """Estimate formation water resistivity (Rw) from a water-bearing interval using Archie's equation.

    This method assumes the interval is 100% water-saturated (Sw=1) and calculates
    Rw. It then fits a minimum trend line to the calculated values to find a
    representative Rw.

    Args:
        phit (np.ndarray or float): Total porosity (fraction).
        rt (np.ndarray or float): True resistivity (ohm.m).
        a (float, optional): Tortuosity factor. Defaults to 1.
        m (float, optional): Cementation exponent. Defaults to 2.

    Returns:
        np.ndarray or float: Estimated formation water resistivity (ohm.m).
    """
    logger.debug("Estimating Rw using Archie's equation")
    rw = pd.Series(phit**m * rt / a)
    _, rw = min_max_line(rw, alpha=0.2)
    logger.debug(f"Estimated Rw range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_waxman_smits(phit, rt, a=1, m=2, B=None, Qv=None):
    """Estimate Rw from a water-bearing interval using the Waxman-Smits equation.

    This method assumes the interval is 100% water-saturated (Sw=1) and calculates
    Rw. It then fits a minimum trend line to the calculated values.

    Args:
        phit (np.ndarray or float): Total porosity (fraction).
        rt (np.ndarray or float): True resistivity (ohm.m).
        a (float, optional): Tortuosity factor. Defaults to 1.
        m (float, optional): Cementation exponent. Defaults to 2.
        B (float, optional): Equivalent conductance of clay exchange cations. Defaults to 2.
        Qv (float, optional): Cation exchange capacity. Defaults to 0.3.

    Returns:
        np.ndarray or float: Estimated formation water resistivity (ohm.m).
    """
    logger.debug("Estimating Rw using Waxman-Smits equation")
    if B is None:
        B = 2
    if Qv is None:
        Qv = 0.3

    rw = pd.Series(1 / ((a / (phit**m * rt)) - (B * Qv)))
    _, rw = min_max_line(rw, alpha=0.2)
    logger.debug(f"Estimated Rw range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_from_shale_trend(rt, phit, vshale, depth, m=1.3):
    """Estimate Rw by extrapolating the resistivity trend from shales into sands.

    This method identifies a resistivity trend with depth in nearby shales and
    assumes this trend represents the Ro (resistivity of 100% water-saturated rock)
    baseline. Rw is then calculated from this Ro trend and formation porosity.

    Args:
        rt (np.ndarray or float): True resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        vshale (np.ndarray or float): Volume of shale (fraction).
        depth (np.ndarray or float): Depth log.
        m (float, optional): Shale cementation or shape factor. Defaults to 1.3.

    Returns:
        np.ndarray or float: Estimated formation water resistivity (ohm.m).
    """
    logger.debug(f"Estimating Rw from shale trend with m={m}")

    # Create a mask for valid, finite data points
    valid_data_mask = np.isfinite(rt) & np.isfinite(phit) & np.isfinite(vshale) & np.isfinite(depth) & (depth > 0)

    # Identify shale intervals
    vsh_threshold = np.nanquantile(vshale, .5)
    shale_mask = (vshale >= vsh_threshold) & valid_data_mask
    rt_shale = rt[shale_mask]
    depth_shale = depth[shale_mask]

    params = np.polyfit(np.log(depth_shale), np.log(rt_shale), 1)
    min_rt = params[0] * np.log(depth) + params[1]

    rw = phit ** m * np.exp(min_rt)
    logger.debug(f"Shale trend Rw range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rt_shale(rt, vshale):
    """Estimate shale resistivity (Rsh) from log data.

    This function identifies shale intervals based on a Vshale cutoff (50th percentile),
    applies a rolling average to the resistivity in those intervals, and then
    propagates this value across the entire log interval.

    Args:
        rt (pd.Series or np.ndarray): True resistivity log (ohm.m).
        vshale (pd.Series or np.ndarray): Volume of shale log (fraction).

    Returns:
        pd.Series: Estimated shale resistivity log (ohm.m).
    """
    # Identify shale intervals
    vshale = remove_outliers(vshale)

    vsh_threshold = np.nanquantile(vshale, 0.5)
    shale_mask = vshale >= vsh_threshold

    # Create a series with phit values only at shale intervals
    rt_shale = rt.where(shale_mask).rolling(31, center=True, min_periods=1).mean()

    # Forward-fill and then back-fill to propagate shale porosity
    rt_shale = rt_shale.ffill().bfill()
    logger.debug(f"Shale porosity range: {rt_shale.min():.3f} - {rt_shale.max():.3f}")
    return rt_shale


def estimate_qv(vcld, phit, rho_clay=2.65, cec_clay=.062):
    """Estimate Qv (cation exchange capacity per unit pore volume) from clay properties.

    Args:
        vcld (np.ndarray or float): Volume of dry clay (fraction of bulk volume).
        phit (np.ndarray or float): Total porosity (fraction).
        rho_clay (float, optional): Bulk density of dry clay (g/cm³). Defaults to 2.65.
        cec_clay (float, optional): Cation exchange capacity of clay (meq/g). Defaults to 0.062.

    Returns:
        np.ndarray or float: Qv in meq/cm³.
    """
    logger.debug(f"Estimating Qv with clay density={rho_clay} g/cm³, CEC={cec_clay} meq/g")
    qv = vcld * rho_clay * cec_clay / phit
    logger.debug(f"Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cm³")
    return qv


def estimate_qvn(vclay, phit, phit_clay):
    """Estimate normalized Qv (Qvn) for the Juhasz (1981) model.

    Args:
        vclay (np.ndarray or float): Volume of clay (fraction).
        phit (np.ndarray or float): Total porosity (fraction).
        phit_clay (np.ndarray or float): Total porosity of the clay (fraction).

    Returns:
        np.ndarray or float: Normalized Qv (Qvn).
    """
    return vclay * phit_clay / phit


def estimate_qv_ward(rt, phit, B, rw, m):
    """Estimate Qv from water-bearing zone logs using the Ward (1973) method.

    This method inverts the Waxman-Smits equation, assuming Sw=1, to solve for Qv.

    Args:
        rt (np.ndarray or float): True resistivity in a water-bearing zone (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        B (float): Equivalent conductance of clay exchange cations (S·m²/meq).
        rw (float): Formation water resistivity (ohm.m).
        m (float): Cementation exponent.

    Returns:
        np.ndarray or float: Qv in meq/cm³.

    References:
        Ward, B. (1973). Internal SIPM Report.
    """
    return 1 / B * ((1 / (rt * phit**m)) - (1 / rw))


def estimate_qv_hill(vclb, phit, water_salinity=10000):
    """Estimate Qv using the empirical relationship from Hill, Shirley, and Klein (1979).

    This method relates Qv to the bulk volume of clay-bound water and water salinity.

    Args:
        vclb (np.ndarray or float): Volume of clay bound water (fraction of bulk volume).
        phit (np.ndarray or float): Total porosity (fraction).
        water_salinity (float, optional): Water salinity in ppm. Defaults to 10000.

    Returns:
        np.ndarray or float: Qv in meq/cm³.

    References:
        Hill, H.J., Shirley, O.J., and Klein, G.E. (1979). Bound Water in Shaly Sands—Its
        Relation to Qv and Other Formation Properties. The Log Analyst, 20(3).
    """
    logger.debug(f"Estimating Qv using Hill method with water salinity={water_salinity}")
    qv = (vclb / phit) / (0.084 * water_salinity**-0.5 + 0.22)
    logger.debug(f"Hill Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cc")
    return qv


def estimate_qv_lavers(phit, a=3.05e-4, b=3.49):
    """Estimate Qv from porosity using the Lavers (1975) empirical equation.

    This method provides a quick estimate of Qv based on a power-law relationship
    with total porosity.

    Args:
        phit (np.ndarray or float): Total porosity (fraction).
        a (float, optional): Empirical constant. Defaults to 3.05e-4.
        b (float, optional): Empirical exponent. Defaults to 3.49.

    Returns:
        np.ndarray or float: Qv in meq/cm³.

    References:
        Lavers, B.A., Smits, L.J.M., and van Baaren, C. (1975). Some fundamental problems of
        formation evaluation in the North Sea. The Log Analyst, 16(3).
    """
    logger.debug(f"Estimating Qv using Lavers method with a={a}, b={b}")
    qv = a * phit**-b
    logger.debug(f"Lavers Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cc")
    return qv


def estimate_bqv(phit, max_phit_clean_sand, C):
    """Estimate BQv (Bulk Volume of clay-bound water) using the Juhasz/Rackley method.

    This method relates the reduction in porosity from a clean sand trend to the
    amount of clay-bound water.

    Args:
        phit (np.ndarray or float): Total porosity (fraction).
        max_phit_clean_sand (float): Maximum porosity of the clean sand trend.
        C (float): A constant that depends on the clay type.

    Returns:
        np.ndarray or float: Bulk volume of clay-bound water (fraction of bulk volume).
    """
    return (max_phit_clean_sand - phit) / (C * phit)


def estimate_m_archie(rt, rw, phit):
    """Estimate the apparent cementation exponent (m) using Archie's equation.

    This function inverts Archie's equation in a water-bearing zone (assuming Sw=1)
    to solve for 'm'.

    Args:
        rt (np.ndarray or float): True resistivity in a water-bearing zone (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).

    Returns:
        np.ndarray or float: Apparent cementation exponent (m).
    """
    logger.debug("Estimating apparent m using Archie's equation")
    m = np.log(rw / rt) / np.log(phit)
    logger.debug(f"Apparent m range: {m.min():.3f} - {m.max():.3f}")
    return m


def estimate_m_indonesian(rt, rw, phie, vsh, rsh):
    """Estimate the apparent cementation exponent (m) using the Indonesian model.

    This function inverts the Indonesian saturation equation in a water-bearing
    zone (assuming Sw=1) to solve for 'm'. It is typically used in shaly intervals.

    Args:
        rt (np.ndarray or float): True resistivity in a water-bearing zone (ohm.m).
        rw (np.ndarray or float): Formation water resistivity (ohm.m).
        phie (np.ndarray or float): Effective porosity (fraction).
        vsh (np.ndarray or float): Volume of shale (fraction).
        rsh (np.ndarray or float): Resistivity of shale (ohm.m).

    Returns:
        np.ndarray or float: Apparent cementation exponent (m).
    """
    logger.debug("Estimating apparent m using Indonesian model")
    m = (2 / np.log(phie)) * np.log(rw**0.5 * ((1 / rt)**0.5 - (vsh**(1 - 0.5 * vsh) / rsh**0.5)))
    logger.debug(f"Indonesian apparent m range: {m.min():.3f} - {m.max():.3f}")
    return m


def qv_phit_xplot(phit, qv):
    """Generate a Qv vs 1/PHIT crossplot.

    This plot is used to analyze the relationship between cation exchange capacity (Qv)
    and porosity, which can help in understanding clay distribution. A best-fit
    straight line is overlaid on the data.

    Args:
        phit (np.ndarray or float): Total porosity (fraction).
        qv (np.ndarray or float): Cation exchange capacity per unit pore volume (meq/cm³).
    """
    fig, ax = plt.subplots()
    ax.set_title("Qv vs 1/PHIT")

    # Prepare data for plotting and fitting
    x = 1 / phit
    y = qv

    # Remove non-finite values for a clean fit
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], marker='.')

    # Fit a straight line using np.polyfit
    m, c = np.polyfit(x[mask], y[mask], 1)
    ax.plot(x[mask], m * x[mask] + c, color='r', linestyle='--', label=f'y = {m:.2f}x + {c:.2f}')

    ax.set_xlabel('1 / PHIT (frac)')
    ax.set_ylabel('Qv (meq/cc)')
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')


def cwa_qvn_xplot(rt, phit, qvn, m=2.0, rw=.2, slope=250):
    """Generate a Cwa vs Qvn plot for shaly sand analysis.

    This plot is used to graphically solve for water saturation in the
    normalized Waxman-Smits (Juhasz) model.

    Args:
        rt (np.ndarray or float): True resistivity of the formation (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        qvn (np.ndarray or float): Normalized Qv.
        m (float, optional): Cementation exponent. Defaults to 2.0.
        rw (float, optional): Formation water resistivity (ohm.m). Used for the trend line. Defaults to 0.2.
        slope (float, optional): Slope of the interpretation line. Defaults to 250.
    """
    x = qvn
    y = 1 / (rt * phit**m)
    if len(x) != len(y):
        logger.warning(f"Length mismatch between Qvn ({len(x)}) and Cwa ({len(y)}). Plot may be incorrect.")

    fig, ax = plt.subplots()
    ax.set_title("Cwa vs Qvn")
    ax.scatter(x, y, marker='.')

    # Add straight line
    x_line = np.linspace(0, 1.0, len(x))
    y_line = 1 / rw + x_line * slope
    ax.plot(x_line, y_line, 'r--', label=f'm= {m} \nrw= {rw} \nslope= {slope}')

    ax.set_xlabel('Qvn (meq/cc)')
    ax.set_ylabel('Cwa (1 / ohm.m)')
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 100)
    ax.minorticks_on()
    ax.grid(True)
    fig.tight_layout()
    ax.legend()
    return fig


def swirr_xplot(swt, phit, c=.0125, label='', log_log=False, title=''):
    """Generate a Buckles plot (SWT vs. PHIT) to estimate irreducible water saturation.

    This crossplot helps identify the hyperbolic trend of irreducible water
    saturation, where PHIT * Swirr = constant (C).

    Args:
        swt (np.ndarray or float): Water saturation (fraction).
        phit (np.ndarray or float): Total porosity (fraction).
        c (float, optional): The Buckles constant (PHIT * Swirr). Used to plot an iso-line. Defaults to 0.0125.
        label (str, optional): Label for the scattered data points. Defaults to ''.
        log_log (bool, optional): If True, both axes will be logarithmic. Defaults to False.
        title (str, optional): Title for the plot. Defaults to ''.
    """
    logger.debug(f"Creating Swirr crossplot with constant c={c}")
    fig, ax = plt.subplots()
    ax.set_title(title)
    sc = ax.scatter(swt, phit, marker='.', label=label)
    if c:
        line_color = sc.get_facecolors()[0]
        line_color[-1] = 0.75
        cphit = np.geomspace(0.0001, 0.4, 30)
        ax.plot(c / cphit, cphit, label=rf'$\phi$ S = {c}',
                color=line_color, linestyle='dashed')
    ax.set_ylabel('PHIT (frac)')
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel('SWT (frac)')
    ax.set_xlim(1e-3, 1)
    ax.legend()
    if log_log:
        ax.set_xscale('log')
        ax.set_yscale('log')
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')
    fig.tight_layout()
    logger.debug("Swirr crossplot created")


def rt_phit_xplot(rt, phit, m=2, rw=.01):
    """Generate an Rt vs. PHIT (Hingle) plot.

    This plot is useful for visualizing the relationship between true resistivity
    and total porosity, often used for identifying water-bearing zones and estimating
    formation water resistivity (Rw) and cementation exponent (m).

    Args:
        rt (np.ndarray or float): True resistivity or deep resistivity log (ohm.m).
        phit (np.ndarray or float): Total porosity (fraction).
        m (float, optional): Cementation exponent for the iso-line. Defaults to 2.
        rw (float, optional): Formation water resistivity for the iso-line (ohm.m). Defaults to 0.01.

    Returns:
        matplotlib.figure.Figure: The RT vs PHIT plot.
    """
    fig, ax = plt.subplots()
    ax.set_title("RT vs PHIT")
    ax.scatter(phit, rt, marker='.')

    # Add iso-lines
    phit_i = np.arange(0, 1, 1 / len(phit))
    rt_i = phit_i**-m * rw
    ax.plot(phit_i, rt_i, 'r--', alpha=0.5, label=f'm= {m} \nrw= {rw}')

    ax.set_ylabel('RT (ohm.m)')
    ax.set_xlabel('PHIT (frac)')
    ax.loglog()
    ax.set_xlim(1e-2, 1)
    ax.set_ylim(1e-2, 1e2)

    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
    ax.legend()
    fig.tight_layout()
    return fig


def pickett_plot(rt, phit, m=-2, min_rw=0.1, shift=.2, title='Pickett Plot'):
    """Generate a Pickett plot (log-log of Rt vs. PHIT).

    This plot is a graphical tool to determine the cementation exponent (m),
    formation water resistivity (Rw), and water saturation (Sw) from log data.

    Args:
        rt (np.ndarray or float): True resistivity log (ohm.m).
        phit (np.ndarray or float): Total porosity log (fraction).
        m (float, optional): The slope of the water-bearing line (should be negative). Defaults to -2.
        min_rw (float, optional): The Rw value for the 100% water saturation line. Defaults to 0.1.
        shift (float, optional): Increment to generate other iso-Sw lines. Defaults to 0.2.
        title (str, optional): Title for the plot. Defaults to 'Pickett Plot'.

    Returns:
        matplotlib.pyplot.Figure: The Pickett plot figure.
    """
    logger.debug(f"Creating Pickett plot with m={m}, min_rw={min_rw}")
    m = m if m < 0 else -m
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)
    ax.scatter(rt, phit, marker='.', color='b')
    # Add iso-lines
    phit_i = np.arange(0, 1, 1 / len(phit))
    for i in np.geomspace(1, 5, num=5):
        c = min_rw + (i - 1) * shift
        sw = round(min_rw / c * 100)
        rt_i = (phit_i**m) * c
        ax.plot(rt_i, phit_i, linestyle='dashed', alpha=0.5, label=f'SW={sw}%')
    # Set up y axis
    ax.set_yscale('log')
    ax.set_ylim(0.01, 1)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
    ax.set_ylabel('PHIT (v/v)')
    # Set up x axis
    ax.set_xscale('log')
    ax.set_xlim(0.01, 1000)
    ax.tick_params(top=True, labeltop=True)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: ('{{:.{:1d}f}}'.format(int(np.maximum(-np.log10(x), 0)))).format(x)))
    ax.set_xlabel('RT (ohm.m)')

    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.3', color='gray')
    fig.tight_layout()
    logger.debug("Pickett plot created")


def RI_plot(sw, rt, ro):
    """Generate a Resistivity Index (RI) vs. Sw plot from lab data.

    This log-log plot is used to determine the saturation exponent (n). The
    slope of the best-fit line through the data is -n. The trend line should
    pass through the (1, 1) point.

    Args:
        sw (np.ndarray or float): Water saturation from core analysis (fraction).
        rt (np.ndarray or float): True resistivity of the core sample at a given Sw (ohm.m).
        ro (np.ndarray or float): Resistivity of the same core sample when 100% water-saturated (ohm.m).

    Returns:
        matplotlib.pyplot.Figure: The Resistivity Index plot figure.
    """
    logger.debug("Creating resistivity index plot")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Resistivity Index Plot')
    ax.scatter(sw, rt / ro)
    ax.set_xlim(0.01, 1)
    ax.set_ylim(1, 1000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('SW')
    ax.set_ylabel('RT/Rw')
    ax.grid(True)
    fig.tight_layout()
    logger.debug("Resistivity index plot created")
    return fig


def FF_plot(phit, ro, rw):
    """Generate a Formation Factor (FF) vs. Porosity plot from lab data.

    This log-log plot is used to determine the tortuosity factor (a) and the
    cementation exponent (m). The slope of the best-fit line is -m, and the
    intercept at PHIT=1 is 'a'.

    Args:
        phit (np.ndarray or float): Porosity from core analysis (fraction).
        ro (np.ndarray or float): Resistivity of the 100% water-saturated core sample (ohm.m).
        rw (np.ndarray or float): Resistivity of the saturating water (ohm.m).

    Returns:
        matplotlib.pyplot.Figure: The Formation Factor plot figure.
    """
    logger.debug("Creating formation factor plot")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Formation Factor Plot')
    ax.scatter(phit, ro / rw)
    ax.set_xlim(0.01, 1)
    ax.set_ylim(1, 1000)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('PHIT')
    ax.set_ylabel('Ro/Rw')
    ax.grid(True)
    fig.tight_layout()
    logger.debug("Formation factor plot created")
    return fig
