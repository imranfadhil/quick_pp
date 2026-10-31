"""Water saturation, irreducible water saturation (Swirr), and saturation height models.

Provides resistivity-based saturation equations (Archie, Waxman-Smits, Dual Water,
Indonesian, Simandoux, Connectivity), Rw / Qv / m estimation utilities, Swirr
correlations (Timur, RQI, semi-log permeability), and capillary pressure / height
above free water level (SHF) calculations.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from tqdm import tqdm

from quick_pp import logger
from quick_pp.utils import min_max_line, remove_outliers

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {"axes.labelsize": 8, "xtick.labelsize": 10, "legend.fontsize": "small"}
)


def archie_saturation(rt, rw, phit, a=1, m=2, n=2):
    """Estimate water saturation based on Archie's model for clean sand.

    Args:
        rt (np.ndarray or float): True resistivity or deep resistivity log [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        a (float, optional): Tortuosity factor. Defaults to 1.0.
        m (float, optional): Cementation exponent. Defaults to 2.0.
        n (float, optional): Saturation exponent. Defaults to 2.0.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Archie, G.E. (1942). The Electrical Resistivity Log as an Aid in
        Determining Some Reservoir Characteristics. Transactions of the AIME,
        146(1), 54-62.

    """
    logger.debug(f"Calculating Archie saturation with a={a}, m={m}, n={n}")
    sw = ((a / (phit**m)) * (rw / rt)) ** (1 / n)
    logger.debug(f"Archie saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def waxman_smits_saturation(rt, rw, phit, Qv=None, B=None, m=2, n=2):
    """Estimate water saturation using the Waxman-Smits model for shaly sands.

    This function iteratively solves the Waxman-Smits equation, which accounts
    for the conductivity of clay minerals in the formation.

    Args:
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        Qv (np.ndarray or float, optional): Cation exchange capacity per unit pore volume [meq/cm³].
                                            Defaults to 0.3 if not provided.
        B (np.ndarray or float, optional): Equivalent conductance of clay exchange cations [S·m²/meq].
                                           If not provided, it is estimated based on Rw at 25°C.
        m (float, optional): Cementation exponent. Defaults to 2.0.
        n (float, optional): Saturation exponent. Defaults to 2.0.

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
    for _ in tqdm(range(50), desc="Waxman-Smits iteration"):
        fx = (
            swt**n + rw * B * Qv * swt ** (n - 1) - (phit**-m * rw / rt)
        )  # Ausburn, 1985
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

    logger.debug(f"Waxman-Smits saturation range: {min(swt)} - {max(swt)}")
    # Post process only valid data
    invalid_mask = ~((rt > 0) & np.isfinite(rt) & (phit > 0) & np.isfinite(phit))
    swt[invalid_mask] = 1.0
    return swt


def normalized_waxman_smits_saturation(
    rt, rw, phit, vshale, phit_shale, rt_shale, m=2, n=2
):
    """Estimate water saturation using the normalized Waxman-Smits model (Juhasz, 1981).

    This is a variation of the Waxman-Smits model that uses normalized Qv (Qvn)
    and derives the B parameter from shale properties.

    Args:
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        vshale (np.ndarray or float): Volume of shale [fraction].
        phit_shale (np.ndarray or float): Porosity of the shale [fraction].
        rt_shale (np.ndarray or float): Resistivity of the shale [ohm.m].
        m (float, optional): Cementation exponent. Defaults to 2.0.
        n (float, optional): Saturation exponent. Defaults to 2.0.

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
    for _ in tqdm(range(50), desc="Normalized Waxman-Smits iteration"):
        fx = swt**n / rw + Qvn * B * swt ** (n - 1) - (phit**-m / rt)  # Juhasz, 1981
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
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        rw (np.ndarray or float): Formation (free) water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        a (float): Tortuosity factor.
        m (float): Cementation exponent.
        n (float): Saturation exponent.
        swb (np.ndarray or float): Bound water saturation [fraction of total porosity].
        rwb (np.ndarray or float): Bound water resistivity [ohm.m].

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
    for _ in tqdm(range(50), desc="Dual water iteration"):
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
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phie (np.ndarray or float): Effective porosity [fraction].
        vsh (np.ndarray or float): Volume of shale [fraction].
        rsh (np.ndarray or float): Resistivity of shale [ohm.m].
        a (float): Tortuosity factor.
        m (float): Cementation exponent. Defaults to 2.0.
        n (float): Saturation exponent.

    Returns:
        np.ndarray or float: Water saturation (fraction).

    References:
        Poupon, A., and Leveaux, J. (1971). Evaluation of Water Saturation in Shaly Formations.
        The Log Analyst, 12(4).
    """
    logger.debug("Calculating Indonesian saturation")
    sw = (
        (1 / rt) ** (1 / 2)
        / ((vsh ** (1 - 0.5 * vsh) / rsh ** (1 / 2)) + (phie**m / (a * rw)) ** (1 / 2))
    ) ** (2 / n)
    logger.debug(f"Indonesian saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def simandoux_saturation(rt, rw, phit, vsh, rsh, a, m):
    """Estimate water saturation using the Simandoux model (1963).

    This is a classic shaly sand model that adds a shale conductivity term
    to the clean sand Archie equation.

    Args:
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        vsh (np.ndarray or float): Volume of shale [fraction].
        rsh (np.ndarray or float): Resistivity of shale [ohm.m].
        a (float): Tortuosity factor.
        m (float): Cementation exponent.

    Returns:
        np.ndarray or float: Water saturation (fraction).
    """
    logger.debug("Calculating Simandoux saturation")
    shale_factor = vsh / rsh
    sw = (a * rw / (2 * phit**m)) * (
        (shale_factor**2 + (4 * phit**m / (a * rw * rt))) ** (1 / 2) - shale_factor
    )
    logger.debug(f"Simandoux saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def connectivity_saturation(rt, rw, phit, mu=2, chi_w=0):
    """Estimate water saturation using the Connectivity Equation (Montaron, 2009).

    This model relates water saturation to porosity and a water connectivity
    index, offering an alternative to traditional shaly sand models.

    Args:
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        mu (float, optional): Conductivity exponent, typically 1.6-2.0. Defaults to 2.0.
        chi_w (float, optional): Water connectivity correction index, typically -0.02 to 0.02.
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
        phit (np.ndarray or float): Total porosity in fraction.
        vsh (np.ndarray or float): Volume of shale in fraction.
        nphi_sh (np.ndarray or float): Neutron porosity reading in a nearby 100% shale interval in fraction.

    Returns:
        np.ndarray or float: Bound water saturation.
    """
    swb = vsh * nphi_sh / phit
    return swb


def estimate_chi_w(s_cw, phit, sigma_cw, sigma_w, mu=2):
    """Estimate water connectivity correction index based on connectivity model Montaron 2009
    Args:
        s_cw (np.ndarray or float): Water saturation, ranges from 0 to 1.
        phit (np.ndarray or float): Total porosity.
        sigma_cw (np.ndarray or float): Conductivity of clay water, ranges from 0 to 1.
        sigma_w (np.ndarray or float): Water conductivity, ranges from 0 to 1.
        mu (float, optional): Conductivity exponent, ranges from 1.6 to 2.0. Defaults to 2.0.

    Returns:
        np.ndarray or float: Water connectivity correction index.

    """
    logger.debug("Calculating water connectivity correction index")
    chi_w = -s_cw * phit * ((sigma_cw / sigma_w) ** (1 / mu) - 1)
    logger.debug(
        f"Water connectivity correction index range: {chi_w.min():.3f} - {chi_w.max():.3f}"
    )
    return chi_w


def estimate_temperature_gradient(tvd, unit="metric"):
    """Estimate formation temperature based on gradient of 25 degC/km or 15 degF/1000ft.

    Args:
        tvd (np.ndarray or float): True vertical depth in m.
        unit (str, optional): The unit system for depth and gradient, either 'metric' or 'imperial'.
                              Defaults to 'metric'.

    Returns:
        np.ndarray or float: Formation temperature in degrees Celsius.
    """
    logger.debug(f"Estimating temperature gradient with unit: {unit}")
    assert unit in ["metric", "imperial"], (
        "Please choose from 'metric' or 'imperial' units."
    )
    temp = 32 + 25 * tvd / 1000 if unit == "metric" else 90 + 15 * tvd / 1000
    logger.debug(f"Temperature range: {temp.min():.1f} - {temp.max():.1f} °C")
    return temp


def estimate_b_waxman_smits(T, rw):
    """Estimate the B parameter (equivalent conductance) for the Waxman-Smits model.

    This function uses the empirical relationship proposed by Juhasz (1981) which
    relates B to formation temperature and water resistivity.

    Args:
        T (np.ndarray or float): Formation temperature in degrees Celsius.
        rw (np.ndarray or float): Formation water resistivity [ohm.m].

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


def estimate_rw_temperature_salinity(temp_formation, water_salinity):
    """Estimate formation water resistivity (Rw) from temperature and salinity.
    Based on Crain’s Petrophysical Handbook, which attributes the algorithm to the Bateman and Konen (1977) method.

    This function uses a common empirical formula to approximate Rw.

    Args:
        temp_formation (np.ndarray or float): Formation temperature [degrees Celsius].
        water_salinity (np.ndarray or float): Water salinity [parts per million (ppm)].

    Returns:
        np.ndarray or float: Estimated formation water resistivity [ohm.m].
    """
    logger.debug("Estimating Rw from temperature and salinity")
    temp_formation = temp_formation * 9 / 5 + 32
    rw = (400000 / (water_salinity * temp_formation)) ** 0.88
    logger.debug(
        f"Formation water resistivity range: {rw.min():.3f} - {rw.max():.3f} ohm.m"
    )
    return rw


def estimate_rw_surface(temp_formation, rw_surface, temp_surface=20):
    """Estimate downhole formation water resistivity (Rw) from surface measurements.

    This uses Arps' formula to adjust resistivity for temperature.

    Args:
        temp_formation (np.ndarray or float): Formation temperature [degrees Celsius].
        rw_surface (float): Water resistivity measured at surface temperature [ohm.m].
        temp_surface (float, optional): Surface temperature [degrees Celsius]. Defaults to 20.

    Returns:
        np.ndarray or float: Estimated formation water resistivity at formation temperature [ohm.m].
    """
    logger.debug(
        "Estimating Rw from surface temperature and resistivity using Arps' formula"
    )
    rw = rw_surface * (temp_surface + 21.5) / (temp_formation + 21.5)
    logger.debug(
        f"Formation water resistivity range: {rw.min():.3f} - {rw.max():.3f} ohm.m"
    )
    return rw


def estimate_rw_archie(phit, rt, a=1, m=2):
    """Estimate formation water resistivity (Rw) from a water-bearing interval using Archie's equation.

    This method assumes the interval is 100% water-saturated (Sw=1) and calculates
    Rw. It then fits a minimum trend line to the calculated values to find a
    representative Rw.

    Args:
        phit (np.ndarray or float): Total porosity [fraction].
        rt (np.ndarray or float): True resistivity [ohm.m].
        a (float, optional): Tortuosity factor. Defaults to 1.0.
        m (float, optional): Cementation exponent. Defaults to 2.0.

    Returns:
        np.ndarray or float: Estimated formation water resistivity [ohm.m].
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
        phit (np.ndarray or float): Total porosity [fraction].
        rt (np.ndarray or float): True resistivity [ohm.m].
        a (float, optional): Tortuosity factor. Defaults to 1.0.
        m (float, optional): Cementation exponent. Defaults to 2.0.
        B (float, optional): Equivalent conductance of clay exchange cations. Defaults to 2.0.
        Qv (float, optional): Cation exchange capacity. Defaults to 0.3.

    Returns:
        np.ndarray or float: Estimated formation water resistivity [ohm.m].
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
        rt (np.ndarray or float): True resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        vshale (np.ndarray or float): Volume of shale [fraction].
        depth (np.ndarray or float): Depth log.
        m (float, optional): Shale cementation or shape factor. Defaults to 1.3.

    Returns:
        np.ndarray or float: Estimated formation water resistivity [ohm.m].
    """
    logger.debug(f"Estimating Rw from shale trend with m={m}")

    # Create a mask for valid, finite data points
    valid_data_mask = (
        np.isfinite(rt)
        & np.isfinite(phit)
        & np.isfinite(vshale)
        & np.isfinite(depth)
        & (depth > 0)
    )

    # Identify shale intervals
    vsh_threshold = np.nanquantile(vshale, 0.5)
    shale_mask = (vshale >= vsh_threshold) & valid_data_mask
    rt_shale = rt[shale_mask]
    depth_shale = depth[shale_mask]

    params = np.polyfit(np.log(depth_shale), np.log(rt_shale), 1)
    min_rt = params[0] * np.log(depth) + params[1]

    rw = phit**m * np.exp(min_rt)
    logger.debug(f"Shale trend Rw range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rt_shale(rt, vshale):
    """Estimate shale resistivity (Rsh) from log data.

    This function identifies shale intervals based on a Vshale cutoff (50th percentile),
    applies a rolling average to the resistivity in those intervals, and then
    propagates this value across the entire log interval.

    Args:
        rt (pd.Series or np.ndarray): True resistivity log [ohm.m].
        vshale (pd.Series or np.ndarray): Volume of shale log [fraction].

    Returns:
        pd.Series: Estimated shale resistivity log [ohm.m].
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


def estimate_qv(vcld, phit, rho_clay=2.65, cec_clay=0.062):
    """Estimate Qv (cation exchange capacity per unit pore volume) from clay properties.

    Args:
        vcld (np.ndarray or float): Volume of dry clay [fraction of bulk volume].
        phit (np.ndarray or float): Total porosity [fraction].
        rho_clay (float, optional): Bulk density of dry clay [g/cm³]. Defaults to 2.65.
        cec_clay (float, optional): Cation exchange capacity of clay [meq/g]. Defaults to 0.062.

    Returns:
        np.ndarray or float: Qv in meq/cm³.
    """
    logger.debug(
        f"Estimating Qv with clay density={rho_clay} g/cm³, CEC={cec_clay} meq/g"
    )
    qv = vcld * rho_clay * cec_clay / phit
    logger.debug(f"Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cm³")
    return qv


def estimate_qvn(vclay, phit, phit_clay):
    """Estimate normalized Qv (Qvn) for the Juhasz (1981) model.

    Args:
        vclay (np.ndarray or float): Volume of clay [fraction].
        phit (np.ndarray or float): Total porosity [fraction].
        phit_clay (np.ndarray or float): Total porosity of the clay [fraction].

    Returns:
        np.ndarray or float: Normalized Qv (Qvn).
    """
    return vclay * phit_clay / phit


def estimate_qv_ward(rt, phit, B, rw, m):
    """Estimate Qv from water-bearing zone logs using the Ward (1973) method.

    This method inverts the Waxman-Smits equation, assuming Sw=1, to solve for Qv.

    Args:
        rt (np.ndarray or float): True resistivity in a water-bearing zone [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        B (float): Equivalent conductance of clay exchange cations [S·m²/meq].
        rw (float): Formation water resistivity [ohm.m].
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
        vclb (np.ndarray or float): Volume of clay bound water [fraction of bulk volume].
        phit (np.ndarray or float): Total porosity [fraction].
        water_salinity (float, optional): Water salinity in ppm. Defaults to 10000.

    Returns:
        np.ndarray or float: Qv in meq/cm³.

    References:
        Hill, H.J., Shirley, O.J., and Klein, G.E. (1979). Bound Water in Shaly Sands—Its
        Relation to Qv and Other Formation Properties. The Log Analyst, 20(3).
    """
    logger.debug(
        f"Estimating Qv using Hill method with water salinity={water_salinity}"
    )
    qv = (vclb / phit) / (0.084 * water_salinity**-0.5 + 0.22)
    logger.debug(f"Hill Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cc")
    return qv


def estimate_qv_lavers(phit, a=3.05e-4, b=3.49):
    """Estimate Qv from porosity using the Lavers (1975) empirical equation.

    This method provides a quick estimate of Qv based on a power-law relationship
    with total porosity.

    Args:
        phit (np.ndarray or float): Total porosity [fraction].
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
        phit (np.ndarray or float): Total porosity [fraction].
        max_phit_clean_sand (float): Maximum porosity of the clean sand trend.
        C (float): A constant that depends on the clay type.

    Returns:
        np.ndarray or float: Bulk volume of clay-bound water [fraction of bulk volume].
    """
    return (max_phit_clean_sand - phit) / (C * phit)


def estimate_m_archie(rt, rw, phit):
    """Estimate the apparent cementation exponent (m) using Archie's equation.

    This function inverts Archie's equation in a water-bearing zone (assuming Sw=1)
    to solve for 'm'.

    Args:
        rt (np.ndarray or float): True resistivity in a water-bearing zone [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].

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
        rt (np.ndarray or float): True resistivity in a water-bearing zone [ohm.m].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phie (np.ndarray or float): Effective porosity [fraction].
        vsh (np.ndarray or float): Volume of shale [fraction].
        rsh (np.ndarray or float): Resistivity of shale [ohm.m].

    Returns:
        np.ndarray or float: Apparent cementation exponent (m).
    """
    logger.debug("Estimating apparent m using Indonesian model")
    m = (2 / np.log(phie)) * np.log(
        rw**0.5 * ((1 / rt) ** 0.5 - (vsh ** (1 - 0.5 * vsh) / rsh**0.5))
    )
    logger.debug(f"Indonesian apparent m range: {m.min():.3f} - {m.max():.3f}")
    return m


def estimate_m_star(m_apparent, b_parameter, qv, rw, phi):
    """Estimate intrinsic cementation exponent (m*) from apparent Archie m in shaly sand.

    Corrects an apparent cementation exponent for excess clay conductivity using
    the Waxman-Smits conductivity factor `1 + B * Qv * Rw`.

    Args:
        m_apparent (np.ndarray or float): Apparent Archie cementation exponent (m).
        b_parameter (np.ndarray or float): Waxman-Smits equivalent clay conductance B [S·m²/meq].
        qv (np.ndarray or float): Cation exchange capacity per unit pore volume [meq/cm³].
        rw (np.ndarray or float): Formation water resistivity [ohm.m].
        phi (np.ndarray or float): Total porosity [fraction].

    Returns:
        np.ndarray or float: Intrinsic Waxman-Smits cementation exponent (m*).
    """
    # Conductivity of water
    cw = 1.0 / rw

    # The factor by which clay increases conductivity
    clay_correction_factor = 1 + (b_parameter * qv / cw)

    # Since F = F* / clay_correction_factor
    # and F = phi**-m, F* = phi**-m*
    # We can derive:
    m_star = m_apparent + np.log(clay_correction_factor) / np.log(phi)

    return m_star


def qv_phit_xplot(phit, qv):
    """Generate a Qv vs 1/PHIT crossplot.

    This plot is used to analyze the relationship between cation exchange capacity (Qv)
    and porosity, which can help in understanding clay distribution. A best-fit
    straight line is overlaid on the data.

    Args:
        phit (np.ndarray or float): Total porosity [fraction].
        qv (np.ndarray or float): Cation exchange capacity per unit pore volume [meq/cm³].
    """
    fig, ax = plt.subplots()
    ax.set_title("Qv vs 1/PHIT")

    # Prepare data for plotting and fitting
    x = 1 / phit
    y = qv

    # Remove non-finite values for a clean fit
    mask = np.isfinite(x) & np.isfinite(y)
    ax.scatter(x[mask], y[mask], marker=".")

    # Fit a straight line using np.polyfit
    m, c = np.polyfit(x[mask], y[mask], 1)
    ax.plot(
        x[mask],
        m * x[mask] + c,
        color="r",
        linestyle="--",
        label=f"y = {m:.2f}x + {c:.2f}",
    )

    ax.set_xlabel("1 / PHIT (frac)")
    ax.set_ylabel("Qv (meq/cc)")
    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def cwa_qvn_xplot(rt, phit, qvn, m=2.0, rw=0.2, slope=250):
    """Generate a Cwa vs Qvn plot for shaly sand analysis.

    This plot is used to graphically solve for water saturation in the
    normalized Waxman-Smits (Juhasz) model.

    Args:
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        qvn (np.ndarray or float): Normalized Qv.
        m (float, optional): Cementation exponent. Defaults to 2.0.
        rw (float, optional): Formation water resistivity [ohm.m]. Used for the trend line. Defaults to 0.2.
        slope (float, optional): Slope of the interpretation line. Defaults to 250.0.
    """
    x = qvn
    y = 1 / (rt * phit**m)
    if len(x) != len(y):
        logger.warning(
            f"Length mismatch between Qvn ({len(x)}) and Cwa ({len(y)}). Plot may be incorrect."
        )

    fig, ax = plt.subplots()
    ax.set_title("Cwa vs Qvn")
    ax.scatter(x, y, marker=".")

    # Add straight line
    x_line = np.linspace(0, 1.0, len(x))
    y_line = 1 / rw + x_line * slope
    ax.plot(x_line, y_line, "r--", label=f"m= {m} \nrw= {rw} \nslope= {slope}")

    ax.set_xlabel("Qvn (meq/cc)")
    ax.set_ylabel("Cwa (1 / ohm.m)")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 100)
    ax.minorticks_on()
    ax.grid(True)
    fig.tight_layout()
    ax.legend()
    return fig


def cwa_vclay_phit_xplot(rt, phit, vclay, m=2.0, rw=0.2, slope=250):
    """Generate a Cwa vs Vclay/PHIT ratio plot for shaly sand analysis.

    This plot is used to graphically solve for water saturation in shaly sand models
    where the clay volume to porosity ratio is a key parameter.

    Args:
        rt (np.ndarray or float): True resistivity of the formation [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        vclay (np.ndarray or float): Volume of clay [fraction].
        m (float, optional): Cementation exponent. Defaults to 2.0.
        rw (float, optional): Formation water resistivity [ohm.m]. Used for the trend line. Defaults to 0.2.
        slope (float, optional): Slope of the interpretation line. Defaults to 250.0.
    """
    logger.debug("Creating Cwa vs Vclay/PHIT ratio plot")

    # Calculate Cwa (Conductivity of water-saturated rock)
    cwa = 1 / (rt * phit**m)

    # Calculate Vclay/PHIT ratio
    vclay_phit_ratio = vclay / phit

    fig, ax = plt.subplots()
    ax.set_title("Cwa vs Vclay/PHIT Ratio")
    ax.scatter(vclay_phit_ratio, cwa, marker=".")

    # Add a trend line for interpretation
    x_line = np.linspace(
        0, np.nanmax(vclay_phit_ratio[np.isfinite(vclay_phit_ratio)]), 100
    )
    y_line = 1 / rw + x_line * slope
    ax.plot(x_line, y_line, "r--", label=f"m= {m} \nrw= {rw} \nslope= {slope}")

    ax.set_xlabel("Vclay / PHIT Ratio")
    ax.set_ylabel("Cwa (1 / ohm.m)")
    ax.minorticks_on()
    ax.grid(True)
    fig.tight_layout()
    ax.legend()
    logger.debug("Cwa vs Vclay/PHIT ratio plot created")


def swirr_xplot(swt, phit, c=0.0125, label="", log_log=False, title=""):
    """Generate a Buckles plot (SWT vs. PHIT) to estimate irreducible water saturation.

    This crossplot helps identify the hyperbolic trend of irreducible water
    saturation, where PHIT * Swirr = constant (C).

    Args:
        swt (np.ndarray or float): Water saturation [fraction].
        phit (np.ndarray or float): Total porosity [fraction].
        c (float, optional): The Buckles constant (PHIT * Swirr) for the iso-line. Defaults to 0.0125.
        label (str, optional): Label for the scattered data points. Defaults to ''.
        log_log (bool, optional): If True, both axes will be on a logarithmic scale. Defaults to False.
        title (str, optional): Title for the plot. Defaults to ''.
    """
    logger.debug(f"Creating Swirr crossplot with constant c={c}")
    fig, ax = plt.subplots()
    ax.set_title(title)
    sc = ax.scatter(swt, phit, marker=".", label=label)
    if c:
        line_color = sc.get_facecolors()[0]
        line_color[-1] = 0.75
        cphit = np.geomspace(0.0001, 0.4, 30)
        ax.plot(
            c / cphit,
            cphit,
            label=rf"$\phi$ S = {c}",
            color=line_color,
            linestyle="dashed",
        )
    ax.set_ylabel("PHIT (frac)")
    ax.set_ylim(1e-3, 1)
    ax.set_xlabel("SWT (frac)")
    ax.set_xlim(1e-3, 1)
    ax.legend()
    if log_log:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")
    fig.tight_layout()
    logger.debug("Swirr crossplot created")


def rt_phit_xplot(rt, phit, m=2, rw=0.01):
    """Generate an Rt vs. PHIT (Hingle) plot.

    This plot is useful for visualizing the relationship between true resistivity
    and total porosity, often used for identifying water-bearing zones and estimating
    formation water resistivity (Rw) and cementation exponent (m).

    Args:
        rt (np.ndarray or float): True resistivity or deep resistivity log [ohm.m].
        phit (np.ndarray or float): Total porosity [fraction].
        m (float, optional): Cementation exponent for the iso-line. Defaults to 2.0.
        rw (float, optional): Formation water resistivity for the iso-line [ohm.m]. Defaults to 0.01.

    Returns:
        matplotlib.figure.Figure: The generated RT vs PHIT plot.
    """
    fig, ax = plt.subplots()
    ax.set_title("RT vs PHIT")
    ax.scatter(phit, rt, marker=".")

    # Add iso-lines
    phit_i = np.arange(0, 1, 1 / len(phit))
    rt_i = phit_i**-m * rw
    ax.plot(phit_i, rt_i, "r--", alpha=0.5, label=f"m= {m} \nrw= {rw}")

    ax.set_ylabel("RT (ohm.m)")
    ax.set_xlabel("PHIT (frac)")
    ax.loglog()
    ax.set_xlim(1e-2, 1)
    ax.set_ylim(1e-2, 1e2)

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )
    ax.legend()
    fig.tight_layout()
    return fig


def pickett_plot(rt, phit, m=-2, min_rw=0.1, shift=0.2, title="Pickett Plot"):
    """Generate a Pickett plot (log-log of Rt vs. PHIT).

    This plot is a graphical tool to determine the cementation exponent (m),
    formation water resistivity (Rw), and water saturation (Sw) from log data.

    Args:
        rt (np.ndarray or float): True resistivity log [ohm.m].
        phit (np.ndarray or float): Total porosity log [fraction].
        m (float, optional): The slope of the water-bearing line (should be negative). Defaults to -2.
        min_rw (float, optional): The Rw value for the 100% water saturation line. Defaults to 0.1.
        shift (float, optional): Increment to generate other iso-Sw lines. Defaults to 0.2.
        title (str, optional): Title for the plot. Defaults to 'Pickett Plot'.

    Returns:
        None: The function generates a plot but does not return any object.
    """
    logger.debug(f"Creating Pickett plot with m={m}, min_rw={min_rw}")
    m = m if m < 0 else -m
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title(title)
    ax.scatter(rt, phit, marker=".", color="b")
    # Add iso-lines
    phit_i = np.arange(0, 1, 1 / len(phit))
    for i in np.geomspace(1, 5, num=5):
        c = min_rw + (i - 1) * shift
        sw = round(min_rw / c * 100)
        rt_i = (phit_i**m) * c
        ax.plot(rt_i, phit_i, linestyle="dashed", alpha=0.5, label=f"SW={sw}%")
    # Set up y axis
    ax.set_yscale("log")
    ax.set_ylim(0.01, 1)
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )
    ax.set_ylabel("PHIT (v/v)")
    # Set up x axis
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1000)
    ax.tick_params(top=True, labeltop=True)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )
    ax.set_xlabel("RT (ohm.m)")

    ax.legend()
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")
    fig.tight_layout()
    logger.debug("Pickett plot created")


def RI_plot(sw, rt, ro):
    """Generate a Resistivity Index (RI) vs. Sw plot from lab data.

    This log-log plot is used to determine the saturation exponent (n). The
    slope of the best-fit line through the data is -n. The trend line should
    pass through the (1, 1) point.

    Args:
        sw (np.ndarray or float): Water saturation from core analysis [fraction].
        rt (np.ndarray or float): True resistivity of the core sample at a given Sw [ohm.m].
        ro (np.ndarray or float): Resistivity of the same core sample when 100% water-saturated [ohm.m].

    Returns:
        None: The function generates a plot but does not return any object.
    """
    logger.debug("Creating resistivity index plot")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Resistivity Index Plot")
    ax.scatter(sw, rt / ro)
    ax.set_xlim(0.01, 1)
    ax.set_ylim(1, 1000)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("SW")
    ax.set_ylabel("RT/Rw")
    ax.grid(True)
    fig.tight_layout()
    logger.debug("Resistivity index plot created")


def FF_plot(phit, ro, rw):
    """Generate a Formation Factor (FF) vs. Porosity plot from lab data.

    This log-log plot is used to determine the tortuosity factor (a) and the
    cementation exponent (m). The slope of the best-fit line is -m, and the
    intercept at PHIT=1 is 'a'.

    Args:
        phit (np.ndarray or float): Porosity from core analysis [fraction].
        ro (np.ndarray or float): Resistivity of the 100% water-saturated core sample [ohm.m].
        rw (np.ndarray or float): Resistivity of the saturating water [ohm.m].

    Returns:
        None: The function generates a plot but does not return any object.
    """
    logger.debug("Creating formation factor plot")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title("Formation Factor Plot")
    ax.scatter(phit, ro / rw)
    ax.set_xlim(0.01, 1)
    ax.set_ylim(1, 1000)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("PHIT")
    ax.set_ylabel("Ro/Rw")
    ax.grid(True)
    fig.tight_layout()
    logger.debug("Formation factor plot created")


# ==============================================================================
# Multi-Well Contact & Transition Zone Optimizer (SHM vs. Resistivity & MDT/RFT)
# ==============================================================================


def _gravity_conversion_factor(unit: str = "imperial") -> float:
    """Return hydrostatic pressure gradient conversion factor from g/cc to psi/depth_unit."""
    return 0.433 * 3.28084 if str(unit).strip().lower() in ("metric", "m") else 0.433


def estimate_mdt_fwl(
    pressure_df: pd.DataFrame = None,
    tvdss=None,
    pressure=None,
    fluid_type=None,
    depth_col: str = "TVDSS",
    pressure_col: str = "FORMATION_PRESSURE",
    fluid_col: str = "FLUID_TYPE",
    water_label: str = "WATER",
    hc_label: str = "OIL",
    unit: str = "imperial",
) -> dict:
    """Estimate Free Water Level (FWL) and fluid densities from MDT/RFT formation pressure data.

    Fits linear pressure-depth trends $P(z) = m_w z + c_w$ (aquifer water leg) and
    $P(z) = m_{hc} z + c_{hc}$ (hydrocarbon leg) and solves for their intersection depth
    ($P_c = P_{hc} - P_w = 0$):
      $$\\text{FWL}_{MDT} = \\frac{c_{hc} - c_w}{m_w - m_{hc}}$$

    Args:
        pressure_df (pd.DataFrame, optional): DataFrame containing depth, formation pressure,
            and fluid type columns.
        tvdss (array-like, optional): True vertical depth subsea if `pressure_df` is not provided.
        pressure (array-like, optional): Formation pressure (psi) if `pressure_df` is not provided.
        fluid_type (array-like, optional): Fluid label per pressure point (e.g., `'WATER'`, `'OIL'`, `'GAS'`).
        depth_col (str, optional): Depth column name in `pressure_df`. Defaults to `'TVDSS'`.
        pressure_col (str, optional): Pressure column name in `pressure_df`. Defaults to `'FORMATION_PRESSURE'`.
        fluid_col (str, optional): Fluid type column name in `pressure_df`. Defaults to `'FLUID_TYPE'`.
        water_label (str, optional): Label identifying aquifer water points. Defaults to `'WATER'`.
        hc_label (str, optional): Label identifying hydrocarbon points (e.g., `'OIL'` or `'GAS'`).
            If not found, uses any non-water hydrocarbon points (`'OIL'` or `'GAS'`). Defaults to `'OIL'`.
        unit (str, optional): Depth unit (`'imperial'`/`'ft'` or `'metric'`/`'m'`). Defaults to `'imperial'`.

    Returns:
        dict: Dictionary with keys `'fwl'`, `'pressure_at_fwl'`, `'water_gradient'`, `'hc_gradient'`,
              `'water_intercept'`, `'hc_intercept'`, `'gw'`, and `'ghc'`.
    """
    if pressure_df is not None:
        d_col = depth_col if depth_col in pressure_df.columns else "DEPTH"
        z = np.asarray(pressure_df[d_col], dtype=float)
        p = np.asarray(pressure_df[pressure_col], dtype=float)
        fl = pressure_df[fluid_col].astype(str).str.upper().to_numpy()
    else:
        if tvdss is None or pressure is None or fluid_type is None:
            raise ValueError(
                "Provide either `pressure_df` or (`tvdss`, `pressure`, `fluid_type`)."
            )
        z = np.asarray(tvdss, dtype=float)
        p = np.asarray(pressure, dtype=float)
        fl = pd.Series(fluid_type).astype(str).str.upper().to_numpy()

    valid = np.isfinite(z) & np.isfinite(p)
    w_mask = valid & (fl == water_label.upper())
    hc_mask = valid & (fl == hc_label.upper())
    if np.sum(hc_mask) < 2:
        hc_mask = valid & np.isin(fl, ["OIL", "GAS", "HC", "CONDENSATE"])

    if np.sum(w_mask) < 2 or np.sum(hc_mask) < 2:
        raise ValueError(
            "At least 2 valid pressure points in both water and hydrocarbon legs are required to fit MDT FWL."
        )

    m_w, c_w = np.polyfit(z[w_mask], p[w_mask], 1)
    m_hc, c_hc = np.polyfit(z[hc_mask], p[hc_mask], 1)

    if np.isclose(m_w, m_hc, atol=1e-6):
        raise ValueError(
            f"Water gradient ({m_w:.4f}) and hydrocarbon gradient ({m_hc:.4f}) are nearly parallel."
        )

    fwl_mdt = float((c_hc - c_w) / (m_w - m_hc))
    p_fwl = float(m_w * fwl_mdt + c_w)
    conv = _gravity_conversion_factor(unit)

    return {
        "fwl": fwl_mdt,
        "pressure_at_fwl": p_fwl,
        "water_gradient": float(m_w),
        "hc_gradient": float(m_hc),
        "water_intercept": float(c_w),
        "hc_intercept": float(c_hc),
        "gw": float(m_w / conv),
        "ghc": float(m_hc / conv),
    }


def _normalize_shm_params_map(shm_params, model: str = "leverett_j") -> dict:
    """Normalize `shm_params` into a lookup dict `{rock_flag_key: param_dict}` and `'default'` fallback."""
    model_clean = model.strip().lower()
    is_skelt = model_clean in ("skelt_harrison", "skelt")

    if shm_params is None:
        default_p = (
            {"a": 0.85, "b": 35.0, "c": 1.5, "d": 2.0, "swirr": 0.0}
            if is_skelt
            else {"a": 0.15, "b": 1.5, "swirr": 0.0}
        )
        return {"default": default_p, "by_rock": {}}

    # Case 1: List of dicts (e.g., output of core_analysis.auto_j_params)
    if isinstance(shm_params, list):
        by_rock = {}
        for item in shm_params:
            if isinstance(item, dict) and "ROCK_FLAG" in item:
                rf_val = item["ROCK_FLAG"]
                by_rock[str(rf_val)] = item
                try:
                    by_rock[int(float(rf_val))] = item
                    by_rock[float(rf_val)] = item
                except Exception:
                    pass
        default_item = shm_params[0] if shm_params else {"a": 0.15, "b": 1.5}
        return {"default": default_item, "by_rock": by_rock}

    # Case 2: Dict
    if isinstance(shm_params, dict):
        # Single global parameter dict: e.g. {"a": 0.15, "b": 1.5}
        if "a" in shm_params and "b" in shm_params:
            return {"default": shm_params, "by_rock": {}}

        # Mapping of rock_flag -> dict or tuple
        by_rock = {}
        default_item = None
        for k, v in shm_params.items():
            if isinstance(v, (tuple, list)):
                if is_skelt and len(v) >= 4:
                    p_dict = {
                        "a": float(v[0]),
                        "b": float(v[1]),
                        "c": float(v[2]),
                        "d": float(v[3]),
                        "swirr": float(v[4]) if len(v) > 4 else 0.0,
                    }
                else:
                    p_dict = {
                        "a": float(v[0]),
                        "b": float(v[1]),
                        "swirr": float(v[2]) if len(v) > 2 else 0.0,
                    }
            elif isinstance(v, dict):
                p_dict = v
            else:
                continue

            if default_item is None:
                default_item = p_dict
            by_rock[str(k)] = p_dict
            try:
                by_rock[int(float(k))] = p_dict
                by_rock[float(k)] = p_dict
            except Exception:
                pass

        return {
            "default": default_item or {"a": 0.15, "b": 1.5},
            "by_rock": by_rock,
        }

    # Case 3: Tuple/sequence of parameters (a, b) or (a, b, c, d)
    if isinstance(shm_params, (tuple, list)):
        if is_skelt and len(shm_params) >= 4:
            return {
                "default": {
                    "a": float(shm_params[0]),
                    "b": float(shm_params[1]),
                    "c": float(shm_params[2]),
                    "d": float(shm_params[3]),
                },
                "by_rock": {},
            }
        return {
            "default": {"a": float(shm_params[0]), "b": float(shm_params[1])},
            "by_rock": {},
        }

    raise ValueError(f"Unsupported shm_params format: {type(shm_params)}")


def _extract_rock_param_arrays(
    n: int,
    rock_flag_arr: np.ndarray = None,
    shm_params=None,
    model: str = "leverett_j",
    non_reservoir_flags=None,
):
    """Map per-sample rock flags to parameter arrays (`a`, `b`, `c`, `d`, `swirr`, `is_non_res`)."""
    norm_map = _normalize_shm_params_map(shm_params, model=model)
    default_p = norm_map["default"]
    by_rock = norm_map["by_rock"]

    a_arr = np.full(n, float(default_p.get("a", 0.15)), dtype=float)
    b_arr = np.full(n, float(default_p.get("b", 1.5)), dtype=float)
    c_arr = np.full(n, float(default_p.get("c", 1.5)), dtype=float)
    d_arr = np.full(n, float(default_p.get("d", 1.5)), dtype=float)
    swirr_arr = np.full(n, float(default_p.get("swirr", 0.0)), dtype=float)
    is_non_res = np.zeros(n, dtype=bool)

    non_res_set = set()
    if non_reservoir_flags is not None:
        if np.isscalar(non_reservoir_flags):
            non_res_set.add(non_reservoir_flags)
        else:
            non_res_set.update(non_reservoir_flags)

    if rock_flag_arr is not None and len(by_rock) > 0:
        for i in range(n):
            rf = rock_flag_arr[i]
            if pd.isna(rf) or rf in non_res_set:
                is_non_res[i] = True
                continue
            p = by_rock.get(rf)
            if p is None:
                p = by_rock.get(str(rf))
            if p is None:
                try:
                    p = by_rock.get(int(float(rf)))
                except Exception:
                    p = None
            if p is None:
                is_non_res[i] = True
                continue
            a_arr[i] = float(p.get("a", a_arr[i]))
            b_arr[i] = float(p.get("b", b_arr[i]))
            c_arr[i] = float(p.get("c", c_arr[i]))
            d_arr[i] = float(p.get("d", d_arr[i]))
            if "swirr" in p:
                swirr_arr[i] = float(p["swirr"])
    elif rock_flag_arr is not None and len(non_res_set) > 0:
        for i in range(n):
            if pd.isna(rock_flag_arr[i]) or rock_flag_arr[i] in non_res_set:
                is_non_res[i] = True

    return a_arr, b_arr, c_arr, d_arr, swirr_arr, is_non_res


def predict_sw_shm(
    df: pd.DataFrame = None,
    fwl: float | dict = None,
    model: str = "leverett_j",
    shm_params=None,
    depth=None,
    phit=None,
    perm=None,
    phie=None,
    rock_flag=None,
    swirr=None,
    ift: float = 30.0,
    theta: float = 30.0,
    gw: float = 1.05,
    ghc: float = 0.80,
    depth_col: str = "DEPTH",
    phit_col: str = "PHIT",
    perm_col: str = "PERM",
    phie_col: str = None,
    rock_flag_col: str = "ROCK_FLAG",
    well_col: str = "WELL_NAME",
    non_reservoir_flags=None,
    unit: str = "imperial",
    clip: bool = True,
) -> np.ndarray | pd.Series:
    """Predict Saturation-Height Model water saturation ($S_{w,SHM}$) using Leverett-$J$, Skelt-Harrison, or Cuddy.

    Supports single-well and multi-well evaluation, global or per-well `fwl` dictionaries,
    and rock-type (`ROCK_FLAG` / FZI) specific capillary parameters.

    Args:
        df (pd.DataFrame, optional): Input well log DataFrame. If None, arrays (`depth`, `phit`,
            `perm`, etc.) must be provided directly.
        fwl (float or dict): Free Water Level in true vertical depth (scalar float, or dict mapping
            well name to well FWL).
        model (str, optional): Saturation-height formulation:
            - `'leverett_j'` (or `'j'`): Leverett $J$-function $J = a \\cdot S_{wn}^{-b}$
            - `'skelt_harrison'` (or `'skelt'`): Skelt-Harrison $S_w = 1 - a \\exp(-(b / (h + d))^c)$
            - `'cuddy'`: Cuddy's FOIL function $S_w = a / (P_c^b \\cdot \\phi_t)$
            Defaults to `'leverett_j'`.
        shm_params (dict, list[dict], or tuple, optional): Model parameters:
            - For `'leverett_j'`: `{'a': 0.15, 'b': 1.5}` or list of per-rock-type dicts from `auto_j_params`.
            - For `'skelt_harrison'`: `{'a': 0.85, 'b': 35.0, 'c': 1.5, 'd': 2.0}` or per-rock-type mapping.
            - For `'cuddy'`: `{'a': 0.12, 'b': 0.65}`.
        depth (array-like, optional): TVD / TVDSS array if `df` is None.
        phit (array-like, optional): Total porosity (frac) array if `df` is None.
        perm (array-like, optional): Permeability (mD) array if `df` is None.
        phie (array-like, optional): Effective porosity (frac) array for CBW conversion to $S_{wt}$.
        rock_flag (array-like, optional): Discrete rock type flag array if `df` is None.
        swirr (float or array-like, optional): Irreducible water saturation fraction to un-normalize
            $S_{wn} \\to S_w = S_{wirr} + (1 - S_{wirr}) S_{wn}$. Defaults to None.
        ift (float, optional): Reservoir interfacial tension (dynes/cm). Defaults to 30.0.
        theta (float, optional): Reservoir contact angle (degrees). Defaults to 30.0.
        gw (float, optional): Formation water density (g/cc). Defaults to 1.05.
        ghc (float, optional): Hydrocarbon density (g/cc). Defaults to 0.80.
        depth_col (str, optional): Depth column in `df` (tries `'TVDSS'`, `'TVD'`, `'DEPTH'`).
            Defaults to `'DEPTH'`.
        phit_col (str, optional): Porosity column in `df`. Defaults to `'PHIT'`.
        perm_col (str, optional): Permeability column in `df` (falls back to `'CPERM'`).
            Defaults to `'PERM'`.
        phie_col (str, optional): Optional effective porosity column in `df`. Defaults to None.
        rock_flag_col (str, optional): Rock type column in `df`. Defaults to `'ROCK_FLAG'`.
        well_col (str, optional): Well name column in `df` when `fwl` is a dict. Defaults to `'WELL_NAME'`.
        non_reservoir_flags (list or int, optional): Rock flag values representing non-reservoir
            shale/tight intervals where $S_{w,SHM}$ is forced to 1.0. Defaults to None.
        unit (str, optional): `'imperial'`/`'ft'` or `'metric'`/`'m'`. Defaults to `'imperial'`.
        clip (bool, optional): Clip $S_{w,SHM}$ to $[0.0, 1.0]$ and enforce 1.0 at/below FWL.
            Defaults to True.

    Returns:
        pd.Series or np.ndarray: Predicted water saturation $S_{w,SHM}$ (frac).
    """
    if fwl is None:
        raise ValueError("`fwl` (Free Water Level) must be provided to predict_sw_shm.")

    return_series = False
    series_index = None

    if df is not None:
        return_series = True
        series_index = df.index
        d_col = depth_col
        if d_col not in df.columns:
            for cand in ("TVDSS", "TVD", "DEPTH", "depth", "tvdss"):
                if cand in df.columns:
                    d_col = cand
                    break
        depth_arr = np.asarray(df[d_col], dtype=float)
        phit_arr = (
            np.asarray(df[phit_col], dtype=float)
            if phit_col in df.columns
            else np.full(len(df), 0.15, dtype=float)
        )
        p_col = (
            perm_col
            if perm_col in df.columns
            else ("CPERM" if "CPERM" in df.columns else None)
        )
        perm_arr = (
            np.asarray(df[p_col], dtype=float)
            if p_col is not None
            else np.full(len(df), 10.0, dtype=float)
        )
        phie_arr = (
            np.asarray(df[phie_col], dtype=float)
            if (phie_col and phie_col in df.columns)
            else None
        )
        rf_arr = (
            df[rock_flag_col].to_numpy()
            if rock_flag_col in df.columns
            else (np.asarray(rock_flag) if rock_flag is not None else None)
        )
        if isinstance(fwl, dict) and well_col in df.columns:
            default_fwl = float(next(iter(fwl.values())))
            fwl_arr = (
                df[well_col].map(lambda w: float(fwl.get(w, default_fwl))).to_numpy(dtype=float)
            )
        else:
            fwl_arr = np.full(
                len(df),
                float(next(iter(fwl.values())) if isinstance(fwl, dict) else fwl),
                dtype=float,
            )
    else:
        if depth is None:
            raise ValueError("Either `df` or `depth` must be provided to predict_sw_shm.")
        if isinstance(depth, pd.Series):
            return_series = True
            series_index = depth.index
        depth_arr = np.asarray(depth, dtype=float)
        n_pts = len(depth_arr)
        phit_arr = (
            np.broadcast_to(np.asarray(phit, dtype=float), (n_pts,))
            if phit is not None
            else np.full(n_pts, 0.15, dtype=float)
        )
        perm_arr = (
            np.broadcast_to(np.asarray(perm, dtype=float), (n_pts,))
            if perm is not None
            else np.full(n_pts, 10.0, dtype=float)
        )
        phie_arr = (
            np.broadcast_to(np.asarray(phie, dtype=float), (n_pts,))
            if phie is not None
            else None
        )
        rf_arr = np.asarray(rock_flag) if rock_flag is not None else None
        fwl_arr = np.broadcast_to(np.asarray(fwl, dtype=float), (n_pts,))

    n = len(depth_arr)
    a_arr, b_arr, c_arr, d_arr, swirr_map_arr, is_non_res = _extract_rock_param_arrays(
        n,
        rock_flag_arr=rf_arr,
        shm_params=shm_params,
        model=model,
        non_reservoir_flags=non_reservoir_flags,
    )

    if swirr is not None:
        if isinstance(swirr, str) and df is not None and swirr in df.columns:
            swirr_use = np.asarray(df[swirr], dtype=float)
        else:
            swirr_use = np.broadcast_to(np.asarray(swirr, dtype=float), (n,))
    else:
        swirr_use = swirr_map_arr

    model_clean = model.strip().lower()
    conv = _gravity_conversion_factor(unit)
    h = np.maximum(fwl_arr - depth_arr, 0.0)

    if model_clean in ("leverett_j", "j"):
        pc = h * (gw - ghc) * conv
        denom = max(float(ift) * abs(np.cos(np.radians(theta))), 1e-9)
        j_val = (
            0.21665
            * (pc / denom)
            * np.sqrt(np.maximum(perm_arr, 0.0) / np.maximum(phit_arr, 1e-6))
        )
        b_abs = np.maximum(np.abs(b_arr), 1e-6)
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            swn = np.where(
                (h > 0) & (j_val > 0) & (a_arr > 0) & (perm_arr > 0) & (phit_arr > 0),
                (a_arr / j_val) ** (1.0 / b_abs),
                1.0,
            )
        sw_pred = swirr_use + (1.0 - swirr_use) * swn
    elif model_clean in ("skelt_harrison", "skelt"):
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sw_pred = 1.0 - a_arr * np.exp(-((b_arr / np.maximum(h + d_arr, 1e-6)) ** c_arr))
        if np.any(swirr_use > 0):
            sw_pred = np.maximum(sw_pred, swirr_use)
    elif model_clean == "cuddy":
        b_abs = np.maximum(np.abs(b_arr), 1e-6)
        pc = h * (gw - ghc) * conv
        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
            sw_pred = np.where(
                (h > 0) & (pc > 0) & (phit_arr > 0),
                a_arr / ((pc ** b_abs) * np.maximum(phit_arr, 1e-6)),
                1.0,
            )
        sw_pred = swirr_use + (1.0 - swirr_use) * sw_pred
    else:
        raise ValueError(
            f"Unsupported SHM model '{model}'. Choose from 'leverett_j', 'skelt_harrison', or 'cuddy'."
        )

    if phie_arr is not None:
        swb = np.clip(1.0 - (phie_arr / np.maximum(phit_arr, 1e-6)), 0.0, 1.0)
        sw_pred = sw_pred * (1.0 - swb) + swb

    sw_pred = np.where(is_non_res | (fwl_arr - depth_arr <= 0), 1.0, sw_pred)
    if clip:
        sw_pred = np.clip(sw_pred, 0.0, 1.0)

    if return_series:
        return pd.Series(sw_pred, index=series_index, name="SW_SHM")
    return sw_pred


calc_sw_shm = predict_sw_shm


def analyze_transition_zone(
    fwl: float,
    model: str = "leverett_j",
    shm_params=None,
    phit: float | np.ndarray = 0.18,
    perm: float | np.ndarray = 50.0,
    ift: float = 30.0,
    theta: float = 30.0,
    gw: float = 1.05,
    ghc: float = 0.80,
    swirr: float = 0.0,
    sw_owc_threshold: float = 0.95,
    sw_ttz_tol: float = 0.05,
    unit: str = "imperial",
) -> dict:
    """Compute theoretical capillary entry height (OWC), Top of Transition Zone (TTZ), and thickness.

    For a given Free Water Level (`fwl`), rock quality ($\\phi_t, k$), and fluid properties:
      - **Free Water Level (`fwl`)**: Depth where capillary pressure $P_c = 0$.
      - **Oil-Water / Hydrocarbon-Water Contact (`owc`)**: Depth where capillary pressure equals
        the pore-throat displacement / entry pressure $P_{ce}$ (for Leverett-$J$, $J_{entry} = a$ at
        $S_{wn} = 1.0$; for Skelt-Harrison, depth where $S_{w,SHM} = S_{w,owc}$).
      - **Top of Transition Zone (`ttz`)**: Depth above which water saturation reaches within
        `sw_ttz_tol` of irreducible water saturation ($S_{wn} \\le \\text{sw\\_ttz\\_tol}$).
      - **Transition Zone Thickness (`transition_zone_thickness`)**: $\\text{OWC} - \\text{TTZ}$.
      - **Capillary Fringe / Threshold Height (`capillary_entry_height`)**: $\\text{FWL} - \\text{OWC}$.

    Args:
        fwl (float): Free Water Level in true vertical depth (ft or m).
        model (str, optional): `'leverett_j'` or `'skelt_harrison'`. Defaults to `'leverett_j'`.
        shm_params (dict, list, or tuple, optional): SHM parameters.
        phit (float or array-like, optional): Representative reservoir porosity (frac). Defaults to 0.18.
        perm (float or array-like, optional): Representative reservoir permeability (mD). Defaults to 50.0.
        ift (float, optional): Interfacial tension (dynes/cm). Defaults to 30.0.
        theta (float, optional): Contact angle (degrees). Defaults to 30.0.
        gw (float, optional): Water density (g/cc). Defaults to 1.05.
        ghc (float, optional): Hydrocarbon density (g/cc). Defaults to 0.80.
        swirr (float, optional): Irreducible water saturation fraction. Defaults to 0.0.
        sw_owc_threshold (float, optional): Saturation threshold defining OWC for Skelt-Harrison.
            Defaults to 0.95.
        sw_ttz_tol (float, optional): Normalized saturation tolerance above $S_{wirr}$ defining
            the Top of Transition Zone (TTZ). Defaults to 0.05.
        unit (str, optional): `'imperial'`/`'ft'` or `'metric'`/`'m'`. Defaults to `'imperial'`.

    Returns:
        dict: Dictionary containing `'fwl'`, `'owc'`, `'ttz'`, `'capillary_entry_height'`,
              `'transition_zone_thickness'`, and `'total_height_to_ttz'`.
    """
    norm_map = _normalize_shm_params_map(shm_params, model=model)
    p = norm_map["default"]
    model_clean = model.strip().lower()
    conv = _gravity_conversion_factor(unit)
    delta_rho_grad = max((gw - ghc) * conv, 1e-6)

    phi_rep = float(np.nanmedian(phit)) if not np.isscalar(phit) else float(phit)
    perm_rep = float(np.nanmedian(perm)) if not np.isscalar(perm) else float(perm)
    phi_rep = max(phi_rep, 1e-4)
    perm_rep = max(perm_rep, 1e-4)

    if model_clean in ("leverett_j", "j"):
        a = max(float(p.get("a", 0.15)), 1e-6)
        b = max(abs(float(p.get("b", 1.5))), 1e-3)
        denom = float(ift) * abs(np.cos(np.radians(theta)))
        rqi_term = 0.21665 * np.sqrt(perm_rep / phi_rep)

        # Entry capillary pressure at Swn = 1.0 -> J_entry = a
        pce = (a * denom) / max(rqi_term, 1e-9)
        h_entry = float(pce / delta_rho_grad)

        # Top of transition zone at Swn_top = clip(sw_ttz_tol, 0.01, 0.5)
        swn_top = float(np.clip(sw_ttz_tol, 0.02, 0.50))
        j_top = a * (swn_top ** (-b))
        pc_top = (j_top * denom) / max(rqi_term, 1e-9)
        h_top = float(pc_top / delta_rho_grad)
    else:
        a = float(np.clip(p.get("a", 0.85), 0.05, 0.999))
        b = max(float(p.get("b", 35.0)), 1e-3)
        c = max(float(p.get("c", 1.5)), 1e-3)
        d = float(p.get("d", 2.0))

        # Solve 1 - a * exp(-((b / (h + d)) ^ c)) = sw_target
        def _solve_h_skelt(sw_target: float) -> float:
            ratio = np.clip((1.0 - sw_target) / a, 1e-6, 1.0 - 1e-6)
            neg_ln = max(-np.log(ratio), 1e-9)
            h_val = (b / (neg_ln ** (1.0 / c))) - d
            return float(max(h_val, 0.0))

        sw_irr_model = 1.0 - a
        sw_owc_eff = float(np.clip(sw_owc_threshold, sw_irr_model + 0.05, 0.99))
        sw_top_eff = float(np.clip(sw_irr_model + sw_ttz_tol * a, sw_irr_model + 0.01, 0.90))
        h_entry = _solve_h_skelt(sw_owc_eff)
        h_top = max(_solve_h_skelt(sw_top_eff), h_entry)

    owc = float(fwl - h_entry)
    ttz = float(fwl - h_top)
    tz_thickness = float(max(owc - ttz, 0.0))

    return {
        "fwl": float(fwl),
        "owc": owc,
        "ttz": ttz,
        "capillary_entry_height": h_entry,
        "transition_zone_thickness": tz_thickness,
        "total_height_to_ttz": h_top,
    }


def optimize_fwl(
    df: pd.DataFrame | dict,
    model: str = "leverett_j",
    shm_params=None,
    fwl_bounds: tuple = None,
    ift: float = 30.0,
    theta: float = 30.0,
    gw: float = 1.05,
    ghc: float = 0.80,
    swirr=None,
    depth_col: str = "DEPTH",
    sw_col: str = "SWT",
    phit_col: str = "PHIT",
    perm_col: str = "PERM",
    phie_col: str = None,
    rock_flag_col: str = "ROCK_FLAG",
    well_col: str = "WELL_NAME",
    vshale_col: str = "VSHALE",
    phit_cutoff: float = 0.03,
    perm_cutoff: float = 0.01,
    vshale_cutoff: float = 0.50,
    non_reservoir_flags=None,
    loss: str = "rmse",
    pore_volume_weighted: bool = False,
    equal_well_weights: bool = False,
    mdt_fwl: float = None,
    pressure_df: pd.DataFrame = None,
    mdt_tolerance: float = None,
    mdt_penalty_weight: float = 0.0,
    solve_per_well: bool = True,
    solve_per_zone: bool = True,
    zone_col: str = "ZONES",
    optimize_shm_params: bool = False,
    saturation_method: str = "archie",
    archie_params: dict = None,
    unit: str = "imperial",
) -> dict:
    """Solve for Free Water Level (FWL) across multiple wells simultaneously by minimizing
    the error between resistivity-derived $S_w$ and $J$-function / Skelt-Harrison $S_{w,SHM}$.

    This function reconciles log-derived water saturation ($S_{w,log}$), Special Core Analysis
    capillary pressure Saturation-Height Models ($S_{w,SHM}$), and wireline formation tester
    (MDT/RFT) Free Water Levels across one or more wells simultaneously.

    Key features:
      1. **Multi-Well Simultaneous Optimization**: Solves for a field-wide (or fault-block) FWL
         using a two-stage global grid bracket + bounded Brent local refinement so it never gets
         trapped in flat aquifer/caprock objective plateaus.
      2. **Per-Well FWL Diagnostics**: Optionally solves for each well's individual FWL
         (`well_metrics['fwl_well']`) to diagnose hydrodynamic tilt or fault compartmentalization.
      3. **MDT/RFT Pressure Reconciliation**: Accepts an explicit `mdt_fwl` or raw `pressure_df`
         (via `estimate_mdt_fwl`), supporting hard search bounds (`mdt_tolerance`) and/or soft
         regularization (`mdt_penalty_weight`), and reports `fwl_reconciliation_diff`.
      4. **Transition Zone Delineation**: Computes `owc` (capillary entry depth), `ttz` (top of
         transition zone), `transition_zone_thickness`, and per-sample `TRANSITION_ZONE_FLAG`
         (`'FREE_WATER'`, `'CAPILLARY_FRINGE'`, `'TRANSITION_ZONE'`, `'IRREDUCIBLE_HC'`).

    Args:
        df (pd.DataFrame or dict[str, pd.DataFrame]): Multi-well DataFrame (or dict of per-well
            DataFrames) containing depth, porosity, permeability, and resistivity-derived $S_w$
            (or `RT` and `RW` to compute $S_w$ automatically).
        model (str, optional): SHM formulation, `'leverett_j'` (or `'j'`), `'skelt_harrison'`
            (or `'skelt'`), or `'cuddy'`. Defaults to `'leverett_j'`.
        shm_params (dict, list[dict], or tuple, optional): Capillary parameters for the SHM
            (either global or keyed by `ROCK_FLAG`).
        fwl_bounds (tuple[float, float], optional): `(fwl_min, fwl_max)` search range in TVD/TVDSS.
            If None, inferred automatically from the reservoir depth range (or `mdt_fwl ± mdt_tolerance`).
        ift (float, optional): Reservoir interfacial tension (dynes/cm). Defaults to 30.0.
        theta (float, optional): Reservoir contact angle (degrees). Defaults to 30.0.
        gw (float, optional): Formation water density (g/cc). Defaults to 1.05.
        ghc (float, optional): Hydrocarbon density (g/cc). Defaults to 0.80.
        swirr (float, str, or array-like, optional): Irreducible water saturation fraction.
            Defaults to None.
        depth_col (str, optional): Depth column name (`'DEPTH'`, `'TVDSS'`, or `'TVD'`).
            Defaults to `'DEPTH'`.
        sw_col (str, optional): Resistivity-derived water saturation column name (falls back to
            `'SW'` or computes from `'RT'`/`'RW'`). Defaults to `'SWT'`.
        phit_col (str, optional): Total porosity column name. Defaults to `'PHIT'`.
        perm_col (str, optional): Permeability column name (falls back to `'CPERM'`).
            Defaults to `'PERM'`.
        phie_col (str, optional): Optional effective porosity column name. Defaults to None.
        rock_flag_col (str, optional): Rock type flag column name. Defaults to `'ROCK_FLAG'`.
        well_col (str, optional): Well identifier column name. Defaults to `'WELL_NAME'`.
        vshale_col (str, optional): Shale volume column name for reservoir filtering.
            Defaults to `'VSHALE'`.
        phit_cutoff (float, optional): Minimum porosity cutoff for reservoir optimization mask.
            Defaults to 0.03.
        perm_cutoff (float, optional): Minimum permeability cutoff (mD) for optimization mask.
            Defaults to 0.01.
        vshale_cutoff (float, optional): Maximum shale volume cutoff for optimization mask.
            Defaults to 0.50.
        non_reservoir_flags (list or int, optional): Rock flag values excluded from reservoir
            optimization mask. Defaults to None.
        loss (str, optional): Objective loss function: `'rmse'`, `'mae'`, `'huber'`, or `'bvw'`
            (Bulk Volume Water error). Defaults to `'rmse'`.
        pore_volume_weighted (bool, optional): Weight sample residuals by $\\phi_t \\cdot h_i$.
            Defaults to False.
        equal_well_weights (bool, optional): Weight each well equally in the multi-well objective
            regardless of sample count. Defaults to False.
        mdt_fwl (float, optional): Prior Free Water Level from MDT/RFT pressure gradients.
            Defaults to None.
        pressure_df (pd.DataFrame, optional): Raw MDT/RFT pressure DataFrame; if provided,
            `mdt_fwl` (and fluid densities if not overridden) is estimated via `estimate_mdt_fwl`.
        mdt_tolerance (float, optional): Maximum allowed deviation $|\\text{FWL} - \\text{FWL}_{MDT}|$
            used to bound the optimizer when `mdt_fwl` is present. Defaults to None.
        mdt_penalty_weight (float, optional): Soft regularization weight $\\lambda_{MDT} \\ge 0$
            penalizing normalized distance from `mdt_fwl`. Defaults to 0.0.
        solve_per_well (bool, optional): Also compute individual per-well optimal FWLs in
            `well_metrics` to diagnose contact tilt or compartmentalization. Defaults to True.
        optimize_shm_params (bool, optional): Jointly refine global SHM parameters alongside
            `fwl` using L-BFGS-B / Nelder-Mead. Defaults to False.
        saturation_method (str, optional): Method to compute resistivity $S_w$ if `sw_col` is
            absent (`'archie'` or `'waxman_smits'`). Defaults to `'archie'`.
        archie_params (dict, optional): Parameters `{'a': 1.0, 'm': 2.0, 'n': 2.0, 'rw': 0.05}`
            used if resistivity $S_w$ is computed on the fly. Defaults to None.
        unit (str, optional): `'imperial'`/`'ft'` or `'metric'`/`'m'`. Defaults to `'imperial'`.

    Returns:
        dict: Optimization and reconciliation summary containing:
            - `'fwl'`: Optimal multi-well Free Water Level (float)
            - `'owc'`: Representative Oil-Water / Hydrocarbon-Water Contact (float)
            - `'ttz'`: Representative Top of Transition Zone (float)
            - `'capillary_entry_height'`: Capillary fringe height $\\text{FWL} - \\text{OWC}$ (float)
            - `'transition_zone_thickness'`: Transition zone thickness $\\text{OWC} - \\text{TTZ}$ (float)
            - `'rmse'`: Global $S_w$ RMSE across reservoir intervals (float)
            - `'mae'`: Global $S_w$ MAE across reservoir intervals (float)
            - `'r2'`: Global $S_w$ $R^2$ score across reservoir intervals (float)
            - `'bvw_rmse'`: Global Bulk Volume Water ($\\phi_t S_w$) RMSE (float)
            - `'mdt_fwl'`: MDT/RFT Free Water Level if supplied/estimated, else None
            - `'mdt_details'`: Full dict from `estimate_mdt_fwl` if `pressure_df` supplied
            - `'fwl_reconciliation_diff'`: $\\text{FWL}_{opt} - \\text{FWL}_{MDT}$ if `mdt_fwl` available
            - `'shm_params'`: Active or optimized SHM parameter dictionary/list
            - `'well_metrics'`: Per-well reconciliation `pd.DataFrame`
            - `'grid_profile'`: `pd.DataFrame` of candidate `fwl` vs. `objective_loss`
            - `'df'`: Augmented `pd.DataFrame` with `'SW_SHM'`, `'BVW_LOG'`, `'BVW_SHM'`,
              `'HAFWL'`, `'PC_RES'`, and `'TRANSITION_ZONE_FLAG'`
    """
    from scipy.optimize import minimize, minimize_scalar

    # 1. Consolidate input DataFrame (multi-well df or dict of well dfs)
    if isinstance(df, dict):
        parts = []
        for w_name, w_df in df.items():
            tmp = w_df.copy()
            if well_col not in tmp.columns:
                tmp[well_col] = str(w_name)
            parts.append(tmp)
        work_df = pd.concat(parts, ignore_index=True)
    elif isinstance(df, pd.DataFrame):
        work_df = df.copy()
        if well_col not in work_df.columns:
            work_df[well_col] = "WELL_1"
    else:
        raise ValueError("`df` must be a pandas DataFrame or a dict of {well_name: DataFrame}.")

    # Resolve depth column
    d_col = depth_col
    if d_col not in work_df.columns:
        for cand in ("TVDSS", "TVD", "DEPTH", "depth", "tvdss"):
            if cand in work_df.columns:
                d_col = cand
                break
    if d_col not in work_df.columns:
        raise ValueError(f"Depth column '{depth_col}' not found in DataFrame.")

    # Resolve porosity & permeability columns
    if phit_col not in work_df.columns:
        for cand in ("PHIT", "PHIE", "CPORE", "phit"):
            if cand in work_df.columns:
                phit_col = cand
                break
    if perm_col not in work_df.columns:
        for cand in ("PERM", "CPERM", "CORE_PERM", "perm"):
            if cand in work_df.columns:
                perm_col = cand
                break

    # Resolve or compute resistivity-derived Sw column
    if sw_col not in work_df.columns:
        if "SW" in work_df.columns:
            sw_col = "SW"
        elif "SWT" in work_df.columns:
            sw_col = "SWT"
        elif "RT" in work_df.columns and phit_col in work_df.columns:
            ap = archie_params or {}
            rw_val = (
                work_df["RW"].to_numpy(dtype=float)
                if "RW" in work_df.columns
                else float(ap.get("rw", 0.05))
            )
            a_arch = float(ap.get("a", 1.0))
            m_arch = float(ap.get("m", 2.0))
            n_arch = float(ap.get("n", 2.0))
            if saturation_method.strip().lower() == "waxman_smits":
                qv_val = work_df["QV"].to_numpy(dtype=float) if "QV" in work_df.columns else 0.3
                b_val = work_df["B"].to_numpy(dtype=float) if "B" in work_df.columns else None
                work_df[sw_col] = np.clip(
                    waxman_smits_saturation(
                        work_df["RT"].to_numpy(dtype=float),
                        rw_val,
                        work_df[phit_col].to_numpy(dtype=float),
                        Qv=qv_val,
                        B=b_val,
                        m=m_arch,
                        n=n_arch,
                    ),
                    0.0,
                    1.0,
                )
            else:
                work_df[sw_col] = np.clip(
                    archie_saturation(
                        work_df["RT"].to_numpy(dtype=float),
                        rw_val,
                        work_df[phit_col].to_numpy(dtype=float),
                        a=a_arch,
                        m=m_arch,
                        n=n_arch,
                    ),
                    0.0,
                    1.0,
                )
        else:
            raise ValueError(
                f"Water saturation column '{sw_col}' not found and could not be computed from 'RT'."
            )

    # 2. Estimate MDT FWL if pressure_df is provided
    mdt_details = None
    if pressure_df is not None:
        mdt_details = estimate_mdt_fwl(pressure_df, unit=unit)
        if mdt_fwl is None:
            mdt_fwl = mdt_details["fwl"]

    depth_arr = np.asarray(work_df[d_col], dtype=float)
    sw_log_arr = np.clip(np.asarray(work_df[sw_col], dtype=float), 0.0, 1.0)
    phit_arr = (
        np.asarray(work_df[phit_col], dtype=float)
        if phit_col in work_df.columns
        else np.full(len(work_df), 0.15, dtype=float)
    )
    perm_arr = (
        np.asarray(work_df[perm_col], dtype=float)
        if perm_col in work_df.columns
        else np.full(len(work_df), 50.0, dtype=float)
    )
    phie_arr = (
        np.asarray(work_df[phie_col], dtype=float)
        if (phie_col and phie_col in work_df.columns)
        else None
    )
    rf_arr = (
        work_df[rock_flag_col].to_numpy()
        if rock_flag_col in work_df.columns
        else None
    )
    wells_arr = work_df[well_col].astype(str).to_numpy()

    # 3. Build valid reservoir mask for optimization
    model_clean = model.strip().lower()
    is_leverett = model_clean in ("leverett_j", "j")
    _, _, _, _, _, is_non_res = _extract_rock_param_arrays(
        len(work_df),
        rock_flag_arr=rf_arr,
        shm_params=shm_params,
        model=model,
        non_reservoir_flags=non_reservoir_flags,
    )

    res_mask = (
        np.isfinite(depth_arr)
        & np.isfinite(sw_log_arr)
        & np.isfinite(phit_arr)
        & (phit_arr >= phit_cutoff)
        & (~is_non_res)
    )
    if is_leverett:
        res_mask = res_mask & np.isfinite(perm_arr) & (perm_arr >= perm_cutoff)

    for vcol in (vshale_col, "VCLAY", "VSH"):
        if vcol in work_df.columns:
            vsh_vals = np.asarray(work_df[vcol], dtype=float)
            res_mask = res_mask & (np.isnan(vsh_vals) | (vsh_vals <= vshale_cutoff))
            break

    if not np.any(res_mask):
        res_mask = np.isfinite(depth_arr) & np.isfinite(sw_log_arr)

    # Compute sample weights (pore-volume weighting & equal well weighting)
    weights = np.ones(len(work_df), dtype=float)
    if pore_volume_weighted:
        weights = weights * np.maximum(phit_arr, 1e-4)
    if equal_well_weights:
        for w in np.unique(wells_arr[res_mask]):
            w_m = res_mask & (wells_arr == w)
            n_w = np.sum(w_m)
            if n_w > 0:
                weights[w_m] = weights[w_m] / float(n_w)
    w_res = weights[res_mask]
    w_res = w_res / np.maximum(np.sum(w_res), 1e-12)

    # 4. Determine FWL search bounds
    d_min = float(np.nanmin(depth_arr[res_mask]))
    d_max = float(np.nanmax(depth_arr[res_mask]))
    d_span = max(d_max - d_min, 50.0)

    if fwl_bounds is not None:
        fwl_low, fwl_high = float(fwl_bounds[0]), float(fwl_bounds[1])
    elif mdt_fwl is not None and mdt_tolerance is not None:
        fwl_low = float(mdt_fwl - abs(mdt_tolerance))
        fwl_high = float(mdt_fwl + abs(mdt_tolerance))
    else:
        fwl_low = d_min - 0.10 * d_span
        fwl_high = d_max + 0.50 * d_span
        if mdt_fwl is not None:
            fwl_low = min(fwl_low, float(mdt_fwl) - 0.25 * d_span)
            fwl_high = max(fwl_high, float(mdt_fwl) + 0.25 * d_span)

    if fwl_high <= fwl_low:
        fwl_high = fwl_low + 100.0

    loss_clean = loss.strip().lower()

    def _compute_loss(
        cand_fwl: float,
        cand_params=shm_params,
        subset_mask: np.ndarray = res_mask,
        subset_weights: np.ndarray = w_res,
    ) -> float:
        sw_pred = predict_sw_shm(
            df=None,
            fwl=cand_fwl,
            model=model,
            shm_params=cand_params,
            depth=depth_arr[subset_mask],
            phit=phit_arr[subset_mask],
            perm=perm_arr[subset_mask],
            phie=phie_arr[subset_mask] if phie_arr is not None else None,
            rock_flag=rf_arr[subset_mask] if rf_arr is not None else None,
            swirr=swirr[subset_mask]
            if isinstance(swirr, np.ndarray)
            else (
                np.asarray(work_df[swirr], dtype=float)[subset_mask]
                if isinstance(swirr, str) and swirr in work_df.columns
                else swirr
            ),
            ift=ift,
            theta=theta,
            gw=gw,
            ghc=ghc,
            non_reservoir_flags=non_reservoir_flags,
            unit=unit,
            clip=True,
        )
        sw_obs = sw_log_arr[subset_mask]
        if loss_clean == "bvw":
            phi_sub = phit_arr[subset_mask]
            resid = phi_sub * (sw_obs - sw_pred)
            base_err = float(np.sqrt(np.sum(subset_weights * (resid**2))))
        elif loss_clean == "mae":
            resid = np.abs(sw_obs - sw_pred)
            base_err = float(np.sum(subset_weights * resid))
        elif loss_clean == "huber":
            delta = 0.10
            abs_r = np.abs(sw_obs - sw_pred)
            huber = np.where(
                abs_r <= delta, 0.5 * (abs_r**2), delta * (abs_r - 0.5 * delta)
            )
            base_err = float(np.sqrt(2.0 * np.sum(subset_weights * huber)))
        else:
            resid = sw_obs - sw_pred
            base_err = float(np.sqrt(np.sum(subset_weights * (resid**2))))

        if mdt_fwl is not None and mdt_penalty_weight > 0:
            norm_dist = (cand_fwl - float(mdt_fwl)) / max(d_span, 1.0)
            base_err += float(mdt_penalty_weight) * (norm_dist**2)

        return base_err

    # 5. Two-stage global grid search + local bounded refinement for FWL
    n_grid = 250
    grid_fwls = np.linspace(fwl_low, fwl_high, n_grid)
    grid_losses = np.array([_compute_loss(float(f)) for f in grid_fwls], dtype=float)
    best_idx = int(np.argmin(grid_losses))

    left_idx = max(0, best_idx - 2)
    right_idx = min(n_grid - 1, best_idx + 2)
    local_low = float(grid_fwls[left_idx])
    local_high = float(grid_fwls[right_idx])
    if local_high <= local_low:
        local_low, local_high = fwl_low, fwl_high

    opt_res = minimize_scalar(
        _compute_loss,
        bounds=(local_low, local_high),
        method="bounded",
        options={"xatol": 1e-4, "maxiter": 200},
    )
    opt_fwl = (
        float(opt_res.x)
        if opt_res.fun <= grid_losses[best_idx]
        else float(grid_fwls[best_idx])
    )
    active_shm_params = shm_params

    # Optional joint refinement of global SHM parameters + FWL
    if optimize_shm_params:
        norm_p = _normalize_shm_params_map(shm_params, model=model)["default"]
        if is_leverett:
            x0 = np.array(
                [opt_fwl, float(norm_p.get("a", 0.15)), abs(float(norm_p.get("b", 1.5)))],
                dtype=float,
            )
            bounds_joint = [(fwl_low, fwl_high), (1e-4, 5.0), (0.1, 10.0)]

            def _joint_obj_j(x):
                return _compute_loss(float(x[0]), cand_params={"a": float(x[1]), "b": float(x[2])})

            res_j = minimize(_joint_obj_j, x0, method="L-BFGS-B", bounds=bounds_joint)
            if res_j.fun <= _compute_loss(opt_fwl, shm_params):
                opt_fwl = float(res_j.x[0])
                active_shm_params = {"a": round(float(res_j.x[1]), 4), "b": round(float(res_j.x[2]), 4)}
        elif model_clean in ("skelt_harrison", "skelt"):
            x0 = np.array(
                [
                    opt_fwl,
                    float(norm_p.get("a", 0.85)),
                    float(norm_p.get("b", 35.0)),
                    float(norm_p.get("c", 1.5)),
                    float(norm_p.get("d", 2.0)),
                ],
                dtype=float,
            )
            bounds_joint = [
                (fwl_low, fwl_high),
                (0.1, 0.999),
                (0.1, 5000.0),
                (0.2, 5.0),
                (0.0, 200.0),
            ]

            def _joint_obj_sh(x):
                return _compute_loss(
                    float(x[0]),
                    cand_params={
                        "a": float(x[1]),
                        "b": float(x[2]),
                        "c": float(x[3]),
                        "d": float(x[4]),
                    },
                )

            res_sh = minimize(_joint_obj_sh, x0, method="L-BFGS-B", bounds=bounds_joint)
            if res_sh.fun <= _compute_loss(opt_fwl, shm_params):
                opt_fwl = float(res_sh.x[0])
                active_shm_params = {
                    "a": round(float(res_sh.x[1]), 4),
                    "b": round(float(res_sh.x[2]), 4),
                    "c": round(float(res_sh.x[3]), 4),
                    "d": round(float(res_sh.x[4]), 4),
                }

    # 6. Evaluate final SW_SHM and transition zone profile across all samples
    sw_shm_full = np.asarray(
        predict_sw_shm(
            df=work_df,
            fwl=opt_fwl,
            model=model,
            shm_params=active_shm_params,
            swirr=swirr,
            ift=ift,
            theta=theta,
            gw=gw,
            ghc=ghc,
            depth_col=d_col,
            phit_col=phit_col,
            perm_col=perm_col,
            phie_col=phie_col,
            rock_flag_col=rock_flag_col,
            well_col=well_col,
            non_reservoir_flags=non_reservoir_flags,
            unit=unit,
            clip=True,
        ),
        dtype=float,
    )

    tz_info = analyze_transition_zone(
        fwl=opt_fwl,
        model=model,
        shm_params=active_shm_params,
        phit=phit_arr[res_mask],
        perm=perm_arr[res_mask],
        ift=ift,
        theta=theta,
        gw=gw,
        ghc=ghc,
        unit=unit,
    )

    conv = _gravity_conversion_factor(unit)
    hafwl = np.maximum(opt_fwl - depth_arr, 0.0)
    pc_res = hafwl * (gw - ghc) * conv

    tz_flag = np.where(
        depth_arr >= opt_fwl,
        "FREE_WATER",
        np.where(
            depth_arr > tz_info["owc"],
            "CAPILLARY_FRINGE",
            np.where(depth_arr > tz_info["ttz"], "TRANSITION_ZONE", "IRREDUCIBLE_HC"),
        ),
    )

    work_df["SW_SHM"] = sw_shm_full
    work_df["BVW_LOG"] = phit_arr * sw_log_arr
    work_df["BVW_SHM"] = phit_arr * sw_shm_full
    work_df["HAFWL"] = hafwl
    work_df["PC_RES"] = pc_res
    work_df["TRANSITION_ZONE_FLAG"] = tz_flag

    # 7. Global error metrics on reservoir intervals
    sw_obs_res = sw_log_arr[res_mask]
    sw_pred_res = sw_shm_full[res_mask]
    phi_res = phit_arr[res_mask]

    global_rmse = float(np.sqrt(np.mean((sw_obs_res - sw_pred_res) ** 2)))
    global_mae = float(np.mean(np.abs(sw_obs_res - sw_pred_res)))
    global_bvw_rmse = float(
        np.sqrt(np.mean((phi_res * (sw_obs_res - sw_pred_res)) ** 2))
    )
    ss_res = float(np.sum((sw_obs_res - sw_pred_res) ** 2))
    ss_tot = float(np.sum((sw_obs_res - np.mean(sw_obs_res)) ** 2))
    global_r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0

    # 8. Per-well metrics & per-well FWL tilt / compartmentalization diagnostics
    well_rows = []
    for w_name in pd.unique(wells_arr):
        w_mask = res_mask & (wells_arr == w_name)
        if not np.any(w_mask):
            w_mask = (wells_arr == w_name) & np.isfinite(depth_arr) & np.isfinite(sw_log_arr)
        n_w = int(np.sum(w_mask))
        if n_w == 0:
            continue

        w_obs = sw_log_arr[w_mask]
        w_pred = sw_shm_full[w_mask]
        w_phi = phit_arr[w_mask]
        w_perm = perm_arr[w_mask]

        w_rmse = float(np.sqrt(np.mean((w_obs - w_pred) ** 2)))
        w_mae = float(np.mean(np.abs(w_obs - w_pred)))
        w_bvw_rmse = float(np.sqrt(np.mean((w_phi * (w_obs - w_pred)) ** 2)))

        w_tz = analyze_transition_zone(
            fwl=opt_fwl,
            model=model,
            shm_params=active_shm_params,
            phit=w_phi,
            perm=w_perm,
            ift=ift,
            theta=theta,
            gw=gw,
            ghc=ghc,
            unit=unit,
        )

        fwl_well_opt = opt_fwl
        if solve_per_well and n_w >= 3:
            w_weights = np.ones(n_w, dtype=float) / float(n_w)
            w_grid_losses = np.array(
                [
                    _compute_loss(
                        float(f),
                        cand_params=active_shm_params,
                        subset_mask=w_mask,
                        subset_weights=w_weights,
                    )
                    for f in grid_fwls
                ],
                dtype=float,
            )
            w_best_i = int(np.argmin(w_grid_losses))
            w_l = float(grid_fwls[max(0, w_best_i - 2)])
            w_r = float(grid_fwls[min(n_grid - 1, w_best_i + 2)])
            if w_r > w_l:
                w_opt = minimize_scalar(
                    lambda f: _compute_loss(
                        float(f),
                        cand_params=active_shm_params,
                        subset_mask=w_mask,
                        subset_weights=w_weights,
                    ),
                    bounds=(w_l, w_r),
                    method="bounded",
                )
                fwl_well_opt = (
                    float(w_opt.x)
                    if w_opt.fun <= w_grid_losses[w_best_i]
                    else float(grid_fwls[w_best_i])
                )
            else:
                fwl_well_opt = float(grid_fwls[w_best_i])

        well_rows.append(
            {
                well_col: w_name,
                "n_samples": n_w,
                "rmse": w_rmse,
                "mae": w_mae,
                "bvw_rmse": w_bvw_rmse,
                "fwl_global": opt_fwl,
                "fwl_well": fwl_well_opt,
                "fwl_offset_from_global": float(fwl_well_opt - opt_fwl),
                "owc": w_tz["owc"],
                "ttz": w_tz["ttz"],
                "transition_zone_thickness": w_tz["transition_zone_thickness"],
            }
        )

    well_metrics_df = pd.DataFrame(well_rows)
    grid_profile_df = pd.DataFrame(
        {"fwl": grid_fwls, "objective_loss": grid_losses}
    )

    # 9. Per-zone FWL optimization & transition zone contacts
    zone_rows = []
    fwl_by_zone: dict[str, float] = {}
    resolved_zone_col = zone_col if zone_col in work_df.columns else (
        "ZONE" if "ZONE" in work_df.columns else None
    )
    if solve_per_zone and resolved_zone_col is not None:
        zones_raw = work_df[resolved_zone_col].fillna("Unknown").astype(str).to_numpy()
        sw_shm_full = sw_shm_full.copy()
        hafwl = hafwl.copy()
        pc_res = pc_res.copy()
        # Sort unique zones by median depth so zone table is ordered stratigraphically
        unique_zones = [
            z
            for z in pd.unique(zones_raw)
            if z.strip() and z.strip().upper() not in ("NAN", "NONE", "<NA>")
        ]
        unique_zones.sort(
            key=lambda zn: float(
                np.nanmedian(depth_arr[zones_raw == zn])
                if np.any(zones_raw == zn)
                else 0.0
            )
        )
        for z_name in unique_zones:
            z_mask = res_mask & (zones_raw == z_name)
            if int(np.sum(z_mask)) < 3:
                z_mask = (
                    (zones_raw == z_name)
                    & np.isfinite(depth_arr)
                    & np.isfinite(sw_log_arr)
                    & np.isfinite(phit_arr)
                    & (phit_arr > 0)
                    & (~is_non_res)
                )
                if is_leverett:
                    z_mask = z_mask & np.isfinite(perm_arr) & (perm_arr > 0)
            n_z = int(np.sum(z_mask))
            if n_z < 3:
                continue

            z_depths = depth_arr[z_mask]
            z_d_min = float(np.nanmin(z_depths))
            z_d_max = float(np.nanmax(z_depths))
            z_span = max(z_d_max - z_d_min, 25.0)
            z_low = z_d_min - 0.15 * z_span
            z_high = z_d_max + 0.60 * z_span
            if z_high <= z_low:
                z_high = z_low + 50.0

            z_w = weights[z_mask]
            z_w = z_w / np.maximum(np.sum(z_w), 1e-12)
            z_grid = np.linspace(z_low, z_high, 120)
            z_losses = np.array(
                [
                    _compute_loss(
                        float(f),
                        cand_params=active_shm_params,
                        subset_mask=z_mask,
                        subset_weights=z_w,
                    )
                    for f in z_grid
                ],
                dtype=float,
            )
            z_best_i = int(np.argmin(z_losses))
            z_l = float(z_grid[max(0, z_best_i - 2)])
            z_r = float(z_grid[min(len(z_grid) - 1, z_best_i + 2)])
            if z_r > z_l:
                z_opt = minimize_scalar(
                    lambda f: _compute_loss(
                        float(f),
                        cand_params=active_shm_params,
                        subset_mask=z_mask,
                        subset_weights=z_w,
                    ),
                    bounds=(z_l, z_r),
                    method="bounded",
                )
                fwl_zone_opt = (
                    float(z_opt.x)
                    if z_opt.fun <= z_losses[z_best_i]
                    else float(z_grid[z_best_i])
                )
            else:
                fwl_zone_opt = float(z_grid[z_best_i])

            z_phi = phit_arr[z_mask]
            z_perm = perm_arr[z_mask]
            z_tz = analyze_transition_zone(
                fwl=fwl_zone_opt,
                model=model,
                shm_params=active_shm_params,
                phit=z_phi,
                perm=z_perm,
                ift=ift,
                theta=theta,
                gw=gw,
                ghc=ghc,
                unit=unit,
            )

            z_all_mask = zones_raw == z_name
            z_sw_pred_all = np.asarray(
                predict_sw_shm(
                    df=work_df.loc[z_all_mask],
                    fwl=fwl_zone_opt,
                    model=model,
                    shm_params=active_shm_params,
                    swirr=swirr[z_all_mask]
                    if isinstance(swirr, np.ndarray)
                    else swirr,
                    ift=ift,
                    theta=theta,
                    gw=gw,
                    ghc=ghc,
                    depth_col=d_col,
                    phit_col=phit_col,
                    perm_col=perm_col,
                    phie_col=phie_col,
                    rock_flag_col=rock_flag_col,
                    well_col=well_col,
                    non_reservoir_flags=non_reservoir_flags,
                    unit=unit,
                    clip=True,
                ),
                dtype=float,
            )
            # Update SW_SHM, HAFWL, PC_RES, and BVW_SHM for this zone using its zone-specific FWL
            sw_shm_full[z_all_mask] = z_sw_pred_all
            z_hafwl = np.maximum(fwl_zone_opt - depth_arr[z_all_mask], 0.0)
            hafwl[z_all_mask] = z_hafwl
            pc_res[z_all_mask] = z_hafwl * (gw - ghc) * conv

            z_obs = sw_log_arr[z_mask]
            z_pred = sw_shm_full[z_mask]
            z_rmse = float(np.sqrt(np.mean((z_obs - z_pred) ** 2)))
            z_mae = float(np.mean(np.abs(z_obs - z_pred)))
            z_bvw_rmse = float(np.sqrt(np.mean((z_phi * (z_obs - z_pred)) ** 2)))
            z_ss_res = float(np.sum((z_obs - z_pred) ** 2))
            z_ss_tot = float(np.sum((z_obs - np.mean(z_obs)) ** 2))
            z_r2 = float(max(0.0, 1.0 - z_ss_res / z_ss_tot)) if z_ss_tot > 1e-12 else 0.0

            fwl_by_zone[str(z_name)] = round(float(fwl_zone_opt), 2)
            zone_rows.append(
                {
                    "zone": str(z_name),
                    "n_samples": n_z,
                    "d_min": round(z_d_min, 2),
                    "d_max": round(z_d_max, 2),
                    "fwl": round(float(fwl_zone_opt), 2),
                    "owc": round(float(z_tz["owc"]), 2),
                    "ttz": round(float(z_tz["ttz"]), 2),
                    "capillary_entry_height": round(
                        float(z_tz["capillary_entry_height"]), 2
                    ),
                    "transition_zone_thickness": round(
                        float(z_tz["transition_zone_thickness"]), 2
                    ),
                    "rmse": round(z_rmse, 4),
                    "mae": round(z_mae, 4),
                    "bvw_rmse": round(z_bvw_rmse, 4),
                    "r2": round(z_r2, 4),
                }
            )

        if fwl_by_zone:
            work_df["SW_SHM"] = sw_shm_full
            work_df["BVW_SHM"] = phit_arr * sw_shm_full
            work_df["HAFWL"] = hafwl
            work_df["PC_RES"] = pc_res
            sw_obs_res = sw_log_arr[res_mask]
            sw_pred_res = sw_shm_full[res_mask]
            global_rmse = float(np.sqrt(np.mean((sw_obs_res - sw_pred_res) ** 2)))
            global_mae = float(np.mean(np.abs(sw_obs_res - sw_pred_res)))
            global_bvw_rmse = float(
                np.sqrt(np.mean((phi_res * (sw_obs_res - sw_pred_res)) ** 2))
            )
            ss_res = float(np.sum((sw_obs_res - sw_pred_res) ** 2))
            ss_tot = float(np.sum((sw_obs_res - np.mean(sw_obs_res)) ** 2))
            global_r2 = float(max(0.0, 1.0 - ss_res / ss_tot)) if ss_tot > 1e-12 else 0.0

    zone_metrics_df = pd.DataFrame(zone_rows)

    fwl_reconciliation_diff = (
        float(opt_fwl - float(mdt_fwl)) if mdt_fwl is not None else None
    )

    return {
        "fwl": opt_fwl,
        "fwl_by_zone": fwl_by_zone,
        "owc": tz_info["owc"],
        "ttz": tz_info["ttz"],
        "capillary_entry_height": tz_info["capillary_entry_height"],
        "transition_zone_thickness": tz_info["transition_zone_thickness"],
        "rmse": global_rmse,
        "mae": global_mae,
        "r2": global_r2,
        "bvw_rmse": global_bvw_rmse,
        "mdt_fwl": float(mdt_fwl) if mdt_fwl is not None else None,
        "mdt_details": mdt_details,
        "fwl_reconciliation_diff": fwl_reconciliation_diff,
        "shm_params": active_shm_params,
        "well_metrics": well_metrics_df,
        "zone_metrics": zone_metrics_df,
        "grid_profile": grid_profile_df,
        "df": work_df,
    }


def plot_fwl_reconciliation(
    opt_result: dict,
    depth_col: str = "DEPTH",
    sw_col: str = "SWT",
    well_col: str = "WELL_NAME",
    title: str = "Multi-Well Contact & Transition Zone Reconciliation",
):
    """Plot a 3-panel diagnostic figure for Multi-Well FWL Optimization & SHM Reconciliation.

    Panels:
      1. Objective Function Profile (Error vs. Candidate FWL) with optimal FWL, OWC, TTZ, and MDT FWL.
      2. Multi-Well Saturation vs. Depth Profile ($S_{w,log}$ vs. $S_{w,SHM}$) with contact horizons.
      3. $S_{w,log}$ vs. $S_{w,SHM}$ 1:1 Reconciliation Crossplot colored by well.

    Args:
        opt_result (dict): Dictionary returned by `optimize_fwl` (or `ContactOptimizer.result_`).
        depth_col (str, optional): Depth column name. Defaults to `'DEPTH'`.
        sw_col (str, optional): Log water saturation column name. Defaults to `'SWT'`.
        well_col (str, optional): Well column name. Defaults to `'WELL_NAME'`.
        title (str, optional): Figure title.

    Returns:
        matplotlib.figure.Figure: Diagnostic matplotlib Figure.
    """
    df = opt_result["df"]
    grid_df = opt_result["grid_profile"]
    fwl = opt_result["fwl"]
    owc = opt_result["owc"]
    ttz = opt_result["ttz"]
    mdt_fwl = opt_result.get("mdt_fwl")

    d_col = depth_col if depth_col in df.columns else (
        "TVDSS" if "TVDSS" in df.columns else df.columns[0]
    )
    s_col = sw_col if sw_col in df.columns else ("SW" if "SW" in df.columns else "SW_SHM")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    fig.suptitle(title, fontsize=11, fontweight="bold")

    # Panel 1: Objective Loss vs. Candidate FWL
    ax0 = axes[0]
    ax0.plot(grid_df["objective_loss"], grid_df["fwl"], color="navy", lw=1.8, label="Objective Loss")
    ax0.axhline(fwl, color="dodgerblue", ls="-", lw=1.8, label=f"Opt FWL = {fwl:.1f}")
    ax0.axhline(owc, color="forestgreen", ls="--", lw=1.4, label=f"OWC = {owc:.1f}")
    ax0.axhline(ttz, color="darkorange", ls=":", lw=1.4, label=f"TTZ = {ttz:.1f}")
    if mdt_fwl is not None:
        ax0.axhline(mdt_fwl, color="crimson", ls="-.", lw=1.5, label=f"MDT FWL = {mdt_fwl:.1f}")
    ax0.invert_yaxis()
    ax0.set_xlabel("Objective Error (Sw / BVW)")
    ax0.set_ylabel("Depth (TVD / TVDSS)")
    ax0.set_title("FWL Objective Profile")
    ax0.grid(True, ls="--", alpha=0.5)
    ax0.legend(loc="best", fontsize=8)

    # Panel 2: Multi-Well Depth vs. Sw & Sw_SHM
    ax1 = axes[1]
    wells = df[well_col].unique() if well_col in df.columns else ["WELL"]
    cmap = plt.get_cmap("tab10")
    for idx, w in enumerate(wells):
        sub = df[df[well_col] == w].sort_values(d_col) if well_col in df.columns else df.sort_values(d_col)
        color = cmap(idx % 10)
        ax1.scatter(sub[s_col], sub[d_col], s=10, alpha=0.45, color=color, label=f"{w} Sw_log")
        ax1.plot(sub["SW_SHM"], sub[d_col], lw=1.5, color=color, label=f"{w} Sw_SHM")
    ax1.axhline(fwl, color="dodgerblue", ls="-", lw=1.6, label="FWL")
    ax1.axhline(owc, color="forestgreen", ls="--", lw=1.3, label="OWC")
    ax1.axhline(ttz, color="darkorange", ls=":", lw=1.3, label="TTZ")
    ax1.invert_yaxis()
    ax1.set_xlim(0.0, 1.05)
    ax1.set_xlabel("Water Saturation (frac)")
    ax1.set_ylabel("Depth (TVD / TVDSS)")
    ax1.set_title("Multi-Well Transition Zone Profile")
    ax1.grid(True, ls="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=7)

    # Panel 3: 1:1 Reconciliation Crossplot
    ax2 = axes[2]
    for idx, w in enumerate(wells):
        sub = df[df[well_col] == w] if well_col in df.columns else df
        ax2.scatter(sub[s_col], sub["SW_SHM"], s=14, alpha=0.6, color=cmap(idx % 10), label=str(w))
    ax2.plot([0, 1], [0, 1], "k--", lw=1.2, label="1:1 Line")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Resistivity Sw (frac)")
    ax2.set_ylabel("SHM Sw (frac)")
    ax2.set_title(
        f"Sw Reconciliation (RMSE={opt_result['rmse']:.3f}, R²={opt_result['r2']:.2f})"
    )
    ax2.grid(True, ls="--", alpha=0.5)
    ax2.legend(loc="best", fontsize=8)

    fig.tight_layout()
    return fig


class ContactOptimizer:
    """Multi-Well Free Water Level (FWL) & Capillary Transition Zone Optimizer.

    Object-oriented estimator that solves for the field-wide (and per-well) Free Water Level
    by minimizing the discrepancy between resistivity-derived water saturation ($S_w$) and
    capillary pressure Saturation-Height Modeling ($S_{w,SHM}$ via Leverett-$J$, Skelt-Harrison,
    or Cuddy) across multiple wells simultaneously, with optional MDT/RFT pressure reconciliation.
    """

    def __init__(
        self,
        model: str = "leverett_j",
        shm_params=None,
        ift: float = 30.0,
        theta: float = 30.0,
        gw: float = 1.05,
        ghc: float = 0.80,
        swirr=None,
        loss: str = "rmse",
        pore_volume_weighted: bool = False,
        equal_well_weights: bool = False,
        phit_cutoff: float = 0.03,
        perm_cutoff: float = 0.01,
        vshale_cutoff: float = 0.50,
        non_reservoir_flags=None,
        solve_per_well: bool = True,
        optimize_shm_params: bool = False,
        unit: str = "imperial",
    ):
        self.model = model
        self.shm_params = shm_params
        self.ift = ift
        self.theta = theta
        self.gw = gw
        self.ghc = ghc
        self.swirr = swirr
        self.loss = loss
        self.pore_volume_weighted = pore_volume_weighted
        self.equal_well_weights = equal_well_weights
        self.phit_cutoff = phit_cutoff
        self.perm_cutoff = perm_cutoff
        self.vshale_cutoff = vshale_cutoff
        self.non_reservoir_flags = non_reservoir_flags
        self.solve_per_well = solve_per_well
        self.optimize_shm_params = optimize_shm_params
        self.unit = unit

        self.result_: dict = None
        self.fwl_: float = None
        self.owc_: float = None
        self.ttz_: float = None
        self.shm_params_ = None
        self.well_metrics_: pd.DataFrame = None

    def fit(
        self,
        df: pd.DataFrame | dict,
        fwl_bounds: tuple = None,
        mdt_fwl: float = None,
        pressure_df: pd.DataFrame = None,
        mdt_tolerance: float = None,
        mdt_penalty_weight: float = 0.0,
        depth_col: str = "DEPTH",
        sw_col: str = "SWT",
        phit_col: str = "PHIT",
        perm_col: str = "PERM",
        phie_col: str = None,
        rock_flag_col: str = "ROCK_FLAG",
        well_col: str = "WELL_NAME",
        vshale_col: str = "VSHALE",
        saturation_method: str = "archie",
        archie_params: dict = None,
    ) -> "ContactOptimizer":
        """Fit the optimizer across single or multiple wells to solve for FWL and transition zone contacts."""
        self.result_ = optimize_fwl(
            df=df,
            model=self.model,
            shm_params=self.shm_params,
            fwl_bounds=fwl_bounds,
            ift=self.ift,
            theta=self.theta,
            gw=self.gw,
            ghc=self.ghc,
            swirr=self.swirr,
            depth_col=depth_col,
            sw_col=sw_col,
            phit_col=phit_col,
            perm_col=perm_col,
            phie_col=phie_col,
            rock_flag_col=rock_flag_col,
            well_col=well_col,
            vshale_col=vshale_col,
            phit_cutoff=self.phit_cutoff,
            perm_cutoff=self.perm_cutoff,
            vshale_cutoff=self.vshale_cutoff,
            non_reservoir_flags=self.non_reservoir_flags,
            loss=self.loss,
            pore_volume_weighted=self.pore_volume_weighted,
            equal_well_weights=self.equal_well_weights,
            mdt_fwl=mdt_fwl,
            pressure_df=pressure_df,
            mdt_tolerance=mdt_tolerance,
            mdt_penalty_weight=mdt_penalty_weight,
            solve_per_well=self.solve_per_well,
            optimize_shm_params=self.optimize_shm_params,
            saturation_method=saturation_method,
            archie_params=archie_params,
            unit=self.unit,
        )
        self.fwl_ = self.result_["fwl"]
        self.owc_ = self.result_["owc"]
        self.ttz_ = self.result_["ttz"]
        self.shm_params_ = self.result_["shm_params"]
        self.well_metrics_ = self.result_["well_metrics"]
        return self

    def predict(
        self,
        df: pd.DataFrame,
        use_per_well_fwl: bool = False,
        depth_col: str = "DEPTH",
        phit_col: str = "PHIT",
        perm_col: str = "PERM",
        phie_col: str = None,
        rock_flag_col: str = "ROCK_FLAG",
        well_col: str = "WELL_NAME",
    ) -> pd.Series:
        """Predict $S_{w,SHM}$ on a DataFrame using the fitted FWL and SHM parameters."""
        if self.fwl_ is None:
            raise RuntimeError("ContactOptimizer must be fitted with .fit() before calling .predict().")

        fwl_arg = self.fwl_
        if (
            use_per_well_fwl
            and self.well_metrics_ is not None
            and not self.well_metrics_.empty
            and well_col in self.well_metrics_.columns
        ):
            fwl_arg = dict(
                zip(
                    self.well_metrics_[well_col],
                    self.well_metrics_["fwl_well"],
                    strict=False,
                )
            )

        return predict_sw_shm(
            df=df,
            fwl=fwl_arg,
            model=self.model,
            shm_params=self.shm_params_,
            swirr=self.swirr,
            ift=self.ift,
            theta=self.theta,
            gw=self.gw,
            ghc=self.ghc,
            depth_col=depth_col,
            phit_col=phit_col,
            perm_col=perm_col,
            phie_col=phie_col,
            rock_flag_col=rock_flag_col,
            well_col=well_col,
            non_reservoir_flags=self.non_reservoir_flags,
            unit=self.unit,
            clip=True,
        )

    def transform(
        self,
        df: pd.DataFrame,
        use_per_well_fwl: bool = False,
        depth_col: str = "DEPTH",
        phit_col: str = "PHIT",
        perm_col: str = "PERM",
        phie_col: str = None,
        rock_flag_col: str = "ROCK_FLAG",
        well_col: str = "WELL_NAME",
    ) -> pd.DataFrame:
        """Return a copy of `df` augmented with `SW_SHM`, `HAFWL`, `PC_RES`, and `TRANSITION_ZONE_FLAG`."""
        if self.fwl_ is None:
            raise RuntimeError(
                "ContactOptimizer must be fitted with .fit() before calling .transform()."
            )
        out_df = df.copy()
        out_df["SW_SHM"] = self.predict(
            out_df,
            use_per_well_fwl=use_per_well_fwl,
            depth_col=depth_col,
            phit_col=phit_col,
            perm_col=perm_col,
            phie_col=phie_col,
            rock_flag_col=rock_flag_col,
            well_col=well_col,
        )
        d_col = depth_col if depth_col in out_df.columns else (
            "TVDSS" if "TVDSS" in out_df.columns else out_df.columns[0]
        )
        z = np.asarray(out_df[d_col], dtype=float)
        conv = _gravity_conversion_factor(self.unit)
        hafwl = np.maximum(self.fwl_ - z, 0.0)
        out_df["HAFWL"] = hafwl
        out_df["PC_RES"] = hafwl * (self.gw - self.ghc) * conv
        out_df["TRANSITION_ZONE_FLAG"] = np.where(
            z >= self.fwl_,
            "FREE_WATER",
            np.where(
                z > self.owc_,
                "CAPILLARY_FRINGE",
                np.where(z > self.ttz_, "TRANSITION_ZONE", "IRREDUCIBLE_HC"),
            ),
        )
        return out_df

    def summary(self) -> dict:
        """Return a dictionary of fitted contacts, transition zone thickness, and error metrics."""
        if self.result_ is None:
            raise RuntimeError("ContactOptimizer has not been fitted yet.")
        return {k: v for k, v in self.result_.items() if k not in ("df", "grid_profile")}

    def plot(self, **kwargs):
        """Plot the 3-panel Multi-Well FWL & SHM Reconciliation diagnostic figure."""
        if self.result_ is None:
            raise RuntimeError("ContactOptimizer has not been fitted yet.")
        return plot_fwl_reconciliation(self.result_, **kwargs)



def swirr_rqi(
    poro, perm, a: float = 0.25, b: float = 0.50, swirr_min: float = 0.02, swirr_max: float = 0.90
):
    """Estimate irreducible water saturation ($S_{wirr}$) from Reservoir Quality Index (RQI) ratio.

    $$S_{wirr} = \\text{clip}\\left(a \\cdot \\left(\\sqrt{\\frac{k}{\\phi}}\\right)^{-b}, S_{wirr,\\min}, S_{wirr,\\max}\\right)$$

    Args:
        poro (np.ndarray or float): Porosity in fraction (v/v).
        perm (np.ndarray or float): Permeability in mD.
        a (float, optional): Power-law multiplier coefficient. Defaults to 0.25.
        b (float, optional): Power-law exponent. Defaults to 0.50.
        swirr_min (float, optional): Lower physical clamp bound. Defaults to 0.02.
        swirr_max (float, optional): Upper physical clamp bound. Defaults to 0.90.

    Returns:
        np.ndarray or float: Modeled irreducible water saturation (fraction).
    """
    phi_arr = np.asarray(poro, dtype=float)
    k_arr = np.asarray(perm, dtype=float)
    valid = np.isfinite(phi_arr) & np.isfinite(k_arr) & (phi_arr > 0) & (k_arr > 0)
    safe_phi = np.where(valid, np.maximum(phi_arr, 1e-4), 0.1)
    safe_k = np.where(valid, np.maximum(k_arr, 1e-4), 1.0)
    rqi_ratio = np.sqrt(safe_k / safe_phi)
    raw = float(a) * np.power(np.maximum(rqi_ratio, 1e-6), -float(b))
    clipped = np.clip(raw, float(swirr_min), float(swirr_max))
    res = np.where(valid, clipped, np.nan)
    return float(res) if np.ndim(poro) == 0 and np.ndim(perm) == 0 else res


def swirr_semilog_k(
    perm, a: float = 0.35, b: float = 0.10, swirr_min: float = 0.02, swirr_max: float = 0.90
):
    """Estimate irreducible water saturation ($S_{wirr}$) using a semi-log permeability correlation.

    $$S_{wirr} = \\text{clip}\\left(a - b \\cdot \\log_{10}(k), S_{wirr,\\min}, S_{wirr,\\max}\\right)$$

    Args:
        perm (np.ndarray or float): Permeability in mD.
        a (float, optional): Intercept at $k = 1\\text{ mD}$. Defaults to 0.35.
        b (float, optional): Slope per log decade of permeability. Defaults to 0.10.
        swirr_min (float, optional): Lower physical clamp bound. Defaults to 0.02.
        swirr_max (float, optional): Upper physical clamp bound. Defaults to 0.90.

    Returns:
        np.ndarray or float: Modeled irreducible water saturation (fraction).
    """
    k_arr = np.asarray(perm, dtype=float)
    valid = np.isfinite(k_arr) & (k_arr > 0)
    safe_k = np.where(valid, np.maximum(k_arr, 1e-4), 1.0)
    raw = float(a) - float(b) * np.log10(safe_k)
    clipped = np.clip(raw, float(swirr_min), float(swirr_max))
    res = np.where(valid, clipped, np.nan)
    return float(res) if np.ndim(perm) == 0 else res


def swirr_timur(
    poro,
    perm,
    a: float = 0.15,
    b: float = 1.50,
    c: float = 0.50,
    swirr_min: float = 0.02,
    swirr_max: float = 0.90,
):
    """Estimate irreducible water saturation ($S_{wirr}$) using a Generalized Timur-Coates correlation.

    $$S_{wirr} = \\text{clip}\\left(a \\cdot \\frac{\\phi^b}{k^c}, S_{wirr,\\min}, S_{wirr,\\max}\\right)$$

    Args:
        poro (np.ndarray or float): Porosity in fraction (v/v).
        perm (np.ndarray or float): Permeability in mD.
        a (float, optional): Multiplier coefficient. Defaults to 0.15.
        b (float, optional): Porosity exponent. Defaults to 1.50.
        c (float, optional): Permeability exponent. Defaults to 0.50.
        swirr_min (float, optional): Lower physical clamp bound. Defaults to 0.02.
        swirr_max (float, optional): Upper physical clamp bound. Defaults to 0.90.

    Returns:
        np.ndarray or float: Modeled irreducible water saturation (fraction).
    """
    phi_arr = np.asarray(poro, dtype=float)
    k_arr = np.asarray(perm, dtype=float)
    valid = np.isfinite(phi_arr) & np.isfinite(k_arr) & (phi_arr > 0) & (k_arr > 0)
    safe_phi = np.where(valid, np.maximum(phi_arr, 1e-4), 0.1)
    safe_k = np.where(valid, np.maximum(k_arr, 1e-4), 1.0)
    raw = float(a) * (np.power(safe_phi, float(b)) / np.power(safe_k, float(c)))
    clipped = np.clip(raw, float(swirr_min), float(swirr_max))
    res = np.where(valid, clipped, np.nan)
    return float(res) if np.ndim(poro) == 0 and np.ndim(perm) == 0 else res


def fit_swirr_correlation(
    poro,
    perm,
    swirr,
    equation_type: str = "rqi_power",
    swirr_min: float = 0.02,
    swirr_max: float = 0.90,
) -> dict:
    """Fit empirical $S_{wirr}$ correlation parameters (`a`, `b`, `c`) from plug porosity, permeability, and $S_{wirr}$.

    Args:
        poro (array-like): Core porosity in fraction (v/v).
        perm (array-like): Core permeability in mD.
        swirr (array-like): Extracted reference irreducible water saturation (fraction).
        equation_type (str, optional): One of `'rqi_power'`, `'semilog_k'`, or `'timur'`. Defaults to `'rqi_power'`.
        swirr_min (float, optional): Lower physical clamp bound. Defaults to 0.02.
        swirr_max (float, optional): Upper physical clamp bound. Defaults to 0.90.

    Returns:
        dict: Fitted parameters `{'a', 'b', 'c', 'r2', 'rmse', 'count'}`.
    """
    phi_arr = np.asarray(poro, dtype=float).ravel()
    k_arr = np.asarray(perm, dtype=float).ravel()
    sw_arr = np.asarray(swirr, dtype=float).ravel()

    eq = str(equation_type or "rqi_power").strip().lower()
    if eq == "semilog_k":
        mask = np.isfinite(k_arr) & np.isfinite(sw_arr) & (k_arr > 0) & (sw_arr > 0)
    else:
        mask = (
            np.isfinite(phi_arr)
            & np.isfinite(k_arr)
            & np.isfinite(sw_arr)
            & (phi_arr > 0)
            & (k_arr > 0)
            & (sw_arr > 0)
        )

    n = int(np.sum(mask))
    if n == 0:
        return {
            "a": 0.30 if eq != "timur" else 0.80,
            "b": 0.35 if eq == "rqi_power" else (0.08 if eq == "semilog_k" else 1.20),
            "c": 0.45 if eq == "timur" else None,
            "r2": 0.0,
            "rmse": 0.0,
            "count": 0,
        }

    phi_v = np.maximum(phi_arr[mask], 1e-4)
    k_v = np.maximum(k_arr[mask], 1e-4)
    sw_v = np.clip(sw_arr[mask], 1e-3, 0.999)

    a_val, b_val, c_val = 0.30, 0.35, None
    if n == 1:
        if eq == "semilog_k":
            b_val = 0.08
            a_val = float(np.clip(sw_v[0] + b_val * np.log10(k_v[0]), 0.02, 1.5))
        elif eq == "timur":
            b_val, c_val = 1.20, 0.40
            a_val = float(
                np.clip(sw_v[0] * (k_v[0] ** c_val) / (phi_v[0] ** b_val), 0.01, 50.0)
            )
        else:
            b_val = 0.35
            rqi_0 = np.sqrt(k_v[0] / phi_v[0])
            a_val = float(np.clip(sw_v[0] * (rqi_0**b_val), 0.01, 10.0))
    else:
        if eq == "semilog_k":
            x_log = np.log10(k_v)
            if np.std(x_log) > 1e-9:
                slope, intercept = np.polyfit(x_log, sw_v, 1)
                a_val = float(np.clip(intercept, 0.01, 2.0))
                b_val = float(np.clip(-slope, 0.001, 1.0))
            else:
                a_val = float(np.mean(sw_v))
                b_val = 0.05
        elif eq == "timur":
            ln_sw = np.log(sw_v)
            ln_phi = np.log(phi_v)
            ln_k = np.log(k_v)
            if n >= 3 and np.std(ln_phi) > 1e-6 and np.std(ln_k) > 1e-6:
                X = np.column_stack([np.ones(n), ln_phi, -ln_k])
                coeffs, _, _, _ = np.linalg.lstsq(X, ln_sw, rcond=None)
                a_val = float(np.clip(np.exp(coeffs[0]), 1e-4, 100.0))
                b_val = float(np.clip(coeffs[1], -2.0, 4.0))
                c_val = float(np.clip(coeffs[2], 0.01, 2.0))
            else:
                b_val = 1.0
                y_adj = ln_sw - b_val * ln_phi
                if np.std(ln_k) > 1e-6:
                    slope, intercept = np.polyfit(ln_k, y_adj, 1)
                    a_val = float(np.clip(np.exp(intercept), 1e-4, 100.0))
                    c_val = float(np.clip(-slope, 0.01, 2.0))
                else:
                    c_val = 0.40
                    a_val = float(
                        np.clip(
                            np.mean(sw_v * (k_v**c_val) / (phi_v**b_val)), 1e-4, 100.0
                        )
                    )
        else:
            rqi_ratio = np.sqrt(k_v / phi_v)
            ln_x = np.log(np.maximum(rqi_ratio, 1e-6))
            ln_y = np.log(sw_v)
            if np.std(ln_x) > 1e-9:
                slope, intercept = np.polyfit(ln_x, ln_y, 1)
                a_val = float(np.clip(np.exp(intercept), 0.005, 10.0))
                b_val = float(np.clip(-slope, 0.01, 3.0))
            else:
                a_val = float(np.mean(sw_v))
                b_val = 0.25

    if eq == "semilog_k":
        sw_pred = swirr_semilog_k(k_v, a_val, b_val, swirr_min, swirr_max)
    elif eq == "timur":
        sw_pred = swirr_timur(
            phi_v, k_v, a_val, b_val, c_val if c_val is not None else 0.5, swirr_min, swirr_max
        )
    else:
        sw_pred = swirr_rqi(phi_v, k_v, a_val, b_val, swirr_min, swirr_max)

    sw_pred = np.asarray(sw_pred, dtype=float)
    residuals = sw_v - sw_pred
    rmse = float(np.sqrt(np.mean(residuals**2))) if n > 0 else 0.0
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((sw_v - np.mean(sw_v)) ** 2))
    if ss_tot > 1e-12:
        r2 = float(max(0.0, min(1.0, 1.0 - ss_res / ss_tot)))
    else:
        r2 = 1.0 if rmse < 1e-4 else 0.0

    return {
        "a": round(float(a_val), 4),
        "b": round(float(b_val), 4),
        "c": round(float(c_val), 4) if c_val is not None else None,
        "r2": round(float(r2), 4),
        "rmse": round(float(rmse), 4),
        "count": n,
    }
