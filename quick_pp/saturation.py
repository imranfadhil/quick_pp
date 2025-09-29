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
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.

    Returns:
        float: Water saturation.

    """
    logger.debug(f"Calculating Archie saturation with a={a}, m={m}, n={n}")
    sw = ((a / (phit ** m)) * (rw / rt)) ** (1 / n)
    logger.debug(f"Archie saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def waxman_smits_saturation(rt, rw, phit, Qv=None, B=None, m=2, n=2):
    """Estimate water saturation based on Waxman-Smits model for dispersed clay mineral.
    Based on Ausburn, Brian E., and Robert Freedman. "The Waxman-smits Equation For Shaly Sands:
        i. Simple Methods Of Solution
        ii. Error Analysis." The Log Analyst 26 (1985)

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
    return swt


def normalized_waxman_smits_saturation(rt, rw, phit, vshale, phit_shale, rt_shale, m=2, n=2):
    """Estimate water saturation based on Waxman-Smits model for dispersed clay mineral.
    Based on Ausburn, Brian E., and Robert Freedman. "The Waxman-smits Equation For Shaly Sands:
        i. Simple Methods Of Solution
        ii. Error Analysis." The Log Analyst 26 (1985)

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.

        m (float): Saturation factor.
        n (float): Cementartion factor.

    Returns:
        float: Water saturation.
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
    return swt


def dual_water_saturation(rt, rw, phit, a, m, n, swb, rwb):
    """Estimate water saturation based on dual water model, an extension from Waxman-Smits by Clavier, Coates and
    Dumanoir (1977).

    Args:
        rt (float): True resistivity or deep resistivity log in ohm.m.
        rw (float): Formation water resistivity in ohm.m.
        phit (float): Total porosity in fraction.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.
        swb (float): Bound water saturation in fraction.
        rwb (float): Bound water resistivity in ohm.m.

    Returns:
        float: Water saturation.

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
    return swt


def indonesian_saturation(rt, rw, phie, vsh, rsh, a, m, n):
    """Estimate water saturation based on Indonesian model which may work well with fresh formation water.
    Based on Poupon-Leveaux 1971.

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phie (float): Effective porosity.
        vsh (float): Volume of shale.
        rsh (float): Resistivity of shale.
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.

    Returns:
        float: Water saturation.

    """
    logger.debug("Calculating Indonesian saturation")
    sw = ((1 / rt)**(1 / 2) / ((vsh**(1 - 0.5 * vsh) / rsh**(1 / 2)) + (phie**m / (a * rw))**(1 / 2)))**(2 / n)
    logger.debug(f"Indonesian saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def simandoux_saturation(rt, rw, phit, vsh, rsh, a, m):
    """Estimate water saturation based on Simandoux's model.

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.
        vsh (float): Volume of shale.
        rsh (float): Resistivity of shale.
        a (float): Cementation exponent.
        m (float): Saturation exponent.

    Returns:
        float: Water saturation.

    """
    logger.debug("Calculating Simandoux saturation")
    shale_factor = vsh / rsh
    sw = (a * rw / (2 * phit**m)) * ((shale_factor**2 + (4 * phit**m / (a * rw * rt)))**(1 / 2) - shale_factor)
    logger.debug(f"Simandoux saturation range: {sw.min():.3f} - {sw.max():.3f}")
    return sw


def connectivity_saturation(rt, rw, phit, mu=2, chi_w=0):
    """Estimate water saturation based on connectivity model Montaron 2009
    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.
        mu (float): Conductivity exponent, ranges from 1.6 to 2. Defaults to 2.
        chi_w (float): Water connectivity correction index, ranges from -0.02 to 0.02. Defaults to 0.

    Returns:
        float: Water saturation.

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

    Returns:
        float: Formation temperature in degC.
    """
    logger.debug(f"Estimating temperature gradient with unit: {unit}")
    assert unit in ['metric', 'imperial'], "Please choose from 'metric' or 'imperial' units."
    temp = 32 + 25 * tvd / 1000 if unit == 'metric' else 90 + 15 * tvd / 1000
    logger.debug(f"Temperature range: {temp.min():.1f} - {temp.max():.1f} °C")
    return temp


def estimate_b_waxman_smits(T, rw):
    """Estimate B (conductance parameter) for Waxman-Smits model based on Juhasz 1981.

    Args:
        T (float): Temperature in degC.
        rw (float): Formation water resistivity in ohm.m.

    Returns:
        float: B parameter.
    """
    logger.debug("Estimating B parameter for Waxman-Smits (Juhasz 1981)")
    B = (-1.28 + 0.225 * T - 0.0004059 * T**2) / (1 + (0.045 * T - 0.27) * rw**1.23)
    logger.debug(f"B parameter range: {B.min():.3f} - {B.max():.3f}")
    return B


def estimate_rw_temperature_salinity(temperature_gradient, water_salinity):
    """Estimate formation water resistivity based on temperature gradient and water salinity.

    Args:
        temperature_gradient (float): Temperature gradient in degC/meter
        water_salinity (float): Water salinity in ppm

    Returns:
        float: Formation water resistivity.
    """
    logger.debug("Estimating Rw from temperature and salinity")
    rw = (400000 / temperature_gradient / water_salinity)**.88
    logger.debug(f"Formation water resistivity range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_surface(temperature_gradient, rw_surface, temp_surface=20):
    """Estimate formation water resistivity based on surface resistivity and temperature.
    This uses Arps' formula to adjust resistivity for temperature.

    Args:
        temperature_gradient (float): Formation temperature in degC.
        rw_surface (float): Water resistivity at surface temperature in ohm.m.
        temp_surface (float, optional): Surface temperature in degC. Defaults to 20.

    Returns:
        float: Formation water resistivity in ohm.m.
    """
    logger.debug("Estimating Rw from surface temperature and resistivity using Arps' formula")
    rw = rw_surface * (temp_surface + 21.5) / (temperature_gradient + 21.5)
    logger.debug(f"Formation water resistivity range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_archie(phit, rt, a=1, m=2):
    """Estimate water saturation based on Archie's equation.

    Args:
        phit (float): Total porosity.
        rt (float): True resistivity.

    Returns:
        float: Formation water resistivity.
    """
    logger.debug("Estimating Rw using Archie's equation")
    rw = pd.Series(phit**m * rt / a)
    _, rw = min_max_line(rw, alpha=0.2)
    logger.debug(f"Estimated Rw range: {rw.min():.3f} - {rw.max():.3f} ohm.m")
    return rw


def estimate_rw_waxman_smits(phit, rt, a=1, m=2, B=None, Qv=None):
    """Estimate water saturation based on Archie's equation.

    Args:
        phit (float): Total porosity.
        rt (float): True resistivity.

    Returns:
        float: Formation water resistivity.
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
    """Estimate Rw from shale trend.

    Args:
        rt (float): True resistivity.
        phit (float): Total porosity.
        m (float): Shale cementation or shape factor. Defaults to 1.3.
        alpha (float): Alpha value for percentile calculation. Defaults to 0.1.

    Returns:
        float: Formation water resistivity.
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
    """Estimate resistivity of shale.

    Args:
        rt (float): True resistivity.
        vshale (float): Volume of shale.

    Returns:
        float: Resistivity of shale.
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
    """Estimate Qv, cation exchange capacity per unit total pore volume (meq/cm3).

    Args:
        vcld (float): Volume of dry clay in fraction
        phit (float): Total porosity in fraction
        rho_clay (float): Bulk density of clay in g/cc
        cec_clay (float): Cation exchange capacity of clay in meq/g

    Returns:
        float: Qv in meq/cm3.
    """
    logger.debug(f"Estimating Qv with clay density={rho_clay} g/cm³, CEC={cec_clay} meq/g")
    qv = vcld * rho_clay * cec_clay / phit
    logger.debug(f"Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cm³")
    return qv


def estimate_qvn(vclay, phit, phit_clay):
    """Estimate normalized Qv

    Args:
        vclay (float): Volume of clay in fraction
        phit (float): Total porosity in fraction
        phit_clay (float): Total porosity of clay in fraction

    Returns:
        float: Normalized Qv.
    """
    return vclay * phit_clay / phit


def estimate_qv_ward(rt, phit, B, rw, m):
    """Estimate Qv based on B. Ward (SIPM), 1973

    Args:
        rt (float): True resistivity in ohm.m from a water-bearing zone.
        phit (float): Total porosity in fraction.
        B (float): Conductance parameter for Waxman-Smits.
        rw (float): Formation water resistivity in ohm.m.
        m (float): Cementation exponent.

    Returns:
        float: Qv in meq/cm3.
    """
    return 1 / B * ((1 / (rt * phit**m)) - (1 / rw))


def estimate_qv_hill(vclb, phit, water_salinity=10000):
    """Estimating Qv based on Hill et. al, 1979.

    Args:
        vclb (float): Volume of clay bound water in fraction.
        phit (float): Total porosity in fraction.
        water_salinity (float): Water salinity in ppm or meq/cc.

    Returns:
        float: Qv in meq/cc
    """
    logger.debug(f"Estimating Qv using Hill method with water salinity={water_salinity}")
    qv = (vclb / phit) / (0.084 * water_salinity**-0.5 + 0.22)
    logger.debug(f"Hill Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cc")
    return qv


def estimate_qv_lavers(phit, a=3.05e-4, b=3.49):
    """Based on Lavers, 1975.

    Args:
        phit (float): Total porosity in fraction.
        a (float): Constant, defaults to 3.05e-4.
        b (float): Constant, defaults to 3.49.

    Returns:
        float: Qv in meq/cc.
    """
    logger.debug(f"Estimating Qv using Lavers method with a={a}, b={b}")
    qv = a * phit**-b
    logger.debug(f"Lavers Qv range: {qv.min():.3f} - {qv.max():.3f} meq/cc")
    return qv


def estimate_bqv(phit, max_phit_clean_sand, C):
    """Estimate BQv (Bulk volume of clay bound water) based on Juhasz/ Rackley method.

    Args:
        phit (float): Total porosity.
        max_phit_clean_sand (float): Maximum porosity of clean sand.
        C (float): Constant depending on clay type.

    Returns:
        float: Bulk volume of clay bound water.
    """
    return (max_phit_clean_sand - phit) / (C * phit)


def estimate_m_archie(rt, rw, phit):
    """Estimate apparent m (saturation factor) based on Archie's equation in water bearing zone.

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity.

    Returns:
        float: Apparent m parameter.
    """
    logger.debug("Estimating apparent m using Archie's equation")
    m = np.log(rw / rt) / np.log(phit)
    logger.debug(f"Apparent m range: {m.min():.3f} - {m.max():.3f}")
    return m


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
    logger.debug("Estimating apparent m using Indonesian model")
    m = (2 / np.log(phie)) * np.log(rw**0.5 * ((1 / rt)**0.5 - (vsh**(1 - 0.5 * vsh) / rsh**0.5)))
    logger.debug(f"Indonesian apparent m range: {m.min():.3f} - {m.max():.3f}")
    return m


def qv_phit_xplot(phit, qv):
    """Generate BQV plot (Cwa vs 1/PHIT) and fit a best straight line."""
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


def cwa_qvn_xplot(rt, phit, qvn, m=2.0, rw=.2, B=None, C=None, slope=250):
    """Generate Cwa vs Qvn plot."""
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
    """Plot SWT vs PHIT for sand intervales only and estimate Swirr from cross plot.
    Based on Buckles, 1965.

    Args:
        swt (float): Water saturation.
        phit (float): Total porosity.
        c (float): Constant. Defaults to 0.125.
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
    """Generate RT vs PHIT plot.

    This plot is useful for visualizing the relationship between true resistivity
    and total porosity, often used for identifying water-bearing zones and estimating
    formation water resistivity (Rw) and cementation exponent (m).

    Args:
        rt (float): True resistivity or deep resistivity log (ohm.m).
        phit (float): Total porosity (fraction).
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
    """Generate Pickett plot which is used to plot phit and rt at water bearing interval to determine;
        m = The slope of best-fit line crossing the cleanest sand.
        rw = Formation water resistivity. The intercept of the best-fit line at rt when phit = 100%.

    Args:
        rt (float): True resistivity or deep resistivity log.
        phit (float): Total porosity.

    Returns:
        matplotlib.pyplot.Figure: Picket plot.
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
    """Generate resistivity index plot to estimate saturation exponent. The inputs are from lab measurements.
    - Assumed a = 1.
    - Trend line must cross (1, 1) point.
    - The slope of the trend line is the saturation exponent.

    Args:
        sw (float): Water saturation.
        rt (float): True resistivity or deep resistivity log.
        ro (float): Resistivity of 100% water saturated rock.

    Returns:
        matplotlib.pyplot.Figure: RI plot.
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
    """Generate formation factor plot to estimate cementation exponent. The inputs are from lab measurements.
    - Assumed a = 1.
    - Trend line must cross (1, 1) point.
    - The slope of the trend line is the cementation exponent.

    Args:
        phit (float): Total porosity.
        ro (float): Resistivity of 100% water saturated rock.
        rw (float): Formation water resistivity.

    Returns:
        matplotlib.pyplot.Figure: FF plot.
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
