import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from quick_pp.utils import min_max_line
from quick_pp.utils import power_law_func as func

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
    return ((a / (phit ** m)) * (rw / rt)) ** (1 / n)


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

    return swt


def dual_water_saturation(rt, rw, phit, a, m, n, swb, rwb):
    """Estimate water saturation based on dual water model, an extension from Waxman-Smits.
    TODO: Estimate swb and rwb if not provided

    Args:
        rt (float): True resistivity or deep resistivity log.
        rw (float): Formation water resistivity.
        phit (float): Total porosity
        a (float): Cementation exponent.
        m (float): Saturation exponent.
        n (float): Porosity exponent.
        swb (float): Bound water saturation.
        rwb (float): Bound water resistivity.

    Returns:
        float: Water saturation.

    """
    # Initial guess
    swt = 1
    swt_i = 0
    for i in range(50):
        fx = phit**m * swt**n / a * (1 / rw * (swb / swt) * (1 / rwb - 1 / rw)) - 1 / rt
        delta_sat = abs(swt - swt_i) / 2
        swt_i = swt
        swt = np.where(fx < 0, swt + delta_sat, swt - delta_sat)

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
    return ((1 / rt)**(1 / 2) / ((vsh**(1 - 0.5 * vsh) / rsh**(1 / 2)) + (phie**m / (a * rw))**(1 / 2)))**(2 / n)


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
    shale_factor = vsh / rsh
    return (a * rw / (2 * phit**m)) * ((shale_factor**2 + (4 * phit**m / (a * rw * rt)))**(1 / 2) - shale_factor)


def modified_simandoux_saturation():
    """ TODO: Estimate water saturation based on modified Simandoux's model.
    """
    pass


def estimate_temperature_gradient(tvd, unit='metric'):
    """Estimate formation temperature based on gradient of 25 degC/km or 15 degF/1000ft.

    Args:
        tvd (float): True vertical depth in m.

    Returns:
        float: Formation temperature in degC.
    """
    assert unit in ['metric', 'imperial'], "Please choose from 'metric' or 'imperial' units."
    return 32 + 25 * tvd / 1000 if unit == 'metric' else 90 + 15 * tvd / 1000


def estimate_b_waxman_smits(T, rw):
    """Estimate B (conductance parameter) for Waxman-Smits model based on Juhasz 1981.

    Args:
        T (float): Temperature in degC.
        rw (float): Water resistivity in ohm.m.

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
    rw = pd.Series(phit**m * rt / a)
    _, rw = min_max_line(rw, alpha=0.2)
    return rw


def estimate_rw_waxman_smits(phit, rt, a=1, m=2, B=None, Qv=None):
    """Estimate water saturation based on Archie's equation.

    Args:
        phit (float): Total porosity.
        rt (float): True resistivity.

    Returns:
        float: Formation water resistivity.
    """
    if B is None:
        B = 2
    if Qv is None:
        Qv = 0.3

    rw = pd.Series(1 / ((a / (phit**m * rt)) - (B * Qv)))
    _, rw = min_max_line(rw, alpha=0.2)
    return rw


def estimate_rt_water_trend(rt, alpha=0.3):
    """Estimate trend RT of formation water based.

    Args:
        rt (float): True resistivity.
        RES_FLAG (int): Reservoir flag.

    Returns:
        float: Formation water resistivity.
    """
    rt = np.log(rt)
    rt = np.where(rt <= 0, 1e-3, rt)
    min_rt, _ = min_max_line(rt, alpha)
    return np.exp(min_rt)


def estimate_rw_from_shale_trend(rt, phit, m=1.3, alpha=0.1):
    """Estimate Rw from shale trend.

    Args:
        rt (float): True resistivity.
        phit (float): Total porosity.
        m (float): Shale cementation or shape factor. Defaults to 1.3.
        alpha (float): Alpha value for percentile calculation. Defaults to 0.1.

    Returns:
        float: Formation water resistivity.
    """
    min_rt = estimate_rt_water_trend(rt, alpha=alpha)
    min_phit, _ = min_max_line(phit, alpha=alpha)
    return min_phit ** m * np.exp(min_rt)


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
        vclb (float): Volume of clay bound water in fraction.
        phit (float): Total porosity in fraction.
        water_salinity (float): Water salinity in ppm or meq/cc.

    Returns:
        float: Qv in meq/cc
    """
    return (vclb / phit) / (0.084 * water_salinity**-0.5 + 0.22)


def estimate_qv_lavers(phit, a=3.05e-4, b=3.49):
    """Based on Lavers, 1975.

    Args:
        phit (float): Total porosity in fraction.
        a (float): Constant, defaults to 3.05e-4.
        b (float): Constant, defaults to 3.49.

    Returns:
        float: Qv in meq/cc.
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


def swirr_xplot(swt, phit, c=None, q=None, label='', log_log=False):
    """Plot SWT vs PHIT for sand intervales only and estimate Swirr from X-plot.

    Args:
        swt (float): Water saturation.
        phit (float): Total porosity.
        c (float): Constant.
        q (float): Constant.
    """
    sc = plt.scatter(swt, phit, marker='s', label=label)
    if c and q:
        line_color = sc.get_facecolors()[0]
        line_color[-1] = 0.5
        cphit = np.geomspace(0.1, 1.0, 30)
        plt.plot(cphit, func(cphit, c, q), color=line_color, linestyle='dashed')
    plt.xlabel('SWT (frac)')
    plt.ylabel('PHIT (frac)')
    plt.ylim(0.01, .5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    if log_log:
        plt.xscale('log')
        plt.yscale('log')


def pickett_plot(rt, phit, m=-2, min_rw=0.1, shift=.2):
    """Generate Pickett plot which is used to plot phit and rt at water bearing interval to determine;
        m = The slope of best-fit line crossing the cleanest sand.
        rw = Formation water resistivity. The intercept of the best-fit line at rt when phit = 100%.

    Args:
        rt (float): True resistivity or deep resistivity log.
        phit (float): Total porosity.

    Returns:
        matplotlib.pyplot.Figure: Picket plot.
    """
    m = m if m < 0 else -m
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_title('Pickett Plot')
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
    fig.tight_layout()

    return fig


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

    return fig
