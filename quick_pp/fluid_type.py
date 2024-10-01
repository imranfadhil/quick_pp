import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from quick_pp.utils import straight_line_func as func

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)


def fit_pressure_gradient(tvdss, formation_pressure):
    """Fit pressure gradient from formation pressure and true vertical depth subsea.

    Args:
        tvdss (float): True vertical depth subsea.
        formation_pressure (float): Formation pressure.

    Returns:
        tuple: Gradient and intercept of the best-fit line.
    """
    popt, _ = curve_fit(func, tvdss, formation_pressure)
    return round(popt[0], 3), round(popt[1], 3)


def fluid_contact_plot(tvdss, formation_pressue, m, c, fluid_type: str = 'WATER',
                       ylim: tuple = (5000, 3000), xlim: tuple = (1000, 5000),
                       goc: float = 0, owc: float = 0, gwc: float = 0):
    """Generate fluid contact plot which is used to determine the fluid contact in a well.

    Args:
        tvdss (float): True vertical depth subsea.
        formation_pressure (float): Formation pressure.

    Returns:
        matplotlib.pyplot.Figure: Fluid contact plot.
    """
    color_dict = dict(
        OIL=('green', 'o'),
        GAS=('red', 'x'),
        WATER=('blue', '^'),
        HYDROSTATIC=('brown', 's')
    )
    marker = color_dict.get(fluid_type.upper(), ('black', 'o'))
    label = f'{fluid_type} gradient: {round(m, 3)} psi/ft'
    sc = plt.scatter(formation_pressue, tvdss, label=label, c=marker[0], marker=marker[1])

    line_color = sc.get_facecolors()[0]
    line_color[-1] = 0.5
    tvd_pts = np.linspace(ylim[0] - 30, ylim[1] + 30, 30)
    plt.plot(func(tvd_pts, m, c), tvd_pts, color=line_color)

    plt.ylim(ylim)
    plt.ylabel('TVDSS (ft)')
    plt.xlim(xlim)
    plt.xlabel('Formation Pressure (psi)')
    plt.legend()
    plt.title('Fluid Contact Plot')


def gas_composition_analysis(c1, c2, c3, ic4, nc4, ic5, nc5):
    """Analyze gas composition based on the concentration of each component (Haworth 1985).

    Args:
        c1 (float): Concentration of methane in ppm.
        c2 (float): Concentration of ethane in ppm.
        c3 (float): Concentration of propane in ppm.
        ic4 (float): Concentration of isobutane in ppm.
        nc4 (float): Concentration of normal butane in ppm.
        ic5 (float): Concentration of isopentane in ppm.
        nc5 (float): Concentration of normal pentane in ppm.

    Returns:
        pd.DataFrame: Gas composition analysis.
    """
    total = c1 + c2 + c3 + ic4 + nc4 + ic5 + nc5
    # Gas Quality Ration
    gqr = total / (c1 + 2 * c2 + 3 * c3 + 4 * (ic4 + nc4) + 5 * (ic5 + nc5))
    # Wetness ratio
    wh = (c2 + c3 + ic4 + nc4 + ic5 + nc5) / total * 100
    # Balance ration
    bal = (c1 + c2) / (c3 + ic4 + nc4 + ic5 + nc5)
    # Character ratio
    ch = (ic4 + nc4 + ic5 + nc5) / c3

    hc_type_dict = {
        0: 'Non Representative Sample',
        1: 'Non Productive Dry Gas',
        2: 'Potentially Productive Gas',
        3: 'Potentially Productive High GOR Oil',
        4: 'Potentially Productive Condensate',
        5: 'Potentially Productive Oil',
    }
    hc_type = np.zeros(1 if isinstance(c1, int) else len(c1))
    hc_type = np.where((wh < 5) & (bal > 100), 1, hc_type)
    hc_type = np.where((wh > 5) & (wh < 17.5) & (wh < bal) & (bal < 100), 2, hc_type)
    hc_type = np.where((wh > 5) & (wh < 17.5) & (wh > bal) & (ch > 0.5), 3, hc_type)
    hc_type = np.where((wh > 5) & (wh < 17.5) & (wh > bal) & (ch < 0.5), 4, hc_type)
    hc_type = np.where((wh > 17.5) & (wh < 40) & (wh > bal), 5, hc_type)

    return pd.DataFrame({
        'Gas Quality Ratio': gqr,
        'Wetness Ratio': wh,
        'Balance Ratio': bal,
        'Character Ratio': ch,
        'Hydrocarbon Type': [hc_type_dict[x] for x in hc_type],
        'HC_TYPE_FLAG': hc_type
    })
