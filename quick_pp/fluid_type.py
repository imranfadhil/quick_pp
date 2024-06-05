import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from .utils import straight_line_func as func

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
