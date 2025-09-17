import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import curve_fit

from quick_pp.utils import straight_line_func as func
from quick_pp import logger

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
    logger.debug(f"Fitting pressure gradient with {len(tvdss)} data points")
    popt, _ = curve_fit(func, tvdss, formation_pressure)
    gradient, intercept = round(popt[0], 3), round(popt[1], 3)
    logger.debug(f"Fitted pressure gradient: {gradient} psi/ft, intercept: {intercept}")
    return gradient, intercept


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
    logger.debug(f"Creating fluid contact plot for {fluid_type} with gradient {m} psi/ft")
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
    logger.debug("Fluid contact plot created successfully")


def gas_composition_analysis(c1, c2, c3, ic4, nc4, ic5, nc5):
    """Analyze hydrocarbon type based on gas composition (Haworth 1985).

    Hydrocarbon type flag (HC_TYPE_FLAG) is classified into 6 categories:
    0: Non Representative Sample
    1: Non Productive Dry Gas
    2: Potentially Productive Gas
    3: Potentially Productive High GOR Oil
    4: Potentially Productive Condensate
    5: Potentially Productive Oil

    Gas quality flag (GQ_FLAG) is classified into 2 categories:
    0: Not within the range of 0.8 to 1.2
    1: Within the range of 0.8 to 1.2

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
    logger.debug("Performing gas composition analysis using Haworth 1985 method")
    total = c1 + c2 + c3 + ic4 + nc4 + ic5 + nc5

    # Gas Quality Ratio and flag
    gqr = total / (c1 + 2 * c2 + 3 * c3 + 4 * (ic4 + nc4) + 5 * (ic5 + nc5))
    gq_flag = np.where((gqr > 0.8) & (gqr < 1.2), 1, 0)
    logger.debug(f"Gas quality ratio range: {np.min(gqr):.3f} - {np.max(gqr):.3f}")

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

    result_df = pd.DataFrame({
        'Gas Quality Ratio': gqr,
        'GQ_FLAG': gq_flag,
        'Wetness Ratio': wh,
        'Balance Ratio': bal,
        'Character Ratio': ch,
        'Hydrocarbon Type': [hc_type_dict[x] for x in hc_type],
        'HC_TYPE_FLAG': hc_type
    })

    logger.debug(f"Gas composition analysis completed. HC types found: {np.unique(hc_type)}")
    return result_df


def fix_fluid_segregation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating fluid volume fractions from OIL_FLAG and GAS_FLAG")
    df['VOIL'] = df['OIL_FLAG'] * df['VHC']
    df['VGAS'] = df['GAS_FLAG'] * df['VHC']

    for well_name, well_df in tqdm(df.groupby('WELL_NAME'), desc="Fixing fluid segregation"):
        tqdm.write(f"Processing well {well_name}")
        # Fix fluid segregation issues bounded by continuous hydrocarbon intervals
        hc_mask = ((well_df['VHC'] >= 1e-2)).astype(int)
        # Identify continuous hydrocarbon intervals
        hc_groups = (hc_mask.diff() != 0).cumsum()

        for _, group_df in well_df.groupby(hc_groups):
            # Process only hydrocarbon-bearing intervals
            if hc_mask.loc[group_df.index].sum() > 0:
                # If both gas and oil are predicted in the same interval
                if (group_df['GAS_FLAG'] == 1).any() and (group_df['OIL_FLAG'] == 1).any():
                    # Find the deepest depth where gas is predicted
                    last_gas_depth = group_df[group_df['GAS_FLAG'] == 1]['DEPTH'].max()
                    # Identify indices of oil intervals above this deepest gas
                    oil_above_gas_indices = group_df[(group_df['DEPTH'] <= last_gas_depth) & (
                        group_df['OIL_FLAG'] == 1)].index
                    # Re-assign oil volumes to gas for these intervals in the main dataframe
                    df.loc[oil_above_gas_indices, 'VGAS'] = df.loc[oil_above_gas_indices, 'VHC']
                    df.loc[oil_above_gas_indices, 'VOIL'] = 0
                if (group_df['GAS_FLAG'] == 1).any() and (group_df['OIL_FLAG'] == 0).all():
                    df.loc[group_df.index, 'VGAS'] = df.loc[group_df.index, 'VHC']
                    df.loc[group_df.index, 'VHC'] = 0
                if (group_df['OIL_FLAG'] == 1).any() and (group_df['GAS_FLAG'] == 0).all():
                    df.loc[group_df.index, 'VOIL'] = df.loc[group_df.index, 'VHC']
                    df.loc[group_df.index, 'VHC'] = 0

    df['VHC'] = np.where((df['VOIL'] > 0) | (df['VGAS'] > 0), 0, df['VHC'])

    return df
