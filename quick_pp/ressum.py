import numpy as np
import pandas as pd
from scipy.stats import gmean, hmean
from scipy.stats import truncnorm
import random
import matplotlib.pyplot as plt
import ptitprince as pt
from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample
from tqdm import tqdm

from quick_pp import logger


def calc_reservoir_summary(depth, vshale, phit, swt, perm, zones,
                           cutoffs=dict(VSHALE=0.4, PHIT=0.01, SWT=0.9), uom: str = 'ft'):
    """Calculate reservoir summary based on cutoffs on vshale, phit, and swt.

    Args:
        depth (float): Depth either in MD or TVD.
        vshale (float): Volume of shale in fraction.
        phit (float): Total porosity in fraction.
        swt (float): Total water saturation in fraction
        zones (str): Zone names.
        cutoffs (dict, optional): {VSHALE: x, PHIT: y, SWT: z}. Defaults to dict(VSHALE=0.4, PHIT=0.01, SWT=0.9).
        uom (str, optional): Unit of measurement for depth. Defaults to 'ft'.

    Returns:
        pd.Dataframe: Reservoir summary in tabular format.
    """
    logger.debug(f"Starting reservoir summary calculation with {len(depth)} data points, uom={uom}")

    step = 0.1524 if uom == 'm' else 0.5
    df = pd.DataFrame({'depth': depth, 'vshale': vshale, 'phit': phit, 'swt': swt, 'perm': perm, 'zones': zones})
    df['bvo'] = df['phit'] * (1 - df['swt'])
    df['rock_flag'], df['reservoir_flag'], df['pay_flag'] = flag_interval(df['vshale'], df['phit'], df['swt'], cutoffs)
    df['all_flag'] = 1

    logger.debug(
        f"Data flags calculated - Rock: {df['rock_flag'].sum()}, "
        f"Reservoir: {df['reservoir_flag'].sum()}, Pay: {df['pay_flag'].sum()}"
    )

    ressum_df = pd.DataFrame()
    for flag in ['all', 'rock', 'reservoir', 'pay']:
        temp_df = pd.DataFrame()
        # Calculate net thickness
        temp_df[["zones", "net"]] = df.groupby(["zones"])[[f"{flag}_flag"]].agg(
            lambda x: np.nansum(x) * step).reset_index()

        # Average the properties and merge
        flag_df = df[df[f"{flag}_flag"] == 1].copy()
        avg_df = flag_df.groupby(["zones"]).agg({
            "vshale": lambda x: np.nanmean(x),
            "phit": lambda x: np.nanmean(x),
            "swt": lambda x: np.nanmean(x),
            "bvo": lambda x: np.nanmean(x)
        }).reset_index().rename(columns={"vshale": "av_vshale", "phit": "av_phit", "swt": "av_swt", "bvo": "av_bvo"})
        avg_df['perm_am'] = flag_df.groupby('zones')['perm'].agg('mean').reset_index(drop=True)
        avg_df['perm_gm'] = flag_df.groupby('zones')['perm'].agg(gmean, nan_policy='omit').reset_index(drop=True)
        avg_df['perm_hm'] = flag_df.groupby('zones')['perm'].agg(hmean, nan_policy='omit').reset_index(drop=True)
        temp_df = temp_df.merge(avg_df, on=["zones"], how="left", validate="1:1")

        # Calculate gross thickness
        gross = df.groupby(["zones"])[["depth"]].agg(lambda x: np.nanmax(x) - np.nanmin(x)).reset_index().rename(
            columns={"depth": "gross"})
        temp_df = temp_df.merge(gross[["zones", 'gross']], on=["zones"], how="left", validate="1:1")
        temp_df['ntg'] = temp_df['net'] / temp_df['gross']

        # Set the maximum depth as bottom depth
        bottom = df.groupby(["zones"])[["depth"]].agg(lambda x: np.nanmax(x)).reset_index().rename(
            columns={"depth": "bottom"})
        temp_df = temp_df.merge(bottom[["zones", 'bottom']], on=["zones"], how="left", validate="1:1")

        # Set the minimum depth as top depth
        top = df.groupby(["zones"])[["depth"]].agg(lambda x: np.nanmin(x)).reset_index().rename(
            columns={"depth": "top"})
        temp_df = temp_df.merge(top[["zones", 'top']], on=["zones"], how="left", validate="1:1")
        temp_df['flag'] = flag

        # Concat to ressum_df
        ressum_df = pd.concat([ressum_df, temp_df], ignore_index=True)

    ressum_df = ressum_df.round(3)
    ressum_df = ressum_df.sort_values(by=['top'], ignore_index=True)

    # Sort the columns
    cols = ['zones', 'flag', 'top', 'bottom', 'gross', 'net', 'ntg',
            'av_vshale', 'av_phit', 'av_swt', 'av_bvo', 'perm_am', 'perm_gm', 'perm_hm']

    logger.debug(f"Reservoir summary calculation completed with {len(ressum_df)} summary rows")
    return ressum_df[cols]


def flag_interval(vshale, phit, swt, cutoffs: dict):
    """Flag interval based on cutoffs.

    Args:
        vshale (float): Vshale.
        phit (float): Total porosity.
        swt (float): Water saturation.
        cutoffs (list, optional): List of cutoffs. Defaults to [].

    Returns:
        float: Flagged interval.
    """
    if len(cutoffs) != 3:
        logger.error(f"Invalid cutoffs format: expected 3 key-value pairs, got {len(cutoffs)}")
        raise AssertionError('cutoffs must be 3 key-value pairs: {VSHALE: x, PHIT: y, SWT: z}.')

    logger.debug(
        f"Flagging intervals with cutoffs: VSHALE={cutoffs.get('VSHALE')}, "
        f"PHIT={cutoffs.get('PHIT')}, SWT={cutoffs.get('SWT')}"
    )

    rock_flag = np.where(vshale < cutoffs['VSHALE'], 1, 0)
    reservoir_flag = np.where(rock_flag == 1, np.where(phit > cutoffs['PHIT'], 1, 0), 0)
    pay_flag = np.where(reservoir_flag == 1, np.where(swt < cutoffs['SWT'], 1, 0), 0)

    return rock_flag, reservoir_flag, pay_flag


def volumetric_method(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
    recovery_factor_bound: tuple = (),
    random_state=None
):
    """Calculate volume using the volumetric method.

    Args:
        area_bound (tuple): (min, max, mean, std) in acre.
            - Truncated normal distribution parameters for area.
        thickness_bound (tuple): (min, max, mean, std) in feet.
            - Truncated normal distribution parameters for thickness.
        porosity_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for porosity.
        water_saturation_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for water saturation.
        volume_factor_bound (tuple): (min, max) unitless.
            - Uniform distribution parameters for volume factor.
        recovery_factor_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for RF.
        random_state (int, optional): Random seed. Defaults to 123.

    Returns:
        float: Estimated volume in MM bbl.
    """
    logger.debug(f"Starting volumetric method calculation with random_state={random_state}")

    random.seed(random_state)
    a_min_transformed = (area_bound[0] - area_bound[2]) / area_bound[3]
    a_max_transformed = (area_bound[1] - area_bound[2]) / area_bound[3]
    a = truncnorm.rvs(a_min_transformed, a_max_transformed,
                      loc=area_bound[2], scale=area_bound[3], random_state=random_state)

    h_min_transformed = (thickness_bound[0] - thickness_bound[2]) / thickness_bound[3]
    h_max_transformed = (thickness_bound[1] - thickness_bound[2]) / thickness_bound[3]
    h = truncnorm.rvs(h_min_transformed, h_max_transformed,
                      loc=thickness_bound[2], scale=thickness_bound[3], random_state=random_state)

    poro_min_transformed = (porosity_bound[0] - porosity_bound[2]) / porosity_bound[3]
    poro_max_transformed = (porosity_bound[1] - porosity_bound[2]) / porosity_bound[3]
    poro = truncnorm.rvs(poro_min_transformed, poro_max_transformed,
                         loc=porosity_bound[2], scale=porosity_bound[3], random_state=random_state)
    sw_min_transformed = (water_saturation_bound[0] - water_saturation_bound[2]) / water_saturation_bound[3]
    sw_max_transformed = (water_saturation_bound[1] - water_saturation_bound[2]) / water_saturation_bound[3]
    sw = truncnorm.rvs(sw_min_transformed, sw_max_transformed,
                       loc=water_saturation_bound[2], scale=water_saturation_bound[3], random_state=random_state)

    bo = random.uniform(volume_factor_bound[0], volume_factor_bound[1])

    if recovery_factor_bound is not None:
        rf_min_transformed = (recovery_factor_bound[0] - recovery_factor_bound[2]) / recovery_factor_bound[3]
        rf_max_transformed = (recovery_factor_bound[1] - recovery_factor_bound[2]) / recovery_factor_bound[3]
        rf = truncnorm.rvs(rf_min_transformed, rf_max_transformed,
                           loc=recovery_factor_bound[2], scale=recovery_factor_bound[3], random_state=random_state)
    else:
        rf = 1.0

    logger.debug(
        f"Volumetric parameters generated - Area: {a:.2f}, Thickness: {h:.2f}, "
        f"Porosity: {poro:.3f}, Sw: {sw:.3f}, Bo: {bo:.3f}, RF: {rf:.3f}"
    )
    return a, h, poro, sw, bo, rf


def mc_volumetric_method(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple,
    n_try=10000, percentile=[10, 50, 90]
):
    """Monte Carlo simulation for volumetric method.

    Args:
        area_bound (tuple): (min, max, mean, std) in acre.
            - Truncated normal distribution parameters for area.
        thickness_bound (tuple): (min, max, mean, std) in feet.
            - Truncated normal distribution parameters for thickness.
        porosity_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for porosity.
        water_saturation_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for water saturation.
        volume_factor_bound (tuple): (min, max) unitless.
            - Uniform distribution parameters for volume factor.
        n_try (int, optional): Number of trials. Defaults to 10000.
        percentile (list, optional): Percentiles to calculate. Defaults to [10, 50, 90].
    """
    logger.info(f"Starting Monte Carlo simulation with {n_try} trials")

    area = np.empty(0)
    thickness = np.empty(0)
    porosity = np.empty(0)
    water_saturation = np.empty(0)
    volume_factor = np.empty(0)
    volumes = np.empty(0)

    for i in tqdm(range(n_try), desc="Monte Carlo simulation", unit="trials"):
        a, h, poro, sw, bo, _ = volumetric_method(
            area_bound=area_bound,
            thickness_bound=thickness_bound,
            porosity_bound=porosity_bound,
            water_saturation_bound=water_saturation_bound,
            volume_factor_bound=volume_factor_bound
        )
        area = np.append(area, a)
        thickness = np.append(thickness, h)
        porosity = np.append(porosity, poro)
        water_saturation = np.append(water_saturation, sw)
        volume_factor = np.append(volume_factor, bo)
        result = (43560 * 0.1781) * a * h * poro * (1 - sw) / bo
        volumes = np.append(volumes, result * 1e-6)

    logger.info(
        f"Monte Carlo simulation completed. Volume statistics - "
        f"Mean: {np.mean(volumes):.2f} MM bbl, Std: {np.std(volumes):.2f} MM bbl"
    )

    # Plot tornado chart
    data = pd.DataFrame({
        'Area': area,
        'Thickness': thickness,
        'Porosity': porosity,
        'Water Saturation': water_saturation,
        'Volume Factor': volume_factor,
        'Volume': volumes
    })
    fig, axs = plt.subplots(5, 1, figsize=(7, 11), constrained_layout=True)
    pt.RainCloud(data=data, y='Area', ax=axs[0], orient='h', bw=0.1, width_viol=0.5, alpha=0.6, dodge=True)
    axs[0].set_ylabel('Area (acre)')
    pt.RainCloud(data=data, y='Thickness', ax=axs[1], orient='h', bw=0.1, width_viol=0.5, alpha=0.6, dodge=True)
    axs[1].set_ylabel('Thickness (ft)')
    pt.RainCloud(data=data, y='Porosity', ax=axs[2], orient='h', bw=0.1, width_viol=0.5, alpha=0.6, dodge=True)
    axs[2].set_ylabel('Porosity (frac)')
    pt.RainCloud(data=data, y='Water Saturation', ax=axs[3], orient='h', bw=0.1, width_viol=0.5, alpha=0.6, dodge=True)
    axs[3].set_ylabel('Water Saturation (frac)')
    pt.RainCloud(data=data, y='Volume Factor', ax=axs[4], orient='h', bw=0.1, width_viol=0.5, alpha=0.6, dodge=True)
    axs[4].set_ylabel('Volume Factor')
    fig.set_facecolor('aliceblue')
    fig.show()

    plt.figure(figsize=(12, 6))
    a = np.hstack(volumes.tolist())
    _ = plt.hist(a, density=True, bins='auto')  # arguments are passed to np.histogram
    plt.title(f"Volume Estimation ({n_try} samples)")
    plt.xlabel('Volume (MM bbl)')
    for i, pct in enumerate(np.percentile(volumes, percentile)):
        plt.axvline(x=pct, color='r', linestyle='dashed', label=f'P{percentile[i]}: {round(pct)} MM bbl')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    return volumes, area, thickness, porosity, water_saturation, volume_factor


def calculate_volume(x, area, thickness, porosity, water_saturation, volume_factor, recovery_factor=1):
    """Calculate volume using the volumetric method.

    Args:
        area (float): Area in acre.
        thickness (float): Thickness in feet.
        porosity (float): Porosity in fraction.
        water_saturation (float): Water saturation in fraction.
        volume_factor (float): Volume factor.
        recovery_factor (float): Recovery factor in fraction.

    Returns:
        float: Estimated volume in MM bbl.
    """
    return x * (43560 * 0.1781) * area * thickness * porosity * (1 - water_saturation) / (
        volume_factor) * recovery_factor * 1e-6


def sensitivity_analysis(
    area_bound: tuple,
    thickness_bound: tuple,
    porosity_bound: tuple,
    water_saturation_bound: tuple,
    volume_factor_bound: tuple
):
    """Sensitivity analysis for volumetric method.

    Args:
        area_bound (tuple): (min, max, mean, std) in acre.
            - Truncated normal distribution parameters for area.
        thickness_bound (tuple): (min, max, mean, std) in feet.
            - Truncated normal distribution parameters for thickness.
        porosity_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for porosity.
        water_saturation_bound (tuple): (min, max, mean, std) in fraction.
            - Truncated normal distribution parameters for water saturation.
        volume_factor_bound (tuple): (min, max) unitless.
            - Uniform distribution parameters for volume factor.
    """
    logger.info("Starting sensitivity analysis for volumetric method")

    # Define the model inputs
    problem = {
        'num_vars': 5,
        'names': ['area', 'thickness', 'porosity', 'water_saturation', 'volume_factor'],
        'bounds': [
            [area_bound[0], area_bound[1]],
            [thickness_bound[0], thickness_bound[1]],
            [porosity_bound[0], porosity_bound[1]],
            [water_saturation_bound[0], water_saturation_bound[1]],
            [volume_factor_bound[0], volume_factor_bound[1]]
        ]
    }

    logger.debug(f"Problem defined with {problem['num_vars']} variables and {2**10} samples")

    # Generate samples
    param_values = sample(problem, 2**10)

    # evaluate
    x = np.linspace(-1, 1, 100)
    y = np.array([calculate_volume(x, *params) for params in param_values])

    # analyse
    Si = [analyze(problem, Y) for Y in y.T][0]

    # Extract the sensitivity indices
    S1 = Si['S1']
    S1_conf = Si['S1_conf']
    names = problem['names']

    # Sort the indices for better visualization
    sorted_indices = np.argsort(S1)
    sorted_S1 = S1[sorted_indices]
    sorted_S1_conf = S1_conf[sorted_indices]
    sorted_names = [names[i] for i in sorted_indices]

    logger.info(
        f"Sensitivity analysis completed. Top parameter: {sorted_names[-1]} "
        f"(S1={sorted_S1[-1]:.3f})"
    )

    # Plot the tornado chart
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(sorted_S1)), sorted_S1, xerr=sorted_S1_conf,
             align='center', alpha=0.7, color='blue', ecolor='black')
    plt.yticks(range(len(sorted_S1)), sorted_names)
    plt.xlabel('First-order Sensitivity Index')
    plt.title('Tornado Chart of Sensitivity Indices')
    plt.grid(True)
    plt.show()
