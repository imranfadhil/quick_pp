import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt

from quick_pp.utils import length_a_b, line_intersection
from quick_pp.rock_type import estimate_vsh_gr
from quick_pp.config import Config
from quick_pp.ressum import calc_reservoir_summary, flag_interval
from quick_pp import logger

plt.style.use('seaborn-v0_8-paper')


def handle_outer_limit(data, fill=False):
    """ Replace values outside of min and max with null or the min/ max value.

    Args:
        data (pd.DataFrame): Data to be transformed

    Returns:
        (pd.DataFrame): masked dataframe
    """
    data = data.copy()
    for var, var_dict in Config.VARS.items():
        if (('min' in var_dict.keys()) or ('max' in var_dict.keys())):
            if var in data.columns:
                if not fill:
                    data.loc[:, var] = np.where(data[var] < var_dict['min'], np.nan, data[var])
                    data.loc[:, var] = np.where(data[var] > var_dict['max'], np.nan, data[var])
                else:
                    data.loc[:, var] = np.where(data[var] < var_dict['min'], var_dict['min'], data[var])
                    data.loc[:, var] = np.where(data[var] > var_dict['max'], var_dict['max'], data[var])
    return data


def badhole_flagging(data, thold=4):
    """Generate BADHOLE flag using ruptures package. Modified based on https://hallau.world/post/bitsize-from-caliper/

    Args:
        df (pd.DataFrame): Dataframe used to generate the BADHOLE_FLAG
        thold (int, optional): Threshold in difference between CALI and BS to flag as badhole. Defaults to 4.

    Returns:
        pd.DataFrame: Pandas dataframe with BADHOLE_FLAG
    """
    import ruptures as rpt
    import scipy.stats
    from scipy.signal import find_peaks
    df = data.copy()
    if 'CALI' not in df.columns:
        df['BADHOLE'] = 0
        return df

    def reject_outliers(series, m=3):
        # Works on a pandas Series
        series = series.where((series > 6.0) & (series < 21.5))  # Widened range
        d = np.abs(series - series.median())
        mdev = np.nanmedian(d)
        s = d / mdev if mdev else 0.0
        return series.where(s < m, series.median())

    bit_sizes = {
        # 6.0: 6.0,
        8.5: 8.5,
        12.25: 12.25,
        17.5: 17.5
    }

    model = "rbf"
    processed_wells = []

    df['BS'] = np.nan if 'BS' not in df.columns else df['BS']

    for well, well_data in df.groupby('WELL_NAME'):
        well_data = well_data.sort_values(by='DEPTH').copy()
        well_data['BIN'] = 0
        well_data['ESTIMATED_BITSIZE'] = 8.5
        if not well_data[well_data["CALI"].notna()].empty:
            logger.info(f"[feature_transformer `badhole_flag`] Processing {well}.")

            signal_data = well_data['CALI']
            signal_data = reject_outliers(signal_data)
            clean_signal = signal_data.dropna()

            try:
                # Identifying number of clusters/breakpoints from Kernel Density Estimation
                if clean_signal.nunique() > 1:
                    kde = scipy.stats.gaussian_kde(clean_signal)
                    evaluated = kde.evaluate(np.linspace(clean_signal.min(), clean_signal.max(), 100))
                    peaks, _ = find_peaks(evaluated, height=0.1)
                else:
                    peaks = []

                jump = 60
                algo = rpt.BottomUp(model=model, jump=jump).fit(signal_data.to_numpy())
                my_bkps = algo.predict(n_bkps=len(peaks))

                # Determine the bit sizes based on break points
                indices = [0] + my_bkps
                bins = list(zip(indices, indices[1:]))
                ii = pd.IntervalIndex.from_tuples(bins, closed='left')
                well_data['BIN'] = pd.cut(well_data.reset_index().index, bins=ii)
                bits_percentiles = well_data.groupby('BIN')['CALI'].describe(percentiles=[.01])

                estimated_bits = []
                for i in bits_percentiles['1%'].values:
                    bit_ = bit_sizes[min(bit_sizes.keys(), key=lambda k: abs(k - i))]
                    estimated_bits.append(bit_)

                bits_diff_1 = np.array(estimated_bits[1:] + [estimated_bits[-1]])
                bits_diff_neg1 = np.array([estimated_bits[0]] + estimated_bits[:-1])
                estimated_bits = np.array(estimated_bits)
                estimated_bits = np.where(bits_diff_1 == bits_diff_neg1, bits_diff_1, estimated_bits)

                categories = dict(zip(well_data.BIN.unique(), estimated_bits))
                well_data['ESTIMATED_BITSIZE'] = well_data['BIN'].map(categories)
            except Exception as e:
                logger.error(f"[badhole_flagging] Error processing well {well}: {e}.")
                continue

        processed_wells.append(well_data)

    if processed_wells:
        return_df = pd.concat(processed_wells, ignore_index=True)

        bitsize = np.where(return_df.BS.notna(), return_df.BS, return_df.ESTIMATED_BITSIZE)
        return_df['BS'] = bitsize
        absdiff = np.where(return_df.CALI.notna(), abs(return_df.CALI - bitsize), 0)
        return_df['BADHOLE'] = np.where(absdiff < thold, 0, 1)

        # Drop BIN and ESTIMATED_BITSIZE
        return_df.drop(['BIN', 'ESTIMATED_BITSIZE'], axis=1, inplace=True)

        return return_df
    else:
        return df


def neu_den_xplot_hc_correction(
        nphi, rhob, vsh_gr=None,
        dry_min1_point: tuple = (),
        dry_clay_point: tuple = (),
        water_point: tuple = (1.0, 1.0),
        corr_angle: float = 50, buffer=0.0):
    """Estimate correction for neutron porosity and bulk density based on correction angle.

    Args:
        nphi (float): Neutron porosity log in fraction.
        rhob (float): Bulk density log in g/cc.
        vsh_gr (float, optional): Vshale from gamma ray log. Defaults to None.
        dry_min1_point (tuple, optional): Neutron porosity and bulk density of mineral 1 point. Defaults to None.
        dry_clay_point (tuple, optional): Neutron porosity and bulk density of dry clay point. Defaults to None.
        water_point (tuple, optional): Neutron porosity and bulk density of water point. Defaults to (1.0, 1.0).
        corr_angle (float, optional): Correction angle (degree) from east horizontal line. Defaults to 50.
        buffer (float, optional): Buffer to be included in correction. Defaults to 0.0.

    Returns:
        (np.ndarray, np.ndarray, np.ndarray): Corrected neutron porosity, bulk density, and hydrocarbon flag.
    """
    A = dry_min1_point
    C = dry_clay_point
    D = water_point
    rocklithofrac = length_a_b(A, C)

    # Ensure inputs are numpy arrays
    nphi = np.array(nphi)
    rhob = np.array(rhob)
    frac_vsh_gr = np.array(vsh_gr) if vsh_gr is not None else np.zeros_like(nphi)

    # Initialize corrected logs and flags
    nphi_corrected = nphi.copy()
    rhob_corrected = rhob.copy()
    hc_flag = np.zeros_like(nphi, dtype=int)

    # Calculate initial projection and vsh_dn
    var_pts = line_intersection((A, C), (D, (nphi, rhob)))
    projlithofrac = length_a_b(var_pts, C) + buffer
    vsh_dn = 1 - (projlithofrac / rocklithofrac)

    # Identify points needing correction
    correction_mask = (projlithofrac > rocklithofrac) & (nphi < 0.45)
    hc_flag[correction_mask] = 1

    # Iteratively correct points
    # Using a max iteration count to prevent infinite loops
    for i in range(100):
        # Only work on points that still need correction
        active_mask = correction_mask & (vsh_dn <= frac_vsh_gr) & ~np.isnan(frac_vsh_gr)
        if not np.any(active_mask):
            break  # Exit if no more points need correction

        # Apply correction shift
        nphi_corrected[active_mask] += 1e-2 * np.cos(np.radians(corr_angle))
        rhob_corrected[active_mask] += 1e-2 * np.sin(np.radians(corr_angle))

        # Recalculate vsh_dn for the corrected points
        var_pts_corr = line_intersection((A, C), (D, (nphi_corrected[active_mask], rhob_corrected[active_mask])))
        projlithofrac_corr = length_a_b(var_pts_corr, C)
        vsh_dn[active_mask] = 1 - (projlithofrac_corr / rocklithofrac)

    return nphi_corrected, rhob_corrected, hc_flag.astype(float)


def neu_den_xplot_hc_correction_angle(rho_water=1.0, rho_hc=0.3, HI_hc=None):
    """Estimate correction angle for neutron porosity and bulk density based on water and hydrocarbon density.
     The angle is from east horizontal line.

    Args:
        rho_water (float): Fluid density, ranges from 0.8 to 1.2. Defaults to 1.0.
        rho_hc (float): Hydrocarbon density; Gas: 0.1 - 0.3; Oil: 0.4 - 0.8. Defaults to 0.3.
        HI_hc (float): HI hydrocarbon index; Gas: 0.2 - 0.7; Oil: 0.8 - 1.0. Defaults to None.

    Returns:
        float: Correction angle (degree).
    """
    if HI_hc is None:
        HI_hc = 2.2 * rho_hc
    # Estimate correction angle
    corr_angle = np.arctan((rho_water - rho_hc) / (1 - HI_hc))
    return 90 - math.degrees(corr_angle)


def den_correction(nphi, gr, vsh_gr=None, phin_sh=.35, phid_sh=.05, rho_ma=2.68, rho_water=1.0, alpha=0.05):
    """Correct bulk density based on nphi and gamma ray.

    Args:
        nphi (float): Neutron porosity log in fraction.
        gr (float): Gamma ray log in GAPI.
        phin_sh (float, optional): Neutron porosity of shale. Defaults to .35.
        phid_sh (float, optional): Density porosity of shale. Defaults to .05.
        rho_ma (float, optional): Matrix density. Defaults to 2.68.
        rho_water (float, optional): Fluid density. Defaults to 1.0.
        alpha (float, optional): Alpha value for gamma ray normalization. Defaults

    Returns:
        float: Corrected bulk density.
    """
    # Estimate vsh_gr
    vsh_gr = vsh_gr if vsh_gr is not None else estimate_vsh_gr(gr, alpha=alpha)
    # Estimate phid assuming vsh_gr = vsh_dn = (phin - phid) / (phin_sh - phid_sh)
    phid = nphi - (phin_sh - phid_sh) * vsh_gr
    return rho_ma - (rho_ma - rho_water) * phid


def quick_qc(well_data, return_fig=False):
    """Quick QC for well data.

    Args:
        well_data (pd.DataFrame): Well data, meant to process one well at a time.

    Returns:
        pd.DataFrame: Well data with QC flags.
        pd.DataFrame: Reservoir summary.
        plt.Figure: Distribution plot.
        plt.Figure: Depth plot.
    """
    return_df = well_data.copy()
    return_df['ZONES'] = "ALL"
    return_df['QC_FLAG'] = 0
    return_df['PERM'] = return_df['PERM'].clip(lower=1e-3, upper=1e5)

    cutoffs = dict(VSHALE=0.4, PHIT=0.01, SWT=0.9)
    _, return_df['RES_FLAG'], _ = flag_interval(return_df['VCLAY'], return_df['PHIT'], return_df['SWT'], cutoffs)
    return_df['ALL_FLAG'] = 1

    # Check average swt in non-reservoir zone
    return_df['QC_FLAG'] = np.where((return_df['RES_FLAG'] == 0) & (return_df['SWT'] < 0.8), 3, return_df['QC_FLAG'])

    # Summarize
    summary_df = calc_reservoir_summary(
        return_df.DEPTH, return_df.VCLAY, return_df.PHIT, return_df.SWT, return_df.PERM, return_df.ZONES
    )
    summary_df.columns = [col.upper() for col in summary_df.columns]
    for group in ['all', 'reservoir']:
        index = return_df['ALL_FLAG'] == 1 if group == 'all' else return_df['RES_FLAG'] == 1
        data = return_df[index]

        # Calculate average and standard deviation
        temp_df = data.groupby('ZONES').agg({
            'GR': ['mean', 'std'],
            'RT': ['mean', 'std'],
            'NPHI': ['mean', 'std'],
            'RHOB': ['mean', 'std'],
        })
        # Rename columns
        cols_rename = [f'{stat}_{col}' for col, stat in temp_df.columns]
        cols_rename = [col.replace('mean', 'AV').replace('std', 'STD') for col in cols_rename]
        temp_df.columns = cols_rename

        summary_df.loc[summary_df.FLAG == group, ['AV_GR', 'AV_RT', 'AV_NPHI', 'AV_RHOB']] = temp_df[
            ['AV_GR', 'AV_RT', 'AV_NPHI', 'AV_RHOB']].values

        summary_df.loc[summary_df.FLAG == group, ['STD_GR', 'STD_RT', 'STD_NPHI', 'STD_RHOB']] = temp_df[
            ['STD_GR', 'STD_RT', 'STD_NPHI', 'STD_RHOB']].values

        summary_df.loc[summary_df.FLAG == group, 'QC_FLAG_COUNT'] = data.where(
            data['QC_FLAG'] == 1, 0).groupby('ZONES')['QC_FLAG'].sum().values

        summary_df.loc[summary_df.FLAG == group, 'QC_FLAG_mode'] = data.groupby(
            'ZONES')['QC_FLAG'].agg(lambda x: x.mode()).values

    if return_fig:
        # Distribution plot
        dist_fig, axs = plt.subplots(4, 1, figsize=(5, 10))
        for group, data in return_df.groupby('RES_FLAG'):
            label = 'RES' if group == 1 else 'NON-RES'
            axs[0].hist(data['VCLAY'], bins=100, alpha=0.7, label=label, range=[0, 1])
            axs[1].hist(data['PHIT'], bins=100, alpha=0.7, label=label, range=[0, .5])
            axs[2].hist(data['SWT'], bins=100, alpha=0.7, label=label, range=[0, 1])
            axs[3].hist(np.log10(data['PERM']), bins=100, alpha=0.7, label=f'Log PERM_{label}')
        axs[0].set_title('VSHALE Distribution')
        axs[0].legend()

        axs[1].set_title('PHIT Distribution')
        axs[1].legend()

        axs[2].set_title('SWT Distribution')
        axs[2].set_yscale('log')
        axs[2].legend()

        axs[3].set_title('Log PERM Distribution')
        axs[3].set_yscale('log')
        axs[3].legend()

        dist_fig.tight_layout()

        # Depth plot
        depth_fig, axs = plt.subplots(4, 1, figsize=(20, 5), sharex=True)
        axs[0].plot(return_df['DEPTH'], return_df['VCLAY'], label='VSHALE')
        # Compare vsh_gr and vsh_dn
        if 'VSH_GR' in return_df.columns:
            return_df['QC_FLAG'] = np.where(
                abs(return_df['VSH_GR'] - return_df['VCLAY']) > 0.1, 1, return_df['QC_FLAG'])
            axs[0].plot(return_df['DEPTH'], return_df['VSH_GR'], label='VSH_GR')
        axs[0].set_frame_on(False)
        axs[0].set_title('VSHALE')
        axs[0].legend()

        axs[1].plot(return_df['DEPTH'], return_df['PHIT'], label='PHIT')
        # Compare phit_dn and phit_d
        if 'PHID' in return_df.columns:
            return_df['QC_FLAG'] = np.where(abs(return_df['PHIT'] - return_df['PHID']) > 0.1, 2, return_df['QC_FLAG'])
            axs[1].plot(return_df['DEPTH'], return_df['PHID'], label='PHID')
        axs[1].set_frame_on(False)
        axs[1].set_title('PHIT')
        axs[1].legend()

        axs[2].plot(return_df['DEPTH'], return_df['SWT'], label='SWT')
        axs[2].plot(return_df['DEPTH'], return_df['RES_FLAG'], label='RES_FLAG')
        axs[2].set_frame_on(False)
        axs[2].set_title('SWT')
        axs[2].legend()

        axs[3].plot(return_df['DEPTH'], return_df['QC_FLAG'], label='QC_FLAG')
        axs[3].set_frame_on(False)
        axs[3].set_title('QC_FLAG')
        axs[3].legend()

        depth_fig.tight_layout()

        return return_df, summary_df, dist_fig, depth_fig

    else:
        return return_df, summary_df, None, None


def quick_compare(field_data, level='WELL', return_fig=False):
    """Quick comparison of field data.

    Args:
        field_data (pd.DataFrame): Field data.
        level (str, optional): Level of comparison, either 'WELL' or 'ZONE'. Defaults to 'WELL'.
        return_fig (bool, optional): Whether to return figure. Defaults to False.

    Returns:
        pd.DataFrame: Comparison dataframe.
        plt.Figure: Comparison figure.
    """
    assert level in ['WELL', 'ZONE'], "Level must be either 'WELL' or 'ZONE'"
    field_data['ZONES'] = "ALL" if level == 'WELL' else field_data['ZONES']
    compare_df = pd.DataFrame()
    for well, well_data in field_data.groupby('WELL_NAME'):
        logger.info(f"Processing {well}")
        _, summary_df, _, _ = quick_qc(well_data, return_fig=False)
        summary_df.insert(0, 'WELL_NAME', well)
        compare_df = pd.concat([compare_df, summary_df])
    logger.info("Finished processing all wells.")

    if return_fig:
        # Box plot
        curves = ['AV_GR', 'AV_RT', 'AV_NPHI', 'AV_RHOB',
                  'AV_VSHALE', 'AV_PHIT', 'AV_SWT', 'AV_PERM_GM']
        no_curves = len(curves)
        plot_df = compare_df[compare_df.FLAG == 'all']
        idx_percentiles = [int(x * (len(plot_df) - 1)) for x in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]]
        fig, axes = plt.subplots(no_curves, 3, figsize=(7, 2.5 * no_curves),
                                 gridspec_kw={'width_ratios': [1, 3, 5]})
        for i, curve in enumerate(curves):
            # Box plot
            axes[i, 0].set_title(curve)
            axes[i, 0].boxplot(plot_df[curve], labels=[''])

            # Distribution plot
            axes[i, 1].hist(plot_df[curve], bins=25, alpha=0.7, orientation='horizontal')

            # Summary table
            df = plot_df[['WELL_NAME', curve]].sort_values(by=curve, ascending=True).reset_index(drop=True)
            df = df.loc[idx_percentiles, :].round(2)
            df['PERCENTILES'] = ['0%', '10%', '25%', '50%', '75%', '90%', '100%']
            df = df[['PERCENTILES', curve, 'WELL_NAME']]
            axes[i, 2].table(cellText=df.values, colLabels=df.columns, loc='center', cellLoc='center',
                             colColours=['#f2f2f2'] * 3)
            axes[i, 2].axis('off')

        plt.tight_layout()
        return compare_df.reset_index(drop=True), fig

    else:
        return compare_df.reset_index(drop=True), None


def extract_quick_stats(compare_df, flag='all'):
    """Extract quick stats from average comparison dataframe.

    Args:
        compare_df (pd.DataFrame): Comparison dataframe

    Returns:
        pd.DataFrame: Quick stats dataframe
    """
    compare_df = compare_df[compare_df.FLAG == flag].copy()
    # Extract quick stats
    reqs = ['PHIT', 'SWT']
    stats_df = pd.DataFrame()
    for col in compare_df.columns:
        if any([req in col for req in reqs]) or col in ['NET', 'NTG']:
            stats = compare_df[col].describe(percentiles=[0.1, 0.5, 0.9])
            stats_df[col] = stats

    return round(stats_df, 3)
