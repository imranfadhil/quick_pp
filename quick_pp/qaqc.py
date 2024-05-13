import pandas as pd
import numpy as np

from .utils import length_a_b, line_intersection
from .lithology import gr_index
from .rock_type import vsh_gr
from .config import Config


def mask_outside_threshold(data, fill=False):
    """ Replace values outside of min and max with null or the min/ max value.

    Args:
        data (pd.DataFrame): Data to be transformed

    Returns:
        (pd.DataFrame): masked dataframe
    """
    data = data.copy()
    for model_, vars_ in Config.VARS.items():
        for var_ in vars_:
            var = var_['var']
            if (('min' in var_.keys()) or ('max' in var_.keys())):
                if var in data.columns:
                    if not fill:
                        data.loc[:, var] = np.where(data[var] < var_['min'], np.nan, data[var])
                        data.loc[:, var] = np.where(data[var] > var_['max'], np.nan, data[var])
                    else:
                        data.loc[:, var] = np.where(data[var] < var_['min'], var_['min'], data[var])
                        data.loc[:, var] = np.where(data[var] > var_['max'], var_['max'], data[var])
    return data


def badhole_flag(data, thold=4):
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

    def reject_outliers(data, m=3):
        data = np.where((data > 8.5) & (data < 21.5), data, np.nan)
        d = np.abs(data - np.nanmedian(data))
        mdev = np.nanmedian(d)
        s = d / mdev if mdev else 0.0
        data = np.where(s < m, data, np.nanmedian(data))
        return data

    bit_sizes = {
        # 6.0: 6.0,
        8.5: 8.5,
        12.25: 12.25,
        17.5: 17.5
    }

    model = "rbf"
    return_df = pd.DataFrame()

    df['BS'] = np.nan if 'BS' not in df.columns else df['BS']

    for well, well_data in df.groupby('WELL_NAME'):
        # Determine the breakpoints for different bit size
        well_data.sort_values(by='DEPTH', inplace=True)
        well_data['BIN'] = 0
        well_data['ESTIMATED_BITSIZE'] = 8.5
        if not well_data[well_data["CALI"].notna()].empty:
            print(f"[feature_transformer `badhole_flag`] Processing {well}.")

            signal_data = well_data['CALI']
            signal_data = reject_outliers(signal_data)
            try:
                # Identifying number of clusters/breakpoints from Kernel Density Estimation
                # gaussian_kde couldn't handle constant value
                if (len(np.unique(signal_data)) > 1) & (len(signal_data[~np.isnan(signal_data)]) > 1):
                    kde = scipy.stats.gaussian_kde(signal_data[~np.isnan(signal_data)])
                    evaluated = kde.evaluate(np.linspace(signal_data.min(), signal_data.max(), 100))
                    peaks, _ = find_peaks(evaluated, height=0.1)
                else:
                    peaks = []

                jump = 60
                algo = rpt.BottomUp(model=model, jump=jump).fit(signal_data)
                my_bkps = algo.predict(n_bkps=len(peaks))

                # Determine the bit sizes based on break points
                indices = [0] + my_bkps
                bins = list(zip(indices, indices[1:]))
                ii = pd.IntervalIndex.from_tuples(bins, closed='left')
                well_data['BIN'] = pd.cut(well_data.reset_index().index, bins=ii)
                bits_percentiles = well_data.groupby('BIN')['CALI'].describe(percentiles=[.01])

                estimated_bits = []
                for i in bits_percentiles['1%'].values:
                    bit_ = bit_sizes[min(bit_sizes.keys(), key=lambda k: abs(k-i))]
                    estimated_bits.append(bit_)

                bits_diff_1 = np.array(estimated_bits[1:] + [estimated_bits[-1]])
                bits_diff_neg1 = np.array([estimated_bits[0]] + estimated_bits[:-1])
                estimated_bits = np.array(estimated_bits)
                estimated_bits = np.where(bits_diff_1 == bits_diff_neg1, bits_diff_1, estimated_bits)

                categories = dict(zip(well_data.BIN.unique(), estimated_bits))
                well_data['ESTIMATED_BITSIZE'] = well_data.apply(lambda x: categories.get(x.BIN), axis=1)
            except Exception as e:
                print(f"[feature_transformer `badhole_flag`] Error {e}.")
                continue

        return_df = pd.concat([return_df, well_data])

    if not return_df.empty:
        bitsize = np.where(return_df.BS.notna(), return_df.BS, return_df.ESTIMATED_BITSIZE)
        return_df['BS'] = bitsize
        absdiff = np.where(return_df.CALI.notna(), abs(return_df.CALI - bitsize), 0)
        return_df['BADHOLE'] = np.where(absdiff < thold, 0, 1)

        # Drop BIN and ESTIMATED_BITSIZE
        return_df.drop(['BIN', 'ESTIMATED_BITSIZE'], axis=1, inplace=True)

        return return_df
    else:
        return df


def porosity_correction_averaging(nphi, dphi, method='weighted'):
    """Correct porosity using averaging method.

    Args:
        nphi (float): Neutron porosity.
        dphi (float): Density porosity.
        method (str, optional): Averaging method selection from 'weighted', 'arithmetic' or 'gaymard'.
         Defaults to 'weighted'.

    Returns:
        float: Corrected porosity.
    """
    assert method in ['weighted', 'arithmetic', 'gaymard'], "method must be either \
        'weighted', 'arithmetic' or 'gaymard' "

    if method == 'weighted':
        phit = (2 * dphi + nphi) / 3
    elif method == 'arithmetic':
        phit = (dphi + nphi) / 2
    elif method == 'gaymard':
        phit = np.sqrt((dphi**2 + nphi**2) / 2)
    return phit


def neu_den_xplot_hc_correction(
        nphi, rhob, gr=None,
        dry_sand_point: tuple = None,
        dry_clay_point: tuple = None,
        fluid_point: tuple = (1.0, 1.0),
        corr_angle: float = 50, buffer=0.0):
    """Estimate correction for neutron porosity and bulk density based on correction angle.

    Args:
        nphi (float): Neutron porosity log in fraction.
        rhob (float): Bulk density log in g/cc.
        gr (float, optional): Gamma ray log in GAPI. Defaults to None.
        dry_sand_point (tuple, optional): Neutron porosity and bulk density of dry sand point. Defaults to None.
        dry_clay_point (tuple, optional): Neutron porosity and bulk density of dry clay point. Defaults to None.
        fluid_point (tuple, optional): Neutron porosity and bulk density of fluid point. Defaults to (1.0, 1.0).
        corr_angle (float, optional): Correction angle (degree) from east horizontal line. Defaults to 50.
        buffer (float, optional): Buffer to be included in correction. Defaults to 0.0.

    Returns:
        (float, float): Corrected neutron porosity and bulk density.
    """
    A = dry_sand_point
    C = dry_clay_point
    D = fluid_point
    rocklithofrac = length_a_b(A, C)

    frac_vsh_gr = gr_index(gr)
    nphi_corrected = np.empty(0)
    rhob_corrected = np.empty(0)
    hc_flag = np.empty(0)
    for i, point in enumerate(list(zip(nphi, rhob, frac_vsh_gr))):
        var_pt = line_intersection((A, C), (D, (point[0], point[1])))
        projlithofrac = length_a_b(var_pt, C) + buffer
        if projlithofrac > rocklithofrac:
            # Iteration until vsh_dn = vsh_gr
            vsh_dn = 1 - (projlithofrac / rocklithofrac)
            nphi_corr = point[0]
            rhob_corr = point[1]
            shi = 0
            count = 0
            while vsh_dn <= point[2] and not np.isnan(point[2]) and count < 100:
                nphi_corr = point[0] + shi * np.sin(np.radians(corr_angle))
                rhob_corr = point[1] + shi * np.cos(np.radians(corr_angle))
                shi += 0.01
                count += 1
                # Recalculate vsh_dn
                var_pt = line_intersection((A, C), (D, (nphi_corr, rhob_corr)))
                projlithofrac = length_a_b(var_pt, C)
                vsh_dn = 1 - (projlithofrac / rocklithofrac)

            nphi_corrected = np.append(nphi_corrected, nphi_corr)
            rhob_corrected = np.append(rhob_corrected, rhob_corr)
            hc_flag = np.append(hc_flag, 1)
        else:
            nphi_corrected = np.append(nphi_corrected, point[0])
            rhob_corrected = np.append(rhob_corrected, point[1])
            hc_flag = np.append(hc_flag, 0)

    return nphi_corrected, rhob_corrected, hc_flag


def quick_qc(well_data):
    """Quick QC for well data.

    Args:
        well_data (pd.DataFrame): Well data.

    Returns:
        pd.DataFrame: Well data with QC flags.
    """
    return_df = well_data.copy()
    return_df['QC_FLAG'] = 0

    # Compare vsh_gr and vsh_dn
    return_df['VSH_GR'] = vsh_gr(return_df['GR'])
    return_df['QC_FLAG'] = np.where(abs(return_df['VSH_GR'] - return_df['VCLW']) > 0.1, 1, return_df['QC_FLAG'])

    # Compare phit_dn and phit_d
    if 'PHID' in return_df.columns:
        return_df['QC_FLAG'] = np.where(abs(return_df['PHIT'] - return_df['PHID']) > 0.1, 2, return_df['QC_FLAG'])

    # Check average swt in non-reservoir zone
    return_df['QC_FLAG'] = np.where((return_df['SAND_FLAG'] == 0) & (return_df['SWT'] < 0.8), 3, return_df['QC_FLAG'])

    # Summarize
    # return_df['QC_FLAG'] = return_df['QC_FLAG'].astype('category')
    summary_df = return_df.groupby('ZONES').agg({
        'VCLW': 'mean',
        'PHIT': 'mean',
        'SWT': 'mean',
        'SAND_FLAG': 'sum',
        'QC_FLAG': 'sum'
    })
    summary_df['QC_FLAG_mode'] = return_df.groupby('ZONES')['QC_FLAG'].agg(lambda x: x.mode())
    summary_df = summary_df.sort_values(by='QC_FLAG', ascending=False).reset_index().round(3)

    return return_df, summary_df
