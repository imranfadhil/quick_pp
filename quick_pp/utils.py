import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import detrend
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import math
import ruptures.detection as rpt
import scipy.stats as stats
from scipy.signal import find_peaks


def min_max_line(feature, alpha: float = 0.1):
    """Calculates the minimum and maximum line of a feature, grouped based on change points.

    Args:
        feature (float): Input feature to calculate the minimum and maximum line.
        alpha (float, optional): Confidence interval. Defaults to 0.05.

    Returns:
        (float, float): Minimum and maximum line of a feature.
    """
    # Fill missing values with the median
    feature = np.where(np.isnan(feature), np.nanmedian(feature), feature)

    # Estimating number of peaks
    signal_data = feature
    kde = stats.gaussian_kde(signal_data[~np.isnan(signal_data)])
    evaluated = kde.evaluate(np.linspace(np.nanmin(signal_data), np.nanmax(signal_data), 100))
    peaks, _ = find_peaks(evaluated, height=np.nanmedian(evaluated))

    # Detecting change points
    model = "rbf"
    jump = 500
    my_bkps = rpt.BottomUp(model=model, jump=jump).fit_predict(signal_data, n_bkps=len(peaks))
    my_bkps = np.insert(my_bkps, 0, 0)

    list_of_dfs = [feature[my_bkps[i]:my_bkps[i + 1]] for i in range(len(my_bkps) - 1)]

    min_lines = np.empty(0)
    max_lines = np.empty(0)
    # Enumerate over the bins
    for data in list_of_dfs:
        y = np.arange(0, len(data))
        if y.size > 0:
            Y = sm.add_constant(y)
            modeltemp = sm.OLS(data, Y).fit()
            prstd, min_line, max_line = wls_prediction_std(modeltemp, alpha=alpha)
            if not isinstance(min_line, np.ndarray):
                min_line = min_line.to_numpy()
                max_line = max_line.to_numpy()
            min_lines = np.append(min_lines, min_line)
            max_lines = np.append(max_lines, max_line)
        else:
            min_lines = np.append(min_lines, np.nan)
            max_lines = np.append(max_lines, np.nan)

    return min_lines, max_lines


def length_a_b(A: tuple, B: tuple):
    """Calculates the length of line between two points.

    Args:
        A (tuple): Cartesian coordinates of point A.
        B (tuple): Cartesian coordinates of point B.

    Returns:
        float: Length of line between two points.
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))


def line_intersection(line1, line2):
    """Calculates the intersection of two lines.

    Args:
        line1 (tuple of tuples): ((x11, y11), (x12, y12))
        line2 (tuple of tuples): ((x22, y22), (x22, y22))

    Returns:
        float, float: Cartesian coordinates of the intersection of two lines.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        print(f'\r{line1} and {line2} lines do not intersect', end='')
        return np.nan, np.nan

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def zone_flagging(data: pd.DataFrame):
    """Flagging sand zones based on VSHALE, VCLW, and VSH_GR.

    Returns:
        pd.DataFrame: Original DataFrame with ZONE_FLAG and ZONES columns.
    """
    df = data.copy()
    if 'ZONES' not in df.columns:
        df['ZONES'] = 'FORMATION'
    df['ZONES'].fillna('FORMATION', inplace=True)

    return_df = pd.DataFrame()
    for _, well_data in df.groupby('WELL_NAME'):
        # Using VSHALE if available otherwise calculate VSH_GR
        if 'VSHALE' in well_data.columns:
            well_data['vsh_curve'] = well_data['VSHALE']
        else:
            dtr_gr = detrend(well_data[['GR']].fillna(well_data['GR'].median()), axis=0) + well_data['GR'].mean()
            well_data['vsh_curve'] = MinMaxScaler().fit_transform(dtr_gr)
        threshold = np.nanquantile(well_data['vsh_curve'], .7, method='median_unbiased')

        # Estimate ZONE_FLAG using VSH_GR
        well_data['SAND_FLAG'] = np.where(well_data['vsh_curve'] < threshold, 1, 0)
        well_data['ZONES_FLAG'] = np.where(
            well_data['vsh_curve'].rolling(25, win_type='boxcar', center=True).mean() < threshold, 1, 0)
        well_data['ZONES_FLAG'] = well_data['ZONES_FLAG'].fillna(0)

        # Fill in empty ZONES
        no_zones_df = well_data[well_data['ZONES'] == 'FORMATION']
        if not no_zones_df.empty:
            # Assign generic ZONES
            df_temp = pd.DataFrame()
            sand_counter = 1
            shale_counter = 0
            for i, data in no_zones_df.iterrows():
                if data['ZONES_FLAG'] == 1 and shale_counter == 1:
                    sand_counter += 1
                    shale_counter = 0
                if data['ZONES_FLAG'] == 1:
                    data['ZONES'] = f'ZONE_{sand_counter}'
                else:
                    data['ZONES'] = f'ZONE_{sand_counter}'
                    shale_counter = 1
                df_temp = pd.concat([df_temp, pd.DataFrame(data[['DEPTH', 'ZONES']]).T], axis=0)
            df_temp.reset_index(drop=True, inplace=True)
            well_data.loc[well_data['DEPTH'].isin(df_temp['DEPTH']), 'ZONES'] = df_temp['ZONES']

            # Replace small 'ZONE_'s with np.nan and ffill
            remove_sands = well_data[well_data['ZONES'].str.contains('ZONE_', na=False)][['DEPTH', 'ZONES']].groupby(
                'ZONES').count().reset_index()
            remove_sands = remove_sands[remove_sands['DEPTH'] < 70]
            well_data['ZONES'] = well_data['ZONES'].apply(
                lambda x: x if x not in remove_sands['ZONES'].to_list() else np.nan)
            well_data['ZONES'] = well_data['ZONES'].ffill().bfill()

        return_df = pd.concat([return_df, well_data], ignore_index=True)

    return return_df


def power_law_func(x, a, b):
    """Generic power law function.

    Args:
        x (float): Input variable.
        a (float): a constant.
        b (float): b constant.

    Returns:
        float: y = a * x^b
    """
    return a * x**b


def straight_line_func(x, m, c):
    """Generic straight line function.

    Args:
        x (float): Input variable.
        m (float): Slope of the line.
        c (float): Y intercept of the line.

    Returns:
        float: y = m * x + c
    """
    return m * x + c
