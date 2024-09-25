import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import detrend
import math
import ruptures.detection as rpt
import scipy.stats as stats
from scipy.signal import find_peaks


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
        well_data['RES_FLAG'] = np.where(well_data['vsh_curve'] < threshold, 1, 0)
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


def min_max_line(feature, alpha: float = 0.05, auto_bin=False):
    """Calculates the minimum and maximum line of a feature, grouped based on change points.

    Args:
        feature (float): Input feature to calculate the minimum and maximum line.
        alpha (float, optional): Correction in percentage. Defaults to 0.05.
        auto_bin (bool, optional): Automatically bin the feature. Defaults to False.

    Returns:
        (float, float): Minimum and maximum line of a feature.
    """
    # Reset index if feature is a Series
    if isinstance(feature, pd.Series):
        feature.reset_index(drop=True, inplace=True)
    list_of_dfs = [feature]
    if auto_bin:
        # Estimating number of peaks
        signal_data = feature
        kde = stats.gaussian_kde(signal_data[~np.isnan(signal_data)])
        evaluated = kde.evaluate(np.linspace(np.nanmin(signal_data), np.nanmax(signal_data), 100))
        peaks, _ = find_peaks(evaluated, height=np.nanmedian(evaluated))

        # Detecting change points
        model = "l2"
        jump = len(signal_data) // (len(peaks) + 1)
        my_bkps = rpt.BottomUp(model=model, jump=jump).fit_predict(signal_data, n_bkps=len(peaks))
        my_bkps = np.insert(my_bkps, 0, 0)

        list_of_dfs = [feature[my_bkps[i]:my_bkps[i + 1]] for i in range(len(my_bkps) - 1)]

    min_lines = np.empty(0)
    max_lines = np.empty(0)
    # Enumerate over the bins
    for data in list_of_dfs:
        num_data = len(data)
        if num_data > 0:
            (min_line_slope, min_line_intercept), (max_line_slope, max_line_intercept) = fit_trendlines_single(data)
            min_line = straight_line_func(
                np.arange(num_data), min_line_slope, min_line_intercept) + alpha * max(abs(data))
            max_line = straight_line_func(
                np.arange(num_data), max_line_slope, max_line_intercept) - alpha * max(abs(data))
            min_lines = np.append(min_lines, min_line)
            max_lines = np.append(max_lines, max_line)
        else:
            min_lines = np.append(min_lines, np.nan)
            max_lines = np.append(max_lines, np.nan)
    return min_lines, max_lines


def check_trend_line(minimum: bool, pivot: int, slope: float, y: np.array):
    """Check if the trend line is valid. Based on TrendLineAutomation by neurotrader888.
    Computes sum of differences between trend line and feature, return negative val if invalid

    Args:
        minimum (bool): Whether to check for minimum or maximum.
        pivot (int): Anchor point of the trend line.
        slope (float): Slope of the trend line.
        y (np.array): Data to fit the trend line.

    Returns:
        float: Sum of differences between trend line and feature.
    """
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid.
    if (minimum and diffs.max() > 1e-5) or (not minimum and diffs.min() < -1e-5):
        return -1.0

    # Squared sum of diffs between data and line
    err = (diffs ** 2.0).sum()
    return err


def optimize_slope(minimum: bool, pivot: int, init_slope: float, y: np.array):
    """Optimize the slope of the trend line. Based on TrendLineAutomation by neurotrader888.

    Args:
        minimum (bool): Whether to optimize for minimum or maximum.
        pivot (int): Anchor point of the trend line.
        init_slope (float): Initial slope of the trend line.
        y (np.array): Data to fit the trend line.

    Raises:
        Exception: If the derivative fails.

    Returns:
        tuple: Tuple containing the slope and intercept of the optimized trend
    """
    # Optmization variables
    curr_step = 1.0
    min_step = 0.0001

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(minimum, pivot, init_slope, y)
    assert (best_err >= 0.0)  # Shouldn't ever fail with initial slope

    slope_unit = (y.max() - y.min()) / len(y)
    get_derivative = True
    derivative = None
    while curr_step > min_step:

        if get_derivative:
            # Numerical differentiation, increase slope by very small amount to see if error increases/ decreases.
            # Gives us the direction to change slope.
            slope_change = best_slope + slope_unit * min_step
            test_err = check_trend_line(minimum, pivot, slope_change, y)
            derivative = test_err - best_err

            # If increasing by a small amount fails, try decreasing by a small amount
            if test_err < 0.0:
                slope_change = best_slope - slope_unit * min_step
                test_err = check_trend_line(minimum, pivot, slope_change, y)
                derivative = best_err - test_err

            if test_err < 0.0:  # Derivative failed, give up
                raise Exception("Derivative failed. Check your data.")

            get_derivative = False

        if derivative > 0.0:  # Increasing slope increased error
            test_slope = best_slope - slope_unit * curr_step
        else:  # Increasing slope decreased error
            test_slope = best_slope + slope_unit * curr_step

        test_err = check_trend_line(minimum, pivot, test_slope, y)
        if test_err < 0 or test_err >= best_err:
            # Slope failed/ didn't reduce error
            curr_step *= 0.5  # Reduce step size
        else:  # Test slope reduced error
            best_err = test_err
            best_slope = test_slope
            get_derivative = True  # Recompute derivative

    # Optimize done, return best slope and intercept
    return (best_slope, -best_slope * pivot + y[pivot])


def fit_trendlines_single(data: np.array):
    """Find line of best fit (least squared). Based on TrendLineAutomation by neurotrader888.

    Args:
        data (np.array): Curve to fit the trend lines.

    Returns:
        tuple: Tuples containing the slope and intercept of the min and max trend line.
    """
    x = np.arange(len(data))
    coefs = np.polyfit(x, data, 1)
    slope = coefs[0]
    intercept = coefs[1]

    # Get points of line.
    line_points = slope * x + intercept

    # Find upper and lower pivot points
    upper_pivot = (data - line_points).argmax()
    lower_pivot = (data - line_points).argmin()

    # Optimize the slope for both trend lines
    min_line_coefs = optimize_slope(True, lower_pivot, slope, data)
    max_line_coefs = optimize_slope(False, upper_pivot, slope, data)

    return (min_line_coefs, max_line_coefs)
