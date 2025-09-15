import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import detrend
import math
import ruptures.detection as rpt
import scipy.stats as stats
from scipy.signal import find_peaks

from quick_pp import logger


def length_a_b(A: tuple, B: tuple):
    """Calculates the length of line between two points.

    Args:
        A (tuple): Cartesian coordinates of point A.
        B (tuple): Cartesian coordinates of point B.

    Returns:
        float: Length of line between two points.
    """
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))


def line_intersection(line1: tuple, line2: tuple):
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
        logger.error(f'\r{line1} and {line2} lines do not intersect')
        return np.nan, np.nan

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


def angle_between_lines(line1: tuple, line2: tuple):
    """Calculates the convex angle between two lines in degrees.

    Args:
        line1 (tuple of tuples): ((x11, y11), (x12, y12))
        line2 (tuple of tuples): ((x21, y21), (x22, y22))

    Returns:
        float: Angle between the two lines in degrees.
    """
    def slope(line):
        (x1, y1), (x2, y2) = line
        if x2 == x1:  # Vertical line
            return None
        elif y2 == y1:  # Horizontal line
            return 0
        return (y2 - y1) / (x2 - x1)

    m1 = slope(line1)
    m2 = slope(line2)

    if m1 is None and m2 is None:
        return 0.0  # Both lines are vertical, angle is 0 degrees
    elif m1 is None:  # Line1 is vertical
        angle = 90.0 if m2 == 0 else math.degrees(math.atan(abs(1 / m2)))
    elif m2 is None:  # Line2 is vertical
        angle = 90.0 if m1 == 0 else math.degrees(math.atan(abs(1 / m1)))
    else:
        tan_theta = abs((m2 - m1) / (1 + m1 * m2))
        angle = math.degrees(math.atan(tan_theta))

    # Ensure the angle is convex (less than or equal to 180 degrees)
    if angle > 90.0:
        angle = 180.0 - angle

    return angle


def zone_flagging(data: pd.DataFrame, min_zone_thickness: int = 150):
    """Flagging sand zones based on VSHALE, VCLW, and VSH_GR.

    Args:
        data (pd.DataFrame): DataFrame with well log data.
        min_zone_thickness (int): The minimum number of data points for a zone to be kept.

    Returns:
        pd.DataFrame: Original DataFrame with ZONE_FLAG and ZONES columns.
    """
    df = data.copy()
    if 'ZONES' not in df.columns:
        df['ZONES'] = 'FORMATION'
    df['ZONES'].fillna('FORMATION', inplace=True)

    return_df = pd.DataFrame()
    if 'WELL_NAME' not in df.columns:
        df['WELL_NAME'] = 'WELL'
    df['WELL_NAME'].fillna('WELL', inplace=True)
    for _, well_data in df.groupby('WELL_NAME'):
        # Using VSHALE if available otherwise calculate VSH_GR
        if 'VSHALE' in well_data.columns:
            well_data['vsh_curve'] = well_data['VSHALE']
        else:
            dtr_gr = detrend(well_data[['GR']].fillna(well_data['GR'].median()), axis=0) + well_data['GR'].mean()
            well_data['vsh_curve'] = MinMaxScaler().fit_transform(dtr_gr)
        threshold = np.nanquantile(well_data['vsh_curve'], .7, method='median_unbiased')

        # Estimate RES_FLAG (reservoir flag) using vsh_curve
        well_data['RES_FLAG'] = np.where(well_data['vsh_curve'] < threshold, 1, 0)

        # Fill in empty ZONES
        no_zones_df = well_data[well_data['ZONES'] == 'FORMATION']
        if not no_zones_df.empty:
            # Assign initial zone names
            sand_zone_numbers = (well_data['RES_FLAG'].diff().clip(lower=0) == 1).cumsum()

            well_data['ZONES'] = np.where(
                well_data['RES_FLAG'] == 1,
                'ZONE_' + sand_zone_numbers.astype(str),
                np.nan
            )

            # Merge small zones
            zone_counts = well_data['ZONES'].value_counts()
            small_zones = zone_counts[zone_counts < min_zone_thickness].index

            while len(small_zones) > 0:
                well_data['ZONES'] = well_data['ZONES'].replace(small_zones, np.nan).ffill().bfill()
                zone_counts = well_data['ZONES'].value_counts()
                small_zones = zone_counts[zone_counts < min_zone_thickness].index

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


def inv_power_law_func(x, a, b):
    """Generic power law function.

    Args:
        x (float): Input variable.
        a (float): a constant.
        b (float): b constant.

    Returns:
        float: y = a * x^-b
    """
    return a * x**-b


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
            try:
                (min_line_slope, min_line_intercept), (max_line_slope, max_line_intercept) = fit_trendlines_single(data)
                min_line = straight_line_func(
                    np.arange(num_data), min_line_slope, min_line_intercept) + alpha * max(abs(data))
                max_line = straight_line_func(
                    np.arange(num_data), max_line_slope, max_line_intercept) - alpha * max(abs(data))
                min_lines = np.append(min_lines, min_line)
                max_lines = np.append(max_lines, max_line)
            except Exception as e:
                logger.error(f'Error: {e}')
                min_lines = np.append(min_lines, np.nan)
                max_lines = np.append(max_lines, np.nan)
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
