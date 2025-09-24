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
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))


def line_intersection(line1: tuple, line2: tuple):
    """Calculates the intersection of two lines.

    This function is vectorized to handle arrays of points.

    Args:
        line1 (tuple): A tuple containing two points, where each point can be a
                       tuple of coordinates or a NumPy array of coordinates.
                       Example: ((x11, y11), (x12, y12))
        line2 (tuple): A tuple containing two points for the second line.
                       Example: ((x21, y21), (x22, y22))

    Returns:
        (np.ndarray, np.ndarray): A tuple of NumPy arrays (x, y) for the
                                  intersection coordinates. If lines are parallel,
                                  coordinates will be np.nan.
    """
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    # Use np.divide for safe division, returning np.nan where div is 0
    # Create a mask for non-zero divisors to avoid warnings
    mask = div != 0
    d = (det(*line1), det(*line2))
    x = np.divide(det(d, xdiff), div, where=mask, out=np.full_like(div, np.nan, dtype=float))
    y = np.divide(det(d, ydiff), div, where=mask, out=np.full_like(div, np.nan, dtype=float))

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
    """Flagging sand zones based on VSHALE, VCLAY, and VSH_GR.

    Args:
        data (pd.DataFrame): DataFrame with well log data.
        min_zone_thickness (int): The minimum number of data points for a zone to be kept.

    Returns:
        pd.DataFrame: Original DataFrame with ZONE_FLAG and ZONES columns.
    """
    df = data.copy()
    if 'ZONES' not in df.columns:
        df['ZONES'] = 'FORMATION'
    df['ZONES'] = df['ZONES'].fillna('FORMATION')

    return_df = pd.DataFrame()
    if 'WELL_NAME' not in df.columns:
        df['WELL_NAME'] = 'WELL'
    df['WELL_NAME'] = df['WELL_NAME'].fillna('WELL')
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
    # Ensure feature is a numpy array and handle NaNs
    if isinstance(feature, pd.Series):
        feature_np = feature.to_numpy()
    else:
        feature_np = np.array(feature)

    segments = [feature_np]
    if auto_bin:
        signal_data = feature_np[~np.isnan(feature_np)]
        if len(signal_data) < 2:
            logger.warning("Not enough data for auto-binning. Processing as a single segment.")
            segments = [feature_np]
        else:
            # Estimating number of peaks
            try:
                kde = stats.gaussian_kde(signal_data)
                evaluated = kde.evaluate(np.linspace(np.nanmin(signal_data), np.nanmax(signal_data), 100))
                peaks, _ = find_peaks(evaluated, height=np.nanmedian(evaluated))

                # Detecting change points
                model = "l2"
                jump = len(feature_np) // (len(peaks) + 2)  # +2 to be safe
                algo = rpt.BottomUp(model=model, jump=jump).fit(feature_np)
                my_bkps = algo.predict(n_bkps=len(peaks))
                my_bkps = np.insert(my_bkps, 0, 0)

                segments = [feature_np[my_bkps[i]:my_bkps[i + 1]] for i in range(len(my_bkps) - 1)]
            except Exception as e:
                logger.error(f"Auto-binning failed: {e}. Processing as a single segment.")
                segments = [feature_np]

    min_lines_list = []
    max_lines_list = []
    # Enumerate over the bins
    for data_segment in segments:
        num_data = len(data_segment)
        if num_data > 0:
            try:
                clean_data = data_segment[~np.isnan(data_segment)]
                if len(clean_data) < 2:
                    raise ValueError("Not enough valid data in segment to fit trendlines.")

                (min_slope, min_intercept), (max_slope, max_intercept) = fit_trendlines_single(clean_data)
                x_range = np.arange(num_data)
                correction = alpha * np.nanmax(np.abs(clean_data))
                min_line = straight_line_func(x_range, min_slope, min_intercept) + correction
                max_line = straight_line_func(x_range, max_slope, max_intercept) - correction
                min_lines_list.append(min_line)
                max_lines_list.append(max_line)
            except Exception as e:
                logger.error(f'Error fitting trendline for a segment: {e}')
                min_lines_list.append(np.full(num_data, np.nan))
                max_lines_list.append(np.full(num_data, np.nan))
        else:
            min_lines_list.append(np.empty(0))
            max_lines_list.append(np.empty(0))

    return np.concatenate(min_lines_list), np.concatenate(max_lines_list)


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


def remove_straights(log, window: int = 30, threshold: float = 0.001):
    """Removes straight line sections from a log by replacing them with np.nan.

    This function identifies flat or "stuck" tool readings by calculating the
    standard deviation over a rolling window. Where the standard deviation is
    below a given threshold, the original log values in that window are
    flagged for removal.

    Args:
        log (pd.Series or np.array): The input log data.
        window (int, optional): The size of the rolling window. Defaults to 30.
        threshold (float, optional): The standard deviation threshold below which
                                     a segment is considered straight. Defaults to 0.001.

    Returns:
        np.array: The log with straight sections replaced by np.nan.
    """
    if not isinstance(log, pd.Series):
        log_series = pd.Series(log)
    else:
        log_series = log.copy()

    rolling_std = log_series.rolling(window=window, center=True, min_periods=window // 2).std()
    return np.where(rolling_std < threshold, np.nan, log_series.values)


def get_tvd(df, well_coords):
    """_summary_

    Args:
        df (_type_): _description_
        well_coords (_type_): _description_
    """
    import wellpathpy as wpp

    dev_survey = wpp.deviation(md=well_coords['md'], inc=well_coords['incl'], azi=well_coords['azim'])
    # Resample the survey at the exact MD points from the log data
    tvd = dev_survey.minimum_curvature().resample(df['DEPTH'].values).depth
    return tvd
