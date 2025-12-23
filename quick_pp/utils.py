import math

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.signal import detrend, find_peaks

from quick_pp import logger


def r2_score(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculates the R-squared (coefficient of determination) between true and predicted values.

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.
    Returns:
        float: The R-squared value.
    """
    ss_res = np.nansum((y_true - y_pred) ** 2)
    ss_tot = np.nansum((y_true - np.nanmean(y_true)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    return r2


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray):
    """Calculates the Root Mean Squared Error (RMSE) between true and predicted values.

    Args:
        y_true (np.ndarray): The ground truth values.
        y_pred (np.ndarray): The predicted values.
    Returns:
        float: The RMSE value.
    """
    mse = np.nanmean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def robust_scale(series: pd.Series, quantile_range=(25.0, 75.0)):
    """Scales a pandas Series using robust scaling (median and IQR).

    Mirrors sklearn.preprocessing.RobustScaler behaviour for a single series:
    - uses the median and the IQR defined by `quantile_range` (defaults 25-75)
    - preserves NaNs when computing statistics
    - if IQR is zero or NaN, centering is applied but scaling is skipped

    Args:
        series (pd.Series): The input pandas Series to be scaled.
        quantile_range (tuple): pair of percentiles (low, high) used to compute IQR.

    Returns:
        pd.Series: The scaled pandas Series (dtype float).
    """
    q_min, q_max = quantile_range
    median = series.median()
    q1 = series.quantile(q_min / 100.0)
    q3 = series.quantile(q_max / 100.0)
    iqr = q3 - q1

    # Follow sklearn behaviour: if scale is zero or undefined, do not divide
    if iqr == 0 or pd.isna(iqr):
        return (series - median).astype(float)

    return ((series - median) / iqr).astype(float)


def minmax_scale(series: pd.Series):
    """Scales a pandas Series to the range [0, 1] using min-max scaling.

    Args:
        series (pd.Series): The input pandas Series to be scaled.

    Returns:
        pd.Series: The scaled pandas Series with values between 0 and 1.
    """
    min_val = series.min()
    max_val = series.max()
    scaled_series = (series - min_val) / (max_val - min_val)
    return scaled_series


def length_a_b(A: tuple, B: tuple):
    """Calculate the Euclidean distance between two points.

    Args:
        A (tuple): Cartesian coordinates of point A.
        B (tuple): Cartesian coordinates of point B.

    Returns:
        float: The distance between points A and B.
    """
    return np.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B, strict=True)]))


def line_intersection(line1: tuple, line2: tuple):
    """Calculates the intersection of two lines.

    This function is vectorized to handle arrays of points.

    Args:
        line1 (tuple): A tuple of two points defining the first line, e.g., `((x1, y1), (x2, y2))`.
                       Coordinates can be single values or arrays.
        line2 (tuple): A tuple of two points defining the second line.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple of NumPy arrays `(x, y)` for the
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
    x = np.divide(
        det(d, xdiff), div, where=mask, out=np.full_like(div, np.nan, dtype=float)
    )
    y = np.divide(
        det(d, ydiff), div, where=mask, out=np.full_like(div, np.nan, dtype=float)
    )

    return x, y


def angle_between_lines(line1: tuple, line2: tuple):
    """Calculates the convex angle between two lines in degrees.

    Args:
        line1 (tuple): A tuple of two points defining the first line, e.g., `((x1, y1), (x2, y2))`.
        line2 (tuple): A tuple of two points defining the second line.

    Returns:
        float: The angle between the two lines in degrees (between 0 and 90).
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
    """Identify and flag sand zones based on shale volume.

    This function uses a shale volume curve (VSHALE or estimated VSH_GR) to
    delineate reservoir zones. It identifies continuous sand intervals and merges
    smaller zones into larger adjacent ones based on a minimum thickness criterion.

    Args:
        data (pd.DataFrame): The input DataFrame with well log data.
        min_zone_thickness (int, optional): The minimum number of data points for a zone to be considered valid. Defaults to 150.

    Returns:
        pd.DataFrame: The DataFrame with added 'ZONE_FLAG' and updated 'ZONES' columns.
    """
    df = data.copy()
    if "ZONES" not in df.columns:
        df["ZONES"] = "FORMATION"
    df["ZONES"] = df["ZONES"].fillna("FORMATION")

    return_df = pd.DataFrame()
    if "WELL_NAME" not in df.columns:
        df["WELL_NAME"] = "WELL"
    df["WELL_NAME"] = df["WELL_NAME"].fillna("WELL")
    for _, well_data in df.groupby("WELL_NAME"):
        # Using VSHALE if available otherwise calculate VSH_GR
        if "VSHALE" in well_data.columns:
            well_data["vsh_curve"] = well_data["VSHALE"]
        else:
            dtr_gr = (
                detrend(well_data[["GR"]].fillna(well_data["GR"].median()), axis=0)
                + well_data["GR"].mean()
            )
            well_data["vsh_curve"] = minmax_scale(pd.Series(dtr_gr.flatten()))
        threshold = np.nanquantile(
            well_data["vsh_curve"], 0.7, method="median_unbiased"
        )

        # Estimate RES_FLAG (reservoir flag) using vsh_curve
        well_data["RES_FLAG"] = np.where(well_data["vsh_curve"] < threshold, 1, 0)

        # Fill in empty ZONES
        no_zones_df = well_data[well_data["ZONES"] == "FORMATION"]
        if not no_zones_df.empty:
            # Assign initial zone names
            sand_zone_numbers = (
                well_data["RES_FLAG"].diff().clip(lower=0) == 1
            ).cumsum()

            well_data["ZONES"] = np.where(
                well_data["RES_FLAG"] == 1,
                "ZONE_" + sand_zone_numbers.astype(str),
                np.nan,
            )

            # Merge small zones
            zone_counts = well_data["ZONES"].value_counts()
            small_zones = zone_counts[zone_counts < min_zone_thickness].index

            while len(small_zones) > 0:
                well_data["ZONES"] = (
                    well_data["ZONES"].replace(small_zones, np.nan).ffill().bfill()
                )
                zone_counts = well_data["ZONES"].value_counts()
                small_zones = zone_counts[zone_counts < min_zone_thickness].index

        return_df = pd.concat([return_df, well_data], ignore_index=True)

    return return_df


def power_law_func(x, a, b):
    """Generic power law function.

    Args:
        x (np.ndarray or float): Input variable.
        a (float): a constant.
        b (float): b constant.

    Returns:
        np.ndarray or float: `y = a * x^b`
    """
    return a * x**b


def inv_power_law_func(x, a, b):
    """Generic power law function.

    Args:
        x (np.ndarray or float): Input variable.
        a (float): a constant.
        b (float): b constant.

    Returns:
        np.ndarray or float: `y = a * x^-b`
    """
    return a * x**-b


def straight_line_func(x, m, c):
    """Generic straight line function.

    Args:
        x (np.ndarray or float): Input variable.
        m (float): Slope of the line.
        c (float): Y intercept of the line.

    Returns:
        np.ndarray or float: `y = m * x + c`
    """
    return m * x + c


def min_max_line(feature, alpha: float = 0.05, auto_bin=False):
    """Calculate the minimum and maximum trend lines for a given feature.

    This function can optionally segment the data based on change points before
    fitting trend lines to each segment.

    Args:
        feature (np.ndarray or float): Input feature to calculate the minimum and maximum line.
        alpha (float, optional): Correction in percentage. Defaults to 0.05.
        auto_bin (bool, optional): Automatically bin the feature. Defaults to False.

    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing the minimum and maximum trend lines.
    """
    import ruptures.detection as rpt

    # Ensure feature is a numpy array and handle NaNs
    if isinstance(feature, pd.Series):
        feature_np = feature.to_numpy()
    else:
        feature_np = np.array(feature)

    segments = [feature_np]
    if auto_bin:
        signal_data = feature_np[~np.isnan(feature_np)]
        if len(signal_data) < 2:
            logger.warning(
                "Not enough data for auto-binning. Processing as a single segment."
            )
            segments = [feature_np]
        else:
            # Estimating number of peaks
            try:
                kde = stats.gaussian_kde(signal_data)
                evaluated = kde.evaluate(
                    np.linspace(np.nanmin(signal_data), np.nanmax(signal_data), 100)
                )
                peaks, _ = find_peaks(evaluated, height=np.nanmedian(evaluated))

                # Detecting change points
                model = "l2"
                jump = len(feature_np) // (len(peaks) + 2)  # +2 to be safe
                algo = rpt.BottomUp(model=model, jump=jump).fit(feature_np)
                my_bkps = algo.predict(n_bkps=len(peaks))
                my_bkps = np.insert(my_bkps, 0, 0)

                segments = [
                    feature_np[my_bkps[i] : my_bkps[i + 1]]
                    for i in range(len(my_bkps) - 1)
                ]
            except Exception as e:
                logger.error(
                    f"Auto-binning failed: {e}. Processing as a single segment."
                )
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
                    raise ValueError(
                        "Not enough valid data in segment to fit trendlines."
                    )

                (min_slope, min_intercept), (max_slope, max_intercept) = (
                    fit_trendlines_single(clean_data)
                )
                x_range = np.arange(num_data)
                correction = alpha * np.nanmax(np.abs(clean_data))
                min_line = (
                    straight_line_func(x_range, min_slope, min_intercept) + correction
                )
                max_line = (
                    straight_line_func(x_range, max_slope, max_intercept) - correction
                )
                min_lines_list.append(min_line)
                max_lines_list.append(max_line)
            except Exception as e:
                logger.error(f"Error fitting trendline for a segment: {e}")
                min_lines_list.append(np.full(num_data, np.nan))
                max_lines_list.append(np.full(num_data, np.nan))
        else:
            min_lines_list.append(np.empty(0))
            max_lines_list.append(np.empty(0))

    return np.concatenate(min_lines_list), np.concatenate(max_lines_list)


def check_trend_line(minimum: bool, pivot: int, slope: float, y: np.array):
    """Check if a trend line is valid and calculate its error.

    This function, based on TrendLineAutomation by neurotrader888, computes the
    sum of squared differences between a trend line and the data, returning -1.0
    if the line is invalid.

    Args:
        minimum (bool): If True, checks for a minimum trend line (support); otherwise, a maximum (resistance).
        pivot (int): Anchor point of the trend line.
        slope (float): Slope of the trend line.
        y (np.array): Data to fit the trend line.

    Returns:
        float: The sum of squared differences, or -1.0 if the line is invalid.
    """
    # Find the intercept of the line going through pivot point with given slope
    intercept = -slope * pivot + y[pivot]
    line_vals = slope * np.arange(len(y)) + intercept

    diffs = line_vals - y

    # Check to see if the line is valid, return -1 if it is not valid.
    if (minimum and diffs.max() > 1e-5) or (not minimum and diffs.min() < -1e-5):
        return -1.0

    # Squared sum of diffs between data and line
    err = (diffs**2.0).sum()
    return err


def optimize_slope(minimum: bool, pivot: int, init_slope: float, y: np.array):
    """Optimize the slope of a trend line to best fit the data.

    This function, based on TrendLineAutomation by neurotrader888, iteratively
    adjusts the slope to minimize the error while maintaining the line's validity
    as either a support or resistance line.

    Args:
        minimum (bool): If True, optimizes for a minimum trend line; otherwise, a maximum.
        pivot (int): Anchor point of the trend line.
        init_slope (float): Initial slope of the trend line.
        y (np.array): Data to fit the trend line.

    Returns:
        tuple: Tuple containing the slope and intercept of the optimized trend
    """
    # Optmization variables
    curr_step = 1.0
    min_step = 0.0001

    # Initiate at the slope of the line of best fit
    best_slope = init_slope
    best_err = check_trend_line(minimum, pivot, init_slope, y)
    assert best_err >= 0.0  # Shouldn't ever fail with initial slope

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
    """Find the best-fit minimum and maximum trend lines for a single data segment.

    This function, based on TrendLineAutomation by neurotrader888, uses a least-squares
    fit to find an initial trend and then optimizes for the tightest possible support
    and resistance lines.

    Args:
        data (np.ndarray): The data segment to fit the trend lines to.

    Returns:
        tuple: A tuple of tuples containing the (slope, intercept) for the minimum and maximum trend lines.
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
        log (pd.Series or np.ndarray): The input log data.
        window (int, optional): The size of the rolling window. Defaults to 30.
        threshold (float, optional): The standard deviation threshold below which
                                     a segment is considered straight. Defaults to 0.001.

    Returns:
        np.ndarray: The log with straight sections replaced by np.nan.
    """
    if not isinstance(log, pd.Series):
        log_series = pd.Series(log)
    else:
        log_series = log.copy()

    rolling_std = log_series.rolling(
        window=window, center=True, min_periods=window // 2
    ).std()
    return np.where(rolling_std < threshold, np.nan, log_series.values)


def get_tvd(df, well_coords):
    """Calculate true vertical depth (TVD) given well survey data.

    Args:
        df (pd.DataFrame): A DataFrame with a 'DEPTH' column.
        well_coords (pd.DataFrame): A DataFrame with 'md', 'incl', and 'azim' columns for the deviation survey.

    Returns:
        np.ndarray: An array of TVD values corresponding to the input depths.
    """
    import wellpathpy as wpp

    dev_survey = wpp.deviation(
        md=well_coords["md"], inc=well_coords["incl"], azi=well_coords["azim"]
    )
    # Resample the survey at the exact MD points from the log data
    df = df[df.DEPTH.between(well_coords["md"].min(), well_coords["md"].max())]
    tvd = dev_survey.minimum_curvature().resample(df["DEPTH"].values).depth
    return tvd


def remove_outliers(curve):
    """Removes outliers from a curve using the IQR method and forward fills the NaNs.

    Args:
        curve (np.ndarray): The input curve data.

    Returns:
        pd.Series: The curve with outliers replaced by NaN and then forward filled.
    """
    q1, q3 = np.nanquantile(curve, [0.25, 0.75])
    iqr = q3 - q1
    upper_bound = q3 + 1.5 * iqr
    lower_bound = q1 - 1.5 * iqr

    # Replace outliers with NaN and then forward fill
    curve_series = pd.Series(curve)
    return curve_series.where(
        (curve_series >= lower_bound) & (curve_series <= upper_bound)
    )
