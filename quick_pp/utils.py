import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import detrend
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import math


def min_max_line(feature, alpha: float = 0.05, num_bins: int = 1):
    """Calculates the minimum and maximum line of a feature.

    Args:
        feature (float): Input feature to calculate the minimum and maximum line.
        alpha (float, optional): Confidence interval. Defaults to 0.05.
        num_bins (int, optional): Number of bins. Defaults to 1.

    Returns:
        (float, float): Minimum and maximum line of a feature.
    """
    # Fill missing values with the median
    feature = np.where(np.isnan(feature), np.nanmedian(feature), feature)

    # Redefine the bins
    window = int(len(feature) / num_bins)
    bins = np.arange(0, len(feature), window)
    bins = np.append(bins, len(feature))

    min_lines = np.empty(0)
    max_lines = np.empty(0)
    # Enumerate over the bins
    for i, bin in enumerate(bins[:-1]):
        data = feature[bin: bins[i+1]]
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


def sand_flagging(data: pd.DataFrame):
    """Flagging sand zones based on VSHALE, VCLW, and VSH_GR.

    Returns:
        pd.DataFrame: Original DataFrame with SAND_FLAG and ZONES columns.
    """
    df = data.copy()
    if 'ZONES' not in df.columns:
        df['ZONES'] = 'FORMATION'
    else:
        df['ZONES'].fillna('FORMATION', inplace=True)

    return_df = pd.DataFrame()
    for _, well_data in df.groupby('WELL_NAME'):
        # Using VSH_GR
        dtr_gr = detrend(well_data[['GR']].fillna(well_data['GR'].median()), axis=0) + well_data['GR'].mean()
        well_data['VSH_GR'] = MinMaxScaler().fit_transform(dtr_gr)
        threshold = np.nanquantile(well_data['VSH_GR'], .75, method='median_unbiased')

        # Estimate SAND_FLAG using VSH_GR
        well_data['SAND_FLAG'] = np.where(well_data['VSH_GR'] < threshold, 1, -1)
        well_data['SAND_FLAG'] = np.where(
            well_data['SAND_FLAG'].rolling(13, center=True, closed='both').mean() > 0, 1, 0)

        # Fill in empty ZONES
        no_zones_df = well_data[well_data['ZONES'] == 'FORMATION'].copy()
        if not no_zones_df.empty:
            # Assign generic ZONES
            df_temp = pd.DataFrame()
            sand_counter = 1
            shale_counter = 0
            for i, data in no_zones_df.iterrows():
                if data['SAND_FLAG'] == 1 and shale_counter == 1:
                    sand_counter += 1
                    shale_counter = 0
                if data['SAND_FLAG'] == 1:
                    data['ZONES'] = f'SAND_{sand_counter}'
                else:
                    data['ZONES'] = f'SAND_{sand_counter}'
                    shale_counter = 1
                df_temp = pd.concat([df_temp, pd.DataFrame(data[['DEPTH', 'SAND_FLAG', 'ZONES']]).T], axis=0)
            df_temp.reset_index(drop=True, inplace=True)
            well_data.loc[well_data['DEPTH'].isin(df_temp['DEPTH']), 'ZONES'] = df_temp['ZONES']

        return_df = pd.concat([return_df, well_data], ignore_index=True)

    return return_df
