import numpy as np
import statsmodels.api as sm
from statsmodels.sandbox.regression.predstd import wls_prediction_std
import math


def min_max_line(feature, alpha: float = 0.05):
    """Calculates the minimum and maximum line of a feature.

    Args:
        feature (array): _description_
        alpha (float, optional): _description_. Defaults to 0.05.

    Returns:
        (array, array): Minimum and maximum line of a feature.
    """
    y = np.arange(0, len(feature))
    Y = sm.add_constant(y)
    modeltemp = sm.OLS(feature, Y).fit()
    prstd, min_line, max_line = wls_prediction_std(modeltemp, alpha=alpha)
    min_line = min_line.to_numpy()
    max_line = max_line.to_numpy()

    return min_line, max_line


def length_a_b(A: tuple, B: tuple):
    """Calculates the length of line between two points.

    Args:
        A (tuple): _description_
        B (tuple): _description_

    Returns:
        float: Length of line between two points.
    """    
    return math.sqrt(sum([(a - b) ** 2 for a, b in zip(A, B)]))


def line_intersection(line1, line2):
    """Calculates the intersection of two lines.

    Args:
        line1 (tuple of tuples): ((x11, y11), (x12, y12))
        line2 (tuple of tuples): ((x22, y22), (x22, y22))

    Raises:
        Exception: _description_

    Returns:
        float, float: Cartesian coordinates of the intersection of two lines.
    """    
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        raise Exception('lines do not intersect')

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y
