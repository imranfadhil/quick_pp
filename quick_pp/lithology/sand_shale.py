import numpy as np

from .. import Config
from ..utils import min_max_line, length_a_b, line_intersection
from ..porosity import neu_den_xplot_poro_pt
from ..plotter import neutron_density_xplot


def sand_shale_model(nphi, rhob,
                     dry_sand_point: tuple = None,
                     dry_clay_point: tuple = None,
                     fluid_point: tuple = None,
                     wet_clay_point: tuple = None,
                     silt_line_angle: float = None,
                     xplot: bool = False):
    """Estimate lithology volumetrics based on Kuttan's litho-porosity model

    Args:
        df (pd.DataFrame): _description_

    Raises:
        Exception: _description_

    Returns:
        _type_: _description_
    """
    # Initialize the endpoints
    A = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
    C = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
    D = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
    wetclay_pt = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
    silt_line_angle = silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]

    # Redefine wetclay point
    _, rhob_max_line = min_max_line(rhob, 0.05)
    _, nphi_max_line = min_max_line(nphi, 0.05)
    wetclay_RHOB = np.min(rhob_max_line)
    wetclay_NPHI = np.max(nphi_max_line)
    if not all(wetclay_pt):
        wetclay_pt = (wetclay_NPHI, wetclay_RHOB)
    # print(f'#### wetclay_pt: {wetclay_pt}')

    # Define dryclay point
    if not all(C):
        m = (D[1] - wetclay_pt[1]) / (D[0] - wetclay_pt[0])
        dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
        C = (dryclay_NPHI, C[1])
    # print(f'#### dryclay_pt: {C}, {m}')

    vsand, vcld = lithology_fraction(nphi, rhob, A, C, D)

    if xplot:
        fig = neutron_density_xplot(nphi, rhob, A, (0, 0), C, D, wetclay_pt)
        return vsand, vcld, fig
    else:
        return vsand, vcld, None


def lithology_fraction(nphi, rhob,
                       dry_sand_point: tuple = None,
                       dry_clay_point: tuple = None,
                       fluid_point: tuple = None):
    A = dry_sand_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))
    rocklithofrac = length_a_b(A, C)

    vsand = []
    vcld = []
    for i, point in enumerate(E):
        var_pt = line_intersection((A, C), (D, point))
        projlithofrac = length_a_b(var_pt, A)
        vshale = projlithofrac / rocklithofrac

        phit = neu_den_xplot_poro_pt(point[0], point[1], 'ss', None, A, (0, 0), C, D)
        vmatrix = 1 - phit
        vsand.append((1 - vshale) * vmatrix)
        vcld.append(vshale * vmatrix)

    return vsand, vcld
