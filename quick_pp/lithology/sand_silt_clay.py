import numpy as np
import math

from ..config import Config
from ..utils import min_max_line, length_a_b, line_intersection
from ..porosity import neu_den_xplot_poro_pt, clay_porosity
from ..plotter import neutron_density_xplot


def sand_silt_clay_model(nphi, rhob, model: str = 'kuttan',
                         dry_sand_point: tuple = None,
                         dry_silt_point: tuple = None,
                         dry_clay_point: tuple = None,
                         fluid_point: tuple = None,
                         wet_clay_point: tuple = None,
                         silt_line_angle: float = None,
                         xplot: bool = False,
                         normalize: bool = True):
    """Estimate lithology volumetrics based on Kuttan's and simplification of PCSB's litho-porosity model

    Args:
        nphi (float or array): Neutron Porosity log in v/v
        rhob (float or array): Bulk Density log in g/cc
        model (str, optional): Model to choose from 'kuttan' or 'kuttan_modified'. Defaults to 'kuttan'.
        dry_sand_point (tuple, optional): _description_. Defaults to None.
        dry_silt_point (tuple, optional): _description_. Defaults to None.
        dry_clay_point (tuple, optional): _description_. Defaults to None.
        fluid_point (tuple, optional): _description_. Defaults to None.
        wet_clay_point (tuple, optional): _description_. Defaults to None.
        silt_line_angle (float, optional): _description_. Defaults to None.
        xplot (bool, optional): To plot Neutron Density cross plot. Defaults to False.

    Returns:
        _type_: _description_
    """
    assert model in ['kuttan', 'kuttan_modified'], f"'{model}' model is not available."
    # Initialize the endpoints
    A = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
    B = dry_silt_point or Config.SSC_ENDPOINTS["DRY_SILT_POINT"]
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

    # Redefine drysilt point
    drysilt_NPHI = 1 - 1.68*(math.tan(float(silt_line_angle - 90)*math.pi / 180))
    if not all(B):
        B = (drysilt_NPHI, B[1])
    # print(f'#### drysilt_pt: {drysilt_pt}')

    # Define dryclay point
    if not all(C):
        m = (D[1] - wetclay_pt[1]) / (D[0] - wetclay_pt[0])
        dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
        C = (dryclay_NPHI, C[1])
    # print(f'#### dryclay_pt: {C}, {m}')

    if model == 'kuttan':
        vsand, vsilt, vcld = lithology_fraction_kuttan(nphi, rhob, A, B, C, D, normalize)
    else:
        vsand, vsilt, vcld = lithology_fraction_kuttan_modified(nphi, rhob, A, B, C, D, normalize)

    # Calculate vclb: volume of clay bound water
    clay_phit = clay_porosity(rhob_max_line, C[1])
    vclb = vcld * clay_phit

    if xplot:
        fig = neutron_density_xplot(nphi, rhob, A, B, C, D, wetclay_pt)
        return vsand, vsilt, vcld, vclb, fig
    else:
        return vsand, vsilt, vcld, vclb, None


def lithology_fraction_kuttan(nphi, rhob,
                              dry_sand_point: tuple = None,
                              dry_silt_point: tuple = None,
                              dry_clay_point: tuple = None,
                              fluid_point: tuple = None,
                              normalize: bool = True):
    """_summary_

    Args:
        nphi (_type_): _description_
        rhob (_type_): _description_
        dry_sand_point (tuple, optional): _description_. Defaults to None.
        dry_silt_point (tuple, optional): _description_. Defaults to None.
        dry_clay_point (tuple, optional): _description_. Defaults to None.
        fluid_point (tuple, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))
    rocklithofrac = length_a_b(A, C)
    sandsiltfrac = length_a_b(A, B)
    matrix_ratio_x = sandsiltfrac / rocklithofrac

    vsand = []
    vsilt = []
    vcld = []
    for i, point in enumerate(E):
        var_pt = line_intersection((A, C), (D, point))
        projlithofrac = length_a_b(var_pt, A)
        matrix_ratio = projlithofrac / rocklithofrac
        if matrix_ratio < matrix_ratio_x:
            phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', True, A, B, C, D) if normalize else 0
            vmatrix = 1 - phit
            vsilt.append((matrix_ratio / matrix_ratio_x) * vmatrix)
            vsand.append((1 - vsilt[i]) * vmatrix)
            vcld.append(0)
        else:
            phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', False, A, B, C, D) if normalize else 0
            vmatrix = 1 - phit
            vsand.append(0)
            vsilt.append(((1 - matrix_ratio) * vmatrix) / (1 - matrix_ratio_x))
            vcld.append((1 - vsilt[i]) * vmatrix)

    return vsand, vsilt, vcld


def lithology_fraction_kuttan_modified(
        nphi, rhob,
        dry_sand_point: tuple = None,
        dry_silt_point: tuple = None,
        dry_clay_point: tuple = None,
        fluid_point: tuple = None,
        normalize: bool = True):
    """_summary_

    Args:
        nphi (float or array): _description_
        rhob (float or array): _description_
        dry_sand_point (tuple, optional): _description_. Defaults to None.
        dry_silt_point (tuple, optional): _description_. Defaults to None.
        dry_clay_point (tuple, optional): _description_. Defaults to None.
        fluid_point (tuple, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))

    siltclayratio = 0.4  # empirical value
    claysiltfrac = length_a_b(C, B)
    rocklithofrac = length_a_b(A, C)
    sandsiltfrac = length_a_b(A, B)
    matrix_ratio_x = sandsiltfrac / rocklithofrac

    vsand = []
    vsilt = []
    vcld = []
    for i, point in enumerate(E):
        var_pt = line_intersection((A, C), (D, point))
        projlithofrac = length_a_b(var_pt, A)
        matrix_ratio = projlithofrac / rocklithofrac
        if matrix_ratio < matrix_ratio_x:
            phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', True, A, B, C, D) if normalize else 0
            vmatrix = 1 - phit
            vsand.append(((-projlithofrac / sandsiltfrac) + 1) * vmatrix)
            vsilt.append(siltclayratio / sandsiltfrac * projlithofrac * vmatrix)
            vcld.append(((1 - siltclayratio) / sandsiltfrac * projlithofrac) * vmatrix)
        else:
            phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', False, A, B, C, D) if normalize else 0
            vmatrix = 1 - phit
            vsand.append(0)
            vsilt.append((-siltclayratio / claysiltfrac * projlithofrac + (
                rocklithofrac * siltclayratio / claysiltfrac)) * vmatrix)
            vcld.append((siltclayratio / claysiltfrac * projlithofrac + (
                1 - (siltclayratio * rocklithofrac / claysiltfrac))) * vmatrix)

    return vsand, vsilt, vcld
