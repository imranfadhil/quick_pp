import numpy as np
import math
import statistics
from scipy.optimize import minimize
from scipy.signal import detrend
from sklearn.preprocessing import MinMaxScaler

from .utils import min_max_line, length_a_b, line_intersection
from .porosity import neu_den_xplot_poro_pt, clay_porosity
from .plotter import neutron_density_xplot
from .config import Config


class SandSiltClay:
    """Sand-silt-clay model based on Kuttan's litho-porosity model.

     The original is a binary model where reservoir sections (left to the silt line) consist of combination of
     sand-silt while non-reservoir sections (right to the silt line) consist of silt-clay combination.

     PCSB proposed a model based on Kuttan, where the reservoir sections consist of combination of sand-silt-clay
     while non-reservoir sections remain a combination of silt-clay only. This module is a simplified version of the
     original PCSB model."""

    def __init__(self, dry_sand_point: tuple = None, dry_silt_point: tuple = None, dry_clay_point: tuple = None,
                 fluid_point: tuple = None, wet_clay_point: tuple = None, silt_line_angle: float = None):
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_silt_point = dry_silt_point or Config.SSC_ENDPOINTS["DRY_SILT_POINT"]
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]

    def estimate_lithology(self, nphi, rhob, model: str = 'kuttan', xplot: bool = False, normalize: bool = True):
        """Estimate sand silt clay lithology volumetrics.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            model (str, optional): Model to choose from 'kuttan' or 'kuttan_modified'. Defaults to 'kuttan'.
            xplot (bool, optional): To plot Neutron Density cross plot. Defaults to False.
            normalize (bool, optional): To normalize with porosity. Defaults to True.

        Returns:
            (float, float, float, boolean): vsand, vsilt, vcld, vclb, cross-plot if xplot True else None
        """
        assert model in ['kuttan', 'kuttan_modified'], f"'{model}' model is not available."
        # Initialize the endpoints
        A = self.dry_sand_point
        B = self.dry_silt_point
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point
        _, rhob_max_line = min_max_line(rhob, 0.05)
        _, nphi_max_line = min_max_line(nphi, 0.05)
        wetclay_RHOB = np.min(rhob_max_line)
        wetclay_NPHI = np.max(nphi_max_line)
        if not all(self.wet_clay_point):
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
        # print(f'#### wet_clay_point: {wet_clay_point}')

        # Redefine drysilt point
        drysilt_NPHI = 1 - 1.68*(math.tan(float(self.silt_line_angle - 90)*math.pi / 180))
        if not all(B):
            B = self.dry_silt_point = (drysilt_NPHI, B[1])
        # print(f'#### drysilt_pt: {drysilt_pt}')

        # Define dryclay point
        if not all(C):
            m = (D[1] - self.wet_clay_point[1]) / (D[0] - self.wet_clay_point[0])
            dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
            C = self.dry_clay_point = (dryclay_NPHI, C[1])
        # print(f'#### dryclay_pt: {C}, {m}')

        if model == 'kuttan':
            vsand, vsilt, vcld = self.lithology_fraction_kuttan(nphi, rhob, normalize)
        else:
            vsand, vsilt, vcld = self.lithology_fraction_kuttan_modified(nphi, rhob, normalize)

        # Calculate vclb: volume of clay bound water
        clay_phit = clay_porosity(rhob_max_line, C[1])
        vclb = vcld * clay_phit

        if xplot:
            fig = neutron_density_xplot(nphi, rhob, A, B, C, D, self.wet_clay_point)
            return vsand, vsilt, vcld, vclb, fig
        else:
            return vsand, vsilt, vcld, vclb, None

    def lithology_fraction_kuttan(self, nphi, rhob, normalize: bool = True):
        """Estimate lithology volumetrics based on Kuttan's litho-porosity model.

        Args:
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.
            normalize (bool, optional): To normalize with porosity. Defaults to True.

        Returns:
            (float, float, float): vsand, vsilt, vcld
        """
        A = self.dry_sand_point
        B = self.dry_silt_point
        C = self.dry_clay_point
        D = self.fluid_point
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

    def lithology_fraction_kuttan_modified(self, nphi, rhob, normalize: bool = True):
        """Estimate lithology volumetrics based on modified Kuttan's litho-porosity model
        (simplification of PCSB's litho-porosity model).

        Args:
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.
            normalize (bool, optional): To normalize with porosity. Defaults to True.

        Returns:
            (float, float, float): vsand, vsilt, vcld
        """
        A = self.dry_sand_point
        B = self.dry_silt_point
        C = self.dry_clay_point
        D = self.fluid_point
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


class SandShale:
    """This binary model only consider a combination of sand-shale components. """

    def __init__(self, dry_sand_point: tuple = None, dry_clay_point: tuple = None,
                 fluid_point: tuple = None, wet_clay_point: tuple = None, silt_line_angle: float = None):
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]

    def estimate_lithology(self, nphi, rhob, xplot: bool = False):
        """Estimate lithology volumetrics based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            xplot (bool, optional): To plot Neutron Density cross plot. Defaults to False.

        Returns:
            (float, float): vsand, vcld, cross-plot if xplot True else None
        """
        # Initialize the endpoints
        A = self.dry_sand_point
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point
        _, rhob_max_line = min_max_line(rhob, 0.05)
        _, nphi_max_line = min_max_line(nphi, 0.05)
        wetclay_RHOB = np.min(rhob_max_line)
        wetclay_NPHI = np.max(nphi_max_line)
        if not all(self.wet_clay_point):
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
        # print(f'#### wet_clay_point: {wet_clay_point}')

        # Define dryclay point
        if not all(C):
            m = (D[1] - self.wet_clay_point[1]) / (D[0] - self.wet_clay_point[0])
            dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
            C = self.dry_clay_point = (dryclay_NPHI, C[1])
        # print(f'#### dryclay_pt: {C}, {m}')

        vsand, vcld = self.lithology_fraction(nphi, rhob)

        if xplot:
            fig = neutron_density_xplot(nphi, rhob, A, (0, 0), C, D, self.wet_clay_point)
            return vsand, vcld, fig
        else:
            return vsand, vcld, None

    def lithology_fraction(self, nphi, rhob):
        """Estimate sand and shale based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vsand, vcld
        """
        A = self.dry_sand_point
        C = self.dry_clay_point
        D = self.fluid_point
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


class MultiMineral():
    """This multi-mineral model utilizes optimization method to determine each mineral components."""

    def estimate_lithology(self, gr, nphi, rhob):
        """Modified from https://github.com/ruben-charles/petrophysical_evaluation_optimization_methods.git
        This module only takes in gr, nphi and rhob to estimate the volumetric of quartz, calcite, dolomite and shale.

        Args:
            gr (float): Gamma Ray log in GAPI.
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.

        Returns:
            (float, float, float, float, float): vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud
        """
        # Getting default values from Config which may need to be changed based on the dataset
        responses = Config.MINERALS_LOG_VALUE

        def error_recon(volumes, *args):
            vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud = volumes
            GR, NPHI, RHOB = args
            GR_RECON = vol_quartz*responses["GR_QUARTZ"] + vol_calcite*responses["GR_CALCITE"] + \
                vol_dolomite*responses["GR_DOLOMITE"] + vol_shale*responses["GR_SH"] + vol_mud*responses["GR_MUD"]
            NPHI_RECON = vol_quartz*responses["NPHI_QUARTZ"] + vol_calcite*responses["NPHI_CALCITE"] + \
                vol_dolomite*responses["NPHI_DOLOMITE"] + vol_shale*responses["NPHI_SH"] + vol_mud*responses["NPHI_MUD"]
            RHOB_RECON = vol_quartz*responses["RHOB_QUARTZ"] + vol_calcite*responses["RHOB_CALCITE"] + \
                vol_dolomite*responses["RHOB_DOLOMITE"] + vol_shale*responses["RHOB_SH"] + vol_mud*responses["RHOB_MUD"]

            # Some magic numbers to adjust the precision of differents magnitude orders (needs improvement)
            return (GR - GR_RECON)**2 + (NPHI*300 - NPHI_RECON*300)**2 + (RHOB*100 - RHOB_RECON*100)**2

        def constraint_(x):
            return x[0] + x[1] + x[2] + x[3] + x[4] - 1

        constrains = [{"type": "eq", "fun": constraint_}]

        # Mineral volume bounds (quartz, calcite, dolomite, shale, mud)
        bounds = ((0, 1), (0, 1), (0, 0.1), (0, 1), (0, 0.45))

        vol_quartz = []
        vol_calcite = []
        vol_dolomite = []
        vol_shale = []
        vol_mud = []
        for i, input in enumerate(zip(gr, nphi, rhob)):
            res = minimize(error_recon, ((0, 0, 0, 0, 0)),
                           args=input, bounds=bounds, constraints=constrains)
            vol_quartz.append(res.x[0])
            vol_calcite.append(res.x[1])
            vol_dolomite.append(res.x[2])
            vol_shale.append(res.x[3])
            vol_mud.append(res.x[4])

        return vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud


def shale_volume_larinov_tertiary(igr):
    """
    Computes shale volume from gamma ray using Larinov's method for tertiary rocks.

    Parameters
    ----------
    igr : float
        Interval gamma ray [API].

    Returns
    -------
    vshale : float
        Shale volume [fraction].

    """
    return 0.083 * (2**(3.7 * igr) - 1)


def shale_volume_larinov_older(igr):
    """
    Computes shale volume from gamma ray using Larinov's method for older rocks.

    Parameters
    ----------
    igr : float
        Interval gamma ray [API].

    Returns
    -------
    vshale : float
        Shale volume [fraction].

    """
    return 0.33 * (2**(2 * igr) - 1)


def shale_volume_steiber(igr):
    """
    Computes shale volume from gamma ray using Steiber's method.

    Parameters
    ----------
    igr : float
        Interval gamma ray [API].

    Returns
    -------
    vshale : float
        Shale volume [fraction].

    """
    return igr / (3 - 2 * igr)


def gr_index(gr):
    """
    Computes gamma ray index from gamma ray.

    Parameters
    ----------
    gr : float
        Gamma ray [API].

    Returns
    -------
    gr_index : float
        Gamma ray index [API].

    """
    gr = np.where(np.isnan(gr), np.min(gr), gr)
    dtr_gr = detrend(gr, axis=0) + statistics.mean(gr)
    scaler = MinMaxScaler()
    igr = scaler.fit_transform(dtr_gr.reshape(-1, 1))
    return igr
