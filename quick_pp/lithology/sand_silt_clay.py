import numpy as np
import math

from ..rock_type import estimate_vsh_gr
from ..utils import min_max_line, length_a_b, line_intersection
from ..porosity import neu_den_xplot_poro_pt, clay_porosity
from ..config import Config


class SandSiltClay:
    """Sand-silt-clay model based on Kuttan's litho-porosity model.

     The original is a binary model where reservoir sections (left to the silt line) consist of combination of
     sand-silt while non-reservoir sections (right to the silt line) consist of silt-clay combination.

     PCSB proposed a model based on Kuttan, where the reservoir sections consist of combination of sand-silt-clay
     while non-reservoir sections remain a combination of silt-clay only. This module is a simplified version of the
     original PCSB model."""

    def __init__(self, dry_sand_point: tuple = None, dry_silt_point: tuple = None, dry_clay_point: tuple = None,
                 fluid_point: tuple = None, wet_clay_point: tuple = None, silt_line_angle: float = None, **kwargs):
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_silt_point = dry_silt_point or Config.SSC_ENDPOINTS["DRY_SILT_POINT"]
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]

    def estimate_lithology(self, nphi, rhob, gr=None, badhole_flag=None, model: str = 'kuttan', normalize: bool = True):
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

        # Estimate vsh_gr for badhole interval
        vsh_gr = estimate_vsh_gr(gr) if gr is not None else None

        # Redefine wetclay point
        _, rhob_max_line = min_max_line(rhob, 0.1, num_bins=1)
        _, nphi_max_line = min_max_line(nphi, 0.1, num_bins=1)
        wetclay_RHOB = np.nanmin(rhob_max_line)
        wetclay_NPHI = np.nanmax(nphi_max_line)
        if not all(self.wet_clay_point):
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
        # print(f'#### wet_clay_point: {self.wet_clay_point}')

        # Redefine drysilt point
        drysilt_NPHI = 1 - 1.68*(math.tan(float(self.silt_line_angle - 90)*math.pi / 180))
        if not all(B):
            B = self.dry_silt_point = (drysilt_NPHI, B[1])
        # print(f'#### drysilt_pt: {drysilt_pt}')

        # Define dryclay point
        if not all(C):
            # # Calculate dryclay_NPHI given dryclay_RHOB
            # m = (D[1] - self.wet_clay_point[1]) / (D[0] - self.wet_clay_point[0])
            # dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
            # C = self.dry_clay_point = (dryclay_NPHI, C[1])

            # Calculate dryclay point, the intersection between rock line and clay line
            drysilt_RHOB = B[1] if B[1] > np.nanmax(rhob) else np.nanmax(rhob)
            updated_drysilt_pt = (B[0], drysilt_RHOB)
            C = self.dry_clay_point = line_intersection((A, updated_drysilt_pt), (D, self.wet_clay_point))
        # print(f'#### dryclay_pt: {C}, {m}')

        if model == 'kuttan':
            vsand, vsilt, vcld = self.lithology_fraction_kuttan(nphi, rhob, normalize)
        else:
            vsand, vsilt, vcld = self.lithology_fraction_kuttan_modified(
                nphi, rhob, vsh_gr=vsh_gr, badhole_flag=badhole_flag, normalize=normalize
            )

        # Calculate vclb: volume of clay bound water
        clay_phit = clay_porosity(rhob, C[1])
        vclb = vcld * clay_phit

        return vsand, vsilt, vcld, vclb, (nphi_max_line, rhob_max_line)

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

        vsand = np.empty(0)
        vsilt = np.empty(0)
        vcld = np.empty(0)
        for i, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            matrix_ratio = projlithofrac / rocklithofrac
            if matrix_ratio < matrix_ratio_x:
                phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', True, A, B, C, D) if normalize else 0
                vmatrix = 1 - phit
                vsilt = np.append(vsilt, (matrix_ratio / matrix_ratio_x) * vmatrix)
                vsand = np.append(vsand, (1 - vsilt[i]) * vmatrix)
                vcld = np.append(vcld, 0)
            else:
                phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', False, A, B, C, D) if normalize else 0
                vmatrix = 1 - phit
                vsand = np.append(vsand, 0)
                vsilt = np.append(vsilt, ((1 - matrix_ratio) * vmatrix) / (1 - matrix_ratio_x))
                vcld = np.append(vcld, (1 - vsilt[i]) * vmatrix)

        return vsand, vsilt, vcld

    def lithology_fraction_kuttan_modified(self, nphi, rhob, vsh_gr=None, badhole_flag=None, normalize: bool = True):
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
        badhole_flag = badhole_flag if badhole_flag is not None else np.zeros(len(nphi))
        vsh_gr = vsh_gr if vsh_gr is not None else np.zeros(len(nphi))
        E = list(zip(nphi, rhob, badhole_flag, vsh_gr))

        siltclayratio = 0.25  # empirical value
        claysiltfrac = length_a_b(C, B)
        rocklithofrac = length_a_b(A, C)
        sandsiltfrac = length_a_b(A, B)
        matrix_ratio_x = sandsiltfrac / rocklithofrac

        vsand = np.empty(0)
        vsilt = np.empty(0)
        vcld = np.empty(0)
        for i, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            matrix_ratio = projlithofrac / rocklithofrac
            if point[2] == 1:
                phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', True, A, B, C, D) if normalize else 0
                vmatrix = 1 - phit
                vsand = np.append(vsand, (1 - point[3]) * vmatrix)
                vsilt = np.append(vsilt, siltclayratio * point[3] * vmatrix)
                vcld = np.append(vcld, (1 - siltclayratio) * point[3] * vmatrix)
            elif matrix_ratio < matrix_ratio_x:
                phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', True, A, B, C, D) if normalize else 0
                vmatrix = 1 - phit
                vsand = np.append(vsand, ((-projlithofrac / sandsiltfrac) + 1) * vmatrix)
                vsilt = np.append(vsilt, siltclayratio / sandsiltfrac * projlithofrac * vmatrix)
                vcld = np.append(vcld, ((1 - siltclayratio) / sandsiltfrac * projlithofrac) * vmatrix)
            else:
                phit = neu_den_xplot_poro_pt(point[0], point[1], 'ssc', False, A, B, C, D) if normalize else 0
                vmatrix = 1 - phit
                vsand = np.append(vsand, 0)
                vsilt = np.append(vsilt, (-siltclayratio / claysiltfrac * projlithofrac + (
                    rocklithofrac * siltclayratio / claysiltfrac)) * vmatrix)
                vcld = np.append(vcld, (siltclayratio / claysiltfrac * projlithofrac + (
                    1 - (siltclayratio * rocklithofrac / claysiltfrac))) * vmatrix)

        return vsand, vsilt, vcld
