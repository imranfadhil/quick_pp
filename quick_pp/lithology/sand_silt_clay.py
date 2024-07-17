import numpy as np
import math

from quick_pp.utils import min_max_line, length_a_b, line_intersection
from quick_pp.config import Config


class SandSiltClay:
    """Sand-silt-clay model based on Kuttan's litho-porosity model.

     The original is a binary model where reservoir sections (left to the silt line) consist of combination of
     sand-silt while non-reservoir sections (right to the silt line) consist of silt-clay combination.

     This module is modified based on Kuttan, where the reservoir sections consist of combination of sand-silt-clay
     while non-reservoir sections remain a combination of silt-clay only."""

    def __init__(self, dry_sand_point: tuple = None, dry_silt_point: tuple = None, dry_clay_point: tuple = None,
                 fluid_point: tuple = None, wet_clay_point: tuple = None, silt_line_angle: float = None, **kwargs):
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_silt_point = dry_silt_point or Config.SSC_ENDPOINTS["DRY_SILT_POINT"]
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]

    def estimate_lithology(self, nphi, rhob):
        """Estimate sand silt clay lithology volumetrics.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float, float, boolean): vsand, vsilt, vcld, vclb, cross-plot if xplot True else None
        """
        # Initialize the endpoints
        A = self.dry_sand_point
        B = self.dry_silt_point
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point
        _, rhob_max_line = min_max_line(rhob, 0.1)
        _, nphi_max_line = min_max_line(nphi, 0.1)
        wetclay_RHOB = np.nanmin(rhob_max_line)
        wetclay_NPHI = np.nanmax(nphi_max_line)
        if not all(self.wet_clay_point):
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
        # print(f'#### wet_clay_point: {self.wet_clay_point}')

        # Redefine drysilt point
        drysilt_NPHI = 1 - 1.68 * (math.tan(float(self.silt_line_angle - 90) * math.pi / 180))
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

        # Calculate lithology fraction
        vsand, vsilt, vcld = self.lithology_fraction_kuttan_modified(nphi, rhob)

        return vsand, vsilt, vcld, (nphi_max_line, rhob_max_line)

    def lithology_fraction_kuttan_modified(self, nphi, rhob):
        """Estimate lithology volumetrics based on modified Kuttan's litho-porosity model.

        Args:
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.

        Returns:
            (float, float, float): vsand, vsilt, vcld
        """
        A = self.dry_sand_point
        B = self.dry_silt_point
        C = self.dry_clay_point
        D = self.fluid_point
        E = list(zip(nphi, rhob))

        rock_len = length_a_b(A, C)
        res_len = length_a_b(A, B)
        res_ratio = res_len / rock_len

        vsand = np.empty(0)
        vsilt = np.empty(0)
        vcld = np.empty(0)
        for i, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            proj_len = length_a_b(var_pt, A)
            vsand_pt, vsilt_pt, vcld_pt = self.lithology_chart(proj_len, rock_len, res_ratio)
            vsand = np.append(vsand, vsand_pt)
            vsilt = np.append(vsilt, vsilt_pt)
            vcld = np.append(vcld, vcld_pt)

        return vsand, vsilt, vcld

    def lithology_chart(self, proj_len, rock_len, res_ratio):
        """Estimate lithology fraction based on Kuttan's modified chart.

        Args:
            proj_len (float): Length of projected point from dry sand point.
            rock_len (float): Length of rock line.
            res_ratio (float): Ratio of reservoir length to rock length.

        Returns:
            (float, float, float): vsand, vsilt, vcld
        """
        siltclayratio = 0.25  # empirical value
        res_len = rock_len * res_ratio
        non_res_len = rock_len * (1 - res_ratio)
        pt_ratio = proj_len / rock_len
        if pt_ratio <= res_ratio:
            vsand = (-proj_len / res_len) + 1
            vsilt = siltclayratio / res_len * proj_len
            vcld = (1 - siltclayratio) / res_len * proj_len
        else:
            vsand = 0
            vsilt = -siltclayratio / non_res_len * proj_len + (rock_len * siltclayratio / non_res_len)
            vcld = siltclayratio / non_res_len * proj_len + (1 - (siltclayratio * rock_len / non_res_len))

        return vsand, vsilt, vcld
