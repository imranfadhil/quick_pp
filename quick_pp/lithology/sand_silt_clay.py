import numpy as np
import math
from typing import Optional

from quick_pp.utils import min_max_line, length_a_b, line_intersection, remove_outliers
from quick_pp.config import Config
from quick_pp import logger


class SandSiltClay:
    """A ternary lithology model for sand-silt-clay sequences.

    This model is a modification of Kuttan's litho-porosity model. It uses a
    neutron-density crossplot to estimate the volumetric fractions of sand,
    silt, and clay. Unlike the original binary model, this implementation
    considers reservoir sections as a combination of sand, silt, and clay.
    """

    def __init__(
        self,
        dry_sand_point: Optional[tuple[float, float]] = None,
        dry_silt_point: Optional[tuple[float, float]] = None,
        dry_clay_point: Optional[tuple[float, float]] = None,
        fluid_point: Optional[tuple[float, float]] = None,
        wet_clay_point: Optional[tuple[float, float]] = None,
        silt_line_angle: Optional[float] = None,
        **kwargs,
    ):
        """Initializes the SandSiltClay model with specified or default endpoints.

        Args:
            dry_sand_point (tuple, optional): (NPHI, RHOB) for dry sand. Defaults to config values.
            dry_silt_point (tuple, optional): (NPHI, RHOB) for dry silt. Defaults to config values.
            dry_clay_point (tuple, optional): (NPHI, RHOB) for dry clay. Defaults to config values.
            fluid_point (tuple, optional): (NPHI, RHOB) for the formation fluid. Defaults to config values.
            wet_clay_point (tuple, optional): (NPHI, RHOB) for wet clay. Defaults to config values.
            silt_line_angle (float, optional): The angle of the silt line on the crossplot. Defaults to config values.
        """
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_silt_point = dry_silt_point or Config.SSC_ENDPOINTS["DRY_SILT_POINT"]
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = (
            silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]
        )

        logger.debug(
            f"SandSiltClay model initialized with endpoints: sand_point={self.dry_sand_point}, "
            f"silt_point={self.dry_silt_point}, clay_point={self.dry_clay_point}, "
            f"fluid_point={self.fluid_point}, silt_line_angle={self.silt_line_angle}"
        )

    def estimate_lithology(self, nphi, rhob):
        """Estimate sand silt clay lithology volumetrics.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, tuple]: A tuple containing the volumes of sand, silt, and clay,
                                                               and a tuple with the calculated min/max trend lines for NPHI and RHOB.
        """
        logger.info(f"Estimating sand-silt-clay lithology for {len(nphi)} data points")

        # Initialize the endpoints
        B = self.dry_silt_point
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point
        nphi_max_line = None
        rhob_max_line = None
        if not all(self.wet_clay_point):
            rhob_clean = remove_outliers(rhob)
            nphi_clean = remove_outliers(nphi)
            _, rhob_max_line = min_max_line(rhob_clean, 0.1)
            _, nphi_max_line = min_max_line(nphi_clean, 0.1)
            wetclay_RHOB = np.nanmin(rhob_max_line)
            wetclay_NPHI = np.nanmax(nphi_max_line)
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
            logger.debug(
                f"Updated wet clay point to: ({wetclay_NPHI:.3f}, {wetclay_RHOB:.3f})"
            )

        # Redefine drysilt point
        drysilt_NPHI = 1 - 1.68 * (
            math.tan(float(self.silt_line_angle - 90) * math.pi / 180)
        )
        if not all(B):
            B = self.dry_silt_point = (drysilt_NPHI, B[1])
            logger.debug(f"Updated dry silt point to: ({drysilt_NPHI:.3f}, {B[1]:.3f})")

        # Define dryclay point
        if not all(C):
            # Calculate dryclay_NPHI given dryclay_RHOB
            m = (D[1] - self.wet_clay_point[1]) / (D[0] - self.wet_clay_point[0])
            dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
            C = self.dry_clay_point = (dryclay_NPHI, C[1])
            logger.debug(f"Updated dry clay point to: ({dryclay_NPHI:.3f}, {C[1]:.3f})")

        # Calculate lithology fraction
        vsand, vsilt, vcld = self.lithology_fraction_kuttan_modified(nphi, rhob)

        logger.debug(
            f"Sand-silt-clay estimation completed - mean vsand: {vsand.mean():.3f}, "
            f"vsilt: {vsilt.mean():.3f}, vcld: {vcld.mean():.3f}"
        )

        return vsand, vsilt, vcld, (nphi_max_line, rhob_max_line)

    def lithology_fraction_kuttan_modified(self, nphi, rhob):
        """Estimate lithology volumetrics based on modified Kuttan's litho-porosity model.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the volumes of sand, silt, and clay.
        """
        logger.debug("Calculating lithology fractions using modified Kuttan model")

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
            vsand_pt, vsilt_pt, vcld_pt = self.lithology_chart(
                proj_len, rock_len, res_ratio
            )
            vsand = np.append(vsand, vsand_pt)
            vsilt = np.append(vsilt, vsilt_pt)
            vcld = np.append(vcld, vcld_pt)

        logger.debug(f"Kuttan modified calculation completed for {len(vsand)} points")
        return vsand, vsilt, vcld

    def lithology_chart(self, proj_len, rock_len, res_ratio):
        """Estimate lithology fraction based on Kuttan's modified chart.

        Args:
            proj_len (float): The length of the projected point from the dry sand point along the rock line.
            rock_len (float): The total length of the rock line (sand to clay).
            res_ratio (float): The ratio of the reservoir section length (sand to silt) to the total rock line length.

        Returns:
            tuple[float, float, float]: A tuple containing the volumes of sand, silt, and clay.
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
            vsilt = -siltclayratio / non_res_len * proj_len + (
                rock_len * siltclayratio / non_res_len
            )
            vcld = siltclayratio / non_res_len * proj_len + (
                1 - (siltclayratio * rock_len / non_res_len)
            )

        return vsand, vsilt, vcld
