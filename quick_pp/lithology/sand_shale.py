import numpy as np

from quick_pp.utils import min_max_line, length_a_b, line_intersection
from quick_pp.config import Config
from quick_pp import logger


class SandShale:
    """This binary model only consider a combination of sand-shale components. """

    def __init__(self, dry_sand_point: tuple = (float, float), dry_clay_point: tuple = (float, float),
                 fluid_point: tuple = (float, float), wet_clay_point: tuple = (float, float), 
                 silt_line_angle: float = 0, **kwargs):
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]

        logger.debug(
            f"SandShale model initialized with endpoints: sand_point={self.dry_sand_point}, "
            f"clay_point={self.dry_clay_point}, fluid_point={self.fluid_point}, "
            f"wet_clay_point={self.wet_clay_point}"
        )

    def estimate_lithology(self, nphi, rhob):
        """Estimate lithology volumetrics based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            xplot (bool, optional): To plot Neutron Density cross plot. Defaults to False.

        Returns:
            (float, float): vsand, vcld, cross-plot if xplot True else None
        """
        logger.info(f"Estimating sand-shale lithology for {len(nphi)} data points")

        # Initialize the endpoints
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point
        _, rhob_max_line = min_max_line(rhob, 0.05)
        _, nphi_max_line = min_max_line(nphi, 0.05)
        wetclay_RHOB = np.min(rhob_max_line)
        wetclay_NPHI = np.max(nphi_max_line)
        if not all(self.wet_clay_point):
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
            logger.debug(f"Updated wet clay point to: ({wetclay_NPHI:.3f}, {wetclay_RHOB:.3f})")

        # Define dryclay point
        if not all(C):
            m = (D[1] - self.wet_clay_point[1]) / (D[0] - self.wet_clay_point[0])
            dryclay_NPHI = ((C[1] - D[1]) / m) + D[0]
            C = self.dry_clay_point = (dryclay_NPHI, C[1])
            logger.debug(f"Updated dry clay point to: ({dryclay_NPHI:.3f}, {C[1]:.3f})")

        vsand, vcld = self.lithology_fraction(nphi, rhob)

        logger.debug(
            f"Sand-shale estimation completed - mean vsand: {vsand.mean():.3f}, "
            f"vcld: {vcld.mean():.3f}"
        )

        return vsand, vcld

    def lithology_fraction(self, nphi, rhob):
        """Estimate sand and shale based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vsand, vcld
        """
        logger.debug("Calculating sand-shale lithology fractions")

        A = self.dry_sand_point
        C = self.dry_clay_point
        D = self.fluid_point
        E = list(zip(nphi, rhob))
        rocklithofrac = length_a_b(A, C)

        vsand = np.empty(0)
        vcld = np.empty(0)
        for i, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            vshale = projlithofrac / rocklithofrac
            vsand = np.append(vsand, (1 - vshale))
            vcld = np.append(vcld, vshale)

        logger.debug(f"Lithology fraction calculation completed for {len(vsand)} points")
        return vsand, vcld
