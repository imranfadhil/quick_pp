import numpy as np
from typing import Optional

from quick_pp.utils import min_max_line, length_a_b, line_intersection, remove_outliers
from quick_pp.config import Config
from quick_pp import logger


class SandShale:
    """A binary lithology model for sand-shale sequences.

    This model uses a neutron-density crossplot to estimate the volumetric
    fractions of sand and shale. It dynamically defines endpoints if not
    explicitly provided, making it adaptable to different datasets.
    """

    def __init__(
        self,
        dry_sand_point: Optional[tuple[float, float]] = None,
        dry_clay_point: Optional[tuple[float, float]] = None,
        fluid_point: Optional[tuple[float, float]] = None,
        wet_clay_point: Optional[tuple[float, float]] = None,
        silt_line_angle: Optional[float] = None,
        **kwargs,
    ):
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
        """
        Initializes the SandShale model with specified or default endpoints.

        Args:
            dry_sand_point (tuple, optional): (NPHI, RHOB) for dry sand. Defaults to config values.
            dry_clay_point (tuple, optional): (NPHI, RHOB) for dry clay. Defaults to config values.
            fluid_point (tuple, optional): (NPHI, RHOB) for the formation fluid. Defaults to config values.
            wet_clay_point (tuple, optional): (NPHI, RHOB) for wet clay, used for dynamic endpoint calculation. Defaults to config values.
        """
        self.dry_clay_point = dry_clay_point or Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.SSC_ENDPOINTS["FLUID_POINT"]
        self.wet_clay_point = wet_clay_point or Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
        self.silt_line_angle = (
            silt_line_angle or Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]
        )

        logger.debug(
            f"SandShale model initialized with endpoints: sand_point={self.dry_sand_point}, "
            f"clay_point={self.dry_clay_point}, fluid_point={self.fluid_point}, "
            f"wet_clay_point={self.wet_clay_point}"
        )

    def estimate_lithology(self, nphi, rhob):
        """Estimate lithology volumetrics based on neutron density cross plot.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple[np.ndarray, np.ndarray, tuple]: A tuple containing the volumes of sand and clay,
                                                   and a tuple with the calculated min/max trend lines for NPHI and RHOB.
        """
        logger.info(f"Estimating sand-shale lithology for {len(nphi)} data points")

        # Initialize the endpoints
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point
        nphi_max_line = None
        rhob_max_line = None
        if not all(self.wet_clay_point):
            rhob_clean = remove_outliers(rhob)
            nphi_clean = remove_outliers(nphi)
            _, rhob_max_line = min_max_line(rhob_clean, 0.05)
            _, nphi_max_line = min_max_line(nphi_clean, 0.05)
            wetclay_RHOB = np.min(rhob_max_line)
            wetclay_NPHI = np.max(nphi_max_line)
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
            logger.debug(
                f"Updated wet clay point to: ({wetclay_NPHI:.3f}, {wetclay_RHOB:.3f})"
            )

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

        return vsand, vcld, (nphi_max_line, rhob_max_line)

    def lithology_fraction(self, nphi, rhob):
        """Estimate sand and shale based on neutron density cross plot.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the volumes of sand and clay.
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
            projlithofrac = length_a_b(var_pt, C)
            sand_frac = projlithofrac / rocklithofrac
            sand_frac = 0 if var_pt[0] > C[0] else sand_frac
            vsand = np.append(vsand, sand_frac)
            vcld = np.append(vcld, 1 - sand_frac)

        logger.debug(
            f"Lithology fraction calculation completed for {len(vsand)} points"
        )
        return vsand, vcld
