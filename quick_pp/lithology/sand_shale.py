from typing import Optional

import numpy as np

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.utils import line_intersection, min_max_line, remove_outliers


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
        """Initializes the SandShale model with specified or default endpoints.

        Args:
            dry_sand_point (tuple, optional): (NPHI, RHOB) for dry sand. Defaults to config values.
            dry_clay_point (tuple, optional): (NPHI, RHOB) for dry clay. Defaults to config values.
            fluid_point (tuple, optional): (NPHI, RHOB) for the formation fluid. Defaults to config values.
            wet_clay_point (tuple, optional): (NPHI, RHOB) for wet clay, used for dynamic endpoint calculation. Defaults to config values.
            silt_line_angle (float, optional): Silt line angle in degrees. Defaults to config values.
        """
        # Initialize the endpoints
        self.dry_sand_point = dry_sand_point or Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
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
        nphi_arr = np.asarray(nphi, dtype=float)
        rhob_arr = np.asarray(rhob, dtype=float)
        logger.info(f"Estimating sand-shale lithology for {len(nphi_arr)} data points")

        # Initialize the endpoints
        C = self.dry_clay_point
        D = self.fluid_point

        # Redefine wetclay point if not fully specified
        nphi_max_line = None
        rhob_max_line = None
        if self.wet_clay_point is None or any(v is None for v in self.wet_clay_point):
            rhob_clean = remove_outliers(rhob_arr)
            nphi_clean = remove_outliers(nphi_arr)
            _, rhob_max_line = min_max_line(rhob_clean, 0.05)
            _, nphi_max_line = min_max_line(nphi_clean, 0.05)
            wetclay_RHOB = np.nanmin(rhob_max_line)
            wetclay_NPHI = np.nanmax(nphi_max_line)
            self.wet_clay_point = (wetclay_NPHI, wetclay_RHOB)
            logger.debug(
                f"Updated wet clay point to: ({wetclay_NPHI:.3f}, {wetclay_RHOB:.3f})"
            )

        # Define dryclay point if not fully specified
        if C is None or any(v is None for v in C):
            den = float(D[0]) - float(self.wet_clay_point[0])
            if abs(den) < 1e-6:
                dryclay_NPHI = float(self.wet_clay_point[0])
            else:
                m = (float(D[1]) - float(self.wet_clay_point[1])) / den
                dryclay_NPHI = ((float(C[1]) - float(D[1])) / m) + float(D[0])
            C = self.dry_clay_point = (dryclay_NPHI, C[1])
            logger.debug(f"Updated dry clay point to: ({dryclay_NPHI:.3f}, {C[1]:.3f})")

        vsand, vcld = self.lithology_fraction(nphi_arr, rhob_arr)

        logger.debug(
            f"Sand-shale estimation completed - mean vsand: {np.nanmean(vsand):.3f}, "
            f"vcld: {np.nanmean(vcld):.3f}"
        )

        return vsand, vcld, (nphi_max_line, rhob_max_line)

    def lithology_fraction(self, nphi, rhob):
        """Estimate sand and clay fractions based on neutron-density crossplot.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the volumes of sand and clay,
                                           strictly normalized such that vsand + vcld = 1.0
                                           and values are clamped to [0.0, 1.0].
        """
        logger.debug("Calculating sand-shale lithology fractions")

        nphi_arr = np.asarray(nphi, dtype=float)
        rhob_arr = np.asarray(rhob, dtype=float)
        if nphi_arr.shape != rhob_arr.shape:
            raise ValueError("nphi and rhob must have the same shape")

        A = self.dry_sand_point
        C = self.dry_clay_point
        D = self.fluid_point

        var_pt = line_intersection((A, C), (D, (nphi_arr, rhob_arr)))

        # Parametric projection along rock line from A (dry sand, t=0) to C (dry clay, t=1)
        dx = float(C[0]) - float(A[0])
        dy = float(C[1]) - float(A[1])
        denom = dx * dx + dy * dy
        if denom < 1e-12:
            raise ValueError("Dry sand point and dry clay point must not be identical")

        t = ((var_pt[0] - float(A[0])) * dx + (var_pt[1] - float(A[1])) * dy) / denom
        vcld = np.clip(t, 0.0, 1.0)
        vsand = 1.0 - vcld

        # Preserve NaNs where input logs or intersections are non-finite
        invalid = np.isnan(var_pt[0]) | np.isnan(var_pt[1]) | np.isnan(nphi_arr) | np.isnan(rhob_arr)
        if np.any(invalid):
            if np.ndim(vsand) == 0:
                vsand = np.nan
                vcld = np.nan
            else:
                vsand = np.where(invalid, np.nan, vsand)
                vcld = np.where(invalid, np.nan, vcld)

        logger.debug(
            f"Lithology fraction calculation completed for {len(np.atleast_1d(vsand))} points"
        )
        return vsand, vcld
