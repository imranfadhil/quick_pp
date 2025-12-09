from typing import Optional

import numpy as np
import pandas as pd

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.utils import length_a_b, line_intersection


class Carbonate:
    """A model for estimating carbonate lithology from well logs.

    This class implements methods for determining the volumetric fractions of
    calcite, dolomite, and clay using either neutron-density or density-PEF
    crossplots. It supports both single-mineral (limestone or dolostone) and
    dual-mineral (limestone-dolomite) carbonate models.
    """

    def __init__(
        self,
        dry_calc_point: Optional[tuple[float, float]] = None,
        dry_dolo_point: Optional[tuple[float, float]] = None,
        dry_clay_point: Optional[tuple[float, float]] = None,
        fluid_point: Optional[tuple[float, float]] = None,
        **kwargs,
    ):
        """Initializes the Carbonate model with specified or default endpoints.

        Args:
            dry_calc_point (tuple, optional): (NPHI, RHOB) for dry calcite. Defaults to config values.
            dry_dolo_point (tuple, optional): (NPHI, RHOB) for dry dolomite. Defaults to config values.
            dry_clay_point (tuple, optional): (NPHI, RHOB) for dry clay. Defaults to config values.
            fluid_point (tuple, optional): (NPHI, RHOB) for the formation fluid. Defaults to config values.
        """
        # Initialize the endpoints
        self.dry_calc_point = (
            dry_calc_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"]
        )
        self.dry_dolo_point = (
            dry_dolo_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"]
        )
        self.dry_clay_point = (
            dry_clay_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CLAY_POINT"]
        )
        self.fluid_point = fluid_point or Config.CARB_NEU_DEN_ENDPOINTS["FLUID_POINT"]

        logger.debug(
            f"Carbonate model initialized with endpoints: calc={self.dry_calc_point}, "
            f"dolo={self.dry_dolo_point}, clay={self.dry_clay_point}, "
            f"fluid={self.fluid_point}"
        )

    def estimate_lithology(
        self,
        nphi,
        rhob,
        pef=None,
        vsh_gr=None,
        model: str = "single",
        method: str = "neu_den",
        carbonate_type: str = "limestone",
    ):
        """Estimate carbonate and clay lithology volumetrics.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].
            pef (np.ndarray or float, optional): Photoelectric Factor log [barns/electron]. Required for 'den_pef' method. Defaults to None.
            vsh_gr (np.ndarray or float, optional): Volume of shale from gamma ray log [v/v]. Defaults to None.
            model (str, optional): The lithology model, either 'single' (one carbonate type + clay) or
                                   'double' (calcite + dolomite + clay). Defaults to 'single'.
            method (str, optional): The crossplot method for the 'double' model, either 'neu_den' or
                                    'den_pef'. Defaults to 'neu_den'.
            carbonate_type (str, optional): The dominant carbonate type for the 'single' model, either
                                            'limestone' or 'dolostone'. Defaults to 'limestone'.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the volumes of clay, calcite, and dolomite.
        """
        logger.info(
            f"Estimating lithology using {model} model with {method} method for {carbonate_type}"
        )

        if model not in ["single", "double"]:
            logger.error(
                f"Invalid model '{model}'. Available models: 'single', 'double'"
            )
            raise AssertionError(f"'{model}' model is not available.")

        if method not in ["neu_den", "den_pef"]:
            logger.error(
                f"Invalid method '{method}'. Available methods: 'neu_den', 'den_pef'"
            )
            raise AssertionError(f"'{method}' method is not available.")

        vcld = vsh_gr if vsh_gr is not None else np.zeros(len(nphi))
        logger.debug(
            f"Input data length: {len(nphi)}, clay volume provided: {vsh_gr is not None}"
        )

        if model == "single":
            # Estimate vshale
            vcarb = 1 - vcld
            vdolo = vcarb if carbonate_type == "dolostone" else 0
            vcalc = vcarb if carbonate_type == "limestone" else 0
            logger.debug(
                f"Single model results - {pd.DataFrame({'vcld': vcld, 'vcalc': vcalc, 'vdolo': vdolo}).describe()}"
            )
            return vcld, vcalc, vdolo
        elif model == "double":
            # Estimate vshale
            if method == "neu_den":
                logger.debug("Using neutron-density crossplot method")
                vcalc, vdolo = self.lithology_fraction_neu_den(
                    nphi, rhob, model="double"
                )
            else:
                if pef is None:
                    logger.error(
                        "PEF log is required for 'den_pef' method but was not provided"
                    )
                    raise AssertionError("PEF log is required for 'den_pef' method.")
                logger.debug("Using density-PEF crossplot method")
                vcalc, vdolo = self.lithology_fraction_pef(pef, rhob)
            vcalc = vcalc * (1 - vcld)
            vdolo = vdolo * (1 - vcld)
            logger.debug(
                f"Double model results - {pd.DataFrame({'vcld': vcld, 'vcalc': vcalc, 'vdolo': vdolo}).describe()}"
            )
            return vcld, vcalc, vdolo

    def lithology_fraction_neu_den(
        self, nphi, rhob, model: str = "single", carbonate_type: str = "limestone"
    ):
        """Estimate clay and carbonate (either limestone or dolostone) based on neutron density cross plot.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].
            model (str, optional): The lithology model, either 'single' or 'double'. Defaults to 'single'.
            carbonate_type (str, optional): The dominant carbonate type for the 'single' model.
                                            Defaults to 'limestone'.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the volumes of the two lithology components.
                                           For 'single' model: (vclay, vcarbonate).
                                           For 'double' model: (vdolomite, vcalcite).
        """
        logger.debug(
            f"Calculating lithology fractions using neutron-density crossplot "
            f"(model: {model}, type: {carbonate_type})"
        )

        if model == "single":
            A = self.dry_clay_point
            C = (
                self.dry_calc_point
                if carbonate_type == "limestone"
                else self.dry_dolo_point
            )
        else:
            A = self.dry_dolo_point
            C = self.dry_calc_point
        D = self.fluid_point
        E = zip(nphi, rhob, strict=True)
        rocklithofrac = length_a_b(A, C)

        vlitho1 = np.empty(0)
        vlitho2 = np.empty(0)
        for _, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            vfrac = projlithofrac / rocklithofrac

            vlitho1 = np.append(vlitho1, (1 - vfrac))
            vlitho2 = np.append(vlitho2, vfrac)

        logger.debug(f"Neutron-density calculation completed for {len(vlitho1)} points")
        return vlitho1, vlitho2

    def lithology_fraction_pef(self, pef, rhob):
        """Estimate limestone and dolostone based on pef density cross plot. Expecting the inputs are clay corrected.

        Args:
            pef (np.ndarray or float): Photoelectric Factor log [barns/electron].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing the volumes of calcite and dolomite.
        """
        logger.debug("Calculating lithology fractions using density-PEF crossplot")

        A = (Config.MINERALS_LOG_VALUE["PEF_CALCITE"], self.dry_calc_point[1])
        C = (Config.MINERALS_LOG_VALUE["PEF_DOLOMITE"], self.dry_dolo_point[1])
        D = self.fluid_point
        E = zip(pef, rhob, strict=True)
        rocklithofrac = length_a_b(A, C)

        vcalc = np.empty(0)
        vdolo = np.empty(0)
        for _, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            dolo_frac = projlithofrac / rocklithofrac

            vcalc = np.append(vcalc, (1 - dolo_frac))
            vdolo = np.append(vdolo, dolo_frac)

        logger.debug(f"Density-PEF calculation completed for {len(vcalc)} points")
        return vcalc, vdolo

    def clay_correction(self, vcld, nphi, rhob, pef):
        """Apply clay correction to the input logs.

        Args:
            vcld (np.ndarray or float): Volume of clay [v/v].
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].
            pef (np.ndarray or float): Photoelectric Factor log [barns/electron].

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Clay-corrected NPHI, RHOB, and PEF logs.
        """
        logger.debug("Applying clay correction to input logs")

        # Convert to numpy array for vectorized operations
        nphi = np.array(nphi)
        rhob = np.array(rhob)
        pef = np.array(pef)

        nphicc = (nphi - vcld * self.dry_clay_point[0]) / (1 - vcld)
        rhobcc = (rhob - vcld * self.dry_clay_point[1]) / (1 - vcld)
        pefcc = (pef - vcld * Config.MINERALS_LOG_VALUE["PEF_SH"]) / (1 - vcld)

        logger.debug(f"Clay correction completed. Input length: {len(nphi)}")
        return nphicc, rhobcc, pefcc


def sep_vug_poro(phit, phis, dtc=None, model="base", alpha=2.0, p=0.1):
    """Separate vug porosity from total porosity and sonic porosity.
    Base model (Lucia-Conti, 1987)
    Power model (Wang-Lucia, 1993)
    Quadratic model (Wang-Lucia, 1993)

    Args:
        phit (np.ndarray or float): Total porosity [v/v].
        phis (np.ndarray or float): Sonic porosity [v/v].
        dtc (np.ndarray or float, optional): Compressional slowness log [us/ft]. Used by the 'base' model. Defaults to None.
        model (str, optional): The model to use, either 'base', 'power', or 'quadratic'. Defaults to 'base'.
        alpha (float, optional): The scaling factor for the 'power' model. Defaults to 2.0.
        p (float, optional): The empirical coefficient for the 'quadratic' model. Defaults to 0.1.

    Returns:
        np.ndarray or float: The estimated separate vug porosity [v/v].
    """
    logger.info(f"Calculating separate vug porosity using {model} model")

    if model not in ["base", "power", "quadratic"]:
        logger.error(
            f"Invalid model '{model}'. Available models: 'base', 'power', 'quadratic'"
        )
        raise AssertionError(
            'Please choose from "base", "power", or "quadratic" model.'
        )

    if model == "base" and dtc is not None:
        logger.debug("Using base model (Lucia-Conti, 1987) with sonic data")
        return 10 ** (4.09 - 0.145 * (dtc - 141.5 * phit))
    elif model == "power":
        logger.debug(f"Using power model (Wang-Lucia, 1993) with alpha={alpha}")
        return (phit / phis) ** alpha * (phit - phis)
    elif model == "quadratic":
        logger.debug(f"Using quadratic model (Wang-Lucia, 1993) with p={p}")
        return (phit - phis) + p * (phit - phis) ** 2
