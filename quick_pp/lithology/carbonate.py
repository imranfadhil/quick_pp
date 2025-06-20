import numpy as np
import pandas as pd

from quick_pp.utils import length_a_b, line_intersection
from quick_pp.config import Config
from quick_pp import logger


class Carbonate:
    """Carbonate model based on neutron density or density pef cross plots."""

    def __init__(self, dry_calc_point: tuple = (float, float), dry_dolo_point: tuple = (float, float),
                 dry_clay_point: tuple = (float, float), fluid_point: tuple = (float, float), **kwargs):
        # Initialize the endpoints
        self.dry_calc_point = dry_calc_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"]
        self.dry_dolo_point = dry_dolo_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"]
        self.dry_clay_point = dry_clay_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.CARB_NEU_DEN_ENDPOINTS["FLUID_POINT"]

        logger.debug(
            f"Carbonate model initialized with endpoints: calc={self.dry_calc_point}, "
            f"dolo={self.dry_dolo_point}, clay={self.dry_clay_point}, "
            f"fluid={self.fluid_point}"
        )

    def estimate_lithology(self, nphi, rhob, pef=None, vsh_gr=None, model: str = 'single',
                           method: str = 'neu_den', carbonate_type: str = 'limestone'):
        """Estimate sand silt clay lithology volumetrics.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            pef (float, optional): Photoelectric Factor log in barns/electron. Defaults to [0].
            vsh_gr (float, optional): Volume of shale from gamma ray log in v/v. Defaults to None.
            model (str, optional): Model to choose from 'single' or 'double'. Defaults to 'single'.
            method (str, optional): Method for 2 minerals model, to choose from 'neu_den' or 'den_pef'.
                                    Defaults to 'neu_den'.
            carbonate_type (str, optional): Carbonate type to choose from 'limestone' or 'dolostone'.
                                            Defaults to 'limestone'.

        Returns:
            (float, float, float): vcld, vcalc, vdolo
        """
        logger.info(f"Estimating lithology using {model} model with {method} method for {carbonate_type}")

        if model not in ['single', 'double']:
            logger.error(f"Invalid model '{model}'. Available models: 'single', 'double'")
            raise AssertionError(f"'{model}' model is not available.")

        if method not in ['neu_den', 'den_pef']:
            logger.error(f"Invalid method '{method}'. Available methods: 'neu_den', 'den_pef'")
            raise AssertionError(f"'{method}' method is not available.")

        vcld = vsh_gr if vsh_gr is not None else np.zeros(len(nphi))
        logger.debug(f"Input data length: {len(nphi)}, clay volume provided: {vsh_gr is not None}")

        if model == 'single':
            # Estimate vshale
            vcarb = 1 - vcld
            vdolo = vcarb if carbonate_type == 'dolostone' else 0
            vcalc = vcarb if carbonate_type == 'limestone' else 0
            logger.debug(
                f"Single model results - {pd.DataFrame({'vcld': vcld, 'vcalc': vcalc, 'vdolo': vdolo}).describe()}"
            )
            return vcld, vcalc, vdolo
        elif model == 'double':
            # Estimate vshale
            if method == 'neu_den':
                logger.debug("Using neutron-density crossplot method")
                vcalc, vdolo = self.lithology_fraction_neu_den(nphi, rhob, model='double')
            else:
                if pef is None:
                    logger.error("PEF log is required for 'den_pef' method but was not provided")
                    raise AssertionError("PEF log is required for 'den_pef' method.")
                logger.debug("Using density-PEF crossplot method")
                vcalc, vdolo = self.lithology_fraction_pef(pef, rhob)
            vcalc = vcalc * (1 - vcld)
            vdolo = vdolo * (1 - vcld)
            logger.debug(
                f"Double model results - {pd.DataFrame({'vcld': vcld, 'vcalc': vcalc, 'vdolo': vdolo}).describe()}"
            )
            return vcld, vcalc, vdolo

    def lithology_fraction_neu_den(self, nphi, rhob, model: str = 'single', carbonate_type: str = 'limestone'):
        """Estimate clay and carbonate (either limestone or dolostone) based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            model (str, optional): Model to choose from 'single' or 'double'. Defaults to 'single'.
            carbonate_type (str, optional): Carbonate type to choose from 'limestone' or 'dolostone'.

        Returns:
            (float, float): vlitho1, vlitho2
        """
        logger.debug(
            f"Calculating lithology fractions using neutron-density crossplot "
            f"(model: {model}, type: {carbonate_type})"
        )

        if model == 'single':
            A = self.dry_clay_point
            C = self.dry_calc_point if carbonate_type == 'limestone' else self.dry_dolo_point
        else:
            A = self.dry_dolo_point
            C = self.dry_calc_point
        D = self.fluid_point
        E = zip(nphi, rhob)
        rocklithofrac = length_a_b(A, C)

        vlitho1 = np.empty(0)
        vlitho2 = np.empty(0)
        for i, point in enumerate(E):
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
            pef (float): Photoelectric Factor in barns/electron
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vcalc, vdolo
        """
        logger.debug("Calculating lithology fractions using density-PEF crossplot")

        A = (Config.MINERALS_LOG_VALUE['PEF_CALCITE'], self.dry_calc_point[1])
        C = (Config.MINERALS_LOG_VALUE['PEF_DOLOMITE'], self.dry_dolo_point[1])
        D = self.fluid_point
        E = zip(pef, rhob)
        rocklithofrac = length_a_b(A, C)

        vcalc = np.empty(0)
        vdolo = np.empty(0)
        for i, point in enumerate(E):
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
            vcld (float): Volume of clay in v/v
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            pef (float): Photoelectric Factor log in barns/electron

        Returns:
            float, float, float: nphicc, rhobcc, pefcc
        """
        logger.debug("Applying clay correction to input logs")

        # Convert to numpy array for vectorized operations
        nphi = np.array(nphi)
        rhob = np.array(rhob)
        pef = np.array(pef)

        nphicc = (nphi - vcld * self.dry_clay_point[0]) / (1 - vcld)
        rhobcc = (rhob - vcld * self.dry_clay_point[1]) / (1 - vcld)
        pefcc = (pef - vcld * Config.MINERALS_LOG_VALUE['PEF_SH']) / (1 - vcld)

        logger.debug(f"Clay correction completed. Input length: {len(nphi)}")
        return nphicc, rhobcc, pefcc


def sep_vug_poro(phit, phis, dtc=None, model='base', alpha=2.0, p=0.1):
    """Separate vug porosity from total porosity and sonic porosity.
    Base model (Lucia-Conti, 1987)
    Power model (Wang-Lucia, 1993)
    Quadratic model (Wang-Lucia, 1993)

    Args:
        phit (float): Total porosity in v/v
        phis (float): Sonic porosity in v/v
        dtc (float, optional): Compressional slowness log in us/ft. Defaults to None.
        model (str, optional): Model to choose from 'base', 'power', or 'quadratic'. Defaults to 'base'.
        alpha (float, optional): Scaling factor for power model. Defaults to 2.0.
        p (float, optional): Emperical coefficient for quadratic model. Defaults to 0.1.

    Returns:
        float: Separate vug porosity in v/v
    """
    logger.info(f"Calculating separate vug porosity using {model} model")

    if model not in ['base', 'power', 'quadratic']:
        logger.error(f"Invalid model '{model}'. Available models: 'base', 'power', 'quadratic'")
        raise AssertionError('Please choose from "base", "power", or "quadratic" model.')

    if model == 'base' and dtc is not None:
        logger.debug("Using base model (Lucia-Conti, 1987) with sonic data")
        return 10 ** (4.09 - 0.145 * (dtc - 141.5 * phit))
    elif model == 'power':
        logger.debug(f"Using power model (Wang-Lucia, 1993) with alpha={alpha}")
        return (phit / phis) ** alpha * (phit - phis)
    elif model == 'quadratic':
        logger.debug(f"Using quadratic model (Wang-Lucia, 1993) with p={p}")
        return (phit - phis) + p * (phit - phis) ** 2
