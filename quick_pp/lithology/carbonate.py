import numpy as np

from . import shale_volume_steiber, gr_index
from ..utils import length_a_b, line_intersection
from ..qaqc import neu_den_xplot_hc_correction
from ..config import Config


class Carbonate:
    """Sand-silt-clay model based on Kuttan's litho-porosity model.

     The original is a binary model where reservoir sections (left to the silt line) consist of combination of
     sand-silt while non-reservoir sections (right to the silt line) consist of silt-clay combination.

     PCSB proposed a model based on Kuttan, where the reservoir sections consist of combination of sand-silt-clay
     while non-reservoir sections remain a combination of silt-clay only. This module is a simplified version of the
     original PCSB model."""

    def __init__(self, dry_calc_point: tuple = None, dry_dolo_point: tuple = None, dry_clay_point: tuple = None,
                 fluid_point: tuple = None):
        # Initialize the endpoints
        self.dry_calc_point = dry_calc_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"]
        self.dry_dolo_point = dry_dolo_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"]
        self.dry_clay_point = dry_clay_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.CARB_NEU_DEN_ENDPOINTS["FLUID_POINT"]

    def estimate_lithology(self, gr, nphi, rhob, pef: float = [0], model: str = 'single', method: str = 'neu_den',
                           carbonate_type: str = 'limestone'):
        """Estimate sand silt clay lithology volumetrics.

        Args:
            gr (float): Gamma Ray log in API
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            pef (float, optional): Photoelectric Factor log in barns/electron. Defaults to [0].
            model (str, optional): Model to choose from 'single' or 'double'. Defaults to 'single'.
            method (str, optional): Method for 2 minerals model, to choose from 'neu_den' or 'den_pef'.
                                    Defaults to 'neu_den'.
            carbonate_type (str, optional): Carbonate type to choose from 'limestone' or 'dolostone'.
                                            Defaults to 'limestone'.
            xplot (bool, optional): To plot Neutron Density cross plot. Defaults to False.
            normalize (bool, optional): To normalize with porosity. Defaults to True.

        Returns:
            (float, float, float): vcld, vcalc, vdolo
        """
        assert model in ['single', 'double'], f"'{model}' model is not available."
        assert method in ['neu_den', 'den_pef'], f"'{method}' method is not available."

        # Apply HC correction to neutron porosity and bulk density
        carb_point = self.dry_calc_point if carbonate_type == 'limestone' else self.dry_dolo_point
        nphi_hc, rhob_hc = neu_den_xplot_hc_correction(nphi, rhob, gr, carb_point, self.dry_clay_point)
        if model == 'single':
            # Estimate vshale
            vcld, vcarb = self.lithology_fraction_neu_den(nphi_hc, rhob_hc, carbonate_type)
            print(f'## Info for vcld \nmean: {np.mean(vcld)}, min: {np.min(vcld)}, max: {np.max(vcld)}')

            vdolo = vcarb if carbonate_type == 'dolostone' else 0
            vcalc = vcarb if carbonate_type == 'limestone' else 0
            return vcld, vcalc, vdolo
        else:
            # Estimate vshale and apply clay correction
            vcld = shale_volume_steiber(gr_index(gr)).reshape(-1)
            print(f'## Info for vcld \nmean: {np.mean(vcld)}, min: {np.min(vcld)}, max: {np.max(vcld)}')
            nphi_cc, rhob_cc, pef_cc = self.clay_correction(vcld, nphi_hc, rhob_hc, pef)
            print(f'## Info for nphi_cc \nmean: {np.mean(nphi_cc)}, min: {np.min(nphi_cc)}, max: {np.max(nphi_cc)}')
            print(f'## Info for rhob_cc \nmean: {np.mean(rhob_cc)}, min: {np.min(rhob_cc)}, max: {np.max(rhob_cc)}')
            print(f'## Info for pef_cc \nmean: {np.mean(pef_cc)}, min: {np.min(pef_cc)}, max: {np.max(pef_cc)}')
            if method == 'neu_den':
                vcalc, vdolo = self.lithology_fraction_neu_den(nphi_cc, rhob_cc, model='double')
            else:
                vcalc, vdolo = self.lithology_fraction_pef(pef_cc, rhob_cc)
            vcalc = vcalc*(1 - vcld)
            vdolo = vdolo*(1 - vcld)
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
        if model == 'single':
            A = self.dry_calc_point if carbonate_type == 'limestone' else self.dry_dolo_point
            C = self.dry_clay_point
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
            vlitho1 = np.append(vlitho1, vfrac)
            vlitho2 = np.append(vlitho2, 1 - vfrac)

        return vlitho1, vlitho2

    def lithology_fraction_pef(self, pef, rhob):
        """Estimate limestone and dolostone based on pef density cross plot. Expecting the inputs to b clay corrected.

        Args:
            pef (float): Photoelectric Factor in barns/electron
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vcalc, vdolo
        """
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
            vcalc = np.append(vcalc, 1 - dolo_frac)
            vdolo = np.append(vdolo, dolo_frac)

        return vcalc, vdolo

    def clay_correction(self, vcld, nphi, rhob, pef):
        # Convert to numpy array for vectorized operations
        nphi = np.array(nphi)
        rhob = np.array(rhob)
        pef = np.array(pef)

        nphicc = (nphi - vcld*self.dry_clay_point[0]) / (1 - vcld)
        rhobcc = (rhob - vcld*self.dry_clay_point[1]) / (1 - vcld)
        pefcc = (pef - vcld*Config.MINERALS_LOG_VALUE['PEF_SH']) / (1 - vcld)

        return nphicc, rhobcc, pefcc
