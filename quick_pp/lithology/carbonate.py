import numpy as np

from . import shale_volume_steiber, gr_index
from ..utils import length_a_b, line_intersection
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

    def estimate_lithology(self, gr, nphi, rhob, pef: float = [0], model: str = 'single',
                           carbonate_type: str = 'limestone'):
        """Estimate sand silt clay lithology volumetrics.

        Args:
            gr (float): Gamma Ray log in API
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc
            pef (float, optional): Photoelectric Factor log in barns/electron. Defaults to [0].
            model (str, optional): Model to choose from 'single' or 'double'. Defaults to 'single'.
            carbonate_type (str, optional): Carbonate type to choose from 'limestone' or 'dolostone'.
                                            Defaults to 'limestone'.
            xplot (bool, optional): To plot Neutron Density cross plot. Defaults to False.
            normalize (bool, optional): To normalize with porosity. Defaults to True.

        Returns:
            (float, float, float): vcld, vcalc, vdolo
        """
        assert model in ['single', 'double'], f"'{model}' model is not available."

        if model == 'single':
            # Estimate vshale and apply clay correction
            vcld, _ = self.lithology_fraction_neu_den(nphi, rhob, carbonate_type)
            nphi_cc, rhob_cc, pef_cc = self.clay_correction(vcld, nphi, rhob, pef)

            vcld, vcarb = self.lithology_fraction_neu_den(nphi_cc, rhob_cc, carbonate_type)
            print(np.mean(vcld), np.mean(vcarb), len(vcld), len(vcarb))
            vdolo = vcarb if carbonate_type == 'dolostone' else 0
            vcalc = vcarb if carbonate_type == 'limestone' else 0
            return vcld, vcalc, vdolo
        else:
            # Estimate vshale and apply clay correction
            vcld = shale_volume_steiber(gr_index(gr)).reshape(-1)
            print(type(vcld), np.shape(vcld), np.mean(vcld))
            nphi_cc, rhob_cc, pef_cc = self.clay_correction(vcld, nphi, rhob, pef)
            print(np.mean(nphi_cc), np.mean(rhob_cc), np.mean(pef_cc))
            vcalc, vdolo = self.lithology_fraction_pef(pef_cc, rhob_cc)
            vcalc = vcalc*(1 - vcld)
            vdolo = vdolo*(1 - vcld)
            return vcld, vcalc, vdolo

    def lithology_fraction_neu_den(self, nphi, rhob, carbonate_type: str = 'limestone'):
        """Estimate clay and carbonate (either limestone or dolostone) based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vcld, vcarb
        """
        A = self.dry_calc_point if carbonate_type == 'limestone' else self.dry_dolo_point
        C = self.dry_clay_point
        D = self.fluid_point
        E = zip(nphi, rhob)
        rocklithofrac = length_a_b(A, C)

        vcarb = np.empty(0)
        vcld = np.empty(0)
        for i, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            vshale = projlithofrac / rocklithofrac
            vcarb = np.append(vcarb, 1 - vshale)
            vcld = np.append(vcld, vshale)

        return vcld, vcarb

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
        # Convert to numpy array
        nphi = np.array(nphi)
        rhob = np.array(rhob)
        pef = np.array(pef)

        nphicc = (nphi - vcld*self.dry_clay_point[0]) / (1 - vcld)
        rhobcc = (rhob - vcld*self.dry_clay_point[1]) / (1 - vcld)
        pefcc = (pef - vcld*Config.MINERALS_LOG_VALUE['PEF_SH']) / (1 - vcld)

        return nphicc, rhobcc, pefcc
