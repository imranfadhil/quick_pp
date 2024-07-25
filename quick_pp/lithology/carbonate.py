import numpy as np

from quick_pp.lithology import shale_volume_steiber, gr_index
from quick_pp.utils import length_a_b, line_intersection
from quick_pp.porosity import neu_den_xplot_poro_pt, rho_matrix, density_porosity
from quick_pp.config import Config


class Carbonate:
    """Carbonate model based on neutron density or density pef cross plots."""

    def __init__(self, dry_calc_point: tuple = None, dry_dolo_point: tuple = None, dry_clay_point: tuple = None,
                 fluid_point: tuple = None, **kwargs):
        # Initialize the endpoints
        self.dry_calc_point = dry_calc_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"]
        self.dry_dolo_point = dry_dolo_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"]
        self.dry_clay_point = dry_clay_point or Config.CARB_NEU_DEN_ENDPOINTS["DRY_CLAY_POINT"]
        self.fluid_point = fluid_point or Config.CARB_NEU_DEN_ENDPOINTS["FLUID_POINT"]

    def estimate_lithology(self, nphi, rhob, gr, pef=None, normalize: bool = True,
                           model: str = 'single', method: str = 'neu_den', carbonate_type: str = 'limestone'):
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

        if model == 'single':
            # Estimate vshale
            vcld = self.estimate_vcld_from_gr(gr)
            vcarb = 1 - vcld
            vdolo = vcarb if carbonate_type == 'dolostone' else 0
            vcalc = vcarb if carbonate_type == 'limestone' else 0
            if normalize:
                # Estimate total porosity
                rho_ma = rho_matrix(vcalc=vcalc, vdolo=vdolo, vclay=vcld)
                phit = density_porosity(rhob, rho_ma)
                vmatrix = (1 - phit)
                vcalc = vcalc * vmatrix
                vdolo = vdolo * vmatrix
                vcld = vcld * vmatrix
            return vcld, vcalc, vdolo
        elif model == 'double':
            # Estimate vshale
            vcld = self.estimate_vcld_from_gr(gr)
            if method == 'neu_den':
                vcalc, vdolo = self.lithology_fraction_neu_den(nphi, rhob, model='double', normalize=normalize)
            else:
                assert pef is not None, "PEF log is required for 'den_pef' method."
                vcalc, vdolo = self.lithology_fraction_pef(pef, rhob, normalize=normalize)
            vcalc = vcalc * (1 - vcld)
            vdolo = vdolo * (1 - vcld)
            return vcld, vcalc, vdolo

    def lithology_fraction_neu_den(self, nphi, rhob, model: str = 'single', carbonate_type: str = 'limestone',
                                   normalize: bool = True):
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

            phit = neu_den_xplot_poro_pt(point[0], point[1], 'ss', None, A, (0, 0), C, D) if normalize else 0
            vmatrix = 1 - phit
            vlitho1 = np.append(vlitho1, (1 - vfrac) * vmatrix)
            vlitho2 = np.append(vlitho2, vfrac * vmatrix)

        return vlitho1, vlitho2

    def lithology_fraction_pef(self, pef, rhob, normalize: bool = True):
        """Estimate limestone and dolostone based on pef density cross plot. Expecting the inputs are clay corrected.

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

            phit = neu_den_xplot_poro_pt(point[0], point[1], 'ss', None, A, (0, 0), C, D) if normalize else 0
            vmatrix = 1 - phit
            vcalc = np.append(vcalc, (1 - dolo_frac) * vmatrix)
            vdolo = np.append(vdolo, dolo_frac * vmatrix)

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
        # Convert to numpy array for vectorized operations
        nphi = np.array(nphi)
        rhob = np.array(rhob)
        pef = np.array(pef)

        nphicc = (nphi - vcld * self.dry_clay_point[0]) / (1 - vcld)
        rhobcc = (rhob - vcld * self.dry_clay_point[1]) / (1 - vcld)
        pefcc = (pef - vcld * Config.MINERALS_LOG_VALUE['PEF_SH']) / (1 - vcld)

        return nphicc, rhobcc, pefcc

    def estimate_vcld_from_gr(self, gr):
        """Estimate Vclay from Gamma Ray log.

        Args:
            gr (float): Gamma Ray log in API
        Returns:
            float: Vclay in v/v
        """
        return shale_volume_steiber(gr_index(gr)).flatten()
