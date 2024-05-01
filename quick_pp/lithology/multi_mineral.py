from scipy.optimize import minimize
from ..config import Config


class MultiMineral():
    """This multi-mineral model utilizes optimization method to determine each mineral components."""

    def estimate_lithology(self, gr, nphi, rhob, dtc, pef):
        """Modified from https://github.com/ruben-charles/petrophysical_evaluation_optimization_methods.git
        This module only takes in gr, nphi and rhob to estimate the volumetric of quartz, calcite, dolomite and shale.

        Args:
            gr (float): Gamma Ray log in GAPI.
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.

        Returns:
            (float, float, float, float, float): vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud
        """
        # Getting default values from Config which may need to be changed based on the dataset
        responses = Config.MINERALS_LOG_VALUE

        def error_recon(volumes, *args):
            vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud = volumes
            GR, NPHI, RHOB, DTC, PEF = args
            GR_RECON = vol_quartz*responses["GR_QUARTZ"] + vol_calcite*responses["GR_CALCITE"] + \
                vol_dolomite*responses["GR_DOLOMITE"] + vol_shale*responses["GR_SH"] + vol_mud*responses["GR_MUD"]
            NPHI_RECON = vol_quartz*responses["NPHI_QUARTZ"] + vol_calcite*responses["NPHI_CALCITE"] + \
                vol_dolomite*responses["NPHI_DOLOMITE"] + vol_shale*responses["NPHI_SH"] + vol_mud*responses["NPHI_MUD"]
            RHOB_RECON = vol_quartz*responses["RHOB_QUARTZ"] + vol_calcite*responses["RHOB_CALCITE"] + \
                vol_dolomite*responses["RHOB_DOLOMITE"] + vol_shale*responses["RHOB_SH"] + vol_mud*responses["RHOB_MUD"]
            DTC_RECON = vol_quartz*responses["DTC_QUARTZ"] + vol_calcite*responses["DTC_CALCITE"] + \
                vol_dolomite*responses["DTC_DOLOMITE"] + vol_shale*responses["DTC_SH"] + vol_mud*responses["DTC_MUD"]
            PEF_RECON = vol_quartz*responses["PEF_QUARTZ"] + vol_calcite*responses["PEF_CALCITE"] + \
                vol_dolomite*responses["PEF_DOLOMITE"] + vol_shale*responses["PEF_SH"] + vol_mud*responses["PEF_MUD"]

            # Some magic numbers to adjust the precision of differents magnitude orders (needs improvement)
            return (GR - GR_RECON)**2 + (NPHI*300 - NPHI_RECON*300)**2 + (RHOB*100 - RHOB_RECON*100)**2 + \
                (DTC - DTC_RECON)**2 + (PEF - PEF_RECON)**2

        def constraint_(x):
            return x[0] + x[1] + x[2] + x[3] + x[4] - 1

        constrains = [{"type": "eq", "fun": constraint_}]

        # Mineral volume bounds (quartz, calcite, dolomite, shale, mud)
        bounds = ((0, 1), (0, 1), (0, 0.1), (0, 1), (0, 0.45))

        vol_quartz = []
        vol_calcite = []
        vol_dolomite = []
        vol_shale = []
        vol_mud = []
        for i, input in enumerate(zip(gr, nphi, rhob, dtc, pef)):
            res = minimize(error_recon, ((0, 0, 0, 0, 0)),
                           args=input, bounds=bounds, constraints=constrains)
            vol_quartz.append(res.x[0])
            vol_calcite.append(res.x[1])
            vol_dolomite.append(res.x[2])
            vol_shale.append(res.x[3])
            vol_mud.append(res.x[4])

        return vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud
