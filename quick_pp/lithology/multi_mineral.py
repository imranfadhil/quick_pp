from scipy.optimize import minimize
import numpy as np

from quick_pp.config import Config


def constraint_(x):
    return x[0] + x[1] + x[2] + x[3] + x[4] - 1


constrains = [{"type": "eq", "fun": constraint_}]
# Mineral volume bounds (quartz, calcite, dolomite, shale, mud)
bounds = ((0, 1), (0, 1), (0, 0.1), (0, 1), (0, 0.45))
# Getting default values from Config which may need to be changed based on the dataset
responses = Config.MINERALS_LOG_VALUE


class MultiMineral():
    """This multi-mineral model utilizes optimization method to determine each mineral components."""

    def estimate_lithology(self, gr, nphi, rhob, pef, dtc):
        """Modified from https://github.com/ruben-charles/petrophysical_evaluation_optimization_methods.git
        This module only takes in gr, nphi and rhob to estimate the volumetric of quartz, calcite, dolomite and shale.

        Args:
            gr (float): Gamma Ray log in GAPI.
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.
            dtc (float): Compressional slowness log in us/ft.
            pef (float): Photoelectric Factor log in barns/electron.

        Returns:
            (float, float, float, float, float): vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud
        """
        vol_quartz = np.empty(0)
        vol_calcite = np.empty(0)
        vol_dolomite = np.empty(0)
        vol_shale = np.empty(0)
        vol_mud = np.empty(0)
        for i, input in enumerate(zip(gr, nphi, rhob, pef, dtc)):
            if all([not np.isnan(c) for c in input]):
                print(f'\r Using all inputs {input}', end='')
                res = minimizer_4(input[0], input[1], input[2], input[3], input[4])
            elif not np.isnan(input[3]):
                print(f'\r Using PEF {input}', end='')
                res = minimizer_2(input[0], input[1], input[2], input[3])
            elif not np.isnan(input[4]):
                print(f'\r Using DTC {input}', end='')
                res = minimizer_3(input[0], input[1], input[2], input[4])
            else:
                print(f'\r Using GR, RHOB and NPHI {input}', end='')
                res = minimizer_1(input[0], input[1], input[2])
            vol_quartz = np.append(vol_quartz, res.x[0])
            vol_calcite = np.append(vol_calcite, res.x[1])
            vol_dolomite = np.append(vol_dolomite, res.x[2])
            vol_shale = np.append(vol_shale, res.x[3])
            vol_mud = np.append(vol_mud, res.x[4])

        return vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud


def minimizer_1(gr, nphi, rhob):
    def error_recon(volumes, *args):
        vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud = volumes
        GR, NPHI, RHOB = args
        GR_RECON = vol_quartz*responses["GR_QUARTZ"] + vol_calcite*responses["GR_CALCITE"] + \
            vol_dolomite*responses["GR_DOLOMITE"] + vol_shale*responses["GR_SH"] + vol_mud*responses["GR_MUD"]
        NPHI_RECON = vol_quartz*responses["NPHI_QUARTZ"] + vol_calcite*responses["NPHI_CALCITE"] + \
            vol_dolomite*responses["NPHI_DOLOMITE"] + vol_shale*responses["NPHI_SH"] + vol_mud*responses["NPHI_MUD"]
        RHOB_RECON = vol_quartz*responses["RHOB_QUARTZ"] + vol_calcite*responses["RHOB_CALCITE"] + \
            vol_dolomite*responses["RHOB_DOLOMITE"] + vol_shale*responses["RHOB_SH"] + vol_mud*responses["RHOB_MUD"]

        # Some magic numbers to adjust the precision of differents magnitude orders (needs improvement)
        return (GR - GR_RECON)**2 + (NPHI*300 - NPHI_RECON*300)**2 + (RHOB*100 - RHOB_RECON*100)**2

    return minimize(error_recon, ((0, 0, 0, 0, 0)),
                    args=(gr, nphi, rhob), bounds=bounds, constraints=constrains)


def minimizer_2(gr, nphi, rhob, pef):
    def error_recon(volumes, *args):
        vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud = volumes
        GR, NPHI, RHOB, PEF = args
        GR_RECON = vol_quartz*responses["GR_QUARTZ"] + vol_calcite*responses["GR_CALCITE"] + \
            vol_dolomite*responses["GR_DOLOMITE"] + vol_shale*responses["GR_SH"] + vol_mud*responses["GR_MUD"]
        NPHI_RECON = vol_quartz*responses["NPHI_QUARTZ"] + vol_calcite*responses["NPHI_CALCITE"] + \
            vol_dolomite*responses["NPHI_DOLOMITE"] + vol_shale*responses["NPHI_SH"] + vol_mud*responses["NPHI_MUD"]
        RHOB_RECON = vol_quartz*responses["RHOB_QUARTZ"] + vol_calcite*responses["RHOB_CALCITE"] + \
            vol_dolomite*responses["RHOB_DOLOMITE"] + vol_shale*responses["RHOB_SH"] + vol_mud*responses["RHOB_MUD"]
        PEF_RECON = vol_quartz*responses["PEF_QUARTZ"] + vol_calcite*responses["PEF_CALCITE"] + \
            vol_dolomite*responses["PEF_DOLOMITE"] + vol_shale*responses["PEF_SH"] + vol_mud*responses["PEF_MUD"]

        # Some magic numbers to adjust the precision of differents magnitude orders (needs improvement)
        return (GR - GR_RECON)**2 + (NPHI*300 - NPHI_RECON*300)**2 + (RHOB*100 - RHOB_RECON*100)**2 + \
            (PEF - PEF_RECON)**2

    return minimize(error_recon, ((0, 0, 0, 0, 0)),
                    args=(gr, nphi, rhob, pef), bounds=bounds, constraints=constrains)


def minimizer_3(gr, nphi, rhob, dtc):
    def error_recon(volumes, *args):
        vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud = volumes
        GR, NPHI, RHOB, DTC = args
        GR_RECON = vol_quartz*responses["GR_QUARTZ"] + vol_calcite*responses["GR_CALCITE"] + \
            vol_dolomite*responses["GR_DOLOMITE"] + vol_shale*responses["GR_SH"] + vol_mud*responses["GR_MUD"]
        NPHI_RECON = vol_quartz*responses["NPHI_QUARTZ"] + vol_calcite*responses["NPHI_CALCITE"] + \
            vol_dolomite*responses["NPHI_DOLOMITE"] + vol_shale*responses["NPHI_SH"] + vol_mud*responses["NPHI_MUD"]
        RHOB_RECON = vol_quartz*responses["RHOB_QUARTZ"] + vol_calcite*responses["RHOB_CALCITE"] + \
            vol_dolomite*responses["RHOB_DOLOMITE"] + vol_shale*responses["RHOB_SH"] + vol_mud*responses["RHOB_MUD"]
        DTC_RECON = vol_quartz*responses["DTC_QUARTZ"] + vol_calcite*responses["DTC_CALCITE"] + \
            vol_dolomite*responses["DTC_DOLOMITE"] + vol_shale*responses["DTC_SH"] + vol_mud*responses["DTC_MUD"]

        # Some magic numbers to adjust the precision of differents magnitude orders (needs improvement)
        return (GR - GR_RECON)**2 + (NPHI*300 - NPHI_RECON*300)**2 + (RHOB*100 - RHOB_RECON*100)**2 + \
            (DTC - DTC_RECON)**2

    return minimize(error_recon, ((0, 0, 0, 0, 0)),
                    args=(gr, nphi, rhob, dtc), bounds=bounds, constraints=constrains)


def minimizer_4(gr, nphi, rhob, pef, dtc):
    def error_recon(volumes, *args):
        vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud = volumes
        GR, NPHI, RHOB, PEF, DTC = args
        GR_RECON = vol_quartz*responses["GR_QUARTZ"] + vol_calcite*responses["GR_CALCITE"] + \
            vol_dolomite*responses["GR_DOLOMITE"] + vol_shale*responses["GR_SH"] + vol_mud*responses["GR_MUD"]
        NPHI_RECON = vol_quartz*responses["NPHI_QUARTZ"] + vol_calcite*responses["NPHI_CALCITE"] + \
            vol_dolomite*responses["NPHI_DOLOMITE"] + vol_shale*responses["NPHI_SH"] + vol_mud*responses["NPHI_MUD"]
        RHOB_RECON = vol_quartz*responses["RHOB_QUARTZ"] + vol_calcite*responses["RHOB_CALCITE"] + \
            vol_dolomite*responses["RHOB_DOLOMITE"] + vol_shale*responses["RHOB_SH"] + vol_mud*responses["RHOB_MUD"]
        PEF_RECON = vol_quartz*responses["PEF_QUARTZ"] + vol_calcite*responses["PEF_CALCITE"] + \
            vol_dolomite*responses["PEF_DOLOMITE"] + vol_shale*responses["PEF_SH"] + vol_mud*responses["PEF_MUD"]
        DTC_RECON = vol_quartz*responses["DTC_QUARTZ"] + vol_calcite*responses["DTC_CALCITE"] + \
            vol_dolomite*responses["DTC_DOLOMITE"] + vol_shale*responses["DTC_SH"] + vol_mud*responses["DTC_MUD"]

        # Some magic numbers to adjust the precision of differents magnitude orders (needs improvement)
        return (GR - GR_RECON)**2 + (NPHI*300 - NPHI_RECON*300)**2 + (RHOB*100 - RHOB_RECON*100)**2 + \
            (PEF - PEF_RECON)**2 + (DTC - DTC_RECON)**2

    return minimize(error_recon, ((0, 0, 0, 0, 0)),
                    args=(gr, nphi, rhob, pef, dtc), bounds=bounds, constraints=constrains)
