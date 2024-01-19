from scipy.optimize import minimize

from ..config import Config


def multi_mineral_model(gr, nphi, rhob):
    """Modified from https://github.com/ruben-charles/petrophysical_evaluation_optimization_methods.git

    Args:
        df (pd.DataFrame): _description_

    Returns:
        _type_: _description_
    """
    # Getting default values from Config which may need to be changed based on the dataset
    responses = Config.MINERALS_LOG_VALUE

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
    for i, input in enumerate(zip(gr, nphi, rhob)):
        res = minimize(error_recon, ((0, 0, 0, 0, 0)),
                       args=input, bounds=bounds, constraints=constrains)
        vol_quartz.append(res.x[0])
        vol_calcite.append(res.x[1])
        vol_dolomite.append(res.x[2])
        vol_shale.append(res.x[3])
        vol_mud.append(res.x[4])

    return vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud
