class Config(object):

    RESSUM_CUTOFFS = dict(
        VSHALE=0.4,
        PHIT=0.05,
        SWT=0.95
    )

    VARS = dict(
        GR=dict(name="Gamma Ray", unit="GAPI", min=0, max=200),
        RT=dict(name="True Resistivity", unit="ohm.m", min=0, max=2000),
        NPHI=dict(name="Neutron Porosity", unit="v/v", min=-.05, max=1.0),
        RHOB=dict(name="Bulk Density", unit="g/cc", min=1.0, max=3.0),
        PEF=dict(name="Peak Frequency", unit="Hz", min=0, max=10),
        DTC=dict(name="DTC", unit="us/ft", min=20, max=260),
        DTS=dict(name="DTS", unit="us/ft", min=20, max=260),
        VSAND=dict(name="Sand Volume", unit="v/v", min=0, max=1),
        VSILT=dict(name="Silt Volume", unit="v/v", min=0, max=1),
        VCLAY=dict(name="Clay Volume", unit="v/v", min=0, max=1),
        VSHALE=dict(name="Shale Volume", unit="v/v", min=0, max=1),
        PHIT=dict(name="Total Porosity", unit="v/v", min=0, max=0.5),
        PHIE=dict(name="Effective Porosity", unit="v/v", min=0, max=0.5),
        SWT=dict(vname="Total Water Saturation", unit="v/v", min=0, max=1),
        SWE=dict(vname="Effective Water Saturation", unit="v/v", min=0, max=1),
        PERM=dict(name="Permeability", unit="mD", min=0.01, max=100000)
    )

    SSC_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_SAND_POINT=(-0.02, 2.65),
        DRY_SILT_POINT=(None, 2.68),  # None means it will be calculated
        DRY_CLAY_POINT=(None, 2.71),  # None means it will be calculated
        WET_CLAY_POINT=(None, None),  # None means it will be calculated
        SILT_LINE_ANGLE=119  # Deg angle from horizontal
    )

    TS_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_SAND_POINT=(-0.02, 2.65),
        DRY_SHALE_POINT=(0.44, 2.71),
        DRY_SAND_PORO=.26,
        DRY_SHALE_PORO=.1,
    )

    CARB_NEU_DEN_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_CALC_POINT=(0.0, 2.71),
        DRY_DOLO_POINT=(0.01, 2.87),
        DRY_CLAY_POINT=(0.24, 2.78),
    )

    CARB_DEN_PEF_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_CALC_POINT=(5.08, 2.71),
        DRY_DOLO_POINT=(3.14, 2.87),
        DRY_CLAY_POINT=(2.2, 2.78),
    )

    MINERALS_LOG_VALUE = {
        'GR_QUARTZ': 0.0,
        'NPHI_QUARTZ': -0.02,
        'RHOB_QUARTZ': 2.64,
        'DTC_QUARTZ': 52.9,
        'PEF_QUARTZ': 1.8,

        'GR_CALCITE': 0.0,
        'NPHI_CALCITE': 0.0,
        'RHOB_CALCITE': 2.71,
        'DTC_CALCITE': 49.7,
        'PEF_CALCITE': 5.1,

        'GR_DOLOMITE': 0.0,
        'NPHI_DOLOMITE': 0.01,
        'RHOB_DOLOMITE': 2.85,
        'DTC_DOLOMITE': 43.5,
        'PEF_DOLOMITE': 3.1,

        # Generic Shale (e.g., Illite/Smectite mix)
        'GR_SHALE': 150.0,
        'NPHI_SHALE': 0.30,
        'RHOB_SHALE': 2.55,
        'DTC_SHALE': 120.0,
        'PEF_SHALE': 3.5,

        # Kaolinite (as a separate mineral option)
        'GR_KAOLINITE': 35.0,
        'NPHI_KAOLINITE': 0.37,
        'RHOB_KAOLINITE': 2.41,
        'DTC_KAOLINITE': 143.0,
        'PEF_KAOLINITE': 1.8,

        # Feldspar
        'GR_FELDSPAR': 150.0,
        'NPHI_FELDSPAR': 0.02,
        'RHOB_FELDSPAR': 2.56,
        'DTC_FELDSPAR': 70.0,
        'PEF_FELDSPAR': 3.1,

        # Anhydrite
        'GR_ANHYDRITE': 0.0,
        'NPHI_ANHYDRITE': -0.02,
        'RHOB_ANHYDRITE': 2.98,
        'DTC_ANHYDRITE': 50.0,
        'PEF_ANHYDRITE': 5.1,

        # Gypsum
        'GR_GYPSUM': 0.0,
        'NPHI_GYPSUM': 0.6,
        'RHOB_GYPSUM': 2.35,
        'DTC_GYPSUM': 52.0,
        'PEF_GYPSUM': 4.0,

        # Halite
        'GR_HALITE': 0.0,
        'NPHI_HALITE': -0.03,
        'RHOB_HALITE': 2.04,
        'DTC_HALITE': 67.0,
        'PEF_HALITE': 4.7,

        # Pyrite
        'GR_PYRITE': 0.0,
        'NPHI_PYRITE': -0.02,
        'RHOB_PYRITE': 5.0,
        'DTC_PYRITE': 65.0,
        'PEF_PYRITE': 17.0,

        # Coal
        'GR_COAL': 30.0,
        'NPHI_COAL': 0.50,
        'RHOB_COAL': 1.3,
        'DTC_COAL': 150.0,
        'PEF_COAL': 0.2,
    }

    GEOMECHANICS_VALUE = dict(
        # Quartz
        RHOB_QUARTZ=2.65,
        K_QUARTZ=36.6,
        G_QUARTZ=45.0,
        # Shale
        RHOB_SHALE=2.7,
        K_SHALE=21.0,
        G_SHALE=7.0,
        # Calcite
        RHOB_CALCITE=2.71,
        K_CALCITE=75.0,
        G_CALCITE=30.0,
        # Dolomite
        RHOB_DOLOMITE=2.85,
        K_DOLOMITE=100.0,
        G_DOLOMITE=40.0,
        # Cement
        RHOB_CEMENT=2.65,
        K_CEMENT=37.0,
        G_CEMENT=45.0,
        # Brine
        RHOB_BRINE=1.0,
        K_BRINE=2.5,
        # Oil
        RHOB_OIL=0.8,
        K_OIL=1.5,
        # Gas
        RHOB_GAS=0.2,
        K_GAS=0.06,
    )

    @staticmethod
    def vars_units(data):
        """Return list of units for variables exist in data.columns.

        Returns:
            list: List of units for variables exist in data.columns.
        """
        return {k: v['unit'] for k, v in Config.VARS.items() if k in data.columns and 'unit' in v.keys()}
