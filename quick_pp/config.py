class Config(object):

    RESSUM_CUTOFFS = dict(
        VSHALE=0.4,
        PHIT=0.05,
        SWT=0.95
    )

    VARS = dict(
        RAW=[
            dict(var="GR", name="Gamma Ray", unit="GAPI", min=0, max=200),
            dict(var="RT", name="True Resistivity", unit="ohm.m", min=0, max=2000),
            dict(var="NPHI", name="Neutron Porosity", unit="v/v", min=-.05, max=0.45),
            dict(var="RHOB", name="Bulk Density", unit="g/cc", min=1.85, max=2.85)
        ],
        LITHOLOGY=[
            dict(var="VSAND", name="Sand Volume", unit="v/v", min=0, max=1),
            dict(var="VSILT", name="Silt Volume", unit="v/v", min=0, max=1),
            dict(var="VCLAY", name="Clay Volume", unit="v/v", min=0, max=1),
            dict(var="VSHALE", name="Shale Volume", unit="v/v", min=0, max=1)
        ],
        POROSITY=[
            dict(var="PHIT", name="Total Porosity", unit="v/v", min=0, max=0.5),
            dict(var="PHIE", name="Effective Porosity", unit="v/v", min=0, max=0.5)
        ],
        SATURATION=[
            dict(var="SWT", name="Total Water Saturation", unit="v/v", min=0, max=1),
            dict(var="SWE", name="Effective Water Saturation", unit="v/v", min=0, max=1)
        ],
        PERMEABILITY=[
            dict(var="PERM", name="Permeability", unit="mD", min=0.01, max=100000)
        ]
    )

    SSC_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_SAND_POINT=(-0.08, 2.65),
        DRY_SILT_POINT=(None, 2.68),  # None means it will be calculated
        DRY_CLAY_POINT=(None, 2.7),  # None means it will be calculated
        WET_CLAY_POINT=(None, None),  # None means it will be calculated
        SILT_LINE_ANGLE=119  # Deg angle from horizontal
    )

    CARB_NEU_DEN_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_CALC_POINT=(0.0, 2.71),
        DRY_DOLO_POINT=(0.009, 2.87),
        DRY_CLAY_POINT=(0.24, 2.78),
    )

    CARB_DEN_PEF_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_CALC_POINT=(5.08, 2.71),
        DRY_DOLO_POINT=(3.14, 2.87),
        DRY_CLAY_POINT=(2.2, 2.78),
    )

    MINERALS_LOG_VALUE = {
        'GR_QUARTZ': 30.0,
        'NPHI_QUARTZ': -0.04,
        'RHOB_QUARTZ': 2.65,
        'DTC_QUARTZ': 55.0,
        'PEF_QUARTZ': 1.8,

        'GR_CALCITE': 0.0,
        'NPHI_CALCITE': 0.0,
        'RHOB_CALCITE': 2.71,
        'DTC_CALCITE': 47.6,
        'PEF_CALCITE': 5.08,

        'GR_DOLOMITE': 0.0,
        'NPHI_DOLOMITE': 0.009,
        'RHOB_DOLOMITE': 2.87,
        'DTC_DOLOMITE': 43.5,
        'PEF_DOLOMITE': 3.14,

        'GR_SH': 110.0,
        'NPHI_SH': 0.3,
        'RHOB_SH': 2.5,
        'DTC_SH': 110.0,
        'PEF_SH': 2.2,

        'GR_MUD': 0.0,
        'NPHI_MUD': 1.0,
        'RHOB_MUD': 1.3,
        'DTC_MUD': 180.0,
        'PEF_MUD': 0.0
    }

    @staticmethod
    def vars_units(data):
        """Return list of units for variables exist in data.columns.

        Returns:
            list: List of units for variables exist in data.columns.
        """
        return {d['var']: d['unit'] for models_, vars_ in Config.VARS.items() for d in vars_
                if d['var'] in data.columns and 'unit' in d.keys()}
