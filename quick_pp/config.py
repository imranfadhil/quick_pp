"""
Centralized configuration for the quick_pp library.

This module defines the `Config` class, which serves as a namespace for
storing static configuration data used across various modules. This includes
default petrophysical parameters, rock physics model endpoints, and mappings.

Centralizing configuration here allows for easier management and modification
of default values without altering the core logic of the functions.
"""


class Config:
    """A class to hold static configuration data for the quick_pp library."""

    # Default cutoffs for reservoir summary calculations
    RESSUM_CUTOFFS = dict(VSHALE=0.4, PHIT=0.05, SWT=0.95)

    # Defines properties for common well log variables, including names, units, and typical ranges.
    VARS = dict(
        GR=dict(name="Gamma Ray", unit="GAPI", min=0, max=200),
        RT=dict(name="True Resistivity", unit="ohm.m", min=0, max=2000),
        NPHI=dict(name="Neutron Porosity", unit="v/v", min=-0.05, max=1.0),
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
        PERM=dict(name="Permeability", unit="mD", min=0.01, max=100000),
    )

    # Default endpoints for the Sand-Silt-Clay lithology model.
    SSC_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_SAND_POINT=(-0.02, 2.65),
        DRY_SILT_POINT=(None, 2.68),  # None means it will be calculated
        DRY_CLAY_POINT=(None, 2.71),  # None means it will be calculated
        WET_CLAY_POINT=(None, None),  # None means it will be calculated
        SILT_LINE_ANGLE=119,  # Deg angle from horizontal
    )

    # Default endpoints for the Thin-Beds lithology model.
    TS_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_SAND_POINT=(-0.02, 2.65),
        DRY_SHALE_POINT=(0.44, 2.71),
        DRY_SAND_PORO=0.26,
        DRY_SHALE_PORO=0.1,
    )

    # Default endpoints for the Carbonate Neutron-Density crossplot model.
    CARB_NEU_DEN_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_CALC_POINT=(0.0, 2.71),
        DRY_DOLO_POINT=(0.01, 2.87),
        DRY_CLAY_POINT=(0.24, 2.78),
    )

    # Default endpoints for the Carbonate Density-PEF crossplot model.
    CARB_DEN_PEF_ENDPOINTS = dict(
        FLUID_POINT=(1.0, 1.0),
        DRY_CALC_POINT=(5.08, 2.71),
        DRY_DOLO_POINT=(3.14, 2.87),
        DRY_CLAY_POINT=(2.2, 2.78),
    )

    # Typical log responses for various pure minerals.
    MINERALS_LOG_VALUE = {
        "QUARTZ": {"GR": 0.0, "NPHI": -0.02, "RHOB": 2.64, "DTC": 52.9, "PEF": 1.8},
        "CALCITE": {"GR": 0.0, "NPHI": 0.0, "RHOB": 2.71, "DTC": 49.7, "PEF": 5.1},
        "DOLOMITE": {"GR": 0.0, "NPHI": 0.01, "RHOB": 2.85, "DTC": 43.5, "PEF": 3.1},
        "SHALE": {"GR": 150.0, "NPHI": 0.30, "RHOB": 2.55, "DTC": 120.0, "PEF": 3.5},
        "KAOLINITE": {"GR": 35.0, "NPHI": 0.37, "RHOB": 2.41, "DTC": 143.0, "PEF": 1.8},
        "FELDSPAR": {"GR": 150.0, "NPHI": 0.02, "RHOB": 2.56, "DTC": 70.0, "PEF": 3.1},
        "ANHYDRITE": {"GR": 0.0, "NPHI": -0.02, "RHOB": 2.98, "DTC": 50.0, "PEF": 5.1},
        "GYPSUM": {"GR": 0.0, "NPHI": 0.6, "RHOB": 2.35, "DTC": 52.0, "PEF": 4.0},
        "HALITE": {"GR": 0.0, "NPHI": -0.03, "RHOB": 2.04, "DTC": 67.0, "PEF": 4.7},
        "PYRITE": {"GR": 0.0, "NPHI": -0.02, "RHOB": 5.0, "DTC": 65.0, "PEF": 17.0},
        "COAL": {"GR": 30.0, "NPHI": 0.50, "RHOB": 1.3, "DTC": 150.0, "PEF": 0.2},
    }

    # Maps mineral names to their corresponding volume curve mnemonics (e.g., QUARTZ -> VSAND).
    MINERALS_NAME_MAPPING = {
        "QUARTZ": "VSAND",
        "CALCITE": "VCALC",
        "DOLOMITE": "VDOLO",
        "SHALE": "VCLAY",
        "KAOLINITE": "VKAOL",
        "FELDSPAR": "VFELD",
        "ANHYDRITE": "VANHY",
        "GYPSUM": "VGYPS",
        "HALITE": "VHALI",
        "PYRITE": "VPYRI",
        "COAL": "VCOAL",
    }

    # Elastic and density properties for minerals and fluids used in geomechanics and rock physics.
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

    # A dictionary of common geological abbreviations and their expanded forms.
    # This can be expanded with more terms as needed.
    CORE_GEO_ABBREVIATIONS = {
        "ss": "sandstone",
        "sh": "shale",
        "ls": "limestone",
        "dol": "dolomite",
        "cgl": "conglomerate",
        "gr": "grained",
        "crs": "coarse",
        "med": "medium",
        "f": "fine",
        "fg": "fine grain",
        "vf": "very fine",
        "sl": "slightly",
        "spar": "sparry",
        "cmt": "cemented",
        "lam": "laminated",
        "xbed": "cross-bedded",
        "bioturb": "bioturbated",
        "arg": "argillaceous",
        "calc": "calcareous",
        "calcar": "calcareous",
        "calclut": "calcilutite",
        "lut": "lutite",
        "carb": "carbonaceous",
        "glauc": "glauconitic",
        "pyr": "pyritic",
        "gy": "grey",
        "wh": "white",
        "crm": "cream",
        "shale": "shale",
        "siltstone": "siltstone",
        "stylolite": "stylolite",
    }

    # Categories of geological terms (expanded form) used for structured naming.
    # The order of keys in this dictionary determines the final name structure.
    CORE_WORD_CATEGORIES = {
        "FORMATION": [
            "sandstone",
            "shale",
            "siltstone",
            "limestone",
            "dolomite",
            "conglomerate",
            "calcilutite",
            "lutite",
        ],
        "GRAIN_SIZE": [
            "grained",
            "coarse",
            "medium",
            "fine",
            "very fine",
            "rubble",
            "sparry",
        ],
        "MODIFIER": [
            "slightly",
            "laminated",
            "cross-bedded",
            "bioturbated",
            "argillaceous",
            "calcareous",
            "carbonaceous",
            "glauconitic",
            "pyritic",
            "cemented",
            "stylolite",
        ],
    }

    # Define special case descriptions that should be handled directly, not clustered.
    # The key is the phrase to search for (case-insensitive), and the value is the group name to assign.
    CORE_SPECIAL_CASE_DESCRIPTIONS = {
        "no plug possible": "NO_PLUG_POSSIBLE",
        "fractured": "FRACTURED",
        "preserved sample": "PRESERVED_SAMPLE",
        "rubble": "RUBBLE",
    }

    @staticmethod
    def vars_units(data):
        """Return a dictionary of units for variables present in the DataFrame.

        Args:
            data (pd.DataFrame): The DataFrame whose columns will be checked.

        Returns:
            dict: A dictionary mapping variable names (mnemonics) to their units.
        """
        return {
            k: v["unit"]
            for k, v in Config.VARS.items()
            if k in data.columns and "unit" in v.keys()
        }
