"""
Configuration for Machine Learning Pipelines.

This module centralizes the configuration for MLflow tracking, feature definitions,
and model specifications used in the training and prediction pipelines.

Attributes:
    MLFLOW_CONFIG (dict): Contains settings for local and remote MLflow tracking servers.
    RAW_FEATURES (list): A list of raw well log mnemonics used as base features.
    FE_FEATURES (list): A list of feature-engineered column names.
    MODELLING_CONFIG (dict): A nested dictionary defining the specifications for different
                             modelling suites (e.g., clastic, carbonate), including the
                             algorithm, target variables, features, and a description for each model.
"""

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

MLFLOW_CONFIG = {
    "local": {
        "tracking_host": "localhost",
        "tracking_port": 5015,
        "backend_store_uri": "sqlite:///./mlruns/mlflow.db",
        "artifact_location": "./mlruns/",
    },
    "remote": {
        "tracking_host": "<your-mlflow-server>",
        "tracking_port": 5000,
        "backend_store_uri": "postgresql://<username>:<password>@<hostname>:<port>/<database>?sslmode=require",
        "artifact_location": "wasbs://<your-blob-storage-account-name>.blob.core.windows.net/<your-container-name>/",
        # artifact_location='s3://<your-s3-bucket-name>/<your-folder-name>/',
    },
}

RAW_FEATURES = ["GR", "RT", "NPHI", "RHOB"]
PHYSICS_FE_FEATURES = ["RHOMA_APP", "AI", "M_LITH", "N_LITH"]
FE_FEATURES = ["RHOB_INT", "DPHI", "GAS_XOVER"] + PHYSICS_FE_FEATURES

# Default significance level alpha=0.20 corresponds to an 80% prediction interval (P10 to P90)
CONFORMAL_ALPHA = 0.20

# Standard bulk-volume mineral columns subject to closure: sum(V_mineral) + PHIT = 1.0
MINERAL_VOL_COLS = [
    "VSAND",
    "VSILT",
    "VCLAY",
    "VCALC",
    "VDOLO",
    "VKAOL",
    "VFELD",
    "VANHY",
    "VGYPS",
    "VHALI",
    "VPYRI",
    "VCOAL",
]

# Physical bounds for petrophysical properties
PHYSICAL_BOUNDS = {
    "PHIT": (0.0, 0.50),
    "PHIE": (0.0, 0.50),
    "SWT": (0.0, 1.0),
    "SW": (0.0, 1.0),
    "SWE": (0.0, 1.0),
    "LOG_PERM": (-3.0, 5.0),
    "PERM": (1e-3, 1e5),
}

MODELLING_CONFIG = {
    "mock": {
        "POROSAT": {
            "alg": RandomForestRegressor,
            "targets": ["PHIE", "SW"],
            "features": RAW_FEATURES,
            "description": (
                "Mock clastic properties prediction consisting of total porosity and total water saturation."
            ),
        },
    },
    "clastic": {
        "OIL": {
            "alg": RandomForestClassifier,
            "targets": ["OIL_FLAG"],
            "features": RAW_FEATURES + FE_FEATURES,
            "description": "Oil flag classification",
        },
        "GAS": {
            "alg": RandomForestClassifier,
            "targets": ["GAS_FLAG"],
            "features": RAW_FEATURES + FE_FEATURES + ["OIL_FLAG"],
            "description": "Gas flag classification",
        },
        "CLASTIC": {
            "alg": RandomForestRegressor,
            "targets": ["PHIT", "SWT", "LOG_PERM", "VSAND", "VSILT", "VCLAY"],
            "features": RAW_FEATURES + FE_FEATURES,
            "description": (
                "Clastic properties prediction consisting of total porosity, total water saturation, "
                "log permeability, sand volume, silt volume, and clay volume"
            ),
        },
    },
    "carbonate": {
        "OIL": {
            "alg": RandomForestClassifier,
            "targets": ["OIL_FLAG"],
            "features": RAW_FEATURES + FE_FEATURES,
            "description": "Oil flag classification",
        },
        "GAS": {
            "alg": RandomForestClassifier,
            "targets": ["GAS_FLAG"],
            "features": RAW_FEATURES + FE_FEATURES + ["OIL_FLAG"],
            "description": "Gas flag classification",
        },
        "CARBONATE": {
            "alg": RandomForestRegressor,
            "targets": ["PHIT", "SWT", "LOG_PERM", "VCALC", "VCLAY", "VDOLO"],
            "features": RAW_FEATURES + FE_FEATURES,
            "description": (
                "Carbonate properties prediction consisting of total porosity, total water saturation, "
                "log permeability, calcite volume, clay volume, and dolomite volume"
            ),
        },
    },
}
