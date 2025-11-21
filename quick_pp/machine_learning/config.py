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

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

MLFLOW_CONFIG = dict(
    local=dict(
        tracking_host="localhost",
        tracking_port=5015,
        backend_store_uri="sqlite:///./mlruns/mlflow.db",
        artifact_location="./mlruns/",
    ),
    remote=dict(
        tracking_host="<your-mlflow-server>",
        tracking_port=5000,
        backend_store_uri="postgresql://<username>:<password>@<hostname>:<port>/<database>?sslmode=require",
        artifact_location="wasbs://<your-blob-storage-account-name>.blob.core.windows.net/<your-container-name>/",
        # artifact_location='s3://<your-s3-bucket-name>/<your-folder-name>/',
    ),
)

RAW_FEATURES = ["GR", "RT", "NPHI", "RHOB"]
FE_FEATURES = ["RHOB_INT", "DPHI", "GAS_XOVER"]
MODELLING_CONFIG = dict(
    mock=dict(
        POROSAT=dict(
            alg=RandomForestRegressor,
            targets=["PHIE", "SW"],
            features=RAW_FEATURES,
            description=(
                "Mock clastic properties prediction consisting of total porosity and total water saturation."
            ),
        ),
    ),
    clastic=dict(
        OIL=dict(
            alg=RandomForestClassifier,
            targets=["OIL_FLAG"],
            features=RAW_FEATURES + FE_FEATURES,
            description="Oil flag classification",
        ),
        GAS=dict(
            alg=RandomForestClassifier,
            targets=["GAS_FLAG"],
            features=RAW_FEATURES + FE_FEATURES + ["OIL_FLAG"],
            description="Gas flag classification",
        ),
        CLASTIC=dict(
            alg=RandomForestRegressor,
            targets=["PHIT", "SWT", "LOG_PERM", "VSAND", "VSILT", "VCLAY"],
            features=RAW_FEATURES + FE_FEATURES,
            description=(
                "Clastic properties prediction consisting of total porosity, total water saturation, "
                "log permeability, sand volume, silt volume, and clay volume"
            ),
        ),
    ),
    carbonate=dict(
        OIL=dict(
            alg=RandomForestClassifier,
            targets=["OIL_FLAG"],
            features=RAW_FEATURES,
            description="Oil flag classification",
        ),
        GAS=dict(
            alg=RandomForestClassifier,
            targets=["GAS_FLAG"],
            features=RAW_FEATURES + ["OIL_FLAG"],
            description="Gas flag classification",
        ),
        CARBONATE=dict(
            alg=RandomForestRegressor,
            targets=["PHIT", "SWT", "LOG_PERM", "VCALC", "VCLAY", "VDOLO"],
            features=RAW_FEATURES,
            description=(
                "Carbonate properties prediction consisting of total porosity, total water saturation, "
                "log permeability, calcite volume, clay volume, and dolomite volume"
            ),
        ),
    ),
)
