from collections import OrderedDict
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

MLFLOW_CONFIG = dict(
    local=dict(
        tracking_host='localhost',
        tracking_port=5015,
        backend_store_uri='sqlite:///./mlruns/mlflow.db',
        artifact_location='./mlruns/',
    ),
    remote=dict(
        tracking_host='<your-mlflow-server>',
        tracking_port=5000,
        backend_store_uri='postgresql://<username>:<password>@<hostname>:<port>/<database>?sslmode=require',
        artifact_location='wasbs://<your-blob-storage-account-name>.blob.core.windows.net/<your-container-name>/',
        # artifact_location='s3://<your-s3-bucket-name>/<your-folder-name>/',
    ),
)

RAW_FEATURES = ['GR', 'RT', 'NPHI', 'RHOB']
MODELLING_CONFIG = dict(
    clastic=OrderedDict(
        OIL=dict(
            alg=RandomForestClassifier,
            targets=['OIL_FLAG'],
            features=RAW_FEATURES,
            description='Oil flag classification',
        ),
        GAS=dict(
            alg=RandomForestClassifier,
            targets=['GAS_FLAG'],
            features=RAW_FEATURES + ['OIL_FLAG'],
            description='Gas flag classification',
        ),
        CLASTIC=dict(
            alg=RandomForestRegressor,
            targets=['PHIT', 'SWT', 'LOG_PERM', 'VSAND', 'VSILT', 'VCLW'],
            features=RAW_FEATURES,
            description=('Clastic properties prediction consisting of total porosity, total water saturation, '
                         'log permeability, sand volume, silt volume, and clay volume'),
        ),
    ),
    carbonate=OrderedDict(
        OIL=dict(
            alg=RandomForestClassifier,
            targets=['OIL_FLAG'],
            features=RAW_FEATURES,
            description='Oil flag classification',
        ),
        GAS=dict(
            alg=RandomForestClassifier,
            targets=['GAS_FLAG'],
            features=RAW_FEATURES + ['OIL_FLAG'],
            description='Gas flag classification',
        ),
        CARBONATE=dict(
            alg=RandomForestRegressor,
            targets=['PHIT', 'SWT', 'LOG_PERM', 'VCALC', 'VCLW', 'VDOLO'],
            features=RAW_FEATURES,
            description=('Carbonate properties prediction consisting of total porosity, total water saturation, '
                         'log permeability, calcite volume, clay volume, and dolomite volume'),
        ),
    )
)
