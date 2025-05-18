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

raw_features = ['GR', 'RT', 'NPHI', 'RHOB']
MODELLING_CONFIG = dict(
    fluid_type=OrderedDict(
        OIL=dict(
            alg=RandomForestClassifier,
            targets=['OIL_FLAG'],
            features=raw_features,
            description='Oil flag classification',
        ),
        GAS=dict(
            alg=RandomForestClassifier,
            targets=['GAS_FLAG'],
            features=raw_features + ['OIL_FLAG'],
            description='Gas flag classification',
        ),
    ),
    clastic=OrderedDict(
        POROSAT=dict(
            alg=RandomForestRegressor,
            targets=['PHIT', 'SWT'],
            features=raw_features,
            description='Total porosity and total water saturation prediction',
        ),
        LITHO=dict(
            alg=RandomForestRegressor,
            targets=['VSAND', 'VSILT', 'VCLW'],
            features=raw_features + ['PHIT', 'SWT'],
            description='Lithology volumetric prediction consisting of sand, silt, and clay',
        ),
        PERM=dict(
            alg=RandomForestRegressor,
            targets=['LOG_PERM'],
            features=raw_features + ['VSAND', 'VSILT', 'VCLW'],
            description='Permeability prediction',
        ),
    ),
    carbonate=OrderedDict(
        CARBONATE=dict(
            alg=RandomForestRegressor,
            targets=['PHIT', 'SWT', 'LOG_PERM', 'VCALC', 'VCLW', 'VDOLO'],
            features=raw_features,
            description=('Carbonate properties prediction consisting of total porosity, total water saturation, '
                         'log permeability, calcite volume, clay volume, and dolomite volume'),
        ),
    )
)
