import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.models.signature import infer_signature
import importlib.util

from quick_pp.machine_learning.config import MODELLING_CONFIG
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import run_mlflow_server
from quick_pp import logger

# Check if config.py exists in the root directory and update MODELLING_CONFIG if found
root_config_path = Path(os.getcwd(), "config.py")
if root_config_path.exists():
    try:
        spec = importlib.util.spec_from_file_location("root_config", root_config_path)
        root_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_config)
        if hasattr(root_config, "MODELLING_CONFIG"):
            logger.info("Updating MODELLING_CONFIG from root config.py")
            MODELLING_CONFIG.clear()
            MODELLING_CONFIG.update(root_config.MODELLING_CONFIG)
        else:
            logger.warning("MODELLING_CONFIG not found in root config.py, using default MODELLING_CONFIG")
    except Exception as e:
        logger.warning(f"Could not import MODELLING_CONFIG from root config.py: {e}")


def load_data(hash: str):
    """Load data from the specified directory using a hash to identify the file.

    Args:
        hash (str): Hash to identify the file.

    Raises:
        FileNotFoundError: If no file is found with the specified hash.

    Returns:
        pd.DataFrame: Loaded DataFrame.
    """
    data_dir = Path("data/input/")
    matching_files = list(data_dir.glob(f"*{hash}*.parquet"))
    if not matching_files:
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    return pd.read_parquet(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by adding a log-perm column if needed and dropping rows with NaN values.

    Args:
        df (pd.DataFrame): DataFrame to preprocess.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = generate_fe_features(df)
    # Drop duplicates based on WELL_NAME and DEPTH, including duplicated columns
    df = df.drop_duplicates(subset=['WELL_NAME', 'DEPTH'])
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def split_data(df: pd.DataFrame, target_column: list[str], features: list[str], test_size=0.2, random_state=42) -> list:
    """Split the data into training and testing sets.

    Args:
        df (pd.DataFrame): DataFrame to split.
        target_column (list[str]): List of target column names.
        features (list[str]): List of feature column names.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list: List containing the training and testing sets for features and target.
    """
    # Drop rows with NaN in target or features
    return_df = df.dropna(subset=target_column + features)
    X = return_df[features].astype('float')
    y = return_df[target_column].astype('float')
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def train_model(alg, X_train: pd.DataFrame, y_train: pd.DataFrame):
    """Train the model using the specified algorithm.

    Args:
        alg (_type_): Algorithm to use for training.
        X_train (pd.DataFrame): Feature DataFrame.
        y_train (pd.DataFrame): Target DataFrame.

    Returns:
        _type_: _description_
    """
    logger.info(f"Training model: {getattr(alg, '__name__', str(alg))}")
    model = alg(random_state=42)
    model.fit(X_train, y_train)
    logger.debug("Model training complete")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Evaluate the model using the test data.

    Args:
        model (_type_): Trained model to evaluate.
        X_test (pd.DataFrame): Feature DataFrame for testing.
        y_test (pd.DataFrame): Target DataFrame for testing.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    logger.info(f"Evaluating model: {type(model).__name__}")
    y_pred = model.predict(X_test)

    # Check model type
    if hasattr(model, "predict_proba"):
        logger.debug("Classification metrics calculated")
        return dict(
            f1_score=f1_score(y_test, y_pred),
            accuracy=accuracy_score(y_test, y_pred),
        )
    else:
        logger.debug("Regression metrics calculated")
        return dict(
            r2_score=r2_score(y_test, y_pred),
            mean_absolute_error=mean_absolute_error(y_test, y_pred),
        )


# 6. Train pipeline
def train_pipeline(model_config: str, data_hash: str, env: str = 'local'):
    """This function automates the process of training, evaluating, and logging multiple models as defined in
    a configuration, leveraging MLflow for experiment tracking and model management.

    Args:
        model_config (str): Key for the model configuration in the MODELLING_CONFIG dictionary.
        data_hash (str): Hash to identify the data file in the 'data/input/' directory.
        env (str, optional): MLflow environment to use. Defaults to 'local'.

    Raises:
        TypeError: If the targets or features are not lists of strings.
    """
    logger.info(f"Starting train_pipeline with model_config={model_config}, data_hash={data_hash}, env={env}")

    # Check if the model_config exists in MODELLING_CONFIG
    if model_config not in MODELLING_CONFIG:
        error_msg = f"Model configuration '{model_config}' not found in MODELLING_CONFIG"
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Run MLflow server
    run_mlflow_server(env)

    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        alg = model_values['alg']
        targets = model_values['targets']
        features = model_values['features']

        logger.info(f"Processing model: {model_key}")
        if not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)):
            logger.error(f"Targets must be a list of strings, got {targets}")
            raise TypeError(f"Targets must be a list of strings, got {targets}")
        if not (isinstance(features, list) and all(isinstance(f, str) for f in features)):
            logger.error(f"Features must be a list of strings, got {features}")
            raise TypeError(f"Features must be a list of strings, got {features}")

        df = load_data(data_hash)
        df = preprocess_data(df)
        # Skip if targets or features are not in the DataFrame
        if not all(col in df.columns for col in targets + features):
            missing_cols = [col for col in targets + features if col not in df.columns]
            logger.warning(f"Skipping model {model_key} due to missing columns: {missing_cols}")
            continue
        X_train, X_test, y_train, y_test = split_data(df, targets, features)

        mlflow_dir = Path('./mlruns')
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_experiment(model_config)
        with mlflow.start_run(run_name=model_key, description=str(model_values['description'])):
            # Train model
            model = train_model(alg, X_train, y_train)

            # Log metrics
            metrics_dict = evaluate_model(model, X_test, y_test)
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, float(metric_value))
                logger.info(f"Logged metric: {metric_name}={metric_value}")

            # Log model
            reg_model_name = f'{model_config}_{model_key}_{data_hash}'
            signature = infer_signature(X_train, y_train)
            mlflow_sklearn.log_model(
                model, "model", signature=signature, input_example=X_test.sample(5),
                registered_model_name=reg_model_name)
            logger.info(f"Model logged and registered as: {reg_model_name}")
