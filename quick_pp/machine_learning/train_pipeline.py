import importlib.util
import os
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from quick_pp import logger
from quick_pp.machine_learning.config import CONFORMAL_ALPHA, MODELLING_CONFIG
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import (
    ConformalMultiOutputRegressor,
    physics_informed_loss,
    run_mlflow_server,
)

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
            logger.warning(
                "MODELLING_CONFIG not found in root config.py, using default MODELLING_CONFIG"
            )
    except Exception as e:
        logger.warning(f"Could not import MODELLING_CONFIG from root config.py: {e}")


def load_data(hash: str):
    """Load data from the specified directory using a hash to identify the file.

    Args:
        hash (str): A unique hash string contained within the target Parquet filename.

    Raises:
        FileNotFoundError: If no file is found with the specified hash.

    Returns:
        pd.DataFrame: The loaded well log data as a DataFrame.
    """
    data_dir = Path("data/input/")
    matching_files = list(data_dir.glob(f"*{hash}*.parquet"))
    if not matching_files:
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    return pd.read_parquet(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by generating features and cleaning the data.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with engineered features and
                      duplicates removed.
    """
    df = generate_fe_features(df)
    # Drop duplicates based on WELL_NAME and DEPTH, including duplicated columns
    df = df.drop_duplicates(subset=["WELL_NAME", "DEPTH"])
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def split_data(
    df: pd.DataFrame,
    target_column: list[str],
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "WELL_NAME",
    splitter: str = "group_shuffle",
    n_splits: int = 5,
) -> list:
    """Split the data into training and testing sets using group-aware splitting.

    Uses `GroupShuffleSplit` (or `GroupKFold`) grouped by `group_col` (default `'WELL_NAME'`)
    so entire wells are held out together, preventing depth-adjacent spatial autocorrelation
    leakage between train and test sets. If fewer than 2 unique wells exist in `df`,
    automatically falls back to grouping by `'ZONES'` or contiguous depth blocks.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        target_column (list[str]): List of target column names.
        features (list[str]): List of feature column names.
        test_size (float, optional): Proportion of groups/data to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        group_col (str, optional): Column name used to define groups. Defaults to 'WELL_NAME'.
        splitter (str, optional): Splitting strategy: 'group_shuffle' (`GroupShuffleSplit`) or
                                  'group_kfold' (`GroupKFold`). Defaults to 'group_shuffle'.
        n_splits (int, optional): Number of folds when using `GroupKFold` or constructing
                                  contiguous depth blocks. Defaults to 5.

    Returns:
        list: A list containing `[X_train, X_test, y_train, y_test]`.
    """
    # Drop rows with NaN in target or features
    return_df = df.dropna(subset=target_column + features).copy()
    X = return_df[features].astype("float")
    y = return_df[target_column].astype("float")

    # Determine group labels to prevent spatial / well leakage
    if group_col in return_df.columns and return_df[group_col].nunique(dropna=True) >= 2:
        groups = return_df[group_col].astype(str).to_numpy()
        logger.info(
            f"Splitting data by '{group_col}' ({return_df[group_col].nunique()} unique groups) using {splitter}"
        )
    elif "ZONES" in return_df.columns and return_df["ZONES"].nunique(dropna=True) >= 2:
        groups = return_df["ZONES"].astype(str).to_numpy()
        logger.info(
            f"Single well detected; splitting data by 'ZONES' ({return_df['ZONES'].nunique()} zones) using {splitter}"
        )
    else:
        # Construct contiguous depth-interval blocks so adjacent samples do not leak across train/test
        n_blocks = max(2, min(max(n_splits, int(round(1.0 / max(test_size, 0.05)))), len(return_df) // 2))
        if len(return_df) >= 4 and n_blocks >= 2:
            groups = np.floor(np.linspace(0, n_blocks, len(return_df), endpoint=False)).astype(int)
            logger.info(
                f"Single well without multiple zones; splitting by {n_blocks} contiguous depth blocks using {splitter}"
            )
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)

    n_unique_groups = len(np.unique(groups))
    splitter_key = splitter.lower().strip()

    if splitter_key in ("group_kfold", "groupkfold", "gkf") and n_unique_groups >= 2:
        eff_splits = min(max(2, n_splits), n_unique_groups)
        gkf = GroupKFold(n_splits=eff_splits)
        train_idx, test_idx = next(gkf.split(X, y, groups=groups))
    else:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return [
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    ]


def train_model(
    alg,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    use_conformal: bool = True,
    alpha: float = CONFORMAL_ALPHA,
):
    """Train the model using the specified algorithm, wrapping regressors in Conformal Prediction.

    Args:
        alg (callable): The scikit-learn model class to instantiate.
        X_train (pd.DataFrame): The training feature DataFrame.
        y_train (pd.DataFrame): The training target DataFrame.
        use_conformal (bool, optional): Whether to wrap regression models in
            `ConformalMultiOutputRegressor` for calibrated P10–P90 intervals. Defaults to True.
        alpha (float, optional): Significance level (0.20 -> P10 to P90). Defaults to CONFORMAL_ALPHA.

    Returns:
        object: The trained model instance.
    """
    logger.info(f"Training model: {getattr(alg, '__name__', str(alg))}")
    base_model = alg(random_state=42)

    if hasattr(base_model, "predict_proba") or not use_conformal:
        y_fit = (
            y_train.iloc[:, 0]
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1
            else y_train
        )
        base_model.fit(X_train, y_fit)
        logger.debug("Classifier / standard model training complete")
        return base_model

    target_names = list(y_train.columns) if isinstance(y_train, pd.DataFrame) else None
    model = ConformalMultiOutputRegressor(
        base_estimator=base_model,
        alpha=alpha,
        random_state=42,
        enforce_physics=True,
        target_names=target_names,
    )
    model.fit(X_train, y_train)
    logger.debug("Conformal regression model training complete")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Evaluate the model using the test data, including physics loss and interval coverage.

    Args:
        model (object): The trained model to evaluate.
        X_test (pd.DataFrame): The testing feature DataFrame.
        y_test (pd.DataFrame): The testing target DataFrame.

    Returns:
        dict: A dictionary of evaluation metrics (e.g., 'f1_score', 'r2_score',
              'physics_loss', 'p10_p90_coverage').
    """
    logger.info(f"Evaluating model: {type(model).__name__}")
    y_pred = model.predict(X_test)

    # Check model type
    if hasattr(model, "predict_proba"):
        logger.debug("Classification metrics calculated")
        return {
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }
    else:
        logger.debug("Regression metrics calculated")
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "physics_loss": physics_informed_loss(y_test, y_pred, X=X_test),
        }
        if hasattr(model, "predict_interval"):
            _, y_p10, y_p90 = model.predict_interval(X_test)
            y_true_arr = (
                y_test.to_numpy(dtype=float)
                if isinstance(y_test, pd.DataFrame)
                else np.asarray(y_test, dtype=float)
            )
            if y_true_arr.ndim == 1:
                y_true_arr = y_true_arr.reshape(-1, 1)
            y_p10_arr = np.asarray(y_p10, dtype=float).reshape(y_true_arr.shape)
            y_p90_arr = np.asarray(y_p90, dtype=float).reshape(y_true_arr.shape)
            in_bounds = (y_true_arr >= y_p10_arr) & (y_true_arr <= y_p90_arr)
            metrics["p10_p90_coverage"] = float(np.nanmean(in_bounds))
        return metrics


# 6. Train pipeline
# 6. Train pipeline
def is_training_disabled() -> bool:
    """Return True if heavy model training is disabled (e.g. in the public Hugging Face Spaces demo)."""
    return (
        os.getenv("QPP_DEMO_MODE", "false").lower() in ("1", "true", "yes")
        or os.getenv("QPP_DISABLE_TRAINING", "false").lower() in ("1", "true", "yes")
    )


def train_pipeline(model_config: str, data_hash: str, env: str = "local"):
    """Execute the end-to-end model training pipeline.

    This function automates training, evaluating, and logging multiple models as defined
    in the configuration, leveraging MLflow for experiment tracking and model management.

    Args:
        model_config (str): The key for the model configuration (e.g., 'clastic').
        data_hash (str): The unique hash identifying the data file.
        env (str, optional): The MLflow environment ('local' or 'remote'). Defaults to 'local'.

    Raises:
        TypeError: If the targets or features are not lists of strings.
    """
    logger.info(
        f"Starting train_pipeline with model_config={model_config}, data_hash={data_hash}, env={env}"
    )

    if is_training_disabled():
        msg = (
            "Model training (train_pipeline.py) is disabled in the public interactive demo "
            "(QPP_DEMO_MODE=true). Please use pre-trained PIML & Conformal Prediction inference instead."
        )
        logger.warning(msg)
        raise PermissionError(msg)

    # Check if the model_config exists in MODELLING_CONFIG
    if model_config not in MODELLING_CONFIG:
        error_msg = (
            f"Model configuration '{model_config}' not found in MODELLING_CONFIG"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Run MLflow server
    run_mlflow_server(env)

    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        alg = model_values["alg"]
        targets = model_values["targets"]
        features = model_values["features"]

        logger.info(f"Processing model: {model_key}")
        if not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)):
            logger.error(f"Targets must be a list of strings, got {targets}")
            raise TypeError(f"Targets must be a list of strings, got {targets}")
        if not (
            isinstance(features, list) and all(isinstance(f, str) for f in features)
        ):
            logger.error(f"Features must be a list of strings, got {features}")
            raise TypeError(f"Features must be a list of strings, got {features}")

        df = load_data(data_hash)
        df = preprocess_data(df)
        # Skip if targets or features are not in the DataFrame
        if not all(col in df.columns for col in targets + features):
            missing_cols = [col for col in targets + features if col not in df.columns]
            logger.warning(
                f"Skipping model {model_key} due to missing columns: {missing_cols}"
            )
            continue
        X_train, X_test, y_train, y_test = split_data(df, targets, features)

        mlflow_dir = Path("./mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_experiment(model_config)
        with mlflow.start_run(
            run_name=model_key, description=str(model_values["description"])
        ):
            # Train model
            model = train_model(alg, X_train, y_train)

            # Log metrics
            metrics_dict = evaluate_model(model, X_test, y_test)
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, float(metric_value))
                logger.info(f"Logged metric: {metric_name}={metric_value}")

            # Log model
            reg_model_name = f"{model_config}_{model_key}_{data_hash}"
            signature = infer_signature(X_train, y_train)
            mlflow_sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_test.sample(5),
                registered_model_name=reg_model_name,
            )
            logger.info(f"Model logged and registered as: {reg_model_name}")
