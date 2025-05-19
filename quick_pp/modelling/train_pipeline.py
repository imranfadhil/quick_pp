import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score
import mlflow
import mlflow.sklearn as mlflow_sklearn
from mlflow.models.signature import infer_signature

# # Uncomment below 3 lines to run >> if __name__ == "__main__"
# import os
# import sys
# sys.path.append(os.getcwd())

from quick_pp.modelling.config import MODELLING_CONFIG
from quick_pp.modelling.utils import run_mlflow_server


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


def preprocess_data(
        df: pd.DataFrame, target_column: list[str], features: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preprocess the DataFrame by adding a log-perm column if needed and dropping rows with NaN values.

    Args:
        df (pd.DataFrame): DataFrame to preprocess.
        target_column (list[str]): Target column(s) for the model.
        features (list[str]): Feature columns for the model.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: Tuple containing the features DataFrame and target DataFrame.
    """
    # Add log perm if not already present
    if 'LOG_PERM' not in df.columns and 'PERM' in df.columns and (
            'LOG_PERM' in target_column or 'LOG_PERM' in features):
        df['LOG_PERM'] = np.log10(df['PERM'].clip(lower=1e-3))

    # Drop rows with NaN in target or features
    return_df = df.dropna(subset=target_column + features)
    X = return_df[features]
    y = return_df[target_column]
    return X, y


def split_data(X, y, test_size=0.2, random_state=42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the data into training and testing sets.

    Args:
        X (pd.DataFrame): Feature DataFrame.
        y (pd.DataFrame): Target DataFrame.
        test_size (float, optional): Proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing the training and testing sets.
    """
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
    model = alg(random_state=42)
    model.fit(X_train, y_train)
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
    y_pred = model.predict(X_test)

    # Check model type
    if hasattr(model, "predict_proba"):
        return dict(
            f1_score=f1_score(y_test, y_pred),
            accuracy=accuracy_score(y_test, y_pred),
        )
    else:
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
    # Run MLflow server
    run_mlflow_server(env)

    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        alg = model_values['alg']
        targets = model_values['targets']
        features = model_values['features']

        if not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)):
            raise TypeError(f"Targets must be a list of strings, got {targets}")
        if not (isinstance(features, list) and all(isinstance(f, str) for f in features)):
            raise TypeError(f"Features must be a list of strings, got {features}")

        df = load_data(data_hash)
        X, y = preprocess_data(df, targets, features)
        X_train, X_test, y_train, y_test = split_data(X, y)
        mlflow.set_experiment(model_config)
        with mlflow.start_run(run_name=model_key, description=str(model_values['description'])):
            # Train model
            model = train_model(alg, X_train, y_train)

            # Log metrics
            metrics_dict = evaluate_model(model, X_test, y_test)
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, float(metric_value))

            # Log model
            signature = infer_signature(X_train, y_train)
            mlflow_sklearn.log_model(
                model, "model", signature=signature, input_example=X_test.sample(),
                registered_model_name=f'{model_config}_{model_key}')


if __name__ == "__main__":
    import os

    # Set up MLflow
    os.makedirs('./mlruns', exist_ok=True)
    os.makedirs('data/input', exist_ok=True)

    # Example usage
    data_hash = "x2x2"  # Update with your hash for your data in the 'data/input/' folder
    model_config = "carbonate"  # Update with your model config
    train_pipeline(model_config, data_hash)
