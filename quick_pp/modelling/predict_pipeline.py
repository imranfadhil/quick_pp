import pandas as pd
import os
from pathlib import Path
from mlflow.pyfunc import load_model
import mlflow.tracking as mlflow_tracking

# # Uncomment below 2 lines to run >> if __name__ == "__main__"
# import sys
# sys.path.append(os.getcwd())

from quick_pp.modelling.config import MODELLING_CONFIG, RAW_FEATURES
from quick_pp.modelling.utils import get_latest_registered_models, unique_id, run_mlflow_server
from quick_pp.logger import logger


def load_data(hash: str) -> pd.DataFrame:
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
        logger.error(f"No file found in {data_dir} containing hash '{hash}'")
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    logger.info(f"Loading data from {path}")
    return pd.read_parquet(path)


def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Postprocess the DataFrame by inverting LOG_PERM to PERM if needed.

    Args:
        df (pd.DataFrame): DataFrame to postprocess.

    Returns:
        pd.DataFrame: Postprocessed DataFrame.
    """
    # Invert LOG_PERM to PERM if exists
    if 'LOG_PERM' in df.columns and 'PERM' not in df.columns:
        logger.info("Inverting LOG_PERM to PERM")
        df['PERM'] = 10 ** df['LOG_PERM']
    return df


def save_predictions(pred_df: pd.DataFrame, output_file_name: str):
    """Save the predictions DataFrame to a Parquet file.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predictions.
        output_file_name (str): Base name for the output file.
    """
    hash = unique_id(pred_df)
    output_dir = Path("data/output/")
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(f"{output_dir}/{output_file_name}_{hash}.parquet")
    pred_df.to_parquet(output_path, index=False)
    logger.info(f"Predictions saved to {output_path}")


def predict_pipeline(model_config: str, data_hash: str, output_file_name: str, env: str = 'local'):
    """
    Run the prediction pipeline: load the latest models, make predictions on input data, postprocess, and save results.

    Args:
        model_config (str): Model configuration key.
        data_hash (str): Hash to identify the input data file.
        output_file_name (str): Base name for the output predictions file.
        env (str, optional): Environment for MLflow server. Defaults to 'local'.
    """
    logger.info("Starting prediction pipeline")
    # Run MLflow server
    run_mlflow_server(env)

    # Get the information of the latest registered models
    client = mlflow_tracking.MlflowClient()
    latest_rms = get_latest_registered_models(client, model_config, data_hash)

    # Load the data
    data = load_data(data_hash)

    pred_df = data[['DEPTH'] + RAW_FEATURES].copy()
    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        targets = model_values['targets']
        features = model_values['features']
        reg_model_name = f'{model_config}_{model_key}_{data_hash}'
        logger.info(f"Predicting with model: {model_key} | {reg_model_name}")

        # Load the model
        model = load_model(latest_rms[reg_model_name]['model_uri'])

        # Run predictions and concat to pred_df
        preds = model.predict(data[features])
        temp_df = pd.DataFrame(preds, columns=targets)
        pred_df = pd.concat([pred_df, temp_df], axis=1)

    # Postprocess the predictions and save
    pred_df = postprocess_data(pred_df)
    save_predictions(pred_df, output_file_name)
    logger.info("Prediction pipeline completed successfully")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prediction pipeline")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config")
    parser.add_argument("--data", type=str, required=True, help="Path to input data parquet file")
    parser.add_argument("--output", type=str, default='test', help="Path to save predictions parquet file")
    args = parser.parse_args()

    # Set up MLflow
    os.makedirs('data/output', exist_ok=True)

    logger.info(f"Model config: {args.model_config}")
    logger.info(f"Data hash: {args.data}")

    predict_pipeline(args.model_config, args.data, args.output)
