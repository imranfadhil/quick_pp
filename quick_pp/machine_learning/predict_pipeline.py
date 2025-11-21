import pandas as pd
import os
from pathlib import Path
from mlflow.pyfunc import load_model
import mlflow.tracking as mlflow_tracking
from tqdm import tqdm

from quick_pp.machine_learning.config import MODELLING_CONFIG, RAW_FEATURES
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import (
    get_latest_registered_models,
    unique_id,
    run_mlflow_server,
)
from quick_pp.plotter.well_log import plotly_log
from quick_pp.fluid_type import fix_fluid_segregation
from quick_pp import logger


def load_data(hash: str) -> pd.DataFrame:
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
        logger.error(f"No file found in {data_dir} containing hash '{hash}'")
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    logger.info(f"Loading data from {path}")
    return pd.read_parquet(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input DataFrame by generating engineered features.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Raises:
        ValueError: If required columns for feature engineering are missing.

    Returns:
        pd.DataFrame: The DataFrame with added feature-engineered columns.
    """
    df = generate_fe_features(df)
    return df


def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Postprocess the DataFrame by inverting LOG_PERM to PERM if needed.
    This function also calculates hydrocarbon volumes and corrects for fluid segregation.

    Args:
        df (pd.DataFrame): The DataFrame containing model predictions.

    Returns:
        pd.DataFrame: The postprocessed DataFrame with added 'PERM', 'VHC', 'VOIL',
                      and 'VGAS' columns.
    """
    # TODO: Clear predictions at COAL_FLAG interval
    # Invert LOG_PERM to PERM if exists
    if "LOG_PERM" in df.columns and "PERM" not in df.columns:
        logger.info("Inverting LOG_PERM to PERM")
        df["PERM"] = 10 ** df["LOG_PERM"]

    # Generate fluid volume fractions if OIL_FLAG and GAS_FLAG exist
    df["VHC"] = (1 - df.get("SWT", 1)) * df.get("PHIT", 0)
    if "OIL_FLAG" in df.columns and "GAS_FLAG" in df.columns:
        df = fix_fluid_segregation(df)

    return df


def save_predictions(pred_df: pd.DataFrame, output_file_name: str, plot: bool = False):
    """Save the predictions DataFrame to a Parquet file.
    Optionally, generate and save individual well log plots.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predictions.
        output_file_name (str): The base name for the output Parquet and plot files.
        plot (bool, optional): If True, generate and save well log plots. Defaults to False.
    """
    hash = unique_id(pred_df)
    output_dir = Path("data/output/")
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(f"{output_dir}/{output_file_name}_{hash}.parquet")
    pred_df.to_parquet(output_path, index=False)

    if plot:
        output_dir = Path("data/output/plots")
        os.makedirs(output_dir, exist_ok=True)
        for well_name, well_df in tqdm(
            pred_df.groupby("WELL_NAME"),
            desc="Generating plots",
        ):
            fig = plotly_log(
                well_df,
                well_name=well_name,
                column_widths=[1, 1, 1, 1, 1, 1, 0.3, 1, 1],
            )
            plot_path = Path(f"{output_dir}/{well_name}.html")
            fig.write_html(plot_path, config=dict(scrollZoom=True))
            tqdm.write(f"Plot for well {well_name} saved to {plot_path}")
    logger.info(f"Predictions saved to {output_path}")


def predict_pipeline(
    model_config: str,
    data_hash: str,
    output_file_name: str,
    env: str = "local",
    plot_predictions: bool = False,
) -> None:
    """Execute the end-to-end prediction pipeline.

    This function orchestrates loading data, preprocessing, loading the latest registered
    MLflow models, making predictions, postprocessing the results, and saving the output.

    Args:
        model_config (str): The key for the model configuration (e.g., 'clastic', 'carbonate').
        data_hash (str): The unique hash identifying the input data file.
        output_file_name (str): The base name for the output predictions file.
        env (str, optional): Environment for MLflow server. Defaults to 'local'.
        plot_predictions (bool, optional): If True, generate plots for each well. Defaults to False.
    """
    logger.info("Starting prediction pipeline")
    # Run MLflow server
    run_mlflow_server(env)

    # Get the information of the latest registered models
    client = mlflow_tracking.MlflowClient()
    latest_rms = get_latest_registered_models(client, model_config, data_hash)

    # Load the data
    data = load_data(data_hash)
    data = preprocess_data(data)

    pred_df = data[["DEPTH", "WELL_NAME"] + RAW_FEATURES].copy()
    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        targets = model_values["targets"]
        features = model_values["features"]
        reg_model_name = f"{model_config}_{model_key}_{data_hash}"
        logger.info(f"Predicting with model: {model_key} | {reg_model_name}")

        # Load the model
        model = load_model(latest_rms[reg_model_name]["model_uri"])

        # Run predictions and concat to pred_df
        preds = model.predict(data[features].astype("float"))
        temp_df = pd.DataFrame(preds, columns=targets)
        pred_df = pd.concat([pred_df, temp_df], axis=1)

    # Merge specific columns from original data missing from pred_df
    merge_cols = ["WELL_NAME", "DEPTH"]
    missing_cols = ["ROCK_FLAG", "COAL_FLAG", "TIGHT_FLAG"]
    pred_df = pd.merge(
        pred_df, data[merge_cols + missing_cols], on=merge_cols, how="left"
    )

    # Postprocess the predictions and save
    pred_df = postprocess_data(pred_df)
    save_predictions(pred_df, output_file_name, plot=plot_predictions)
    logger.info("Prediction pipeline completed successfully")
