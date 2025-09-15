import pandas as pd
import numpy as np
import os
from pathlib import Path
from mlflow.pyfunc import load_model
import mlflow.tracking as mlflow_tracking
from tqdm import tqdm

# # Uncomment below 2 lines to run >> if __name__ == "__main__"
# import sys
# sys.path.append(os.getcwd())

from quick_pp.machine_learning.config import MODELLING_CONFIG, RAW_FEATURES
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import get_latest_registered_models, unique_id, run_mlflow_server
from quick_pp.plotter.well_log import plotly_log
from quick_pp import logger


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


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by ensuring required columns are present.

    Args:
        df (pd.DataFrame): DataFrame to preprocess.

    Raises:
        ValueError: If required columns are missing.

    Returns:
        pd.DataFrame: Preprocessed DataFrame.
    """
    df = generate_fe_features(df)
    return df


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

    # Generate fluid volume fractions if OIL_FLAG and GAS_FLAG exist
    df['VHC'] = (1 - df.get('SWT', 1)) * df.get('PHIT', 0)
    if 'OIL_FLAG' in df.columns and 'GAS_FLAG' in df.columns:
        df = fix_fluid_segregation(df)

    return df


def fix_fluid_segregation(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Generating fluid volume fractions from OIL_FLAG and GAS_FLAG")
    df['VOIL'] = df['OIL_FLAG'] * df['VHC']
    df['VGAS'] = df['GAS_FLAG'] * df['VHC']

    for well_name, well_df in tqdm(df.groupby('WELL_NAME'), desc="Fixing fluid segregation"):
        tqdm.write(f"Processing well {well_name}")
        # Fix fluid segregation issues bounded by continuous hydrocarbon intervals
        hc_mask = ((well_df['VHC'] >= 1e-2)).astype(int)
        # Identify continuous hydrocarbon intervals
        hc_groups = (hc_mask.diff() != 0).cumsum()

        for _, group_df in well_df.groupby(hc_groups):
            # Process only hydrocarbon-bearing intervals
            if hc_mask.loc[group_df.index].sum() > 0:
                # If both gas and oil are predicted in the same interval
                if (group_df['GAS_FLAG'] == 1).any() and (group_df['OIL_FLAG'] == 1).any():
                    # Find the deepest depth where gas is predicted
                    last_gas_depth = group_df[group_df['GAS_FLAG'] == 1]['DEPTH'].max()
                    # Identify indices of oil intervals above this deepest gas
                    oil_above_gas_indices = group_df[(group_df['DEPTH'] <= last_gas_depth) & (
                        group_df['OIL_FLAG'] == 1)].index
                    # Re-assign oil volumes to gas for these intervals in the main dataframe
                    df.loc[oil_above_gas_indices, 'VGAS'] = df.loc[oil_above_gas_indices, 'VHC']
                    df.loc[oil_above_gas_indices, 'VOIL'] = 0
                if (group_df['GAS_FLAG'] == 1).any() and (group_df['OIL_FLAG'] == 0).all():
                    df.loc[group_df.index, 'VGAS'] = df.loc[group_df.index, 'VHC']
                    df.loc[group_df.index, 'VHC'] = 0
                if (group_df['OIL_FLAG'] == 1).any() and (group_df['GAS_FLAG'] == 0).all():
                    df.loc[group_df.index, 'VOIL'] = df.loc[group_df.index, 'VHC']
                    df.loc[group_df.index, 'VHC'] = 0

    df['VHC'] = np.where((df['VOIL'] > 0) | (df['VGAS'] > 0), 0, df['VHC'])

    return df


def save_predictions(pred_df: pd.DataFrame, output_file_name: str, plot: bool = False):
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

    if plot:
        output_dir = Path("data/output/plots")
        os.makedirs(output_dir, exist_ok=True)
        for well_name, well_df in tqdm(pred_df.groupby('WELL_NAME'), desc="Generating plots", ):
            fig = plotly_log(well_df, well_name=well_name, column_widths=[1, 1, 1, 1, 1, 1, .5, 1, 1],)
            plot_path = Path(f"{output_dir}/{well_name}.html")
            fig.write_html(plot_path, config=dict(scrollZoom=True))
            tqdm.write(f"Plot for well {well_name} saved to {plot_path}")
    logger.info(f"Predictions saved to {output_path}")


def predict_pipeline(
        model_config: str, data_hash: str, output_file_name: str, env: str = 'local', plot_predictions: bool = False
) -> None:
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
    data = preprocess_data(data)

    pred_df = data[['DEPTH', 'WELL_NAME'] + RAW_FEATURES].copy()
    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        targets = model_values['targets']
        features = model_values['features']
        reg_model_name = f'{model_config}_{model_key}_{data_hash}'
        logger.info(f"Predicting with model: {model_key} | {reg_model_name}")

        # Load the model
        model = load_model(latest_rms[reg_model_name]['model_uri'])

        # Run predictions and concat to pred_df
        preds = model.predict(data[features].astype('float'))
        temp_df = pd.DataFrame(preds, columns=targets)
        pred_df = pd.concat([pred_df, temp_df], axis=1)

    # Postprocess the predictions and save
    pred_df = postprocess_data(pred_df)
    save_predictions(pred_df, output_file_name, plot=plot_predictions)
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
