import os
from pathlib import Path

import mlflow.tracking as mlflow_tracking
import pandas as pd
from mlflow.pyfunc import load_model
from tqdm import tqdm

from quick_pp import logger
from quick_pp.fluid_type import fix_fluid_segregation
from quick_pp.machine_learning.config import (
    CONFORMAL_ALPHA,
    FE_FEATURES,
    MODELLING_CONFIG,
    RAW_FEATURES,
)
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import (
    POST_PROCESSING,
    EnforceMineralSum,
    enforce_physical_constraints,
    get_latest_registered_models,
    predict_with_uncertainty,
    run_mlflow_server,
    unique_id,
)
from quick_pp.plotter.well_log import plotly_log


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
    """Postprocess the DataFrame by enforcing single-column physical bounds, jointly normalizing
    ``['VCLAY', 'VSAND', 'VSILT', 'PHIT']`` via ``EnforceMineralSum``, inverting ``LOG_PERM`` to
    ``PERM`` (including P10–P90 bands), and computing fluid volumes.

    Args:
        df (pd.DataFrame): The DataFrame containing model predictions.

    Returns:
        pd.DataFrame: The postprocessed DataFrame with physical bounds enforced and added
                      'PERM', 'PERM_P10', 'PERM_P90', 'VHC', 'VWATER', 'VOIL', and 'VGAS' columns.
    """
    logger.info("Enforcing petrophysical bounds and volumetric closure constraints")
    df = df.copy()

    # 1. Single-column bound post-processing
    for k, fn in POST_PROCESSING.items():
        if k in df.columns:
            df[k] = fn(df[k])

    # 2. Multi-column joint post-processing for mineral volumes + porosity closure
    joint_mineral_poro_cols = [
        c for c in ["VCLAY", "VSAND", "VSILT", "PHIT"] if c in df.columns
    ]
    if len(joint_mineral_poro_cols) >= 2:
        df[joint_mineral_poro_cols] = EnforceMineralSum(
            mineral_cols=["VCLAY", "VSAND", "VSILT"], porosity_col="PHIT"
        )(df[joint_mineral_poro_cols])

    # 3. Full petrophysical constraints (tight-rock Sw, coal intervals, PERM inversion, P10-P90 ordering)
    df = enforce_physical_constraints(df)

    # Generate fluid volume fractions if porosity and water saturation exist
    poro_col = "PHIT" if "PHIT" in df.columns else ("PHIE" if "PHIE" in df.columns else None)
    sw_col = "SWT" if "SWT" in df.columns else ("SW" if "SW" in df.columns else None)

    if poro_col is not None and sw_col is not None:
        df["VHC"] = ((1.0 - df[sw_col]) * df[poro_col]).clip(lower=0.0, upper=1.0)
        df["VWATER"] = (df[sw_col] * df[poro_col]).clip(lower=0.0, upper=1.0)
    else:
        df["VHC"] = (1.0 - df.get("SWT", 1.0)) * df.get("PHIT", 0.0)

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
            fig.write_html(plot_path, config={"scrollZoom": True})
            tqdm.write(f"Plot for well {well_name} saved to {plot_path}")
    logger.info(f"Predictions saved to {output_path}")


def predict_pipeline(
    model_config: str,
    data_hash: str,
    output_file_name: str,
    env: str = "local",
    plot_predictions: bool = False,
    alpha: float = CONFORMAL_ALPHA,
) -> None:
    """Execute the end-to-end prediction pipeline with P10–P90 conformal intervals and physics constraints.

    This function orchestrates loading data, preprocessing, loading the latest registered
    MLflow models, making point and P10–P90 interval predictions, postprocessing the results,
    and saving the output.

    Args:
        model_config (str): The key for the model configuration (e.g., 'clastic', 'carbonate').
        data_hash (str): The unique hash identifying the input data file.
        output_file_name (str): The base name for the output predictions file.
        env (str, optional): Environment for MLflow server. Defaults to 'local'.
        plot_predictions (bool, optional): If True, generate plots for each well. Defaults to False.
        alpha (float, optional): Significance level for conformal intervals (0.20 -> P10–P90).
                                 Defaults to CONFORMAL_ALPHA.
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

    base_cols = ["DEPTH", "WELL_NAME"] + [
        c for c in RAW_FEATURES + FE_FEATURES if c in data.columns
    ]
    pred_df = data[base_cols].copy()
    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]

    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        targets = model_values["targets"]
        features = model_values["features"]
        reg_model_name = f"{model_config}_{model_key}_{data_hash}"
        logger.info(f"Predicting with model: {model_key} | {reg_model_name}")

        # Load the model
        model = load_model(latest_rms[reg_model_name]["model_uri"])

        # Run predictions + P10/P90 uncertainty bands and concat to pred_df
        X_feat = data[features].astype("float")
        preds, preds_p10, preds_p90 = predict_with_uncertainty(
            model, X_feat, targets=targets, alpha=alpha
        )
        temp_df = pd.DataFrame(preds, columns=targets, index=pred_df.index)
        if preds_p10 is not None and preds_p90 is not None:
            for idx, target_name in enumerate(targets):
                temp_df[f"{target_name}_P10"] = preds_p10[:, idx]
                temp_df[f"{target_name}_P90"] = preds_p90[:, idx]

        pred_df = pd.concat([pred_df, temp_df], axis=1)
        # Keep data updated with intermediate predictions (e.g. OIL_FLAG used by GAS model)
        for col in targets:
            data[col] = temp_df[col]

    # Merge specific flag columns from original data missing from pred_df
    merge_cols = ["WELL_NAME", "DEPTH"]
    missing_cols = [
        c
        for c in ["ROCK_FLAG", "COAL_FLAG", "TIGHT_FLAG"]
        if c in data.columns and c not in pred_df.columns
    ]
    if missing_cols:
        pred_df = pd.merge(
            pred_df, data[merge_cols + missing_cols], on=merge_cols, how="left"
        )

    # Postprocess the predictions and save
    pred_df = postprocess_data(pred_df)
    save_predictions(pred_df, output_file_name, plot=plot_predictions)
    logger.info("Prediction pipeline completed successfully")

