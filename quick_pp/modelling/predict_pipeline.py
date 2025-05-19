import pandas as pd
from pathlib import Path
from mlflow.pyfunc import load_model
import mlflow.tracking as mlflow_tracking

# # Uncomment below 3 lines to run >> if __name__ == "__main__"
# import os
# import sys
# sys.path.append(os.getcwd())

from quick_pp.modelling.config import MODELLING_CONFIG
from quick_pp.modelling.utils import get_latest_registered_models, unique_id


def load_data(hash: str) -> pd.DataFrame:
    data_dir = Path("data/input/")
    matching_files = list(data_dir.glob(f"*{hash}*.parquet"))
    if not matching_files:
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    return pd.read_parquet(path)


def postprocess_data(df):
    """Postprocess data if needed."""
    # Invert LOG_PERM to PERM if exists
    if 'LOG_PERM' in df.columns and 'PERM' not in df.columns:
        df['PERM'] = 10 ** df['LOG_PERM']
    return df


def save_predictions(pred_df: pd.DataFrame, output_file_name: str):
    """Save predictions to a parquet file."""
    hash = unique_id(pred_df)
    output_path = Path(f"data/output/{output_file_name}_{hash}.parquet")
    pred_df.to_parquet(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def predict_pipeline(model_config: str, data_hash: str, output_file_name: str):
    """Run the prediction pipeline."""
    client = mlflow_tracking.MlflowClient()
    latest_rms = get_latest_registered_models(client, model_config)

    pred_df = pd.DataFrame()
    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        targets = model_values['targets']
        features = model_values['features']
        reg_model_name = f'{model_config}_{model_key}'
        print(f"Predicting with model: {model_key} | {reg_model_name}")

        # Load the model
        model = load_model(latest_rms[reg_model_name]['model_uri'])

        # Load the data
        data = load_data(data_hash)

        # Run predictions and concat to pred_df
        preds = model.predict(data[features])
        pred_df[targets] = preds

    # Postprocess the predictions and save
    pred_df = postprocess_data(pred_df)
    save_predictions(pred_df, output_file_name)


if __name__ == "__main__":
    import argparse
    import os

    from quick_pp.modelling.utils import run_mlflow_server

    parser = argparse.ArgumentParser(description="Prediction pipeline")
    parser.add_argument("--model-config", type=str, required=True, help="Path to model config")
    parser.add_argument("--data", type=str, required=True, help="Path to input data parquet file")
    parser.add_argument("--output", type=str, default='test', help="Path to save predictions parquet file")
    args = parser.parse_args()

    # Set up MLflow
    os.makedirs('data/output', exist_ok=True)

    run_mlflow_server('local')

    print(args.model_config)
    print(args.data)

    predict_pipeline(args.model_config, args.data, args.output)
