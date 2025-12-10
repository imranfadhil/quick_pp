import os
import socket
from hashlib import sha256
from pathlib import Path
from subprocess import Popen

import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pandas.util import hash_pandas_object

from quick_pp import logger
from quick_pp.machine_learning.config import MLFLOW_CONFIG


def unique_id(df: pd.DataFrame) -> str:
    """Generate a unique ID for the DataFrame based on its content.

    Args:
        df (pd.DataFrame): DataFrame to hash.
    Returns:
        str: An 8-character unique hexadecimal ID for the DataFrame.
    """
    # Hash the DataFrame content and convert to hex string
    uid = sha256(hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[
        :8
    ]
    logger.debug(f"Generated unique_id: {uid}")
    return uid


def is_mlflow_server_running(host, port):
    """Check if the MLflow server is running on the specified host and port.

    Args:
        host (str): Hostname or IP address of the MLflow server.
        port (int): Port number of the MLflow server.
    Returns:
        bool: True if the server is running, False otherwise.
    """
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            logger.debug(f"MLflow server is running at {host}:{port}")
            return True
    except Exception as e:
        logger.debug(f"MLflow server not running at {host}:{port}: {e}")
        return False


def run_mlflow_server(env):
    """Start an MLflow tracking server if not already running.

    This function checks for a running MLflow server based on the environment configuration
    and sets the MLflow tracking URI accordingly.

    Args:
        env (str): The environment key to select the MLflow server configuration from MLFLOW_CONFIG.
    Raises:
        KeyError: If the specified environment is not found in MLFLOW_CONFIG.
    """
    mlruns_dir = Path(str(MLFLOW_CONFIG[env]["artifact_location"]))
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflog_config = MLFLOW_CONFIG[env]
    if not is_mlflow_server_running(
        mlflog_config["tracking_host"], mlflog_config["tracking_port"]
    ):
        cmd_mlflow_server = (
            f"mlflow server --backend-store-uri {mlflog_config['backend_store_uri']} "
            f"--default-artifact-root {mlflog_config['artifact_location']} "
            f"--host {mlflog_config['tracking_host']} "
            f"--port {mlflog_config['tracking_port']}"
        )
        logger.warning(
            f"MLflow server is not running. Starting it now... | {cmd_mlflow_server}"
        )
        Popen(cmd_mlflow_server, shell=True)
        logger.info("MLflow server started successfully.")

    mlflow.set_tracking_uri(
        f"http://{MLFLOW_CONFIG[env]['tracking_host']}:{MLFLOW_CONFIG[env]['tracking_port']}"
    )
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")


def get_model_info(registered_model):
    """Extract key information from a registered MLflow model version object.

    Args:
        registered_model (list[mlflow.entities.model_registry.ModelVersion]): A list containing
            one or more registered model version objects. This function processes the first one.

    Returns:
        dict: A dictionary containing the model's name, run ID, version, URI, and stage.
    """
    model_info = {}
    for model in registered_model:
        model_info["reg_model_name"] = model.name
        model_info["run_id"] = model.run_id
        model_info["version"] = model.version
        model_info["model_uri"] = model.source
        model_info["stage"] = model.current_stage
    logger.debug(f"Extracted model info: {model_info}")
    return model_info


def get_latest_registered_models(
    client: MlflowClient, experiment_name: str, data_hash: str
) -> dict:
    """Get the latest versions of registered models from MLflow for a given experiment and data hash.

    Args:
        client (MlflowClient): MLflow client to interact with the tracking server.
        experiment_name (str): Name of the experiment to filter registered models.
        data_hash (str): The unique hash of the data used to train the models.

    Returns:
        dict: A dictionary where keys are registered model names and values are dicts of their details.
    """
    latest_rm_models = {}
    filter_str = f"name ILIKE '{experiment_name}%' AND name like '%{data_hash}'"
    logger.info(f"Searching for registered models with filter: {filter_str}")
    for rm in client.search_registered_models(filter_string=filter_str):
        latest_rm_info = get_model_info(rm.latest_versions)
        latest_rm_models[latest_rm_info["reg_model_name"]] = latest_rm_info
        logger.info(
            f"Adding {latest_rm_info['reg_model_name']} to latest_rm_models dictionary"
        )
    return latest_rm_models
