import pandas as pd
from pandas.util import hash_pandas_object
import os
import mlflow
from mlflow.tracking import MlflowClient
import socket
from subprocess import Popen
from hashlib import sha256
from pathlib import Path

from quick_pp.modelling.config import MLFLOW_CONFIG
from quick_pp.logger import logger


def unique_id(df: pd.DataFrame) -> str:
    """Generate a unique ID for the DataFrame based on its content.
    Args:
        df (pd.DataFrame): DataFrame to hash.
    Returns:
        str: Unique ID for the DataFrame.
    """
    # Hash the DataFrame content and convert to hex string
    uid = sha256(hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[:8]
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
    """
    Starts an MLflow tracking server if it is not already running for the specified environment,
    and sets the MLflow tracking URI accordingly.
    Args:
        env (str): The environment key to select the MLflow server configuration from MLFLOW_CONFIG.
    Raises:
        KeyError: If the specified environment is not found in MLFLOW_CONFIG.
    Side Effects:
        - Starts an MLflow server process if one is not already running for the given environment.
        - Sets the MLflow tracking URI for the current process.
        - Prints status messages to the console.
    """
    mlruns_dir = Path(str(MLFLOW_CONFIG[env]['artifact_location']))
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflog_config = MLFLOW_CONFIG[env]
    if not is_mlflow_server_running(mlflog_config['tracking_host'], mlflog_config['tracking_port']):
        cmd_mlflow_server = (
            f"mlflow server --backend-store-uri {mlflog_config['backend_store_uri']} "
            f"--default-artifact-root {mlflog_config['artifact_location']} "
            f"--host {mlflog_config['tracking_host']} "
            f"--port {mlflog_config['tracking_port']}"
        )
        logger.warning(f"MLflow server is not running. Starting it now... | {cmd_mlflow_server}")
        Popen(cmd_mlflow_server, shell=True)
        logger.info("MLflow server started successfully.")

    mlflow.set_tracking_uri(
        f"http://{MLFLOW_CONFIG[env]['tracking_host']}:{MLFLOW_CONFIG[env]['tracking_port']}")
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")


def get_model_info(registered_model):
    """Extract information of the registered models to load models automatically for predictions.

    Args:
        registered_model (class): Model that has been loaded and registered
    """
    model_info = {}
    for model in registered_model:
        model_info['reg_model_name'] = model.name
        model_info['run_id'] = model.run_id
        model_info['version'] = model.version
        model_info['model_uri'] = model.source
        model_info['stage'] = model.current_stage
    logger.debug(f"Extracted model info: {model_info}")
    return model_info


def get_latest_registered_models(client: MlflowClient, experiment_name: str, data_hash: str) -> dict:
    """Get the latest registered models from MLflow.

    Args:
        client (MlflowClient): MLflow client to interact with the tracking server.
        experiment_name (str): Name of the experiment to filter registered models.

    Returns:
        dict: Dictionary containing the latest registered models with their details.
    """
    latest_rm_models = {}
    filter_str = f"name ILIKE '{experiment_name}%' AND name like '%{data_hash}'"
    logger.info(f"Searching for registered models with filter: {filter_str}")
    for rm in client.search_registered_models(filter_string=filter_str):
        latest_rm_info = get_model_info(rm.latest_versions)
        latest_rm_models[latest_rm_info['reg_model_name']] = latest_rm_info
        logger.info(f"Adding {latest_rm_info['reg_model_name']} to latest_rm_models dictionary")
    return latest_rm_models
