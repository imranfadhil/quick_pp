"""
Command-Line Interface for the quick_pp library.

This script provides a set of commands to interact with the various components
of the quick_pp application, including running the web app, managing MLflow
servers, and executing machine learning training and prediction pipelines.
"""

import click
import os
import shutil
import socket
import sys
from importlib import metadata, resources
from pathlib import Path
from subprocess import Popen


try:
    quick_ppVersion = metadata.version("quick_pp")
except metadata.PackageNotFoundError:
    quick_ppVersion = "0.0.0"


def is_server_running(host, port):
    """Check if a server is running on the specified host and port.

    Args:
        host (str): The hostname or IP address of the server.
        port (int): The port number of the server.

    Returns:
        bool: True if the server is accessible, False otherwise.
    """
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


@click.group()
@click.version_option(
    prog_name="quick_pp",
    version=quick_ppVersion,
    message="%(prog)s version: %(version)s",
)
def cli():
    """
    Command-line interface for the quick_pp library.

    Provides commands to run the web application, manage MLflow servers,
    and execute machine learning pipelines for training and prediction.
    """
    pass


@click.command()
@click.option("--debug", is_flag=True)
def app(debug):
    """Start the quick_pp web application.

    This launches a Uvicorn server to run the FastAPI backend and the associated qpp assistant module.
    The --debug flag enables auto-reload for development."""
    if not is_server_running("localhost", 6312):
        reload_ = "--reload" if debug else ""
        cmd = f"uvicorn quick_pp.app.backend.main:app --host 0.0.0.0 --port 6312 {reload_}"
        click.echo(f"App is not running. Starting it now... | {cmd}")
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        process.wait()


@click.command()
@click.argument("env", default="local", type=click.Choice(["local", "remote"]))
def mlflow_server(env):
    """Start the MLflow tracking server.

    This command can launch a local server or is intended to be used with a
    pre-configured remote server based on the 'env' argument.
    """
    from quick_pp.machine_learning.utils import run_mlflow_server

    run_mlflow_server(env)


@click.command()
@click.option("--debug", is_flag=True)
def model_deployment(debug):
    """Run the MLflow model deployment."""
    """Serve the latest registered MLflow models via a FastAPI endpoint.

    This makes the trained models available for prediction over a REST API.
    The --debug flag enables auto-reload for development."""
    if not is_server_running("localhost", 5555):
        reload_ = "--reload" if debug else ""
        cmd = f"uvicorn quick_pp.app.backend.mlflow_model_deployment:app --host 0.0.0.0 --port 5555 {reload_}"
        click.echo(f"Model server is not running. Starting it now... | {cmd}")
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        process.wait()


@click.command()
@click.argument("model_config", required=True, type=click.STRING)
@click.argument("data_hash", required=True, type=click.STRING)
@click.argument("env", default="local", type=click.Choice(["local", "remote"]))
def train(model_config, data_hash, env):
    """Execute the model training pipeline.

    This command automates the process of training, evaluating, and logging models
    as defined in the configuration. It sets up the necessary environment by copying
    default configuration and data files if they don't exist, then starts the training
    process using MLflow for experiment tracking.

    MODEL_CONFIG: The key for the modelling suite (e.g., 'clastic', 'carbonate').
    DATA_HASH: The unique hash identifying the input data file for training.
    ENV: The MLflow environment ('local' or 'remote').
    """
    from quick_pp.machine_learning.train_pipeline import train_pipeline

    # Set up the environment by copying default config and data if they don't exist.
    # This allows users to customize them easily.
    config_file = resources.files("quick_pp.machine_learning").joinpath("config.py")
    root_config_file = Path(os.getcwd(), "config.py")
    if os.path.exists(config_file) and not os.path.exists(root_config_file):
        shutil.copyfile(config_file, root_config_file)
        click.echo(f"Copied {config_file} to {root_config_file}")

    # Copy mock_data.parquet into data/input if it doesn't exist
    mock_file = resources.files("quick_pp.machine_learning.mock_data").joinpath(
        "mock_data.parquet"
    )
    data_dir = Path(os.getcwd(), "data", "input")
    os.makedirs(data_dir, exist_ok=True)
    root_mock_file = Path(data_dir, "mock_data.parquet")
    if os.path.exists(mock_file) and not os.path.exists(root_mock_file):
        shutil.copyfile(mock_file, root_mock_file)
        click.echo(f"Copied {mock_file} to {root_mock_file}")

    click.echo(f"Training {model_config} model with data hash {data_hash}")
    train_pipeline(model_config, data_hash, env)


@click.command()
@click.argument("model_config", required=True, type=click.STRING)
@click.argument("data_hash", required=True, type=click.STRING)
@click.argument("output_file_name", required=False, default="test", type=click.STRING)
@click.argument("env", default="local", type=click.Choice(["local", "remote"]))
@click.option("--plot", is_flag=True, help="Generate and save plots for predictions.")
def predict(model_config, data_hash, output_file_name, env, plot):
    """Run the prediction pipeline using the latest registered models.

    This command loads data, fetches the appropriate trained models from the MLflow
    registry, runs predictions, post-processes the results, and saves the output.

    MODEL_CONFIG: The key for the modelling suite whose models will be used.
    DATA_HASH: The unique hash identifying the input data file for prediction.
    OUTPUT_FILE_NAME: The base name for the output predictions file.
    ENV: The MLflow environment ('local' or 'remote').
    """
    from quick_pp.machine_learning.predict_pipeline import predict_pipeline

    click.echo("Running prediction...")
    predict_pipeline(model_config, data_hash, output_file_name, env, plot)


# Add commands to the CLI group
cli.add_command(app)
cli.add_command(mlflow_server)
cli.add_command(model_deployment)
cli.add_command(train)
cli.add_command(predict)


if __name__ == "__main__":
    cli()
