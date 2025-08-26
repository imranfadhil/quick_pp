import click
import os
import shutil
import socket
import sys
from importlib import metadata, resources
from pathlib import Path
from subprocess import Popen


try:
    quick_ppVersion = metadata.version('quick_pp')
except metadata.PackageNotFoundError:
    quick_ppVersion = "0.0.0"


def is_server_running(host, port):
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
    """A handy CLI tool for quick_pp."""
    pass


@click.command()
@click.option('--debug', is_flag=True)
def app(debug):
    """Run the App consisting of API server and qpp assistant module."""
    if not is_server_running('localhost', 8888):
        reload_ = "--reload" if debug else ""
        cmd = f"uvicorn quick_pp.api.main:app --host 0.0.0.0 --port 8888 {reload_}"
        click.echo(f"App is not running. Starting it now... | {cmd}")
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        process.wait()


@click.command()
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def mlflow_server(env):
    """Run the MLflow server."""
    from quick_pp.machine_learning.utils import run_mlflow_server

    run_mlflow_server(env)


@click.command()
@click.option('--debug', is_flag=True)
def model_deployment(debug):
    """Run the MLflow model deployment."""
    if not is_server_running('localhost', 5555):
        reload_ = "--reload" if debug else ""
        cmd = f"uvicorn quick_pp.api.mlflow_model_deployment:app --host 0.0.0.0 --port 5555 {reload_}"
        click.echo(f"Model server is not running. Starting it now... | {cmd}")
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        process.wait()


@click.command()
@click.argument('model_config', required=True, type=click.STRING)
@click.argument('data_hash', required=True, type=click.STRING)
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def train(model_config, data_hash, env):
    """Train the model with the specified parameters."""
    from quick_pp.machine_learning.train_pipeline import train_pipeline

    # Copy config.py into the root directory if it doesn't exist
    config_file = resources.files('quick_pp.modelling').joinpath('config.py')
    root_config_file = Path(os.getcwd(), 'config.py')
    if os.path.exists(config_file) and not os.path.exists(root_config_file):
        shutil.copyfile(config_file, root_config_file)
        click.echo(f"Copied {config_file} to {root_config_file}")

    # Copy mock_data.parquet into data/input if it doesn't exist
    mock_file = resources.files('quick_pp.modelling.mock_data').joinpath('mock_data.parquet')
    data_dir = Path(os.getcwd(), 'data', 'input')
    os.makedirs(data_dir, exist_ok=True)
    root_mock_file = Path(data_dir, 'mock_data.parquet')
    if os.path.exists(mock_file) and not os.path.exists(root_mock_file):
        shutil.copyfile(mock_file, root_mock_file)
        click.echo(f"Copied {mock_file} to {root_mock_file}")

    click.echo(f"Training {model_config} model with data hash {data_hash}")
    train_pipeline(model_config, data_hash, env)


@click.command()
@click.argument('model_config', required=True, type=click.STRING)
@click.argument('data_hash', required=True, type=click.STRING)
@click.argument('output_file_name', required=False, default='test', type=click.STRING)
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def predict(model_config, data_hash, output_file_name, env):
    """Run the prediction."""
    from quick_pp.machine_learning.predict_pipeline import predict_pipeline

    click.echo("Running prediction...")
    predict_pipeline(model_config, data_hash, output_file_name, env)


# Add commands to the CLI group
cli.add_command(app)
cli.add_command(mlflow_server)
cli.add_command(model_deployment)
cli.add_command(train)
cli.add_command(predict)


if __name__ == "__main__":
    cli()
