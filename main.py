import click
from subprocess import Popen
import sys
import socket
import mlflow

from quick_pp.modelling.train_pipeline import train_pipeline
from quick_pp.modelling.config import MLFLOW_CONFIG


def is_server_running(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


@click.group()
def cli():
    """A handy CLI tool."""
    pass


@click.command()
def api_server():
    """Run the API server."""
    cmd = "uvicorn api.main:app --reload"
    print("Starting API server with command: ", cmd)

    if not is_server_running('localhost', 8000):
        print("API server is not running. Starting it now...")
        Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=False)


@click.command()
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def mlflow_server(env):
    cmd_mlflow_server = (f"mlflow server --backend-store-uri {MLFLOW_CONFIG[env]['backend_store_uri']} "
                         f"--default-artifact-root {MLFLOW_CONFIG[env]['artifact_location']} "
                         f"--host {MLFLOW_CONFIG[env]['tracking_host']} "
                         f"--port {MLFLOW_CONFIG[env]['tracking_port']}")
    print(f"Start MLflow server with command: {cmd_mlflow_server}")

    if not is_server_running(MLFLOW_CONFIG[env]['tracking_host'], MLFLOW_CONFIG[env]['tracking_port']):
        print("MLflow server is not running. Starting it now...")
        Popen(cmd_mlflow_server, stdout=sys.stdout, stderr=sys.stderr, shell=False)

    mlflow.set_tracking_uri(
        f"http://{MLFLOW_CONFIG[env]['tracking_host']}:{MLFLOW_CONFIG[env]['tracking_port']}")
    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")


@click.command()
@click.argument('--model_config', required=True)
@click.argument('--data_hash', default=10)
def train(model_config, data_hash):
    """Train the model with the specified parameters."""
    print(f"Training {model_config} model with data hash {data_hash}")
    train_pipeline(model_config, data_hash)


# Add commands to the CLI group
cli.add_command(api_server)
cli.add_command(mlflow_server)
cli.add_command(train)

if __name__ == '__main__':
    cli()
