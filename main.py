import click
from subprocess import Popen
import sys
import socket

import quick_pp
from quick_pp.modelling.train_pipeline import train_pipeline
from quick_pp.modelling.predict_pipeline import predict_pipeline
from quick_pp.modelling.utils import run_mlflow_server


def is_server_running(host, port):
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except Exception:
        return False


@click.group()
@click.version_option(
    prog_name="quick_pp",
    version=quick_pp.__version__,
    message="%(prog)s version: %(version)s",
)
def cli():
    """A handy CLI tool for quick_pp."""
    pass


@click.command()
@click.option('--debug', is_flag=True)
def api_server(debug):
    """Run the API server."""
    if not is_server_running('localhost', 8888):
        reload_ = "--reload" if debug else ""
        cmd = f"uvicorn quick_pp.api.main:app --host 0.0.0.0 --port 8888 {reload_}"
        click.echo(f"API server is not running. Starting it now... | {cmd}")
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        process.wait()
        click.echo("API server started successfully.")


@click.command()
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def mlflow_server(env):
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
        click.echo("Model server started successfully.")


@click.command()
@click.argument('model_config', required=True, type=click.STRING)
@click.argument('data_hash', required=True, type=click.STRING)
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def train(model_config, data_hash, env):
    """Train the model with the specified parameters."""
    click.echo(f"Training {model_config} model with data hash {data_hash}")
    train_pipeline(model_config, data_hash, env)


@click.command()
@click.argument('model_config', required=True, type=click.STRING)
@click.argument('data_hash', required=True, type=click.STRING)
@click.argument('output_file_name', required=False, default='test', type=click.STRING)
@click.argument('env', default='local', type=click.Choice(['local', 'remote']))
def predict(model_config, data_hash, output_file_name, env):
    """Run the prediction."""
    click.echo("Running prediction...")
    predict_pipeline(model_config, data_hash, output_file_name, env)


# Add commands to the CLI group
cli.add_command(api_server)
cli.add_command(mlflow_server)
cli.add_command(model_deployment)
cli.add_command(train)
cli.add_command(predict)

if __name__ == '__main__':
    cli()
