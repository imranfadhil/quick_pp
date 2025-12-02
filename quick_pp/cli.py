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
import webbrowser
from importlib import metadata, resources
from pathlib import Path
from subprocess import Popen


try:
    quick_ppVersion = metadata.version("quick_pp")
except metadata.PackageNotFoundError:
    quick_ppVersion = "0.0.0"

# Global defaults for backend and frontend locations used across commands
BACKEND_HOST = "localhost"
BACKEND_PORT = 6312
FRONTEND_DIR = Path(__file__).parent / "app" / "frontend"


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
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open browser after starting backend",
)
def backend(debug, no_open):
    """Start the quick_pp web application.

    This launches a Uvicorn server to run the FastAPI backend and the associated qpp assistant module.
    The --debug flag enables auto-reload for development."""
    if not is_server_running(BACKEND_HOST, BACKEND_PORT):
        reload_ = "--reload" if debug else ""
        cmd = f"uvicorn quick_pp.app.backend.main:app --host 0.0.0.0 --port {BACKEND_PORT} {reload_}"
        click.echo(f"App is not running. Starting it now... | {cmd}")
        process = Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True)
        # Open browser to backend URL after starting (default)
        try:
            if not no_open:
                webbrowser.open(f"http://{BACKEND_HOST}:{BACKEND_PORT}")
        except Exception:
            pass
        process.wait()


@click.command()
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open browser after starting frontend",
)
def frontend(no_open):
    """Start the quick_pp frontend development server.

    This launches the SvelteKit development server for the frontend application."""
    frontend_dir = FRONTEND_DIR

    if not frontend_dir.exists():
        click.echo(f"Error: Frontend directory not found at {frontend_dir}")
        return

    # Check if node_modules exists, if not suggest installing dependencies
    if not (frontend_dir / "node_modules").exists():
        click.echo(
            "Warning: node_modules not found. Attempting to run 'npm install' now..."
        )
        try:
            install_cmd = "npm install"
            click.echo(f"Running: (in {frontend_dir}) {install_cmd}")
            p_install = Popen(
                install_cmd,
                stdout=sys.stdout,
                stderr=sys.stderr,
                shell=True,
                cwd=str(frontend_dir),
            )
            p_install.wait()
            if p_install.returncode != 0:
                click.echo(
                    "npm install failed. Please run it manually and re-run this command."
                )
                return
            click.echo("npm install completed successfully.")
        except Exception as e:
            click.echo(f"Failed to run npm install: {e}")
            click.echo(f"Run manually: cd {frontend_dir} && npm install")
            return

    click.echo(f"Starting frontend development server from {frontend_dir}...")

    # Change to frontend directory and run npm run dev
    cmd = "npm run dev"
    process = Popen(
        cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True, cwd=str(frontend_dir)
    )
    # Open browser to frontend dev URL (SvelteKit default 5173) unless disabled
    try:
        if not no_open:
            webbrowser.open("http://localhost:5173")
    except Exception:
        pass
    process.wait()


@click.command()
@click.option("--debug", is_flag=True)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open browser after starting services",
)
def app(debug, no_open):
    """Start both backend and frontend development servers.

    This command will start the backend (uvicorn) on port 6312 and the
    frontend (`npm run dev`) from `quick_pp/app/frontend` if they are not
    already running. Use `--debug` to enable uvicorn's reload mode.
    """
    processes = []
    try:
        # Start backend if not running
        if not is_server_running(BACKEND_HOST, BACKEND_PORT):
            reload_ = "--reload" if debug else ""
            backend_cmd = f"uvicorn quick_pp.app.backend.main:app --host 0.0.0.0 --port {BACKEND_PORT} {reload_}"
            click.echo(f"Starting backend... | {backend_cmd}")
            p_backend = Popen(
                backend_cmd, stdout=sys.stdout, stderr=sys.stderr, shell=True
            )
            processes.append(p_backend)
        else:
            click.echo("Backend already running on localhost:6312")

        # Start frontend if available and node modules installed
        frontend_dir = FRONTEND_DIR
        if not frontend_dir.exists():
            click.echo(f"Frontend directory not found at {frontend_dir}")
        else:
            if not (frontend_dir / "node_modules").exists():
                click.echo(
                    "Warning: node_modules not found. Attempting to run 'npm install' now..."
                )
                try:
                    install_cmd = "npm install"
                    click.echo(f"Running: (in {frontend_dir}) {install_cmd}")
                    p_install = Popen(
                        install_cmd,
                        stdout=sys.stdout,
                        stderr=sys.stderr,
                        shell=True,
                        cwd=str(frontend_dir),
                    )
                    p_install.wait()
                    if p_install.returncode != 0:
                        click.echo(
                            "npm install failed. Please run it manually and re-run this command."
                        )
                    else:
                        click.echo("npm install completed successfully.")
                        click.echo(
                            f"Starting frontend development server from {frontend_dir}..."
                        )
                        cmd = "npm run dev"
                        p_front = Popen(
                            cmd,
                            stdout=sys.stdout,
                            stderr=sys.stderr,
                            shell=True,
                            cwd=str(frontend_dir),
                        )
                        processes.append(p_front)
                except Exception as e:
                    click.echo(f"Failed to run npm install: {e}")
                    click.echo(f"Run manually: cd {frontend_dir} && npm install")
            else:
                click.echo(
                    f"Starting frontend development server from {frontend_dir}..."
                )
                cmd = "npm run dev"
                p_front = Popen(
                    cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    shell=True,
                    cwd=str(frontend_dir),
                )
                processes.append(p_front)

        # Open browser(s) after launching processes (default behavior)
        try:
            if not no_open:
                # backend
                if any(p == p_backend for p in processes) or is_server_running(
                    BACKEND_HOST, BACKEND_PORT
                ):
                    webbrowser.open(f"http://{BACKEND_HOST}:{BACKEND_PORT}")
                # frontend (SvelteKit default dev port)
                if any(p == p_front for p in processes) or frontend_dir.exists():
                    webbrowser.open("http://localhost:5173")
        except Exception:
            pass

        if not processes:
            click.echo(
                "Nothing to start: backend may be running and frontend missing or uninstalled."
            )
            return

        # Wait for started processes; handle KeyboardInterrupt to terminate them.
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            click.echo("Shutting down servers...")
            for p in processes:
                try:
                    p.terminate()
                except Exception:
                    pass
    except Exception as e:
        click.echo(f"Error while starting app components: {e}")


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
cli.add_command(backend)
cli.add_command(mlflow_server)
cli.add_command(model_deployment)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(frontend)
cli.add_command(app)


if __name__ == "__main__":
    cli()
