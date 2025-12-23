"""
Command-Line Interface for the quick_pp library.

This script provides a set of commands to interact with the various components
of the quick_pp application, including running the web app, managing MLflow
servers, and executing machine learning training and prediction pipelines.
"""

import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from importlib import metadata, resources
from pathlib import Path
from subprocess import Popen

import click


def on_starting(server):
    """Gunicorn hook: initialize DBConnector in master process before forking.

    This performs application-level initialization (like running migrations or
    creating a DB connector) in the master process. Use with caution: some DB
    drivers are not fork-safe, so forking after opening DB connections may be
    problematic. Consider disabling `preload_app` if you need per-worker DB
    engines.
    """
    try:
        server.log.info("gunicorn on_starting: initializing DBConnector")
        from quick_pp.database.db_connector import DBConnector

        db_url = os.environ.get("QPP_DATABASE_URL")
        DBConnector(db_url=db_url)
        server.log.info("DBConnector initialized in gunicorn master process")
    except Exception as exc:  # pragma: no cover - environment-specific
        try:
            server.log.exception(
                "Failed to initialize DBConnector in gunicorn master: %s", exc
            )
        except Exception:
            pass


try:
    quick_ppVersion = metadata.version("quick_pp")
except metadata.PackageNotFoundError:
    quick_ppVersion = "0.0.0"

# Global defaults for backend and frontend locations used across commands
BACKEND_HOST = "localhost"
BACKEND_PORT = 6312
FRONTEND_PORT = 5469
FRONTEND_DIR = Path(__file__).parent / "app" / "frontend"
DOCKER_DIR = Path(__file__).parent / "app" / "docker"


def get_gunicorn_worker_flag(debug: bool = False) -> str:
    """Return a gunicorn workers flag based on available CPUs.

    - If `debug`/reload is enabled, return an empty string (reload and
        multiple workers don't mix well for development).
    - Otherwise return a string like `--workers N` where N is the number
        of CPUs (at least 1).
    """
    if debug:
        return ""

    # Defensive: if using SQLite, prefer a single worker to avoid file locking
    db_url = os.environ.get("QPP_DATABASE_URL")
    if "sqlite" in (db_url or "").lower():
        # Do not pass a workers flag; gunicorn defaults to single-process.
        return ""

    try:
        cpus = os.cpu_count() or 1
    except Exception:
        cpus = 1
    workers = max(1, int(cpus / 3))
    return f"--workers {workers}"


def start_process(cmd, cwd=None, shell=False, env=None):
    """Start a subprocess in a new process group/session for graceful shutdown.

    On Windows, use CREATE_NEW_PROCESS_GROUP and on POSIX use setsid so we can
    signal the whole group (uvicorn master + workers).
    """
    # If cmd is a list, pass it directly (recommended for uvicorn to avoid shell)
    if os.name == "nt":
        return subprocess.Popen(
            cmd, stdout=sys.stdout, stderr=sys.stderr, shell=shell, cwd=cwd, env=env
        )
    else:
        return subprocess.Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=shell,
            cwd=cwd,
            env=env,
            preexec_fn=os.setsid if not shell else None,
        )


def stop_process_gracefully(proc, timeout: float = 5.0):
    """Attempt graceful shutdown of a process started with `start_process`.

    - On Windows: send CTRL_BREAK_EVENT to the process group.
    - On POSIX: send SIGINT to the process group.
    Falls back to terminate/kill if the process doesn't exit within `timeout` seconds.
    """
    try:
        if proc.poll() is not None:
            return

        if os.name == "nt":
            try:
                proc.send_signal(signal.CTRL_BREAK_EVENT)
            except Exception:
                pass
        else:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGINT)
            except Exception:
                pass

        # wait for graceful exit
        start = time.time()
        while True:
            if proc.poll() is not None:
                return
            if time.time() - start > timeout:
                break
            time.sleep(0.1)

        try:
            proc.terminate()
        except Exception:
            pass

        try:
            proc.wait(2)
        except Exception:
            try:
                proc.kill()
            except Exception:
                pass
            try:
                proc.wait()
            except Exception:
                pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


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
        if os.name != "nt":
            argv = [
                sys.executable,
                "-m",
                "gunicorn",
                "-k",
                "uvicorn.workers.UvicornWorker",
                "quick_pp.app.backend.main:app",
                "--bind",
                f"0.0.0.0:{BACKEND_PORT}",
                "--preload",
                "--config",
                str(Path(__file__).resolve()),
            ]
            # worker_flag may be empty or like "--workers N"
            worker_flag = get_gunicorn_worker_flag(debug)
            if worker_flag:
                argv.extend(worker_flag.split())
        else:
            # Build argv list for uvicorn to avoid shell and improve signal delivery
            argv = [
                sys.executable,
                "-m",
                "uvicorn",
                "quick_pp.app.backend.main:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(BACKEND_PORT),
            ]

        if debug:
            argv.append("--reload")
        click.echo(f"App is not running. Starting it now... | {argv}")

        try:
            process = start_process(argv, shell=False)
            # Open browser to backend URL after starting (default)
            try:
                if not no_open:
                    webbrowser.open(f"http://{BACKEND_HOST}:{BACKEND_PORT}")
            except Exception as e:
                click.echo(f"Failed to open browser: {e}")

            # Wait for the process and handle exit codes
            exit_code = process.wait()
            if exit_code != 0:
                click.echo(f"Backend process exited with code: {exit_code}")
                sys.exit(exit_code)

        except (KeyboardInterrupt, click.exceptions.Abort):
            click.echo("Shutting down backend server...")
            try:
                stop_process_gracefully(process)
            except Exception:
                pass
        except Exception as e:
            click.echo(f"Failed to start backend server: {e}")
            sys.exit(1)
    else:
        click.echo(f"Backend is already running on {BACKEND_HOST}:{BACKEND_PORT}")
        if not no_open:
            try:
                webbrowser.open(f"http://{BACKEND_HOST}:{BACKEND_PORT}")
            except Exception as e:
                click.echo(f"Failed to open browser: {e}")


@click.command()
@click.option(
    "--dev",
    is_flag=True,
    default=False,
    help="Run `npm run dev`",
)
@click.option(
    "--force-install",
    is_flag=True,
    default=False,
    help="Force npm install to run even if node_modules exists",
)
@click.option(
    "--no-open",
    is_flag=True,
    default=False,
    help="Do not open browser after starting frontend",
)
def frontend(dev, force_install, no_open):
    """Start the quick_pp frontend server.

    By default, launches the SvelteKit development server (npm run dev).
    Use --prod to run the production build (requires building first)."""
    frontend_dir = FRONTEND_DIR

    if not frontend_dir.exists():
        click.echo(f"Error: Frontend directory not found at {frontend_dir}")
        return

    # Check if Node.js is installed
    try:
        node_check = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        )
        if node_check.returncode != 0:
            click.echo("Error: Node.js is not installed or not in PATH.")
            click.echo("Please install Node.js from https://nodejs.org/")
            return
    except (subprocess.TimeoutExpired, FileNotFoundError):
        click.echo("Error: Node.js is not installed or not in PATH.")
        click.echo("Please install Node.js from https://nodejs.org/")
        return

    # Check if running production build
    if not dev:
        build_dir = frontend_dir / "build"
        if not build_dir.exists():
            click.echo(
                "Frontend build not found. Please ensure the frontend is pre-built and included in the package."
            )
            return
        click.echo(f"Starting frontend production server on port {FRONTEND_PORT}...")

        # Set environment variables for the production server
        env = os.environ.copy()
        env["PORT"] = str(FRONTEND_PORT)
        env["HOST"] = "0.0.0.0"

        # Run the built Node.js server
        cmd = ["node", "build/index.js"]
        process = start_process(cmd, cwd=str(frontend_dir), shell=False, env=env)

        # Open browser to production URL unless disabled
        try:
            if not no_open:
                time.sleep(1)  # Give server a moment to start
                webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
        except Exception:
            pass

        try:
            process.wait()
        except KeyboardInterrupt:
            click.echo("\nShutting down frontend server...")
            stop_process_gracefully(process)
        return

    # Development mode (default)
    # Check if we need to run npm install
    should_install = force_install or not (frontend_dir / "node_modules").exists()

    if should_install:
        if force_install:
            click.echo(
                "Force install requested. Cleaning node_modules and package-lock.json..."
            )
            # Remove node_modules and package-lock.json for a clean install
            node_modules = frontend_dir / "node_modules"
            package_lock = frontend_dir / "package-lock.json"
            if node_modules.exists():
                shutil.rmtree(node_modules)
                click.echo("Removed node_modules")
            if package_lock.exists():
                package_lock.unlink()
                click.echo("Removed package-lock.json")
            click.echo("Running 'npm install'...")
        else:
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
@click.option(
    "--open",
    is_flag=True,
    default=False,
    help="Open browser after starting services",
)
def app(open):
    """Start both backend and frontend development servers.

    This command will start the backend (uvicorn) on port 6312 and the
    frontend production server from `quick_pp/app/frontend` if they are not
    already running.
    """
    processes = []
    try:
        # Start backend if not running
        if not is_server_running(BACKEND_HOST, BACKEND_PORT):
            if os.name != "nt":
                argv = [
                    sys.executable,
                    "-m",
                    "gunicorn",
                    "-k",
                    "uvicorn.workers.UvicornWorker",
                    "quick_pp.app.backend.main:app",
                    "--bind",
                    f"0.0.0.0:{BACKEND_PORT}",
                    "--preload",
                    "--config",
                    str(Path(__file__).resolve()),
                ]
                worker_flag = get_gunicorn_worker_flag(False)  # No debug for production
                if worker_flag:
                    argv.extend(worker_flag.split())
            else:
                argv = [
                    sys.executable,
                    "-m",
                    "uvicorn",
                    "quick_pp.app.backend.main:app",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(BACKEND_PORT),
                ]
            click.echo(f"Starting backend... | {' '.join(argv)}")
            p_backend = start_process(argv, shell=False)
            processes.append(p_backend)
        else:
            click.echo("Backend already running on localhost:6312")

        # Start frontend if available
        # Check if Node.js is installed
        try:
            node_check = subprocess.run(
                ["node", "--version"], capture_output=True, text=True, timeout=5
            )
            if node_check.returncode != 0:
                click.echo("Error: Node.js is not installed or not in PATH.")
                click.echo("Please install Node.js from https://nodejs.org/")
                return
        except (subprocess.TimeoutExpired, FileNotFoundError):
            click.echo("Error: Node.js is not installed or not in PATH.")
            click.echo("Please install Node.js from https://nodejs.org/")
            return

        frontend_dir = FRONTEND_DIR
        build_dir = frontend_dir / "build"
        if not build_dir.exists():
            click.echo(
                f"Frontend build not found at {build_dir}. Skipping frontend start."
            )
        else:
            click.echo(
                f"Starting frontend production server on port {FRONTEND_PORT}..."
            )

            # Set environment variables for the production server
            env = os.environ.copy()
            env["PORT"] = str(FRONTEND_PORT)
            env["HOST"] = "0.0.0.0"

            # Run the built Node.js server
            cmd = ["node", "build/index.js"]
            p_front = start_process(cmd, cwd=str(frontend_dir), shell=False, env=env)
            processes.append(p_front)

        # Open browser(s) after launching processes (default behavior)
        try:
            if open:
                # backend
                if any(p == p_backend for p in processes) or is_server_running(
                    BACKEND_HOST, BACKEND_PORT
                ):
                    webbrowser.open(f"http://{BACKEND_HOST}:{BACKEND_PORT}")
                # frontend production port
                if any(p == p_front for p in processes) or (
                    frontend_dir.exists() and (frontend_dir / "build").exists()
                ):
                    webbrowser.open(f"http://localhost:{FRONTEND_PORT}")
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
        except (KeyboardInterrupt, click.exceptions.Abort):
            click.echo("Shutting down servers...")
            for p in processes:
                try:
                    stop_process_gracefully(p)
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
        if os.name != "nt":
            argv = [
                sys.executable,
                "-m",
                "gunicorn",
                "-k",
                "uvicorn.workers.UvicornWorker",
                "quick_pp.app.backend.mlflow_model_deployment:app",
                "--bind",
                "0.0.0.0:5555",
                "--preload",
                "--config",
                str(Path(__file__).resolve()),
            ]
        else:
            argv = [
                sys.executable,
                "-m",
                "uvicorn",
                "quick_pp.app.backend.mlflow_model_deployment:app",
                "--host",
                "0.0.0.0",
                "--port",
                "5555",
            ]
        if debug:
            argv.append("--reload")
        worker_flag = get_gunicorn_worker_flag(debug)
        if worker_flag:
            argv.extend(worker_flag.split())
        click.echo(
            f"Model server is not running. Starting it now... | {' '.join(argv)}"
        )
        process = start_process(argv, shell=False)
        try:
            process.wait()
        except (KeyboardInterrupt, click.exceptions.Abort):
            stop_process_gracefully(process)


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


@click.command()
@click.argument(
    "action", type=click.Choice(["up", "down", "build", "logs", "ps", "restart"])
)
@click.option(
    "--detach",
    "-d",
    is_flag=True,
    default=False,
    help="Run in detached mode (background)",
)
@click.option(
    "--build",
    is_flag=True,
    default=False,
    help="Build images before starting containers",
)
@click.option(
    "--no-cache",
    is_flag=True,
    default=False,
    help="Do not use cache when building images (for up/build)",
)
@click.option(
    "--profile",
    multiple=True,
    help="Enable specific profiles (e.g., --profile langflow)",
)
@click.option(
    "--service",
    help="Target specific service (postgres, qpp-backend, qpp-frontend, langflow)",
)
@click.option(
    "--follow",
    "-f",
    is_flag=True,
    default=False,
    help="Follow log output (for logs command)",
)
@click.option(
    "--env-file",
    type=click.Path(exists=True),
    default=None,
    help="Path to .env file (defaults to .env in repo root if it exists)",
)
def docker(action, detach, build, no_cache, profile, service, follow, env_file):
    """Manage Docker services using docker-compose.

    ACTIONS:
    \b
    up      - Start services
    down    - Stop and remove services
    build   - Build or rebuild services
    logs    - View service logs
    ps      - List running services
    restart - Restart services

    EXAMPLES:
    \b
    quick-pp docker up -d                    # Start all services in background
    quick-pp docker up --profile langflow    # Start with Langflow enabled
    quick-pp docker logs --service postgres  # View postgres logs
    quick-pp docker logs -f                  # Follow all logs
    quick-pp docker down                     # Stop all services
    """
    docker_dir = DOCKER_DIR

    if not docker_dir.exists():
        click.echo(f"Error: Docker directory not found at {docker_dir}")
        return

    if not (docker_dir / "docker-compose.yaml").exists():
        click.echo(f"Error: docker-compose.yaml not found in {docker_dir}")
        return

    # Build the docker-compose command
    cmd_parts = ["docker", "compose"]

    # Add env-file option (defaults to .env in repo root if it exists)
    if env_file:
        cmd_parts.extend(["--env-file", str(env_file)])
    else:
        # Try to find .env in repo root (parent of quick_pp package)
        repo_root = Path(__file__).parent.parent
        default_env_file = repo_root / ".env"
        if default_env_file.exists():
            cmd_parts.extend(["--env-file", str(default_env_file)])
            click.echo(f"Using .env file: {default_env_file}")

    # Add profile options
    for prof in profile:
        cmd_parts.extend(["--profile", prof])

    # Add the action
    cmd_parts.append(action)

    # For build, --no-cache must come after --build and before service name
    if action == "up":
        if detach:
            cmd_parts.append("-d")
        if build:
            cmd_parts.append("--build")
    elif action == "build":
        if no_cache:
            cmd_parts.append("--no-cache")
        if service:
            cmd_parts.append(service)
    elif action == "logs":
        if follow:
            cmd_parts.append("-f")
        if service:
            cmd_parts.append(service)
    elif action in ["down", "restart", "ps"]:
        if service:
            cmd_parts.append(service)

    # Join command parts
    cmd = " ".join(cmd_parts)

    click.echo(f"Running: (in {docker_dir}) {cmd}")

    try:
        process = Popen(
            cmd,
            stdout=sys.stdout,
            stderr=sys.stderr,
            shell=True,
            cwd=str(docker_dir),
        )

        if action == "up" and not detach:
            try:
                process.wait()
            except (KeyboardInterrupt, click.exceptions.Abort):
                click.echo("\nShutting down services...")
                # Run docker-compose down to gracefully stop services
                down_cmd = "docker-compose down"
                down_process = Popen(
                    down_cmd,
                    stdout=sys.stdout,
                    stderr=sys.stderr,
                    shell=True,
                    cwd=str(docker_dir),
                )
                down_process.wait()
        else:
            exit_code = process.wait()
            if exit_code != 0:
                click.echo(f"Docker command exited with code: {exit_code}")
                sys.exit(exit_code)

    except Exception as e:
        click.echo(f"Failed to run docker command: {e}")
        sys.exit(1)


@click.command()
def docker_publish():
    """Build and push the qpp-backend Docker image to Docker Hub as 'latest'."""
    docker_dir = DOCKER_DIR
    dockerfile_path = docker_dir / "Dockerfile"
    image_tag = "imranfadhil86/quick_pp-backend:latest"
    click.echo(f"Building Docker image: {image_tag}")
    build_cmd = [
        "docker",
        "build",
        "-f",
        str(dockerfile_path),
        "-t",
        image_tag,
        str(docker_dir.parent.parent.parent),  # repo root
    ]
    try:
        subprocess.check_call(build_cmd)
    except subprocess.CalledProcessError as e:
        click.echo(f"Docker build failed: {e}")
        sys.exit(1)
    click.echo(f"Pushing Docker image: {image_tag}")
    docker_username = os.environ.get("DOCKER_USERNAME")
    docker_password = os.environ.get("DOCKER_PASSWORD")
    if not docker_username or not docker_password:
        click.echo(
            "DOCKER_USERNAME and DOCKER_PASSWORD environment variables are required for docker login."
        )
        sys.exit(1)
    login_cmd = ["docker", "login", "--username", docker_username, "--password-stdin"]
    try:
        login_proc = subprocess.Popen(
            login_cmd, stdin=subprocess.PIPE, stdout=sys.stdout, stderr=sys.stderr
        )
        login_proc.communicate(input=docker_password.encode())
        if login_proc.returncode != 0:
            click.echo("Docker login failed.")
            sys.exit(1)
    except Exception as e:
        click.echo(f"Docker login error: {e}")
        sys.exit(1)
    push_cmd = ["docker", "push", image_tag]
    try:
        subprocess.check_call(push_cmd)
    except subprocess.CalledProcessError as e:
        click.echo(f"Docker push failed: {e}")
        sys.exit(1)
    click.echo(f"Successfully built and pushed {image_tag}")


# Add commands to the CLI group
cli.add_command(backend)
cli.add_command(mlflow_server)
cli.add_command(model_deployment)
cli.add_command(train)
cli.add_command(predict)
cli.add_command(frontend)
cli.add_command(app)
cli.add_command(docker)
cli.add_command(docker_publish)


if __name__ == "__main__":
    cli()
