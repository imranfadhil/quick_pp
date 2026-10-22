"""Unit tests for all CLI commands and options in `quick_pp.cli`."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from click.testing import CliRunner

import quick_pp
from quick_pp import cli as cli_mod
from quick_pp.cli import cli, get_gunicorn_worker_flag


class DummyProcess:
    """Lightweight mock subprocess handle."""

    def __init__(self, returncode: int = 0):
        self.returncode = returncode
        self.pid = 99999

    def wait(self, timeout=None):
        return self.returncode

    def poll(self):
        return self.returncode

    def terminate(self):
        pass

    def kill(self):
        pass


def test_cli_version_and_all_subcommand_help():
    """Verify `--version`, `--help`, and `--help` on all 11 CLI subcommands."""
    runner = CliRunner()

    res_ver = runner.invoke(cli, ["--version"])
    assert res_ver.exit_code == 0
    assert quick_pp.__version__ in res_ver.output

    res_help = runner.invoke(cli, ["--help"])
    assert res_help.exit_code == 0

    subcommands = [
        "backend",
        "frontend",
        "app",
        "mlflow-server",
        "model-deployment",
        "train",
        "predict",
        "docker",
        "docker-publish",
        "mcp",
        "create-admin",
    ]
    for sub in subcommands:
        sub_res = runner.invoke(cli, [sub, "--help"])
        assert sub_res.exit_code == 0, f"`quick_pp {sub} --help` failed: {sub_res.output}"


def test_get_gunicorn_worker_flag(monkeypatch):
    """Verify worker flag logic for debug mode, SQLite, and PostgreSQL."""
    assert get_gunicorn_worker_flag(debug=True) == ""

    monkeypatch.setenv("QPP_DATABASE_URL", "sqlite:///./data/quick_pp.db")
    assert get_gunicorn_worker_flag(debug=False) == ""

    monkeypatch.setenv(
        "QPP_DATABASE_URL", "postgresql+psycopg2://user:pass@localhost:5432/quick_pp"
    )
    flag = get_gunicorn_worker_flag(debug=False)
    assert flag.startswith("--workers ")


def test_backend_command_options(monkeypatch):
    """Verify `quick_pp backend` with `--debug` and `--no-open`, when stopped and when already running."""
    runner = CliRunner()
    started = []
    opened = []

    monkeypatch.setattr(cli_mod, "is_server_running", lambda host, port: False)
    monkeypatch.setattr(
        cli_mod,
        "start_process",
        lambda cmd, cwd=None, shell=False, env=None: (
            started.append(cmd) or DummyProcess(0)
        ),
    )
    monkeypatch.setattr(cli_mod.webbrowser, "open", lambda url: opened.append(url))

    # 1. Start with --debug and --no-open
    res = runner.invoke(cli, ["backend", "--debug", "--no-open"])
    assert res.exit_code == 0
    assert len(started) == 1
    assert "--reload" in started[0]
    assert opened == []

    # 2. Already running without --no-open -> opens browser
    monkeypatch.setattr(cli_mod, "is_server_running", lambda host, port: True)
    res_running = runner.invoke(cli, ["backend"])
    assert res_running.exit_code == 0
    assert "already running" in res_running.output
    assert opened == [f"http://{cli_mod.BACKEND_HOST}:{cli_mod.BACKEND_PORT}"]


def test_frontend_command_prod_and_dev_options(monkeypatch, tmp_path):
    """Verify `quick_pp frontend` in default production mode and `--dev` / `--force-install` mode."""
    runner = CliRunner()
    fake_frontend = tmp_path / "frontend"
    (fake_frontend / "build").mkdir(parents=True)
    (fake_frontend / "build" / "index.js").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(cli_mod, "FRONTEND_DIR", fake_frontend)
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="v20.0.0"),
    )

    started_procs = []
    monkeypatch.setattr(
        cli_mod,
        "start_process",
        lambda cmd, cwd=None, shell=False, env=None: (
            started_procs.append((cmd, cwd, env)) or DummyProcess(0)
        ),
    )

    # 1. Default production mode with --no-open
    res_prod = runner.invoke(cli, ["frontend", "--no-open"])
    assert res_prod.exit_code == 0
    assert len(started_procs) == 1
    cmd, cwd, env = started_procs[0]
    assert cmd == ["node", "build/index.js"]
    assert env["PORT"] == str(cli_mod.FRONTEND_PORT)
    assert env["ORIGIN"] == f"http://localhost:{cli_mod.FRONTEND_PORT}"

    # 2. Development mode with --dev, --force-install, --no-open
    popen_calls = []
    (fake_frontend / "node_modules").mkdir(exist_ok=True)
    (fake_frontend / "package-lock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cli_mod,
        "Popen",
        lambda cmd, **kw: popen_calls.append((cmd, kw)) or DummyProcess(0),
    )

    res_dev = runner.invoke(cli, ["frontend", "--dev", "--force-install", "--no-open"])
    assert res_dev.exit_code == 0
    assert any("npm install" in str(c[0]) for c in popen_calls)
    assert any("npm run dev" in str(c[0]) for c in popen_calls)


def test_app_command_starts_both_and_handles_open_flag(monkeypatch, tmp_path):
    """Verify `quick_pp app` and `quick_pp app --open` both when starting fresh and when servers are running."""
    runner = CliRunner()
    fake_frontend = tmp_path / "frontend"
    (fake_frontend / "build").mkdir(parents=True)
    (fake_frontend / "build" / "index.js").write_text("// stub", encoding="utf-8")

    monkeypatch.setattr(cli_mod, "FRONTEND_DIR", fake_frontend)
    monkeypatch.setattr(
        cli_mod.subprocess,
        "run",
        lambda *a, **kw: SimpleNamespace(returncode=0, stdout="v20.0.0"),
    )

    started = []
    opened = []
    monkeypatch.setattr(cli_mod, "is_server_running", lambda host, port: False)
    monkeypatch.setattr(
        cli_mod,
        "start_process",
        lambda cmd, cwd=None, shell=False, env=None: (
            started.append((cmd, cwd, env)) or DummyProcess(0)
        ),
    )
    monkeypatch.setattr(cli_mod.webbrowser, "open", lambda url: opened.append(url))

    res = runner.invoke(cli, ["app", "--open"])
    assert res.exit_code == 0
    assert len(started) == 2
    assert len(opened) == 2

    # When both backend and frontend are already running, `app --open` must not raise UnboundLocalError
    started.clear()
    opened.clear()
    monkeypatch.setattr(cli_mod, "is_server_running", lambda host, port: True)
    res_already = runner.invoke(cli, ["app", "--open"])
    assert res_already.exit_code == 0
    assert "Backend already running" in res_already.output
    assert "Frontend already running" in res_already.output
    assert len(started) == 0
    assert len(opened) == 2


def test_mlflow_and_model_deployment_and_mcp_commands(monkeypatch):
    """Verify `mlflow-server`, `model-deployment --debug`, and `mcp` CLI commands."""
    runner = CliRunner()

    # 1. mlflow-server
    mlflow_envs = []
    monkeypatch.setattr(
        "quick_pp.machine_learning.utils.run_mlflow_server",
        lambda env: mlflow_envs.append(env),
    )
    res_mlf = runner.invoke(cli, ["mlflow-server", "local"])
    assert res_mlf.exit_code == 0
    assert mlflow_envs == ["local"]

    # 2. model-deployment --debug
    md_started = []
    monkeypatch.setattr(cli_mod, "is_server_running", lambda host, port: False)
    monkeypatch.setattr(
        cli_mod,
        "start_process",
        lambda cmd, cwd=None, shell=False, env=None: (
            md_started.append(cmd) or DummyProcess(0)
        ),
    )
    res_md = runner.invoke(cli, ["model-deployment", "--debug"])
    assert res_md.exit_code == 0
    assert len(md_started) == 1
    assert "--reload" in md_started[0]

    # 3. mcp
    mcp_called = []
    monkeypatch.setattr(
        "quick_pp.app.backend.mcp_server.petro_mcp_server.run_stdio",
        lambda: mcp_called.append(True),
    )
    res_mcp = runner.invoke(cli, ["mcp"])
    assert res_mcp.exit_code == 0
    assert mcp_called == [True]


def test_train_and_predict_commands(monkeypatch, tmp_path):
    """Verify `quick_pp train` and `quick_pp predict --plot` commands."""
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)

    train_calls = []
    monkeypatch.setattr(
        "quick_pp.machine_learning.train_pipeline.train_pipeline",
        lambda model_config, data_hash, env: train_calls.append(
            (model_config, data_hash, env)
        ),
    )
    res_train = runner.invoke(cli, ["train", "clastic", "mock", "local"])
    assert res_train.exit_code == 0
    assert train_calls == [("clastic", "mock", "local")]
    assert (tmp_path / "config.py").exists()
    assert (tmp_path / "data" / "input" / "mock_data.parquet").exists()

    predict_calls = []
    monkeypatch.setattr(
        "quick_pp.machine_learning.predict_pipeline.predict_pipeline",
        lambda model_config, data_hash, output_file_name, env, plot: predict_calls.append(
            (model_config, data_hash, output_file_name, env, plot)
        ),
    )
    res_pred = runner.invoke(
        cli, ["predict", "clastic", "mock", "out_test", "local", "--plot"]
    )
    assert res_pred.exit_code == 0
    assert predict_calls == [("clastic", "mock", "out_test", "local", True)]


def test_docker_and_docker_publish_commands(monkeypatch, tmp_path):
    """Verify `quick_pp docker` options (`up -d`, `logs -f`, `--service`, `--profile`, `--env-file`) and `docker-publish`."""
    runner = CliRunner()
    monkeypatch.chdir(tmp_path)
    custom_env = tmp_path / "custom.env"
    custom_env.write_text("WEB_PORT=8080\n", encoding="utf-8")

    popen_calls = []
    monkeypatch.setattr(
        cli_mod,
        "Popen",
        lambda cmd, **kw: popen_calls.append((cmd, kw)) or DummyProcess(0),
    )

    # 1. docker up -d --profile tunnel --service qpp-backend --env-file custom.env
    res_up = runner.invoke(
        cli,
        [
            "docker",
            "up",
            "-d",
            "--profile",
            "tunnel",
            "--service",
            "qpp-backend",
            "--env-file",
            str(custom_env),
        ],
    )
    assert res_up.exit_code == 0
    cmd, kw = popen_calls[-1]
    assert cmd[:2] == ["docker", "compose"]
    assert "--env-file" in cmd
    assert "--profile" in cmd and "tunnel" in cmd
    assert "up" in cmd and "-d" in cmd and "qpp-backend" in cmd
    assert kw["env"]["QPP_HOST_DATA_DIR"] == str((tmp_path / "data").resolve())
    assert kw["env"]["QPP_HOST_UPLOADS_DIR"] == str((tmp_path / "uploads").resolve())

    # 2. docker logs -f --service postgres
    res_logs = runner.invoke(
        cli, ["docker", "logs", "-f", "--service", "postgres"]
    )
    assert res_logs.exit_code == 0
    cmd_logs, _ = popen_calls[-1]
    assert cmd_logs[-3:] == ["logs", "-f", "postgres"]

    # 3. docker down
    res_down = runner.invoke(cli, ["docker", "down"])
    assert res_down.exit_code == 0
    cmd_down, _ = popen_calls[-1]
    assert cmd_down[-1] == "down"

    # 4. docker-publish with credentials
    check_calls = []
    monkeypatch.setenv("DOCKER_USERNAME", "testuser")
    monkeypatch.setenv("DOCKER_PASSWORD", "testpass")
    monkeypatch.setattr(
        cli_mod.subprocess,
        "check_call",
        lambda cmd: check_calls.append(cmd) or 0,
    )
    login_mock = MagicMock()
    login_mock.communicate.return_value = (b"", b"")
    login_mock.returncode = 0
    monkeypatch.setattr(cli_mod.subprocess, "Popen", lambda *a, **kw: login_mock)

    res_pub = runner.invoke(cli, ["docker-publish"])
    assert res_pub.exit_code == 0
    assert len(check_calls) == 2
    assert check_calls[0][:2] == ["docker", "build"]
    assert check_calls[1] == ["docker", "push", "imranfadhil86/quick_pp-backend:latest"]


def test_create_admin_command(monkeypatch, tmp_path):
    """Verify `quick_pp create-admin` creates and updates admin accounts with `--reset-mfa`."""
    runner = CliRunner()
    db_file = tmp_path / "cli_admin_test.db"
    monkeypatch.setenv("QPP_DATABASE_URL", f"sqlite:///{db_file.as_posix()}")

    res_create = runner.invoke(
        cli,
        [
            "create-admin",
            "--username",
            "cli_admin",
            "--email",
            "cli_admin@example.com",
            "--password",
            "AdminPass@2026!",
        ],
    )
    assert res_create.exit_code == 0
    assert "Created new administrator" in res_create.output
    assert "cli_admin" in res_create.output

    res_update = runner.invoke(
        cli,
        [
            "create-admin",
            "--username",
            "cli_admin",
            "--email",
            "cli_admin@example.com",
            "--password",
            "NewAdminPass@2026!",
            "--reset-mfa",
        ],
    )
    assert res_update.exit_code == 0
    assert "Updated existing administrator" in res_update.output

