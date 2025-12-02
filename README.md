# quick_pp

Lightweight toolkit for quick-look petrophysical estimation and exploration.

This repository contains the Python backend and SvelteKit frontend used by the
quick_pp application, plus utilities for running ML training and simple
petrophysical workflows.

Goals of this README
- Give developers and users the minimal, practical steps to get the app
    running locally (backend, frontend and optional Docker services).

Project components
- Backend: FastAPI application, data services, model endpoints and plotting APIs (in `quick_pp/app/backend`).
- Frontend: SvelteKit UI (in `quick_pp/app/frontend`) providing data visualisations and tools.
- Docker: Compose assets to run backend + Postgres for development (`quick_pp/app/docker`).
- CLI: `quick_pp` CLI wrapper that starts services, runs training, prediction and deployment tasks (`quick_pp/cli.py`).
- Machine learning: training/prediction pipelines and MLflow integration (`quick_pp/machine_learning`).

Prerequisites
- Python 3.11+ (for backend and CLI)
- Node.js 18+ and npm or yarn (for frontend)
- Docker & Docker Compose (optional, for the packaged backend + DB)

.env & Database (SQLite vs PostgreSQL)
- The application reads DB and other secrets from environment variables. For
    local development create a `.env` file in the repo root or use `quick_pp/app/docker/.env`
    when running the bundled Docker Compose stack.

- Minimal `.env` examples

    SQLite (quick local testing)
    ```bash
    QPP_DATABASE_URL=sqlite:///./data/local.db
    QPP_SECRET_KEY=change-this-to-a-random-string
    ```

    PostgreSQL (recommended for realistic usage / Docker)
    ```bash
    QPP_DATABASE_URL=postgresql://qpp_user:qpp_pass@postgres:5432/quick_pp
    QPP_SECRET_KEY=replace-with-secure-value
    # if you run DB externally, replace host with reachable hostname or IP
    ```

    Which to choose
    - SQLite: easiest for quick, single-user experiments. No external DB
        server required but limited in concurrency and not recommended for
        multi-container deployments.
    - PostgreSQL: recommended for Docker and production-like setups; the
        `quick_pp/app/docker/docker-compose.yaml` in the repo is configured to create a
        Postgres service and a matching `.env` template.

    Security note
    - Never commit secrets (`QPP_SECRET_KEY`, DB passwords) to version control.
        Use environment-specific `.env` files excluded via `.gitignore` or a
        secrets manager.

Quick checklist
- Ports: backend API `6312`, frontend dev `5173`, MLflow UI `5015`, model server `5555`.
- Backend CLI entrypoint: `python main.py` (or `quick_pp` if installed).

Clone & Python setup
1. Clone the repo and create a venv:

```bash
git clone https://github.com/imranfadhil/quick_pp.git
cd quick_pp
python -m venv .venv
# mac/linux
source .venv/bin/activate
# windows (cmd.exe)
.venv\Scripts\activate
```

2. Install Python dependencies:

```bash
pip install -r requirements.txt
# (optional) install package editable for CLI convenience
pip install -e .
```

Using Docker (recommended for a complete local stack)
- The repo provides Docker assets in `quick_pp/app/docker/` to start the backend and
    a Postgres data volume.

Quick docker compose (from repo root):

```bash
cd quick_pp/app/docker
docker-compose up -d
```

This will bring up services configured for development. Logs can be checked
with `docker-compose logs -f` in the same folder.

Frontend (SvelteKit)
1. Install frontend dependencies and run the dev server:

```bash
cd quick_pp/app/frontend
npm install
# Ensure Plotly is available for the UI components
npm install plotly.js-dist-min --save
npm run dev
```

2. Open the frontend at `http://localhost:5173` (SvelteKit default).

Start the app using the project CLI
- From the repo root you can use the included CLI which orchestrates backend
    and frontend processes. Example (starts backend and, if available, frontend):

```bash
python main.py app
# or (if installed) the user-facing command
quick_pp app
```

Start backend only (dev):

```bash
python main.py backend --debug
```

Start frontend only (dev):

```bash
python main.py frontend
```

Common commands
- Run MLflow tracking UI (local): `python main.py mlflow_server`
- Deploy model server: `python main.py model_deployment`
- Train/predict via CLI: see `python main.py --help` or `quick_pp --help`

Testing
- Run unit tests with `pytest` in the repo root:

```bash
pytest -q
```

Troubleshooting & tips
- If the frontend does not render charts, ensure `plotly.js-dist-min` is
    installed in `quick_pp/app/frontend` (some components do dynamic imports).
- If the backend fails to start behind Docker, check `quick_pp/app/docker/.env` and
    the Postgres volumes under `quick_pp/app/docker/data/`.
- Use the CLI `python main.py` for convenience; it will open browser
    windows for the services it starts unless `--no-open` is provided.

Further reading
- API docs are available when the backend is running: `http://localhost:6312/docs`.
- Project documentation: https://quick-pp.readthedocs.io/en/latest/index.html

License
- See the `LICENSE` file in the repository root.

Contributions and feedback welcome — open an issue or a PR with improvements.

Jupyter Notebooks
-----------------
The repository includes several example notebooks under `notebooks/` that
demonstrate data handling, EDA and basic petrophysical workflows. Recommended
workflow for exploring the project locally:

1. Start the backend API (see CLI commands above) if a notebook calls the API.
2. Open a Python environment with the project dependencies installed.
3. Launch JupyterLab or Jupyter Notebook and open the notebooks in `notebooks/`.

Key notebooks:
- `01_data_handler.ipynb` — create and inspect a mock `qppp` project file.
- `02_EDA.ipynb` — quick exploratory data analysis patterns used in demos.
- `03_*` series — interpretation examples (porosity, saturation, rock typing).

Machine learning (Train / Predict / Deploy)
-----------------------------------------
The project includes ML training and prediction utilities integrated with
MLflow. High-level steps and helpful details:

1. Prepare input data
     - Training expects a Parquet file in `data/input/<data_hash>___.parquet`.
     - The feature set required by each modelling config is defined in
         `quick_pp/machine_learning/config.py` (or `MODELLING_CONFIG` used by
         training code). Ensure input columns match the configured features.

2. Train a model (local)

```bash
# from repo root, with virtualenv active
python main.py train <model_config> <data_hash>
# example
python main.py train mock mock
```

3. Run predictions

```bash
python main.py predict <model_config> <data_hash> [output_name] [--plot]
# example
python main.py predict mock mock results_test --plot
```

4. Deploy model server (serves registered MLflow models)

```bash
python main.py model_deployment
```

Notes:
- MLflow UI (tracking server) is available with `python main.py mlflow_server`.
- The `--plot` flag in `predict` saves visual outputs (if supported by the
    predict pipeline).
- For production or reproducible experiments, register models in MLflow and
    configure the model registry settings used by the deployment code.
