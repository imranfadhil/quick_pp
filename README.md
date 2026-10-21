# quick_pp

Comprehensive Python library and full-stack web platform for deterministic and probabilistic petrophysical evaluation, Special Core Analysis (SCAL) & relative permeability modeling, multi-well Free Water Level (FWL) contact optimization, stratigraphic log normalization, rock physics, and static-to-dynamic geomodeling upscaling.

This repository contains the core `quick_pp` scientific Python package, the FastAPI backend (`quick_pp/app/backend`), and the SvelteKit interactive workstation frontend (`quick_pp/app/frontend`), plus MLflow-integrated machine learning pipelines (`quick_pp/machine_learning`).

## Core Capabilities

- **Multi-Mineral & Deterministic Petrophysics** (`quick_pp.lithology`, `quick_pp.porosity`, `quick_pp.saturation`, `quick_pp.permeability`):
  Sand-Silt-Clay (modified Kuttan), Carbonate, and multi-mineral inversion solver with Simandoux, Indonesian, Dual-Water, Waxman-Smits, Juhasz, and Archie saturation models.
- **Multi-Well Contact & Transition Zone Optimization** (`quick_pp.saturation.optimize_fwl`, `ContactOptimizer`):
  Simultaneous multi-well Free Water Level (FWL), Oil-Water Contact (OWC), and Top of Transition Zone (TTZ) solver reconciling resistivity-derived $S_w$, capillary pressure Saturation-Height Models ($S_{w,\text{SHM}}$ via Leverett-$J$, Skelt-Harrison, and Cuddy), and wireline formation tester (MDT/RFT) pressure gradients.
- **Special Core Analysis (SCAL) & Relative Permeability** (`quick_pp.scal`, `quick_pp.core_analysis`):
  Laboratory-to-reservoir core plug corrections (hydrostatic/uniaxial overburden compaction, Klinkenberg gas slippage, and Dean-Stark clay-bound water reconciliation), capillary pressure scaling, two-phase relative permeability models (**Corey**, **Brooks-Corey**, and **LET**), automated SCAL curve fitting (`fit_relperm`), and Eclipse/Intersect `SWOF` table generation (`generate_swof_table`).
- **Probabilistic Uncertainty & Monte Carlo Engine** (`quick_pp.uncertainty`):
  Vectorized Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`) for volumetric hydrocarbon in-place uncertainty (`mc_volumetric_uncertainty`), log-level CPI uncertainty propagation (`run_log_uncertainty` producing `P10`/`P50`/`P90` curves), and stochastic reservoir summaries (`calc_stochastic_reservoir_summary`).
- **Log Blocking & Static-to-Dynamic Upscaling** (`quick_pp.upscaling`):
  Petrophysically rigorous blocking (`upscale_logs`, `block_logs`) onto cellular geomodel grids with thickness-weighted arithmetic porosity/shale averaging, pore-volume weighted saturation averaging (preserving `BVHC`), directional & Cardwell-Parsons (1945) power-law permeability bounds ($k_h, k_{\text{eff}}, k_v, k_v/k_h$), and majority-vote facies upscaling.
- **Multi-Well QA/QC & Stratigraphic Normalization** (`quick_pp.qaqc`):
  Zone-by-zone multi-well quantile and histogram log normalization (`stratigraphic_normalization` / `normalize_by_zone`), borehole environmental and invasion corrections (`apply_environmental_corrections`), neutron-density hydrocarbon corrections, and bad-hole flagging.
- **Rock Physics & Gassmann Fluid Substitution** (`quick_pp.rock_physics`):
  Elastic moduli, shear-wave velocity ($V_s$) prediction, Gassmann fluid substitution, and seismic rock physics templates.

## Goals of this README
- Give developers and users the practical steps to run the package and web application locally (backend, frontend, and optional Docker services).

## Project Components
- **Core Package (`quick_pp/`)**: Petrophysics, SCAL, uncertainty, upscaling, rock typing, QA/QC, and rock physics modules exported directly in `quick_pp`.
- **Backend (`quick_pp/app/backend`)**: FastAPI application exposing REST endpoints for LAS handling, QA/QC & stratigraphic normalization, lithology, porosity, saturation & FWL optimization, rock typing, SCAL & relative permeability, permeability transforms, SHF, probabilistic uncertainty, log blocking/upscaling, and interactive plotting.
- **Frontend (`quick_pp/app/frontend`)**: SvelteKit UI providing interactive multi-step petrophysical workflows and Plotly visualisations.
- **Docker (`quick_pp/app/docker`)**: Compose assets to run backend + Postgres for development.
- **CLI (`quick_pp/cli.py`)**: `quick_pp` CLI wrapper that starts services, runs training, prediction, and deployment tasks.
- **Machine Learning (`quick_pp/machine_learning`)**: Training/prediction pipelines and MLflow integration.

Prerequisites
- Python 3.11+ (for backend and CLI)
- Node.js 18+ and npm or yarn (for frontend)
- Docker & Docker Compose (optional, for the packaged backend + DB)
- uv (recommended, fast Python package and environment manager from https://github.com/astral-sh/uv)

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
- Ports: backend API `6312`, frontend dev `5173`, frontend prod `5469`, MLflow UI `5015`, model server `5555`.
- Backend CLI entrypoint: `python main.py` (or `quick_pp` if installed).

Clone & Python setup
1. Clone the repo and create a venv:

```bash
git clone https://github.com/imranfadhil/quick_pp.git
cd quick_pp
uv venv --python 3.11
# mac/linux
source .venv/bin/activate
# windows (cmd.exe)
.venv\Scripts\activate
```

2. Install Python dependencies via modular `pyproject.toml` extras (`uv` or `pip`):

```bash
# Core petrophysics, SCAL, uncertainty & upscaling only
uv pip install -e ".[core]"

# Add machine learning & conformal prediction (MLflow, MAPIE, imbalanced-learn)
uv pip install -e ".[ml]"

# Add FastAPI web backend, MCP server & database layer
uv pip install -e ".[app]"

# Everything (core + ml + app + Jupyter notebooks)
uv pip install -e ".[all]"
```

Or install directly from PyPI using modular extras:

```bash
pip install quick-pp[core]
pip install quick-pp[ml]
pip install quick-pp[app]
pip install quick-pp[all]
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
The frontend is a SvelteKit application. For the best onboarding experience, use the CLI to start the frontend (see CLI commands below). The CLI handles dependency installation, building, and running the server automatically.

For manual setup or advanced development:
1. Install dependencies:

```bash
cd quick_pp/app/frontend
npm install
# Ensure Plotly is available for the UI components
npm install plotly.js-dist-min --save
```

2. Run the dev server:

```bash
npm run dev
```

3. Open the frontend at `http://localhost:5173` (SvelteKit default).

For production build:
- Run `npm run build` to build the app.
- Run `npm run preview` to preview the production build locally.
- The CLI can start the production server automatically (see CLI commands below).

Start the app using the project CLI
- From the repo root you can use the included CLI which orchestrates backend
    and frontend processes. Example (starts backend and, if available, frontend production server):

```bash
python main.py app
# or (if installed) the user-facing command
quick_pp app
```

Start backend only (dev):

```bash
python main.py backend --debug
# or (if installed) the user-facing command
quick_pp backend --debug
```

Start frontend only (dev):

```bash
python main.py frontend --dev
# or (if installed) the user-facing command
quick_pp frontend --dev
```

Start frontend only (prod):

```bash
python main.py frontend
# or (if installed) the user-facing command
quick_pp frontend
```

Common commands
- Run MLflow tracking UI (local): `python main.py mlflow_server`
- Deploy model server: `python main.py model_deployment`
- Train/predict via CLI: see `python main.py --help` or `quick_pp --help`
- Manage Docker services: `python main.py docker up -d` (see `python main.py docker --help` for options)

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
- `03_*` series — deterministic interpretation examples (porosity, saturation, rock typing).
- `04_fdp_and_piml_showcase.ipynb` — **End-to-End FDP & PIML Master Showcase** demonstrating `run_cpi_uncertainty` (`run_log_uncertainty`), `optimize_fwl` (`ContactOptimizer`), `CoreyRelPerm` & `SWOF` generation, `block_logs` (`upscale_logs`), and `ConformalRegressor` / `ConformalMultiOutputRegressor`.
- `04_1_probabilistic_cpi_and_fwl_optimization.ipynb` — Latin Hypercube CPI uncertainty propagation, stochastic STOIIP volumetrics, and multi-well FWL/OWC/TTZ optimization.
- `04_2_scal_relperm_and_log_blocking.ipynb` — SCAL lab-to-reservoir core corrections, `CoreyRelPerm` curve fitting, Eclipse `SWOF` export, and static-to-dynamic log blocking (`block_logs`).
- `04_3_piml_and_conformal_prediction.ipynb` — Physics-Informed Machine Learning (PIML) feature engineering, locally weighted conformal prediction (`ConformalRegressor`), and volumetric closure (`EnforceMineralSum`).

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
