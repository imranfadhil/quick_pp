# quick_pp

[![CI](https://github.com/imranfadhil/quick_pp/actions/workflows/ci.yml/badge.svg)](https://github.com/imranfadhil/quick_pp/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/quick-pp.svg)](https://badge.fury.io/py/quick-pp)
[![Documentation Status](https://readthedocs.org/projects/quick-pp/badge/?version=latest)](https://quick-pp.readthedocs.io/en/latest/?badge=latest)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Comprehensive Python library and full-stack web platform for deterministic and probabilistic petrophysical evaluation, unified borehole environmental & radial invasion QC, Special Core Analysis (SCAL) & relative permeability modeling, multi-well Free Water Level (FWL) & Formation Pressure Testing (FPT) contact analysis, stratigraphic log normalization, source-rock TOC geochemistry, core-log calibration & depth matching, automated CPI/FDP report generation with subsurface gate checks, bi-directional petrophysical handover packaging, rock physics, and static-to-dynamic geomodeling upscaling.

This repository contains the core `quick_pp` scientific Python package, the FastAPI backend (`quick_pp/app/backend`), and the SvelteKit interactive workstation frontend (`quick_pp/app/frontend`), plus MLflow-integrated machine learning pipelines (`quick_pp/machine_learning`).

## Core Capabilities

- **Logging Environment, Borehole Corrections, Sensor Confidence & Radial Invasion** (`quick_pp.environment`, `quick_pp.borehole_corrections`, `quick_pp.qaqc`, `quick_pp.conditioning`, `InspectorQaqc.svelte`):
  Depth-zoned **Hole Sections** (`BS`, mud system, mud weight, salinity, $R_{mf}$), onshore/offshore BHT-calibrated P&T profiles, generic `WL`/`LWD` borehole environmental corrections (`*_ENV_CORR` with spine-and-ribs `DRHO` double-compensation protection), perturbation-propagated per-sample uncertainty (`*_SIGMA`) and sensor confidence (`*_QC_CONF`), 2/3-curve step-profile radial invasion inversion ($R_t, R_{xo}, d_i$ & $R_{xo}/R_t$ vs. $R_{mf}/R_w$ movable-HC classification), 1D bounded Huber nuclear-log resolution harmonization (`GR_ENH`, `NPHI_ENH`, `RHOB_ENH`), missing-data imputation, and zone-by-zone stratigraphic quantile normalization.
- **Multi-Mineral, Clastic, Thin-Bed & Continuous $\rho_{fl}$ Petrophysics** (`quick_pp.lithology`, `quick_pp.porosity`, `quick_pp.saturation`, `quick_pp.permeability`):
  Sand-Silt-Clay (modified Kuttan), Carbonate, selectable-log SLSQP multi-mineral solver, and **Thomas-Stieber** laminated/dispersed/structural thin-bed evaluation (`VSH_LAM`, `VSH_DIS`, `VSH_STR`, `PHIT_SAND`, `RSAND_TS`), 3-tier coupled apparent fluid-density ($\rho_{fl}$) solver, adaptive-shoulder coal detection with reversible interval manual overrides, empirical single-well permeability (`Choo`, `Kozeny-Carman`, `Timur`, `Tixier`, `Coates`), Simandoux / Indonesian / Dual-Water / Waxman-Smits / Juhasz / Archie saturation, water-leg $m$ calibration & continuous $m(\phi)$ (`Borai`, `Shell/Lucia`), and multi-criteria reservoir fluid typing (`GAS`, `OIL`, `UNCONFIRMED`, `WATER`).
- **Source Rock & Total Organic Carbon (TOC) Geochemistry** (`quick_pp.source_rock`, `InspectorToc.svelte`):
  Passey et al. (1990) $\Delta\log R$ baseline separation across sonic, density, and neutron logs, automated lean-shale baseline detection, Hood et al. (1975) Level of Organic Metamorphism (`LOM`) maturity conversion from $R_o$ (%), and core pyrolysis TOC calibration.
- **Rock Typing, Power-Law `PERM_TRANS` & Supervised Classification** (`quick_pp.rock_type`, `MultiWellRockTyping.svelte`, `MultiWellPermTransform.svelte`):
  Dual Core (`CPORE`, `CPERM`) and Log (`PHIT`, `PERM`) Discrete Rock Typing (`ROCK_FLAG`) via Amaefule `FZI`/`RQI`, Winland $r_{35}$, and Lucia carbonate `RFN` with GMM, Modified Lorenz, or K-Means cutoff discovery and `ROCK_FLAG = 0` non-reservoir isolation; rock-type partitioned power-law permeability transforms ($k = a_k \cdot \phi^{b_k}$ → `PERM_TRANS`); and standalone supervised `RandomForest` / Gradient Boosting rock-type classification across uncored wells.
- **Multi-Well Contact, SHF, Stick Plot & Formation Pressure Testing (FPT)** (`quick_pp.saturation`, `quick_pp.fluid_type`, `MultiWellShf.svelte`, `MultiWellFpt.svelte`, `StickPlotCanvas.svelte`):
  Multi-well Leverett-$J$, Skelt-Harrison, and Cuddy Saturation-Height Models ($S_{w,\text{SHF}}$), global and per-zone Free Water Level (FWL) optimization, interactive `TVDSS` Stick Plot structural/stratigraphic cross-sections (`GR`/`COAL` tracks & fluid limits `GUT`/`GDT`/`GOC`/`GWC`, `OUT`/`ODT`/`OWC`, `WUT`/`FWL`), and zone-filtered MDT/RFT pressure gradient regression on true `TVDSS` (`FormationPressureAnalyzer` with OLS/Huber/RANSAC, automatic mobility/supercharge flagging, in-situ fluid density, excess pressure, and covariance $\pm 1\sigma$ contacts).
- **Special Core Analysis (SCAL), Core Calibration & Multi-Log Grain Density** (`quick_pp.scal`, `quick_pp.core_analysis`, `InspectorScal.svelte`, `MultiWellSwirr.svelte`):
  Laboratory-to-reservoir core plug corrections (hydrostatic/uniaxial/log-integrated overburden compaction, Klinkenberg gas slippage, and Dean-Stark clay-bound water reconciliation preserving `CPORE_AMB` and `CPERM_AIR` lineage), interactive core-to-log depth matching, multi-log Huber/ridge core grain-density ($\rho_{ma} \to \text{RHOM}$) calibration, explicit fluid-system lab-to-reservoir $P_c$ scaling, PRT Pore-Throat Size Distributions (`plot_ptsd_by_prt`), field-wide `Swirr` fitting, **Corey** / **Brooks-Corey** / **LET** relative permeability, and Eclipse/Intersect `SWOF` table generation (`generate_swof_table`).
- **Probabilistic Volumetrics, Cutoff Sensitivity & Log Blocking** (`quick_pp.ressum`, `quick_pp.uncertainty`, `quick_pp.upscaling`, `MultiWellVolumetrics.svelte`):
  PRMS 95% HCPV empirical cutoff sensitivity sweeps (`cutoffs_analysis`), Iman-Conover correlated Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`) for STOIIP/GIIP (`AREA` & `GRV` modes, Log-Average vs. 1D Hypsometric SHF integration, Tornado ranking), log-level CPI uncertainty propagation (`P10`/`P50`/`P90` curves utilizing `*_SIGMA`), and Cardwell-Parsons (1945) power-law ($\omega \in [-1, 1]$) cellular log blocking (`upscale_logs`, `block_logs`).
- **High-Throughput Multi-Format Log & Ancillary Import Engine** (`quick_pp.las_handler`, `BulkImporterModal.svelte`, `BulkAncillaryImporter.svelte`):
  Fast two-stage hybrid LAS 2.0/3.0, DLIS and LIS79 (`dlisio`), CSV, and Parquet log importer with automated mnemonic alias mapping, unit standardization, `RAW_*` source preservation, interval-bounded multi-run splicing & depth regularization, and bulk ancillary ingestion (Hole Sections, Formation Tops, Fluid Contacts, Minimum-Curvature Surveys, MDT/RFT Pressure & Mobility Tests, Core RCA plugs, and SCAL $P_c$ & RelPerm tables).
- **Automated FDP & Single-Well Report Generation** (`quick_pp/app/backend/services/report.py`, `quick_pp.plotter.report_plots`, `InspectorReport.svelte`, `ReportPreviewCanvas.svelte`):
  10-section Field Development Plan (`FDP_FULL`), Single-Well CPI (`SINGLE_WELL_CPI`), `QAQC_SUMMARY`, and `RESSUM_ONLY` report generator with Borehole Env/Invasion QC & `*_SIGMA`/`*_QC_CONF` audits, automated subsurface QA/QC gate checks (`CHK-01`+) & anomaly remediation logs, publication-grade headless Matplotlib figures, SSOT-grounded AI narrative synthesis (`executive`, `technical`, `concise`, `audit`), and print-ready HTML / multi-page A4 PDF export.
- **Bi-Directional Petrophysical Handover Package Builder & Importer** (`quick_pp.handover`, `HandoverImporter.svelte`):
  Zero-friction `.zip` / `.qpp` handover archive export & persistent project re-hydration preserving LAS 2.0 logs (`~Well` headers + `~Parameter` CPI settings), Hole Sections, Formation Tops, Fluid Contacts & Perforation Intervals (`TVD` & `TVDSS`), Deviation Surveys, Core RCA & MDT LAS point-sets, SCAL tables, Eclipse `SWOF` `.inc` decks, PRMS/SEC Zonal Reservoir Summaries, Multi-Tab Master Excel Workbooks (`.xlsx`), and automated SLB Techlog (`import_to_techlog.py`) & Emerson Geolog (`import_to_geolog.tcl`) loader scripts.
- **Rock Physics & Gassmann Fluid Substitution** (`quick_pp.rock_physics`):
  Elastic moduli, shear-wave velocity ($V_s$) prediction, Gassmann fluid substitution, and seismic rock physics templates.

## Goals of this README
- Give developers and users the practical steps to run the package and web application locally (backend, frontend, and optional Docker services).

## Project Components
- **Core Package (`quick_pp/`)**: Petrophysics, SCAL, uncertainty, upscaling, rock typing, QA/QC, multi-format LAS/DLIS handling (`las_handler.py`), handover package builder (`handover.py`), report plot exporters (`plotter/report_plots.py`), and rock physics modules exported directly in `quick_pp`.
- **Backend (`quick_pp/app/backend`)**: FastAPI application exposing REST endpoints for multi-format log import (`/import/logs`), bulk ancillary & SCAL ingestion (`/ancillary`), QA/QC & stratigraphic normalization, lithology (Sand-Silt-Clay, Multi-Mineral, Thomas-Stieber), porosity, saturation & FWL optimization, rock typing, SCAL & relative permeability, permeability transforms, SHF, multi-well Stick Plots, probabilistic uncertainty, log blocking/upscaling, automated FDP/CPI report generation (`/projects/{project_id}/reports`), and petrophysical handover `.zip` export (`/projects/{project_id}/export`).
- **Frontend (`quick_pp/app/frontend`)**: SvelteKit 5 UI providing interactive Single-Well and Multi-Well petrophysical workflows, core-to-log depth matching, parity crossplots, Stick Plot cross-sections, report workbench, and Canvas/Plotly visualizations.
- **Docker (`quick_pp/app/docker`)**: Compose assets to run backend + Postgres + frontend for development or production deployment.
- **CLI (`quick_pp/cli.py`)**: `quick_pp` CLI wrapper that starts services (`quick_pp app`, `quick_pp backend`, `quick_pp frontend`, `quick_pp docker`, `quick_pp mcp`), runs training, prediction, and deployment tasks.
- **Machine Learning (`quick_pp/machine_learning`)**: Training/prediction pipelines, Physics-Informed Conformal Prediction, and MLflow integration.

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


## ☕ Support & Sponsorship

`quick_pp` is an independently maintained open-source project. If this library has saved you time in your research, coursework, or subsurface workflows, please consider:
* Starring ⭐ the repository to help other geoscientists discover it.
* Citing the package in your academic papers or technical posters.
* [Sponsoring the project on GitHub](https://github.com/sponsors/imranfadhil) to support ongoing development and new tutorials.


## 🤝 Contributing

Contributions are warmly welcome! Whether you are a petrophysicist fixing an equation, adding a tutorial notebook, or improving documentation, your help makes `quick_pp` better for everyone.

To keep the library fast, lightweight, and maintainable, please keep these guidelines in mind:
* **Discuss first**: Before building large new features or proposing major architectural changes, please [open an Issue](https://github.com/imranfadhil/quick_pp/issues) or [Discussion](https://github.com/imranfadhil/quick_pp/discussions) so we can ensure it aligns with `quick_pp`'s lightweight design philosophy.
* **Keep dependencies minimal**: Avoid adding new external packages when standard library or existing core dependencies (`numpy`, `scipy`, `pandas`) can achieve the result.
* **Standards & Tests**: Ensure all unit tests pass (`pytest -q`), code is formatted with Black & Ruff (`ruff check quick_pp tests`), and public functions are typed (`mypy quick_pp`).

See our comprehensive [CONTRIBUTING.md](CONTRIBUTING.md) guide for local environment setup (`uv pip install -e ".[dev]"`), coding standards, and PR workflows.

## ✨ Contributors

Thanks to all the wonderful people who contribute to `quick_pp`!

<a href="https://github.com/imranfadhil/quick_pp/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=imranfadhil/quick_pp" alt="Contributors" />
</a>

## 📖 Citation

If you use `quick_pp` in your academic research, coursework, publications, or technical evaluations, please cite it using the metadata in [CITATION.cff](CITATION.cff) or the BibTeX entry below:

```bibtex
@software{quick_pp2026,
  author       = {Fadhil, Imran},
  title        = {{quick\_pp: Practical Open-Source Petrophysics, Special Core Analysis (SCAL), and Subsurface Machine Learning}},
  year         = {2026},
  publisher    = {GitHub},
  journal      = {GitHub repository},
  howpublished = {\url{https://github.com/imranfadhil/quick_pp}},
  version      = {0.3.8}
}
```
