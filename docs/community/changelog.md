# Changelog

## 0.3.3 (2026)
- Added **Formation Pressure Testing (FPT) Fluid Gradient & Contact Analysis** (`quick_pp/fluid_type.py`, `quick_pp/app/backend/services/fpt.py`, `MultiWellFpt.svelte`, `tests/test_fluid_type_fpt.py`) supporting multi-well pressure-depth regression (OLS, Huber, RANSAC), automatic mobility (`TIGHT`) and supercharge residual quality flagging, unsupervised fluid phase clustering, in-situ fluid density conversion (`gradient_to_density`, `density_to_gradient`), Monte Carlo P10/P50/P90 fluid contact intersections (`GOC`, `OWC`/`FWL`, `GWC`), excess pressure profiling, per-zone analysis, and one-click TVDSS-to-MD contact writing into well fluid contact tables.
- Added **Multi-Well Irreducible Water Saturation (`Swirr`) Module** (`quick_pp/saturation.py`, `MultiWellSwirr.svelte`, `InspectorPerm.svelte`) with multi-well `Swirr` correlation fitting (`fit_swirr_correlation`) and continuous curve prediction using **RQI (`swirr_rqi`)**, **Semi-log Permeability (`swirr_semilog_k`)**, and **Timur (`swirr_timur`)** models, plus direct `SWIRR` integration in Single-Well Timur, Coates, and Tixier permeability calculations.
- Added **Standalone Multi-Well Rock Type Classifier & Continuous Rock Flagging** (`quick_pp/rock_type.py`, `MultiWellRockTyping.svelte`, `InspectorRockTyping.svelte`) to train, cross-validate, cache, and apply supervised `RandomForestClassifier` rock-type models (`train_and_validate_rock_type_classifier`, `predict_with_saved_rt_classifier`) using raw logs (`GR`, `NPHI`, `RHOB`, `RT`, `DTC`, `PEF`) and calculated petrophysical curves (`PHIT`, `VSAND`, `VSHALE`, `VSILT`, `VCLAY`, `VDL`), with core-vs-predicted rock flag accuracy & confusion matrix evaluation (`compute_rock_flag_comparison`, `ROCK_FLAG_CORE`, `ROCK_FLAG_PRED`).
- Added **Core-to-Log Depth Match Workbench & Core Correction Lineage** (`CoreDepthShiftPanel.svelte`, `wellData.svelte.ts`, `quick_pp/scal.py`) with interactive piecewise linear shift tables (`RAW_DEPTH`, `SHIFT_M` -> `DEPTH_SHIFTED`), cross-correlation **Auto-Suggest Shift** against `PHIT` or `RHOB`, ambient vs. reservoir-corrected core storage (`CPORE_AMB` / `CPERM_AIR` alongside overburden- and Klinkenberg-corrected `CPORE` / `CPERM`), original vs. shifted core point ghost markers on log tracks and crossplots, and database persistence to `core_samples`.
- Added **1:1 Estimation vs. Core Parity Crossplot & Curve Comparison** (`CrossplotCanvas.svelte`, `InspectorLithoPoro.svelte`, `InspectorPerm.svelte`) to evaluate `PHIT` vs. `CPORE` and log10 `PERM` vs. `CPERM` against a 1:1 parity line with $\pm 0.03\text{–}0.05\text{ v/v}$ porosity and $\pm 0.5\text{–}1.0$ decade permeability envelopes, live error metrics (`N`, `RMSE`, `MAE`, $R^2$, `Bias`), and multi-curve comparison overlays (`PHIT_COMP`, `PHIE_COMP`, `PERM_COMP`).
- Added **Automated FDP & Single-Well Report Generation** (`quick_pp/app/backend/services/report.py`, `quick_pp/plotter/report_plots.py`, `InspectorReport.svelte`, `ReportPreviewCanvas.svelte`) supporting 10-section Field Development Plan (`FDP_FULL`) and streamlined Single-Well CPI (`SINGLE_WELL_CPI`) reports, headless high-DPI Matplotlib figures, grounded AI narrative generation & enhancement (`executive`, `technical`, `concise`, `audit`), MCP report tools, and print-ready HTML / multi-page A4 PDF compilation.
- Added **Petrophysical Handover Package Builder** (`quick_pp/handover.py`, `quick_pp/las_handler.py`, `/projects/{project_id}/export`, `tests/test_handover.py`) generating structured `.zip` handover archives for SLB Techlog (`import_to_techlog.py`) and Emerson Geolog (`import_to_geolog.tcl`) with populated LAS 2.0 `~Well` headers (`EKB`, `EDF`, `EGL`, `WDEP`, coordinates), continuous `TVDSS`/`SWIRR`/`ROCK_FLAG_CORE`/`ROCK_FLAG_PRED`/`VDL` and `P10`/`P50`/`P90` uncertainty curves, Formation Tops, Fluid Contacts & Perforation Intervals (with interpolated `TVD` and `TVDSS`), Well Headers & Datum summary, Deviation Surveys, Core RCA point-sets (`.csv` and `*_RCA.las` with `RAW_DEPTH`, `SHIFT_M`, `CPORE_AMB`, `CPERM_AIR`), MDT/RFT Pressure Tests (`.csv` and `*_MDT.las`), SCAL Capillary Pressure & Relative Permeability tables, Eclipse `SWOF` `.inc` decks, PRMS/SEC Zonal Reservoir Summaries, Project Parameters manifest, and Multi-Tab Master Excel Workbooks (`.xlsx`).
- Added **High-Throughput Multi-Format Log & Ancillary Import Engine** (`quick_pp/las_handler.py`, `quick_pp/app/backend/services/las_handler.py`, `quick_pp/app/backend/services/ancillary.py`, `BulkImporterModal.svelte`, `BulkAncillaryImporter.svelte`, `tests/test_import_engine.py`) with two-stage hybrid LAS 2.0/3.0, DLIS (`dlisio`), CSV, and Parquet ingestion, mnemonic alias mapping, unit standardization, multi-run splicing & depth regularization (preserving raw missing gaps without default interpolation), and bulk ancillary/SCAL/perforation table import.
- Added **Multi-Well Stick Plot Visualization** (`StickPlotCanvas.svelte`, `POST /workspace/stick_plot`) for structural (TVD/TVDSS/MD) and stratigraphic (flattened on datum or formation top) multi-well cross-sections with customizable tracks, lithology/fluid fills, and formation top correlation lines.
- Added **Multi-Well Data QA/QC & Conditioning Module** (`MultiWellQaqc.svelte`, `POST /workspace/multi_well_qaqc/*`) for field-wide curve distributions, automated washout/badhole flagging, environmental corrections, and despiking/smoothing conditioning across project wells.
- Added **Thomas-Stieber Laminated Sand-Shale Evaluation** (`POST /workspace/wells/{well_name}/thomas_stieber`, `InspectorLithoPoro.svelte`, `CrossplotCanvas.svelte`, `LogTrackCanvas.svelte`) with interactive $\phi_t$ vs. $V_{\text{sh}}$ ternary diagram (`Clean Sand`, `Pure Shale`, `100% Dispersed`, `100% Structural` apexes) and continuous `VSH_LAM`, `VSH_DISP`, `VSH_STR`, `PHIT_TS`, `PHIE_TS`, and `PHISAND_TS` curves.
- Enhanced **Perforation Intervals & Strict TVD / TVDSS Resolution** (`Well.add_perforations`, `_resolve_well_tvd_array`, `LogTrackCanvas.svelte`) to strictly distinguish True Vertical Depth (`TVD`) from True Vertical Depth Sub-Sea (`TVDSS = TVD - datum_elevation`) across logs, surveys, tops, contacts, perforations, core samples, and pressure tests.
- Fixed project deletion cascade integrity (`models.py`, `database.py`, `db_connector.py`) to cleanly remove child records across both SQLite (`PRAGMA foreign_keys=ON`) and PostgreSQL.

## 0.3.2 (2026)
- Fixed frontend API base URL resolution (`resolveApiBase` in `client.ts`) when running the standalone production server on port `5469` (`quick_pp app` and `quick_pp frontend`), routing requests directly to the FastAPI backend on port `6312`.
- Fixed `quick_pp docker up` container volume mounts (`docker-compose.yaml`) to mount the complete `quick_pp` package (`core_analysis.py`, `saturation.py`, `scal.py`, `uncertainty.py`, `upscaling.py`, etc.) and use the pre-built frontend bundle (`build/index.js`).
- Improved `DBConnector` and FastAPI startup (`main.py`) to automatically create parent directories for any file-backed `sqlite:///` database URL and ensure schema initialization completes before bootstrapping the default administrator.
- Updated `seed_demo.py` so non-demo logins (`QPP_DEMO_MODE=false`) prompt for initial password and QR code MFA setup instead of using an un-scanned random TOTP secret.
- Fixed CLI edge cases across `quick_pp app --open`, `quick_pp model-deployment`, and `quick_pp docker`, added automated Docker Hub image publishing to `scripts/release.py`, and added a comprehensive CLI unit test suite (`tests/test_cli.py`).

## 0.3.1 (2026)
- Added **Multi-Well Conformal ML Model Training** (`MultiWellMlTrainer.svelte`, `train_multi_well_conformal_model`) to train, evaluate, and persist custom Physics-Informed Machine Learning (PIML) conformal prediction suites across project wells for Single-Well Lithology & Porosity inference.
- Added **Automated Badhole Flagging & Lithology Mitigation** (`badhole_flagging` in `quick_pp.qaqc`, `InspectorQaqc.svelte`, `InspectorLithoPoro.svelte`) with caliper, density correction (`DRHO`), and density/neutron washout detection, log track shading, and badhole mitigation in Single-Well Lithology & Porosity evaluation.
- Separated **Compute (Preview) and Save (Persist)** workflows across Single-Well and Multi-Well workbench modules (`MultiWellRockTyping`, `MultiWellPermTransform`, `MultiWellShf`, `MultiWellMlTrainer`, and Single-Well inspectors) to allow in-memory parameter tuning and visual inspection before committing curves or project parameters to the database.
- Improved workbench **ParameterSection Accordion UI** for a cleaner layout and streamlined section toggling across Single-Well and Multi-Well views.
- Fixed packaged frontend production build configuration and added wheel entrypoint verification (`quick_pp/app/frontend/build/index.js`) to `scripts/release.py`.

## 0.3.0 (2026)
- Added full-stack **SvelteKit 5 Subsurface Petrophysical Workbench** (`quick_pp/app/frontend`) with Single-Well and Multi-Well Field evaluation scopes, synchronized multi-track log canvas, interactive crossplots, and in-place curve editing.
- Added **Multi-Well Saturation-Height Function (SHF)** SCAL capillary pressure curve fitting for **Leverett-$J$** (`auto_j_params`), **Skelt-Harrison** (`auto_skelt_harrison_params`), and **Cuddy $BVW$** (`auto_cuddy_params`), alongside FZI rock typing and core-log permeability transforms.
- Added **Per-Zone Free Water Level (`FWL`) Optimization** (`solve_per_zone=True`, `fwl_by_zone`, `zone_metrics`) for reconciling compartmentalized contacts across reservoir zones.
- Added **Log-Level CPI Uncertainty Propagation (`P10`/`P50`/`P90`)** and stochastic reservoir summary tables directly in the Single-Well Reservoir Summary inspector.
- Added **Cognito-style User Access Management (UAM)** with invite-only onboarding, mandatory TOTP Multi-Factor Authentication (`MFA`), role-based project access control (`admin`/`editor`/`viewer`), and isolated **Live Demo Mode**.
- Added native **Petrophysical AI Assistant & FastAPI MCP Server** (`quick_pp mcp`, `/api/v1/assistant/chat`), removing legacy Langflow dependencies.
- Vectorized multi-mineral lithology optimization, added automated coal flagging, and improved neutron-density hydrocarbon corrections.

## 0.2.106 (2026)
- Added probabilistic uncertainty & Monte Carlo engine (`quick_pp.uncertainty`) with vectorized Latin Hypercube Sampling (`run_cpi_uncertainty`, `mc_volumetric_uncertainty`).
- Added Special Core Analysis & Relative Permeability module (`quick_pp.scal`) with lab-to-reservoir overburden, Klinkenberg, and Dean-Stark corrections, Corey/Brooks-Corey/LET models, and Eclipse `SWOF` table export.
- Added static-to-dynamic geomodel log blocking & upscaling (`quick_pp.upscaling`) with pore-volume weighted saturation and Cardwell-Parsons permeability bounds.
- Added multi-well Free Water Level (`FWL`), Oil-Water Contact (`OWC`), and transition zone optimizer (`ContactOptimizer`, `optimize_fwl`).
- Added multi-well stratigraphic normalization and environmental corrections (`quick_pp.qaqc`).
- Added Physics-Informed Machine Learning (PIML) feature engineering and locally weighted conformal prediction (`ConformalRegressor`, `ConformalMultiOutputRegressor`, `EnforceMineralSum`).
- Refactored `pyproject.toml` into modular extras (`core`, `ml`, `app`, `all`) and migrated documentation to MkDocs-Material.

## 0.2.2 (2025-05-23)
- Added `qpp_assistant` module.
- Updated requirements to use Python 3.11.
- Updated README and documentation.

## 0.2.1 (2025-05-20)
- Refactored the API module.
- Added ML modelling module.
- Added CLI using `click`.
- Refactored the well log plotter module.

## 0.1.18 (2024-11-14)
- Added stick plot.
- Updated rock typing module.
- Added Skelt-Harrison saturation function.
- Added sensitivity analysis in `ressum` module.

## 0.1.13 (2024-10-04)
- Added hydrocarbon typing based on gas composition.
- Added geomechanics module.
- Added quick compare function.
- Updated `estimate_vsh_gr`, `porosity`, and `qaqc` modules.

## 0.1.11 (2024-07-17)
- Updated requirements to fix security risks.

## 0.1.5 (2024-06-05)
- Added core calibration module.

## 0.1.0 (2023-10-08)
- First release on PyPI.
