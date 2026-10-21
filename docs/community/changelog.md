# Changelog

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

