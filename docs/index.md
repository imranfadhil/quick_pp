# quick_pp: Integrated Petrophysical Evaluation & Subsurface Modeling

[![PyPI version](https://img.shields.io/pypi/v/quick_pp.svg)](https://pypi.org/project/quick-pp/)
[![Documentation Status](https://readthedocs.org/projects/quick-pp/badge/?version=latest)](https://quick-pp.readthedocs.io/en/latest/?version=latest)

**`quick_pp`** is a comprehensive Python package and full-stack web platform for deterministic and probabilistic petrophysical evaluation, unified borehole environmental & radial invasion QC, Special Core Analysis (SCAL) & relative permeability modeling, multi-well Free Water Level (FWL) & Formation Pressure Testing (FPT) contact analysis, stratigraphic log normalization, source-rock TOC geochemistry, automated CPI/FDP report generation with subsurface gate checks, and static-to-dynamic geomodeling upscaling.

---

## Core Capabilities

- **Logging Environment, Borehole Corrections, Sensor Confidence & Radial Invasion** ([`quick_pp.environment`](api/qaqc.md), [`quick_pp.borehole_corrections`](api/qaqc.md), [`quick_pp.qaqc`](api/qaqc.md), [`quick_pp.conditioning`](api/conditioning.md)):
  Depth-zoned **Hole Sections** (`BS`, mud system, mud weight, salinity, $R_{mf}$), onshore/offshore BHT-calibrated P&T profiles, generic `WL`/`LWD` borehole environmental corrections (`*_ENV_CORR` with spine-and-ribs `DRHO` double-compensation protection), perturbation-propagated per-sample uncertainty (`*_SIGMA`) and sensor confidence (`*_QC_CONF`), 2/3-curve step-profile radial invasion inversion ($R_t, R_{xo}, d_i$ & $R_{xo}/R_t$ vs. $R_{mf}/R_w$ movable-HC classification), 1D bounded Huber nuclear-log resolution harmonization (`GR_ENH`, `NPHI_ENH`, `RHOB_ENH`), missing-data imputation, and zone-by-zone stratigraphic quantile normalization.
- **Multi-Mineral, Clastic, Thin-Bed & Continuous $\rho_{fl}$ Petrophysics** ([`quick_pp.lithology`](api/lithology.md), [`quick_pp.porosity`](api/porosity.md), [`quick_pp.saturation`](api/saturation.md), [`quick_pp.permeability`](api/permeability.md)):
  Sand-Silt-Clay, Carbonate, selectable-log SLSQP Multi-Mineral solver, and Thomas-Stieber laminated/dispersed/structural thin-bed evaluation (`VSH_LAM`, `VSH_DIS`, `VSH_STR`, `PHIT_SAND`, `RSAND_TS`), 3-tier coupled apparent fluid-density ($\rho_{fl}$) solver, adaptive-shoulder coal detection with reversible interval manual overrides, empirical single-well permeability (`Choo`, `Kozeny-Carman`, `Timur`, `Tixier`, `Coates`), Archie / Waxman-Smits / Simandoux / Indonesian / Dual-Water saturation, water-leg $m$ calibration & continuous $m(\phi)$ (`Borai`, `Shell/Lucia`), and multi-criteria reservoir fluid typing (`GAS`, `OIL`, `UNCONFIRMED`, `WATER`).
- **Source Rock & Total Organic Carbon (TOC) Geochemistry** ([`quick_pp.source_rock`](api/source_rock.md)):
  Passey et al. (1990) $\Delta\log R$ baseline separation across sonic, density, and neutron logs, automated lean-shale baseline detection, Hood et al. (1975) Level of Organic Metamorphism (`LOM`) maturity conversion from $R_o\,\%$, and core pyrolysis TOC calibration.
- **Rock Typing, Power-Law `PERM_TRANS` & Supervised Classification** ([`quick_pp.rock_type`](api/rock_type.md)):
  Dual Core (`CPORE`, `CPERM`) and Log (`PHIT`, `PERM`) Discrete Rock Typing (`ROCK_FLAG`) via Amaefule `FZI`/`RQI`, Winland $r_{35}$, and Lucia carbonate `RFN` with GMM, Modified Lorenz, or K-Means cutoff discovery and `ROCK_FLAG = 0` non-reservoir isolation; rock-type partitioned power-law permeability transforms ($k = a_k \cdot \phi^{b_k} \to \text{PERM\_TRANS}$); and supervised Random Forest / Gradient Boosting rock-type classification across uncored wells.
- **Multi-Well Contact, SHF, Stick Plot & Formation Pressure Testing (FPT)** ([`quick_pp.saturation`](api/saturation.md), [`quick_pp.fluid_type`](api/rock_type.md)):
  Multi-well Leverett-$J$, Skelt-Harrison, and Cuddy Saturation-Height Functions ($S_{w,\text{SHF}}$), global and per-zone Free Water Level ($\text{FWL}$) optimization, `TVDSS` Stick Plot geological correlation (`GR`/`COAL` tracks & fluid limits `GUT`/`GDT`/`GOC`/`GWC`, `OUT`/`ODT`/`OWC`, `WUT`/`FWL`), and zone-filtered MDT/RFT pressure gradient regression on true `TVDSS` (`FormationPressureAnalyzer` with OLS/Huber/RANSAC, excess pressure, in-situ densities, and covariance $\pm 1\sigma$ contacts).
- **Special Core Analysis (SCAL), Core Calibration & Multi-Log Grain Density** ([`quick_pp.scal`](api/scal.md), [`quick_pp.core_analysis`](api/core_analysis.md)):
  Interactive core-to-log depth shifting, overburden stress (scalar or log-integrated `SV_EST`/`NCS_EST`) and Klinkenberg corrections, multi-log Huber/ridge core grain-density ($\rho_{ma} \to \text{RHOM}$) calibration, explicit fluid-system lab-to-reservoir $P_c$ scaling, PRT Pore-Throat Size Distributions (`plot_ptsd_by_prt`), field-wide $S_{\text{wirr}}$ fitting, **Corey** / **Brooks-Corey** / **LET** relative permeability, and Eclipse `SWOF` deck export.
- **Probabilistic Volumetrics, Cutoff Sensitivity & Log Blocking** ([`quick_pp.ressum`](api/ressum.md), [`quick_pp.uncertainty`](api/uncertainty.md), [`quick_pp.upscaling`](api/upscaling.md)):
  PRMS 95% HCPV empirical cutoff sensitivity sweeps (`cutoffs_analysis`), Iman-Conover correlated Latin Hypercube Monte Carlo STOIIP/GIIP (`AREA` & `GRV` modes, Log-Average vs. 1D Hypsometric SHF integration, Tornado ranking), log-level CPI uncertainty propagation (`P10`/`P50`/`P90` utilizing `*_SIGMA`), and Cardwell-Parsons ($\omega \in [-1, 1]$) cellular log blocking.
- **Multi-Format Log Import, Bi-Directional Handover Packages & Automated CPI/FDP Reports** ([`quick_pp.las_handler`](api/plotter.md), [`quick_pp.handover`](api/plotter.md), [`quick_pp.plotter.report_plots`](api/plotter.md)):
  Hybrid LAS 2.0/3.0, DLIS, LIS79, CSV, and Parquet ingestion with multi-run splicing; bi-directional `.zip` / `.qpp` handover archive export & persistent project import (with SLB Techlog & Emerson Geolog loader scripts, Excel workbooks, LAS point-sets & `SWOF` decks); and automated 10-section `FDP_FULL`, `SINGLE_WELL_CPI`, `QAQC_SUMMARY`, and `RESSUM_ONLY` reports with Borehole Env/Invasion QC audits, automated subsurface gate checks (`CHK-01`+), SSOT-grounded AI commentary, and multi-page A4 PDF export.
- **Physics-Informed ML & Conformal Prediction** ([`quick_pp.machine_learning`](api/machine_learning.md)):
  Domain physics features (`RHOMA_APP`, `AI`, Burke's `M_LITH` & `N_LITH`), locally weighted split/cross-conformal intervals (`ConformalRegressor`, `ConformalMultiOutputRegressor`), and mineral volumetric closure (`EnforceMineralSum`).

---

## Quick Installation

=== "uv (Recommended)"

    ```bash
    # Core petrophysics, SCAL, uncertainty & upscaling
    uv pip install "quick-pp[core]"

    # Machine learning & conformal prediction
    uv pip install "quick-pp[ml]"

    # Full-stack FastAPI backend, MCP server & database layer
    uv pip install "quick-pp[app]"

    # Everything (core + ml + app + Jupyter notebooks)
    uv pip install "quick-pp[all]"
    ```

=== "pip"

    ```bash
    pip install "quick-pp[core]"
    pip install "quick-pp[ml]"
    pip install "quick-pp[app]"
    pip install "quick-pp[all]"
    ```
