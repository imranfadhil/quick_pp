# quick_pp: Integrated Petrophysical Evaluation & Subsurface Modeling

[![PyPI version](https://img.shields.io/pypi/v/quick_pp.svg)](https://pypi.org/project/quick-pp/)
[![Documentation Status](https://readthedocs.org/projects/quick-pp/badge/?version=latest)](https://quick-pp.readthedocs.io/en/latest/?version=latest)

**`quick_pp`** is a comprehensive Python package and full-stack web platform for deterministic and probabilistic petrophysical evaluation, Special Core Analysis (SCAL) & relative permeability modeling, multi-well Free Water Level (FWL) & Formation Pressure Testing (FPT) contact analysis, stratigraphic log normalization, core-log calibration & depth matching, automated FDP report generation, and static-to-dynamic geomodeling upscaling.

---

## Core Capabilities

- **Multi-Mineral, Clastic & Thomas-Stieber Petrophysics** ([`quick_pp.lithology`](api/lithology.md), [`quick_pp.porosity`](api/porosity.md), [`quick_pp.saturation`](api/saturation.md), [`quick_pp.permeability`](api/permeability.md)):
  Sand-Silt-Clay (modified Kuttan), Carbonate, optimization-based Multi-Mineral solver, and Thomas-Stieber laminated sand-shale evaluation (`VSH_LAM`, `VSH_DISP`, `VSH_STR`, `PHIT_TS`, `PHIE_TS`, `PHISAND_TS`) with Simandoux, Indonesian, Dual-Water, Waxman-Smits, Juhasz, and Archie saturation models.
- **Multi-Well Contact, Swirr & Formation Pressure Testing (FPT)** ([`quick_pp.saturation`](api/saturation.md), [`quick_pp.fluid_type`](api/rock_type.md)):
  Simultaneous multi-well Free Water Level ($\text{FWL}$), Oil-Water Contact ($\text{OWC}$), and Top of Transition Zone ($\text{TTZ}$) solver reconciling resistivity-derived $S_w$, capillary pressure Saturation-Height Models ($S_{w,\text{SHM}}$ via Leverett-$J$, Skelt-Harrison, and Cuddy), multi-well irreducible water saturation (`Swirr` via RQI, Semi-log $k$, and Timur correlations), and wireline formation tester (MDT/RFT) pressure gradient regression (`FormationPressureAnalyzer` with OLS/Huber/RANSAC, automatic supercharge/tight flagging, in-situ fluid density, and Monte Carlo P10/P50/P90 GOC/OWC/GWC contact intersections).
- **Rock Typing & Supervised Rock Flag Classification** ([`quick_pp.rock_type`](api/rock_type.md)):
  Hydraulic Flow Unit (`FZI`, `RQI`, `GHE`), Winland $R_{35}$, and Pittman pore-throat rock typing, plus standalone multi-well `RandomForestClassifier` rock-type training, cross-validation, and uncored well prediction (`train_and_validate_rock_type_classifier`, `predict_with_saved_rt_classifier`) using both raw logs and calculated petrophysical curves (`PHIT`, `VSAND`, `VSHALE`, `VSILT`, `VCLAY`, `VDL`).
- **Special Core Analysis (SCAL), Core Calibration & Depth Matching** ([`quick_pp.scal`](api/scal.md), [`quick_pp.core_analysis`](api/core_analysis.md)):
  Laboratory-to-reservoir core plug corrections (hydrostatic/uniaxial overburden compaction, Klinkenberg gas slippage, and Dean-Stark clay-bound water reconciliation preserving ambient `CPORE_AMB`/`CPERM_AIR` lineage), interactive core-to-log depth shifting (piecewise linear anchor tables & automated cross-correlation shift suggestion), 1:1 Estimation vs. Core parity crossplots (`PHIT` vs `CPORE`, `PERM` vs `CPERM`), capillary pressure scaling, two-phase relative permeability models (**Corey**, **Brooks-Corey**, and **LET**), automated SCAL curve fitting (`fit_relperm`), and Eclipse/Intersect `SWOF` table generation (`generate_swof_table`).
- **Probabilistic Uncertainty & Monte Carlo Engine** ([`quick_pp.uncertainty`](api/uncertainty.md)):
  Vectorized Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`) for volumetric hydrocarbon in-place uncertainty (`mc_volumetric_uncertainty`), log-level CPI uncertainty propagation (`run_cpi_uncertainty` producing `P10`/`P50`/`P90` curves), and stochastic reservoir summaries (`calc_stochastic_reservoir_summary`).
- **Log Blocking & Static-to-Dynamic Upscaling** ([`quick_pp.upscaling`](api/upscaling.md)):
  Petrophysically rigorous blocking (`upscale_logs`, `block_logs`) onto cellular geomodel grids with thickness-weighted arithmetic porosity/shale averaging, pore-volume weighted saturation averaging (preserving `BVHC`), directional & Cardwell-Parsons (1945) power-law permeability bounds ($k_h, k_{\text{eff}}, k_v, k_v/k_h$), and majority-vote facies upscaling.
- **Multi-Well QA/QC, Stratigraphic Normalization & Stick Plots** ([`quick_pp.qaqc`](api/qaqc.md)):
  Zone-by-zone multi-well quantile and histogram log normalization (`stratigraphic_normalization` / `normalize_by_zone`), borehole environmental and invasion corrections (`apply_environmental_corrections`), neutron-density hydrocarbon corrections, automated washout/bad-hole flagging (`badhole_flagging`), and multi-well structural/stratigraphic Stick Plot cross-sections.
- **High-Throughput Multi-Format Log Import, Handover Packages & Automated FDP Reports** ([`quick_pp.las_handler`](api/plotter.md), [`quick_pp.handover`](api/plotter.md), [`quick_pp.plotter.report_plots`](api/plotter.md)):
  Fast hybrid LAS 2.0/3.0, DLIS (`dlisio`), CSV, and Parquet ingestion with mnemonic mapping, unit standardization, and multi-run splicing; zero-friction `.zip` handover package generation (`build_handover_archive_bytes`) with LAS 2.0 exports (`~Well` headers + `~Parameter` CPI settings), Formation Tops, Fluid Contacts & Perforations (with `TVD` and `TVDSS`), Core RCA & MDT LAS point-sets, multi-tab Excel workbooks, Eclipse `SWOF` decks, and ready-to-run SLB Techlog (`import_to_techlog.py`) & Emerson Geolog (`import_to_geolog.tcl`) loader scripts; plus automated 10-section FDP & Single-Well CPI report generation with publication-grade figures and A4 PDF / print HTML export.
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
