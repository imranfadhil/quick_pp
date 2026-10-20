# `quick_pp` Example & Showcase Notebooks

This directory contains interactive Jupyter notebooks organized into numbered workflow series (`01` through `04`) and standalone topic examples covering data ingestion, exploratory data analysis, deterministic petrophysics, rock physics, and **Field Development Planning (FDP) & Physics-Informed Machine Learning (PIML)**.

---

## `01` & `02` Series — Data Ingestion & Exploratory Data Analysis (EDA)

- **[`01_data_handler.ipynb`](01_data_handler.ipynb)** — Ingest raw LAS files and core datasets (e.g., `VOLVE` clastic or `COSTA` carbonate fields) into a local SQLite database or `.qppp` project file using `DBConnector` and `Project`.
- **[`02_EDA.ipynb`](02_EDA.ipynb)** — Generate multi-well and single-well profiling reports, cross-well log distribution comparisons, and data completeness/QAQC summaries.

---

## `03` Series — Deterministic Petrophysical Workflows (Carbonate & Clastic)

### Carbonate Evaluation (`03_1` – `03_6`)
- **[`03_1_carbonate_lithology_porosity.ipynb`](03_1_carbonate_lithology_porosity.ipynb)** — Carbonate mineral volumes (`VCALC`, `VDOLO`, `VCLAY`), hydrocarbon density/neutron corrections, and total/effective porosity (`PHIT`, `PHIE`) via `Carbonate`.
- **[`03_2_carbonate_rock_typing_fzi.ipynb`](03_2_carbonate_rock_typing_fzi.ipynb)** — Core-calibrated carbonate rock typing comparing Flow Zone Indicator (`FZI`), Winland $R_{35}$, and Lucia Rock Fabric Number (`RFN`), plus permeability transforms.
- **[`03_3_carbonate_saturation_j_fzi.ipynb`](03_3_carbonate_saturation_j_fzi.ipynb)** — Resistivity water saturation (`archie_saturation`) vs. capillary Leverett-$J$ / `FZI` saturation-height modeling (`leverett_j`, `skelt_harrison`) and Pickett/Buckles crossplots.
- **[`03_4_carbonate_single.ipynb`](03_4_carbonate_single.ipynb)** — End-to-end deterministic single-well carbonate CPI workflow (lithology, porosity, rock typing, permeability, water saturation, and interactive `plotly_log` display).
- **[`03_5_carbonate_batch.ipynb`](03_5_carbonate_batch.ipynb)** — Multi-well batch carbonate interpretation, automated reservoir summaries (`calc_reservoir_summary`), and multi-well `quick_qc` diagnostics.
- **[`03_6_carbonate_volume_uncertainty.ipynb`](03_6_carbonate_volume_uncertainty.ipynb)** — Multi-well fluid contact stick plots (`stick_plot`), zone statistical summaries (`quick_compare`, `extract_quick_stats`), cutoff sensitivity (`cutoffs_analysis`), and Monte Carlo volumetric estimation.

### Clastic Evaluation (`03_qpp_*`)
- **[`03_qpp_ssc_single.ipynb`](03_qpp_ssc_single.ipynb)** — Single-well clastic evaluation using the modified Kuttan **Sand-Silt-Clay (`SandSiltClay`)** ternary model, Choo permeability, and Waxman-Smits / Dual-Water saturation.
- **[`03_qpp_ssc_batch.ipynb`](03_qpp_ssc_batch.ipynb)** — Multi-well batch execution of the `SandSiltClay` interpretation pipeline with field-wide reservoir summary tables and QC plots.
- **[`03_qpp_ss_single.ipynb`](03_qpp_ss_single.ipynb)** — Single-well binary **Sand-Shale (`SandShale`)** deterministic interpretation workflow on Volve (`15-9-19-A`) well logs.
- **[`03_qpp_ts_single.ipynb`](03_qpp_ts_single.ipynb)** — Laminated **Thin-Bed (`ThinBeds`)** sand-shale evaluation with macroscopic anisotropy resistivity modeling and thin-bed saturation corrections.

---

## `04` Series — FDP & Physics-Informed ML (PIML) Showcases

- **[`04_fdp_and_piml_showcase.ipynb`](04_fdp_and_piml_showcase.ipynb)** — **End-to-End Master Showcase** integrating multi-well QA/QC & stratigraphic normalization (`apply_environmental_corrections`, `normalize_by_zone`), probabilistic CPI (`run_cpi_uncertainty`), multi-well FWL optimization (`optimize_fwl`), SCAL (`CoreyRelPerm`), log blocking (`block_logs`), and conformal PIML (`ConformalRegressor`, `ConformalMultiOutputRegressor`).
- **[`04_1_probabilistic_cpi_and_fwl_optimization.ipynb`](04_1_probabilistic_cpi_and_fwl_optimization.ipynb)** — Latin Hypercube parameter sampling (`sample_lhs_parameters`), log-level CPI uncertainty (`run_cpi_uncertainty`), stochastic STOIIP volumetrics (`calc_stochastic_reservoir_summary`), MDT pressure intersection (`estimate_mdt_fwl`), and multi-well FWL/OWC/TTZ optimization (`ContactOptimizer`).
- **[`04_2_scal_relperm_and_log_blocking.ipynb`](04_2_scal_relperm_and_log_blocking.ipynb)** — Overburden & Kozeny-Carman core corrections (`overburden_poroperm_correction`), MICP lab-to-reservoir scaling (`pc_lab_to_reservoir`), `CoreyRelPerm` curve fitting & Eclipse `SWOF` export (`generate_swof_table`), and static-to-dynamic cellular log blocking (`block_logs`, `expand_blocked_logs`).
- **[`04_3_piml_and_conformal_prediction.ipynb`](04_3_piml_and_conformal_prediction.ipynb)** — Domain rock-physics feature engineering (`generate_fe_features`), locally weighted split & multi-output conformal prediction (`ConformalRegressor`, `ConformalMultiOutputRegressor`), and volumetric closure constraints (`EnforceMineralSum`, `enforce_physical_constraints`, `physics_informed_loss`).

---

## Standalone Topic, Rock Physics & Visualization Notebooks

### Lithology, Fluid Contacts & REST API Examples
- **[`sand_silt_clay-examples.ipynb`](sand_silt_clay-examples.ipynb)** — Standalone `SandSiltClay` model, hydrocarbon correction, porosity, permeability, and Waxman-Smits saturation without a database wrapper.
- **[`sand_shale-examples.ipynb`](sand_shale-examples.ipynb)** — Standalone binary `SandShale` model and neutron-density crossplot porosity.
- **[`multi_mineral-examples.ipynb`](multi_mineral-examples.ipynb)** — Optimization-based multi-mineral inversion (`MultiMineral`) for clastic and carbonate (`COSTA`) mineral assemblages.
- **[`fluid_contact-example.ipynb`](fluid_contact-example.ipynb)** — Wireline formation tester (`MDT`/`RFT`) pressure-depth gradient fitting (`fit_pressure_gradient`, `fluid_contact_plot`) for `GOC` and `OWC` identification.
- **[`API_sand_silt_clay_examples.ipynb`](API_sand_silt_clay_examples.ipynb)** — Calling the `quick_pp` FastAPI backend endpoints (`/api/v1/...`) over HTTP for petrophysical evaluation.

### Rock Physics & Geomechanics (`rp_*`)
- **[`rp_geomech.ipynb`](rp_geomech.ipynb)** — Mohr-Coulomb failure envelopes (`plot_mohrs_circle`), rock strength (`estimate_rock_strength`), dynamic-to-static elastic moduli, and in-situ stress profiles.
- **[`rp_geophysics.ipynb`](rp_geophysics.ipynb)** — Geophysical well-tie & seismic utilities (`segysak` SEG-Y inspection, acoustic impedance, reflection coefficients, wavelets, and synthetic seismograms).
- **[`rp_rpm.ipynb`](rp_rpm.ipynb)** — Gassmann fluid substitution, Voigt-Reuss-Hill / Hashin-Shtrikman bounds, and Rock Physics Templates (`RPT`: $V_p/V_s$ vs. Acoustic Impedance).

### Interactive Plotting Utilities
- **[`well_plotter.ipynb`](well_plotter.ipynb)** — Customizing multi-track interactive Plotly well log displays (`plotly_log`, `TRACE_DEFS`, `COLOR_DICT`).
- **[`cross_plotter.ipynb`](cross_plotter.ipynb)** — Interactive Plotly petrophysical crossplots with multi-dimensional color/size encoding and overlays.

