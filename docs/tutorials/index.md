# Visual Tutorials (Jupyter Notebooks)

All notebooks in `notebooks/03_*.ipynb` and `notebooks/04_*.ipynb` are rendered directly into this documentation site with their full code, mathematical explanations, and visual plots.

## 04 Series — Field Development Planning (FDP) & Physics-Informed ML

| Tutorial | Highlights |
| :--- | :--- |
| [**04 Master Showcase: FDP & PIML**](04_fdp_and_piml_showcase.ipynb) | End-to-end integration of Latin Hypercube CPI uncertainty, multi-well FWL optimization, SCAL relative permeability & Eclipse `SWOF` export, geomodel log blocking, and conformal ML prediction. |
| [**04.1 Probabilistic CPI & FWL Optimization**](04_1_probabilistic_cpi_and_fwl_optimization.ipynb) | Vectorized LHS CPI uncertainty (`run_cpi_uncertainty`), stochastic STOIIP tornado sensitivity (`mc_volumetric_uncertainty`), and simultaneous multi-well `ContactOptimizer` (`optimize_fwl`). |
| [**04.2 SCAL, RelPerm & Log Blocking**](04_2_scal_relperm_and_log_blocking.ipynb) | Lab-to-reservoir overburden/Klinkenberg/Dean-Stark core corrections, `CoreyRelPerm` & `let_relperm` fitting (`fit_relperm`), Eclipse `SWOF` tables, and static-to-dynamic `block_logs` upscaling. |
| [**04.3 PIML & Conformal Prediction**](04_3_piml_and_conformal_prediction.ipynb) | Physics-informed domain features (`RHOMA_APP`, `AI`, `M_LITH`, `N_LITH`), locally weighted `ConformalRegressor` & `ConformalMultiOutputRegressor`, and `EnforceMineralSum`. |

---

## 03 Series — Deterministic & Carbonate Petrophysics

| Tutorial | Highlights |
| :--- | :--- |
| [**03.1 Carbonate Lithology & Porosity**](03_1_carbonate_lithology_porosity.ipynb) | Neutron-density crossplot & PEF carbonate lithology and porosity interpretation. |
| [**03.2 Carbonate Rock Typing (FZI)**](03_2_carbonate_rock_typing_fzi.ipynb) | Flow Zone Indicator (`calc_fzi`), pore throat radius (`estimate_pore_throat`), and hydraulic rock typing. |
| [**03.3 Carbonate Saturation (J-Function)**](03_3_carbonate_saturation_j_fzi.ipynb) | Capillary pressure Leverett-$J$ water saturation modeling reconciled against Archie $S_w$. |
| [**03.4 Carbonate Single-Well Evaluation**](03_4_carbonate_single.ipynb) | Complete single-well carbonate CPI workflow. |
| [**03.5 Carbonate Multi-Well Batch**](03_5_carbonate_batch.ipynb) | Multi-well batch carbonate processing and reservoir summary generation. |
| [**03.6 Carbonate Volume Uncertainty**](03_6_carbonate_volume_uncertainty.ipynb) | Monte Carlo hydrocarbon pore volume and STOIIP uncertainty analysis. |
| [**Clastic Sand-Silt-Clay (Single Well)**](03_qpp_ssc_single.ipynb) | Modified Kuttan ternary Sand-Silt-Clay evaluation on a single clastic well. |
| [**Clastic Sand-Silt-Clay (Multi-Well Batch)**](03_qpp_ssc_batch.ipynb) | Multi-well batch Sand-Silt-Clay petrophysical evaluation. |
| [**Sand-Shale Evaluation**](03_qpp_ss_single.ipynb) | Deterministic Sand-Shale interpretation workflow. |
| [**Thin-Bed Evaluation**](03_qpp_ts_single.ipynb) | Laminated thin-bed sand-shale analysis (Thomas-Stieber). |

