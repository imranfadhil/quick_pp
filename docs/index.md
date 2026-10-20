# quick_pp: Integrated Petrophysical Evaluation & Subsurface Modeling

[![PyPI version](https://img.shields.io/pypi/v/quick_pp.svg)](https://pypi.org/project/quick-pp/)
[![Documentation Status](https://readthedocs.org/projects/quick-pp/badge/?version=latest)](https://quick-pp.readthedocs.io/en/latest/?version=latest)

**`quick_pp`** is a comprehensive Python package and full-stack web platform for deterministic and probabilistic petrophysical evaluation, Special Core Analysis (SCAL) & relative permeability modeling, multi-well Free Water Level (FWL) contact optimization, stratigraphic log normalization, rock physics, and static-to-dynamic geomodeling upscaling.

---

## Core Capabilities

- **Multi-Mineral & Deterministic Petrophysics** ([`quick_pp.lithology`](api/lithology.md), [`quick_pp.porosity`](api/porosity.md), [`quick_pp.saturation`](api/saturation.md), [`quick_pp.permeability`](api/permeability.md)):
  Sand-Silt-Clay (modified Kuttan), Carbonate, and multi-mineral inversion solver with Simandoux, Indonesian, Dual-Water, Waxman-Smits, Juhasz, and Archie saturation models.
- **Multi-Well Contact & Transition Zone Optimization** ([`quick_pp.saturation.optimize_fwl`](api/saturation.md)):
  Simultaneous multi-well Free Water Level ($\text{FWL}$), Oil-Water Contact ($\text{OWC}$), and Top of Transition Zone ($\text{TTZ}$) solver reconciling resistivity-derived $S_w$, capillary pressure Saturation-Height Models ($S_{w,\text{SHM}}$ via Leverett-$J$, Skelt-Harrison, and Cuddy), and wireline formation tester (MDT/RFT) pressure gradients.
- **Special Core Analysis (SCAL) & Relative Permeability** ([`quick_pp.scal`](api/scal.md), [`quick_pp.core_analysis`](api/core_analysis.md)):
  Laboratory-to-reservoir core plug corrections (hydrostatic/uniaxial overburden compaction, Klinkenberg gas slippage, and Dean-Stark clay-bound water reconciliation), capillary pressure scaling, two-phase relative permeability models (**Corey**, **Brooks-Corey**, and **LET**), automated SCAL curve fitting (`fit_relperm`), and Eclipse/Intersect `SWOF` table generation (`generate_swof_table`).
- **Probabilistic Uncertainty & Monte Carlo Engine** ([`quick_pp.uncertainty`](api/uncertainty.md)):
  Vectorized Latin Hypercube Sampling (`scipy.stats.qmc.LatinHypercube`) for volumetric hydrocarbon in-place uncertainty (`mc_volumetric_uncertainty`), log-level CPI uncertainty propagation (`run_cpi_uncertainty` producing `P10`/`P50`/`P90` curves), and stochastic reservoir summaries (`calc_stochastic_reservoir_summary`).
- **Log Blocking & Static-to-Dynamic Upscaling** ([`quick_pp.upscaling`](api/upscaling.md)):
  Petrophysically rigorous blocking (`upscale_logs`, `block_logs`) onto cellular geomodel grids with thickness-weighted arithmetic porosity/shale averaging, pore-volume weighted saturation averaging (preserving `BVHC`), directional & Cardwell-Parsons (1945) power-law permeability bounds ($k_h, k_{\text{eff}}, k_v, k_v/k_h$), and majority-vote facies upscaling.
- **Multi-Well QA/QC & Stratigraphic Normalization** ([`quick_pp.qaqc`](api/qaqc.md)):
  Zone-by-zone multi-well quantile and histogram log normalization (`stratigraphic_normalization` / `normalize_by_zone`), borehole environmental and invasion corrections (`apply_environmental_corrections`), neutron-density hydrocarbon corrections, and bad-hole flagging.
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

    # Full-stack FastAPI backend & Celery worker
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

