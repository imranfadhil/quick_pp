# Quickstart & CLI

There are two primary ways to use `quick_pp`: as a Python scientific library or via the `quick_pp` Command-Line Interface (CLI) and web workstation.

## Using `quick_pp` as a Python Library

Core petrophysics, SCAL, uncertainty, upscaling, and conformal ML classes/functions are exposed directly at the top-level `quick_pp` namespace:

```python
import quick_pp as qpp

# 1. Multi-well QA/QC & stratigraphic normalization
df_norm, shifts = qpp.stratigraphic_normalization(
    df, well_col="WELL_NAME", zone_col="ZONE", curves=["GR", "NPHI", "RHOB"]
)

# 2. Probabilistic CPI Monte Carlo uncertainty (P10 / P50 / P90 logs)
cpi_mc = qpp.run_cpi_uncertainty(df_well, n_scenarios=200, seed=42)

# 3. Multi-well Free Water Level (FWL) & transition zone reconciliation
fwl_res = qpp.optimize_fwl(df_multiwell, mdt_df=df_mdt, model="leverett_j")

# 4. SCAL relative permeability & Eclipse SWOF export
relperm = qpp.CoreyRelPerm(swirr=0.15, sorw=0.22, nw=3.2, no=2.4, krw_max=0.35, kro_max=0.85)
swof_df, swof_str = qpp.generate_swof_table(relperm, pc_entry=1.8, lam=1.6)

# 5. Static-to-dynamic geomodel log blocking
df_blocked = qpp.block_logs(df_well, n_layers=20, kh_anisotropy_ratio=0.1)
```

---

## Command-Line Interface (CLI)

When installed with `quick-pp[app]` or `quick-pp[all]`, the `quick_pp` CLI manages the full-stack web platform and MLflow pipelines:

```bash
# Start both FastAPI backend (:6312) and SvelteKit frontend (:5469)
quick_pp app

# Start FastAPI backend only with auto-reload
quick_pp backend --debug

# Start SvelteKit frontend in development mode (:5173)
quick_pp frontend --dev

# Launch local MLflow tracking UI (:5015)
quick_pp mlflow_server

# Show all CLI commands
quick_pp --help
```

---

## Citation & Publication

An article demonstrating the use of `quick_pp` was published in *Petrophysics Journal* — Vol. 66, No. 5 (October 2025), pp. 807–838:

> Fadhil, I. (2025). [*Enhanced Learning Experience for New Petrophysicists Using Open-Source Carbonate Data and Python Programming*](https://www.researchgate.net/publication/396099755_Enhanced_Learning_Experience_for_New_Petrophysicists_Using_Open-Source_Carbonate_Data_and_Python_Programming). *Petrophysics*, 66(5), 807–838.

