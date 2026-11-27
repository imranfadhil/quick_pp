# Logging Environment, Borehole Corrections, QA/QC & Normalization (`quick_pp.environment`, `quick_pp.borehole_corrections`, `quick_pp.qaqc`)

Unified logging-environment physics, tool-profile (`WL` / `LWD`) borehole environmental corrections with per-sample uncertainty (`*_SIGMA`) and sensor confidence (`*_QC_CONF`), 2/3-curve radial resistivity invasion inversion, neutron-density hydrocarbon corrections, 4-criterion bad-hole flagging, and multi-well zone-by-zone quantile/histogram normalization.

Environmental correction outputs use distinct `*_ENV_CORR` mnemonics alongside `*_CORR_DELTA`, `*_SIGMA`, `*_QC_CONF`, `*_QC_STATUS`, `HOLE_SEVERITY`, `ENV_QC_REASON`, and radial invasion curves (`INVASION_FLAG`, `INVASION_PROFILE`, `DI_IN`). Density reconstruction has its own `RHOB_DEN_CORR` output. Downstream calculations use raw curves unless `use_env_corrected` / `confidence_weighting` (`confidence_weighted_curve`) is enabled.

## Well Header & Hole Sections metadata

The **Well & Ancillary** step (`InspectorAncillary.svelte`, Step 1 in the Single-Well workflow) manages both the Well Header metadata (`mud_type` `WBM`/`OBM`/`SBM`, `obm_base_density`, `rm`, `rmf`, `rmc`, `mud_filtrate_salinity`, `mud_filtrate_density`, water depth, air gap, and BHT records) and depth-zoned **Hole Sections** (`top`, `bottom`, `bit_size`, `mud_type`, `mud_weight`, `mud_salinity`, `rmf`, `rmf_temp`). These synthesize nominal bit size (`BS`) and depth-specific mud properties for `badhole_flagging`, `apply_environmental_corrections`, and the continuous apparent fluid-density ($\rho_{fl}$) solver in [Porosity Evaluation](porosity.md#continuous-apparent-fluid-density-rho_fl).

## Hydrocarbon correction targets

The crossplot hydrocarbon correction defaults to `target_mode="bulk"`: `VSH_GR` is
compared with bulk clay volume `vcl0`, consistent with density reconstruction.
Explicit `target_mode="solid"` compares with the binary clay fraction of solids,
`Vcl / (1 - porosity)`. In the standard invaded-zone iteration, `target_mode="solid"`
compares the clay fraction of solids, while `target_mode="bulk"` compares bulk clay.
Changing the mode changes both the target and the `vsh_gr` flagging criterion.
Re-run the correction to update existing curves.

The solid-relative binary projection is not the same as the piecewise
sand-silt-clay (SSC) `VSH_DN` displayed in the workbench, so matching the target
does not guarantee matching SSC `VSH_DN`. Both modes assume the GR-derived
value is a suitable target; anomalous GR may invalidate that assumption.

## Invaded-zone hydrocarbon correction

The workspace HC-correction endpoint offers an optional **Standard invaded-zone**
method alongside the existing crossplot correction. It searches invaded-zone
hydrocarbon saturation `S_hi` from 0 to 1 in 0.01 increments and selects the
candidate whose corrected binary clay estimate is closest to `VSH_GR`.
The density and neutron shifts, light/heavy hydrocarbon coefficients, mud-filtrate
hydrogen index, and lithology-scaled excavation term follow the supplied method
equations. The selected saturation is returned as `SHC_INV`.

The method requires well-header `mud_filtrate_density` (g/cc) and
`mud_filtrate_salinity` (ppm), editable in the Well Header panel in **Well & Ancillary**. Lithology
coefficient `K` defaults to 1. Density porosity is estimated from the dry-sand
density endpoint and mud-filtrate density; corrected `VSH_DN` is evaluated in the
matrix/clay/mud-filtrate crossplot triangle.

---

## Unified Logging Environment (`quick_pp.environment`)

::: quick_pp.environment

## Borehole Corrections, Sensor Confidence & Radial Invasion (`quick_pp.borehole_corrections`)

::: quick_pp.borehole_corrections

## Bad-Hole Flagging, Normalization & Curve Reconstruction (`quick_pp.qaqc`)

::: quick_pp.qaqc
