# Porosity Evaluation (`quick_pp.porosity`)

Total and effective porosity calculations from density, neutron-density crossplot, sonic, and core calibration.

::: quick_pp.porosity

## Continuous apparent fluid density ($\rho_{fl}$)

Density porosity is normally computed with a static fluid density
($\rho_{fl} = 1.0$ g/cc). Because the density tool is a shallow-reading pad
device, it actually measures the flushed-zone fluid mixture of invading mud
filtrate ($\rho_{mf}$) and residual unflushed hydrocarbon ($\rho_{h\_app}$).
Assuming a constant 1.0 g/cc therefore introduces systematic porosity errors of
roughly 1.5 to 8+ p.u. in high-salinity WBM, OBM/SBM, and light-oil/gas
intervals.

`estimate_continuous_rho_fluid` and `solve_continuous_rho_fluid_and_phid`
implement a vectorized fixed-point solver that estimates a continuous
$\rho_{fl}(z)$ and the corresponding corrected density porosity:

$$\rho_{fl}(z) = S_{xo}(z)\,\rho_{mf}(z) + \left(1 - S_{xo}(z)\right)\rho_{h\_app}(z)$$

$$\phi_D(z) = \frac{\rho_{ma}(z) - \rho_b(z)}{\rho_{ma}(z) - \rho_{fl}(z)}$$

Because $S_{xo}$ depends on $\phi_D$, the pair is iterated until
$\max_z |\phi_D^{(k+1)} - \phi_D^{(k)}| < 10^{-4}$ (default `max_iter=8`).

### Tiered fallback

The solver auto-degrades **per depth sample**, so a single run can mix tiers:

| Tier | Condition | $S_{xo}$ model |
| :--- | :--- | :--- |
| 1 | Valid `Rxo` + `Rmf`, WBM, `BADHOLE == 0` | Archie from shallow resistivity |
| 2 | `Rxo` missing/flagged, valid `Rt` | $S_{xo} = S_w^{c_{flush}}$ (Poupon, default $c_{flush}=0.20$) |
| 3 | No usable resistivity | Zonal constant $S_{xo} = 1 - S_{hr}$ |

The tier actually used at each depth is returned in `RHO_FL_TIER`
(1 = Rxo, 2 = $S_w^{c}$, 3 = zonal).

### Filtrate and hydrocarbon densities

* **WBM** — `brine_density` (Batzle & Wang, 1992) evaluated at the downhole
  temperature and pressure. Salinity comes from `mud_salinity_ppm`, or is
  back-calculated from a surface `Rmf` with `salinity_from_rmf` (Arps
  temperature correction + Bateman-Konen salinity correlation). With neither
  available the solver falls back to 5,000 ppm (fresh) or 150,000 ppm (saline).
* **OBM/SBM** — `obm_filtrate_density` applies thermal expansion and pressure
  compressibility to the surface base-oil density (default 0.82 g/cc).
* **Hydrocarbon** — true in-situ density from `estimate_fluid_properties`
  (Standing / Batzle & Wang for oil, real-gas EOS for gas), converted to the
  gamma-gamma apparent response with the Gaymard & Poupon (1968) relations
  ($\rho_{o\_app} = 1.07\rho_o - 0.07$, $\rho_{g\_app} = 1.14\rho_g - 0.14$).

### Implementation conventions

The specification left a few behaviours undefined; the following conventions are
used and are worth knowing when interpreting results:

* **Water zones** collapse the hydrocarbon term to the filtrate/water density,
  so $\rho_{fl}$ in a water leg equals $\rho_{mf}$ (1.00 g/cc for a default WBM)
  rather than the formation-water density.
* **Tier 1 lower clip** uses the zonal residual saturation when `Rt` is absent.
  Clipping to `Sw` would force $S_{xo} = 1.0$ (no correction) whenever `Sw`
  defaults to 1.0.
* **`PHID_RAW_100`** is the reference density porosity at $\rho_{fl} = 1.0$,
  i.e. $(\rho_{ma} - \rho_b)/(\rho_{ma} - 1.0)$.
* **Tier 3 Option 3A** (Gaymard-Poupon NPHI gas correction) is not implemented;
  Tier 3 uses the zonal $S_{hr}$ option only.
* **`RHO_FL_TIER`** is stored as a float (1.0/2.0/3.0) for curve-database
  compatibility.

### Output curves

| Mnemonic | Description |
| :--- | :--- |
| `PHID` | Corrected density porosity (replaces the static-$\rho_{fl}$ value) |
| `PHID_RAW_100` | Density porosity at $\rho_{fl} = 1.0$ (QC reference) |
| `RHO_FL` | Continuous apparent fluid density (g/cc) |
| `RHO_MF` | Continuous mud-filtrate density (g/cc) |
| `SXO_EST` | Estimated flushed-zone water saturation |
| `RHO_FL_TIER` | Solver tier used per sample (1/2/3) |

These five diagnostic curves (`RHO_FL`, `RHO_MF`, `SXO_EST`, `PHID_RAW_100`,
`RHO_FL_TIER`) plus `PHID` are QC outputs and are **not** drawn on the well log.
The porosity comparison endpoint returns them under `rho_fl_diagnostics`, and the
**Porosity Methods vs Core** plot renders them: `RHO_FL`/`RHO_MF` on the density
axis alongside `RHOM` and the scalar fluid density line, and `PHID_RAW_100` vs
`PHID` on the porosity axis to show $\Delta\phi_D$.

