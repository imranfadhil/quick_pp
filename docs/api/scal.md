# Special Core Analysis & Relative Permeability (`quick_pp.scal`)

Lab-to-reservoir core plug corrections (overburden stress compaction, Klinkenberg gas slippage, Dean-Stark clay-bound water), capillary pressure scaling, Corey / Brooks-Corey / LET relative permeability models, automated curve fitting (`fit_relperm`), and Eclipse/Intersect `SWOF` table export (`generate_swof_table`).

## Grain / Matrix Density Correlation

Core-calibrated matrix density trends that override the mineral-model $\rho_{ma}$ in the density-porosity calculation.

* **Anchored single-driver form** (`fit_grain_density_correlation`, `grain_density_from_correlation`):

    $$\rho_{ma}(z) = \rho_{ma,\text{clean}} + k \cdot \left(\text{driver}(z) - \text{driver}_{\text{clean}}\right)$$

    The slope $k$ is fitted from paired core grain density (CGD) and a gamma driver, with the clean end member anchored at quartz ($\rho_{ma} \approx 2.65$ g/cc).

* **Multi-log ridge / Huber form** (`fit_grain_density_correlation_multivariate`, `grain_density_from_correlation_multivariate`):

    $$\rho_{ma}(z) = b_0 + \sum_j \beta_j \frac{\text{log}_j(z) - c_j}{s_j}$$

    Drivers (GR/CGR/SGR, RHOB, NPHI, Burke $M$-parameter, acoustic impedance) are robustly standardized with median/IQR, then fitted by ridge regression with optional Huber IRLS reweighting. Prefer this form when the gamma driver is **heteroscedastic**: total GR mixes K/Th/U sources, so a single-log OLS slope is dominated by high-gamma radioactive outliers (mica, glauconite, heavy minerals).

    Diagnostics returned with the fit:

    - **`r2` / `rmse` / `mae`** — conventional fit quality.
    - **`robust_rmse` / `robust_mae` / `resid_mad`** — outlier-resistant residual spread.
    - **`vif`** — variance inflation factor per driver; values above ~10 flag collinear channels (e.g. GR vs. CGR vs. SGR).
    - **`breusch_pagan_stat` / `breusch_pagan_p`** — heteroscedasticity test; a small $p$ confirms non-constant residual variance.
    - **`native_slopes`** — coefficients re-expressed in g/cc per driver unit.

::: quick_pp.scal

