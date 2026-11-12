# Technical Implementation Specification: 1D Constrained Deconvolution & Resolution Harmonization (`GR`, `NPHI`, `RHOB`) for `quick_pp`

**Target Codebase:** `quick_pp` (Python $3.11+$, `numpy`, `scipy`, `pandas`)  
**Proposed Module Path:** `quick_pp/conditioning.py` (or `quick_pp/deconvolution.py`, integrated with `quick_pp.qaqc`)  
**Downstream Consumers:** `quick_pp.lithology`, `quick_pp.porosity`, `quick_pp.saturation` (specifically Thomas-Stieber laminated shaly-sand workflows: `VSH_LAM`, `VSH_DISP`, `PHIT_TS`, `PHIE_TS`, `PHISAND_TS`)

---

## 1. Architectural Objectives & Petrophysical Design Principles

In Low-Resistivity Low-Contrast (LRLC) reservoirs, legacy nuclear logs (`GR`, `NPHI`, `RHOB`) suffer from two distinct vertical resolution problems:
1. **Shoulder-Bed Averaging:** Intrinsic tool vertical resolutions ($0.38\text{ m}$ to $0.90\text{ m}$, or $1.25\text{ ft}$ to $3.0\text{ ft}$) smear thin sand-shale interbeds toward intermediate shaly-sand values.
2. **Inter-Sensor Resolution Mismatch:** Because `GR`, `NPHI`, and `RHOB` investigate different vertical rock volumes, bed-boundary transitions shift at different rates. On a Thomas-Stieber ($\phi_t$ vs. $V_{sh}$) crossplot, this mismatch scatters laminated points into unphysical "boomerang" loops inside the dispersed-clay region.

To address both issues cleanly within `quick_pp`, the implementation follows five engineering constraints:

* **Raw-Log Enhancement Only:** Deconvolution and resolution matching are strictly executed on conditioned raw logs (`GR`, `NPHI`, `RHOB`) *prior* to non-linear petrophysical transforms ($V_{sh}$, $\phi_t$, $S_w$).
* **Target Resolution Harmonization Over Blind Over-Sharpening:** Rather than attempting deconvolution toward an unachievable Dirac impulse $\delta(z)$—which amplifies high-frequency noise near the Nyquist limit—all three curves are deconvolved onto a **common target vertical resolution** ($FWHM_{\text{target}}$), subject to the Shannon-Nyquist sampling bound of the input LAS depth step $\Delta z$.
* **Edge-Preserving & Bounded Inversion:** Unconstrained linear inverse filters (pure $L_2$ Tikhonov or Wiener filters) create Gibbs ringing ("ears") adjacent to tight streaks, coals, and high-contrast bed boundaries. The primary solver enforces hard petrophysical box bounds ($\mathbf{l} \le \mathbf{x} \le \mathbf{u}$) and a **Huber edge-preserving penalty** that suppresses sensor noise on flat beds, preserves sharp bed boundaries, and avoids artificial stair-casing across fining-upward trends.
* **Bad-Hole Isolation & Bulk-Volume Conservation:** Washout intervals (`CALI`, `DRHO`) are downweighted in the data-fidelity term so borehole rugosity spikes are never propagated into adjacent reservoir sands, while rolling integral checks guarantee conservation of bulk rock properties.
* **Zero Heavy Dependencies & Raw Lineage Preservation:** Built exclusively with vectorized `numpy` and `scipy` (`scipy.optimize.minimize` with `L-BFGS-B` and exact analytical adjoint gradients). Raw curves (`GR`, `NPHI`, `RHOB`) remain untouched; outputs are emitted as explicit columns (`GR_ENH`, `NPHI_ENH`, `RHOB_ENH`) alongside forward-reconstructed curves (`GR_RECON`, `NPHI_RECON`, `RHOB_RECON`) for QAQC parity inspection.

---

## 2. Forward Tool Physics & Differential Kernel Formulation

### 2.1 Linear Domain Assumptions & Physical Box Constraints

Within a borehole free of severe washouts, each nuclear log measurement $y(z)$ is modeled as a 1D linear convolution of the true formation property $e(z)$ with a normalized Vertical Response Function (VRF) $h_{\text{tool}}(z)$ plus additive statistical counting noise $n(z)$:

$$y(z) = (h_{\text{tool}} * e)(z) + n(z) = \int_{-\infty}^{+\infty} h_{\text{tool}}(\zeta)\, e(z - \zeta)\, d\zeta + n(z), \qquad \int_{-\infty}^{+\infty} h_{\text{tool}}(\zeta)\, d\zeta = 1$$

| Curve | Physical Domain | Linearity Validity & Practical Caveats | Default Hard Bounds $[l, u]$ |
| :--- | :--- | :--- | :--- |
| `GR` | Natural radioactivity ($\text{API}$) | Linear in bulk radioactivity. (Optional mass-weighted mode $\rho_b \cdot \text{GR}$ can be toggled when high-density barite or coal beds are present, though direct $\text{API}$ deconvolution is standard for clastic LRLC). | $[0.0,\; 300.0]\;\text{API}$ |
| `RHOB` | Bulk electron density ($\text{g/cm}^3$) | Strictly linear in bulk density over reservoir intervals when mudcake compensation is valid (`\|DRHO\|` $\le 0.15\;\text{g/cm}^3$). | $[1.75,\; 3.05]\;\text{g/cm}^3$ *(extend $l$ to $1.15\;\text{g/cm}^3$ if coals exist)* |
| `NPHI` | Apparent hydrogen index ($\text{v/v}$ decimal) | Approximately linear across clastic porosities ($0.0 \le \phi_N \le 0.45$). Epithermal/excavation non-linearities occur in gas or very high porosity shales, handled via tight physical bounds. | $[-0.03,\; 0.65]\;\text{v/v}$ |

### 2.2 Parametric Vertical Response Functions (VRFs)

When exact service-company chartbook VRF tables are unavailable for legacy wells, the tool's vertical response is parameterized by its **Full Width at Half Maximum ($FWHM$)**. Two analytical VRF families and one empirical option should be supported:

1. **Gaussian VRF (Default & Semigroup-Compatible):**
   $$h_{\text{gauss}}(z; \sigma) = \frac{1}{\sqrt{2\pi}\,\sigma} \exp\left(-\frac{z^2}{2\sigma^2}\right), \qquad \sigma = \frac{FWHM}{2\sqrt{2\ln 2}} \approx \frac{FWHM}{2.35482}$$
2. **Symmetric Double-Exponential (Laplacian) VRF (Better tail representation for scintillation `GR` and `NPHI`):**
   $$h_{\text{exp}}(z; b) = \frac{1}{2b} \exp\left(-\frac{|z|}{b}\right), \qquad b = \frac{FWHM}{2\ln 2} \approx \frac{FWHM}{1.38629}$$
3. **Custom Discrete Kernel:** User-supplied 1D array sampled at $\Delta z$ (supports asymmetric axial response functions if logging speed and tool asymmetry are known from legacy headers).

### 2.3 Default Legacy Tool Resolutions & Target Harmonization

Typical legacy wireline logging suites logged at standard speeds ($9\text{--}18\;\text{m/min}$ or $1800\text{--}3600\;\text{ft/hr}$) exhibit the following intrinsic resolutions:

| Legacy Tool / Mnemonic | Typical Legacy Tool Family | Default $FWHM_{\text{tool}}$ ($\text{m}$) | Default $FWHM_{\text{tool}}$ ($\text{ft}$) |
| :--- | :--- | :--- | :--- |
| `GR` | Scintillation Gamma Ray (SGR / NGT / GR) | $0.76\;\text{m}$ | $2.50\;\text{ft}$ |
| `NPHI` | Compensated Thermal Neutron (CNL / SNP) | $0.55\;\text{m}$ | $1.80\;\text{ft}$ |
| `RHOB` | Litho-Density / Compensated Formation Density (LDT / FDC) | $0.40\;\text{m}$ | $1.30\;\text{ft}$ |

#### Target Resolution Matching ($FWHM_{\text{target}}$) & The Nyquist Guardrail
Instead of attempting to invert $e(z)$ at infinite resolution, we define the **target-resolution formation log** $x(z) = (h_{\text{target}} * e)(z)$ characterized by $FWHM_{\text{target}}$. Under the Gaussian VRF model, convolution forms an exact additive variance semigroup:

$$h_{\text{tool}} = h_{\text{eff}} * h_{\text{target}} \implies \sigma_{\text{eff}} = \sqrt{\max\left(\sigma_{\text{tool}}^2 - \sigma_{\text{target}}^2,\; 0\right)}$$

Thus, the forward relationship directly linking the desired harmonized curve $x(z)$ to the measured legacy curve $y(z)$ is:

$$y(z) = (h_{\text{eff}} * x)(z) + n(z)$$

* **Nyquist Sampling Constraint:** Standard legacy LAS files have depth step $\Delta z = 0.1524\;\text{m}$ ($0.5\;\text{ft}$). By the Shannon-Nyquist theorem, any spatial wavelength shorter than $2\Delta z$ is aliased. Therefore, `quick_pp` must enforce a hard floor on target resolution:
  $$FWHM_{\text{target}} \ge \kappa_{\text{nyq}} \cdot \Delta z, \qquad \kappa_{\text{nyq}} = 2.0 \quad (\text{e.g., } FWHM_{\text{target}} \ge 0.3048\;\text{m} \text{ or } 1.0\;\text{ft} \text{ for } 0.5\text{-ft LAS})$$
  If high-sampling raw tapes ($\Delta z = 0.0762\;\text{m}$ or $0.25\;\text{ft}$) are loaded, $FWHM_{\text{target}}$ can safely reach $0.18\text{--}0.25\;\text{m}$ ($0.6\text{--}0.8\;\text{ft}$).
* **Two Harmonization Modes:**
  1. **`mode="match_finest"` (Conservative):** Sets $FWHM_{\text{target}} = \min(FWHM_{\text{RHOB}}) \approx 0.40\;\text{m}$ ($1.30\;\text{ft}$). Here $\sigma_{\text{eff, RHOB}} = 0$ (`RHOB` is lightly denoised only), while `NPHI` and `GR` are sharpened to match `RHOB`'s exact vertical footprint.
  2. **`mode="enhance_all"` (Recommended for LRLC):** Sets $FWHM_{\text{target}} = 2.0 \cdot \Delta z$ ($0.3048\;\text{m}$ / $1.0\;\text{ft}$ on standard LAS), sharpening all three logs (`GR`, `NPHI`, `RHOB`) onto an identical $1.0\text{-ft}$ vertical footprint.

---

## 3. Mathematical Formulation of 1D Constrained Deconvolution

### 3.1 Exact Discrete Forward & Adjoint Operators ($\mathbf{H}$ and $\mathbf{H}^T$)

Let $\mathbf{x} \in \mathbb{R}^N$ be the discrete target-resolution log on a uniform grid $z_i = z_0 + i\Delta z$, and let $\mathbf{k} \in \mathbb{R}^{2M+1}$ be the truncated discrete kernel ($M = \lceil 3.5\,\sigma_{\text{eff}} / \Delta z \rceil$) normalized such that $\sum_{m=0}^{2M} k_m = 1$.

Zero-padding at the top and base of a logged interval causes severe artificial boundary droop. Instead, define the **edge-replicate padding operator** $\mathbf{P} \in \mathbb{R}^{(N+2M) \times N}$:

$$(\mathbf{P}\mathbf{x})_j = \begin{cases} x_0, & 0 \le j < M \\ x_{j-M}, & M \le j < M + N \\ x_{N-1}, & M + N \le j < N + 2M \end{cases}$$

Followed by the valid-mode cross-correlation operator $\mathbf{C} \in \mathbb{R}^{N \times (N+2M)}$:

$$(\mathbf{C}\mathbf{u})_i = \sum_{m=0}^{2M} k_m\, u_{i+m}, \qquad i = 0, \dots, N-1$$

The forward operator is $\mathbf{H}\mathbf{x} = \mathbf{C}(\mathbf{P}\mathbf{x})$. For gradient-based optimization (`L-BFGS-B`) to converge without line-search failures, $\mathbf{H}^T\mathbf{r}$ must equal the exact algebraic transpose of $\mathbf{H}$ to machine precision ($\approx 10^{-15}$):

1. Compute full-mode convolution $\tilde{\mathbf{u}} = \mathbf{C}^T\mathbf{r} \in \mathbb{R}^{N+2M}$, where $\tilde{u}_j = (\mathbf{r} * \mathbf{k})_j$ (`scipy.signal.convolve(r, k, mode="full")`).
2. Fold padded boundary weights via $\mathbf{P}^T\tilde{\mathbf{u}} \in \mathbb{R}^N$:
   $$(\mathbf{H}^T\mathbf{r})_i = \begin{cases} \displaystyle\sum_{j=0}^{M} \tilde{u}_j, & i = 0 \\[8pt] \tilde{u}_{i+M}, & 1 \le i \le N-2 \\[8pt] \displaystyle\sum_{j=M+N-1}^{N+2M-1} \tilde{u}_j, & i = N-1 \end{cases}$$

*(Note: This formulation guarantees $\langle \mathbf{H}\mathbf{x}, \mathbf{r} \rangle = \langle \mathbf{x}, \mathbf{H}^T\mathbf{r} \rangle$ identically for both symmetric and asymmetric kernels).*

### 3.2 Constrained Variational Objective Function (Production Solver)

We solve the box-constrained convex optimization problem:

$$\min_{\mathbf{l} \le \mathbf{x} \le \mathbf{u}} \; \mathcal{J}(\mathbf{x}) = \underbrace{\frac{1}{2} \left\| \mathbf{W}_d (\mathbf{H}\mathbf{x} - \mathbf{y}) \right\|_2^2}_{\text{Weighted Data Fidelity}} + \underbrace{\lambda_{\text{reg}} \sum_{i=0}^{N-2} \phi_{\delta}(x_{i+1} - x_i)}_{\text{Edge-Preserving Huber Prior}} + \underbrace{\frac{\lambda_{\text{vol}}}{2} \left\| \mathbf{B}(\mathbf{x} - \mathbf{y}) \right\|_2^2}_{\text{Bulk-Volume Conservation Tether}}$$

#### Term 1: Bad-Hole Reliability Weights ($\mathbf{W}_d = \text{diag}(\mathbf{w})$)
Washouts cause sharp density drops (`RHOB` $\to 1.9\;\text{g/cm}^3$) and neutron spikes (`NPHI` $\to 0.45$). If $w_i = 1$ in a washout, deconvolution misinterprets the spike as a smeared ultra-porous thin bed and creates negative/positive ringing lobes in adjacent clean beds. Define continuous reliability weights $w_i \in [\epsilon_w, 1]$ (with $\epsilon_w = 0.05$):

$$w_i = w_{\text{cali}, i} \cdot w_{\text{drho}, i}$$

$$w_{\text{cali}, i} = \exp\left(-\max\left(0,\; \frac{\Delta d_{\text{hole}, i} - \tau_{\text{cali}}}{s_{\text{cali}}}\right)^2\right), \qquad w_{\text{drho}, i} = \exp\left(-\max\left(0,\; \frac{|\text{DRHO}_i| - \tau_{\text{drho}}}{s_{\text{drho}}}\right)^2\right)$$

where $\Delta d_{\text{hole}, i} = \text{CALI}_i - \text{BS}_i$ (default $\tau_{\text{cali}} = 2.0\;\text{in}$ or $50\;\text{mm}$, $s_{\text{cali}} = 1.5\;\text{in}$; default $\tau_{\text{drho}} = 0.10\;\text{g/cm}^3$, $s_{\text{drho}} = 0.08\;\text{g/cm}^3$; $w_{\text{drho}, i} = 1$ for `GR`).

#### Term 2: Normalized Huber Edge-Preserving Regularization ($\phi_{\delta}$)
Pure $L_2$ gradient penalties ($\|\mathbf{D}_1\mathbf{x}\|_2^2$) smear sharp bed boundaries and cause Gibbs overshoot. Pure $L_1$ Total Variation ($\|\mathbf{D}_1\mathbf{x}\|_1$) turns smooth fining-upward turbidite sequences into blocky staircases, distorting dispersed vs. laminar clay trends on Thomas-Stieber diagrams. The **normalized Huber function** combines both regimes seamlessly:

$$\phi_{\delta}(d) = \begin{cases} \dfrac{d^2}{2\delta}, & |d| \le \delta \\[8pt] |d| - \dfrac{\delta}{2}, & |d| > \delta \end{cases}, \qquad \phi'_{\delta}(d) = \text{clip}\left(\frac{d}{\delta},\; -1,\; +1\right)$$

* For small sample-to-sample fluctuations ($|x_{i+1} - x_i| \le \delta$), $\phi_{\delta}$ acts quadratically to smooth counting noise and preserve gradual geological ramps.
* Across sharp bed boundaries ($|x_{i+1} - x_i| > \delta$), $\phi_{\delta}$ acts linearly like Total Variation, permitting crisp jumps without penalizing contrast amplitude—eliminating Gibbs "ears" next to tight carbonate or coal streaks.

#### Term 3: Rolling Bulk-Volume Conservation Tether ($\mathbf{B}$)
Let $\mathbf{B}$ be a wide boxcar/Gaussian averaging operator over window $L_{\text{vol}} \approx 3.0\;\text{m}$ ($10\;\text{ft}$, roughly $4\times FWHM_{\text{tool}}$). Because $\mathbf{H}$ is unit-sum, $\mathbf{B}\mathbf{x} \approx \mathbf{B}\mathbf{y}$ holds naturally in good hole, while a mild weight ($\lambda_{\text{vol}} \approx 0.05$) anchors macro-interval bulk density and hydrogen index across downweighted washout zones so net reservoir pore volume never drifts.

#### Exact Analytical Gradient ($\nabla \mathcal{J}(\mathbf{x})$)
Let $d_i = x_{i+1} - x_i$ and $g_i = \text{clip}(d_i / \delta, -1, 1)$ for $i = 0, \dots, N-2$. Then:

$$\nabla \mathcal{J}(\mathbf{x}) = \mathbf{H}^T \left( \mathbf{w}^2 \odot (\mathbf{H}\mathbf{x} - \mathbf{y}) \right) + \lambda_{\text{reg}} \mathbf{D}_1^T \mathbf{g} + \lambda_{\text{vol}} \mathbf{B}^T \mathbf{B}(\mathbf{x} - \mathbf{y})$$

where $(\mathbf{D}_1^T \mathbf{g})_0 = -g_0$, $(\mathbf{D}_1^T \mathbf{g})_i = g_{i-1} - g_i$ for $1 \le i \le N-2$, and $(\mathbf{D}_1^T \mathbf{g})_{N-1} = g_{N-2}$.  
Solved via `scipy.optimize.minimize(..., method="L-BFGS-B", jac=True, bounds=bounds)`, a $1500\;\text{m}$ ($10{,}000\text{-sample}$) well converges in $30\text{--}60$ iterations ($\approx 15\text{--}35\;\text{ms}$ in pure NumPy/SciPy).

### 3.3 Fast Spectral Baseline (Optional Preview Mode: `method="wiener"`)

For rapid interactive slider previews in the `quick_pp` SvelteKit UI before committing constrained `L-BFGS-B` runs, a closed-form FFT Tikhonov-Wiener filter can be offered:

$$\hat{X}(\omega) = \frac{H_{\text{eff}}^*(\omega)}{|H_{\text{eff}}(\omega)|^2 + \gamma \left(2 - 2\cos(\omega \Delta z)\right)} Y(\omega)$$

clipped post-hoc to $[\mathbf{l}, \mathbf{u}]$, where $\gamma \approx (\hat{\sigma}_n / \sigma_{\text{signal}})^2$. Document clearly in docstrings that `method="constrained_huber"` (default) must be used for final Thomas-Stieber evaluation to avoid edge overshoot.

---

## 4. Automated Noise Estimation & Morozov Discrepancy Auto-Tuning

Manual tuning of regularization weights across tens of legacy wells is error-prone. `quick_pp` should automatically calibrate $\hat{\sigma}_n$, $\delta$, and $\lambda_{\text{reg}}$ per log curve over valid good-hole samples ($w_i > 0.8$):

1. **Robust Noise Standard Deviation ($\hat{\sigma}_n$) via Second-Order Differences:**
   First differences $\mathbf{D}_1 \mathbf{y}$ are contaminated by geological ramps, whereas second differences $(\mathbf{D}_2 \mathbf{y})_i = y_{i+1} - 2y_i + y_{i-1}$ cancel linear trends. For white noise $n_i \sim \mathcal{N}(0, \sigma_n^2)$, $\text{Var}(n_{i+1} - 2n_i + n_{i-1}) = 6\sigma_n^2$. Using the Median Absolute Deviation (MAD) makes $\hat{\sigma}_n$ immune to sharp bed boundaries:
   $$\hat{\sigma}_n = \max\left(\frac{\text{median}\left(|\mathbf{D}_2 \mathbf{y}_{\text{good}}|\right)}{0.67449 \cdot \sqrt{6}},\; \sigma_{n, \min}\right)$$
   *(Floor $\sigma_{n, \min}$ prevents zero-noise singularities on synthetic or heavily pre-filtered LAS curves: $0.5\;\text{API}$ for `GR`, $0.005\;\text{g/cm}^3$ for `RHOB`, $0.003\;\text{v/v}$ for `NPHI`).*

2. **Automatic Huber Threshold ($\delta$):**
   Set $\delta = c_{\delta} \cdot \hat{\sigma}_n$ with default $c_{\delta} = 2.0$. Variations below $2\hat{\sigma}_n$ are treated as sensor noise/gradual ramps; steps exceeding $2\hat{\sigma}_n$ are preserved as bed boundaries.

3. **Morozov Discrepancy Principle for $\lambda_{\text{reg}}$:**
   A physically valid deconvolved log $\mathbf{x}^*(\lambda_{\text{reg}})$ must reproduce the measured raw log $\mathbf{y}$ upon forward re-convolution to within the statistical noise level—no closer (which overfits noise into fake laminations) and no looser (which under-resolves thin beds). Define weighted RMS residual:
   $$r_{\text{rms}}(\lambda_{\text{reg}}) = \sqrt{\frac{\sum_{i=0}^{N-1} w_i^2 \left((\mathbf{H}\mathbf{x}^*)_i - y_i\right)^2}{\sum_{i=0}^{N-1} w_i^2}}$$
   Using 5–7 steps of log-space bisection (warm-starting each `L-BFGS-B` call from the previous $\mathbf{x}^*$), select $\lambda_{\text{reg}}$ such that:
   $$r_{\text{rms}}(\lambda_{\text{reg}}) \approx \tau_{\text{morozov}} \cdot \hat{\sigma}_n, \qquad \tau_{\text{morozov}} = 1.05$$

---

## 5. Pre-Conditioning Pipeline & Edge-Case Guards

Before invoking deconvolution, the orchestration pipeline must run four deterministic guards:

1. **Depth Grid Uniformity & Unit Detection:**
   Inspect $\Delta z = \text{median}(\text{diff}(\text{DEPTH}))$. Auto-detect metric ($\Delta z \approx 0.1524\;\text{m}$ or $0.0762\;\text{m}$) vs. imperial ($\Delta z \approx 0.5\;\text{ft}$ or $0.25\;\text{ft}$) if `depth_uom` is omitted, and convert default tool $FWHM$ values into matching depth units. Raise a clear `ValueError` (or resample linearly onto a uniform grid with a warning) if $|\max(\text{diff}(z)) - \min(\text{diff}(z))| > 10^{-3}\,\Delta z$.
2. **NaN Gap Segmentation:**
   * Gaps $\le 2$ samples: Linearly interpolate prior to deconvolution, then restore `NaN` masks afterward.
   * Gaps $> 2$ samples (e.g., casing shoes, unlogged intervals): Split the log into contiguous valid segments of length $\ge 2M + 5$ and deconvolve each segment independently so `NaN` boundaries never contaminate adjacent valid data.
3. **Micro-Depth Alignment Diagnostic (Pre- & Post-Enhancement):**
   Enhancing vertical resolution amplifies depth-shift errors! A $0.15\;\text{m}$ ($0.5\;\text{ft}$) depth mismatch between `RHOB_ENH` and `NPHI_ENH` creates severe false gas/shale spikes at every bed boundary. Include a diagnostic helper `check_interlog_depth_alignment()` that computes normalized cross-correlation between $|\partial_z \text{GR}|$, $|\partial_z \text{RHOB}|$, and $|\partial_z \text{NPHI}|$ over lags $[-5\Delta z, +5\Delta z]$ and flags a warning if peak correlation occurs at a non-zero lag.
4. **Coal & Tight-Streak Halo Protection:**
   If carbonaceous shales or coals ($\text{RHOB} < 1.85\;\text{g/cm}^3$, $\text{NPHI} > 0.45\;\text{v/v}$) or tight carbonate stringers ($\text{RHOB} > 2.68\;\text{g/cm}^3$, $\text{NPHI} < 0.04\;\text{v/v}$) are detected, verify that `method="constrained_huber"` is active and adaptively expand local lower/upper bounds only within the flagged non-reservoir lithology while keeping adjacent sand bounds strictly enforced.

---

## 6. `quick_pp` API Specification & Reference Implementation Blueprint

### 6.1 Module Layout & Naming Conventions

To align with existing `quick_pp` dataframe workflows (`df` containing `DEPTH`, `GR`, `NPHI`, `RHOB`, and optional `CALI`, `BS`, `DRHO`), add `quick_pp/conditioning.py`:

* **Output Columns Added to DataFrame (Non-Destructive):**
  * `GR_ENH`, `NPHI_ENH`, `RHOB_ENH`: Resolution-matched & enhanced logs ready for `quick_pp.lithology` and `quick_pp.porosity`.
  * `GR_RECON`, `NPHI_RECON`, `RHOB_RECON`: Forward-convolved $\mathbf{H}\mathbf{x}^*$ curves for visual & quantitative residual QC (`y - y_recon`).
  * `DECONV_WEIGHT`: Composite borehole reliability weight $w_i \in [0.05, 1.0]$.

### 6.2 Reference Implementation (`quick_pp/conditioning.py`)

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import convolve, correlate


VrfType = Literal["gaussian", "double_exp"]
HarmonizeMode = Literal["enhance_all", "match_finest"]
SolverMethod = Literal["constrained_huber", "wiener"]


@dataclass(frozen=True)
class ToolResolutionSpec:
    """Vertical resolution and physical constraint specification for a single nuclear log."""

    curve_name: str
    fwhm_m: float
    lower_bound: float
    upper_bound: float
    noise_floor: float
    vrf_type: VrfType = "gaussian"
    apply_drho_weight: bool = True


DEFAULT_LEGACY_SPECS: dict[str, ToolResolutionSpec] = {
    "GR": ToolResolutionSpec(
        curve_name="GR",
        fwhm_m=0.76,
        lower_bound=0.0,
        upper_bound=300.0,
        noise_floor=0.5,
        vrf_type="gaussian",
        apply_drho_weight=False,
    ),
    "NPHI": ToolResolutionSpec(
        curve_name="NPHI",
        fwhm_m=0.55,
        lower_bound=-0.03,
        upper_bound=0.65,
        noise_floor=0.003,
        vrf_type="gaussian",
        apply_drho_weight=True,
    ),
    "RHOB": ToolResolutionSpec(
        curve_name="RHOB",
        fwhm_m=0.40,
        lower_bound=1.75,
        upper_bound=3.05,
        noise_floor=0.005,
        vrf_type="gaussian",
        apply_drho_weight=True,
    ),
}


@dataclass
class CurveDeconvDiagnostics:
    """Quantitative QAQC diagnostics emitted per deconvolved log curve."""

    curve_name: str
    intrinsic_fwhm_m: float
    target_fwhm_m: float
    effective_fwhm_m: float
    estimated_noise_std: float
    lambda_reg: float
    huber_delta: float
    weighted_recon_rmse: float
    relative_volume_bias_pct: float
    converged: bool
    iterations: int


@dataclass
class HarmonizationResult:
    """Container returned by harmonize_nuclear_logs with conditioned DataFrame and QC metrics."""

    df: pd.DataFrame
    target_fwhm_m: float
    depth_step_m: float
    diagnostics: dict[str, CurveDeconvDiagnostics] = field(default_factory=dict)
    alignment_lags_samples: dict[str, int] = field(default_factory=dict)


def build_vrf_kernel(
    effective_fwhm: float,
    dz: float,
    vrf_type: VrfType = "gaussian",
    truncate_stds: float = 3.5,
) -> np.ndarray:
    """Construct a unit-sum discrete symmetric 1D Vertical Response Function kernel."""
    if effective_fwhm <= 1e-6:
        return np.array([1.0], dtype=np.float64)

    if vrf_type == "gaussian":
        sigma = effective_fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        half_len = max(1, int(np.ceil(truncate_stds * sigma / dz)))
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64) * dz
        kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    elif vrf_type == "double_exp":
        scale_b = effective_fwhm / (2.0 * np.log(2.0))
        half_len = max(1, int(np.ceil(5.0 * scale_b / dz)))
        offsets = np.arange(-half_len, half_len + 1, dtype=np.float64) * dz
        kernel = np.exp(-np.abs(offsets) / scale_b)
    else:
        raise ValueError(f"Unsupported vrf_type: {vrf_type}")

    kernel /= np.sum(kernel)
    return kernel


def apply_vrf_forward(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute H @ x using edge-replicated padding to eliminate top/bottom log droop."""
    half_len = (len(kernel) - 1) // 2
    if half_len == 0:
        return x.copy()
    x_pad = np.pad(x, pad_width=half_len, mode="edge")
    return correlate(x_pad, kernel, mode="valid")


def apply_vrf_adjoint(r: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute exact algebraic transpose H.T @ r matching apply_vrf_forward to machine precision."""
    half_len = (len(kernel) - 1) // 2
    if half_len == 0:
        return r.copy()
    u_full = convolve(r, kernel, mode="full")
    n = len(r)
    out = np.empty(n, dtype=np.float64)
    out[0] = np.sum(u_full[: half_len + 1])
    if n > 2:
        out[1:-1] = u_full[half_len + 1 : half_len + n - 1]
    if n > 1:
        out[-1] = np.sum(u_full[half_len + n - 1 :])
    return out


def estimate_noise_mad(
    y: np.ndarray,
    weights: np.ndarray,
    noise_floor: float,
) -> float:
    """Estimate sensor noise std via 2nd-order finite-difference MAD over good-hole samples."""
    good_mask = weights > 0.8
    y_eval = y[good_mask] if np.sum(good_mask) >= 15 else y
    if len(y_eval) < 5:
        return noise_floor
    d2 = y_eval[2:] - 2.0 * y_eval[1:-1] + y_eval[:-2]
    sigma_est = float(np.median(np.abs(d2)) / (0.67448975 * np.sqrt(6.0)))
    return max(sigma_est, noise_floor)


def compute_borehole_weights(
    df: pd.DataFrame,
    cali_col: str | None = "CALI",
    bs_col: str | None = "BS",
    drho_col: str | None = "DRHO",
    cali_tol: float = 2.0,
    cali_scale: float = 1.5,
    drho_tol: float = 0.10,
    drho_scale: float = 0.08,
    min_weight: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (w_cali, w_both) reliability weight arrays in [min_weight, 1.0]."""
    n = len(df)
    w_cali = np.ones(n, dtype=np.float64)
    if cali_col and bs_col and cali_col in df.columns and bs_col in df.columns:
        excess = (df[cali_col].to_numpy(dtype=float) - df[bs_col].to_numpy(dtype=float))
        excess = np.nan_to_num(excess, nan=0.0)
        over = np.maximum(0.0, (excess - cali_tol) / cali_scale)
        w_cali = np.exp(-(over ** 2))

    w_drho = np.ones(n, dtype=np.float64)
    if drho_col and drho_col in df.columns:
        abs_drho = np.abs(np.nan_to_num(df[drho_col].to_numpy(dtype=float), nan=0.0))
        over_d = np.maximum(0.0, (abs_drho - drho_tol) / drho_scale)
        w_drho = np.exp(-(over_d ** 2))

    w_cali = np.clip(w_cali, min_weight, 1.0)
    w_both = np.clip(w_cali * w_drho, min_weight, 1.0)
    return w_cali, w_both


def _solve_segment_huber(
    y: np.ndarray,
    weights: np.ndarray,
    kernel: np.ndarray,
    vol_kernel: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    lambda_reg: float,
    huber_delta: float,
    lambda_vol: float = 0.05,
    x0: np.ndarray | None = None,
    maxiter: int = 150,
) -> tuple[np.ndarray, bool, int]:
    """Solve 1D bounded Huber-regularized deconvolution on a contiguous valid segment."""
    n = len(y)
    w_sq = weights ** 2
    x_init = np.clip(y.copy() if x0 is None else x0.copy(), lower_bound, upper_bound)
    bounds = [(lower_bound, upper_bound)] * n

    def objective_and_grad(x: np.ndarray) -> tuple[float, np.ndarray]:
        # 1. Weighted data fidelity term
        res = apply_vrf_forward(x, kernel) - y
        w_res = w_sq * res
        loss_fid = 0.5 * float(np.dot(res, w_res))
        grad_fid = apply_vrf_adjoint(w_res, kernel)

        # 2. Normalized Huber edge-preserving prior
        if n > 1 and lambda_reg > 0.0:
            diffs = x[1:] - x[:-1]
            abs_d = np.abs(diffs)
            quad_mask = abs_d <= huber_delta
            huber_vals = np.where(
                quad_mask,
                0.5 * (diffs ** 2) / huber_delta,
                abs_d - 0.5 * huber_delta,
            )
            loss_reg = lambda_reg * float(np.sum(huber_vals))

            g = np.clip(diffs / huber_delta, -1.0, 1.0)
            grad_reg = np.empty(n, dtype=np.float64)
            grad_reg[0] = -g[0]
            if n > 2:
                grad_reg[1:-1] = g[:-1] - g[1:]
            grad_reg[-1] = g[-1]
            grad_reg *= lambda_reg
        else:
            loss_reg = 0.0
            grad_reg = np.zeros(n, dtype=np.float64)

        # 3. Rolling bulk-volume tether
        if lambda_vol > 0.0 and len(vol_kernel) > 1:
            vol_diff = apply_vrf_forward(x - y, vol_kernel)
            loss_vol = 0.5 * lambda_vol * float(np.dot(vol_diff, vol_diff))
            grad_vol = lambda_vol * apply_vrf_adjoint(vol_diff, vol_kernel)
        else:
            loss_vol = 0.0
            grad_vol = np.zeros(n, dtype=np.float64)

        return loss_fid + loss_reg + loss_vol, grad_fid + grad_reg + grad_vol

    opt = minimize(
        objective_and_grad,
        x_init,
        method="L-BFGS-B",
        jac=True,
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6},
    )
    return opt.x, bool(opt.success), int(opt.nit)


def deconvolve_curve_1d(
    y: np.ndarray,
    dz_m: float,
    spec: ToolResolutionSpec,
    target_fwhm_m: float,
    weights: np.ndarray | None = None,
    lambda_reg: float | None = None,
    morozov_tau: float = 1.05,
    huber_delta_mult: float = 2.0,
    lambda_vol: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, CurveDeconvDiagnostics]:
    """Deconvolve a 1D nuclear curve with NaN segmentation and Morozov auto-regularization."""
    n = len(y)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)

    eff_sq = max(spec.fwhm_m ** 2 - target_fwhm_m ** 2, 0.0)
    eff_fwhm_m = float(np.sqrt(eff_sq))
    kernel = build_vrf_kernel(eff_fwhm_m, dz_m, vrf_type=spec.vrf_type)
    vol_kernel = build_vrf_kernel(3.0, dz_m, vrf_type="gaussian")

    # Interpolate tiny <=2 sample gaps; split across larger NaN intervals
    y_work = pd.Series(y, dtype=float).interpolate(limit=2, limit_area="inside").to_numpy()
    orig_nan = np.isnan(y)
    valid_mask = ~np.isnan(y_work)

    x_out = np.full(n, np.nan, dtype=np.float64)
    y_recon = np.full(n, np.nan, dtype=np.float64)

    if not np.any(valid_mask):
        diag = CurveDeconvDiagnostics(
            curve_name=spec.curve_name,
            intrinsic_fwhm_m=spec.fwhm_m,
            target_fwhm_m=target_fwhm_m,
            effective_fwhm_m=eff_fwhm_m,
            estimated_noise_std=np.nan,
            lambda_reg=np.nan,
            huber_delta=np.nan,
            weighted_recon_rmse=np.nan,
            relative_volume_bias_pct=np.nan,
            converged=False,
            iterations=0,
        )
        return x_out, y_recon, diag

    sigma_n = estimate_noise_mad(y_work[valid_mask], weights[valid_mask], spec.noise_floor)
    huber_delta = huber_delta_mult * sigma_n

    # Identify contiguous valid slices
    padded = np.concatenate(([False], valid_mask, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])

    total_iters = 0
    all_converged = True
    chosen_lambda = lambda_reg if lambda_reg is not None else sigma_n

    for s, e in zip(starts, ends):
        seg_y = y_work[s:e]
        seg_w = weights[s:e]
        if len(seg_y) < len(kernel) + 4:
            x_out[s:e] = np.clip(seg_y, spec.lower_bound, spec.upper_bound)
            y_recon[s:e] = apply_vrf_forward(x_out[s:e], kernel)
            continue

        if lambda_reg is None:
            # 6-step log-bisection using Morozov discrepancy principle
            target_rmse = morozov_tau * sigma_n
            lo_log, hi_log = np.log10(1e-4 * sigma_n), np.log10(20.0 * sigma_n)
            x_warm = None
            for _ in range(6):
                mid_log = 0.5 * (lo_log + hi_log)
                cand_lam = 10.0 ** mid_log
                x_cand, conv, nit = _solve_segment_huber(
                    seg_y, seg_w, kernel, vol_kernel,
                    spec.lower_bound, spec.upper_bound,
                    cand_lam, huber_delta, lambda_vol=lambda_vol, x0=x_warm,
                )
                x_warm = x_cand
                total_iters += nit
                seg_rec = apply_vrf_forward(x_cand, kernel)
                rmse = float(np.sqrt(np.sum((seg_w * (seg_rec - seg_y)) ** 2) / np.sum(seg_w ** 2)))
                if rmse < target_rmse:
                    lo_log = mid_log  # Under-regularized (overfitting noise) -> increase lambda
                else:
                    hi_log = mid_log  # Over-regularized -> decrease lambda
            chosen_lambda = 10.0 ** (0.5 * (lo_log + hi_log))
            seg_x, conv, nit = _solve_segment_huber(
                seg_y, seg_w, kernel, vol_kernel,
                spec.lower_bound, spec.upper_bound,
                chosen_lambda, huber_delta, lambda_vol=lambda_vol, x0=x_warm,
            )
            total_iters += nit
        else:
            seg_x, conv, nit = _solve_segment_huber(
                seg_y, seg_w, kernel, vol_kernel,
                spec.lower_bound, spec.upper_bound,
                lambda_reg, huber_delta, lambda_vol=lambda_vol,
            )
            total_iters += nit

        all_converged = all_converged and conv
        x_out[s:e] = seg_x
        y_recon[s:e] = apply_vrf_forward(seg_x, kernel)

    x_out[orig_nan] = np.nan
    y_recon[orig_nan] = np.nan

    eval_mask = ~orig_nan
    w_eval = weights[eval_mask]
    final_rmse = float(
        np.sqrt(np.sum((w_eval * (y_recon[eval_mask] - y[eval_mask])) ** 2) / np.sum(w_eval ** 2))
    )
    denom_mean = float(np.mean(np.abs(y[eval_mask]))) + 1e-12
    vol_bias_pct = 100.0 * float(np.mean(x_out[eval_mask] - y[eval_mask])) / denom_mean

    diag = CurveDeconvDiagnostics(
        curve_name=spec.curve_name,
        intrinsic_fwhm_m=spec.fwhm_m,
        target_fwhm_m=target_fwhm_m,
        effective_fwhm_m=eff_fwhm_m,
        estimated_noise_std=sigma_n,
        lambda_reg=float(chosen_lambda),
        huber_delta=huber_delta,
        weighted_recon_rmse=final_rmse,
        relative_volume_bias_pct=vol_bias_pct,
        converged=all_converged,
        iterations=total_iters,
    )
    return x_out, y_recon, diag


def check_interlog_depth_alignment(
    df: pd.DataFrame,
    ref_col: str = "RHOB_ENH",
    compare_cols: tuple[str, ...] = ("GR_ENH", "NPHI_ENH"),
    max_lag_samples: int = 5,
) -> dict[str, int]:
    """Detect residual micro-depth shifts via cross-correlation of absolute vertical gradients."""
    lags_found: dict[str, int] = {}
    if ref_col not in df.columns:
        return lags_found

    ref_grad = np.abs(np.gradient(np.nan_to_num(df[ref_col].to_numpy(dtype=float))))
    ref_grad -= np.mean(ref_grad)

    for col in compare_cols:
        if col not in df.columns:
            continue
        cmp_grad = np.abs(np.gradient(np.nan_to_num(df[col].to_numpy(dtype=float))))
        cmp_grad -= np.mean(cmp_grad)
        xcorr = correlate(cmp_grad, ref_grad, mode="full")
        mid = len(ref_grad) - 1
        window = xcorr[mid - max_lag_samples : mid + max_lag_samples + 1]
        best_lag = int(np.argmax(window) - max_lag_samples)
        lags_found[col] = best_lag
        if best_lag != 0:
            warnings.warn(
                f"Micro-depth shift detected: {col} lags {ref_col} by {best_lag} sample(s). "
                "Depth-match raw curves before running Thomas-Stieber!",
                UserWarning,
                stacklevel=2,
            )
    return lags_found


def harmonize_nuclear_logs(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    depth_uom: Literal["m", "ft"] = "m",
    mode: HarmonizeMode = "enhance_all",
    target_fwhm_m: float | None = None,
    specs: dict[str, ToolResolutionSpec] | None = None,
    nyquist_factor: float = 2.0,
) -> HarmonizationResult:
    """Enhance and resolution-match GR, NPHI, and RHOB onto an identical vertical footprint."""
    out_df = df.copy()
    active_specs = specs or DEFAULT_LEGACY_SPECS

    z_raw = out_df[depth_col].to_numpy(dtype=np.float64)
    dz_raw = float(np.median(np.diff(z_raw)))
    if np.max(np.abs(np.diff(z_raw) - dz_raw)) > 1e-2 * dz_raw:
        raise ValueError(f"{depth_col} must be uniformly sampled prior to 1D deconvolution.")

    dz_m = dz_raw * 0.3048 if depth_uom == "ft" else dz_raw
    nyquist_floor_m = nyquist_factor * dz_m

    present_specs = {k: v for k, v in active_specs.items() if k in out_df.columns}
    if target_fwhm_m is None:
        if mode == "match_finest":
            finest = min(s.fwhm_m for s in present_specs.values())
            target_fwhm_m = max(finest, nyquist_floor_m)
        else:
            target_fwhm_m = nyquist_floor_m
    elif target_fwhm_m < nyquist_floor_m:
        warnings.warn(
            f"Requested target_fwhm_m ({target_fwhm_m:.3f} m) violates Nyquist floor "
            f"({nyquist_floor_m:.3f} m); clamping to {nyquist_floor_m:.3f} m.",
            UserWarning,
            stacklevel=2,
        )
        target_fwhm_m = nyquist_floor_m

    w_cali, w_both = compute_borehole_weights(out_df)
    out_df["DECONV_WEIGHT"] = w_both

    diagnostics: dict[str, CurveDeconvDiagnostics] = {}
    for curve_name, spec in present_specs.items():
        w_curve = w_both if spec.apply_drho_weight else w_cali
        x_enh, y_rec, diag = deconvolve_curve_1d(
            y=out_df[curve_name].to_numpy(dtype=np.float64),
            dz_m=dz_m,
            spec=spec,
            target_fwhm_m=target_fwhm_m,
            weights=w_curve,
        )
        out_df[f"{curve_name}_ENH"] = x_enh
        out_df[f"{curve_name}_RECON"] = y_rec
        diagnostics[curve_name] = diag

    lags = check_interlog_depth_alignment(out_df)
    return HarmonizationResult(
        df=out_df,
        target_fwhm_m=target_fwhm_m,
        depth_step_m=dz_m,
        diagnostics=diagnostics,
        alignment_lags_samples=lags,
    )
```

### 6.3 Seamless Handoff to `quick_pp` Thomas-Stieber Evaluation

In downstream notebooks or `quick_pp` pipelines, pass the harmonized curves (`GR_ENH`, `NPHI_ENH`, `RHOB_ENH`) directly into shale-volume and porosity estimators prior to Thomas-Stieber partitioning:

```python
from quick_pp.conditioning import harmonize_nuclear_logs

# 1. Harmonize raw legacy GR, NPHI, and RHOB onto a 0.305 m (1.0 ft) common vertical footprint
harm_res = harmonize_nuclear_logs(well_df, depth_col="DEPTH", depth_uom="m", mode="enhance_all")
df_cond = harm_res.df

# 2. Compute VSH and PHIT from resolution-matched curves (no boomerang crossplot scatter)
# Pass df_cond["GR_ENH"], df_cond["NPHI_ENH"], df_cond["RHOB_ENH"] into quick_pp.lithology / porosity
# before calling Thomas-Stieber (VSH_LAM, VSH_DISP, PHIT_TS, PHISAND_TS).
```

---

## 7. Unit Testing, Synthetic LRLC Benchmark & QAQC Acceptance Criteria

Add `tests/test_conditioning_deconv.py` to your `pytest` suite covering five mandatory verification gates:

| Test Case | Setup & Verification Method | Quantitative Pass Criterion |
| :--- | :--- | :--- |
| **1. Exact Adjoint Identity (`test_vrf_adjoint_exactness`)** | Generate random vectors $\mathbf{x}, \mathbf{r} \in \mathbb{R}^{200}$ and both odd-length Gaussian and asymmetric kernels $\mathbf{k}$. Compare $\langle \mathbf{H}\mathbf{x}, \mathbf{r} \rangle$ vs. $\langle \mathbf{x}, \mathbf{H}^T\mathbf{r} \rangle$, plus `scipy.optimize.check_grad` on `_solve_segment_huber`. | Absolute inner-product error $< 10^{-13}$; analytical gradient error $< 10^{-7}$. |
| **2. Constant & Ramp Invariance (`test_boundary_droop_free`)** | Pass a constant shale baseline ($\text{GR} = 110\;\text{API}$) and a linear fining-upward ramp across top/bottom log boundaries. | Zero top/base boundary droop ($< 10^{-5}$ error on constant log; no stair-casing artifacts on ramp). |
| **3. Thin-Bed Step Recovery & Gibbs Suppression (`test_lrlc_thin_bed_no_overshoot`)** | Construct a synthetic $30\;\text{m}$ laminated LRLC sand-shale earth model ($0.45\;\text{m}$ alternating beds: clean sand $\rho_b = 2.22\;\text{g/cm}^3, \phi_N = 0.27, \text{GR} = 25\;\text{API}$; shale $\rho_b = 2.55\;\text{g/cm}^3, \phi_N = 0.36, \text{GR} = 115\;\text{API}$; plus one tight calcite streak $\rho_b = 2.71\;\text{g/cm}^3$). Convolve each with its respective legacy $FWHM_{\text{tool}}$ and add realistic Gaussian noise. | Peak overshoot/undershoot adjacent to tight calcite streak $< 2.0\%$ of step contrast; $> 55\%$ reduction in MAE against $0.305\;\text{m}$ target truth. |
| **4. Washout Isolation (`test_washout_ringing_isolation`)** | Inject a $0.6\;\text{m}$ washout spike ($\text{CALI} - \text{BS} = 5.5\;\text{in}$, $\text{DRHO} = -0.28\;\text{g/cm}^3$, $\text{RHOB} = 1.82\;\text{g/cm}^3$) inside a homogeneous $2.35\;\text{g/cm}^3$ sand bed. | Adjacent good-hole samples ($> 0.6\;\text{m}$ from washout) deviate by $< 0.01\;\text{g/cm}^3$ from true sand density (zero washout halo). |
| **5. Bulk-Volume & Thomas-Stieber Collapse (`test_ts_boomerang_collapse`)** | Compute $\phi_t$ and $V_{sh}$ on the mismatched synthetic legacy suite vs. `harmonize_nuclear_logs` output. Check macro-interval integral bias and point-to-line distance from the theoretical laminated sand-shale line on the Thomas-Stieber crossplot. | Relative bulk integral bias `abs(relative_volume_bias_pct)` $< 0.5\%$; RMS perpendicular scatter off the Thomas-Stieber laminated trend reduced by $> 65\%$. |

---

## 8. Optional Frontend / Inspector Hook (`quick_pp/app`)

If exposing this module in `quick_pp`'s FastAPI backend and SvelteKit 5 UI, provide a lightweight **Resolution Conditioning Inspector** view displaying:
1. **3-Track Overlay:** Raw curve (`dashed`), Enhanced curve (`solid`), and `DECONV_WEIGHT` shading in a narrow callout track.
2. **Forward Parity Residual Track:** Plotting $y_i - (\mathbf{H}\mathbf{x}^*)_i$ bounded by $\pm 2\hat{\sigma}_n$ bands so the petrophysicist can visually confirm that residuals consist purely of uncorrelated counting noise rather than unmodeled bed boundaries.
3. **Live Before/After Thomas-Stieber Crossplot:** Showing raw mismatched points collapsing onto the laminated line as resolution harmonization is applied.