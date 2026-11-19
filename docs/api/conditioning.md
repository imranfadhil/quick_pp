# Raw Log Resolution Enhancement (`quick_pp.conditioning`)

1D constrained deconvolution and inter-sensor resolution harmonization for the
nuclear logs `GR`, `NPHI`, and `RHOB`.

## Why this exists

`GR`, `NPHI`, and `RHOB` investigate different vertical rock volumes. Their
intrinsic vertical resolutions differ (legacy tools: `GR` ≈ 0.76 m, `NPHI` ≈
0.55 m, `RHOB` ≈ 0.40 m FWHM), so bed-boundary transitions shift at different
rates on the same depth grid. On a Thomas-Stieber $\phi_t$ vs. $V_{sh}$
crossplot this mismatch scatters laminated points into unphysical "boomerang"
loops that reach into the dispersed-clay region.

`harmonize_nuclear_logs` deconvolves all three curves onto a **common target
vertical resolution** so their transitions align. This is not a
Thomas-Stieber-only feature: it operates on raw logs *before* any non-linear
transform, so lithology inversion, porosity, saturation, permeability, and rock
typing all consume better-aligned inputs. Thomas-Stieber is simply the module
where the mismatch is most visible.

## Design principles

- **Harmonization over blind over-sharpening.** Curves are deconvolved onto a
  common `FWHM_target`, never toward a Dirac impulse, and never below the
  Shannon-Nyquist floor `2·Δz`.
- **Edge-preserving and bounded.** A normalized Huber penalty suppresses noise
  on flat beds while permitting crisp jumps at bed boundaries, and hard
  petrophysical box bounds prevent unphysical excursions.
- **Raw curves are never modified.** Outputs are additive: `<CURVE>_ENH`
  (enhanced), `<CURVE>_RECON` (forward-convolved parity), and `DECONV_WEIGHT`
  (composite borehole reliability weight).

## Forward model

Each measurement is modelled as the true formation property convolved with a
Gaussian vertical response function (VRF) plus noise:

$$
\mathbf{y} = \mathbf{H}\mathbf{x} + \mathbf{n}
$$

The effective kernel links the target resolution to the measurement:

$$
\mathbf{H}_{\text{eff}} = \text{VRF}\!\left(\sqrt{\text{FWHM}_{\text{tool}}^2 - \text{FWHM}_{\text{target}}^2}\right)
$$

Boundary handling uses edge-replicate padding, so constant and linear-ramp logs
are reproduced exactly (no top/base droop).

## Inverse problem

$$
\min_{\mathbf{x}} \; \tfrac{1}{2}\|\mathbf{W}(\mathbf{H}\mathbf{x} - \mathbf{y})\|_2^2
\;+\; \lambda \sum_i \phi_\delta\!\left((\mathbf{D}_1\mathbf{x})_i\right)
\;+\; \tfrac{\lambda_v}{2}\|\mathbf{H}_v(\mathbf{x} - \mathbf{y})\|_2^2
\quad \text{s.t.} \quad \mathbf{l} \le \mathbf{x} \le \mathbf{u}
$$

with the normalized Huber function

$$
\phi_\delta(d) =
\begin{cases}
\dfrac{d^2}{2\delta}, & |d| \le \delta \\[4pt]
|d| - \dfrac{\delta}{2}, & |d| > \delta
\end{cases}
$$

and $\delta = c_\delta \hat{\sigma}_n$. The third term is a wide rolling
bulk-volume conservation tether that keeps the macro-interval integral stable.

Solved with `scipy.optimize.minimize(method="L-BFGS-B", jac=True)` using an
analytically exact adjoint of the forward operator.

## Noise estimation and Morozov auto-tuning

Sensor noise is estimated from the median absolute deviation of second
differences, which cancels linear geological trends:

$$
\hat{\sigma}_n = \frac{\operatorname{MAD}(\mathbf{D}_2\mathbf{y})}{0.6745\sqrt{6}}
$$

When `lambda_reg` is omitted, $\lambda$ is auto-tuned by log-space bisection so
the weighted reconstruction RMSE converges to $\tau\hat{\sigma}_n$
(Morozov discrepancy principle, default $\tau = 1.05$).

!!! warning "Deviation from the reference specification"
    In densely laminated intervals (beds only a few samples thick) most samples
    sit near a transition, so the second-difference MAD is inflated by bed
    signal rather than counting noise. The estimate is therefore capped at
    `noise_cap_mult` times the tool noise floor (default
    `DEFAULT_NOISE_CAP_MULT = 3.0`; pass `None` to disable the cap). Without the
    cap, the Morozov target is set too high and `NPHI`/`RHOB` are
    over-regularized (MAE regresses versus the raw logs). The cap preserves the
    intent of the discrepancy principle: the target must reflect true sensor
    noise.

## Pre-conditioning guards

1. **Depth-grid uniformity.** Non-uniform sampling is linearly resampled onto a
   uniform grid with a warning, or rejected when `resample_nonuniform=False`.
2. **Nyquist floor.** `FWHM_target ≥ nyquist_factor · Δz` (default factor 2.0).
   Requests below the floor are clamped with a warning.
3. **NaN segmentation.** Gaps of ≤ 2 samples are interpolated; larger gaps split
   the log into contiguous segments solved independently, so `NaN` boundaries
   never contaminate adjacent valid data.
4. **Inter-log depth alignment.** `check_interlog_depth_alignment` cross-correlates
   $|\partial_z \text{GR}|$, $|\partial_z \text{RHOB}|$, and $|\partial_z \text{NPHI}|$
   over lags $[-5\Delta z, +5\Delta z]$ and warns when the peak correlation sits
   at a non-zero lag. Enhancing resolution amplifies depth-shift errors, so this
   diagnostic matters.
5. **Borehole reliability weights.** Caliper excess over bit size and the
   density correction `DRHO` produce continuous weights in `[0, 1]`. Washouts
   are downweighted so a spike is not misread as an ultra-porous thin bed.

## Usage

```python
from quick_pp.conditioning import harmonize_nuclear_logs

result = harmonize_nuclear_logs(
    df,
    depth_col="DEPTH",
    mode="enhance_all",          # or "match_finest"
    method="constrained_huber",  # or "wiener" for a fast preview
    noise_cap_mult=3.0,          # None disables the MAD noise cap
)

result.df["GR_ENH"]        # enhanced GR
result.df["RHOB_RECON"]    # forward-convolved parity curve
result.df["DECONV_WEIGHT"] # composite borehole reliability weight
result.target_fwhm_m       # common target resolution (m)
result.diagnostics["GR"]   # per-curve convergence, lambda, noise, volume bias
result.alignment_lags_samples
result.warnings
```

`mode="enhance_all"` sharpens every curve onto the Nyquist floor (0.3048 m for a
0.1524 m LAS). `mode="match_finest"` sharpens onto the finest intrinsic tool
resolution (0.40 m, set by `RHOB`), which is a gentler, more conservative
harmonization.

## Verification gates

| Gate | Criterion |
| --- | --- |
| Adjoint identity | $\langle \mathbf{H}\mathbf{x}, \mathbf{r}\rangle = \langle \mathbf{x}, \mathbf{H}^T\mathbf{r}\rangle$ to machine precision |
| Constant & ramp invariance | Zero boundary droop on constant logs; no stair-casing on ramps |
| Thin-bed recovery | MAE reduction vs. target truth; Gibbs overshoot < 2% of step contrast |
| Washout isolation | Ringing lobes adjacent to washouts < 0.01 g/cc |
| Bulk volume & TS collapse | `abs(relative_volume_bias_pct) < 0.5%`; Thomas-Stieber laminated-line scatter reduced |

## Workspace integration

The single-well **Data & QAQC** step exposes this through
`POST /quick_pp/workspace/projects/{project_id}/wells/{well_name}/resolution_enhancement`.
It is a raw-log conditioning step, so it lives alongside Badhole Flagging and
Environmental Corrections rather than inside any single downstream module, and
its outputs are grouped under the QAQC & Conditioning curve group.

`use_enhanced_logs=true` makes the downstream evaluators consume
`GR_ENH`/`NPHI_ENH`/`RHOB_ENH` instead of the raw curves. It is honored by:

| Endpoint | How the enhanced curves enter |
| --- | --- |
| `thomas_stieber` | NPHI/RHOB via `_select_nphi_rhob` |
| `lithology` | NPHI/RHOB for SSC and multi-mineral; GR for VSH_GR and badhole repair |
| `porosity` | NPHI/RHOB via `_select_nphi_rhob` |
| `saturation` | NPHI/RHOB for the Juhasz Qvn route; inherits enhancement transitively elsewhere through PHIT/VCLAY |

Permeability takes no flag of its own: it consumes only `PHIT`, `VCLAY`, and
`VSILT`, so it follows whatever the lithology and porosity runs produced.

The toggle is explicit and defaults to off, so enhancement is always a
deliberate choice rather than an implicit side effect of running it. The
**Resolution Enhanced vs. Raw** crossplot in the right pane renders each nuclear
curve against depth with the raw trace, the enhanced trace, and the
forward-convolved `*_RECON` parity curve, which makes the deconvolution
verifiable rather than merely asserted.

::: quick_pp.conditioning
