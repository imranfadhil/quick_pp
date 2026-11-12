"""Raw log resolution enhancement and inter-sensor resolution harmonization.

Implements 1D constrained deconvolution of legacy nuclear logs (`GR`, `NPHI`,
`RHOB`) onto a common target vertical resolution so that thin-bed (laminated
shaly-sand) evaluation is not corrupted by shoulder-bed averaging or by
inter-sensor vertical response mismatch.

Two distinct vertical resolution problems are addressed:

1. **Shoulder-bed averaging** - intrinsic tool vertical resolutions
   (:math:`0.38`-:math:`0.90` m) smear thin sand-shale interbeds toward
   intermediate shaly-sand values.
2. **Inter-sensor resolution mismatch** - `GR`, `NPHI`, and `RHOB` investigate
   different vertical rock volumes, so bed-boundary transitions shift at
   different rates. On a Thomas-Stieber (:math:`\\phi_t` vs. :math:`V_{sh}`)
   crossplot this mismatch scatters laminated points into unphysical
   "boomerang" loops inside the dispersed-clay region.

Design constraints (see the technical specification for the full derivation):

* **Raw-log enhancement only** - deconvolution runs on conditioned raw logs
  *before* non-linear petrophysical transforms.
* **Target resolution harmonization** - all curves are deconvolved onto a
  common :math:`FWHM_{\\text{target}}` subject to the Shannon-Nyquist sampling
  bound of the input LAS depth step.
* **Edge-preserving and bounded inversion** - hard petrophysical box bounds plus
  a normalized Huber edge-preserving penalty that suppresses counting noise on
  flat beds, preserves sharp bed boundaries, and avoids stair-casing.
* **Bad-hole isolation and bulk-volume conservation** - washout intervals are
  downweighted in the data-fidelity term and a rolling bulk-volume tether
  anchors macro-interval bulk properties.
* **Zero heavy dependencies** - vectorized `numpy`/`scipy` only. Raw curves are
  never modified; outputs are emitted as explicit columns.

Typical usage::

    from quick_pp.conditioning import harmonize_nuclear_logs

    result = harmonize_nuclear_logs(well_df, depth_col="DEPTH", mode="enhance_all")
    df_cond = result.df  # adds GR_ENH, NPHI_ENH, RHOB_ENH, *_RECON, DECONV_WEIGHT
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.signal import convolve, correlate

from quick_pp import logger

VrfType = Literal["gaussian", "double_exp"]
HarmonizeMode = Literal["enhance_all", "match_finest"]
SolverMethod = Literal["constrained_huber", "wiener"]

#: Shannon-Nyquist guardrail multiplier applied to the input depth step.
DEFAULT_NYQUIST_FACTOR = 2.0

#: Rolling bulk-volume conservation window (metres), roughly 4x a legacy tool FWHM.
DEFAULT_VOLUME_WINDOW_M = 3.0

#: Default weight of the rolling bulk-volume conservation tether.
DEFAULT_LAMBDA_VOL = 0.05

#: Default Morozov discrepancy target multiplier.
DEFAULT_MOROZOV_TAU = 1.05

#: Default Huber threshold multiplier applied to the estimated noise standard deviation.
DEFAULT_HUBER_DELTA_MULT = 2.0

#: Default cap on the MAD noise estimate, expressed as a multiple of the tool noise floor.
#: Dense thin-bed sequences (beds only a few samples thick) contaminate second differences
#: with bed signal, inflating the MAD estimate and over-regularizing the inversion. Capping
#: the estimate keeps the Morozov target anchored to a physically plausible sensor noise level.
DEFAULT_NOISE_CAP_MULT = 3.0

#: Minimum reliability weight assigned to any sample.
DEFAULT_MIN_WEIGHT = 0.05

_METRES_PER_FOOT = 0.3048


@dataclass(frozen=True)
class ToolResolutionSpec:
    """Vertical resolution and physical constraint specification for one nuclear log.

    Attributes:
        curve_name: Log mnemonic the specification applies to.
        fwhm_m: Intrinsic tool full width at half maximum (metres).
        lower_bound: Hard lower petrophysical bound for the deconvolved curve.
        upper_bound: Hard upper petrophysical bound for the deconvolved curve.
        noise_floor: Minimum plausible sensor noise standard deviation, which
            prevents zero-noise singularities on synthetic or pre-filtered curves.
        vrf_type: Vertical response function family (``"gaussian"`` or ``"double_exp"``).
        apply_drho_weight: Whether the density-correction reliability weight applies.
    """

    curve_name: str
    fwhm_m: float
    lower_bound: float
    upper_bound: float
    noise_floor: float
    vrf_type: VrfType = "gaussian"
    apply_drho_weight: bool = True


#: Default legacy wireline tool resolutions and physical bounds.
DEFAULT_LEGACY_SPECS: Dict[str, ToolResolutionSpec] = {
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

    def as_dict(self) -> Dict[str, float | bool | str]:
        """Return the diagnostics as a plain dictionary for JSON serialization."""
        return {
            "curve_name": self.curve_name,
            "intrinsic_fwhm_m": self.intrinsic_fwhm_m,
            "target_fwhm_m": self.target_fwhm_m,
            "effective_fwhm_m": self.effective_fwhm_m,
            "estimated_noise_std": self.estimated_noise_std,
            "lambda_reg": self.lambda_reg,
            "huber_delta": self.huber_delta,
            "weighted_recon_rmse": self.weighted_recon_rmse,
            "relative_volume_bias_pct": self.relative_volume_bias_pct,
            "converged": self.converged,
            "iterations": self.iterations,
        }


@dataclass
class HarmonizationResult:
    """Container returned by :func:`harmonize_nuclear_logs`.

    Attributes:
        df: Copy of the input DataFrame with enhanced, reconstructed, and weight columns.
        target_fwhm_m: Common target vertical resolution used for all curves (metres).
        depth_step_m: Uniform depth step of the input grid (metres).
        depth_uom: Depth unit detected or supplied (``"m"`` or ``"ft"``).
        diagnostics: Per-curve deconvolution diagnostics keyed by curve mnemonic.
        alignment_lags_samples: Residual micro-depth shift per enhanced curve (samples).
        warnings: Human-readable warnings raised during harmonization.
    """

    df: pd.DataFrame
    target_fwhm_m: float
    depth_step_m: float
    depth_uom: str = "m"
    diagnostics: Dict[str, CurveDeconvDiagnostics] = field(default_factory=dict)
    alignment_lags_samples: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Vertical response function kernels and exact forward / adjoint operators
# ---------------------------------------------------------------------------


def build_vrf_kernel(
    effective_fwhm: float,
    dz: float,
    vrf_type: VrfType = "gaussian",
    truncate_stds: float = 3.5,
) -> np.ndarray:
    """Construct a unit-sum discrete symmetric 1D vertical response function kernel.

    Args:
        effective_fwhm: Effective (residual) full width at half maximum in depth units.
        dz: Uniform depth step in the same units as ``effective_fwhm``.
        vrf_type: ``"gaussian"`` or ``"double_exp"`` (symmetric Laplacian).
        truncate_stds: Gaussian truncation half-width in standard deviations.

    Returns:
        np.ndarray: Odd-length, unit-sum kernel sampled at ``dz``. Returns ``[1.0]``
        when ``effective_fwhm`` is negligible (no residual smearing to invert).
    """
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
        raise ValueError(f"Unsupported vrf_type: {vrf_type!r}")

    kernel /= np.sum(kernel)
    return kernel


def apply_vrf_forward(x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute ``H @ x`` using edge-replicated padding to eliminate top/base log droop.

    Args:
        x: Target-resolution log samples.
        kernel: Unit-sum discrete VRF kernel.

    Returns:
        np.ndarray: Forward-modelled (tool-resolution) log of the same length as ``x``.
    """
    half_len = (len(kernel) - 1) // 2
    if half_len == 0:
        return np.asarray(x, dtype=np.float64).copy()
    x_pad = np.pad(np.asarray(x, dtype=np.float64), pad_width=half_len, mode="edge")
    return correlate(x_pad, kernel, mode="valid")


def apply_vrf_adjoint(r: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Compute the exact algebraic transpose ``H.T @ r`` of :func:`apply_vrf_forward`.

    The result matches the forward operator to machine precision, which is required
    for `L-BFGS-B` to converge without line-search failures.

    Args:
        r: Residual vector at tool resolution.
        kernel: Unit-sum discrete VRF kernel.

    Returns:
        np.ndarray: Adjoint vector of the same length as ``r``.
    """
    half_len = (len(kernel) - 1) // 2
    r = np.asarray(r, dtype=np.float64)
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


# ---------------------------------------------------------------------------
# Noise estimation and borehole reliability weights
# ---------------------------------------------------------------------------


def estimate_noise_mad(
    y: np.ndarray,
    weights: np.ndarray,
    noise_floor: float,
    cap_mult: Optional[float] = DEFAULT_NOISE_CAP_MULT,
) -> float:
    """Estimate sensor noise standard deviation via 2nd-order difference MAD.

    Second differences cancel linear geological trends, so the median absolute
    deviation of :math:`\\mathbf{D}_2\\mathbf{y}` is immune to isolated sharp bed
    boundaries. For white noise :math:`\\mathrm{Var}(n_{i+1} - 2n_i + n_{i-1}) = 6\\sigma_n^2`.

    In densely laminated intervals (beds only a few samples thick) most samples sit
    near a transition, so the MAD is inflated by bed signal rather than counting
    noise. ``cap_mult`` bounds the estimate to a multiple of ``noise_floor`` so the
    Morozov target stays anchored to a physically plausible sensor noise level.

    Args:
        y: Log samples.
        weights: Per-sample reliability weights; samples with ``w > 0.8`` are used.
        noise_floor: Minimum returned value, preventing zero-noise singularities.
        cap_mult: Optional cap expressed as a multiple of ``noise_floor``. ``None``
            disables the cap and returns the raw MAD estimate.

    Returns:
        float: Estimated noise standard deviation, floored at ``noise_floor``.
    """
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    good_mask = weights > 0.8
    y_eval = y[good_mask] if np.sum(good_mask) >= 15 else y
    y_eval = y_eval[np.isfinite(y_eval)]
    if len(y_eval) < 5:
        return float(noise_floor)
    d2 = y_eval[2:] - 2.0 * y_eval[1:-1] + y_eval[:-2]
    sigma_est = float(np.median(np.abs(d2)) / (0.67448975 * np.sqrt(6.0)))
    sigma_est = max(sigma_est, float(noise_floor))
    if cap_mult is not None and cap_mult > 0.0:
        sigma_est = min(sigma_est, float(cap_mult) * float(noise_floor))
    return sigma_est


def compute_borehole_weights(
    df: pd.DataFrame,
    cali_col: Optional[str] = "CALI",
    bs_col: Optional[str] = "BS",
    drho_col: Optional[str] = "DRHO",
    cali_tol: float = 2.0,
    cali_scale: float = 1.5,
    drho_tol: float = 0.10,
    drho_scale: float = 0.08,
    min_weight: float = DEFAULT_MIN_WEIGHT,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute continuous borehole reliability weights from caliper and density correction.

    Washouts cause sharp density drops and neutron spikes. If a washout sample is
    fully trusted, deconvolution misinterprets the spike as a smeared ultra-porous
    thin bed and creates ringing lobes in adjacent clean beds.

    Args:
        df: Well DataFrame containing optional ``CALI``, ``BS``, and ``DRHO`` columns.
        cali_col: Caliper column name (inches).
        bs_col: Bit-size column name (inches).
        drho_col: Density-correction column name (g/cc).
        cali_tol: Allowable ``CALI - BS`` excess before downweighting (inches).
        cali_scale: Gaussian decay scale for the caliper excess (inches).
        drho_tol: Allowable ``|DRHO|`` before downweighting (g/cc).
        drho_scale: Gaussian decay scale for the density correction (g/cc).
        min_weight: Floor applied to both returned weight arrays.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(w_cali, w_both)`` where ``w_cali`` excludes
        the density-correction term (used for `GR`) and ``w_both`` includes it.
    """
    n = len(df)
    w_cali = np.ones(n, dtype=np.float64)
    if cali_col and bs_col and cali_col in df.columns and bs_col in df.columns:
        excess = df[cali_col].to_numpy(dtype=float) - df[bs_col].to_numpy(dtype=float)
        excess = np.nan_to_num(excess, nan=0.0)
        over = np.maximum(0.0, (excess - cali_tol) / cali_scale)
        w_cali = np.exp(-(over**2))

    w_drho = np.ones(n, dtype=np.float64)
    if drho_col and drho_col in df.columns:
        abs_drho = np.abs(np.nan_to_num(df[drho_col].to_numpy(dtype=float), nan=0.0))
        over_d = np.maximum(0.0, (abs_drho - drho_tol) / drho_scale)
        w_drho = np.exp(-(over_d**2))

    w_cali = np.clip(w_cali, min_weight, 1.0)
    w_both = np.clip(w_cali * w_drho, min_weight, 1.0)
    return w_cali, w_both


# ---------------------------------------------------------------------------
# Constrained Huber solver
# ---------------------------------------------------------------------------


def _solve_segment_huber(
    y: np.ndarray,
    weights: np.ndarray,
    kernel: np.ndarray,
    vol_kernel: np.ndarray,
    lower_bound: float,
    upper_bound: float,
    lambda_reg: float,
    huber_delta: float,
    lambda_vol: float = DEFAULT_LAMBDA_VOL,
    x0: Optional[np.ndarray] = None,
    maxiter: int = 150,
) -> Tuple[np.ndarray, bool, int]:
    """Solve 1D bounded Huber-regularized deconvolution on a contiguous valid segment.

    Minimizes the weighted data-fidelity term, a normalized Huber edge-preserving
    prior, and a rolling bulk-volume conservation tether, subject to hard box bounds.

    Args:
        y: Measured tool-resolution samples for the segment.
        weights: Per-sample reliability weights in ``[0, 1]``.
        kernel: Effective VRF kernel linking the target log to the measurement.
        vol_kernel: Wide averaging kernel for the bulk-volume tether.
        lower_bound: Hard lower bound for the deconvolved log.
        upper_bound: Hard upper bound for the deconvolved log.
        lambda_reg: Huber regularization weight.
        huber_delta: Huber transition threshold in log units.
        lambda_vol: Bulk-volume conservation tether weight.
        x0: Optional warm-start solution.
        maxiter: Maximum `L-BFGS-B` iterations.

    Returns:
        tuple[np.ndarray, bool, int]: ``(x, converged, iterations)``.
    """
    n = len(y)
    w_sq = weights**2
    x_init = np.clip(y.copy() if x0 is None else x0.copy(), lower_bound, upper_bound)
    bounds = [(lower_bound, upper_bound)] * n

    def objective_and_grad(x: np.ndarray) -> Tuple[float, np.ndarray]:
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
                0.5 * (diffs**2) / huber_delta,
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

        # 3. Rolling bulk-volume conservation tether
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


def _weighted_rmse(recon: np.ndarray, y: np.ndarray, weights: np.ndarray) -> float:
    """Return the reliability-weighted RMS residual between reconstruction and measurement."""
    denom = float(np.sum(weights**2))
    if denom <= 0.0:
        return float("nan")
    return float(np.sqrt(np.sum((weights * (recon - y)) ** 2) / denom))


def _empty_diagnostics(
    spec: ToolResolutionSpec,
    target_fwhm_m: float,
    effective_fwhm_m: float,
) -> CurveDeconvDiagnostics:
    """Build an all-NaN diagnostics record for a curve with no valid samples."""
    return CurveDeconvDiagnostics(
        curve_name=spec.curve_name,
        intrinsic_fwhm_m=spec.fwhm_m,
        target_fwhm_m=target_fwhm_m,
        effective_fwhm_m=effective_fwhm_m,
        estimated_noise_std=np.nan,
        lambda_reg=np.nan,
        huber_delta=np.nan,
        weighted_recon_rmse=np.nan,
        relative_volume_bias_pct=np.nan,
        converged=False,
        iterations=0,
    )


def deconvolve_curve_1d(
    y: np.ndarray,
    dz_m: float,
    spec: ToolResolutionSpec,
    target_fwhm_m: float,
    weights: Optional[np.ndarray] = None,
    lambda_reg: Optional[float] = None,
    morozov_tau: float = DEFAULT_MOROZOV_TAU,
    huber_delta_mult: float = DEFAULT_HUBER_DELTA_MULT,
    lambda_vol: float = DEFAULT_LAMBDA_VOL,
    volume_window_m: float = DEFAULT_VOLUME_WINDOW_M,
    noise_cap_mult: Optional[float] = DEFAULT_NOISE_CAP_MULT,
    maxiter: int = 150,
) -> Tuple[np.ndarray, np.ndarray, CurveDeconvDiagnostics]:
    """Deconvolve a 1D nuclear curve with NaN segmentation and Morozov auto-regularization.

    Small gaps (``<= 2`` samples) are linearly interpolated before deconvolution and
    their ``NaN`` masks restored afterwards. Larger gaps split the log into contiguous
    valid segments that are deconvolved independently so ``NaN`` boundaries never
    contaminate adjacent valid data.

    When ``lambda_reg`` is ``None`` the regularization weight is auto-tuned by
    log-space bisection on the Morozov discrepancy principle, warm-starting each
    `L-BFGS-B` call from the previous solution.

    Args:
        y: Measured tool-resolution samples (may contain ``NaN``).
        dz_m: Uniform depth step in metres.
        spec: Tool resolution and physical bound specification.
        target_fwhm_m: Common target vertical resolution in metres.
        weights: Optional per-sample reliability weights in ``[0, 1]``.
        lambda_reg: Explicit Huber regularization weight; auto-tuned when ``None``.
        morozov_tau: Morozov discrepancy target multiplier.
        huber_delta_mult: Huber threshold multiplier applied to the noise estimate.
        lambda_vol: Bulk-volume conservation tether weight.
        volume_window_m: Rolling bulk-volume averaging window in metres.
        noise_cap_mult: Cap on the MAD noise estimate as a multiple of the tool noise
            floor; ``None`` disables the cap.
        maxiter: Maximum `L-BFGS-B` iterations per solve.

    Returns:
        tuple[np.ndarray, np.ndarray, CurveDeconvDiagnostics]: ``(x_enh, y_recon, diagnostics)``
        where ``x_enh`` is the enhanced target-resolution log and ``y_recon`` is the
        forward-convolved reconstruction for QAQC parity inspection.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    eff_sq = max(spec.fwhm_m**2 - target_fwhm_m**2, 0.0)
    eff_fwhm_m = float(np.sqrt(eff_sq))
    kernel = build_vrf_kernel(eff_fwhm_m, dz_m, vrf_type=spec.vrf_type)
    vol_kernel = build_vrf_kernel(volume_window_m, dz_m, vrf_type="gaussian")

    # Interpolate tiny <=2 sample gaps; split across larger NaN intervals
    y_work = (
        pd.Series(y, dtype=float).interpolate(limit=2, limit_area="inside").to_numpy()
    )
    orig_nan = np.isnan(y)
    valid_mask = ~np.isnan(y_work)

    x_out = np.full(n, np.nan, dtype=np.float64)
    y_recon = np.full(n, np.nan, dtype=np.float64)

    if not np.any(valid_mask):
        return x_out, y_recon, _empty_diagnostics(spec, target_fwhm_m, eff_fwhm_m)

    sigma_n = estimate_noise_mad(
        y_work[valid_mask],
        weights[valid_mask],
        spec.noise_floor,
        cap_mult=noise_cap_mult,
    )
    huber_delta = huber_delta_mult * sigma_n

    # Identify contiguous valid slices
    padded = np.concatenate(([False], valid_mask, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])

    total_iters = 0
    all_converged = True
    chosen_lambda = float(lambda_reg) if lambda_reg is not None else float(sigma_n)

    for s, e in zip(starts, ends, strict=False):
        seg_y = y_work[s:e]
        seg_w = weights[s:e]
        if len(seg_y) < len(kernel) + 4:
            x_out[s:e] = np.clip(seg_y, spec.lower_bound, spec.upper_bound)
            y_recon[s:e] = apply_vrf_forward(x_out[s:e], kernel)
            continue

        if lambda_reg is None:
            # Log-bisection on the Morozov discrepancy principle
            target_rmse = morozov_tau * sigma_n
            lo_log, hi_log = np.log10(1e-4 * sigma_n), np.log10(20.0 * sigma_n)
            x_warm: Optional[np.ndarray] = None
            for _ in range(6):
                mid_log = 0.5 * (lo_log + hi_log)
                cand_lam = 10.0**mid_log
                x_cand, _conv, nit = _solve_segment_huber(
                    seg_y,
                    seg_w,
                    kernel,
                    vol_kernel,
                    spec.lower_bound,
                    spec.upper_bound,
                    cand_lam,
                    huber_delta,
                    lambda_vol=lambda_vol,
                    x0=x_warm,
                    maxiter=maxiter,
                )
                x_warm = x_cand
                total_iters += nit
                seg_rec = apply_vrf_forward(x_cand, kernel)
                rmse = _weighted_rmse(seg_rec, seg_y, seg_w)
                if rmse < target_rmse:
                    # Under-regularized (overfitting noise) -> increase lambda
                    lo_log = mid_log
                else:
                    # Over-regularized -> decrease lambda
                    hi_log = mid_log
            chosen_lambda = 10.0 ** (0.5 * (lo_log + hi_log))
            seg_x, conv, nit = _solve_segment_huber(
                seg_y,
                seg_w,
                kernel,
                vol_kernel,
                spec.lower_bound,
                spec.upper_bound,
                chosen_lambda,
                huber_delta,
                lambda_vol=lambda_vol,
                x0=x_warm,
                maxiter=maxiter,
            )
            total_iters += nit
        else:
            seg_x, conv, nit = _solve_segment_huber(
                seg_y,
                seg_w,
                kernel,
                vol_kernel,
                spec.lower_bound,
                spec.upper_bound,
                float(lambda_reg),
                huber_delta,
                lambda_vol=lambda_vol,
                maxiter=maxiter,
            )
            total_iters += nit

        all_converged = all_converged and conv
        x_out[s:e] = seg_x
        y_recon[s:e] = apply_vrf_forward(seg_x, kernel)

    x_out[orig_nan] = np.nan
    y_recon[orig_nan] = np.nan

    eval_mask = ~orig_nan
    w_eval = weights[eval_mask]
    final_rmse = _weighted_rmse(y_recon[eval_mask], y[eval_mask], w_eval)
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


def deconvolve_curve_wiener(
    y: np.ndarray,
    dz_m: float,
    spec: ToolResolutionSpec,
    target_fwhm_m: float,
    weights: Optional[np.ndarray] = None,
    gamma: Optional[float] = None,
    noise_cap_mult: Optional[float] = DEFAULT_NOISE_CAP_MULT,
) -> Tuple[np.ndarray, np.ndarray, CurveDeconvDiagnostics]:
    """Fast closed-form FFT Tikhonov-Wiener deconvolution for interactive previews.

    Solves the spectral filter

    .. math::
        \\hat{X}(\\omega) = \\frac{H_{\\text{eff}}^*(\\omega)}
        {|H_{\\text{eff}}(\\omega)|^2 + \\gamma (2 - 2\\cos(\\omega \\Delta z))} Y(\\omega)

    and clips the result to the physical bounds. This is a preview-only method:
    it is not edge-preserving and can overshoot adjacent to high-contrast beds, so
    ``method="constrained_huber"`` must be used for final Thomas-Stieber evaluation.

    Args:
        y: Measured tool-resolution samples (may contain ``NaN``).
        dz_m: Uniform depth step in metres.
        spec: Tool resolution and physical bound specification.
        target_fwhm_m: Common target vertical resolution in metres.
        weights: Optional per-sample reliability weights (used only for the noise estimate).
        gamma: Explicit regularization factor; derived from the noise-to-signal ratio when ``None``.
        noise_cap_mult: Cap on the MAD noise estimate as a multiple of the tool
            noise floor; ``None`` disables the cap.

    Returns:
        tuple[np.ndarray, np.ndarray, CurveDeconvDiagnostics]: ``(x_enh, y_recon, diagnostics)``.
    """
    y = np.asarray(y, dtype=np.float64)
    n = len(y)
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)

    eff_sq = max(spec.fwhm_m**2 - target_fwhm_m**2, 0.0)
    eff_fwhm_m = float(np.sqrt(eff_sq))

    y_work = (
        pd.Series(y, dtype=float).interpolate(limit=2, limit_area="inside").to_numpy()
    )
    orig_nan = np.isnan(y)
    valid_mask = ~np.isnan(y_work)
    x_out = np.full(n, np.nan, dtype=np.float64)
    y_recon = np.full(n, np.nan, dtype=np.float64)

    if not np.any(valid_mask):
        return x_out, y_recon, _empty_diagnostics(spec, target_fwhm_m, eff_fwhm_m)

    sigma_n = estimate_noise_mad(
        y_work[valid_mask],
        weights[valid_mask],
        spec.noise_floor,
        cap_mult=noise_cap_mult,
    )
    signal_std = float(np.nanstd(y_work[valid_mask]))
    if gamma is None:
        gamma = (sigma_n / max(signal_std, 1e-12)) ** 2

    kernel = build_vrf_kernel(eff_fwhm_m, dz_m, vrf_type=spec.vrf_type)
    half_len = (len(kernel) - 1) // 2

    padded = np.concatenate(([False], valid_mask, [False]))
    starts = np.flatnonzero(padded[1:] & ~padded[:-1])
    ends = np.flatnonzero(~padded[1:] & padded[:-1])

    for s, e in zip(starts, ends, strict=False):
        seg = y_work[s:e]
        m = len(seg)
        if m < 4:
            x_out[s:e] = np.clip(seg, spec.lower_bound, spec.upper_bound)
            y_recon[s:e] = apply_vrf_forward(x_out[s:e], kernel)
            continue

        # Centred full-length kernel so the circular FFT convolution is symmetric
        k_full = np.zeros(m, dtype=np.float64)
        k_full[: half_len + 1] = kernel[half_len:]
        if half_len > 0:
            k_full[m - half_len :] = kernel[:half_len]

        h = np.fft.rfft(k_full)
        y_hat = np.fft.rfft(seg)
        omega = 2.0 * np.pi * np.fft.rfftfreq(m, d=dz_m)
        reg = gamma * (2.0 - 2.0 * np.cos(omega * dz_m))
        x_hat = np.conj(h) / (np.abs(h) ** 2 + reg + 1e-12) * y_hat
        seg_x = np.clip(np.fft.irfft(x_hat, n=m), spec.lower_bound, spec.upper_bound)
        x_out[s:e] = seg_x
        y_recon[s:e] = apply_vrf_forward(seg_x, kernel)

    x_out[orig_nan] = np.nan
    y_recon[orig_nan] = np.nan

    eval_mask = ~orig_nan
    final_rmse = _weighted_rmse(y_recon[eval_mask], y[eval_mask], weights[eval_mask])
    denom_mean = float(np.mean(np.abs(y[eval_mask]))) + 1e-12
    vol_bias_pct = 100.0 * float(np.mean(x_out[eval_mask] - y[eval_mask])) / denom_mean

    diag = CurveDeconvDiagnostics(
        curve_name=spec.curve_name,
        intrinsic_fwhm_m=spec.fwhm_m,
        target_fwhm_m=target_fwhm_m,
        effective_fwhm_m=eff_fwhm_m,
        estimated_noise_std=sigma_n,
        lambda_reg=float(gamma),
        huber_delta=np.nan,
        weighted_recon_rmse=final_rmse,
        relative_volume_bias_pct=vol_bias_pct,
        converged=True,
        iterations=0,
    )
    return x_out, y_recon, diag


# ---------------------------------------------------------------------------
# Micro-depth alignment diagnostic
# ---------------------------------------------------------------------------


def check_interlog_depth_alignment(
    df: pd.DataFrame,
    ref_col: str = "RHOB_ENH",
    compare_cols: Sequence[str] = ("GR_ENH", "NPHI_ENH"),
    max_lag_samples: int = 5,
    warn: bool = True,
) -> Dict[str, int]:
    """Detect residual micro-depth shifts via cross-correlation of absolute vertical gradients.

    Enhancing vertical resolution amplifies depth-shift errors: a half-foot mismatch
    between ``RHOB_ENH`` and ``NPHI_ENH`` creates severe false gas/shale spikes at
    every bed boundary.

    Args:
        df: DataFrame containing the enhanced curves.
        ref_col: Reference curve used as the alignment datum.
        compare_cols: Curves compared against ``ref_col``.
        max_lag_samples: Maximum absolute lag searched, in samples.
        warn: Emit a ``UserWarning`` when a non-zero lag is detected.

    Returns:
        dict[str, int]: Best-correlation lag in samples per compared curve.
    """
    lags_found: Dict[str, int] = {}
    if ref_col not in df.columns:
        return lags_found

    ref_grad = np.abs(np.gradient(np.nan_to_num(df[ref_col].to_numpy(dtype=float))))
    ref_grad = ref_grad - np.mean(ref_grad)

    for col in compare_cols:
        if col not in df.columns:
            continue
        cmp_grad = np.abs(np.gradient(np.nan_to_num(df[col].to_numpy(dtype=float))))
        cmp_grad = cmp_grad - np.mean(cmp_grad)
        xcorr = correlate(cmp_grad, ref_grad, mode="full")
        mid = len(ref_grad) - 1
        window = xcorr[mid - max_lag_samples : mid + max_lag_samples + 1]
        best_lag = int(np.argmax(window) - max_lag_samples)
        lags_found[col] = best_lag
        if best_lag != 0 and warn:
            warnings.warn(
                f"Micro-depth shift detected: {col} lags {ref_col} by {best_lag} sample(s). "
                "Depth-match raw curves before running Thomas-Stieber!",
                UserWarning,
                stacklevel=2,
            )
    return lags_found


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _detect_depth_uom(dz_raw: float) -> str:
    """Infer metric vs. imperial depth units from the median depth step.

    Args:
        dz_raw: Median depth step in the file's native units.

    Returns:
        str: ``"ft"`` when the step matches a common imperial increment, else ``"m"``.
    """
    imperial_steps = (0.25, 0.5, 1.0, 2.0)
    metric_steps = (0.0762, 0.1524, 0.3048, 0.6096)
    ft_err = min(abs(dz_raw - s) for s in imperial_steps)
    m_err = min(abs(dz_raw - s) for s in metric_steps)
    return "ft" if ft_err < m_err else "m"


def _resample_to_uniform_grid(
    df: pd.DataFrame,
    depth_col: str,
    columns: Sequence[str],
) -> Tuple[pd.DataFrame, float]:
    """Linearly resample selected columns onto a uniform grid at the median depth step.

    Args:
        df: Input DataFrame sorted by depth.
        depth_col: Depth column name.
        columns: Curve columns to resample.

    Returns:
        tuple[pd.DataFrame, float]: ``(resampled_df, step)`` with ``DEPTH`` plus the
        requested columns, and the uniform step in the input depth units.
    """
    z = df[depth_col].to_numpy(dtype=np.float64)
    diffs = np.diff(z)
    step = float(np.median(diffs[diffs > 0]))
    grid = np.arange(z[0], z[-1] + step * 0.5, step)
    out = pd.DataFrame({depth_col: grid})
    for col in columns:
        vals = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=np.float64)
        finite = np.isfinite(vals)
        if finite.sum() < 2:
            out[col] = np.nan
            continue
        out[col] = np.interp(grid, z[finite], vals[finite], left=np.nan, right=np.nan)
    return out, step


def harmonize_nuclear_logs(
    df: pd.DataFrame,
    depth_col: str = "DEPTH",
    depth_uom: Optional[Literal["m", "ft"]] = None,
    mode: HarmonizeMode = "enhance_all",
    target_fwhm_m: Optional[float] = None,
    specs: Optional[Dict[str, ToolResolutionSpec]] = None,
    curves: Optional[Sequence[str]] = None,
    method: SolverMethod = "constrained_huber",
    nyquist_factor: float = DEFAULT_NYQUIST_FACTOR,
    lambda_reg: Optional[float] = None,
    lambda_vol: float = DEFAULT_LAMBDA_VOL,
    morozov_tau: float = DEFAULT_MOROZOV_TAU,
    huber_delta_mult: float = DEFAULT_HUBER_DELTA_MULT,
    volume_window_m: float = DEFAULT_VOLUME_WINDOW_M,
    noise_cap_mult: Optional[float] = DEFAULT_NOISE_CAP_MULT,
    resample_nonuniform: bool = True,
    check_alignment: bool = True,
) -> HarmonizationResult:
    """Enhance and resolution-match `GR`, `NPHI`, and `RHOB` onto an identical vertical footprint.

    The raw curves are never modified. Enhanced curves are written as
    ``<CURVE>_ENH``, forward-convolved reconstructions as ``<CURVE>_RECON``, and the
    composite borehole reliability weight as ``DECONV_WEIGHT``.

    Args:
        df: Well DataFrame containing ``depth_col`` and the target curves.
        depth_col: Depth column name.
        depth_uom: Depth unit (``"m"`` or ``"ft"``); auto-detected from the depth step when ``None``.
        mode: ``"enhance_all"`` sharpens every curve onto the Nyquist floor;
            ``"match_finest"`` sharpens onto the finest intrinsic tool resolution.
        target_fwhm_m: Explicit target resolution in metres, overriding ``mode``.
        specs: Tool resolution specifications keyed by curve mnemonic.
        curves: Restrict processing to these mnemonics (defaults to all present specs).
        method: ``"constrained_huber"`` (production) or ``"wiener"`` (fast preview).
        nyquist_factor: Multiplier applied to the depth step for the target-resolution floor.
        lambda_reg: Explicit Huber regularization weight; auto-tuned per curve when ``None``.
        lambda_vol: Bulk-volume conservation tether weight.
        morozov_tau: Morozov discrepancy target multiplier.
        huber_delta_mult: Huber threshold multiplier applied to the noise estimate.
        volume_window_m: Rolling bulk-volume averaging window in metres.
        noise_cap_mult: Cap on the MAD noise estimate as a multiple of the tool noise
            floor; ``None`` disables the cap.
        resample_nonuniform: Linearly resample onto a uniform grid (with a warning) when
            the input depth step is not uniform.
        check_alignment: Run the inter-log micro-depth alignment diagnostic on the outputs.

    Returns:
        HarmonizationResult: Container with the conditioned DataFrame, target resolution,
        depth step, per-curve diagnostics, alignment lags, and collected warnings.

    Raises:
        ValueError: If the depth column is missing, fewer than two samples are present,
            or the depth grid is non-uniform and ``resample_nonuniform`` is ``False``.
    """
    if depth_col not in df.columns:
        raise ValueError(f"Depth column {depth_col!r} not found in DataFrame")
    if len(df) < 2:
        raise ValueError("At least two depth samples are required for deconvolution")

    out_df = df.copy()
    active_specs = specs or DEFAULT_LEGACY_SPECS
    collected_warnings: List[str] = []

    z_raw = pd.to_numeric(out_df[depth_col], errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(z_raw)):
        raise ValueError(f"{depth_col} contains non-finite values")
    if np.any(np.diff(z_raw) <= 0):
        out_df = out_df.sort_values(depth_col).reset_index(drop=True)
        z_raw = pd.to_numeric(out_df[depth_col], errors="coerce").to_numpy(
            dtype=np.float64
        )

    dz_raw = float(np.median(np.diff(z_raw)))
    if dz_raw <= 0:
        raise ValueError(f"{depth_col} must be strictly increasing")

    resolved_uom = depth_uom or _detect_depth_uom(dz_raw)
    dz_m = dz_raw * _METRES_PER_FOOT if resolved_uom == "ft" else dz_raw

    present_specs = {
        k: v
        for k, v in active_specs.items()
        if k in out_df.columns and (curves is None or k in curves)
    }
    if not present_specs:
        raise ValueError(
            "None of the requested nuclear curves are present in the DataFrame. "
            f"Available: {list(out_df.columns)}"
        )

    # Guard 1: depth grid uniformity
    if np.max(np.abs(np.diff(z_raw) - dz_raw)) > 1e-3 * dz_raw:
        if not resample_nonuniform:
            raise ValueError(
                f"{depth_col} must be uniformly sampled prior to 1D deconvolution."
            )
        msg = (
            f"{depth_col} is not uniformly sampled (median step {dz_raw:.4f}); "
            "linearly resampling onto a uniform grid before deconvolution."
        )
        collected_warnings.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        out_df, dz_raw = _resample_to_uniform_grid(
            out_df, depth_col, list(present_specs.keys())
        )
        dz_m = dz_raw * _METRES_PER_FOOT if resolved_uom == "ft" else dz_raw

    # Guard 2: Nyquist target-resolution floor
    nyquist_floor_m = nyquist_factor * dz_m
    if target_fwhm_m is None:
        if mode == "match_finest":
            finest = min(s.fwhm_m for s in present_specs.values())
            target_fwhm_m = max(finest, nyquist_floor_m)
        else:
            target_fwhm_m = nyquist_floor_m
    elif target_fwhm_m < nyquist_floor_m:
        msg = (
            f"Requested target_fwhm_m ({target_fwhm_m:.3f} m) violates the Nyquist floor "
            f"({nyquist_floor_m:.3f} m); clamping to {nyquist_floor_m:.3f} m."
        )
        collected_warnings.append(msg)
        warnings.warn(msg, UserWarning, stacklevel=2)
        target_fwhm_m = nyquist_floor_m

    w_cali, w_both = compute_borehole_weights(out_df)
    out_df["DECONV_WEIGHT"] = w_both

    diagnostics: Dict[str, CurveDeconvDiagnostics] = {}
    solver = deconvolve_curve_wiener if method == "wiener" else deconvolve_curve_1d

    for curve_name, spec in present_specs.items():
        w_curve = w_both if spec.apply_drho_weight else w_cali
        y = pd.to_numeric(out_df[curve_name], errors="coerce").to_numpy(
            dtype=np.float64
        )
        if solver is deconvolve_curve_wiener:
            x_enh, y_rec, diag = solver(
                y=y,
                dz_m=dz_m,
                spec=spec,
                target_fwhm_m=target_fwhm_m,
                weights=w_curve,
                noise_cap_mult=noise_cap_mult,
            )
        else:
            x_enh, y_rec, diag = solver(
                y=y,
                dz_m=dz_m,
                spec=spec,
                target_fwhm_m=target_fwhm_m,
                weights=w_curve,
                lambda_reg=lambda_reg,
                morozov_tau=morozov_tau,
                huber_delta_mult=huber_delta_mult,
                lambda_vol=lambda_vol,
                volume_window_m=volume_window_m,
                noise_cap_mult=noise_cap_mult,
            )
        out_df[f"{curve_name}_ENH"] = x_enh
        out_df[f"{curve_name}_RECON"] = y_rec
        diagnostics[curve_name] = diag
        logger.debug(
            f"[conditioning] {curve_name}: FWHM {spec.fwhm_m:.3f} -> {target_fwhm_m:.3f} m, "
            f"lambda={diag.lambda_reg:.4g}, rmse={diag.weighted_recon_rmse:.4g}, "
            f"vol_bias={diag.relative_volume_bias_pct:+.3f}%"
        )

    lags: Dict[str, int] = {}
    if check_alignment:
        ref_col = (
            "RHOB_ENH"
            if "RHOB_ENH" in out_df.columns
            else f"{next(iter(present_specs))}_ENH"
        )
        compare = tuple(f"{c}_ENH" for c in present_specs if f"{c}_ENH" != ref_col)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            lags = check_interlog_depth_alignment(
                out_df, ref_col=ref_col, compare_cols=compare, warn=True
            )
            for item in caught:
                collected_warnings.append(str(item.message))

    return HarmonizationResult(
        df=out_df,
        target_fwhm_m=float(target_fwhm_m),
        depth_step_m=float(dz_m),
        depth_uom=resolved_uom,
        diagnostics=diagnostics,
        alignment_lags_samples=lags,
        warnings=collected_warnings,
    )


def plot_resolution_qc(
    result: HarmonizationResult,
    curves: Sequence[str] = ("GR", "NPHI", "RHOB"),
    depth_range: Optional[Tuple[float, float]] = None,
    figsize: Tuple[float, float] = (9.0, 8.0),
):
    """Plot raw vs. enhanced curves with forward-parity residual tracks.

    Produces one column per curve: the raw curve (dashed), the enhanced curve
    (solid), and the forward-parity residual :math:`y - Hx^*` bounded by
    :math:`\\pm 2\\hat{\\sigma}_n` bands so residuals can be visually confirmed to
    consist of uncorrelated counting noise rather than unmodelled bed boundaries.

    Args:
        result: Output of :func:`harmonize_nuclear_logs`.
        curves: Curve mnemonics to plot.
        depth_range: Optional ``(top, bottom)`` depth window.
        figsize: Matplotlib figure size.

    Returns:
        matplotlib.figure.Figure: The generated QAQC figure.
    """
    import matplotlib.pyplot as plt

    df = result.df
    depth_col = "DEPTH" if "DEPTH" in df.columns else df.columns[0]
    depth = pd.to_numeric(df[depth_col], errors="coerce").to_numpy(dtype=float)

    mask = np.isfinite(depth)
    if depth_range is not None:
        mask &= (depth >= depth_range[0]) & (depth <= depth_range[1])

    plotted = [c for c in curves if c in df.columns and f"{c}_ENH" in df.columns]
    if not plotted:
        raise ValueError("No enhanced curves available to plot")

    fig, axes = plt.subplots(
        1, len(plotted), figsize=figsize, sharey=True, squeeze=False
    )
    axes = axes[0]
    for ax, name in zip(axes, plotted, strict=False):
        raw = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
        enh = pd.to_numeric(df[f"{name}_ENH"], errors="coerce").to_numpy(dtype=float)
        rec = pd.to_numeric(df[f"{name}_RECON"], errors="coerce").to_numpy(dtype=float)
        diag = result.diagnostics.get(name)
        sigma = diag.estimated_noise_std if diag is not None else np.nan

        ax.plot(raw[mask], depth[mask], "k--", lw=0.9, label=f"{name} (raw)")
        ax.plot(enh[mask], depth[mask], "r-", lw=1.3, label=f"{name}_ENH")
        ax.plot(
            rec[mask],
            depth[mask],
            color="tab:blue",
            lw=0.8,
            alpha=0.7,
            label=f"{name}_RECON",
        )
        if np.isfinite(sigma):
            ax.fill_betweenx(
                depth[mask],
                raw[mask] - 2.0 * sigma,
                raw[mask] + 2.0 * sigma,
                color="tab:blue",
                alpha=0.12,
                label=r"$\pm 2\hat{\sigma}_n$",
            )
        ax.set_title(f"{name}  (FWHM {result.target_fwhm_m:.2f} m)")
        ax.grid(True, ls=":", lw=0.5, color="gray")
        ax.legend(fontsize=7, loc="upper right")
        ax.invert_yaxis()
    axes[0].set_ylabel(f"DEPTH ({result.depth_uom})")
    fig.tight_layout()
    return fig
