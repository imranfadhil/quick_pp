"""Raw-log inter-curve depth alignment QC and depth-shift conditioning.

Detects constant (bulk run tie-in) and depth-varying (cable stretch, yo-yo
stick-slip, splice jump) depth mismatches across wireline/LWD curves using
gap-masked, badhole-aware normalized cross-correlation of smoothed bed-boundary
gradients (``|df/dz|``) — plus direct correlation for same-physics pairs (such
as auxiliary GR runs or resistivity array curves ``RXO``/``RM`` vs ``RT``).

Sign convention
---------------
Matches ``curve_edit`` ``depth_shift`` (+Down / -Up):
If a target curve is recorded *too shallow* by ``0.5 m`` (a bed boundary sits at
``1999.5 m`` on the target curve and at ``2000.0 m`` on the reference curve),
``suggested_shift`` is ``+0.5 m``. Applying ``+0.5 m`` shifts the target curve
downward (``z_new = z_old + 0.5``) so the boundary lands at ``2000.0 m``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

M_PER_FT = 0.3048
FT_PER_M = 3.280839895013123

LOG_SCALE_MNEMONICS = {
    "RT",
    "RD",
    "RDEP",
    "LLD",
    "ILD",
    "RLLD",
    "AT90",
    "RESD",
    "RM",
    "ILM",
    "LLS",
    "RMED",
    "RLA3",
    "AT30",
    "RESM",
    "RXO",
    "MSFL",
    "MLL",
    "RS",
    "RSHAL",
    "RLA1",
    "AT10",
    "RESS",
    "PERM",
}

GR_FAMILY_MNEMONICS = {"GR", "SGR", "CGR", "GRC", "GAMMA"}

DEFAULT_CANDIDATE_CURVES: Tuple[str, ...] = (
    "GR",
    "SGR",
    "CGR",
    "RHOB",
    "NPHI",
    "PEF",
    "RT",
    "RM",
    "RXO",
    "DTC",
    "DT",
    "DTS",
)

RES_SHALLOW_MED_CANDIDATES: Tuple[str, ...] = (
    "RXO",
    "MSFL",
    "RSHAL",
    "RS",
    "RLA1",
    "RM",
    "ILM",
    "RMED",
    "RLA3",
)


def _is_ft(depth_unit: str) -> bool:
    return str(depth_unit or "").strip().lower() in ("ft", "f", "feet", "foot")


def _is_log_curve(name: str) -> bool:
    up = name.strip().upper()
    if up in LOG_SCALE_MNEMONICS:
        return True
    for pfx in ("RT_", "RD_", "RXO_", "RM_", "RES"):
        if up.startswith(pfx):
            return True
    return False


def _is_gr_curve(name: str) -> bool:
    up = name.strip().upper()
    if up in GR_FAMILY_MNEMONICS:
        return True
    return up.startswith("GR_") and not up.endswith(
        (
            "_ENV_CORR",
            "_SIGMA",
            "_QC_CONF",
            "_QC_STATUS",
            "_CORR_DELTA",
            "_ENH",
            "_RECON",
            "_NORM",
            "_ORI",
        )
    )


def _dilate_false_mask(valid: np.ndarray, radius: int) -> np.ndarray:
    """Erode ``valid`` by ``radius`` samples around every False/NaN boundary."""
    if radius <= 0 or valid.size == 0:
        return valid.copy()
    out = valid.copy()
    n = valid.size
    for r in range(1, radius + 1):
        out[: n - r] &= valid[r:]
        out[r:] &= valid[: n - r]
    return out


def _prepare_feature_signal(
    values: np.ndarray,
    curve_name: str,
    dz_m: float,
    badhole_mask: Optional[np.ndarray] = None,
    mode: str = "gradient",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build gap-safe standardized feature signal and valid sample mask.

    Args:
        values: Raw numeric values on a uniform or near-uniform depth grid.
        curve_name: Curve mnemonic (controls log10 scaling for resistivity).
        dz_m: Median sample spacing in meters.
        badhole_mask: Optional boolean mask where True indicates badhole to exclude.
        mode: ``'gradient'`` for ``|df/dz|`` bed-boundary strength (cross-physics),
            or ``'direct'`` for high-pass demeaned signal (same-physics curves).
    """
    arr = np.asarray(values, dtype=float)
    n = arr.size
    finite = np.isfinite(arr)
    if _is_log_curve(curve_name):
        finite &= arr > 0
    if finite.sum() < 12:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=bool)

    work = np.empty(n, dtype=float)
    if _is_log_curve(curve_name):
        work[finite] = np.log10(np.clip(arr[finite], 1e-3, 1e5))
        work[~finite] = np.nan
    else:
        work[:] = np.where(finite, arr, np.nan)

    # Interpolate across internal NaNs purely so smoothing doesn't bleed NaNs,
    # then strictly mask out NaNs + a 2-sample guard band around every gap edge.
    s_interp = pd.Series(work).interpolate(limit_direction="both").to_numpy(dtype=float)
    sigma_samples = float(np.clip(0.22 / max(dz_m, 0.02), 0.8, 2.5))
    smoothed = gaussian_filter1d(s_interp, sigma=sigma_samples, mode="nearest")

    guard_valid = _dilate_false_mask(finite, radius=2)
    if badhole_mask is not None and badhole_mask.size == n:
        guard_valid = guard_valid & (~badhole_mask)

    if guard_valid.sum() < 12:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=bool)

    if mode == "direct":
        hp_win = max(15, int(round(12.0 / max(dz_m, 0.05))))
        baseline = (
            pd.Series(smoothed)
            .rolling(window=hp_win, center=True, min_periods=3)
            .median()
            .bfill()
            .ffill()
            .to_numpy(dtype=float)
        )
        feat = smoothed - baseline
        lo, hi = np.percentile(feat[guard_valid], [0.5, 99.5])
        feat = np.clip(feat, lo, hi)
    else:
        feat = np.abs(np.gradient(smoothed))
        hi = float(np.percentile(feat[guard_valid], 99.5))
        if hi > 0:
            feat = np.minimum(feat, hi)

    mu = float(np.mean(feat[guard_valid]))
    sd = float(np.std(feat[guard_valid]))
    if sd < 1e-9:
        return np.zeros(n, dtype=float), np.zeros(n, dtype=bool)

    normed = np.zeros(n, dtype=float)
    normed[guard_valid] = (feat[guard_valid] - mu) / sd
    return normed, guard_valid


def _ncc_at_lags(
    ref_sig: np.ndarray,
    ref_valid: np.ndarray,
    tgt_sig: np.ndarray,
    tgt_valid: np.ndarray,
    max_lag_samples: int,
    min_overlap: int = 16,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute overlap-normalized cross-correlation across integer lags ``[-L, +L]``.

    Positive lag ``+k`` means shifting ``tgt`` downward by ``+k`` samples aligns it
    with ``ref``: ``tgt[i - k]`` matches ``ref[i]``.
    """
    n = ref_sig.size
    max_l = max(1, min(int(max_lag_samples), max(1, n // 3)))
    lags = np.arange(-max_l, max_l + 1, dtype=int)
    corrs = np.full(lags.size, np.nan, dtype=float)
    counts = np.zeros(lags.size, dtype=int)

    for idx_l, k in enumerate(lags):
        if k >= 0:
            r_sl = slice(k, n)
            t_sl = slice(0, n - k)
        else:
            r_sl = slice(0, n + k)
            t_sl = slice(-k, n)
        ov = ref_valid[r_sl] & tgt_valid[t_sl]
        cnt = int(ov.sum())
        counts[idx_l] = cnt
        if cnt < min_overlap:
            continue
        rv = ref_sig[r_sl][ov]
        tv = tgt_sig[t_sl][ov]
        rv_c = rv - np.mean(rv)
        tv_c = tv - np.mean(tv)
        denom = float(np.sqrt(np.sum(rv_c * rv_c) * np.sum(tv_c * tv_c)))
        if denom > 1e-12:
            corrs[idx_l] = float(np.sum(rv_c * tv_c) / denom)

    return lags, corrs, counts


def _refine_peak_and_confidence(
    lags: np.ndarray,
    corrs: np.ndarray,
    counts: np.ndarray,
) -> Dict[str, float]:
    """Parabolic sub-sample peak interpolation and SNR-based confidence score."""
    finite = np.isfinite(corrs)
    if not finite.any():
        return {
            "lag_samples": 0.0,
            "peak_corr": 0.0,
            "zero_lag_corr": 0.0,
            "snr": 0.0,
            "confidence": 0.0,
            "overlap_samples": 0,
        }

    safe_corrs = np.where(finite, corrs, -np.inf)
    best_i = int(np.argmax(safe_corrs))
    r0 = float(corrs[best_i])
    sub_delta = 0.0
    if 0 < best_i < len(corrs) - 1:
        y_m1, y_0, y_p1 = corrs[best_i - 1], corrs[best_i], corrs[best_i + 1]
        if np.isfinite(y_m1) and np.isfinite(y_p1):
            denom = y_m1 - 2.0 * y_0 + y_p1
            if abs(denom) > 1e-9 and denom < 0:
                sub_delta = float(np.clip(0.5 * (y_m1 - y_p1) / denom, -0.5, 0.5))
                r0 = float(y_0 - 0.25 * (y_m1 - y_p1) * sub_delta)

    zero_idx = np.flatnonzero(lags == 0)
    zero_corr = (
        float(corrs[zero_idx[0]])
        if zero_idx.size > 0 and np.isfinite(corrs[zero_idx[0]])
        else 0.0
    )

    # Exclude +/-2 samples around peak when computing background correlation floor
    bg_mask = finite & (np.abs(np.arange(len(corrs)) - best_i) > 2)
    if bg_mask.sum() >= 3:
        bg_med = float(np.median(corrs[bg_mask]))
        bg_mad = float(np.median(np.abs(corrs[bg_mask] - bg_med))) * 1.4826
        bg_std = max(bg_mad, float(np.std(corrs[bg_mask])), 0.03)
        snr = max(0.0, (r0 - bg_med) / bg_std)
    else:
        snr = max(0.0, r0 * 4.0)

    n_ov = int(counts[best_i])
    corr_term = float(np.clip((r0 - 0.12) / 0.55, 0.0, 1.0))
    snr_term = float(np.clip((snr - 1.2) / 3.8, 0.0, 1.0))
    len_term = float(np.clip(n_ov / 80.0, 0.2, 1.0))
    conf = float(np.clip((0.50 * corr_term + 0.50 * snr_term) * len_term, 0.0, 1.0))

    # When zero-lag correlation is already virtually indistinguishable from the peak,
    # avoid reporting noise-induced fractional micro-lags when SNR is weak.
    lag_samp = float(lags[best_i]) + sub_delta
    if r0 - zero_corr < 0.015 and abs(lag_samp) < 0.65:
        lag_samp = 0.0

    return {
        "lag_samples": lag_samp,
        "peak_corr": float(np.clip(r0, -1.0, 1.0)),
        "zero_lag_corr": float(np.clip(zero_corr, -1.0, 1.0)),
        "snr": round(snr, 2),
        "confidence": round(conf, 3),
        "overlap_samples": n_ov,
    }


def _discover_target_curves(df: pd.DataFrame, ref_curve: str) -> List[str]:
    """Pick raw/spliced petrophysical curves present in ``df`` for alignment QC."""
    found: List[str] = []
    seen = {ref_curve.upper()}
    for c in DEFAULT_CANDIDATE_CURVES:
        if c in df.columns and c.upper() not in seen:
            if pd.to_numeric(df[c], errors="coerce").notna().sum() >= 20:
                found.append(c)
                seen.add(c.upper())

    # Also include auxiliary multi-run raw curves such as GR_1, GR_2, RT_2, DTC_2
    skip_suffixes = (
        "_ENV_CORR",
        "_SIGMA",
        "_QC_CONF",
        "_QC_STATUS",
        "_CORR_DELTA",
        "_ENH",
        "_RECON",
        "_NORM",
        "_ORI",
        "_IMP",
        "_PRED",
        "_ML",
        "_SYN",
        "_HC",
        "_DEN_CORR",
        "_EST",
    )
    for col in df.columns:
        up = str(col).strip().upper()
        if up in seen or up in {"DEPTH", "DEPT", "TVD", "TVDSS", "BS", "CALI", "DRHO"}:
            continue
        if up.endswith(skip_suffixes) or up.startswith(("ENV_", "INVASION_")):
            continue
        for pfx in (
            "GR_",
            "RHOB_",
            "NPHI_",
            "RT_",
            "RD_",
            "RXO_",
            "RM_",
            "DT_",
            "DTC_",
        ):
            if up.startswith(pfx):
                if pd.to_numeric(df[col], errors="coerce").notna().sum() >= 20:
                    found.append(col)
                    seen.add(up)
                break
    return found


def evaluate_depth_alignment(
    df: pd.DataFrame,
    ref_curve: str = "GR",
    target_curves: Optional[Sequence[str]] = None,
    max_shift: float = 3.0,
    window_size: float = 40.0,
    window_step: float = 10.0,
    exclude_badhole: bool = True,
    depth_unit: str = "m",
) -> Dict[str, Any]:
    """Evaluate bulk and sliding-window depth alignment across raw well logs.

    Args:
        df: Well DataFrame with ``DEPTH`` sorted or sortable ascending.
        ref_curve: Reference depth curve (default ``'GR'``; falls back to ``'RHOB'``).
        target_curves: Explicit list of curves to check; auto-discovered if ``None``.
        max_shift: Maximum search shift in ``depth_unit`` (default ``3.0 m`` / ``~10 ft``).
        window_size: Sliding window height in ``depth_unit`` for drift profile.
        window_step: Sliding window step in ``depth_unit``.
        exclude_badhole: Mask ``BADHOLE >= 1`` samples during correlation.
        depth_unit: Unit of ``df['DEPTH']`` and shift arguments (``'m'`` or ``'ft'``).

    Returns:
        Structured diagnostic dictionary with per-curve bulk shift, correlogram,
        sliding-window drift profile, resistivity-array tie-in check, and summary.
    """
    is_ft = _is_ft(depth_unit)
    u_label = "ft" if is_ft else "m"
    to_m = M_PER_FT if is_ft else 1.0

    if df is None or df.empty or "DEPTH" not in df.columns:
        return {
            "reference_curve": ref_curve,
            "depth_unit": u_label,
            "step": 0.0,
            "step_m": 0.0,
            "total_samples": 0,
            "curves": [],
            "summary": {
                "evaluated_count": 0,
                "aligned_count": 0,
                "minor_count": 0,
                "off_depth_count": 0,
                "variable_drift_count": 0,
                "worst_curve": None,
                "max_abs_shift_m": 0.0,
            },
        }

    work = df.copy()
    work["DEPTH"] = pd.to_numeric(work["DEPTH"], errors="coerce")
    work = work.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)
    n = len(work)
    if n < 20:
        return {
            "reference_curve": ref_curve,
            "depth_unit": u_label,
            "step": 0.0,
            "step_m": 0.0,
            "total_samples": n,
            "curves": [],
            "summary": {
                "evaluated_count": 0,
                "aligned_count": 0,
                "minor_count": 0,
                "off_depth_count": 0,
                "variable_drift_count": 0,
                "worst_curve": None,
                "max_abs_shift_m": 0.0,
            },
        }

    depths = work["DEPTH"].to_numpy(dtype=float)
    diffs = np.diff(depths)
    pos_diffs = diffs[diffs > 1e-6]
    dz = (
        float(np.median(pos_diffs))
        if pos_diffs.size > 0
        else (0.5 if is_ft else 0.1524)
    )
    dz_m = dz * to_m

    # Resolve reference curve
    eff_ref = ref_curve.strip().upper()
    if (
        eff_ref not in work.columns
        or pd.to_numeric(work[eff_ref], errors="coerce").notna().sum() < 20
    ):
        for fallback in ("GR", "RHOB", "NPHI", "DT", "DTC"):
            if (
                fallback in work.columns
                and pd.to_numeric(work[fallback], errors="coerce").notna().sum() >= 20
            ):
                eff_ref = fallback
                break

    if eff_ref not in work.columns:
        return {
            "reference_curve": eff_ref,
            "depth_unit": u_label,
            "step": round(dz, 4),
            "step_m": round(dz_m, 4),
            "total_samples": n,
            "curves": [],
            "summary": {
                "evaluated_count": 0,
                "aligned_count": 0,
                "minor_count": 0,
                "off_depth_count": 0,
                "variable_drift_count": 0,
                "worst_curve": None,
                "max_abs_shift_m": 0.0,
            },
        }

    bh_mask: Optional[np.ndarray] = None
    if exclude_badhole and "BADHOLE" in work.columns:
        bh_mask = (
            pd.to_numeric(work["BADHOLE"], errors="coerce").fillna(0.0).to_numpy()
            >= 0.5
        )
        # Only apply badhole mask if enough clean samples remain
        if int((~bh_mask).sum()) < 25:
            bh_mask = None

    targets = (
        [
            c.strip().upper()
            for c in target_curves
            if c.strip().upper() in work.columns and c.strip().upper() != eff_ref
        ]
        if target_curves is not None
        else _discover_target_curves(work, eff_ref)
    )

    ref_vals = pd.to_numeric(work[eff_ref], errors="coerce").to_numpy(dtype=float)
    ref_grad_sig, ref_grad_valid = _prepare_feature_signal(
        ref_vals, eff_ref, dz_m=dz_m, badhole_mask=bh_mask, mode="gradient"
    )
    ref_dir_sig, ref_dir_valid = _prepare_feature_signal(
        ref_vals, eff_ref, dz_m=dz_m, badhole_mask=bh_mask, mode="direct"
    )

    # Also prepare RT reference if RT is present, so shallow/medium resistivity (RXO/RM)
    # can be checked directly against deep RT when their GR correlation is weaker.
    rt_col = "RT" if "RT" in work.columns else None
    rt_dir_sig = rt_dir_valid = rt_grad_sig = rt_grad_valid = None
    if rt_col and pd.to_numeric(work[rt_col], errors="coerce").notna().sum() >= 20:
        rt_vals = pd.to_numeric(work[rt_col], errors="coerce").to_numpy(dtype=float)
        rt_dir_sig, rt_dir_valid = _prepare_feature_signal(
            rt_vals, rt_col, dz_m=dz_m, badhole_mask=bh_mask, mode="direct"
        )
        rt_grad_sig, rt_grad_valid = _prepare_feature_signal(
            rt_vals, rt_col, dz_m=dz_m, badhole_mask=bh_mask, mode="gradient"
        )

    max_lag_samples = max(2, int(round(max(float(max_shift), dz * 2.0) / dz)))
    win_samples = max(30, int(round(max(float(window_size), dz * 30.0) / dz)))
    step_samples = max(8, int(round(max(float(window_step), dz * 8.0) / dz)))

    minor_thresh_m = max(0.55 * dz_m, 0.08)
    off_thresh_m = max(1.50 * dz_m, 0.23)

    curve_results: List[Dict[str, Any]] = []

    for col in targets:
        t_vals = pd.to_numeric(work[col], errors="coerce").to_numpy(dtype=float)
        same_gr = _is_gr_curve(eff_ref) and _is_gr_curve(col)
        is_shal_res = (
            col in RES_SHALLOW_MED_CANDIDATES and rt_col is not None and col != rt_col
        )

        t_grad_sig, t_grad_valid = _prepare_feature_signal(
            t_vals, col, dz_m=dz_m, badhole_mask=bh_mask, mode="gradient"
        )

        # Primary correlation against eff_ref
        use_r_sig, use_r_val = ref_grad_sig, ref_grad_valid
        use_t_sig, use_t_val = t_grad_sig, t_grad_valid
        paired_ref = eff_ref
        corr_method = "gradient"

        lags, corrs, counts = _ncc_at_lags(
            use_r_sig, use_r_val, use_t_sig, use_t_val, max_lag_samples
        )
        peak = _refine_peak_and_confidence(lags, corrs, counts)

        if same_gr:
            t_dir_sig, t_dir_valid = _prepare_feature_signal(
                t_vals, col, dz_m=dz_m, badhole_mask=bh_mask, mode="direct"
            )
            lags_d, corrs_d, counts_d = _ncc_at_lags(
                ref_dir_sig, ref_dir_valid, t_dir_sig, t_dir_valid, max_lag_samples
            )
            peak_d = _refine_peak_and_confidence(lags_d, corrs_d, counts_d)
            if peak_d["confidence"] >= peak["confidence"]:
                use_r_sig, use_r_val = ref_dir_sig, ref_dir_valid
                use_t_sig, use_t_val = t_dir_sig, t_dir_valid
                lags, corrs, counts, peak = lags_d, corrs_d, counts_d, peak_d
                corr_method = "direct"

        # For shallow/medium resistivity, check direct tie-in against RT as well
        rt_tie_shift_m: Optional[float] = None
        rt_tie_conf: Optional[float] = None
        if is_shal_res and rt_dir_sig is not None and rt_grad_sig is not None:
            t_dir_sig, t_dir_valid = _prepare_feature_signal(
                t_vals, col, dz_m=dz_m, badhole_mask=bh_mask, mode="direct"
            )
            lags_rt, corrs_rt, counts_rt = _ncc_at_lags(
                rt_dir_sig, rt_dir_valid, t_dir_sig, t_dir_valid, max_lag_samples
            )
            peak_rt = _refine_peak_and_confidence(lags_rt, corrs_rt, counts_rt)
            rt_tie_shift_m = round(peak_rt["lag_samples"] * dz_m, 4)
            rt_tie_conf = peak_rt["confidence"]
            if eff_ref == rt_col and peak_rt["confidence"] >= peak["confidence"]:
                use_r_sig, use_r_val = rt_dir_sig, rt_dir_valid
                use_t_sig, use_t_val = t_dir_sig, t_dir_valid
                lags, corrs, counts, peak = lags_rt, corrs_rt, counts_rt, peak_rt
                corr_method = "direct"

        # Sliding-window drift profile along depth
        windows: List[Dict[str, float]] = []
        if n >= win_samples:
            starts = list(range(0, n - win_samples + 1, step_samples))
            if (
                starts[-1] != n - win_samples
                and (n - win_samples - starts[-1]) >= step_samples // 2
            ):
                starts.append(n - win_samples)
            win_max_lag = min(max_lag_samples, max(2, win_samples // 4))
            for st_i in starts:
                en_i = st_i + win_samples
                w_lags, w_corrs, w_counts = _ncc_at_lags(
                    use_r_sig[st_i:en_i],
                    use_r_val[st_i:en_i],
                    use_t_sig[st_i:en_i],
                    use_t_val[st_i:en_i],
                    win_max_lag,
                    min_overlap=max(12, win_samples // 4),
                )
                w_peak = _refine_peak_and_confidence(w_lags, w_corrs, w_counts)
                if w_peak["overlap_samples"] <= 0:
                    continue
                center_d = float(0.5 * (depths[st_i] + depths[en_i - 1]))
                w_shift = w_peak["lag_samples"] * dz
                windows.append(
                    {
                        "depth": round(center_d, 3),
                        "depth_m": round(center_d * to_m, 3),
                        "shift": round(w_shift, 4),
                        "shift_m": round(w_shift * to_m, 4),
                        "shift_samples": round(w_peak["lag_samples"], 2),
                        "peak_corr": round(w_peak["peak_corr"], 3),
                        "confidence": w_peak["confidence"],
                    }
                )

        bulk_samp = peak["lag_samples"]
        bulk_shift = bulk_samp * dz
        bulk_shift_m = bulk_samp * dz_m
        conf = peak["confidence"]

        reliable_wins = [w for w in windows if w["confidence"] >= 0.40]
        if reliable_wins:
            abs_shifts_m = np.array(
                [abs(w["shift_m"]) for w in reliable_wins], dtype=float
            )
            signed_shifts_m = np.array(
                [w["shift_m"] for w in reliable_wins], dtype=float
            )
            p95_drift_m = float(np.percentile(abs_shifts_m, 95))
            max_drift_m = float(np.max(abs_shifts_m))
            drift_spread_m = (
                float(
                    np.percentile(signed_shifts_m, 90)
                    - np.percentile(signed_shifts_m, 10)
                )
                if len(reliable_wins) >= 3
                else 0.0
            )
        else:
            p95_drift_m = abs(bulk_shift_m)
            max_drift_m = abs(bulk_shift_m)
            drift_spread_m = 0.0

        variable_drift = bool(
            len(reliable_wins) >= 3 and drift_spread_m > max(1.5 * dz_m, 0.25)
        )

        if conf < 0.28:
            status = "LOW_SIGNAL"
        elif abs(bulk_shift_m) > off_thresh_m or (
            len(reliable_wins) >= 2 and p95_drift_m > off_thresh_m * 1.25
        ):
            status = "OFF_DEPTH"
        elif abs(bulk_shift_m) > minor_thresh_m or variable_drift:
            status = "MINOR"
        else:
            status = "ALIGNED"

        finite_lag_mask = np.isfinite(corrs)
        curve_results.append(
            {
                "curve": col,
                "reference_curve": paired_ref,
                "method": corr_method,
                "samples": int(peak["overlap_samples"]),
                "shift_samples": round(bulk_samp, 2),
                "suggested_shift": round(bulk_shift, 4),
                "suggested_shift_m": round(bulk_shift_m, 4),
                "p95_abs_drift": round(p95_drift_m / to_m, 4),
                "p95_abs_drift_m": round(p95_drift_m, 4),
                "max_abs_drift_m": round(max_drift_m, 4),
                "drift_spread_m": round(drift_spread_m, 4),
                "variable_drift": variable_drift,
                "peak_corr": round(peak["peak_corr"], 3),
                "zero_lag_corr": round(peak["zero_lag_corr"], 3),
                "snr": peak["snr"],
                "confidence": conf,
                "status": status,
                "rt_tie_shift_m": rt_tie_shift_m,
                "rt_tie_confidence": rt_tie_conf,
                "correlogram": {
                    "lags": [
                        round(float(lag * dz), 4) for lag in lags[finite_lag_mask]
                    ],
                    "lags_m": [
                        round(float(lag * dz_m), 4) for lag in lags[finite_lag_mask]
                    ],
                    "corrs": [round(float(c), 4) for c in corrs[finite_lag_mask]],
                },
                "windows": windows,
            }
        )

    aligned_cnt = sum(1 for r in curve_results if r["status"] == "ALIGNED")
    minor_cnt = sum(1 for r in curve_results if r["status"] == "MINOR")
    off_cnt = sum(1 for r in curve_results if r["status"] == "OFF_DEPTH")
    var_cnt = sum(1 for r in curve_results if r["variable_drift"])
    confident_rows = [r for r in curve_results if r["status"] != "LOW_SIGNAL"]
    worst_row = (
        max(
            confident_rows,
            key=lambda r: max(abs(r["suggested_shift_m"]), r["p95_abs_drift_m"]),
        )
        if confident_rows
        else None
    )

    return {
        "reference_curve": eff_ref,
        "depth_unit": u_label,
        "step": round(dz, 4),
        "step_m": round(dz_m, 4),
        "total_samples": n,
        "curves": curve_results,
        "summary": {
            "evaluated_count": len(curve_results),
            "aligned_count": aligned_cnt,
            "minor_count": minor_cnt,
            "off_depth_count": off_cnt,
            "variable_drift_count": var_cnt,
            "worst_curve": worst_row["curve"] if worst_row else None,
            "max_abs_shift_m": round(
                max(
                    (
                        max(abs(r["suggested_shift_m"]), r["p95_abs_drift_m"])
                        for r in confident_rows
                    ),
                    default=0.0,
                ),
                4,
            ),
        },
    }


def _build_monotonic_shift_profile(
    depths: np.ndarray,
    windows: Sequence[Dict[str, Any]],
    fallback_shift: float,
    min_conf: float = 0.40,
) -> np.ndarray:
    """Interpolate window shifts onto ``depths`` while enforcing monotonic ``z + shift(z)``."""
    pts = [
        (float(w["depth"]), float(w["shift"]), float(w["confidence"]))
        for w in windows
        if w.get("confidence", 0.0) >= min_conf
        and np.isfinite(w.get("depth", np.nan))
        and np.isfinite(w.get("shift", np.nan))
    ]
    if len(pts) < 2:
        return np.full_like(depths, float(fallback_shift), dtype=float)

    pts.sort(key=lambda t: t[0])
    w_d = np.array([p[0] for p in pts], dtype=float)
    w_s = np.array([p[1] for p in pts], dtype=float)
    if len(w_s) >= 3:
        w_s = (
            pd.Series(w_s)
            .rolling(3, center=True, min_periods=1)
            .median()
            .to_numpy(dtype=float)
        )

    shift_arr = np.interp(depths, w_d, w_s, left=w_s[0], right=w_s[-1])

    # Enforce strict monotonicity of warped depth z' = z + shift(z) so dz'/dz >= 0.25
    warped = depths + shift_arr
    dz = np.diff(depths)
    min_step = 0.25 * np.maximum(dz, 1e-6)
    for i in range(1, warped.size):
        if warped[i] < warped[i - 1] + min_step[i - 1]:
            warped[i] = warped[i - 1] + min_step[i - 1]
    return warped - depths


def apply_depth_alignment(
    df: pd.DataFrame,
    alignment_report: Dict[str, Any],
    curves_to_shift: Optional[Sequence[str]] = None,
    mode: str = "bulk",
    manual_shifts: Optional[Dict[str, float]] = None,
    min_confidence: float = 0.35,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Any]]]:
    """Apply bulk or dynamic depth shifts to curves in ``df``.

    Args:
        df: Input well DataFrame with ``DEPTH``.
        alignment_report: Output from :func:`evaluate_depth_alignment`.
        curves_to_shift: Optional explicit list of curves to shift; defaults to all
            ``OFF_DEPTH`` and ``MINOR`` curves meeting ``min_confidence``.
        mode: ``'bulk'`` (constant shift) or ``'dynamic'`` (monotonic window profile).
        manual_shifts: Optional ``{curve: shift_in_df_depth_unit}`` overrides.
        min_confidence: Minimum alignment confidence required when auto-selecting curves.

    Returns:
        ``(shifted_df, applied_info)`` where ``applied_info[curve]`` summarizes the
        applied shift mode, mean shift, and min/max shift.
    """
    out = df.copy()
    if out.empty or "DEPTH" not in out.columns:
        return out, {}

    depth_arr = pd.to_numeric(out["DEPTH"], errors="coerce").to_numpy(dtype=float)
    valid_d = np.isfinite(depth_arr)
    if valid_d.sum() < 2:
        return out, {}

    row_by_curve = {
        str(r["curve"]).upper(): r for r in (alignment_report.get("curves") or [])
    }
    manual_map = {
        str(k).upper(): float(v)
        for k, v in (manual_shifts or {}).items()
        if v is not None
    }

    if curves_to_shift is not None:
        selected = [c.strip().upper() for c in curves_to_shift if c and c.strip()]
    elif manual_map:
        selected = list(manual_map.keys())
    else:
        selected = [
            r["curve"]
            for r in (alignment_report.get("curves") or [])
            if r.get("status") in ("OFF_DEPTH", "MINOR")
            and float(r.get("confidence", 0.0)) >= min_confidence
        ]

    applied: Dict[str, Dict[str, Any]] = {}
    eff_mode = "dynamic" if str(mode).strip().lower() == "dynamic" else "bulk"

    for col in selected:
        if col not in out.columns:
            continue
        val_arr = pd.to_numeric(out[col], errors="coerce").to_numpy(dtype=float)
        valid = valid_d & np.isfinite(val_arr)
        if valid.sum() < 2:
            continue

        diag = row_by_curve.get(col)
        if col in manual_map:
            shift_vec = np.full(depth_arr.size, manual_map[col], dtype=float)
            used_mode = "manual"
        elif eff_mode == "dynamic" and diag and len(diag.get("windows") or []) >= 2:
            shift_vec = _build_monotonic_shift_profile(
                depth_arr,
                diag["windows"],
                fallback_shift=float(diag.get("suggested_shift", 0.0)),
            )
            used_mode = "dynamic"
        elif diag is not None:
            s_val = float(diag.get("suggested_shift", 0.0))
            shift_vec = np.full(depth_arr.size, s_val, dtype=float)
            used_mode = "bulk"
        else:
            continue

        if np.max(np.abs(shift_vec[valid])) < 1e-6:
            continue

        shifted_depths = depth_arr[valid] + shift_vec[valid]
        order = np.argsort(shifted_depths, kind="stable")
        interp_vals = np.interp(
            depth_arr,
            shifted_depths[order],
            val_arr[valid][order],
            left=np.nan,
            right=np.nan,
        )
        out[col] = interp_vals
        applied[col] = {
            "mode": used_mode,
            "mean_shift": round(float(np.mean(shift_vec[valid])), 4),
            "min_shift": round(float(np.min(shift_vec[valid])), 4),
            "max_shift": round(float(np.max(shift_vec[valid])), 4),
        }

    return out, applied
