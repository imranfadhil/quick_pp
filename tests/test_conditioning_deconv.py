"""Tests for 1D constrained deconvolution and resolution harmonization.

Covers the five mandatory verification gates from the technical specification:
1. Exact adjoint identity and analytical gradient correctness.
2. Constant and ramp invariance (boundary droop free).
3. Thin-bed step recovery and Gibbs overshoot suppression.
4. Washout ringing isolation.
5. Bulk-volume conservation and Thomas-Stieber scatter collapse.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import check_grad

from quick_pp.conditioning import (
    DEFAULT_LEGACY_SPECS,
    ToolResolutionSpec,
    apply_vrf_adjoint,
    apply_vrf_forward,
    build_vrf_kernel,
    check_interlog_depth_alignment,
    compute_borehole_weights,
    deconvolve_curve_1d,
    deconvolve_curve_wiener,
    estimate_noise_mad,
    harmonize_nuclear_logs,
)

DZ = 0.1524


def _synthetic_lrlc(n: int = 400, seed: int = 42, noise_mult: float = 3.0):
    """Build a 30 m laminated LRLC earth model with a tight calcite streak."""
    rng = np.random.default_rng(seed)
    depth = 1000.0 + np.arange(n) * DZ
    bed = (np.floor((depth - depth[0]) / 0.45) % 2).astype(int)
    truth = {
        "GR": np.where(bed == 0, 25.0, 115.0),
        "NPHI": np.where(bed == 0, 0.27, 0.36),
        "RHOB": np.where(bed == 0, 2.22, 2.55),
    }
    streak = (depth > 1030.0) & (depth < 1030.6)
    truth["RHOB"][streak] = 2.71
    truth["NPHI"][streak] = 0.02
    truth["GR"][streak] = 12.0

    df = pd.DataFrame({"DEPTH": depth})
    for name, spec in DEFAULT_LEGACY_SPECS.items():
        kernel = build_vrf_kernel(spec.fwhm_m, DZ)
        measured = apply_vrf_forward(truth[name], kernel)
        df[name] = measured + rng.normal(0.0, spec.noise_floor * noise_mult, n)
    return df, truth, streak


# ---------------------------------------------------------------------------
# Gate 1: exact adjoint identity and analytical gradient
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fwhm", [0.0, 0.2, 0.4, 0.76])
def test_vrf_adjoint_exactness(fwhm):
    """Inner products must match to machine precision for symmetric kernels."""
    rng = np.random.default_rng(0)
    kernel = build_vrf_kernel(fwhm, DZ)
    x = rng.normal(size=200)
    r = rng.normal(size=200)
    lhs = float(np.dot(apply_vrf_forward(x, kernel), r))
    rhs = float(np.dot(x, apply_vrf_adjoint(r, kernel)))
    assert abs(lhs - rhs) < 1e-13


def test_vrf_adjoint_exactness_asymmetric_kernel():
    """The adjoint identity must hold for asymmetric (custom discrete) kernels."""
    rng = np.random.default_rng(1)
    kernel = rng.random(9)
    kernel /= kernel.sum()
    x = rng.normal(size=150)
    r = rng.normal(size=150)
    lhs = float(np.dot(apply_vrf_forward(x, kernel), r))
    rhs = float(np.dot(x, apply_vrf_adjoint(r, kernel)))
    assert abs(lhs - rhs) < 1e-13


def test_analytical_gradient_matches_finite_difference():
    """The analytical Huber objective gradient must match numerical differentiation."""
    from quick_pp.conditioning import _solve_segment_huber

    rng = np.random.default_rng(2)
    n = 60
    y = rng.normal(2.4, 0.05, n)
    weights = np.ones(n)
    kernel = build_vrf_kernel(0.25, DZ)
    vol_kernel = build_vrf_kernel(3.0, DZ)

    captured = {}

    def objective(x):
        res = apply_vrf_forward(x, kernel) - y
        loss_fid = 0.5 * float(np.dot(res, weights**2 * res))
        diffs = x[1:] - x[:-1]
        abs_d = np.abs(diffs)
        delta = 0.01
        huber = np.where(abs_d <= delta, 0.5 * diffs**2 / delta, abs_d - 0.5 * delta)
        loss_reg = 0.5 * float(np.sum(huber))
        vol_diff = apply_vrf_forward(x - y, vol_kernel)
        loss_vol = 0.5 * 0.05 * float(np.dot(vol_diff, vol_diff))
        return loss_fid + loss_reg + loss_vol

    # Recover the analytical gradient by calling the solver's inner objective
    import quick_pp.conditioning as cond

    orig_minimize = cond.minimize

    def spy_minimize(fun, x0, **kwargs):
        captured["fun"] = fun
        return orig_minimize(fun, x0, **kwargs)

    cond.minimize = spy_minimize
    try:
        _solve_segment_huber(
            y, weights, kernel, vol_kernel, 1.75, 3.05, 0.5, 0.01, lambda_vol=0.05
        )
    finally:
        cond.minimize = orig_minimize

    x0 = np.clip(y, 1.75, 3.05)
    err = check_grad(lambda x: captured["fun"](x)[0], lambda x: captured["fun"](x)[1], x0)
    assert err < 1e-5


# ---------------------------------------------------------------------------
# Gate 2: constant and ramp invariance (boundary droop free)
# ---------------------------------------------------------------------------


def test_boundary_droop_free_constant():
    """A constant shale baseline must survive forward convolution with zero droop."""
    kernel = build_vrf_kernel(0.4, DZ)
    constant = np.full(300, 110.0)
    recon = apply_vrf_forward(constant, kernel)
    assert np.max(np.abs(recon - 110.0)) < 1e-5


def test_boundary_droop_free_ramp():
    """A linear fining-upward ramp must be reproduced without stair-casing."""
    kernel = build_vrf_kernel(0.4, DZ)
    ramp = np.linspace(30.0, 120.0, 300)
    recon = apply_vrf_forward(ramp, kernel)
    # Interior samples must remain on the ramp (edge replication preserves linearity)
    assert np.max(np.abs(recon[20:-20] - ramp[20:-20])) < 1e-6


def test_constant_log_deconvolution_is_stable():
    """Deconvolving a constant log must return the same constant within bounds."""
    y = np.full(200, 110.0)
    spec = DEFAULT_LEGACY_SPECS["GR"]
    x, recon, diag = deconvolve_curve_1d(y, DZ, spec, 0.3048)
    assert np.all(np.isfinite(x))
    assert np.max(np.abs(x - 110.0)) < 1.0
    assert diag.converged


# ---------------------------------------------------------------------------
# Gate 3: thin-bed step recovery and Gibbs suppression
# ---------------------------------------------------------------------------


def test_lrlc_thin_bed_no_overshoot():
    """Peak overshoot adjacent to a tight calcite streak must stay below 2% of contrast."""
    df, truth, streak = _synthetic_lrlc(seed=7)
    result = harmonize_nuclear_logs(df, mode="enhance_all")

    si = np.flatnonzero(streak)
    lo, hi = si.min() - 10, si.max() + 10
    for name in ("GR", "NPHI", "RHOB"):
        enh = result.df[f"{name}_ENH"].to_numpy()[lo:hi]
        local_truth = truth[name][lo:hi]
        contrast = abs(truth[name][streak].mean() - truth[name][~streak].mean())
        overshoot = max(0.0, enh.max() - local_truth.max())
        undershoot = max(0.0, local_truth.min() - enh.min())
        assert overshoot / contrast < 0.02, f"{name} overshoot {overshoot / contrast:.3%}"
        assert undershoot / contrast < 0.02, f"{name} undershoot {undershoot / contrast:.3%}"


def test_lrlc_thin_bed_mae_reduction():
    """Enhancement must reduce MAE against the target-resolution truth for every curve."""
    df, truth, _ = _synthetic_lrlc(seed=42)
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    target_kernel = build_vrf_kernel(result.target_fwhm_m, DZ)

    for name in ("GR", "NPHI", "RHOB"):
        target_truth = apply_vrf_forward(truth[name], target_kernel)
        raw_mae = float(np.mean(np.abs(df[name].to_numpy() - target_truth)))
        enh_mae = float(
            np.mean(np.abs(result.df[f"{name}_ENH"].to_numpy() - target_truth))
        )
        assert enh_mae < raw_mae, f"{name}: enh MAE {enh_mae} !< raw MAE {raw_mae}"


def test_enhanced_curves_respect_physical_bounds():
    """Deconvolved curves must never leave their hard petrophysical box bounds."""
    df, _, _ = _synthetic_lrlc(seed=11)
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    for name, spec in DEFAULT_LEGACY_SPECS.items():
        enh = result.df[f"{name}_ENH"].to_numpy()
        finite = enh[np.isfinite(enh)]
        assert finite.min() >= spec.lower_bound - 1e-9
        assert finite.max() <= spec.upper_bound + 1e-9


# ---------------------------------------------------------------------------
# Gate 4: washout ringing isolation
# ---------------------------------------------------------------------------


def test_washout_ringing_isolation():
    """Good-hole samples beyond 0.6 m of a washout must stay within 0.01 g/cc of truth."""
    n = 300
    depth = 1000.0 + np.arange(n) * DZ
    rhob_true = np.full(n, 2.35)
    washout = (depth > 1015.0) & (depth < 1015.6)
    rhob_true[washout] = 1.82

    rng = np.random.default_rng(7)
    spec = DEFAULT_LEGACY_SPECS["RHOB"]
    y = apply_vrf_forward(rhob_true, build_vrf_kernel(spec.fwhm_m, DZ))
    y = y + rng.normal(0.0, spec.noise_floor, n)

    cali = np.full(n, 8.5)
    cali[washout] = 14.0
    drho = np.zeros(n)
    drho[washout] = -0.28
    df = pd.DataFrame(
        {"DEPTH": depth, "RHOB": y, "CALI": cali, "BS": np.full(n, 8.5), "DRHO": drho}
    )

    result = harmonize_nuclear_logs(df, mode="enhance_all", curves=["RHOB"])
    enh = result.df["RHOB_ENH"].to_numpy()

    far = (depth < 1014.4) | (depth > 1016.2)
    deviation = np.abs(enh[far] - 2.35)
    assert deviation.max() < 0.01, f"washout halo {deviation.max():.5f} g/cc"


def test_borehole_weights_downweight_washout():
    """Caliper and DRHO washout indicators must drive the reliability weight down."""
    n = 50
    cali = np.full(n, 8.5)
    cali[20:25] = 14.0
    drho = np.zeros(n)
    drho[20:25] = -0.28
    df = pd.DataFrame({"CALI": cali, "BS": np.full(n, 8.5), "DRHO": drho})
    w_cali, w_both = compute_borehole_weights(df)
    assert w_cali[22] < 0.1
    assert w_both[22] < 0.1
    assert w_cali[0] == pytest.approx(1.0)
    assert w_both[0] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Gate 5: bulk-volume conservation and Thomas-Stieber collapse
# ---------------------------------------------------------------------------


def test_bulk_volume_conservation():
    """Relative bulk integral bias must stay below 0.5% for every curve."""
    df, _, _ = _synthetic_lrlc(seed=3)
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    for name, diag in result.diagnostics.items():
        assert abs(diag.relative_volume_bias_pct) < 0.5, (
            f"{name} volume bias {diag.relative_volume_bias_pct:.3f}%"
        )


def test_ts_boomerang_collapse():
    """Resolution harmonization must reduce Thomas-Stieber laminated-line scatter.

    Uses the real :class:`~quick_pp.lithology.thin_beds.ThinBeds` neutron-density
    crossplot projection, which is the mechanism that produces the unphysical
    "boomerang" loops when GR, NPHI, and RHOB are measured at different vertical
    footprints.
    """
    from quick_pp.lithology.thin_beds import ThinBeds

    tb = ThinBeds(
        dry_sand_poro=0.28,
        dry_shale_poro=0.10,
        dry_sand_point=(-0.02, 2.65),
        dry_clay_point=(0.44, 2.71),
        fluid_point=(1.0, 1.0),
    )

    def ts_scatter(gr, nphi, rhob):
        """RMS distance from the theoretical laminated sand-shale line."""
        vsh = np.clip((gr - 25.0) / (115.0 - 25.0), 0.0, 1.0)
        phit = np.asarray(tb.estimate_litho_poro(nphi, rhob)[2], dtype=float)
        line = 0.28 + (0.10 - 0.28) * vsh
        return float(np.sqrt(np.mean((phit - line) ** 2)))

    df, _, _ = _synthetic_lrlc(n=500, seed=5)
    result = harmonize_nuclear_logs(df, mode="enhance_all")

    raw_scatter = ts_scatter(
        df["GR"].to_numpy(), df["NPHI"].to_numpy(), df["RHOB"].to_numpy()
    )
    enh_scatter = ts_scatter(
        result.df["GR_ENH"].to_numpy(),
        result.df["NPHI_ENH"].to_numpy(),
        result.df["RHOB_ENH"].to_numpy(),
    )
    reduction = 100.0 * (1.0 - enh_scatter / raw_scatter)
    assert reduction > 20.0, (
        f"TS scatter reduction {reduction:.1f}% below 20% "
        f"(raw {raw_scatter:.5f} -> enh {enh_scatter:.5f})"
    )


# ---------------------------------------------------------------------------
# Edge cases and orchestration guards
# ---------------------------------------------------------------------------


def test_nan_gap_segmentation_preserves_mask():
    """NaN gaps must be restored after deconvolution and never bridged."""
    df, _, _ = _synthetic_lrlc(n=200, seed=9)
    df.loc[50:70, ["GR", "NPHI", "RHOB"]] = np.nan
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    for name in ("GR", "NPHI", "RHOB"):
        enh = result.df[f"{name}_ENH"].to_numpy()
        assert np.all(np.isnan(enh[50:71]))


def test_small_gap_interpolation_restores_nan():
    """Gaps of <= 2 samples are interpolated then re-masked."""
    df, _, _ = _synthetic_lrlc(n=200, seed=10)
    df.loc[100, "GR"] = np.nan
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    assert np.isnan(result.df["GR_ENH"].to_numpy()[100])


def test_nonuniform_grid_raises_when_resampling_disabled():
    """A non-uniform depth grid must raise when resampling is disabled."""
    df, _, _ = _synthetic_lrlc(n=100, seed=12)
    df.loc[50, "DEPTH"] = df.loc[50, "DEPTH"] + 0.05
    with pytest.raises(ValueError, match="uniformly sampled"):
        harmonize_nuclear_logs(df, mode="enhance_all", resample_nonuniform=False)


def test_nonuniform_grid_resamples_with_warning():
    """A non-uniform depth grid is resampled with a warning when enabled."""
    df, _, _ = _synthetic_lrlc(n=100, seed=13)
    df.loc[50, "DEPTH"] = df.loc[50, "DEPTH"] + 0.05
    with pytest.warns(UserWarning, match="not uniformly sampled"):
        result = harmonize_nuclear_logs(df, mode="enhance_all")
    assert "GR_ENH" in result.df.columns


def test_nonuniform_grid_preserves_original_depths():
    """Resampling must not leak the internal uniform grid into the returned depths.

    Depth-keyed consumers (the UI merges computed curves by exact depth) silently drop
    every enhanced sample when the returned grid differs from the well grid.
    """
    df, _, _ = _synthetic_lrlc(n=200, seed=17)
    rng = np.random.default_rng(17)
    df["DEPTH"] = df["DEPTH"] + np.cumsum(rng.normal(0.0, 0.0004, len(df)))
    with pytest.warns(UserWarning, match="not uniformly sampled"):
        result = harmonize_nuclear_logs(df, mode="enhance_all")
    np.testing.assert_allclose(
        result.df["DEPTH"].to_numpy(), df["DEPTH"].to_numpy(), atol=1e-9
    )
    for name in ("GR", "NPHI", "RHOB"):
        enh = result.df[f"{name}_ENH"].to_numpy()
        assert np.isfinite(enh).all()
        assert np.sqrt(np.mean((enh - df[name].to_numpy()) ** 2)) > 0.0


def test_zone_gaps_are_not_interpolated_across():
    """Excluded depth intervals must stay NaN instead of being bridged by a straight line."""
    df, _, _ = _synthetic_lrlc(n=400, seed=18)
    keep = (df["DEPTH"] < 1015.0) | (df["DEPTH"] > 1045.0)
    filtered = df[keep].reset_index(drop=True)
    with pytest.warns(UserWarning, match="not uniformly sampled"):
        result = harmonize_nuclear_logs(filtered, mode="enhance_all")
    # The returned frame must stay on the filtered grid, never grow across the gap.
    assert len(result.df) == len(filtered)
    np.testing.assert_allclose(
        result.df["DEPTH"].to_numpy(), filtered["DEPTH"].to_numpy(), atol=1e-9
    )
    for name in ("GR", "NPHI", "RHOB"):
        assert np.isfinite(result.df[f"{name}_ENH"].to_numpy()).all()


def test_borehole_weights_survive_resampling():
    """Caliper/density-correction weights must be derived from the measured frame."""
    df, _, _ = _synthetic_lrlc(n=200, seed=19)
    rng = np.random.default_rng(19)
    df["DEPTH"] = df["DEPTH"] + np.cumsum(rng.normal(0.0, 0.0004, len(df)))
    df["BS"] = 8.5
    df["CALI"] = 8.5
    df.loc[80:100, "CALI"] = 12.0
    with pytest.warns(UserWarning, match="not uniformly sampled"):
        result = harmonize_nuclear_logs(df, mode="enhance_all")
    weight = result.df["DECONV_WEIGHT"].to_numpy()
    assert np.isfinite(weight).all()
    assert weight.min() < 0.9


def test_curve_already_at_target_resolution_warns():
    """A curve finer than the target must report the pass-through instead of silently no-oping."""
    df, _, _ = _synthetic_lrlc(n=200, seed=20)
    with pytest.warns(UserWarning, match="already at or below"):
        result = harmonize_nuclear_logs(df, mode="match_finest")
    assert result.diagnostics["RHOB"].effective_fwhm_m == pytest.approx(0.0)
    # No residual smearing to invert, so the output stays close to the raw curve
    # (only the volume tether and Huber penalty perturb it).
    raw = df["RHOB"].to_numpy()
    enh = result.df["RHOB_ENH"].to_numpy()
    assert np.sqrt(np.mean((enh - raw) ** 2)) < 0.3 * np.std(raw)


def test_nyquist_floor_clamping():
    """A target resolution below the Nyquist floor must be clamped with a warning."""
    df, _, _ = _synthetic_lrlc(n=200, seed=14)
    with pytest.warns(UserWarning, match="Nyquist floor"):
        result = harmonize_nuclear_logs(df, target_fwhm_m=0.05)
    assert result.target_fwhm_m == pytest.approx(2.0 * DZ)


def test_match_finest_mode_uses_finest_tool():
    """'match_finest' must target the finest intrinsic tool resolution."""
    df, _, _ = _synthetic_lrlc(n=200, seed=15)
    result = harmonize_nuclear_logs(df, mode="match_finest")
    assert result.target_fwhm_m == pytest.approx(DEFAULT_LEGACY_SPECS["RHOB"].fwhm_m)


def test_imperial_depth_unit_detection():
    """A 0.5 ft depth step must be detected as imperial and converted to metres."""
    n = 200
    depth = 3000.0 + np.arange(n) * 0.5
    df = pd.DataFrame({"DEPTH": depth})
    for name, spec in DEFAULT_LEGACY_SPECS.items():
        kernel = build_vrf_kernel(spec.fwhm_m / 0.3048, 0.5)
        df[name] = apply_vrf_forward(np.full(n, 50.0), kernel)
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    assert result.depth_uom == "ft"
    assert result.depth_step_m == pytest.approx(0.5 * 0.3048)


def test_missing_curves_raise():
    """Requesting curves absent from the DataFrame must raise a clear error."""
    df = pd.DataFrame({"DEPTH": np.arange(10) * DZ, "GR": np.zeros(10)})
    with pytest.raises(ValueError, match="None of the requested nuclear curves"):
        harmonize_nuclear_logs(df, curves=["NPHI", "RHOB"])


def test_wiener_preview_method_runs():
    """The fast spectral preview must produce bounded, finite output."""
    df, _, _ = _synthetic_lrlc(n=200, seed=16)
    result = harmonize_nuclear_logs(df, mode="enhance_all", method="wiener")
    for name, spec in DEFAULT_LEGACY_SPECS.items():
        enh = result.df[f"{name}_ENH"].to_numpy()
        finite = enh[np.isfinite(enh)]
        assert finite.size > 0
        assert finite.min() >= spec.lower_bound - 1e-9
        assert finite.max() <= spec.upper_bound + 1e-9


def test_wiener_direct_call_matches_signature():
    """The Wiener solver must be callable directly with a spec and target resolution."""
    y = np.full(120, 60.0)
    x, recon, diag = deconvolve_curve_wiener(y, DZ, DEFAULT_LEGACY_SPECS["GR"], 0.3048)
    assert x.shape == y.shape
    assert recon.shape == y.shape
    assert diag.curve_name == "GR"


def test_noise_estimate_respects_floor_and_cap():
    """The MAD noise estimate must be floored and capped around the tool noise floor."""
    spec = DEFAULT_LEGACY_SPECS["RHOB"]
    rng = np.random.default_rng(17)
    # Pure noise at the floor level
    y = rng.normal(2.4, spec.noise_floor, 400)
    est = estimate_noise_mad(y, np.ones(400), spec.noise_floor)
    assert spec.noise_floor <= est <= 3.0 * spec.noise_floor

    # Zero-variance input must return the floor
    flat = np.full(400, 2.4)
    assert estimate_noise_mad(flat, np.ones(400), spec.noise_floor) == pytest.approx(
        spec.noise_floor
    )


def test_noise_cap_mult_is_threaded_to_solver():
    """``noise_cap_mult`` must reach the noise estimator for both solvers."""
    df, _, _ = _synthetic_lrlc(n=300, seed=21)

    capped = harmonize_nuclear_logs(df, mode="enhance_all", noise_cap_mult=1.0)
    uncapped = harmonize_nuclear_logs(df, mode="enhance_all", noise_cap_mult=None)
    for name in ("GR", "NPHI", "RHOB"):
        floor = DEFAULT_LEGACY_SPECS[name].noise_floor
        assert capped.diagnostics[name].estimated_noise_std == pytest.approx(floor)
        assert uncapped.diagnostics[name].estimated_noise_std > floor

    wiener_capped = harmonize_nuclear_logs(
        df, mode="enhance_all", method="wiener", noise_cap_mult=1.0
    )
    wiener_uncapped = harmonize_nuclear_logs(
        df, mode="enhance_all", method="wiener", noise_cap_mult=None
    )
    for name in ("GR", "NPHI", "RHOB"):
        floor = DEFAULT_LEGACY_SPECS[name].noise_floor
        assert wiener_capped.diagnostics[name].estimated_noise_std == pytest.approx(
            floor
        )
        assert wiener_uncapped.diagnostics[name].estimated_noise_std > floor


def test_depth_alignment_detects_shift():
    """A deliberate one-sample shift must be reported by the alignment diagnostic."""
    n = 300
    depth = 1000.0 + np.arange(n) * DZ
    base = np.where((np.arange(n) // 5) % 2 == 0, 2.2, 2.6)
    df = pd.DataFrame(
        {
            "DEPTH": depth,
            "RHOB_ENH": base,
            "NPHI_ENH": np.roll(base, 1),
        }
    )
    lags = check_interlog_depth_alignment(df, warn=False)
    assert lags["NPHI_ENH"] != 0


def test_custom_specs_and_curves_subset():
    """Custom specs and a curve subset must be honored."""
    df, _, _ = _synthetic_lrlc(n=200, seed=18)
    custom = {
        "GR": ToolResolutionSpec(
            curve_name="GR",
            fwhm_m=0.9,
            lower_bound=0.0,
            upper_bound=250.0,
            noise_floor=0.5,
        )
    }
    result = harmonize_nuclear_logs(df, specs=custom, curves=["GR"])
    assert "GR_ENH" in result.df.columns
    assert "NPHI_ENH" not in result.df.columns
    assert result.diagnostics["GR"].intrinsic_fwhm_m == pytest.approx(0.9)


def test_diagnostics_and_result_metadata():
    """The result container must expose consistent diagnostics and metadata."""
    df, _, _ = _synthetic_lrlc(n=200, seed=19)
    result = harmonize_nuclear_logs(df, mode="enhance_all")
    assert result.target_fwhm_m > 0
    assert result.depth_step_m == pytest.approx(DZ)
    assert set(result.diagnostics) == {"GR", "NPHI", "RHOB"}
    for diag in result.diagnostics.values():
        assert diag.converged
        assert diag.iterations > 0
        assert np.isfinite(diag.weighted_recon_rmse)
        assert isinstance(diag.as_dict(), dict)


def test_raw_curves_are_not_modified():
    """Harmonization must never mutate the input raw curves."""
    df, _, _ = _synthetic_lrlc(n=200, seed=20)
    original = df.copy()
    harmonize_nuclear_logs(df, mode="enhance_all")
    pd.testing.assert_frame_equal(df, original)
