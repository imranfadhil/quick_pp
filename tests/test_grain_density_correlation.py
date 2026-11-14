"""Tests for the multivariate (ridge / Huber) grain-density correlation."""

import numpy as np
import pytest

from quick_pp.scal import (
    fit_grain_density_correlation_multivariate,
    grain_density_from_correlation_multivariate,
)


def _synthetic(n=120, seed=7):
    """Build a heteroscedastic multi-log dataset around a known linear trend."""
    rng = np.random.default_rng(seed)
    gr = rng.normal(60, 30, n)
    cgr = 0.8 * gr + rng.normal(5, 12, n)
    rhob = 2.35 - 0.0008 * gr + rng.normal(0, 0.05, n)
    nphi = 0.15 + 0.0006 * gr + rng.normal(0, 0.04, n)
    cgd = 2.60 + 0.0012 * gr + 0.35 * (rhob - 2.35) + rng.normal(0, 0.012, n)
    return np.column_stack([gr, cgr, rhob, nphi]), cgd


def test_multivariate_fit_recovers_trend():
    """The multi-log fit must recover most of the variance of a known linear composite."""
    x, y = _synthetic()
    fit = fit_grain_density_correlation_multivariate(
        x, y, driver_names=["GR", "CGR", "RHOB", "NPHI"]
    )

    assert fit["count"] == len(y)
    assert fit["r2"] > 0.8
    assert fit["rmse"] < 0.02
    assert fit["robust"] is True
    assert fit["n_iter"] > 0
    assert fit["driver_names"] == ["GR", "CGR", "RHOB", "NPHI"]
    assert len(fit["coef"]) == 4
    assert len(fit["native_slopes"]) == 4
    # RHOB is the strongest secondary driver in the synthetic trend.
    assert fit["native_slopes"][2] > 0.1


def test_multivariate_beats_single_driver():
    """Several drivers must explain more grain-density variance than GR alone."""
    x, y = _synthetic()
    multi = fit_grain_density_correlation_multivariate(
        x, y, driver_names=["GR", "CGR", "RHOB", "NPHI"]
    )
    single = fit_grain_density_correlation_multivariate(
        x[:, :1], y, driver_names=["GR"], robust=False
    )
    assert multi["r2"] > single["r2"]


def test_multivariate_huber_downweights_outliers():
    """Huber reweighting must keep residuals near the bulk when gross outliers are injected."""
    x, y = _synthetic()
    spiked = y.copy()
    spiked[:6] = 2.95

    robust = fit_grain_density_correlation_multivariate(
        x, spiked, driver_names=["GR", "CGR", "RHOB", "NPHI"], robust=True
    )
    ols = fit_grain_density_correlation_multivariate(
        x, spiked, driver_names=["GR", "CGR", "RHOB", "NPHI"], robust=False
    )
    assert robust["robust_rmse"] < ols["robust_rmse"]


def test_multivariate_diagnostics_reported():
    """VIF and Breusch-Pagan diagnostics must be present and physically bounded."""
    x, y = _synthetic()
    fit = fit_grain_density_correlation_multivariate(
        x, y, driver_names=["GR", "CGR", "RHOB", "NPHI"]
    )

    vif = fit["vif"]
    assert set(vif) == {"GR", "CGR", "RHOB", "NPHI"}
    assert all(v >= 1.0 for v in vif.values())
    # GR and CGR are intentionally collinear in the fixture.
    assert vif["GR"] > 2.0
    assert 0.0 <= fit["breusch_pagan_p"] <= 1.0
    assert fit["resid_mad"] > 0.0


def test_multivariate_prediction_roundtrip_and_missing_driver():
    """Applying the fit reproduces the trend and falls back to the intercept for NaN drivers."""
    x, y = _synthetic()
    fit = fit_grain_density_correlation_multivariate(
        x, y, driver_names=["GR", "CGR", "RHOB", "NPHI"]
    )
    pred = grain_density_from_correlation_multivariate(
        x, fit, driver_names=fit["driver_names"]
    )

    assert pred.shape == y.shape
    assert np.corrcoef(pred, y)[0, 1] > 0.85
    assert pred.min() >= fit["rho_ma_min"] and pred.max() <= fit["rho_ma_max"]

    gappy = x.copy()
    gappy[:, 2] = np.nan
    pred_gap = grain_density_from_correlation_multivariate(
        gappy, fit, driver_names=fit["driver_names"]
    )
    assert np.all(np.isfinite(pred_gap))


def test_multivariate_accepts_dict_and_insufficient_data():
    """Dict input is supported and an empty sample set returns a zero-effect fit."""
    x, y = _synthetic()
    fit = fit_grain_density_correlation_multivariate(
        {"GR": x[:, 0], "RHOB": x[:, 2]}, y
    )
    assert fit["driver_names"] == ["GR", "RHOB"]
    assert fit["count"] == len(y)

    empty = fit_grain_density_correlation_multivariate(
        x, np.full(len(y), np.nan), driver_names=["GR", "CGR", "RHOB", "NPHI"]
    )
    assert empty["count"] == 0
    assert empty["coef"] == [0.0, 0.0, 0.0, 0.0]


def test_multivariate_rejects_shape_mismatch():
    """Driver/sample or driver/name cardinality mismatches must raise."""
    x, y = _synthetic(n=20)
    with pytest.raises(ValueError):
        fit_grain_density_correlation_multivariate(
            x, y[:10], driver_names=["GR", "CGR", "RHOB", "NPHI"]
        )
    with pytest.raises(ValueError):
        fit_grain_density_correlation_multivariate(x, y, driver_names=["GR", "CGR"])
