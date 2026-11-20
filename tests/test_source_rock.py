import numpy as np
import pandas as pd

from quick_pp.source_rock import (
    auto_detect_shale_baseline,
    calibrate_toc_to_core,
    estimate_toc_passey,
    hood_lom_to_ro,
    hood_ro_to_lom,
    passey_delta_log_r,
)


def test_hood_ro_lom_conversion():
    # Test typical maturation milestones
    assert 6.0 <= hood_ro_to_lom(0.4) <= 7.0
    assert 9.0 <= hood_ro_to_lom(0.8) <= 10.0
    assert 10.0 <= hood_ro_to_lom(1.0) <= 11.0
    assert 12.5 <= hood_ro_to_lom(2.0) <= 14.0

    # Roundtrip test
    for ro in [0.5, 0.7, 0.9, 1.2, 1.5]:
        lom = hood_ro_to_lom(ro)
        ro_back = hood_lom_to_ro(lom)
        assert abs(ro_back - ro) < 0.02


def test_passey_delta_log_r_sonic():
    # In baseline shale: R=2.0, DT=80 -> Delta Log R = 0
    dlogr = passey_delta_log_r(2.0, 80.0, r_baseline=2.0, poro_baseline=80.0, porosity_type="sonic")
    assert abs(dlogr) < 1e-6

    # In organic shale with 1 decade resistivity and 50 us/ft separation:
    # dlogr = log10(20/2) + 0.02 * (130 - 80) = 1.0 + 1.0 = 2.0
    dlogr_sep = passey_delta_log_r(20.0, 130.0, r_baseline=2.0, poro_baseline=80.0, porosity_type="sonic")
    assert abs(dlogr_sep - 2.0) < 1e-6


def test_passey_delta_log_r_density():
    # In baseline shale: R=2.0, RHOB=2.50
    dlogr = passey_delta_log_r(2.0, 2.50, r_baseline=2.0, poro_baseline=2.50, porosity_type="density")
    assert abs(dlogr) < 1e-6

    # In organic shale with RHOB drop: RHOB = 2.30 (drop of 0.20 g/cc)
    # Default coeff = -2.5
    # dlogr = log10(20/2) + (-2.5) * (2.30 - 2.50) = 1.0 + (-2.5 * -0.2) = 1.0 + 0.5 = 1.5
    dlogr_den = passey_delta_log_r(20.0, 2.30, r_baseline=2.0, poro_baseline=2.50, porosity_type="density")
    assert abs(dlogr_den - 1.5) < 1e-6


def test_passey_delta_log_r_neutron():
    # In baseline shale: R=2.0, NPHI=0.35
    dlogr = passey_delta_log_r(2.0, 0.35, r_baseline=2.0, poro_baseline=0.35, porosity_type="neutron")
    assert abs(dlogr) < 1e-6

    # In organic shale with NPHI increase: NPHI = 0.55 (+0.20)
    # Default coeff = 4.0
    # dlogr = log10(20/2) + 4.0 * (0.55 - 0.35) = 1.0 + 0.8 = 1.8
    dlogr_neu = passey_delta_log_r(20.0, 0.55, r_baseline=2.0, poro_baseline=0.35, porosity_type="neutron")
    assert abs(dlogr_neu - 1.8) < 1e-6


def test_estimate_toc_passey():
    # With DlogR = 2.0, LOM = 10.5, TOC_base = 0.8
    # factor = 10^(2.297 - 0.1688 * 10.5) = 10^0.5246 = 3.3465
    # TOC = 2.0 * 3.3465 + 0.8 = 7.493 wt%
    toc = estimate_toc_passey(2.0, lom=10.5, toc_baseline=0.8)
    assert 7.4 < toc < 7.6

    # Negative DlogR should clamp to background when clamp_negative=True
    toc_neg = estimate_toc_passey(-1.0, lom=10.5, toc_baseline=0.8, clamp_negative=True)
    assert abs(toc_neg - 0.8) < 1e-6


def test_estimate_toc_series():
    dlogr_s = pd.Series([0.0, 1.0, 2.0, -0.5])
    toc_s = estimate_toc_passey(dlogr_s, ro=0.8, toc_baseline=0.5)
    assert len(toc_s) == 4
    assert toc_s.iloc[0] == 0.5  # baseline
    assert toc_s.iloc[3] == 0.5  # clamped negative
    assert toc_s.iloc[2] > toc_s.iloc[1] > toc_s.iloc[0]


def test_auto_detect_shale_baseline():
    n = 100
    df = pd.DataFrame({
        "DEPTH": np.linspace(1000, 1100, n),
        "RD": np.concatenate([np.full(50, 2.0), np.full(50, 25.0)]),
        "DT": np.concatenate([np.full(50, 85.0), np.full(50, 120.0)]),
        "VSHALE": np.concatenate([np.full(50, 0.85), np.full(50, 0.20)]),
    })
    res = auto_detect_shale_baseline(df, res_col="RD", poro_col="DT", vsh_col="VSHALE")
    assert res["status"] == "auto_detected"
    assert abs(res["r_baseline"] - 2.0) < 0.1
    assert abs(res["poro_baseline"] - 85.0) < 0.1


def test_calibrate_toc_to_core():
    dlogr = np.array([0.5, 1.0, 1.5, 2.0, 2.5])
    # True relation: Core_TOC = 3.0 * DlogR + 0.8
    core_toc = 3.0 * dlogr + 0.8
    cal = calibrate_toc_to_core(dlogr, core_toc)
    assert cal["success"] is True
    assert abs(cal["slope"] - 3.0) < 0.05
    assert abs(cal["intercept"] - 0.8) < 0.05
    assert cal["r2"] > 0.99
    assert cal["implied_lom"] is not None
