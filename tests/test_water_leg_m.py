import numpy as np
import pytest

from quick_pp.saturation import (
    calibrate_water_leg_m,
    compute_continuous_m,
    estimate_m_borai,
    estimate_m_shell_lucia,
)


def test_estimate_m_borai():
    phi = np.array([0.10, 0.20, 0.30])
    m_borai = estimate_m_borai(phi)
    assert len(m_borai) == 3
    # Borai formula: 2.2 - 0.035 / (phi + 0.042)
    assert np.all(m_borai > 1.8)
    assert np.all(m_borai < 2.2)


def test_estimate_m_shell_lucia():
    phi = np.array([0.05, 0.10, 0.20])
    m_shell = estimate_m_shell_lucia(phi)
    assert len(m_shell) == 3
    # Shell / Lucia formula: 1.87 + 0.019 / phi
    # phi = 0.05 -> m = 1.87 + 0.38 = 2.25
    # phi = 0.20 -> m = 1.87 + 0.095 = 1.965
    assert pytest.approx(m_shell[0], abs=0.01) == 2.25
    assert pytest.approx(m_shell[2], abs=0.01) == 1.965
    # Test m increases at low porosity
    assert m_shell[0] > m_shell[1] > m_shell[2]


def test_compute_continuous_m():
    phi = np.array([0.15, 0.25])
    m_cal = compute_continuous_m(phi, method="calibrated", slope=1.5, intercept=1.7)
    assert pytest.approx(m_cal[0], abs=0.01) == 1.5 * 0.15 + 1.7

    m_bor = compute_continuous_m(phi, method="borai")
    assert np.all(m_bor > 1.9) and np.all(m_bor < 2.2)

    m_sh = compute_continuous_m(phi, method="shell_lucia")
    assert np.all(m_sh > 1.9) and np.all(m_sh < 2.1)


def test_calibrate_water_leg_m_tortuosity_factor():
    n = 60
    phit = np.linspace(0.15, 0.30, n)
    rw, a, true_m = 0.05, 0.81, 2.0
    rt = a * rw / (phit**true_m)

    # Archie: ignoring a biases m; passing a recovers the true value
    biased = calibrate_water_leg_m(rt=rt, rw=rw, phit=phit, method="archie")
    fixed = calibrate_water_leg_m(rt=rt, rw=rw, phit=phit, method="archie", a=a)
    assert abs(fixed["mean_m"] - true_m) < 0.01
    assert abs(biased["mean_m"] - true_m) > 0.05

    # Indonesian with Vsh=0 reduces to Archie, so `a` must be honoured there too
    indo = calibrate_water_leg_m(
        rt=rt, rw=rw, phit=phit, vsh=np.zeros(n), method="indonesian", a=a
    )
    assert abs(indo["mean_m"] - true_m) < 0.01

    # Invalid `a` falls back to 1.0
    assert (
        calibrate_water_leg_m(rt=rt, rw=rw, phit=phit, a=-1.0)["mean_m"]
        == biased["mean_m"]
    )


def test_calibrate_water_leg_m():
    # Synthetic clean water zone
    n = 100
    phit = np.linspace(0.15, 0.30, n)
    rw = 0.05
    # True m = 2.0 -> Ro = a * Rw / phi^m
    a = 1.0
    true_m = 2.0
    rt = a * rw / (phit**true_m)
    vsh = np.full(n, 0.05)

    res = calibrate_water_leg_m(rt=rt, rw=rw, phit=phit, vsh=vsh, method="archie")
    assert res["count"] == n
    assert pytest.approx(res["mean_m"], abs=0.05) == true_m
    assert res["slope"] is not None
    assert res["intercept"] is not None
