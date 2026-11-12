"""Tests for the continuous apparent fluid density (RHO_FL) iterative solver."""

import time

import numpy as np
import pytest

from quick_pp.porosity import (
    RHO_FL_TIER_RXO,
    RHO_FL_TIER_SW_FALLBACK,
    RHO_FL_TIER_ZONAL,
    estimate_continuous_rho_fluid,
    salinity_from_rmf,
    solve_continuous_rho_fluid_and_phid,
)

RHO_MA = 2.65


def _phid(rhob, rho_fl, rho_ma=RHO_MA):
    return (rho_ma - rhob) / (rho_ma - rho_fl)


# --------------------------------------------------------------------------- salinity
def test_salinity_from_rmf_monotonic_and_bounded():
    fresh = salinity_from_rmf(1.0, 25.0)
    saline = salinity_from_rmf(0.05, 25.0)
    assert 0.0 <= fresh < saline <= 350_000.0
    # A lower Rmf at the same temperature implies a higher salinity
    assert salinity_from_rmf(0.2, 25.0) > salinity_from_rmf(0.5, 25.0)
    with pytest.raises(ValueError, match="rmf"):
        salinity_from_rmf(0.0)


# --------------------------------------------------------------------------- tier 1
def test_high_salinity_wbm_aquifer():
    """180,000 ppm WBM in a water zone yields rho_fl ~ 1.10 and lifts PHID ~2 p.u."""
    n = 200
    rhob = np.full(n, 2.14)
    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob,
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="WBM",
        rho_mf=1.10,
        rho_w=1.10,
        fluid_zone=np.array(["WATER"] * n),
    )
    assert np.allclose(result["RHO_FL"], 1.10, atol=1e-6)
    assert np.allclose(result["SXO_EST"], 1.0, atol=1e-6)
    assert np.allclose(result["RHO_FL_TIER"], RHO_FL_TIER_ZONAL)

    phid_corr = result["PHID"][0]
    phid_raw = result["PHID_RAW_100"][0]
    assert phid_corr == pytest.approx(_phid(2.14, 1.10), abs=1e-6)
    assert phid_raw == pytest.approx(_phid(2.14, 1.0), abs=1e-6)
    # ~2 p.u. uplift versus the static 1.0 g/cc assumption
    assert (phid_corr - phid_raw) * 100 == pytest.approx(2.0, abs=0.2)


def test_tier1_rxo_archie_sxo():
    """A valid Rxo + Rmf routes to Tier 1 and honours Archie's Sxo."""
    n = 50
    rhob = np.full(n, 2.30)
    rxo = np.full(n, 5.0)
    rmf = np.full(n, 0.05)
    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob,
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="WBM",
        rho_mf=1.0,
        rho_w=1.02,
        rho_h_app=0.7,
        rxo=rxo,
        rmf_downhole=rmf,
        fluid_zone=np.array(["OIL"] * n),
    )
    assert np.allclose(result["RHO_FL_TIER"], RHO_FL_TIER_RXO)
    sxo = result["SXO_EST"][0]
    phid = result["PHID"][0]
    # Tier 1 must not collapse to Sxo = 1 when Rt is unavailable
    assert 0.0 < sxo < 1.0
    # Reported Sxo, rho_fl, and PHID must be mutually consistent
    assert result["RHO_FL"][0] == pytest.approx(sxo * 1.0 + (1 - sxo) * 0.7, abs=1e-6)
    assert phid == pytest.approx(_phid(2.30, result["RHO_FL"][0]), abs=1e-6)
    # Archie Sxo from Rxo/Rmf, clipped to the zonal residual lower bound
    archie_sxo = ((1.0 * 0.05) / (max(phid, 0.01) ** 2 * 5.0)) ** 0.5
    assert sxo == pytest.approx(np.clip(archie_sxo, 0.85, 1.0), abs=1e-6)


# --------------------------------------------------------------------------- tier 2
def test_obm_oil_leg_coupling():
    """OBM (rho_mf=0.80) with Sw=0.15 converges near 0.83 g/cc and lowers PHID."""
    n = 100
    rhob = np.full(n, 2.35)
    # Choose Rt/Rw so the Archie Sw lands on 0.15 at the converged porosity.
    phid_guess = _phid(2.35, 0.83)
    rw = 0.05
    rt = (1.0 * rw) / (max(phid_guess, 0.01) ** 2 * 0.15**2)
    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob,
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="OBM",
        rho_mf=0.80,
        rho_w=1.02,
        rt=np.full(n, rt),
        rw_downhole=np.full(n, rw),
        fluid_zone=np.array(["OIL"] * n),
    )
    assert np.allclose(result["RHO_FL_TIER"], RHO_FL_TIER_SW_FALLBACK)
    assert result["SXO_EST"][0] == pytest.approx(0.15, abs=0.02)
    assert result["RHO_FL"][0] == pytest.approx(0.83, abs=0.02)

    phid_corr = result["PHID"][0]
    phid_raw = result["PHID_RAW_100"][0]
    # Lower fluid density -> lower density porosity than the 1.0 g/cc assumption
    assert phid_corr < phid_raw
    assert (phid_raw - phid_corr) * 100 == pytest.approx(2.5, abs=1.0)


def test_wbm_tier2_poupon_coupling():
    n = 40
    rhob = np.full(n, 2.30)
    rw = 0.05
    rt = np.full(n, 20.0)
    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob,
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="WBM",
        rho_mf=1.0,
        rho_w=1.02,
        rho_h_app=0.7,
        rt=rt,
        rw_downhole=np.full(n, rw),
        flush_exponent=0.2,
        fluid_zone=np.array(["OIL"] * n),
    )
    assert np.allclose(result["RHO_FL_TIER"], RHO_FL_TIER_SW_FALLBACK)
    phid = result["PHID"][0]
    sw = np.clip(((1.0 * rw) / (max(phid, 0.01) ** 2 * 20.0)) ** 0.5, 0.02, 1.0)
    assert result["SXO_EST"][0] == pytest.approx(np.clip(sw**0.2, sw, 1.0), abs=1e-4)


# --------------------------------------------------------------------------- tier 3
def test_tier3_zonal_residual_saturation():
    n = 30
    rhob = np.full(n, 2.30)
    zone = np.array(["GAS"] * 10 + ["OIL"] * 10 + ["WATER"] * 10)
    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob,
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="WBM",
        rho_mf=1.0,
        rho_w=1.02,
        rho_h_app=0.7,
        fluid_zone=zone,
        default_shr_gas=0.25,
        default_shr_oil=0.15,
    )
    assert np.allclose(result["RHO_FL_TIER"], RHO_FL_TIER_ZONAL)
    assert result["SXO_EST"][0] == pytest.approx(0.75)
    assert result["SXO_EST"][10] == pytest.approx(0.85)
    assert result["SXO_EST"][20] == pytest.approx(1.0)
    # Water zone collapses the hydrocarbon term to formation water
    assert result["RHO_FL"][20] == pytest.approx(1.0)
    # Gas zone mixes filtrate with the apparent hydrocarbon density
    assert result["RHO_FL"][0] == pytest.approx(0.75 * 1.0 + 0.25 * 0.7, abs=1e-6)


# --------------------------------------------------------------------------- degradation
def test_badhole_rxo_graceful_degradation():
    """A washout flagged BADHOLE=1 must drop Tier 1 -> Tier 2 without NaN spikes."""
    n = 100
    rhob = np.full(n, 2.32)
    rxo = np.full(n, 2.0)
    rmf = np.full(n, 0.05)
    rt = np.full(n, 20.0)
    rw = np.full(n, 0.05)
    badhole = np.zeros(n, dtype=bool)
    badhole[40:60] = True  # 10 m washout at 0.5 m sampling

    result = solve_continuous_rho_fluid_and_phid(
        rhob=rhob,
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="WBM",
        rho_mf=1.0,
        rho_w=1.02,
        rho_h_app=0.7,
        rxo=rxo,
        rmf_downhole=rmf,
        rt=rt,
        rw_downhole=rw,
        badhole_flag=badhole,
        fluid_zone=np.array(["OIL"] * n),
    )
    tier = result["RHO_FL_TIER"]
    assert np.all(tier[:40] == RHO_FL_TIER_RXO)
    assert np.all(tier[40:60] == RHO_FL_TIER_SW_FALLBACK)
    assert np.all(tier[60:] == RHO_FL_TIER_RXO)
    assert np.all(np.isfinite(result["RHO_FL"]))
    assert np.all(np.isfinite(result["PHID"]))
    # No discontinuity in rho_fl across the washout boundary
    assert abs(result["RHO_FL"][39] - result["RHO_FL"][40]) < 0.05


def test_missing_resistivity_falls_back_to_tier3():
    n = 20
    result = solve_continuous_rho_fluid_and_phid(
        rhob=np.full(n, 2.3),
        rhoma=RHO_MA,
        tvd=np.arange(n, dtype=float),
        mud_type="WBM",
        rho_mf=1.0,
        fluid_zone=np.array(["OIL"] * n),
    )
    assert np.all(result["RHO_FL_TIER"] == RHO_FL_TIER_ZONAL)
    assert np.all(np.isfinite(result["RHO_FL"]))


# --------------------------------------------------------------------------- convergence
def test_iteration_convergence_speed_and_no_division_warnings():
    n = 10_000
    rng = np.random.default_rng(42)
    rhob = rng.uniform(2.0, 2.6, n)
    # Force rhob -> rhoma at a few samples to exercise the divide-by-zero guard
    rhob[:5] = RHO_MA
    rxo = rng.uniform(0.5, 5.0, n)
    rmf = np.full(n, 0.05)
    rt = rng.uniform(5.0, 50.0, n)
    rw = np.full(n, 0.05)

    with np.errstate(all="raise"):
        start = time.perf_counter()
        result = solve_continuous_rho_fluid_and_phid(
            rhob=rhob,
            rhoma=RHO_MA,
            tvd=np.arange(n, dtype=float),
            mud_type="WBM",
            rho_mf=1.0,
            rho_w=1.02,
            rho_h_app=0.7,
            rxo=rxo,
            rmf_downhole=rmf,
            rt=rt,
            rw_downhole=rw,
            fluid_zone=np.array(["OIL"] * n),
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert elapsed_ms < 150.0
    assert np.all(np.isfinite(result["PHID"]))
    assert np.all(np.isfinite(result["RHO_FL"]))
    assert np.all(result["PHID"] >= 0.0)
    assert np.all(result["PHID"] <= 0.60)


def test_empty_input_returns_empty_arrays():
    result = solve_continuous_rho_fluid_and_phid(
        rhob=np.array([]), rhoma=RHO_MA, tvd=np.array([])
    )
    assert result["PHID"].size == 0
    assert result["RHO_FL_TIER"].size == 0


# --------------------------------------------------------------------------- wrapper
def test_estimate_continuous_rho_fluid_wbm_salinity_profile():
    n = 50
    tvd = np.linspace(1000.0, 3000.0, n)
    temp = 25.0 + 30.0 * tvd / 1000.0
    pres = 0.1013 + 10.2 * tvd / 1000.0
    result = estimate_continuous_rho_fluid(
        rhob=np.full(n, 2.35),
        rhoma=RHO_MA,
        tvd=tvd,
        temp_c=temp,
        pressure_mpa=pres,
        mud_type="WBM",
        salinity_ppm=180000.0,
        fluid_zone=np.array(["WATER"] * n),
    )
    assert np.allclose(result["SALINITY_PPM"], 180000.0)
    # Downhole brine density decreases with temperature along the profile
    assert result["RHO_MF"][0] > result["RHO_MF"][-1]
    assert np.all(result["RHO_MF"] > 1.0)


def test_estimate_continuous_rho_fluid_obm_profile():
    n = 50
    tvd = np.linspace(1000.0, 3000.0, n)
    temp = 25.0 + 30.0 * tvd / 1000.0
    pres = 0.1013 + 10.2 * tvd / 1000.0
    result = estimate_continuous_rho_fluid(
        rhob=np.full(n, 2.35),
        rhoma=RHO_MA,
        tvd=tvd,
        temp_c=temp,
        pressure_mpa=pres,
        mud_type="OBM",
        obm_base_density=0.82,
        fluid_zone=np.array(["OIL"] * n),
    )
    assert np.allclose(result["SALINITY_PPM"], 0.0)
    assert np.all((result["RHO_MF"] > 0.75) & (result["RHO_MF"] < 0.85))


def test_estimate_continuous_rho_fluid_rmf_fallback():
    n = 20
    tvd = np.linspace(1000.0, 2000.0, n)
    temp = 25.0 + 30.0 * tvd / 1000.0
    pres = 0.1013 + 10.2 * tvd / 1000.0
    result = estimate_continuous_rho_fluid(
        rhob=np.full(n, 2.35),
        rhoma=RHO_MA,
        tvd=tvd,
        temp_c=temp,
        pressure_mpa=pres,
        mud_type="WBM",
        rmf=0.05,
        rmf_temp_c=25.0,
        fluid_zone=np.array(["WATER"] * n),
    )
    assert np.all(result["SALINITY_PPM"] > 100_000.0)
    assert np.all(result["RHO_MF"] > 1.0)
