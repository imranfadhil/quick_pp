"""Unit tests for SCAL core corrections and core analysis functions (`quick_pp.core_analysis`)."""

import numpy as np
import pandas as pd
import pytest

from quick_pp.core_analysis import (
    correct_core_data,
    fit_j_curve,
    fit_poroperm_curve,
    humidity_to_total_porosity,
    klinkenberg_permeability_correction,
    leverett_j,
    normalize_sw,
    overburden_porosity_correction,
    perm_transform,
    skelt_harrison_func,
)


def test_overburden_porosity_correction():
    """Verify exponential and power-law overburden stress compaction on core porosity."""
    poro_amb = 0.20
    ncs = 3000.0

    # Exponential compaction: 0.20 * exp(-4e-6 * 3000)
    poro_exp = overburden_porosity_correction(poro_amb, ncs, compaction_coeff=4e-6)
    expected_exp = 0.20 * np.exp(-4e-6 * 3000.0)
    np.testing.assert_allclose(poro_exp, expected_exp, rtol=1e-6)
    assert poro_exp < poro_amb

    # Power-law compaction: 0.20 * (3000 / 400)^(-0.03)
    poro_pow = overburden_porosity_correction(
        poro_amb, ncs, compaction_coeff=0.03, ambient_stress=400.0, method="power_law"
    )
    expected_pow = 0.20 * ((3000.0 / 400.0) ** (-0.03))
    np.testing.assert_allclose(poro_pow, expected_pow, rtol=1e-6)
    assert poro_pow < poro_amb

    # Zero stress difference returns unchanged porosity
    np.testing.assert_allclose(
        overburden_porosity_correction(poro_amb, 400.0, ambient_stress=400.0),
        poro_amb,
    )


def test_klinkenberg_permeability_correction():
    """Verify Klinkenberg gas slippage permeability correction with empirical and explicit b-factor."""
    k_air = np.array([1.0, 10.0, 100.0])
    mean_pressure = 50.0

    # Default Jones & Owens (1980) empirical b-factor
    k_liq = klinkenberg_permeability_correction(k_air, mean_pressure)
    expected_b = 12.64 * (k_air ** (-0.33))
    expected_k = k_air / (1.0 + expected_b / mean_pressure)
    np.testing.assert_allclose(k_liq, expected_k, rtol=1e-6)
    assert np.all(k_liq < k_air)

    # Explicit b-factor
    k_liq_explicit = klinkenberg_permeability_correction(10.0, 50.0, b_factor=5.0)
    np.testing.assert_allclose(k_liq_explicit, 10.0 / (1.0 + 5.0 / 50.0), rtol=1e-6)

    # Zero or negative k_air returns 0.0 safely
    assert klinkenberg_permeability_correction(0.0, 50.0) == 0.0


def test_humidity_to_total_porosity():
    """Verify reconciling humidity-dried (PHIE) and oven-dried (PHIT) core plugs with CBW."""
    # Humidity-dried plug (PHIE = 0.15) + CBW (0.04) -> PHIT = 0.19
    phit = humidity_to_total_porosity(0.15, 0.04, oven_dried=False)
    np.testing.assert_allclose(phit, 0.19, rtol=1e-6)

    # Oven-dried plug (PHIT = 0.19) - CBW (0.04) -> PHIE = 0.15
    phie = humidity_to_total_porosity(0.19, 0.04, oven_dried=True)
    np.testing.assert_allclose(phie, 0.15, rtol=1e-6)

    # Clipped safely to [0.0, 1.0]
    assert humidity_to_total_porosity(0.02, 0.05, oven_dried=True) == 0.0
    assert humidity_to_total_porosity(0.98, 0.05, oven_dried=False) == 1.0


def test_correct_core_data_pipeline_helper():
    """Verify DataFrame helper preserves backup columns and applies overburden, CBW, and Klinkenberg corrections."""
    df = pd.DataFrame(
        {
            "CPORE": [0.15, 0.20],
            "CPERM": [10.0, 100.0],
            "PHIT": [0.18, 0.22],
            "PHIE": [0.15, 0.20],
            "NCS": [2500.0, 3000.0],
        }
    )

    res = correct_core_data(
        df,
        net_confining_stress="NCS",
        mean_pore_pressure=50.0,
        cbw=True,
        replace_ori=False,
    )

    assert "CPORE_AMB" in res.columns
    assert "CPERM_AIR" in res.columns
    np.testing.assert_allclose(res["CPORE_AMB"], df["CPORE"])
    np.testing.assert_allclose(res["CPERM_AIR"], df["CPERM"])

    expected_poro = overburden_porosity_correction(df["CPORE"], df["NCS"]) + (
        df["PHIT"] - df["PHIE"]
    )
    np.testing.assert_allclose(res["CPORE"], expected_poro, rtol=1e-6)
    assert np.all(res["CPERM"] < res["CPERM_AIR"])

    # Error when cbw=True but PHIT/PHIE are missing
    with pytest.raises(ValueError, match="Cannot infer CBW"):
        correct_core_data(df[["CPORE", "CPERM"]], cbw=True)


def test_core_fitting_and_transforms():
    """Verify Leverett-J, poroperm fitting, Skelt-Harrison, and Sw normalization."""
    poro = np.array([0.10, 0.15, 0.20, 0.25])
    perm = perm_transform(poro, 10000.0, 3.0)
    a, b = fit_poroperm_curve(poro, perm)
    np.testing.assert_allclose(a, 10000.0, rtol=1e-2)
    np.testing.assert_allclose(b, 3.0, rtol=1e-2)

    sw = np.array([0.2, 0.4, 0.6, 0.8, 1.0])
    j_vals = 0.15 * (sw ** (-1.5))
    ja, jb = fit_j_curve(sw, j_vals)
    np.testing.assert_allclose(ja, 0.15, atol=1e-2)
    np.testing.assert_allclose(jb, 1.5, atol=1e-2)

    swn = normalize_sw(sw)
    assert swn[0] == 0.0
    assert swn[-1] == 1.0

    j_calc = leverett_j(pc=10.0, ift=30.0, theta=0.0, perm=100.0, phit=0.25)
    np.testing.assert_allclose(j_calc, 0.21665 * (10.0 / 30.0) * 20.0, rtol=1e-5)

    sh_sw = skelt_harrison_func(h=100.0, a=0.8, b=50.0, c=1.5, d=5.0)
    assert 0.0 < sh_sw < 1.0

