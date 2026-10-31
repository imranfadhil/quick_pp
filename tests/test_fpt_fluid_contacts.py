"""Unit tests for the Formation Pressure Test (FPT) fluid gradient & contact analyzer."""

import numpy as np
import pandas as pd
import pytest

from quick_pp.fluid_type import (
    FormationPressureAnalyzer,
    InvalidFluidGradientError,
    density_to_gradient,
    gradient_to_density,
)


def _three_phase_df(noise: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """Synthetic gas / oil / water column (psi, ft) with GOC=5000 ft and OWC=5300 ft."""
    rng = np.random.default_rng(seed)
    m_g, m_o, m_w = 0.08, 0.33, 0.45
    p_owc = 2400.0
    owc, goc = 5300.0, 5000.0
    c_w = p_owc - m_w * owc
    c_o = p_owc - m_o * owc
    p_goc = m_o * goc + c_o
    c_g = p_goc - m_g * goc
    rows = []
    for z in np.arange(4850, 4990, 20):
        rows.append((z, m_g * z + c_g, "GAS"))
    for z in np.arange(5020, 5290, 30):
        rows.append((z, m_o * z + c_o, "OIL"))
    for z in np.arange(5320, 5500, 30):
        rows.append((z, m_w * z + c_w, "WATER"))
    df = pd.DataFrame(rows, columns=["TVDSS", "PRESSURE", "FLUID_TYPE"])
    if noise:
        df["PRESSURE"] += rng.normal(0, noise, len(df))
    df["TEST_QUALITY"] = "VALID"
    return df


def test_density_conversions_match_spec_constants():
    assert gradient_to_density(0.433527, "psi", "ft") == pytest.approx(1.0, rel=1e-4)
    assert gradient_to_density(1.42233, "psi", "m") == pytest.approx(1.0, rel=1e-4)
    assert gradient_to_density(0.0980665, "bar", "m") == pytest.approx(1.0, rel=1e-6)
    assert gradient_to_density(9.80665, "kPa", "m") == pytest.approx(1.0, rel=1e-6)
    assert density_to_gradient(gradient_to_density(0.33, "psi", "ft"), "psi", "ft") == pytest.approx(0.33)


def test_three_phase_contacts_exact():
    fpa = FormationPressureAnalyzer(_three_phase_df(), pressure_unit="psi", depth_unit="ft")
    fpa.fit_all()
    assert fpa.gradients["OIL"].gradient == pytest.approx(0.33, abs=1e-9)
    contacts = fpa.estimate_contacts(n_simulations=500)
    assert set(contacts) == {"GOC", "OWC"}  # GWC suppressed when oil present
    assert contacts["GOC"].tvdss_p50 == pytest.approx(5000.0, abs=1e-6)
    assert contacts["OWC"].tvdss_p50 == pytest.approx(5300.0, abs=1e-6)
    assert contacts["OWC"].pressure_at_contact == pytest.approx(2400.0, abs=1e-6)


def test_noisy_contacts_have_ordered_uncertainty_band():
    fpa = FormationPressureAnalyzer(_three_phase_df(noise=1.5, seed=3), "psi", "ft")
    fpa.fit_all()
    owc = fpa.estimate_contacts(n_simulations=2000)["OWC"]
    assert owc.tvdss_p10 < owc.tvdss_p50 < owc.tvdss_p90
    assert abs(owc.tvdss_p50 - 5300.0) < 40.0
    assert owc.std_tvdss > 0


def test_inverted_gradient_raises_and_is_skipped():
    df = _three_phase_df()
    # Swap labels so 'WATER' is lighter than 'OIL'
    df["FLUID_TYPE"] = df["FLUID_TYPE"].replace({"OIL": "WATER", "WATER": "OIL"})
    df = df[df["FLUID_TYPE"] != "GAS"]
    fpa = FormationPressureAnalyzer(df, "psi", "ft")
    fpa.fit_all()
    with pytest.raises(InvalidFluidGradientError):
        fpa.solve_contact(fpa.gradients["OIL"], fpa.gradients["WATER"])
    assert "OWC" not in fpa.estimate_contacts()
    assert any("OWC" in w for w in fpa.warnings)


def test_single_water_point_with_fixed_gradient_anchor():
    df = _three_phase_df()
    water_idx = df.index[df["FLUID_TYPE"] == "WATER"]
    df = df.drop(water_idx[1:])  # keep a single aquifer point
    fpa = FormationPressureAnalyzer(df, "psi", "ft")
    fpa.fit_all(overrides={"WATER": {"gradient": 0.45}})
    assert fpa.gradients["WATER"].method == "fixed_gradient"
    assert fpa.estimate_contacts()["OWC"].tvdss_p50 == pytest.approx(5300.0, abs=1e-6)


def test_manual_regional_aquifer_line_without_water_points():
    df = _three_phase_df()
    df = df[df["FLUID_TYPE"] != "WATER"]
    fpa = FormationPressureAnalyzer(df, "psi", "ft")
    fpa.fit_all(
        overrides={"WATER": {"gradient": 0.45, "anchor_tvdss": 5400.0, "anchor_pressure": 2445.0}}
    )
    assert fpa.gradients["WATER"].method == "manual"
    assert fpa.estimate_contacts()["OWC"].tvdss_p50 == pytest.approx(5300.0, abs=1e-6)


def test_supercharged_outlier_flagged_and_excluded():
    df = _three_phase_df()
    oil_idx = df.index[df["FLUID_TYPE"] == "OIL"]
    df.loc[oil_idx[2], "PRESSURE"] += 60.0  # supercharged point
    fpa = FormationPressureAnalyzer(df, "psi", "ft")
    fpa.auto_classify_points(ransac_residual_threshold=5.0, assign_unassigned=False)
    assert fpa.df.loc[oil_idx[2], "TEST_QUALITY"] == "SUPERCHARGED"
    fpa.fit_all()
    assert fpa.gradients["OIL"].gradient == pytest.approx(0.33, abs=1e-6)


def test_auto_assign_phases_and_tight_flag():
    df = _three_phase_df()
    df["MOBILITY"] = 50.0
    df.loc[df.index[0], "MOBILITY"] = 0.05
    df["FLUID_TYPE"] = "UNASSIGNED"
    fpa = FormationPressureAnalyzer(df, "psi", "ft")
    fpa.auto_classify_points(min_mobility=0.5)
    assert fpa.df.loc[df.index[0], "TEST_QUALITY"] == "TIGHT"
    # Interior points (away from contacts) should be classified correctly
    assigned = fpa.df["FLUID_TYPE"]
    assert assigned.iloc[3] == "GAS"
    assert assigned.iloc[11] == "OIL"
    assert assigned.iloc[-2] == "WATER"


def test_excess_pressure_zero_on_water_line_and_metric_units():
    df = _three_phase_df()
    # Convert to bar + m
    df["TVDSS"] = df["TVDSS"] * 0.3048
    df["PRESSURE"] = df["PRESSURE"] * 0.0689476
    fpa = FormationPressureAnalyzer(df, "bar", "m")
    fpa.fit_all()
    assert fpa.gradients["WATER"].density_gcc == pytest.approx(
        gradient_to_density(0.45, "psi", "ft"), rel=1e-4
    )
    owc = fpa.estimate_contacts()["OWC"]
    assert owc.tvdss_p50 == pytest.approx(5300.0 * 0.3048, abs=1e-3)
    exc = fpa.calculate_excess_pressure()
    water_mask = fpa.df["FLUID_TYPE"] == "WATER"
    assert np.allclose(exc[water_mask], 0.0, atol=1e-6)
    assert (exc[fpa.df["FLUID_TYPE"] == "OIL"] > 0).all()


def test_plot_formation_pressure_renders():
    import matplotlib

    matplotlib.use("Agg")
    from quick_pp.plotter.plotter import plot_formation_pressure

    df = _three_phase_df(noise=1.0)
    df["WELL_NAME"] = ["W1" if i % 2 else "W2" for i in range(len(df))]
    df.loc[df.index[4], "TEST_QUALITY"] = "TIGHT"
    fpa = FormationPressureAnalyzer(df, "psi", "ft")
    fpa.fit_all()
    fpa.estimate_contacts(n_simulations=200)
    fig = plot_formation_pressure(fpa, show_excess=True)
    assert len(fig.axes) == 2
