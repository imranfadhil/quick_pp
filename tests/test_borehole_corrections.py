"""Tests for the unified environment model and generic borehole corrections."""

import numpy as np
import pandas as pd
import pytest

from quick_pp.borehole_corrections import (
    ENV_QC_REASON_BITS,
    apply_environmental_corrections,
    classify_invasion_profile,
    confidence_weighted_curve,
    detect_flatline,
    detect_sonic_cycle_skip,
    detect_spikes,
    invert_invasion_profile,
    resistivity_borehole_correction,
)
from quick_pp.environment import (
    Environment,
    MudInterval,
    arps_resistivity,
    environment_from_config,
)


def _well(n=60, cali=8.5):
    depth = np.linspace(2000.0, 2000.0 + 0.5 * (n - 1), n)
    return pd.DataFrame(
        {
            "DEPTH": depth,
            "BS": 8.5,
            "CALI": cali,
            "GR": 60.0 + 10 * np.sin(depth),
            "NPHI": 0.22 + 0.02 * np.sin(depth / 3),
            "RHOB": 2.35 + 0.03 * np.cos(depth / 2),
            "DRHO": 0.02,
        }
    )


# ---------------------------------------------------------------- environment
def test_onshore_vs_offshore_temperature():
    on = environment_from_config({"surface_temp": 25.0, "temp_gradient_c_per_km": 30.0})
    off = environment_from_config(
        {"water_depth": 1000.0, "kb_elev": 25.0, "temp_gradient_c_per_km": 30.0}
    )
    assert np.isclose(on.temperature_c(2000.0), 25.0 + 60.0)
    # Offshore: 4 degC at seabed (1025 m), gradient only below the mudline.
    assert np.isclose(off.temperature_c(1025.0), 4.0)
    assert np.isclose(off.temperature_c(2025.0), 34.0)
    assert off.temperature_c(2025.0) < on.temperature_c(2025.0)
    # Formation pressure offshore uses seawater above the mudline.
    assert off.formation_pressure_psi(2025.0) < on.formation_pressure_psi(2025.0)


def test_bht_calibration_and_assumptions():
    env = environment_from_config(
        {"surface_temp": 20.0, "bht": 110.0, "bht_unit": "degC", "bht_depth": 3000.0}
    )
    assert np.isclose(env.gradient_c_per_km, 30.0, atol=0.01)
    assert any("BHT" in a for a in env.assumptions)
    env2 = environment_from_config({})
    assert any("gradient" in a for a in env2.assumptions)
    assert any("Mud weight" in a for a in env2.assumptions)


def test_arps_and_mud_program_obm():
    assert arps_resistivity(1.0, 25.0, 100.0) < 1.0
    env = Environment(
        mud_program=[
            MudInterval(
                top=0,
                bottom=1500,
                bit_size_in=12.25,
                mud_type="WBM",
                rmf=0.5,
                rmf_temp_c=25.0,
            ),
            MudInterval(
                top=1500,
                bottom=4000,
                bit_size_in=8.5,
                mud_type="OBM",
                mud_weight_ppg=12.0,
            ),
        ]
    )
    d = np.array([1000.0, 2000.0])
    m = env.mud_arrays(d, env.temperature_c(d))
    assert list(m["bit_size_in"]) == [12.25, 8.5]
    assert list(m["is_oil"]) == [False, True]
    assert m["rmf"][0] < 0.5 and np.isnan(m["rmf"][1])
    assert m["salinity_ppm"][1] == 0.0
    with pytest.raises(ValueError, match="mud_type"):
        MudInterval.from_dict({"top": 0, "bottom": 1, "mud_type": "XYZ"})


# ---------------------------------------------------------------- resistivity
def test_invasion_inversion_recovers_step_profile():
    rt_true, rxo_true, di = 40.0, 4.0, 30.0
    radii = {"shallow": 10.0, "medium": 30.0, "deep": 80.0}

    def app(r):
        j = 1 - np.exp(-((di / r) ** 2))
        return 1.0 / ((1 - j) / rt_true + j / rxo_true)

    inv = invert_invasion_profile(
        np.array([app(radii["deep"])]),
        rm=np.array([app(radii["medium"])]),
        rxo=np.array([app(radii["shallow"])]),
    )
    assert inv["state"][0] == 1
    assert np.isclose(inv["rt"][0], rt_true, rtol=0.1)
    assert np.isclose(inv["rxo"][0], rxo_true, rtol=0.1)
    assert 20 <= inv["di_in"][0] <= 45


def test_borehole_correction_skips_obm():
    r = np.array([10.0, 10.0])
    out = resistivity_borehole_correction(
        r,
        np.array([12.0, 12.0]),
        np.array([0.05, np.nan]),
        0.015,
        np.array([False, True]),
    )
    assert out[0] > 10.0  # conductive mud lowered the reading
    assert out[1] == 10.0  # OBM: unchanged


def test_invasion_profile_classification():
    # Fresh mud (Rmf/Rw = 10): water zone shows Rxo/Rt ~ 10, HC zone much lower.
    res = classify_invasion_profile(
        rt=np.array([1.0, 20.0, 1.0, 1.0]),
        rxo=np.array([10.0, 60.0, 80.0, 1.0]),
        rmf=1.0,
        rw=0.1,
    )
    assert list(res["code"]) == [1, 2, 3, 4]
    # Rmf ~ Rw: uninformative
    assert (
        classify_invasion_profile(np.array([1.0]), np.array([3.0]), 0.1, 0.1)["code"][0]
        == 0
    )


# ---------------------------------------------------------------- flags
def test_signal_flags():
    v = np.full(50, 60.0) + np.random.default_rng(0).normal(0, 0.5, 50)
    v[20] = 160.0
    assert detect_spikes(v, tol=15.0)[20] and detect_spikes(v, tol=15.0).sum() == 1
    f = np.arange(50, dtype=float)
    f[10:30] = 5.0
    flat = detect_flatline(f)
    assert flat[10:30].all() and not flat[:9].any() and not flat[31:].any()
    dt = np.full(40, 80.0)
    dt[15] = 130.0
    assert detect_sonic_cycle_skip(dt)[15]


# ---------------------------------------------------------------- pipeline
def test_no_caliper_reduces_confidence():
    df = _well()
    with_cali = apply_environmental_corrections(df, depth_unit="m")
    no_cali = apply_environmental_corrections(
        df.drop(columns=["CALI", "DRHO"]), depth_unit="m"
    )
    for log in ("GR", "NPHI", "RHOB"):
        assert no_cali[f"{log}_QC_CONF"].mean() < with_cali[f"{log}_QC_CONF"].mean()
    assert (no_cali["ENV_QC_REASON"] & ENV_QC_REASON_BITS["NO_CALIPER"]).all()
    assert any("Caliper" in a for a in no_cali.attrs["env_assumptions"])


def test_sigma_columns_wl_vs_lwd_and_obm():
    df = _well(cali=np.r_[np.full(30, 8.5), np.full(30, 12.5)])
    wl = apply_environmental_corrections(df, depth_unit="m", tool_type="WL")
    lwd = apply_environmental_corrections(df, depth_unit="m", tool_type="LWD")
    for out in (wl, lwd):
        for log in ("GR", "NPHI", "RHOB"):
            assert f"{log}_SIGMA" in out.columns
            assert (out[f"{log}_SIGMA"] > 0).all()
        assert out["HOLE_SEVERITY"].iloc[-1] > out["HOLE_SEVERITY"].iloc[0]
    assert lwd.attrs["env_tool_profile"] == "LWD"
    # Washout reduces density confidence
    assert wl["RHOB_QC_CONF"].iloc[-1] < wl["RHOB_QC_CONF"].iloc[0]
    with pytest.raises(ValueError, match="tool_type"):
        apply_environmental_corrections(df, tool_type="XYZ")


def test_environment_object_and_resistivity_outputs():
    df = _well()
    df["RT"] = 30.0
    df["RM"] = 12.0
    df["RXO"] = 3.0
    env = environment_from_config(
        {
            "water_depth": 500.0,
            "mud_type": "WBM",
            "rmf": 0.3,
            "rmf_temp": 24.0,
            "mud_density": 1.2,
        }
    )
    out = apply_environmental_corrections(
        df, depth_unit="m", environment=env, formation_salinity_ppm=60000
    )
    for col in (
        "RT_ENV_CORR",
        "RXO_EST",
        "DI_EST",
        "INVASION_FLAG",
        "INVASION_PROFILE",
        "RT_SIGMA",
        "ENV_TEMP_C",
        "ENV_RM",
        "ENV_RMF",
        "ENV_RW",
    ):
        assert col in out.columns
    assert (out["RT_ENV_CORR"] > 30.0).all()
    assert (out["INVASION_FLAG"] == 1).all()
    # Offshore: seabed 4 degC + 30 degC/km * ~1.5 km
    assert 40 < out["ENV_TEMP_C"].iloc[0] < 55
    # WBM: ENV_RM > ENV_RMF > ENV_RW (60000 ppm saline formation water vs 0.3 ohm.m @ 24C Rmf)
    assert (
        out["ENV_RM"].notna().all()
        and out["ENV_RMF"].notna().all()
        and out["ENV_RW"].notna().all()
    )
    assert (out["ENV_RM"] > out["ENV_RMF"]).all()
    assert (out["ENV_RMF"] > out["ENV_RW"]).all()

    obm = apply_environmental_corrections(
        df, depth_unit="m", mud_type="OBM", res_tool="laterolog"
    )
    assert (obm["INVASION_PROFILE"] == 0).all()
    assert (
        obm["ENV_RM"].isna().all()
        and obm["ENV_RMF"].isna().all()
        and obm["ENV_RW"].notna().all()
    )
    assert (obm["ENV_QC_REASON"] & ENV_QC_REASON_BITS["RES_MUD_MISMATCH"]).all()
    assert obm["RT_QC_CONF"].mean() < out["RT_QC_CONF"].mean()


def test_confidence_weighted_curve():
    v = np.full(50, 2.40)
    v[20:25] = 2.00  # washout-affected
    conf = np.ones(50)
    conf[20:25] = 0.1
    out = confidence_weighted_curve(v, conf, depth=np.arange(50.0))
    assert np.allclose(out[20:25], 2.40)
    assert np.allclose(out[:20], 2.40)
    mid = confidence_weighted_curve(
        v, np.where(conf < 1, 0.5, 1.0), depth=np.arange(50.0)
    )
    assert 2.0 < mid[22] < 2.4
