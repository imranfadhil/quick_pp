"""Tests for raw inter-log depth alignment QC and bulk/dynamic shift conditioning."""

import numpy as np
import pandas as pd

from quick_pp.borehole_corrections import (
    ENV_QC_REASON_BITS,
    apply_environmental_corrections,
)
from quick_pp.depth_alignment import apply_depth_alignment, evaluate_depth_alignment


def _synthetic_multibed_well(
    z_start: float = 2000.0, z_stop: float = 2080.0, dz: float = 0.15
) -> pd.DataFrame:
    z = np.arange(z_start, z_stop, dz)
    beds = (
        (np.sin(2.0 * np.pi * (z - z_start) / 9.0) > 0.0).astype(float)
        + 0.6 * (np.sin(2.0 * np.pi * (z - z_start) / 3.7) > 0.25).astype(float)
        + 0.4 * (np.cos(2.0 * np.pi * (z - z_start) / 5.3) < -0.3).astype(float)
    )
    gr = 25.0 + 55.0 * beds
    rhob = 2.22 + 0.18 * beds
    nphi = 0.28 - 0.08 * beds
    rt = 40.0 * (10.0 ** (-0.55 * beds))
    dtc = 95.0 - 18.0 * beds
    return pd.DataFrame(
        {
            "DEPTH": z,
            "GR": gr,
            "RHOB": rhob,
            "NPHI": nphi,
            "RT": rt,
            "RXO": rt * 0.45,
            "DTC": dtc,
            "CALI": np.full_like(z, 8.5),
            "BS": np.full_like(z, 8.5),
            "DRHO": np.full_like(z, 0.01),
        }
    )


def test_aligned_well_reports_zero_shift():
    df = _synthetic_multibed_well()
    rep = evaluate_depth_alignment(df, ref_curve="GR", depth_unit="m")
    assert rep["summary"]["off_depth_count"] == 0
    for r in rep["curves"]:
        assert r["status"] in ("ALIGNED", "LOW_SIGNAL")
        assert abs(r["suggested_shift_m"]) < 0.10


def test_bulk_shift_detection_and_correction_with_nan_gaps():
    df = _synthetic_multibed_well()
    z = df["DEPTH"].to_numpy()
    # Record DTC 0.60 m too shallow -> requires +0.60 m downward shift
    dtc_true = df["DTC"].to_numpy().copy()
    df["DTC"] = np.interp(z + 0.60, z, dtc_true)
    # Introduce NaN gap in the middle to verify gap-guard masking doesn't corrupt lag
    df.loc[120:135, "DTC"] = np.nan

    rep = evaluate_depth_alignment(df, ref_curve="GR", depth_unit="m")
    dtc_diag = next(r for r in rep["curves"] if r["curve"] == "DTC")
    assert dtc_diag["status"] == "OFF_DEPTH"
    assert dtc_diag["confidence"] >= 0.5
    assert np.isclose(dtc_diag["suggested_shift_m"], 0.60, atol=0.10)

    shifted_df, applied = apply_depth_alignment(
        df, rep, curves_to_shift=["DTC"], mode="bulk"
    )
    assert "DTC" in applied
    post = evaluate_depth_alignment(shifted_df, ref_curve="GR", depth_unit="m")
    dtc_post = next(r for r in post["curves"] if r["curve"] == "DTC")
    assert dtc_post["status"] == "ALIGNED"
    assert abs(dtc_post["suggested_shift_m"]) < 0.10


def test_dynamic_cable_stretch_alignment_preserves_monotonicity():
    df = _synthetic_multibed_well(z_start=2000.0, z_stop=2150.0, dz=0.15)
    z = df["DEPTH"].to_numpy()
    rhob_true = df["RHOB"].to_numpy().copy()
    # Linearly increasing cable stretch from 0.0 m at top to +0.90 m at bottom
    stretch = 0.90 * (z - z[0]) / (z[-1] - z[0])
    df["RHOB"] = np.interp(z + stretch, z, rhob_true)

    rep = evaluate_depth_alignment(
        df, ref_curve="GR", window_size=35.0, window_step=10.0, depth_unit="m"
    )
    rhob_diag = next(r for r in rep["curves"] if r["curve"] == "RHOB")
    assert rhob_diag["status"] in ("OFF_DEPTH", "MINOR")
    assert rhob_diag["variable_drift"] is True

    dyn_df, applied = apply_depth_alignment(
        df, rep, curves_to_shift=["RHOB"], mode="dynamic"
    )
    assert applied["RHOB"]["mode"] == "dynamic"
    post = evaluate_depth_alignment(
        dyn_df, ref_curve="GR", window_size=35.0, window_step=10.0, depth_unit="m"
    )
    rhob_post = next(r for r in post["curves"] if r["curve"] == "RHOB")
    assert rhob_post["p95_abs_drift_m"] < rhob_diag["p95_abs_drift_m"] * 0.5


def test_env_corrections_flags_off_depth_curve_and_widens_boundary_sigma():
    df = _synthetic_multibed_well()
    z = df["DEPTH"].to_numpy()
    aligned_out = apply_environmental_corrections(df, depth_unit="m")

    df_off = df.copy()
    df_off["NPHI"] = np.interp(z - 0.75, z, df["NPHI"].to_numpy())
    off_out = apply_environmental_corrections(df_off, depth_unit="m")

    assert any(
        "Off-depth log detected: NPHI" in a for a in off_out.attrs["env_assumptions"]
    )
    assert (off_out["ENV_QC_REASON"] & ENV_QC_REASON_BITS["DEPTH_MISMATCH"]).any()
    assert off_out["NPHI_SIGMA"].max() > aligned_out["NPHI_SIGMA"].max()
