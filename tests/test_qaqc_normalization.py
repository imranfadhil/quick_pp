"""Unit tests for global and stratigraphic (zone-by-zone) multi-well log normalization
and borehole environmental/invasion corrections (`quick_pp.qaqc`).
"""

import numpy as np
import pandas as pd
import pytest

from quick_pp.qaqc import (
    apply_environmental_corrections,
    badhole_flagging,
    density_mudcake_correction,
    gr_borehole_correction,
    log_normalization,
    normalize_by_zone,
    nphi_environmental_correction,
    resistivity_invasion_correction,
)


def _make_multi_well_stratigraphic_df(n_per_zone: int = 200) -> pd.DataFrame:
    """Create a synthetic 3-well, 2-zone dataset with distinct formation lithologies
    and systematic tool calibration drift (shift + stretch) on target wells.
    """
    rng = np.random.default_rng(42)

    # True geological distributions in key well ("WELL-KEY"):
    # - ZONE_SAND: clean sandstone reservoir (low GR ~ 20-80 GAPI, higher NPHI ~ 0.12-0.28)
    # - ZONE_SHALE: marine shale caprock (high GR ~ 90-150 GAPI, higher NPHI ~ 0.25-0.42)
    gr_sand_base = np.linspace(20.0, 80.0, n_per_zone)
    nphi_sand_base = np.linspace(0.12, 0.28, n_per_zone)

    gr_shale_base = np.linspace(90.0, 150.0, n_per_zone)
    nphi_shale_base = np.linspace(0.25, 0.42, n_per_zone)

    # Permute deterministically so depth order isn't purely monotonic
    idx_sand = rng.permutation(n_per_zone)
    idx_shale = rng.permutation(n_per_zone)

    gr_sand_key = gr_sand_base[idx_sand]
    nphi_sand_key = nphi_sand_base[idx_sand]
    gr_shale_key = gr_shale_base[idx_shale]
    nphi_shale_key = nphi_shale_base[idx_shale]

    # WELL-DRIFT-1: tool calibration error in ZONE_SAND (+30 GAPI shift, 1.5x gain stretch)
    # and different drift in ZONE_SHALE (-15 GAPI shift, 0.8x gain compression)
    gr_sand_w2 = 30.0 + 1.5 * gr_sand_key
    nphi_sand_w2 = 0.05 + 1.2 * nphi_sand_key
    gr_shale_w2 = -15.0 + 0.8 * gr_shale_key
    nphi_shale_w2 = -0.03 + 0.9 * nphi_shale_key

    # WELL-DRIFT-2: negative shift and compression
    gr_sand_w3 = -10.0 + 0.75 * gr_sand_key
    nphi_sand_w3 = -0.02 + 0.85 * nphi_sand_key
    gr_shale_w3 = 25.0 + 1.25 * gr_shale_key
    nphi_shale_w3 = 0.04 + 1.15 * nphi_shale_key

    depths = np.arange(n_per_zone * 2, dtype=float) * 0.5 + 1000.0
    zones = ["ZONE_SAND"] * n_per_zone + ["ZONE_SHALE"] * n_per_zone

    df_key = pd.DataFrame(
        {
            "WELL_NAME": "WELL-KEY",
            "DEPTH": depths,
            "ZONES": zones,
            "GR": np.concatenate([gr_sand_key, gr_shale_key]),
            "NPHI": np.concatenate([nphi_sand_key, nphi_shale_key]),
        }
    )
    df_w2 = pd.DataFrame(
        {
            "WELL_NAME": "WELL-DRIFT-1",
            "DEPTH": depths,
            "ZONES": zones,
            "GR": np.concatenate([gr_sand_w2, gr_shale_w2]),
            "NPHI": np.concatenate([nphi_sand_w2, nphi_shale_w2]),
        }
    )
    df_w3 = pd.DataFrame(
        {
            "WELL_NAME": "WELL-DRIFT-2",
            "DEPTH": depths,
            "ZONES": zones,
            "GR": np.concatenate([gr_sand_w3, gr_shale_w3]),
            "NPHI": np.concatenate([nphi_sand_w3, nphi_shale_w3]),
        }
    )

    return pd.concat([df_key, df_w2, df_w3], ignore_index=True)


def test_normalize_by_zone_with_key_well_preserves_stratigraphic_contrast():
    """Verify normalize_by_zone aligns target wells to key_well per zone without smearing lithology across zones."""
    df = _make_multi_well_stratigraphic_df(n_per_zone=200)
    quantiles = (0.05, 0.95)

    res = normalize_by_zone(
        df, key_well="WELL-KEY", zone_col="ZONES", quantiles=quantiles
    )

    assert "GR_NORM" in res.columns
    assert "NPHI_NORM" in res.columns
    # Original columns remain untouched when inplace=False
    np.testing.assert_allclose(res["GR"], df["GR"])
    np.testing.assert_allclose(res["NPHI"], df["NPHI"])

    # Key well's normalized curves should equal its original curves
    key_mask = res["WELL_NAME"] == "WELL-KEY"
    np.testing.assert_allclose(
        res.loc[key_mask, "GR_NORM"], res.loc[key_mask, "GR"], rtol=1e-6
    )
    np.testing.assert_allclose(
        res.loc[key_mask, "NPHI_NORM"], res.loc[key_mask, "NPHI"], rtol=1e-6
    )

    # Because target wells are exact affine transforms of WELL-KEY within each zone,
    # quantile normalization recovers the exact key well curves in every zone!
    for well in ("WELL-DRIFT-1", "WELL-DRIFT-2"):
        for zone in ("ZONE_SAND", "ZONE_SHALE"):
            w_mask = (res["WELL_NAME"] == well) & (res["ZONES"] == zone)
            k_mask = (res["WELL_NAME"] == "WELL-KEY") & (res["ZONES"] == zone)

            np.testing.assert_allclose(
                res.loc[w_mask, "GR_NORM"].to_numpy(),
                res.loc[k_mask, "GR"].to_numpy(),
                rtol=1e-5,
                atol=1e-5,
            )
            np.testing.assert_allclose(
                res.loc[w_mask, "NPHI_NORM"].to_numpy(),
                res.loc[k_mask, "NPHI"].to_numpy(),
                rtol=1e-5,
                atol=1e-5,
            )

    # Verify lithological contrast between ZONE_SAND and ZONE_SHALE is preserved (no smearing)
    sand_mean_gr = res.loc[res["ZONES"] == "ZONE_SAND", "GR_NORM"].mean()
    shale_mean_gr = res.loc[res["ZONES"] == "ZONE_SHALE", "GR_NORM"].mean()
    assert np.isclose(sand_mean_gr, 50.0, atol=1.0)
    assert np.isclose(shale_mean_gr, 120.0, atol=1.0)
    assert shale_mean_gr - sand_mean_gr > 60.0


def test_normalize_by_zone_field_median_and_missing_key_well_fallback():
    """Verify field-median reference mode when key_well is None or missing in a specific zone."""
    df = _make_multi_well_stratigraphic_df(n_per_zone=100)
    quantiles = (0.05, 0.95)

    # 1. key_well=None uses multi-well median of (P05, P95) per zone
    res_median = normalize_by_zone(
        df, key_well=None, zone_col="ZONES", quantiles=quantiles
    )
    for zone in ("ZONE_SAND", "ZONE_SHALE"):
        z_df = res_median[res_median["ZONES"] == zone]
        p05_per_well = [
            np.nanquantile(w_df["GR_NORM"], 0.05)
            for _, w_df in z_df.groupby("WELL_NAME")
        ]
        p95_per_well = [
            np.nanquantile(w_df["GR_NORM"], 0.95)
            for _, w_df in z_df.groupby("WELL_NAME")
        ]
        # All wells in the zone now share identical P05 and P95 equal to the field median
        np.testing.assert_allclose(p05_per_well, p05_per_well[0], rtol=1e-5)
        np.testing.assert_allclose(p95_per_well, p95_per_well[0], rtol=1e-5)

    # 2. Remove WELL-KEY from ZONE_SHALE: ZONE_SAND uses WELL-KEY, ZONE_SHALE falls back to field median
    df_partial = df[
        ~((df["WELL_NAME"] == "WELL-KEY") & (df["ZONES"] == "ZONE_SHALE"))
    ].copy()
    res_partial = normalize_by_zone(
        df_partial, key_well="WELL-KEY", zone_col="ZONES", quantiles=quantiles
    )
    shale_wells = res_partial[res_partial["ZONES"] == "ZONE_SHALE"]
    p05_shale = [
        np.nanquantile(w_df["GR_NORM"], 0.05)
        for _, w_df in shale_wells.groupby("WELL_NAME")
    ]
    np.testing.assert_allclose(p05_shale[0], p05_shale[1], rtol=1e-5)


def test_normalize_by_zone_50_wells_histogram_matching_and_report():
    """Verify multi-well histogram/CDF normalization across 50+ vintage wells and drift report generation."""
    rng = np.random.default_rng(101)
    n_wells = 55
    n_pts = 80

    base_gr_sand = np.linspace(25.0, 75.0, n_pts)
    base_gr_shale = np.linspace(95.0, 145.0, n_pts)

    frames = []
    for i in range(n_wells):
        well_name = "WELL-00" if i == 0 else f"WELL-{i:02d}"
        if i == 0:
            gr_sand = base_gr_sand
            gr_shale = base_gr_shale
        else:
            # Non-linear + affine vintage tool drift across 55 wells
            shift = rng.uniform(-25.0, 35.0)
            scale = rng.uniform(0.7, 1.4)
            gr_sand = shift + scale * (base_gr_sand ** 1.05)
            gr_shale = (shift - 10.0) + scale * (base_gr_shale ** 1.03)

        frames.append(
            pd.DataFrame(
                {
                    "WELL_NAME": well_name,
                    "DEPTH": np.arange(n_pts * 2, dtype=float),
                    "ZONES": ["ZONE_A"] * n_pts + ["ZONE_B"] * n_pts,
                    "GR": np.concatenate([gr_sand, gr_shale]),
                }
            )
        )

    field_df = pd.concat(frames, ignore_index=True)

    norm_df, report_df = normalize_by_zone(
        field_df,
        key_well="WELL-00",
        zone_col="ZONES",
        logs=["GR"],
        method="histogram",
        return_report=True,
    )

    assert len(report_df) == n_wells * 2
    assert set(
        [
            "WELL_NAME",
            "ZONE",
            "LOG",
            "METHOD",
            "REFERENCE",
            "P_LOW_ORIG",
            "P_HIGH_ORIG",
            "SCALE_FACTOR",
            "DRIFT_SHIFT",
        ]
    ).issubset(report_df.columns)

    # Every single one of the 55 wells should match WELL-00's distribution in both zones
    key_sand = norm_df[
        (norm_df["WELL_NAME"] == "WELL-00") & (norm_df["ZONES"] == "ZONE_A")
    ]["GR"].to_numpy()
    w50_sand = norm_df[
        (norm_df["WELL_NAME"] == "WELL-50") & (norm_df["ZONES"] == "ZONE_A")
    ]["GR_NORM"].to_numpy()
    np.testing.assert_allclose(w50_sand, key_sand, rtol=1e-2, atol=0.5)


def test_normalize_by_zone_inplace_nans_and_validation():
    """Verify inplace=True, NaN handling, constant curve fallback, and input validation."""
    df = _make_multi_well_stratigraphic_df(n_per_zone=50)
    # Inject NaNs
    df.loc[[0, 10, 105, 250], "GR"] = np.nan

    res_inplace = normalize_by_zone(
        df, key_well="WELL-KEY", zone_col="ZONES", quantiles=(0.10, 0.90), inplace=True
    )
    assert res_inplace.loc[[0, 10, 105, 250], "GR"].isna().all()
    np.testing.assert_allclose(
        res_inplace["GR"].dropna(), res_inplace["GR_NORM"].dropna()
    )

    # Validation errors
    with pytest.raises(ValueError, match="quantiles must satisfy"):
        normalize_by_zone(df, "WELL-KEY", "ZONES", quantiles=(0.95, 0.05))

    with pytest.raises(ValueError, match="Zone column"):
        normalize_by_zone(df, "WELL-KEY", zone_col="MISSING_ZONE")

    with pytest.raises(ValueError, match="Well column"):
        normalize_by_zone(df, "WELL-KEY", well_col="MISSING_WELL")

    with pytest.raises(ValueError, match="None of the specified logs"):
        normalize_by_zone(df, "WELL-KEY", logs=["NON_EXISTENT_LOG"])


def test_global_log_normalization():
    """Verify global log_normalization() for minmax, quantile, and zscore methods."""
    df = pd.DataFrame(
        {
            "GR": [20.0, 45.0, 70.0, 95.0, 120.0],
            "NPHI": [0.10, 0.15, 0.20, 0.25, 0.30],
        }
    )

    # Min-max scaling -> [0, 1]
    res_mm = log_normalization(df, logs=["GR", "NPHI"], method="minmax")
    np.testing.assert_allclose(res_mm["GR_NORM"], [0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_allclose(res_mm["NPHI_NORM"], [0.0, 0.25, 0.5, 0.75, 1.0])

    # Z-score scaling -> mean 0, std 1
    res_zs = log_normalization(df, logs="GR", method="zscore")
    assert np.isclose(res_zs["GR_NORM"].mean(), 0.0, atol=1e-7)
    assert np.isclose(res_zs["GR_NORM"].std(), 1.0, atol=1e-7)

    # Quantile scaling
    res_q = log_normalization(
        df, logs=["GR"], method="quantile", quantiles=(0.25, 0.75)
    )
    assert np.isclose(res_q.loc[1, "GR_NORM"], 0.0)
    assert np.isclose(res_q.loc[3, "GR_NORM"], 1.0)

    with pytest.raises(ValueError, match="Unsupported normalization method"):
        log_normalization(df, method="invalid_method")


def test_environmental_and_invasion_corrections():
    """Verify borehole GR, environmental NPHI, mudcake RHOB, and tornado-chart RT invasion corrections."""
    # 1. GR Borehole correction: washout (12.5" vs 8.5" bit) and heavy mud (14 ppg) attenuate GR -> corrected GR > raw GR
    gr_nom = gr_borehole_correction(80.0, cali=8.5, bitsize=8.5, mud_weight=9.0)
    gr_wash = gr_borehole_correction(80.0, cali=12.5, bitsize=8.5, mud_weight=14.0)
    assert np.isclose(gr_nom, 80.0, rtol=1e-4)
    assert gr_wash > 80.0

    # 2. NPHI environmental correction: enlarged borehole adds mud hydrogen -> corrected NPHI < raw NPHI
    nphi_nom = nphi_environmental_correction(
        0.22, cali=8.5, bitsize=8.5, temp=75.0, pressure=0.0, mud_weight=9.0
    )
    nphi_wash = nphi_environmental_correction(
        0.22, cali=12.5, bitsize=8.5, temp=75.0, pressure=0.0, mud_weight=9.0
    )
    assert np.isclose(nphi_nom, 0.22, atol=1e-5)
    assert nphi_wash < 0.22

    # 3. Density mudcake/washout correction: washout reads light mudcake -> corrected RHOB > raw RHOB
    rhob_nom = density_mudcake_correction(2.35, cali=8.5, bitsize=8.5, rho_mudcake=1.5)
    rhob_wash = density_mudcake_correction(
        2.35, cali=11.5, bitsize=8.5, rho_mudcake=1.5, drho=0.25
    )
    assert np.isclose(rhob_nom, 2.35, atol=1e-5)
    assert rhob_wash > 2.35

    # 4. Resistivity invasion correction:
    # Row 0: Conductive invasion in hydrocarbon sand (Rt=50 > Rm=20 > Rxo=5) -> Rt_corr > 50, flag = 1
    # Row 1: Resistive invasion in water sand (Rt=2 < Rm=6 < Rxo=15) -> Rt_corr < 2, flag = 2
    # Row 2: Uninvaded shale (Rt=5, Rm=5, Rxo=5) -> Rt_corr == 5, flag = 0
    rt = np.array([50.0, 2.0, 5.0])
    rm = np.array([20.0, 6.0, 5.0])
    rxo = np.array([5.0, 15.0, 5.0])
    rt_corr, inv_flag = resistivity_invasion_correction(rt, rm=rm, rxo=rxo)
    assert rt_corr[0] > 50.0 and inv_flag[0] == 1
    assert rt_corr[1] < 2.0 and inv_flag[1] == 2
    assert np.isclose(rt_corr[2], 5.0) and inv_flag[2] == 0

    # 5. End-to-end DataFrame pipeline via apply_environmental_corrections
    df = pd.DataFrame(
        {
            "WELL_NAME": ["W1", "W1", "W1"],
            "DEPTH": [2000.0, 2001.0, 2002.0],
            "BS": [8.5, 8.5, 8.5],
            "CALI": [8.5, 12.0, 10.5],
            "GR": [60.0, 60.0, 90.0],
            "NPHI": [0.20, 0.25, 0.30],
            "RHOB": [2.35, 2.20, 2.45],
            "DRHO": [0.02, 0.22, 0.05],
            "RT": [45.0, 2.5, 8.0],
            "RM": [20.0, 6.0, 8.0],
            "RXO": [6.0, 14.0, 8.0],
        }
    )
    corr_df = apply_environmental_corrections(df, mud_weight=11.5, inplace=False)
    for col in ("GR_CORR", "NPHI_CORR", "RHOB_CORR", "RT_CORR", "INVASION_FLAG"):
        assert col in corr_df.columns
    assert corr_df.loc[1, "GR_CORR"] > corr_df.loc[1, "GR"]
    assert corr_df.loc[1, "RHOB_CORR"] > corr_df.loc[1, "RHOB"]
    assert corr_df.loc[0, "RT_CORR"] > corr_df.loc[0, "RT"]
    assert corr_df.loc[0, "INVASION_FLAG"] == 1
    assert corr_df.loc[1, "INVASION_FLAG"] == 2


def test_badhole_flagging_multi_criteria_and_coal_protection():
    """Verify all 4 badhole flagging criteria (CALI-BS, DRHO, RHOB/NPHI bounds, rugosity) and coal protection."""
    n = 30
    df = pd.DataFrame(
        {
            "DEPTH": np.linspace(1000.0, 1014.5, n),
            "CALI": np.full(n, 8.5),
            "DRHO": np.full(n, 0.02),
            "RHOB": np.full(n, 2.35),
            "NPHI": np.full(n, 0.20),
            "COAL_FLAG": np.zeros(n, dtype=int),
        }
    )
    # Row 5: Differential caliper washout (|12.0 - 8.5| = 3.5 >= 2.5)
    df.loc[5, "CALI"] = 12.0
    # Row 10: High DRHO pad standoff (0.24 > 0.15)
    df.loc[10, "DRHO"] = 0.24
    # Row 15: Low RHOB pad standoff (1.70 < 1.95), not coal
    df.loc[15, "RHOB"] = 1.70
    # Row 18: Coal bed with low RHOB (1.45) and high NPHI (0.50) -> protected when exclude_coal=True
    df.loc[18, "RHOB"] = 1.45
    df.loc[18, "NPHI"] = 0.50
    df.loc[18, "COAL_FLAG"] = 1
    # Row 24: Local rugosity spike on RHOB drop (2.05 vs 2.35 rolling median -> >8% drop)
    df.loc[24, "RHOB"] = 2.05

    res = badhole_flagging(
        df,
        thold=2.5,
        use_cali=True,
        auto_bitsize=False,
        bitsize_default=8.5,
        use_drho=True,
        drho_threshold=0.15,
        use_rhob_nphi=True,
        rhob_min=1.95,
        rhob_max=2.95,
        nphi_max=0.45,
        exclude_coal=True,
        use_rugosity=True,
        rugosity_window=9,
        rhob_trend_factor=0.08,
        cali_spike_threshold=1.0,
    )

    assert "BADHOLE" in res.columns
    assert "BS" in res.columns
    assert "DCALI" in res.columns
    assert res.loc[5, "BADHOLE"] == 1
    assert np.isclose(res.loc[5, "DCALI"], 3.5)
    assert res.loc[10, "BADHOLE"] == 1
    assert res.loc[15, "BADHOLE"] == 1
    # Row 18 is coal, so exclude_coal=True protects it from RHOB/NPHI and rugosity badhole flagging
    assert res.loc[18, "BADHOLE"] == 0
    assert res.loc[24, "BADHOLE"] == 1
    assert res.loc[0, "BADHOLE"] == 0

    summary = res.attrs.get("badhole_summary")
    assert summary is not None
    assert summary["flagged_count"] == 4
    assert summary["cali_flagged"] == 1
    assert summary["drho_flagged"] == 1
    assert summary["rhob_nphi_flagged"] == 1
    assert summary["rugosity_flagged"] >= 1

