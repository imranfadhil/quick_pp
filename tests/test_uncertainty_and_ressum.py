"""Unit tests for PRMS/SEC reservoir summary metrics, volumetric uncertainty,
and Latin Hypercube log-level petrophysical uncertainty propagation (`quick_pp.ressum` and `quick_pp.uncertainty`).
"""

import numpy as np
import pandas as pd

from quick_pp.plotter.well_log import plotly_log
from quick_pp.ressum import (
    calc_reservoir_summary,
    mc_volumetric_method,
    mc_volumetric_uncertainty as ressum_mc_unc,
)
from quick_pp.uncertainty import (
    calc_stochastic_reservoir_summary,
    mc_volumetric_uncertainty,
    run_log_uncertainty,
)


def _make_synthetic_well_df(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    depth = np.linspace(2000.0, 2000.0 + 0.5 * (n - 1), n)
    phit = rng.uniform(0.05, 0.30, size=n)
    vclay = rng.uniform(0.05, 0.40, size=n) * (1.0 - phit)
    vsilt = rng.uniform(0.05, 0.30, size=n) * (1.0 - phit)
    vsand = np.maximum(0.0, 1.0 - phit - vclay - vsilt)

    rhob = (1.0 - phit) * 2.65 + phit * 1.0
    nphi = phit + 0.15 * vclay
    gr = 20.0 + 110.0 * vclay
    rt = np.exp(rng.uniform(0.5, 4.0, size=n))
    dtc = (1.0 - phit) * 55.0 + phit * 189.0
    swt = np.clip(rng.uniform(0.2, 1.0, size=n), 0.0, 1.0)
    log_perm = -1.0 + 12.0 * phit - 3.0 * vclay

    return pd.DataFrame(
        {
            "WELL_NAME": ["WELL-A"] * n,
            "DEPTH": depth,
            "GR": gr,
            "RT": rt,
            "NPHI": nphi,
            "RHOB": rhob,
            "DTC": dtc,
            "PHIT": phit,
            "SWT": swt,
            "LOG_PERM": log_perm,
            "VSAND": vsand,
            "VSILT": vsilt,
            "VCLAY": vclay,
        }
    )


def test_calc_reservoir_summary_pore_volume_weighting_and_perm_upscaling():
    """Verify PRMS/SEC pore-volume weighted Sw and kh/kv permeability upscaling proxies in calc_reservoir_summary."""
    # Two pay layers in Zone A with equal thickness (h = 1.0 m):
    # Layer 1 (high quality sand): phit = 0.30, swt = 0.20, perm = 1000.0 mD
    # Layer 2 (poor quality sand): phit = 0.10, swt = 0.80, perm = 10.0 mD
    # Arithmetic Sw = (0.20 + 0.80) / 2 = 0.50 (WRONG for reserves!)
    # Pore-Volume Weighted Sw = (0.30*0.20*1 + 0.10*0.80*1) / (0.30*1 + 0.10*1) = 0.14 / 0.40 = 0.35
    # Thickness-weighted phit = (0.30 + 0.10) / 2 = 0.20
    # BVHC = phit_avg * (1 - sw_pv) = 0.20 * (1 - 0.35) = 0.13
    # Perm Arith (k_h) = (1000 + 10) / 2 = 505.0 mD
    # Perm Geom (k_eff) = sqrt(1000 * 10) = 100.0 mD
    # Perm Harm (k_v) = 2 / (1/1000 + 1/10) = 19.802 mD
    depth = np.array([1000.0, 1001.0])
    vshale = np.array([0.10, 0.20])
    phit = np.array([0.30, 0.10])
    swt = np.array([0.20, 0.80])
    perm = np.array([1000.0, 10.0])
    zones = np.array(["ZONE_A", "ZONE_A"])

    cutoffs = {"VSHALE": 0.40, "PHIT": 0.05, "SWT": 0.90}
    summary = calc_reservoir_summary(depth, vshale, phit, swt, perm, zones, cutoffs=cutoffs)

    pay_row = summary[(summary["zones"] == "ZONE_A") & (summary["flag"] == "pay")].iloc[0]

    # Verify pore-volume weighted Sw (0.35), NOT arithmetic mean (0.50)
    np.testing.assert_allclose(pay_row["av_swt"], 0.35, atol=1e-3)
    np.testing.assert_allclose(pay_row["av_sw_pay"], 0.35, atol=1e-3)
    np.testing.assert_allclose(pay_row["av_phit"], 0.20, atol=1e-3)
    np.testing.assert_allclose(pay_row["av_phit_pay"], 0.20, atol=1e-3)

    # Volumetric identity: av_bvhc == av_phit * (1 - av_swt)
    np.testing.assert_allclose(
        pay_row["av_bvhc"],
        pay_row["av_phit"] * (1.0 - pay_row["av_swt"]),
        atol=1e-3,
    )

    # Verify permeability upscaling proxies (k_harm <= k_geom <= k_arith)
    np.testing.assert_allclose(pay_row["av_perm_arith_pay"], 505.0, atol=1e-3)
    np.testing.assert_allclose(pay_row["av_perm_geom_pay"], 100.0, atol=1e-3)
    np.testing.assert_allclose(pay_row["av_perm_harm_pay"], 19.802, atol=1e-3)
    assert pay_row["av_perm_harm_pay"] <= pay_row["av_perm_geom_pay"] <= pay_row["av_perm_arith_pay"]


def test_lhs_volumetric_and_log_uncertainty_module():
    """Verify Latin Hypercube volumetric uncertainty and 100-realization log-level uncertainty
    for both SandSiltClay and Carbonate + waxman_smits_saturation, CPI curve output
    (SW_P10, SW_P50, SW_P90), well_log.py integration, and stochastic reservoir summary.
    """
    # 1. Volumetric LHS uncertainty check (and re-export / delegation in ressum.py)
    assert ressum_mc_unc is mc_volumetric_uncertainty
    vol_res = mc_volumetric_uncertainty(
        area_bound=(500, 2500, 1560, 400),
        thickness_bound=(10, 216, 89, 74),
        porosity_bound=(0.066, 0.261, 0.177, 0.05),
        water_saturation_bound=(0.228, 0.90, 0.55, 0.15),
        volume_factor_bound=(1.2, 1.4),
        n_try=1000,
        seed=42,
        plot=False,
    )
    assert len(vol_res["volumes"]) == 1000
    assert (
        0.0
        < vol_res["percentiles"]["P10"]
        < vol_res["percentiles"]["P50"]
        < vol_res["percentiles"]["P90"]
    )

    volumes_tuple = mc_volumetric_method(
        area_bound=(500, 2500, 1560, 400),
        thickness_bound=(10, 216, 89, 74),
        porosity_bound=(0.066, 0.261, 0.177, 0.05),
        water_saturation_bound=(0.228, 0.90, 0.55, 0.15),
        volume_factor_bound=(1.2, 1.4),
        n_try=1000,
        seed=42,
        plot=False,
    )
    np.testing.assert_allclose(volumes_tuple[0], vol_res["volumes"])

    # 2. 100 realizations of SandSiltClay + waxman_smits_saturation
    well_df = _make_synthetic_well_df(40)
    well_df["ZONES"] = ["ZONE_A"] * 20 + ["ZONE_B"] * 20

    cpi_ssc, reals_ssc = run_log_uncertainty(
        well_df,
        model="ssc",
        saturation_method="waxman_smits",
        n_samples=100,
        seed=42,
        return_realizations=True,
    )
    assert reals_ssc["SW"].shape == (100, 40)
    for col in (
        "SW_P10",
        "SW_P50",
        "SW_P90",
        "SWT_P10",
        "SWT_P50",
        "SWT_P90",
        "PHIT_P10",
        "PHIT_P50",
        "PHIT_P90",
    ):
        assert col in cpi_ssc.columns, f"Missing CPI uncertainty curve: {col}"
        assert cpi_ssc[col].notna().all()

    assert np.all(cpi_ssc["SW_P10"] <= cpi_ssc["SW_P50"] + 1e-9)
    assert np.all(cpi_ssc["SW_P50"] <= cpi_ssc["SW_P90"] + 1e-9)
    assert np.mean(cpi_ssc["SW_P90"] - cpi_ssc["SW_P10"]) > 0.01

    # 3. 100 realizations of Carbonate + waxman_smits_saturation
    cpi_carb = run_log_uncertainty(
        well_df,
        model="carbonate",
        saturation_method="waxman_smits",
        n_samples=100,
        seed=42,
    )
    assert np.all(cpi_carb["SW_P10"] <= cpi_carb["SW_P50"] + 1e-9)
    assert np.all(cpi_carb["SW_P50"] <= cpi_carb["SW_P90"] + 1e-9)
    assert np.all(cpi_carb["PHIT_P10"] <= cpi_carb["PHIT_P50"] + 1e-9)
    assert np.all(cpi_carb["PHIT_P50"] <= cpi_carb["PHIT_P90"] + 1e-9)

    # 4. Verify well_log.py plots SW_P10, SW_P50, SW_P90 and SWT_P10, SWT_P50, SWT_P90
    fig = plotly_log(cpi_carb.drop(columns=["SWT"]), well_name="CARB-UNC")
    trace_names = {t.name for t in fig.data if t.name}
    for expected_trace in (
        "SW_P10",
        "SW_P50",
        "SW_P90",
        "SWT_P10",
        "SWT_P50",
        "SWT_P90",
        "PHIT_P10",
        "PHIT_P50",
        "PHIT_P90",
    ):
        assert expected_trace in trace_names, f"Missing {expected_trace} in plotly_log"

    # 5. Stochastic reservoir summary & FastAPI schema/service integration
    stoch_res = calc_stochastic_reservoir_summary(
        well_df,
        model="carbonate",
        saturation_method="waxman_smits",
        n_samples=50,
        seed=42,
    )
    summary_df = stoch_res["summary_df"]
    assert "av_swt_p10" in summary_df.columns
    assert "av_swt_p50" in summary_df.columns
    assert "av_swt_p90" in summary_df.columns
    assert "stoiip_p50_mmbbl" in summary_df.columns
    assert "ZONE_A" in stoch_res["volumetrics"]

