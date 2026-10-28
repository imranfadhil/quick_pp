"""Unit tests for SCAL / Core Corrections (`quick_pp.scal` & `quick_pp.core_analysis`)
and Multi-Well Contact & Transition Zone Optimization (`quick_pp.saturation`).
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from quick_pp import scal
from quick_pp.core_analysis import (
    auto_cuddy_params,
    auto_j_params,
    auto_skelt_harrison_params,
    correct_core_data,
    cuddy_func,
    dean_stark_cbw_correction,
    dean_stark_saturation_correction,
    estimate_cbw_from_qv,
    fit_cuddy_curve,
    fit_j_curve,
    fit_poroperm_curve,
    fit_skelt_harrison_curve,
    humidity_to_total_porosity,
    klinkenberg_permeability_correction,
    leverett_j,
    normalize_sw,
    overburden_permeability_correction,
    overburden_poroperm_correction,
    overburden_porosity_correction,
    pc_lab_to_reservoir,
    perm_transform,
    skelt_harrison_func,
    sw_shf_leverett_j,
    sw_skelt_harrison,
)
from quick_pp.saturation import (
    ContactOptimizer,
    analyze_transition_zone,
    archie_saturation,
    estimate_mdt_fwl,
    optimize_fwl,
    plot_fwl_reconciliation,
    predict_sw_shm,
)


def test_overburden_porosity_and_permeability_correction():
    """Verify exponential, power-law, and Kozeny-Carman overburden stress compaction on porosity and permeability."""
    poro_amb = 0.20
    perm_amb = 150.0
    ncs = 3000.0

    # Exponential porosity compaction: 0.20 * exp(-4e-6 * 3000)
    poro_exp = overburden_porosity_correction(poro_amb, ncs, compaction_coeff=4e-6)
    expected_exp = 0.20 * np.exp(-4e-6 * 3000.0)
    np.testing.assert_allclose(poro_exp, expected_exp, rtol=1e-6)
    assert poro_exp < poro_amb

    # Power-law porosity compaction: 0.20 * (3000 / 400)^(-0.03)
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

    # Exponential permeability compaction: 150 * exp(-3e-5 * 3000)
    perm_exp = overburden_permeability_correction(perm_amb, ncs, gamma=3e-5)
    np.testing.assert_allclose(perm_exp, 150.0 * np.exp(-3e-5 * 3000.0), rtol=1e-6)
    assert perm_exp < perm_amb

    # Power-law permeability compaction
    perm_pow = overburden_permeability_correction(
        perm_amb, ncs, gamma=0.08, ambient_stress=400.0, method="power_law"
    )
    np.testing.assert_allclose(
        perm_pow, 150.0 * ((3000.0 / 400.0) ** (-0.08)), rtol=1e-6
    )

    # Kozeny-Carman porosity-coupled permeability compaction
    perm_kc = overburden_permeability_correction(
        perm_amb,
        ncs,
        method="kozeny_carman",
        core_poro=poro_amb,
        poro_compaction_coeff=4e-6,
    )
    expected_kc = (
        perm_amb
        * ((poro_exp / poro_amb) ** 3.0)
        * (((1.0 - poro_amb) / (1.0 - poro_exp)) ** 2.0)
    )
    np.testing.assert_allclose(perm_kc, expected_kc, rtol=1e-6)

    # Joint overburden_poroperm_correction
    p_joint, k_joint = overburden_poroperm_correction(
        poro_amb,
        perm_amb,
        ncs,
        poro_compaction_coeff=4e-6,
        perm_compaction_coeff=3e-5,
    )
    np.testing.assert_allclose(p_joint, poro_exp, rtol=1e-6)
    np.testing.assert_allclose(k_joint, perm_exp, rtol=1e-6)
    assert scal.overburden_poroperm_correction is overburden_poroperm_correction


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

    # Sampath & Keighin (1982) porosity-dependent b-factor
    poro = np.array([0.10, 0.15, 0.22])
    k_sk = klinkenberg_permeability_correction(
        k_air, mean_pressure, method="sampath_keighin", poro=poro
    )
    expected_b_sk = 13.85 * ((k_air / poro) ** (-0.53))
    np.testing.assert_allclose(
        k_sk, k_air / (1.0 + expected_b_sk / mean_pressure), rtol=1e-6
    )

    # Explicit b-factor
    k_liq_explicit = klinkenberg_permeability_correction(10.0, 50.0, b_factor=5.0)
    np.testing.assert_allclose(k_liq_explicit, 10.0 / (1.0 + 5.0 / 50.0), rtol=1e-6)

    # Zero or negative k_air returns 0.0 safely
    assert klinkenberg_permeability_correction(0.0, 50.0) == 0.0


def test_humidity_and_dean_stark_cbw_corrections():
    """Verify reconciling humidity-dried vs. Dean-Stark/oven-dried core porosity and water saturation."""
    # Humidity-dried plug (PHIE = 0.15) + CBW (0.04) -> PHIT = 0.19
    phit = humidity_to_total_porosity(0.15, 0.04, oven_dried=False)
    np.testing.assert_allclose(phit, 0.19, rtol=1e-6)

    # Oven-dried plug (PHIT = 0.19) - CBW (0.04) -> PHIE = 0.15
    phie = humidity_to_total_porosity(0.19, 0.04, oven_dried=True)
    np.testing.assert_allclose(phie, 0.15, rtol=1e-6)

    # Clipped safely to [0.0, 1.0]
    assert humidity_to_total_porosity(0.02, 0.05, oven_dried=True) == 0.0
    assert humidity_to_total_porosity(0.98, 0.05, oven_dried=False) == 1.0

    # Dean-Stark total Sw (Swt = 0.40) with Swb = 0.20 -> Swe = (0.40 - 0.20) / (1 - 0.20) = 0.25
    swe = dean_stark_cbw_correction(0.40, swb=0.20, to_effective=True)
    np.testing.assert_allclose(swe, 0.25, rtol=1e-6)

    # Inverse: Humidity-dried/capillary Swe = 0.25 with Swb = 0.20 -> Swt = 0.25*(0.80) + 0.20 = 0.40
    swt_back = dean_stark_saturation_correction(
        swe, phit=0.20, phie=0.16, to_effective=False
    )
    np.testing.assert_allclose(swt_back, 0.40, rtol=1e-6)

    # Dean-Stark water volume erroneously reported on humidity-dried porosity (Sw_app = 0.50, Swb = 0.20)
    # True Swt = 0.50 * (1 - 0.20) = 0.40 -> Swe = (0.40 - 0.20) / 0.80 = 0.25
    swe_from_hum = dean_stark_cbw_correction(
        0.50, swb=0.20, to_effective=True, reported_on_humidity_poro=True
    )
    np.testing.assert_allclose(swe_from_hum, 0.25, rtol=1e-6)

    # Hill-Shirley-Klein CBW from Qv
    cbw_qv = estimate_cbw_from_qv(qv=0.5, phit=0.20, water_salinity=10000.0)
    assert 0.0 < cbw_qv < 0.20


def test_pc_lab_to_reservoir_and_correct_core_data_pipeline():
    """Verify lab-to-reservoir Pc conversion and DataFrame pipeline helper."""
    # Lab air-brine (72 dynes/cm, 0 deg) -> Reservoir oil-brine (30 dynes/cm, 30 deg)
    pc_lab = np.array([10.0, 50.0])
    pc_res = pc_lab_to_reservoir(
        pc_lab,
        ift_lab=72.0,
        theta_lab=0.0,
        ift_res=30.0,
        theta_res=30.0,
        poro_amb=0.20,
        perm_amb=100.0,
        poro_res=0.19,
        perm_res=85.0,
    )
    fluid_ratio = (30.0 * np.cos(np.radians(30.0))) / 72.0
    stress_ratio = np.sqrt((100.0 / 0.20) / (85.0 / 0.19))
    np.testing.assert_allclose(pc_res, pc_lab * fluid_ratio * stress_ratio, rtol=1e-6)

    df = pd.DataFrame(
        {
            "CPORE": [0.15, 0.20],
            "CPERM": [10.0, 100.0],
            "CSW": [0.45, 0.35],
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
        perm_compaction_coeff=3e-5,
        correct_dean_stark=True,
        replace_ori=False,
    )

    assert "CPORE_AMB" in res.columns
    assert "CPERM_AIR" in res.columns
    assert "CSW_RAW" in res.columns
    np.testing.assert_allclose(res["CPORE_AMB"], df["CPORE"])
    np.testing.assert_allclose(res["CPERM_AIR"], df["CPERM"])

    expected_poro = overburden_porosity_correction(df["CPORE"], df["NCS"]) + (
        df["PHIT"] - df["PHIE"]
    )
    np.testing.assert_allclose(res["CPORE"], expected_poro, rtol=1e-6)
    assert np.all(res["CPERM"] < res["CPERM_AIR"])
    assert np.all(res["CSW"] < res["CSW_RAW"])

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

    core_df = pd.DataFrame(
        {
            "ROCK_FLAG": [1] * len(sw),
            "Well": ["W1"] * len(sw),
            "Sample": ["S1"] * len(sw),
            "SWN": sw,
            "J": j_vals,
        }
    )
    j_param_list = auto_j_params(core_df)
    assert len(j_param_list) == 1
    assert j_param_list[0]["rmse"] < 0.01

    swn = normalize_sw(sw)
    assert swn[0] == 0.0
    assert swn[-1] == 1.0

    j_calc = leverett_j(pc=10.0, ift=30.0, theta=0.0, perm=100.0, phit=0.25)
    np.testing.assert_allclose(j_calc, 0.21665 * (10.0 / 30.0) * 20.0, rtol=1e-5)

    sh_sw = skelt_harrison_func(h=100.0, a=0.8, b=50.0, c=1.5, d=5.0)
    assert 0.0 < sh_sw < 1.0

    # Verify sw_shf_leverett_j and sw_skelt_harrison handle water zone (depth >= fwl) safely
    assert sw_shf_leverett_j(100.0, 0.20, 8600.0, 8550.0, 30.0, 30.0, 1.05, 0.80, 0.15, 1.5) == 1.0
    assert sw_skelt_harrison(8600.0, 8550.0, 0.85, 35.0, 1.5, 2.0) == 1.0

    # Verify Skelt-Harrison and Cuddy curve fitting & auto-params per Rock Type
    h_grid = np.array([2.0, 5.0, 10.0, 20.0, 40.0, 80.0, 150.0])
    sw_sh_true = skelt_harrison_func(h_grid, 0.85, 25.0, 1.5, 2.0)
    sa, sb, sc, sd = fit_skelt_harrison_curve(sw_sh_true, h_grid)
    np.testing.assert_allclose(
        skelt_harrison_func(h_grid, sa, sb, sc, sd), sw_sh_true, atol=0.02
    )

    pc_grid = np.array([0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0])
    bvw_true = cuddy_func(pc_grid, 0.14, 0.38)
    ca, cb = fit_cuddy_curve(bvw_true, pc_grid)
    np.testing.assert_allclose(ca, 0.14, atol=0.01)
    np.testing.assert_allclose(cb, 0.38, atol=0.01)

    scal_df = pd.DataFrame(
        {
            "ROCK_FLAG": [1] * len(h_grid),
            "Well": ["W1"] * len(h_grid),
            "Sample": ["S1"] * len(h_grid),
            "SW": sw_sh_true,
            "H": h_grid,
            "PC": pc_grid,
            "BVW": bvw_true,
        }
    )
    sh_params_auto = auto_skelt_harrison_params(scal_df)
    assert len(sh_params_auto) == 1
    assert sh_params_auto[0]["rmse"] < 0.02

    cuddy_params_auto = auto_cuddy_params(scal_df)
    assert len(cuddy_params_auto) == 1
    assert cuddy_params_auto[0]["rmse"] < 0.01

    # Verify outlier dampening and strict non-crossing across multiple Rock Types
    swn_eval = np.geomspace(0.01, 1.0, 50)
    multi_rt_rows = []
    for rf, (a_true, b_true) in enumerate(
        [(1.4, 0.75), (0.85, 0.95), (0.50, 0.60), (0.28, 1.35), (0.18, 0.70)],
        start=1,
    ):
        for s_val in np.linspace(0.08, 0.85, 12):
            multi_rt_rows.append(
                {
                    "ROCK_FLAG": rf,
                    "Well": "W1",
                    "Sample": f"S{rf}",
                    "SWN": s_val,
                    "J": a_true * (s_val ** (-b_true)),
                }
            )
        # Inject severe boundary & interior outliers
        multi_rt_rows.append(
            {"ROCK_FLAG": rf, "Well": "W1", "Sample": f"S{rf}", "SWN": 0.03, "J": 120.0}
        )
        multi_rt_rows.append(
            {"ROCK_FLAG": rf, "Well": "W1", "Sample": f"S{rf}", "SWN": 0.96, "J": 0.002}
        )
    multi_rt_df = pd.DataFrame(multi_rt_rows)
    multi_j_fits = auto_j_params(multi_rt_df)
    assert len(multi_j_fits) == 5
    curves = [
        p["a"] * (swn_eval ** (-p["b"]))
        for p in sorted(multi_j_fits, key=lambda d: d["ROCK_FLAG"])
    ]
    for idx_c in range(len(curves) - 1):
        diff = curves[idx_c] - curves[idx_c + 1]
        assert np.all(diff > 0) or np.all(diff < 0)


def _make_multi_well_contact_dataset(
    true_fwl: float = 8550.0, model: str = "leverett_j"
) -> tuple[pd.DataFrame, list[dict], pd.DataFrame]:
    """Generate a synthetic 3-well dataset across the capillary transition zone and aquifer,
    along with matching MDT/RFT formation pressure measurements.
    """
    rng = np.random.default_rng(123)
    ift, theta, gw, ghc = 32.0, 30.0, 1.10, 0.82

    j_params = [
        {"ROCK_FLAG": 1, "a": 0.12, "b": 1.65},  # High-quality rock (low entry pressure)
        {"ROCK_FLAG": 2, "a": 0.28, "b": 1.35},  # Moderate-quality rock (higher entry pressure)
    ]
    sh_params = {
        1: {"a": 0.88, "b": 25.0, "c": 1.6, "d": 2.0},
        2: {"a": 0.78, "b": 60.0, "c": 1.3, "d": 5.0},
    }

    well_configs = [
        ("WELL-CREST", np.linspace(8350.0, 8530.0, 90)),
        ("WELL-FLANK", np.linspace(8420.0, 8590.0, 90)),
        ("WELL-DOWNSLOPE", np.linspace(8480.0, 8640.0, 90)),
    ]

    dfs = []
    for w_name, depths in well_configs:
        n = len(depths)
        rf = rng.choice([1, 2], size=n, p=[0.6, 0.4])
        phit = np.where(rf == 1, rng.uniform(0.18, 0.26, n), rng.uniform(0.10, 0.17, n))
        perm = np.where(rf == 1, rng.uniform(80.0, 350.0, n), rng.uniform(8.0, 45.0, n))
        vsh = rng.uniform(0.02, 0.20, n)

        w_df = pd.DataFrame(
            {
                "WELL_NAME": w_name,
                "DEPTH": depths,
                "PHIT": phit,
                "PERM": perm,
                "ROCK_FLAG": rf,
                "VSHALE": vsh,
            }
        )
        active_params = j_params if model == "leverett_j" else sh_params
        sw_true = predict_sw_shm(
            df=w_df,
            fwl=true_fwl,
            model=model,
            shm_params=active_params,
            ift=ift,
            theta=theta,
            gw=gw,
            ghc=ghc,
        )
        # Add realistic small log measurement noise and compute corresponding RT log
        sw_noisy = np.clip(sw_true + rng.normal(0.0, 0.015, size=n), 0.02, 1.0)
        rw = 0.025
        rt = rw / ((phit ** 2.0) * (sw_noisy ** 2.0))
        w_df["RW"] = rw
        w_df["RT"] = rt
        w_df["SWT"] = archie_saturation(rt, rw, phit, a=1.0, m=2.0, n=2.0).clip(0.0, 1.0)
        dfs.append(w_df)

    multi_well_df = pd.concat(dfs, ignore_index=True)

    # Synthetic MDT/RFT pressure data intersecting at true_fwl = 8550.0 ft
    conv = 0.433
    m_w = gw * conv
    m_hc = ghc * conv
    p_at_fwl = 3850.0
    c_w = p_at_fwl - m_w * true_fwl
    c_hc = p_at_fwl - m_hc * true_fwl

    z_hc = np.array([8380.0, 8430.0, 8480.0, 8520.0])
    z_w = np.array([8565.0, 8590.0, 8620.0, 8650.0])
    pressure_df = pd.DataFrame(
        {
            "TVDSS": np.concatenate([z_hc, z_w]),
            "FORMATION_PRESSURE": np.concatenate([m_hc * z_hc + c_hc, m_w * z_w + c_w]),
            "FLUID_TYPE": ["OIL"] * len(z_hc) + ["WATER"] * len(z_w),
        }
    )

    return (
        multi_well_df,
        (j_params if model == "leverett_j" else sh_params),
        pressure_df,
    )


def test_multi_well_fwl_optimizer_leverett_j_and_mdt_reconciliation():
    """Verify optimize_fwl and ContactOptimizer recover true FWL across multiple wells for Leverett-J."""
    true_fwl = 8550.0
    df, j_params, pressure_df = _make_multi_well_contact_dataset(
        true_fwl=true_fwl, model="leverett_j"
    )

    # 1. Verify MDT FWL estimation from formation pressure data
    mdt_res = estimate_mdt_fwl(pressure_df)
    np.testing.assert_allclose(mdt_res["fwl"], true_fwl, atol=0.5)
    np.testing.assert_allclose(mdt_res["gw"], 1.10, atol=1e-2)
    np.testing.assert_allclose(mdt_res["ghc"], 0.82, atol=1e-2)

    # 2. Run simultaneous multi-well FWL optimization (dropping SWT to also verify RT->Archie auto-eval)
    df_no_swt = df.drop(columns=["SWT"])
    opt = optimize_fwl(
        df=df_no_swt,
        model="leverett_j",
        shm_params=j_params,
        ift=32.0,
        theta=30.0,
        gw=1.10,
        ghc=0.82,
        pressure_df=pressure_df,
        pore_volume_weighted=True,
    )

    # Recovered FWL should be within ~2 ft of true 8550.0 ft across the 3 noisy wells
    np.testing.assert_allclose(opt["fwl"], true_fwl, atol=3.0)
    assert abs(opt["fwl_reconciliation_diff"]) < 3.0
    assert opt["rmse"] < 0.03
    assert opt["r2"] > 0.95
    assert opt["ttz"] < opt["owc"] <= opt["fwl"]
    assert opt["transition_zone_thickness"] > 0.0

    # Per-well diagnostics table has all 3 wells and accurate per-well FWLs
    well_metrics = opt["well_metrics"]
    assert len(well_metrics) == 3
    np.testing.assert_allclose(well_metrics["fwl_well"], true_fwl, atol=5.0)

    # Augmented DataFrame contains required columns and valid transition zone flags
    aug_df = opt["df"]
    for col in ("SW_SHM", "BVW_LOG", "BVW_SHM", "HAFWL", "PC_RES", "TRANSITION_ZONE_FLAG"):
        assert col in aug_df.columns
    assert set(aug_df["TRANSITION_ZONE_FLAG"].unique()).issubset(
        {"FREE_WATER", "CAPILLARY_FRINGE", "TRANSITION_ZONE", "IRREDUCIBLE_HC"}
    )

    # Diagnostic plot creation
    fig = plot_fwl_reconciliation(opt)
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_contact_optimizer_class_and_skelt_harrison_multi_well():
    """Verify ContactOptimizer class API and Skelt-Harrison multi-well FWL optimization."""
    true_fwl = 8550.0
    df, sh_params, pressure_df = _make_multi_well_contact_dataset(
        true_fwl=true_fwl, model="skelt_harrison"
    )

    optimizer = ContactOptimizer(
        model="skelt_harrison",
        shm_params=sh_params,
        gw=1.10,
        ghc=0.82,
        loss="huber",
        equal_well_weights=True,
    )
    optimizer.fit(df, pressure_df=pressure_df, mdt_tolerance=80.0, mdt_penalty_weight=0.05)

    np.testing.assert_allclose(optimizer.fwl_, true_fwl, atol=3.0)
    assert optimizer.ttz_ < optimizer.owc_ <= optimizer.fwl_

    sw_pred = optimizer.predict(df)
    assert len(sw_pred) == len(df)
    assert np.all((sw_pred >= 0.0) & (sw_pred <= 1.0))

    transformed = optimizer.transform(df, use_per_well_fwl=True)
    assert "SW_SHM" in transformed.columns
    assert "TRANSITION_ZONE_FLAG" in transformed.columns

    summary = optimizer.summary()
    assert "fwl" in summary and "well_metrics" in summary

    tz_dict = analyze_transition_zone(
        fwl=optimizer.fwl_,
        model="skelt_harrison",
        shm_params=sh_params[1],
        gw=1.10,
        ghc=0.82,
    )
    assert tz_dict["ttz"] < tz_dict["owc"] <= tz_dict["fwl"]

    # Verify per-zone FWL optimization on stacked reservoir zones with distinct FWLs
    z1_df, j_params, _ = _make_multi_well_contact_dataset(
        true_fwl=8450.0, model="leverett_j"
    )
    z1_df["ZONES"] = "UPPER_SAND"
    z2_df, _, _ = _make_multi_well_contact_dataset(
        true_fwl=8580.0, model="leverett_j"
    )
    z2_df["ZONES"] = "LOWER_SAND"
    stacked_df = pd.concat([z1_df, z2_df], ignore_index=True)

    opt_z = optimize_fwl(
        df=stacked_df,
        model="leverett_j",
        shm_params=j_params,
        ift=32.0,
        theta=30.0,
        gw=1.10,
        ghc=0.82,
        solve_per_zone=True,
        zone_col="ZONES",
    )
    assert "fwl_by_zone" in opt_z
    assert "UPPER_SAND" in opt_z["fwl_by_zone"]
    assert "LOWER_SAND" in opt_z["fwl_by_zone"]
    np.testing.assert_allclose(opt_z["fwl_by_zone"]["UPPER_SAND"], 8450.0, atol=5.0)
    np.testing.assert_allclose(opt_z["fwl_by_zone"]["LOWER_SAND"], 8580.0, atol=5.0)
    assert len(opt_z["zone_metrics"]) == 2
