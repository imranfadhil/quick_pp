"""Unit tests for multi-mineral inversion (`quick_pp.lithology.multi_mineral`),
including vectorized Active-Set Bounded Least Squares in `Optim.optimize()`,
`joblib.Parallel` batching, and SLSQP non-linear conductivity models.
"""

import numpy as np

from quick_pp.lithology.multi_mineral import (
    MultiMineral,
    Optim,
    dual_water_conductivity,
    sandsiltclay_waxman_smits_conductivity,
)


def test_vectorized_multimineral_and_optim_active_set_and_nonlinear_conductivity():
    """Verify vectorized Active-Set Bounded Least Squares in Optim.optimize() and MultiMineral,
    parallel batching via joblib.Parallel, and SLSQP fallback for non-linear conductivity models.
    """
    rng = np.random.default_rng(123)
    n_samples = 500

    # Generate synthetic ground-truth volumes on the simplex: [QUARTZ, CALCITE, DOLOMITE, SHALE, OIL, GAS, WATER]
    raw_v = rng.dirichlet(np.ones(7), size=n_samples)
    mm = MultiMineral(minerals=["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"])
    log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC"]
    C = mm._build_response_matrix(log_names)  # (5, 7)

    # Synthetic logs Y = V @ C.T
    Y = raw_v @ C.T
    gr, nphi, rhob, pef, dtc = Y[:, 0], Y[:, 1], Y[:, 2], Y[:, 3], Y[:, 4]

    # 1. Fast vectorized MultiMineral.estimate_lithology() on linear system
    res_df = mm.estimate_lithology(gr=gr, nphi=nphi, rhob=rhob, pef=pef, dtc=dtc)
    vol_cols = ["VSAND", "VCALC", "VDOLO", "VCLAY", "VOIL", "VGAS", "VWATER"]
    V_est = res_df[vol_cols].to_numpy()

    assert res_df["OPTIMIZER_SUCCESS"].all()
    assert np.all(V_est >= -1e-7) and np.all(V_est <= 1.0 + 1e-7)
    np.testing.assert_allclose(np.sum(V_est, axis=1), np.ones(n_samples), atol=1e-5)
    np.testing.assert_allclose(res_df["GR_RECONSTRUCTED"].to_numpy(), gr, atol=1e-2)
    np.testing.assert_allclose(res_df["RHOB_RECONSTRUCTED"].to_numpy(), rhob, atol=1e-2)

    # 2. Direct Optim.optimize(A, b) matrix interface & joblib.Parallel batching
    optim = Optim(A=C, b=Y.T, n_jobs=2)
    V_opt = optim.optimize()  # shape (7, n_samples)
    assert V_opt.shape == (7, n_samples)
    np.testing.assert_allclose(np.sum(V_opt, axis=0), np.ones(n_samples), atol=1e-5)
    assert np.all((V_opt >= -1e-7) & (V_opt <= 1.0 + 1e-7))

    # 3. Non-linear conductivity (sandsiltclay_waxman_smits & dual_water) using SLSQP
    n_nl = 15
    phit_nl = np.sum(raw_v[:n_nl, 4:], axis=1)
    swt_nl = np.clip(raw_v[:n_nl, 6] / np.maximum(phit_nl, 1e-4), 1e-3, 1.0)
    vclay_nl = raw_v[:n_nl, 3]

    ct_ws = sandsiltclay_waxman_smits_conductivity(phit_nl, swt_nl, vclay_nl, rw=0.05)
    rt_ws = 1.0 / np.maximum(ct_ws, 1e-4)

    mm_ws = MultiMineral(
        minerals=["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"],
        conductivity_model="sandsiltclay_waxman_smits",
        conductivity_params={"rw": 0.05},
        n_jobs=2,
    )
    res_ws = mm_ws.estimate_lithology(
        gr=gr[:n_nl],
        nphi=nphi[:n_nl],
        rhob=rhob[:n_nl],
        pef=pef[:n_nl],
        dtc=dtc[:n_nl],
        rt=rt_ws,
    )
    V_ws = res_ws[vol_cols].to_numpy()
    np.testing.assert_allclose(np.sum(V_ws, axis=1), np.ones(n_nl), atol=1e-4)

    ct_dw = dual_water_conductivity(phit_nl, swt_nl, vclay_nl, rw=0.05, rwb=0.15)
    rt_dw = 1.0 / np.maximum(ct_dw, 1e-4)
    mm_dw = MultiMineral(
        minerals=["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"],
        conductivity_model="dual_water",
        conductivity_params={"rw": 0.05, "rwb": 0.15},
    )
    res_dw = mm_dw.estimate_lithology(
        gr=gr[:n_nl],
        nphi=nphi[:n_nl],
        rhob=rhob[:n_nl],
        pef=pef[:n_nl],
        dtc=dtc[:n_nl],
        rt=rt_dw,
    )
    V_dw = res_dw[vol_cols].to_numpy()
    np.testing.assert_allclose(np.sum(V_dw, axis=1), np.ones(n_nl), atol=1e-4)


def test_optim_solve_fast_unity_constraint_weighting():
    """Verify Optim.solve_fast() weights the unity constraint row by lambda=100.0
    so that raw nnls solutions natively satisfy sum(V_i) ~= 1.0 without distorting
    tool reconstruction.
    """
    rng = np.random.default_rng(99)
    mm = MultiMineral(minerals=["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"])
    log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC"]
    C = mm._build_response_matrix(log_names)  # (5, 7)

    # Normalize rows of A_mat and perturb b_vec so an unweighted nnls sum would drift from 1.0
    row_scales = np.max(np.abs(C), axis=1, keepdims=True)
    A_norm = C / np.where(row_scales > 0, row_scales, 1.0)

    v_true = rng.dirichlet(np.ones(7), size=25)  # (25, 7)
    b_clean = A_norm @ v_true.T  # (5, 25)
    # Add multiplicative and additive measurement bias so unconstrained NNLS sum != 1.0
    b_noisy = b_clean * 1.15 + rng.normal(0.0, 0.02, size=b_clean.shape)

    optim = Optim(A=A_norm, b=b_noisy)

    # Raw NNLS with weak weight=1.0 vs strong penalty weight=100.0 (before post-hoc division)
    v_raw_weak = optim.solve_fast(unity_weight=1.0, normalize=False)
    v_raw_strong = optim.solve_fast(unity_weight=100.0, normalize=False)

    weak_sum_err = np.max(np.abs(np.sum(v_raw_weak, axis=0) - 1.0))
    strong_sum_err = np.max(np.abs(np.sum(v_raw_strong, axis=0) - 1.0))
    assert strong_sum_err < 1e-3, f"Expected native NNLS sum within 1e-3 of 1.0, got max diff {strong_sum_err}"
    assert strong_sum_err < weak_sum_err

    # Normalized output from default solve_fast()
    v_fast = optim.solve_fast()
    assert v_fast.shape == (7, 25)
    assert np.all(v_fast >= 0.0)
    np.testing.assert_allclose(np.sum(v_fast, axis=0), np.ones(25), atol=1e-6)




def test_complex_lithology_minerals_and_spectral_gr():
    """Verify newly added minerals (KFELD, GLAUCONITE, ILLITE) are resolved using
    dynamic log ingestion with Spectral Gamma Ray (POTA, THOR, URAN).
    """
    minerals = ["QUARTZ", "KFELD", "GLAUCONITE", "ILLITE"]
    fluids = ["WATER"]
    fluid_props = {"WATER": {"density": 1.0, "nphi": 1.0, "pef": 0.36, "dtc": 190.0, "u": 0.36}}
    mm = MultiMineral(minerals=minerals, fluid_properties=fluid_props)
    mm.fluids = fluids
    mm.bounds = mm.mineral_bounds + tuple({"WATER": (0.0, 1.0)}[f] for f in fluids)

    log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC", "POTA", "THOR"]
    C = mm._build_response_matrix(log_names)

    # 3 geological scenarios:
    # 1. Clean sandstone (90% Quartz, 10% Water)
    # 2. Arkosic sand with high K-Feldspar (60% Quartz, 30% K-Feldspar, 10% Water)
    # 3. Greensand with Glauconite (50% Quartz, 40% Glauconite, 10% Water)
    v_true = np.array([
        [0.90, 0.00, 0.00, 0.00, 0.10],
        [0.60, 0.30, 0.00, 0.00, 0.10],
        [0.50, 0.00, 0.40, 0.00, 0.10],
    ])

    logs_synth = v_true @ C.T
    gr = logs_synth[:, 0]
    nphi = logs_synth[:, 1]
    rhob = logs_synth[:, 2]
    pef = logs_synth[:, 3]
    dtc = logs_synth[:, 4]
    pota = logs_synth[:, 5]
    thor = logs_synth[:, 6]

    res_df = mm.estimate_lithology(
        gr=gr,
        nphi=nphi,
        rhob=rhob,
        pef=pef,
        dtc=dtc,
        pota=pota,
        thor=thor,
    )

    # Ensure expected mapped column names exist
    assert "VSAND" in res_df.columns
    assert "VKFELD" in res_df.columns
    assert "VGLAUC" in res_df.columns
    assert "VILLI" in res_df.columns
    assert "VWATER" in res_df.columns
    assert "POTA_RECONSTRUCTED" in res_df.columns
    assert "THOR_RECONSTRUCTED" in res_df.columns

    # Check that K-Feldspar scenario correctly isolates VKFELD > 0.25 without confusing with ILLITE
    assert res_df["VKFELD"].iloc[1] > 0.25
    assert res_df["VILLI"].iloc[1] < 0.05

    # Check that Glauconite scenario correctly isolates VGLAUC > 0.35
    assert res_df["VGLAUC"].iloc[2] > 0.35
    assert res_df["VILLI"].iloc[2] < 0.05


def test_custom_bounds_and_pef_linearization():
    """Verify custom bounds clamp accessory minerals and linearize_pef computes exact U."""
    # Constrain GLAUCONITE to maximum 20%
    custom_bounds = {"GLAUCONITE": (0.0, 0.20)}
    mm = MultiMineral(
        minerals=["QUARTZ", "GLAUCONITE", "SHALE"],
        mineral_bounds=custom_bounds,
        linearize_pef=True,
    )

    # Synthetic logs forcing high glauconite
    gr = np.array([180.0, 180.0])
    nphi = np.array([0.25, 0.25])
    rhob = np.array([2.80, 2.80])
    pef = np.array([5.50, 5.50])
    dtc = np.array([90.0, 90.0])

    res_df = mm.estimate_lithology(gr=gr, nphi=nphi, rhob=rhob, pef=pef, dtc=dtc, linearize_pef=True)

    # Verify GLAUCONITE volume does not exceed the custom upper bound of 0.20
    assert np.all(res_df["VGLAUC"] <= 0.20 + 1e-4)
    assert np.all(res_df["OPTIMIZER_SUCCESS"])
    assert "PEF_RECONSTRUCTED" in res_df.columns


def test_borehole_washout_weighting():
    """Verify bad-hole weighting de-weights corrupted RHOB and PEF in washed-out intervals."""
    mm = MultiMineral(minerals=["QUARTZ", "CALCITE", "SHALE"])

    # 2 depth points:
    # Point 0: Good hole
    # Point 1: Severe washout (cali = 12.5 in, bit size = 8.5 in; density falsely drops to 1.80)
    gr = np.array([5.0, 5.0])
    nphi = np.array([0.02, 0.02])
    rhob = np.array([2.60, 1.80])   # point 1 has severe washout density artifact
    pef = np.array([1.9, 0.5])      # point 1 has severe PEF artifact
    dtc = np.array([55.0, 55.0])    # DTC is unaffected by shallow washout

    cali = np.array([8.5, 12.5])
    bs = 8.5

    # With bad hole weighting active, point 1 down-weights RHOB/PEF, relying on GR/DTC/NPHI
    res_df = mm.estimate_lithology(
        gr=gr,
        nphi=nphi,
        rhob=rhob,
        pef=pef,
        dtc=dtc,
        cali=cali,
        bs=bs,
        bad_hole_weight=0.01,
    )

    # With bad hole weighting, point 1 avoids the false porosity blowout caused by rhob=1.80
    assert res_df["PHIT_CONSTRUCTED"].iloc[1] < 0.05
    assert res_df["RHOB_RECONSTRUCTED"].iloc[1] > 2.60
    assert res_df["OPTIMIZER_SUCCESS"].iloc[1]
