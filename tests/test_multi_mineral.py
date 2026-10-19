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


