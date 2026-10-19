"""Unit tests for physics-informed ML feature engineering, conformal prediction intervals,
volumetric closure constraints, and P10–P90 well log uncertainty band plotting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from quick_pp.machine_learning.config import FE_FEATURES, PHYSICS_FE_FEATURES
from quick_pp.machine_learning.feature_engineering import (
    acoustic_impedance_feature,
    apparent_matrix_density,
    generate_fe_features,
    mn_lithology_parameters,
)
from quick_pp.machine_learning.predict_pipeline import postprocess_data
from quick_pp.machine_learning.train_pipeline import evaluate_model, train_model
from quick_pp.machine_learning.utils import (
    ConformalMultiOutputRegressor,
    enforce_physical_constraints,
    physics_informed_loss,
    predict_with_uncertainty,
)
from quick_pp.plotter.well_log import plotly_log


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


def test_physics_derived_features():
    """Verify apparent matrix density, acoustic impedance, and M-N lithology parameters."""
    df = _make_synthetic_well_df(60)
    fe_df = generate_fe_features(df)

    for col in FE_FEATURES:
        assert col in fe_df.columns, f"Missing engineered feature: {col}"
        assert fe_df[col].notna().all(), f"NaN values found in feature: {col}"

    for col in PHYSICS_FE_FEATURES:
        assert col in fe_df.columns

    # Pure quartz water-filled sandstone check: rhob = 2.65*(1 - 0.2) + 1.0*0.2 = 2.32, nphi = 0.2
    rhoma = apparent_matrix_density(np.array([2.32]), np.array([0.20]), rho_fluid=1.0)
    np.testing.assert_allclose(rhoma, np.array([2.65]), atol=1e-3)

    # Acoustic impedance with and without sonic log
    ai_measured = acoustic_impedance_feature(df["RHOB"], df["DTC"])
    ai_gardner = acoustic_impedance_feature(df["RHOB"], None)
    assert np.all(ai_measured > 1000.0)
    assert np.all(ai_gardner > 1000.0)

    # M-N lithology parameters
    m_val, n_val = mn_lithology_parameters(df["RHOB"], df["NPHI"], df["DTC"])
    assert np.all(np.isfinite(m_val))
    assert np.all(np.isfinite(n_val))


def test_conformal_prediction_and_closure():
    """Verify ConformalMultiOutputRegressor produces P10-P90 intervals and enforces sum(V_mineral) + PHIT = 1.0."""
    df = generate_fe_features(_make_synthetic_well_df(80))
    targets = ["PHIT", "SWT", "LOG_PERM", "VSAND", "VSILT", "VCLAY"]
    features = ["GR", "RT", "NPHI", "RHOB"] + FE_FEATURES

    X_train = df[features].iloc[:60]
    y_train = df[targets].iloc[:60]
    X_test = df[features].iloc[60:]
    y_test = df[targets].iloc[60:]

    model = train_model(RandomForestRegressor, X_train, y_train, use_conformal=True)
    assert isinstance(model, ConformalMultiOutputRegressor)

    metrics = evaluate_model(model, X_test, y_test)
    assert "r2_score" in metrics
    assert "mean_absolute_error" in metrics
    assert "physics_loss" in metrics
    assert "p10_p90_coverage" in metrics
    assert 0.0 <= metrics["p10_p90_coverage"] <= 1.0

    preds, p10, p90 = predict_with_uncertainty(model, X_test, targets=targets)
    assert preds.shape == (20, len(targets))
    assert p10.shape == (20, len(targets))
    assert p90.shape == (20, len(targets))

    # Monotonicity of P10 <= Pred <= P90
    assert np.all(p10 <= preds + 1e-9)
    assert np.all(preds <= p90 + 1e-9)
    # Non-trivial uncertainty band width
    assert np.all(np.mean(p90 - p10, axis=0) > 0.0)

    # Volumetric closure on predictions: VSAND + VSILT + VCLAY + PHIT == 1.0
    pred_df = pd.DataFrame(preds, columns=targets)
    closure_sum = pred_df[["VSAND", "VSILT", "VCLAY", "PHIT"]].sum(axis=1)
    np.testing.assert_allclose(closure_sum.to_numpy(), np.ones(len(pred_df)), atol=1e-6)


def test_physical_constraints_and_postprocess():
    """Verify physical bounds: PHIE <= PHIT, positive PERM, tight-rock Sw = 1.0, coal clearing, and closure."""
    raw_pred_df = pd.DataFrame(
        {
            "WELL_NAME": ["W1", "W1", "W1", "W1"],
            "DEPTH": [1000.0, 1001.0, 1002.0, 1003.0],
            "PHIT": [0.20, 0.01, 0.25, -0.05],  # row 1 tight, row 3 negative porosity
            "PHIT_P10": [0.15, -0.02, 0.20, -0.10],
            "PHIT_P90": [0.25, 0.03, 0.30, 0.02],
            "PHIE": [0.28, 0.05, 0.22, 0.01],  # row 0 violates PHIE > PHIT
            "PHIE_P10": [0.22, 0.01, 0.18, -0.01],
            "PHIE_P90": [0.32, 0.08, 0.27, 0.05],
            "SWT": [0.35, 0.25, 0.40, 1.20],  # row 1 has unphysical Sw spike in tight rock
            "SWT_P10": [0.25, 0.15, 0.30, 0.95],
            "SWT_P90": [0.45, 0.35, 0.50, 1.30],
            "LOG_PERM": [2.0, 1.5, 2.5, -4.0],
            "LOG_PERM_P10": [1.5, 1.0, 2.0, -4.5],
            "LOG_PERM_P90": [2.5, 2.0, 3.0, -3.5],
            "VSAND": [0.60, 0.50, 0.50, 0.40],
            "VSILT": [0.20, 0.30, 0.20, 0.30],
            "VCLAY": [0.30, 0.40, 0.20, 0.50],
            "TIGHT_FLAG": [0.0, 1.0, 0.0, 0.0],
            "COAL_FLAG": [0.0, 0.0, 1.0, 0.0],  # row 2 is coal
            "OIL_FLAG": [1, 1, 1, 0],
            "GAS_FLAG": [0, 0, 1, 0],
        }
    )

    loss_before = physics_informed_loss(raw_pred_df[["PHIT", "PHIE", "SWT", "VSAND", "VSILT", "VCLAY"]], raw_pred_df[["PHIT", "PHIE", "SWT", "VSAND", "VSILT", "VCLAY"]])
    post_df = postprocess_data(raw_pred_df)
    loss_after = physics_informed_loss(post_df[["PHIT", "PHIE", "SWT", "VSAND", "VSILT", "VCLAY"]], post_df[["PHIT", "PHIE", "SWT", "VSAND", "VSILT", "VCLAY"]])
    assert loss_after < loss_before

    # 1. Porosity hierarchy & non-negativity: 0 <= PHIE <= PHIT
    assert (post_df["PHIT"] >= 0.0).all()
    assert (post_df["PHIE"] >= 0.0).all()
    assert (post_df["PHIE"] <= post_df["PHIT"] + 1e-9).all()

    # 2. Positive permeability & P10-P90 inversion
    assert "PERM" in post_df.columns
    assert "PERM_P10" in post_df.columns
    assert "PERM_P90" in post_df.columns
    assert (post_df["PERM"] > 0.0).all()
    assert (post_df["PERM_P10"] <= post_df["PERM"] + 1e-9).all()
    assert (post_df["PERM"] <= post_df["PERM_P90"] + 1e-9).all()

    # 3. Tight rock (row 1) and zero-porosity rock (row 3) have SWT clamped to 1.0
    assert post_df.loc[1, "SWT"] == 1.0
    assert post_df.loc[3, "SWT"] == 1.0
    assert post_df.loc[1, "OIL_FLAG"] == 0

    # 4. Coal interval (row 2) cleared of reservoir effective porosity and HC flags
    assert post_df.loc[2, "PHIE"] == 0.0
    assert post_df.loc[2, "SWT"] == 1.0
    assert post_df.loc[2, "OIL_FLAG"] == 0
    assert post_df.loc[2, "GAS_FLAG"] == 0

    # 5. Volumetric closure: VSAND + VSILT + VCLAY + PHIT == 1.0
    closure = post_df[["VSAND", "VSILT", "VCLAY", "PHIT"]].sum(axis=1)
    np.testing.assert_allclose(closure.to_numpy(), np.ones(len(post_df)), atol=1e-6)


def test_plotly_log_uncertainty_bands():
    """Verify plotly_log adds P10–P90 shaded uncertainty band traces on the parent curve x-axis."""
    df = _make_synthetic_well_df(30)
    df["PHIT_P10"] = np.maximum(0.0, df["PHIT"] - 0.03)
    df["PHIT_P90"] = np.minimum(0.5, df["PHIT"] + 0.03)
    df["SWT_P10"] = np.maximum(0.0, df["SWT"] - 0.08)
    df["SWT_P90"] = np.minimum(1.0, df["SWT"] + 0.08)
    df["LOG_PERM_P10"] = df["LOG_PERM"] - 0.3
    df["LOG_PERM_P90"] = df["LOG_PERM"] + 0.3
    df = postprocess_data(df)

    fig = plotly_log(df, well_name="WELL-A")
    trace_map = {t.name: t for t in fig.data if t.name}

    for curve in ("PHIT", "PERM", "SWT"):
        assert f"{curve}_P10" in trace_map, f"Missing {curve}_P10 trace"
        assert f"{curve}_P90" in trace_map, f"Missing {curve}_P90 trace"
        assert trace_map[f"{curve}_P90"].fill == "tonextx"
        assert trace_map[f"{curve}_P10"].xaxis == trace_map[curve].xaxis
        assert trace_map[f"{curve}_P90"].xaxis == trace_map[curve].xaxis


def test_group_split_data():
    """Verify split_data prevents well leakage using GroupShuffleSplit and GroupKFold."""
    from quick_pp.machine_learning.train_pipeline import split_data

    # 1. Multi-well dataset: ensure zero overlap of WELL_NAME between train and test
    wells = []
    for i in range(5):
        w_df = _make_synthetic_well_df(20)
        w_df["WELL_NAME"] = f"WELL-{i + 1}"
        wells.append(w_df)
    multi_df = pd.concat(wells, ignore_index=True)

    features = ["GR", "RT", "NPHI", "RHOB"]
    targets = ["PHIT", "SWT"]

    X_train, X_test, y_train, y_test = split_data(
        multi_df, targets, features, test_size=0.2, splitter="group_shuffle"
    )
    train_wells = set(multi_df.loc[X_train.index, "WELL_NAME"])
    test_wells = set(multi_df.loc[X_test.index, "WELL_NAME"])
    assert len(train_wells) >= 1 and len(test_wells) >= 1
    assert train_wells.isdisjoint(test_wells), "Well leakage detected in GroupShuffleSplit!"

    X_train_kf, X_test_kf, _, _ = split_data(
        multi_df, targets, features, splitter="group_kfold", n_splits=5
    )
    train_wells_kf = set(multi_df.loc[X_train_kf.index, "WELL_NAME"])
    test_wells_kf = set(multi_df.loc[X_test_kf.index, "WELL_NAME"])
    assert train_wells_kf.isdisjoint(test_wells_kf), "Well leakage detected in GroupKFold!"

    # 2. Single-well dataset fallback: splits by contiguous depth blocks without error
    single_df = _make_synthetic_well_df(50)
    X_tr_s, X_te_s, y_tr_s, y_te_s = split_data(
        single_df, targets, features, test_size=0.2, splitter="group_shuffle"
    )
    assert len(X_tr_s) + len(X_te_s) == len(single_df)
    assert set(X_tr_s.index).isdisjoint(set(X_te_s.index))

