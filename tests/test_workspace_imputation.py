import numpy as np
import pandas as pd
import pytest
from fastapi import HTTPException

from quick_pp.app.backend.services import workspace


@pytest.fixture
def wells(monkeypatch):
    frames = [
        pd.DataFrame({
            "WELL_NAME": ["A"] * 3,
            "DEPTH": [3.0, 1.0, 2.0],
            "GR": [30.0, 10.0, np.nan],
            "RHOB": [3.0, 1.0, 2.0],
        }),
        pd.DataFrame({
            "WELL_NAME": ["B"] * 3,
            "DEPTH": [1.0, 2.0, 3.0],
            "GR": [100.0, np.nan, np.nan],
            "RHOB": [1.0, 2.0, 3.0],
        }),
    ]
    saved = {}

    def load(_project, well_names, min_depth, max_depth, zones):
        assert min_depth is None and max_depth is None and zones is None
        return [df.copy() for df in frames if not well_names or df["WELL_NAME"].iloc[0] in well_names]

    monkeypatch.setattr(workspace, "_load_multi_well_dfs", load)
    monkeypatch.setattr(workspace, "_save_curves_to_well",
                        lambda _project, name, df: saved.update({name: df.copy()}) or len(df))
    return frames, saved


def test_pooled_mean_saves_separate_curves_and_preserves_observations(wells):
    frames, saved = wells
    result = workspace.workspace_impute(1, workspace.WorkspaceImputationRequest(
        curves=["GR"], method="mean", well_names=["A", "B"]
    ))
    assert saved["A"]["GR_IMP"].tolist() == pytest.approx([30, 10, 140 / 3])
    assert saved["B"]["GR_IMP"].tolist() == pytest.approx([100, 140 / 3, 140 / 3])
    assert list(frames[0].columns) == ["WELL_NAME", "DEPTH", "GR", "RHOB"]
    assert result["summaries"][0]["filled"] == 1
    assert result["summaries"][1]["filled"] == 2


@pytest.mark.parametrize(
    ("method", "expected_a", "expected_b"),
    [
        ("ffill", [30, 10, 10], [100, 100, 100]),
        ("bfill", [30, 10, 30], [100, np.nan, np.nan]),
    ],
)
def test_directional_fill_uses_depth_order_and_does_not_cross_wells(wells, method, expected_a, expected_b):
    _, saved = wells
    workspace.workspace_impute(1, workspace.WorkspaceImputationRequest(curves=["GR"], method=method))
    np.testing.assert_allclose(saved["A"]["GR_IMP"], expected_a, equal_nan=True)
    np.testing.assert_allclose(saved["B"]["GR_IMP"], expected_b, equal_nan=True)


@pytest.mark.parametrize("method", ["median", "most_frequent", "knn", "iterative"])
def test_methods_preserve_observed_values(wells, method):
    _, saved = wells
    workspace.workspace_impute(1, workspace.WorkspaceImputationRequest(
        curves=["GR", "RHOB"], method=method, well_names=["A"]
    ))
    assert saved["A"]["GR_IMP"].iloc[0] == 30
    assert saved["A"]["GR_IMP"].iloc[1] == 10
    assert np.isfinite(saved["A"]["GR_IMP"].iloc[2])
    np.testing.assert_array_equal(saved["A"]["RHOB_IMP"], [3, 1, 2])
    assert "B" not in saved


def test_rejects_all_missing_and_existing_output_before_saving(wells):
    frames, saved = wells
    frames[0]["EMPTY"] = np.nan
    frames[1]["EMPTY"] = np.nan
    with pytest.raises(HTTPException, match="No observed values"):
        workspace.workspace_impute(1, workspace.WorkspaceImputationRequest(curves=["EMPTY"], method="mean"))
    frames[1]["GR_IMP"] = 1.0
    with pytest.raises(HTTPException, match="already exists"):
        workspace.workspace_impute(1, workspace.WorkspaceImputationRequest(curves=["GR"], method="mean"))
    assert not saved


def test_matrix_depth_bins_put_deeper_values_first(wells):
    result = workspace.workspace_missing_matrix(
        1, workspace.WorkspaceMissingMatrixRequest(well_name="A")
    )
    index = result["curves"].index("GR")
    assert result["min_depth"] == 1
    assert result["max_depth"] == 3
    assert result["matrix"][index] == [0, 1, 0]
    assert result["samples"] == [1, 1, 1]


def test_pooled_model_can_fill_curve_absent_in_one_well(wells):
    frames, saved = wells
    frames[1] = frames[1].drop(columns="GR")
    workspace.workspace_impute(1, workspace.WorkspaceImputationRequest(
        curves=["GR"], method="median"
    ))
    np.testing.assert_array_equal(saved["B"]["GR_IMP"], [20, 20, 20])


def test_matrix_distinguishes_empty_bins_and_fully_missing_curves(wells):
    frames, _ = wells
    frames[0]["DEPTH"] = [3.0, 1.0, 1.0]
    frames[0]["EMPTY"] = [None, None, None]
    result = workspace.workspace_missing_matrix(
        1, workspace.WorkspaceMissingMatrixRequest(well_name="A")
    )
    assert result["samples"] == [1, 0, 2]
    assert result["matrix"][result["curves"].index("EMPTY")] == [1, None, 1]
