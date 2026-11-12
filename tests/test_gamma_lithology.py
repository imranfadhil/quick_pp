import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quick_pp.app.backend.main import app
from quick_pp.app.backend.services import workspace
from quick_pp.lithology import spectral_cgr


@pytest.fixture
def client_and_state(monkeypatch):
    state = {
        "df": pd.DataFrame(
            {
                "DEPTH": [1000.0, 1001.0, 1002.0],
                "GR": [20.0, 70.0, 120.0],
                "CGR": [10.0, 40.0, 70.0],
                "THOR": [1.0, 4.0, 7.0],
                "POTA": [0.5, 1.5, 2.0],
                "RHOB": [2.32, 2.325, 2.36],
                "NPHI": [0.2, 0.2, 0.2],
                "BADHOLE": [0.0, 0.0, 0.0],
            }
        )
    }

    def update(_pid, _well, output):
        state["df"] = state["df"].copy()
        for col in output:
            if col != "DEPTH":
                state["df"][col] = output[col].to_numpy()

    monkeypatch.setattr(
        workspace, "_get_filtered_well_df", lambda *args, **kwargs: state["df"].copy()
    )
    monkeypatch.setattr(workspace, "_update_in_memory_well_cache", update)
    monkeypatch.setattr(
        workspace, "_record_well_model_config", lambda *args, **kwargs: None
    )
    return TestClient(app), state


def test_spectral_cgr_invalid_inputs():
    np.testing.assert_allclose(
        spectral_cgr([10, -1, 5, np.nan], [2, 3, 0, 1], 4, 16),
        [72, np.nan, 20, np.nan],
        equal_nan=True,
    )
    with pytest.raises(ValueError, match="coefficients"):
        spectral_cgr([10], [2], 0, 16)


@pytest.mark.parametrize(
    "source,extra,expected",
    [
        ("gr", {"gr_clean": 20, "gr_clay": 120}, [0, 0.5, 1]),
        (
            "cgr",
            {"gamma_curve": "CGR", "gamma_clean": 10, "gamma_shale": 70},
            [0, 0.5, 1],
        ),
        ("derived_cgr", {"gamma_clean": 12, "gamma_shale": 60}, [0, 28 / 48, 1]),
    ],
)
def test_gamma_lithology_is_independent_of_porosity(
    client_and_state, source, extra, expected
):
    client, state = client_and_state
    root = "/quick_pp/workspace/projects/1/wells/W"
    payload = {"model": "vsh_gr", "gamma_source": source, **extra}
    lith = client.post(root + "/lithology", json=payload)
    assert lith.status_code == 200, lith.text
    data = lith.json()["data"]
    np.testing.assert_allclose(data["VCLAY"], expected)
    np.testing.assert_allclose(np.array(data["VSAND"]) + data["VCLAY"], 1)
    if source == "derived_cgr":
        assert data["CGR_SYN"] == [12, 40, 60]
    original = state["df"][["VCLAY", "VSAND"]].copy()
    poro = client.post(
        root + "/porosity", json={"lithology_model": "vsh_gr", "method": "den"}
    )
    assert poro.status_code == 200, poro.text
    pd.testing.assert_frame_equal(state["df"][["VCLAY", "VSAND"]], original)
    state["df"]["RHOB"] = [2.5, 2.5, 2.5]
    again = client.post(root + "/lithology", json=payload)
    assert again.status_code == 200, again.text
    np.testing.assert_allclose(again.json()["data"]["VCLAY"], expected)


def test_gamma_missing_curve_and_invalid_endpoints(client_and_state):
    client, _ = client_and_state
    path = "/quick_pp/workspace/projects/1/wells/W/lithology"
    response = client.post(
        path, json={"model": "vsh_gr", "gamma_source": "cgr", "gamma_curve": "MISSING"}
    )
    assert response.status_code == 400
    response = client.post(
        path,
        json={
            "model": "vsh_gr",
            "gamma_source": "derived_cgr",
            "gamma_clean": 20,
            "gamma_shale": 20,
        },
    )
    assert response.status_code == 400


def test_derived_gamma_missing_sample_clears_previous_lithology(client_and_state):
    client, state = client_and_state
    path = "/quick_pp/workspace/projects/1/wells/W/lithology"
    payload = {
        "model": "vsh_gr",
        "gamma_source": "derived_cgr",
        "gamma_clean": 12,
        "gamma_shale": 60,
    }
    assert client.post(path, json=payload).status_code == 200
    state["df"].loc[1, "THOR"] = np.nan
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["VCLAY"][1] is None
    assert pd.isna(state["df"].loc[1, "VCLAY"])
    assert pd.isna(state["df"].loc[1, "CGR_SYN"])


def test_ssc_lithology_survives_porosity_calculation(client_and_state):
    client, state = client_and_state
    root = "/quick_pp/workspace/projects/1/wells/W"
    lith = client.post(root + "/lithology", json={"model": "ssc"})
    assert lith.status_code == 200, lith.text
    cols = ["VSAND", "VSILT", "VCLAY"]
    original = state["df"][cols].copy()
    np.testing.assert_allclose(original.sum(axis=1), 1)
    poro = client.post(
        root + "/porosity", json={"lithology_model": "ssc", "method": "neu_den"}
    )
    assert poro.status_code == 200, poro.text
    pd.testing.assert_frame_equal(state["df"][cols], original)


def test_multi_mineral_reports_matrix_relative_volumes(client_and_state, monkeypatch):
    client, state = client_and_state
    monkeypatch.setattr(
        workspace.MultiMineral,
        "estimate_lithology",
        lambda *args, **kwargs: pd.DataFrame(
            {
                "VSAND": [0.4, 0.6, 0.0],
                "VCLAY": [0.4, 0.2, 0.0],
                "PHIT_CONSTRUCTED": [0.2, 0.2, 1.0],
            }
        ),
    )
    root = "/quick_pp/workspace/projects/1/wells/W"
    lith = client.post(root + "/lithology", json={"model": "multi_mineral"})
    assert lith.status_code == 200, lith.text
    data = lith.json()["data"]
    assert data["VSAND"] == pytest.approx([0.5, 0.75, None], nan_ok=True)
    assert data["VCLAY"] == pytest.approx([0.5, 0.25, None], nan_ok=True)
    assert data["PHIT_CONSTRUCTED"] == [0.2, 0.2, 1.0]
    original = state["df"][["VSAND", "VCLAY"]].copy()
    poro = client.post(
        root + "/porosity", json={"lithology_model": "multi_mineral", "method": "den"}
    )
    assert poro.status_code == 200, poro.text
    pd.testing.assert_frame_equal(state["df"][["VSAND", "VCLAY"]], original)
