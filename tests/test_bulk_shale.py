import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quick_pp.app.backend.main import app
from quick_pp.app.backend.services import workspace
from quick_pp.lithology.bulk_shale import bulk_shale_density, spectral_cgr


def test_bulk_shale_mixtures_close_and_respond_to_density_and_pore_parameters():
    vsh = np.array([0.0, 0.5, 1.0])
    clean_pore = np.array([0.2, 0.15, 0.0])
    shale_phi = 0.2
    clean_solid = 1 - vsh - clean_pore
    shale_solid = vsh * (1 - shale_phi)
    phit = clean_pore + vsh * shale_phi
    rhob = clean_solid * 2.65 + shale_solid * 2.7 + phit
    result = bulk_shale_density(
        [20, 70, 120], rhob, 20, 120, 2.65, 2.7, 1.0, shale_phi, 0.5
    )
    np.testing.assert_allclose(result["VSHALE"], vsh)
    np.testing.assert_allclose(result["VCLAY"], shale_solid)
    np.testing.assert_allclose(result["VSAND"], clean_solid)
    np.testing.assert_allclose(result["PHIT"], phit)
    np.testing.assert_allclose(result["PHIE"], phit - 0.5 * vsh * shale_phi)
    np.testing.assert_allclose(result["VCLAY"] + result["VSAND"] + result["PHIT"], 1)
    changed = bulk_shale_density(
        [20, 70, 120], rhob, 20, 120, 2.65, 2.8, 1.0, shale_phi, 0.5
    )
    assert changed["PHIT"][1] != pytest.approx(result["PHIT"][1])
    assert changed["RHOM"][1] != pytest.approx(result["RHOM"][1])


def test_spectral_cgr_and_invalid_mixtures_are_not_silently_substituted():
    result = spectral_cgr([10, -1, 5, np.nan], [2, 3, 0, 1], 4, 16)
    np.testing.assert_allclose(result, [72, np.nan, 20, np.nan], equal_nan=True)
    with pytest.raises(ValueError, match="coefficients"):
        spectral_cgr([10], [2], 0, 16)
    result = bulk_shale_density([20, 70, 120], [2.4, 3.2, np.nan], 20, 120, 2.65, 2.7, 1, .2, 1)
    assert np.isfinite(result["PHIT"][0])
    assert np.isnan(result["PHIT"][1:]).all()
    with pytest.raises(ValueError, match="Shale gamma endpoint"):
        bulk_shale_density([20], [2.4], 20, 20, 2.65, 2.7, 1, .2, 1)


@pytest.fixture
def bulk_client(monkeypatch):
    base = pd.DataFrame({
        "DEPTH": [1000.0, 1001.0, 1002.0],
        "GR": [20.0, 70.0, 120.0],
        "CGR": [10.0, 40.0, 70.0],
        "THOR": [1.0, 4.0, 7.0],
        "POTA": [0.5, 1.5, 2.0],
        "RHOB": [2.32, 2.325, 2.36],
        "NPHI": [0.2, 0.2, 0.2],
        "BADHOLE": [0.0, 1.0, 0.0],
        "RT": [10.0, 10.0, 10.0],
        "SWT": [0.2, 0.2, 0.2],
        "ZONES": ["A", "A", "A"],
    })
    state = {"df": base}

    def update(_pid, _well, output):
        state["df"] = state["df"].copy()
        for col in output:
            if col != "DEPTH":
                state["df"][col] = output[col].to_numpy()

    monkeypatch.setattr(workspace, "_get_filtered_well_df", lambda *args, **kwargs: state["df"].copy())
    monkeypatch.setattr(workspace, "_update_in_memory_well_cache", update)
    monkeypatch.setattr(workspace, "_record_well_model_config", lambda *args, **kwargs: None)
    return TestClient(app), state


def test_bulk_lithology_and_porosity_endpoints_are_consistent(bulk_client):
    client, _ = bulk_client
    path = "/quick_pp/workspace/projects/1/wells/W/lithology"
    payload = {"model": "vsh_bulk", "badhole_treatment": "none", "gamma_clean": 20,
               "gamma_shale": 120, "enable_coal_flagging": False}
    lith = client.post(path, json=payload)
    assert lith.status_code == 200, lith.text
    poro = client.post(path.replace("lithology", "porosity"), json={
        **payload, "lithology_model": "vsh_bulk", "method": "den",
    })
    assert poro.status_code == 200, poro.text
    ld, pd_ = lith.json()["data"], poro.json()["data"]
    for curve in ("VSHALE", "VSAND", "VCLAY", "PHIT", "PHIE", "PHID", "RHOM"):
        np.testing.assert_allclose(ld[curve], pd_[curve])
    np.testing.assert_allclose(ld["VSHALE"], [0, 0.5, 1])
    np.testing.assert_allclose(np.array(ld["VSAND"]) + ld["VCLAY"] + ld["PHIT"], 1)
    assert ld["VCLAY"][2] == pytest.approx(.8)
    assert ld["VSHALE"][2] == 1


def test_bulk_source_selection_badhole_and_errors(bulk_client):
    client, _ = bulk_client
    path = "/quick_pp/workspace/projects/1/wells/W/lithology"
    payload = {"model": "vsh_bulk", "badhole_treatment": "ignore",
               "gamma_source": "cgr", "gamma_curve": "CGR",
               "gamma_clean": 10, "gamma_shale": 70}
    response = client.post(path, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["data"]["VSHALE"] == [0, None, 1]
    response = client.post(path, json={**payload, "gamma_source": "derived_cgr",
                                       "gamma_clean": 12, "gamma_shale": 60})
    assert response.status_code == 200, response.text
    assert response.json()["data"]["CGR_SYN"] == [12, 40, 60]
    assert client.post(path, json={**payload, "gamma_curve": "MISSING"}).status_code == 400
    assert client.post(path, json={**payload, "badhole_treatment": "vsh_gr"}).status_code == 400
    assert client.post(path, json={**payload, "use_hc_corrected": True}).status_code == 400
    assert client.post(path.replace("lithology", "porosity"), json={
        **payload, "lithology_model": "vsh_bulk", "method": "neu_den"
    }).status_code == 400


def test_bulk_shale_is_used_for_shale_saturation_and_summary(bulk_client, monkeypatch):
    client, _ = bulk_client
    root = "/quick_pp/workspace/projects/1/wells/W"
    response = client.post(root + "/lithology", json={
        "model": "vsh_bulk", "badhole_treatment": "none",
        "gamma_clean": 20, "gamma_shale": 120,
    })
    assert response.status_code == 200, response.text
    captured = []
    original = workspace.simandoux_saturation

    def capture(*args, **kwargs):
        captured.extend(args[3])
        return original(*args, **kwargs)

    monkeypatch.setattr(workspace, "simandoux_saturation", capture)
    sat = client.post(root + "/saturation", json={
        "method": "simandoux", "lithology_model": "vsh_bulk",
    })
    assert sat.status_code == 200, sat.text
    np.testing.assert_allclose(captured, [0, .5, 1])
    summary = client.post(root + "/reservoir_summary", json={
        "lithology_model": "vsh_bulk",
        "cut_offs": {"VSHALE": .45, "PHIT": .05, "SWT": .8},
    })
    assert summary.status_code == 200, summary.text
    assert summary.json()["flags"]["data"]["RES_FLAG"] == [1, 0, 0]
