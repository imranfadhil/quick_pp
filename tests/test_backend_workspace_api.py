import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quick_pp.app.backend.main import app
from quick_pp.app.backend.services import database as db_service
from quick_pp.app.backend.services import rock_typing as rt_service
from quick_pp.app.backend.services import workspace as ws_service
from quick_pp.database.models import Base
from quick_pp.database.objects import Project, Well


@pytest.fixture
def client_and_project(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    proj = Project(db_session=session, name="workspace_test_proj")
    proj.save()

    depths = np.linspace(1000.0, 1009.0, 10)
    df = pd.DataFrame(
        {
            "DEPTH": depths,
            "TVD": depths,
            "WELL_NAME": ["Well-A"] * 10,
            "GR": np.linspace(30.0, 110.0, 10),
            "NPHI": np.linspace(0.12, 0.28, 10),
            "RHOB": np.linspace(2.25, 2.55, 10),
            "RT": np.linspace(25.0, 4.0, 10),
            "DTC": np.linspace(62.0, 52.0, 10),
            "ZONES": ["Zone-1"] * 5 + ["Zone-2"] * 5,
            "CPORE": [
                0.22,
                np.nan,
                0.19,
                np.nan,
                0.17,
                np.nan,
                0.15,
                np.nan,
                0.13,
                np.nan,
            ],
            "CPERM": [
                150.0,
                np.nan,
                80.0,
                np.nan,
                40.0,
                np.nan,
                15.0,
                np.nan,
                5.0,
                np.nan,
            ],
        }
    )
    well_a = Well(
        db_session=session, project_id=proj.project_id, name="Well-A", uwi="WELL-A-1"
    )
    well_a.save()
    proj.update_data(df)
    proj.save()

    well_a.add_formation_tops(
        [{"name": "Zone-1", "depth": 1000.0}, {"name": "Zone-2", "depth": 1005.0}]
    )
    session.commit()

    # Also add an empty well to verify multi-well endpoints skip empty wells gracefully
    well_empty = Well(
        db_session=session, project_id=proj.project_id, name="Well-Empty", uwi="EMPTY-1"
    )
    well_empty.save()
    session.commit()

    class DummyCtx:
        def __init__(self, s):
            self._s = s

        def __enter__(self):
            return self._s

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db_service, "get_db", lambda: DummyCtx(session))
    monkeypatch.setattr(ws_service, "get_db", lambda: DummyCtx(session))
    monkeypatch.setattr(rt_service, "get_db", lambda: DummyCtx(session))
    db_service.invalidate_well_cache()
    ws_service._batch_previews.clear()

    client = TestClient(app)
    yield client, proj.project_id
    session.close()


def test_bulk_shale_save_removes_invalid_prior_values(client_and_project):
    client, pid = client_and_project
    path = f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology"
    payload = {
        "model": "vsh_bulk", "badhole_treatment": "none", "save": True,
        "gamma_clean": 0.0, "gamma_shale": 1000.0,
    }
    initial = client.post(path, json=payload)
    assert initial.status_code == 200, initial.text
    assert initial.json()["data"]["PHIT"][-1] is not None
    replacement = client.post(path, json={**payload, "gamma_shale": 110.0})
    assert replacement.status_code == 200, replacement.text
    assert replacement.json()["data"]["PHIT"][-1] is None

    from sqlalchemy import select

    from quick_pp.database.models import Curve, CurveData
    from quick_pp.database.models import Well as WellModel

    with ws_service.get_db() as session:
        rows = session.execute(
            select(CurveData.depth)
            .join(Curve, CurveData.curve_id == Curve.curve_id)
            .join(WellModel, Curve.well_id == WellModel.well_id)
            .where(WellModel.project_id == pid, WellModel.name == "Well-A", Curve.mnemonic == "PHIT")
        ).scalars().all()
    assert 1009.0 not in rows
    assert rows


@pytest.mark.parametrize(
    "method",
    ["sonic_wyllie", "sonic_hunt_raymer", "avg_weighted", "avg_arithmetic", "avg_gaymard"],
)
def test_workspace_porosity_methods_use_lithology_and_fluid(client_and_project, method):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    lith = client.post(f"{base}/lithology", json={"model": "ssc"})
    assert lith.status_code == 200, lith.text
    response = client.post(
        f"{base}/porosity",
        json={"method": method, "lithology_model": "ssc", "fluid_point": [1.0, 1.1], "sonic_fluid_dt": 210},
    )
    assert response.status_code == 200, response.text
    result = response.json()["data"]
    lith_data = lith.json()["data"]
    vols = np.array([lith_data[c][0] for c in ("VSAND", "VSILT", "VCLAY")])
    vols /= vols.sum()
    if method.startswith("sonic_"):
        dt_ma = np.dot(vols, [52.9, 52.9, 120.0])
        if method == "sonic_wyllie":
            expected = (62.0 - dt_ma) / (210.0 - dt_ma)
        else:
            c = dt_ma / (2 * 210.0) - 1
            expected = -c - np.sqrt(c * c + dt_ma / 62.0 - 1)
    else:
        phid = result["PHID"][0]
        assert phid == pytest.approx((result["RHOM"][0] - 2.25) / (result["RHOM"][0] - 1.1))
        expected = {
            "avg_weighted": (2 * phid + 0.12) / 3,
            "avg_arithmetic": (phid + 0.12) / 2,
            "avg_gaymard": np.sqrt((phid**2 + 0.12**2) / 2),
        }[method]
    assert result["PHIT"][0] == pytest.approx(np.clip(expected, 0, 1), abs=1e-6)


def test_sonic_missing_samples_stay_null_and_missing_curve_is_rejected(client_and_project, monkeypatch):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    assert client.post(f"{base}/lithology", json={"model": "ssc"}).status_code == 200
    original = ws_service._get_filtered_well_df

    def missing_sample(*args, **kwargs):
        df = original(*args, **kwargs).copy()
        df.loc[df.index[1], "DTC"] = np.nan
        return df

    monkeypatch.setattr(ws_service, "_get_filtered_well_df", missing_sample)
    result = client.post(f"{base}/porosity", json={"method": "sonic_wyllie", "lithology_model": "ssc"})
    assert result.status_code == 200, result.text
    assert result.json()["data"]["PHIT"][1] is None

    def missing_curve(*args, **kwargs):
        return original(*args, **kwargs).drop(columns="DTC")

    monkeypatch.setattr(ws_service, "_get_filtered_well_df", missing_curve)
    result = client.post(f"{base}/porosity", json={"method": "sonic_wyllie", "lithology_model": "ssc"})
    assert result.status_code == 400
    assert "requires a DTC or DT curve" in result.json()["detail"]


def test_sonic_uses_extended_mineral_and_rejects_invalid_fluid_dt(client_and_project, monkeypatch):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    original = ws_service._get_filtered_well_df

    def illite_lithology(*args, **kwargs):
        df = original(*args, **kwargs).copy()
        for column in ws_service.ALL_LITHOLOGY_VOL_COLS:
            df[column] = 0.0
        df["VILLI"] = 1.0
        df["DTC"] = 130.0
        return df

    monkeypatch.setattr(ws_service, "_get_filtered_well_df", illite_lithology)
    payload = {"method": "sonic_wyllie", "lithology_model": "multi_mineral", "sonic_fluid_dt": 210}
    result = client.post(f"{base}/porosity", json=payload)
    assert result.status_code == 200, result.text
    assert result.json()["data"]["PHIT"][0] == pytest.approx((130 - 110) / (210 - 110))
    result = client.post(f"{base}/porosity", json={**payload, "sonic_fluid_dt": 100})
    assert result.status_code == 400
    assert "No valid sonic porosity results" in result.json()["detail"]


def test_sonic_save_clears_previously_saved_porosity_at_missing_samples(client_and_project, monkeypatch):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    assert client.post(f"{base}/lithology", json={"model": "ssc", "save": True}).status_code == 200
    assert client.post(f"{base}/porosity", json={"method": "den", "lithology_model": "ssc", "save": True}).status_code == 200
    original = ws_service._get_filtered_well_df

    def missing_sample(*args, **kwargs):
        df = original(*args, **kwargs).copy()
        df.loc[df.index[1], "DTC"] = np.nan
        return df

    monkeypatch.setattr(ws_service, "_get_filtered_well_df", missing_sample)
    response = client.post(f"{base}/porosity", json={"method": "sonic_wyllie", "lithology_model": "ssc", "save": True})
    assert response.status_code == 200, response.text
    from sqlalchemy import select

    from quick_pp.database.models import Curve, CurveData
    from quick_pp.database.models import Well as WellModel

    with ws_service.get_db() as session:
        depths = session.execute(
            select(CurveData.depth)
            .join(Curve, CurveData.curve_id == Curve.curve_id)
            .join(WellModel, Curve.well_id == WellModel.well_id)
            .where(WellModel.project_id == pid, WellModel.name == "Well-A", Curve.mnemonic == "PHIT")
        ).scalars().all()
    assert 1001.0 not in depths
    assert 1000.0 in depths


@pytest.mark.parametrize("method", ["sonic_wyllie", "avg_arithmetic"])
def test_new_porosity_methods_honor_gr_badhole_treatment(client_and_project, monkeypatch, method):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    original = ws_service._get_filtered_well_df

    def with_badhole(*args, **kwargs):
        df = original(*args, **kwargs).copy()
        df["BADHOLE"] = 0
        df.loc[df.index[0], "BADHOLE"] = 1
        return df

    monkeypatch.setattr(ws_service, "_get_filtered_well_df", with_badhole)
    response = client.post(
        f"{base}/porosity",
        json={"method": method, "lithology_model": "vsh_gr", "badhole_treatment": "vsh_gr", "gr_clean": 20, "gr_clay": 120},
    )
    assert response.status_code == 200, response.text
    data = response.json()["data"]
    if method == "sonic_wyllie":
        dt_ma = 0.9 * 52.9 + 0.1 * 120.0
        expected = (62.0 - dt_ma) / (190.0 - dt_ma)
    else:
        expected = (data["PHID"][0] + 0.12) / 2
    assert data["PHIT"][0] == pytest.approx(expected)


def test_porosity_comparison_is_isolated_and_matches_active_method(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    assert client.post(f"{base}/lithology", json={"model": "ssc"}).status_code == 200
    payload = {"method": "avg_weighted", "lithology_model": "ssc", "sonic_fluid_dt": 210}
    active = client.post(f"{base}/porosity", json=payload)
    assert active.status_code == 200, active.text
    before = ws_service._get_filtered_well_df(pid, "Well-A").copy()
    before_config = dict(ws_service._ephemeral_well_configs.get((pid, "Well-A"), {}))
    comparison = client.post(f"{base}/porosity_comparison", json=payload)
    assert comparison.status_code == 200, comparison.text
    body = comparison.json()
    assert len(body["curves"]) == 7
    assert body["curves"]["avg_weighted"] == pytest.approx(active.json()["data"]["PHIT"])
    pd.testing.assert_frame_equal(before, ws_service._get_filtered_well_df(pid, "Well-A"))
    assert before_config == ws_service._ephemeral_well_configs.get((pid, "Well-A"), {})
    assert client.post(f"{base}/porosity_comparison", json={**payload, "save": True}).status_code == 400
    assert client.post(f"{base}/porosity_comparison", json={**payload, "lithology_model": "vsh_bulk"}).status_code == 400


def test_porosity_comparison_reports_unavailable_sonic_methods(client_and_project, monkeypatch):
    client, pid = client_and_project
    original = ws_service._get_filtered_well_df
    monkeypatch.setattr(
        ws_service, "_get_filtered_well_df",
        lambda *args, **kwargs: original(*args, **kwargs).drop(columns="DTC", errors="ignore"),
    )
    response = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/porosity_comparison",
        json={"lithology_model": "ssc"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["curves"]) == 5
    assert set(body["errors"]) == {"sonic_wyllie", "sonic_hunt_raymer"}


@pytest.mark.parametrize("use_ts", [False, True])
def test_permeability_comparison_is_isolated_and_matches_active_method(client_and_project, use_ts):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"
    payload = {
        "method": "choo", "m": 2.2, "A": 800000, "B": 2.5, "C": 1.5,
        "s": 0.02, "swirr": 0.25, "use_thomas_stieber": use_ts,
        "depths": list(np.linspace(1000, 1009, 10)),
        "phit": [0.2] * 10, "vclay": [0.1] * 10, "vsilt": [0.05] * 10,
        "phit_sand": [0.3] * 10, "vsh_lam": [0.2] * 10, "vsh_dis": [0.04] * 10,
    }
    active = client.post(f"{base}/permeability", json=payload)
    assert active.status_code == 200, active.text
    before = ws_service._get_filtered_well_df(pid, "Well-A").copy()
    before_config = dict(ws_service._ephemeral_well_configs.get((pid, "Well-A"), {}))
    comparison = client.post(f"{base}/permeability_comparison", json=payload)
    assert comparison.status_code == 200, comparison.text
    body = comparison.json()
    assert set(body["curves"]) == {"choo", "kozeny_carman", "timur", "tixier", "coates"}
    assert body["curves"]["choo"] == pytest.approx(active.json()["data"]["PERM"])
    pd.testing.assert_frame_equal(before, ws_service._get_filtered_well_df(pid, "Well-A"))
    assert before_config == ws_service._ephemeral_well_configs.get((pid, "Well-A"), {})
    for method in ("kozeny_carman", "timur", "tixier", "coates"):
        independent = client.post(f"{base}/permeability", json={**payload, "method": method})
        assert independent.status_code == 200, independent.text
        assert body["curves"][method] == pytest.approx(independent.json()["data"]["PERM"])
    assert client.post(f"{base}/permeability_comparison", json={**payload, "save": True}).status_code == 400


def test_permeability_comparison_reports_per_method_errors(client_and_project, monkeypatch):
    client, pid = client_and_project
    original = ws_service._get_filtered_well_df

    def no_lithology(*args, **kwargs):
        df = original(*args, **kwargs).copy()
        df["PHIT"] = 0.2
        return df.drop(columns=["NPHI", "RHOB", "VCLAY", "VSILT", "VSHALE"], errors="ignore")

    monkeypatch.setattr(ws_service, "_get_filtered_well_df", no_lithology)
    response = client.post(f"/quick_pp/workspace/projects/{pid}/wells/Well-A/permeability_comparison", json={})
    assert response.status_code == 200, response.text
    body = response.json()
    assert set(body["curves"]) == {"kozeny_carman", "timur", "tixier", "coates"}
    assert "VCLAY" in body["errors"]["choo"]


def test_project_summary_endpoint(client_and_project):
    client, pid = client_and_project
    r = client.get(f"/quick_pp/database/projects/{pid}/summary")
    assert r.status_code == 200
    body = r.json()
    assert body["project_id"] == pid
    assert body["well_count"] == 2
    assert "Zone-1" in body["all_zones"]
    assert "NPHI" in body["all_curves"]
    well_a = next(w for w in body["wells"] if w["name"] == "Well-A")
    assert well_a["min_depth"] == 1000.0
    assert well_a["max_depth"] == 1009.0
    assert well_a["total_depth"] == 9.0
    assert well_a["row_count"] == 10
    assert well_a["core_sample_count"] == 5
    assert well_a["core_availability_pct"] == 50.0


def test_workspace_zone_filter_requires_zone_data(monkeypatch):
    merged = pd.DataFrame(
        {"DEPTH": [1000.0, 1001.0], "NPHI": [0.2, 0.25], "RHOB": [2.4, 2.5]}
    )
    monkeypatch.setattr(
        ws_service, "_get_merged_dataframe_sync", lambda *args, **kwargs: merged
    )

    with pytest.raises(ValueError, match="no zone data or formation tops"):
        ws_service._get_filtered_well_df(1, "Well-A", zones=["Zone-1"])
    assert len(ws_service._get_filtered_well_df(1, "Well-A")) == 2


def test_zone_filtered_lithology_rejects_missing_zone_data(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    merged = pd.DataFrame(
        {"DEPTH": [1000.0, 1001.0], "NPHI": [0.2, 0.25], "RHOB": [2.4, 2.5]}
    )
    monkeypatch.setattr(
        ws_service, "_get_merged_dataframe_sync", lambda *args, **kwargs: merged
    )

    response = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology",
        json={"zones": ["Zone-1"]},
    )
    assert response.status_code == 400
    assert "no zone data or formation tops" in response.json()["detail"]


def test_workspace_zone_filter_uses_matching_rows(monkeypatch):
    merged = pd.DataFrame({"DEPTH": [1000.0, 1001.0], "ZONES": ["Zone-1", "Zone-2"]})
    monkeypatch.setattr(
        ws_service, "_get_merged_dataframe_sync", lambda *args, **kwargs: merged
    )

    filtered = ws_service._get_filtered_well_df(1, "Well-A", zones=["Zone-1"])
    assert filtered["DEPTH"].tolist() == [1000.0]
    with pytest.raises(ValueError, match="No data rows remain"):
        ws_service._get_filtered_well_df(1, "Well-A", zones=["Missing"])


def test_batch_preview_is_isolated_and_apply_checks_staleness(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation"
    payload = {
        "well_names": ["Well-A", "Well-Empty"],
        "zones": ["Zone-1", "Missing"],
        "lithology": {"model": "ssc"},
        "porosity": {"method": "den", "lithology_model": "ssc"},
        "permeability": {"method": "choo"},
        "saturation": {"method": "archie"},
        "parameter_snapshots": {
            "litho-poro": {"litho": {"model": "ssc"}, "ts": {}},
            "perm": {"method": "choo"},
            "saturation": {"method": "archie"},
        },
    }
    original = db_service._get_merged_dataframe_sync(pid, "Well-A")
    response = client.post(f"{endpoint}/preview", json=payload)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    ready = next(
        r for r in results if r["well_name"] == "Well-A" and r["zone"] == "Zone-1"
    )
    assert ready["status"] == "ready", ready
    assert ready["rows"] == 5
    assert ready["metrics"]["PHIT"]["valid"] > 0
    assert all(r["status"] == "error" for r in results if r is not ready)
    profile = client.get(
        f"{endpoint}/{response.json()['preview_id']}/profile",
        params={"well_name": "Well-A", "zone": "Zone-1"},
    ).json()["profiles"]["PHIT"]
    pd.testing.assert_frame_equal(
        original, db_service._get_merged_dataframe_sync(pid, "Well-A")
    )
    apply_body = {
        "preview_id": response.json()["preview_id"],
        "fingerprints": {"Well-A": {"Zone-1": "invalid"}},
    }
    stale = client.post(f"{endpoint}/apply", json=apply_body)
    assert stale.status_code == 200, stale.text
    assert stale.json()["results"][0]["status"] == "error"
    apply_body["fingerprints"]["Well-A"]["Zone-1"] = ready["fingerprint"]
    monkeypatch.setattr(
        ws_service, "_batch_evaluate_zone",
        lambda *args: pytest.fail("Apply must not calculate again"),
    )
    applied = client.post(f"{endpoint}/apply", json=apply_body)
    assert applied.status_code == 200, applied.text
    assert applied.json()["results"][0]["status"] == "saved", applied.text
    repeated = client.post(f"{endpoint}/apply", json=apply_body).json()["results"][0]
    assert repeated["status"] == "error"
    assert "already saved" in repeated["error"]
    db_service.invalidate_well_cache()
    saved = db_service._get_merged_dataframe_sync(pid, "Well-A")
    assert saved.loc[saved["ZONES"] == "Zone-1", "PHIT"].notna().any()
    expected = pd.DataFrame(profile).set_index("DEPTH")["after"]
    actual = saved.set_index("DEPTH").loc[expected.index, "PHIT"]
    np.testing.assert_allclose(actual, expected, equal_nan=True)
    assert saved.loc[saved["ZONES"] == "Zone-2", "PHIT"].isna().all()
    assert saved.loc[saved["ZONES"] == "Zone-1", "PHIT_SH"].notna().any()
    assert saved.loc[saved["ZONES"] == "Zone-1", "SWT_ARCHIE"].notna().any()
    assert saved.loc[saved["ZONES"] == "Zone-1", "RW"].notna().any()
    config = client.get(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/bundle"
    ).json()["config"]
    assert config["module_params"]["perm"]["zone:Zone-1"]["params"]["method"] == "choo"


def test_batch_apply_multiple_zones_and_uses_new_porosity(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation"
    prior = pd.DataFrame(
        {
            "DEPTH": np.linspace(1000, 1009, 10),
            "PHIT": [0.01] * 10,
            "PERM": [0.01] * 10,
        }
    )
    ws_service._save_curves_to_well(pid, "Well-A", prior)
    payload = {
        "well_names": ["Well-A"],
        "zones": ["Zone-1", "Zone-2"],
        "lithology": {"model": "ssc"},
        "porosity": {"method": "den", "lithology_model": "ssc"},
        "permeability": {"method": "choo"},
        "saturation": {"method": "archie"},
    }
    response = client.post(f"{endpoint}/preview", json=payload)
    assert response.status_code == 200, response.text
    results = response.json()["results"]
    assert all(r["status"] == "ready" for r in results), results
    assert all(r["metrics"]["PHIT"]["after"] > 0.01 for r in results)
    assert all(r["metrics"]["PERM"]["after"] != pytest.approx(0.01) for r in results)
    ws_service._save_curves_to_well(
        pid,
        "Well-A",
        pd.DataFrame(
            {"DEPTH": prior["DEPTH"], "PHIT": [0.55] * 10, "PHIT_SH": [0.55] * 10}
        ),
    )
    changed_response = client.post(f"{endpoint}/preview", json=payload).json()
    changed = changed_response["results"]
    for before, after in zip(results, changed, strict=False):
        assert before["metrics"]["PERM"]["after"] == pytest.approx(
            after["metrics"]["PERM"]["after"]
        )
        assert before["metrics"]["SWT"]["after_all"] == pytest.approx(
            after["metrics"]["SWT"]["after_all"]
        )
    results = changed
    fingerprints = {"Well-A": {r["zone"]: r["fingerprint"] for r in results}}
    applied = client.post(
        f"{endpoint}/apply", json={"preview_id": changed_response["preview_id"], "fingerprints": fingerprints}
    )
    assert applied.status_code == 200, applied.text
    assert [r["status"] for r in applied.json()["results"]] == ["saved", "saved"], (
        applied.text
    )


def test_batch_preview_supports_thomas_stieber_and_rejects_mismatched_correction(
    client_and_project,
):
    client, pid = client_and_project
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation/preview"
    payload = {
        "well_names": ["Well-A"],
        "zones": ["Zone-1"],
        "lithology": {"model": "ssc"},
        "porosity": {"lithology_model": "ssc"},
        "permeability": {"use_thomas_stieber": True},
        "saturation": {"use_thomas_stieber": True},
        "thomas_stieber": {"use_hc_corrected": False},
    }
    response = client.post(endpoint, json=payload)
    assert response.status_code == 200, response.text
    assert response.json()["results"][0]["status"] == "ready", response.text
    payload["lithology"]["use_hc_corrected"] = True
    response = client.post(endpoint, json=payload)
    assert response.json()["results"][0]["status"] == "error"
    assert "settings do not match" in response.json()["results"][0]["error"]


def test_batch_preview_rejects_blank_zone(client_and_project):
    client, pid = client_and_project
    response = client.post(
        f"/quick_pp/workspace/projects/{pid}/batch_evaluation/preview",
        json={
            "well_names": ["Well-A"],
            "zones": [" "],
            "lithology": {},
            "porosity": {},
            "permeability": {},
            "saturation": {},
        },
    )
    assert response.status_code == 422


def test_batch_full_profile_is_loaded_on_demand_and_preview_expires(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    depths = np.arange(301, dtype=float) + 1000
    merged = pd.DataFrame({
        "DEPTH": depths, "ZONES": ["Zone-1"] * 300 + ["Zone-2"],
        "GR": np.full(301, 60.0), "NPHI": np.full(301, 0.2),
        "RHOB": np.full(301, 2.4), "RT": np.full(301, 10.0),
        "PHIT": np.full(301, 0.2),
    })
    merged.loc[100, "PHIT"] = np.nan
    monkeypatch.setattr(ws_service, "_get_merged_dataframe_sync", lambda *a, **k: merged.copy())
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation"
    payload = {
        "well_names": ["Well-A"], "zones": ["Zone-1"],
        "lithology": {}, "porosity": {"lithology_model": "ssc"},
        "permeability": {}, "saturation": {},
    }
    response = client.post(f"{endpoint}/preview", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["results"][0]["status"] == "ready", body
    assert "profiles" not in body["results"][0]
    profile_url = f"{endpoint}/{body['preview_id']}/profile"
    profile = client.get(profile_url, params={"well_name": "Well-A", "zone": "Zone-1"})
    assert profile.status_code == 200, profile.text
    for points in profile.json()["profiles"].values():
        assert [p["DEPTH"] for p in points] == depths[:300].tolist()
    assert profile.json()["profiles"]["PHIT"][100]["before"] is None
    assert client.get(profile_url, params={"well_name": "Well-A", "zone": "Zone-2"}).status_code == 404
    assert client.get(profile_url.replace(f"projects/{pid}", f"projects/{pid + 1}"),
                      params={"well_name": "Well-A", "zone": "Zone-1"}).status_code == 404
    unknown = client.post(
        f"{endpoint}/apply",
        json={"preview_id": body["preview_id"], "fingerprints": {"Well-A": {"Zone-2": "x"}}},
    ).json()["results"][0]
    assert unknown["status"] == "error"
    assert "not in this preview" in unknown["error"]
    ws_service._batch_previews[body["preview_id"]].expires_at = 0
    assert client.get(profile_url, params={"well_name": "Well-A", "zone": "Zone-1"}).status_code == 404
    assert client.post(f"{endpoint}/apply", json={"preview_id": body["preview_id"], "fingerprints": {}}).status_code == 404


def test_batch_preview_release_and_storage_limit(client_and_project, monkeypatch):
    client, pid = client_and_project
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation"
    payload = {
        "well_names": ["Well-A"], "zones": ["Zone-1"],
        "lithology": {}, "porosity": {"lithology_model": "ssc"},
        "permeability": {}, "saturation": {},
    }
    monkeypatch.setattr(ws_service, "_BATCH_PREVIEW_MAX_COUNT", 1)
    first = client.post(f"{endpoint}/preview", json=payload)
    assert first.status_code == 200, first.text
    preview_id = first.json()["preview_id"]
    assert client.post(f"{endpoint}/preview", json=payload).status_code == 503
    assert client.delete(f"{endpoint}/{preview_id}").status_code == 200
    assert client.get(
        f"{endpoint}/{preview_id}/profile",
        params={"well_name": "Well-A", "zone": "Zone-1"},
    ).status_code == 404
    assert client.post(f"{endpoint}/preview", json=payload).status_code == 200


def test_batch_apply_detects_real_input_changes_and_preserves_other_zones(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    depths = np.linspace(1000, 1009, 10)
    ws_service._save_curves_to_well(
        pid, "Well-A", pd.DataFrame({"DEPTH": depths, "PHIT": [0.17] * 10})
    )
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation"
    payload = {
        "well_names": ["Well-A"],
        "zones": ["Zone-1"],
        "lithology": {},
        "porosity": {"lithology_model": "ssc"},
        "permeability": {},
        "saturation": {},
    }
    preview_response = client.post(f"{endpoint}/preview", json=payload).json()
    preview = preview_response["results"][0]
    ws_service._save_curves_to_well(
        pid, "Well-A", pd.DataFrame({"DEPTH": [1000.0], "RHOB": [2.35]})
    )
    apply = {
        "preview_id": preview_response["preview_id"],
        "fingerprints": {"Well-A": {"Zone-1": preview["fingerprint"]}},
    }
    stale = client.post(f"{endpoint}/apply", json=apply).json()["results"][0]
    assert stale["status"] == "error"
    assert "preview again" in stale["error"]
    fresh_response = client.post(f"{endpoint}/preview", json=payload).json()
    fresh = fresh_response["results"][0]
    apply["preview_id"] = fresh_response["preview_id"]
    apply["fingerprints"]["Well-A"]["Zone-1"] = fresh["fingerprint"]
    assert (
        client.post(f"{endpoint}/apply", json=apply).json()["results"][0]["status"]
        == "saved"
    )
    db_service.invalidate_well_cache()
    saved = db_service._get_merged_dataframe_sync(pid, "Well-A")
    assert saved.loc[saved["ZONES"] == "Zone-2", "PHIT"].tolist() == pytest.approx(
        [0.17] * 5
    )


def test_batch_apply_rolls_back_curves_if_parameter_save_fails(
    client_and_project, monkeypatch
):
    from contextlib import contextmanager

    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    original_get_db = ws_service.get_db

    @contextmanager
    def transactional_db():
        with original_get_db() as session:
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise

    monkeypatch.setattr(ws_service, "get_db", transactional_db)
    endpoint = f"/quick_pp/workspace/projects/{pid}/batch_evaluation"
    payload = {
        "well_names": ["Well-A"],
        "zones": ["Zone-1"],
        "lithology": {},
        "porosity": {"lithology_model": "ssc"},
        "permeability": {},
        "saturation": {},
        "parameter_snapshots": {"perm": {"method": "choo"}},
    }
    preview_response = client.post(f"{endpoint}/preview", json=payload).json()
    preview = preview_response["results"][0]
    assert preview["status"] == "ready", preview

    def fail_params(*args, **kwargs):
        raise ValueError("snapshot failure")

    monkeypatch.setattr(ws_service, "_merge_module_params", fail_params)
    response = client.post(
        f"{endpoint}/apply",
        json={
            "preview_id": preview_response["preview_id"],
            "fingerprints": {"Well-A": {"Zone-1": preview["fingerprint"]}},
        },
    )
    assert response.json()["results"][0]["status"] == "error"
    db_service.invalidate_well_cache()
    saved = db_service._get_merged_dataframe_sync(pid, "Well-A")
    assert "PHIT" not in saved.columns or saved["PHIT"].isna().all()


def test_imputation_persists_new_curve_and_matrix(client_and_project, monkeypatch):
    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    endpoint = f"/quick_pp/workspace/projects/{pid}/qaqc"
    matrix = client.post(f"{endpoint}/missing_matrix", json={"well_name": "Well-A"})
    assert matrix.status_code == 200, matrix.text
    matrix_data = matrix.json()
    assert matrix_data["matrix"][matrix_data["curves"].index("CPORE")] == [1, 0] * 5
    response = client.post(
        f"{endpoint}/impute",
        json={"well_names": ["Well-A"], "curves": ["CPORE"], "method": "mean"},
    )
    assert response.status_code == 200, response.text
    assert response.json()["summaries"][0]["filled"] == 5
    db_service.invalidate_well_cache()
    merged = db_service._get_merged_dataframe_sync(pid, "Well-A")
    assert merged["CPORE"].isna().sum() == 5
    assert merged["CPORE_IMP"].notna().sum() == 10
    assert merged.loc[merged["CPORE"].isna(), "CPORE_IMP"].tolist() == pytest.approx(
        [0.172] * 5
    )


def test_imputation_persists_only_filtered_depths(client_and_project, monkeypatch):
    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    endpoint = f"/quick_pp/workspace/projects/{pid}/qaqc/impute"
    response = client.post(
        endpoint,
        json={
            "well_names": ["Well-A"],
            "curves": ["CPORE"],
            "method": "mean",
            "min_depth": 1002,
            "max_depth": 1006,
        },
    )
    assert response.status_code == 200, response.text
    assert response.json()["summaries"][0]["filled"] == 2
    db_service.invalidate_well_cache()
    merged = db_service._get_merged_dataframe_sync(pid, "Well-A")
    assert merged.loc[~merged["DEPTH"].between(1002, 1006), "CPORE_IMP"].isna().all()
    assert merged.loc[merged["DEPTH"].between(1002, 1006), "CPORE_IMP"].notna().all()


def test_imputation_replacement_preserves_full_original_backup(client_and_project, monkeypatch):
    client, pid = client_and_project
    monkeypatch.setattr(ws_service, "is_demo_mode", lambda: False)
    endpoint = f"/quick_pp/workspace/projects/{pid}/qaqc/impute"
    original = db_service._get_merged_dataframe_sync(pid, "Well-A")["CPORE"].copy()
    request = {
        "well_names": ["Well-A"], "curves": ["CPORE"], "method": "linear",
        "min_depth": 1002, "max_depth": 1006, "replace_original": True,
    }
    response = client.post(endpoint, json=request)
    assert response.status_code == 200, response.text
    assert response.json()["summaries"][0]["target_curve"] == "CPORE"
    db_service.invalidate_well_cache()
    merged = db_service._get_merged_dataframe_sync(pid, "Well-A")
    np.testing.assert_allclose(merged["CPORE_ORI"], original, equal_nan=True)
    np.testing.assert_allclose(
        merged.loc[merged["DEPTH"].between(1002, 1006), "CPORE"],
        [0.19, 0.18, 0.17, 0.16, 0.15],
    )
    np.testing.assert_allclose(
        merged.loc[~merged["DEPTH"].between(1002, 1006), "CPORE"],
        original.loc[~merged["DEPTH"].between(1002, 1006)],
        equal_nan=True,
    )
    request.update(method="mean", min_depth=1000, max_depth=1009)
    response = client.post(endpoint, json=request)
    assert response.status_code == 200, response.text
    db_service.invalidate_well_cache()
    merged = db_service._get_merged_dataframe_sync(pid, "Well-A")
    np.testing.assert_allclose(merged["CPORE_ORI"], original, equal_nan=True)


def test_merged_well_data_columnar_and_cache(client_and_project):
    client, pid = client_and_project
    # Columnar request
    r = client.get(
        f"/quick_pp/database/projects/{pid}/wells/Well-A/merged?format=columnar&min_depth=1002&max_depth=1006"
    )
    assert r.status_code == 200
    body = r.json()
    assert body["row_count"] == 5
    assert "DEPTH" in body["columns"]
    assert len(body["data"]["DEPTH"]) == 5

    # Verify cache was populated
    stats = client.get("/quick_pp/database/cache/stats").json()
    assert stats["cache_size"] == 1

    # Verify /merged/generate returns synchronously by default
    r_gen = client.post(
        f"/quick_pp/database/projects/{pid}/wells/Well-A/merged/generate"
    )
    assert r_gen.status_code == 200
    gen_body = r_gen.json()
    assert gen_body["task_id"] == "sync"
    assert gen_body["status"] == "success"
    assert len(gen_body["result"]) == 10

    # Verify /workspace/projects/{pid}/wells/Well-A/bundle single-request bundle endpoint
    r_bundle = client.get(f"/quick_pp/workspace/projects/{pid}/wells/Well-A/bundle")
    assert r_bundle.status_code == 200
    bundle = r_bundle.json()
    assert bundle["row_count"] == 10
    assert "DEPT" in bundle["curves"]
    assert "GR" in bundle["curves"]
    assert "GR" in bundle["stats"]
    assert len(bundle["tops"]) == 2


def test_end_to_end_workspace_compute_workflow(client_and_project):
    client, pid = client_and_project

    # 1. HC Correction (save=True)
    r_hc = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/hc_correction",
        json={"rho_water": 1.0, "rho_hc": 0.8, "hi_hc": 0.9, "save": True},
    )
    assert r_hc.status_code == 200
    hc_body = r_hc.json()
    assert hc_body["saved"] is True
    assert hc_body["saved_rows"] == 10
    assert len(hc_body["data"]["NPHI_HC"]) == 10

    # 2. Lithology SSC (save=True, using HC corrected logs)
    r_lith = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology",
        json={
            "model": "ssc",
            "use_hc_corrected": True,
            "dry_silt_point": [0.151, 2.74],
            "silt_line_angle": 116.0,
            "save": True,
        },
    )
    assert r_lith.status_code == 200
    lith_body = r_lith.json()
    assert lith_body["saved_rows"] == 10
    assert "VSAND" in lith_body["data"]
    assert "VCLAY" in lith_body["data"]
    assert "VSHALE" in lith_body["data"]
    for vclay, vsilt, vshale in zip(
        lith_body["data"]["VCLAY"],
        lith_body["data"]["VSILT"],
        lith_body["data"]["VSHALE"],
        strict=True,
    ):
        assert np.isclose(vshale, vclay + vsilt)

    assert lith_body["crossplot"]["dry_silt_point"][1] == 2.68
    expected_silt_nphi = 1.0 + ((2.68 - 1.0) / np.tan(np.deg2rad(116.0)))
    assert np.isclose(lith_body["crossplot"]["dry_silt_point"][0], expected_silt_nphi)

    # 3. Porosity (save=True)
    r_poro = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/porosity",
        json={"method": "neu_den", "use_hc_corrected": True, "save": True},
    )
    assert r_poro.status_code == 200
    poro_body = r_poro.json()
    assert poro_body["saved_rows"] == 10
    assert poro_body["core_metrics"]["count"] == 5
    assert poro_body["core_metrics"]["rmse"] is not None

    # 4. Permeability Choo (save=True)
    r_perm = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/permeability",
        json={"method": "choo", "save": True},
    )
    assert r_perm.status_code == 200
    perm_body = r_perm.json()
    assert perm_body["saved_rows"] == 10
    assert perm_body["core_metrics"]["count"] == 5

    # 5. Saturation (single-pass Archie and alternative models, save=True)
    r_sat = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/saturation",
        json={"method": "waxman_smits", "water_salinity": 35000.0, "save": True},
    )
    assert r_sat.status_code == 200
    sat_body = r_sat.json()
    assert sat_body["saved_rows"] == 10
    assert len(sat_body["data"]["SWT"]) == 10
    assert len(sat_body["data"]["SWT_ARCHIE"]) == 10
    assert len(sat_body["data"]["SWT_WAXMAN"]) == 10

    r_sat_crossplot = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/saturation",
        json={
            "method": "waxman_smits",
            "qv_method": "crossplot",
            "cec_clay": 0.2,
            "rho_dry_clay": 2.71,
            "water_salinity": 35000.0,
            "save": False,
        },
    )
    assert r_sat_crossplot.status_code == 200
    assert len(r_sat_crossplot.json()["data"]["SWT_WAXMAN"]) == 10

    # 6. Rock Typing across project (including empty well) & Perm Transform
    r_rt_winland = client.post(
        f"/quick_pp/workspace/projects/{pid}/rock_typing/compute",
        json={"method": "winland", "cutoffs": [0.5, 2.0, 5.0], "save": False},
    )
    assert r_rt_winland.status_code == 200
    assert r_rt_winland.json()["method"] == "winland"
    assert "r35" in r_rt_winland.json()["data"]

    r_rt_lucia = client.post(
        f"/quick_pp/workspace/projects/{pid}/rock_typing/compute",
        json={"method": "lucia", "cutoffs": [1.5, 2.5, 3.5], "save": False},
    )
    assert r_rt_lucia.status_code == 200
    assert r_rt_lucia.json()["method"] == "lucia"
    assert r_rt_lucia.json()["higher_is_better"] is False
    assert "rfn" in r_rt_lucia.json()["data"]

    r_rt_gmm = client.post(
        f"/quick_pp/workspace/projects/{pid}/rock_typing/compute",
        json={
            "method": "fzi",
            "auto_find_cutoffs": True,
            "cutoff_method": "gmm",
            "auto_k": True,
            "lucia_vuggy_override": True,
            "phi_sec_cutoff": 0.01,
            "save": False,
        },
    )
    assert r_rt_gmm.status_code == 200
    assert "diagnostics" in r_rt_gmm.json()
    assert r_rt_gmm.json()["diagnostics"]["carbonate"]["has_sonic_data"] is True

    r_rt_lorenz = client.post(
        f"/quick_pp/workspace/projects/{pid}/rock_typing/compute",
        json={
            "method": "fzi",
            "auto_find_cutoffs": True,
            "cutoff_method": "lorenz",
            "auto_k": True,
            "save": False,
        },
    )
    assert r_rt_lorenz.status_code == 200
    assert "lorenz" in r_rt_lorenz.json()["diagnostics"]

    r_rt = client.post(
        f"/quick_pp/workspace/projects/{pid}/rock_typing/compute",
        json={"method": "fzi", "cutoffs": [0.5, 1.5, 3.0], "save": True},
    )
    assert r_rt.status_code == 200
    assert r_rt.json()["saved_wells"] == 1

    r_pt = client.post(
        f"/quick_pp/workspace/projects/{pid}/perm_transform/compute",
        json={"save": True},
    )
    assert r_pt.status_code == 200
    assert r_pt.json()["saved_wells"] == 1

    # 7. Reservoir Summary + Upscaling + Uncertainty
    r_res = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/reservoir_summary",
        json={
            "cut_offs": {"VSHALE": 0.6, "PHIT": 0.01, "SWT": 0.99},
            "include_upscaling": True,
            "cell_size": 2.0,
            "include_uncertainty": True,
            "n_samples": 40,
        },
    )
    assert r_res.status_code == 200
    res_body = r_res.json()
    assert len(res_body["summary"]) >= 1
    assert "upscaling" in res_body
    assert res_body["upscaling"]["n_cells"] >= 1
    assert "uncertainty" in res_body
    assert len(res_body["uncertainty"]["zone_summary"]) >= 1
    z0 = res_body["uncertainty"]["zone_summary"][0]
    assert z0.get("ZONE") is not None
    assert z0.get("STOIIP_MMBBL_P50") is not None

    # 7b. Multi-well Volumetric Assessment (Area mode & GRV mode)
    r_vol_area = client.post(
        f"/quick_pp/workspace/projects/{pid}/multi_well_volumetrics",
        json={"mode": "area", "n_samples": 200, "seed": 42, "save_to_project": True},
    )
    assert r_vol_area.status_code == 200
    vol_area_body = r_vol_area.json()
    assert vol_area_body["mode"] == "area"
    assert len(vol_area_body["well_summary"]) >= 1
    assert len(vol_area_body["zonal_summary"]) >= 1
    assert "field_totals" in vol_area_body
    assert vol_area_body["field_totals"]["stoiip"]["p50"] > 0
    assert len(vol_area_body["field_totals"]["exceedance_curve"]) == 100
    assert len(vol_area_body["zonal_summary"][0]["tornado"]) >= 1

    r_vol_grv = client.post(
        f"/quick_pp/workspace/projects/{pid}/multi_well_volumetrics",
        json={"mode": "grv", "n_samples": 200, "seed": 42},
    )
    assert r_vol_grv.status_code == 200
    assert r_vol_grv.json()["mode"] == "grv"
    assert r_vol_grv.json()["field_totals"]["stoiip"]["p50"] > 0

    # 8. Saturation Height Function (Leverett-J / Skelt-Harrison / Cuddy)
    r_shf = client.post(
        f"/quick_pp/workspace/projects/{pid}/shf/compute",
        json={
            "fwl": 1020.0,
            "model": "leverett_j",
            "leverett_a": 0.2,
            "leverett_b": -0.5,
            "save": True,
        },
    )
    assert r_shf.status_code == 200
    shf_body = r_shf.json()
    assert shf_body["saved_wells"] == 1
    assert len(shf_body["data"]["shf"]) == 10
    assert "scal_data" in shf_body
    assert len(shf_body["scal_data"]["pc"]) > 0

    for shf_model in ("skelt_harrison", "cuddy"):
        r_m = client.post(
            f"/quick_pp/workspace/projects/{pid}/shf/compute",
            json={
                "fwl": 1020.0,
                "fwl_by_zone": {"Unknown": 1015.0},
                "model": shf_model,
                "save": False,
            },
        )
        assert r_m.status_code == 200
        m_body = r_m.json()
        assert m_body["model"] == shf_model
        assert m_body["fwl_by_zone"].get("Unknown") == 1015.0
        assert len(m_body["fits"]) >= 1
        assert len(m_body["data"]["shf"]) == 10

    # 9. Columnar PUT well data
    r_put = client.put(
        f"/quick_pp/database/projects/{pid}/wells/Well-A/data",
        json={"data": {"DEPTH": [1000.0, 1001.0], "CUSTOM_CURVE": [1.5, 2.5]}},
    )
    assert r_put.status_code == 200
    assert r_put.json()["rows"] == 2


def test_dynamic_hc_correction_returns_depth_aligned_fluid_curves(client_and_project):
    client, pid = client_and_project
    response = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/hc_correction",
        json={
            "mode": "dynamic",
            "temp_grad": 35.0,
            "pres_grad": 12.0,
            "gas_gravity": 0.7,
            "oil_api": 32.0,
            "gor": 250.0,
            "salinity_ppm": 45000.0,
        },
    )

    assert response.status_code == 200
    curves = response.json()["data"]
    for curve in (
        "TEMP_FL",
        "PRES_FL",
        "RHO_W",
        "RHO_O",
        "RHO_G",
        "HI_W",
        "HI_O",
        "HI_G",
        "RHO_H",
        "HI_H",
        "Z_FACTOR",
    ):
        assert len(curves[curve]) == 10
    np.testing.assert_allclose(curves["RHO_H"], curves["RHO_O"])
    np.testing.assert_allclose(curves["HI_H"], curves["HI_O"])
    assert response.json()["stats"]["RHO_H"]["count"] == 10


def test_dynamic_hc_correction_rejects_measured_depth_without_tvd(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    well_df = ws_service._get_filtered_well_df(
        pid, "Well-A", min_depth=None, max_depth=None, zones=None
    )
    well_df = well_df.drop(
        columns=[
            "DEPT_TVD",
            "DEPTH_TVD",
            "TVDSS",
            "DEPT_TVDSS",
            "DEPTH_TVDSS",
            "TVD_SS",
        ],
        errors="ignore",
    )
    well_df["TVD"] = np.nan
    monkeypatch.setattr(
        ws_service, "_get_filtered_well_df", lambda *args, **kwargs: well_df
    )

    response = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/hc_correction",
        json={"mode": "dynamic"},
    )

    assert response.status_code == 400
    assert "TVD or TVDSS curve" in response.json()["detail"]
    assert "Measured depth" in response.json()["detail"]


def test_dynamic_hc_correction_accepts_tvdss(client_and_project, monkeypatch):
    client, pid = client_and_project
    well_df = ws_service._get_filtered_well_df(
        pid, "Well-A", min_depth=None, max_depth=None, zones=None
    )
    well_df = well_df.drop(columns=["TVD"])
    well_df["TVDSS"] = np.linspace(900.0, 909.0, len(well_df))
    monkeypatch.setattr(
        ws_service, "_get_filtered_well_df", lambda *args, **kwargs: well_df
    )

    response = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/hc_correction",
        json={"mode": "dynamic"},
    )

    assert response.status_code == 200
    assert len(response.json()["data"]["RHO_H"]) == len(well_df)


@pytest.mark.parametrize("method", ["crossplot", "invaded_zone_standard"])
def test_hc_correction_defaults_to_solid_target(client_and_project, method):
    client, pid = client_and_project
    endpoint = f"/quick_pp/workspace/projects/{pid}/wells/Well-A/hc_correction"
    params = {
        "correction_method": method,
        "mud_filtrate_density": 1.0,
        "mud_filtrate_salinity": 0.0,
        "rho_hc": 0.1,
        "hi_hc": 0.22,
        "gr_clean": 30.0,
        "gr_clay": 110.0,
        "flag_mode": "vsh_gr",
        "vsh_tol": 0.0,
    }
    results = []
    for mode in (None, "solid", "bulk"):
        payload = dict(params)
        if mode is not None:
            payload["target_mode"] = mode
        response = client.post(endpoint, json=payload)
        assert response.status_code == 200, response.text
        results.append(response.json()["data"])

    default, solid, bulk = results
    for curve in ("NPHI_HC", "RHOB_HC", "HC_FLAG"):
        np.testing.assert_allclose(default[curve], solid[curve])
    assert not np.allclose(default["RHOB_HC"], bulk["RHOB_HC"])


def test_standard_invaded_zone_correction_returns_saturation_curve(client_and_project):
    client, pid = client_and_project
    endpoint = f"/quick_pp/workspace/projects/{pid}/wells/Well-A/hc_correction"
    missing_metadata = client.post(
        endpoint,
        json={"correction_method": "invaded_zone_standard"},
    )
    assert missing_metadata.status_code == 400
    assert "mud-filtrate density" in missing_metadata.json()["detail"]

    response = client.post(
        endpoint,
        json={
            "mode": "dynamic",
            "correction_method": "invaded_zone_standard",
            "mud_filtrate_density": 1.0,
            "mud_filtrate_salinity": 200000.0,
            "lithology_coeff": 1.0,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["hc_summary"]["correction_method"] == "invaded_zone_standard"
    assert len(body["data"]["SHC_INV"]) == 10
    assert body["stats"]["SHC_INV"]["count"] > 0


def test_saturation_model_comparison_curves_and_archie_default(client_and_project):
    client, pid = client_and_project
    endpoint = f"/quick_pp/workspace/projects/{pid}/wells/Well-A/saturation"
    fallback = {
        "depths": np.linspace(1000.0, 1009.0, 10).tolist(),
        "phit": np.linspace(0.12, 0.25, 10).tolist(),
        "phie": np.full(10, 0.01).tolist(),
        "vclay": np.linspace(0.1, 0.4, 10).tolist(),
        "save": False,
    }

    default_response = client.post(endpoint, json=fallback)
    assert default_response.status_code == 200, default_response.text
    default_body = default_response.json()
    assert default_body["method"] == "archie"
    assert default_body["data"]["SWT"] == default_body["data"]["SWT_ARCHIE"]
    expected_archie = np.sqrt(
        np.asarray(default_body["data"]["RW"], dtype=float)
        / (
            np.asarray(default_body["data"]["RT"], dtype=float)
            * np.asarray(fallback["phit"], dtype=float) ** 2
        )
    ).clip(0.0, 1.0)
    np.testing.assert_allclose(
        default_body["data"]["SWT_ARCHIE"], expected_archie, equal_nan=True
    )

    model_curves = {
        "waxman_smits": "SWT_WAXMAN",
        "simandoux": "SWT_SIMANDOUX",
        "indonesian": "SWT_INDONESIAN",
        "dual_water": "SWT_DUAL_WATER",
    }
    for method, curve in model_curves.items():
        response = client.post(endpoint, json={**fallback, "method": method})
        assert response.status_code == 200
        body = response.json()
        assert body["method"] == method
        assert len(body["data"][curve]) == 10
        assert any(value is not None for value in body["data"][curve])
        assert body["data"]["SWT"] == body["data"][curve]
        assert len(body["data"]["SWT_ARCHIE"]) == 10


def test_petrophysical_mcp_server_and_agent_assistant(client_and_project):
    client, pid = client_and_project

    # 1. Provider catalog & MCP tools listing
    r_prov = client.get("/quick_pp/assistant/providers")
    assert r_prov.status_code == 200
    prov_body = r_prov.json()
    provider_ids = [p["id"] for p in prov_body["providers"]]
    assert "native_mcp" in provider_ids
    assert "openrouter" in provider_ids
    assert "ollama" in provider_ids
    tool_names = [t["name"] for t in prov_body["mcp_tools"]]
    assert "inspect_well" in tool_names
    assert "run_well_qaqc" in tool_names
    assert "run_multi_mineral" in tool_names
    assert "run_shf_evaluation" in tool_names
    assert "evaluate_with_conformal_bounds" in tool_names
    assert "control_workbench_ui" in tool_names

    # 2. Standard MCP JSON-RPC 2.0 protocol (initialize + tools/list + tools/call)
    r_init = client.post(
        "/quick_pp/mcp",
        json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
    )
    assert r_init.status_code == 200
    assert r_init.json()["result"]["serverInfo"]["name"] == "quick_pp-petrophysics-mcp"

    r_list = client.post(
        "/quick_pp/mcp",
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    assert r_list.status_code == 200
    assert len(r_list.json()["result"]["tools"]) >= 6

    r_call = client.post(
        "/quick_pp/mcp",
        json={
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "inspect_well",
                "arguments": {"project_id": pid, "well_name": "Well-A"},
            },
        },
    )
    assert r_call.status_code == 200
    structured = r_call.json()["result"]["structuredContent"]
    assert structured["well_name"] == "Well-A"
    assert structured["row_count"] == 10
    assert len(structured["zonal_breakdown"]) == 2

    # 3. Agentic Assistant end-to-end evaluation (QAQC + Multi-Mineral + SHF + On-the-fly Conformal Bounds)
    r_chat = client.post(
        "/quick_pp/assistant/chat",
        json={
            "message": "Run a full evaluation (qaqc, multi-mineral, shf, and conformal prediction bounds) and explain.",
            "project_id": pid,
            "well_name": "Well-A",
            "alpha": 0.20,
            "provider": "native_mcp",
        },
    )
    assert r_chat.status_code == 200
    chat_body = r_chat.json()
    executed_names = [tc["name"] for tc in chat_body["tool_calls"]]
    assert "inspect_well" in executed_names
    assert "run_well_qaqc" in executed_names
    assert "run_multi_mineral" in executed_names
    assert "run_shf_evaluation" in executed_names
    assert "evaluate_with_conformal_bounds" in executed_names

    # Verify live computed curves include P10/P90 conformal bounds and QAQC / SHF curves
    curves = chat_body["computed_curves"]
    for expected_col in (
        "NPHI_HC",
        "RHOB_HC",
        "PHIT",
        "PHIT_P10",
        "PHIT_P90",
        "SWT",
        "SW_P10",
        "SW_P90",
        "PERM",
        "PERM_P10",
        "PERM_P90",
        "SHF",
    ):
        assert expected_col in curves
        assert len(curves[expected_col]) == 10

    # Verify P10 <= Prediction <= P90 monotonicity across all samples
    for p10, pred, p90 in zip(
        curves["PHIT_P10"], curves["PHIT"], curves["PHIT_P90"], strict=False
    ):
        assert p10 <= pred + 1e-6
        assert pred <= p90 + 1e-6

    assert "Conformal Prediction Bounds" in chat_body["reply"]


def test_badhole_flagging_and_litho_poro_badhole_treatments(client_and_project):
    """Verify /badhole_flagging endpoint and all 4 badhole_treatment modes in /lithology and /porosity."""
    client, pid = client_and_project

    # Inject CALI, DRHO, and a washout spike at depth index 2 (1002.0 m) on Well-A
    cali_vals = [8.5, 8.5, 13.0, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5, 8.5]
    drho_vals = [0.01, 0.01, 0.25, 0.02, 0.01, 0.01, 0.02, 0.01, 0.01, 0.01]
    rhob_vals = [2.25, 2.28, 1.65, 2.34, 2.38, 2.41, 2.45, 2.48, 2.52, 2.55]
    r_put = client.put(
        f"/quick_pp/database/projects/{pid}/wells/Well-A/data",
        json={
            "data": {
                "DEPTH": list(np.linspace(1000.0, 1009.0, 10)),
                "CALI": cali_vals,
                "DRHO": drho_vals,
                "RHOB": rhob_vals,
            }
        },
    )
    assert r_put.status_code == 200

    # 1. Run /badhole_flagging
    r_bh = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/badhole_flagging",
        json={
            "use_cali": True,
            "cali_threshold": 2.5,
            "auto_bitsize": False,
            "bitsize_default": 8.5,
            "use_drho": True,
            "drho_threshold": 0.15,
            "use_rhob_nphi": True,
            "rhob_min": 1.95,
            "rhob_max": 2.95,
            "nphi_max": 0.45,
            "exclude_coal": True,
            "use_rugosity": False,
            "save": True,
        },
    )
    assert r_bh.status_code == 200
    bh_body = r_bh.json()
    assert bh_body["badhole_summary"]["flagged_count"] == 1
    assert bh_body["data"]["BADHOLE"][2] == 1.0
    assert bh_body["data"]["BADHOLE"][0] == 0.0
    assert np.isclose(bh_body["data"]["DCALI"][2], 4.5)

    # 2. Treatment = 'ignore': index 2 should be None (NaN) in lithology and porosity outputs
    r_lith_ign = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology",
        json={"model": "ssc", "badhole_treatment": "ignore", "save": False},
    )
    assert r_lith_ign.status_code == 200
    lith_ign = r_lith_ign.json()["data"]
    assert lith_ign["VCLAY"][2] is None
    assert lith_ign["VSAND"][2] is None
    assert lith_ign["VSHALE"][2] is None
    assert lith_ign["VCLAY"][1] is not None

    r_poro_ign = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/porosity",
        json={"method": "neu_den", "badhole_treatment": "ignore", "save": False},
    )
    assert r_poro_ign.status_code == 200
    poro_ign = r_poro_ign.json()
    assert poro_ign["data"]["PHIT"][2] is None
    assert poro_ign["data"]["PHIE"][2] is None
    # Index 2 had a core sample (CPORE=0.19), so excluding badhole leaves 4 core comparison points
    assert poro_ign["core_metrics"]["count"] == 4

    # 3. Treatment = 'interpolate': index 2 is linearly interpolated between index 1 and index 3
    r_lith_interp = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology",
        json={"model": "ssc", "badhole_treatment": "interpolate", "save": False},
    )
    assert r_lith_interp.status_code == 200
    lith_interp = r_lith_interp.json()["data"]
    assert np.isclose(
        lith_interp["VCLAY"][2],
        0.5 * (lith_interp["VCLAY"][1] + lith_interp["VCLAY"][3]),
        atol=1e-4,
    )
    assert np.isclose(
        lith_interp["VSHALE"][2],
        lith_interp["VCLAY"][2] + lith_interp["VSILT"][2],
        atol=1e-6,
    )

    r_poro_interp = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/porosity",
        json={"method": "neu_den", "badhole_treatment": "interpolate", "save": False},
    )
    assert r_poro_interp.status_code == 200
    poro_interp = r_poro_interp.json()["data"]
    assert np.isclose(
        poro_interp["PHIT"][2],
        0.5 * (poro_interp["PHIT"][1] + poro_interp["PHIT"][3]),
        atol=1e-4,
    )

    # 4. Treatment = 'vsh_gr': index 2 uses GR-based VCLAY, VSILT=0, and reconstructed porosity
    r_lith_gr = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology",
        json={
            "model": "ssc",
            "gr_clean": 30.0,
            "gr_clay": 110.0,
            "badhole_treatment": "vsh_gr",
            "save": True,
        },
    )
    assert r_lith_gr.status_code == 200
    lith_gr = r_lith_gr.json()["data"]
    assert lith_gr["VSILT"][2] == 0.0
    assert lith_gr["VCLAY"][2] is not None and 0.0 <= lith_gr["VCLAY"][2] <= 1.0
    assert np.isclose(lith_gr["VSHALE"][2], lith_gr["VCLAY"][2])

    r_lith_vsh_gr = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/lithology",
        json={
            "model": "vsh_gr",
            "gr_clean": 30.0,
            "gr_clay": 110.0,
            "save": False,
        },
    )
    assert r_lith_vsh_gr.status_code == 200
    lith_vsh_gr = r_lith_vsh_gr.json()["data"]
    assert np.allclose(lith_vsh_gr["VSHALE"], lith_vsh_gr["VCLAY"])

    r_poro_gr = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/porosity",
        json={
            "method": "neu_den",
            "gr_clean": 30.0,
            "gr_clay": 110.0,
            "badhole_treatment": "vsh_gr",
            "save": True,
        },
    )
    assert r_poro_gr.status_code == 200
    poro_gr = r_poro_gr.json()["data"]
    # Without vsh_gr reconstruction, RHOB=1.65 at index 2 would yield anomalously high porosity (>0.40)
    # With vsh_gr reconstruction via den_correction, PHIT[2] is realistic (~0.10 - 0.30)
    assert poro_gr["PHIT"][2] is not None
    assert 0.05 < poro_gr["PHIT"][2] < 0.35


def test_workspace_stick_plot(client_and_project):
    client, pid = client_and_project
    res = client.post(
        f"/quick_pp/workspace/projects/{pid}/stick_plot",
        json={"max_points_per_well": 500},
    )
    assert res.status_code == 200
    payload = res.json()
    assert payload["project_id"] == pid
    assert "wells" in payload
    assert len(payload["wells"]) == 1
    w = payload["wells"][0]
    assert w["well_name"] == "Well-A"
    assert "tvd" in w and len(w["tvd"]) == 10
    assert "bvo" in w and len(w["bvo"]) == 10
    assert "top_reservoir_tvd" in w
    assert "cum_hcpv" in w
    assert len(w["tops"]) == 2


def test_stick_plot_batches_ancillary_depths_using_full_well(
    client_and_project, monkeypatch
):
    client, pid = client_and_project
    get_ancillary = Project.get_well_ancillary_data
    get_merged = ws_service._get_merged_dataframe_sync
    resolve_tvd = ws_service._resolve_well_tvd_array
    resolved_depths = []

    def nonlinear_tvd(project_id, well_name):
        df = get_merged(project_id, well_name)
        if not df.empty:
            tvd = 1000 + (df["DEPTH"] - 1000) ** 2 * 0.1
            df["TVD"] = tvd
            df["TVDSS"] = tvd
        return df

    def ancillary_with_contacts_and_perfs(project, well_name):
        anc = get_ancillary(project, well_name)
        anc["fluid_contacts"] = [{"name": "OWC", "depth": 1003.5}]
        anc["perforations"] = [
            {"name": "P1", "top": 1008.0, "bottom": 1007.0},
            {"name": "P2", "top": 999.0},
        ]
        return anc

    def track_resolutions(md_values, well_df, surveys_df, df_elev=0.0):
        resolved_depths.append(md_values.copy())
        return resolve_tvd(md_values, well_df, surveys_df, df_elev)

    monkeypatch.setattr(
        Project, "get_well_ancillary_data", ancillary_with_contacts_and_perfs
    )
    monkeypatch.setattr(ws_service, "_get_merged_dataframe_sync", nonlinear_tvd)
    monkeypatch.setattr(ws_service, "_resolve_well_tvd_array", track_resolutions)

    res = client.post(
        f"/quick_pp/workspace/projects/{pid}/stick_plot",
        json={"min_depth": 1003, "max_depth": 1006},
    )
    assert res.status_code == 200, res.text
    well = res.json()["wells"][0]
    assert well["contacts"] == [
        {"name": "OWC", "type": "OWC", "depth": 1003.5, "tvd": 1001.25}
    ]
    assert well["tops"] == [
        {"name": "Zone-1", "depth": 1000.0, "tvd": 1000.0},
        {"name": "Zone-2", "depth": 1005.0, "tvd": 1002.5},
    ]
    assert well["perforations"] == [
        {
            "name": "P2",
            "top": 999.0,
            "bottom": 999.0,
            "top_tvd": 999.5,
            "bottom_tvd": 999.5,
            "description": "",
        },
        {
            "name": "P1",
            "top": 1007.0,
            "bottom": 1008.0,
            "top_tvd": 1004.9,
            "bottom_tvd": 1006.4,
            "description": "",
        },
    ]
    assert well["fluid_flag"] == [-1, 0, 0, 0]
    assert len(resolved_depths) == 2
    np.testing.assert_array_equal(
        resolved_depths[1], [1003.5, 1000, 1005, 1007, 1008, 999, 999]
    )


@pytest.mark.parametrize(
    ("well_df", "surveys"),
    [
        (pd.DataFrame({"DEPTH": [100, 200, 300], "TVDSS": [-90, -180, -260]}), None),
        (
            pd.DataFrame({"DEPTH": [100, 200, 300]}),
            pd.DataFrame({"md": [100, 200, 300], "tvd": [90, 180, 260]}),
        ),
        (
            pd.DataFrame({"DEPTH": [100, 200, 300]}),
            pd.DataFrame(
                {"md": [0, 200, 300], "inc": [5, 15, 30], "azim": [0, 10, 20]}
            ),
        ),
    ],
)
def test_batched_tvd_resolution_matches_individual_depths(well_df, surveys):
    depths = np.array([50.0, 150.0, 300.0, 350.0])
    expected = np.concatenate(
        [
            ws_service._resolve_well_tvd_array(np.array([depth]), well_df, surveys)
            for depth in depths
        ]
    )
    actual = ws_service._resolve_well_tvd_array(depths, well_df, surveys)
    np.testing.assert_allclose(actual, expected)


def test_hc_correction_angle_and_den_correction_protects_hc_rows(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    r_hc = client.post(
        f"{base}/hc_correction",
        json={
            "rho_water": 1.0,
            "rho_hc": 0.3,
            "hi_hc": 0.66,
            "gr_clean": 30.0,
            "gr_clay": 110.0,
            "save": True,
        },
    )
    assert r_hc.status_code == 200
    hc_body = r_hc.json()
    # Angle measured from the NPHI axis: atan2(0.7, 0.34) ~= 64.09 deg
    assert hc_body["corr_angle"] == pytest.approx(64.09, abs=0.01)
    hc_flag = np.asarray(hc_body["data"]["HC_FLAG"], dtype=float)
    assert hc_flag.sum() >= 1
    nphi_hc = np.asarray(hc_body["data"]["NPHI_HC"], dtype=float)
    rhob_hc = np.asarray(hc_body["data"]["RHOB_HC"], dtype=float)

    r_bad_mode = client.post(f"{base}/hc_correction", json={"flag_mode": "bogus"})
    assert r_bad_mode.status_code == 400

    r_den = client.post(
        f"{base}/den_correction",
        json={"scope": "full", "gr_clean": 30.0, "gr_clay": 110.0, "save": False},
    )
    assert r_den.status_code == 200
    den_body = r_den.json()
    den_data = den_body["data"]
    assert "RHOB_DEN_CORR" in den_data
    assert "RHOB_CORR" not in den_data
    np.testing.assert_allclose(
        np.asarray(den_data["RHOB_DEN_CORR"], dtype=float),
        np.asarray(den_data["RHOB_HC"], dtype=float),
    )
    hc_rows = hc_flag >= 0.5
    np.testing.assert_allclose(
        np.asarray(den_data["HC_FLAG"], dtype=float)[hc_rows], 1.0
    )
    np.testing.assert_allclose(
        np.asarray(den_data["DEN_CORR_FLAG"], dtype=float)[hc_rows], 0.0
    )
    np.testing.assert_allclose(
        np.asarray(den_data["NPHI_HC"], dtype=float)[hc_rows], nphi_hc[hc_rows]
    )
    np.testing.assert_allclose(
        np.asarray(den_data["RHOB_HC"], dtype=float)[hc_rows], rhob_hc[hc_rows]
    )
    assert den_body["den_summary"]["hc_protected_count"] == int(hc_rows.sum())
