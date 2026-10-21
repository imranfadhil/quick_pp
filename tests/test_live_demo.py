"""End-to-end verification tests for the Hugging Face Spaces Interactive Live Demo spec."""

from __future__ import annotations

import io
import os
import pytest
from fastapi.testclient import TestClient

from quick_pp.app.backend import main as backend_main
from quick_pp.app.backend.main import app
from quick_pp.machine_learning.train_pipeline import train_pipeline


@pytest.fixture(scope="module")
def seeded_demo_paths(tmp_path_factory):
    """Seed an isolated SQLite demo database once (cached in .pytest_cache) and copy per test module."""
    import shutil
    from pathlib import Path
    from deploy.demo.seed_demo import seed_demo_database

    cache_dir = Path(__file__).resolve().parent.parent / ".pytest_cache" / "demo_seed_v1"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_db = cache_dir / "quick_pp_seed.db"
    cached_model = cache_dir / "pretrained_models.joblib"

    prev_conn = backend_main._db_connector_instance
    backend_main._db_connector_instance = None

    if not cached_db.exists() or not cached_model.exists():
        if cached_db.exists():
            cached_db.unlink()
        seed_demo_database(
            db_url=f"sqlite:///{cached_db.as_posix()}",
            model_path=cached_model,
            max_rows_per_well=80,
        )

    demo_dir = tmp_path_factory.mktemp("live_demo")
    db_file = demo_dir / "quick_pp_demo.db"
    shutil.copy2(cached_db, db_file)
    db_url = f"sqlite:///{db_file.as_posix()}"

    yield db_url, str(cached_model)
    if backend_main._db_connector_instance is not None:
        try:
            backend_main._db_connector_instance.dispose()
        except Exception:
            pass
    backend_main._db_connector_instance = prev_conn


@pytest.fixture(autouse=True)
def enable_demo_mode(monkeypatch, seeded_demo_paths):
    db_url, model_path = seeded_demo_paths
    monkeypatch.setenv("QPP_DEMO_MODE", "true")
    monkeypatch.setenv("QPP_DISABLE_TRAINING", "true")
    monkeypatch.setenv("QPP_DATABASE_URL", db_url)
    monkeypatch.setenv("QPP_PRETRAINED_MODELS_PATH", model_path)
    backend_main._db_connector_instance = None


def test_zero_cold_start_auth_and_preseeded_projects():
    """Verify visitors bypass login in QPP_DEMO_MODE and see pre-seeded Volve & Costa wells."""
    client = TestClient(app)

    me_res = client.get("/quick_pp/auth/me")
    assert me_res.status_code == 200
    user = me_res.json()["user"]
    assert user["role"] == "admin"
    assert user["status"] == "CONFIRMED"
    assert user["demo_mode"] is True

    # Even with a stale/invalid Bearer token or cookie (e.g. non-Incognito browser), demo mode bypasses auth
    stale_res = client.get(
        "/quick_pp/auth/me",
        headers={
            "Authorization": "Bearer expired.or.invalid.jwt.token",
            "Cookie": "qpp_access_token=stale_cookie_token",
        },
    )
    assert stale_res.status_code == 200
    assert stale_res.json()["user"]["demo_mode"] is True

    login_res = client.post(
        "/quick_pp/auth/login",
        json={"username": "admin", "password": "any"},
    )
    assert login_res.status_code == 200
    assert login_res.json()["status"] == "AUTHENTICATED"
    assert login_res.json()["user"]["demo_mode"] is True

    proj_res = client.get("/quick_pp/database/projects")
    assert proj_res.status_code == 200
    projects = proj_res.json()
    assert len(projects) >= 1
    proj_id = projects[0]["project_id"]

    summary_res = client.get(f"/quick_pp/workspace/projects/{proj_id}/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["well_count"] >= 3
    well_names = [w["name"] for w in summary["wells"]]
    assert "15-9-19-A" in well_names
    assert "HW-3" in well_names

    bundle_res = client.get(f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/bundle")
    assert bundle_res.status_code == 200
    bundle = bundle_res.json()
    for curve in ("GR", "NPHI", "RHOB", "RT", "PHIT", "PHIT_P10", "PHIT_P90", "SW", "PERM"):
        assert curve in bundle["curves"]
    assert len(bundle["tops"]) >= 2
    assert len(bundle["contacts"]) >= 1


def test_deterministic_and_probabilistic_inspector_loop():
    """Verify single-well and multi-well interactive inspector endpoints."""
    client = TestClient(app)
    proj_id = client.get("/quick_pp/database/projects").json()[0]["project_id"]

    # 1. Lithology with Vshale GR endpoints
    lith_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/lithology",
        json={
            "model": "ssc",
            "gr_clean": 15.0,
            "gr_clay": 115.0,
            "save": True,
        },
    )
    assert lith_res.status_code == 200
    assert "VCLAY" in lith_res.json()["data"]

    # 2. Porosity
    poro_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/porosity",
        json={"method": "neu_den", "save": True},
    )
    assert poro_res.status_code == 200
    assert "PHIT" in poro_res.json()["data"]

    # 3. Saturation with Archie m, n, Rw scrubber override
    sat_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/saturation",
        json={
            "method": "archie",
            "m": 1.9,
            "n": 2.05,
            "rw": 0.04,
            "save": True,
        },
    )
    assert sat_res.status_code == 200
    assert "SWT" in sat_res.json()["data"]

    # 4. Permeability
    perm_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/permeability",
        json={"method": "coates", "swirr": 0.15, "save": True},
    )
    assert perm_res.status_code == 200
    assert "PERM" in perm_res.json()["data"]

    # 5. Reservoir Summary + Upscaling + Monte Carlo Uncertainty
    ressum_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/reservoir_summary",
        json={
            "cut_offs": {"VSHALE": 0.4, "PHIT": 0.08, "SWT": 0.65},
            "include_upscaling": True,
            "cell_size": 2.0,
            "include_uncertainty": True,
            "n_samples": 25,
        },
    )
    assert ressum_res.status_code == 200
    rjson = ressum_res.json()
    assert "summary" in rjson
    assert "flags" in rjson
    assert "blocked_cells" in rjson["upscaling"]
    assert "cpi" in rjson["uncertainty"]

    # 6. Multi-Well Rock Typing, Perm Transform & SHF
    rt_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/rock_typing/compute",
        json={"cutoffs": [0.5, 1.2, 2.5], "higher_is_better": True, "save": True},
    )
    assert rt_res.status_code == 200
    assert "rock_type_counts" in rt_res.json()

    pt_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/perm_transform/compute",
        json={"save": True},
    )
    assert pt_res.status_code == 200
    assert "perm_pred" in pt_res.json()["data"]

    shf_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/shf/compute",
        json={"cutoffs": [0.5, 1.2, 2.5], "fwl": 3915.0, "save": True},
    )
    assert shf_res.status_code == 200
    assert "shf" in shf_res.json()["data"]


def test_pretrained_ml_inference_and_readonly_training_guardrail():
    """Verify pre-trained conformal inference works while heavy model training is blocked."""
    client = TestClient(app)
    proj_id = client.get("/quick_pp/database/projects").json()[0]["project_id"]

    ml_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/ml_inference",
        json={"model_config": "clastic", "alpha": 0.20, "save": True},
    )
    assert ml_res.status_code == 200
    ml_data = ml_res.json()["data"]
    for col in ("PHIT", "PHIT_P10", "PHIT_P90", "SW", "SW_P10", "SW_P90", "PERM", "PERM_P10", "PERM_P90"):
        assert col in ml_data

    with pytest.raises(PermissionError):
        train_pipeline("clastic", "demo_hash")


def test_ephemeral_session_upload_cap_and_reset():
    """Verify > 5 MB LAS upload is rejected with 413 and demo reset clears ephemeral session."""
    client = TestClient(app)
    proj_id = client.get("/quick_pp/database/projects").json()[0]["project_id"]

    oversized_bytes = b"~Version\n" + (b"0" * (5 * 1024 * 1024 + 1024))
    up_res = client.post(
        f"/quick_pp/database/projects/{proj_id}/read_las",
        files={"files": ("oversized_well.las", io.BytesIO(oversized_bytes), "text/plain")},
    )
    assert up_res.status_code == 413

    reset_res = client.post(f"/quick_pp/workspace/projects/{proj_id}/demo/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "reset"

