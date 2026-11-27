"""End-to-end verification tests for the Hugging Face Spaces Interactive Live Demo spec."""

from __future__ import annotations

import io

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

    cache_dir = (
        Path(__file__).resolve().parent.parent / ".pytest_cache" / "demo_seed_v1"
    )
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
    from quick_pp.app.backend.services.demo_guard import reset_demo_sessions_for_testing

    db_url, model_path = seeded_demo_paths
    monkeypatch.setenv("QPP_DEMO_MODE", "true")
    monkeypatch.setenv("QPP_DISABLE_TRAINING", "true")
    monkeypatch.setenv("QPP_DATABASE_URL", db_url)
    monkeypatch.setenv("QPP_PRETRAINED_MODELS_PATH", model_path)
    backend_main._db_connector_instance = None
    reset_demo_sessions_for_testing()


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
    proj_id = next(p["project_id"] for p in projects if p["name"] == "VOLVE_COSTA_DEMO")

    summary_res = client.get(f"/quick_pp/workspace/projects/{proj_id}/summary")
    assert summary_res.status_code == 200
    summary = summary_res.json()
    assert summary["well_count"] >= 3
    well_names = [w["name"] for w in summary["wells"]]
    assert "15-9-19-A" in well_names
    assert "HW-3" in well_names

    bundle_res = client.get(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/bundle"
    )
    assert bundle_res.status_code == 200
    bundle = bundle_res.json()
    for curve in (
        "GR",
        "NPHI",
        "RHOB",
        "RT",
        "PHIT",
        "PHIT_P10",
        "PHIT_P90",
        "SW",
        "PERM",
    ):
        assert curve in bundle["curves"]
    assert len(bundle["tops"]) >= 2
    assert len(bundle["contacts"]) >= 1


def test_deterministic_and_probabilistic_inspector_loop():
    """Verify single-well and multi-well interactive inspector endpoints."""
    client = TestClient(app)
    projects = client.get("/quick_pp/database/projects").json()
    proj_id = next(p["project_id"] for p in projects if p["name"] == "VOLVE_COSTA_DEMO")

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


def test_pretrained_ml_inference_and_readonly_training_guardrail(monkeypatch):
    """Verify pre-trained conformal inference works while heavy model training is blocked."""
    client = TestClient(app)
    projects = client.get("/quick_pp/database/projects").json()
    proj_id = next(p["project_id"] for p in projects if p["name"] == "VOLVE_COSTA_DEMO")

    models_res = client.get(f"/quick_pp/workspace/projects/{proj_id}/ml_models")
    assert models_res.status_code == 200
    models_info = models_res.json()
    assert models_info["pretrained_available"] is True
    assert models_info["fallback_active"] is False
    suite_keys = [s["key"] for s in models_info["suites"]]
    assert "clastic" in suite_keys
    assert "carbonate" in suite_keys

    ml_res = client.post(
        f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/ml_inference",
        json={"model_config": "clastic", "alpha": 0.20, "save": True},
    )
    assert ml_res.status_code == 200
    ml_json = ml_res.json()
    assert ml_json["pretrained_used"] is True
    assert ml_json["fallback_used"] is False
    ml_data = ml_json["data"]
    for col in (
        "PHIT",
        "PHIT_P10",
        "PHIT_P90",
        "SW",
        "SW_P10",
        "SW_P90",
        "PERM",
        "PERM_P10",
        "PERM_P90",
    ):
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
        files={
            "files": ("oversized_well.las", io.BytesIO(oversized_bytes), "text/plain")
        },
    )
    assert up_res.status_code == 413

    reset_res = client.post(f"/quick_pp/workspace/projects/{proj_id}/demo/reset")
    assert reset_res.status_code == 200
    assert reset_res.json()["status"] == "reset"


def test_online_users_heartbeat_capacity_gate_and_expiry(monkeypatch):
    """Verify online user counter, lag warning threshold, max-users soft gate, and session leave/expiry."""
    from quick_pp.app.backend.services import demo_guard

    monkeypatch.setenv("QPP_DEMO_MAX_USERS", "2")
    monkeypatch.setenv("QPP_DEMO_LAG_WARNING_USERS", "2")
    monkeypatch.setenv("QPP_DEMO_SESSION_TTL_SECONDS", "30")

    client = TestClient(app)

    # Session 1 joins: 1 online, no lag warning yet
    hb1 = client.post(
        "/quick_pp/demo/heartbeat",
        headers={"X-Demo-Session-Id": "demo_user_01"},
        json={"session_id": "demo_user_01"},
    )
    assert hb1.status_code == 200
    d1 = hb1.json()
    assert d1["admitted"] is True
    assert d1["online_users"] == 1
    assert d1["max_users"] == 2
    assert d1["lag_warning"] is False

    # Session 2 joins: 2 online -> hits lag warning threshold (2)
    hb2 = client.post(
        "/quick_pp/demo/heartbeat",
        headers={"X-Demo-Session-Id": "demo_user_02"},
    )
    assert hb2.status_code == 200
    d2 = hb2.json()
    assert d2["admitted"] is True
    assert d2["online_users"] == 2
    assert d2["lag_warning"] is True

    # Session 3 attempts to join while 2/2 slots are full -> soft waiting room (admitted=False)
    hb3 = client.post(
        "/quick_pp/demo/heartbeat",
        headers={"X-Demo-Session-Id": "demo_user_03"},
    )
    assert hb3.status_code == 200
    d3 = hb3.json()
    assert d3["admitted"] is False
    assert d3["at_capacity"] is True
    assert d3["online_users"] == 2

    # Direct API request from unadmitted Session 3 receives 429 DEMO_AT_CAPACITY
    blocked_api = client.get(
        "/quick_pp/database/projects",
        headers={"X-Demo-Session-Id": "demo_user_03"},
    )
    assert blocked_api.status_code == 429
    assert blocked_api.json()["code"] == "DEMO_AT_CAPACITY"

    # Existing admitted Session 1 continues to work normally
    ok_api = client.get(
        "/quick_pp/database/projects",
        headers={"X-Demo-Session-Id": "demo_user_01"},
    )
    assert ok_api.status_code == 200

    # Session 1 pauses/leaves -> frees a slot immediately so Session 3 is admitted
    leave_res = client.post(
        "/quick_pp/demo/leave",
        json={"session_id": "demo_user_01"},
    )
    assert leave_res.status_code == 200
    assert leave_res.json()["online_users"] == 1

    hb3_retry = client.post(
        "/quick_pp/demo/heartbeat",
        headers={"X-Demo-Session-Id": "demo_user_03"},
    )
    assert hb3_retry.json()["admitted"] is True
    assert hb3_retry.json()["online_users"] == 2

    # Expired heartbeats are evicted automatically
    evicted = demo_guard.evict_expired_sessions(now=9_999_999_999.0)
    assert set(evicted) == {"demo_user_02", "demo_user_03"}
    assert client.get("/quick_pp/demo/status").json()["online_users"] == 0


def test_demo_compute_concurrency_semaphore(monkeypatch):
    """Verify heavy compute requests return 429 DEMO_COMPUTE_BUSY when compute slots are saturated."""
    from quick_pp.app.backend.services import demo_guard

    monkeypatch.setenv("QPP_DEMO_MAX_CONCURRENT_COMPUTE", "1")
    monkeypatch.setenv("QPP_DEMO_COMPUTE_TIMEOUT_SECONDS", "0.05")

    client = TestClient(app)
    projects = client.get("/quick_pp/database/projects").json()
    proj_id = projects[0]["project_id"]

    # Hold the single compute slot to simulate a long-running solver
    assert demo_guard.acquire_compute_slot_sync(timeout=0.1) is True
    try:
        # Heavy compute route sheds load cleanly with 429
        busy_res = client.post(
            f"/quick_pp/workspace/projects/{proj_id}/wells/15-9-19-A/porosity",
            json={"method": "neu_den", "save": False},
        )
        assert busy_res.status_code == 429
        assert busy_res.json()["code"] == "DEMO_COMPUTE_BUSY"

        # Lightweight project listing remains fast and unblocked
        fast_res = client.get("/quick_pp/database/projects")
        assert fast_res.status_code == 200
    finally:
        demo_guard.release_compute_slot_sync()


def test_demo_endpoint_security_ssrf_uam_and_preseeded_guards(monkeypatch):
    """Verify SSRF base_url, UAM routes, and pre-seeded project/well deletion are blocked in demo mode."""
    from quick_pp.app.backend.services.demo_guard import validate_external_llm_target

    client = TestClient(app)
    projects = client.get("/quick_pp/database/projects").json()
    proj_id = next(p["project_id"] for p in projects if p["name"] == "VOLVE_COSTA_DEMO")

    # 1. Admin UAM routes must return 403 even though demo visitor has role='admin'
    uam_res = client.get("/quick_pp/uam/users")
    assert uam_res.status_code == 403

    # 2. SSRF via custom base_url (e.g. DigitalOcean metadata 169.254.169.254) or local Ollama must return 403
    ssrf_res = client.post(
        "/quick_pp/assistant/chat",
        json={
            "message": "Inspect well",
            "provider": "openrouter",
            "base_url": "http://169.254.169.254/metadata/v1",
            "api_key": "dummy",
        },
    )
    assert ssrf_res.status_code == 403

    ollama_res = client.post(
        "/quick_pp/assistant/chat",
        json={"message": "Inspect well", "provider": "ollama"},
    )
    assert ollama_res.status_code == 403

    # Even outside demo mode, server env OPENROUTER_API_KEY is never sent to a custom base_url
    monkeypatch.setenv("QPP_DEMO_MODE", "false")
    url_out, key_out = validate_external_llm_target(
        provider="openrouter",
        caller_base_url="https://evil.example.com/v1",
        caller_api_key=None,
        env_openrouter_key="sk-or-secret-server-key",
    )
    assert url_out == "https://evil.example.com/v1"
    assert key_out == ""
    monkeypatch.setenv("QPP_DEMO_MODE", "true")

    # 3. Pre-seeded showcase projects & wells cannot be deleted, renamed, or created in DB
    assert (
        client.post("/quick_pp/database/projects", json={"name": "NewProj"}).status_code
        == 403
    )
    assert (
        client.patch(
            f"/quick_pp/database/projects/{proj_id}", json={"name": "Hacked"}
        ).status_code
        == 403
    )
    assert client.delete(f"/quick_pp/database/projects/{proj_id}").status_code == 403
    assert (
        client.delete(
            f"/quick_pp/database/projects/{proj_id}/wells/15-9-19-A"
        ).status_code
        == 403
    )
    assert (
        client.patch(
            f"/quick_pp/database/projects/{proj_id}/wells/15-9-19-A",
            json={"name": "RENAMED"},
        ).status_code
        == 403
    )


def test_three_tier_ephemeral_artifact_cleanup(tmp_path, monkeypatch):
    """Verify immediate upload deletion, session-leave cleanup, ring-buffer caps, and idle reset."""
    import time

    from quick_pp.app.backend.services import database as db_svc
    from quick_pp.app.backend.services import demo_guard, las_handler, report

    # Isolate report & staging directories to tmp_path so tests never touch workspace files
    fake_reports_root = tmp_path / "reports"
    fake_reports_root.mkdir(parents=True, exist_ok=True)
    fake_staging_dir = tmp_path / "import_staging"
    fake_staging_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(report, "REPORTS_STATIC_ROOT", fake_reports_root)
    monkeypatch.setattr(las_handler, "STAGING_DIR", fake_staging_dir)
    monkeypatch.setenv("QPP_DEMO_MAX_REPORTS", "2")
    monkeypatch.setenv("QPP_DEMO_ZERO_USERS_RESET_SECONDS", "10")

    client = TestClient(app)

    # Register session and simulate session-owned staged upload + report folder
    sid = "demo_cleanup_sess"
    client.post("/quick_pp/demo/heartbeat", headers={"X-Demo-Session-Id": sid})

    staged_file = fake_staging_dir / "fid1_test.las"
    staged_file.write_text("~Version\n", encoding="utf-8")
    las_handler._STAGED_FILES["fid1"] = {
        "file_id": "fid1",
        "path": str(staged_file),
        "filename": "test.las",
        "created_at": time.time(),
        "demo_session_id": sid,
    }

    rpt_dir = fake_reports_root / "rpt_sess01"
    rpt_dir.mkdir()
    (rpt_dir / "report.md").write_text("# Demo Report", encoding="utf-8")
    report._REPORTS_STORE["rpt_sess01"] = {
        "report_id": "rpt_sess01",
        "_demo_session_id": sid,
        "_created_ts": time.time(),
    }

    # Tier 2: Leaving the session immediately deletes both the staged upload and report folder
    leave_out = client.post("/quick_pp/demo/leave", json={"session_id": sid}).json()
    assert leave_out["purged"]["reports_deleted"] == 1
    assert leave_out["purged"]["staged_files_deleted"] == 1
    assert not staged_file.exists()
    assert not rpt_dir.exists()

    # Tier 3: Ring-buffer cap keeps at most QPP_DEMO_MAX_REPORTS (2) report folders
    for idx in range(4):
        rid = f"rpt_ring0{idx}"
        rdir = fake_reports_root / rid
        rdir.mkdir()
        (rdir / "report.json").write_text("{}", encoding="utf-8")
        report._REPORTS_STORE[rid] = {
            "report_id": rid,
            "_created_ts": 1000.0 + idx,
        }
    demo_guard.enforce_report_storage_limits(now=1010.0)
    remaining_dirs = sorted(p.name for p in fake_reports_root.iterdir() if p.is_dir())
    assert remaining_dirs == ["rpt_ring02", "rpt_ring03"]

    # Global zero-users idle sweep resets ephemeral DataFrames back to clean seed state
    db_svc._ephemeral_wells[999] = {"TEMP_WELL": None}
    demo_guard.mark_demo_state_dirty()
    sweep_res = demo_guard.run_periodic_demo_cleanup(now=time.time() + 60.0)
    assert sweep_res["memory_reset"] is True
    assert 999 not in db_svc._ephemeral_wells
