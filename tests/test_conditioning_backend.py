"""Backend API tests for the resolution enhancement endpoint and Thomas-Stieber handoff."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quick_pp.app.backend.main import app
from quick_pp.app.backend.services import database as db_service
from quick_pp.app.backend.services import workspace as ws_service
from quick_pp.conditioning import (
    DEFAULT_LEGACY_SPECS,
    apply_vrf_forward,
    build_vrf_kernel,
)
from quick_pp.database.models import Base
from quick_pp.database.objects import Project, Well

DZ = 0.1524


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

    proj = Project(db_session=session, name="conditioning_test_proj")
    proj.save()

    n = 300
    depth = 1000.0 + np.arange(n) * DZ
    bed = (np.floor((depth - depth[0]) / 0.45) % 2).astype(int)
    truth = {
        "GR": np.where(bed == 0, 25.0, 115.0),
        "NPHI": np.where(bed == 0, 0.27, 0.36),
        "RHOB": np.where(bed == 0, 2.22, 2.55),
    }
    rng = np.random.default_rng(7)
    measured = {}
    for name, spec in DEFAULT_LEGACY_SPECS.items():
        kernel = build_vrf_kernel(spec.fwhm_m, DZ)
        measured[name] = apply_vrf_forward(truth[name], kernel) + rng.normal(
            0.0, spec.noise_floor * 3.0, n
        )

    df = pd.DataFrame(
        {
            "DEPTH": depth,
            "WELL_NAME": ["Well-A"] * n,
            "GR": measured["GR"],
            "NPHI": measured["NPHI"],
            "RHOB": measured["RHOB"],
            "RT": np.full(n, 10.0),
            "CALI": np.full(n, 8.5),
            "BS": np.full(n, 8.5),
            "DRHO": np.zeros(n),
            "ZONES": ["Zone-1"] * n,
        }
    )
    well_a = Well(
        db_session=session, project_id=proj.project_id, name="Well-A", uwi="WELL-A-1"
    )
    well_a.save()
    proj.update_data(df)
    proj.save()
    well_a.add_formation_tops([{"name": "Zone-1", "depth": 1000.0}])
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
    db_service.invalidate_well_cache()

    client = TestClient(app)
    yield client, proj.project_id
    session.close()


def test_resolution_enhancement_endpoint(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    response = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "enhance_all", "save": False},
    )
    assert response.status_code == 200, response.text
    body = response.json()

    assert body["target_fwhm_m"] > 0
    assert body["depth_uom"] == "m"
    assert body["depth_step_m"] == pytest.approx(0.1524, abs=1e-6)

    for curve in ("GR_ENH", "NPHI_ENH", "RHOB_ENH", "DECONV_WEIGHT"):
        assert curve in body["data"], f"missing {curve}"
        assert any(v is not None for v in body["data"][curve]), f"{curve} is all-null"

    assert set(body["diagnostics"]) == {"GR", "NPHI", "RHOB"}
    for diag in body["diagnostics"].values():
        assert diag["converged"] is True
        assert diag["iterations"] > 0
        assert abs(diag["relative_volume_bias_pct"]) < 0.5


def test_resolution_enhancement_feeds_thomas_stieber(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    enh = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "enhance_all", "save": False},
    )
    assert enh.status_code == 200, enh.text

    ts = client.post(
        f"{base}/thomas_stieber",
        json={"use_existing_phit_vsh": False, "use_enhanced_logs": True},
    )
    assert ts.status_code == 200, ts.text
    for curve in ("VSH_LAM", "VSH_DIS", "PHIT_SAND"):
        assert curve in ts.json()["data"]

    # Enhanced logs must override pre-computed PHIT/VCLAY so the TS partition
    # actually reflects the harmonized nuclear curves.
    raw = client.post(
        f"{base}/thomas_stieber",
        json={"use_existing_phit_vsh": False, "use_enhanced_logs": False},
    )
    assert raw.status_code == 200, raw.text
    enh_lam = np.asarray(ts.json()["data"]["VSH_LAM"], dtype=float)
    raw_lam = np.asarray(raw.json()["data"]["VSH_LAM"], dtype=float)
    both = np.isfinite(enh_lam) & np.isfinite(raw_lam)
    assert both.any()
    assert not np.allclose(enh_lam[both], raw_lam[both], atol=1e-6)


def test_resolution_enhancement_save_and_wiener(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    saved = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "match_finest", "method": "wiener", "save": True},
    )
    assert saved.status_code == 200, saved.text
    assert saved.json()["saved_rows"] > 0


def test_resolution_enhancement_noise_cap_mult(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    capped = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "enhance_all", "save": False, "noise_cap_mult": 1.0},
    )
    assert capped.status_code == 200, capped.text
    uncapped = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "enhance_all", "save": False, "noise_cap_mult": None},
    )
    assert uncapped.status_code == 200, uncapped.text

    for curve in ("GR", "NPHI", "RHOB"):
        floor = DEFAULT_LEGACY_SPECS[curve].noise_floor
        assert capped.json()["diagnostics"][curve][
            "estimated_noise_std"
        ] == pytest.approx(floor)
        assert (
            uncapped.json()["diagnostics"][curve]["estimated_noise_std"] > floor
        )


def test_resolution_enhancement_rejects_missing_curves(client_and_project):
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    response = client.post(
        f"{base}/resolution_enhancement",
        json={"curves": ["SP", "CALI"], "save": False},
    )
    assert response.status_code == 400
    assert "None of the requested nuclear curves" in response.json()["detail"]


@pytest.mark.parametrize(
    "endpoint,payload,curve",
    [
        ("lithology", {"model": "ssc"}, "VCLAY"),
        ("porosity", {"method": "den"}, "PHIT"),
        # Saturation reads NPHI/RHOB directly only for the Juhasz Qvn route; its
        # default CEC route inherits enhancement transitively via PHIT/VCLAY.
        ("saturation", {"method": "waxman_smits", "qv_method": "qvn"}, "SWT"),
    ],
)
def test_enhanced_logs_toggle_changes_downstream_results(
    client_and_project, endpoint, payload, curve
):
    """The enhanced-logs toggle must actually change lithology, porosity, and saturation."""
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    enh = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "enhance_all", "save": True},
    )
    assert enh.status_code == 200, enh.text

    raw = client.post(
        f"{base}/{endpoint}",
        json={**payload, "save": False, "use_enhanced_logs": False},
    )
    assert raw.status_code == 200, raw.text
    enhanced = client.post(
        f"{base}/{endpoint}",
        json={**payload, "save": False, "use_enhanced_logs": True},
    )
    assert enhanced.status_code == 200, enhanced.text

    raw_vals = np.asarray(raw.json()["data"][curve], dtype=float)
    enh_vals = np.asarray(enhanced.json()["data"][curve], dtype=float)
    both = np.isfinite(raw_vals) & np.isfinite(enh_vals)
    assert both.any(), f"{curve} has no overlapping finite samples"
    assert not np.allclose(raw_vals[both], enh_vals[both], atol=1e-9), (
        f"{curve} is unchanged by use_enhanced_logs"
    )


def test_enhanced_logs_toggle_drives_vsh_gr_model(client_and_project):
    """The VSH_GR lithology model must read GR_ENH when the toggle is on."""
    client, pid = client_and_project
    base = f"/quick_pp/workspace/projects/{pid}/wells/Well-A"

    enh = client.post(
        f"{base}/resolution_enhancement",
        json={"mode": "enhance_all", "save": True},
    )
    assert enh.status_code == 200, enh.text

    payload = {"model": "vsh_gr", "gamma_source": "gr", "gr_clean": 20.0, "gr_clay": 120.0}
    raw = client.post(f"{base}/lithology", json={**payload, "save": False})
    assert raw.status_code == 200, raw.text
    enhanced = client.post(
        f"{base}/lithology",
        json={**payload, "save": False, "use_enhanced_logs": True},
    )
    assert enhanced.status_code == 200, enhanced.text

    raw_vals = np.asarray(raw.json()["data"]["VCLAY"], dtype=float)
    enh_vals = np.asarray(enhanced.json()["data"]["VCLAY"], dtype=float)
    both = np.isfinite(raw_vals) & np.isfinite(enh_vals)
    assert both.any()
    assert not np.allclose(raw_vals[both], enh_vals[both], atol=1e-9)
