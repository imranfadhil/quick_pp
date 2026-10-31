"""API tests for the multi-well Formation Pressure & Contacts (FPT) module.

Endpoint functions are invoked directly (sync FastAPI handlers) to avoid event-loop / HTTP
transport dependencies; request bodies are validated through the same Pydantic models.
"""

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quick_pp.app.backend.services import database as db_service
from quick_pp.app.backend.services import fpt as fpt_service
from quick_pp.app.backend.services import workspace as ws_service
from quick_pp.database.models import Base
from quick_pp.database.objects import Project, Well

PSI_PER_M_OIL = 0.33 / 0.3048
PSI_PER_M_WATER = 0.45 / 0.3048
OWC_M = 1006.0
P_OWC = 2100.0


def _p(z: float, grad: float) -> float:
    return P_OWC + grad * (z - OWC_M)


@pytest.fixture
def fpt_client(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()

    proj = Project(db_session=session, name="fpt_test_proj")
    proj.save()
    depths = np.linspace(990.0, 1015.0, 51)
    wells = []
    for wn in ("FPT-1", "FPT-2"):
        w = Well(db_session=session, project_id=proj.project_id, name=wn, uwi=wn)
        w.save()
        wells.append(w)
    df = pd.concat(
        [
            pd.DataFrame(
                {"DEPTH": depths, "WELL_NAME": wn, "GR": 50.0, "ZONES": "Sand-A"}
            )
            for wn in ("FPT-1", "FPT-2")
        ]
    )
    proj.update_data(df)
    proj.save()
    for w in wells:
        w.add_well_surveys(
            [{"md": 0.0, "inc": 0.0, "azim": 0.0}, {"md": 2000.0, "inc": 0.0, "azim": 0.0}]
        )
        w.add_formation_tops([{"name": "Sand-A", "depth": 990.0}])
    # Oil points in FPT-1, water points in FPT-2 (field-wide compartment)
    wells[0].add_pressure_tests(
        [
            {"depth": z, "pressure": _p(z, PSI_PER_M_OIL), "fluid_type": "OIL", "mobility": 25.0}
            for z in (995.0, 998.0, 1001.0, 1004.0)
        ]
        + [{"depth": 1002.5, "pressure": _p(1002.5, PSI_PER_M_OIL) + 40.0, "fluid_type": "OIL"}]
    )
    wells[1].add_pressure_tests(
        [
            {"depth": z, "pressure": _p(z, PSI_PER_M_WATER), "fluid_type": "water"}
            for z in (1008.0, 1010.0, 1012.0, 1014.0)
        ]
    )
    session.commit()

    class DummyCtx:
        def __enter__(self):
            return session

        def __exit__(self, *exc):
            return False

    for mod in (db_service, ws_service, fpt_service):
        monkeypatch.setattr(mod, "get_db", lambda: DummyCtx())
    db_service.invalidate_well_cache()
    yield proj.project_id, session
    session.close()


def _points(pid):
    return fpt_service.fpt_list_points(pid, well_names=None)


def _analyze(pid, **body):
    return fpt_service.fpt_analyze(pid, fpt_service.FPTAnalysisRequest(**body))


def test_points_resolve_tvdss_and_zone(fpt_client):
    pid, _ = fpt_client
    body = _points(pid)
    assert body["n_points"] == 9
    assert body["n_missing_tvdss"] == 0
    pt = next(p for p in body["points"] if p["md"] == 995.0)
    assert pt["tvdss"] == pytest.approx(995.0, abs=0.05)
    assert pt["zone"] == "Sand-A"
    assert {p["fluid_type"] for p in body["points"]} == {"OIL", "WATER"}


def test_analyze_patch_and_apply_contacts(fpt_client):
    pid, session = fpt_client
    pts = _points(pid)["points"]
    outlier = next(p for p in pts if p["md"] == 1002.5)

    # Unsaved UI exclusion of the supercharged point via point_overrides
    res = _analyze(
        pid,
        pressure_unit="psi",
        depth_unit="m",
        point_overrides={str(outlier["id"]): {"test_quality": "SUPERCHARGED"}},
        n_simulations=300,
    )
    assert res["gradients"]["OIL"]["density_gcc"] == pytest.approx(0.33 / 0.433527, rel=1e-3)
    owc = res["contacts"]["OWC"]
    assert owc["tvdss_p50"] == pytest.approx(OWC_M, abs=0.1)
    assert owc["tvdss_p50_m"] == pytest.approx(OWC_M, abs=0.1)
    assert "OIL" in res["plot_lines"] and "WATER" in res["plot_lines"]
    assert next(p for p in res["points"] if p["id"] == outlier["id"])["test_quality_stored"] == "VALID"

    # Same analysis in psi/ft returns contact in ft
    r_ft = _analyze(
        pid,
        depth_unit="ft",
        point_overrides={str(outlier["id"]): {"test_quality": "EXCLUDED"}},
    )
    assert r_ft["gradients"]["OIL"]["gradient"] == pytest.approx(0.33, abs=1e-4)
    assert r_ft["contacts"]["OWC"]["tvdss_p50"] == pytest.approx(OWC_M / 0.3048, abs=0.5)

    # Auto-classify flags the +40 psi point as supercharged
    r_auto = _analyze(pid, auto_classify={"ransac_residual_threshold": 0, "assign_unassigned": False})
    assert next(p for p in r_auto["points"] if p["id"] == outlier["id"])["test_quality"] == "SUPERCHARGED"

    # Persist the exclusion
    r = fpt_service.fpt_patch_points(
        pid,
        fpt_service.FPTPointsPatchRequest(
            updates=[{"test_id": outlier["id"], "well_name": "FPT-1", "test_quality": "SUPERCHARGED"}]
        ),
    )
    assert r["updated"] == 1
    session.expire_all()
    pts2 = _points(pid)["points"]
    assert next(p for p in pts2 if p["id"] == outlier["id"])["test_quality"] == "SUPERCHARGED"

    # Write contact into wells as FWL (MD == TVDSS for vertical survey)
    r = fpt_service.fpt_apply_contacts(
        pid, fpt_service.FPTApplyContactsRequest(contacts={"OWC": owc["tvdss_p50_m"]})
    )
    assert r["n_wells_written"] == 2
    session.expire_all()
    w1 = Project.load(session, project_id=pid).get_well("FPT-1")
    contacts = w1.get_fluid_contacts()
    assert contacts.loc[contacts["name"] == "FWL", "depth"].iloc[0] == pytest.approx(OWC_M, abs=0.2)


def test_router_registered():
    from quick_pp.app.backend.main import app

    paths = set(app.openapi().get("paths", {}).keys())
    for suffix in ("/fpt/points", "/fpt/analyze", "/fpt/apply-contacts"):
        assert f"/quick_pp/workspace/projects/{{project_id}}{suffix}" in paths


def test_single_well_save_preserves_interpretation(fpt_client):
    pid, session = fpt_client
    w2 = Project.load(session, project_id=pid).get_well("FPT-2")
    df = w2.get_pressure_tests()
    assert set(df["fluid_type"]) == {"WATER"}
    # Re-save rows (as the single-well grid does in replace mode) keeping interpretation fields
    w2.add_pressure_tests(df.drop(columns=["test_id"]).to_dict(orient="records"), replace=True)
    session.commit()
    session.expire_all()
    assert set(w2.get_pressure_tests()["fluid_type"]) == {"WATER"}


def test_analyze_by_zone(fpt_client):
    pid, session = fpt_client
    proj = Project.load(session, project_id=pid)
    w2 = proj.get_well("FPT-2")
    # Add tops and points in a distinct deeper zone Sand-B (1800m, completely different pressure)
    w2.add_formation_tops([{"name": "Sand-B", "depth": 1800.0}])
    w2.add_pressure_tests(
        [
            {"depth": 1805.0, "pressure": 3500.0, "fluid_type": "OIL", "zone": "Sand-B"},
            {"depth": 1810.0, "pressure": 3500.0 + PSI_PER_M_OIL * 5.0, "fluid_type": "OIL", "zone": "Sand-B"},
            {"depth": 1815.0, "pressure": 3500.0 + PSI_PER_M_OIL * 10.0, "fluid_type": "WATER", "zone": "Sand-B"},
            {"depth": 1820.0, "pressure": 3500.0 + PSI_PER_M_OIL * 10.0 + PSI_PER_M_WATER * 5.0, "fluid_type": "WATER", "zone": "Sand-B"},
        ]
    )
    session.commit()
    session.expire_all()

    # Analyze with group_by_zone=True
    r = _analyze(pid, group_by_zone=True)
    assert "by_zone" in r
    assert "Sand-A" in r["by_zone"]
    assert "Sand-B" in r["by_zone"]
    assert r["active_zone"] == "ALL"

    # Sand-B gradients are independently fitted and realistic
    g_b = r["by_zone"]["Sand-B"]["gradients"]
    assert g_b["OIL"]["gradient"] == pytest.approx(PSI_PER_M_OIL, abs=0.05)
    assert 0.65 <= g_b["OIL"]["density_gcc"] <= 0.85
    assert g_b["WATER"]["gradient"] == pytest.approx(PSI_PER_M_WATER, abs=0.05)
    assert 0.95 <= g_b["WATER"]["density_gcc"] <= 1.15

    # Sand-B contact estimated independently without cross-zone crosstalk
    c_b = r["by_zone"]["Sand-B"]["contacts"]["OWC"]
    assert c_b["tvdss_p50"] == pytest.approx(1815.0, abs=1.0)

    # When using RANSAC or excluding the outlier, Sand-A is also cleanly fitted
    r_ransac = _analyze(pid, group_by_zone=True, regression_method="ransac")
    g_a_clean = r_ransac["by_zone"]["Sand-A"]["gradients"]
    assert g_a_clean["OIL"]["gradient"] == pytest.approx(PSI_PER_M_OIL, abs=0.05)
    assert 0.65 <= g_a_clean["OIL"]["density_gcc"] <= 0.85

    # Query with specific active_zone="Sand-B"
    r_b = _analyze(pid, group_by_zone=True, active_zone="Sand-B")
    assert r_b["active_zone"] == "Sand-B"
    assert r_b["gradients"]["OIL"]["gradient"] == pytest.approx(PSI_PER_M_OIL, abs=0.05)
    assert r_b["contacts"]["OWC"]["tvdss_p50"] == pytest.approx(1815.0, abs=1.0)

