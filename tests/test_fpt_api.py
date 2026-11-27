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
            [
                {"md": 0.0, "inc": 0.0, "azim": 0.0},
                {"md": 2000.0, "inc": 0.0, "azim": 0.0},
            ]
        )
        w.add_formation_tops([{"name": "Sand-A", "depth": 990.0}])
    # Oil points in FPT-1, water points in FPT-2 (field-wide compartment)
    wells[0].add_pressure_tests(
        [
            {
                "depth": z,
                "pressure": _p(z, PSI_PER_M_OIL),
                "fluid_type": "OIL",
                "mobility": 25.0,
            }
            for z in (995.0, 998.0, 1001.0, 1004.0)
        ]
        + [
            {
                "depth": 1002.5,
                "pressure": _p(1002.5, PSI_PER_M_OIL) + 40.0,
                "fluid_type": "OIL",
            }
        ]
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
    assert res["gradients"]["OIL"]["density_gcc"] == pytest.approx(
        0.33 / 0.433527, rel=1e-3
    )
    owc = res["contacts"]["OWC"]
    assert owc["tvdss_p50"] == pytest.approx(OWC_M, abs=0.1)
    assert owc["tvdss_p50_m"] == pytest.approx(OWC_M, abs=0.1)
    assert "OIL" in res["plot_lines"] and "WATER" in res["plot_lines"]
    assert (
        next(p for p in res["points"] if p["id"] == outlier["id"])[
            "test_quality_stored"
        ]
        == "VALID"
    )

    # Same analysis in psi/ft returns contact in ft
    r_ft = _analyze(
        pid,
        depth_unit="ft",
        point_overrides={str(outlier["id"]): {"test_quality": "EXCLUDED"}},
    )
    assert r_ft["gradients"]["OIL"]["gradient"] == pytest.approx(0.33, abs=2e-4)
    assert r_ft["contacts"]["OWC"]["tvdss_p50"] == pytest.approx(
        OWC_M / 0.3048, abs=0.5
    )

    # Auto-classify flags the +40 psi point as supercharged
    r_auto = _analyze(
        pid, auto_classify={"ransac_residual_threshold": 0, "assign_unassigned": False}
    )
    assert (
        next(p for p in r_auto["points"] if p["id"] == outlier["id"])["test_quality"]
        == "SUPERCHARGED"
    )

    # Persist the exclusion
    r = fpt_service.fpt_patch_points(
        pid,
        fpt_service.FPTPointsPatchRequest(
            updates=[
                {
                    "test_id": outlier["id"],
                    "well_name": "FPT-1",
                    "test_quality": "SUPERCHARGED",
                }
            ]
        ),
    )
    assert r["updated"] == 1
    session.expire_all()
    pts2 = _points(pid)["points"]
    assert (
        next(p for p in pts2 if p["id"] == outlier["id"])["test_quality"]
        == "SUPERCHARGED"
    )

    # Write contact into wells as FWL (MD == TVDSS for vertical survey)
    r = fpt_service.fpt_apply_contacts(
        pid, fpt_service.FPTApplyContactsRequest(contacts={"OWC": owc["tvdss_p50_m"]})
    )
    assert r["n_wells_written"] == 2
    session.expire_all()
    w1 = Project.load(session, project_id=pid).get_well("FPT-1")
    contacts = w1.get_fluid_contacts()
    assert contacts.loc[contacts["name"] == "FWL", "depth"].iloc[0] == pytest.approx(
        OWC_M, abs=0.2
    )


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
    w2.add_pressure_tests(
        df.drop(columns=["test_id"]).to_dict(orient="records"), replace=True
    )
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
            {
                "depth": 1805.0,
                "pressure": 3500.0,
                "fluid_type": "OIL",
                "zone": "Sand-B",
            },
            {
                "depth": 1810.0,
                "pressure": 3500.0 + PSI_PER_M_OIL * 5.0,
                "fluid_type": "OIL",
                "zone": "Sand-B",
            },
            {
                "depth": 1815.0,
                "pressure": 3500.0 + PSI_PER_M_OIL * 10.0,
                "fluid_type": "WATER",
                "zone": "Sand-B",
            },
            {
                "depth": 1820.0,
                "pressure": 3500.0 + PSI_PER_M_OIL * 10.0 + PSI_PER_M_WATER * 5.0,
                "fluid_type": "WATER",
                "zone": "Sand-B",
            },
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


def test_zone_filtering_case_insensitive_and_aliases(fpt_client):
    pid, _ = fpt_client
    # Lowercase / uppercase zone filter should cleanly match Sand-A
    r_lower = _analyze(pid, zones=["sand-a"], n_simulations=100)
    assert r_lower["n_points"] == 9
    assert all("Sand-A" in p["zone"] for p in r_lower["points"])

    r_upper = _analyze(pid, zones=["SAND-A"], n_simulations=100)
    assert r_upper["n_points"] == 9

    # Alias / substring filter: 'Sand-A Formation' matching 'Sand-A'
    r_alias = _analyze(pid, zones=["Sand-A Formation"], n_simulations=100)
    assert r_alias["n_points"] == 9


def test_zone_resolution_from_well_logs_and_unit_alignment():
    # Test _resolve_fpt_point_zone directly
    from quick_pp.app.backend.services.fpt import _resolve_fpt_point_zone

    # Case 1: Stored zone wins
    z, src = _resolve_fpt_point_zone(1000.0, "Fulmar", None, [])
    assert z == "Fulmar"
    assert src == "stored"

    # Case 2: Resolved from well_df ZONES curve with depth in meters
    wdf_m = pd.DataFrame({
        "DEPTH": np.linspace(2400.0, 2500.0, 101),
        "ZONES": ["Fulmar"] * 60 + ["Skagerrak"] * 41,
    })
    z_m, src_m = _resolve_fpt_point_zone(2420.0, None, wdf_m, [])
    assert z_m == "Fulmar"
    assert src_m == "well_logs"

    # Case 3: md in feet (7939.6 ft ~ 2420 m) while well_df is in meters
    z_ft, src_ft = _resolve_fpt_point_zone(7939.6, None, wdf_m, [])
    assert z_ft == "Fulmar"
    assert src_ft == "well_logs"

    # Case 4: Fallback to formation tops with unit alignment (md in ft, tops in m)
    tops_m = [
        {"name": "Fulmar", "depth": 2400.0},
        {"name": "Skagerrak", "depth": 2492.0},
    ]
    z_top, src_top = _resolve_fpt_point_zone(7940.0, None, None, tops_m)
    assert z_top == "Fulmar"
    assert src_top == "tops"


def test_zone_overrides_case_insensitive_and_fallback(fpt_client):
    pid, _ = fpt_client

    # 1. Zone override matching case-insensitively ('sand-a' matching 'Sand-A')
    res_ov = _analyze(
        pid,
        group_by_zone=True,
        overrides_by_zone={
            "sand-a": {
                "OIL": {
                    "density_gcc": 0.80,
                    "mode": "fixed",
                }
            }
        },
        n_simulations=100,
    )
    assert "Sand-A" in res_ov["by_zone"]
    sand_a_oil = res_ov["by_zone"]["Sand-A"]["gradients"]["OIL"]
    assert sand_a_oil["density_gcc"] == pytest.approx(0.80, abs=0.01)
    assert sand_a_oil["method"] == "fixed_gradient"

    # 2. Substring / alias zone override ('Sand-A Reservoir' matching 'Sand-A')
    res_alias_ov = _analyze(
        pid,
        group_by_zone=True,
        overrides_by_zone={
            "Sand-A Reservoir": {
                "OIL": {
                    "density_gcc": 0.72,
                    "mode": "fixed",
                }
            }
        },
        n_simulations=100,
    )
    assert "Sand-A" in res_alias_ov["by_zone"]
    sand_a_alias_oil = res_alias_ov["by_zone"]["Sand-A"]["gradients"]["OIL"]
    assert sand_a_alias_oil["density_gcc"] == pytest.approx(0.72, abs=0.01)


def test_tvdss_feet_unit_reconciliation_preventing_inflation(fpt_client):
    """Verify that TVDSS provided in feet (e.g. 8090 ft) for a metric well (MD ~ 2491.7 m)
    does not shoot to 26542 ft (~3.28x / 4x) when analyzed in display unit 'ft'.
    """
    pid, session = fpt_client
    w = Well(db_session=session, project_id=pid, name="21-25-2", uwi="21-25-2", depth_uom="m")
    w.save()

    # Pressure tests with MD in metres (2491.7 m) and TVDSS stored in feet (8090.0 ft)
    w.add_pressure_tests(
        [
            {"depth": 2491.7, "pressure": 5848.0, "tvdss": 8090.0, "fluid_type": "OIL"},
            {"depth": 2507.0, "pressure": 5863.0, "tvdss": 8139.4, "fluid_type": "OIL"},
            {"depth": 2516.3, "pressure": 5869.0, "tvdss": 8164.4, "fluid_type": "OIL"},
            {"depth": 2551.8, "pressure": 5892.0, "tvdss": 8287.5, "fluid_type": "OIL"},
        ]
    )
    session.commit()

    pts_res = fpt_service.fpt_list_points(pid, well_names=["21-25-2"])
    pts = pts_res["points"]
    p0 = next(p for p in pts if p["md"] == pytest.approx(2491.7, abs=0.1))
    # Stored canonical TVDSS should be converted to metres (~2465.8 m), NOT left at 8090 m
    assert p0["tvdss"] == pytest.approx(8090.0 * 0.3048, abs=0.5)

    # When analyzed in feet ('ft'):
    # TVDSS should display as 8090 ft, NOT shoot to 26542 ft!
    res_ft = _analyze(pid, well_names=["21-25-2"], depth_unit="ft", pressure_unit="psi", n_simulations=100)
    p0_ft = next(p for p in res_ft["points"] if p["md"] == pytest.approx(2491.7, abs=0.1))
    assert p0_ft["tvdss_display"] == pytest.approx(8090.0, abs=1.0)
    assert p0_ft["tvdss_display"] < 10000.0  # Must not shoot to ~26542 ft!

    # Oil gradient must be ~0.22 psi/ft (not 0.0711 psi/ft)
    oil_grad = res_ft["gradients"]["OIL"]["gradient"]
    assert 0.18 < oil_grad < 0.28
    # Oil density must be ~0.51 g/cc (not 0.164 g/cc)
    oil_rho = res_ft["gradients"]["OIL"]["density_gcc"]
    assert 0.45 < oil_rho < 0.65


def test_resolve_well_tvd_array_unit_alignment():
    """Verify _resolve_well_tvd_array reconciles when well_df TVDSS is in feet and target MD in metres."""
    # Well log dataframe with DEPTH in metres (~2500m) and TVDSS in feet (~8090ft)
    depths_m = np.linspace(2480.0, 2560.0, 81)
    tvdss_ft = depths_m * (1.0 / 0.3048) - 85.0  # ~8050 to 8315 ft
    well_df = pd.DataFrame({"DEPTH": depths_m, "TVDSS": tvdss_ft})

    target_md_m = np.array([2491.7, 2507.0, 2551.8])
    resolved_m = ws_service._resolve_well_tvd_array(target_md_m, well_df, None)

    # Resolved TVDSS must be in metres (~2465 m), not in feet (~8090 ft)
    assert np.all(resolved_m < 3000.0)
    assert np.all(resolved_m > 2400.0)
    expected_m = (target_md_m * (1.0 / 0.3048) - 85.0) * 0.3048
    assert resolved_m == pytest.approx(expected_m, abs=0.5)




