import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quick_pp.app.backend.main import app
from quick_pp.app.backend.services import database as db_service
from quick_pp.app.backend.services import rock_typing as rt_service
from quick_pp.app.backend.services import workspace as ws_service
from quick_pp.database.models import Base
from quick_pp.database.objects import Project, Well
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool


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
            "WELL_NAME": ["Well-A"] * 10,
            "GR": np.linspace(30.0, 110.0, 10),
            "NPHI": np.linspace(0.12, 0.28, 10),
            "RHOB": np.linspace(2.25, 2.55, 10),
            "RT": np.linspace(25.0, 4.0, 10),
            "ZONES": ["Zone-1"] * 5 + ["Zone-2"] * 5,
            "CPORE": [0.22, np.nan, 0.19, np.nan, 0.17, np.nan, 0.15, np.nan, 0.13, np.nan],
            "CPERM": [150.0, np.nan, 80.0, np.nan, 40.0, np.nan, 15.0, np.nan, 5.0, np.nan],
        }
    )
    well_a = Well(db_session=session, project_id=proj.project_id, name="Well-A", uwi="WELL-A-1")
    well_a.save()
    proj.update_data(df)
    proj.save()

    well_a.add_formation_tops(
        [{"name": "Zone-1", "depth": 1000.0}, {"name": "Zone-2", "depth": 1005.0}]
    )
    session.commit()

    # Also add an empty well to verify multi-well endpoints skip empty wells gracefully
    well_empty = Well(db_session=session, project_id=proj.project_id, name="Well-Empty", uwi="EMPTY-1")
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

    client = TestClient(app)
    yield client, proj.project_id
    session.close()


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
    r_bundle = client.get(
        f"/quick_pp/workspace/projects/{pid}/wells/Well-A/bundle"
    )
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
        json={"model": "ssc", "use_hc_corrected": True, "save": True},
    )
    assert r_lith.status_code == 200
    lith_body = r_lith.json()
    assert lith_body["saved_rows"] == 10
    assert "VSAND" in lith_body["data"]
    assert "VCLAY" in lith_body["data"]

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

    # 5. Saturation (single-pass Archie + Waxman-Smits, save=True)
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

    # 6. Rock Typing across project (including empty well) & Perm Transform
    r_rt = client.post(
        f"/quick_pp/workspace/projects/{pid}/rock_typing/compute",
        json={"cutoffs": [0.5, 1.5, 3.0], "save": True},
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


def test_petrophysical_mcp_server_and_agent_assistant(client_and_project):
    client, pid = client_and_project

    # 1. Provider catalog & MCP tools listing
    r_prov = client.get("/quick_pp/assistant/providers")
    assert r_prov.status_code == 200
    prov_body = r_prov.json()
    provider_ids = [p["id"] for p in prov_body["providers"]]
    assert "native_mcp" in provider_ids
    assert "openai" in provider_ids
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
    assert len(r_list.json()["result"]["tools"]) == 6

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
    for p10, pred, p90 in zip(curves["PHIT_P10"], curves["PHIT"], curves["PHIT_P90"]):
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



