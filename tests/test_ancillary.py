from uuid import uuid4

from fastapi import FastAPI
from fastapi.testclient import TestClient

from quick_pp.app.backend.router import api_router

app = FastAPI()
app.include_router(api_router)

client = TestClient(app)


def test_well_delete_clears_tops_and_surveys_before_reupload():
    project_name = f"WellDeleteAncillary-{uuid4().hex}"
    well_name = f"Well-{uuid4().hex}"
    uwi = f"UWI-{uuid4().hex}"
    project_res = client.post(
        "/quick_pp/database/projects", json={"name": project_name}
    )
    assert project_res.status_code == 200
    pid = project_res.json()["project_id"]

    well_res = client.post(
        f"/quick_pp/database/projects/{pid}/wells",
        json={"name": well_name, "uwi": uwi},
    )
    assert well_res.status_code == 200

    tops_res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/formation_tops",
        json={"tops": [{"name": "Top-A", "depth": 1000.0}]},
    )
    assert tops_res.status_code == 200
    surveys_res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/well_surveys",
        json={"surveys": [{"md": 1000.0, "inc": 15.0, "azim": 90.0}]},
    )
    assert surveys_res.status_code == 200

    delete_res = client.delete(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}"
    )
    assert delete_res.status_code == 200

    reupload_res = client.post(
        f"/quick_pp/database/projects/{pid}/wells",
        json={"name": well_name, "uwi": uwi},
    )
    assert reupload_res.status_code == 200

    tops_after = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/formation_tops"
    )
    surveys_after = client.get(
        f"/quick_pp/database/projects/{pid}/well_surveys",
        params={"well_name": well_name},
    )
    assert tops_after.status_code == 200
    assert tops_after.json()["tops"] == []
    assert surveys_after.status_code == 200
    assert surveys_after.json()["well_surveys"] == []


def test_formation_tops_lifecycle():
    # Create a project
    res = client.post("/quick_pp/database/projects", json={"name": "TestProj"})
    assert res.status_code == 200
    project = res.json()
    pid = project["project_id"]

    # Create a well
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells", json={"name": "Well-1", "uwi": "U1"}
    )
    assert res.status_code == 200

    # Add tops
    tops = {
        "tops": [{"name": "Top-A", "depth": 1000}, {"name": "Top-B", "depth": 1100}]
    }
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/Well-1/formation_tops", json=tops
    )
    assert res.status_code == 200

    # List tops
    res = client.get(f"/quick_pp/database/projects/{pid}/wells/Well-1/formation_tops")
    assert res.status_code == 200
    data = res.json()
    assert len(data.get("tops", [])) >= 2

    # Delete a top
    res = client.delete(
        f"/quick_pp/database/projects/{pid}/wells/Well-1/formation_tops/Top-A"
    )
    assert res.status_code == 200

    # Verify deletion
    res = client.get(f"/quick_pp/database/projects/{pid}/wells/Well-1/formation_tops")
    assert res.status_code == 200
    tops_after = res.json().get("tops", [])
    assert not any(t["name"] == "Top-A" for t in tops_after)


def test_preview_and_core_samples():
    res = client.post("/quick_pp/database/projects", json={"name": "Proj2"})
    assert res.status_code == 200
    pid = res.json()["project_id"]
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells", json={"name": "W2", "uwi": "U2"}
    )
    assert res.status_code == 200

    # Test CSV preview upload
    csv_content = "name,depth\nTop1,100.5\nTop2,200.0\n"
    files = {"file": ("tops.csv", csv_content)}
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W2/formation_tops/preview",
        files=files,
    )
    assert res.status_code == 200
    data = res.json()
    assert "preview" in data and len(data["preview"]) == 2

    # Test core sample add and get
    payload = {
        "sample_name": "Plug-1",
        "depth": 2500.5,
        "measurements": [{"property_name": "POR", "value": 0.21, "unit": "v/v"}],
        "relperm_data": [{"saturation": 0.1, "kr": 0.01, "phase": "water"}],
        "pc_data": [{"saturation": 0.1, "pressure": 12.3}],
    }
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W2/core_samples", json=payload
    )
    assert res.status_code == 200

    # list core samples
    res = client.get(f"/quick_pp/database/projects/{pid}/wells/W2/core_samples")
    assert res.status_code == 200
    cs = res.json().get("core_samples", [])
    assert any(s["sample_name"] == "Plug-1" for s in cs)

    # get sample details
    res = client.get(f"/quick_pp/database/projects/{pid}/wells/W2/core_samples/Plug-1")
    assert res.status_code == 200
    sample = res.json().get("core_sample")
    assert sample and sample["sample_name"] == "Plug-1"


def test_ancillary_bulk_replace_and_delete_all():
    res = client.post("/quick_pp/database/projects", json={"name": "BulkDeleteProj"})
    assert res.status_code == 200
    pid = res.json()["project_id"]
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells", json={"name": "W-Bulk", "uwi": "UB"}
    )
    assert res.status_code == 200

    # 1. Formation tops: add 3, delete 2 via replace=True, then delete all
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/formation_tops",
        json={
            "tops": [
                {"name": "T1", "depth": 1000.0},
                {"name": "T2", "depth": 1100.0},
                {"name": "T3", "depth": 1200.0},
            ],
            "replace": True,
        },
    )
    assert res.status_code == 200
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/formation_tops",
        json={"tops": [{"name": "T2", "depth": 1100.0}], "replace": True},
    )
    assert res.status_code == 200
    tops_list = client.get(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/formation_tops"
    ).json()["tops"]
    assert len(tops_list) == 1 and tops_list[0]["name"] == "T2"

    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/formation_tops",
        json={"tops": [], "replace": True},
    )
    assert res.status_code == 200
    assert (
        client.get(
            f"/quick_pp/database/projects/{pid}/wells/W-Bulk/formation_tops"
        ).json()["tops"]
        == []
    )

    # 2. Fluid contacts: add 2, then clear all via replace=True
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/fluid_contacts",
        json={
            "contacts": [
                {"name": "GOC", "depth": 1050.0},
                {"name": "OWC", "depth": 1150.0},
            ],
            "replace": True,
        },
    )
    assert res.status_code == 200
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/fluid_contacts",
        json={"contacts": [], "replace": True},
    )
    assert res.status_code == 200
    assert (
        client.get(
            f"/quick_pp/database/projects/{pid}/wells/W-Bulk/fluid_contacts"
        ).json()["fluid_contacts"]
        == []
    )

    # 3. Pressure tests: add 2, then clear all via replace=True
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/pressure_tests",
        json={
            "tests": [
                {"depth": 1010.0, "pressure": 3100.0},
                {"depth": 1020.0, "pressure": 3120.0},
            ],
            "replace": True,
        },
    )
    assert res.status_code == 200
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/W-Bulk/pressure_tests",
        json={"tests": [], "replace": True},
    )
    assert res.status_code == 200
    assert (
        client.get(
            f"/quick_pp/database/projects/{pid}/wells/W-Bulk/pressure_tests"
        ).json()["pressure_tests"]
        == []
    )


def test_robust_unique_constraints():
    """Verify that composite unique constraints support real-world petrophysical data:
    1. Formation tops with repeated names at different depths (fault repeat).
    2. Fluid contacts with same contact type (OWC) in different zones.
    3. Pressure tests with multiple pre-tests at the exact same depth station.
    4. Core samples with same plug number in different cores (different depths).
    5. Core samples with different names (e.g. horiz/vert) at the same depth.
    6. Capillary pressure with multiple pressure steps at irreducible saturation plateau.
    """
    res = client.post("/quick_pp/database/projects", json={"name": "RobustUniqueProj"})
    assert res.status_code == 200
    pid = res.json()["project_id"]

    well_name = "Well-Robust"
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells",
        json={"name": well_name, "uwi": "UWI-ROBUST-1"},
    )
    assert res.status_code == 200

    # 1. Formation Tops: same name at 1000m and 1500m
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/formation_tops",
        json={
            "tops": [
                {"name": "Heimdal", "depth": 1000.0},
                {"name": "Heimdal", "depth": 1500.0},
            ],
            "replace": False,
        },
    )
    assert res.status_code == 200
    tops_res = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/formation_tops"
    ).json().get("tops", [])
    heimdal_depths = sorted([t["depth"] for t in tops_res if t["name"] == "Heimdal"])
    assert len(heimdal_depths) == 2
    assert abs(heimdal_depths[0] - 1000.0) < 0.1
    assert abs(heimdal_depths[1] - 1500.0) < 0.1

    # 2. Fluid Contacts: OWC in Zone-1 and OWC in Zone-2
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/fluid_contacts",
        json={
            "contacts": [
                {"name": "OWC", "depth": 1200.0, "zone": "Zone-1", "contact_type": "OWC"},
                {"name": "OWC", "depth": 1400.0, "zone": "Zone-2", "contact_type": "OWC"},
            ],
            "replace": False,
        },
    )
    assert res.status_code == 200
    contacts_res = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/fluid_contacts"
    ).json().get("fluid_contacts", [])
    owc_depths = sorted([c["depth"] for c in contacts_res if "OWC" in c["name"]])
    assert len(owc_depths) == 2

    # 3. Pressure Tests: Repeat pre-tests at 2000.0m with 3000 psi and 3050 psi
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/pressure_tests",
        json={
            "tests": [
                {"depth": 2000.0, "pressure": 3000.0, "test_quality": "VALID"},
                {"depth": 2000.0, "pressure": 3050.0, "test_quality": "SUPERCHARGED"},
            ],
            "replace": False,
        },
    )
    assert res.status_code == 200
    tests_res = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/pressure_tests"
    ).json().get("pressure_tests", [])
    tests_at_2000 = [t for t in tests_res if abs(t["depth"] - 2000.0) < 0.1]
    assert len(tests_at_2000) == 2
    pressures_at_2000 = sorted([t["pressure"] for t in tests_at_2000])
    assert abs(pressures_at_2000[0] - 3000.0) < 1.0
    assert abs(pressures_at_2000[1] - 3050.0) < 1.0

    # 4. Core Samples: Plug '1' in Core 1 (1500m) and Plug '1' in Core 2 (2500m)
    res1 = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples",
        json={
            "sample_name": "Plug-1",
            "depth": 1500.0,
            "measurements": [{"property_name": "CPORE", "value": 0.22, "unit": "v/v"}],
        },
    )
    assert res1.status_code == 200
    res2 = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples",
        json={
            "sample_name": "Plug-1",
            "depth": 2500.0,
            "measurements": [{"property_name": "CPORE", "value": 0.18, "unit": "v/v"}],
        },
    )
    assert res2.status_code == 200

    # 5. Core Samples: Vertical plug at the same depth 1500m
    res3 = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples",
        json={
            "sample_name": "Plug-1V",
            "depth": 1500.0,
            "measurements": [{"property_name": "CPERMV", "value": 15.0, "unit": "mD"}],
        },
    )
    assert res3.status_code == 200

    samples_res = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples"
    ).json().get("core_samples", [])
    assert len(samples_res) >= 3

    # 6. Capillary Pressure: Plateau at Sw = 0.12 across multiple pressure steps (50 & 100 psi)
    res_pc = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/scal_pc",
        json={
            "well_name": well_name,
            "sample_name": "Plug-1",
            "depth": 1500.0,
            "pc_data": [
                {"saturation": 0.12, "pressure": 50.0, "experiment_type": "micp", "cycle": "drainage"},
                {"saturation": 0.12, "pressure": 100.0, "experiment_type": "micp", "cycle": "drainage"},
            ],
            "append": True,
        },
    )
    assert res_pc.status_code == 200


def test_ancillary_commit_import_modes():
    """Verify that commit_ancillary_data honors 'merge' (default) and 'replace' import modes."""
    from uuid import uuid4

    pname = f"ImportModeProj-{uuid4().hex}"
    res = client.post("/quick_pp/database/projects", json={"name": pname})
    assert res.status_code == 200
    pid = res.json()["project_id"]

    well_name = "Well-ImportMode"
    res = client.post(
        f"/quick_pp/database/projects/{pid}/wells",
        json={"name": well_name, "uwi": f"UWI-{uuid4().hex}"},
    )
    assert res.status_code == 200

    # 1. Initial RCA import (samples at shallow depth 1000m and 1005m)
    res_rca_1 = client.post(
        "/quick_pp/import/ancillary/commit",
        json={
            "project_id": pid,
            "well_name": well_name,
            "data_type": "rca",
            "import_mode": "merge",
            "column_mappings": [
                {"source_column": "sample", "target_column": "SAMPLE_NAME"},
                {"source_column": "depth", "target_column": "DEPTH"},
                {"source_column": "poro", "target_column": "CPORE"},
            ],
            "rows": [
                {"sample": "S-1", "depth": 1000.0, "poro": 0.20},
                {"sample": "S-2", "depth": 1005.0, "poro": 0.25},
            ],
        },
    )
    assert res_rca_1.status_code == 200
    assert res_rca_1.json()["import_mode"] == "merge"

    # Verify initial samples
    samples = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples"
    ).json().get("core_samples", [])
    assert len(samples) == 2
    depths = sorted([s["depth"] for s in samples])
    assert abs(depths[0] - 1000.0) < 0.1
    assert abs(depths[1] - 1005.0) < 0.1

    # 2. Second RCA import with deeper depth (1010m) and updated 1000m sample using default "merge" mode
    # Verifies user's concern: new data from deeper depths must NOT overwrite shallower depths!
    res_rca_2 = client.post(
        "/quick_pp/import/ancillary/commit",
        json={
            "project_id": pid,
            "well_name": well_name,
            "data_type": "rca",
            "import_mode": "merge",
            "column_mappings": [
                {"source_column": "sample", "target_column": "SAMPLE_NAME"},
                {"source_column": "depth", "target_column": "DEPTH"},
                {"source_column": "poro", "target_column": "CPORE"},
            ],
            "rows": [
                {"sample": "S-1", "depth": 1000.0, "poro": 0.22},
                {"sample": "S-3", "depth": 1010.0, "poro": 0.30},
            ],
        },
    )
    assert res_rca_2.status_code == 200

    # Verify all 3 samples are present: shallow (1000m, 1005m) preserved and deeper (1010m) appended!
    samples_merged = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples"
    ).json().get("core_samples", [])
    assert len(samples_merged) == 3
    merged_depths = sorted([s["depth"] for s in samples_merged])
    assert abs(merged_depths[0] - 1000.0) < 0.1
    assert abs(merged_depths[1] - 1005.0) < 0.1
    assert abs(merged_depths[2] - 1010.0) < 0.1

    # 3. Third RCA import with "replace" mode (only sample at 2000m)
    res_rca_3 = client.post(
        "/quick_pp/import/ancillary/commit",
        json={
            "project_id": pid,
            "well_name": well_name,
            "data_type": "rca",
            "import_mode": "replace",
            "column_mappings": [
                {"source_column": "sample", "target_column": "SAMPLE_NAME"},
                {"source_column": "depth", "target_column": "DEPTH"},
                {"source_column": "poro", "target_column": "CPORE"},
            ],
            "rows": [
                {"sample": "S-New", "depth": 2000.0, "poro": 0.15},
            ],
        },
    )
    assert res_rca_3.status_code == 200

    # In "replace" mode, previous samples are wiped and only the new one remains
    samples_replaced = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples"
    ).json().get("core_samples", [])
    assert len(samples_replaced) == 1
    assert abs(samples_replaced[0]["depth"] - 2000.0) < 0.1


def test_rca_import_with_qv_without_cperm():
    res = client.post("/quick_pp/database/projects", json={"name": "QvProj"})
    assert res.status_code == 200
    pid = res.json()["project_id"]
    well_name = "QvWell"
    res_w = client.post(
        f"/quick_pp/database/projects/{pid}/wells",
        json={"name": well_name, "uwi": "QV-1"},
    )
    assert res_w.status_code == 200

    # Import RCA sample with QV, CPORE, but NO CPERM
    res_rca = client.post(
        "/quick_pp/import/ancillary/commit",
        json={
            "project_id": pid,
            "well_name": well_name,
            "data_type": "rca",
            "import_mode": "replace",
            "column_mappings": [
                {"source_column": "sample", "target_column": "SAMPLE_NAME"},
                {"source_column": "depth", "target_column": "DEPTH"},
                {"source_column": "poro", "target_column": "CPORE"},
                {"source_column": "qv_core", "target_column": "QV"},
            ],
            "rows": [
                {"sample": "Plug-Qv-1", "depth": 1550.0, "poro": 0.22, "qv_core": 0.35},
            ],
        },
    )
    assert res_rca.status_code == 200

    # Verify detail endpoint
    res_detail = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples/Plug-Qv-1"
    )
    assert res_detail.status_code == 200
    detail = res_detail.json().get("core_sample")
    assert detail is not None
    meas = detail.get("measurements", [])
    qv_meas = next((m for m in meas if m.get("property") == "QV" or m.get("property_name") == "QV"), None)
    assert qv_meas is not None
    assert abs(qv_meas["value"] - 0.35) < 1e-4

    # Verify core samples list endpoint
    list_res = client.get(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/core_samples"
    )
    assert list_res.status_code == 200
    samples = list_res.json().get("core_samples", [])
    assert len(samples) == 1
    assert samples[0]["sample_name"] == "Plug-Qv-1"
    assert abs(samples[0]["depth"] - 1550.0) < 1e-4


def test_calculate_tvd_endpoints_routing():
    project_name = f"TVDTestProj-{uuid4().hex}"
    well_name = f"Well-{uuid4().hex}"
    uwi = f"UWI-{uuid4().hex}"

    # Create project and well
    p_res = client.post("/quick_pp/database/projects", json={"name": project_name})
    assert p_res.status_code == 200
    pid = p_res.json()["project_id"]

    w_res = client.post(
        f"/quick_pp/database/projects/{pid}/wells",
        json={"name": well_name, "uwi": uwi},
    )
    assert w_res.status_code == 200

    # Add well surveys
    surveys = [
        {"md": 0.0, "inc": 0.0, "azim": 0.0},
        {"md": 500.0, "inc": 2.0, "azim": 45.0},
        {"md": 1000.0, "inc": 10.0, "azim": 90.0},
    ]
    s_res = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/well_surveys",
        json={"surveys": surveys},
    )
    assert s_res.status_code == 200

    # 1. Test canonical endpoint: /quick_pp/database/projects/{pid}/well_surveys/calculate_tvd?well_name=...
    res1 = client.post(
        f"/quick_pp/database/projects/{pid}/well_surveys/calculate_tvd",
        params={"well_name": well_name},
    )
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["success"] is True
    assert data1["wells_processed"] == 1

    # 2. Test well path endpoint: /quick_pp/database/projects/{pid}/wells/{well_name}/well_surveys/calculate_tvd
    res2 = client.post(
        f"/quick_pp/database/projects/{pid}/wells/{well_name}/well_surveys/calculate_tvd"
    )
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["success"] is True

    # 3. Test compatibility endpoint: /quick_pp/well_surveys/calculate_tvd?project_id=...&well_name=...
    res3 = client.post(
        "/quick_pp/well_surveys/calculate_tvd",
        params={"project_id": pid, "well_name": well_name},
    )
    assert res3.status_code == 200
    data3 = res3.json()
    assert data3["success"] is True



