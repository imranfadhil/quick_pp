from fastapi.testclient import TestClient

from quick_pp.app.backend.router import api_router
from fastapi import FastAPI


app = FastAPI()
app.include_router(api_router)

client = TestClient(app)


def test_formation_tops_lifecycle(tmp_path):
    # Initialize DB connector to SQLite in tmp path
    db_file = tmp_path / "test_qpp.db"
    res = client.post(
        "/quick_pp/database/init",
        json={"db_url": f"sqlite:///{db_file}", "setup": True},
    )
    assert res.status_code == 200

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


def test_preview_and_core_samples(tmp_path):
    # reuse same test app; create project and well
    db_file = tmp_path / "test_qpp2.db"
    res = client.post(
        "/quick_pp/database/init",
        json={"db_url": f"sqlite:///{db_file}", "setup": True},
    )
    assert res.status_code == 200

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
