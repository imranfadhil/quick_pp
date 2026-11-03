from contextlib import contextmanager
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quick_pp.app.backend.services.ancillary import delete_fluid_contacts_project
from quick_pp.app.backend.services.workspace import (
    WorkspaceModuleParamsEntry,
    WorkspaceModuleParamsRequest,
    workspace_delete_project_module_params,
    workspace_delete_well_module_params,
    workspace_save_project_module_params,
    workspace_save_module_params,
)
from quick_pp.database.models import Base
from quick_pp.database.objects import Project, Well


@pytest.fixture
def test_db(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)

    @contextmanager
    def _get_test_db():
        session = Session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    monkeypatch.setattr("quick_pp.app.backend.services.ancillary.get_db", _get_test_db)
    monkeypatch.setattr("quick_pp.app.backend.services.workspace.get_db", _get_test_db)
    monkeypatch.setattr("quick_pp.app.backend.services.workspace.is_demo_mode", lambda: False)

    session = Session()
    proj = Project(db_session=session, name="test_reset_proj")
    proj.save()

    w1 = Well(db_session=session, project_id=proj.project_id, name="W-1", uwi="U-1")
    w1.save()

    # Add 1 manual contact and 2 auto contacts
    w1.add_fluid_contacts(
        [
            {
                "name": "OWC",
                "depth": 2400.0,
                "source": "Log Evaluation (Midpoint)",
                "contact_type": "OWC",
            },
            {
                "name": "ODT",
                "depth": 2390.0,
                "source": "Log Evaluation (Lowest proved oil)",
                "contact_type": "ODT",
            },
            {
                "name": "Manual_FWL",
                "depth": 2410.0,
                "source": "Handover Import",
                "contact_type": "FWL",
            },
        ]
    )

    yield {"session": session, "project_id": proj.project_id, "well_name": w1.name}


def test_delete_auto_fluid_contacts(test_db):
    pid = test_db["project_id"]

    # Delete auto-only contacts
    res = delete_fluid_contacts_project(project_id=pid, auto_only=True)
    assert res["deleted_count"] == 2
    assert "W-1" in res["wells_affected"]

    # Check remaining contacts in well
    session = test_db["session"]
    proj = Project.load(session, project_id=pid)
    w = proj.get_well("W-1")
    remaining = w.get_fluid_contacts().to_dict(orient="records")
    assert len(remaining) == 1
    assert remaining[0]["name"] == "Manual_FWL"


def test_project_module_params_delete(test_db):
    pid = test_db["project_id"]

    # Save 2 entries
    req = WorkspaceModuleParamsRequest(
        entries=[
            WorkspaceModuleParamsEntry(
                module="rock-typing",
                key="all",
                selection={"zones": None},
                label="All zones",
                params={"num_clusters": 3},
            ),
            WorkspaceModuleParamsEntry(
                module="swirr-corr",
                key="zone:A",
                selection={"zones": ["A"]},
                label="Zone A",
                params={"referencePcPsi": 100.0},
            ),
        ]
    )
    saved = workspace_save_project_module_params(pid, req)
    assert "rock-typing" in saved["module_params"]
    assert "swirr-corr" in saved["module_params"]

    # Delete specific module
    del_res = workspace_delete_project_module_params(pid, module="rock-typing")
    assert "rock-typing" not in del_res["module_params"]
    assert "swirr-corr" in del_res["module_params"]

    # Delete all remaining
    del_all = workspace_delete_project_module_params(pid)
    assert del_all["module_params"] == {}


def test_well_module_params_delete(test_db):
    pid = test_db["project_id"]
    wn = test_db["well_name"]

    req = WorkspaceModuleParamsRequest(
        entries=[
            WorkspaceModuleParamsEntry(
                module="litho-poro",
                key="well",
                selection={"zones": None},
                label="Whole well",
                params={"model": "ssc"},
            ),
            WorkspaceModuleParamsEntry(
                module="saturation",
                key="zone:A",
                selection={"zones": ["A"]},
                label="Zone A",
                params={"m": 2.0, "n": 2.0},
            ),
        ]
    )
    saved = workspace_save_module_params(pid, wn, req)
    assert "litho-poro" in saved["module_params"]
    assert "saturation" in saved["module_params"]

    # Delete saturation
    del_res = workspace_delete_well_module_params(pid, wn, module="saturation")
    assert "saturation" not in del_res["module_params"]
    assert "litho-poro" in del_res["module_params"]
