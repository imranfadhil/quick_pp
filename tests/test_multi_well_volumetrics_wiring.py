from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from quick_pp.app.backend.services import database as db_service
from quick_pp.app.backend.services import workspace as ws_service
from quick_pp.app.backend.services.workspace import WorkspaceMultiWellVolumetricsRequest
from quick_pp.app.backend.utils import db as util_db
from quick_pp.database.models import Base
from quick_pp.database.objects import Project


@pytest.fixture
def test_setup(monkeypatch):
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    proj = Project(db_session=session, name="volumetrics_test_proj")
    proj.save()

    depths = np.linspace(2400.0, 2450.0, 50)
    # Well with interpreted curves (VCLAY, PHIT, SWT, SHF, PERM_TRANS)
    df = pd.DataFrame(
        {
            "DEPTH": depths,
            "TVD": depths,
            "TVDSS": depths,
            "WELL_NAME": ["Well-1"] * len(depths),
            "VCLAY": [0.15] * len(depths),
            "PHIT": [0.22] * len(depths),
            "SWT": [0.35] * len(depths),
            "SHF": [0.25] * len(depths),
            "PERM_TRANS": [150.0] * len(depths),
            "ZONES": ["West Phalcon B"] * len(depths),
        }
    )
    proj.update_data(df, group_by="WELL_NAME")
    proj.save()

    @contextmanager
    def mock_get_db():
        yield session

    monkeypatch.setattr(db_service, "get_db", mock_get_db)
    monkeypatch.setattr(ws_service, "get_db", mock_get_db)
    monkeypatch.setattr(util_db, "get_db", mock_get_db)
    monkeypatch.setattr(ws_service, "_load_multi_well_dfs", lambda *a, **kw: [df.copy()])

    return proj.project_id


def test_volumetrics_inherit_shf_contact_and_exported_format(test_setup):
    """Verify contact pickup, 4 rows (ALL, ROCK, RES, PAY), PHIT, PERM, and SW_SHF."""
    project_id = test_setup

    payload = {
        "cut_offs": {"VSHALE": 0.40, "PHIT": 0.05, "SWT": 0.80},
        "mode": "area",
        "compute_oil": False,
        "compute_gas": True,
        "contact_source": "inherit",
        "reservoir_type": "gas_only",
        "owc": None,
        "goc": None,
        "gwc": None,
        "contacts_by_zone": {},
        "saturation_method": "shf",
        "fwl_uncertainty": 5.0,
        "shf_params": {
            "fwl": 2430.0,
            "fwl_by_zone": {"West Phalcon B": 2425.0},
            "fluid_system": "gas_water",
            "a": 0.35,
            "b": -0.45,
            "swirr": 0.12,
        },
        "zones": ["West Phalcon B"],
        "n_samples": 500,
        "seed": 42,
    }

    req = WorkspaceMultiWellVolumetricsRequest(**payload)
    data = ws_service.workspace_multi_well_volumetrics(project_id, req)

    # 1. Verify contact inheritance
    contacts = data.get("contacts", {})
    assert contacts.get("source") == "Inherited from Step 5 SHF"
    assert contacts.get("gwc") in (2425.0, 2430.0) or contacts.get("fwl") == 2430.0

    # 2. Verify well summary contains 4 rows per zone (ALL, ROCK, RES, PAY)
    well_sum = data.get("well_summary", [])
    assert len(well_sum) == 4, f"Expected 4 cutoff rows, got {len(well_sum)}"

    flags = [r["flag"] for r in well_sum]
    assert flags == ["ALL", "ROCK", "RES", "PAY"]

    # 3. Verify single net and ntg columns exist and are consistent
    for r in well_sum:
        assert "net" in r
        assert "ntg" in r
        assert r["net"] > 0
        assert 0.0 <= r["ntg"] <= 1.0

    # 4. Verify PHIT is used
    for r in well_sum:
        assert "phit" in r
        assert r["phit"] > 0

    # 5. Verify PERM averaging is not zero
    for r in well_sum:
        assert "perm_geom" in r
        assert r["perm_geom"] > 0, f"Expected non-zero perm_geom for {r['flag']}, got {r['perm_geom']}"

    # 6. Verify sw_shf is populated when SHF is available
    for r in well_sum:
        assert "sw_shf" in r
        assert r["sw_shf"] is not None
        assert 0.0 < r["sw_shf"] <= 1.0

    # Verify curves used
    pay_row = [r for r in well_sum if r["flag"] == "PAY"][0]
    assert pay_row["curves_used"]["vsh"] == "VCLAY"
    assert pay_row["curves_used"]["phi"] == "PHIT"
    assert pay_row["curves_used"]["sw"] == "SHF"
    assert pay_row["curves_used"]["perm"] == "PERM_TRANS"


def test_volumetrics_log_average_curve_pickup(test_setup):
    """Verify that log_average saturation method picks up SWT and standard curves."""
    project_id = test_setup

    payload = {
        "cut_offs": {"VSHALE": 0.40, "PHIT": 0.05, "SWT": 0.80},
        "mode": "area",
        "compute_oil": True,
        "compute_gas": True,
        "contact_source": "inherit",
        "reservoir_type": "auto",
        "contacts_by_zone": {},
        "saturation_method": "log_average",
        "shf_params": {
            "fwl": 2440.0,
            "fluid_system": "oil_water",
        },
        "zones": ["West Phalcon B"],
        "n_samples": 500,
        "seed": 42,
    }

    req = WorkspaceMultiWellVolumetricsRequest(**payload)
    data = ws_service.workspace_multi_well_volumetrics(project_id, req)

    well_sum = data.get("well_summary", [])
    assert len(well_sum) == 4
    pay_row = [r for r in well_sum if r["flag"] == "PAY"][0]
    assert pay_row["curves_used"]["sw"] == "SWT"
    assert pay_row["net"] > 0
    assert pay_row["perm_geom"] > 0


def test_cutoffs_sensitivity_endpoint_and_return(test_setup):
    """Verify cutoffs_analysis returns sensitivity data and optimal cutoffs."""
    project_id = test_setup
    req = ws_service.WorkspaceMultiWellCutoffsSensitivityRequest(percentile=95.0)
    sens = ws_service.workspace_multi_well_cutoffs_sensitivity(project_id, req)
    assert "optimal_cutoffs" in sens
    assert "vsh_sensitivity" in sens
    assert "phi_sensitivity" in sens
    assert "sw_sensitivity" in sens
    assert sens["optimal_cutoffs"]["vshale"] > 0
    assert sens["optimal_cutoffs"]["phit"] > 0
    assert sens["optimal_cutoffs"]["swt"] > 0
    assert len(sens["vsh_sensitivity"]) > 0
    assert len(sens["phi_sensitivity"]) > 0
    assert len(sens["sw_sensitivity"]) > 0


def test_single_well_cutoffs_sensitivity_endpoint(test_setup, monkeypatch):
    """Verify single-well cutoffs sensitivity endpoint returns PRMS optimal cutoffs and sweep curves."""
    project_id = test_setup
    depths = np.linspace(2400.0, 2450.0, 50)
    well_df = pd.DataFrame(
        {
            "DEPTH": depths,
            "WELL_NAME": ["Well-1"] * len(depths),
            "VCLAY": [0.15] * len(depths),
            "PHIT": [0.22] * len(depths),
            "SWT": [0.35] * len(depths),
            "ZONES": ["Zone A"] * len(depths),
        }
    )
    monkeypatch.setattr(ws_service, "_get_filtered_well_df", lambda *a, **kw: well_df.copy())
    req = ws_service.WorkspaceWellCutoffsSensitivityRequest(percentile=95.0)
    sens = ws_service.workspace_well_cutoffs_sensitivity(project_id, "Well-1", req)

    assert "optimal_cutoffs" in sens
    assert "vsh_sensitivity" in sens
    assert "phi_sensitivity" in sens
    assert "sw_sensitivity" in sens
    assert sens["percentile"] == 95.0
    assert sens["optimal_cutoffs"]["vshale"] > 0
    assert sens["optimal_cutoffs"]["phit"] > 0
    assert sens["optimal_cutoffs"]["swt"] > 0
    assert len(sens["vsh_sensitivity"]) > 0
    assert len(sens["phi_sensitivity"]) > 0
    assert len(sens["sw_sensitivity"]) > 0

    # Also verify workspace_reservoir_summary automatically attaches cutoffs_sensitivity
    res_req = ws_service.WorkspaceReservoirSummaryRequest(
        cut_offs={"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8}
    )
    res_summary = ws_service.workspace_reservoir_summary(project_id, "Well-1", res_req)
    assert "cutoffs_sensitivity" in res_summary
    assert "optimal_cutoffs" in res_summary["cutoffs_sensitivity"]
    assert len(res_summary["cutoffs_sensitivity"]["vsh_sensitivity"]) > 0

