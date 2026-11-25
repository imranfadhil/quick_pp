from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quick_pp.app.backend.services.report import (
    REPORT_TYPE_PRESETS,
    SECTION_ID_ALIASES,
    SECTION_TEMPLATE_MAP,
    build_report_context,
    validate_fdp_report_payload,
)
from quick_pp.app.backend.services.report import (
    _jinja_env as jinja_env,
)


def test_phase2_section_registration_and_presets():
    """Verify Section 10 gate checks and presets registration."""
    assert "10_gate_checks" in SECTION_TEMPLATE_MAP
    assert "11_appendix_cpi" in SECTION_TEMPLATE_MAP
    assert SECTION_ID_ALIASES.get("gate_checks") == "10_gate_checks"
    assert SECTION_ID_ALIASES.get("appendix_cpi") == "11_appendix_cpi"
    assert "10_gate_checks" in REPORT_TYPE_PRESETS["FDP_FULL"]
    assert "11_appendix_cpi" in REPORT_TYPE_PRESETS["FDP_FULL"]


def test_phase2_gate_checks_enrichment_and_rendering():
    """Verify CHK-01 through CHK-08 enrichment and Section 10 template rendering."""
    full_df = pd.DataFrame(
        {
            "DEPTH": np.linspace(2800, 3000, 201),
            "ZONES": ["Target Sand"] * 201,
            "RT": np.linspace(10, 50, 201),
            "PHIE": np.full(201, 0.18),
            "PHIT": np.full(201, 0.22),
            "SWT": np.full(201, 0.35),
            "SHF": np.full(201, 0.36),
            "PERM": np.full(201, 150.0),
        }
    )

    ctx = {
        "_full_df": full_df,
        "target_zones": ["Target Sand"],
        "sw_params": {"fwl": 2940.0, "median_target_tvdss": 2940.0},
        "poro_params": {
            "core_count": 50,
            "total_core_count": 50,
            "core_r2": 0.75,
            "core_rmse": 0.025,
        },
        "uncertainty_summary": [
            {
                "zone": "Target Sand",
                "p90_hcpt": 1.5,
                "p50_hcpt": 2.5,
                "p10_hcpt": 3.8,
                "p50_phie": 0.18,
            }
        ],
        "shf_params": {"shf_rmse": 0.045},
        "fluid_contacts_rows": [
            {"well_name": "W-1", "contact_type": "FWL", "depth": 2940.0}
        ],
        "anomaly_flags": [
            {
                "category": "Core Parity Alert",
                "target": "W-1",
                "message": "Plugs unshifted",
                "mitigation": "Apply shift",
                "severity": "HIGH",
            }
        ],
    }

    checks = validate_fdp_report_payload(ctx)
    check_ids = [c["id"] for c in checks]
    assert "CHK-01" in check_ids
    assert "CHK-02" in check_ids
    assert "CHK-03" in check_ids
    assert "CHK-04" in check_ids
    assert "CHK-05" in check_ids
    assert "CHK-06" in check_ids
    assert "CHK-07" in check_ids
    assert "CHK-08" in check_ids

    # Verify all enriched fields exist
    for c in checks:
        assert "category" in c
        assert "criterion" in c
        assert "measured" in c
        assert "status" in c
        assert "action_note" in c

    # Render Section 10 template
    ctx["gate_checks"] = checks
    template = jinja_env.get_template(SECTION_TEMPLATE_MAP["10_gate_checks"])
    rendered = template.render(**ctx)
    assert "# 10. Petrophysical QA/QC Gate-Check & Peer-Review Audit Log" in rendered
    assert "Table 10.1" in rendered or "CHK-01" in rendered
    assert "CHK-07" in rendered
    assert "CHK-08" in rendered
    assert "Core Parity Alert" in rendered


def test_phase2_well_core_parity_diagnostics():
    """Verify per-well core parity calculations and conditional table rendering in Section 4 and 5."""
    # Synthetic well data with 2 wells: W-1 (aligned) and W-2 (misaligned)
    n = 50
    depths = np.linspace(2800, 2900, n)
    # Well 1: strong correlation
    cx1 = np.linspace(0.10, 0.25, n)
    phit1 = cx1 + 0.005
    # Well 2: low correlation / high residual
    cx2 = np.linspace(0.10, 0.25, n)
    phit2 = cx2[::-1] + 0.08

    df1 = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-01"] * n,
            "DEPTH": depths,
            "ZONES": ["Target Sand"] * n,
            "CPORE": cx1,
            "PHIT": phit1,
            "PHIE": phit1 - 0.02,
            "CPERM": 10 ** (10 * cx1),
            "PERM": 10 ** (10 * phit1),
            "SHIFT_M": [1.25] * n,
        }
    )
    df2 = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-02"] * n,
            "DEPTH": depths,
            "ZONES": ["Target Sand"] * n,
            "CPORE": cx2,
            "PHIT": phit2,
            "PHIE": phit2 - 0.02,
            "CPERM": 10 ** (10 * cx2),
            "PERM": 10 ** (10 * phit2),
            "SHIFT_M": [-2.0] * n,
        }
    )
    full_df = pd.concat([df1, df2], ignore_index=True)

    # Mock DB loading
    with (
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
        patch(
            "quick_pp.app.backend.services.report._load_project_wells_fast"
        ) as mock_wells,
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            side_effect=lambda pid, wn: full_df[full_df["WELL_NAME"] == wn],
        ),
    ):
        mock_wells.return_value = (
            "Test Project",
            [
                {
                    "id": 1,
                    "name": "WELL-01",
                    "uwi": "W01",
                    "core_sample_count": n,
                    "pressure_test_count": 0,
                    "tops": [],
                    "well_config": {},
                },
                {
                    "id": 2,
                    "name": "WELL-02",
                    "uwi": "W02",
                    "core_sample_count": n,
                    "pressure_test_count": 0,
                    "tops": [],
                    "well_config": {},
                },
            ],
        )
        session_mock = MagicMock()
        session_mock.execute.return_value.all.return_value = []
        mock_db.return_value.__enter__.return_value = session_mock

        ctx = build_report_context(
            project_id=1,
            target_zones=["Target Sand"],
            enabled_section_ids={"04_lithology_porosity", "05_rock_typing_perm"},
        )

    assert "well_core_parity" in ctx
    parity = ctx["well_core_parity"]
    assert len(parity) == 2
    w1_par = next(p for p in parity if p["well_name"] == "WELL-01")

    assert w1_par["n_plugs"] == n
    assert w1_par["r2_phit"] > 0.90
    assert w1_par["status"] == "ALIGNED"

    # Verify template rendering for section 04 and 05
    t4 = jinja_env.get_template(SECTION_TEMPLATE_MAP["04_lithology_porosity"])
    # Force multi-well core R2 < 0.40 to trigger conditional parity table
    ctx["poro_params"]["core_r2"] = 0.25
    ctx["is_single_well"] = False
    rend4 = t4.render(**ctx, assets={})
    assert "Well-by-Well Core-to-Log Porosity Parity" in rend4
    assert "WELL-01" in rend4
    assert "WELL-02" in rend4

    t5 = jinja_env.get_template(SECTION_TEMPLATE_MAP["05_rock_typing_perm"])
    ctx["poro_params"]["core_perm_r2"] = 0.25
    rend5 = t5.render(**ctx, assets={})
    assert "Well-by-Well Core-to-Log Permeability Parity" in rend5
    assert "WELL-01" in rend5


def test_phase2_spatial_zone_mapping_and_contact_summary():
    """Verify spatial zone mapping maps 'All Zones' contacts to intersecting formations."""
    n = 100
    # Well with known depth window
    full_df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-01"] * n,
            "DEPTH": np.linspace(2500, 2700, n),
            "TVD": np.linspace(2500, 2700, n),
            "TVDSS": np.linspace(2480, 2680, n),
            "ZONES": ["West Phalian B"] * n,
            "GR": np.full(n, 45.0),
            "RT": np.full(n, 20.0),
            "RHOB": np.full(n, 2.35),
            "NPHI": np.full(n, 0.15),
        }
    )

    with (
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
        patch(
            "quick_pp.app.backend.services.report._load_project_wells_fast"
        ) as mock_wells,
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            return_value=full_df,
        ),
    ):
        mock_wells.return_value = (
            "Spatial Zone Project",
            [
                {
                    "id": 1,
                    "name": "WELL-01",
                    "uwi": "W01",
                    "core_sample_count": 0,
                    "pressure_test_count": 0,
                    "tops": [],
                    "well_config": {},
                }
            ],
        )
        session_mock = MagicMock()
        # Mock committed fluid contacts with "All Zones" at depth 2550 m TVDss
        fc_record1 = MagicMock()
        fc_record1.well_id = 1
        fc_record1.name = "OWC"
        fc_record1.depth = 2550.0
        fc_record1.zone = "All Zones"
        fc_record1.contact_type = "OWC"
        fc_record1.fault_block = "Main Block"
        fc_record1.source = "MDT Pressures"
        fc_record1.remarks = "Discrete water gradient intersection"

        fc_record2 = MagicMock()
        fc_record2.well_id = 1
        fc_record2.name = "FWL"
        fc_record2.depth = 2560.0
        fc_record2.zone = "All Zones"
        fc_record2.contact_type = "FWL"
        fc_record2.fault_block = "Main Block"
        fc_record2.source = "Capillary model"
        fc_record2.remarks = ""

        session_mock.execute.return_value.all.return_value = [fc_record1, fc_record2]
        mock_db.return_value.__enter__.return_value = session_mock

        ctx = build_report_context(
            project_id=1,
            target_zones=["West Phalian B"],
            enabled_section_ids={"06_saturation_contacts"},
        )

    rows = ctx.get("fluid_contacts_rows", [])
    assert len(rows) == 2
    # Both records should be mapped to West Phalian B by TVDss intersection
    assert rows[0]["zone"] == "West Phalian B"
    assert "Mapped from 'All Zones' by TVDss intersection" in rows[0]["remarks"]
    assert rows[1]["zone"] == "West Phalian B"

    # Fluid contacts summary should not contain orphan "All Zones"
    summary = ctx.get("fluid_contacts_summary", [])
    zones_in_summary = [s["zone"] for s in summary]
    assert "All Zones" not in zones_in_summary
    assert "West Phalian B" in zones_in_summary


def test_phase2_multi_curve_badhole_and_norm_shifts():
    """Verify multi-curve bad-hole condition evaluation and delta_gr/delta_rhob/delta_nphi calculation."""
    n = 100
    # Construct curves with bad-hole conditions: CALI - BS > 2, abs(DRHO) > 0.15, RHOB < 1.95
    cali = np.full(n, 8.5)
    cali[10:25] = 11.0  # Washout (CALI - BS = 2.5 > 2.0)
    drho = np.zeros(n)
    drho[30:40] = 0.20  # Pad standoff
    rhob = np.full(n, 2.40)
    rhob[50:60] = 1.85  # Low density washout

    gr = np.full(n, 50.0)
    gr_norm = gr + 5.0  # +5 gAPI shift
    rhob_norm = rhob - 0.02  # -0.02 g/cc shift
    nphi = np.full(n, 0.20)
    nphi_norm = nphi + 0.015  # +0.015 v/v shift

    full_df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-QAQC"] * n,
            "DEPTH": np.linspace(2000, 2100, n),
            "ZONES": ["Target Sand"] * n,
            "CALI": cali,
            "BS": np.full(n, 8.5),
            "DRHO": drho,
            "RHOB": rhob,
            "RHOB_NORM": rhob_norm,
            "GR": gr,
            "GR_NORM": gr_norm,
            "NPHI": nphi,
            "NPHI_NORM": nphi_norm,
            "RT": np.full(n, 20.0),
        }
    )

    with (
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
        patch(
            "quick_pp.app.backend.services.report._load_project_wells_fast"
        ) as mock_wells,
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            return_value=full_df,
        ),
    ):
        mock_wells.return_value = (
            "QAQC Project",
            [
                {
                    "id": 1,
                    "name": "WELL-QAQC",
                    "uwi": "WQA",
                    "core_sample_count": 0,
                    "pressure_test_count": 0,
                    "tops": [],
                    "well_config": {},
                }
            ],
        )
        session_mock = MagicMock()
        session_mock.execute.return_value.all.return_value = []
        mock_db.return_value.__enter__.return_value = session_mock

        ctx = build_report_context(
            project_id=1,
            target_zones=["Target Sand"],
            enabled_section_ids={"03_qaqc"},
        )

    qaqc_table = ctx.get("qaqc_table", [])
    assert len(qaqc_table) == 1
    row = qaqc_table[0]
    # Total bad-hole samples: 15 (from CALI) + 10 (from DRHO) + 10 (from RHOB) = 35
    assert row["badhole_count"] == 35
    assert row["badhole_pct"] == 35.0
    assert row["delta_gr"] == "+5.0"
    assert row["delta_rhob"] == "-0.020"
    assert row["delta_nphi"] == "+0.015"

    qp = ctx.get("qaqc_params", {})
    assert "badhole_criteria_used" in qp
    assert "CALI - BS >" in qp["badhole_criteria_used"]


def test_phase2_raw_eval_curves_and_core_status():
    """Verify Table 2.1 receives separated raw logs vs evaluated curves and RCA/SCAL status."""
    n = 50
    full_df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-INV"] * n,
            "DEPTH": np.linspace(2000, 2050, n),
            "ZONES": ["Target Sand"] * n,
            "GR": np.full(n, 50.0),
            "RT": np.full(n, 20.0),
            "RHOB": np.full(n, 2.40),
            "NPHI": np.full(n, 0.20),
            "CALI": np.full(n, 8.5),
            "DT": np.full(n, 70.0),
            "SP": np.full(n, -40.0),
            "CPORE": np.full(n, 0.18),
            "CPERM": np.full(n, 50.0),
            "PC": np.full(n, 2.5),
            "KRW": np.full(n, 0.2),
        }
    )

    with (
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
        patch(
            "quick_pp.app.backend.services.report._load_project_wells_fast"
        ) as mock_wells,
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            return_value=full_df,
        ),
    ):
        mock_wells.return_value = (
            "Inventory Project",
            [
                {
                    "id": 1,
                    "name": "WELL-INV",
                    "uwi": "WINV",
                    "core_sample_count": 25,
                    "scal_sample_count": 10,
                    "pressure_test_count": 5,
                    "tops": [],
                    "well_config": {},
                }
            ],
        )
        session_mock = MagicMock()
        session_mock.execute.return_value.all.return_value = []
        mock_db.return_value.__enter__.return_value = session_mock

        ctx = build_report_context(
            project_id=1,
            target_zones=["Target Sand"],
            enabled_section_ids={"02_data_inventory"},
        )

    wells = ctx.get("wells", [])
    assert len(wells) == 1
    w = wells[0]
    assert "raw_curves" in w
    assert "eval_curves" in w
    assert "core_status" in w

    # Verify raw curves contain DT, SP, GR, RT, CALI and do NOT contain VCLAY, PHIT, etc.
    for raw_c in ["GR", "RT", "RHOB", "NPHI", "CALI", "DT", "SP"]:
        assert raw_c in w["raw_curves"]
    assert "VCLAY" not in w["raw_curves"]

    # Verify evaluated curves contain VCLAY, PHIT, PERM, SWT
    for ev_c in ["VCLAY", "PHIT", "PERM", "SWT"]:
        assert ev_c in w["eval_curves"]

    # Verify core_status
    assert "RCA: 50 pts" in w["core_status"]
    assert "SCAL: 10 pts" in w["core_status"]

    # Render Section 02 template
    t2 = jinja_env.get_template(SECTION_TEMPLATE_MAP["02_data_inventory"])
    rend2 = t2.render(**ctx, assets={})
    assert "RCA: 50 pts | SCAL: 10 pts" in rend2
    assert "DT" in rend2
