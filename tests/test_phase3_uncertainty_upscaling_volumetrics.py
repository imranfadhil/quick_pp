from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quick_pp.app.backend.services.report import (
    SECTION_TEMPLATE_MAP,
    build_report_context,
    validate_fdp_report_payload,
)
from quick_pp.app.backend.services.report import (
    _jinja_env as jinja_env,
)


def test_phase3_monte_carlo_physics_and_multi_row_table():
    """Verify physics-based Monte Carlo engine generates 4 rows per zone with sensitivity drivers."""
    n = 100
    depths = np.linspace(2500, 2600, n)
    full_df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-01"] * n,
            "DEPTH": depths,
            "ZONES": ["Main Sand"] * n,
            "GR": np.linspace(30, 80, n),
            "RT": np.linspace(40, 10, n),
            "RHOB": np.linspace(2.30, 2.45, n),
            "NPHI": np.full(n, 0.18),
            "PHIT": np.full(n, 0.20),
            "PHIE": np.full(n, 0.17),
            "VCLAY": np.full(n, 0.15),
            "_REPORT_VSHALE": np.full(n, 0.15),
            "SWT": np.linspace(0.20, 0.60, n),
            "PERM": np.full(n, 120.0),
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
            "MC Test Project",
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
        session_mock.execute.return_value.all.return_value = []
        mock_db.return_value.__enter__.return_value = session_mock

        ctx = build_report_context(
            project_id=1,
            target_zones=["Main Sand"],
            enabled_section_ids={"08_ressum_uncertainty"},
        )

    unc_list = ctx.get("uncertainty_summary", [])
    assert len(unc_list) == 4

    metrics = [u["metric"] for u in unc_list]
    assert "Net Pay (m)" in metrics
    assert "Mean Effective Porosity Φe (v/v)" in metrics
    assert "Mean Water Saturation Sw (v/v)" in metrics
    assert "Hydrocarbon Pore Thickness HCPT (m)" in metrics

    # Verify all rows retain p90_hcpt, p50_hcpt, p10_hcpt, p50_phie for gate-check compatibility
    for u in unc_list:
        assert "p90_hcpt" in u
        assert "p50_hcpt" in u
        assert "p10_hcpt" in u
        assert "p50_phie" in u
        assert u["p90_hcpt"] <= u["p50_hcpt"] <= u["p10_hcpt"]

    # Verify dynamic Tornado sensitivity drivers
    assert "uncertainty_drivers_text" in ctx
    drivers_text = ctx["uncertainty_drivers_text"]
    assert "Hydrocarbon pore thickness (HCPT) variance" in drivers_text

    # Verify QA/QC gate-check CHK-04 and CHK-05 pass cleanly
    checks = validate_fdp_report_payload(ctx)
    chk04 = next(c for c in checks if c["id"] == "CHK-04")
    chk05 = next(c for c in checks if c["id"] == "CHK-05")
    assert chk04["status"] == "PASS"
    assert chk05["status"] == "PASS"

    # Verify Section 8 template rendering
    t8 = jinja_env.get_template(SECTION_TEMPLATE_MAP["08_ressum_uncertainty"])
    rend8 = t8.render(**ctx, assets={})
    assert "Net Pay (m)" in rend8
    assert "Mean Effective Porosity Φe (v/v)" in rend8
    assert "Mean Water Saturation Sw (v/v)" in rend8
    assert "Hydrocarbon Pore Thickness HCPT (m)" in rend8
    assert "Tornado Sensitivity Drivers" in rend8


def test_phase3_upscaling_dynamic_grid_and_dual_scope_tables():
    """Verify dynamic grid parameters, dual-scope blocked metrics, and Table 9.1 / 9.2 rendering."""
    n = 100
    depths = np.linspace(2000, 2100, n)  # 100 m logged
    full_df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-UP"] * n,
            "DEPTH": depths,
            "ZONES": ["Target Zone"] * n,
            "GR": np.linspace(40, 70, n),
            "RT": np.linspace(30, 15, n),
            "RHOB": np.linspace(2.35, 2.40, n),
            "NPHI": np.full(n, 0.18),
            "PHIT": np.full(n, 0.20),
            "PHIE": np.full(n, 0.18),
            "VCLAY": np.full(n, 0.10),
            "_REPORT_VSHALE": np.full(n, 0.10),
            "SWT": np.full(n, 0.35),
            "PERM": np.full(n, 150.0),
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
            "Upscaling Project",
            [
                {
                    "id": 1,
                    "name": "WELL-UP",
                    "uwi": "WUP",
                    "core_sample_count": 0,
                    "pressure_test_count": 0,
                    "tops": [],
                    "well_config": {
                        "upscaling_cell_size": 2.5,
                        "upscaling_omega": 0.40,
                    },
                }
            ],
        )
        session_mock = MagicMock()
        session_mock.execute.return_value.all.return_value = []
        mock_db.return_value.__enter__.return_value = session_mock

        ctx = build_report_context(
            project_id=1,
            target_zones=["Target Zone"],
            enabled_section_ids={"09_upscaling"},
        )

    up_params = ctx.get("upscaling_params", {})
    assert up_params.get("cell_size") == 2.5
    assert up_params.get("omega") == 0.40
    assert up_params.get("basis") == "Project Upscaling Config"

    up_tbl = ctx.get("upscaling_table", [])
    assert len(up_tbl) == 1
    row = up_tbl[0]

    # Verify All-Cell metrics
    assert "gross_m" in row
    assert row["gross_m"] == 100.0
    assert "phit_up" in row
    assert "swt_up" in row
    assert "k_arith" in row
    assert "k_geom" in row
    assert "k_harm" in row
    assert "kv_kh" in row

    # Verify Net-Reservoir metrics & conservation audit
    assert "n_res_cells" in row
    assert "ntg_res_up" in row
    assert "phie_res_up" in row
    assert "phie_res_fine" in row
    assert "delta_phi_res" in row
    assert "conservation_status" in row
    assert row["conservation_status"] == "✓ CONSERVED"

    # Verify upscaling summary
    up_summary = ctx.get("upscaling_summary", {})
    assert "mean_kv_kh" in up_summary
    assert up_summary["mean_kv_kh"] > 0.0

    # Verify Section 9 template rendering
    t9 = jinja_env.get_template(SECTION_TEMPLATE_MAP["09_upscaling"])
    rend9 = t9.render(**ctx, assets={})
    assert "Table 9.1: All-Cell Blocked Cellular Grid Architecture" in rend9
    assert (
        "Table 9.2: Active Net-Reservoir Cellular Blocked Averages & Volumetric Conservation Audit"
        in rend9
    )
    assert "✓ CONSERVED" in rend9
    assert f"{up_summary['mean_kv_kh']:.3f}" in rend9


def test_phase3_volumetrics_fallback_and_readiness_note():
    """Verify auto-volumetrics when Area/Bg configured, and Readiness Note when Area absent."""
    n = 50
    full_df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-VOL"] * n,
            "DEPTH": np.linspace(2500, 2550, n),
            "ZONES": ["Gas Sand"] * n,
            "GR": np.full(n, 40.0),
            "RT": np.full(n, 50.0),
            "RHOB": np.full(n, 2.30),
            "NPHI": np.full(n, 0.15),
            "PHIT": np.full(n, 0.22),
            "PHIE": np.full(n, 0.20),
            "VCLAY": np.full(n, 0.10),
            "_REPORT_VSHALE": np.full(n, 0.10),
            "SWT": np.full(n, 0.25),
            "PERM": np.full(n, 200.0),
        }
    )

    # Case A: Area configured -> Auto-computes GIIP / STOIIP
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
            "Volumetrics Project",
            [
                {
                    "id": 1,
                    "name": "WELL-VOL",
                    "uwi": "WVOL",
                    "core_sample_count": 0,
                    "pressure_test_count": 0,
                    "tops": [],
                    "well_config": {
                        "area_acres": 640.0,
                        "bg": 0.0045,
                        "reservoir_fluid": "gas",
                    },
                }
            ],
        )
        session_mock = MagicMock()
        session_mock.execute.return_value.all.return_value = []
        mock_db.return_value.__enter__.return_value = session_mock

        ctx_vol = build_report_context(
            project_id=1,
            target_zones=["Gas Sand"],
            enabled_section_ids={"08_ressum_uncertainty"},
        )

    vol_sum = ctx_vol.get("volumetric_summary")
    assert vol_sum is not None
    assert "field_totals" in vol_sum
    assert vol_sum["field_totals"]["giip"]["p50"] > 0.0

    t8 = jinja_env.get_template(SECTION_TEMPLATE_MAP["08_ressum_uncertainty"])
    rend8_vol = t8.render(**ctx_vol, assets={})
    assert (
        "Volumetric Hydrocarbon Resources (SPE-PRMS Latin Hypercube Monte Carlo)"
        in rend8_vol
    )
    assert "FIELD ROLLUP (TOTAL)" in rend8_vol

    # Case B: Area absent -> Volumetric Resource In-Place Readiness Note renders
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
            "Readiness Project",
            [
                {
                    "id": 1,
                    "name": "WELL-VOL",
                    "uwi": "WVOL",
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

        ctx_readiness = build_report_context(
            project_id=1,
            target_zones=["Gas Sand"],
            enabled_section_ids={"08_ressum_uncertainty"},
        )

    assert ctx_readiness.get("volumetric_summary") is None
    readiness = ctx_readiness.get("volumetric_readiness")
    assert readiness is not None
    assert readiness["mean_hcpt"] > 0.0
    assert readiness["mean_net_pay"] > 0.0

    rend8_readiness = t8.render(**ctx_readiness, assets={})
    assert "Volumetric Resource In-Place Readiness Note" in rend8_readiness
    assert "Cumulative Hydrocarbon Pore Thickness (HCPT)" in rend8_readiness
    assert "Mean HCPT per Evaluated Zone" in rend8_readiness
    assert f"{readiness['mean_hcpt']:.2f}" in rend8_readiness
    # Section 8.4 and the Executive Summary must quote the same cumulative HCPT.
    assert readiness["column_hcpt"] == readiness["total_hcpt"]
    assert f"{readiness['column_hcpt']:.2f}" in rend8_readiness
