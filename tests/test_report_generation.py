"""End-to-end tests for the FDP Petrophysical Report Generation Module."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from quick_pp.app.backend.main import app
from quick_pp.app.backend.mcp_server import petro_mcp_server
from quick_pp.app.backend.schemas.report import (
    DEFAULT_REPORT_SECTIONS,
    ReportAiEnhanceRequest,
    ReportGenerationRequest,
)
from quick_pp.app.backend.services.report import (
    compile_report_to_pdf,
    enhance_report_with_ai_service,
    generate_petrophysical_report_service,
    get_compact_report_context,
    render_report_print_html,
    update_report_section_service,
)
from quick_pp.plotter import report_plots


@pytest.fixture
def synthetic_well_df() -> pd.DataFrame:
    """Create a realistic synthetic petrophysical well DataFrame across two zones."""
    n = 120
    depths = np.linspace(2000.0, 2060.0, n)
    zones = ["UPPER_SAND" if d < 2030.0 else "LOWER_SAND" for d in depths]
    vclay = np.clip(0.18 + 0.12 * np.sin(np.linspace(0, 4, n)), 0.05, 0.75)
    vsilt = np.clip(0.10 + 0.05 * np.cos(np.linspace(0, 3, n)), 0.02, 0.30)
    phit = np.clip(0.24 - 0.15 * vclay, 0.06, 0.30)
    phie = np.clip(phit * (1.0 - 0.4 * vclay), 0.03, 0.28)
    vsand = np.clip(1.0 - vclay - vsilt - phit, 0.05, 0.85)
    swt = np.clip(0.25 + 0.55 * ((depths - 2000.0) / 60.0), 0.12, 0.98)
    shf = np.clip(swt + 0.02 * np.sin(np.linspace(0, 6, n)), 0.10, 0.99)
    perm = np.clip(10.0 ** (-1.0 + 14.0 * phie), 0.05, 2500.0)

    df = pd.DataFrame(
        {
            "DEPTH": depths,
            "WELL_NAME": ["WELL-A"] * n,
            "ZONES": zones,
            "GR": 25.0 + 95.0 * vclay,
            "GR_NORM": 24.0 + 92.0 * vclay,
            "RT": np.clip(1.5 / ((phit**2) * (swt**2)), 0.5, 250.0),
            "NPHI": phit + 0.12 * vclay,
            "RHOB": 2.65 * (1.0 - phit) + 1.0 * phit,
            "DTC": 55.0 + 130.0 * phit,
            "CALI": np.where((depths > 2022) & (depths < 2025), 11.2, 8.5),
            "BADHOLE": np.where((depths > 2022) & (depths < 2025), 1.0, 0.0),
            "VCLAY": vclay,
            "VSILT": vsilt,
            "VSAND": vsand,
            "PHIT": phit,
            "PHIE": phie,
            "PERM": perm,
            "SWT": swt,
            "SHF": shf,
            "ROCK_FLAG": np.where(phie >= 0.16, 1, np.where(phie >= 0.11, 2, 3)),
            "CPORE": np.where(np.arange(n) % 10 == 0, phit * 1.01, np.nan),
            "CPERM": np.where(np.arange(n) % 10 == 0, perm * 1.05, np.nan),
        }
    )
    return df


def test_report_schemas_default_sections():
    req = ReportGenerationRequest(project_id="1", report_type="FDP_FULL")
    assert len(DEFAULT_REPORT_SECTIONS) == 10
    assert req.cutoffs["vsh"] == 0.40
    assert req.cutoffs["phie"] == 0.08
    assert req.cutoffs["sw"] == 0.65


def test_headless_plot_exporters(tmp_path: Path, synthetic_well_df: pd.DataFrame):
    wells_meta = [
        {
            "name": "WELL-A",
            "curves": [
                "GR",
                "CALI",
                "RT",
                "NPHI",
                "RHOB",
                "DTC",
                "VCLAY",
                "PHIT",
                "PHIE",
                "PERM",
                "SWT",
                "SHF",
            ],
            "core_sample_count": 12,
            "formation_top_count": 2,
        }
    ]

    p1 = Path(
        report_plots.generate_inventory_heatmap("99", wells_meta, output_dir=tmp_path)
    )
    p2 = Path(
        report_plots.generate_qaqc_norm_plot(
            synthetic_well_df, ["WELL-A"], output_dir=tmp_path
        )
    )
    p3 = Path(
        report_plots.generate_nd_crossplot(
            synthetic_well_df, ["WELL-A"], output_dir=tmp_path
        )
    )
    p5 = Path(
        report_plots.generate_rock_typing_perm_plot(
            synthetic_well_df, ["WELL-A"], output_dir=tmp_path
        )
    )
    p6 = Path(
        report_plots.generate_pickett_plot(
            synthetic_well_df, ["WELL-A"], output_dir=tmp_path
        )
    )
    p7 = Path(
        report_plots.generate_shf_j_function_plot(
            synthetic_well_df, output_dir=tmp_path
        )
    )
    p8 = Path(
        report_plots.generate_uncertainty_cutoff_plot(
            synthetic_well_df,
            ["WELL-A"],
            cutoffs={"vsh": 0.4, "phie": 0.08, "sw": 0.65},
            output_dir=tmp_path,
        )
    )
    p9 = Path(
        report_plots.generate_upscaling_comparison_plot(
            synthetic_well_df,
            "WELL-A",
            cutoffs={"vsh": 0.4, "phie": 0.08, "sw": 0.65},
            output_dir=tmp_path,
        )
    )
    p10 = Path(
        report_plots.generate_well_cpi_plot(
            synthetic_well_df,
            "WELL-A",
            cutoffs={"vsh": 0.4, "phie": 0.08, "sw": 0.65},
            output_dir=tmp_path,
        )
    )

    for p in (p1, p2, p3, p5, p6, p7, p8, p9, p10):
        assert p.exists() and p.stat().st_size > 1000

    # Also test sparse DataFrame where SHF, CPORE, CPERM, BADHOLE are absent
    sparse_df = synthetic_well_df[
        ["DEPTH", "WELL_NAME", "GR", "RT", "NPHI", "RHOB"]
    ].copy()
    p6_sparse = Path(
        report_plots.generate_pickett_plot(sparse_df, ["WELL-A"], output_dir=tmp_path)
    )
    p10_sparse = Path(
        report_plots.generate_well_cpi_plot(sparse_df, "WELL-A", output_dir=tmp_path)
    )
    assert p6_sparse.exists() and p10_sparse.exists()


def test_end_to_end_report_service_and_mcp_tools(synthetic_well_df: pd.DataFrame):
    mock_summary = {
        "project_id": 99,
        "name": "Volve Synthetic Field",
        "wells": [
            {
                "name": "WELL-A",
                "tops": [
                    {"name": "UPPER_SAND", "depth": 2000.0},
                    {"name": "LOWER_SAND", "depth": 2030.0},
                ],
                "contacts": [{"name": "OWC", "depth": 2046.0}],
            }
        ],
    }

    with (
        patch(
            "quick_pp.app.backend.services.report.get_project_summary",
            return_value=mock_summary,
        ),
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            return_value=synthetic_well_df,
        ),
    ):
        # 1. Test compact report context
        ctx = get_compact_report_context("99", well_names=["WELL-A"])
        assert ctx["project_name"] == "Volve Synthetic Field"
        assert len(ctx["well_names"]) == 1

        # 2. Test full FDP report generation
        req = ReportGenerationRequest(
            project_id="99",
            report_type="FDP_FULL",
            well_names=["WELL-A"],
            include_ai_summary=True,
        )
        res = generate_petrophysical_report_service(req)
        assert res.report_id
        assert "1. Executive Summary" in res.markdown_content
        assert "<table" in res.html_content
        assert len(res.asset_urls) >= 8

        # 3. Test section update (both short alias and canonical ID)
        updated = update_report_section_service(
            res.report_id,
            "exec_summary",
            "# 1. Executive Summary\n\nCustom updated executive summary text.",
        )
        assert "Custom updated executive summary text." in updated.markdown_content

        # 4. Test AI enhancement
        enhanced = enhance_report_with_ai_service(
            res.report_id, ReportAiEnhanceRequest(provider="native_mcp")
        )
        assert enhanced.report_id == res.report_id

        # 5. Test A4 print HTML & PDF export
        print_html = render_report_print_html(res.report_id)
        assert "<!DOCTYPE html>" in print_html
        assert "@page" in print_html

        pdf_path = compile_report_to_pdf(res.report_id)
        assert pdf_path.exists() and pdf_path.stat().st_size > 500

        # 6. Test MCP tools registration & execution
        mcp_out = petro_mcp_server.call_tool(
            "generate_petrophysical_report",
            {
                "project_id": 99,
                "report_type": "QAQC_SUMMARY",
                "include_ai_summary": True,
            },
        )
        assert mcp_out["tool"] == "generate_petrophysical_report"
        assert mcp_out["report_id"]

        # 7. Test FastAPI endpoints via TestClient
        client = TestClient(app)
        resp = client.post(
            "/quick_pp/projects/99/reports/generate",
            json={
                "project_id": "99",
                "report_type": "RESSUM_ONLY",
                "well_names": ["WELL-A"],
                "include_ai_summary": True,
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        # 8. Test Single-Well CPI Dossier context & numbering (not FDP)
        sw_req = ReportGenerationRequest(
            project_id="99",
            report_type="SINGLE_WELL_CPI",
            well_names=["WELL-A"],
            include_ai_summary=True,
        )
        sw_res = generate_petrophysical_report_service(sw_req)
        assert sw_res.report_type == "SINGLE_WELL_CPI"
        assert "Well WELL-A — Petrophysical CPI & Evaluation Dossier" in sw_res.title
        assert (
            "Single-Well Petrophysical CPI & Formation Evaluation Dossier"
            in sw_res.markdown_content
        )
        assert "Field Development Plan" not in sw_res.markdown_content
        assert "# 1. Well Petrophysical Evaluation Summary" in sw_res.markdown_content
        assert (
            "# 2. Borehole Environment & Log Curve Quality Control"
            in sw_res.markdown_content
        )
        assert (
            "# 6. Composite Petrophysical Evaluation Log (CPI)"
            in sw_res.markdown_content
        )
        sw_print_html = render_report_print_html(sw_res.report_id)
        assert "Target Well:" in sw_print_html
