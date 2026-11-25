from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd

from quick_pp.app.backend.services.report import (
    SECTION_TEMPLATE_MAP,
    _normalize_zone_name,
    _pluralize,
    build_report_context,
)
from quick_pp.app.backend.services.report import (
    _jinja_env as jinja_env,
)
from quick_pp.plotter.report_plots import generate_fpt_pressure_plot


def test_phase4_stratigraphic_lexicon_and_capitalization():
    """Verify Task 4.2 stratigraphic lexicon corrections and casing preservation."""
    # Split or camel-case artifacts from logs/headers
    assert _normalize_zone_name("West Phalian B") == "Westphalian B"
    assert _normalize_zone_name("west phalian b") == "Westphalian B"
    assert _normalize_zone_name("WESTPHALIAN_B") == "Westphalian B"
    assert _normalize_zone_name("WESTPHALIAN B") == "Westphalian B"
    assert _normalize_zone_name("Steph Anian") == "Stephanian"
    assert _normalize_zone_name("Namur Ian") == "Namurian"
    assert _normalize_zone_name("Vise An") == "Visean"
    assert _normalize_zone_name("Kimme Ridg Ian") == "Kimmeridgian"
    assert _normalize_zone_name("Rot Liegend") == "Rotliegend"
    assert _normalize_zone_name("Zech Stein") == "Zechstein"
    assert _normalize_zone_name("Stat Fjord") == "Statfjord"

    # Already valid or mixed casing preserved
    assert _normalize_zone_name("Westphalian B") == "Westphalian B"
    assert _normalize_zone_name("Brent") == "Brent"
    assert _normalize_zone_name("Main Sand") == "Main Sand"
    assert _normalize_zone_name("Target Formation II") == "Target Formation II"
    assert _normalize_zone_name("") == "Unknown"
    assert _normalize_zone_name("nan") == "Unknown"
    assert _normalize_zone_name(None) == "Unknown"


def test_phase4_editorial_polish_and_pluralization():
    """Verify Task 4.3 dynamic pluralization and elimination of static (s) boilerplate."""
    # Pluralization helper
    assert _pluralize(1, "well") == "well"
    assert _pluralize(4, "well") == "wells"
    assert _pluralize(1, "zone") == "zone"
    assert _pluralize(31, "zone") == "zones"
    assert _pluralize(0, "plug") == "plugs"
    assert _pluralize(1, "realization") == "realization"
    assert _pluralize(1000, "realization") == "realizations"

    # Check template rendering for zero '(s)' boilerplate
    dummy_ctx = {
        "project_name": "Murdoch Field",
        "wells": [
            {
                "name": f"W{i}",
                "uwi": f"W0{i}",
                "kb": 25.0,
                "has_tvd": True,
                "min_depth": 2000.0,
                "max_depth": 2100.0,
                "td": 100.0,
                "raw_curves": ["GR", "NPHI", "RHOB"],
                "eval_curves": ["VCLAY", "PHIE", "SWT"],
                "core_status": "None",
                "has_pressure": "Yes",
                "formation_top_count": 2,
            }
            for i in range(1, 5)
        ],
        "well_names": ["W1", "W2", "W3", "W4"],
        "all_zones": ["Westphalian B"],
        "target_zones": ["Westphalian B"],
        "total_logged_thickness": 450.0,
        "zonal_summary": [
            {
                "zone": "Westphalian B",
                "well_count": 4,
                "gross": 100.0,
                "net_res": 80.0,
                "ntg_res": 0.80,
                "vsh_res": 0.12,
                "phie_res": 0.18,
                "perm_geom_res": 150.0,
                "net_pay": 60.0,
                "ntg": 0.60,
                "phie": 0.19,
                "sw": 0.25,
                "hcpt": 11.4,
            }
        ],
        "poro_params": {"total_core_count": 54, "core_count": 28},
        "upscaling_params": {"cell_size": 2.0, "omega": 0.33},
        "upscaling_summary": {
            "mean_kv_kh": 0.285,
            "min_kv_kh": 0.150,
            "max_kv_kh": 0.420,
        },
        "upscaling_table": [
            {
                "well_name": "W1",
                "fine_samples": 500,
                "n_cells": 50,
                "gross_m": 100.0,
                "phit_up": 0.18,
                "swt_up": 0.35,
                "k_arith": 120.0,
                "k_geom": 75.0,
                "k_harm": 34.2,
                "kv_kh": 0.285,
                "n_res_cells": 40,
                "ntg_res_up": 0.80,
                "phie_res_up": 0.18,
                "phie_res_fine": 0.18,
                "swt_res_up": 0.35,
                "swt_res_fine": 0.35,
                "delta_phi_res": 0.0,
                "conservation_status": "✓ CONSERVED",
            }
        ],
        "include_ai_commentary": False,
        "custom_notes": None,
        "assets": {},
    }

    t1 = jinja_env.get_template(SECTION_TEMPLATE_MAP["01_executive_summary"])
    rend1 = t1.render(**dummy_ctx)
    assert "well(s)" not in rend1
    assert "evaluated wells" in rend1

    t2 = jinja_env.get_template(SECTION_TEMPLATE_MAP["02_data_inventory"])
    rend2 = t2.render(**dummy_ctx)
    assert "well(s)" not in rend2
    assert "zone(s)" not in rend2
    assert "**4** wells were evaluated" in rend2
    assert "**1** stratigraphic zone" in rend2

    t9 = jinja_env.get_template(SECTION_TEMPLATE_MAP["09_upscaling"])
    rend9 = t9.render(**dummy_ctx)
    assert "well(s)" not in rend9
    assert "Kv / Kh = 0.285" in rend9 or "K_v / K_h = 0.285" in rend9
    assert "0.150–0.420" in rend9
    assert "evaluated 4 wells" in rend9


def test_phase4_fpt_gathering_and_gradient_analysis():
    """Verify Task 4.1 FPT station gathering, fluid gradient regression, and SHF density feedback."""
    # Synthetic FPT stations spanning Gas and Water columns in 2 wells
    # Gas gradient ~ 0.09 psi/ft (0.20 g/cc) from 3600 to 3690 m TVDss
    # Water gradient ~ 0.45 psi/ft (1.04 g/cc) from 3700 to 3760 m TVDss
    # Contact at ~3697.0 m TVDss
    fpt_points = [
        # Gas stations in W1
        {"id": 1, "well_name": "44-22-1", "md": 3610.0, "tvdss": 3610.0, "pressure": 5300.0, "mobility": 25.0, "fluid_type": "GAS", "test_quality": "VALID"},
        {"id": 2, "well_name": "44-22-1", "md": 3640.0, "tvdss": 3640.0, "pressure": 5309.0, "mobility": 30.0, "fluid_type": "GAS", "test_quality": "VALID"},
        {"id": 3, "well_name": "44-22-1", "md": 3670.0, "tvdss": 3670.0, "pressure": 5318.0, "mobility": 20.0, "fluid_type": "GAS", "test_quality": "VALID"},
        {"id": 4, "well_name": "44-22-1", "md": 3690.0, "tvdss": 3690.0, "pressure": 5324.0, "mobility": 15.0, "fluid_type": "GAS", "test_quality": "VALID"},
        # Non-valid station in W1
        {"id": 5, "well_name": "44-22-1", "md": 3695.0, "tvdss": 3695.0, "pressure": 4900.0, "mobility": 0.01, "fluid_type": "UNASSIGNED", "test_quality": "TIGHT"},
        # Water stations in W2
        {"id": 6, "well_name": "44-22-2", "md": 3710.0, "tvdss": 3710.0, "pressure": 5345.0, "mobility": 45.0, "fluid_type": "WATER", "test_quality": "VALID"},
        {"id": 7, "well_name": "44-22-2", "md": 3730.0, "tvdss": 3730.0, "pressure": 5374.5, "mobility": 60.0, "fluid_type": "WATER", "test_quality": "VALID"},
        {"id": 8, "well_name": "44-22-2", "md": 3750.0, "tvdss": 3750.0, "pressure": 5404.0, "mobility": 50.0, "fluid_type": "WATER", "test_quality": "VALID"},
        {"id": 9, "well_name": "44-22-2", "md": 3770.0, "tvdss": 3770.0, "pressure": 5433.5, "mobility": 40.0, "fluid_type": "WATER", "test_quality": "VALID"},
    ]

    full_df = pd.DataFrame({
        "WELL_NAME": ["44-22-1"] * 50 + ["44-22-2"] * 50,
        "DEPTH": np.linspace(3600, 3780, 100),
        "ZONES": ["Westphalian B"] * 100,
        "GR": np.full(100, 45.0),
        "RT": np.full(100, 25.0),
        "RHOB": np.full(100, 2.35),
        "NPHI": np.full(100, 0.16),
        "PHIT": np.full(100, 0.19),
        "PHIE": np.full(100, 0.17),
        "VCLAY": np.full(100, 0.12),
        "_REPORT_VSHALE": np.full(100, 0.12),
        "SWT": np.full(100, 0.30),
        "PERM": np.full(100, 150.0),
    })

    with (
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
        patch("quick_pp.app.backend.services.report._load_project_wells_fast") as mock_wells,
        patch("quick_pp.app.backend.services.report._get_merged_dataframe_sync", return_value=full_df),
    ):
        mock_wells.return_value = (
            "Murdoch FPT Project",
            [
                {
                    "id": 1,
                    "name": "44-22-1",
                    "uwi": "W01",
                    "core_sample_count": 0,
                    "pressure_test_count": 5,
                    "tops": [],
                    "well_config": {"fpt_points": fpt_points},
                },
                {
                    "id": 2,
                    "name": "44-22-2",
                    "uwi": "W02",
                    "core_sample_count": 0,
                    "pressure_test_count": 4,
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
            target_zones=["Westphalian B"],
            enabled_section_ids={"06_saturation_contacts", "07_shf_scal"},
        )

    fpt_sum = ctx.get("fpt_summary")
    assert fpt_sum is not None
    assert fpt_sum["has_data"] is True
    assert fpt_sum["total_stations"] == 9
    assert fpt_sum["valid_stations"] == 8

    # Per-well station breakdown
    well_rows = fpt_sum["well_fpt_rows"]
    assert len(well_rows) == 2
    w1_row = next(r for r in well_rows if r["well_name"] == "44-22-1")
    assert w1_row["total_count"] == 5
    assert w1_row["valid_count"] == 4
    assert "3610.0" in w1_row["depth_range"]

    # Fluid gradients
    grad_rows = fpt_sum["fpt_gradient_rows"]
    assert len(grad_rows) >= 2
    gas_row = next(r for r in grad_rows if r["fluid_type"] == "Gas")
    water_row = next(r for r in grad_rows if r["fluid_type"] == "Water")
    assert 0.15 <= gas_row["density_gcc"] <= 0.30
    assert 0.95 <= water_row["density_gcc"] <= 1.15
    assert gas_row["r_squared"] >= 0.95
    assert water_row["r_squared"] >= 0.95

    # Contact intersection
    contacts = fpt_sum["fpt_contacts"]
    assert len(contacts) > 0
    c0 = contacts[0]
    assert 3680.0 <= c0["tvdss_p50"] <= 3715.0
    assert "RECONCILED" in c0["status"] or "Δ =" in c0["status"]

    # Measured fluid densities feedback into SHF params (REG-02)
    shf_p = ctx.get("shf_params", {})
    assert shf_p.get("gw") == water_row["density_gcc"]
    assert "Measured FPT" in shf_p.get("gw_basis", "")
    assert shf_p.get("ghc") == gas_row["density_gcc"]
    assert "Measured FPT" in shf_p.get("ghc_basis", "")

    # Section 6 template rendering
    t6 = jinja_env.get_template(SECTION_TEMPLATE_MAP["06_saturation_contacts"])
    rend6 = t6.render(**ctx, assets={"fpt_pressure_plot": "fig_6_2_fpt_pressure_plot.png"})
    assert "Formation Pressure Testing (RFT/MDT/XPT)" in rend6
    assert "Table 6.3: Multi-Well Formation Pressure Regimes" in rend6
    assert "Table 6.4: Pressure-Gradient Contact Reconciliation" in rend6
    assert "Figure 6.2: Multi-Well Formation Pressure vs. TVDss" in rend6


def test_phase4_fpt_pressure_plot_generation(tmp_path: Path):
    """Verify headless generation of Figure 6.2 two-panel diagnostic plot."""
    fpt_points = [
        {"id": 1, "well_name": "44-22-1", "md": 3610.0, "tvdss": 3610.0, "pressure": 5300.0, "test_quality": "VALID"},
        {"id": 2, "well_name": "44-22-1", "md": 3640.0, "tvdss": 3640.0, "pressure": 5309.0, "test_quality": "VALID"},
        {"id": 3, "well_name": "44-22-1", "md": 3670.0, "tvdss": 3670.0, "pressure": 5318.0, "test_quality": "VALID"},
        {"id": 4, "well_name": "44-22-1", "md": 3690.0, "tvdss": 3690.0, "pressure": 5100.0, "test_quality": "TIGHT"},
        {"id": 5, "well_name": "44-22-2", "md": 3710.0, "tvdss": 3710.0, "pressure": 5345.0, "test_quality": "VALID"},
        {"id": 6, "well_name": "44-22-2", "md": 3750.0, "tvdss": 3750.0, "pressure": 5404.0, "test_quality": "VALID"},
    ]
    fpt_summary = {
        "has_data": True,
        "fpt_gradient_rows": [
            {"fluid_type": "Gas", "gradient_psi_ft": 0.09, "density_gcc": 0.21, "datum_pressure": 5326.0},
            {"fluid_type": "Water", "gradient_psi_ft": 0.45, "density_gcc": 1.04, "datum_pressure": 5326.0},
        ],
        "fpt_contacts": [
            {"contact_type": "GWC / FWL", "tvdss_p50": 3697.0},
        ],
    }

    plot_file = generate_fpt_pressure_plot(
        fpt_points=fpt_points,
        fpt_summary=fpt_summary,
        output_dir=tmp_path,
    )
    p = Path(plot_file)
    assert p.exists()
    assert p.stat().st_size > 1000
