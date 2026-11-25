from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from jinja2 import Environment, FileSystemLoader

from quick_pp.app.backend.services.report import build_report_context


@pytest.fixture
def jinja_env():
    templates_dir = Path("quick_pp/app/backend/template/reports")
    return Environment(loader=FileSystemLoader(str(templates_dir)))


@pytest.fixture
def sample_gas_df():
    # Synthetic dataframe with depths around 3600-3750m TVDss (Murdoch-like)
    n = 100
    depths = np.linspace(3650.0, 3750.0, n)
    return pd.DataFrame(
        {
            "DEPTH": depths,
            "TVDSS": depths,
            "WELL_NAME": ["WELL-1"] * n,
            "ZONES": ["West Phalian B"] * n,
            "GR": np.linspace(20, 100, n),
            "RT": np.linspace(50, 2, n),
            "PHIT": np.full(n, 0.18),
            "PHIE": np.full(n, 0.16),
            "SWT": np.linspace(0.15, 0.95, n),
            "PERM": np.full(n, 150.0),
            "CPERM": np.full(n, 150.0),
            "VSHALE": np.full(n, 0.15),
            "_REPORT_VSHALE": np.full(n, 0.15),
        }
    )


def test_fwl_tier_hierarchy(sample_gas_df):
    mock_contact = MagicMock()
    mock_contact.well_id = 101
    mock_contact.name = "Murdoch FWL"
    mock_contact.depth = 3697.0
    mock_contact.zone = "West Phalian B"
    mock_contact.contact_type = "FWL"
    mock_contact.fault_block = "Block-A"
    mock_contact.source = "FWL Optimizer (Capillary Pc = 0 datum)"
    mock_contact.remarks = "Committed"

    wells_meta = [{"id": 101, "name": "WELL-1", "well_config": {}}]

    with (
        patch(
            "quick_pp.app.backend.services.report._load_project_wells_fast",
            return_value=("Murdoch Gas Project", wells_meta),
        ),
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            return_value=sample_gas_df,
        ),
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
    ):
        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = [mock_contact]
        mock_db.return_value.__enter__.return_value = mock_session

        ctx = build_report_context(
            1, well_names=["WELL-1"], target_zones=["West Phalian B"]
        )
        assert ctx["sw_params"]["fwl"] == 3697.0
        assert ctx["shf_params"]["fwl"] == 3697.0
        assert "Committed Fluid Contact" in ctx["sw_params"]["fwl_basis"]
        assert "Committed Fluid Contact" in ctx["shf_params"]["fwl_basis"]

        # Case 2: Tier 1 - Request Override overrides DB contact
        ctx_override = build_report_context(
            1, well_names=["WELL-1"], target_zones=["West Phalian B"], fwl_tvdss=3690.5
        )
        assert ctx_override["sw_params"]["fwl"] == 3690.5
        assert ctx_override["sw_params"]["fwl_basis"] == "User Request Override"
        assert ctx_override["shf_params"]["fwl"] == 3690.5


def test_phase_aware_gas_fluid_densities(sample_gas_df):
    mock_contact = MagicMock()
    mock_contact.well_id = 101
    mock_contact.name = "GWC"
    mock_contact.depth = 3697.0
    mock_contact.zone = "West Phalian B"
    mock_contact.contact_type = "GWC"
    mock_contact.fault_block = None
    mock_contact.source = "Log"
    mock_contact.remarks = ""

    wells_meta = [{"id": 101, "name": "WELL-1", "well_config": {}}]

    with (
        patch(
            "quick_pp.app.backend.services.report._load_project_wells_fast",
            return_value=("Murdoch Gas Field", wells_meta),
        ),
        patch(
            "quick_pp.app.backend.services.report._get_merged_dataframe_sync",
            return_value=sample_gas_df,
        ),
        patch("quick_pp.app.backend.services.report.get_db") as mock_db,
    ):
        mock_session = MagicMock()
        mock_session.execute.return_value.all.return_value = [mock_contact]
        mock_db.return_value.__enter__.return_value = mock_session

        ctx = build_report_context(
            1, well_names=["WELL-1"], target_zones=["West Phalian B"]
        )
        assert ctx["shf_params"]["reservoir_fluid"] == "gas"
        assert ctx["shf_params"]["ghc"] == 0.22
        assert ctx["shf_params"]["gw"] == 1.05
        assert ctx["shf_params"]["delta_rho"] == 0.83
        assert "Gas" in ctx["shf_params"]["ghc_basis"]


def test_let_vs_corey_template_rendering(jinja_env):
    template = jinja_env.get_template("sections/07_shf_scal.md.j2")

    # Case 1: Active LET model
    ctx_let = {
        "report_type": "FDP_FULL",
        "sw_params": {"fwl": 3697.0, "fwl_basis": "Committed Fluid Contact"},
        "shf_params": {
            "fwl": 3697.0,
            "fwl_basis": "Committed Fluid Contact",
            "ift_cos_theta": 26.0,
            "ift": 30.0,
            "theta": 30.0,
            "ift_lab_cos_theta": 367.7,
            "ift_lab": 480.0,
            "theta_lab": 140.0,
            "pc_scale_factor": 0.0707,
            "pc_lab_ref": 25.0,
            "pc_res_ref": 1.768,
            "gw": 1.05,
            "ghc": 0.22,
            "ghc_basis": "Default Gas Density",
            "reconciliation_rmse": 0.045,
            "reconciliation_bias": -0.010,
        },
        "poro_params": {"core_corrected": False},
        "scal_params": {
            "is_analogue": True,
            "model": "let",
            "source_label": "Regional Analogue / Screening LET Parameterization",
            "swirr": 0.15,
            "sor": 0.20,
            "krw_max": 0.45,
            "kro_max": 0.90,
            "lw": 2.0,
            "ew": 1.5,
            "tw": 2.0,
            "lo": 2.0,
            "eo": 1.5,
            "to": 2.0,
            "pc_entry": 1.5,
            "pc_max": 50.0,
        },
        "include_ai_commentary": False,
        "custom_notes": None,
        "assets": {},
    }
    rendered_let = template.render(ctx_let)
    # Check that LET equations appear
    assert "S_e^{L_w}" in rendered_let
    assert "E_w (1 - S_e)^{T_w}" in rendered_let
    assert "LET Water Parameters" in rendered_let
    assert "LET Oil Parameters" in rendered_let
    # Check that Corey exponents are NOT rendered
    assert "Corey Water / Oil Exponents" not in rendered_let
    assert "(S_e)^{N_w}" not in rendered_let


def test_gamma_endpoint_deduplication(jinja_env):
    template = jinja_env.get_template("sections/04_lithology_porosity.md.j2")

    ctx = {
        "report_type": "FDP_FULL",
        "target_zones": ["West Phalian B"],
        "litho_params": {
            "model_key": "vsh_gr",
            "model_name": "Spectral Gamma-Ray (Vsh_GR)",
            "rho_clay": 2.68,
            "nphi_clay": 0.35,
            "gr_clean": 6.5,
            "gr_shale": 88.5,
            "gamma": {
                "gamma_source": "derived_cgr",
                "thorium_curve": "THOR",
                "potassium_curve": "POTA",
                "cgr_th_coefficient": 4.0,
                "cgr_k_coefficient": 16.0,
                "gamma_clean": 6.5,
                "gamma_shale": 88.5,
            },
        },
        "poro_params": {
            "method_name": "Density Porosity",
            "method_key": "density",
            "use_hc_corrected": False,
            "total_core_count": 0,
            "core_count": 0,
            "rho_ma": 2.65,
            "core_ref": "2.65 g/cc Matrix",
            "rho_fl": 1.0,
            "rhoma_count": 0,
            "core_shift_applied": False,
            "core_corrected": False,
        },
        "include_ai_commentary": False,
        "custom_notes": None,
        "assets": {},
    }
    rendered = template.render(ctx)
    assert "Clean / shale gamma endpoints" in rendered
    assert "6.5 / 88.5" in rendered
    assert "Clean GR Endpoint" not in rendered
    assert "Shale GR Endpoint" not in rendered


def test_saturation_waxman_smits_parameters(jinja_env):
    template = jinja_env.get_template("sections/06_saturation_contacts.md.j2")

    ctx = {
        "report_type": "FDP_FULL",
        "sw_params": {
            "model_key": "waxman_smits",
            "model_name": "Waxman-Smits Shaly-Sand",
            "fwl": 3697.0,
            "fwl_basis": "Committed Fluid Contact / FWL Optimizer (Pc = 0 datum)",
            "salinity": 42000,
            "rw": 0.055,
            "temp": 126.0,
            "temp_basis": "Computed from Geothermal Gradient (3.0 °C/100m @ 3700 m TVDss)",
            "a": 1.0,
            "m": 1.95,
            "n": 2.0,
            "b_conductance": 14.8,
            "qv": 0.45,
            "qv_method": "Juhasz Qvn Empirical Correlation",
            "phish": 0.22,
        },
        "fluid_contacts_rows": [],
        "fluid_contacts_summary": [],
        "include_ai_commentary": False,
        "custom_notes": None,
        "assets": {},
    }
    rendered = template.render(ctx)
    assert "Active Waxman-Smits Shaly-Sand" in rendered
    assert "3697.0" in rendered
    assert "Committed Fluid Contact / FWL Optimizer" in rendered
    assert "Clay Counter-Ion Conductance" in rendered
    assert "14.8" in rendered
    assert "126.0" in rendered
    assert "Computed from Geothermal Gradient" in rendered
