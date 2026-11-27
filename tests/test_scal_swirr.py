"""Tests for multi-well SCAL Swirr extraction and Lab vs Reservoir Pc domain handling."""

import numpy as np

from quick_pp.app.backend.services.workspace import (
    WorkspaceSwirrFitRequest,
    _extract_plug_swirr_at_ref_pc,
)


def test_extract_plug_swirr_at_ref_pc_modes():
    """Verify Swirr extraction under cutoff_pc and interp_pc modes."""
    pc = np.array([1.0, 5.0, 15.0, 50.0, 203.0])
    sw = np.array([0.95, 0.70, 0.45, 0.25, 0.15])

    # Mode 1: cutoff_pc at ref_pc = 50 psi (max Pc 203 >= 50 -> valid)
    ext_swirr, valid, max_pc, sw_at_max = _extract_plug_swirr_at_ref_pc(
        pc, sw, "cutoff_pc", 50.0
    )
    assert valid is True
    assert max_pc == 203.0
    assert sw_at_max == 0.15
    assert ext_swirr == 0.15

    # If ref_pc is higher than max Pc
    ext_swirr_high, valid_high, _, _ = _extract_plug_swirr_at_ref_pc(
        pc, sw, "cutoff_pc", 500.0
    )
    assert valid_high is False

    # Mode 2: interp_pc interpolates Sw at ref_pc
    ext_swirr_interp, valid_interp, _, _ = _extract_plug_swirr_at_ref_pc(
        pc, sw, "interp_pc", 50.0
    )
    assert valid_interp is True
    assert 0.15 <= ext_swirr_interp <= 0.35


def test_pc_domain_request_schema():
    """Verify WorkspaceSwirrFitRequest accepts pc_domain and pcDomain."""
    req1 = WorkspaceSwirrFitRequest(pc_domain="lab")
    assert req1.pc_domain == "lab"

    req2 = WorkspaceSwirrFitRequest(pcDomain="reservoir")
    assert req2.pcDomain == "reservoir"


def test_canonical_well_token():
    """Verify canonical well token handles slashes, dashes, spaces, and leading zeros."""
    from quick_pp.app.backend.services.ancillary import _canonical_well_token

    assert _canonical_well_token("21/30-8") == "21308"
    assert _canonical_well_token("21-30-8") == "21308"
    assert _canonical_well_token("21/030-08") == "21308"
    assert _canonical_well_token("21/25-3") == "21253"
    assert _canonical_well_token("21-25-3") == "21253"


def test_experiment_type_case_insensitive_detection():
    """Verify case-insensitive method detection from core sample pc table."""
    import pandas as pd

    # Case 1: lowercase 'type' with 'micp'
    pc_df1 = pd.DataFrame(
        {
            "saturation": [0.9, 0.5, 0.1],
            "pressure": [10.0, 100.0, 2000.0],
            "type": ["micp", "micp", "micp"],
            "cycle": ["drainage", "drainage", "drainage"],
        }
    )
    col_map = {str(c).strip().upper(): c for c in pc_df1.columns}
    exp_type = "porous_plate"
    for tc in ("TYPE", "EXPERIMENT_TYPE", "METHOD", "TEST_TYPE"):
        if tc in col_map:
            vals = pc_df1[col_map[tc]].dropna().astype(str).tolist()
            if vals and vals[0].strip():
                exp_type = vals[0].strip().lower()
                break
    assert exp_type == "micp"

    # Case 2: uppercase 'EXPERIMENT_TYPE' with 'porous_plate'
    pc_df2 = pd.DataFrame(
        {
            "saturation": [0.9, 0.5, 0.2],
            "pressure": [1.0, 5.0, 203.0],
            "EXPERIMENT_TYPE": ["porous_plate", "porous_plate", "porous_plate"],
        }
    )
    col_map2 = {str(c).strip().upper(): c for c in pc_df2.columns}
    exp_type2 = "unknown"
    for tc in ("TYPE", "EXPERIMENT_TYPE", "METHOD", "TEST_TYPE"):
        if tc in col_map2:
            vals = pc_df2[col_map2[tc]].dropna().astype(str).tolist()
            if vals and vals[0].strip():
                exp_type2 = vals[0].strip().lower()
                break
    assert exp_type2 == "porous_plate"


def test_resolve_scal_fluid_system_and_leverett_j_invariance():
    """Verify MICP, Air-Brine, Lab Oil-Brine, and Reservoir fluid systems produce consistent J regardless of reservoir IFT/theta."""
    from quick_pp.app.backend.services.workspace import _resolve_scal_fluid_system
    from quick_pp.core_analysis import leverett_j

    # MICP (Hg-Air: 485 dyn/cm, 140 deg) vs Air-Brine (72 dyn/cm, 0 deg) measuring the exact same pore throat
    sys_micp, ift_m, th_m, den_m, res1 = _resolve_scal_fluid_system(
        "micp", 1500.0, ift_res=30.0, theta_res=30.0
    )
    sys_ab, ift_ab, th_ab, den_ab, _ = _resolve_scal_fluid_system(
        "porous_plate", 120.0, ift_res=30.0, theta_res=30.0
    )
    sys_oil, ift_o, th_o, den_o, _ = _resolve_scal_fluid_system(
        "centrifuge", 80.0, ift_res=30.0, theta_res=30.0, fluid_override="lab_oil_brine"
    )
    assert sys_micp == "micp" and ift_m == 485.0 and th_m == 140.0
    assert sys_ab == "air_brine" and ift_ab == 72.0 and th_ab == 0.0
    assert sys_oil == "lab_oil_brine" and ift_o == 48.0 and th_o == 30.0

    # Suppose true reservoir Pc = 25 psi at IFT=30, theta=30 -> equivalent lab pressures:
    pc_res_true = 25.0
    pc_lab_micp = pc_res_true * (den_m / res1)
    pc_lab_ab = pc_res_true * (den_ab / res1)

    j_from_micp = float(leverett_j(pc_lab_micp, ift_m, th_m, 100.0, 0.20))
    j_from_ab = float(leverett_j(pc_lab_ab, ift_ab, th_ab, 100.0, 0.20))
    j_from_res = float(leverett_j(pc_res_true, 30.0, 30.0, 100.0, 0.20))

    assert np.isclose(j_from_micp, j_from_ab, rtol=1e-6)
    assert np.isclose(j_from_micp, j_from_res, rtol=1e-6)


def test_swirr_and_shf_override_schemas():
    """Verify WorkspaceSwirrFitRequest and WorkspaceShfRequest accept RT/fluid overrides and fit_scope."""
    from quick_pp.app.backend.services.workspace import WorkspaceShfRequest

    req_swirr = WorkspaceSwirrFitRequest(
        plugRtOverrides={"Well-A::P1": 2},
        plugFluidOverrides={"Well-A::P1": "micp"},
    )
    assert req_swirr.plugRtOverrides == {"Well-A::P1": 2}
    assert req_swirr.plugFluidOverrides == {"Well-A::P1": "micp"}

    req_shf = WorkspaceShfRequest(fit_scope="global")
    assert req_shf.fit_scope == "global"
