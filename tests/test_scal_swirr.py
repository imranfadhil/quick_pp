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
