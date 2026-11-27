"""Tests for Petrophysical Single-Source-of-Truth (SSOT) synchronization and grounding."""

from __future__ import annotations

import re
from pathlib import Path

from quick_pp.app.backend.services.ssot_reference import (
    STEP_ALIASES,
    SSOTReferenceRegistry,
    get_step_reference,
    query_workflow_reference,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCS_WORKFLOWS = REPO_ROOT / "docs" / "workflows"
FRONTEND_SHELL = (
    REPO_ROOT / "quick_pp" / "app" / "frontend" / "src" / "lib" / "components" / "shell"
)


def test_single_well_step_1_and_2_swapped_in_ssot():
    """Step 1 must be Well & Ancillary and Step 2 must be Data & QA/QC across SSOT."""
    reg = SSOTReferenceRegistry()

    sw1_num = reg.get_step_reference("single-well", "1")
    sw1_id = reg.get_step_reference("single-well", "ancillary")
    assert "{#sw-step-1-ancillary}" in sw1_num
    assert sw1_num == sw1_id
    assert "Hole Sections" in sw1_num
    assert "Minimum Curvature" in sw1_num

    sw2_num = reg.get_step_reference("single-well", "2")
    sw2_id = reg.get_step_reference("single-well", "qaqc")
    assert "{#sw-step-2-qaqc}" in sw2_num
    assert sw2_num == sw2_id
    assert "BADHOLE" in sw2_num
    assert "INVASION_PROFILE" in sw2_num


def test_all_step_aliases_resolve_to_existing_markdown_anchors():
    """Every alias in STEP_ALIASES must resolve to a parsed markdown anchor with non-empty body."""
    reg = SSOTReferenceRegistry()
    reg._ensure_loaded()

    for scope, alias_map in STEP_ALIASES.items():
        anchors = reg._cache[scope]["anchors"]
        for alias, target_anchor in alias_map.items():
            assert target_anchor in anchors, (
                f"Alias '{alias}' in scope '{scope}' points to missing anchor '{target_anchor}'"
            )
            section_text = reg.get_step_reference(scope, alias)
            lines = [ln.strip() for ln in section_text.splitlines() if ln.strip()]
            assert len(lines) >= 2, (
                f"Anchor '{target_anchor}' for '{scope}:{alias}' has an empty body"
            )


def test_permeability_ssot_accuracy_and_fzi_scope_placement():
    """Single-well perm must reflect Choo/KC/Timur/Tixier/Coates accurately; FZI belongs in multi-well."""
    sw_perm = get_step_reference("single-well", "perm")
    assert "{#sw-step-4-perm}" in sw_perm
    # Choo equation: A * phit^B / 10^(6*Vclay + 3*Vsilt) and B = m * ((2/C) + 1) + 2
    assert "6 V_{\\text{clay}} + 3 V_{\\text{silt}}" in sw_perm
    assert "\\frac{2}{C} + 1" in sw_perm
    # Kozeny-Carman, Timur, Tixier, Coates
    assert "5 S^2 (1 - \\phi_t)^2" in sw_perm
    assert "0.136 \\cdot \\phi_t^{4.4}" in sw_perm
    assert "62.5 \\cdot \\phi_t^6" in sw_perm
    # FZI should not be claimed as a single-well Step 4 output curve
    output_line = next(ln for ln in sw_perm.splitlines() if "**Output Curves:**" in ln)
    assert "FZI" not in output_line
    assert "RQI" not in output_line

    # FZI and power-law PERM_TRANS live in Multi-Well Steps 3 & 4
    mw_rt = get_step_reference("multi-well", "fzi")
    assert "{#mw-step-3-rock-typing}" in mw_rt
    assert "0.0314" in mw_rt
    assert "ROCK_FLAG" in mw_rt

    mw_pt = get_step_reference("multi-well", "perm-transform")
    assert "{#mw-step-4-perm-transform}" in mw_pt
    assert "PERM_TRANS" in mw_pt
    assert "a_k \\cdot \\phi^{b_k}" in mw_pt


def test_frontend_stepper_and_guide_anchors_match_ssot_markdown():
    """Anchors referenced in WorkflowStepper.svelte and WorkflowGuideSheet.svelte must match SSOT docs."""
    reg = SSOTReferenceRegistry()
    reg._ensure_loaded()
    all_doc_anchors = set(reg._cache["single-well"]["anchors"]) | set(
        reg._cache["multi-well"]["anchors"]
    )

    stepper_src = (FRONTEND_SHELL / "WorkflowStepper.svelte").read_text(
        encoding="utf-8"
    )
    guide_src = (FRONTEND_SHELL / "WorkflowGuideSheet.svelte").read_text(
        encoding="utf-8"
    )

    stepper_anchors = set(
        re.findall(
            r"'(sw-(?:step|standalone)-[\w\-]+|mw-(?:step|standalone)-[\w\-]+)'",
            stepper_src,
        )
    )
    guide_ids = set(
        re.findall(
            r'id=["\']((?:sw|mw)-(?:level-1-2-diagrams|step-[\w\-]+|standalone-[\w\-]+))["\']',
            guide_src,
        )
    )

    assert stepper_anchors, "Expected step anchors in WorkflowStepper.svelte"
    assert guide_ids, "Expected section IDs in WorkflowGuideSheet.svelte"
    assert stepper_anchors.issubset(all_doc_anchors), (
        f"Stepper anchors missing in docs/workflows: {stepper_anchors - all_doc_anchors}"
    )
    assert guide_ids == all_doc_anchors, (
        f"Mismatch between WorkflowGuideSheet IDs and docs/workflows anchors: "
        f"only_in_guide={guide_ids - all_doc_anchors}, only_in_docs={all_doc_anchors - guide_ids}"
    )

    # Verify Choo formula in WorkflowGuideSheet matches quick_pp implementation
    assert "10^(6·V_clay + 3·V_silt)" in guide_src
    assert "PERM_TRANS" in guide_src


def test_query_workflow_reference_finds_latest_enhancements():
    """Query engine should retrieve sections for Hole Sections, radial invasion, and cutoff sensitivity."""
    hs_hits = query_workflow_reference(
        "hole sections bit size rmf", scope="single-well"
    )
    assert hs_hits and hs_hits[0]["anchor"] == "sw-step-1-ancillary"

    inv_hits = query_workflow_reference(
        "radial invasion rxo di sigma sensor confidence", scope="single-well"
    )
    assert inv_hits and inv_hits[0]["anchor"] == "sw-step-2-qaqc"

    sens_hits = query_workflow_reference(
        "cutoff sensitivity prms hcpv cardwell parsons", scope="single-well"
    )
    assert sens_hits and sens_hits[0]["anchor"] == "sw-step-6-ressum"
