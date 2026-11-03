from quick_pp.app.backend.services.chat import (
    PRESET_QUICK_ACTIONS,
    SYSTEM_PROMPT,
    AgentChatRequest,
    _handle_unrecognized_native_prompt,
    _plan_native_tool_calls,
)


def test_balmoral_sandstone_does_not_randomly_run_tools():
    """Verify that external regional queries do not trigger random tools or blind fallbacks."""
    req = AgentChatRequest(
        message="what is the average porosity in balmoral sandstone",
        project_id=1,
        well_name="30-7a-8",
        provider="native_mcp",
    )
    planned = _plan_native_tool_calls(req)
    # Must NOT plan any tools (no random inspect, no conformal bounds, no irrelevant SSOT chapters)
    assert planned == []


def test_unrecognized_native_prompt_highlights_low_confidence_and_quick_actions():
    """Verify that unhandled prompts return low confidence warning and preset quick actions."""
    req = AgentChatRequest(
        message="what is the average porosity in balmoral sandstone",
        well_name="30-7a-8",
        provider="native_mcp",
    )
    reply, actions = _handle_unrecognized_native_prompt(req)

    # Must highlight answer confidence
    assert "Answer Confidence: Low" in reply
    assert "Why the agent is not confident in this answer" in reply
    assert "Balmoral" in reply or "balmoral" in reply

    # Must return all 7 preset quick actions
    assert len(actions) == 7
    assert len(actions) == len(PRESET_QUICK_ACTIONS)
    assert any(act["label"] == "Inspect Well" for act in actions)
    assert any(act["label"] == "Run QAQC" for act in actions)
    assert any(act["label"] == "Multi-Mineral" for act in actions)
    assert any(act["label"] == "Conformal Bounds" for act in actions)
    assert "Preset Quick Actions" in reply


def test_valid_petrophysical_prompts_still_route_correctly():
    """Verify that legitimate petrophysical commands still trigger the correct tools."""
    req_inspect = AgentChatRequest(
        message="Inspect this well and check curve inventory",
        project_id=1,
        well_name="30-7a-8",
        provider="native_mcp",
    )
    planned_inspect = _plan_native_tool_calls(req_inspect)
    assert len(planned_inspect) == 1
    assert planned_inspect[0]["name"] == "inspect_well"

    req_qaqc = AgentChatRequest(
        message="Run QAQC badhole detection on this well",
        project_id=1,
        well_name="30-7a-8",
        provider="native_mcp",
    )
    planned_qaqc = _plan_native_tool_calls(req_qaqc)
    assert len(planned_qaqc) == 1
    assert planned_qaqc[0]["name"] == "run_well_qaqc"


def test_system_prompt_mandates_answer_confidence_assessment():
    """Verify that the LLM system prompt conditions the agent to assess confidence and highlight gaps in answers."""
    assert "SELF-ASSESSED ANSWER CONFIDENCE & EPISTEMIC HUMILITY" in SYSTEM_PROMPT
    assert "High Confidence" in SYSTEM_PROMPT
    assert "Medium Confidence" in SYSTEM_PROMPT
    assert "Low Confidence" in SYSTEM_PROMPT
    assert "Answer Confidence: [Low / Medium]" in SYSTEM_PROMPT
    assert "Why I am not confident in this answer" in SYSTEM_PROMPT
    assert "Balmoral" in SYSTEM_PROMPT

