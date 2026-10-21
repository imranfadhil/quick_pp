"""Petrophysical Agentic Assistant & Multi-Provider LLM + MCP Service for quick_pp.

Self-contained agentic assistant that:
1. Supports a live UI provider/model switcher (`native_mcp`, `openai`, `anthropic`, `gemini`, `ollama`)
2. Executes MCP tools in-process (`inspect_well`, `run_well_qaqc`, `run_multi_mineral`, `run_shf_evaluation`,
   `evaluate_with_conformal_bounds`, `control_workbench_ui`)
3. Returns structured tool execution cards, live-computed depth-aligned curves (`computed_curves`),
   Svelte UI navigation actions (`ui_actions`), and markdown explanations with conformal prediction bounds.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional

import requests
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from quick_pp import logger
from quick_pp.app.backend.mcp_server import petro_mcp_server

router = APIRouter(tags=["Agentic Assistant & MCP"])


# ---------------------------------------------------------------------------
# Provider Catalog for UI Drawer Switcher
# ---------------------------------------------------------------------------


PROVIDERS_CATALOG = [
    {
        "id": "native_mcp",
        "name": "Built-in Petrophysical Agent (No API Key Needed)",
        "requires_key": False,
        "default_model": "qpp-domain-agent-v1",
        "models": ["qpp-domain-agent-v1"],
        "description": "Deterministic & Conformal Petrophysical Agent running directly on quick_pp MCP tools.",
    },
    {
        "id": "openai",
        "name": "OpenAI / Compatible (GPT-4o / o3-mini)",
        "requires_key": True,
        "default_model": "gpt-4o-mini",
        "models": ["gpt-4o-mini", "gpt-4o", "o3-mini"],
        "env_key": "OPENAI_API_KEY",
    },
    {
        "id": "gemini",
        "name": "Google Gemini (2.5 Flash / Pro)",
        "requires_key": True,
        "default_model": "gemini-2.5-flash",
        "models": ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
        "env_key": "GEMINI_API_KEY",
    },
    {
        "id": "anthropic",
        "name": "Anthropic Claude (Sonnet / Haiku)",
        "requires_key": True,
        "default_model": "claude-3-5-sonnet-latest",
        "models": ["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"],
        "env_key": "ANTHROPIC_API_KEY",
    },
    {
        "id": "ollama",
        "name": "Local Ollama (Llama 3.2 / Qwen 2.5)",
        "requires_key": False,
        "default_model": "qwen2.5:7b",
        "models": ["qwen2.5:7b", "llama3.2:latest", "mistral:latest"],
        "default_base_url": "http://localhost:11434/v1",
    },
]

SYSTEM_PROMPT = """You are the quick_pp Petrophysical AI Assistant embedded inside the Svelte Subsurface Workbench.
You have direct access to the Petrophysical MCP Server tools:
- `inspect_well`: Inspect well curves, depth range, formation tops, fluid contacts, core count, and zonal stats.
- `run_well_qaqc`: Run `qaqc.py` (outer limit checks, washout/badhole flagging, tight/coal detection, and NPHI/RHOB hydrocarbon correction).
- `run_multi_mineral`: Run `multi_mineral.py` or `sand_silt_clay.py` solver for mineral volumes and porosity (`PHIT`, `PHIE`).
- `run_shf_evaluation`: Run `shf.py` FZI rock typing, Leverett-J saturation height function (`SHF`), and Free Water Level (`FWL`) reconciliation.
- `evaluate_with_conformal_bounds`: Calibrate `ConformalMultiOutputRegressor` on the fly and generate P10–P90 conformal prediction intervals (`PHIT_P10`/`P90`, `SW_P10`/`P90`, `PERM_P10`/`P90`) with volumetric closure and zonal pay confidence diagnostics.
- `control_workbench_ui`: Switch workbench tabs, filter zones/depths, or scroll the well log canvas to a target depth.

Always ground your interpretation in the quantitative tool outputs: cite depth intervals, zonal means, P10–P90 conformal interval widths, and physical constraints (volumetric closure, washout impact, FWL transition zones).
"""


# ---------------------------------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------------------------------


class ChatHistoryItem(BaseModel):
    role: str = Field(..., description="'user' or 'assistant'")
    content: str = Field(..., description="Message text")


class AgentChatRequest(BaseModel):
    message: str = Field(..., description="User prompt or petrophysical instruction")
    project_id: Optional[int] = Field(None, description="Active project ID in workbench")
    well_name: Optional[str] = Field(None, description="Active well name in workbench")
    scope: Optional[str] = Field("single-well", description="Active workbench scope")
    step: Optional[str] = Field("litho-poro", description="Active workflow step")
    min_depth: Optional[float] = Field(None, description="Active minimum depth filter")
    max_depth: Optional[float] = Field(None, description="Active maximum depth filter")
    selected_zones: Optional[List[str]] = Field(None, description="Active zone filter")
    alpha: float = Field(0.20, ge=0.02, le=0.50, description="Conformal significance level alpha (0.20 = P10-P90)")
    provider: str = Field(
        "native_mcp",
        description="LLM provider: 'native_mcp', 'openai', 'gemini', 'anthropic', or 'ollama'",
    )
    model: Optional[str] = Field(None, description="Model identifier for selected provider")
    api_key: Optional[str] = Field(None, description="Optional API key from UI drawer")
    base_url: Optional[str] = Field(None, description="Optional custom base URL (e.g., Ollama / vLLM)")
    history: List[ChatHistoryItem] = Field(default_factory=list)


class DirectToolCallRequest(BaseModel):
    name: str = Field(..., description="MCP tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Built-in Petrophysical Domain Agent (Zero-API-Key Planner + Explainer)
# ---------------------------------------------------------------------------


def _plan_native_tool_calls(req: AgentChatRequest) -> List[Dict[str, Any]]:
    """Determine which MCP tools to invoke based on user intent and workbench context."""
    q = req.message.lower()
    pid = req.project_id
    well = req.well_name
    calls: List[Dict[str, Any]] = []

    # Check for explicit UI navigation / depth jump requests
    depth_match = re.search(
        r"(?:scroll|jump|go|zoom|center)\s+(?:to\s+)?(?:depth\s+)?(\d+(?:\.\d+)?)", q
    )
    if depth_match:
        calls.append(
            {
                "name": "control_workbench_ui",
                "arguments": {"scroll_to_depth": float(depth_match.group(1))},
            }
        )

    if pid is None:
        return calls

    common_filter = {
        "project_id": int(pid),
        "well_name": well,
        "min_depth": req.min_depth,
        "max_depth": req.max_depth,
        "zones": req.selected_zones if req.selected_zones else None,
    }

    wants_full_workflow = any(
        kw in q
        for kw in (
            "full evaluation",
            "full workflow",
            "complete evaluation",
            "run all",
            "everything",
            "end-to-end",
            "full petrophysical",
        )
    )
    wants_inspect = any(
        kw in q
        for kw in (
            "inspect",
            "summary",
            "overview",
            "what curves",
            "available",
            "tops",
            "zones",
            "describe",
            "check well",
        )
    )
    wants_qaqc = any(
        kw in q
        for kw in (
            "qaqc",
            "qc",
            "badhole",
            "washout",
            "caliper",
            "hc correction",
            "hydrocarbon correction",
            "environmental",
            "outlier",
            "clean",
        )
    )
    wants_multimin = any(
        kw in q
        for kw in (
            "multi_mineral",
            "multi-mineral",
            "multimineral",
            "mineral",
            "lithology",
            "quartz",
            "calcite",
            "dolomite",
            "ssc",
            "sand silt clay",
            "vclay",
        )
    )
    wants_shf = any(
        kw in q
        for kw in (
            "shf",
            "saturation height",
            "leverett",
            "j-function",
            "j function",
            "fwl",
            "free water level",
            "capillary",
            "transition zone",
        )
    )
    wants_conformal = any(
        kw in q
        for kw in (
            "conformal",
            "uncertainty",
            "p10",
            "p90",
            "bounds",
            "interval",
            "confidence",
            "explain",
            "interpret",
            "pay",
            "evaluate",
        )
    )

    if wants_full_workflow and well:
        calls.extend(
            [
                {"name": "inspect_well", "arguments": common_filter},
                {
                    "name": "run_well_qaqc",
                    "arguments": {**common_filter, "save": True},
                },
                {
                    "name": "run_multi_mineral",
                    "arguments": {
                        **common_filter,
                        "model": "multi_mineral",
                        "save": True,
                    },
                },
                {
                    "name": "run_shf_evaluation",
                    "arguments": {
                        "project_id": int(pid),
                        "well_name": well,
                        "optimize_fwl_first": True,
                        "save": True,
                    },
                },
                {
                    "name": "evaluate_with_conformal_bounds",
                    "arguments": {
                        **common_filter,
                        "alpha": req.alpha,
                        "save": True,
                    },
                },
            ]
        )
        return calls

    if wants_inspect:
        calls.append({"name": "inspect_well", "arguments": common_filter})

    if wants_qaqc and well:
        calls.append(
            {
                "name": "run_well_qaqc",
                "arguments": {
                    **common_filter,
                    "run_env_correction": "env" in q,
                    "save": True,
                },
            }
        )

    if wants_multimin and well:
        model_type = "ssc" if ("ssc" in q or "sand silt" in q) else "multi_mineral"
        calls.append(
            {
                "name": "run_multi_mineral",
                "arguments": {
                    **common_filter,
                    "model": model_type,
                    "save": True,
                },
            }
        )

    if wants_shf:
        fwl_match = re.search(r"fwl\s*(?:=|of|at)?\s*(\d+(?:\.\d+)?)", q)
        fwl_val = float(fwl_match.group(1)) if fwl_match else 1020.0
        calls.append(
            {
                "name": "run_shf_evaluation",
                "arguments": {
                    "project_id": int(pid),
                    "well_name": well,
                    "fwl": fwl_val,
                    "optimize_fwl_first": "optim" in q,
                    "save": True,
                },
            }
        )

    if wants_conformal and well:
        calls.append(
            {
                "name": "evaluate_with_conformal_bounds",
                "arguments": {
                    **common_filter,
                    "alpha": req.alpha,
                    "save": True,
                },
            }
        )

    # Default fallback if user asks a general question on an active well: inspect + conformal bounds
    if not calls and well:
        calls.append({"name": "inspect_well", "arguments": common_filter})
        calls.append(
            {
                "name": "evaluate_with_conformal_bounds",
                "arguments": {
                    **common_filter,
                    "alpha": req.alpha,
                    "save": True,
                },
            }
        )
    elif not calls and pid is not None:
        calls.append({"name": "inspect_well", "arguments": {"project_id": int(pid)}})

    return calls


def _synthesize_petrophysical_explanation(
    question: str,
    tool_records: List[Dict[str, Any]],
) -> str:
    """Synthesize a rich, quantitative petrophysical markdown report from executed MCP tool results."""
    if not tool_records:
        return (
            "No active project or well is currently selected. Please select a project and well in the "
            "top context bar, or ask me to inspect a specific `project_id`."
        )

    sections: List[str] = []

    for rec in tool_records:
        t_name = rec["name"]
        res = rec.get("result") or {}
        if "error" in res:
            sections.append(f"### ⚠️ `{t_name}` Notice\n- {res['error']}")
            continue

        if t_name == "inspect_well":
            dr = res.get("depth_range") or {}
            curves = ", ".join(f"`{c}`" for c in res.get("available_curves", [])[:14])
            sections.append(
                f"### 🔍 Well Inspection: `{res.get('well_name')}`" + (f" (Project `{res.get('project_name')}`)\n" if res.get("project_name") else "\n") +
                f"- **Depth Interval**: `{dr.get('min')} – {dr.get('max')} m` (`{res.get('row_count')}` samples)\n"
                f"- **Core Calibration Points**: `{res.get('core_sample_count', 0)}` samples\n"
                f"- **Available Curves**: {curves}"
            )
            zb = res.get("zonal_breakdown") or []
            if zb:
                rows = [
                    "| Zone | Interval (m) | Gross (m) | GR Mean | NPHI Mean | RHOB Mean |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: |",
                ]
                for z in zb:
                    rows.append(
                        f"| **{z['zone']}** | `{z['top']}–{z['bottom']}` | `{z['gross_thickness']}` | "
                        f"`{z.get('GR_mean', '—')}` | `{z.get('NPHI_mean', '—')}` | `{z.get('RHOB_mean', '—')}` |"
                    )
                sections.append("\n".join(rows))

        elif t_name == "run_well_qaqc":
            bh = res.get("badhole") or {}
            hc = res.get("hc_correction") or {}
            outliers = res.get("outliers_by_curve") or {}
            outlier_txt = (
                ", ".join(f"`{k}` ({v} pts)" for k, v in outliers.items())
                if outliers
                else "None (all curves within physical bounds)"
            )
            bh_intervals = bh.get("intervals") or []
            bh_detail = (
                ", ".join(f"`{iv['top']}–{iv['bottom']} m`" for iv in bh_intervals[:3])
                if bh_intervals
                else "No severe washouts detected"
            )
            sections.append(
                f"### 🛡️ Log QAQC & Hydrocarbon Correction (`qaqc.py`)\n"
                f"- **Out-of-Limit Spikes**: {outlier_txt}\n"
                f"- **Badhole / Washout Flag (`BADHOLE`)**: `{bh.get('flagged_count', 0)}` samples "
                f"(`{bh.get('flagged_pct', 0)}%`) — {bh_detail}\n"
                f"- **Tight Streaks / Coal Flags**: `{res.get('tight_streak_samples', 0)}` tight pts, "
                f"`{res.get('coal_samples', 0)}` coal pts\n"
                f"- **NPHI/RHOB Hydrocarbon Correction**: Rotated at `{hc.get('corr_angle_deg', '—')}°`; "
                f"`{hc.get('hc_affected_samples', 0)}` samples (`{hc.get('hc_affected_pct', 0)}%`) flagged & corrected (`NPHI_HC`, `RHOB_HC` synced to Track 3)."
            )

        elif t_name == "run_multi_mineral":
            st = res.get("stats") or {}
            diag = res.get("solver_diagnostics") or {}
            phit_m = (st.get("PHIT") or {}).get("mean", "—")
            phie_m = (st.get("PHIE") or {}).get("mean", "—")
            vclay_m = (st.get("VCLAY") or {}).get("mean", "—")
            vsand_m = (st.get("VSAND") or {}).get("mean", "—")
            conv = diag.get("convergence_rate_pct", 100.0)
            err = diag.get("mean_reconstruction_error", "—")
            sections.append(
                f"### 🪨 Lithology & Porosity Optimization (`{res.get('model')}`)\n"
                f"- **Minerals Solved**: {', '.join(f'`{m}`' for m in res.get('minerals', []))}\n"
                f"- **Solver Convergence**: `{conv}%` (Mean log reconstruction error: `{err}`)\n"
                f"- **Mean Volumetric Fractions**: `VSAND/VQTZ = {vsand_m}`, `VCLAY/VSHA = {vclay_m}`, "
                f"`PHIT = {phit_m} v/v`, `PHIE = {phie_m} v/v` (synced to Track 4)."
            )

        elif t_name == "run_shf_evaluation":
            rec_cmp = res.get("sw_vs_shf_reconciliation") or {}
            shf_st = (res.get("stats") or {}).get("SHF") or {}
            cmp_line = (
                f"- **Sw vs. SHF Reconciliation**: RMSE = `{rec_cmp.get('rmse_sw_vs_shf')}` v/v, "
                f"MAE = `{rec_cmp.get('mae_sw_vs_shf')}` v/v (Mean bias: `{rec_cmp.get('mean_bias_sw_minus_shf')}`)"
                if rec_cmp
                else "- **SHF Curve**: Synced to Track 6 (`SHF` dashed red curve)."
            )
            sections.append(
                f"### 💧 Saturation Height Function (`shf.py`)\n"
                f"- **Free Water Level (FWL)**: `{res.get('fwl_used')} m` | **FZI Cutoffs**: `{res.get('fzi_cutoffs')}`\n"
                f"- **SHF Summary**: Mean = `{shf_st.get('mean', '—')} v/v`, P50 = `{shf_st.get('p50', '—')} v/v`\n"
                f"{cmp_line}"
            )

        elif t_name == "evaluate_with_conformal_bounds":
            conf_pct = res.get("confidence_level_pct", 80.0)
            iw = res.get("interval_width_summary") or {}
            q_hats = res.get("conformal_quantiles_q_hat") or {}
            anchors = res.get("core_anchors") or {}
            sections.append(
                f"### 📊 Conformal Prediction Bounds (`{conf_pct}%` P10–P90 Calibration)\n"
                f"- **On-the-Fly Calibration**: Locally Weighted Split Conformal (`ConformalMultiOutputRegressor`) "
                f"calibrated across `{res.get('calibration_samples')}` depth samples "
                f"(`{anchors.get('core_porosity_samples', 0)}` `CPORE` & `{anchors.get('core_permeability_samples', 0)}` `CPERM` core anchors).\n"
                f"- **Normalized Conformal Quantiles (q̂₁₋ₐ)**: "
                f"`PHIT = {q_hats.get('PHIT', '—')}`, `SWT = {q_hats.get('SWT', '—')}`, `LOG_PERM = {q_hats.get('LOG_PERM', '—')}`\n"
                f"- **Mean P10–P90 Interval Widths**:\n"
                f"  - Porosity (`PHIT_P10` to `PHIT_P90`): **`±{round(iw.get('phit_mean_width_vv', 0) / 2, 4)} v/v`** (total width `{iw.get('phit_mean_width_vv')} v/v`)\n"
                f"  - Water Saturation (`SW_P10` to `SW_P90`): **`±{round(iw.get('sw_mean_width_vv', 0) / 2, 4)} v/v`** (total width `{iw.get('sw_mean_width_vv')} v/v`)\n"
                f"  - Permeability (`PERM_P90 / PERM_P10`): **`{iw.get('perm_median_p90_p10_factor')}×` spread**"
            )

            z_conf = res.get("zonal_conformal_summary") or []
            if z_conf:
                z_rows = [
                    "| Zone | Interval (m) | PHIT Mean `[P10, P90]` | Sw Mean `[P10, P90]` | Certifiable Pay (P10) | Upside Pay (P90) |",
                    "| :--- | :--- | :---: | :---: | :---: | :---: |",
                ]
                for z in z_conf:
                    pb = z["phit_p10_p90_band"]
                    sb = z["sw_p10_p90_band"]
                    z_rows.append(
                        f"| **{z['zone']}** | `{z['top']}–{z['bottom']}` | "
                        f"`{z['phit_mean']}` `[{pb[0]}, {pb[1]}]` | "
                        f"`{z['sw_mean']}` `[{sb[0]}, {sb[1]}]` | "
                        f"**`{round(z['certifiable_pay_fraction'] * 100, 1)}%`** | "
                        f"`{round(z['upside_pay_fraction'] * 100, 1)}%` |"
                    )
                sections.append("\n".join(z_rows))

            hi_unc = res.get("high_uncertainty_intervals") or []
            if hi_unc:
                hi_str = ", ".join(
                    f"`{iv['top']}–{iv['bottom']} m` ({iv['thickness']} m)"
                    for iv in hi_unc[:3]
                )
                sections.append(
                    f"> **Physical Interpretation of Conformal Widths**: Uncertainty bands widen most in "
                    f"{hi_str} due to higher ensemble tree dispersion σ̂(x) across heterogeneous/tight "
                    f"transitions, while remaining tight in clean reservoir intervals where volumetric closure "
                    f"(∑ V_mineral + PHIT = 1.0) constrains the solution."
                )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# External LLM Provider Orchestration (OpenAI / Ollama / Gemini / Anthropic)
# ---------------------------------------------------------------------------


def _run_openai_compatible_agent(
    req: AgentChatRequest,
    context_summary: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """Run tool-calling loop against OpenAI or Ollama (`/v1/chat/completions`)."""
    provider = req.provider.lower()
    if provider == "ollama":
        base_url = (req.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        api_key = req.api_key or "ollama"
        model = req.model or "qwen2.5:7b"
    else:
        base_url = (req.base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")).rstrip("/")
        api_key = req.api_key or os.getenv("OPENAI_API_KEY", "")
        model = req.model or "gpt-4o-mini"
        if not api_key:
            raise ValueError(
                "OpenAI API key is required. Enter it in the AI Assistant settings or set OPENAI_API_KEY."
            )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context_summary}
    ]
    for h in req.history[-6:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": req.message})

    tools = petro_mcp_server.get_openai_tools()
    executed_tools: List[Dict[str, Any]] = []

    for _ in range(4):
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": 0.2,
        }
        resp = requests.post(
            f"{base_url}/chat/completions", headers=headers, json=payload, timeout=45
        )
        resp.raise_for_status()
        data = resp.json()
        choice = data["choices"][0]["message"]
        messages.append(choice)

        tool_calls = choice.get("tool_calls") or []
        if not tool_calls:
            return choice.get("content") or "", executed_tools

        for tc in tool_calls:
            fn_name = tc["function"]["name"]
            raw_args = tc["function"].get("arguments") or "{}"
            fn_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            try:
                tool_out = petro_mcp_server.call_tool(fn_name, fn_args)
            except Exception as e:
                tool_out = {"error": str(e)}
            executed_tools.append(
                {"name": fn_name, "arguments": fn_args, "result": tool_out}
            )
            compact_res = {k: v for k, v in tool_out.items() if k != "computed_curves"}
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "name": fn_name,
                    "content": json.dumps(compact_res, default=str),
                }
            )

    return messages[-1].get("content") or "", executed_tools


def _run_anthropic_agent(
    req: AgentChatRequest,
    context_summary: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """Run tool-calling loop against Anthropic Messages API."""
    api_key = req.api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Anthropic API key is required. Enter it in the AI Assistant settings or set ANTHROPIC_API_KEY."
        )
    model = req.model or "claude-3-5-sonnet-latest"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    messages: List[Dict[str, Any]] = [
        {"role": "user", "content": f"[Workbench Context: {context_summary}]\n\n{req.message}"}
    ]
    tools = petro_mcp_server.get_anthropic_tools()
    executed_tools: List[Dict[str, Any]] = []

    for _ in range(4):
        payload = {
            "model": model,
            "max_tokens": 2048,
            "system": SYSTEM_PROMPT,
            "messages": messages,
            "tools": tools,
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=45,
        )
        resp.raise_for_status()
        data = resp.json()
        content_blocks = data.get("content", [])
        messages.append({"role": "assistant", "content": content_blocks})

        tool_uses = [b for b in content_blocks if b.get("type") == "tool_use"]
        if not tool_uses:
            text_parts = [b.get("text", "") for b in content_blocks if b.get("type") == "text"]
            return "\n".join(text_parts), executed_tools

        tool_results_blocks = []
        for tu in tool_uses:
            fn_name = tu["name"]
            fn_args = tu.get("input") or {}
            try:
                tool_out = petro_mcp_server.call_tool(fn_name, fn_args)
            except Exception as e:
                tool_out = {"error": str(e)}
            executed_tools.append(
                {"name": fn_name, "arguments": fn_args, "result": tool_out}
            )
            compact_res = {k: v for k, v in tool_out.items() if k != "computed_curves"}
            tool_results_blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tu["id"],
                    "content": json.dumps(compact_res, default=str),
                }
            )
        messages.append({"role": "user", "content": tool_results_blocks})

    return "", executed_tools


def _run_gemini_agent(
    req: AgentChatRequest,
    context_summary: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """Run tool-calling loop against Google Gemini REST API."""
    api_key = req.api_key or os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError(
            "Gemini API key is required. Enter it in the AI Assistant settings or set GEMINI_API_KEY."
        )
    model = req.model or "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"

    # Execute planned MCP tools first and let Gemini synthesize or call additional tools
    executed_tools: List[Dict[str, Any]] = []
    for call_spec in _plan_native_tool_calls(req):
        try:
            t_out = petro_mcp_server.call_tool(call_spec["name"], call_spec["arguments"])
        except Exception as e:
            t_out = {"error": str(e)}
        executed_tools.append(
            {
                "name": call_spec["name"],
                "arguments": call_spec["arguments"],
                "result": t_out,
            }
        )

    compact_context = [
        {
            "tool": r["name"],
            "result": {k: v for k, v in r["result"].items() if k != "computed_curves"},
        }
        for r in executed_tools
    ]
    prompt = (
        f"{SYSTEM_PROMPT}\n\n{context_summary}\n\n"
        f"Executed MCP Tool Results:\n{json.dumps(compact_context, default=str, indent=2)}\n\n"
        f"User Request: {req.message}"
    )
    resp = requests.post(
        url,
        json={"contents": [{"parts": [{"text": prompt}]}]},
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()
    candidates = data.get("candidates") or []
    if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        text = "\n".join(p.get("text", "") for p in parts if "text" in p)
        if text.strip():
            return text, executed_tools

    return _synthesize_petrophysical_explanation(req.message, executed_tools), executed_tools


# ---------------------------------------------------------------------------
# FastAPI Routes (`/quick_pp/assistant/*` & `/quick_pp/mcp/*`)
# ---------------------------------------------------------------------------


@router.get("/assistant/providers", summary="List supported LLM & MCP providers for UI switcher")
async def list_assistant_providers():
    """Return available agent providers, models, and MCP tool definitions for the Svelte UI drawer."""
    providers_out = []
    for p in PROVIDERS_CATALOG:
        env_k = p.get("env_key")
        has_env_key = bool(os.getenv(env_k)) if env_k else True
        providers_out.append({**p, "server_key_configured": has_env_key})
    return {
        "providers": providers_out,
        "mcp_tools": petro_mcp_server.list_tools(),
    }


@router.post("/assistant/chat", summary="Run Petrophysical Agentic Assistant with MCP tools")
async def assistant_chat(req: AgentChatRequest):
    """Execute petrophysical MCP tools and return explanation, conformal bounds, computed curves, and UI actions."""
    context_summary = (
        f"Active Workbench Context: project_id={req.project_id}, well_name={req.well_name}, "
        f"scope={req.scope}, step={req.step}, depth_filter=({req.min_depth}, {req.max_depth}), "
        f"selected_zones={req.selected_zones}, conformal_alpha={req.alpha}"
    )

    provider = (req.provider or "native_mcp").lower()
    executed_tools: List[Dict[str, Any]] = []
    reply_text = ""
    used_fallback = False

    try:
        if provider in ("openai", "ollama"):
            reply_text, executed_tools = _run_openai_compatible_agent(req, context_summary)
        elif provider == "anthropic":
            reply_text, executed_tools = _run_anthropic_agent(req, context_summary)
        elif provider == "gemini":
            reply_text, executed_tools = _run_gemini_agent(req, context_summary)
        else:
            # Built-in Petrophysical MCP Agent
            planned = _plan_native_tool_calls(req)
            for call_spec in planned:
                try:
                    t_out = petro_mcp_server.call_tool(
                        call_spec["name"], call_spec["arguments"]
                    )
                except Exception as tool_err:
                    logger.warning(f"MCP tool '{call_spec['name']}' error: {tool_err}")
                    t_out = {"error": str(tool_err)}
                executed_tools.append(
                    {
                        "name": call_spec["name"],
                        "arguments": call_spec["arguments"],
                        "result": t_out,
                    }
                )
            reply_text = _synthesize_petrophysical_explanation(req.message, executed_tools)
    except Exception as exc:
        logger.warning(f"External provider '{provider}' failed ({exc}); falling back to native_mcp agent.")
        used_fallback = True
        if not executed_tools:
            for call_spec in _plan_native_tool_calls(req):
                try:
                    t_out = petro_mcp_server.call_tool(
                        call_spec["name"], call_spec["arguments"]
                    )
                    executed_tools.append(
                        {
                            "name": call_spec["name"],
                            "arguments": call_spec["arguments"],
                            "result": t_out,
                        }
                    )
                except Exception as tool_err:
                    executed_tools.append(
                        {
                            "name": call_spec["name"],
                            "arguments": call_spec["arguments"],
                            "result": {"error": str(tool_err)},
                        }
                    )
        fallback_body = _synthesize_petrophysical_explanation(req.message, executed_tools)
        reply_text = (
            f"*(Note: Provider `{provider}` returned `{exc}`; executed built-in Petrophysical MCP Agent)*\n\n"
            + fallback_body
        )

    # Aggregate computed curves, stats, and UI actions across all executed tools for live Svelte sync
    merged_curves: Dict[str, List[Any]] = {}
    merged_stats: Dict[str, Any] = {}
    ui_actions: List[Dict[str, Any]] = []
    compact_tool_calls: List[Dict[str, Any]] = []

    for rec in executed_tools:
        res = rec.get("result") or {}
        if isinstance(res, dict):
            if "computed_curves" in res and isinstance(res["computed_curves"], dict):
                merged_curves.update(res["computed_curves"])
            if "stats" in res and isinstance(res["stats"], dict):
                merged_stats.update(res["stats"])
            if "ui_action" in res and isinstance(res["ui_action"], dict):
                ui_actions.append(res["ui_action"])
            compact_res = {k: v for k, v in res.items() if k != "computed_curves"}
        else:
            compact_res = res

        compact_tool_calls.append(
            {
                "name": rec["name"],
                "arguments": rec["arguments"],
                "summary": compact_res,
            }
        )

    return {
        "reply": reply_text,
        "provider": provider,
        "used_fallback": used_fallback,
        "tool_calls": compact_tool_calls,
        "computed_curves": merged_curves,
        "stats": merged_stats,
        "ui_actions": ui_actions,
    }


@router.get("/mcp/tools", summary="List registered Petrophysical MCP tools")
async def list_mcp_tools():
    """Return MCP tool definitions and JSON schemas."""
    return {"server": petro_mcp_server.name, "tools": petro_mcp_server.list_tools()}


@router.post("/mcp/call", summary="Directly execute a Petrophysical MCP tool")
async def call_mcp_tool_endpoint(req: DirectToolCallRequest):
    """Execute a specific MCP tool by name and return structured results + computed curves."""
    try:
        return petro_mcp_server.call_tool(req.name, req.arguments)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MCP tool execution failed: {e}") from e


@router.post("/mcp", summary="Standard MCP JSON-RPC 2.0 Streamable HTTP Endpoint")
async def mcp_jsonrpc_endpoint(payload: Dict[str, Any]):
    """Standard Model Context Protocol JSON-RPC 2.0 endpoint (`initialize`, `tools/list`, `tools/call`)."""
    resp = petro_mcp_server.handle_jsonrpc(payload)
    return resp or {"jsonrpc": "2.0", "result": {}}
