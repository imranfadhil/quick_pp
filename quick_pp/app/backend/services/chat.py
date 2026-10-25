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
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import socket
import urllib3.connection as urllib3_conn
import urllib3.util.connection as urllib3_cn
from contextlib import contextmanager
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from quick_pp import logger
from quick_pp.app.backend.mcp_server import petro_mcp_server

router = APIRouter(tags=["Agentic Assistant & MCP"])

_REPO_ROOT_ENV = Path(__file__).resolve().parents[4] / ".env"


def _get_env_key(name: str, default: str = "") -> str:
    """Load root `.env` if running locally and return environment variable value."""
    if _REPO_ROOT_ENV.is_file():
        load_dotenv(_REPO_ROOT_ENV, override=False)
    return os.getenv(name, default).strip()


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
        "id": "openrouter",
        "name": "OpenRouter (DeepSeek / Gemini / Claude / GPT)",
        "requires_key": True,
        "default_model": "deepseek/deepseek-chat",
        "models": [
            "deepseek/deepseek-chat",
            "deepseek/deepseek-r1",
            "google/gemini-2.5-flash",
            "google/gemini-2.5-pro",
            "anthropic/claude-3.7-sonnet",
            "openai/gpt-4o-mini",
            "qwen/qwen-2.5-72b-instruct",
        ],
        "env_key": "OPENROUTER_API_KEY",
        "default_base_url": "https://openrouter.ai/api/v1",
    },
    {
        "id": "ollama",
        "name": "Local Ollama (Qwen 2.5 / DeepSeek R1 / Llama 3.2)",
        "requires_key": False,
        "default_model": "qwen2.5:7b",
        "models": ["qwen2.5:7b", "deepseek-r1:8b", "llama3.2:latest", "mistral:latest"],
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
- `get_project_report_context`: Retrieve compact project wells, zones, QAQC flags, rock types, FWL, and P10/P50/P90 sums for report drafting.
- `generate_petrophysical_report`: Generate a complete FDP Petrophysical Evaluation Report (`FDP_FULL`, `SINGLE_WELL_CPI`, `QAQC_SUMMARY`, `RESSUM_ONLY`) with high-DPI plots and open the Report Workbench.
- `update_report_section`: Rewrite or refine a specific section (`exec_summary`, `data_inventory`, `qaqc`, `litho_poro`, `rock_perm`, `saturation`, `shf_scal`, `ressum_uncert`, `upscaling`, `appendix_cpi`) in the active report.
- `export_report_to_pdf`: Compile the active report into a downloadable A4 PDF document.

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
    project_id: Optional[int] = Field(
        None, description="Active project ID in workbench"
    )
    well_name: Optional[str] = Field(None, description="Active well name in workbench")
    scope: Optional[str] = Field("single-well", description="Active workbench scope")
    step: Optional[str] = Field("litho-poro", description="Active workflow step")
    min_depth: Optional[float] = Field(None, description="Active minimum depth filter")
    max_depth: Optional[float] = Field(None, description="Active maximum depth filter")
    selected_zones: Optional[List[str]] = Field(None, description="Active zone filter")
    alpha: float = Field(
        0.20,
        ge=0.02,
        le=0.50,
        description="Conformal significance level alpha (0.20 = P10-P90)",
    )
    provider: str = Field(
        "native_mcp",
        description="LLM provider: 'native_mcp', 'openai', 'gemini', 'anthropic', or 'ollama'",
    )
    model: Optional[str] = Field(
        None, description="Model identifier for selected provider"
    )
    api_key: Optional[str] = Field(None, description="Optional API key from UI drawer")
    base_url: Optional[str] = Field(
        None, description="Optional custom base URL (e.g., Ollama / vLLM)"
    )
    history: List[ChatHistoryItem] = Field(default_factory=list)
    active_report_id: Optional[str] = Field(
        None, description="Active report ID in workbench if any"
    )


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

    # Report Generation & PDF Export Intents
    wants_pdf_export = any(
        kw in q
        for kw in (
            "export pdf",
            "export report to pdf",
            "compile pdf",
            "download pdf",
            "export to pdf",
            "generate pdf",
        )
    )
    wants_section_rewrite = any(
        kw in q
        for kw in (
            "rewrite section",
            "update section",
            "refine section",
            "rewrite the executive summary",
            "rewrite executive summary",
            "update executive summary",
            "rewrite qaqc section",
            "rewrite saturation section",
        )
    )
    wants_report = any(
        kw in q
        for kw in (
            "fdp report",
            "petrophysical report",
            "generate report",
            "build report",
            "create report",
            "draft report",
            "write report",
            "single well cpi report",
            "qaqc summary report",
            "ressum report",
        )
    )

    if wants_pdf_export:
        calls.append(
            {
                "name": "export_report_to_pdf",
                "arguments": {
                    "project_id": int(pid),
                    "report_id": req.active_report_id or "latest",
                },
            }
        )
        return calls

    if wants_section_rewrite:
        sec_map = {
            "exec": "exec_summary",
            "executive": "exec_summary",
            "inventory": "data_inventory",
            "qaqc": "qaqc",
            "litho": "litho_poro",
            "porosity": "litho_poro",
            "rock": "rock_perm",
            "perm": "rock_perm",
            "saturation": "saturation",
            "fwl": "saturation",
            "shf": "shf_scal",
            "scal": "shf_scal",
            "ressum": "ressum_uncert",
            "uncertainty": "ressum_uncert",
            "upscal": "upscaling",
            "appendix": "appendix_cpi",
            "cpi": "appendix_cpi",
        }
        target_sec = "exec_summary"
        for key_sub, sid in sec_map.items():
            if key_sub in q:
                target_sec = sid
                break
        from quick_pp.app.backend.services.report import (
            build_report_context,
            synthesize_domain_ai_commentary,
        )

        ctx = build_report_context(
            project_id=str(pid),
            well_names=[well] if well else None,
        )
        ai_comm = synthesize_domain_ai_commentary(ctx)
        comm_text = ai_comm.get("executive_summary", "")
        if target_sec == "qaqc":
            comm_text = ai_comm.get("qaqc_commentary", comm_text)
        elif target_sec == "litho_poro":
            comm_text = ai_comm.get("litho_poro_commentary", comm_text)
        elif target_sec == "rock_perm":
            comm_text = ai_comm.get("rock_typing_commentary", comm_text)
        elif target_sec == "saturation":
            comm_text = ai_comm.get("saturation_commentary", comm_text)
        elif target_sec == "shf_scal":
            comm_text = ai_comm.get("shf_commentary", comm_text)
        elif target_sec == "ressum_uncert":
            comm_text = ai_comm.get("uncertainty_commentary", comm_text)
        elif target_sec == "upscaling":
            comm_text = ai_comm.get("upscaling_commentary", comm_text)

        custom_md = (
            f"## AI-Refined Section (`{target_sec}`)\n\n"
            f"> **Prompt Note**: {req.message}\n\n"
            f"{comm_text}\n"
        )
        calls.append(
            {
                "name": "update_report_section",
                "arguments": {
                    "project_id": int(pid),
                    "report_id": req.active_report_id or "latest",
                    "section_id": target_sec,
                    "markdown_content": custom_md,
                },
            }
        )
        return calls

    if wants_report:
        rep_type = "FDP_FULL"
        if "single_well" in q or "single-well" in q or "cpi" in q:
            rep_type = "SINGLE_WELL_CPI"
        elif "qaqc" in q and "summary" in q:
            rep_type = "QAQC_SUMMARY"
        elif "ressum" in q and "only" in q:
            rep_type = "RESSUM_ONLY"
        calls.append(
            {
                "name": "generate_petrophysical_report",
                "arguments": {
                    "project_id": int(pid),
                    "report_type": rep_type,
                    "well_names": [well]
                    if (well and rep_type == "SINGLE_WELL_CPI")
                    else None,
                    "include_ai_summary": True,
                },
            }
        )
        return calls

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
                f"### 🔍 Well Inspection: `{res.get('well_name')}`"
                + (
                    f" (Project `{res.get('project_name')}`)\n"
                    if res.get("project_name")
                    else "\n"
                )
                + f"- **Depth Interval**: `{dr.get('min')} – {dr.get('max')} m` (`{res.get('row_count')}` samples)\n"
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

        elif t_name == "get_project_report_context":
            sections.append(
                f"### 📋 Project Report Context (`{res.get('project_name')}`)\n"
                f"- **Evaluated Wells**: `{res.get('well_count', 0)}` (`{', '.join(res.get('well_names', []))}`)\n"
                f"- **Fluid Contacts**: FWL = `{res.get('fluid_contacts', {}).get('fwl')} m`, "
                f"OWC = `{res.get('fluid_contacts', {}).get('owc')} m`"
            )

        elif t_name == "generate_petrophysical_report":
            st_sum = res.get("stats_summary") or {}
            assets = res.get("assets") or []
            sections.append(
                f"### 📄 FDP Petrophysical Evaluation Report Generated (`{res.get('report_id')}`)\n"
                f"- **Report Template**: `{res.get('report_type', 'FDP_FULL')}`\n"
                f"- **Wells Included**: `{st_sum.get('well_count', 0)}` wells | **Zonal Pay Rows**: `{st_sum.get('ressum_row_count', 0)}`\n"
                f"- **Mean Pay Metrics**: Net Pay = `{st_sum.get('mean_net_pay_m', '—')} m`, "
                f"`PHIE = {st_sum.get('mean_phie_pay', '—')} v/v`, `Sw = {st_sum.get('mean_sw_pay', '—')} v/v`\n"
                f"- **High-DPI Figures Exported**: `{len(assets)}` PNG assets rendered and embedded in the Report Workbench."
            )

        elif t_name == "update_report_section":
            sections.append(
                f"### ✏️ Report Section Updated (`{res.get('section_id')}`)\n"
                f"- **Report ID**: `{res.get('report_id')}`\n"
                f"- **Updated Timestamp**: `{res.get('updated_at')}`\n"
                f"- The live preview in the Report Workbench has been refreshed with the updated Markdown."
            )

        elif t_name == "export_report_to_pdf":
            sections.append(
                f"### 🖨️ PDF Export Compiled (`{res.get('report_id')}`)\n"
                f"- **Download PDF**: `{res.get('pdf_url')}`\n"
                f"- **A4 Print View**: `{res.get('print_url')}`"
            )

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# External LLM Provider Orchestration (OpenRouter & Local Ollama)
# ---------------------------------------------------------------------------


@contextmanager
def _force_ipv4_context():
    """Force IPv4 (AF_INET) and clamp TCP MSS to avoid Docker IPv6/MTU blackhole read timeouts."""
    orig_gai = urllib3_cn.allowed_gai_family
    orig_getaddrinfo = socket.getaddrinfo
    orig_sock_opts = list(urllib3_conn.HTTPConnection.default_socket_options)

    def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
        return orig_getaddrinfo(host, port, socket.AF_INET, type, proto, flags)

    urllib3_cn.allowed_gai_family = lambda: socket.AF_INET
    socket.getaddrinfo = _ipv4_getaddrinfo
    if os.name != "nt" and hasattr(socket, "TCP_MAXSEG"):
        urllib3_conn.HTTPConnection.default_socket_options = orig_sock_opts + [
            (socket.IPPROTO_TCP, socket.TCP_MAXSEG, 1360)
        ]
    try:
        yield
    finally:
        urllib3_cn.allowed_gai_family = orig_gai
        socket.getaddrinfo = orig_getaddrinfo
        urllib3_conn.HTTPConnection.default_socket_options = orig_sock_opts


def _run_openai_compatible_agent(
    req: AgentChatRequest,
    context_summary: str,
) -> tuple[str, List[Dict[str, Any]]]:
    """Run petrophysical MCP synthesis and tool-calling against OpenRouter or Local Ollama (`/v1/chat/completions`)."""
    provider = req.provider.lower()
    if provider == "ollama":
        base_url = (
            req.base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        ).rstrip("/")
        api_key = req.api_key or "ollama"
        model = req.model or "qwen2.5:7b"
    else:
        base_url = (
            req.base_url
            or _get_env_key("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        api_key = (req.api_key or "").strip() or _get_env_key("OPENROUTER_API_KEY", "")
        model = req.model or "deepseek/deepseek-chat"
        if not api_key:
            raise ValueError(
                "OpenRouter API key is required. Enter it in the AI Assistant settings or set OPENROUTER_API_KEY."
            )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://github.com/imranfadhil/quick_pp",
        "X-Title": "quick_pp Petrophysical Workbench",
    }

    # Execute planned petrophysical MCP tools first so computed curves, conformal bounds, and UI actions are always ready
    executed_tools: List[Dict[str, Any]] = []
    for call_spec in _plan_native_tool_calls(req):
        try:
            t_out = petro_mcp_server.call_tool(
                call_spec["name"], call_spec["arguments"]
            )
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
            "result": {
                k: v
                for k, v in r["result"].items()
                if k not in ("computed_curves", "report_data")
            },
        }
        for r in executed_tools
    ]

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n{context_summary}\n\n"
                f"Executed MCP Tool Results:\n{json.dumps(compact_context, default=str, separators=(',', ':'))[:6000]}"
            ),
        }
    ]
    for h in req.history[-6:]:
        messages.append({"role": h.role, "content": h.content})
    messages.append({"role": "user", "content": req.message})

    tools = petro_mcp_server.get_openai_tools()
    use_tools = not any(tag in model.lower() for tag in ("deepseek-r1", ":free"))

    with _force_ipv4_context(), requests.Session() as session:
        for _ in range(3):
            payload: Dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": 0.2,
            }
            if use_tools and not executed_tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"

            resp = session.post(
                f"{base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=(10, 60),
            )
            if not resp.ok and "tools" in payload:
                # Some OpenRouter models do not support tool schemas; retry plain completion
                payload.pop("tools", None)
                payload.pop("tool_choice", None)
                resp = session.post(
                    f"{base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=(10, 60),
                )
            resp.raise_for_status()
            data = resp.json()
            choice = (data.get("choices") or [{}])[0].get("message") or {}
            messages.append(choice)

            tool_calls = choice.get("tool_calls") or []
            if not tool_calls:
                text = choice.get("content") or ""
                if text.strip():
                    return text, executed_tools
                break

            for tc in tool_calls:
                fn_name = tc["function"]["name"]
                raw_args = tc["function"].get("arguments") or "{}"
                fn_args = (
                    json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                )
                try:
                    tool_out = petro_mcp_server.call_tool(fn_name, fn_args)
                except Exception as e:
                    tool_out = {"error": str(e)}
                executed_tools.append(
                    {"name": fn_name, "arguments": fn_args, "result": tool_out}
                )
                compact_res = {
                    k: v
                    for k, v in tool_out.items()
                    if k not in ("computed_curves", "report_data")
                }
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "name": fn_name,
                        "content": json.dumps(compact_res, default=str),
                    }
                )

    return _synthesize_petrophysical_explanation(
        req.message, executed_tools
    ), executed_tools


# ---------------------------------------------------------------------------
# FastAPI Routes (`/quick_pp/assistant/*` & `/quick_pp/mcp/*`)
# ---------------------------------------------------------------------------


@router.get(
    "/assistant/providers", summary="List supported LLM & MCP providers for UI switcher"
)
async def list_assistant_providers():
    """Return available agent providers, models, and MCP tool definitions for the Svelte UI drawer."""
    providers_out = []
    for p in PROVIDERS_CATALOG:
        env_k = p.get("env_key")
        has_env_key = bool(_get_env_key(env_k)) if env_k else True
        providers_out.append({**p, "server_key_configured": has_env_key})
    return {
        "providers": providers_out,
        "mcp_tools": petro_mcp_server.list_tools(),
    }


@router.post(
    "/assistant/chat", summary="Run Petrophysical Agentic Assistant with MCP tools"
)
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
        if provider in ("openrouter", "ollama", "openai", "gemini", "anthropic"):
            reply_text, executed_tools = _run_openai_compatible_agent(
                req, context_summary
            )
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
            reply_text = _synthesize_petrophysical_explanation(
                req.message, executed_tools
            )
    except Exception as exc:
        logger.warning(
            f"External provider '{provider}' failed ({exc}); falling back to native_mcp agent."
        )
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
        fallback_body = _synthesize_petrophysical_explanation(
            req.message, executed_tools
        )
        reply_text = (
            f"*(Note: Provider `{provider}` returned `{exc}`; executed built-in Petrophysical MCP Agent)*\n\n"
            + fallback_body
        )

    # Aggregate computed curves, stats, report_data, and UI actions across all executed tools for live Svelte sync
    merged_curves: Dict[str, List[Any]] = {}
    merged_stats: Dict[str, Any] = {}
    ui_actions: List[Dict[str, Any]] = []
    compact_tool_calls: List[Dict[str, Any]] = []
    latest_report_data: Optional[Dict[str, Any]] = None

    for rec in executed_tools:
        res = rec.get("result") or {}
        if isinstance(res, dict):
            if "computed_curves" in res and isinstance(res["computed_curves"], dict):
                merged_curves.update(res["computed_curves"])
            if "stats" in res and isinstance(res["stats"], dict):
                merged_stats.update(res["stats"])
            if "report_data" in res and isinstance(res["report_data"], dict):
                latest_report_data = res["report_data"]
            if "ui_action" in res and isinstance(res["ui_action"], dict):
                ui_actions.append(res["ui_action"])
            compact_res = {
                k: v
                for k, v in res.items()
                if k not in ("computed_curves", "report_data")
            }
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
        "report_data": latest_report_data,
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
        raise HTTPException(
            status_code=500, detail=f"MCP tool execution failed: {e}"
        ) from e


@router.post("/mcp", summary="Standard MCP JSON-RPC 2.0 Streamable HTTP Endpoint")
async def mcp_jsonrpc_endpoint(payload: Dict[str, Any]):
    """Standard Model Context Protocol JSON-RPC 2.0 endpoint (`initialize`, `tools/list`, `tools/call`)."""
    resp = petro_mcp_server.handle_jsonrpc(payload)
    return resp or {"jsonrpc": "2.0", "result": {}}
