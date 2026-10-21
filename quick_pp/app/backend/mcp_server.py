"""Petrophysical MCP Server & Agentic Tool Registry for quick_pp.

Exposes compute-by-reference petrophysical tools over:
1. Standard Model Context Protocol (MCP) JSON-RPC 2.0 (Streamable HTTP `/quick_pp/mcp` & stdio `quick_pp mcp`)
2. Direct in-process Python invocation for the built-in Petrophysical Agentic Assistant
3. FastAPI REST discovery and execution endpoints

All tools operate directly on the server-side well DataFrame cache (`_get_merged_dataframe_sync`)
so neither the LLM nor the frontend ever has to pass 10,000+ log rows through tool arguments.
"""

from __future__ import annotations

import inspect
import json
import math
import sys
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from quick_pp import logger
from quick_pp.app.backend.services.database import (
    _get_merged_dataframe_sync,
    dataframe_to_columnar,
    dataframe_to_records,
    filter_merged_dataframe,
    get_project_summary,
    invalidate_well_cache,
)
from quick_pp.app.backend.services.rock_typing import _compute_fzi_data_sync
from quick_pp.app.backend.services.workspace import (
    WorkspaceHCCorrectionRequest,
    WorkspaceLithologyRequest,
    WorkspaceOptimizeFwlRequest,
    WorkspacePermeabilityRequest,
    WorkspacePorosityRequest,
    WorkspaceReservoirSummaryRequest,
    WorkspaceSaturationRequest,
    WorkspaceShfRequest,
    _curve_summary_stats,
    _get_filtered_well_df,
    _save_curves_to_well,
    _select_nphi_rhob,
    workspace_hc_correction,
    workspace_lithology,
    workspace_optimize_fwl,
    workspace_shf_compute,
)
from quick_pp.app.backend.utils.db import get_db
from quick_pp.config import Config
from quick_pp.database import objects as db_objects
from quick_pp.lithology.multi_mineral import MultiMineral
from quick_pp.lithology.sand_silt_clay import SandSiltClay
from quick_pp.machine_learning.config import CONFORMAL_ALPHA, FE_FEATURES, RAW_FEATURES
from quick_pp.machine_learning.feature_engineering import (
    acoustic_impedance_feature,
    apparent_matrix_density,
    coal_flagging,
    density_porosity as fe_density_porosity,
    gas_xover,
    mn_lithology_parameters,
    rhob_integral,
    rock_flag_gr,
    tight_streak_flagging,
)
from quick_pp.machine_learning.utils import (
    ConformalMultiOutputRegressor,
    EnforceMineralSum,
    enforce_physical_constraints,
)
from quick_pp.permeability import choo_permeability
from quick_pp.porosity import (
    density_porosity,
    estimate_shale_porosity,
    neu_den_xplot_poro,
    rho_matrix,
)
from quick_pp.qaqc import (
    apply_environmental_corrections,
    badhole_flagging,
    handle_outer_limit,
    neu_den_xplot_hc_correction,
)
from quick_pp.ressum import calc_reservoir_summary
from quick_pp.saturation import (
    archie_saturation,
    estimate_b_waxman_smits,
    estimate_qvn,
    estimate_rw_temperature_salinity,
    estimate_temperature_gradient,
    waxman_smits_saturation,
)


# ---------------------------------------------------------------------------
# Tool Metadata & Registry
# ---------------------------------------------------------------------------


class MCPToolDefinition(BaseModel):
    name: str
    description: str
    input_schema: Dict[str, Any]
    fn: Callable[..., Dict[str, Any]] = Field(exclude=True)


class PetrophysicalMCPServer:
    """Dual-mode MCP Server supporting JSON-RPC 2.0 (stdio / HTTP) and in-process agent tool calls."""

    def __init__(self, name: str = "quick_pp-petrophysics-mcp", version: str = "1.0.0"):
        self.name = name
        self.version = version
        self._tools: Dict[str, MCPToolDefinition] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        schema_model: type[BaseModel],
        fn: Callable[..., Dict[str, Any]],
    ) -> None:
        schema = schema_model.model_json_schema()
        # Clean up title/defs for OpenAI / Anthropic / Gemini function calling compatibility
        clean_schema = {
            "type": "object",
            "properties": schema.get("properties", {}),
            "required": schema.get("required", []),
        }
        if "$defs" in schema:
            clean_schema["$defs"] = schema["$defs"]
        self._tools[name] = MCPToolDefinition(
            name=name,
            description=description,
            input_schema=clean_schema,
            fn=fn,
        )

    def list_tools(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "inputSchema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def get_openai_tools(self) -> List[Dict[str, Any]]:
        """Format registered tools for OpenAI / Groq / Ollama / LiteLLM tool calling."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.input_schema,
                },
            }
            for t in self._tools.values()
        ]

    def get_anthropic_tools(self) -> List[Dict[str, Any]]:
        """Format registered tools for Anthropic Messages API."""
        return [
            {
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            }
            for t in self._tools.values()
        ]

    def get_gemini_tools(self) -> List[Dict[str, Any]]:
        """Format registered tools for Google Gemini generateContent API."""
        declarations = []
        for t in self._tools.values():
            props = {}
            for k, v in t.input_schema.get("properties", {}).items():
                prop_type = v.get("type", "string")
                if isinstance(prop_type, list):
                    prop_type = next((pt for pt in prop_type if pt != "null"), "string")
                if "anyOf" in v:
                    non_null = [a for a in v["anyOf"] if a.get("type") != "null"]
                    prop_type = non_null[0].get("type", "string") if non_null else "string"
                gem_prop: Dict[str, Any] = {
                    "type": prop_type.upper(),
                    "description": v.get("description", ""),
                }
                if prop_type == "array":
                    items_spec = v.get("items", {"type": "string"})
                    item_type = items_spec.get("type", "string")
                    gem_prop["items"] = {"type": item_type.upper()}
                props[k] = gem_prop

            declarations.append(
                {
                    "name": t.name,
                    "description": t.description,
                    "parameters": {
                        "type": "OBJECT",
                        "properties": props,
                        "required": t.input_schema.get("required", []),
                    },
                }
            )
        return [{"function_declarations": declarations}]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        if name not in self._tools:
            raise ValueError(f"Unknown MCP tool: '{name}'. Available: {list(self._tools.keys())}")
        tool = self._tools[name]
        args = arguments or {}
        return tool.fn(**args)

    def handle_jsonrpc(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process a standard MCP JSON-RPC 2.0 request."""
        req_id = payload.get("id")
        method = payload.get("method", "")
        params = payload.get("params") or {}

        # Notifications have no id and expect no response
        if req_id is None and method.startswith("notifications/"):
            return None

        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": {"name": self.name, "version": self.version},
                }
            elif method == "tools/list":
                result = {"tools": self.list_tools()}
            elif method == "tools/call":
                t_name = params.get("name")
                t_args = params.get("arguments") or {}
                out = self.call_tool(t_name, t_args)
                # Strip heavy raw columnar arrays from external MCP text output if present
                compact_out = {k: v for k, v in out.items() if k != "computed_curves"}
                result = {
                    "content": [
                        {
                            "type": "text",
                            "text": json.dumps(compact_out, default=str, indent=2),
                        }
                    ],
                    "structuredContent": compact_out,
                    "isError": False,
                }
            elif method == "ping":
                result = {}
            else:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Method not found: {method}"},
                }

            return {"jsonrpc": "2.0", "id": req_id, "result": result}
        except Exception as exc:
            logger.exception(f"MCP JSON-RPC error in {method}: {exc}")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    def run_stdio(self) -> None:
        """Run standard MCP server over stdin/stdout for Claude Desktop / Cursor / VS Code."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                resp = self.handle_jsonrpc(req)
                if resp is not None:
                    sys.stdout.write(json.dumps(resp, default=str) + "\n")
                    sys.stdout.flush()
            except Exception as e:
                err_resp = {
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                }
                sys.stdout.write(json.dumps(err_resp) + "\n")
                sys.stdout.flush()


# ---------------------------------------------------------------------------
# Tool Input Schemas
# ---------------------------------------------------------------------------


class InspectWellArgs(BaseModel):
    project_id: int = Field(..., description="Project ID in quick_pp database")
    well_name: Optional[str] = Field(
        None,
        description="Well name to inspect. If omitted, inspects the project summary and first available well.",
    )
    min_depth: Optional[float] = Field(None, description="Optional minimum depth filter (m)")
    max_depth: Optional[float] = Field(None, description="Optional maximum depth filter (m)")
    zones: Optional[List[str]] = Field(None, description="Optional list of zone names to filter")


class RunQaqcArgs(BaseModel):
    project_id: int = Field(..., description="Project ID")
    well_name: str = Field(..., description="Well name to run QAQC on")
    run_badhole: bool = Field(
        True, description="Detect washout / badhole intervals from CALI and bit size"
    )
    run_hc_correction: bool = Field(
        True, description="Apply NPHI/RHOB hydrocarbon correction (NPHI_HC, RHOB_HC, HC_FLAG)"
    )
    run_env_correction: bool = Field(
        False, description="Apply borehole environmental corrections (GR_CORR, NPHI_CORR, RHOB_CORR)"
    )
    rho_hc: float = Field(0.8, description="Hydrocarbon density (g/cc) for HC correction")
    hi_hc: float = Field(0.9, description="Hydrocarbon hydrogen index for HC correction")
    min_depth: Optional[float] = Field(None, description="Optional minimum depth filter")
    max_depth: Optional[float] = Field(None, description="Optional maximum depth filter")
    zones: Optional[List[str]] = Field(None, description="Optional zone filter")
    save: bool = Field(True, description="Persist QAQC corrected curves to well database")


class RunMultiMineralArgs(BaseModel):
    project_id: int = Field(..., description="Project ID")
    well_name: str = Field(..., description="Well name to evaluate")
    model: str = Field(
        "multi_mineral",
        description="Lithology solver: 'multi_mineral' (constrained optimization) or 'ssc' (Sand-Silt-Clay)",
    )
    minerals: List[str] = Field(
        default_factory=lambda: ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"],
        description="Minerals to solve for when model='multi_mineral'",
    )
    porosity_method: str = Field(
        "neutron_density", description="Porosity method: 'density', 'neutron_density', or 'sonic'"
    )
    use_hc_corrected: bool = Field(
        True, description="Use NPHI_HC and RHOB_HC if available"
    )
    min_depth: Optional[float] = Field(None, description="Optional minimum depth filter")
    max_depth: Optional[float] = Field(None, description="Optional maximum depth filter")
    zones: Optional[List[str]] = Field(None, description="Optional zone filter")
    save: bool = Field(True, description="Persist computed mineral & porosity curves to database")


class RunShfArgs(BaseModel):
    project_id: int = Field(..., description="Project ID")
    well_name: Optional[str] = Field(
        None, description="Optional focal well name for detailed zonal SHF vs Sw comparison"
    )
    fwl: float = Field(1020.0, description="Free Water Level (FWL) depth in meters/feet")
    optimize_fwl_first: bool = Field(
        False, description="Automatically optimize multi-well FWL from log Sw before computing SHF"
    )
    cutoffs: List[float] = Field(
        default_factory=lambda: [0.5, 1.5, 3.0],
        description="FZI rock typing cutoffs",
    )
    ift: float = Field(30.0, description="Interfacial tension (dynes/cm)")
    theta: float = Field(30.0, description="Contact angle (degrees)")
    gw: float = Field(1.05, description="Water specific gravity")
    ghc: float = Field(0.80, description="Hydrocarbon specific gravity")
    save: bool = Field(True, description="Persist ROCK_FLAG and SHF curves to project wells")


class ConformalEvaluationArgs(BaseModel):
    project_id: int = Field(..., description="Project ID")
    well_name: str = Field(..., description="Well name to evaluate with conformal prediction bounds")
    alpha: float = Field(
        CONFORMAL_ALPHA,
        ge=0.02,
        le=0.50,
        description="Significance level alpha (0.20 = 80% P10-P90 interval, 0.10 = 90% P05-P95 interval)",
    )
    lithology_model: str = Field(
        "clastic", description="Target suite: 'clastic' or 'carbonate'"
    )
    water_salinity: float = Field(
        35000.0, description="Formation water salinity (ppm) for baseline physics calibration"
    )
    min_depth: Optional[float] = Field(None, description="Optional minimum depth filter")
    max_depth: Optional[float] = Field(None, description="Optional maximum depth filter")
    zones: Optional[List[str]] = Field(None, description="Optional zone filter")
    save: bool = Field(
        True,
        description="Persist point predictions and P10/P90 conformal bounds (PHIT_P10/P90, SW_P10/P90, PERM_P10/P90) to database",
    )


class WorkbenchUiControlArgs(BaseModel):
    scope: Optional[str] = Field(
        None, description="Switch workbench scope: 'single-well' or 'multi-well'"
    )
    step: Optional[str] = Field(
        None,
        description="Switch active workflow step: 'qaqc', 'ancillary', 'litho-poro', 'perm', 'saturation', 'scal', 'ressum', 'inventory', 'rock-typing', 'perm-transform', or 'shf'",
    )
    scroll_to_depth: Optional[float] = Field(
        None, description="Center and highlight a specific depth on the LogTrackCanvas"
    )
    min_depth: Optional[float] = Field(
        None, description="Set global minimum depth filter on the workbench"
    )
    max_depth: Optional[float] = Field(
        None, description="Set global maximum depth filter on the workbench"
    )
    selected_zones: Optional[List[str]] = Field(
        None, description="Filter workbench to specific geological zones"
    )
    clear_filters: bool = Field(
        False, description="Clear all active depth and zone filters"
    )
    right_pane_tab: Optional[str] = Field(
        None, description="Switch right pane view: 'crossplot', 'grid', or 'split'"
    )


# ---------------------------------------------------------------------------
# Helper: Contiguous Depth Interval Extraction
# ---------------------------------------------------------------------------


def _extract_flagged_intervals(
    depths: np.ndarray, flag_mask: np.ndarray, max_intervals: int = 8
) -> List[Dict[str, float]]:
    """Group boolean flag mask into contiguous (top, bottom, thickness, sample_count) depth intervals."""
    idx = np.where(flag_mask)[0]
    if len(idx) == 0 or len(depths) == 0:
        return []
    step = float(np.nanmedian(np.abs(np.diff(depths)))) if len(depths) > 1 else 0.1524
    intervals: List[Dict[str, float]] = []
    start_i = idx[0]
    prev_i = idx[0]
    for cur_i in idx[1:]:
        if cur_i <= prev_i + 2:
            prev_i = cur_i
        else:
            top_d = float(depths[start_i])
            bot_d = float(depths[prev_i])
            intervals.append(
                {
                    "top": round(top_d, 2),
                    "bottom": round(bot_d, 2),
                    "thickness": round(max(abs(bot_d - top_d), step), 2),
                    "samples": int(prev_i - start_i + 1),
                }
            )
            start_i = cur_i
            prev_i = cur_i

    top_d = float(depths[start_i])
    bot_d = float(depths[prev_i])
    intervals.append(
        {
            "top": round(top_d, 2),
            "bottom": round(bot_d, 2),
            "thickness": round(max(abs(bot_d - top_d), step), 2),
            "samples": int(prev_i - start_i + 1),
        }
    )
    intervals.sort(key=lambda x: x["thickness"], reverse=True)
    return intervals[:max_intervals]


# ---------------------------------------------------------------------------
# Tool Implementations
# ---------------------------------------------------------------------------


def tool_inspect_well(
    project_id: int,
    well_name: Optional[str] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Inspect a project and well: available curves, depth intervals, formation tops, core count, and zonal stats."""
    proj_summary = get_project_summary(project_id)
    wells = proj_summary.get("wells", [])
    if not wells:
        return {
            "project_id": project_id,
            "error": f"No wells found in project {project_id}",
        }

    target_well = well_name or wells[0]["name"]
    df = _get_filtered_well_df(
        project_id, target_well, min_depth=min_depth, max_depth=max_depth, zones=zones
    )

    depths = pd.to_numeric(df["DEPTH"], errors="coerce").dropna()
    key_curves = [
        c
        for c in (
            "GR",
            "RT",
            "NPHI",
            "RHOB",
            "DTC",
            "PEF",
            "CALI",
            "VCLAY",
            "VSAND",
            "VSILT",
            "PHIT",
            "PHIE",
            "PERM",
            "SWT",
            "SHF",
            "CPORE",
            "CPERM",
            "PHIT_P10",
            "PHIT_P90",
            "SW_P10",
            "SW_P90",
        )
        if c in df.columns
    ]
    curve_stats = {c: _curve_summary_stats(df[c]) for c in key_curves}

    # Zonal breakdown if ZONES column has values
    zonal_breakdown: List[Dict[str, Any]] = []
    if "ZONES" in df.columns and df["ZONES"].notna().any():
        for z_name, z_grp in df.groupby("ZONES", dropna=True):
            z_d = pd.to_numeric(z_grp["DEPTH"], errors="coerce").dropna()
            if z_d.empty:
                continue
            entry: Dict[str, Any] = {
                "zone": str(z_name),
                "top": round(float(z_d.min()), 2),
                "bottom": round(float(z_d.max()), 2),
                "gross_thickness": round(float(z_d.max() - z_d.min()), 2),
                "samples": int(len(z_grp)),
            }
            for col in ("GR", "NPHI", "RHOB", "RT", "VCLAY", "PHIT", "PHIE", "SWT", "PERM"):
                if col in z_grp.columns:
                    s = pd.to_numeric(z_grp[col], errors="coerce").dropna()
                    if not s.empty:
                        entry[f"{col}_mean"] = round(float(s.mean()), 4)
                        entry[f"{col}_p50"] = round(float(np.percentile(s, 50)), 4)
            zonal_breakdown.append(entry)
        zonal_breakdown.sort(key=lambda x: x["top"])

    well_meta = next((w for w in wells if w["name"] == target_well), {})
    return {
        "tool": "inspect_well",
        "project_id": project_id,
        "project_name": proj_summary.get("project_name") or proj_summary.get("name"),
        "project_well_names": [w["name"] for w in wells],
        "well_name": target_well,
        "row_count": int(len(df)),
        "depth_range": {
            "min": round(float(depths.min()), 2) if not depths.empty else None,
            "max": round(float(depths.max()), 2) if not depths.empty else None,
        },
        "available_curves": [
            c for c in df.columns if c not in ("WELL_NAME", "ZONES", "DEPTH")
        ],
        "core_sample_count": well_meta.get("core_sample_count", 0),
        "formation_tops": well_meta.get("tops", []),
        "fluid_contacts": well_meta.get("contacts", []),
        "curve_stats": curve_stats,
        "zonal_breakdown": zonal_breakdown,
    }


def tool_run_well_qaqc(
    project_id: int,
    well_name: str,
    run_badhole: bool = True,
    run_hc_correction: bool = True,
    run_env_correction: bool = False,
    rho_hc: float = 0.8,
    hi_hc: float = 0.9,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Run petrophysical QAQC (`qaqc.py`): outer limit check, badhole/washout flagging, environmental & HC corrections."""
    df = _get_filtered_well_df(
        project_id, well_name, min_depth=min_depth, max_depth=max_depth, zones=zones
    ).copy()
    if "WELL_NAME" not in df.columns:
        df["WELL_NAME"] = well_name

    depths = pd.to_numeric(df["DEPTH"], errors="coerce").to_numpy(dtype=float)
    out_df = pd.DataFrame({"DEPTH": df["DEPTH"]})
    save_cols: List[str] = []

    # 1. Outer physical limit audit (Config.VARS)
    limit_clipped = handle_outer_limit(df, fill=False)
    outliers_by_curve: Dict[str, int] = {}
    for var in Config.VARS.keys():
        if var in df.columns:
            orig_valid = int(pd.to_numeric(df[var], errors="coerce").notna().sum())
            after_valid = int(pd.to_numeric(limit_clipped[var], errors="coerce").notna().sum())
            if orig_valid > after_valid:
                outliers_by_curve[var] = orig_valid - after_valid

    # 2. Badhole / Washout Flagging & tight/coal flags
    badhole_summary: Dict[str, Any] = {"available": False, "flagged_count": 0, "intervals": []}
    if run_badhole:
        if "CALI" in df.columns and df["CALI"].notna().any():
            bh_df = badhole_flagging(df)
            bh_series = pd.to_numeric(bh_df.get("BADHOLE", 0), errors="coerce").fillna(0)
            out_df["BADHOLE"] = bh_series
            save_cols.append("BADHOLE")
            bh_mask = bh_series.to_numpy(dtype=float) >= 1.0
            badhole_summary = {
                "available": True,
                "cali_present": True,
                "flagged_count": int(bh_mask.sum()),
                "flagged_pct": round(100.0 * float(bh_mask.mean()), 2),
                "intervals": _extract_flagged_intervals(depths, bh_mask),
            }
        else:
            # Infer density-spike / washout anomalies where RHOB < 2.05 or NPHI > 0.42
            rhob_s = pd.to_numeric(df.get("RHOB"), errors="coerce")
            nphi_s = pd.to_numeric(df.get("NPHI"), errors="coerce")
            bh_mask = ((rhob_s < 2.0) | (nphi_s > 0.42)).fillna(False).to_numpy(dtype=bool)
            out_df["BADHOLE"] = bh_mask.astype(float)
            save_cols.append("BADHOLE")
            badhole_summary = {
                "available": True,
                "cali_present": False,
                "note": "CALI curve absent; inferred washout/density anomalies from RHOB < 2.0 g/cc or NPHI > 0.42 v/v",
                "flagged_count": int(bh_mask.sum()),
                "flagged_pct": round(100.0 * float(bh_mask.mean()), 2),
                "intervals": _extract_flagged_intervals(depths, bh_mask),
            }

    # Tight streak & Coal flagging from feature_engineering
    tight_count = 0
    coal_count = 0
    if "RHOB" in df.columns:
        rhob_s = pd.to_numeric(df["RHOB"], errors="coerce")
        t_flag = tight_streak_flagging(rhob_s, df.get("RT"))
        out_df["TIGHT_FLAG"] = t_flag
        save_cols.append("TIGHT_FLAG")
        tight_count = int((t_flag >= 1.0).sum())
        if "NPHI" in df.columns:
            nphi_s = pd.to_numeric(df["NPHI"], errors="coerce")
            c_flag = coal_flagging(nphi_s, rhob_s)
            out_df["COAL_FLAG"] = c_flag
            save_cols.append("COAL_FLAG")
            coal_count = int((c_flag >= 1.0).sum())

    # 3. Hydrocarbon Correction on NPHI/RHOB crossplot
    hc_summary: Dict[str, Any] = {"applied": False}
    if run_hc_correction and "NPHI" in df.columns and "RHOB" in df.columns:
        hc_req = WorkspaceHCCorrectionRequest(
            rho_water=1.0,
            rho_hc=rho_hc,
            hi_hc=hi_hc,
            min_depth=min_depth,
            max_depth=max_depth,
            zones=zones,
            save=False,
        )
        hc_res = workspace_hc_correction(project_id, well_name, hc_req)
        hc_col_data = hc_res.get("data", {})
        for c in ("NPHI_HC", "RHOB_HC", "HC_FLAG"):
            if c in hc_col_data:
                out_df[c] = hc_col_data[c]
                if c in ("NPHI_HC", "RHOB_HC", "HC_FLAG"):
                    save_cols.append(c)
        hc_flag_arr = pd.to_numeric(out_df.get("HC_FLAG"), errors="coerce").fillna(0).to_numpy() >= 1.0
        hc_summary = {
            "applied": True,
            "corr_angle_deg": hc_res.get("corr_angle"),
            "hc_affected_samples": int(hc_flag_arr.sum()),
            "hc_affected_pct": round(100.0 * float(hc_flag_arr.mean()), 2),
            "hc_intervals": _extract_flagged_intervals(depths, hc_flag_arr),
            "stats": hc_res.get("stats", {}),
        }

    # 4. Environmental Corrections
    env_summary: Dict[str, Any] = {"applied": False}
    if run_env_correction:
        corr_df = apply_environmental_corrections(df)
        env_cols = [
            c
            for c in ("GR_CORR", "NPHI_CORR", "RHOB_CORR", "RT_CORR")
            if c in corr_df.columns
        ]
        for c in env_cols:
            out_df[c] = corr_df[c]
            save_cols.append(c)
        env_summary = {"applied": True, "corrected_curves": env_cols}

    saved_rows = 0
    if save and save_cols:
        unique_save_cols = list(dict.fromkeys(save_cols))
        saved_rows = _save_curves_to_well(
            project_id, well_name, out_df[["DEPTH"] + unique_save_cols]
        )

    stats = {col: _curve_summary_stats(out_df[col]) for col in dict.fromkeys(save_cols)}
    col_payload = dataframe_to_columnar(out_df)

    return {
        "tool": "run_well_qaqc",
        "project_id": project_id,
        "well_name": well_name,
        "saved": save,
        "saved_rows": saved_rows,
        "outliers_by_curve": outliers_by_curve,
        "badhole": badhole_summary,
        "tight_streak_samples": tight_count,
        "coal_samples": coal_count,
        "hc_correction": hc_summary,
        "environmental_correction": env_summary,
        "stats": stats,
        "computed_curves": col_payload.get("data", {}),
        "ui_action": {
            "scope": "single-well",
            "step": "qaqc",
        },
    }


def tool_run_multi_mineral(
    project_id: int,
    well_name: str,
    model: str = "multi_mineral",
    minerals: Optional[List[str]] = None,
    porosity_method: str = "neutron_density",
    use_hc_corrected: bool = True,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Execute `multi_mineral.py` (or `sand_silt_clay.py`) optimization and porosity construction on a well."""
    mineral_list = minerals or ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"]
    lith_req = WorkspaceLithologyRequest(
        model=model,
        minerals=mineral_list,
        porosity_method=porosity_method,
        use_hc_corrected=use_hc_corrected,
        min_depth=min_depth,
        max_depth=max_depth,
        zones=zones,
        save=save,
    )
    lith_res = workspace_lithology(project_id, well_name, lith_req)
    col_data = dict(lith_res.get("data", {}))
    stats = dict(lith_res.get("stats", {}))

    # Map multi_mineral columns (VQTZ -> VSAND, VSHA -> VCLAY, PHIT_CONSTRUCTED -> PHIT/PHIE)
    # so the standard Track 4 in LogTrackCanvas renders both SSC and MultiMineral seamlessly!
    df_out = pd.DataFrame(col_data)
    extra_save_cols: List[str] = []
    if model == "multi_mineral" and not df_out.empty:
        if "VQTZ" in df_out.columns and "VSAND" not in df_out.columns:
            df_out["VSAND"] = df_out["VQTZ"]
            extra_save_cols.append("VSAND")
        if "VSHA" in df_out.columns and "VCLAY" not in df_out.columns:
            df_out["VCLAY"] = df_out["VSHA"]
            extra_save_cols.append("VCLAY")
        if ("VCALC" in df_out.columns or "VCAL" in df_out.columns) and "VSILT" not in df_out.columns:
            vcal_s = pd.to_numeric(
                df_out.get("VCALC", df_out.get("VCAL", 0.0)), errors="coerce"
            ).fillna(0.0)
            vdol_s = pd.to_numeric(
                df_out.get("VDOLO", df_out.get("VDOL", 0.0)), errors="coerce"
            ).fillna(0.0)
            df_out["VSILT"] = (vcal_s + vdol_s).clip(0.0, 1.0)
            extra_save_cols.append("VSILT")
        if "PHIT_CONSTRUCTED" in df_out.columns:
            df_out["PHIT"] = pd.to_numeric(df_out["PHIT_CONSTRUCTED"], errors="coerce").clip(0.0, 0.5)
            vsh = pd.to_numeric(df_out.get("VCLAY", 0.0), errors="coerce").fillna(0.0)
            df_out["PHIE"] = (df_out["PHIT"] * (1.0 - 0.35 * vsh)).clip(lower=0.0)
            extra_save_cols.extend(["PHIT", "PHIE"])

        if save and extra_save_cols and "DEPTH" in df_out.columns:
            _save_curves_to_well(project_id, well_name, df_out[["DEPTH"] + extra_save_cols])
        for c in extra_save_cols:
            stats[c] = _curve_summary_stats(df_out[c])
        col_data = dataframe_to_columnar(df_out).get("data", {})

    # Compute solver diagnostics if AVG_ERROR / OPTIMIZER_SUCCESS exist
    diagnostics: Dict[str, Any] = {}
    if "AVG_ERROR" in df_out.columns:
        err_s = pd.to_numeric(df_out["AVG_ERROR"], errors="coerce").dropna()
        if not err_s.empty:
            diagnostics["mean_reconstruction_error"] = round(float(err_s.mean()), 5)
            diagnostics["p90_reconstruction_error"] = round(float(np.percentile(err_s, 90)), 5)
    if "OPTIMIZER_SUCCESS" in df_out.columns:
        succ_s = pd.to_numeric(df_out["OPTIMIZER_SUCCESS"], errors="coerce").dropna()
        if not succ_s.empty:
            diagnostics["convergence_rate_pct"] = round(100.0 * float(succ_s.mean()), 2)

    return {
        "tool": "run_multi_mineral",
        "project_id": project_id,
        "well_name": well_name,
        "model": model,
        "minerals": mineral_list if model == "multi_mineral" else ["VSAND", "VSILT", "VCLAY"],
        "porosity_method": porosity_method,
        "saved": save,
        "saved_rows": lith_res.get("saved_rows", 0),
        "solver_diagnostics": diagnostics,
        "stats": stats,
        "computed_curves": col_data,
        "ui_action": {
            "scope": "single-well",
            "step": "litho-poro",
        },
    }


def tool_run_shf_evaluation(
    project_id: int,
    well_name: Optional[str] = None,
    fwl: float = 1020.0,
    optimize_fwl_first: bool = False,
    cutoffs: Optional[List[float]] = None,
    ift: float = 30.0,
    theta: float = 30.0,
    gw: float = 1.05,
    ghc: float = 0.80,
    save: bool = True,
) -> Dict[str, Any]:
    """Execute Saturation Height Function (`shf.py`) evaluation and optional FWL optimization across project wells."""
    fzi_cutoffs = cutoffs or [0.5, 1.5, 3.0]
    proj_summary = get_project_summary(project_id)
    wells = [w["name"] for w in proj_summary.get("wells", [])]
    focal_well = well_name or (wells[0] if wells else None)

    # Ensure PHIT and PERM exist in focal well so FZI/SHF can run even on raw wells
    if focal_well:
        wdf = _get_merged_dataframe_sync(project_id, focal_well)
        if not wdf.empty:
            from quick_pp.app.backend.services.workspace import (
                workspace_permeability,
                workspace_porosity,
            )

            if "PHIT" not in wdf.columns:
                try:
                    workspace_porosity(
                        project_id,
                        focal_well,
                        WorkspacePorosityRequest(method="neu_den", save=True),
                    )
                    wdf = _get_merged_dataframe_sync(project_id, focal_well)
                except Exception as e:
                    logger.debug(f"Auto-initializing PHIT for SHF: {e}")

            if "PERM" not in wdf.columns:
                perm_method = (
                    "choo"
                    if ("VCLAY" in wdf.columns and "VSILT" in wdf.columns)
                    else "timur"
                )
                try:
                    workspace_permeability(
                        project_id,
                        focal_well,
                        WorkspacePermeabilityRequest(method=perm_method, save=True),
                    )
                except Exception as e:
                    logger.debug(f"Auto-initializing PERM for SHF: {e}")

    fwl_opt_summary: Optional[Dict[str, Any]] = None
    effective_fwl = float(fwl)
    if optimize_fwl_first:
        try:
            opt_res = workspace_optimize_fwl(
                project_id,
                WorkspaceOptimizeFwlRequest(
                    ift=ift, theta=theta, gw=gw, ghc=ghc, save=False
                ),
            )
            effective_fwl = float(opt_res.get("fwl", fwl))
            fwl_opt_summary = {
                "optimized_fwl": effective_fwl,
                "owc": opt_res.get("owc"),
                "ttz": opt_res.get("ttz"),
                "transition_zone_thickness": opt_res.get("transition_zone_thickness"),
                "rmse": opt_res.get("rmse"),
            }
        except Exception as e:
            fwl_opt_summary = {"note": f"Used user FWL {fwl}m ({e})"}

    # Default Leverett-J parameters per FZI rock type if no SCAL PC data is loaded
    default_fits = {
        "1": {"a": 0.12, "b": -0.38},
        "2": {"a": 0.22, "b": -0.45},
        "3": {"a": 0.38, "b": -0.55},
        "4": {"a": 0.65, "b": -0.70},
    }
    shf_req = WorkspaceShfRequest(
        cutoffs=fzi_cutoffs,
        fwl=effective_fwl,
        ift=ift,
        theta=theta,
        gw=gw,
        ghc=ghc,
        fits=default_fits,
        save=save,
    )
    shf_res = workspace_shf_compute(project_id, shf_req)
    shf_data = shf_res.get("data", {})

    # Extract focal well SHF & ROCK_FLAG curves for live frontend update
    computed_curves: Dict[str, List[Any]] = {}
    focal_stats: Dict[str, Any] = {}
    reconciliation: Dict[str, Any] = {}

    if focal_well and shf_data.get("well_names"):
        wn_arr = shf_data["well_names"]
        d_arr = shf_data["depths"]
        shf_arr = shf_data["shf"]
        rf_arr = shf_data["rock_flags"]

        w_depths = [d for wn, d in zip(wn_arr, d_arr) if wn == focal_well]
        w_shf = [s for wn, s in zip(wn_arr, shf_arr) if wn == focal_well]
        w_rf = [r for wn, r in zip(wn_arr, rf_arr) if wn == focal_well]

        if w_depths:
            w_df = pd.DataFrame({"DEPTH": w_depths, "SHF": w_shf, "ROCK_FLAG": w_rf})
            computed_curves = dataframe_to_columnar(w_df).get("data", {})
            focal_stats = {
                "SHF": _curve_summary_stats(w_df["SHF"]),
                "ROCK_FLAG": _curve_summary_stats(w_df["ROCK_FLAG"]),
            }

            # Compare SHF against log SWT if present
            full_wdf = _get_merged_dataframe_sync(project_id, focal_well)
            if "SWT" in full_wdf.columns:
                merged_cmp = pd.merge(
                    full_wdf[["DEPTH", "SWT"]], w_df[["DEPTH", "SHF"]], on="DEPTH", how="inner"
                ).dropna()
                if not merged_cmp.empty:
                    diff = (
                        pd.to_numeric(merged_cmp["SWT"], errors="coerce")
                        - pd.to_numeric(merged_cmp["SHF"], errors="coerce")
                    ).dropna()
                    if not diff.empty:
                        reconciliation = {
                            "compared_samples": int(len(diff)),
                            "rmse_sw_vs_shf": round(float(np.sqrt(np.mean(diff**2))), 4),
                            "mae_sw_vs_shf": round(float(np.mean(np.abs(diff))), 4),
                            "mean_bias_sw_minus_shf": round(float(diff.mean()), 4),
                        }

    return {
        "tool": "run_shf_evaluation",
        "project_id": project_id,
        "well_name": focal_well,
        "fwl_used": effective_fwl,
        "fwl_optimization": fwl_opt_summary,
        "fzi_cutoffs": fzi_cutoffs,
        "leverett_j_fits": shf_res.get("fits", {}),
        "saved": save,
        "saved_wells": shf_res.get("saved_wells", 0),
        "sw_vs_shf_reconciliation": reconciliation,
        "stats": focal_stats,
        "computed_curves": computed_curves,
        "ui_action": {
            "scope": "single-well",
            "step": "saturation",
        },
    }


def tool_evaluate_with_conformal_bounds(
    project_id: int,
    well_name: str,
    alpha: float = CONFORMAL_ALPHA,
    lithology_model: str = "clastic",
    water_salinity: float = 35000.0,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
    save: bool = True,
) -> Dict[str, Any]:
    """Calibrate `ConformalMultiOutputRegressor` on the fly and compute P10-P90 conformal prediction bounds.

    Workflow:
    1. Extracts raw well logs (`GR`, `RT`, `NPHI`, `RHOB`, `DTC`) and generates physics-informed features
       (`RHOMA_APP`, `AI`, `M_LITH`, `N_LITH`, `DPHI`, `GAS_XOVER`, `TIGHT_FLAG`, `COAL_FLAG`).
    2. Constructs high-fidelity reference labels from core data (`CPORE`, `CPERM`) blended with
       physics-evaluated baseline curves (`VSAND`, `VSILT`, `VCLAY`, `PHIT`, `PHIE`, `SWT`, `LOG_PERM`).
    3. Fits `ConformalMultiOutputRegressor` (Locally Weighted Split Conformal Prediction normalized by
       ensemble tree standard deviation $\\sigma(x)$) at significance level `alpha` (default 0.20 = P10-P90).
    4. Enforces physical closure and hierarchy (`enforce_physical_constraints`) so:
       - $\\sum V_{\\text{mineral}} + \\text{PHIT} = 1.0$
       - $0 \\le \\text{PHIE} \\le \\text{PHIT}$
       - $\\text{P10} \\le \\text{Prediction} \\le \\text{P90}$ everywhere.
    5. Computes zonal conformal interval widths and explains where uncertainty is widest (e.g., washout/tight zones)
       vs narrowest (clean homogeneous pay).
    """
    df = _get_filtered_well_df(
        project_id, well_name, min_depth=min_depth, max_depth=max_depth, zones=zones
    ).copy()
    if "WELL_NAME" not in df.columns:
        df["WELL_NAME"] = well_name

    nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=True)
    gr = (
        pd.to_numeric(df["GR"], errors="coerce")
        if "GR" in df.columns
        else pd.Series(50.0, index=df.index, dtype=float)
    )
    rt = (
        pd.to_numeric(df["RT"], errors="coerce")
        if "RT" in df.columns
        else (
            pd.to_numeric(df["RD"], errors="coerce")
            if "RD" in df.columns
            else pd.Series(10.0, index=df.index, dtype=float)
        )
    )
    dtc = (
        pd.to_numeric(df["DTC"], errors="coerce")
        if "DTC" in df.columns
        else None
    )

    valid_mask = nphi.notna() & rhob.notna() & gr.notna() & rt.notna()
    if valid_mask.sum() < 3:
        raise ValueError(
            f"Need at least 3 valid (GR, RT, NPHI, RHOB) depth samples for conformal calibration; found {valid_mask.sum()}"
        )

    sub = df.loc[valid_mask].copy().reset_index(drop=True)
    nphi_v = nphi.loc[valid_mask].reset_index(drop=True)
    rhob_v = rhob.loc[valid_mask].reset_index(drop=True)
    gr_v = gr.loc[valid_mask].reset_index(drop=True)
    rt_v = rt.loc[valid_mask].reset_index(drop=True).clip(lower=0.05)
    dtc_v = dtc.loc[valid_mask].reset_index(drop=True) if dtc is not None else None
    depths_v = pd.to_numeric(sub["DEPTH"], errors="coerce").to_numpy(dtype=float)

    # 1. Physics Feature Engineering
    feat_df = pd.DataFrame(
        {
            "GR": gr_v,
            "RT": np.log10(rt_v),
            "NPHI": nphi_v,
            "RHOB": rhob_v,
            "DPHI": fe_density_porosity(rhob_v),
            "GAS_XOVER": gas_xover(rhob_v, nphi_v),
            "RHOMA_APP": apparent_matrix_density(rhob_v, nphi_v),
            "AI": acoustic_impedance_feature(rhob_v, dtc_v),
            "TIGHT_FLAG": tight_streak_flagging(rhob_v, rt_v),
            "COAL_FLAG": coal_flagging(nphi_v, rhob_v),
        }
    )
    m_lith, n_lith = mn_lithology_parameters(rhob_v, nphi_v, dtc_v)
    feat_df["M_LITH"] = m_lith
    feat_df["N_LITH"] = n_lith
    feat_df = feat_df.ffill().bfill().fillna(0.0)

    # 2. Construct Reference Calibration Targets (Physics Baseline + Core Anchors where available)
    ssc_model = SandSiltClay(
        dry_sand_point=(-0.02, 2.65),
        dry_silt_point=(None, 2.68),
        dry_clay_point=(0.33, 2.70),
        fluid_point=(1.0, 1.0),
        silt_line_angle=119.0,
    )
    vsand_ref, vsilt_ref, vclay_ref, _ = ssc_model.estimate_lithology(nphi_v, rhob_v)
    vsand_ref = np.clip(np.asarray(vsand_ref, dtype=float), 0.0, 1.0)
    vsilt_ref = np.clip(np.asarray(vsilt_ref, dtype=float), 0.0, 1.0)
    vclay_ref = np.clip(np.asarray(vclay_ref, dtype=float), 0.0, 1.0)

    rho_ma = rho_matrix(vsand=vsand_ref, vsilt=vsilt_ref, vclay=vclay_ref)
    phid_ref = np.asarray(density_porosity(rhob_v.to_numpy(), rho_ma, 1.0), dtype=float)
    phit_ref = np.clip(
        np.asarray(
            neu_den_xplot_poro(
                nphi_v,
                rhob_v,
                model="ssc",
                dry_min1_point=(-0.02, 2.65),
                dry_silt_point=ssc_model.dry_silt_point,
                dry_clay_point=ssc_model.dry_clay_point,
                fluid_point=(1.0, 1.0),
            ),
            dtype=float,
        ),
        0.005,
        0.48,
    )
    phit_sh = np.asarray(estimate_shale_porosity(nphi_v.to_numpy(), phid_ref), dtype=float)
    phie_ref = np.clip(phit_ref - vclay_ref * phit_sh, 0.0, phit_ref)

    # Anchor PHIT to Core Porosity (CPORE) where present
    core_poro_anchored = 0
    for c_col in ("CPORE", "CORE_PORO"):
        if c_col in sub.columns:
            cp = pd.to_numeric(sub[c_col], errors="coerce")
            if cp.notna().any():
                cp_frac = np.where(cp > 1.0, cp / 100.0, cp)
                c_mask = np.isfinite(cp_frac) & (cp_frac > 0) & (cp_frac < 0.5)
                phit_ref = np.where(c_mask, cp_frac, phit_ref)
                phie_ref = np.minimum(phie_ref, phit_ref)
                core_poro_anchored = int(c_mask.sum())
                break

    perm_ref = np.clip(
        np.asarray(choo_permeability(vclay_ref, vsilt_ref, phit_ref), dtype=float),
        1e-3,
        1e5,
    )
    # Anchor PERM to Core Permeability (CPERM) where present
    core_perm_anchored = 0
    for k_col in ("CPERM", "CORE_PERM"):
        if k_col in sub.columns:
            ck = pd.to_numeric(sub[k_col], errors="coerce")
            if ck.notna().any():
                k_mask = np.isfinite(ck) & (ck > 0)
                perm_ref = np.where(k_mask, np.clip(ck, 1e-3, 1e5), perm_ref)
                core_perm_anchored = int(k_mask.sum())
                break
    log_perm_ref = np.log10(perm_ref)

    temp = estimate_temperature_gradient(depths_v, "metric")
    rw = estimate_rw_temperature_salinity(temp, water_salinity)
    b_ws = estimate_b_waxman_smits(temp, rw)
    qvn = estimate_qvn(vclay_ref, phit_ref, 2.70)
    swt_ref = np.clip(
        np.asarray(
            waxman_smits_saturation(
                rt_v.to_numpy(), rw, phit_ref, b_ws, qvn, m=2.0
            ),
            dtype=float,
        ),
        0.0,
        1.0,
    )

    target_cols = ["PHIT", "PHIE", "SWT", "LOG_PERM", "VSAND", "VSILT", "VCLAY"]
    y_ref_df = pd.DataFrame(
        {
            "PHIT": phit_ref,
            "PHIE": phie_ref,
            "SWT": swt_ref,
            "LOG_PERM": log_perm_ref,
            "VSAND": vsand_ref,
            "VSILT": vsilt_ref,
            "VCLAY": vclay_ref,
        }
    )
    y_ref_df = enforce_physical_constraints(y_ref_df)[target_cols]

    # 3. Fit Locally Weighted ConformalMultiOutputRegressor on the fly
    from sklearn.ensemble import ExtraTreesRegressor

    base_rf = ExtraTreesRegressor(
        n_estimators=16, max_depth=10,
        min_samples_leaf=5,
        max_features=0.85,
        bootstrap=True,
        random_state=42,
        n_jobs=1,
    )
    conformal_model = ConformalMultiOutputRegressor(
        base_estimator=base_rf,
        alpha=alpha,
        cv=0,
        random_state=42,
        enforce_physics=True,
        target_names=target_cols,
    )
    max_cal = 1200
    if len(feat_df) > max_cal:
        cal_idx = np.linspace(0, len(feat_df) - 1, max_cal, dtype=int)
        conformal_model.fit(
            feat_df.iloc[cal_idx].reset_index(drop=True),
            y_ref_df.iloc[cal_idx].reset_index(drop=True),
        )
    else:
        conformal_model.fit(feat_df, y_ref_df)
    y_pred, y_p10, y_p90 = conformal_model.predict_interval(feat_df, alpha=alpha)

    # Assemble full depth-aligned output DataFrame
    pred_sub = pd.DataFrame(y_pred, columns=target_cols)
    for idx, t_name in enumerate(target_cols):
        pred_sub[f"{t_name}_P10"] = y_p10[:, idx]
        pred_sub[f"{t_name}_P90"] = y_p90[:, idx]
    pred_sub["TIGHT_FLAG"] = feat_df["TIGHT_FLAG"].to_numpy()
    pred_sub["COAL_FLAG"] = feat_df["COAL_FLAG"].to_numpy()
    pred_sub = enforce_physical_constraints(pred_sub)

    # Alias SW_P10 / SW_P90 so LogTrackCanvas Track 6 renders the purple conformal bands immediately!
    pred_sub["SW_P10"] = pred_sub["SWT_P10"]
    pred_sub["SW_P90"] = pred_sub["SWT_P90"]
    pred_sub["BVHC"] = ((1.0 - pred_sub["SWT"]) * pred_sub["PHIT"]).clip(0.0, 0.5)
    pred_sub["BVHC_P10"] = ((1.0 - pred_sub["SWT_P90"]) * pred_sub["PHIT_P10"]).clip(0.0, 0.5)
    pred_sub["BVHC_P90"] = ((1.0 - pred_sub["SWT_P10"]) * pred_sub["PHIT_P90"]).clip(0.0, 0.5)

    out_df = pd.DataFrame({"DEPTH": df["DEPTH"]})
    all_out_cols = [
        "PHIT",
        "PHIT_P10",
        "PHIT_P90",
        "PHIE",
        "PHIE_P10",
        "PHIE_P90",
        "SWT",
        "SW_P10",
        "SW_P90",
        "PERM",
        "PERM_P10",
        "PERM_P90",
        "VSAND",
        "VSILT",
        "VCLAY",
        "BVHC",
        "BVHC_P10",
        "BVHC_P90",
    ]
    for col in all_out_cols:
        if col in pred_sub.columns:
            s_full = pd.Series(np.nan, index=df.index, dtype=float)
            s_full.loc[valid_mask] = pred_sub[col].to_numpy(dtype=float)
            out_df[col] = s_full

    saved_rows = 0
    if save:
        saved_rows = _save_curves_to_well(project_id, well_name, out_df)

    # 4. Zonal Conformal Prediction Diagnostics & Pay Confidence Breakdown
    phit_width = (pred_sub["PHIT_P90"] - pred_sub["PHIT_P10"]).to_numpy(dtype=float)
    sw_width = (pred_sub["SW_P90"] - pred_sub["SW_P10"]).to_numpy(dtype=float)
    perm_ratio = (
        pred_sub["PERM_P90"].clip(lower=1e-3) / pred_sub["PERM_P10"].clip(lower=1e-3)
    ).to_numpy(dtype=float)

    zonal_conformal: List[Dict[str, Any]] = []
    zones_col = sub["ZONES"] if "ZONES" in sub.columns and sub["ZONES"].notna().any() else pd.Series(["Full Interval"] * len(sub))
    for z_name, z_idx in zones_col.groupby(zones_col).groups.items():
        idx_arr = np.asarray(list(z_idx), dtype=int)
        z_depths = depths_v[idx_arr]
        z_phit = pred_sub["PHIT"].iloc[idx_arr]
        z_phit_p10 = pred_sub["PHIT_P10"].iloc[idx_arr]
        z_phit_p90 = pred_sub["PHIT_P90"].iloc[idx_arr]
        z_sw = pred_sub["SWT"].iloc[idx_arr]
        z_sw_p10 = pred_sub["SW_P10"].iloc[idx_arr]
        z_sw_p90 = pred_sub["SW_P90"].iloc[idx_arr]
        z_perm = pred_sub["PERM"].iloc[idx_arr]

        # Certifiable Net Pay: even under conservative P10 porosity & P90 water saturation, still reservoir pay
        high_conf_pay = (z_phit_p10 >= 0.08) & (z_sw_p90 <= 0.65)
        possible_pay = (z_phit_p90 >= 0.08) & (z_sw_p10 <= 0.65)

        zonal_conformal.append(
            {
                "zone": str(z_name),
                "top": round(float(np.min(z_depths)), 2),
                "bottom": round(float(np.max(z_depths)), 2),
                "samples": int(len(idx_arr)),
                "phit_mean": round(float(z_phit.mean()), 4),
                "phit_p10_p90_band": [
                    round(float(z_phit_p10.mean()), 4),
                    round(float(z_phit_p90.mean()), 4),
                ],
                "phit_mean_width": round(float(phit_width[idx_arr].mean()), 4),
                "sw_mean": round(float(z_sw.mean()), 4),
                "sw_p10_p90_band": [
                    round(float(z_sw_p10.mean()), 4),
                    round(float(z_sw_p90.mean()), 4),
                ],
                "sw_mean_width": round(float(sw_width[idx_arr].mean()), 4),
                "perm_geom_mean_mD": round(
                    float(10.0 ** np.mean(np.log10(np.clip(z_perm, 1e-3, 1e5)))), 2
                ),
                "certifiable_pay_fraction": round(float(high_conf_pay.mean()), 3),
                "upside_pay_fraction": round(float(possible_pay.mean()), 3),
            }
        )
    zonal_conformal.sort(key=lambda x: x["top"])

    # Identify highest-uncertainty depth interval (where trees disagree most)
    high_unc_thresh = float(np.percentile(sw_width, 80)) if len(sw_width) > 1 else 0.15
    high_unc_intervals = _extract_flagged_intervals(
        depths_v, sw_width >= high_unc_thresh, max_intervals=5
    )

    q_hats = {
        t: round(float(q), 4)
        for t, q in zip(target_cols, getattr(conformal_model, "conformal_q_", []))
    }
    stats = {col: _curve_summary_stats(out_df[col]) for col in all_out_cols if col in out_df.columns}
    col_payload = dataframe_to_columnar(out_df)

    return {
        "tool": "evaluate_with_conformal_bounds",
        "project_id": project_id,
        "well_name": well_name,
        "confidence_level_pct": round(100.0 * (1.0 - alpha), 1),
        "alpha": alpha,
        "calibration_samples": int(len(sub)),
        "core_anchors": {
            "core_porosity_samples": core_poro_anchored,
            "core_permeability_samples": core_perm_anchored,
        },
        "conformal_quantiles_q_hat": q_hats,
        "interval_width_summary": {
            "phit_mean_width_vv": round(float(np.mean(phit_width)), 4),
            "sw_mean_width_vv": round(float(np.mean(sw_width)), 4),
            "perm_median_p90_p10_factor": round(float(np.median(perm_ratio)), 2),
        },
        "zonal_conformal_summary": zonal_conformal,
        "high_uncertainty_intervals": high_unc_intervals,
        "saved": save,
        "saved_rows": saved_rows,
        "stats": stats,
        "computed_curves": col_payload.get("data", {}),
        "ui_action": {
            "scope": "single-well",
            "step": "litho-poro",
        },
    }


def tool_control_workbench_ui(
    scope: Optional[str] = None,
    step: Optional[str] = None,
    scroll_to_depth: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    selected_zones: Optional[List[str]] = None,
    clear_filters: bool = False,
    right_pane_tab: Optional[str] = None,
) -> Dict[str, Any]:
    """Navigate or filter the Svelte petrophysical workbench UI."""
    ui_action: Dict[str, Any] = {}
    if scope in ("single-well", "multi-well"):
        ui_action["scope"] = scope
    if step:
        ui_action["step"] = step
    if scroll_to_depth is not None:
        ui_action["scroll_to_depth"] = float(scroll_to_depth)
    if min_depth is not None or max_depth is not None:
        ui_action["depth_filter"] = {"min": min_depth, "max": max_depth}
    if selected_zones is not None:
        ui_action["selected_zones"] = selected_zones
    if clear_filters:
        ui_action["clear_filters"] = True
    if right_pane_tab in ("crossplot", "grid", "split"):
        ui_action["right_pane_tab"] = right_pane_tab

    return {
        "tool": "control_workbench_ui",
        "status": "applied",
        "ui_action": ui_action,
    }


# ---------------------------------------------------------------------------
# Singleton MCP Server Instance
# ---------------------------------------------------------------------------


petro_mcp_server = PetrophysicalMCPServer()

petro_mcp_server.register_tool(
    name="inspect_well",
    description=(
        "Inspect a well and project in quick_pp: returns available log curves, depth range, "
        "formation tops, fluid contacts, core sample counts, and zonal P10/P50/P90 statistics."
    ),
    schema_model=InspectWellArgs,
    fn=tool_inspect_well,
)

petro_mcp_server.register_tool(
    name="run_well_qaqc",
    description=(
        "Run petrophysical QAQC (`qaqc.py`) on a well: outer physical limit checks, caliper/washout "
        "badhole flagging (`BADHOLE`), tight streak & coal detection, and NPHI/RHOB hydrocarbon correction "
        "(`NPHI_HC`, `RHOB_HC`, `HC_FLAG`). Live-updates the Svelte workbench tracks."
    ),
    schema_model=RunQaqcArgs,
    fn=tool_run_well_qaqc,
)

petro_mcp_server.register_tool(
    name="run_multi_mineral",
    description=(
        "Execute optimization-based Multi-Mineral (`multi_mineral.py`) or Sand-Silt-Clay (`ssc`) lithology "
        "and porosity evaluation on a well. Computes mineral volumes (`VQTZ`/`VSAND`, `VCAL`, `VDOL`, `VSHA`/`VCLAY`), "
        "`PHIT`, `PHIE`, and solver convergence diagnostics."
    ),
    schema_model=RunMultiMineralArgs,
    fn=tool_run_multi_mineral,
)

petro_mcp_server.register_tool(
    name="run_shf_evaluation",
    description=(
        "Execute Saturation Height Function (`shf.py`) modeling: computes FZI rock types (`ROCK_FLAG`), "
        "fits Leverett-J parameters, optionally optimizes Free Water Level (`optimize_fwl`), computes `SHF`, "
        "and reconciles `SHF` against log water saturation `SWT`."
    ),
    schema_model=RunShfArgs,
    fn=tool_run_shf_evaluation,
)

petro_mcp_server.register_tool(
    name="evaluate_with_conformal_bounds",
    description=(
        "Calibrate `ConformalMultiOutputRegressor` on the fly using physics-informed features and core/log targets "
        "to generate heteroscedastic P10-P90 conformal prediction bounds (`PHIT_P10`/`P90`, `SW_P10`/`P90`, "
        "`PERM_P10`/`P90`, `BVHC_P10`/`P90`) with volumetric closure and zonal pay confidence diagnostics."
    ),
    schema_model=ConformalEvaluationArgs,
    fn=tool_evaluate_with_conformal_bounds,
)

petro_mcp_server.register_tool(
    name="control_workbench_ui",
    description=(
        "Control the Svelte petrophysical workbench UI: switch workflow steps ('qaqc', 'litho-poro', 'saturation', "
        "'shf'), scroll/zoom the log canvas to a specific depth, or filter by depth interval and geological zones."
    ),
    schema_model=WorkbenchUiControlArgs,
    fn=tool_control_workbench_ui,
)
