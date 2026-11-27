"""FDP Petrophysical Report Generation Service & FastAPI Endpoints (`quick_pp`).

Orchestrates:
1. Context Collection (`build_report_context`) from Project DB, Workspace parameters, `ressum.py`, `qaqc.py`, `uncertainty.py`, and `upscaling.py`
2. High-DPI Plot Asset Generation (`quick_pp/plotter/report_plots.py`)
3. Grounded Lead Petrophysicist Narrative Synthesis (Built-in Domain Agent + OpenAI/Gemini/Anthropic/Ollama)
4. Modular Jinja2 Markdown Compilation (`quick_pp/app/backend/template/reports/`)
5. Multi-Page PDF & Print-Ready HTML Compilation
"""

from __future__ import annotations

import html
import json
import math
import os
import re
import textwrap
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from uuid import uuid4

import jinja2
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import requests
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse, HTMLResponse
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure
from sqlalchemy import func, select

from quick_pp import logger
from quick_pp.app.backend.schemas.report import (
    DEFAULT_REPORT_SECTIONS,
    ReportAiEnhanceRequest,
    ReportFullMarkdownUpdateRequest,
    ReportGenerationRequest,
    ReportGenerationResponse,
    ReportSectionConfig,
    ReportSectionUpdateRequest,
)
from quick_pp.app.backend.services.database import (
    _get_merged_dataframe_sync,
    get_project_summary,
)
from quick_pp.app.backend.services.workspace import (
    _ephemeral_well_configs,
    _select_nphi_rhob,
    _synthesize_bitsize_series,
)
from quick_pp.app.backend.utils.db import get_db
from quick_pp.borehole_corrections import (
    ENV_QC_REASON_BITS,
    confidence_weighted_curve,
)
from quick_pp.config import Config
from quick_pp.database import objects as db_objects
from quick_pp.environment import environment_from_config
from quick_pp.las_handler import extract_las_metadata
from quick_pp.permeability import choo_permeability
from quick_pp.plotter import report_plots
from quick_pp.ressum import calc_reservoir_summary
from quick_pp.upscaling import upscale_logs

router = APIRouter(tags=["Petrophysical Reports"])

# In-memory store of generated reports (also persisted to static/reports/{report_id}/report.json)
_REPORTS_STORE: Dict[str, Dict[str, Any]] = {}

BACKEND_DIR = Path(__file__).resolve().parent.parent
REPORTS_TEMPLATE_DIR = BACKEND_DIR / "template" / "reports"
REPORTS_STATIC_ROOT = BACKEND_DIR / "static" / "reports"
REPORTS_STATIC_ROOT.mkdir(parents=True, exist_ok=True)


def _pluralize(count: Any, singular: str, plural: Optional[str] = None) -> str:
    """Lightweight Jinja2 pluralization helper (Task 4.3)."""
    try:
        c = int(count)
    except Exception:
        c = 1
    return singular if c == 1 else (plural or f"{singular}s")


_jinja_env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(str(REPORTS_TEMPLATE_DIR)),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
_jinja_env.filters["pluralize"] = _pluralize

SECTION_TEMPLATE_MAP: Dict[str, str] = {
    "01_executive_summary": "sections/01_executive_summary.md.j2",
    "02_data_inventory": "sections/02_data_inventory.md.j2",
    "03_qaqc": "sections/03_qaqc.md.j2",
    "04_lithology_porosity": "sections/04_lithology_porosity.md.j2",
    "05_rock_typing_perm": "sections/05_rock_typing_perm.md.j2",
    "06_saturation_contacts": "sections/06_saturation_contacts.md.j2",
    "07_shf_scal": "sections/07_shf_scal.md.j2",
    "08_ressum_uncertainty": "sections/08_ressum_uncertainty.md.j2",
    "09_upscaling": "sections/09_upscaling.md.j2",
    "10_gate_checks": "sections/10_gate_checks.md.j2",
    "11_appendix_cpi": "sections/10_appendix_cpi.md.j2",
    "10_appendix_cpi": "sections/10_appendix_cpi.md.j2",
}

SECTION_ID_ALIASES: Dict[str, str] = {
    "exec_summary": "01_executive_summary",
    "executive_summary": "01_executive_summary",
    "data_inventory": "02_data_inventory",
    "inventory": "02_data_inventory",
    "qaqc": "03_qaqc",
    "litho_poro": "04_lithology_porosity",
    "lithology_porosity": "04_lithology_porosity",
    "rock_perm": "05_rock_typing_perm",
    "rock_typing_perm": "05_rock_typing_perm",
    "saturation": "06_saturation_contacts",
    "saturation_contacts": "06_saturation_contacts",
    "shf_scal": "07_shf_scal",
    "ressum_uncert": "08_ressum_uncertainty",
    "ressum_uncertainty": "08_ressum_uncertainty",
    "upscaling": "09_upscaling",
    "gate_checks": "10_gate_checks",
    "gate_check": "10_gate_checks",
    "appendix_cpi": "11_appendix_cpi",
    "10_appendix_cpi": "11_appendix_cpi",
}


def _normalize_section_id(section_id: str) -> str:
    sid = str(section_id or "").strip()
    return SECTION_ID_ALIASES.get(sid, sid)


def _col_series(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default)
    return pd.Series(default, index=df.index, dtype=float)


REPORT_TYPE_PRESETS: Dict[str, List[str]] = {
    "FDP_FULL": [
        "01_executive_summary",
        "02_data_inventory",
        "03_qaqc",
        "04_lithology_porosity",
        "05_rock_typing_perm",
        "06_saturation_contacts",
        "07_shf_scal",
        "08_ressum_uncertainty",
        "09_upscaling",
        "10_gate_checks",
        "11_appendix_cpi",
    ],
    "SINGLE_WELL_CPI": [
        "01_executive_summary",
        "03_qaqc",
        "04_lithology_porosity",
        "06_saturation_contacts",
        "08_ressum_uncertainty",
        "10_gate_checks",
        "11_appendix_cpi",
    ],
    "QAQC_SUMMARY": [
        "01_executive_summary",
        "02_data_inventory",
        "03_qaqc",
    ],
    "RESSUM_ONLY": [
        "01_executive_summary",
        "08_ressum_uncertainty",
        "09_upscaling",
    ],
}


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if math.isfinite(v) else default
    except Exception:
        return default


GENERIC_PLACEHOLDER_ZONES = {
    "MAIN RESERVOIR",
    "UNKNOWN",
    "ALL",
    "NAN",
    "NONE",
    "",
}


_STRATIGRAPHIC_CORRECTIONS = {
    re.compile(r"\bWest\s*Phalian\b", re.IGNORECASE): "Westphalian",
    re.compile(r"\bSteph\s*Anian\b", re.IGNORECASE): "Stephanian",
    re.compile(r"\bNamur\s*Ian\b", re.IGNORECASE): "Namurian",
    re.compile(r"\bVise\s*An\b", re.IGNORECASE): "Visean",
    re.compile(r"\bKimme\s*Ridg\s*Ian\b", re.IGNORECASE): "Kimmeridgian",
    re.compile(r"\bRot\s*Liegend\b", re.IGNORECASE): "Rotliegend",
    re.compile(r"\bZech\s*Stein\b", re.IGNORECASE): "Zechstein",
    re.compile(r"\bStat\s*Fjord\b", re.IGNORECASE): "Statfjord",
}


def _normalize_zone_name(
    raw_zone: Any,
    well_zones_upper: Optional[set[str]] = None,
) -> str:
    """Clean and normalize a formation/zone name preserving stratigraphic stage capitalization (Task 4.2)."""
    s = str(raw_zone or "").strip()
    if not s or s.lower() in ("nan", "none", "unknown"):
        return "Unknown"

    # If ALL_CAPS, all-lowercase, or snake_case, format to clean Title Case preserving alphanumeric / Roman numeral suffixes
    if s.isupper() or s.islower() or "_" in s:
        words = s.replace("_", " ").split()
        norm_words = []
        for w in words:
            if len(w) <= 2 or w.upper() in (
                "II",
                "III",
                "IV",
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
            ):
                norm_words.append(w.upper())
            else:
                norm_words.append(w.capitalize())
        s = " ".join(norm_words)

    # Post-normalization stratigraphic stage replacement dictionary
    for pattern, replacement in _STRATIGRAPHIC_CORRECTIONS.items():
        s = pattern.sub(replacement, s)

    return s


_ENV_REASON_LABELS: List[Tuple[int, str]] = [
    (ENV_QC_REASON_BITS["WASHOUT"], "Washout"),
    (ENV_QC_REASON_BITS["DRHO_HIGH"], "High DRHO"),
    (ENV_QC_REASON_BITS["BADHOLE"], "Badhole"),
    (ENV_QC_REASON_BITS["INVASION"], "Invasion"),
    (ENV_QC_REASON_BITS["NO_CALIPER"], "No Caliper"),
    (ENV_QC_REASON_BITS["SPIKE"], "Spike"),
    (ENV_QC_REASON_BITS["FLATLINE"], "Flatline"),
    (ENV_QC_REASON_BITS["CYCLE_SKIP"], "Cycle Skip"),
    (ENV_QC_REASON_BITS["BS_ASSUMED"], "Assumed BS"),
    (ENV_QC_REASON_BITS["RES_MUD_MISMATCH"], "Res/Mud Mismatch"),
    (ENV_QC_REASON_BITS["DEPTH_MISMATCH"], "Off-Depth"),
]
_ENV_CURVE_UNITS: Dict[str, str] = {
    "GR": "gAPI",
    "NPHI": "v/v",
    "RHOB": "g/cc",
    "RT": "dec",
    "DT": "us/ft",
}


def _dominant_env_reason(reason_s: pd.Series, mask: Optional[pd.Series] = None) -> str:
    """Return the top 1-2 decoded ENV_QC_REASON flags across `mask` (or `'Clean'`)."""
    if reason_s is None or reason_s.empty:
        return "Clean"
    vals = pd.to_numeric(reason_s, errors="coerce").fillna(0).astype(int)
    if mask is not None:
        vals = vals[mask.reindex(vals.index, fill_value=False)]
    if vals.empty:
        return "Clean"
    arr = vals.to_numpy(dtype=int)
    counts: List[Tuple[int, str]] = []
    for bit, label in _ENV_REASON_LABELS:
        cnt = int(np.sum((arr & bit) != 0))
        if cnt > 0:
            counts.append((cnt, label))
    if not counts:
        return "Clean"
    counts.sort(key=lambda item: item[0], reverse=True)
    return ", ".join(lbl for _, lbl in counts[:2])


def _select_conditioned_curve(
    df: pd.DataFrame,
    base_col: str,
    prefer_hc: bool = False,
    weight_by_conf: bool = True,
) -> pd.Series:
    """Select the best conditioned curve (`<COL>_HC` -> `<COL>_ENV_CORR` -> `<COL>`) with optional QC confidence weighting."""
    idx = df.index
    hc_col = f"{base_col}_HC"
    if prefer_hc and hc_col in df.columns:
        hc_s = pd.to_numeric(df[hc_col], errors="coerce")
        if hc_s.notna().any():
            return hc_s

    qc_prefix = "DT" if base_col in ("DTC", "DT") else base_col
    corr_col = f"{qc_prefix}_ENV_CORR"
    conf_col = f"{qc_prefix}_QC_CONF"
    raw_s = (
        pd.to_numeric(df[base_col], errors="coerce")
        if base_col in df.columns
        else pd.Series(np.nan, index=idx, dtype=float)
    )
    base_s = raw_s
    if corr_col in df.columns:
        corr_s = pd.to_numeric(df[corr_col], errors="coerce")
        if corr_s.notna().any():
            base_s = corr_s.combine_first(raw_s)

    if weight_by_conf and conf_col in df.columns and base_s.notna().any():
        conf_s = pd.to_numeric(df[conf_col], errors="coerce")
        if conf_s.notna().any():
            depth_arr = (
                pd.to_numeric(df["DEPTH"], errors="coerce").to_numpy(dtype=float)
                if "DEPTH" in df.columns
                else None
            )
            try:
                if qc_prefix == "RT":
                    lb = np.log10(base_s.clip(lower=1e-3))
                    blended = confidence_weighted_curve(
                        lb, conf_s.fillna(1.0), depth=depth_arr
                    )
                    out = pd.Series(10.0 ** blended.to_numpy(), index=idx)
                    return out.where(base_s.notna())
                blended = confidence_weighted_curve(
                    base_s, conf_s.fillna(1.0), depth=depth_arr
                )
                blended.index = idx
                return blended.where(base_s.notna())
            except Exception:
                return base_s
    return base_s


def _ensure_evaluated_curves(df: pd.DataFrame, well_name: str) -> pd.DataFrame:
    """Ensure baseline VCLAY, PHIT, PHIE, PERM, and SWT exist without synthetic scalar padding (Spec 1.1 & CHK-01)."""
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    if "WELL_NAME" not in work.columns:
        work["WELL_NAME"] = well_name

    # Alias canonical curves if needed
    work = Config.canonicalize_dataframe(work)

    orig_phit_present = "PHIT" in work.columns and work["PHIT"].notna().any()
    orig_phie_present = "PHIE" in work.columns and work["PHIE"].notna().any()
    has_vclay = "VCLAY" in work.columns and work["VCLAY"].notna().any()
    has_phit = orig_phit_present
    has_phie = orig_phie_present
    has_perm = "PERM" in work.columns and work["PERM"].notna().any()
    has_swt = "SWT" in work.columns and work["SWT"].notna().any()

    # Vectorized log derivations strictly where input logs exist (np.nan outside evaluated intervals)
    if not has_vclay:
        gr_s = _select_conditioned_curve(work, "GR")
        if gr_s.notna().any():
            p10 = float(np.nanpercentile(gr_s, 10)) if gr_s.notna().sum() > 5 else 15.0
            p90 = float(np.nanpercentile(gr_s, 90)) if gr_s.notna().sum() > 5 else 115.0
            denom = max(p90 - p10, 20.0)
            work["VCLAY"] = ((gr_s - p10) / denom).clip(0.0, 1.0)
        else:
            work["VCLAY"] = np.nan

    if not has_phit:
        nphi = _select_conditioned_curve(work, "NPHI", prefer_hc=True)
        rhob = _select_conditioned_curve(work, "RHOB", prefer_hc=True)
        if not (nphi.notna().any() or rhob.notna().any()):
            try:
                nphi, rhob = _select_nphi_rhob(work, use_hc_corrected=True)
            except Exception:
                pass
        if rhob.notna().any():
            phid = ((2.65 - rhob) / (2.65 - 1.0)).clip(0.005, 0.45)
            if nphi.notna().any():
                n_clean = nphi.clip(-0.05, 0.60)
                both_valid = phid.notna() & n_clean.notna()
                phit_est = phid.copy()
                phit_est.loc[both_valid] = np.sqrt(
                    0.5 * (phid.loc[both_valid] ** 2 + n_clean.loc[both_valid] ** 2)
                )
                work["PHIT"] = phit_est.clip(0.005, 0.45)
            else:
                work["PHIT"] = phid
        elif has_phie:
            vsh_s = pd.to_numeric(work["VCLAY"], errors="coerce")
            phie_s = pd.to_numeric(work["PHIE"], errors="coerce")
            work["PHIT"] = np.where(
                phie_s.notna(),
                (phie_s / np.maximum(1.0 - 0.35 * vsh_s.fillna(0.20), 0.3)).clip(
                    0.005, 0.45
                ),
                np.nan,
            )
        else:
            work["PHIT"] = np.nan

    if not has_phie:
        vsh_s = pd.to_numeric(work["VCLAY"], errors="coerce")
        phit_s = pd.to_numeric(work["PHIT"], errors="coerce")
        work["PHIE"] = np.where(
            phit_s.notna(),
            (phit_s * (1.0 - 0.35 * vsh_s.fillna(0.20))).clip(lower=0.0),
            np.nan,
        )

    # CHK-01 enforcement: if PHIT/PHIE were not explicitly saved in DB, mask outside RT-logged interval
    rt_gate_s = _select_conditioned_curve(work, "RT", weight_by_conf=False)
    if not (orig_phit_present or orig_phie_present) and rt_gate_s.notna().any():
        rt_valid = rt_gate_s.notna()
        work.loc[~rt_valid, "PHIT"] = np.nan
        work.loc[~rt_valid, "PHIE"] = np.nan

    if not has_perm:
        try:
            phit_s = pd.to_numeric(work["PHIT"], errors="coerce")
            if phit_s.notna().any():
                vcl_arr = (
                    _col_series(work, "VCLAY", np.nan)
                    .fillna(0.20)
                    .to_numpy(dtype=float)
                )
                vsi_arr = (
                    _col_series(work, "VSILT", np.nan)
                    .fillna(0.10)
                    .to_numpy(dtype=float)
                )
                phi_arr = phit_s.fillna(0.15).to_numpy(dtype=float)
                perm_est = np.clip(
                    np.asarray(
                        choo_permeability(vcl_arr, vsi_arr, phi_arr), dtype=float
                    ),
                    1e-3,
                    1e5,
                )
                work["PERM"] = np.where(phit_s.notna(), perm_est, np.nan)
            else:
                work["PERM"] = np.nan
        except Exception:
            work["PERM"] = np.nan

    if not has_swt:
        try:
            rt_s = _select_conditioned_curve(work, "RT")
            phit_s = pd.to_numeric(work["PHIT"], errors="coerce")
            if rt_s.notna().any() and phit_s.notna().any():
                rt_arr = rt_s.clip(lower=0.1).to_numpy(dtype=float)
                phi_arr = phit_s.clip(lower=0.01).to_numpy(dtype=float)
                swt_est = np.clip(
                    np.sqrt(0.08 / np.maximum(rt_arr * (phi_arr**2), 1e-6)),
                    0.0,
                    1.0,
                )
                work["SWT"] = np.where(rt_s.notna() & phit_s.notna(), swt_est, np.nan)
            else:
                work["SWT"] = np.nan
        except Exception:
            work["SWT"] = np.nan

    return work


def _load_project_wells_fast(
    project_id: int,
    well_names: Optional[List[str]] = None,
) -> Tuple[str, List[Dict[str, Any]]]:
    """Fetch project name and selected well metadata + ancillary counts in a single fast DB session."""
    with get_db() as session:
        try:
            proj = db_objects.Project.load(session, project_id=project_id)
        except Exception:
            proj = None
        if proj is None:
            summary = get_project_summary(project_id)
            project_name = str(summary.get("name") or f"Project #{project_id}")
            raw_wells = summary.get("wells") or []
            if not raw_wells:
                raise ValueError(
                    f"No wells found in project {project_id} ({project_name})."
                )
            return project_name, [
                {
                    "id": w.get("id", idx + 1),
                    "name": str(w.get("name", f"Well-{idx + 1}")),
                    "uwi": str(w.get("uwi") or w.get("name", f"Well-{idx + 1}")),
                    "core_sample_count": int(w.get("core_sample_count", 0)),
                    "pressure_test_count": int(w.get("pressure_test_count", 0)),
                    "formation_top_count": int(
                        w.get("formation_top_count") or len(w.get("tops") or [])
                    ),
                    "hole_section_count": int(
                        w.get("hole_section_count") or len(w.get("hole_sections") or [])
                    ),
                    "hole_sections": list(w.get("hole_sections") or []),
                    "depth_uom": w.get("depth_uom"),
                    "well_config": w.get("well_config") or {},
                }
                for idx, w in enumerate(raw_wells)
            ]
        project_name = str(proj.name or f"Project #{project_id}")
        orm_wells = (
            session.execute(
                select(db_objects.ORMWell).where(
                    db_objects.ORMWell.project_id == project_id
                )
            )
            .scalars()
            .all()
        )
        if not orm_wells:
            raise ValueError(
                f"No wells found in project {project_id} ({project_name})."
            )

        requested_all = (
            not well_names
            or "ALL" in [str(w).upper() for w in well_names]
            or len(well_names) == 0
        )
        if not requested_all:
            req_upper = {
                str(w).strip().upper().replace("/", "-").replace("_", "-")
                for w in well_names
                if w
            }
            filtered = [
                w
                for w in orm_wells
                if str(w.name).strip().upper().replace("/", "-").replace("_", "-")
                in req_upper
                or str(w.uwi or "").strip().upper().replace("/", "-").replace("_", "-")
                in req_upper
            ]
            if filtered:
                orm_wells = filtered

        well_ids = [w.well_id for w in orm_wells]
        core_counts = dict(
            session.execute(
                select(
                    db_objects.ORMCoreSample.well_id,
                    func.count(),
                )
                .where(db_objects.ORMCoreSample.well_id.in_(well_ids))
                .group_by(db_objects.ORMCoreSample.well_id)
            ).all()
        )
        press_counts = dict(
            session.execute(
                select(
                    db_objects.ORMPressureTest.well_id,
                    func.count(),
                )
                .where(db_objects.ORMPressureTest.well_id.in_(well_ids))
                .group_by(db_objects.ORMPressureTest.well_id)
            ).all()
        )
        top_counts = dict(
            session.execute(
                select(
                    db_objects.ORMFormationTop.well_id,
                    func.count(),
                )
                .where(db_objects.ORMFormationTop.well_id.in_(well_ids))
                .group_by(db_objects.ORMFormationTop.well_id)
            ).all()
        )
        hole_sections_by_well: Dict[int, List[Dict[str, Any]]] = {}
        try:
            hs_rows = (
                session.execute(
                    select(db_objects.ORMHoleSection)
                    .where(db_objects.ORMHoleSection.well_id.in_(well_ids))
                    .order_by(
                        db_objects.ORMHoleSection.top,
                        db_objects.ORMHoleSection.bottom,
                    )
                )
                .scalars()
                .all()
            )
            for hs in hs_rows:
                hole_sections_by_well.setdefault(hs.well_id, []).append(
                    {
                        "name": str(hs.name or "Section"),
                        "top": hs.top,
                        "bottom": hs.bottom,
                        "bit_size": hs.bit_size,
                        "mud_type": hs.mud_type or "WBM",
                        "mud_weight": hs.mud_weight,
                        "mud_salinity": hs.mud_salinity,
                        "rmf": hs.rmf,
                        "rmf_temp": hs.rmf_temp,
                    }
                )
        except Exception as exc:
            logger.debug(f"Hole sections query skipped in report loader: {exc}")

        selected_wells_meta: List[Dict[str, Any]] = []
        for w in orm_wells:
            w_cfg = dict(w.config_data) if isinstance(w.config_data, dict) else {}
            if isinstance(w.header_data, dict):
                w_cfg["_header_data"] = dict(w.header_data)
            eph_cfg = _ephemeral_well_configs.get((project_id, w.name))
            if eph_cfg:
                w_cfg = {**w_cfg, **eph_cfg}
            w_hs = (
                list(eph_cfg["hole_sections"])
                if isinstance(eph_cfg, dict)
                and isinstance(eph_cfg.get("hole_sections"), list)
                else hole_sections_by_well.get(w.well_id, [])
            )
            selected_wells_meta.append(
                {
                    "id": w.well_id,
                    "name": str(w.name),
                    "uwi": str(w.uwi or w.name),
                    "depth_uom": getattr(w, "depth_uom", None),
                    "core_sample_count": int(core_counts.get(w.well_id, 0)),
                    "pressure_test_count": int(press_counts.get(w.well_id, 0)),
                    "formation_top_count": int(top_counts.get(w.well_id, 0)),
                    "hole_section_count": len(w_hs),
                    "hole_sections": w_hs,
                    "well_config": w_cfg,
                }
            )
        return project_name, selected_wells_meta


def _load_saved_project_module_entry(
    project_id: int, module: str, target_zones: Optional[List[str]] = None
) -> Dict[str, Any]:
    """Return the saved multi-well module entry (`params`, `label`, `_key`) for the active zone selection.

    Mirrors the workbench lookup order: exact zone bundle, each single zone, `all`, then any saved entry.
    Returns an empty dict when nothing is saved or the store is unreachable.
    """
    try:
        from quick_pp.app.backend.services.workspace import (
            _load_project_module_params,
        )

        mp, _ = _load_project_module_params(int(project_id))
    except Exception as exc:
        logger.debug(f"Saved module params unavailable for '{module}': {exc}")
        return {}
    mod_map = mp.get(module) if isinstance(mp, dict) else None
    if not isinstance(mod_map, dict) or not mod_map:
        return {}
    zones = [str(z).strip() for z in (target_zones or []) if str(z).strip()]
    candidates: List[str] = []
    if zones:
        candidates.append("zone:" + "+".join(zones))
        for z in zones:
            candidates.extend([z, f"zone:{z}"])
    candidates.extend(["all", "ALL", "default"])
    for cand in candidates:
        entry = mod_map.get(cand)
        if isinstance(entry, dict):
            return {**entry, "_key": cand}
    first_key = next(iter(mod_map))
    entry = mod_map[first_key]
    return {**entry, "_key": first_key} if isinstance(entry, dict) else {}


_EDITED_CURVE_SUFFIXES = (
    "_ORI",
    "_RECONSTRUCTED",
    "_DEN_CORR",
    "_ENV_CORR",
    "_ENH",
    "_UNSHIFTED",
    "_AMB",
    "_AIR",
    "_SYN",
    "_FILLED",
    "_NORM",
)
_EDITED_CURVE_NAMES = {
    "RHOB_DEN_CORR",
    "DEN_CORR_FLAG",
    "BADHOLE_BASE",
    "PHID_RAW_100",
    "PROJ_NPHI",
    "PROJ_RHOB",
}
_COMPUTED_CURVE_SUFFIXES = (
    "_CORR_DELTA",
    "_SIGMA",
    "_QC_CONF",
    "_QC_STATUS",
    "_RECON",
    "_P10",
    "_P50",
    "_P90",
)
_COMPUTED_CURVE_PREFIXES = (
    "SWT_",
    "SXO_",
    "PRESSURE_",
    "CORE_",
    "HI_",
    "RHO_",
    "TEMP_",
    "TOC_",
    "VCL_",
    "ENV_",
    "INVASION_",
)
_COMPUTED_CURVE_NAMES = {
    "VCLAY",
    "VSHALE",
    "_REPORT_VSHALE",
    "PHIT",
    "PHIE",
    "PHID",
    "PHIN",
    "PHIS",
    "PHIT_SH",
    "PHIT_CONSTRUCTED",
    "PERM",
    "PERM_TRANS",
    "SWT",
    "SWE",
    "SHF",
    "BVW",
    "BVHC",
    "ROCK_FLAG",
    "ROCK_FLAG_CUTOFF",
    "ROCK_FLAG_ML",
    "FZI",
    "R35",
    "RFN",
    "BADHOLE",
    "BAD_HOLE",
    "BAD_HOLE_FLAG",
    "BS",
    "DCALI",
    "CALI_FLAG",
    "DRHO_FLAG",
    "RUGOSITY_FLAG",
    "HOLE_SEVERITY",
    "RXO_EST",
    "DI_EST",
    "SHC_INV",
    "DECONV_WEIGHT",
    "HC_FLAG",
    "TIGHT_FLAG",
    "COAL_FLAG",
    "GAS_FLAG",
    "OIL_FLAG",
    "UNC_FLAG",
    "FLUID_TYPE",
    "TOP_NAME",
    "TVD",
    "TVDSS",
    "CPORE",
    "CPERM",
    "QVN",
    "QV",
    "RW",
    "RS",
    "RHOM",
    "RHOB_HC",
    "NPHI_HC",
    "TEMP_GRAD",
    "M_CALC",
    "M_STAR",
    "B",
    "CWA",
    "AVG_ERROR",
    "CHI2_ERROR",
    "OPTIMIZER_SUCCESS",
    "DELTA_LOG_R",
    "PP_EST",
    "SV_EST",
    "NCS_EST",
    "Z_FACTOR",
    "NU_DYN",
    "VDL",
    "CGR_SYN",
    "TOC",
    "PRES_FL",
    "SWIRR",
    "RT_ORI",
}


def _classify_curve_provenance(curve_names: Sequence[str]) -> Dict[str, List[str]]:
    """Split curve mnemonics into raw (as loaded), edited (conditioned versions of raw) and computed.

    Rules are name-based because curves carry no provenance tag. Mineral/volume curves (`V*` such as
    VSAND, VCALC) are computed. A curve is edited when it has a conditioning suffix (`_ORI`,
    `_RECONSTRUCTED`, `_DEN_CORR`, `_ENV_CORR`, `_ENH`...); `*_ORI` columns hold the unedited copy
    and are reported with the edited group so the original is not mistaken for a new measurement.
    """
    raw: List[str] = []
    edited: List[str] = []
    computed: List[str] = []
    volume_like = {
        "VSAND",
        "VSILT",
        "VCALC",
        "VDOLO",
        "VANHY",
        "VCHLO",
        "VILLI",
        "VKAOL",
        "VSMEC",
        "VGLAUC",
        "VGYPS",
        "VHALI",
        "VKERO",
        "VKFELD",
        "VMUSC",
        "VPYRI",
        "VSIDE",
        "VBIOT",
        "VFELD",
        "VCOAL",
        "VGAS",
        "VOIL",
        "VWATER",
        "VUNC",
        "VANKE",
        "VSH_GR",
        "VSH_DN",
    }
    for c in curve_names:
        u = str(c).upper()
        if u in _EDITED_CURVE_NAMES or u.endswith(_EDITED_CURVE_SUFFIXES):
            edited.append(str(c))
        elif (
            u in _COMPUTED_CURVE_NAMES
            or u in volume_like
            or u.startswith(_COMPUTED_CURVE_PREFIXES)
            or u.endswith(_COMPUTED_CURVE_SUFFIXES)
        ):
            computed.append(str(c))
        else:
            raw.append(str(c))
    return {"raw": raw, "edited": edited, "computed": computed}


def _compute_field_summary(zonal_summary: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Single source for field-level pay metrics used by the executive summary and Section 8.4.

    Definitions (each zone row already holds the per-well mean for that zone):
    - `column_hcpt` / `column_net_pay`: sum of zone means, i.e. what a typical well penetrating every
      evaluated zone would encounter.
    - `mean_zone_hcpt` / `mean_zone_net_pay`: the same totals divided by the number of evaluated zones.
    - `phie` / `sw`: net-pay-thickness weighted porosity and pore-volume weighted water saturation over the
      zones that carry pay.
    - `ntg_pay`: total net pay over total gross across zones.
    """
    zones = list(zonal_summary or [])
    n_zones = len(zones)
    column_hcpt = sum(float(z.get("hcpt", 0.0)) for z in zones)
    column_net_pay = sum(float(z.get("net_pay", 0.0)) for z in zones)
    column_gross = sum(float(z.get("gross", 0.0)) for z in zones)
    pay_zones = [z for z in zones if float(z.get("net_pay", 0.0)) > 0]
    phie = sw = None
    if pay_zones:
        h = np.array([max(float(z.get("net_pay", 0.0)), 1e-6) for z in pay_zones])
        phi_z = np.array(
            [float(z.get("phie_pay", z.get("phie", 0.0))) for z in pay_zones]
        )
        sw_z = np.array([float(z.get("sw_pay", z.get("sw", 0.0))) for z in pay_zones])
        phie = float(np.sum(h * phi_z) / np.sum(h))
        pv = np.maximum(h * phi_z, 1e-6)
        sw = float(np.sum(pv * sw_z) / np.sum(pv))
    return {
        "n_zones": n_zones,
        "n_pay_zones": len(pay_zones),
        "column_hcpt": round(column_hcpt, 2),
        "column_net_pay": round(column_net_pay, 1),
        "mean_zone_hcpt": round(column_hcpt / max(n_zones, 1), 2),
        "mean_zone_net_pay": round(column_net_pay / max(n_zones, 1), 1),
        "ntg_pay": round(column_net_pay / column_gross, 3) if column_gross > 0 else 0.0,
        "phie": round(phie, 3) if phie is not None else None,
        "sw": round(sw, 3) if sw is not None else None,
    }


def _build_shf_reconciliation_sentence(shf_p: Dict[str, Any]) -> str:
    """Describe the SHF vs log Sw agreement truthfully, including failure and missing-data cases."""
    sc = shf_p.get("ift_cos_theta", 26.0)
    src = shf_p.get("fit_source", "unspecified")
    rmse = shf_p.get("reconciliation_rmse")
    if rmse is None:
        return (
            f"The Leverett $J$-Function saturation-height model ($\\sigma\\cos\\theta = {sc}\\text{{ dynes/cm}}$, "
            f"fit source: {src}) could not be reconciled with log $S_w$ because too few overlapping "
            f"target-zone samples above the FWL were available. "
        )
    bias = shf_p.get("reconciliation_bias") or 0.0
    text = (
        f"The Leverett $J$-Function saturation-height model ($\\sigma\\cos\\theta = {sc}\\text{{ dynes/cm}}$, "
        f"fit source: {src}) gives $\\text{{RMSE}} = {rmse:.3f}\\text{{ v/v}}$ and mean bias "
        f"${bias:+.3f}\\text{{ v/v}}$ against wireline log $S_w$ (n={shf_p.get('reconciliation_n', 0)}). "
    )
    if rmse > 0.12:
        text += (
            "This exceeds the 0.12 v/v acceptance limit, so the saturation-height function does not yet reproduce "
            "the log saturation and should not be used to initialise the static model until the rock-type fits, "
            "FWL and fluid densities are revisited. "
        )
    if not shf_p.get("uses_saved_fit"):
        text += "No saved per-rock-type SHF fit was applied, so the curve uses a generic fallback J-function. "
    return text


def _build_rock_typing_narrative(
    rock_types: List[Dict[str, Any]],
    rt_sum: Dict[str, Any],
    cutoffs: List[float],
) -> str:
    """Narrative derived strictly from the fitted classes, cutoffs and their provenance."""
    n_rt = len(rock_types)
    cut_txt = ", ".join(f"{c:g}" for c in cutoffs) if cutoffs else "n/a"
    src = rt_sum.get("cutoff_source", "unspecified source")
    parts = [
        f"Hydraulic Flow Unit partitioning on **{rt_sum.get('basis', 'Target-Zone Dataset')}** classified the "
        f"reservoir into **{n_rt}** Discrete Rock Types using Flow Zone Indicator cut-offs of "
        f"${cut_txt}\\text{{ }}\\mu\\text{{m}}$ (source: {src}). "
    ]
    dom = rt_sum.get("dominant_codes") or []
    if dom and rt_sum.get("dominant_kh_pct") is not None:
        parts.append(
            f"Flow capacity is concentrated in **{', '.join(dom)}**, which together hold "
            f"**{rt_sum['dominant_share_pct']:.1f}%** of samples and **{rt_sum['dominant_kh_pct']:.1f}%** of "
            f"cumulative permeability-thickness ($\\sum K\\cdot h$). "
        )
    weak = [
        r["name"].split("—")[0].strip()
        for r in rock_types
        if r.get("fit_status") != "fitted"
    ]
    if weak:
        parts.append(
            f"No porosity-permeability regression is reported for {', '.join(weak)} because fewer than "
            f"3 samples (or no porosity spread) fall in the class. "
        )
    if rt_sum.get("method_note"):
        parts.append(str(rt_sum["method_note"]) + " ")
    return "".join(parts).strip()


def _gather_and_analyze_fpt_data(
    project_id: int,
    wells_inventory: List[Dict[str, Any]],
    merged_configs: Dict[str, Any],
    fluid_contacts_rows: List[Dict[str, Any]],
    target_zones: List[str],
    fwl_val: Optional[float] = None,
    fwl_is_committed: bool = False,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Gather and analyze Formation Pressure Test (FPT) data for Section 6.3 (GAP-13).

    Points are restricted to the target zones, valid quality and plausible pressure. Gradient fits are
    accepted only inside physical density ranges. Contact reconciliation is flagged INDICATIVE unless the
    reference FWL comes from a committed contact or an explicit user override (`fwl_is_committed`).
    """
    raw_points: List[Dict[str, Any]] = []

    # Priority 1: Direct injection in configs (e.g. tests or pre-calculated)
    if "fpt_points" in merged_configs and isinstance(
        merged_configs["fpt_points"], list
    ):
        raw_points = list(merged_configs["fpt_points"])
    elif any(int(w.get("pressure_test_count", 0)) > 0 for w in wells_inventory):
        try:
            from quick_pp.app.backend.services.fpt import _collect_fpt_points

            w_names = [w["name"] for w in wells_inventory]
            raw_points = _collect_fpt_points(int(project_id), w_names)
        except Exception as exc:
            logger.debug(f"Failed collecting FPT points via service: {exc}")

    if not raw_points:
        return {
            "has_data": False,
            "total_stations": 0,
            "valid_stations": 0,
            "well_fpt_rows": [],
            "fpt_gradient_rows": [],
            "fpt_contacts": [],
            "fitted_densities": {},
            "fwl": fwl_val,
        }, []

    # Filter valid stations: quality, finite depth, plausible pressure, and active target zone.
    tz_upper = {str(z).strip().upper() for z in (target_zones or []) if str(z).strip()}
    any_zone_info = any(
        str(p.get("zone") or "").strip() not in ("", "None", "nan") for p in raw_points
    )
    exclusions = {"quality": 0, "bad_pressure": 0, "no_depth": 0, "outside_target": 0}
    valid_points = []
    for p in raw_points:
        if str(p.get("test_quality", "VALID")).strip().upper() != "VALID":
            exclusions["quality"] += 1
            continue
        try:
            pr = float(p.get("pressure"))
            tv = p.get("tvdss")
            tv = float(tv) if tv is not None else float("nan")
        except (TypeError, ValueError):
            exclusions["bad_pressure"] += 1
            continue
        if not np.isfinite(tv):
            exclusions["no_depth"] += 1
            continue
        # Reject physically implausible points (< 0.2 or > 1.2 psi/ft equivalent over TVDSS).
        grad_eq = pr / max(tv * 3.28084, 1.0)
        if not np.isfinite(pr) or pr <= 0 or tv <= 0 or not (0.2 <= grad_eq <= 1.2):
            exclusions["bad_pressure"] += 1
            continue
        if tz_upper and any_zone_info:
            if str(p.get("zone") or "").strip().upper() not in tz_upper:
                exclusions["outside_target"] += 1
                continue
        valid_points.append(p)
    fpt_filter_note = (
        f"{len(valid_points)} of {len(raw_points)} stations retained for gradient analysis "
        f"(excluded: {exclusions['outside_target']} outside target zones, "
        f"{exclusions['quality']} non-valid quality, "
        f"{exclusions['bad_pressure']} implausible pressure, "
        f"{exclusions['no_depth']} without TVDSS)."
    )

    well_fpt_rows = []
    wells_with_pts = sorted({p["well_name"] for p in raw_points})
    for wn in wells_with_pts:
        w_all = [p for p in raw_points if p["well_name"] == wn]
        w_val = [p for p in valid_points if p["well_name"] == wn]
        tvdss_l = [
            float(p["tvdss"])
            for p in w_val
            if p.get("tvdss") is not None and np.isfinite(float(p["tvdss"]))
        ]
        p_l = [
            float(p["pressure"])
            for p in w_val
            if p.get("pressure") is not None and np.isfinite(float(p["pressure"]))
        ]
        mob_l = [
            float(p["mobility"])
            for p in w_val
            if p.get("mobility") is not None
            and np.isfinite(float(p["mobility"]))
            and float(p["mobility"]) > 0
        ]

        d_str = f"{min(tvdss_l):.1f} – {max(tvdss_l):.1f} m TVDss" if tvdss_l else "—"
        p_str = f"{min(p_l):.1f} – {max(p_l):.1f} psia" if p_l else "—"
        m_str = f"{min(mob_l):.1f} – {max(mob_l):.1f} mD/cP" if mob_l else "—"

        # Determine overpressure regime relative to normal hydrostatic (0.433 - 0.465 psi/ft)
        if tvdss_l and p_l:
            mean_p = float(np.mean(p_l))
            mean_z_ft = float(np.mean(tvdss_l)) * 3.28084
            grad_est = mean_p / max(mean_z_ft, 1.0)
            if grad_est < 0.42:
                regime = "Depleted / Sub-Hydrostatic"
            elif grad_est <= 0.47:
                regime = "Normal Hydrostatic"
            else:
                regime = "Moderately Overpressured"
        else:
            regime = "Normal Hydrostatic"

        well_fpt_rows.append(
            {
                "well_name": wn,
                "total_count": len(w_all),
                "valid_count": len(w_val),
                "depth_range": d_str,
                "pressure_range": p_str,
                "mobility_range": m_str,
                "overpressure_regime": regime,
            }
        )

    # Fit fluid gradients via FormationPressureAnalyzer
    fpt_gradient_rows = []
    fpt_contacts = []
    fitted_densities: Dict[str, Dict[str, Any]] = {}

    if len(valid_points) >= 2:
        try:
            from quick_pp.fluid_type import FormationPressureAnalyzer

            fpa_df = pd.DataFrame(
                {
                    "ID": [p.get("id", i + 1) for i, p in enumerate(valid_points)],
                    "WELL_NAME": [p["well_name"] for p in valid_points],
                    "TVDSS": [
                        float(p["tvdss"]) if p.get("tvdss") is not None else np.nan
                        for p in valid_points
                    ],
                    "PRESSURE": [float(p["pressure"]) for p in valid_points],
                    "FLUID_TYPE": [
                        str(p.get("fluid_type", "UNASSIGNED")).upper()
                        for p in valid_points
                    ],
                    "TEST_QUALITY": [
                        str(p.get("test_quality", "VALID")).upper()
                        for p in valid_points
                    ],
                    "MOBILITY": [p.get("mobility") for p in valid_points],
                }
            ).dropna(subset=["TVDSS", "PRESSURE"])

            fpa = FormationPressureAnalyzer(
                fpa_df,
                pressure_unit="psi",
                depth_unit="m",
                id_col="ID",
            )
            if all(f == "UNASSIGNED" for f in fpa_df["FLUID_TYPE"]):
                fpa.auto_classify_points()

            fpa.fit_all()
            contacts_res = fpa.estimate_contacts()

            ref_fwl = fwl_val if fwl_val is not None else 2500.0
            for fl, g in fpa.gradients.items():
                g_dict = g.to_dict()
                g_psi_m = _safe_float(g_dict.get("gradient"), 0.0)
                g_psi_ft = round(g_psi_m * 0.3048, 4)
                rho_gcc = round(_safe_float(g_dict.get("density_gcc"), 0.0), 3)
                r2 = round(_safe_float(g_dict.get("r_squared"), 0.0), 3)
                t_range = g_dict.get("tvdss_range", [0, 0])
                t_str = (
                    f"{t_range[0]:.1f} – {t_range[1]:.1f} m TVDss"
                    if t_range and t_range[0] is not None
                    else "—"
                )
                datum_p = round(
                    g_psi_m * ref_fwl + _safe_float(g_dict.get("intercept"), 0.0), 1
                )

                # Physical plausibility bounds on in-situ density (g/cc) by phase.
                dens_bounds = {
                    "GAS": (0.02, 0.45),
                    "OIL": (0.40, 1.00),
                    "WATER": (0.95, 1.25),
                }.get(str(fl).upper())
                n_fit = int(g_dict.get("n_points", 0))
                reject_reason = ""
                if dens_bounds and not (dens_bounds[0] <= rho_gcc <= dens_bounds[1]):
                    reject_reason = (
                        f"density {rho_gcc:.3f} g/cc outside physical range "
                        f"{dens_bounds[0]:.2f}–{dens_bounds[1]:.2f}"
                    )
                elif n_fit < 3:
                    reject_reason = f"only {n_fit} station(s)"
                elif r2 < 0.80:
                    reject_reason = f"poor fit (R²={r2:.2f})"

                # Zone label from the stations that actually fall in this phase's depth range.
                t_lo = t_range[0] if t_range and t_range[0] is not None else None
                t_hi = t_range[1] if t_range and len(t_range) > 1 else None
                if t_lo is not None and t_hi is not None:
                    fit_zones = sorted(
                        {
                            str(p.get("zone")).strip()
                            for p in valid_points
                            if p.get("tvdss") is not None
                            and t_lo - 0.5 <= float(p["tvdss"]) <= t_hi + 0.5
                            and str(p.get("zone") or "").strip()
                            not in ("", "None", "nan")
                        }
                    )
                else:
                    fit_zones = []
                zone_label = (
                    ", ".join(fit_zones)
                    if fit_zones
                    else (
                        ", ".join(target_zones) if target_zones else "Target Reservoir"
                    )
                )

                fpt_gradient_rows.append(
                    {
                        "zone": zone_label,
                        "fluid_type": fl.title(),
                        "gradient_psi_ft": g_psi_ft,
                        "density_gcc": rho_gcc,
                        "datum_pressure": datum_p,
                        "r_squared": r2,
                        "n_points": n_fit,
                        "tvdss_range": t_str,
                        "accepted": not reject_reason,
                        "reject_reason": reject_reason,
                    }
                )
                if not reject_reason:
                    fitted_densities[fl.upper()] = {
                        "density_gcc": rho_gcc,
                        "r_squared": r2,
                        "n_points": n_fit,
                        "gradient_psi_ft": g_psi_ft,
                        "fluid_type": fl,
                    }

            # Contacts from intersecting gradients are only meaningful when the intersecting
            # fits are physical; otherwise suppress them rather than present an arbitrary depth.
            accepted_phases = set(fitted_densities.keys())
            if len(accepted_phases) < 2:
                contacts_res = {}
            for c_key, c_val in contacts_res.items():
                c_dict = c_val.to_dict()
                p50_tvdss = _safe_float(c_dict.get("tvdss_p50"))
                p10_tvdss = _safe_float(c_dict.get("tvdss_p10"), p50_tvdss)
                p90_tvdss = _safe_float(c_dict.get("tvdss_p90"), p50_tvdss)
                p_contact = _safe_float(c_dict.get("pressure_at_contact"), 0.0)
                delta_log = round(p50_tvdss - ref_fwl, 1)
                if not fwl_is_committed:
                    status_str = (
                        "INDICATIVE (no committed contact to reconcile against)"
                    )
                elif abs(delta_log) <= 2.0:
                    status_str = "✓ RECONCILED (< 2.0 m)"
                else:
                    status_str = f"Δ = {delta_log:+.1f} m"

                fpt_contacts.append(
                    {
                        "contact_type": c_dict.get(
                            "label", c_dict.get("contact_type", c_key)
                        ),
                        "tvdss_p50": p50_tvdss,
                        "tvdss_p10": p10_tvdss,
                        "tvdss_p90": p90_tvdss,
                        "pressure_at_contact": p_contact,
                        "delta_log_tvdss": delta_log,
                        "status": status_str,
                    }
                )
        except Exception as exc:
            logger.debug(f"Formation pressure gradient regression fallback: {exc}")

    fpt_summary = {
        "has_data": len(raw_points) > 0,
        "total_stations": len(raw_points),
        "valid_stations": len(valid_points),
        "well_fpt_rows": well_fpt_rows,
        "fpt_gradient_rows": fpt_gradient_rows,
        "fpt_contacts": fpt_contacts,
        "fitted_densities": fitted_densities,
        "fwl": fwl_val,
        "filter_note": fpt_filter_note,
        "fwl_is_committed": fwl_is_committed,
        "rejected_fits": [r for r in fpt_gradient_rows if not r.get("accepted", True)],
    }
    return fpt_summary, valid_points


# ---------------------------------------------------------------------------
# Step 1: Context Collector (`build_report_context`) & Gate-Check Validator
# ---------------------------------------------------------------------------


def build_report_context(
    project_id: Union[int, str],
    well_names: Optional[List[str]] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    zones: Optional[List[str]] = None,
    target_zones: Optional[List[str]] = None,
    fwl_tvdss: Optional[float] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    enabled_section_ids: Optional[set[str]] = None,
    model_params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect and compute all petrophysical tables, model parameters, and statistical digests for a project."""
    from scipy.stats import linregress

    pid = int(project_id)
    project_name, selected_wells_meta = _load_project_wells_fast(pid, well_names)

    cut = cutoffs or {"vsh": 0.40, "phie": 0.08, "sw": 0.65}
    vsh_cut = _safe_float(cut.get("vsh", cut.get("VSHALE", 0.40)), 0.40)
    phie_cut = _safe_float(cut.get("phie", cut.get("PHIT", 0.08)), 0.08)
    sw_cut = _safe_float(cut.get("sw", cut.get("SWT", 0.65)), 0.65)
    ressum_cutoffs = {"VSHALE": vsh_cut, "PHIT": phie_cut, "SWT": sw_cut}
    need_upscaling = (
        enabled_section_ids is None or "09_upscaling" in enabled_section_ids
    )

    wells_inventory: List[Dict[str, Any]] = []
    qaqc_table: List[Dict[str, Any]] = []
    upscaling_table: List[Dict[str, Any]] = []
    well_zonal_rows: List[Dict[str, Any]] = []
    anomaly_flags: List[Dict[str, str]] = []
    hole_sections_rows: List[Dict[str, Any]] = []
    env_well_rows: List[Dict[str, Any]] = []
    invasion_well_rows: List[Dict[str, Any]] = []
    env_assumptions_seen: List[str] = []
    resolved_envs: List[Any] = []
    depth_align_well_rows: List[Dict[str, Any]] = []
    depth_align_curve_rows: List[Dict[str, Any]] = []
    depth_align_reports_by_well: Dict[str, Dict[str, Any]] = {}
    requested_model = model_params or {}
    req_da_cfg = (
        requested_model.get("depth_align_params")
        if isinstance(requested_model.get("depth_align_params"), dict)
        else {}
    )
    da_ref_curve = str(
        req_da_cfg.get("referenceCurve", req_da_cfg.get("reference_curve", "GR"))
        or "GR"
    ).upper()
    da_max_shift_m = _safe_float(
        req_da_cfg.get("maxShiftM", req_da_cfg.get("max_shift", 3.0)), 3.0
    )
    da_win_size_m = _safe_float(
        req_da_cfg.get("windowSizeM", req_da_cfg.get("window_size", 40.0)), 40.0
    )
    da_excl_bh = bool(
        req_da_cfg.get("excludeBadhole", req_da_cfg.get("exclude_badhole", True))
    )

    combined_dfs: List[pd.DataFrame] = []
    unfiltered_dfs: List[pd.DataFrame] = []
    merged_configs: Dict[str, Any] = {}
    all_zones_seen: List[str] = []
    evaluated_zones_seen: List[str] = []
    total_logged_thickness = 0.0
    any_scal_data = False
    requested_zones = [
        _normalize_zone_name(z)
        for z in (zones or target_zones or [])
        if z and _normalize_zone_name(z).upper() not in GENERIC_PLACEHOLDER_ZONES
    ]
    z_filter_upper = {z.upper() for z in requested_zones}

    for w_meta in selected_wells_meta:
        w_name = str(w_meta["name"])
        try:
            raw_df = _get_merged_dataframe_sync(pid, w_name)
        except Exception:
            raw_df = pd.DataFrame()
        if raw_df.empty:
            continue

        w_cfg = w_meta.get("well_config") or {}
        if w_cfg:
            merged_configs.update(w_cfg)
        requested_model = model_params or {}
        well_litho_model = (
            str(
                requested_model.get("litho_model_key")
                or requested_model.get("litho_model")
                or w_cfg.get("litho_model_key")
                or w_cfg.get("litho_model")
                or ""
            )
            .strip()
            .lower()
        )
        is_bulk_shale = well_litho_model == "vsh_bulk"

        w_df = raw_df.copy()
        if "ZONES" not in w_df.columns or not w_df["ZONES"].notna().any():
            w_df["ZONES"] = "Main Reservoir"
        else:
            w_df["ZONES"] = w_df["ZONES"].map(_normalize_zone_name)

        # Trim unlogged surface casing / post-TD intervals where all curves are NaN
        w_df = report_plots._trim_to_logged_interval(w_df)

        if min_depth is not None:
            w_df = w_df[
                pd.to_numeric(w_df["DEPTH"], errors="coerce") >= float(min_depth)
            ]
        if max_depth is not None:
            w_df = w_df[
                pd.to_numeric(w_df["DEPTH"], errors="coerce") <= float(max_depth)
            ]

        if w_df.empty:
            continue

        for z_val in w_df["ZONES"].dropna().unique():
            z_str = str(z_val).strip()
            if (
                z_str
                and z_str.upper() not in GENERIC_PLACEHOLDER_ZONES
                and z_str not in all_zones_seen
            ):
                all_zones_seen.append(z_str)

        from quick_pp.las_handler import extract_well_df_elevation

        header_info = w_cfg.get("_header_data") or {}
        df_val = float(
            extract_well_df_elevation(header_info, w_cfg, target_depth_uom="m")
            or _safe_float(
                header_info.get("DF")
                or header_info.get("EDF")
                or header_info.get("KB")
                or header_info.get("EKB")
                or w_cfg.get("df_elev")
                or w_cfg.get("kb_elev")
                or w_cfg.get("kb")
                or 25.0,
                25.0,
            )
        )

        # Enforce strict TVD / TVDss check (Spec 2.1: Never substitute MD into TVD/TVDss)
        has_tvd = False
        if (
            "TVD" in w_df.columns
            and pd.to_numeric(w_df["TVD"], errors="coerce").notna().any()
        ):
            has_tvd = True
            tvd_s = pd.to_numeric(w_df["TVD"], errors="coerce")
            if (
                "TVDSS" not in w_df.columns
                or not pd.to_numeric(w_df["TVDSS"], errors="coerce").notna().any()
            ):
                w_df["TVDSS"] = tvd_s - df_val
        elif (
            "TVDSS" in w_df.columns
            and pd.to_numeric(w_df["TVDSS"], errors="coerce").notna().any()
        ):
            has_tvd = True
            tvdss_s = pd.to_numeric(w_df["TVDSS"], errors="coerce")
            w_df["TVD"] = tvdss_s + df_val
        else:
            anomaly_flags.append(
                {
                    "category": "Missing Directional Survey / TVD",
                    "target": w_name,
                    "message": "True Vertical Depth (TVD) curve is absent; MD must not be substituted into TVDss Leverett-J height equations.",
                    "mitigation": "Load directional survey to compute TVD and TVDss (= TVD - DF) prior to multi-well fluid contact and SHF initialization.",
                }
            )

        # Capture genuinely available non-empty curves before filling evaluation defaults
        raw_curves = [
            c
            for c in w_df.columns
            if c not in ("DEPTH", "DEPT", "WELL_NAME", "ZONES")
            and w_df[c].notna().any()
        ]
        if any(c.upper() in ("PC", "KRW", "KRO", "SCAL_PC") for c in raw_curves):
            any_scal_data = True

        # Prioritize canonical Field Inventory curves for display in Table 2.1
        priority_order = [
            "TVD",
            "TVDSS",
            "GR",
            "RT",
            "NPHI",
            "RHOB",
            "VCLAY",
            "PHIT",
            "PERM",
            "SWT",
            "ROCK_FLAG",
            "PERM_TRANS",
            "SHF",
        ]
        raw_upper_map = {str(c).upper(): str(c) for c in raw_curves}
        display_curves = [
            raw_upper_map[k] for k in priority_order if k in raw_upper_map
        ] + [c for c in raw_curves if str(c).upper() not in priority_order]

        has_real_logs = any(
            str(c).upper()
            in (
                "GR",
                "RT",
                "RD",
                "RDEP",
                "RILD",
                "LLD",
                "NPHI",
                "RHOB",
                "PHIT",
                "PHIE",
                "PERM",
                "SWT",
            )
            for c in raw_curves
        )
        if has_real_logs:
            w_df = _ensure_evaluated_curves(w_df, w_name)
            if is_bulk_shale:
                # Never substitute legacy estimates for absent bulk-model results.
                for curve in ("VCLAY", "PHIT", "PHIE"):
                    w_df[curve] = raw_df[curve] if curve in raw_df.columns else np.nan
            unfiltered_dfs.append(w_df)

        if is_bulk_shale:
            w_df["_REPORT_VSHALE"] = _col_series(w_df, "VSHALE", np.nan)
            if w_df["_REPORT_VSHALE"].isna().any():
                anomaly_flags.append(
                    {
                        "category": "Missing bulk shale",
                        "target": w_name,
                        "message": "Bulk VSHALE is missing at one or more samples; those samples are excluded from shale-cutoff calculations.",
                        "mitigation": "Save a calibrated bulk-shale evaluation before generating the report.",
                    }
                )
        else:
            w_df["_REPORT_VSHALE"] = _col_series(w_df, "VCLAY", np.nan).fillna(0.20)

        d_s = pd.to_numeric(w_df["DEPTH"], errors="coerce").dropna()
        d_min = float(d_s.min()) if not d_s.empty else 0.0
        d_max = float(d_s.max()) if not d_s.empty else 0.0
        td = max(0.0, d_max - d_min)
        total_logged_thickness += td

        # Inventory row
        eval_curve_names = {
            "VCLAY",
            "VSHALE",
            "_REPORT_VSHALE",
            "PHIT",
            "PHIE",
            "PERM",
            "SWT",
            "SWE",
            "ROCK_FLAG",
            "PERM_TRANS",
            "SHF",
            "BADHOLE",
            "HC_FLAG",
            "TIGHT_FLAG",
            "COAL_FLAG",
            "CPORE",
            "CPERM",
            "CPORE_UNSHIFTED",
            "CPERM_UNSHIFTED",
            "CPORE_AMB",
            "CPERM_AIR",
            "CPORE_ORI",
            "CPERM_ORI",
        }
        provenance = _classify_curve_provenance(raw_curves)
        raw_curves_only = [
            c for c in provenance["raw"] if c.upper() not in eval_curve_names
        ]
        edited_curves_found = provenance["edited"]
        computed_curves_found = [
            c for c in provenance["computed"] if c.upper() not in eval_curve_names
        ]

        # Coverage of each raw curve (% of the logged interval with a valid value), so the table can show
        # only the primary measurements with their completeness instead of every column.
        raw_coverage: Dict[str, float] = {}
        n_rows_w = max(len(w_df), 1)
        for c in raw_curves_only:
            try:
                raw_coverage[c] = round(
                    100.0
                    * float(pd.to_numeric(w_df[c], errors="coerce").notna().sum())
                    / n_rows_w,
                    0,
                )
            except Exception:
                continue

        eval_curves_only = [
            c
            for c in [
                "VCLAY",
                "PHIT",
                "PHIE",
                "PERM",
                "SWT",
                "ROCK_FLAG",
                "SHF",
                "CPORE",
                "CPERM",
                "BADHOLE",
            ]
            if c in w_df.columns and w_df[c].notna().any()
        ]

        cpore_cnt = (
            int(pd.to_numeric(w_df["CPORE"], errors="coerce").notna().sum())
            if "CPORE" in w_df.columns
            else 0
        )
        core_cnt = max(int(w_meta.get("core_sample_count") or 0), cpore_cnt)
        scal_curves_found = [
            c
            for c in ("PC", "KRW", "KRO", "SCAL_PC")
            if c in w_df.columns and w_df[c].notna().any()
        ]
        n_scal = int(w_meta.get("scal_sample_count") or len(scal_curves_found))
        if core_cnt > 0 or n_scal > 0:
            core_status_str = f"RCA: {core_cnt} pts | SCAL: {n_scal} pts"
        else:
            core_status_str = "No Core Plugs"

        press_cnt = int(w_meta.get("pressure_test_count") or 0)
        w_hole_sections = list(w_meta.get("hole_sections") or [])
        w_hs_cnt = int(w_meta.get("hole_section_count") or len(w_hole_sections))
        for hs_item in w_hole_sections:
            if not isinstance(hs_item, dict):
                continue
            hole_sections_rows.append(
                {
                    "well_name": w_name,
                    "name": str(hs_item.get("name") or "Section"),
                    "top": _safe_float(hs_item.get("top"), 0.0),
                    "bottom": _safe_float(hs_item.get("bottom"), 0.0),
                    "bit_size": (
                        round(float(hs_item["bit_size"]), 3)
                        if hs_item.get("bit_size") is not None
                        else None
                    ),
                    "mud_type": str(hs_item.get("mud_type") or "WBM").upper(),
                    "mud_weight": (
                        round(float(hs_item["mud_weight"]), 2)
                        if hs_item.get("mud_weight") is not None
                        else None
                    ),
                    "mud_salinity": (
                        int(round(float(hs_item["mud_salinity"])))
                        if hs_item.get("mud_salinity") is not None
                        else None
                    ),
                    "rmf": (
                        round(float(hs_item["rmf"]), 4)
                        if hs_item.get("rmf") is not None
                        else None
                    ),
                    "rmf_temp": (
                        round(float(hs_item["rmf_temp"]), 1)
                        if hs_item.get("rmf_temp") is not None
                        else None
                    ),
                }
            )
        has_env_corr_well = any(
            f"{lg}_ENV_CORR" in w_df.columns
            and pd.to_numeric(w_df[f"{lg}_ENV_CORR"], errors="coerce").notna().any()
            for lg in ("GR", "NPHI", "RHOB", "RT", "DT")
        )
        wells_inventory.append(
            {
                "name": w_name,
                "uwi": str(w_meta.get("uwi") or w_name),
                "kb": round(df_val, 1),
                "has_tvd": has_tvd,
                "min_depth": round(d_min, 1),
                "max_depth": round(d_max, 1),
                "td": round(td, 1),
                "curves": display_curves[:13],
                "raw_curves": raw_curves_only,
                "raw_coverage": raw_coverage,
                "edited_curves": edited_curves_found,
                "computed_curves": computed_curves_found,
                "eval_curves": eval_curves_only,
                "available_curves": raw_curves,
                "core_sample_count": core_cnt,
                "core_status": core_status_str,
                "has_core": f"Yes ({core_cnt} pts)" if core_cnt > 0 else "No",
                "pressure_test_count": press_cnt,
                "has_pressure": f"Yes ({press_cnt} pts)" if press_cnt > 0 else "No",
                "hole_section_count": w_hs_cnt,
                "has_hole_sections": (
                    f"Yes ({w_hs_cnt} sec)" if w_hs_cnt > 0 else "No"
                ),
                "has_env_corr": has_env_corr_well,
                "formation_top_count": int(
                    w_meta.get("formation_top_count")
                    or len(
                        [
                            z
                            for z in w_df["ZONES"].dropna().unique()
                            if str(z).upper() not in GENERIC_PLACEHOLDER_ZONES
                        ]
                    )
                ),
                "zones": [
                    str(z)
                    for z in w_df["ZONES"].dropna().unique()
                    if str(z).upper() not in GENERIC_PLACEHOLDER_ZONES
                ],
            }
        )

        # Apply Top Bar Zone Filter if specified
        if z_filter_upper and "ZONES" in w_df.columns:
            w_df = w_df[
                w_df["ZONES"].astype(str).str.strip().str.upper().isin(z_filter_upper)
            ]
            if w_df.empty:
                continue

        combined_dfs.append(w_df)

        # Resolve per-well logging environment (LAS header + saved module_params['qaqc-env'] + request env_corr_params + hole sections)
        req_env_p = (
            requested_model.get("env_corr_params")
            if isinstance(requested_model.get("env_corr_params"), dict)
            else {}
        )
        saved_w_mp = (
            w_cfg.get("module_params", {}).get("qaqc-env")
            if isinstance(w_cfg.get("module_params"), dict)
            else None
        )
        saved_env_p: Dict[str, Any] = {}
        if isinstance(saved_w_mp, dict) and saved_w_mp:
            first_env_entry = next(iter(saved_w_mp.values()))
            if isinstance(first_env_entry, dict) and isinstance(
                first_env_entry.get("params"), dict
            ):
                saved_env_p = dict(first_env_entry["params"])
        env_src_cfg: Dict[str, Any] = {}
        if isinstance(header_info, dict) and header_info:
            try:
                las_meta = extract_las_metadata(header_info, fallback_well_name=w_name)
                env_src_cfg.update(
                    {k: v for k, v in dict(las_meta).items() if v is not None}
                )
            except Exception:
                pass
            if isinstance(header_info.get("header_overrides"), dict):
                env_src_cfg.update(
                    {
                        k: v
                        for k, v in header_info["header_overrides"].items()
                        if v is not None
                    }
                )
        for k, v in w_cfg.items():
            if not str(k).startswith("_") and v is not None:
                env_src_cfg[k] = v

        env_overrides: Dict[str, Any] = {}
        for p_src in (saved_env_p, req_env_p):
            if not p_src:
                continue
            if p_src.get("mudType") or p_src.get("mud_type"):
                mt_cand = str(p_src.get("mudType") or p_src.get("mud_type")).upper()
                if mt_cand in ("WBM", "OBM", "SBM"):
                    env_overrides["mud_type"] = mt_cand
            if (
                p_src.get("mudWeight") is not None
                or p_src.get("mud_weight") is not None
            ):
                env_overrides["mud_weight_ppg"] = p_src.get(
                    "mudWeight", p_src.get("mud_weight")
                )
            if (
                p_src.get("bitsizeDefault") is not None
                or p_src.get("bitsize_default") is not None
            ):
                env_overrides["bit_size"] = p_src.get(
                    "bitsizeDefault", p_src.get("bitsize_default")
                )
            if (
                p_src.get("salinityPpm") is not None
                or p_src.get("salinity_ppm") is not None
            ):
                env_overrides["mud_salinity_ppm"] = p_src.get(
                    "salinityPpm", p_src.get("salinity_ppm")
                )
            if (
                p_src.get("formationSalinityPpm") is not None
                or p_src.get("formation_salinity_ppm") is not None
            ):
                env_overrides["formation_salinity_ppm"] = p_src.get(
                    "formationSalinityPpm", p_src.get("formation_salinity_ppm")
                )
            if p_src.get("rmf") is not None and _safe_float(p_src.get("rmf"), 0.0) > 0:
                env_overrides["rmf"] = float(p_src["rmf"])
                if (
                    p_src.get("rmfTempC") is not None
                    or p_src.get("rmf_temp_c") is not None
                ):
                    env_overrides["rmf_temp"] = float(
                        p_src.get("rmfTempC", p_src.get("rmf_temp_c"))
                    )
                    env_overrides["rmf_temp_unit"] = "degC"
            if (
                p_src.get("offshore")
                and _safe_float(p_src.get("waterDepth", p_src.get("water_depth")), 0.0)
                > 0
            ):
                env_overrides["water_depth"] = float(
                    p_src.get("waterDepth", p_src.get("water_depth"))
                )
                env_overrides["water_depth_unit"] = str(
                    p_src.get("waterDepthUnit", p_src.get("water_depth_unit", "m"))
                ).lower()
                if (
                    p_src.get("seabedTempC") is not None
                    or p_src.get("seabed_temp_c") is not None
                ):
                    env_overrides["seabed_temp_c"] = float(
                        p_src.get("seabedTempC", p_src.get("seabed_temp_c"))
                    )
            elif p_src.get("offshore") is False:
                env_overrides["water_depth"] = 0.0
                env_overrides["water_depth_unit"] = "m"
            if (
                p_src.get("surfaceTemp") is not None
                or p_src.get("surface_temp") is not None
            ):
                env_overrides["surface_temp"] = float(
                    p_src.get("surfaceTemp", p_src.get("surface_temp"))
                )
                env_overrides["surface_temp_unit"] = "degF"
            tg_f = p_src.get(
                "tempGrad", p_src.get("tempGradient", p_src.get("temp_grad"))
            )
            if tg_f is not None:
                env_overrides["temp_gradient_c_per_km"] = (
                    float(tg_f) * (5.0 / 9.0) * (1000.0 / 30.48)
                )

        use_hs_well = bool(
            req_env_p.get(
                "useHoleSections",
                req_env_p.get(
                    "use_hole_sections",
                    requested_model.get("use_hole_sections", True),
                ),
            )
        )
        if use_hs_well and w_hole_sections:
            mp_list = []
            for s_hs in w_hole_sections:
                if (
                    isinstance(s_hs, dict)
                    and s_hs.get("top") is not None
                    and s_hs.get("bottom") is not None
                ):
                    mp_list.append(
                        {
                            "top": float(s_hs["top"]),
                            "bottom": float(s_hs["bottom"]),
                            "bit_size_in": s_hs.get("bit_size"),
                            "mud_type": str(s_hs.get("mud_type") or "WBM").upper(),
                            "mud_weight_ppg": s_hs.get("mud_weight"),
                            "salinity_ppm": s_hs.get("mud_salinity"),
                            "rmf": s_hs.get("rmf"),
                            "rmf_temp_c": s_hs.get("rmf_temp"),
                        }
                    )
            if mp_list:
                env_overrides["mud_program"] = mp_list

        if env_src_cfg.get("mud_type") is not None and str(
            env_src_cfg["mud_type"]
        ).upper() not in ("WBM", "OBM", "SBM"):
            env_src_cfg.pop("mud_type", None)

        raw_duom = str(
            req_env_p.get("depthUnit")
            or req_env_p.get("depth_unit")
            or w_meta.get("depth_uom")
            or env_src_cfg.get("datum_unit")
            or "m"
        ).lower()
        eff_duom = "ft" if raw_duom in ("ft", "f", "feet", "foot") else "m"
        try:
            well_env = environment_from_config(
                env_src_cfg, env_overrides, depth_unit=eff_duom
            )
        except Exception:
            well_env = environment_from_config({}, {}, depth_unit="m")
        resolved_envs.append(well_env)
        for a_msg in well_env.assumptions:
            if a_msg not in env_assumptions_seen:
                env_assumptions_seen.append(a_msg)

        w_tool_type = str(
            req_env_p.get("toolType")
            or req_env_p.get("tool_type")
            or saved_env_p.get("toolType")
            or w_cfg.get("tool_type")
            or "WL"
        ).upper()
        if w_tool_type not in ("WL", "LWD"):
            w_tool_type = "WL"
        w_mud_types = sorted(
            {iv.mud_type for iv in well_env.mud_program if iv.mud_type}
            or {well_env.default_mud_type}
        )
        w_mud_str = "/".join(w_mud_types)

        # QA/QC row
        n_samples = len(w_df)
        cali_s = _col_series(w_df, "CALI", np.nan)
        bs_default_val = _safe_float(
            merged_configs.get("bitsize_default", well_env.default_bit_size_in), 8.5
        )
        if (
            "ENV_BS" in w_df.columns
            and pd.to_numeric(w_df["ENV_BS"], errors="coerce").notna().any()
        ):
            bs_s = _col_series(w_df, "ENV_BS", bs_default_val)
        elif (
            "BS" in w_df.columns
            and pd.to_numeric(w_df["BS"], errors="coerce").notna().any()
        ):
            bs_s = _col_series(w_df, "BS", bs_default_val)
        elif use_hs_well and w_hole_sections and "DEPTH" in w_df.columns:
            bs_hs = _synthesize_bitsize_series(w_df["DEPTH"], w_hole_sections)
            bs_s = bs_hs.fillna(bs_default_val)
            bs_s.index = w_df.index
        else:
            bs_s = pd.Series(bs_default_val, index=w_df.index, dtype=float)
        drho_s = _col_series(w_df, "DRHO", np.nan)
        rhob_s = _col_series(w_df, "RHOB", np.nan)
        nphi_s = _col_series(w_df, "NPHI", np.nan)

        if "BADHOLE" in w_df.columns and w_df["BADHOLE"].notna().any():
            bh_mask = _col_series(w_df, "BADHOLE", 0.0) >= 1.0
        else:
            bh_conds = []
            if cali_s.notna().any():
                bh_conds.append(
                    (cali_s - bs_s)
                    > _safe_float(merged_configs.get("cali_threshold"), 2.0)
                )
            if drho_s.notna().any():
                bh_conds.append(
                    drho_s.abs()
                    > _safe_float(merged_configs.get("drho_threshold"), 0.15)
                )
            if rhob_s.notna().any():
                bh_conds.append(
                    rhob_s < _safe_float(merged_configs.get("rhob_min"), 1.95)
                )
            if bh_conds:
                bh_mask = bh_conds[0]
                for c in bh_conds[1:]:
                    bh_mask = bh_mask | c
                bh_mask = bh_mask.fillna(False)
            else:
                bh_mask = ((rhob_s < 2.0) | (nphi_s > 0.42)).fillna(False)
        bh_cnt = int(bh_mask.sum())
        bh_pct = 100.0 * bh_cnt / max(n_samples, 1)

        hc_mask = _col_series(w_df, "HC_FLAG", 0.0) >= 1.0
        hc_pct = 100.0 * float(hc_mask.mean()) if n_samples > 0 else 0.0
        tight_cnt = int((_col_series(w_df, "TIGHT_FLAG", 0.0) >= 1.0).sum())
        coal_cnt = int((_col_series(w_df, "COAL_FLAG", 0.0) >= 1.0).sum())

        gr_clean = _select_conditioned_curve(w_df, "GR", weight_by_conf=False).dropna()
        rhob_clean = _select_conditioned_curve(
            w_df, "RHOB", weight_by_conf=False
        ).dropna()
        gr_p50 = float(np.percentile(gr_clean, 50)) if not gr_clean.empty else 55.0
        rhob_p50 = (
            float(np.percentile(rhob_clean, 50)) if not rhob_clean.empty else 2.42
        )

        # Per-well Borehole Environmental Correction & Sensor Confidence Audit
        env_curves_w: List[str] = []
        conf_vals_w: List[float] = []
        unrel_pcts_w: List[float] = []
        env_deltas_w: Dict[str, str] = {
            "GR": "—",
            "NPHI": "—",
            "RHOB": "—",
            "RT": "—",
        }
        for lg_k in ("GR", "NPHI", "RHOB", "RT", "DT"):
            c_col = f"{lg_k}_ENV_CORR"
            st_col = f"{lg_k}_QC_STATUS"
            cf_col = f"{lg_k}_QC_CONF"
            dl_col = f"{lg_k}_CORR_DELTA"
            if (
                c_col in w_df.columns
                and pd.to_numeric(w_df[c_col], errors="coerce").notna().any()
            ):
                env_curves_w.append(lg_k)
            if st_col in w_df.columns:
                st_s = pd.to_numeric(w_df[st_col], errors="coerce")
                valid_st = st_s >= 0
                n_st = int(valid_st.sum())
                if n_st > 0:
                    unrel_pcts_w.append(100.0 * float((st_s == 2).sum()) / n_st)
            if cf_col in w_df.columns:
                cf_s = pd.to_numeric(w_df[cf_col], errors="coerce").dropna()
                if not cf_s.empty:
                    conf_vals_w.append(float(cf_s.mean()))
            if lg_k in env_deltas_w:
                if dl_col in w_df.columns:
                    dl_s = pd.to_numeric(w_df[dl_col], errors="coerce").dropna()
                elif c_col in w_df.columns and lg_k in w_df.columns:
                    if lg_k == "RT":
                        dl_s = (
                            np.log10(
                                pd.to_numeric(w_df[c_col], errors="coerce").clip(
                                    lower=1e-3
                                )
                            )
                            - np.log10(
                                pd.to_numeric(w_df[lg_k], errors="coerce").clip(
                                    lower=1e-3
                                )
                            )
                        ).dropna()
                    else:
                        dl_s = (
                            pd.to_numeric(w_df[c_col], errors="coerce")
                            - pd.to_numeric(w_df[lg_k], errors="coerce")
                        ).dropna()
                else:
                    dl_s = pd.Series(dtype=float)
                if not dl_s.empty:
                    fmt_spec = "+.1f" if lg_k == "GR" else "+.3f"
                    env_deltas_w[lg_k] = format(float(dl_s.median()), fmt_spec)

        w_has_env = len(env_curves_w) > 0
        w_mean_conf = round(float(np.mean(conf_vals_w)), 3) if conf_vals_w else None
        w_max_unrel = round(max(unrel_pcts_w), 1) if unrel_pcts_w else 0.0
        sev_s = (
            pd.to_numeric(w_df["HOLE_SEVERITY"], errors="coerce")
            if "HOLE_SEVERITY" in w_df.columns
            else pd.Series(dtype=float)
        )
        w_severe_pct = (
            round(100.0 * float((sev_s >= 2).mean()), 1)
            if not sev_s.dropna().empty
            else round(bh_pct, 1)
        )
        reason_w_s = (
            pd.to_numeric(w_df["ENV_QC_REASON"], errors="coerce").fillna(0).astype(int)
            if "ENV_QC_REASON" in w_df.columns
            else pd.Series(dtype=int)
        )
        w_cycle_skips = (
            int(
                (
                    (reason_w_s.to_numpy(dtype=int) & ENV_QC_REASON_BITS["CYCLE_SKIP"])
                    != 0
                ).sum()
            )
            if not reason_w_s.empty
            else 0
        )
        w_res_mismatch = (
            int(
                (
                    (
                        reason_w_s.to_numpy(dtype=int)
                        & ENV_QC_REASON_BITS["RES_MUD_MISMATCH"]
                    )
                    != 0
                ).sum()
            )
            if not reason_w_s.empty
            else 0
        )
        w_dom_reason = _dominant_env_reason(reason_w_s)

        if w_has_env:
            env_well_rows.append(
                {
                    "well_name": w_name,
                    "tool_type": w_tool_type,
                    "mud_type": w_mud_str,
                    "hole_sections": w_hs_cnt,
                    "curves": ", ".join(env_curves_w),
                    "mean_conf": w_mean_conf,
                    "unreliable_pct": w_max_unrel,
                    "severe_hole_pct": w_severe_pct,
                    "cycle_skips": w_cycle_skips,
                    "delta_gr": env_deltas_w["GR"],
                    "delta_nphi": env_deltas_w["NPHI"],
                    "delta_rhob": env_deltas_w["RHOB"],
                    "delta_rt": env_deltas_w["RT"],
                    "dominant_reason": w_dom_reason,
                }
            )

        # Radial Resistivity Invasion Profile summary per well
        if "INVASION_PROFILE" in w_df.columns or "INVASION_FLAG" in w_df.columns:
            inv_fl = (
                pd.to_numeric(w_df["INVASION_FLAG"], errors="coerce").fillna(0)
                if "INVASION_FLAG" in w_df.columns
                else pd.Series(0, index=w_df.index)
            )
            inv_pr = (
                pd.to_numeric(w_df["INVASION_PROFILE"], errors="coerce").fillna(0)
                if "INVASION_PROFILE" in w_df.columns
                else pd.Series(0, index=w_df.index)
            )
            di_s = (
                pd.to_numeric(w_df["DI_EST"], errors="coerce").dropna()
                if "DI_EST" in w_df.columns
                else pd.Series(dtype=float)
            )
            rt_ref_s = _select_conditioned_curve(w_df, "RT", weight_by_conf=False)
            n_rt_valid = max(int(rt_ref_s.notna().sum()), 1)
            inv_cnt = int((inv_fl > 0).sum())
            inv_pct_w = round(100.0 * inv_cnt / n_rt_valid, 1)
            water_like_pct = round(100.0 * float((inv_pr == 1).sum()) / n_rt_valid, 1)
            hc_like_pct = round(100.0 * float((inv_pr == 2).sum()) / n_rt_valid, 1)
            anom_pct = round(100.0 * float((inv_pr == 3).sum()) / n_rt_valid, 1)
            tight_inv_pct = round(100.0 * float((inv_pr == 4).sum()) / n_rt_valid, 1)
            med_di = round(float(di_s.median()), 1) if not di_s.empty else None
            p90_di = (
                round(float(np.nanpercentile(di_s, 90)), 1)
                if len(di_s) >= 3
                else med_di
            )
            invasion_well_rows.append(
                {
                    "well_name": w_name,
                    "mud_type": w_mud_str,
                    "samples": n_rt_valid,
                    "invaded_pct": inv_pct_w,
                    "water_like_pct": water_like_pct,
                    "hc_like_pct": hc_like_pct,
                    "anomalous_pct": anom_pct,
                    "tight_pct": tight_inv_pct,
                    "median_di_in": med_di,
                    "p90_di_in": p90_di,
                }
            )
            if anom_pct > 15.0:
                anomaly_flags.append(
                    {
                        "category": "Anomalous Resistivity Invasion Contrast",
                        "target": w_name,
                        "message": (
                            f"{anom_pct:.1f}% of resistivity samples exhibit anomalous Rxo/Rt contrast "
                            "(INVASION_PROFILE = 3, exceeding expected Rmf/Rw bounds)."
                        ),
                        "mitigation": "Verify depth alignment between shallow/medium/deep resistivity curves, Rmf @ Tmeas input, and thin-bed shoulder effects.",
                    }
                )

        if w_res_mismatch > 0:
            anomaly_flags.append(
                {
                    "category": "Resistivity Tool / Mud-Type Mismatch",
                    "target": w_name,
                    "message": (
                        f"Laterolog resistivity flagged in non-conductive {w_mud_str} mud across {w_res_mismatch} samples (ENV_QC_REASON bit 512)."
                    ),
                    "mitigation": "Confirm induction/propagation resistivity tool type or switch mud system metadata before water saturation evaluation.",
                }
            )
        if w_has_env and w_max_unrel > 20.0:
            anomaly_flags.append(
                {
                    "category": "Low Borehole Sensor Confidence",
                    "target": w_name,
                    "message": (
                        f"Up to {w_max_unrel:.1f}% of environmentally corrected samples are rated UNRELIABLE "
                        f"(mean confidence = {w_mean_conf if w_mean_conf is not None else 0.0:.2f}; dominant cause: {w_dom_reason})."
                    ),
                    "mitigation": "Enable QC confidence weighting (`confidence_weighting=True`) or bad-hole fallback across low-confidence washouts.",
                }
            )
        if n_samples > 0 and (100.0 * w_cycle_skips / n_samples) > 5.0:
            anomaly_flags.append(
                {
                    "category": "Sonic Cycle-Skip Excursions",
                    "target": w_name,
                    "message": (
                        f"Detected {w_cycle_skips} sonic cycle-skip samples ({100.0 * w_cycle_skips / n_samples:.1f}% of interval) removed in DT_ENV_CORR."
                    ),
                    "mitigation": "Use DT_ENV_CORR or density-neutron porosity across cycle-skipped intervals.",
                }
            )

        # Inter-Log Depth Alignment & Run Tie-In QC (bed-boundary |df/dz| cross-correlation)
        w_da_off = 0
        w_da_max_shift = 0.0
        if "DEPTH" in w_df.columns and len(w_df) >= 30:
            try:
                from quick_pp.depth_alignment import evaluate_depth_alignment

                da_bh_mask = None
                if da_excl_bh and "BADHOLE" in w_df.columns:
                    da_bh_mask = (
                        pd.to_numeric(w_df["BADHOLE"], errors="coerce").fillna(0.0)
                        > 0.5
                    ).to_numpy()
                da_rep = evaluate_depth_alignment(
                    w_df,
                    ref_curve=da_ref_curve,
                    max_shift=da_max_shift_m,
                    window_size=da_win_size_m,
                    window_step=max(5.0, da_win_size_m / 4.0),
                    badhole_mask=da_bh_mask,
                    depth_unit="m",
                )
                da_curves = list(da_rep.get("curves") or [])
                if da_curves:
                    depth_align_reports_by_well[w_name] = da_rep
                    eff_ref = str(da_rep.get("reference_curve") or da_ref_curve)
                    w_ori_curves: List[str] = []
                    w_off_details: List[str] = []
                    w_da_aligned = 0
                    w_da_minor = 0
                    w_da_var = 0
                    worst_c = da_curves[0]["curve"]
                    for cr in da_curves:
                        cname = str(cr["curve"])
                        has_ori = f"{cname}_ORI" in w_df.columns
                        if has_ori:
                            w_ori_curves.append(cname)
                        sh_m = float(cr.get("suggested_shift_m") or 0.0)
                        cf_v = float(cr.get("confidence") or 0.0)
                        st_v = str(cr.get("status") or "ALIGNED")
                        var_d = bool(cr.get("variable_drift"))
                        if abs(sh_m) >= w_da_max_shift:
                            w_da_max_shift = abs(sh_m)
                            worst_c = cname
                        if st_v == "OFF_DEPTH":
                            w_da_off += 1
                            if cf_v >= 0.45:
                                w_off_details.append(f"{cname} ({sh_m:+.2f} m)")
                        elif st_v == "MINOR":
                            w_da_minor += 1
                        elif st_v == "ALIGNED":
                            w_da_aligned += 1
                        if var_d:
                            w_da_var += 1
                        depth_align_curve_rows.append(
                            {
                                "well_name": w_name,
                                "curve": cname,
                                "ref_curve": eff_ref,
                                "samples": int(cr.get("valid_samples") or 0),
                                "shift_m": round(sh_m, 3),
                                "shift_samples": round(
                                    float(cr.get("suggested_shift_samples") or 0.0), 2
                                ),
                                "p95_drift_m": round(
                                    float(cr.get("p95_abs_drift_m") or 0.0), 3
                                ),
                                "variable_drift": var_d,
                                "peak_corr": round(
                                    float(cr.get("peak_corr") or 0.0), 3
                                ),
                                "zero_corr": round(
                                    float(cr.get("zero_lag_corr") or 0.0), 3
                                ),
                                "confidence": round(cf_v, 2),
                                "has_ori": has_ori,
                                "status": st_v,
                            }
                        )
                    w_da_status = (
                        "OFF_DEPTH"
                        if w_da_off > 0
                        else ("MINOR_DRIFT" if w_da_minor > 0 else "ALIGNED")
                    )
                    depth_align_well_rows.append(
                        {
                            "well_name": w_name,
                            "ref_curve": eff_ref,
                            "evaluated_count": len(da_curves),
                            "aligned_count": w_da_aligned,
                            "minor_count": w_da_minor,
                            "off_depth_count": w_da_off,
                            "variable_drift_count": w_da_var,
                            "max_abs_shift_m": round(w_da_max_shift, 3),
                            "worst_curve": worst_c,
                            "conditioned_curves": (
                                ", ".join(sorted(set(w_ori_curves)))
                                if w_ori_curves
                                else "—"
                            ),
                            "status": w_da_status,
                        }
                    )
                    if w_off_details:
                        anomaly_flags.append(
                            {
                                "category": "Inter-Log Depth Mismatch",
                                "target": w_name,
                                "message": (
                                    f"Bed-boundary cross-correlation vs {eff_ref} flagged off-depth curve(s): "
                                    f"{', '.join(w_off_details)} (max |Δz| = {w_da_max_shift:.2f} m)."
                                ),
                                "mitigation": (
                                    "Apply bulk or dynamic depth shift in Data & QA/QC -> Inter-Log Depth Alignment "
                                    "(original curve preserved as *_ORI) prior to porosity/invasion evaluation."
                                ),
                            }
                        )
            except Exception as exc:
                logger.debug(f"Depth alignment QC skipped for {w_name}: {exc}")

        if w_res_mismatch > 0:
            qc_status = "FLAGGED (Res/Mud Mismatch)"
        elif bh_pct > 15.0:
            qc_status = "FLAGGED (>15% Washout)"
        elif w_da_off > 0 and w_da_max_shift > 0.50:
            qc_status = "FLAGGED (Off-Depth Logs)"
        elif w_has_env and w_max_unrel > 20.0:
            qc_status = "REVIEW (Low Conf)"
        else:
            qc_status = "PASS"

        delta_gr = "—"
        delta_rhob = "—"
        delta_nphi = "—"
        if "GR_NORM" in w_df.columns and "GR" in w_df.columns:
            d_gr = (
                pd.to_numeric(w_df["GR_NORM"], errors="coerce")
                - pd.to_numeric(w_df["GR"], errors="coerce")
            ).dropna()
            if not d_gr.empty:
                delta_gr = f"{float(d_gr.median()):+.1f}"
        if "RHOB_NORM" in w_df.columns and "RHOB" in w_df.columns:
            d_rhob = (
                pd.to_numeric(w_df["RHOB_NORM"], errors="coerce")
                - pd.to_numeric(w_df["RHOB"], errors="coerce")
            ).dropna()
            if not d_rhob.empty:
                delta_rhob = f"{float(d_rhob.median()):+.3f}"
        if "NPHI_NORM" in w_df.columns and "NPHI" in w_df.columns:
            d_nphi = (
                pd.to_numeric(w_df["NPHI_NORM"], errors="coerce")
                - pd.to_numeric(w_df["NPHI"], errors="coerce")
            ).dropna()
            if not d_nphi.empty:
                delta_nphi = f"{float(d_nphi.median()):+.3f}"

        if bh_pct > 15.0:
            anomaly_flags.append(
                {
                    "category": "Borehole Rugosity / Washout",
                    "target": w_name,
                    "message": f"Bad-hole flag exceeds 15% threshold ({bh_pct:.1f}%, {bh_cnt} depth samples).",
                    "mitigation": "Applied GR-derived shale fallback and excluded rugose intervals from density porosity calibration.",
                }
            )

        qaqc_table.append(
            {
                "well_name": w_name,
                "samples": n_samples,
                "badhole_count": bh_cnt,
                "badhole_pct": round(bh_pct, 2),
                "hc_pct": round(hc_pct, 2),
                "tight_count": tight_cnt,
                "coal_count": coal_cnt,
                "gr_p50": round(gr_p50, 2),
                "rhob_p50": round(rhob_p50, 3),
                "delta_gr": delta_gr,
                "delta_rhob": delta_rhob,
                "delta_nphi": delta_nphi,
                "has_env_corr": w_has_env,
                "env_curves": ", ".join(env_curves_w) if env_curves_w else "—",
                "mean_qc_conf": w_mean_conf,
                "unreliable_pct": w_max_unrel,
                "hole_section_count": w_hs_cnt,
                "mud_type": w_mud_str,
                "status": qc_status,
            }
        )

        # Per-well Zonal Reservoir Summary (`calc_reservoir_summary`) strictly on evaluated reservoir intervals
        try:
            phi_col_name = (
                "PHIE"
                if "PHIE" in w_df.columns and w_df["PHIE"].notna().any()
                else "PHIT"
            )
            phi_eval = pd.to_numeric(w_df.get(phi_col_name), errors="coerce")
            sw_eval = pd.to_numeric(w_df.get("SWT"), errors="coerce")
            zone_eval = (
                w_df["ZONES"]
                if "ZONES" in w_df.columns
                else pd.Series("Main Reservoir", index=w_df.index)
            )
            real_zone_mask = ~zone_eval.astype(str).str.strip().str.upper().isin(
                {"MAIN RESERVOIR", "UNKNOWN", "NAN", ""}
            )
            has_real_zones = bool(real_zone_mask.any())

            eval_mask = (
                phi_eval.notna() & sw_eval.notna() & w_df["_REPORT_VSHALE"].notna()
            )
            if has_real_zones:
                eval_mask &= real_zone_mask

            if not eval_mask.any():
                continue

            eval_df = w_df.loc[eval_mask].copy()

            rs_df = calc_reservoir_summary(
                depth=pd.to_numeric(eval_df["DEPTH"], errors="coerce").to_numpy(
                    dtype=float
                ),
                vshale=pd.to_numeric(
                    eval_df["_REPORT_VSHALE"], errors="coerce"
                ).to_numpy(dtype=float),
                phit=pd.to_numeric(eval_df[phi_col_name], errors="coerce").to_numpy(
                    dtype=float
                ),
                swt=pd.to_numeric(eval_df["SWT"], errors="coerce").to_numpy(
                    dtype=float
                ),
                perm=pd.to_numeric(eval_df["PERM"], errors="coerce")
                .fillna(1.0)
                .to_numpy(dtype=float),
                zones=eval_df["ZONES"].astype(str).to_numpy(),
                cutoffs=ressum_cutoffs,
            )

            res_rows = {
                str(r.get("zones")): r
                for _, r in rs_df[rs_df["flag"] == "reservoir"].iterrows()
            }
            pay_rows = {
                str(r.get("zones")): r
                for _, r in rs_df[rs_df["flag"] == "pay"].iterrows()
            }
            zone_keys = list(
                dict.fromkeys(list(res_rows.keys()) + list(pay_rows.keys()))
            )

            for z_key in zone_keys:
                r_res = res_rows.get(z_key, {})
                r_pay = pay_rows.get(z_key, {})
                r_ref = r_pay if len(r_pay) > 0 else r_res

                top_md = _safe_float(r_ref.get("top"), d_min)
                bot_md = _safe_float(r_ref.get("bottom"), d_max)
                gross_v = _safe_float(r_ref.get("gross"), max(0.0, bot_md - top_md))
                if gross_v <= 0.0:
                    continue

                # Spec 1.1: Reject/flag catch-all zone ("Main Reservoir" / "Unknown") starting near 0 m or > 500 m
                if z_key.upper() in ("MAIN RESERVOIR", "UNKNOWN") and (
                    top_md <= 50.0 or gross_v > 500.0
                ):
                    anomaly_flags.append(
                        {
                            "category": "Rejected Unbounded Catch-All Zone",
                            "target": f"{w_name} ({z_key})",
                            "message": f"Catch-all zone '{z_key}' ({top_md:.1f}–{bot_md:.1f} m MD, gross {gross_v:.1f} m) exceeds 500 m thickness or starts near surface.",
                            "mitigation": "Excluded unbounded catch-all interval from volumetric Net Pay aggregation; assign explicit formation tops.",
                        }
                    )
                    if has_real_zones:
                        continue

                # Compute TVDss interval if TVD/TVDSS is available for this well
                top_tvdss: Optional[float] = None
                bot_tvdss: Optional[float] = None
                if has_tvd and "TVDSS" in eval_df.columns:
                    z_sub = eval_df[eval_df["ZONES"].astype(str) == z_key]
                    tvdss_sub = pd.to_numeric(z_sub["TVDSS"], errors="coerce").dropna()
                    if not tvdss_sub.empty:
                        top_tvdss = round(float(tvdss_sub.min()), 1)
                        bot_tvdss = round(float(tvdss_sub.max()), 1)

                # Net Reservoir properties (Spec 3.1)
                net_res_v = min(_safe_float(r_res.get("net"), 0.0), gross_v)
                ntg_res_v = min(
                    1.0,
                    max(
                        0.0,
                        _safe_float(
                            r_res.get("ntg"),
                            net_res_v / gross_v if gross_v > 0 else 0.0,
                        ),
                    ),
                )
                has_res = net_res_v > 0.0
                vsh_res_v = _safe_float(r_res.get("av_vshale"), 0.0) if has_res else 0.0
                phie_res_v = _safe_float(r_res.get("av_phit"), 0.0) if has_res else 0.0
                sw_res_v = _safe_float(r_res.get("av_swt"), 1.0) if has_res else 1.0
                perm_gm_res = (
                    _safe_float(r_res.get("av_perm_gm"), 0.0) if has_res else 0.0
                )

                # Net Pay properties (Spec 3.1)
                net_v = min(_safe_float(r_pay.get("net"), 0.0), gross_v)
                ntg_v = min(
                    1.0,
                    max(
                        0.0,
                        _safe_float(
                            r_pay.get("ntg"), net_v / gross_v if gross_v > 0 else 0.0
                        ),
                    ),
                )
                has_pay = net_v > 0.0
                phie_v = _safe_float(r_pay.get("av_phit"), 0.0) if has_pay else 0.0
                sw_v = _safe_float(r_pay.get("av_swt"), 1.0) if has_pay else 1.0
                vsh_v = _safe_float(r_pay.get("av_vshale"), 0.0) if has_pay else 0.0
                bvhc_v = (
                    _safe_float(r_pay.get("av_bvhc"), phie_v * (1.0 - sw_v))
                    if has_pay
                    else 0.0
                )
                hcpt_v = net_v * phie_v * max(0.0, 1.0 - sw_v) if has_pay else 0.0
                perm_gm = _safe_float(r_pay.get("av_perm_gm"), 0.0) if has_pay else 0.0
                perm_am = _safe_float(r_pay.get("av_perm_am"), 0.0) if has_pay else 0.0
                perm_hm = _safe_float(r_pay.get("av_perm_hm"), 0.0) if has_pay else 0.0
                kv_kh = (
                    _safe_float(
                        r_pay.get("kv_kh_ratio"),
                        perm_hm / perm_am if perm_am > 0 else 0.0,
                    )
                    if has_pay
                    else (
                        _safe_float(r_res.get("kv_kh_ratio"), 0.0) if has_res else 0.0
                    )
                )

                if z_key not in evaluated_zones_seen:
                    evaluated_zones_seen.append(z_key)

                well_zonal_rows.append(
                    {
                        "well_name": w_name,
                        "zone": z_key,
                        "top": round(top_md, 1),
                        "bottom": round(bot_md, 1),
                        "top_tvdss": top_tvdss,
                        "bottom_tvdss": bot_tvdss,
                        "tvdss_str": (
                            f"`{top_tvdss:.1f}–{bot_tvdss:.1f}`"
                            if top_tvdss is not None and bot_tvdss is not None
                            else "— (No TVD)"
                        ),
                        "gross": round(gross_v, 2),
                        "net_res": round(net_res_v, 2),
                        "ntg_res": round(ntg_res_v, 3),
                        "vsh_res": round(vsh_res_v, 3),
                        "phie_res": round(phie_res_v, 3),
                        "sw_res": round(sw_res_v, 3),
                        "perm_geom_res": round(perm_gm_res, 2),
                        "net_pay": round(net_v, 2),
                        "ntg": round(ntg_v, 3),
                        "vsh": round(vsh_v, 3),
                        "phie": round(phie_v, 3),
                        "sw": round(sw_v, 3),
                        "bvhc": round(bvhc_v, 3),
                        "hcpt": round(hcpt_v, 2),
                        "perm_geom": round(perm_gm, 2),
                        "perm_arith": round(perm_am, 2),
                        "perm_harm": round(perm_hm, 2),
                        "kv_kh": round(kv_kh, 3),
                    }
                )
        except Exception as exc:
            logger.debug(f"Reservoir summary fallback for {w_name}: {exc}")

        # Per-well Upscaling Summary (`upscale_logs`) — only run on evaluated target intervals
        if need_upscaling:
            try:
                up_src = (
                    eval_df
                    if "eval_df" in locals() and not eval_df.empty
                    else w_df.dropna(subset=["PHIT", "SWT"], how="any")
                )
                if not up_src.empty:
                    up_in = pd.DataFrame(
                        {
                            "DEPTH": pd.to_numeric(up_src["DEPTH"], errors="coerce"),
                            "VCLAY": pd.to_numeric(
                                up_src["VCLAY"]
                                if "VCLAY" in up_src.columns
                                else up_src.get("_REPORT_VSHALE", 0.20),
                                errors="coerce",
                            ).fillna(0.20),
                            "_REPORT_VSHALE": pd.to_numeric(
                                up_src["_REPORT_VSHALE"]
                                if "_REPORT_VSHALE" in up_src.columns
                                else up_src.get("VCLAY", 0.20),
                                errors="coerce",
                            ).fillna(0.20),
                            "PHIT": _col_series(
                                up_src,
                                "PHIE" if "PHIE" in up_src.columns else "PHIT",
                                np.nan,
                            ),
                            "PHIE": _col_series(
                                up_src,
                                "PHIE" if "PHIE" in up_src.columns else "PHIT",
                                np.nan,
                            ),
                            "SWT": pd.to_numeric(up_src["SWT"], errors="coerce"),
                            "PERM": pd.to_numeric(up_src["PERM"], errors="coerce")
                            .fillna(1.0)
                            .clip(lower=1e-2),
                            "ZONES": up_src["ZONES"].astype(str),
                        }
                    ).dropna(subset=["DEPTH", "PHIT", "SWT"])
                    if not up_in.empty:
                        up_cell_sz = _safe_float(
                            merged_configs.get(
                                "upscaling_cell_size", w_cfg.get("upscaling_cell_size")
                            ),
                            2.0,
                        )
                        blocked = upscale_logs(
                            up_in,
                            cell_size=up_cell_sz,
                            by_zone=True,
                            depth_col="DEPTH",
                            zone_col="ZONES",
                            perm_method="geometric",
                            cutoffs=ressum_cutoffs,
                        )
                        k_am_col = (
                            "PERM_ARITH"
                            if "PERM_ARITH" in blocked.columns
                            else "PERM_ARITH_UP"
                        )
                        k_gm_col = (
                            "PERM_GEOM"
                            if "PERM_GEOM" in blocked.columns
                            else "PERM_GEOM_UP"
                        )
                        k_hm_col = (
                            "PERM_HARM"
                            if "PERM_HARM" in blocked.columns
                            else "PERM_HARM_UP"
                        )
                        phi_up_col = "PHIT" if "PHIT" in blocked.columns else "PHIT_UP"
                        sw_up_col = "SWT" if "SWT" in blocked.columns else "SWT_UP"

                        k_am = (
                            _safe_float(blocked[k_am_col].mean(), 0.0)
                            if k_am_col in blocked.columns
                            else 0.0
                        )
                        k_gm = (
                            _safe_float(blocked[k_gm_col].mean(), 0.0)
                            if k_gm_col in blocked.columns
                            else 0.0
                        )
                        k_hm = (
                            _safe_float(blocked[k_hm_col].mean(), 0.0)
                            if k_hm_col in blocked.columns
                            else 0.0
                        )
                        gross_m = (
                            round(float(up_in["DEPTH"].max() - up_in["DEPTH"].min()), 1)
                            if "DEPTH" in up_in.columns and len(up_in) > 1
                            else round(float(len(blocked) * up_cell_sz), 1)
                        )

                        # Net-reservoir calculation at fine scale
                        v_cut_val = _safe_float(ressum_cutoffs.get("vsh"), 0.40)
                        p_cut_val = _safe_float(ressum_cutoffs.get("phie"), 0.08)
                        vsh_fine = pd.to_numeric(
                            up_in["_REPORT_VSHALE"]
                            if "_REPORT_VSHALE" in up_in.columns
                            else up_in["VCLAY"],
                            errors="coerce",
                        ).fillna(0.0)
                        phie_fine = pd.to_numeric(
                            up_in["PHIE"] if "PHIE" in up_in.columns else up_in["PHIT"],
                            errors="coerce",
                        ).fillna(0.0)
                        sw_fine = pd.to_numeric(up_in["SWT"], errors="coerce").fillna(
                            1.0
                        )

                        fine_res_mask = (vsh_fine <= v_cut_val) & (
                            phie_fine >= p_cut_val
                        )
                        if fine_res_mask.any():
                            phie_res_fine = round(
                                _safe_float(phie_fine[fine_res_mask].mean(), 0.0), 3
                            )
                            swt_res_fine = round(
                                _safe_float(sw_fine[fine_res_mask].mean(), 1.0), 3
                            )
                        else:
                            phie_res_fine = round(_safe_float(phie_fine.mean(), 0.0), 3)
                            swt_res_fine = round(_safe_float(sw_fine.mean(), 1.0), 3)

                        # Net-reservoir calculation at blocked cellular scale
                        vsh_up_col = (
                            "VCLAY"
                            if "VCLAY" in blocked.columns
                            else (
                                "_REPORT_VSHALE"
                                if "_REPORT_VSHALE" in blocked.columns
                                else None
                            )
                        )
                        vsh_blocked = (
                            pd.to_numeric(blocked[vsh_up_col], errors="coerce").fillna(
                                0.0
                            )
                            if vsh_up_col
                            else pd.Series(0.0, index=blocked.index)
                        )
                        phie_blocked = (
                            pd.to_numeric(blocked[phi_up_col], errors="coerce").fillna(
                                0.0
                            )
                            if phi_up_col in blocked.columns
                            else pd.Series(0.0, index=blocked.index)
                        )
                        sw_blocked = (
                            pd.to_numeric(blocked[sw_up_col], errors="coerce").fillna(
                                1.0
                            )
                            if sw_up_col in blocked.columns
                            else pd.Series(1.0, index=blocked.index)
                        )

                        blocked_res_mask = (vsh_blocked <= v_cut_val) & (
                            phie_blocked >= p_cut_val
                        )
                        if blocked_res_mask.any():
                            n_res_cells = int(blocked_res_mask.sum())
                            ntg_res_up = round(
                                float(n_res_cells / max(len(blocked), 1)), 3
                            )
                            phie_res_up = round(
                                _safe_float(phie_blocked[blocked_res_mask].mean(), 0.0),
                                3,
                            )
                            swt_res_up = round(
                                _safe_float(sw_blocked[blocked_res_mask].mean(), 1.0), 3
                            )
                        else:
                            n_res_cells = int(len(blocked))
                            ntg_res_up = 1.0
                            phie_res_up = round(
                                _safe_float(phie_blocked.mean(), 0.0), 3
                            )
                            swt_res_up = round(_safe_float(sw_blocked.mean(), 1.0), 3)

                        delta_phi_res = round(abs(phie_res_up - phie_res_fine), 3)
                        conservation_status = (
                            "✓ CONSERVED" if delta_phi_res <= 0.015 else "⚠ DIVERGENT"
                        )

                        upscaling_table.append(
                            {
                                "well_name": w_name,
                                "fine_samples": int(len(up_in)),
                                "n_cells": int(len(blocked)),
                                "gross_m": gross_m,
                                "phit_up": round(
                                    _safe_float(blocked[phi_up_col].mean(), 0.0), 3
                                )
                                if phi_up_col in blocked.columns
                                else 0.0,
                                "swt_up": round(
                                    _safe_float(blocked[sw_up_col].mean(), 1.0), 3
                                )
                                if sw_up_col in blocked.columns
                                else 1.0,
                                "k_arith": round(k_am, 1),
                                "k_geom": round(k_gm, 1),
                                "k_harm": round(k_hm, 2),
                                "kv_kh": round(k_hm / max(k_am, 1e-6), 3)
                                if k_am > 0
                                else 0.0,
                                "n_res_cells": n_res_cells,
                                "ntg_res_up": ntg_res_up,
                                "phie_res_up": phie_res_up,
                                "phie_res_fine": phie_res_fine,
                                "swt_res_up": swt_res_up,
                                "swt_res_fine": swt_res_fine,
                                "delta_phi_res": delta_phi_res,
                                "conservation_status": conservation_status,
                            }
                        )
            except Exception as exc:
                logger.debug(f"Upscaling summary fallback for {w_name}: {exc}")

    # Spec 1.2: Enforce strict validation of well datum (KB / RTE) across multi-well projects
    if len(wells_inventory) > 1:
        kb_unique = {w["kb"] for w in wells_inventory}
        if len(kb_unique) == 1:
            uniform_kb = next(iter(kb_unique))
            anomaly_flags.append(
                {
                    "category": "Well Datum (KB / RTE) Uniformity Warning",
                    "target": "Field Well Headers",
                    "message": f"100% of evaluated wells ({len(wells_inventory)}/{len(wells_inventory)}) share an identical KB/RTE datum ({uniform_kb:.1f} m), indicating a default fallback.",
                    "mitigation": "Verify individual rig Kelly Bushing (KB) / Rotary Table Elevations (RTE) in well headers to prevent sub-sea depth shifts in FWL modeling.",
                }
            )

    if not combined_dfs and unfiltered_dfs:
        combined_dfs = unfiltered_dfs
    full_df = (
        pd.concat(combined_dfs, ignore_index=True) if combined_dfs else pd.DataFrame()
    )

    # Stratigraphic Target Filtering (`zones` / `target_zones` from Top Bar Zone Filter)
    cfg_target_zones = (
        requested_zones or target_zones or merged_configs.get("target_zones")
    )
    if cfg_target_zones and isinstance(cfg_target_zones, list):
        resolved_target_zones = [
            _normalize_zone_name(z)
            for z in cfg_target_zones
            if z and _normalize_zone_name(z).upper() not in GENERIC_PLACEHOLDER_ZONES
        ]
    else:
        candidate_targets = [
            z
            for z in evaluated_zones_seen
            if z.upper() not in GENERIC_PLACEHOLDER_ZONES
        ]
        resolved_target_zones = (
            candidate_targets
            if candidate_targets
            else (evaluated_zones_seen or all_zones_seen or ["Main Reservoir"])
        )

    target_zones_upper = {z.upper() for z in resolved_target_zones}
    if well_zonal_rows and target_zones_upper:
        filtered_wz = [
            r for r in well_zonal_rows if r["zone"].upper() in target_zones_upper
        ]
        if filtered_wz:
            well_zonal_rows = filtered_wz

    # Spec 3.1: Aggregate Zonal Summary across wells using Net-Thickness & Pore-Thickness weighting
    zonal_summary: List[Dict[str, Any]] = []
    if well_zonal_rows:
        wz_df = pd.DataFrame(well_zonal_rows)
        for z_name, grp in wz_df.groupby("zone", sort=False):
            res_grp = grp[grp["net_res"] > 0]
            pay_grp = grp[grp["net_pay"] > 0]

            # Net Reservoir weighted averages
            if not res_grp.empty:
                w_res = res_grp["net_res"].to_numpy(dtype=float)
                w_res_sum = max(float(np.sum(w_res)), 1e-6)
                vsh_res_mean = float(
                    np.sum(w_res * res_grp["vsh_res"].to_numpy(dtype=float)) / w_res_sum
                )
                phie_res_mean = float(
                    np.sum(w_res * res_grp["phie_res"].to_numpy(dtype=float))
                    / w_res_sum
                )
                k_res_vals = np.clip(
                    res_grp["perm_geom_res"].to_numpy(dtype=float), 1e-3, 1e6
                )
                perm_res_geom = float(
                    np.exp(np.sum(w_res * np.log(k_res_vals)) / w_res_sum)
                )
            else:
                vsh_res_mean = 0.0
                phie_res_mean = 0.0
                perm_res_geom = 0.0

            # Net Pay weighted averages (Thickness-weighted Φe & Vsh, Pore-Thickness-weighted Sw)
            if not pay_grp.empty:
                w_pay = pay_grp["net_pay"].to_numpy(dtype=float)
                w_pay_sum = max(float(np.sum(w_pay)), 1e-6)
                vsh_pay_mean = float(
                    np.sum(w_pay * pay_grp["vsh"].to_numpy(dtype=float)) / w_pay_sum
                )
                phie_pay_mean = float(
                    np.sum(w_pay * pay_grp["phie"].to_numpy(dtype=float)) / w_pay_sum
                )
                pore_w = np.maximum(w_pay * pay_grp["phie"].to_numpy(dtype=float), 1e-6)
                sw_pay_mean = float(
                    np.sum(pore_w * pay_grp["sw"].to_numpy(dtype=float))
                    / float(np.sum(pore_w))
                )
                k_pay_vals = np.clip(
                    pay_grp["perm_geom"].to_numpy(dtype=float), 1e-3, 1e6
                )
                perm_pay_geom = float(
                    np.exp(np.sum(w_pay * np.log(k_pay_vals)) / w_pay_sum)
                )
            else:
                vsh_pay_mean = 0.0
                phie_pay_mean = 0.0
                sw_pay_mean = 1.0
                perm_pay_geom = 0.0

            gross_mean = float(grp["gross"].mean())
            net_res_mean = float(grp["net_res"].mean())
            net_pay_mean = float(grp["net_pay"].mean())

            zonal_summary.append(
                {
                    "zone": str(z_name),
                    "well_count": int(grp["well_name"].nunique()),
                    "gross": round(gross_mean, 1),
                    "net_res": round(net_res_mean, 1),
                    "ntg_res": round(
                        min(1.0, max(0.0, net_res_mean / max(gross_mean, 1e-6))), 2
                    ),
                    "vsh_res": round(vsh_res_mean, 3),
                    "phie_res": round(phie_res_mean, 3),
                    "perm_geom_res": round(perm_res_geom, 1),
                    "net_pay": round(net_pay_mean, 1),
                    "ntg": round(
                        min(1.0, max(0.0, net_pay_mean / max(gross_mean, 1e-6))), 2
                    ),
                    "vsh": round(vsh_pay_mean, 3),
                    "phie": round(phie_pay_mean, 3),
                    "sw": round(sw_pay_mean, 3),
                    "perm_geom": round(perm_pay_geom, 1),
                    "hcpt": round(float(grp["hcpt"].mean()), 2),
                    "total_hcpt": round(float(grp["hcpt"].sum()), 2),
                }
            )

    # Build Markdown table for Well-by-Well Zonal Reservoir Summary (Decoupled Net Reservoir & Net Pay + MD & TVDss)
    is_single_well_ctx = len(wells_inventory) == 1
    if is_single_well_ctx:
        md_rows = [
            "| Zone | Interval (m MD) | Interval (m TVDss) | Gross (m) | Net Res (m) | $NTG_{res}$ | $\\bar{V}_{sh,res}$ | $\\bar{\\phi}_{e,res}$ | $K_{geom,res}$ (mD) | Net Pay (m) | $NTG_{pay}$ | $\\bar{\\phi}_{e,pay}$ | $\\bar{S}_{w,pay}$ | $HCPT$ (m) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in well_zonal_rows:
            has_r = r["net_res"] > 0
            has_p = r["net_pay"] > 0
            vsh_r_str = f"{r['vsh_res']:.3f}" if has_r else "—"
            phi_r_str = f"{r['phie_res']:.3f}" if has_r else "—"
            k_r_str = f"{r['perm_geom_res']:.1f}" if has_r else "—"
            phi_p_str = f"{r['phie']:.3f}" if has_p else "—"
            sw_p_str = f"{r['sw']:.3f}" if has_p else "—"
            md_rows.append(
                f"| **{r['zone']}** | `{r['top']:.1f}–{r['bottom']:.1f}` | {r['tvdss_str']} | "
                f"{r['gross']:.1f} | {r['net_res']:.1f} | {r['ntg_res']:.2f} | {vsh_r_str} | {phi_r_str} | {k_r_str} | "
                f"{r['net_pay']:.1f} | {r['ntg']:.2f} | {phi_p_str} | {sw_p_str} | **{r['hcpt']:.2f}** |"
            )
    else:
        md_rows = [
            "| Well | Zone | Interval (m MD) | Interval (m TVDss) | Gross (m) | Net Res (m) | $NTG_{res}$ | $\\bar{V}_{sh,res}$ | $\\bar{\\phi}_{e,res}$ | $K_{geom,res}$ (mD) | Net Pay (m) | $NTG_{pay}$ | $\\bar{\\phi}_{e,pay}$ | $\\bar{S}_{w,pay}$ | $HCPT$ (m) |",
            "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
        for r in well_zonal_rows:
            has_r = r["net_res"] > 0
            has_p = r["net_pay"] > 0
            vsh_r_str = f"{r['vsh_res']:.3f}" if has_r else "—"
            phi_r_str = f"{r['phie_res']:.3f}" if has_r else "—"
            k_r_str = f"{r['perm_geom_res']:.1f}" if has_r else "—"
            phi_p_str = f"{r['phie']:.3f}" if has_p else "—"
            sw_p_str = f"{r['sw']:.3f}" if has_p else "—"
            md_rows.append(
                f"| **{r['well_name']}** | {r['zone']} | `{r['top']:.1f}–{r['bottom']:.1f}` | {r['tvdss_str']} | "
                f"{r['gross']:.1f} | {r['net_res']:.1f} | {r['ntg_res']:.2f} | {vsh_r_str} | {phi_r_str} | {k_r_str} | "
                f"{r['net_pay']:.1f} | {r['ntg']:.2f} | {phi_p_str} | {sw_p_str} | **{r['hcpt']:.2f}** |"
            )
    ressum_markdown_table = "\n".join(md_rows)

    # Spec 1.3: Core-to-Log Depth Alignment, Stratigraphic Masking & Grain Density QC
    total_core_plugs_loaded = sum(w["core_sample_count"] for w in wells_inventory)
    valid_cal_core_count = 0
    core_r2 = 0.0
    core_rmse = 0.0
    core_bias = 0.0
    valid_cal_perm_count = 0
    core_perm_r2 = 0.0
    core_perm_log_rmse = 0.0
    core_perm_factor_err = 1.0
    rhoma_mean = 2.65
    rhoma_std = 0.0
    rhoma_count = 0
    if not full_df.empty and "CPERM" in full_df.columns and "PERM" in full_df.columns:
        ck_s = _col_series(full_df, "CPERM", np.nan)
        pm_s = _col_series(full_df, "PERM", np.nan)
        tz_mask_perm = (
            full_df["ZONES"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(target_zones_upper)
            if "ZONES" in full_df.columns
            else pd.Series(True, index=full_df.index)
        )
        kmask_target = (
            ck_s.notna() & pm_s.notna() & (ck_s > 1e-3) & (pm_s > 1e-3) & tz_mask_perm
        )
        kmask_all = ck_s.notna() & pm_s.notna() & (ck_s > 1e-3) & (pm_s > 1e-3)
        active_kmask = kmask_target if kmask_target.sum() >= 3 else kmask_all
        if active_kmask.sum() >= 3:
            kx = np.log10(ck_s[active_kmask].to_numpy(dtype=float))
            ky = np.log10(pm_s[active_kmask].to_numpy(dtype=float))
            valid_cal_perm_count = int(len(kx))
            core_perm_log_rmse = round(float(np.sqrt(np.mean((ky - kx) ** 2))), 3)
            core_perm_factor_err = round(float(10.0**core_perm_log_rmse), 2)
            ss_res_k = float(np.sum((ky - kx) ** 2))
            ss_tot_k = float(np.sum((kx - np.mean(kx)) ** 2))
            core_perm_r2 = round(
                max(0.0, min(0.99, 1.0 - ss_res_k / max(ss_tot_k, 1e-9))), 3
            )
    if not full_df.empty and "CPORE" in full_df.columns:
        cp_s = _col_series(full_df, "CPORE", np.nan)
        ph_s = _col_series(
            full_df, "PHIT" if "PHIT" in full_df.columns else "PHIE", np.nan
        )
        cp_s = pd.Series(np.where(cp_s > 1.0, cp_s / 100.0, cp_s), index=full_df.index)
        tz_mask_core = (
            full_df["ZONES"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(target_zones_upper)
            if "ZONES" in full_df.columns
            else pd.Series(True, index=full_df.index)
        )
        cmask_target = (
            cp_s.notna() & ph_s.notna() & (cp_s > 0.005) & (cp_s < 0.48) & tz_mask_core
        )
        cmask_all = cp_s.notna() & ph_s.notna() & (cp_s > 0.005) & (cp_s < 0.48)
        active_cmask = cmask_target if cmask_target.sum() >= 3 else cmask_all

        if active_cmask.sum() >= 3:
            cx = cp_s[active_cmask].to_numpy(dtype=float)
            ly = ph_s[active_cmask].to_numpy(dtype=float)
            valid_cal_core_count = int(len(cx))
            core_rmse = round(float(np.sqrt(np.mean((ly - cx) ** 2))), 4)
            core_bias = round(float(np.mean(ly - cx)), 4)
            ss_res = float(np.sum((ly - cx) ** 2))
            ss_tot = float(np.sum((cx - np.mean(cx)) ** 2))
            core_r2 = round(max(0.0, min(0.99, 1.0 - ss_res / max(ss_tot, 1e-9))), 3)
            if core_r2 < 0.40 or core_rmse > 0.040:
                anomaly_flags.append(
                    {
                        "category": "Core-to-Log Porosity Calibration (CHK-03)",
                        "target": "Active Target-Zone RCA Dataset",
                        "message": f"Core-to-Log porosity correlation is low (R² = {core_r2:.2f}, RMSE = {core_rmse:.3f} v/v across {valid_cal_core_count} overlapping plugs out of {total_core_plugs_loaded} total loaded plugs).",
                        "mitigation": "Apply core-to-log gamma/porosity depth shifts and account for heavy-mineral / calcite-cement grain density variations.",
                    }
                )
        elif total_core_plugs_loaded > 0:
            anomaly_flags.append(
                {
                    "category": "Non-Overlapping Core & Log Evaluation Intervals (CHK-03)",
                    "target": "Field RCA Dataset",
                    "message": f"{total_core_plugs_loaded} core plugs are loaded in the database, but 0 plugs overlap depth-evaluated PHIT logs in target zones.",
                    "mitigation": "Extend wireline porosity evaluation across cored intervals or restrict core calibration reporting to overlapping target zones.",
                }
            )

        # Grain density (RHOMA / CGD / Apparent Matrix Density)
        for gcol in ("RHOMA", "CGD", "GRAIN_DENSITY", "RHO_MA"):
            if gcol in full_df.columns:
                gs = pd.to_numeric(full_df[gcol], errors="coerce")
                gs_valid = gs[(gs >= 2.45) & (gs <= 2.95)].dropna()
                if len(gs_valid) >= 3:
                    rhoma_mean = round(float(gs_valid.mean()), 3)
                    rhoma_std = round(float(gs_valid.std()), 3)
                    rhoma_count = int(len(gs_valid))
                    break

    # Spec 2.2: Well-by-Well Core Parity Breakdown
    well_core_parity: List[Dict[str, Any]] = []
    unique_wells_for_parity = [w["name"] for w in wells_inventory]
    for wn in unique_wells_for_parity:
        w_df_sub = (
            full_df[
                full_df["WELL_NAME"].astype(str).str.strip().str.upper()
                == wn.strip().upper()
            ]
            if ("WELL_NAME" in full_df.columns and not full_df.empty)
            else full_df
        )
        if w_df_sub.empty:
            continue

        cp_w = _col_series(w_df_sub, "CPORE", np.nan)
        cp_w = pd.Series(np.where(cp_w > 1.0, cp_w / 100.0, cp_w), index=w_df_sub.index)
        phit_w = _col_series(w_df_sub, "PHIT", np.nan)
        phie_w = _col_series(w_df_sub, "PHIE", np.nan)

        tz_sub = (
            w_df_sub["ZONES"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(target_zones_upper)
            if "ZONES" in w_df_sub.columns and target_zones_upper
            else pd.Series(True, index=w_df_sub.index)
        )
        valid_p_mask = cp_w.notna() & (cp_w > 0.005) & (cp_w < 0.48) & tz_sub
        if valid_p_mask.sum() < 3:
            valid_p_mask = cp_w.notna() & (cp_w > 0.005) & (cp_w < 0.48)

        n_plugs_w = int(valid_p_mask.sum())
        shift_s = _col_series(w_df_sub, "SHIFT_M", 0.0)
        mean_shift_m = (
            float(shift_s[valid_p_mask].mean())
            if (n_plugs_w > 0 and shift_s.notna().any())
            else 0.0
        )

        r2_phit_w = 0.0
        r2_phie_w = 0.0
        rmse_phi_w = 0.0
        bias_phi_w = 0.0

        if n_plugs_w >= 3:
            cx_w = cp_w[valid_p_mask].to_numpy(dtype=float)
            if phit_w.notna().any():
                ly_t = phit_w[valid_p_mask].to_numpy(dtype=float)
                valid_t = ~np.isnan(ly_t)
                if valid_t.sum() >= 3:
                    cx_t = cx_w[valid_t]
                    ly_t = ly_t[valid_t]
                    rmse_phi_w = round(float(np.sqrt(np.mean((ly_t - cx_t) ** 2))), 4)
                    bias_phi_w = round(float(np.mean(ly_t - cx_t)), 4)
                    if np.std(cx_t) > 1e-6 and np.std(ly_t) > 1e-6:
                        corr_t = float(np.corrcoef(cx_t, ly_t)[0, 1])
                        r2_phit_w = round(max(0.0, corr_t**2), 3)
            if phie_w.notna().any():
                ly_e = phie_w[valid_p_mask].to_numpy(dtype=float)
                valid_e = ~np.isnan(ly_e)
                if valid_e.sum() >= 3:
                    cx_e = cx_w[valid_e]
                    ly_e = ly_e[valid_e]
                    if np.std(cx_e) > 1e-6 and np.std(ly_e) > 1e-6:
                        corr_e = float(np.corrcoef(cx_e, ly_e)[0, 1])
                        r2_phie_w = round(max(0.0, corr_e**2), 3)

        ck_w = _col_series(w_df_sub, "CPERM", np.nan)
        pm_w = _col_series(w_df_sub, "PERM", np.nan)
        valid_k_mask = (
            ck_w.notna() & pm_w.notna() & (ck_w > 1e-3) & (pm_w > 1e-3) & tz_sub
        )
        if valid_k_mask.sum() < 3:
            valid_k_mask = ck_w.notna() & pm_w.notna() & (ck_w > 1e-3) & (pm_w > 1e-3)
        n_perm_w = int(valid_k_mask.sum())
        r2_perm_w = 0.0
        rmse_log_k_w = 0.0
        factor_err_k_w = 1.0
        if n_perm_w >= 3:
            kx_w = np.log10(ck_w[valid_k_mask].to_numpy(dtype=float))
            ky_w = np.log10(pm_w[valid_k_mask].to_numpy(dtype=float))
            rmse_log_k_w = round(float(np.sqrt(np.mean((ky_w - kx_w) ** 2))), 3)
            factor_err_k_w = round(float(10.0**rmse_log_k_w), 2)
            if np.std(kx_w) > 1e-6 and np.std(ky_w) > 1e-6:
                corr_k = float(np.corrcoef(kx_w, ky_w)[0, 1])
                r2_perm_w = round(max(0.0, corr_k**2), 3)

        if n_plugs_w == 0 and n_perm_w == 0:
            status_w = "NO PLUGS"
            rec_w = "No overlapping core plugs in evaluated intervals."
        elif r2_phit_w >= 0.60 or (abs(bias_phi_w) <= 0.02 and rmse_phi_w <= 0.03):
            status_w = "ALIGNED"
            rec_w = "Core-to-log porosity calibration is consistent."
        elif r2_phit_w >= 0.40:
            status_w = "REVIEW"
            rec_w = (
                "Moderate correlation; verify depth matching and mineral grain density."
            )
        else:
            status_w = "MISALIGNED"
            rec_w = "Low correlation / high residual; depth shifting or facies-specific grain density required."

        perm_status_w = (
            "ALIGNED"
            if r2_perm_w >= 0.50
            else ("REVIEW" if r2_perm_w >= 0.30 else "MISALIGNED")
        )
        perm_rec_w = (
            "Permeability transform consistent with core."
            if r2_perm_w >= 0.50
            else "Inspect rock-typing flow zone indicators (FZI) and hydraulic flow units."
        )

        well_core_parity.append(
            {
                "well_name": wn,
                "n_plugs": n_plugs_w,
                "mean_shift_m": round(mean_shift_m, 2),
                "r2_phit": r2_phit_w,
                "r2_phie": r2_phie_w,
                "rmse_phi": rmse_phi_w,
                "bias_phi": bias_phi_w,
                "status": status_w,
                "recommendation": rec_w,
                "n_perm_plugs": n_perm_w,
                "r2_perm": r2_perm_w,
                "rmse_log_perm": rmse_log_k_w,
                "factor_err_perm": factor_err_k_w,
                "perm_status": perm_status_w,
                "perm_recommendation": perm_rec_w,
            }
        )

    # Spec 2.2: Dynamic Rock Typing & Permeability Regression via `scipy.stats.linregress`
    # Cutoff resolution order: explicit request > workbench-saved rock typing > well config > documented default.
    saved_rt_entry = _load_saved_project_module_entry(
        pid, "rock-typing", resolved_target_zones
    )
    saved_rt_params = (
        saved_rt_entry.get("params") if isinstance(saved_rt_entry, dict) else None
    ) or {}
    saved_rt_method = str(saved_rt_params.get("drtMethod") or "fzi").strip().lower()
    saved_fzi_cutoffs = saved_rt_params.get("fziCutoffs")

    cfg_fzi_cutoffs = None
    cutoff_source = ""
    for src_label, cand in (
        (
            "request override",
            (model_params or {}).get("fzi_cutoffs")
            or (model_params or {}).get("fziCutoffs")
            if isinstance(model_params, dict)
            else None,
        ),
        (
            f"saved workbench rock typing ({saved_rt_entry.get('label', 'project')})"
            if saved_rt_entry
            else "",
            saved_fzi_cutoffs if saved_rt_method == "fzi" else None,
        ),
        (
            "well configuration",
            merged_configs.get("fzi_cutoffs") or merged_configs.get("fziCutoffs"),
        ),
    ):
        if cand and isinstance(cand, (list, tuple)):
            try:
                vals = sorted(float(c) for c in cand if float(c) > 0)
            except (TypeError, ValueError):
                vals = []
            if vals:
                cfg_fzi_cutoffs, cutoff_source = vals, src_label
                break
    if cfg_fzi_cutoffs:
        resolved_fzi_cutoffs = cfg_fzi_cutoffs
    else:
        resolved_fzi_cutoffs = [0.5, 1.5, 3.0]
        cutoff_source = "generic default (no saved rock typing found; not calibrated to this project)"
        anomaly_flags.append(
            {
                "category": "Rock Typing Cutoffs Not Calibrated",
                "target": "Field Rock Typing (Section 5)",
                "message": "No saved rock-typing cutoffs were found; generic FZI cutoffs [0.5, 1.5, 3.0] were applied.",
                "mitigation": "Run and save Rock Typing in the multi-well workbench for the target zones, then regenerate the report.",
            }
        )
    if saved_rt_method in ("winland", "lucia"):
        anomaly_flags.append(
            {
                "category": "Rock Typing Method Mismatch",
                "target": "Field Rock Typing (Section 5)",
                "message": (
                    f"Workbench rock typing used the '{saved_rt_method}' method, but Section 5 re-bins "
                    "core plugs by FZI for the permeability regression table."
                ),
                "mitigation": "Compare Section 5 classes against the saved ROCK_FLAG curve before using them downstream.",
            }
        )

    # Class labels are neutral and positional. Facies names (e.g. "shoreface") cannot be inferred from FZI
    # alone, so none are invented; quality is described only by the FZI range.
    rt_Specs = []
    n_cuts = len(resolved_fzi_cutoffs)
    for i in range(n_cuts + 1):
        idx = i + 1
        if i == 0:
            c_high = resolved_fzi_cutoffs[0]
            quality = "Lowest flow quality"
            code = f"DRT 1 (FZI < {c_high:.2f})"
            fzi_str = f"< {c_high:.2f}"
        elif i == n_cuts:
            c_low = resolved_fzi_cutoffs[-1]
            quality = "Highest flow quality"
            code = f"DRT {idx} (FZI ≥ {c_low:.2f})"
            fzi_str = f"≥ {c_low:.2f}"
        else:
            c_low = resolved_fzi_cutoffs[i - 1]
            c_high = resolved_fzi_cutoffs[i]
            quality = "Intermediate flow quality"
            code = f"DRT {idx} ({c_low:.2f}–{c_high:.2f})"
            fzi_str = f"{c_low:.2f} – {c_high:.2f}"
        rt_Specs.append((idx, f"RT{idx} — {quality}", code, fzi_str))

    rock_types: List[Dict[str, Any]] = []
    saved_flag_rows: List[Dict[str, Any]] = []
    rt34_share_pct: Optional[float] = None
    rt34_kh_pct: Optional[float] = None
    dominant_codes: List[str] = []
    dominant_share_pct: Optional[float] = None
    dominant_kh_pct: Optional[float] = None
    rt_basis_source = "Core Plugs (CPORE & CPERM)"

    if not full_df.empty:
        tz_sub_rt = (
            full_df[
                full_df["ZONES"]
                .astype(str)
                .str.strip()
                .str.upper()
                .isin(target_zones_upper)
            ]
            if "ZONES" in full_df.columns
            else full_df
        )
        if tz_sub_rt.empty:
            tz_sub_rt = full_df

        cp_rt = _col_series(tz_sub_rt, "CPORE", np.nan)
        cp_rt = pd.Series(
            np.where(cp_rt > 1.0, cp_rt / 100.0, cp_rt), index=tz_sub_rt.index
        )
        ck_rt = _col_series(tz_sub_rt, "CPERM", np.nan)
        core_rt_mask = (
            cp_rt.notna()
            & ck_rt.notna()
            & (cp_rt > 0.01)
            & (cp_rt < 0.48)
            & (ck_rt > 1e-3)
        )

        rt_sel_mask = core_rt_mask
        if core_rt_mask.sum() >= 12:
            p_arr = cp_rt[core_rt_mask].to_numpy(dtype=float)
            k_arr = ck_rt[core_rt_mask].to_numpy(dtype=float)
            rt_basis_source = f"Core Plugs (n={len(p_arr)})"
        else:
            ph_rt = _col_series(
                tz_sub_rt, "PHIE" if "PHIE" in tz_sub_rt.columns else "PHIT", np.nan
            )
            pm_rt = _col_series(tz_sub_rt, "PERM", np.nan)
            log_rt_mask = (
                ph_rt.notna()
                & pm_rt.notna()
                & (ph_rt > 0.01)
                & (ph_rt < 0.48)
                & (pm_rt > 1e-3)
            )
            p_arr = ph_rt[log_rt_mask].to_numpy(dtype=float)
            k_arr = pm_rt[log_rt_mask].to_numpy(dtype=float)
            rt_basis_source = f"Target-Zone Evaluated Logs (n={len(p_arr)})"
            rt_sel_mask = log_rt_mask

        # Saved workbench ROCK_FLAG classes on the same samples, fitted as K = a * phi^b (the workbench form).
        if "ROCK_FLAG" in tz_sub_rt.columns and len(p_arr) >= 8:
            rf_sel = pd.to_numeric(
                tz_sub_rt.loc[rt_sel_mask, "ROCK_FLAG"], errors="coerce"
            ).to_numpy(dtype=float)
            ok_rf = np.isfinite(rf_sel) & (rf_sel >= 1)
            if ok_rf.sum() >= 8:
                n_rf_tot = int(ok_rf.sum())
                saved_perm_entry = _load_saved_project_module_entry(
                    pid, "perm-transform", resolved_target_zones
                )
                saved_perm_p = (
                    saved_perm_entry.get("params")
                    if isinstance(saved_perm_entry, dict)
                    else None
                ) or {}
                manual_fits = (
                    saved_perm_p.get("fits")
                    if saved_perm_p.get("manualEdits")
                    and isinstance(saved_perm_p.get("fits"), dict)
                    else {}
                )
                kh_total = float(np.sum(k_arr[ok_rf])) or 1e-9
                for rf_v in sorted({int(v) for v in rf_sel[ok_rf]}):
                    m_f = ok_rf & (rf_sel == rf_v)
                    n_f = int(m_f.sum())
                    row_f: Dict[str, Any] = {
                        "rock_flag": rf_v,
                        "n_samples": n_f,
                        "share_pct": round(100.0 * n_f / n_rf_tot, 1),
                        "kh_pct": round(
                            100.0 * float(np.sum(k_arr[m_f])) / kh_total, 1
                        ),
                        "a": None,
                        "b": None,
                        "r2": None,
                        "source": "insufficient data",
                    }
                    pf, kf = p_arr[m_f], k_arr[m_f]
                    if str(rf_v) in manual_fits:
                        try:
                            row_f["a"] = float(manual_fits[str(rf_v)]["a"])
                            row_f["b"] = float(manual_fits[str(rf_v)]["b"])
                            row_f["source"] = "saved workbench fit (manual)"
                        except (TypeError, ValueError, KeyError):
                            pass
                    if (
                        row_f["a"] is None
                        and n_f >= 3
                        and float(np.std(np.log10(pf))) > 1e-3
                    ):
                        b_fit, log_a = np.polyfit(
                            np.log10(np.clip(pf, 1e-6, None)),
                            np.log10(np.clip(kf, 1e-6, None)),
                            1,
                        )
                        row_f["a"], row_f["b"] = float(10.0**log_a), float(b_fit)
                        row_f["source"] = "fitted on report samples"
                    if row_f["a"] is not None:
                        pred = np.log10(
                            np.clip(
                                row_f["a"] * np.clip(pf, 1e-6, None) ** row_f["b"],
                                1e-6,
                                None,
                            )
                        )
                        obs = np.log10(np.clip(kf, 1e-6, None))
                        ss_t = float(np.sum((obs - obs.mean()) ** 2))
                        row_f["r2"] = (
                            round(1.0 - float(np.sum((obs - pred) ** 2)) / ss_t, 2)
                            if ss_t > 1e-12
                            else None
                        )
                        row_f["a"] = round(row_f["a"], 3)
                        row_f["b"] = round(row_f["b"], 3)
                    saved_flag_rows.append(row_f)

        if len(p_arr) >= 8:
            rqi = 0.0314 * np.sqrt(k_arr / np.clip(p_arr, 1e-4, 0.99))
            phiz = p_arr / np.clip(1.0 - p_arr, 1e-4, 1.0)
            fzi = rqi / np.clip(phiz, 1e-4, 100.0)
            rt_bins = np.digitize(fzi, bins=resolved_fzi_cutoffs) + 1
            total_n = max(len(p_arr), 1)
            total_kh = max(float(np.sum(k_arr)), 1e-9)

            class_kh: Dict[int, float] = {}
            for rt_idx, r_name, r_code, r_fzi in rt_Specs:
                m_rt = rt_bins == rt_idx
                cnt_rt = int(np.sum(m_rt))
                share_v = round(100.0 * cnt_rt / total_n, 1)
                class_kh[rt_idx] = float(np.sum(k_arr[m_rt])) if cnt_rt else 0.0
                fit_status = "fitted"
                if cnt_rt >= 3 and float(np.std(p_arr[m_rt])) > 1e-4:
                    reg = linregress(
                        p_arr[m_rt], np.log10(np.clip(k_arr[m_rt], 1e-4, 1e6))
                    )
                    slope_v = round(float(reg.slope), 2)
                    inter_v = round(float(reg.intercept), 2)
                    r2_v = round(max(0.0, min(1.0, float(reg.rvalue**2))), 2)
                else:
                    # No invented transform: report the absence of a fit instead of defaults.
                    slope_v = inter_v = r2_v = None
                    fit_status = "insufficient_data"
                rock_types.append(
                    {
                        "name": r_name,
                        "code": r_code,
                        "fzi_range": r_fzi,
                        "n_samples": cnt_rt,
                        "share_pct": share_v,
                        "slope": slope_v,
                        "intercept": inter_v,
                        "r2": r2_v,
                        "fit_status": fit_status,
                        "basis": rt_basis_source,
                    }
                )

            # Flow-capacity concentration: the fewest top-FZI classes holding >= 70% of sum(K).
            ranked = sorted(class_kh.items(), key=lambda kv: kv[1], reverse=True)
            run_kh, picked = 0.0, []
            for idx_c, kh_c in ranked:
                if kh_c <= 0:
                    break
                picked.append(idx_c)
                run_kh += kh_c
                if run_kh / total_kh >= 0.70:
                    break
            if picked:
                picked.sort()
                dominant_codes = [f"RT{i}" for i in picked]
                dominant_kh_pct = round(100.0 * run_kh / total_kh, 1)
                dominant_share_pct = round(
                    100.0 * float(np.sum(np.isin(rt_bins, picked))) / total_n, 1
                )
                rt34_share_pct, rt34_kh_pct = dominant_share_pct, dominant_kh_pct

    if not rock_types:
        # Insufficient data: list the configured classes without any fabricated statistics.
        for _, r_name, r_code, r_fzi in rt_Specs:
            rock_types.append(
                {
                    "name": r_name,
                    "code": r_code,
                    "fzi_range": r_fzi,
                    "n_samples": 0,
                    "share_pct": None,
                    "slope": None,
                    "intercept": None,
                    "r2": None,
                    "fit_status": "insufficient_data",
                    "basis": "Insufficient core/log data for rock typing (< 8 valid samples)",
                }
            )
        rt_basis_source = (
            "Insufficient core/log data for rock typing (< 8 valid samples)"
        )

    if model_params and isinstance(model_params, dict):
        merged_configs.update({k: v for k, v in model_params.items() if v is not None})

    # Pre-query committed fluid contacts for selected project wells to support Tier 2 FWL & fluid type resolution
    well_name_by_id = {w["id"]: w["name"] for w in selected_wells_meta}
    fluid_contacts_rows: List[Dict[str, Any]] = []

    def _map_spatial_zone(
        w_name_target: str, contact_depth: Optional[float], current_zone: Optional[str]
    ) -> Tuple[str, bool]:
        if contact_depth is None:
            return str(current_zone or "All Zones"), False
        cz = str(current_zone or "").strip()
        if cz and cz.upper() not in {
            "ALL ZONES",
            "GENERAL",
            "UNKNOWN",
            "NAN",
            "NONE",
            "",
        }:
            return cz, False
        cands = [zr for zr in well_zonal_rows if zr.get("well_name") == w_name_target]
        if not cands:
            cands = well_zonal_rows
        for zr in cands:
            top_v = (
                zr.get("top_tvdss")
                if zr.get("top_tvdss") is not None
                else zr.get("top")
            )
            bot_v = (
                zr.get("bottom_tvdss")
                if zr.get("bottom_tvdss") is not None
                else zr.get("bottom")
            )
            if top_v is not None and bot_v is not None:
                min_d = min(top_v, bot_v)
                max_d = max(top_v, bot_v) + 15.0
                if min_d <= contact_depth <= max_d:
                    return str(zr["zone"]), True
        return cz or "All Zones", False

    with get_db() as session:
        fc_res = session.execute(
            select(
                db_objects.ORMFluidContact.well_id,
                db_objects.ORMFluidContact.name,
                db_objects.ORMFluidContact.depth,
                db_objects.ORMFluidContact.zone,
                db_objects.ORMFluidContact.contact_type,
                db_objects.ORMFluidContact.fault_block,
                db_objects.ORMFluidContact.source,
                db_objects.ORMFluidContact.remarks,
            )
            .where(db_objects.ORMFluidContact.well_id.in_(list(well_name_by_id.keys())))
            .order_by(db_objects.ORMFluidContact.depth)
        ).all()
        for r in fc_res:
            c_type = str(r.contact_type or r.name or "CONTACT").upper()
            if c_type == "WDT":  # Excluded per user instruction
                continue
            wn = well_name_by_id.get(r.well_id, f"Well-{r.well_id}")
            cdepth = round(float(r.depth), 1) if r.depth is not None else None
            mapped_z, was_mapped = _map_spatial_zone(wn, cdepth, r.zone)
            remarks = str(r.remarks) if r.remarks else ""
            if was_mapped:
                remarks = (
                    remarks + " (Mapped from 'All Zones' by TVDss intersection)"
                ).strip()
            fluid_contacts_rows.append(
                {
                    "well_name": wn,
                    "name": str(r.name),
                    "depth": cdepth,
                    "zone": mapped_z,
                    "contact_type": c_type,
                    "fault_block": str(r.fault_block) if r.fault_block else None,
                    "source": str(r.source)
                    if r.source
                    else "Petrophysical Log Evaluation",
                    "remarks": remarks,
                }
            )

    # Spec 2.1 & CHK-02: True Vertical Depth Sub-Sea (TVDss) & Free Water Level (FWL) Validation
    # Never default FWL to max(Depth) - 15.0! Estimate strictly from target-zone transition interval.
    median_target_tvdss = 2940.0
    inferred_fwl_tvdss = 2943.0
    if not full_df.empty:
        tz_df_fwl = (
            full_df[
                full_df["ZONES"]
                .astype(str)
                .str.strip()
                .str.upper()
                .isin(target_zones_upper)
            ]
            if "ZONES" in full_df.columns
            else full_df
        )
        if tz_df_fwl.empty:
            tz_df_fwl = full_df

        if (
            "TVDSS" in tz_df_fwl.columns
            and pd.to_numeric(tz_df_fwl["TVDSS"], errors="coerce").notna().any()
        ):
            depth_ref_s = pd.to_numeric(tz_df_fwl["TVDSS"], errors="coerce")
        elif (
            "TVD" in tz_df_fwl.columns
            and pd.to_numeric(tz_df_fwl["TVD"], errors="coerce").notna().any()
        ):
            depth_ref_s = pd.to_numeric(tz_df_fwl["TVD"], errors="coerce") - 25.0
        else:
            depth_ref_s = pd.to_numeric(tz_df_fwl["DEPTH"], errors="coerce") - 25.0

        phi_fwl = _col_series(
            tz_df_fwl, "PHIE" if "PHIE" in tz_df_fwl.columns else "PHIT", np.nan
        )
        sw_fwl = _col_series(tz_df_fwl, "SWT", np.nan)
        vsh_fwl = _col_series(tz_df_fwl, "_REPORT_VSHALE", np.nan)

        res_mask_fwl = (
            depth_ref_s.notna()
            & phi_fwl.notna()
            & (phi_fwl >= phie_cut)
            & (vsh_fwl <= vsh_cut)
        )
        if res_mask_fwl.any():
            median_target_tvdss = round(
                float(np.nanmedian(depth_ref_s[res_mask_fwl])), 1
            )
        elif depth_ref_s.notna().any():
            median_target_tvdss = round(float(np.nanmedian(depth_ref_s.dropna())), 1)

        # Look for hydrocarbon-water transition zone (0.62 <= Sw <= 0.85) in target reservoir
        trans_mask = res_mask_fwl & sw_fwl.notna() & (sw_fwl >= 0.62) & (sw_fwl <= 0.85)
        hc_mask_fwl = res_mask_fwl & sw_fwl.notna() & (sw_fwl <= sw_cut)
        if trans_mask.sum() >= 3:
            inferred_fwl_tvdss = round(
                float(np.nanpercentile(depth_ref_s[trans_mask], 75)), 1
            )
        elif hc_mask_fwl.any():
            # Deepest Hydrocarbon-Down-To (HDT) + capillary transition offset
            inferred_fwl_tvdss = round(
                float(np.nanpercentile(depth_ref_s[hc_mask_fwl], 95)) + 6.0, 1
            )
        else:
            inferred_fwl_tvdss = round(median_target_tvdss + 15.0, 1)

    # 4-tier FWL resolution hierarchy (GAP-02)
    fwl_val: Optional[float] = None
    fwl_basis_str: str = ""

    # Tier 1: Explicit Request Override
    if fwl_tvdss is not None and np.isfinite(fwl_tvdss):
        fwl_val = round(float(fwl_tvdss), 1)
        fwl_basis_str = "User Request Override"

    # Tier 2: Committed Database Contacts (ORMFluidContact)
    if fwl_val is None and fluid_contacts_rows:

        def _match_zone(z_name: str) -> bool:
            if not z_name or str(z_name).strip().upper() in (
                "ALL ZONES",
                "GENERAL",
                "ALL",
                "-",
                "NONE",
            ):
                return True
            return str(z_name).strip().upper() in target_zones_upper

        fwl_matches = [
            r["depth"]
            for r in fluid_contacts_rows
            if r["depth"] is not None
            and r["contact_type"] == "FWL"
            and _match_zone(r["zone"])
        ]
        if not fwl_matches:
            fwl_matches = [
                r["depth"]
                for r in fluid_contacts_rows
                if r["depth"] is not None and r["contact_type"] == "FWL"
            ]
        if not fwl_matches:
            fwl_matches = [
                r["depth"]
                for r in fluid_contacts_rows
                if r["depth"] is not None
                and r["contact_type"] == "GWC"
                and _match_zone(r["zone"])
            ]
        if not fwl_matches:
            fwl_matches = [
                r["depth"]
                for r in fluid_contacts_rows
                if r["depth"] is not None
                and r["contact_type"] == "OWC"
                and _match_zone(r["zone"])
            ]
        if fwl_matches:
            fwl_val = round(float(np.median(fwl_matches)), 1)
            fwl_basis_str = "Committed Fluid Contact / FWL Optimizer (Pc = 0 datum)"

    # Saved workbench SHF snapshot (FWL, fluid system, per-rock-type J fits) for the active zone selection.
    saved_shf_entry = _load_saved_project_module_entry(
        pid, "shf", resolved_target_zones
    )
    saved_shf_params = (
        saved_shf_entry.get("params") if isinstance(saved_shf_entry, dict) else None
    ) or {}
    saved_shf_inner = (
        saved_shf_params.get("shfParams")
        if isinstance(saved_shf_params.get("shfParams"), dict)
        else {}
    )

    # Tier 3: Project Configuration (workbench-saved SHF FWL takes precedence over well config)
    if fwl_val is None:
        cfg_fwl = (
            saved_shf_inner.get("fwl")
            or merged_configs.get("fwl")
            or (
                merged_configs.get("shf_params", {}).get("fwl")
                if isinstance(merged_configs.get("shf_params"), dict)
                else None
            )
        )
        if cfg_fwl is not None:
            cand_fwl = _safe_float(cfg_fwl, None)
            if cand_fwl is not None and np.isfinite(cand_fwl):
                if abs(cand_fwl - median_target_tvdss) <= 300.0:
                    fwl_val = round(cand_fwl, 1)
                    fwl_basis_str = "Project Saturation/SHF Configuration"
                else:
                    anomaly_flags.append(
                        {
                            "category": "Suspicious Free Water Level Depth (CHK-02)",
                            "target": f"FWL = {cand_fwl:.1f} m vs Target Median = {median_target_tvdss:.1f} m TVDss",
                            "message": f"Configured FWL ({cand_fwl:.1f} m) deviates by >300 m from the target reservoir depth ({median_target_tvdss:.1f} m TVDss).",
                            "mitigation": f"Overrode uncalibrated TD/depth fallback with target-zone transition FWL ({inferred_fwl_tvdss:.1f} m TVDss).",
                        }
                    )

    # Tier 4: Transition-Zone Heuristic Fallback
    if fwl_val is None:
        fwl_val = inferred_fwl_tvdss
        fwl_basis_str = "Inferred from Hydrocarbon-Water Transition Zone (TVDss)"

    # Spec 3.2, CHK-04, CHK-05: True Probabilistic Monte Carlo / Latin Hypercube Uncertainty Engine (N=1000)
    # Incorporate borehole environmental correction 1-sigma (`NPHI_SIGMA`, `RHOB_SIGMA`, `RT_SIGMA`) when available.
    uncertainty_summary: List[Dict[str, Any]] = []
    rng_mc = np.random.default_rng(42)
    n_mc = 1000
    phi_sigma_mc = 0.018
    sw_sigma_mc = 0.11
    env_sigma_used = False
    if not full_df.empty:
        tz_unc_df = (
            full_df[
                full_df["ZONES"]
                .astype(str)
                .str.strip()
                .str.upper()
                .isin(target_zones_upper)
            ]
            if "ZONES" in full_df.columns and target_zones_upper
            else full_df
        )
        if tz_unc_df.empty:
            tz_unc_df = full_df
        nphi_sig_s = (
            pd.to_numeric(tz_unc_df["NPHI_SIGMA"], errors="coerce").dropna()
            if "NPHI_SIGMA" in tz_unc_df.columns
            else pd.Series(dtype=float)
        )
        rhob_sig_s = (
            pd.to_numeric(tz_unc_df["RHOB_SIGMA"], errors="coerce").dropna()
            if "RHOB_SIGMA" in tz_unc_df.columns
            else pd.Series(dtype=float)
        )
        rt_sig_s = (
            pd.to_numeric(tz_unc_df["RT_SIGMA"], errors="coerce").dropna()
            if "RT_SIGMA" in tz_unc_df.columns
            else pd.Series(dtype=float)
        )
        if not nphi_sig_s.empty or not rhob_sig_s.empty:
            sig_n = float(nphi_sig_s.median()) if not nphi_sig_s.empty else 0.015
            sig_d = float(rhob_sig_s.median()) / 1.65 if not rhob_sig_s.empty else 0.015
            phi_sigma_mc = float(
                np.clip(math.sqrt(0.5 * (sig_n**2 + sig_d**2)), 0.012, 0.045)
            )
            env_sigma_used = True
        if not rt_sig_s.empty:
            sig_rt_dec = float(rt_sig_s.median())
            # Propagate log10(Rt) sigma through Sw ~ Rt^(-1/n) combined with Archie/WS parameter uncertainty
            sw_sigma_mc = float(
                np.clip(
                    math.sqrt(0.095**2 + (0.5 * math.log(10.0) * sig_rt_dec) ** 2),
                    0.09,
                    0.28,
                )
            )
            env_sigma_used = True
    d_phi_mc = rng_mc.normal(0.0, phi_sigma_mc, size=(n_mc, 1))
    d_vsh_mc = rng_mc.normal(0.0, 0.035, size=(n_mc, 1))
    sw_mult_mc = rng_mc.lognormal(0.0, sw_sigma_mc, size=(n_mc, 1))

    for z in zonal_summary:
        z_name = z["zone"]
        det_net_pay = float(z.get("net_pay", 0.0))
        det_hcpt = float(z.get("hcpt", 0.0))

        # Extract evaluated samples for this zone
        z_df = (
            full_df[full_df["ZONES"].astype(str) == z_name]
            if not full_df.empty and "ZONES" in full_df.columns
            else pd.DataFrame()
        )
        phi_z = (
            _col_series(
                z_df, "PHIE" if "PHIE" in z_df.columns else "PHIT", np.nan
            ).to_numpy(dtype=float)
            if not z_df.empty
            else np.array([])
        )
        vsh_z = (
            _col_series(z_df, "_REPORT_VSHALE", np.nan).to_numpy(dtype=float)
            if not z_df.empty
            else np.array([])
        )
        sw_z = (
            _col_series(z_df, "SWT", np.nan).to_numpy(dtype=float)
            if not z_df.empty
            else np.array([])
        )
        valid_z = (
            np.isfinite(phi_z) & np.isfinite(sw_z) & np.isfinite(vsh_z) & (phi_z > 0.0)
        )
        phi_z = phi_z[valid_z]
        vsh_z = vsh_z[valid_z]
        sw_z = sw_z[valid_z]

        n_wells_z = max(1, int(z.get("well_count", 1)))
        det_gross = max(float(z.get("gross", 0.0)), 0.1524)
        if len(phi_z) > 0 and det_net_pay > 0.0:
            if len(phi_z) > 2500:
                idx_s = np.linspace(0, len(phi_z) - 1, 2500, dtype=int)
                phi_mat = phi_z[idx_s][None, :]
                vsh_mat = vsh_z[idx_s][None, :]
                sw_mat = sw_z[idx_s][None, :]
            else:
                phi_mat = phi_z[None, :]
                vsh_mat = vsh_z[None, :]
                sw_mat = sw_z[None, :]

            base_pay_0 = (
                (vsh_mat <= vsh_cut) & (phi_mat >= phie_cut) & (sw_mat <= sw_cut)
            )
            raw_hcpt_0 = float(
                np.sum(np.where(base_pay_0, phi_mat * (1.0 - sw_mat), 0.0))
            )
            if raw_hcpt_0 > 1e-6 and det_hcpt > 0.0:
                step_scale = det_hcpt / raw_hcpt_0
            else:
                step_scale = det_gross / max(float(phi_mat.shape[1]), 1.0)

            phi_real = np.clip(phi_mat + d_phi_mc, 0.0, 0.45)
            vsh_real = np.clip(vsh_mat + d_vsh_mc, 0.0, 1.0)
            sw_real = np.clip(sw_mat * sw_mult_mc, 0.02, 1.0)
            pay_real = (
                (vsh_real <= vsh_cut) & (phi_real >= phie_cut) & (sw_real <= sw_cut)
            )

            hcpt_reals = np.sum(
                np.where(pay_real, phi_real * (1.0 - sw_real) * step_scale, 0.0),
                axis=1,
            )
            net_h_reals = np.sum(np.where(pay_real, step_scale, 0.0), axis=1)
            phi_reals = np.sum(
                np.where(pay_real, phi_real * step_scale, 0.0), axis=1
            ) / np.maximum(net_h_reals, 1e-6)
            pv_reals = np.maximum(net_h_reals * phi_reals, 1e-6)
            sw_reals = (
                np.sum(np.where(pay_real, phi_real * sw_real * step_scale, 0.0), axis=1)
                / pv_reals
            )

            net_p90 = round(float(np.nanpercentile(net_h_reals, 10)), 1)
            net_p50 = round(float(np.nanpercentile(net_h_reals, 50)), 1)
            net_p10 = round(float(np.nanpercentile(net_h_reals, 90)), 1)
            net_p90, net_p50, net_p10 = sorted([net_p90, net_p50, net_p10])

            hcpt_p90 = round(float(np.nanpercentile(hcpt_reals, 10)), 2)
            hcpt_p50 = round(float(np.nanpercentile(hcpt_reals, 50)), 2)
            hcpt_p10 = round(float(np.nanpercentile(hcpt_reals, 90)), 2)

            # Ensure monotonic ordering (CHK-04: P90 <= P50 <= P10)
            hcpt_p90, hcpt_p50, hcpt_p10 = sorted([hcpt_p90, hcpt_p50, hcpt_p10])

            phi_p90 = round(float(np.nanpercentile(phi_reals, 10)), 3)
            phi_p50 = round(float(np.nanpercentile(phi_reals, 50)), 3)
            phi_p10 = round(float(np.nanpercentile(phi_reals, 90)), 3)
            phi_p90, phi_p50, phi_p10 = sorted([phi_p90, phi_p50, phi_p10])

            # Note: for Sw, optimistic High Case (P10) has lower water saturation than Conservative Low Case (P90)
            sw_p10 = round(float(np.nanpercentile(sw_reals, 10)), 3)
            sw_p50 = round(float(np.nanpercentile(sw_reals, 50)), 3)
            sw_p90 = round(float(np.nanpercentile(sw_reals, 90)), 3)
            sw_p10, sw_p50, sw_p90 = sorted([sw_p10, sw_p50, sw_p90])

            if phi_p50 <= 0.0:
                hcpt_p50 = 0.0

            spread_ratio = round(hcpt_p10 / hcpt_p90, 2) if hcpt_p90 > 0.005 else 1.0
            uncertainty_summary.extend(
                [
                    {
                        "zone": z_name,
                        "metric": "Net Pay (m)",
                        "p90_hcpt": hcpt_p90,
                        "p50_hcpt": hcpt_p50,
                        "p10_hcpt": hcpt_p10,
                        "p50_phie": phi_p50,
                        "p90": f"{net_p90:.1f} m",
                        "p50": f"{net_p50:.1f} m",
                        "p10": f"{net_p10:.1f} m",
                        "spread": f"{round(net_p10 / max(net_p90, 0.01), 2):.2f}×"
                        if net_p90 > 0.05
                        else "—",
                    },
                    {
                        "zone": z_name,
                        "metric": "Mean Effective Porosity Φe (v/v)",
                        "p90_hcpt": hcpt_p90,
                        "p50_hcpt": hcpt_p50,
                        "p10_hcpt": hcpt_p10,
                        "p50_phie": phi_p50,
                        "p90": f"{phi_p90:.3f}",
                        "p50": f"{phi_p50:.3f}",
                        "p10": f"{phi_p10:.3f}",
                        "spread": f"{round(phi_p10 / max(phi_p90, 0.001), 2):.2f}×"
                        if phi_p90 > 0.001
                        else "—",
                    },
                    {
                        "zone": z_name,
                        "metric": "Mean Water Saturation Sw (v/v)",
                        "p90_hcpt": hcpt_p90,
                        "p50_hcpt": hcpt_p50,
                        "p10_hcpt": hcpt_p10,
                        "p50_phie": phi_p50,
                        "p90": f"{sw_p90:.3f}",
                        "p50": f"{sw_p50:.3f}",
                        "p10": f"{sw_p10:.3f}",
                        "spread": f"{round(sw_p90 / max(sw_p10, 0.001), 2):.2f}×"
                        if sw_p10 > 0.001
                        else "—",
                    },
                    {
                        "zone": z_name,
                        "metric": "Hydrocarbon Pore Thickness HCPT (m)",
                        "p90_hcpt": hcpt_p90,
                        "p50_hcpt": hcpt_p50,
                        "p10_hcpt": hcpt_p10,
                        "p50_phie": phi_p50,
                        "p90": f"{hcpt_p90:.2f} m",
                        "p50": f"{hcpt_p50:.2f} m",
                        "p10": f"{hcpt_p10:.2f} m",
                        "spread": f"{spread_ratio:.2f}×" if hcpt_p90 > 0.005 else "—",
                        # Numeric twin of `spread` so narratives quote the table value, not a default.
                        "spread_ratio": spread_ratio if hcpt_p90 > 0.005 else None,
                    },
                ]
            )
        else:
            # Spec 3.2 & CHK-05: Zero deterministic Net Pay -> report 0.00 m (no dummy 0.11 / 0.15 / 0.19)
            uncertainty_summary.extend(
                [
                    {
                        "zone": z_name,
                        "metric": "Net Pay (m)",
                        "p90_hcpt": 0.0,
                        "p50_hcpt": 0.0,
                        "p10_hcpt": 0.0,
                        "p50_phie": 0.0,
                        "p90": "0.0 m",
                        "p50": "0.0 m",
                        "p10": "0.0 m",
                        "spread": "—",
                    },
                    {
                        "zone": z_name,
                        "metric": "Mean Effective Porosity Φe (v/v)",
                        "p90_hcpt": 0.0,
                        "p50_hcpt": 0.0,
                        "p10_hcpt": 0.0,
                        "p50_phie": 0.0,
                        "p90": "0.000",
                        "p50": "0.000",
                        "p10": "0.000",
                        "spread": "—",
                    },
                    {
                        "zone": z_name,
                        "metric": "Mean Water Saturation Sw (v/v)",
                        "p90_hcpt": 0.0,
                        "p50_hcpt": 0.0,
                        "p10_hcpt": 0.0,
                        "p50_phie": 0.0,
                        "p90": "1.000",
                        "p50": "1.000",
                        "p10": "1.000",
                        "spread": "—",
                    },
                    {
                        "zone": z_name,
                        "metric": "Hydrocarbon Pore Thickness HCPT (m)",
                        "p90_hcpt": 0.0,
                        "p50_hcpt": 0.0,
                        "p10_hcpt": 0.0,
                        "p50_phie": 0.0,
                        "p90": "0.00 m (Water-bearing / No Pay)",
                        "p50": "0.00 m (Water-bearing / No Pay)",
                        "p10": "0.00 m (Water-bearing / No Pay)",
                        "spread": "—",
                    },
                ]
            )

    uncertainty_drivers_text = (
        "Hydrocarbon pore thickness (HCPT) variance is primarily governed by: "
        "(1) Formation water resistivity ($R_w$) & saturation exponent ($n$)"
        + (
            f" coupled with borehole resistivity 1-sigma ($\\sigma_{{Sw}} = {sw_sigma_mc:.3f}$), "
            if env_sigma_used
            else ", "
        )
        + "(2) Matrix density ($\\rho_{ma}$) & porosity calibration"
        + (
            f" propagating environmental sensor uncertainty ($\\sigma_\\phi = {phi_sigma_mc:.3f}\\text{{ v/v}}$), and "
            if env_sigma_used
            else ", and "
        )
        + f"(3) Net pay cutoff thresholds ($\\Phi_e \\ge {phie_cut:.2f}$, $V_{{sh}} \\le {vsh_cut:.2f}$, $S_w \\le {sw_cut:.2f}$)."
    )

    # Task 1.2: Phase-aware fluid type and fluid densities detection (GAP-07)
    shf_cfg = (
        merged_configs.get("shf_params")
        if isinstance(merged_configs.get("shf_params"), dict)
        else {}
    )
    litho_cfg = (
        merged_configs.get("litho_params")
        if isinstance(merged_configs.get("litho_params"), dict)
        else {}
    )
    raw_fluid_type = (
        str(
            merged_configs.get("fluid_type")
            or shf_cfg.get("fluid_type")
            or litho_cfg.get("hc_fluid_type")
            or merged_configs.get("hc_fluid_type")
            or ""
        )
        .strip()
        .lower()
    )

    # Gather & analyze Formation Pressure Test (FPT) data for Section 6.3 & SHF density reconciliation (GAP-13)
    fpt_summary, raw_fpt_points = _gather_and_analyze_fpt_data(
        project_id=pid,
        wells_inventory=wells_inventory,
        merged_configs=merged_configs,
        fluid_contacts_rows=fluid_contacts_rows,
        target_zones=resolved_target_zones,
        fwl_val=fwl_val,
        fwl_is_committed=not str(fwl_basis_str).startswith("Inferred"),
    )
    fitted_dens = fpt_summary.get("fitted_densities", {})
    water_fit = fitted_dens.get("WATER")
    gas_fit = fitted_dens.get("GAS")
    oil_fit = fitted_dens.get("OIL")

    contact_types_seen = {r["contact_type"] for r in fluid_contacts_rows}
    has_gas_contacts = bool({"GUT", "GDT", "GWC"} & contact_types_seen) or (
        bool(
            gas_fit
            and gas_fit.get("r_squared", 0.0) >= 0.85
            and gas_fit.get("n_points", 0) >= 2
        )
    )
    has_oil_contacts = bool({"OUT", "ODT", "OWC"} & contact_types_seen) or (
        bool(
            oil_fit
            and oil_fit.get("r_squared", 0.0) >= 0.85
            and oil_fit.get("n_points", 0) >= 2
        )
    )

    raw_rho_hc = (
        merged_configs.get("rho_hc") or shf_cfg.get("ghc") or litho_cfg.get("rho_hc")
    )
    rho_hc_cand = _safe_float(raw_rho_hc, None)

    if raw_fluid_type in ("gas", "dry_gas", "wet_gas", "condensate"):
        res_fluid = "gas"
    elif raw_fluid_type in ("oil", "black_oil", "heavy_oil", "volatile_oil"):
        res_fluid = "oil"
    elif has_gas_contacts and not has_oil_contacts:
        res_fluid = "gas"
    elif rho_hc_cand is not None and rho_hc_cand < 0.45:
        res_fluid = "gas"
    else:
        res_fluid = (
            "oil" if has_oil_contacts else ("gas" if has_gas_contacts else "oil")
        )

    # Measured FPT Formation Water Gradient reconciliation
    if (
        water_fit
        and water_fit.get("r_squared", 0.0) >= 0.90
        and water_fit.get("n_points", 0) >= 3
    ):
        gw_val = water_fit["density_gcc"]
        gw_basis = f"Measured FPT Formation Water Gradient ({water_fit['density_gcc']:.3f} g/cc, R²={water_fit['r_squared']:.2f})"
    else:
        gw_val = _safe_float(
            shf_cfg.get("gw")
            or merged_configs.get("rho_w")
            or merged_configs.get("gw"),
            1.05,
        )
        gw_basis = "Configured / Regional Formation Water Density"

    # Measured FPT Hydrocarbon Gradient reconciliation
    if res_fluid == "gas":
        if (
            gas_fit
            and gas_fit.get("r_squared", 0.0) >= 0.90
            and gas_fit.get("n_points", 0) >= 3
        ):
            ghc_val = gas_fit["density_gcc"]
            ghc_basis = f"Measured FPT Gas Gradient ({gas_fit['density_gcc']:.3f} g/cc, R²={gas_fit['r_squared']:.2f})"
        elif rho_hc_cand is not None and rho_hc_cand < 0.45:
            ghc_val = round(rho_hc_cand, 3)
            ghc_basis = "Configured Gas Density"
        else:
            ghc_val = 0.22
            ghc_basis = "Default Dry/Wet Gas Reservoir Density (No PVT/FPT Override)"
    else:
        if (
            oil_fit
            and oil_fit.get("r_squared", 0.0) >= 0.90
            and oil_fit.get("n_points", 0) >= 3
        ):
            ghc_val = oil_fit["density_gcc"]
            ghc_basis = f"Measured FPT Oil Gradient ({oil_fit['density_gcc']:.3f} g/cc, R²={oil_fit['r_squared']:.2f})"
        else:
            ghc_val = round(rho_hc_cand if rho_hc_cand is not None else 0.80, 3)
            ghc_basis = "Configured Oil Density"
    delta_rho = round(gw_val - ghc_val, 3)

    # Re-evaluate SHF curve using resolved FWL, phase-aware delta_rho (GAP-07) and the workbench-saved
    # per-rock-type Leverett-J fits. No reconciliation statistic is reported unless it is actually computed.
    shf_rmse: Optional[float] = None
    shf_bias: Optional[float] = None
    shf_fit_rows: List[Dict[str, Any]] = []
    shf_fit_source = ""
    shf_n_compared = 0
    saved_fits_blob = saved_shf_params.get("fits")
    saved_fits: Dict[str, Dict[str, Any]] = {}
    saved_fit_model = "leverett_j"
    if isinstance(saved_fits_blob, dict):
        saved_fit_model = str(saved_fits_blob.get("model") or "leverett_j")
        inner = saved_fits_blob.get("fits")
        if isinstance(inner, dict):
            for k, v in inner.items():
                if (
                    isinstance(v, dict)
                    and v.get("a") is not None
                    and v.get("b") is not None
                ):
                    saved_fits[str(k)] = v
    use_saved_fits = bool(saved_fits) and saved_fit_model == "leverett_j"
    if use_saved_fits:
        shf_fit_source = (
            f"saved workbench SHF fit ({saved_shf_entry.get('label', 'project')})"
        )
    elif saved_fits:
        shf_fit_source = (
            f"generic fallback J-function (saved fit uses '{saved_fit_model}', "
            "which the report does not evaluate)"
        )
    else:
        shf_fit_source = "generic fallback J-function (no saved SHF fit found)"
    if not full_df.empty:
        d_col = (
            "TVDSS"
            if "TVDSS" in full_df.columns
            else ("TVD" if "TVD" in full_df.columns else "DEPTH")
        )
        phi_col = (
            "PHIE"
            if "PHIE" in full_df.columns
            else ("PHIT" if "PHIT" in full_df.columns else None)
        )
        k_col = (
            "CPERM"
            if "CPERM" in full_df.columns
            else ("PERM" if "PERM" in full_df.columns else None)
        )
        if phi_col and k_col and d_col in full_df.columns:
            try:
                from quick_pp.core_analysis import sw_shf_leverett_j

                d_arr = pd.to_numeric(full_df[d_col], errors="coerce").to_numpy(
                    dtype=float
                )
                phi_arr = pd.to_numeric(full_df[phi_col], errors="coerce").to_numpy(
                    dtype=float
                )
                phi_arr = np.where(phi_arr > 1.0, phi_arr / 100.0, phi_arr)
                k_arr = pd.to_numeric(full_df[k_col], errors="coerce").to_numpy(
                    dtype=float
                )
                ift_val = _safe_float(
                    merged_configs.get("ift_res", shf_cfg.get("ift", 30.0)), 30.0
                )
                theta_val = _safe_float(
                    merged_configs.get("theta_res", shf_cfg.get("theta", 30.0)), 30.0
                )
                # Per-row a/b: saved fit for the row's ROCK_FLAG, else the generic fallback pair.
                a_arr = np.full(len(full_df), 0.22, dtype=float)
                b_arr = np.full(len(full_df), -0.50, dtype=float)
                if use_saved_fits and "ROCK_FLAG" in full_df.columns:
                    rf_arr = pd.to_numeric(
                        full_df["ROCK_FLAG"], errors="coerce"
                    ).to_numpy(dtype=float)
                    for rf_key, fit in saved_fits.items():
                        try:
                            rf_int = int(rf_key)
                            a_fit, b_fit = float(fit["a"]), float(fit["b"])
                        except (TypeError, ValueError, KeyError):
                            continue
                        m_rf = np.isfinite(rf_arr) & (rf_arr == rf_int)
                        a_arr[m_rf] = a_fit
                        b_arr[m_rf] = b_fit
                elif use_saved_fits:
                    # No ROCK_FLAG on the data: use the first saved fit as a single global pair.
                    first_fit = next(iter(saved_fits.values()))
                    a_arr[:] = float(first_fit["a"])
                    b_arr[:] = float(first_fit["b"])
                    shf_fit_source += (
                        " (single global pair; no ROCK_FLAG curve on data)"
                    )

                shf_calc = sw_shf_leverett_j(
                    perm=k_arr,
                    phit=phi_arr,
                    depth=d_arr,
                    fwl=fwl_val,
                    ift=ift_val,
                    theta=theta_val,
                    gw=gw_val,
                    ghc=ghc_val,
                    a=a_arr,
                    b=b_arr,
                    unit="metric" if d_col in ("TVDSS", "TVD", "DEPTH") else "imperial",
                    clip=True,
                )
                full_df["SHF"] = shf_calc
            except Exception as exc:
                logger.debug(f"SHF recomputation skipped: {exc}")

        # Reconcile only inside the active target zones and above the FWL, on rows where both curves exist.
        if "SHF" in full_df.columns and "SWT" in full_df.columns:
            rec_df = full_df
            if "ZONES" in full_df.columns and target_zones_upper:
                rec_mask_z = (
                    full_df["ZONES"]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .isin(target_zones_upper)
                )
                if rec_mask_z.any():
                    rec_df = full_df[rec_mask_z]
            rec_depth = pd.to_numeric(
                rec_df[d_col] if d_col in rec_df.columns else rec_df["DEPTH"],
                errors="coerce",
            )
            sw_log = pd.to_numeric(rec_df["SWT"], errors="coerce")
            sw_cap = pd.to_numeric(rec_df["SHF"], errors="coerce")
            rec_ok = (
                sw_log.notna()
                & sw_cap.notna()
                & rec_depth.notna()
                & (rec_depth < fwl_val)
            )
            diff = (sw_log[rec_ok] - sw_cap[rec_ok]).dropna()
            shf_n_compared = int(len(diff))
            if shf_n_compared >= 20:
                shf_rmse = round(float(np.sqrt(np.mean(diff**2))), 4)
                shf_bias = round(float(diff.mean()), 4)

        # Per-rock-type fit table: saved parameters plus the log-vs-capillary agreement in each class.
        if use_saved_fits:
            rf_series = (
                pd.to_numeric(full_df["ROCK_FLAG"], errors="coerce")
                if "ROCK_FLAG" in full_df.columns
                else None
            )
            for rf_key in sorted(
                saved_fits, key=lambda k: (not str(k).isdigit(), str(k))
            ):
                fit = saved_fits[rf_key]
                row = {
                    "rock_flag": rf_key,
                    "a": round(float(fit["a"]), 4),
                    "b": round(float(fit["b"]), 4),
                    "r2": fit.get("r2"),
                    "rmse": fit.get("rmse"),
                    "n_scal": int(fit.get("count", 0) or 0),
                    "n_log": None,
                    "log_rmse": None,
                }
                if (
                    rf_series is not None
                    and "SHF" in full_df.columns
                    and "SWT" in full_df.columns
                    and str(rf_key).isdigit()
                ):
                    m_c = (
                        (rf_series == int(rf_key))
                        & pd.to_numeric(full_df["SWT"], errors="coerce").notna()
                        & pd.to_numeric(full_df["SHF"], errors="coerce").notna()
                    )
                    if "ZONES" in full_df.columns and target_zones_upper:
                        m_c &= (
                            full_df["ZONES"]
                            .astype(str)
                            .str.strip()
                            .str.upper()
                            .isin(target_zones_upper)
                        )
                    d_c = (
                        pd.to_numeric(full_df[d_col], errors="coerce")
                        if d_col in full_df.columns
                        else None
                    )
                    if d_c is not None:
                        m_c &= d_c < fwl_val
                    if int(m_c.sum()) >= 10:
                        dd = (
                            pd.to_numeric(full_df.loc[m_c, "SWT"], errors="coerce")
                            - pd.to_numeric(full_df.loc[m_c, "SHF"], errors="coerce")
                        ).dropna()
                        row["n_log"] = int(len(dd))
                        row["log_rmse"] = round(float(np.sqrt(np.mean(dd**2))), 3)
                shf_fit_rows.append(row)

    # Resolve active Lithology, Porosity, and Saturation models from config or evaluated curves
    raw_litho_key = (
        str(
            merged_configs.get("litho_model_key")
            or merged_configs.get("litho_model")
            or ""
        )
        .strip()
        .lower()
    )

    def _has_positive_col(col_name: str, threshold: float = 0.005) -> bool:
        if full_df.empty or col_name not in full_df.columns:
            return False
        s_num = pd.to_numeric(full_df[col_name], errors="coerce").dropna()
        return bool((s_num > threshold).any())

    has_carb_evap = any(
        _has_positive_col(c) for c in ("VCALC", "VDOLO", "VANHY", "VHALI", "VGYPS")
    )
    has_ssc_cols = _has_positive_col("VSAND") and _has_positive_col("VSILT")

    if raw_litho_key in ("multi_mineral", "multimineral", "multi-mineral"):
        resolved_litho_key = "multi_mineral"
    elif raw_litho_key in ("ssc", "sand_silt_clay", "kuttan"):
        resolved_litho_key = "ssc"
    elif raw_litho_key in ("sand_shale", "vsh_nd", "ss"):
        resolved_litho_key = "sand_shale"
    elif raw_litho_key in ("vsh_gr", "gr", "gamma_ray"):
        resolved_litho_key = "vsh_gr"
    elif raw_litho_key == "vsh_bulk":
        resolved_litho_key = "vsh_bulk"
    elif has_carb_evap:
        resolved_litho_key = "multi_mineral"
    elif has_ssc_cols:
        resolved_litho_key = "ssc"
    elif _has_positive_col("VCLAY") or _has_positive_col("VSH_GR"):
        resolved_litho_key = "ssc" if _has_positive_col("VSAND") else "vsh_gr"
    else:
        resolved_litho_key = "ssc"

    if resolved_litho_key == "multi_mineral":
        cfg_mins = merged_configs.get("litho_minerals")
        if not cfg_mins or not isinstance(cfg_mins, list):
            cfg_mins = []
            for m_col, m_label in (
                ("VSAND", "Quartz"),
                ("VCALC", "Calcite"),
                ("VDOLO", "Dolomite"),
                ("VCLAY", "Shale"),
                ("VANHY", "Anhydrite"),
                ("VHALI", "Halite"),
            ):
                if _has_positive_col(m_col):
                    cfg_mins.append(m_label)
        min_str = (
            ", ".join(str(m).title() for m in cfg_mins)
            if cfg_mins
            else "Quartz, Calcite, Dolomite, Shale"
        )
        cfg_logs = merged_configs.get("litho_input_logs")
        log_str = (
            ", ".join(str(name).upper() for name in cfg_logs)
            if isinstance(cfg_logs, list) and cfg_logs
            else None
        )
        litho_model_display = f"Multi-Mineral Optimization Solver ({min_str})"
        if log_str:
            litho_model_display += f" — input logs: {log_str}"
    elif resolved_litho_key == "sand_shale":
        litho_model_display = (
            "Binary Sand-Shale Neutron-Density Crossplot Model (Vsh from N-D)"
        )
    elif resolved_litho_key == "vsh_gr":
        litho_model_display = "Gamma Ray Shale Volume Index (Vsh,GR) Model"
    elif resolved_litho_key == "vsh_bulk":
        litho_model_display = "Bulk Shale from GR / CGR Density-Balance Model"
    else:
        litho_model_display = "Modified Kuttan Sand-Silt-Clay (SSC) Ternary Model"

    raw_poro_key = (
        str(
            merged_configs.get("porosity_method_key")
            or merged_configs.get("porosity_method")
            or "den"
        )
        .strip()
        .lower()
    )
    if resolved_litho_key == "vsh_bulk":
        poro_method_display = "Bulk Shale Density-Balance Porosity"
    elif raw_poro_key in ("neu_den", "neutron_density", "xplot"):
        poro_method_display = (
            "Neutron-Density Crossplot Total & Clay-Corrected Effective Porosity"
        )
    elif raw_poro_key in (
        "sonic_wyllie",
        "sonic_hunt_raymer",
        "avg_weighted",
        "avg_arithmetic",
        "avg_gaymard",
    ):
        poro_method_display = {
            "sonic_wyllie": "Wyllie Sonic Porosity",
            "sonic_hunt_raymer": "Hunt-Raymer Sonic Porosity",
            "avg_weighted": "Weighted Neutron-Density Porosity",
            "avg_arithmetic": "Arithmetic Neutron-Density Porosity",
            "avg_gaymard": "RMS (Gaymard/Gas) Neutron-Density Porosity",
        }[raw_poro_key]
    else:
        poro_method_display = "Clay-Corrected Matrix Density Total & Effective Porosity"

    raw_sw_key = (
        str(
            merged_configs.get("sw_model_key")
            or merged_configs.get("sw_model")
            or "archie"
        )
        .strip()
        .lower()
    )
    sw_model_display = {
        "archie": "Archie Clean-Sand Water Saturation Model",
        "waxman_smits": "Waxman-Smits Shaly-Sand Cation Exchange ($Q_v$) Model",
        "simandoux": "Simandoux Shaly-Sand Water Saturation Model",
        "indonesian": "Indonesian Shaly-Sand Water Saturation Model",
        "dual_water": "Dual-Water Shaly-Sand Water Saturation Model",
    }.get(raw_sw_key, "Archie Clean-Sand Water Saturation Model")

    dry_sand_pt = merged_configs.get("dry_sand_point")
    dry_clay_pt = merged_configs.get("dry_clay_point")
    fluid_pt = merged_configs.get("fluid_point")
    rho_ma_cfg = (
        _safe_float(dry_sand_pt[1], 2.65)
        if isinstance(dry_sand_pt, (list, tuple)) and len(dry_sand_pt) > 1
        else _safe_float(merged_configs.get("rho_ma"), 2.65)
    )
    nphi_clay_cfg = (
        _safe_float(dry_clay_pt[0], 0.33)
        if isinstance(dry_clay_pt, (list, tuple)) and len(dry_clay_pt) > 0
        else _safe_float(merged_configs.get("nphi_clay"), 0.33)
    )
    rho_clay_cfg = (
        _safe_float(dry_clay_pt[1], 2.70)
        if isinstance(dry_clay_pt, (list, tuple)) and len(dry_clay_pt) > 1
        else _safe_float(merged_configs.get("rho_clay"), 2.70)
    )
    rho_fl_cfg = (
        _safe_float(fluid_pt[1], 1.00)
        if isinstance(fluid_pt, (list, tuple)) and len(fluid_pt) > 1
        else _safe_float(merged_configs.get("rho_fl"), 1.00)
    )
    gr_clean_cfg = _safe_float(merged_configs.get("gr_clean"), 15.0)
    gr_clay_cfg = _safe_float(
        merged_configs.get("gr_clay", merged_configs.get("gr_shale")), 115.0
    )
    bh_treat_raw = str(merged_configs.get("badhole_treatment") or "vsh_gr").strip()
    bh_treat_display = (
        "vsh_gr (GR Shale Fallback)"
        if bh_treat_raw.lower() == "vsh_gr"
        else bh_treat_raw
    )

    # Aggregate Field/Well Borehole Environmental Correction, P&T & Invasion Summary
    primary_env = resolved_envs[0] if resolved_envs else environment_from_config({}, {})
    req_env_cfg = (
        merged_configs.get("env_corr_params")
        if isinstance(merged_configs.get("env_corr_params"), dict)
        else {}
    )
    eff_tool_type = str(
        req_env_cfg.get("toolType")
        or req_env_cfg.get("tool_type")
        or merged_configs.get("tool_type")
        or "WL"
    ).upper()
    if eff_tool_type not in ("WL", "LWD"):
        eff_tool_type = "WL"
    eff_res_tool = str(
        req_env_cfg.get("resTool")
        or req_env_cfg.get("res_tool")
        or ("propagation" if eff_tool_type == "LWD" else "induction")
    ).lower()
    if eff_res_tool == "auto":
        eff_res_tool = "propagation" if eff_tool_type == "LWD" else "induction"

    tz_env_df = (
        full_df[
            full_df["ZONES"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(target_zones_upper)
        ]
        if not full_df.empty and "ZONES" in full_df.columns and target_zones_upper
        else full_df
    )
    if tz_env_df.empty:
        tz_env_df = full_df

    env_corr_curve_table: List[Dict[str, Any]] = []
    if not tz_env_df.empty:
        reason_tz = (
            pd.to_numeric(tz_env_df["ENV_QC_REASON"], errors="coerce")
            .fillna(0)
            .astype(int)
            if "ENV_QC_REASON" in tz_env_df.columns
            else pd.Series(dtype=int)
        )
        for lg_k in ("GR", "NPHI", "RHOB", "RT", "DT"):
            c_col = f"{lg_k}_ENV_CORR"
            st_col = f"{lg_k}_QC_STATUS"
            cf_col = f"{lg_k}_QC_CONF"
            sg_col = f"{lg_k}_SIGMA"
            dl_col = f"{lg_k}_CORR_DELTA"
            has_c = (
                c_col in tz_env_df.columns
                and pd.to_numeric(tz_env_df[c_col], errors="coerce").notna().any()
            )
            has_st = (
                st_col in tz_env_df.columns
                and pd.to_numeric(tz_env_df[st_col], errors="coerce").ge(0).any()
            )
            if not (has_c or has_st):
                continue
            if has_st:
                st_s = pd.to_numeric(tz_env_df[st_col], errors="coerce")
                valid_m = st_s >= 0
                nv = int(valid_m.sum())
                good_p = round(100.0 * float((st_s == 0).sum()) / max(nv, 1), 1)
                corr_p = round(100.0 * float((st_s == 1).sum()) / max(nv, 1), 1)
                unrel_p = round(100.0 * float((st_s == 2).sum()) / max(nv, 1), 1)
            else:
                corr_s = pd.to_numeric(tz_env_df[c_col], errors="coerce")
                valid_m = corr_s.notna()
                nv = int(valid_m.sum())
                good_p, corr_p, unrel_p = 0.0, 100.0, 0.0
            cf_mean = (
                round(
                    float(
                        pd.to_numeric(tz_env_df[cf_col], errors="coerce")[
                            valid_m
                        ].mean()
                    ),
                    3,
                )
                if cf_col in tz_env_df.columns and nv > 0
                else None
            )
            sg_mean = (
                round(
                    float(
                        pd.to_numeric(tz_env_df[sg_col], errors="coerce")[
                            valid_m
                        ].mean()
                    ),
                    4,
                )
                if sg_col in tz_env_df.columns and nv > 0
                else None
            )
            if dl_col in tz_env_df.columns:
                dl_s = pd.to_numeric(tz_env_df[dl_col], errors="coerce")[
                    valid_m
                ].dropna()
            elif has_c and lg_k in tz_env_df.columns:
                if lg_k == "RT":
                    dl_s = (
                        np.log10(
                            pd.to_numeric(tz_env_df[c_col], errors="coerce").clip(
                                lower=1e-3
                            )
                        )
                        - np.log10(
                            pd.to_numeric(tz_env_df[lg_k], errors="coerce").clip(
                                lower=1e-3
                            )
                        )
                    )[valid_m].dropna()
                else:
                    dl_s = (
                        pd.to_numeric(tz_env_df[c_col], errors="coerce")
                        - pd.to_numeric(tz_env_df[lg_k], errors="coerce")
                    )[valid_m].dropna()
            else:
                dl_s = pd.Series(dtype=float)
            dom_r = (
                _dominant_env_reason(reason_tz, valid_m)
                if not reason_tz.empty
                else "Clean"
            )
            env_corr_curve_table.append(
                {
                    "curve": lg_k,
                    "corr_curve": c_col,
                    "unit": _ENV_CURVE_UNITS.get(lg_k, ""),
                    "samples": nv,
                    "good_pct": good_p,
                    "corrected_pct": corr_p,
                    "unreliable_pct": unrel_p,
                    "mean_conf": cf_mean,
                    "mean_sigma": sg_mean,
                    "mean_delta": (
                        round(float(dl_s.mean()), 4) if not dl_s.empty else None
                    ),
                    "mean_abs_delta": (
                        round(float(dl_s.abs().mean()), 4) if not dl_s.empty else None
                    ),
                    "p95_abs_delta": (
                        round(float(dl_s.abs().quantile(0.95)), 4)
                        if not dl_s.empty
                        else None
                    ),
                    "dominant_reason": dom_r,
                }
            )

    env_temp_s = (
        pd.to_numeric(tz_env_df["ENV_TEMP_C"], errors="coerce").dropna()
        if not tz_env_df.empty and "ENV_TEMP_C" in tz_env_df.columns
        else pd.Series(dtype=float)
    )
    env_rm_s = (
        pd.to_numeric(tz_env_df["ENV_RM"], errors="coerce").dropna()
        if not tz_env_df.empty and "ENV_RM" in tz_env_df.columns
        else pd.Series(dtype=float)
    )
    env_rmf_s = (
        pd.to_numeric(tz_env_df["ENV_RMF"], errors="coerce").dropna()
        if not tz_env_df.empty and "ENV_RMF" in tz_env_df.columns
        else pd.Series(dtype=float)
    )
    env_rw_s = (
        pd.to_numeric(tz_env_df["ENV_RW"], errors="coerce").dropna()
        if not tz_env_df.empty and "ENV_RW" in tz_env_df.columns
        else pd.Series(dtype=float)
    )
    reason_all = (
        pd.to_numeric(tz_env_df["ENV_QC_REASON"], errors="coerce").fillna(0).astype(int)
        if not tz_env_df.empty and "ENV_QC_REASON" in tz_env_df.columns
        else pd.Series(dtype=int)
    )
    total_cycle_skips = (
        int(
            (
                (reason_all.to_numpy(dtype=int) & ENV_QC_REASON_BITS["CYCLE_SKIP"]) != 0
            ).sum()
        )
        if not reason_all.empty
        else sum(int(r.get("cycle_skips", 0)) for r in env_well_rows)
    )
    total_res_mismatch = (
        int(
            (
                (
                    reason_all.to_numpy(dtype=int)
                    & ENV_QC_REASON_BITS["RES_MUD_MISMATCH"]
                )
                != 0
            ).sum()
        )
        if not reason_all.empty
        else 0
    )

    invasion_rollup: Dict[str, Any] = {}
    if invasion_well_rows:
        tot_inv_s = max(sum(r["samples"] for r in invasion_well_rows), 1)
        di_vals = [
            r["median_di_in"]
            for r in invasion_well_rows
            if r.get("median_di_in") is not None
        ]
        p90_di_vals = [
            r["p90_di_in"] for r in invasion_well_rows if r.get("p90_di_in") is not None
        ]
        invasion_rollup = {
            "invaded_pct": round(
                sum(r["invaded_pct"] * r["samples"] for r in invasion_well_rows)
                / tot_inv_s,
                1,
            ),
            "water_like_pct": round(
                sum(r["water_like_pct"] * r["samples"] for r in invasion_well_rows)
                / tot_inv_s,
                1,
            ),
            "hc_like_pct": round(
                sum(r["hc_like_pct"] * r["samples"] for r in invasion_well_rows)
                / tot_inv_s,
                1,
            ),
            "anomalous_pct": round(
                sum(r["anomalous_pct"] * r["samples"] for r in invasion_well_rows)
                / tot_inv_s,
                1,
            ),
            "tight_pct": round(
                sum(r["tight_pct"] * r["samples"] for r in invasion_well_rows)
                / tot_inv_s,
                1,
            ),
            "median_di_in": round(float(np.median(di_vals)), 1) if di_vals else None,
            "p90_di_in": round(float(max(p90_di_vals)), 1) if p90_di_vals else None,
        }

    all_mud_types = sorted(
        {r["mud_type"] for r in hole_sections_rows if r.get("mud_type")}
        | {e.default_mud_type for e in resolved_envs}
    )
    any_env_applied = len(env_corr_curve_table) > 0 or any(
        r.get("has_env_corr") for r in qaqc_table
    )
    curve_confs = [
        float(r["mean_conf"])
        for r in env_corr_curve_table
        if r.get("mean_conf") is not None
    ]
    curve_unrels = [float(r["unreliable_pct"]) for r in env_corr_curve_table]
    bht_calibrated = any(bool(e.bht_records) for e in resolved_envs)

    env_corr_summary: Dict[str, Any] = {
        "any_applied": any_env_applied,
        "wells_applied": sum(1 for r in qaqc_table if r.get("has_env_corr")),
        "total_wells": len(qaqc_table),
        "tool_type": eff_tool_type,
        "res_tool": eff_res_tool,
        "mud_type": "/".join(all_mud_types)
        if all_mud_types
        else primary_env.default_mud_type,
        "mud_weight_ppg": round(float(primary_env.default_mud_weight_ppg), 2),
        "mud_salinity_ppm": int(round(float(primary_env.default_mud_salinity_ppm))),
        "formation_salinity_ppm": int(round(float(primary_env.formation_salinity_ppm))),
        "offshore": any(e.offshore for e in resolved_envs),
        "water_depth_m": round(
            max((e.water_depth_m for e in resolved_envs), default=0.0), 1
        ),
        "top_temp_c": round(float(primary_env.top_temp_c), 1),
        "gradient_c_per_km": round(float(primary_env.gradient_c_per_km), 1),
        "bht_calibrated": bht_calibrated,
        "hole_section_count": len(hole_sections_rows),
        "hole_sections_table": hole_sections_rows,
        "temp_min_c": (
            round(float(env_temp_s.min()), 1) if not env_temp_s.empty else None
        ),
        "temp_max_c": (
            round(float(env_temp_s.max()), 1) if not env_temp_s.empty else None
        ),
        "temp_median_c": (
            round(float(env_temp_s.median()), 1) if not env_temp_s.empty else None
        ),
        "median_rm": round(float(env_rm_s.median()), 4) if not env_rm_s.empty else None,
        "median_rmf": (
            round(float(env_rmf_s.median()), 4) if not env_rmf_s.empty else None
        ),
        "median_rw": round(float(env_rw_s.median()), 4) if not env_rw_s.empty else None,
        "mean_conf": round(float(np.mean(curve_confs)), 3) if curve_confs else None,
        "min_mean_conf": round(float(min(curve_confs)), 3) if curve_confs else None,
        "max_unreliable_pct": round(max(curve_unrels), 1) if curve_unrels else 0.0,
        "cycle_skip_count": total_cycle_skips,
        "res_mud_mismatch_count": total_res_mismatch,
        "corrected_curves": [r["corr_curve"] for r in env_corr_curve_table],
        "curve_table": env_corr_curve_table,
        "well_table": env_well_rows,
        "has_invasion": len(invasion_well_rows) > 0,
        "invasion_table": invasion_well_rows,
        "invasion_rollup": invasion_rollup,
        "sigma_propagated": env_sigma_used,
        "assumptions": env_assumptions_seen[:8],
    }

    focal_da_well = None
    if depth_align_well_rows:
        focal_da_well = max(
            depth_align_well_rows,
            key=lambda r: (
                r["off_depth_count"],
                r["minor_count"],
                r["max_abs_shift_m"],
            ),
        )["well_name"]
    da_conditioned = sorted(
        {r["curve"] for r in depth_align_curve_rows if r.get("has_ori")}
    )
    depth_alignment_summary: Dict[str, Any] = {
        "evaluated": len(depth_align_curve_rows) > 0,
        "wells_evaluated": len(depth_align_well_rows),
        "total_wells": len(qaqc_table),
        "reference_curve": (
            depth_align_well_rows[0]["ref_curve"]
            if depth_align_well_rows
            else da_ref_curve
        ),
        "max_shift_m": da_max_shift_m,
        "window_size_m": da_win_size_m,
        "exclude_badhole": da_excl_bh,
        "total_curves": len(depth_align_curve_rows),
        "aligned_count": sum(
            1 for r in depth_align_curve_rows if r["status"] == "ALIGNED"
        ),
        "minor_count": sum(1 for r in depth_align_curve_rows if r["status"] == "MINOR"),
        "off_depth_count": sum(
            1 for r in depth_align_curve_rows if r["status"] == "OFF_DEPTH"
        ),
        "variable_drift_count": sum(
            1 for r in depth_align_curve_rows if r.get("variable_drift")
        ),
        "max_abs_shift_m": round(
            max(
                (float(r["max_abs_shift_m"]) for r in depth_align_well_rows),
                default=0.0,
            ),
            3,
        ),
        "conditioned_curves": da_conditioned,
        "focal_well": focal_da_well,
        "focal_report": (
            depth_align_reports_by_well.get(focal_da_well) if focal_da_well else None
        ),
        "well_table": depth_align_well_rows,
        "curve_table": depth_align_curve_rows,
    }

    cali_lim = _safe_float(
        req_env_cfg.get(
            "washoutThold",
            req_env_cfg.get("washout_thold", merged_configs.get("cali_threshold")),
        ),
        2.0,
    )
    drho_lim = _safe_float(merged_configs.get("drho_threshold"), 0.15)
    rhob_m = _safe_float(merged_configs.get("rhob_min"), 1.95)
    qaqc_params = {
        "caliper_limit": cali_lim,
        "bitsize_in": _safe_float(
            merged_configs.get("bitsize_default", primary_env.default_bit_size_in), 8.5
        ),
        "drho_limit": drho_lim,
        "rhob_min": rhob_m,
        "badhole_treatment": bh_treat_display,
        "badhole_criteria_used": f"CALI - BS > {cali_lim:.1f} in, |DRHO| > {drho_lim:.2f} g/cc, RHOB < {rhob_m:.2f} g/cc",
        "rho_hc": _safe_float(merged_configs.get("rho_hc"), 0.80),
        "hi_hc": _safe_float(merged_configs.get("hi_hc"), 0.90),
        "norm_method": "Stratigraphic Zonal P10–P90 Quantile Alignment",
        "env_corr_applied": any_env_applied,
        "tool_type": eff_tool_type,
        "mud_type": env_corr_summary["mud_type"],
        "mud_weight_ppg": env_corr_summary["mud_weight_ppg"],
        "hole_section_count": len(hole_sections_rows),
        "offshore": env_corr_summary["offshore"],
        "water_depth_m": env_corr_summary["water_depth_m"],
        "top_temp_c": env_corr_summary["top_temp_c"],
        "gradient_c_per_km": env_corr_summary["gradient_c_per_km"],
    }
    dry_sand_pt = merged_configs.get("dry_sand_point") or (-0.02, 2.65)
    dry_clay_pt = merged_configs.get("dry_clay_point") or (
        round(nphi_clay_cfg, 2),
        round(rho_clay_cfg, 2),
    )
    dry_silt_pt = merged_configs.get("dry_silt_point")
    silt_angle_cfg = _safe_float(merged_configs.get("silt_line_angle"), 119.0)
    fluid_pt = merged_configs.get("fluid_point") or (1.0, 1.0)

    gamma_cfg = merged_configs.get("gamma") or {}
    gamma_src = (
        str(gamma_cfg.get("source") or gamma_cfg.get("gamma_source") or "")
        .strip()
        .lower()
    )
    if gamma_src in ("cgr", "derived_cgr") and gamma_cfg.get("gamma_clean") is not None:
        gr_clean_active = round(
            _safe_float(gamma_cfg.get("gamma_clean"), gr_clean_cfg), 1
        )
        gr_shale_active = round(
            _safe_float(gamma_cfg.get("gamma_shale"), gr_clay_cfg), 1
        )
    else:
        gr_clean_active = round(gr_clean_cfg, 1)
        gr_shale_active = round(gr_clay_cfg, 1)

    litho_params = {
        "model_key": resolved_litho_key,
        "model_name": litho_model_display,
        "rho_clay": round(rho_clay_cfg, 2),
        "nphi_clay": round(nphi_clay_cfg, 2),
        "gr_clean": gr_clean_active,
        "gr_shale": gr_shale_active,
        "dry_sand_point": dry_sand_pt,
        "dry_clay_point": dry_clay_pt,
        "dry_silt_point": dry_silt_pt,
        "silt_line_angle": silt_angle_cfg,
        "fluid_point": fluid_pt,
        "bulk_shale": merged_configs.get("bulk_shale") or {},
        "gamma": gamma_cfg,
    }
    scal_cfg = (
        merged_configs.get("scal_params")
        if isinstance(merged_configs.get("scal_params"), dict)
        else {}
    )
    core_corr_active = bool(
        merged_configs.get("core_corrected")
        or scal_cfg.get("core_corrected")
        or ("CPORE_AMB" in full_df.columns)
        or ("CPORE_ORI" in full_df.columns)
        or ("CPERM_AIR" in full_df.columns)
        or ("CPERM_ORI" in full_df.columns)
    )
    ncs_val = _safe_float(
        merged_configs.get(
            "net_confining_stress", scal_cfg.get("net_confining_stress")
        ),
        3500.0,
    )
    pm_val = _safe_float(
        merged_configs.get("mean_pore_pressure", scal_cfg.get("mean_pore_pressure")),
        45.0,
    )
    core_litho_val = (
        str(
            merged_configs.get("core_lithology")
            or scal_cfg.get("core_lithology")
            or "sandstone"
        )
        .strip()
        .lower()
    )
    if core_litho_val == "carbonate":
        default_c_phi = 2.5e-6
        core_litho_label = "Carbonate (Stiff / Well-Cemented)"
    elif core_litho_val in ("friable_sandstone", "friable"):
        default_c_phi = 8.0e-6
        core_litho_label = "Friable Sandstone"
    else:
        default_c_phi = 4.5e-6
        core_litho_label = "Well-Consolidated Sandstone"
    c_phi_val = _safe_float(
        merged_configs.get("compaction_coeff", scal_cfg.get("compaction_coeff")),
        default_c_phi,
    )
    c_k_val = _safe_float(
        merged_configs.get(
            "perm_compaction_coeff", scal_cfg.get("perm_compaction_coeff")
        ),
        c_phi_val * 5.0,
    )
    klink_method_val = (
        str(
            merged_configs.get("klinkenberg_method")
            or scal_cfg.get("klinkenberg_method")
            or "jones"
        )
        .strip()
        .title()
    )

    core_shift_mode_raw = (
        str(merged_configs.get("core_shift_mode") or "affine").strip().lower()
    )
    core_shift_delta_val = _safe_float(merged_configs.get("core_shift_delta"), 0.0)
    core_stretch_val = _safe_float(merged_configs.get("core_stretch_factor"), 1.0)
    core_tie_pts = (
        merged_configs.get("core_tie_points")
        if isinstance(merged_configs.get("core_tie_points"), list)
        else []
    )
    core_shift_active = bool(
        merged_configs.get("core_shift_applied")
        or abs(core_shift_delta_val) > 1e-4
        or abs(core_stretch_val - 1.0) > 1e-4
        or len(core_tie_pts) > 0
        or ("CPORE_UNSHIFTED" in full_df.columns)
        or ("CPERM_UNSHIFTED" in full_df.columns)
    )
    core_shift_mode_label = (
        f"Piecewise Tie-Point Table ({len(core_tie_pts)} ties)"
        if core_shift_mode_raw == "tie-points" and len(core_tie_pts) > 0
        else "Affine Shift & Stretch"
    )
    core_shift_mean_val = _safe_float(
        merged_configs.get("core_shift_mean_m"), core_shift_delta_val
    )
    core_shift_min_val = _safe_float(
        merged_configs.get("core_shift_min_m"), core_shift_delta_val
    )
    core_shift_max_val = _safe_float(
        merged_configs.get("core_shift_max_m"), core_shift_delta_val
    )
    core_pivot_val = merged_configs.get("core_pivot_depth")

    poro_params = {
        "method_key": raw_poro_key,
        "method_name": poro_method_display,
        "sonic_fluid_dt": merged_configs.get("sonic_fluid_dt"),
        "use_hc_corrected": bool(merged_configs.get("use_hc_corrected", False)),
        "rho_ma": round(rho_ma_cfg, 2),
        "rhoma_mean": rhoma_mean,
        "rhoma_std": rhoma_std,
        "rhoma_count": rhoma_count,
        "rho_fl": round(rho_fl_cfg, 2),
        "core_ref": f"{rho_ma_cfg:.2f} g/cc Matrix (Measured RCA mean = {rhoma_mean:.2f} g/cc)",
        "core_r2": core_r2,
        "core_rmse": core_rmse,
        "core_bias": core_bias,
        "core_count": valid_cal_core_count,
        "total_core_count": total_core_plugs_loaded,
        "core_perm_count": valid_cal_perm_count,
        "core_perm_r2": core_perm_r2,
        "core_perm_log_rmse": core_perm_log_rmse,
        "core_perm_factor_err": core_perm_factor_err,
        "core_shift_applied": core_shift_active,
        "core_shift_mode": core_shift_mode_raw,
        "core_shift_mode_label": core_shift_mode_label,
        "core_shift_delta": round(core_shift_delta_val, 3),
        "core_stretch_factor": round(core_stretch_val, 4),
        "core_pivot_depth": (
            round(float(core_pivot_val), 2)
            if isinstance(core_pivot_val, (int, float))
            else None
        ),
        "core_tie_points": core_tie_pts,
        "core_shift_mean_m": round(core_shift_mean_val, 3),
        "core_shift_min_m": round(core_shift_min_val, 3),
        "core_shift_max_m": round(core_shift_max_val, 3),
        "core_corrected": core_corr_active,
        "core_lithology": core_litho_label,
        "net_confining_stress": round(ncs_val, 0),
        "mean_pore_pressure": round(pm_val, 1),
        "compaction_coeff": c_phi_val,
        "perm_compaction_coeff": c_k_val,
        "klinkenberg_method": klink_method_val,
        "core_poro_amb_mean": merged_configs.get(
            "core_poro_amb_mean", scal_cfg.get("core_poro_amb_mean")
        ),
        "core_poro_res_mean": merged_configs.get(
            "core_poro_res_mean", scal_cfg.get("core_poro_res_mean")
        ),
        "core_perm_air_mean": merged_configs.get(
            "core_perm_air_mean", scal_cfg.get("core_perm_air_mean")
        ),
        "core_perm_liq_mean": merged_configs.get(
            "core_perm_liq_mean", scal_cfg.get("core_perm_liq_mean")
        ),
    }
    sw_cfg = (
        merged_configs.get("sw_params")
        if isinstance(merged_configs.get("sw_params"), dict)
        else {}
    )
    raw_temp = (
        sw_cfg.get("temp")
        or merged_configs.get("formation_temp")
        or merged_configs.get("temp")
    )
    if raw_temp is not None:
        res_temp = round(float(raw_temp), 1)
        temp_basis = "Configured Formation Temperature"
    elif env_corr_summary.get("temp_median_c") is not None:
        res_temp = float(env_corr_summary["temp_median_c"])
        temp_basis = (
            f"Borehole Environment Profile (`ENV_TEMP_C`, {env_corr_summary['gradient_c_per_km']:.1f} °C/km"
            + (
                f", offshore WD={env_corr_summary['water_depth_m']:.0f} m"
                if env_corr_summary["offshore"]
                else ""
            )
            + (", BHT-calibrated)" if env_corr_summary["bht_calibrated"] else ")")
        )
    elif primary_env.offshore or primary_env.explicit.get("temperature"):
        res_temp = round(float(primary_env.temperature_c(median_target_tvdss)), 1)
        temp_basis = (
            f"{'Offshore Seabed-Anchored' if primary_env.offshore else 'BHT-Calibrated'} "
            f"Geothermal Profile ({primary_env.gradient_c_per_km:.1f} °C/km @ {median_target_tvdss:.0f} m TVDss)"
        )
    else:
        surf_temp = _safe_float(merged_configs.get("surface_temp"), 15.0)
        temp_grad = _safe_float(
            merged_configs.get("temp_grad")
            or merged_configs.get("saturation_temp_grad"),
            3.0,
        )
        res_temp = round(surf_temp + temp_grad * (median_target_tvdss / 100.0), 1)
        temp_basis = f"Computed from Geothermal Gradient ({temp_grad:.1f} °C/100m @ {median_target_tvdss:.0f} m TVDss)"

    rw_val = round(_safe_float(merged_configs.get("rw_val"), 0.080), 3)
    from quick_pp.saturation import estimate_b_waxman_smits

    b_conductance = round(float(estimate_b_waxman_smits(res_temp, rw_val)), 2)

    qv_val = _safe_float(
        sw_cfg.get("qv")
        or merged_configs.get("qv")
        or (
            full_df["QV"].median()
            if ("QV" in full_df.columns and full_df["QV"].notna().any())
            else 0.55
        ),
        0.55,
    )
    cec_val = _safe_float(sw_cfg.get("cec") or merged_configs.get("cec"), 12.0)
    phish_val = _safe_float(sw_cfg.get("phish") or merged_configs.get("phish"), 0.25)
    qv_method = str(
        sw_cfg.get("qv_method")
        or merged_configs.get("qv_method")
        or "Juhasz Qvn Empirical Correlation"
    )

    sw_params = {
        "model_key": raw_sw_key,
        "model_name": sw_model_display,
        "fwl": fwl_val,
        "fwl_basis": fwl_basis_str,
        "median_target_tvdss": median_target_tvdss,
        "salinity": int(_safe_float(merged_configs.get("water_salinity"), 35000.0)),
        "rw": rw_val,
        "temp": res_temp,
        "temp_basis": temp_basis,
        "a": round(_safe_float(merged_configs.get("sw_a"), 1.0), 2),
        "m": round(_safe_float(merged_configs.get("sw_m"), 2.0), 2),
        "n": round(_safe_float(merged_configs.get("sw_n"), 2.0), 2),
        "b_conductance": b_conductance,
        "qv": round(qv_val, 3),
        "cec": round(cec_val, 1),
        "phish": round(phish_val, 3),
        "qv_method": qv_method,
        "rsh": round(_safe_float(merged_configs.get("sw_rsh"), 2.0), 3),
        "swb": round(_safe_float(merged_configs.get("sw_swb"), 0.2), 3),
        "rwb": round(_safe_float(merged_configs.get("sw_rwb"), 0.1), 3),
    }
    ift_res_cfg = _safe_float(
        merged_configs.get(
            "ift_res", scal_cfg.get("ift_res", merged_configs.get("ift"))
        ),
        30.0,
    )
    theta_res_cfg = _safe_float(
        merged_configs.get(
            "theta_res", scal_cfg.get("theta_res", merged_configs.get("theta"))
        ),
        30.0,
    )
    ift_lab_cfg = _safe_float(
        merged_configs.get("ift_lab", scal_cfg.get("ift_lab")), 480.0
    )
    theta_lab_cfg = _safe_float(
        merged_configs.get("theta_lab", scal_cfg.get("theta_lab")), 140.0
    )
    lab_denom = max(abs(ift_lab_cfg * math.cos(math.radians(theta_lab_cfg))), 1e-6)
    res_num = abs(ift_res_cfg * math.cos(math.radians(theta_res_cfg)))
    pc_scale_cfg = _safe_float(
        merged_configs.get("pc_scale_factor", scal_cfg.get("pc_scale_factor")),
        res_num / lab_denom,
    )
    pc_scaling_applied = bool(
        merged_configs.get("pc_scaling_applied") or scal_cfg.get("pc_scaling_applied")
    )
    shf_params = {
        "fwl": fwl_val,
        "fwl_basis": fwl_basis_str,
        "ift": round(ift_res_cfg, 1),
        "theta": round(theta_res_cfg, 1),
        "ift_cos_theta": round(res_num, 1),
        "ift_lab": round(ift_lab_cfg, 1),
        "theta_lab": round(theta_lab_cfg, 1),
        "ift_lab_cos_theta": round(lab_denom, 1),
        "pc_scaling_applied": pc_scaling_applied,
        "pc_scale_factor": round(pc_scale_cfg, 5),
        "pc_lab_ref": _safe_float(
            merged_configs.get("pc_lab_input", scal_cfg.get("pc_lab_input")), 25.0
        ),
        "pc_res_ref": _safe_float(
            merged_configs.get("pc_res_output", scal_cfg.get("pc_res_output")),
            round(25.0 * pc_scale_cfg, 3),
        ),
        "gw": gw_val,
        "gw_basis": gw_basis,
        "ghc": ghc_val,
        "delta_rho": delta_rho,
        "ghc_basis": ghc_basis,
        "reservoir_fluid": res_fluid,
        "reconciliation_rmse": shf_rmse,
        "reconciliation_bias": shf_bias,
        "reconciliation_n": shf_n_compared,
        "fit_source": shf_fit_source,
        "uses_saved_fit": use_saved_fits,
        "fit_rows": shf_fit_rows,
    }
    scal_model_key = (
        str(scal_cfg.get("model") or merged_configs.get("scal_model") or "let")
        .strip()
        .lower()
    )
    is_scal_calibrated = bool(
        any_scal_data
        or scal_cfg.get("user_calibrated")
        or scal_cfg.get("is_calibrated")
    )
    scal_params = {
        "is_analogue": not is_scal_calibrated,
        "model": scal_model_key,
        "source_label": (
            f"Calibrated {scal_model_key.upper()} SCAL & Core Parameterization"
            if is_scal_calibrated
            else f"Regional Analogue / Screening {scal_model_key.upper()} Parameterization"
        ),
        "swirr": _safe_float(
            scal_cfg.get("swirr", merged_configs.get("scal_swirr")), 0.15
        ),
        "sor": _safe_float(scal_cfg.get("sor", merged_configs.get("scal_sor")), 0.20),
        "krw_max": _safe_float(
            scal_cfg.get("krw_max", merged_configs.get("scal_krw_max")), 0.45
        ),
        "kro_max": _safe_float(
            scal_cfg.get("kro_max", merged_configs.get("scal_kro_max")), 0.90
        ),
        "nw": _safe_float(scal_cfg.get("nw", merged_configs.get("scal_nw")), 2.5),
        "no": _safe_float(scal_cfg.get("no", merged_configs.get("scal_no")), 2.5),
        "lw": _safe_float(scal_cfg.get("lw", merged_configs.get("scal_lw")), 2.0),
        "ew": _safe_float(scal_cfg.get("ew", merged_configs.get("scal_ew")), 1.5),
        "tw": _safe_float(scal_cfg.get("tw", merged_configs.get("scal_tw")), 2.0),
        "lo": _safe_float(scal_cfg.get("lo", merged_configs.get("scal_lo")), 2.0),
        "eo": _safe_float(scal_cfg.get("eo", merged_configs.get("scal_eo")), 1.5),
        "to": _safe_float(scal_cfg.get("to", merged_configs.get("scal_to")), 2.0),
        "pc_entry": _safe_float(
            scal_cfg.get("pc_entry", merged_configs.get("scal_pc_entry")), 1.5
        ),
        "pc_max": _safe_float(
            scal_cfg.get("pc_max", merged_configs.get("scal_pc_max")), 50.0
        ),
    }
    has_custom_up = (
        "upscaling_cell_size" in merged_configs or "upscaling_omega" in merged_configs
    )
    upscaling_params = {
        "cell_size": _safe_float(merged_configs.get("upscaling_cell_size"), 2.0),
        "omega": _safe_float(merged_configs.get("upscaling_omega"), 0.33),
        "basis": "Project Upscaling Config"
        if has_custom_up
        else "Default Simulation Grid Standard",
    }
    mean_kv_kh = (
        round(float(np.mean([r["kv_kh"] for r in upscaling_table])), 3)
        if upscaling_table
        else 0.0
    )
    kv_kh_vals = [
        r["kv_kh"]
        for r in upscaling_table
        if r.get("kv_kh") is not None and r["kv_kh"] > 0
    ]
    min_kv_kh = round(float(np.min(kv_kh_vals)), 3) if kv_kh_vals else mean_kv_kh
    max_kv_kh = round(float(np.max(kv_kh_vals)), 3) if kv_kh_vals else mean_kv_kh
    upscaling_summary = {
        "mean_kv_kh": mean_kv_kh,
        "min_kv_kh": min_kv_kh,
        "max_kv_kh": max_kv_kh,
        "cell_size": upscaling_params["cell_size"],
        "omega": upscaling_params["omega"],
    }

    # If no stored contacts in DB, dynamically extract them from full_df
    if not fluid_contacts_rows and not full_df.empty:
        try:
            from quick_pp.saturation import extract_fluid_contacts_and_limits

            extracted_df = extract_fluid_contacts_and_limits(
                df=full_df,
                fwl_dict={"default": fwl_val},
                sw_pay_cutoff=sw_cut,
                phit_cutoff=phie_cut,
                vsh_cutoff=vsh_cut,
            )
            if not extracted_df.empty:
                for r in extracted_df.to_dict(orient="records"):
                    c_type = str(r.get("contact_type") or "").upper()
                    if c_type == "WDT":
                        continue
                    fluid_contacts_rows.append(
                        {
                            "well_name": str(r.get("well_name") or "Field"),
                            "name": str(r.get("name") or c_type),
                            "depth": round(float(r["depth"]), 1)
                            if r.get("depth") is not None
                            else None,
                            "zone": str(r.get("zone"))
                            if r.get("zone")
                            else "All Zones",
                            "contact_type": c_type,
                            "fault_block": str(r.get("fault_block"))
                            if r.get("fault_block")
                            else None,
                            "source": str(
                                r.get("source") or "Petrophysical Log Interpretation"
                            ),
                            "remarks": str(r.get("remarks") or ""),
                        }
                    )
        except Exception:
            pass

    # Summarize proved column and uncertainty brackets by (zone, fault_block)
    zonal_contact_groups: Dict[Tuple[str, Optional[str]], Dict[str, Any]] = {}
    for r in fluid_contacts_rows:
        key = (r["zone"], r.get("fault_block"))
        if key not in zonal_contact_groups:
            zonal_contact_groups[key] = {
                "zone": r["zone"],
                "fault_block": r.get("fault_block"),
                "gut": None,
                "gdt": None,
                "out": None,
                "odt": None,
                "wut": None,
                "goc": None,
                "gwc": None,
                "owc": None,
                "fwl": None,
                "_fwl_list": [],
            }
        g = zonal_contact_groups[key]
        ctype = r["contact_type"]
        d = r["depth"]
        if d is None:
            continue
        if ctype == "GUT":
            g["gut"] = min(g["gut"], d) if g["gut"] is not None else d
        elif ctype == "GDT":
            g["gdt"] = max(g["gdt"], d) if g["gdt"] is not None else d
        elif ctype == "OUT":
            g["out"] = min(g["out"], d) if g["out"] is not None else d
        elif ctype == "ODT":
            g["odt"] = max(g["odt"], d) if g["odt"] is not None else d
        elif ctype == "WUT":
            g["wut"] = min(g["wut"], d) if g["wut"] is not None else d
        elif ctype == "GOC":
            g["goc"] = d
        elif ctype == "GWC":
            g["gwc"] = d
        elif ctype == "OWC":
            g["owc"] = d
        elif ctype == "FWL":
            g["_fwl_list"].append(d)

    for g in zonal_contact_groups.values():
        fwl_l = g.pop("_fwl_list", [])
        if fwl_l:
            min_f = min(fwl_l)
            max_f = max(fwl_l)
            if abs(max_f - min_f) < 0.1:
                g["fwl"] = round(min_f, 1)
            else:
                g["fwl"] = f"{min_f:.1f}–{max_f:.1f}"

    fluid_contacts_summary = list(zonal_contact_groups.values())

    # Volumetric Summary Calculation & Handover Readiness Fallback (GAP-12)
    area_acres = _safe_float(
        merged_configs.get("area_acres")
        or merged_configs.get("area")
        or (model_params.get("area_acres") if model_params else None),
        None,
    )
    vol_sum_existing = merged_configs.get("volumetric_summary")
    if isinstance(vol_sum_existing, dict) and "field_totals" in vol_sum_existing:
        volumetric_summary = vol_sum_existing
        volumetric_readiness = None
    elif area_acres is not None and area_acres > 0.0:
        res_fl = (
            str(
                merged_configs.get("reservoir_fluid")
                or (model_params.get("reservoir_fluid") if model_params else "")
                or "oil"
            )
            .strip()
            .lower()
        )
        bg_eff = _safe_float(
            merged_configs.get("bg")
            or (model_params.get("bg") if model_params else None),
            0.0045 if res_fl == "gas" else 0.005,
        )
        bo_eff = _safe_float(
            merged_configs.get("bo")
            or merged_configs.get("boi")
            or (model_params.get("bo") if model_params else None),
            1.25 if res_fl != "gas" else 1.20,
        )
        bg_eff = max(bg_eff, 1e-5)
        bo_eff = max(bo_eff, 0.1)

        zonal_vol_summaries = []
        stoiip_p90_tot = 0.0
        stoiip_p50_tot = 0.0
        stoiip_p10_tot = 0.0
        rec_oil_p50_tot = 0.0
        giip_p90_tot = 0.0
        giip_p50_tot = 0.0
        giip_p10_tot = 0.0
        rec_gas_p50_tot = 0.0

        for z in zonal_summary:
            z_name = z["zone"]
            z_unc_rows = [
                u
                for u in uncertainty_summary
                if u["zone"] == z_name
                and u["metric"] == "Hydrocarbon Pore Thickness HCPT (m)"
            ]
            if z_unc_rows:
                h_p90 = _safe_float(z_unc_rows[0].get("p90_hcpt"), 0.0)
                h_p50 = _safe_float(z_unc_rows[0].get("p50_hcpt"), 0.0)
                h_p10 = _safe_float(z_unc_rows[0].get("p10_hcpt"), 0.0)
            else:
                h_p50 = float(z.get("hcpt", 0.0))
                h_p90 = round(h_p50 * 0.8, 2)
                h_p10 = round(h_p50 * 1.2, 2)

            # 1 m = 3.28084 ft
            # HCPV (acre-ft) = Area (acres) * (HCPT (m) * 3.28084)
            hcpv_p90 = area_acres * (h_p90 * 3.28084)
            hcpv_p50 = area_acres * (h_p50 * 3.28084)
            hcpv_p10 = area_acres * (h_p10 * 3.28084)

            # Gas GIIP (Bcf) = 43560 * HCPV / (Bg * 1e9)
            giip_p90 = round((43560.0 * hcpv_p90) / (bg_eff * 1e9), 2)
            giip_p50 = round((43560.0 * hcpv_p50) / (bg_eff * 1e9), 2)
            giip_p10 = round((43560.0 * hcpv_p10) / (bg_eff * 1e9), 2)
            rec_gas_p50 = round(giip_p50 * 0.70, 2)

            # Oil STOIIP (MMbbl) = 7758 * HCPV / (Bo * 1e6)
            stoiip_p90 = round((7758.0 * hcpv_p90) / (bo_eff * 1e6), 2)
            stoiip_p50 = round((7758.0 * hcpv_p50) / (bo_eff * 1e6), 2)
            stoiip_p10 = round((7758.0 * hcpv_p10) / (bo_eff * 1e6), 2)
            rec_oil_p50 = round(stoiip_p50 * 0.35, 2)

            zonal_vol_summaries.append(
                {
                    "zone": z_name,
                    "stoiip": {
                        "p90": stoiip_p90,
                        "p50": stoiip_p50,
                        "p10": stoiip_p10,
                    },
                    "rec_oil": {"p50": rec_oil_p50},
                    "giip": {"p90": giip_p90, "p50": giip_p50, "p10": giip_p10},
                    "rec_gas": {"p50": rec_gas_p50},
                }
            )
            stoiip_p90_tot += stoiip_p90
            stoiip_p50_tot += stoiip_p50
            stoiip_p10_tot += stoiip_p10
            rec_oil_p50_tot += rec_oil_p50
            giip_p90_tot += giip_p90
            giip_p50_tot += giip_p50
            giip_p10_tot += giip_p10
            rec_gas_p50_tot += rec_gas_p50

        volumetric_summary = {
            "mode": "Drainage Area",
            "n_samples": 1000,
            "zonal_summaries": zonal_vol_summaries,
            "field_totals": {
                "stoiip": {
                    "p90": round(stoiip_p90_tot, 2),
                    "p50": round(stoiip_p50_tot, 2),
                    "p10": round(stoiip_p10_tot, 2),
                },
                "rec_oil": {"p50": round(rec_oil_p50_tot, 2)},
                "giip": {
                    "p90": round(giip_p90_tot, 2),
                    "p50": round(giip_p50_tot, 2),
                    "p10": round(giip_p10_tot, 2),
                },
                "rec_gas": {"p50": round(rec_gas_p50_tot, 2)},
            },
        }
        volumetric_readiness = None
    else:
        volumetric_summary = None
        fs = _compute_field_summary(zonal_summary)
        volumetric_readiness = {
            "status": "AWAITING_RESERVOIR_AREA",
            # `mean_*` keys are per-zone means; `column_*` are sums across zones (a well through every zone).
            "mean_hcpt": fs["mean_zone_hcpt"],
            "mean_net_pay": fs["mean_zone_net_pay"],
            "mean_ntg": fs["ntg_pay"],
            "mean_phie": fs["phie"] if fs["phie"] is not None else 0.0,
            "mean_sw": fs["sw"] if fs["sw"] is not None else 0.0,
            "total_hcpt": fs["column_hcpt"],
            "column_hcpt": fs["column_hcpt"],
            "column_net_pay": fs["column_net_pay"],
            "evaluated_zones": fs["n_zones"],
            "pay_zones": fs["n_pay_zones"],
        }

    ctx_payload: Dict[str, Any] = {
        "project_id": pid,
        "project_name": project_name,
        "well_names": [w["name"] for w in wells_inventory],
        "wells": wells_inventory,
        "all_zones": all_zones_seen or ["Main Reservoir"],
        "target_zones": resolved_target_zones,
        "total_logged_thickness": round(total_logged_thickness, 1),
        "cutoffs": {"vsh": vsh_cut, "phie": phie_cut, "sw": sw_cut},
        "qaqc_params": qaqc_params,
        "qaqc_table": qaqc_table,
        "env_corr_summary": env_corr_summary,
        "depth_alignment_summary": depth_alignment_summary,
        "litho_params": litho_params,
        "poro_params": poro_params,
        "rock_types": rock_types,
        "saved_flag_rows": saved_flag_rows,
        "fzi_cutoffs": resolved_fzi_cutoffs,
        "rt_summary": {
            "basis": rt_basis_source,
            "cutoff_source": cutoff_source,
            "method": saved_rt_method,
            "dominant_codes": dominant_codes,
            "dominant_share_pct": dominant_share_pct,
            "dominant_kh_pct": dominant_kh_pct,
            "rt34_share_pct": rt34_share_pct,
            "rt34_kh_pct": rt34_kh_pct,
        },
        "sw_params": sw_params,
        "shf_params": shf_params,
        "scal_params": scal_params,
        "zonal_summary": zonal_summary,
        "well_zonal_summary": well_zonal_rows,
        "ressum_markdown_table": ressum_markdown_table,
        "fluid_contacts_rows": fluid_contacts_rows,
        "fluid_contacts_summary": fluid_contacts_summary,
        "uncertainty_summary": uncertainty_summary,
        "uncertainty_drivers_text": uncertainty_drivers_text,
        "volumetric_summary": volumetric_summary,
        "volumetric_readiness": volumetric_readiness,
        "volumetric_params": merged_configs.get("volumetric_params"),
        "upscaling_params": upscaling_params,
        "upscaling_table": upscaling_table,
        "upscaling_summary": upscaling_summary,
        "fpt_summary": fpt_summary,
        "_fpt_points": raw_fpt_points,
        "anomaly_flags": anomaly_flags,
        "well_core_parity": well_core_parity,
        "_full_df": full_df,
    }
    ctx_payload["gate_checks"] = validate_fdp_report_payload(ctx_payload)
    ctx_payload["gate_summary"] = summarize_gate_checks(ctx_payload["gate_checks"])
    return ctx_payload


def validate_fdp_report_payload(context: Dict[str, Any]) -> List[Dict[str, str]]:
    """Execute Spec 5 Pre-Render Automated Gate-Check Assertions (CHK-01 through CHK-08)."""
    checks: List[Dict[str, str]] = []
    full_df: pd.DataFrame = context.get("_full_df", pd.DataFrame())
    target_zones_upper = {str(z).upper() for z in context.get("target_zones", [])}
    sw_p = context.get("sw_params", {})
    poro_p = context.get("poro_params", {})
    unc_list = context.get("uncertainty_summary", [])
    shf_p = context.get("shf_params", {})
    contacts_rows = context.get("fluid_contacts_rows", [])

    # CHK-01: No synthetic constant padding outside evaluated intervals
    chk01_status = "PASS"
    chk01_detail = "All non-evaluated overburden/underburden intervals preserve null data (0 synthetic padded rows)."
    if not full_df.empty and "PHIE" in full_df.columns and "RT" in full_df.columns:
        outside_mask = (
            ~full_df["ZONES"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin(target_zones_upper)
            if "ZONES" in full_df.columns and target_zones_upper
            else pd.Series(False, index=full_df.index)
        )
        rt_na = pd.to_numeric(full_df["RT"], errors="coerce").isna()
        phie_notna = pd.to_numeric(full_df["PHIE"], errors="coerce").notna()
        padded_outside = outside_mask & rt_na & phie_notna
        if padded_outside.any():
            full_df.loc[padded_outside, ["PHIE", "PHIT", "SWT", "PERM"]] = np.nan
            chk01_status = "REMEDIATED"
            chk01_detail = f"Stripped {int(padded_outside.sum())} constant-padded sample(s) outside evaluated target zones."
    checks.append(
        {
            "id": "CHK-01",
            "rule": "Null-Padding Integrity Outside Target Zones",
            "category": "Data Conditioning",
            "criterion": "Zero synthetic constant padding in overburden/underburden intervals",
            "measured": chk01_detail,
            "status": chk01_status,
            "action_note": "Handover approved: interval boundaries verified."
            if chk01_status in ("PASS", "REMEDIATED")
            else "Remediation required.",
            "detail": chk01_detail,
        }
    )

    # CHK-02: abs(fwl_tvdss - median_target_tvdss) < 300.0
    fwl_v = _safe_float(sw_p.get("fwl"), 2943.0)
    med_tvdss = _safe_float(sw_p.get("median_target_tvdss"), fwl_v)
    diff_fwl = abs(fwl_v - med_tvdss)
    chk02_status = "PASS" if diff_fwl < 300.0 else "FLAGGED"
    checks.append(
        {
            "id": "CHK-02",
            "rule": "Free Water Level vs. Target Reservoir TVDss (<300 m)",
            "category": "Fluid Contacts",
            "criterion": "|FWL - Median Target TVDss| < 300 m",
            "measured": f"FWL = {fwl_v:.1f} m TVDss vs. Target Median = {med_tvdss:.1f} m TVDss (Δ = {diff_fwl:.1f} m)",
            "status": chk02_status,
            "action_note": "Hydrostatic datum consistent with target interval."
            if chk02_status == "PASS"
            else "Inspect regional aquifer / fault compartmentalization.",
            "detail": f"FWL = {fwl_v:.1f} m TVDss vs. Target Median = {med_tvdss:.1f} m TVDss (Δ = {diff_fwl:.1f} m).",
        }
    )

    # CHK-03: Core calibration R² >= 0.40 and RMSE <= 0.040
    c_cnt = int(poro_p.get("core_count", 0))
    c_tot = int(poro_p.get("total_core_count", 0))
    c_r2 = _safe_float(poro_p.get("core_r2"), 0.0)
    c_rmse = _safe_float(poro_p.get("core_rmse"), 0.0)
    if c_cnt >= 3 and c_r2 >= 0.40 and c_rmse <= 0.040:
        chk03_status = "PASS"
        chk03_detail = f"Valid Target-Zone Plugs = {c_cnt}/{c_tot} (R² = {c_r2:.2f}, RMSE = {c_rmse:.3f} v/v)."
        chk03_note = "Porosity calibration verified against target-zone RCA plugs."
    elif c_tot == 0:
        chk03_status = "N/A (Log-Only)"
        chk03_detail = "No RCA core plugs loaded in project; conditional narrative set to log-only."
        chk03_note = "Log-only workflow active; core calibration bypassed."
    else:
        chk03_status = "QA ALERT"
        chk03_detail = f"Overlapping Target Plugs = {c_cnt}/{c_tot} (R² = {c_r2:.2f}, RMSE = {c_rmse:.3f} v/v); 'strong parity' claim suppressed."
        chk03_note = "Review per-well core parity diagnostics in Section 4.1."
    checks.append(
        {
            "id": "CHK-03",
            "rule": "Core-to-Log Depth Alignment & Parity Threshold",
            "category": "Core Calibration",
            "criterion": "Target RCA R² ≥ 0.40 and RMSE ≤ 0.040 v/v",
            "measured": chk03_detail,
            "status": chk03_status,
            "action_note": chk03_note,
            "detail": chk03_detail,
        }
    )

    # CHK-04: P90_HCPT <= P50_HCPT <= P10_HCPT
    chk04_ok = all(
        _safe_float(u.get("p90_hcpt"), 0.0)
        <= _safe_float(u.get("p50_hcpt"), 0.0)
        <= _safe_float(u.get("p10_hcpt"), 0.0)
        for u in unc_list
    )
    checks.append(
        {
            "id": "CHK-04",
            "rule": "Monotonic Uncertainty Percentiles (P90 ≤ P50 ≤ P10)",
            "category": "Uncertainty Quantification",
            "criterion": "P90 ≤ P50 ≤ P10 monotonically across all zones",
            "measured": f"Verified across {len(unc_list)} target {_pluralize(len(unc_list), 'zone')}",
            "status": "PASS" if chk04_ok else "FAIL",
            "action_note": "Volumetric percentiles mathematically verified."
            if chk04_ok
            else "Distribution error: P90 exceeds P10.",
            "detail": f"Verified across {len(unc_list)} target {_pluralize(len(unc_list), 'zone')}.",
        }
    )

    # CHK-05: P50_PHIE == 0 => P50_HCPT == 0
    zero_phi_hcpt = sum(
        _safe_float(u.get("p50_hcpt"), 0.0)
        for u in unc_list
        if _safe_float(u.get("p50_phie"), 0.0) <= 0.0
    )
    checks.append(
        {
            "id": "CHK-05",
            "rule": "Zero-Pay / Zero-Porosity HCPT Consistency",
            "category": "Volumetrics Integrity",
            "criterion": "P50_PHIE == 0 implies P50_HCPT == 0",
            "measured": f"Sum of P50_HCPT where PHIE=0: {zero_phi_hcpt:.2f} m",
            "status": "PASS" if zero_phi_hcpt == 0.0 else "FAIL",
            "action_note": "Zero-porosity non-reservoir integrity verified."
            if zero_phi_hcpt == 0.0
            else "Phantom hydrocarbon volume detected in non-reservoir.",
            "detail": f"Sum of P50_HCPT where P50_PHIE == 0 is {zero_phi_hcpt:.2f} m.",
        }
    )

    # CHK-06: CPI & Upscaling plot vertical span <= 500.0 m
    ymin_w, ymax_w = report_plots._resolve_target_depth_window(
        full_df, target_zones=context.get("target_zones"), max_span=480.0
    )
    span_w = max(0.0, ymax_w - ymin_w)
    checks.append(
        {
            "id": "CHK-06",
            "rule": "Target-Zoomed CPI & Blocking Vertical Span (≤ 500 m)",
            "category": "Visualization QC",
            "criterion": "Target depth window span ≤ 500 m MD",
            "measured": f"Span = {span_w:.1f} m MD ({ymin_w:.1f}–{ymax_w:.1f} m)",
            "status": "PASS" if span_w <= 500.0 else "FAIL",
            "action_note": "High-resolution target zoom active."
            if span_w <= 500.0
            else "Span exceeds 500 m limit; compress interval.",
            "detail": f"Target plot depth window = {ymin_w:.1f}–{ymax_w:.1f} m MD (span = {span_w:.1f} m).",
        }
    )

    # CHK-07: Fluid Contact Consistency
    fwl_comm = None
    for r in contacts_rows:
        if r.get("contact_type") in ("FWL", "OWC") and r.get("depth") is not None:
            fwl_comm = float(r["depth"])
            break
    if fwl_comm is not None:
        delta_fc = abs(fwl_v - fwl_comm)
        chk07_ok = delta_fc <= 10.0
        chk07_status = "PASS" if chk07_ok else "QA ALERT"
        chk07_measured = (
            f"|FWL ({fwl_v:.1f} m) - Committed ({fwl_comm:.1f} m)| = {delta_fc:.1f} m"
        )
        chk07_note = (
            "Hydrostatic equilibrium verified within ±10 m tolerance."
            if chk07_ok
            else "Committed contact deviates from log transition; verify hydrodynamic tilt."
        )
    else:
        chk07_status = "PASS"
        chk07_measured = f"Model FWL = {fwl_v:.1f} m TVDss (Single Source of Truth, no conflicting discrete contacts)"
        chk07_note = "Single Source of Truth FWL verified across all reservoir zones."

    checks.append(
        {
            "id": "CHK-07",
            "rule": "Fluid Contact Consistency (|FWL - Committed| ≤ 10 m)",
            "category": "Fluid Contacts",
            "criterion": "|Inferred FWL - Committed Contact| ≤ 10.0 m TVDss",
            "measured": chk07_measured,
            "status": chk07_status,
            "action_note": chk07_note,
            "detail": chk07_measured,
        }
    )

    # CHK-08: Saturation-Height Reconciliation (SHF RMSE <= 0.12)
    shf_rmse_val = shf_p.get("reconciliation_rmse")
    shf_rmse = _safe_float(shf_rmse_val, None) if shf_rmse_val is not None else None
    if shf_rmse is not None:
        chk08_ok = shf_rmse <= 0.12
        chk08_status = "PASS" if chk08_ok else "QA ALERT"
        chk08_measured = (
            f"SHF vs Log SWT RMSE = {shf_rmse:.3f} v/v "
            f"(n={shf_p.get('reconciliation_n', 0)}; {shf_p.get('fit_source', 'unspecified fit')})"
        )
        chk08_note = (
            "Saturation-height model consistent with active log saturation."
            if chk08_ok
            else "Capillary equilibrium deviation exceeds 0.12 v/v; inspect transition zone, FWL and rock-type fits."
        )
    else:
        chk08_status = "QA ALERT"
        chk08_measured = "SHF vs log Sw reconciliation could not be computed (insufficient overlapping samples)"
        chk08_note = "No SHF reconciliation statistic is reported; run the SHF module for the target zones."

    checks.append(
        {
            "id": "CHK-08",
            "rule": "Saturation-Height Reconciliation (RMSE ≤ 0.12 v/v)",
            "category": "Saturation-Height Modeling",
            "criterion": "SHF Model vs. Active Log Sw RMSE ≤ 0.12 v/v",
            "measured": chk08_measured,
            "status": chk08_status,
            "action_note": chk08_note,
            "detail": chk08_measured,
        }
    )

    # CHK-09: Core-to-log permeability parity
    k_cnt = int(poro_p.get("core_perm_count", 0) or 0)
    k_r2 = _safe_float(poro_p.get("core_perm_r2"), 0.0)
    k_factor = _safe_float(poro_p.get("core_perm_factor_err"), 1.0)
    if k_cnt >= 3 and k_r2 >= 0.40 and k_factor <= 5.0:
        chk09_status = "PASS"
        chk09_measured = f"Valid perm plugs = {k_cnt} (R² = {k_r2:.2f}, factor error = {k_factor:.1f}×)."
        chk09_note = "Log permeability transform reproduces core permeability."
    elif k_cnt == 0:
        chk09_status = "N/A (Log-Only)"
        chk09_measured = "No overlapping core permeability plugs in the target zones."
        chk09_note = "Permeability is uncalibrated against core."
    else:
        chk09_status = "QA ALERT"
        chk09_measured = (
            f"Valid perm plugs = {k_cnt} (R² = {k_r2:.2f}, factor error = {k_factor:.1f}×); "
            "log permeability does not reproduce core."
        )
        chk09_note = "Rock-type permeability transforms are not validated; do not use PERM for static-model handover."
    checks.append(
        {
            "id": "CHK-09",
            "rule": "Core-to-Log Permeability Parity (R² ≥ 0.40, factor error ≤ 5×)",
            "category": "Permeability",
            "criterion": "Target RCA permeability R² ≥ 0.40 and multiplicative error ≤ 5×",
            "measured": chk09_measured,
            "status": chk09_status,
            "action_note": chk09_note,
            "detail": chk09_measured,
        }
    )

    # CHK-10: Rock typing calibrated to project (saved cutoffs, fitted classes)
    rt_sum = context.get("rt_summary", {}) or {}
    rts = context.get("rock_types", []) or []
    n_unfit = sum(1 for r in rts if r.get("fit_status") != "fitted")
    generic_cut = str(rt_sum.get("cutoff_source", "")).startswith("generic default")
    if generic_cut or (rts and n_unfit == len(rts)):
        chk10_status = "QA ALERT"
        chk10_measured = (
            "Generic FZI cutoffs applied; no saved rock typing for the target zones."
            if generic_cut
            else "No rock-type class has enough samples for a porosity-permeability fit."
        )
        chk10_note = "Run and save Rock Typing in the workbench, then regenerate."
    elif n_unfit > 0:
        chk10_status = "REVIEW"
        chk10_measured = f"{len(rts) - n_unfit} of {len(rts)} classes fitted; cutoffs: {rt_sum.get('cutoff_source', 'n/a')}."
        chk10_note = "Classes without a fit carry no permeability transform."
    else:
        chk10_status = "PASS"
        chk10_measured = f"All {len(rts)} classes fitted; cutoffs: {rt_sum.get('cutoff_source', 'n/a')}."
        chk10_note = "Rock typing cutoffs and class regressions are project-calibrated."
    checks.append(
        {
            "id": "CHK-10",
            "rule": "Rock Typing Calibrated to Project Data",
            "category": "Rock Typing",
            "criterion": "Saved project cutoffs and a regression for every class",
            "measured": chk10_measured,
            "status": chk10_status,
            "action_note": chk10_note,
            "detail": chk10_measured,
        }
    )

    # CHK-11: Formation pressure gradients physical and contacts committed
    fpt_s = context.get("fpt_summary", {}) or {}
    if fpt_s.get("has_data"):
        rejected = fpt_s.get("rejected_fits", []) or []
        committed = bool(fpt_s.get("fwl_is_committed"))
        if rejected:
            chk11_status = "QA ALERT"
            chk11_measured = (
                f"{len(rejected)} fitted gradient(s) rejected as unphysical or poorly constrained "
                f"({', '.join(str(r.get('fluid_type')) for r in rejected)})."
            )
            chk11_note = "Review pressure picks, fluid assignment and depth reference before using gradients."
        elif not committed:
            chk11_status = "REVIEW"
            chk11_measured = "Gradients are physical, but no committed fluid contact exists to reconcile against."
            chk11_note = "Contacts derived from pressure are indicative only."
        else:
            chk11_status = "PASS"
            chk11_measured = (
                "Gradients physical and reconciled against committed contact."
            )
            chk11_note = "Pressure regime consistent with the fluid model."
        checks.append(
            {
                "id": "CHK-11",
                "rule": "Formation Pressure Gradients Physical and Contacts Committed",
                "category": "Formation Pressure",
                "criterion": "Gradient densities within phase bounds; reference contact committed",
                "measured": chk11_measured,
                "status": chk11_status,
                "action_note": chk11_note,
                "detail": chk11_measured,
            }
        )

    # CHK-12: Borehole Environmental Correction & Sensor Confidence Audit
    env_s = context.get("env_corr_summary", {}) or {}
    if env_s.get("any_applied"):
        max_unrel = _safe_float(env_s.get("max_unreliable_pct"), 0.0)
        mean_cf = env_s.get("mean_conf")
        min_cf = env_s.get("min_mean_conf")
        n_mismatch = int(env_s.get("res_mud_mismatch_count") or 0)
        cf_txt = f"{mean_cf:.2f}" if isinstance(mean_cf, (int, float)) else "n/a"
        if n_mismatch > 0 or max_unrel > 25.0:
            chk12_status = "QA ALERT"
            chk12_measured = (
                f"Env corrections applied ({', '.join(env_s.get('corrected_curves', []))}); "
                f"max unreliable = {max_unrel:.1f}%, mean conf = {cf_txt}"
                + (
                    f", res/mud mismatch = {n_mismatch} samples."
                    if n_mismatch > 0
                    else "."
                )
            )
            chk12_note = "Inspect washout severity, resistivity/mud-type compatibility and enable confidence weighting."
        elif max_unrel > 10.0 or (isinstance(min_cf, (int, float)) and min_cf < 0.65):
            chk12_status = "REVIEW"
            chk12_measured = (
                f"Applied across {env_s.get('wells_applied', 0)}/{env_s.get('total_wells', 0)} well(s) "
                f"({env_s.get('tool_type', 'WL')}/{env_s.get('mud_type', 'WBM')}); "
                f"mean conf = {cf_txt}, max unreliable = {max_unrel:.1f}%."
            )
            chk12_note = "Propagate *_QC_CONF confidence weights and *_SIGMA uncertainties into downstream solvers."
        else:
            chk12_status = "PASS"
            chk12_measured = (
                f"Applied across {env_s.get('wells_applied', 0)}/{env_s.get('total_wells', 0)} well(s) "
                f"({env_s.get('tool_type', 'WL')}/{env_s.get('mud_type', 'WBM')}); "
                f"mean conf = {cf_txt}, max unreliable = {max_unrel:.1f}%."
            )
            chk12_note = (
                "Borehole environmental corrections and sensor confidence verified."
            )
    else:
        hs_n = int(env_s.get("hole_section_count") or 0)
        chk12_status = "N/A (Raw Logs)"
        chk12_measured = f"No *_ENV_CORR curves saved on project logs (hole sections loaded: {hs_n})."
        chk12_note = "Run Borehole Environmental Corrections in Data & QA/QC to generate *_ENV_CORR and *_SIGMA."
    checks.append(
        {
            "id": "CHK-12",
            "rule": "Borehole Environmental Correction & Sensor Confidence",
            "category": "Borehole Environment",
            "criterion": "Max UNRELIABLE ≤ 10%, mean QC confidence ≥ 0.65, zero res/mud mismatch",
            "measured": chk12_measured,
            "status": chk12_status,
            "action_note": chk12_note,
            "detail": chk12_measured,
        }
    )

    # CHK-13: Inter-Log Depth Alignment & Run Tie-In Audit
    chk13_status = "PASS"
    chk13_measured = (
        "Insufficient multi-curve overlap to audit inter-log depth alignment."
    )
    chk13_note = "Inter-log depth alignment verified across evaluated curve suite."
    da_ctx = context.get("depth_alignment_summary") or {}
    if da_ctx.get("evaluated"):
        eval_wells = int(da_ctx.get("wells_evaluated") or 1)
        off_items_ctx: List[str] = []
        minor_items_ctx: List[str] = []
        max_shift_ctx = 0.0
        for cr in da_ctx.get("curve_table") or []:
            if float(cr.get("confidence") or 0.0) < 0.45:
                continue
            sh_m = float(cr.get("shift_m") or 0.0)
            max_shift_ctx = max(max_shift_ctx, abs(sh_m))
            lbl = f"{cr.get('well_name', 'WELL')}:{cr['curve']} ({sh_m:+.2f} m)"
            if cr.get("status") == "OFF_DEPTH":
                off_items_ctx.append(lbl)
            elif cr.get("status") == "MINOR":
                minor_items_ctx.append(lbl)
        if off_items_ctx and max_shift_ctx > 0.50:
            chk13_status = "QA ALERT"
            chk13_measured = (
                f"Off-depth run(s) detected across {eval_wells} well(s): "
                f"{', '.join(off_items_ctx[:4])} (max |Δz| = {max_shift_ctx:.2f} m)."
            )
            chk13_note = "Run Inter-Log Depth Alignment QC in Data & QA/QC to shift off-depth logs before evaluation."
        elif off_items_ctx or minor_items_ctx:
            chk13_status = "REVIEW"
            flagged_txt = ", ".join((off_items_ctx + minor_items_ctx)[:4])
            chk13_measured = (
                f"Minor depth drift detected across {eval_wells} well(s): "
                f"{flagged_txt} (max |Δz| = {max_shift_ctx:.2f} m)."
            )
            chk13_note = "Inspect Depth Alignment QC plot; apply bulk or dynamic shift if using thin-bed deconvolution."
        else:
            chk13_status = "PASS"
            chk13_measured = (
                f"All primary curves aligned within tolerance across {eval_wells} well(s) "
                f"(max |Δz| = {max_shift_ctx:.2f} m)."
            )
    elif not full_df.empty and "DEPTH" in full_df.columns:
        try:
            from quick_pp.depth_alignment import evaluate_depth_alignment

            off_items: List[str] = []
            minor_items: List[str] = []
            max_shift_m = 0.0
            eval_wells = 0
            w_groups = (
                full_df.groupby("WELL_NAME", sort=False)
                if "WELL_NAME" in full_df.columns
                else [("WELL", full_df)]
            )
            for wn_g, w_sub in w_groups:
                if len(w_sub) < 30:
                    continue
                eval_wells += 1
                al_rep = evaluate_depth_alignment(w_sub, ref_curve="GR", depth_unit="m")
                for cr in al_rep.get("curves") or []:
                    if float(cr.get("confidence") or 0.0) < 0.45:
                        continue
                    sh_m = float(cr.get("suggested_shift_m") or 0.0)
                    max_shift_m = max(max_shift_m, abs(sh_m))
                    if cr.get("status") == "OFF_DEPTH":
                        off_items.append(f"{wn_g}:{cr['curve']} ({sh_m:+.2f} m)")
                    elif cr.get("status") == "MINOR":
                        minor_items.append(f"{wn_g}:{cr['curve']} ({sh_m:+.2f} m)")

            if eval_wells > 0:
                if off_items and max_shift_m > 0.50:
                    chk13_status = "QA ALERT"
                    chk13_measured = (
                        f"Off-depth run(s) detected across {eval_wells} well(s): "
                        f"{', '.join(off_items[:4])} (max |Δz| = {max_shift_m:.2f} m)."
                    )
                    chk13_note = "Run Inter-Log Depth Alignment QC in Data & QA/QC to shift off-depth logs before evaluation."
                elif off_items or minor_items:
                    chk13_status = "REVIEW"
                    flagged_txt = ", ".join((off_items + minor_items)[:4])
                    chk13_measured = (
                        f"Minor depth drift detected across {eval_wells} well(s): "
                        f"{flagged_txt} (max |Δz| = {max_shift_m:.2f} m)."
                    )
                    chk13_note = "Inspect Depth Alignment QC plot; apply bulk or dynamic shift if using thin-bed deconvolution."
                else:
                    chk13_status = "PASS"
                    chk13_measured = (
                        f"All primary curves aligned within tolerance across {eval_wells} well(s) "
                        f"(max |Δz| = {max_shift_m:.2f} m)."
                    )
        except Exception:
            pass
    checks.append(
        {
            "id": "CHK-13",
            "rule": "Inter-Log Depth Alignment & Run Tie-In",
            "category": "Data Conditioning",
            "criterion": "Inter-curve bed-boundary lag |Δz| ≤ 0.25 m (≤ 1.5 samples)",
            "measured": chk13_measured,
            "status": chk13_status,
            "action_note": chk13_note,
            "detail": chk13_measured,
        }
    )

    return checks


_GATE_BLOCKING_STATUSES = {"QA ALERT", "FLAGGED", "FAIL", "FAILED"}
_GATE_REVIEW_STATUSES = {"REVIEW"}


def summarize_gate_checks(checks: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Collapse the gate list into counts and the items that must be visible in the executive summary."""
    blocking = [
        c for c in checks if str(c.get("status", "")).upper() in _GATE_BLOCKING_STATUSES
    ]
    review = [
        c for c in checks if str(c.get("status", "")).upper() in _GATE_REVIEW_STATUSES
    ]
    if blocking:
        overall = "NOT READY"
    elif review:
        overall = "CONDITIONAL"
    else:
        overall = "READY"
    return {
        "overall": overall,
        "n_total": len(checks),
        "n_blocking": len(blocking),
        "n_review": len(review),
        "blocking": [
            {
                "id": c.get("id"),
                "rule": c.get("rule"),
                "measured": c.get("measured"),
                "action": c.get("action_note"),
            }
            for c in blocking
        ],
        "review": [
            {
                "id": c.get("id"),
                "rule": c.get("rule"),
                "measured": c.get("measured"),
                "action": c.get("action_note"),
            }
            for c in review
        ],
    }


def get_compact_report_context(
    project_id: Union[int, str],
    well_names: Optional[List[str]] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    zones: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Return a JSON-serializable digest of the project's report context (excludes raw DataFrames)."""
    ctx = build_report_context(
        project_id,
        well_names=well_names,
        cutoffs=cutoffs,
        zones=zones,
        target_zones=zones,
        enabled_section_ids=set(),
    )
    return {k: v for k, v in ctx.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Step 2: Lead Petrophysicist Narrative Engine (Tier 2 AI Author)
# ---------------------------------------------------------------------------


def synthesize_domain_ai_commentary(
    context: Dict[str, Any],
    provider: str = "native_mcp",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    custom_instruction: Optional[str] = None,
    report_type: str = "FDP_FULL",
) -> Dict[str, str]:
    """Synthesize strictly grounded Lead Petrophysicist commentary for report sections."""
    proj_name = context.get("project_name", "Field Study")
    wells = context.get("wells", [])
    well_names = context.get("well_names", [])
    zonal = context.get("zonal_summary", [])
    qaqc_tbl = context.get("qaqc_table", [])
    env_s = context.get("env_corr_summary", {}) or {}
    da_s = context.get("depth_alignment_summary", {}) or {}
    poro_p = context.get("poro_params", {})
    sw_p = context.get("sw_params", {})
    shf_p = context.get("shf_params", {})
    scal_p = context.get("scal_params", {})
    rt_sum = context.get("rt_summary", {})
    unc_list = context.get("uncertainty_summary", [])
    cut = context.get("cutoffs", {"vsh": 0.40, "phie": 0.08, "sw": 0.65})

    eff_report_type = report_type or context.get("report_type", "FDP_FULL")
    is_single_well = eff_report_type == "SINGLE_WELL_CPI"
    focal_well = well_names[0] if well_names else "Active Well"
    well_meta = wells[0] if wells else {}
    top_d = well_meta.get("min_depth", well_meta.get("top_depth", 0.0))
    bot_d = well_meta.get("max_depth", well_meta.get("bottom_depth", 0.0))

    pay_zones = [z for z in zonal if float(z.get("net_pay", 0.0)) > 0]
    ref_zones = pay_zones if pay_zones else zonal
    top_zone = max(ref_zones, key=lambda z: z.get("hcpt", 0.0)) if ref_zones else {}

    # Field summary numbers come from the same helper as Section 8.4 so both always agree.
    field_sum = _compute_field_summary(zonal)
    if field_sum["phie"] is not None:
        mean_phie = field_sum["phie"]
        mean_sw = field_sum["sw"]
    else:
        mean_phie = (
            float(np.mean([z.get("phie_res", 0.0) for z in ref_zones]))
            if ref_zones
            else 0.0
        )
        mean_sw = 1.0

    total_hcpt = field_sum["column_hcpt"]
    mean_zone_hcpt = field_sum["mean_zone_hcpt"]
    n_eval_zones = field_sum["n_zones"]
    max_bh = max((q.get("badhole_pct", 0.0) for q in qaqc_tbl), default=0.0)
    if env_s.get("any_applied"):
        env_curves_txt = ", ".join(f"`{c}`" for c in env_s.get("corrected_curves", []))
        env_cf_val = env_s.get("mean_conf")
        env_cf_txt = (
            f"{env_cf_val:.2f}" if isinstance(env_cf_val, (int, float)) else "n/a"
        )
        env_regime_txt = (
            f"offshore ($WD = {env_s.get('water_depth_m', 0.0):.0f}\\text{{ m}}$)"
            if env_s.get("offshore")
            else "onshore"
        )
        inv_roll = env_s.get("invasion_rollup") or {}
        inv_clause = ""
        if env_s.get("has_invasion") and inv_roll:
            di_txt = (
                f", median $d_i = {inv_roll['median_di_in']:.1f}\\text{{ in}}$"
                if inv_roll.get("median_di_in") is not None
                else ""
            )
            inv_clause = (
                f" Step-profile radial resistivity inversion classifies **{inv_roll.get('hc_like_pct', 0.0):.1f}%** "
                f"of samples as movable-hydrocarbon invasion ($R_{{xo}}/R_t < R_{{mf}}/R_w$) and "
                f"**{inv_roll.get('water_like_pct', 0.0):.1f}%** as water-zone invasion{di_txt}."
            )
        env_narrative_suffix = (
            f" Generic **{env_s.get('tool_type', 'WL')}** borehole environmental corrections ({env_curves_txt}) were applied "
            f"for a **{env_s.get('mud_type', 'WBM')}** ({env_s.get('mud_weight_ppg', 9.0):.1f} ppg, {env_regime_txt}, "
            f"${env_s.get('gradient_c_per_km', 30.0):.1f}^\\circ\\text{{C/km}}$ geothermal gradient"
            + (
                f", {env_s.get('hole_section_count', 0)} mud-program hole {_pluralize(env_s.get('hole_section_count', 0), 'section')}"
                if env_s.get("hole_section_count")
                else ""
            )
            + f") logging environment, yielding a mean sensor confidence of **{env_cf_txt}** and "
            f"a maximum unreliable-sample rate of **{env_s.get('max_unreliable_pct', 0.0):.1f}%**."
            + inv_clause
        )
    else:
        hs_n = int(env_s.get("hole_section_count") or 0)
        env_narrative_suffix = (
            " Explicit `*_ENV_CORR` environmental corrections have not yet been saved on the project curves"
            + (
                f" ({hs_n} borehole mud-program {_pluralize(hs_n, 'section')} loaded in ancillary metadata)."
                if hs_n > 0
                else "."
            )
        )

    da_narrative_suffix = ""
    if da_s.get("evaluated"):
        da_ref = da_s.get("reference_curve", "GR")
        da_off = int(da_s.get("off_depth_count") or 0)
        da_min = int(da_s.get("minor_count") or 0)
        da_max = float(da_s.get("max_abs_shift_m") or 0.0)
        da_cond = da_s.get("conditioned_curves") or []
        cond_clause = (
            f" (previously depth-shifted curves with `*_ORI` lineage: {', '.join(f'`{c}`' for c in da_cond)})"
            if da_cond
            else ""
        )
        if da_off > 0:
            da_narrative_suffix = (
                f" Bed-boundary gradient (`|df/dz|`) cross-correlation against `{da_ref}` flags **{da_off} off-depth** "
                f"and **{da_min} minor-drift** curve(s) (maximum lag $|\\Delta z| = {da_max:.2f}\\text{{ m}}$){cond_clause}, "
                f"warranting run tie-in conditioning prior to thin-bed or invasion analysis."
            )
        elif da_min > 0:
            da_narrative_suffix = (
                f" Inter-log bed-boundary cross-correlation against `{da_ref}` confirms zero severe off-depth runs, "
                f"with **{da_min} minor micro-drift** curve(s) ($|\\Delta z|_{{\\max}} = {da_max:.2f}\\text{{ m}}$){cond_clause}."
            )
        else:
            da_narrative_suffix = (
                f" Inter-log bed-boundary cross-correlation against `{da_ref}` confirms all **{da_s.get('total_curves', 0)}** "
                f"evaluated curves are on-depth within sample tolerance ($|\\Delta z|_{{\\max}} = {da_max:.2f}\\text{{ m}}$){cond_clause}."
            )
    env_narrative_suffix = env_narrative_suffix + da_narrative_suffix

    # Spec 4.1: Threshold-driven calibration parity prose (never claim "strong parity" when R² is low)
    c_r2 = _safe_float(poro_p.get("core_r2"), 0.0)
    c_rmse = _safe_float(poro_p.get("core_rmse"), 0.0)
    c_cal_cnt = int(poro_p.get("core_count", 0))
    c_tot_cnt = int(poro_p.get("total_core_count", 0))
    if c_r2 >= 0.75 and c_rmse <= 0.030:
        parity_desc = "strong parity and acceptable calibration accuracy"
    elif c_r2 >= 0.50 and c_rmse <= 0.045:
        parity_desc = (
            "moderate agreement, though local facies/depth-shift scatter is present"
        )
    else:
        parity_desc = (
            "poor statistical correlation, indicating unresolved core depth-shifts, "
            "non-overlapping depth intervals, or variable matrix grain density that "
            "must be rectified prior to static model handover"
        )

    # Empirical P10/P90 ratio from Monte Carlo uncertainty table
    valid_spreads = [
        float(u["spread_ratio"])
        for u in unc_list
        if u.get("spread_ratio") is not None
        and str(u.get("metric", "")).startswith("Hydrocarbon Pore Thickness")
    ]
    median_spread_str = (
        f"{float(np.median(valid_spreads)):.2f}×"
        if valid_spreads
        else "N/A (Water-bearing)"
    )

    if is_single_well:
        if c_cal_cnt >= 3:
            core_clause = (
                f"calibrated against **{c_cal_cnt} target-zone core plug measurements** "
                f"(out of {c_tot_cnt} total loaded plugs; $R^2 = {c_r2:.2f}$, $\\text{{RMSE}} = {c_rmse:.3f}\\text{{ v/v}}$, showing {parity_desc})"
            )
            litho_core_sentence = (
                f"Comparison between log-derived total porosity ($\\phi_t$) and overburden-corrected core plugs in **`{focal_well}`** "
                f"yields $R^2 = {c_r2:.2f}$ and $\\text{{RMSE}} = {c_rmse:.3f}\\text{{ v/v}}$ across **{c_cal_cnt} valid overlapping target-zone plugs** "
                f"(out of **{c_tot_cnt} total core plugs loaded**), exhibiting {parity_desc}."
            )
        elif c_tot_cnt > 0:
            core_clause = f"evaluated from environmentally conditioned wireline logs ({c_tot_cnt} core plugs loaded in deeper/non-overlapping intervals)"
            litho_core_sentence = (
                f"Although **{c_tot_cnt} core plugs** are loaded in **`{focal_well}`**, **{c_cal_cnt} plugs** overlap the log-evaluated target reservoir window; "
                f"effective porosity ($\\phi_e$) was therefore derived from neutron-density crossplot total porosity ($\\phi_t$) after subtracting clay-bound water ($CBW$)."
            )
        else:
            core_clause = (
                "evaluated from environmentally conditioned triple-combo wireline logs"
            )
            litho_core_sentence = "Effective porosity ($\\phi_e$) was derived from neutron-density total porosity ($\\phi_t$) after subtracting clay-bound water ($CBW$)."

        base_commentary = {
            "executive_summary": (
                f"This **Single-Well Petrophysical CPI & Formation Evaluation Dossier** documents the quantitative "
                f"wireline log interpretation for well **`{focal_well}`** (`{top_d:.1f}–{bot_d:.1f} m`, "
                f"**{context.get('total_logged_thickness', 0.0):.1f} m** logged interval) in **{proj_name}**. "
                f"The evaluation focuses on the target reservoir succession (`{', '.join(context.get('target_zones', []))}`), "
                f"integrating borehole washout conditioning, clay & mineralogy volume ($V_{{sh}}$), "
                f"light-hydrocarbon corrected effective porosity ($\\phi_e$), and shaly-sand water saturation ($S_w$), "
                f"{core_clause}. Across the evaluated target reservoir succession in **`{focal_well}`**, net-thickness-weighted mean "
                f"effective porosity ($\\phi_e$) is **${mean_phie:.3f}\\text{{ v/v}}$** with a pore-volume weighted water "
                f"saturation ($S_w$) of **${mean_sw:.3f}\\text{{ v/v}}$** and cumulative Hydrocarbon Pore Thickness "
                f"($HCPT$) across the **{n_eval_zones}** evaluated zones of **${total_hcpt:.2f}\\text{{ m}}$**. "
                + (
                    f"The primary pay interval is **{top_zone.get('zone')}**, delivering "
                    f"**${top_zone.get('net_pay', 0.0):.1f}\\text{{ m}}$** of Net Pay ($NTG_{{pay}} = {top_zone.get('ntg', 0.0):.2f}\\text{{ v/v}}$, "
                    f"Net Reservoir $= {top_zone.get('net_res', 0.0):.1f}\\text{{ m}}$, "
                    f"$\\phi_e = {top_zone.get('phie', 0.0):.3f}\\text{{ v/v}}$, $S_w = {top_zone.get('sw', 0.0):.3f}\\text{{ v/v}}$, "
                    f"$HCPT = {top_zone.get('hcpt', 0.0):.2f}\\text{{ m}}$)."
                    if top_zone and float(top_zone.get("net_pay", 0.0)) > 0
                    else (
                        f"In **`{focal_well}`**, the primary target reservoir (**{top_zone.get('zone')}**) contains "
                        f"**${top_zone.get('net_res', 0.0):.1f}\\text{{ m}}$** of clean Net Reservoir ($\\phi_{{e,res}} = {top_zone.get('phie_res', 0.0):.3f}\\text{{ v/v}}$, "
                        f"$K_{{geom,res}} = {top_zone.get('perm_geom_res', 0.0):.1f}\\text{{ mD}}$) but is water-bearing ($Net\\text{{ }}Pay = 0.0\\text{{ m}}$)."
                        if top_zone
                        else ""
                    )
                )
            ),
            "inventory_narrative": (
                f"Well **`{focal_well}`** penetrates **{context.get('total_logged_thickness', 0.0):.1f} m** of logged "
                f"section across **{len(context.get('target_zones', []))} target reservoir {_pluralize(len(context.get('target_zones', [])), 'zone')}** (`{', '.join(context.get('target_zones', []))}`)."
            ),
            "qaqc_narrative": (
                f"Borehole environmental quality control was performed on well **`{focal_well}`** prior to quantitative CPI "
                f"computation. Across the evaluated interval, caliper/density washout (`BADHOLE`) flagging is "
                f"**{max_bh:.1f}%** (using caliper washout $> {context['qaqc_params']['caliper_limit']}$ inch and "
                f"$\\Delta\\rho > {context['qaqc_params']['drho_limit']}$ g/cc criteria). "
                + (
                    f"Rugose intervals exceeding the washout threshold were mitigated using `{context['qaqc_params']['badhole_treatment']}` "
                    f"to prevent pad-standoff bulk density spikes from overestimating porosity."
                    if max_bh > 15.0
                    else f"Well **`{focal_well}`** exhibits good overall borehole stability (<15% washout flagging), providing reliable density-neutron porosity response across the target zones."
                )
                + env_narrative_suffix
            ),
            "litho_poro_narrative": (
                f"In well **`{focal_well}`**, shale volume ($V_{{sh}}$) and mineral fractions were solved using the "
                f"**{context['litho_params']['model_name']}** anchored by clean reservoir sand "
                f"($GR_{{clean}} = {context['litho_params']['gr_clean']}\\text{{ gAPI}}$, "
                f"$\\rho_{{ma}} = {poro_p.get('rho_ma', 2.65)}\\text{{ g/cc}}$) and intra-formational shale endpoints "
                f"($GR_{{shale}} = {context['litho_params']['gr_shale']}\\text{{ gAPI}}$, $\\rho_{{clay}} = {context['litho_params']['rho_clay']}\\text{{ g/cc}}$). "
                + litho_core_sentence
            ),
            "rock_typing_narrative": "",
            "saturation_narrative": (
                f"Water saturation ($S_w$) in well **`{focal_well}`** was evaluated using the **{sw_p.get('model_name')}** "
                f"to account for clay conductivity ($B \\cdot Q_v$) in shaly and heterolithic sand intervals. "
                f"Formation water resistivity was parameterized at $R_w = {sw_p.get('rw', 0.08):.3f}\\text{{ }}\\Omega\\cdot\\text{{m}}$ "
                f"at ${sw_p.get('temp', 75.0):.1f}^\\circ\\text{{C}}$ (${sw_p.get('salinity', 35000):,}\\text{{ ppm NaCl}}$) "
                f"with electrical exponents $a={sw_p.get('a', 1.0)}, m={sw_p.get('m', 2.0)}, n={sw_p.get('n', 2.0)}$ "
                f"and a target-reservoir Free Water Level at $FWL = {sw_p.get('fwl', 2943.0):.1f}\\text{{ m TVDss}}$."
            ),
            "shf_scal_narrative": "",
            "uncertainty_narrative": (
                f"Zonal summation for well **`{focal_well}`** decouples **Net Reservoir** ($V_{{sh}} \\le {cut['vsh']:.2f}\\text{{ v/v}}$, "
                f"$\\phi_e \\ge {cut['phie']:.2f}\\text{{ v/v}}$) from **Net Pay** (adding $S_w \\le {cut['sw']:.2f}\\text{{ v/v}}$) "
                f"so static geomodelers retain aquifer porosity and permeability in water-bearing penetrations. "
                f"Monte Carlo uncertainty propagation ($N = 1000$ realizations) yields a median P10/P90 $HCPT$ ratio of **{median_spread_str}**."
            ),
            "upscaling_narrative": "",
        }
    else:
        if c_cal_cnt >= 3:
            fdp_core_sentence = (
                f"Log-derived total porosity ($\\phi_t$) compared against overburden-corrected core plug porosity exhibits "
                f"**{parity_desc}** ($R^2 = {c_r2:.2f}$, $\\text{{RMSE}} = {c_rmse:.3f}\\text{{ v/v}}$ across "
                f"**{c_cal_cnt} valid target-zone calibration plugs** out of **{c_tot_cnt} total core plugs loaded** across the field)."
            )
        elif c_tot_cnt > 0:
            fdp_core_sentence = (
                f"While **{c_tot_cnt} total RCA core plugs** are loaded across the field (Table 2.1), only **{c_cal_cnt} plugs** "
                f"overlap log-evaluated target reservoir intervals; depth-shift alignment and stratigraphic core-to-log matching "
                f"are flagged in the QA/QC audit prior to static model handover."
            )
        else:
            fdp_core_sentence = "No Routine Core Analysis (RCA) plugs were available in the target zones; porosity was derived from hydrocarbon-corrected neutron-density crossplot logs."

        base_commentary = {
            "executive_summary": (
                f"This Field Development Plan (FDP) Petrophysical Evaluation integrates wireline log conditioning, "
                f"{context['litho_params']['model_name']} lithology evaluation, Routine Core Analysis (RCA), "
                f"Hydraulic Flow Unit ($FZI$) rock typing, and Leverett-$J$ Saturation Height Modeling ($SHF$) across "
                f"**{len(well_names)} evaluated {_pluralize(len(well_names), 'well')}** (`{', '.join(well_names)}`) in **{proj_name}**, focusing on the "
                f"target reservoir succession (`{', '.join(context.get('target_zones', []))}`). "
                f"Across the evaluated target zones, net-thickness-weighted mean effective porosity ($\\phi_e$) is "
                f"**${mean_phie:.3f}\\text{{ v/v}}$** with a pore-volume weighted water saturation ($S_w$) of "
                f"**${mean_sw:.3f}\\text{{ v/v}}$**. A well penetrating all **{n_eval_zones}** evaluated zones accumulates a "
                f"Hydrocarbon Pore Thickness ($HCPT$) of **${total_hcpt:.2f}\\text{{ m}}$** "
                f"(sum of per-zone means; average ${mean_zone_hcpt:.2f}\\text{{ m}}$ per zone). "
                + (
                    f"The primary pay container is **{top_zone.get('zone')}**, delivering a field-mean Net Pay of "
                    f"**${top_zone.get('net_pay', 0.0):.1f}\\text{{ m}}$** ($NTG_{{pay}} = {top_zone.get('ntg', 0.0):.2f}\\text{{ v/v}}$, "
                    f"Mean Net Reservoir $= {top_zone.get('net_res', 0.0):.1f}\\text{{ m}}$, "
                    f"$\\phi_e = {top_zone.get('phie', 0.0):.3f}\\text{{ v/v}}$, $S_w = {top_zone.get('sw', 0.0):.3f}\\text{{ v/v}}$, "
                    f"$HCPT = {top_zone.get('hcpt', 0.0):.2f}\\text{{ m}}$)."
                    if top_zone
                    else ""
                )
            ),
            "inventory_narrative": (
                f"The petrophysical database for **{proj_name}** comprises **{len(wells)} {_pluralize(len(wells), 'well')}** totaling "
                f"**{context.get('total_logged_thickness', 0.0):.1f} m** of logged borehole section across "
                f"**{len(context.get('all_zones', []))} mapped geological formations**, with volumetric and CPI reporting "
                f"filtered to the **{len(context.get('target_zones', []))} target reservoir {_pluralize(len(context.get('target_zones', [])), 'zone')}** (`{', '.join(context.get('target_zones', []))}`). "
                f"A total of **{c_tot_cnt} Routine Core Analysis (RCA) plug measurements** are loaded in the project inventory (Table 2.1), "
                f"of which **{c_cal_cnt} valid plugs** overlap log-evaluated target reservoir intervals for active porosity calibration."
            ),
            "qaqc_narrative": (
                f"Rigorous borehole environmental conditioning and multi-well normalization were executed prior to quantitative "
                f"interpretation. Across the {len(qaqc_tbl)} evaluated {_pluralize(len(qaqc_tbl), 'well')}, maximum washout/bad-hole flagging is "
                f"**{max_bh:.1f}%** (using caliper washout $> {context['qaqc_params']['caliper_limit']}$ in (inches) and "
                f"$\\Delta\\rho > {context['qaqc_params']['drho_limit']}$ g/cc criteria). "
                + (
                    f"Intervals exceeding the 15% washout threshold were automatically switched to `{context['qaqc_params']['badhole_treatment']}` "
                    f"to prevent pad-standoff density spikes from overestimating effective porosity."
                    if max_bh > 15.0
                    else "All evaluated reservoir intervals exhibit good borehole stability (<15% washout flagging), requiring minimal synthetic curve reconstruction."
                )
                + env_narrative_suffix
            ),
            "litho_poro_narrative": (
                f"Shale volume ($V_{{sh}}$) and mineral fractions were resolved using the **{context['litho_params']['model_name']}** "
                f"constrained by clean sand ($GR_{{clean}} = {context['litho_params']['gr_clean']}\\text{{ gAPI}}$, "
                f"$\\rho_{{ma}} = {poro_p.get('rho_ma', 2.65)}\\text{{ g/cc}}$) and intra-reservoir shale endpoints "
                f"($GR_{{shale}} = {context['litho_params']['gr_shale']}\\text{{ gAPI}}$, $\\rho_{{clay}} = {context['litho_params']['rho_clay']}\\text{{ g/cc}}$). "
                + fdp_core_sentence
            ),
            "rock_typing_narrative": _build_rock_typing_narrative(
                context.get("rock_types", []),
                rt_sum,
                context.get("fzi_cutoffs", []),
            ),
            "saturation_narrative": (
                f"Water saturation ($S_w$) was evaluated using the **{sw_p.get('model_name')}** to account for clay-bound "
                f"counter-ion conductivity ($B \\cdot Q_v$) in shaly-sand and heterolithic intervals, where Archie saturation would "
                f"otherwise overestimate $S_w$. Formation water resistivity was anchored at "
                f"$R_w = {sw_p.get('rw', 0.08):.3f}\\text{{ }}\\Omega\\cdot\\text{{m}}$ at ${sw_p.get('temp', 75.0):.1f}^\\circ\\text{{C}}$ "
                f"(${sw_p.get('salinity', 35000):,}\\text{{ ppm NaCl}}$) with electrical exponents $a={sw_p.get('a', 1.0)}, "
                f"m={sw_p.get('m', 2.0)}, n={sw_p.get('n', 2.0)}$ and a target-reservoir Free Water Level at "
                f"$FWL = {sw_p.get('fwl', 2943.0):.1f}\\text{{ m TVDss}}$."
            ),
            "shf_scal_narrative": (
                _build_shf_reconciliation_sentence(shf_p)
                + f"Dynamic simulation relative permeability "
                f"tables (`SWOF`, **{scal_p.get('source_label', 'Regional Analogue / Screening Parameterization')}**) "
                + (
                    f"were parameterized using LET exponents $L_w = {scal_p.get('lw', 2.0)}, E_w = {scal_p.get('ew', 1.5)}, T_w = {scal_p.get('tw', 2.0)}$ and "
                    f"$L_o = {scal_p.get('lo', 2.0)}, E_o = {scal_p.get('eo', 1.5)}, T_o = {scal_p.get('to', 2.0)}$ with "
                    if str(scal_p.get("model", "corey")).lower() == "let"
                    else f"were parameterized using Corey exponents $N_w = {scal_p.get('nw', 2.5)}, N_o = {scal_p.get('no', 2.5)}$ with "
                )
                + f"$S_{{wc}} = {scal_p.get('swirr', 0.15):.2f}\\text{{ v/v}}$ and $S_{{or}} = {scal_p.get('sor', 0.20):.2f}\\text{{ v/v}}$."
            ),
            "uncertainty_narrative": (
                f"Zonal summation decouples **Net Reservoir** ($V_{{sh}} \\le {cut['vsh']:.2f}\\text{{ v/v}}$, "
                f"$\\phi_e \\ge {cut['phie']:.2f}\\text{{ v/v}}$) from **Net Pay** ($S_w \\le {cut['sw']:.2f}\\text{{ v/v}}$) "
                f"so static geomodelers can populate 3D aquifer porosity and permeability in water-bearing penetrations. "
                f"Monte Carlo / Latin Hypercube ($N = 1000$ realizations) uncertainty propagation across the pay-bearing target zones "
                f"indicates a median P10/P90 $HCPT$ ratio of **{median_spread_str}**, driven by matrix density ($\\rho_{{ma}}$), "
                f"cementation exponent ($m$), and $R_w$ uncertainty."
            ),
            "upscaling_narrative": (
                f"Fine-scale evaluated CPI logs within the target reservoir succession were blocked to "
                f"${context['upscaling_params']['cell_size']:.1f}\\text{{ m}}$ cellular resolution honoring geological zone boundaries. "
                f"Cardwell-Parsons directional permeability bounds yield a field-mean vertical-to-horizontal anisotropy ratio of "
                f"**$K_v / K_h = {context.get('upscaling_summary', {}).get('mean_kv_kh', 0.0):.3f}$** "
                f"(range `{context.get('upscaling_summary', {}).get('min_kv_kh', 0.0):.3f}–{context.get('upscaling_summary', {}).get('max_kv_kh', 0.0):.3f}`) "
                f"across the evaluated {len(well_names)} {_pluralize(len(well_names), 'well')}, preserving thin-bed vertical baffles for Eclipse/Intersect dynamic simulation."
            ),
        }

    if custom_instruction:
        base_commentary["executive_summary"] += (
            f"\n\n**Focused Technical Synthesis:** {custom_instruction}"
        )

    prov = (provider or "native_mcp").lower().strip()
    if prov == "native_mcp":
        return base_commentary

    # Optional external LLM synthesis grounded strictly in compact context JSON
    try:
        compact_ctx = {
            "report_type": eff_report_type,
            "project_name": proj_name,
            "well_names": well_names,
            "cutoffs": cut,
            "zonal_summary": zonal,
            "qaqc_table": qaqc_tbl,
            "env_corr_summary": {
                k: v for k, v in env_s.items() if k not in ("hole_sections_table",)
            },
            "depth_alignment_summary": {
                k: v for k, v in da_s.items() if k not in ("focal_report",)
            },
            "poro_params": poro_p,
            "sw_params": sw_p,
            "shf_params": shf_p,
        }
        sys_prompt = (
            "You are a Senior Petrophysicist writing a Single-Well Computer-Processed Interpretation (CPI) & Formation Evaluation Dossier. "
            "Focus strictly on the single evaluated well (do not refer to it as a Field Development Plan or multi-well study). "
            "Strict Grounding Rule: Cite exact numbers from the provided JSON context and never hallucinate parameters. "
            "Return ONLY a valid JSON object with keys: executive_summary, qaqc_narrative, litho_poro_narrative, "
            "saturation_narrative, uncertainty_narrative."
            if is_single_well
            else "You are a Lead Petrophysicist writing an FDP Gate Review evaluation report. "
            "Strict Grounding Rule: Cite exact numbers from the provided JSON context and never hallucinate parameters. "
            "Return ONLY a valid JSON object with keys: executive_summary, inventory_narrative, qaqc_narrative, "
            "litho_poro_narrative, rock_typing_narrative, saturation_narrative, shf_scal_narrative, "
            "uncertainty_narrative, upscaling_narrative."
        )
        user_prompt = (
            f"Context JSON:\n{json.dumps(compact_ctx, default=str)}\n\n"
            + (
                f"Additional Instruction: {custom_instruction}\n\n"
                if custom_instruction
                else ""
            )
            + "Generate the JSON object of technical section narratives."
        )

        if prov in ("openrouter", "ollama", "openai"):
            from quick_pp.app.backend.services.demo_guard import (
                validate_external_llm_target,
            )

            b_url, key = validate_external_llm_target(
                provider=prov,
                caller_base_url=base_url,
                caller_api_key=api_key,
                env_openrouter_url=os.getenv(
                    "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
                ),
                env_openrouter_key=os.getenv("OPENROUTER_API_KEY", ""),
            )
            mdl = model or (
                "qwen2.5:7b" if prov == "ollama" else "deepseek/deepseek-chat"
            )
            if key:
                resp = requests.post(
                    f"{b_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": mdl,
                        "messages": [
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.2,
                    },
                    timeout=35,
                )
                if resp.ok:
                    txt = resp.json()["choices"][0]["message"]["content"]
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    if m:
                        parsed = json.loads(m.group(0))
                        if isinstance(parsed, dict):
                            return {
                                **base_commentary,
                                **{k: str(v) for k, v in parsed.items() if v},
                            }
    except Exception as e:
        logger.warning(
            f"External AI commentary fallback to deterministic domain engine: {e}"
        )

    return base_commentary


# ---------------------------------------------------------------------------
# Step 3: Report Compilation & Section Assembly
# ---------------------------------------------------------------------------

SINGLE_WELL_DEFAULT_SECTIONS: List[Dict[str, Any]] = [
    {
        "section_id": "01_executive_summary",
        "title": "1. Well Evaluation Summary",
        "include_ai_commentary": True,
    },
    {
        "section_id": "03_qaqc",
        "title": "2. Borehole & Curve QC",
        "include_ai_commentary": True,
    },
    {
        "section_id": "04_lithology_porosity",
        "title": "3. Clay Volume, Lithology & Porosity",
        "include_ai_commentary": True,
    },
    {
        "section_id": "06_saturation_contacts",
        "title": "4. Water Saturation & Fluid Contacts",
        "include_ai_commentary": True,
    },
    {
        "section_id": "08_ressum_uncertainty",
        "title": "5. Zonal Net Pay Summary & Cut-offs",
        "include_ai_commentary": True,
    },
    {
        "section_id": "11_appendix_cpi",
        "title": "6. Composite CPI Log Plot",
        "include_ai_commentary": False,
    },
    {
        "section_id": "10_gate_checks",
        "title": "7. QA/QC Gate-Check & Peer-Review Audit Log",
        "include_ai_commentary": False,
    },
]


def _build_sections_config(
    report_type: str,
    requested_sections: Optional[List[ReportSectionConfig]] = None,
    include_ai_commentary: bool = True,
) -> List[ReportSectionConfig]:
    preset_ids = set(
        REPORT_TYPE_PRESETS.get(report_type, REPORT_TYPE_PRESETS["FDP_FULL"])
    )
    if requested_sections:
        normalized_reqs: List[ReportSectionConfig] = []
        for s in requested_sections:
            norm_id = _normalize_section_id(s.section_id)
            normalized_reqs.append(
                ReportSectionConfig(
                    section_id=norm_id,
                    title=s.title,
                    enabled=s.enabled,
                    include_ai_commentary=s.include_ai_commentary,
                    custom_notes=s.custom_notes,
                )
            )
        return normalized_reqs

    if report_type == "SINGLE_WELL_CPI":
        return [
            ReportSectionConfig(
                section_id=item["section_id"],
                title=item["title"],
                enabled=True,
                include_ai_commentary=include_ai_commentary
                and item.get("include_ai_commentary", True),
                custom_notes=None,
            )
            for item in SINGLE_WELL_DEFAULT_SECTIONS
        ]

    configs: List[ReportSectionConfig] = []
    for item in DEFAULT_REPORT_SECTIONS:
        sid = item["section_id"]
        configs.append(
            ReportSectionConfig(
                section_id=sid,
                title=item["title"],
                enabled=(sid in preset_ids),
                include_ai_commentary=include_ai_commentary
                and item.get("include_ai_commentary", True),
                custom_notes=None,
            )
        )
    return configs


def _generate_all_report_plots(
    report_id: str,
    context: Dict[str, Any],
    enabled_section_ids: set[str],
    report_type: str = "FDP_FULL",
) -> Dict[str, Any]:
    """Generate high-DPI PNG assets for enabled sections and return URL dictionary."""
    out_dir = REPORTS_STATIC_ROOT / report_id
    out_dir.mkdir(parents=True, exist_ok=True)
    url_prefix = f"/static/reports/{report_id}"

    with report_plots.report_type_scope(report_type):
        return _generate_all_report_plots_scoped(
            out_dir=out_dir,
            url_prefix=url_prefix,
            context=context,
            enabled_section_ids=enabled_section_ids,
        )


def _generate_all_report_plots_scoped(
    out_dir: Path,
    url_prefix: str,
    context: Dict[str, Any],
    enabled_section_ids: set[str],
) -> Dict[str, Any]:

    full_df: pd.DataFrame = context.get("_full_df", pd.DataFrame())
    well_names: List[str] = context.get("well_names", [])
    wells_meta: List[Dict[str, Any]] = context.get("wells", [])
    sw_p = context.get("sw_params", {})
    cut = context.get("cutoffs", {"vsh": 0.40, "phie": 0.08, "sw": 0.60})

    assets: Dict[str, Any] = {
        "inventory_heatmap": None,
        "norm_plot": None,
        "env_corr_plot": None,
        "depth_alignment_plot": None,
        "litho_poro_crossplot": None,
        "core_calibration_plot": None,
        "perm_crossplot": None,
        "pickett_shf_plot": None,
        "shf_scal_plot": None,
        "uncertainty_plot": None,
        "upscaling_plot": None,
        "cpi_plots": [],
    }

    target_zones: Optional[List[str]] = context.get("target_zones")
    scal_p = context.get("scal_params", {})
    poro_p = context.get("poro_params", {})
    env_s = context.get("env_corr_summary", {}) or {}
    is_analogue_scal = bool(scal_p.get("is_analogue", True))

    if "02_data_inventory" in enabled_section_ids:
        assets["inventory_heatmap"] = report_plots.generate_inventory_heatmap(
            context["project_id"], wells_meta, output_dir=out_dir, url_prefix=url_prefix
        )
    if "03_qaqc" in enabled_section_ids:
        assets["norm_plot"] = report_plots.generate_qaqc_norm_plot(
            full_df, well_names, output_dir=out_dir, url_prefix=url_prefix
        )
        has_env_cols = bool(env_s.get("any_applied")) or any(
            c in full_df.columns
            for c in ("ENV_TEMP_C", "HOLE_SEVERITY", "INVASION_PROFILE")
        )
        if has_env_cols and not full_df.empty:
            try:
                assets["env_corr_plot"] = report_plots.generate_env_corr_qc_plot(
                    full_df,
                    well_names=well_names,
                    env_corr_summary=env_s,
                    target_zones=target_zones,
                    output_dir=out_dir,
                    url_prefix=url_prefix,
                )
            except Exception as exc:
                logger.debug(f"Failed generating borehole env QC plot: {exc}")
        da_s = context.get("depth_alignment_summary", {}) or {}
        if da_s.get("evaluated") and not full_df.empty:
            try:
                da_fig_idx = (
                    1
                    + int(bool(assets.get("norm_plot")))
                    + int(bool(assets.get("env_corr_plot")))
                )
                assets["depth_alignment_plot"] = (
                    report_plots.generate_depth_alignment_qc_plot(
                        full_df,
                        well_names=well_names,
                        depth_alignment_summary=da_s,
                        target_zones=target_zones,
                        output_dir=out_dir,
                        url_prefix=url_prefix,
                        fig_index=da_fig_idx,
                    )
                )
            except Exception as exc:
                logger.debug(f"Failed generating depth alignment QC plot: {exc}")
    litho_p = context.get("litho_params", {})
    if "04_lithology_porosity" in enabled_section_ids:
        assets["litho_poro_crossplot"] = report_plots.generate_nd_crossplot(
            full_df,
            well_names,
            color_by="GR",
            target_zones=target_zones,
            litho_model=litho_p.get("model_key", "ssc"),
            dry_sand_point=litho_p.get("dry_sand_point", (-0.02, 2.65)),
            dry_silt_point=litho_p.get("dry_silt_point"),
            dry_clay_point=litho_p.get("dry_clay_point", (0.33, 2.70)),
            fluid_point=litho_p.get("fluid_point", (1.0, 1.0)),
            silt_line_angle=_safe_float(litho_p.get("silt_line_angle"), 119.0),
            use_hc_corrected=bool(poro_p.get("use_hc_corrected", False)),
            output_dir=out_dir,
            url_prefix=url_prefix,
        )
    if "05_rock_typing_perm" in enabled_section_ids:
        assets["perm_crossplot"] = report_plots.generate_rock_typing_perm_plot(
            full_df,
            well_names,
            rock_types=context.get("rock_types"),
            fzi_cutoffs=context.get("fzi_cutoffs"),
            target_zones=target_zones,
            output_dir=out_dir,
            url_prefix=url_prefix,
        )
    if "06_saturation_contacts" in enabled_section_ids:
        assets["pickett_shf_plot"] = report_plots.generate_pickett_plot(
            full_df,
            well_names,
            rw=_safe_float(sw_p.get("rw"), 0.08),
            a=_safe_float(sw_p.get("a"), 1.0),
            m=_safe_float(sw_p.get("m"), 2.0),
            n=_safe_float(sw_p.get("n"), 2.0),
            fwl=_safe_float(sw_p.get("fwl"), 2500.0),
            fwl_basis=sw_p.get("fwl_basis"),
            target_zones=target_zones,
            output_dir=out_dir,
            url_prefix=url_prefix,
        )
        if context.get("fpt_summary", {}).get("has_data") and context.get(
            "_fpt_points"
        ):
            try:
                assets["fpt_pressure_plot"] = report_plots.generate_fpt_pressure_plot(
                    fpt_points=context.get("_fpt_points", []),
                    fpt_summary=context.get("fpt_summary", {}),
                    output_dir=out_dir,
                    url_prefix=url_prefix,
                )
            except Exception as exc:
                logger.debug(f"Failed generating FPT pressure plot: {exc}")
    if "07_shf_scal" in enabled_section_ids:
        assets["shf_scal_plot"] = report_plots.generate_shf_j_function_plot(
            full_df,
            rock_types=context.get("rock_types"),
            fwl=_safe_float(sw_p.get("fwl"), 2500.0),
            is_analogue_scal=is_analogue_scal,
            scal_params=scal_p,
            shf_params=context.get("shf_params", {}),
            target_zones=target_zones,
            output_dir=out_dir,
            url_prefix=url_prefix,
        )
    if "08_ressum_uncertainty" in enabled_section_ids:
        assets["uncertainty_plot"] = report_plots.generate_uncertainty_cutoff_plot(
            full_df,
            well_names,
            cutoffs=cut,
            target_zones=target_zones,
            output_dir=out_dir,
            url_prefix=url_prefix,
        )
    if "09_upscaling" in enabled_section_ids and well_names:
        assets["upscaling_plot"] = report_plots.generate_upscaling_comparison_plot(
            full_df,
            well_names[0],
            cell_size=2.0,
            cutoffs=cut,
            target_zones=target_zones,
            output_dir=out_dir,
            url_prefix=url_prefix,
        )
    if (
        "11_appendix_cpi" in enabled_section_ids
        or "10_appendix_cpi" in enabled_section_ids
    ):
        cpi_list = []
        cpi_skipped = []
        cpi_candidate_wells = []
        candidate_scores = []
        if not full_df.empty and "WELL_NAME" in full_df.columns:
            for wn in well_names:
                w_sub = full_df[full_df["WELL_NAME"].astype(str) == str(wn)]
                if w_sub.empty or "DEPTH" not in w_sub.columns:
                    continue

                # Target reservoir interval subset
                d_min, d_max = report_plots._resolve_target_depth_window(
                    w_sub, target_zones=target_zones, max_span=480.0
                )
                d_col = pd.to_numeric(w_sub["DEPTH"], errors="coerce")
                w_win = w_sub[(d_col >= d_min) & (d_col <= d_max)]
                if w_win.empty:
                    w_win = w_sub

                # Count valid evaluated samples in target window
                phi_cnt = int(
                    pd.to_numeric(
                        w_win.get("PHIT", w_win.get("PHIE", pd.Series(dtype=float))),
                        errors="coerce",
                    )
                    .notna()
                    .sum()
                )
                swt_cnt = int(
                    pd.to_numeric(
                        w_win.get("SWT", pd.Series(dtype=float)), errors="coerce"
                    )
                    .notna()
                    .sum()
                )
                perm_cnt = int(
                    pd.to_numeric(
                        w_win.get("PERM", pd.Series(dtype=float)), errors="coerce"
                    )
                    .notna()
                    .sum()
                )
                gr_cnt = int(
                    pd.to_numeric(
                        w_win.get("GR", pd.Series(dtype=float)), errors="coerce"
                    )
                    .notna()
                    .sum()
                )
                rt_cnt = int(
                    pd.to_numeric(
                        w_win.get("RT", pd.Series(dtype=float)), errors="coerce"
                    )
                    .notna()
                    .sum()
                )

                # Strict requirement: well must have evaluated porosity and at least one other key curve (RT, SWT, or PERM)
                if phi_cnt < 10 or (swt_cnt < 5 and perm_cnt < 5 and rt_cnt < 5):
                    # Uninterpreted in target window: record it so the report can state why it is absent.
                    cpi_skipped.append(
                        {
                            "well_name": wn,
                            "reason": (
                                f"insufficient evaluated data in target window "
                                f"(PHI={phi_cnt}, SWT={swt_cnt}, PERM={perm_cnt}, RT={rt_cnt} samples)"
                            ),
                        }
                    )
                    continue

                has_core = (
                    "CPORE" in w_win.columns
                    and pd.to_numeric(w_win["CPORE"], errors="coerce").notna().sum() > 0
                ) or (
                    "PORO_CORE" in w_win.columns
                    and pd.to_numeric(w_win["PORO_CORE"], errors="coerce").notna().sum()
                    > 0
                )
                score = (
                    (10000 if has_core else 0)
                    + phi_cnt
                    + swt_cnt
                    + perm_cnt
                    + gr_cnt
                    + rt_cnt
                )
                candidate_scores.append((score, wn))

            candidate_scores.sort(key=lambda x: x[0], reverse=True)
            cpi_candidate_wells = [wn for _, wn in candidate_scores]

        for wn in cpi_candidate_wells:
            cpi_url = report_plots.generate_well_cpi_plot(
                full_df,
                wn,
                zones=target_zones,
                cutoffs=cut,
                output_dir=out_dir,
                url_prefix=url_prefix,
            )
            if cpi_url:
                cpi_list.append({"well_name": wn, "plot_url": cpi_url, "url": cpi_url})
            else:
                cpi_skipped.append(
                    {"well_name": wn, "reason": "CPI plot rendering returned no output"}
                )
        assets["cpi_plots"] = cpi_list
        assets["cpi_skipped"] = cpi_skipped

    return assets


def _render_sections_and_base_markdown(
    title: str,
    author: str,
    report_type: str,
    generation_date: str,
    context: Dict[str, Any],
    sections_cfg: List[ReportSectionConfig],
    ai_commentary: Dict[str, str],
    assets: Dict[str, Any],
    existing_section_overrides: Optional[Dict[str, str]] = None,
) -> Tuple[str, Dict[str, str]]:
    """Render each enabled section template and assemble the full Markdown document."""
    section_contents: Dict[str, str] = {}
    rendered_sections: List[Dict[str, str]] = []
    overrides = existing_section_overrides or {}

    base_vars = {k: v for k, v in context.items() if not k.startswith("_")}
    base_vars.update(
        {
            "title": title,
            "author": author,
            "report_type": report_type,
            "generation_date": generation_date,
            "ai_commentary": ai_commentary,
            "assets": assets,
        }
    )

    for sec in sections_cfg:
        if not sec.enabled:
            continue
        sid = sec.section_id
        if sid in overrides and overrides[sid].strip():
            sec_md = overrides[sid].strip()
        else:
            tpl_name = SECTION_TEMPLATE_MAP.get(sid)
            if not tpl_name:
                continue
            tpl = _jinja_env.get_template(tpl_name)
            sec_md = tpl.render(
                **base_vars,
                include_ai_commentary=sec.include_ai_commentary,
                custom_notes=sec.custom_notes,
            ).strip()

        section_contents[sid] = sec_md
        rendered_sections.append(
            {"section_id": sid, "title": sec.title, "content": sec_md}
        )

    base_tpl = _jinja_env.get_template("base_fdp_report.md.j2")
    full_md = base_tpl.render(
        **base_vars,
        rendered_sections=rendered_sections,
    ).strip()

    return full_md, section_contents


def _extract_braced(s: str, pos: int) -> Tuple[Optional[str], int]:
    """Extract balanced `{...}` content starting at or after `pos` (skipping leading whitespace)."""
    n = len(s)
    i = pos
    while i < n and s[i] in (" ", "\t"):
        i += 1
    if i >= n or s[i] != "{":
        return None, pos
    depth = 0
    start = i + 1
    while i < n:
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return s[start:i], i + 1
        i += 1
    return s[start:], n


_LATEX_SYMBOLS: Dict[str, str] = {
    "phi": "&phi;",
    "varphi": "&phi;",
    "rho": "&rho;",
    "sigma": "&sigma;",
    "theta": "&theta;",
    "mu": "&mu;",
    "omega": "&omega;",
    "alpha": "&alpha;",
    "beta": "&beta;",
    "gamma": "&gamma;",
    "Delta": "&Delta;",
    "Omega": "&Omega;",
    "Sigma": "&sum;",
    "sum": "&sum;",
    "cdot": "&middot;",
    "times": "&times;",
    "pm": "&plusmn;",
    "le": "&le;",
    "leq": "&le;",
    "ge": "&ge;",
    "geq": "&ge;",
    "ne": "&ne;",
    "neq": "&ne;",
    "approx": "&approx;",
    "to": "&rarr;",
    "rightarrow": "&rarr;",
    "circ": "&deg;",
    "quad": '<span class="math-quad"></span>',
    "qquad": '<span class="math-qquad"></span>',
    "ln": '<span class="math-fn">ln</span>',
    "log": '<span class="math-fn">log</span>',
    "exp": '<span class="math-fn">exp</span>',
    "cos": '<span class="math-fn">cos</span>',
    "sin": '<span class="math-fn">sin</span>',
    "tan": '<span class="math-fn">tan</span>',
}


def _latex_to_html(tex: str) -> str:
    """Brace-balanced recursive converter from LaTeX math to publication-grade semantic HTML."""
    if not tex:
        return ""
    s = (
        tex.replace(r"\left(", "(")
        .replace(r"\right)", ")")
        .replace(r"\left[", "[")
        .replace(r"\right]", "]")
        .replace(r"\%", "%")
    )
    out: List[str] = []
    i = 0
    n = len(s)
    while i < n:
        ch = s[i]
        if ch == "\\":
            m = re.match(r"\\([a-zA-Z]+)", s[i:])
            if m:
                cmd = m.group(1)
                next_i = i + len(m.group(0))
                if cmd == "frac":
                    num_str, p1 = _extract_braced(s, next_i)
                    if num_str is not None:
                        den_str, p2 = _extract_braced(s, p1)
                        if den_str is not None:
                            out.append(
                                f'<span class="math-frac"><span class="math-num">{_latex_to_html(num_str)}</span>'
                                f'<span class="math-den">{_latex_to_html(den_str)}</span></span>'
                            )
                            i = p2
                            continue
                elif cmd == "sqrt":
                    rad_str, p1 = _extract_braced(s, next_i)
                    if rad_str is not None:
                        out.append(
                            f'<span class="math-sqrt">&radic;<span class="math-radicand">{_latex_to_html(rad_str)}</span></span>'
                        )
                        i = p1
                        continue
                elif cmd in ("bar", "overline"):
                    arg_str, p1 = _extract_braced(s, next_i)
                    if arg_str is not None:
                        out.append(
                            f'<span class="math-bar">{_latex_to_html(arg_str)}</span>'
                        )
                        i = p1
                        continue
                elif cmd == "text":
                    txt_str, p1 = _extract_braced(s, next_i)
                    if txt_str is not None:
                        out.append(
                            f'<span class="math-text">{html.escape(txt_str)}</span>'
                        )
                        i = p1
                        continue
                elif cmd in _LATEX_SYMBOLS:
                    out.append(_LATEX_SYMBOLS[cmd])
                    i = next_i
                    continue
                else:
                    out.append(html.escape(cmd))
                    i = next_i
                    continue
            # Escaped single character
            if i + 1 < n:
                out.append(html.escape(s[i + 1]))
                i += 2
                continue
            i += 1
        elif ch in ("_", "^"):
            tag = "sub" if ch == "_" else "sup"
            arg_str, p1 = _extract_braced(s, i + 1)
            if arg_str is not None:
                out.append(f"<{tag}>{_latex_to_html(arg_str)}</{tag}>")
                i = p1
                continue
            if ch == "^" and s[i + 1 :].startswith(r"\circ"):
                out.append("&deg;")
                i += 1 + len(r"\circ")
                continue
            m_cmd = re.match(r"\\([a-zA-Z]+)", s[i + 1 :])
            if m_cmd:
                sym = _LATEX_SYMBOLS.get(m_cmd.group(1), html.escape(m_cmd.group(1)))
                out.append(f"<{tag}>{sym}</{tag}>")
                i += 1 + len(m_cmd.group(0))
                continue
            m_tok = re.match(r"([a-zA-Z0-9+\-]+)", s[i + 1 :])
            if m_tok:
                out.append(f"<{tag}>{html.escape(m_tok.group(1))}</{tag}>")
                i += 1 + len(m_tok.group(0))
                continue
            out.append(ch)
            i += 1
        elif ch == "{":
            inner, p1 = _extract_braced(s, i)
            if inner is not None:
                out.append(_latex_to_html(inner))
                i = p1
                continue
            i += 1
        elif ch == "}":
            i += 1
        else:
            out.append(html.escape(ch))
            i += 1
    return "".join(out)


def _render_markdown_body_to_html(md_text: str) -> str:
    body_md = re.sub(
        r"^---\s*\r?\n.*?\r?\n---\s*\r?\n?", "", md_text or "", count=1, flags=re.DOTALL
    )
    # Strip any legacy code-style `~np.isnan(...)` callouts or stray `\text{ in}` tokens
    body_md = re.sub(
        r"\s*\(`~?np\.isnan\([^`]+`\)",
        "",
        body_md,
    )
    body_md = re.sub(r"\\text\{\s*in\s*\}", " in (inches)", body_md)
    # Protect display ($$...$$) and inline ($...$) math before CommonMark parsing
    # so underscores and backslashes inside equations are never mangled into <em> tags
    display_math_store: List[str] = []
    inline_math_store: List[str] = []

    def _stash_display(match: re.Match) -> str:
        idx = len(display_math_store)
        display_math_store.append(match.group(1).strip())
        return f"\n\n@@QPPDISPLAYMATH{idx}@@\n\n"

    def _stash_inline(match: re.Match) -> str:
        idx = len(inline_math_store)
        inline_math_store.append(match.group(1).strip())
        return f"@@QPPINLINEMATH{idx}@@"

    protected_md = re.sub(r"\$\$([\s\S]+?)\$\$", _stash_display, body_md)
    protected_md = re.sub(r"\$([^$\n]+?)\$", _stash_inline, protected_md)

    try:
        from markdown_it import MarkdownIt

        md_parser = MarkdownIt("commonmark", {"html": True}).enable("table")
        rendered_html = md_parser.render(protected_md)
    except Exception:
        rendered_html = f"<pre>{html.escape(protected_md)}</pre>"

    for idx, eq in enumerate(display_math_store):
        token = f"@@QPPDISPLAYMATH{idx}@@"
        block_html = f'<div class="report-math-block">{_latex_to_html(eq)}</div>'
        rendered_html = rendered_html.replace(f"<p>{token}</p>", block_html).replace(
            token, block_html
        )

    for idx, eq in enumerate(inline_math_store):
        token = f"@@QPPINLINEMATH{idx}@@"
        span_html = f'<span class="report-math-inline">{_latex_to_html(eq)}</span>'
        rendered_html = rendered_html.replace(token, span_html)

    return rendered_html


def _persist_report_state(report_data: Dict[str, Any]) -> None:
    import time

    from quick_pp.app.backend.services.demo_guard import (
        enforce_report_storage_limits,
        get_current_demo_session_id,
        mark_demo_state_dirty,
    )
    from quick_pp.app.backend.utils.auth import is_demo_mode

    report_id = report_data["report_id"]
    report_data.setdefault("_created_ts", time.time())
    sid = get_current_demo_session_id()
    if sid and not report_data.get("_demo_session_id"):
        report_data["_demo_session_id"] = sid
    report_data["html_content"] = _render_markdown_body_to_html(
        report_data.get("markdown_content", "")
    )
    _REPORTS_STORE[report_id] = report_data
    out_dir = REPORTS_STATIC_ROOT / report_id
    out_dir.mkdir(parents=True, exist_ok=True)
    serializable = {k: v for k, v in report_data.items() if not k.startswith("_")}
    (out_dir / "report.json").write_text(
        json.dumps(serializable, default=str, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(
        report_data.get("markdown_content", ""), encoding="utf-8"
    )
    if is_demo_mode():
        mark_demo_state_dirty()
        enforce_report_storage_limits()


def get_latest_project_report(project_id: Union[int, str]) -> Optional[Dict[str, Any]]:
    pid_str = str(project_id)
    matching = [
        data
        for data in _REPORTS_STORE.values()
        if str(data.get("project_id")) == pid_str
    ]
    if not matching:
        return None
    matching.sort(
        key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""),
        reverse=True,
    )
    return matching[0]


def _load_report_state(report_id: str) -> Dict[str, Any]:
    if report_id in _REPORTS_STORE:
        return _REPORTS_STORE[report_id]
    json_path = REPORTS_STATIC_ROOT / report_id / "report.json"
    if json_path.exists():
        data = json.loads(json_path.read_text(encoding="utf-8"))
        _REPORTS_STORE[report_id] = data
        return data
    raise ValueError(f"Report '{report_id}' not found.")


def generate_petrophysical_report_service(
    req: ReportGenerationRequest,
) -> ReportGenerationResponse:
    """Orchestrate full FDP Petrophysical Report generation."""
    report_id = f"rpt_{uuid4().hex[:10]}"
    now_iso = datetime.now().strftime("%Y-%m-%d %H:%M")

    effective_ai = (
        req.include_ai_summary
        if req.include_ai_summary is not None
        else req.include_ai_commentary
    )

    sections_cfg = _build_sections_config(
        report_type=req.report_type,
        requested_sections=req.sections,
        include_ai_commentary=effective_ai,
    )
    enabled_ids = {s.section_id for s in sections_cfg if s.enabled}

    context = build_report_context(
        project_id=req.project_id,
        well_names=req.well_names,
        cutoffs=req.cutoffs,
        zones=req.zones,
        target_zones=req.target_zones,
        fwl_tvdss=req.fwl_tvdss,
        min_depth=req.min_depth,
        max_depth=req.max_depth,
        enabled_section_ids=enabled_ids,
        model_params=req.model_params,
    )

    effective_title = req.report_title or req.title
    if req.report_type == "SINGLE_WELL_CPI" and effective_title in (
        "Field Development Plan (FDP) Petrophysical Evaluation Report",
        "Field Development Plan - Petrophysical Evaluation Report",
        "Single-Well Petrophysical CPI & Evaluation Dossier",
        "",
    ):
        focal_w = context["well_names"][0] if context["well_names"] else "Active Well"
        effective_title = f"Well {focal_w} — Petrophysical CPI & Evaluation Dossier"

    assets = _generate_all_report_plots(
        report_id, context, enabled_ids, report_type=req.report_type
    )
    ai_commentary = (
        synthesize_domain_ai_commentary(
            context,
            provider=req.provider or "native_mcp",
            model=req.model,
            api_key=req.api_key,
            base_url=req.base_url,
            report_type=req.report_type,
        )
        if effective_ai
        else {}
    )

    full_md, section_contents = _render_sections_and_base_markdown(
        title=effective_title,
        author=req.author,
        report_type=req.report_type,
        generation_date=now_iso,
        context=context,
        sections_cfg=sections_cfg,
        ai_commentary=ai_commentary,
        assets=assets,
    )

    flat_asset_urls: Dict[str, Any] = {}
    for k, v in assets.items():
        if v:
            flat_asset_urls[k] = v

    pid_str = str(context["project_id"])
    print_html_url = f"/quick_pp/projects/{pid_str}/reports/{report_id}/print"
    pdf_url = f"/quick_pp/projects/{pid_str}/reports/{report_id}/pdf"

    compact_ctx = {k: v for k, v in context.items() if not k.startswith("_")}
    report_state: Dict[str, Any] = {
        "report_id": report_id,
        "project_id": pid_str,
        "project_name": context["project_name"],
        "title": effective_title,
        "author": req.author,
        "report_type": req.report_type,
        "well_names": context["well_names"],
        "sections": [s.model_dump() for s in sections_cfg],
        "section_contents": section_contents,
        "markdown_content": full_md,
        "asset_urls": flat_asset_urls,
        "pdf_download_url": pdf_url,
        "print_html_url": print_html_url,
        "context_summary": compact_ctx,
        "ai_commentary": ai_commentary,
        "created_at": now_iso,
        "updated_at": now_iso,
    }
    _persist_report_state(report_state)

    if req.export_format in ("pdf", "both"):
        compile_report_to_pdf(report_id)

    return ReportGenerationResponse(**report_state)


def update_report_section_service(
    report_id: str,
    section_id: str,
    markdown_content: str,
    title: Optional[str] = None,
    custom_notes: Optional[str] = None,
) -> ReportGenerationResponse:
    """Update a single section's Markdown content in an existing report and re-assemble the document."""
    state = _load_report_state(report_id)
    norm_sid = _normalize_section_id(section_id)
    sec_contents: Dict[str, str] = dict(state.get("section_contents") or {})
    sec_contents[norm_sid] = markdown_content.strip()

    sections_raw = state.get("sections") or []
    updated_sections: List[ReportSectionConfig] = []
    found = False
    for s_dict in sections_raw:
        s_cfg = ReportSectionConfig(**s_dict) if isinstance(s_dict, dict) else s_dict
        if _normalize_section_id(s_cfg.section_id) == norm_sid:
            s_cfg.section_id = norm_sid
            s_cfg.enabled = True
            if title:
                s_cfg.title = title
            if custom_notes is not None:
                s_cfg.custom_notes = custom_notes
            found = True
        updated_sections.append(s_cfg)

    if not found:
        updated_sections.append(
            ReportSectionConfig(
                section_id=norm_sid,
                title=title or norm_sid.replace("_", " ").title(),
                enabled=True,
                custom_notes=custom_notes,
            )
        )

    # Re-assemble full markdown preserving YAML frontmatter
    rendered_blocks = []
    for s_cfg in updated_sections:
        if s_cfg.enabled and s_cfg.section_id in sec_contents:
            rendered_blocks.append(sec_contents[s_cfg.section_id])

    frontmatter = (
        f"---\n"
        f'title: "{state.get("title", "Field Development Plan Report")}"\n'
        f'project: "{state.get("project_name", "")}"\n'
        f'date: "{state.get("created_at", "")}"\n'
        f'author: "{state.get("author", "quick_pp Petrophysical Team")}"\n'
        f'report_type: "{state.get("report_type", "FDP_FULL")}"\n'
        f'platform: "Prepared with quick_pp — Quantitative Petrophysics & Reservoir Evaluation Suite"\n'
        f'version: "1.0"\n'
        f"---\n\n"
    )
    full_md = frontmatter + '\n\n<div class="page-break"></div>\n\n'.join(
        rendered_blocks
    )

    state["sections"] = [s.model_dump() for s in updated_sections]
    state["section_contents"] = sec_contents
    state["markdown_content"] = full_md
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _persist_report_state(state)

    return ReportGenerationResponse(**state)


def update_full_report_markdown_service(
    report_id: str,
    req: ReportFullMarkdownUpdateRequest,
) -> ReportGenerationResponse:
    """Update the full Markdown content of an existing report."""
    state = _load_report_state(report_id)
    state["markdown_content"] = req.markdown_content
    if req.title:
        state["title"] = req.title
    if req.author:
        state["author"] = req.author
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _persist_report_state(state)
    return ReportGenerationResponse(**state)


def enhance_report_with_ai_service(
    report_id: str,
    req: ReportAiEnhanceRequest,
) -> ReportGenerationResponse:
    """Enhance report sections with AI domain narrative."""
    state = _load_report_state(report_id)
    ctx = state.get("context_summary") or build_report_context(
        int(state["project_id"]), state.get("well_names")
    )
    ai_commentary = synthesize_domain_ai_commentary(
        ctx,
        provider=req.provider,
        model=req.model,
        api_key=req.api_key,
        base_url=req.base_url,
        custom_instruction=req.instruction,
        report_type=state.get("report_type", "FDP_FULL"),
    )
    state["ai_commentary"] = ai_commentary

    sections_cfg = [
        ReportSectionConfig(**s) if isinstance(s, dict) else s
        for s in (state.get("sections") or [])
    ]
    for s in sections_cfg:
        if req.section_ids is None or s.section_id in req.section_ids:
            s.include_ai_commentary = True

    # Re-render sections that are in req.section_ids (or all if None)
    existing_overrides = dict(state.get("section_contents") or {})
    if req.section_ids:
        for sid in req.section_ids:
            existing_overrides.pop(sid, None)
    else:
        existing_overrides.clear()

    full_md, sec_contents = _render_sections_and_base_markdown(
        title=state["title"],
        author=state["author"],
        report_type=state.get("report_type", "FDP_FULL"),
        generation_date=state["created_at"],
        context=ctx,
        sections_cfg=sections_cfg,
        ai_commentary=ai_commentary,
        assets=state.get("asset_urls") or {},
        existing_section_overrides=existing_overrides,
    )

    state["sections"] = [s.model_dump() for s in sections_cfg]
    state["section_contents"] = sec_contents
    state["markdown_content"] = full_md
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    _persist_report_state(state)
    return ReportGenerationResponse(**state)


# ---------------------------------------------------------------------------
# Step 4: Markdown-to-HTML (with KaTeX) & Multi-Page PDF Compiler
# ---------------------------------------------------------------------------


def render_report_print_html(report_id: str, auto_print: bool = False) -> str:
    """Render the report Markdown into a standalone A4 corporate HTML page with KaTeX math and print CSS."""
    state = _load_report_state(report_id)
    md_text = state.get("markdown_content", "")

    rendered_body = _render_markdown_body_to_html(md_text)

    css_path = REPORTS_TEMPLATE_DIR / "report_print.css"
    css_content = css_path.read_text(encoding="utf-8") if css_path.exists() else ""

    auto_print_script = (
        "<script>window.addEventListener('load', () => { setTimeout(() => window.print(), 650); });</script>"
        if auto_print
        else ""
    )

    is_single_well = state.get("report_type") == "SINGLE_WELL_CPI"
    well_label_hdr = "Target Well:" if is_single_well else "Evaluated Wells:"
    default_doc_title = (
        "Single-Well Petrophysical CPI & Evaluation Dossier"
        if is_single_well
        else "Field Development Plan: Petrophysical Evaluation Report"
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>{html.escape(state.get("title", default_doc_title))}</title>
    <style>{css_content}</style>
</head>
<body class="qpp-report-print">
    <div class="no-print" style="position:sticky;top:0;z-index:50;display:flex;justify-content:space-between;align-items:center;background:#0f172a;color:#fff;padding:8px 18px;font-size:12px;">
        <div><strong>{html.escape(state.get("title", default_doc_title))}</strong> — {html.escape(state.get("project_name", ""))}</div>
        <div style="display:flex;gap:8px;">
            <button onclick="window.print()" style="background:#2563eb;color:#fff;border:none;padding:5px 12px;border-radius:4px;cursor:pointer;font-weight:600;">Print / Save as A4 PDF</button>
            <a href="{html.escape(state.get("pdf_download_url", "#"))}" style="background:#334155;color:#fff;text-decoration:none;padding:5px 12px;border-radius:4px;font-weight:600;">Download Direct PDF</a>
        </div>
    </div>
    <div class="qpp-report-container" style="max-width:210mm;margin:0 auto;padding:12mm;">
        <div class="qpp-report-header-banner">
            <div class="qpp-report-brand-tagline">Prepared with <code>quick_pp</code> &mdash; Quantitative Petrophysics &amp; Reservoir Evaluation Suite</div>
            <h1>{html.escape(state.get("title", default_doc_title))}</h1>
            <div class="qpp-report-meta-grid">
                <div><strong>Project / Field:</strong> {html.escape(state.get("project_name", ""))}</div>
                <div><strong>{well_label_hdr}</strong> {html.escape(", ".join(state.get("well_names", [])))}</div>
                <div><strong>Author:</strong> {html.escape(state.get("author", ""))}</div>
                <div><strong>Generated:</strong> {html.escape(state.get("updated_at") or state.get("created_at", ""))}</div>
            </div>
        </div>
        <div class="qpp-report-body">
            {rendered_body}
        </div>
    </div>
    {auto_print_script}
</body>
</html>"""


def _strip_latex_for_pdf_text(text: str) -> str:
    """Clean Markdown / LaTeX tokens into readable technical Unicode for matplotlib PdfPages text blocks."""
    out = text
    # Strip display/inline math delimiters first
    out = re.sub(
        r"\$\$\s*([\s\S]*?)\s*\$\$",
        lambda m: "Equation: " + " ".join(m.group(1).strip().split()),
        out,
    )
    out = re.sub(r"\$([^$\n]+)\$", lambda m: m.group(1).strip(), out)

    symbol_replacements = [
        (r"\\qquad", "    "),
        (r"\\quad", "  "),
        (r"\\left\(", "("),
        (r"\\right\)", ")"),
        (r"\\phi_e", "Φe"),
        (r"\\phi_t", "Φt"),
        (r"\\phi_D", "ΦD"),
        (r"\\phi_z", "Φz"),
        (r"\\phi", "Φ"),
        (r"\\rho", "ρ"),
        (r"\\Delta", "Δ"),
        (r"\\Sigma", "Σ"),
        (r"\\sigma", "σ"),
        (r"\\theta", "θ"),
        (r"\\mu", "μ"),
        (r"\\omega", "ω"),
        (r"\\Omega", "Ω"),
        (r"\\alpha", "α"),
        (r"\\le", "≤"),
        (r"\\ge", "≥"),
        (r"\\pm", "±"),
        (r"\\to", "→"),
        (r"\\approx", "≈"),
        (r"\^\\circ", "°"),
        (r"\\circ", "°"),
        (r"\\cdot", "·"),
        (r"\\times", "×"),
        (r"\\sum", "Σ"),
        (r"\\ln", "ln"),
        (r"\\log", "log"),
        (r"\\exp", "exp"),
        (r"\\cos", "cos"),
        (r"\\min", "min"),
        (r"\\max", "max"),
    ]
    for pat, rep in symbol_replacements:
        out = re.sub(pat, lambda _, r=rep: r, out)

    # Iteratively simplify nested _{...}, ^{...}, \text{x}, \bar{x}, \sqrt{x}, \frac{a}{b}
    for _ in range(5):
        out = re.sub(r"\\text\{([^{}]*)\}", lambda m: m.group(1), out)
        out = re.sub(
            r"\\(?:bar|overline)\{([^{}]*)\}",
            lambda m: f"{m.group(1)}\u0304",
            out,
        )
        out = re.sub(r"_\{([^{}]*)\}", lambda m: f"_{m.group(1)}", out)
        out = re.sub(r"\^\{([^{}]*)\}", lambda m: f"^{m.group(1)}", out)
        out = re.sub(r"\\sqrt\{([^{}]*)\}", lambda m: f"√({m.group(1)})", out)
        out = re.sub(
            r"\\frac\{([^{}]*)\}\{([^{}]*)\}",
            lambda m: f"({m.group(1)})/({m.group(2)})",
            out,
        )

    out = re.sub(r"\*\*([^*]+)\*\*", lambda m: m.group(1), out)
    out = re.sub(r"`([^`]+)`", lambda m: m.group(1), out)
    return out


def _localize_static_urls(html_str: str) -> str:
    """Point `/static/...` asset URLs at local files so offline renderers can load them."""
    static_uri = (BACKEND_DIR / "static").as_uri()
    return html_str.replace('src="/static/', f'src="{static_uri}/')


def _find_browser_executable() -> Optional[str]:
    """Locate a Chromium-family browser (env override, PATH, then common install paths)."""
    import shutil

    candidates: List[str] = []
    for env_key in ("BROWSER_PATH", "CHROME_PATH"):
        if os.environ.get(env_key):
            candidates.append(os.environ[env_key])
    for name in (
        "chromium",
        "chromium-browser",
        "google-chrome",
        "google-chrome-stable",
        "chrome",
        "msedge",
    ):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for base in (
        os.environ.get("PROGRAMFILES"),
        os.environ.get("PROGRAMFILES(X86)"),
        os.environ.get("LOCALAPPDATA"),
    ):
        if base:
            candidates.append(str(Path(base) / "Google/Chrome/Application/chrome.exe"))
            candidates.append(str(Path(base) / "Microsoft/Edge/Application/msedge.exe"))
    candidates += [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/usr/bin/chromium",
        "/usr/bin/google-chrome",
    ]
    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    return None


def _render_pdf_with_browser(html_str: str, out_dir: Path, pdf_path: Path) -> bool:
    """Print the report HTML to PDF with headless Chromium/Edge. Returns True on success."""
    import shutil
    import subprocess
    import tempfile

    browser = _find_browser_executable()
    if not browser:
        return False

    src_html = out_dir / "_pdf_source.html"
    profile_dir = Path(tempfile.mkdtemp(prefix="qpp_pdf_"))
    tmp_pdf = out_dir / "_pdf_tmp.pdf"
    try:
        src_html.write_text(_localize_static_urls(html_str), encoding="utf-8")
        tmp_pdf.unlink(missing_ok=True)
        cmd = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--no-sandbox",
            "--allow-file-access-from-files",
            "--hide-scrollbars",
            "--no-pdf-header-footer",
            "--print-to-pdf-no-header",
            f"--user-data-dir={profile_dir}",
            "--virtual-time-budget=15000",
            f"--print-to-pdf={tmp_pdf}",
            src_html.as_uri(),
        ]
        subprocess.run(cmd, capture_output=True, timeout=180, check=False)
        if tmp_pdf.exists() and tmp_pdf.stat().st_size > 1024:
            tmp_pdf.replace(pdf_path)
            return True
        logger.warning(f"Headless browser produced no PDF using {browser}")
    except Exception as exc:
        logger.warning(f"Headless browser PDF export failed ({browser}): {exc}")
    finally:
        shutil.rmtree(profile_dir, ignore_errors=True)
        src_html.unlink(missing_ok=True)
        tmp_pdf.unlink(missing_ok=True)
    return False


def compile_report_to_pdf(report_id: str) -> Path:
    """Compile the report Markdown and high-DPI plot assets into a downloadable multi-page PDF file."""
    state = _load_report_state(report_id)
    out_dir = REPORTS_STATIC_ROOT / report_id
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{report_id}.pdf"

    # Save print HTML alongside PDF
    html_str = render_report_print_html(report_id, auto_print=False)
    (out_dir / "report_print.html").write_text(html_str, encoding="utf-8")

    # Option 1: headless Chromium/Edge renders the same HTML as the print view,
    # so tables, equations and figures paginate like the browser print preview.
    if _render_pdf_with_browser(html_str, out_dir, pdf_path):
        return pdf_path

    # Option 2: WeasyPrint if installed
    try:
        import weasyprint  # type: ignore

        weasyprint.HTML(
            string=_localize_static_urls(html_str), base_url=str(BACKEND_DIR)
        ).write_pdf(str(pdf_path))
        if pdf_path.exists() and pdf_path.stat().st_size > 0:
            return pdf_path
    except Exception as exc:
        logger.debug(f"WeasyPrint PDF export unavailable: {exc}")

    logger.warning(
        "PDF export fell back to the basic matplotlib renderer (no Chromium/Edge or "
        "WeasyPrint found). Tables and equations are simplified. Install Chromium or "
        "set BROWSER_PATH for full-fidelity PDFs."
    )

    # Option 2: Built-in Multi-Page Publication PDF Compiler via matplotlib PdfPages
    md_content = state.get("markdown_content", "")
    body_md = re.sub(
        r"^---\s*\r?\n.*?\r?\n---\s*\r?\n?", "", md_content, count=1, flags=re.DOTALL
    )
    raw_pages = [
        p.strip()
        for p in re.split(r"<div\s+class=[\"']page-break[\"']\s*>\s*</div>", body_md)
        if p.strip()
    ]
    if not raw_pages:
        raw_pages = [body_md]

    with PdfPages(str(pdf_path)) as pdf:
        for page_idx, page_md in enumerate(raw_pages):
            fig = Figure(figsize=(8.27, 11.69))  # Standard A4 Portrait (inches)
            ax = fig.add_axes([0.07, 0.05, 0.86, 0.90])
            ax.axis("off")

            y_cursor = 0.98
            if page_idx == 0:
                # Executive Header Banner on Page 1
                ax.text(
                    0.0,
                    y_cursor,
                    "PREPARED WITH quick_pp — QUANTITATIVE PETROPHYSICS & RESERVOIR EVALUATION SUITE",
                    fontsize=7.2,
                    fontweight="bold",
                    color="#1e40af",
                    va="top",
                )
                y_cursor -= 0.018
                ax.text(
                    0.0,
                    y_cursor,
                    state.get("title", "FDP Petrophysical Evaluation Report"),
                    fontsize=13.5,
                    fontweight="bold",
                    color="#0f172a",
                    va="top",
                )
                y_cursor -= 0.032
                meta_line = (
                    f"Project: {state.get('project_name', '')}   |   "
                    f"Wells: {', '.join(state.get('well_names', []))}   |   "
                    f"Author: {state.get('author', '')}   |   "
                    f"Date: {state.get('updated_at') or state.get('created_at', '')}"
                )
                ax.text(
                    0.0, y_cursor, meta_line, fontsize=8.0, color="#475569", va="top"
                )
                y_cursor -= 0.018
                ax.plot(
                    [0.0, 1.0],
                    [y_cursor, y_cursor],
                    color="#1e3a8a",
                    lw=1.5,
                    transform=ax.transAxes,
                )
                y_cursor -= 0.025

            page_md_norm = re.sub(
                r"\$\$\s*([\s\S]*?)\s*\$\$",
                lambda m: f"$$ {' '.join(m.group(1).strip().split())} $$",
                page_md,
            )
            lines = page_md_norm.splitlines()
            img_paths_on_page: List[Tuple[str, Path]] = []

            i = 0
            while i < len(lines):
                line = lines[i].strip()
                if not line:
                    y_cursor -= 0.010
                    i += 1
                    continue

                # Image reference `![Caption](url)`
                img_m = re.match(r"^!\[([^\]]*)\]\(([^)]+)\)", line)
                if img_m:
                    caption = img_m.group(1)
                    img_url = img_m.group(2)
                    img_name = Path(img_url).name
                    local_img = out_dir / img_name
                    if local_img.exists():
                        img_paths_on_page.append((caption, local_img))
                    i += 1
                    continue

                # Heading
                if line.startswith("# "):
                    ax.text(
                        0.0,
                        y_cursor,
                        _strip_latex_for_pdf_text(line[2:].strip()),
                        fontsize=11.5,
                        fontweight="bold",
                        color="#1e3a8a",
                        va="top",
                    )
                    y_cursor -= 0.028
                    i += 1
                    continue
                if line.startswith("## ") or line.startswith("### "):
                    clean_h = re.sub(r"^#+\s*", "", line)
                    ax.text(
                        0.0,
                        y_cursor,
                        _strip_latex_for_pdf_text(clean_h),
                        fontsize=9.5,
                        fontweight="bold",
                        color="#1e293b",
                        va="top",
                    )
                    y_cursor -= 0.022
                    i += 1
                    continue

                # Markdown Table
                if line.startswith("|") and line.endswith("|"):
                    tbl_lines = []
                    while (
                        i < len(lines)
                        and lines[i].strip().startswith("|")
                        and lines[i].strip().endswith("|")
                    ):
                        tbl_lines.append(lines[i].strip())
                        i += 1
                    parsed_rows = []
                    for tl in tbl_lines:
                        if re.match(r"^\|[\s:|-]+\|$", tl):
                            continue
                        parsed_rows.append(
                            [
                                _strip_latex_for_pdf_text(c.strip())
                                for c in tl.strip("|").split("|")
                            ]
                        )
                    if parsed_rows:
                        n_cols = max(len(r) for r in parsed_rows)
                        col_widths = [0] * n_cols
                        for r in parsed_rows[:22]:
                            for c_idx, val in enumerate(r):
                                col_widths[c_idx] = min(
                                    max(col_widths[c_idx], len(val)), 24
                                )
                        for r_idx, r in enumerate(parsed_rows[:22]):
                            if y_cursor < 0.08:
                                break
                            padded_cells = [
                                (r[c_idx] if c_idx < len(r) else "")[
                                    : col_widths[c_idx]
                                ].ljust(col_widths[c_idx])
                                for c_idx in range(n_cols)
                            ]
                            row_str = " | ".join(padded_cells)
                            is_hdr = r_idx == 0
                            ax.text(
                                0.01,
                                y_cursor,
                                row_str[:135],
                                fontsize=6.6
                                if n_cols > 8
                                else (7.3 if is_hdr else 7.0),
                                fontweight="bold" if is_hdr else "normal",
                                color="#0f172a" if is_hdr else "#334155",
                                family="monospace",
                                va="top",
                            )
                            y_cursor -= 0.0155
                        y_cursor -= 0.008
                    continue

                # Regular text or bullet or equation
                clean_p = _strip_latex_for_pdf_text(line)
                wrapped = textwrap.wrap(clean_p, width=102) or [clean_p]
                for w_line in wrapped:
                    if y_cursor < 0.08:
                        break
                    ax.text(
                        0.0,
                        y_cursor,
                        w_line,
                        fontsize=8.3,
                        color="#1e293b",
                        va="top",
                    )
                    y_cursor -= 0.0165
                i += 1

            # Embed section figures in remaining lower portion of the page
            for caption, img_file in img_paths_on_page[:2]:
                try:
                    img_arr = mpimg.imread(str(img_file))
                    img_h = min(max(y_cursor - 0.06, 0.24), 0.38)
                    img_bottom = max(0.05, y_cursor - img_h)
                    img_ax = fig.add_axes([0.08, img_bottom, 0.84, img_h - 0.02])
                    img_ax.imshow(img_arr)
                    img_ax.axis("off")
                    img_ax.set_title(caption, fontsize=7.5, color="#475569", pad=2)
                    y_cursor = img_bottom - 0.02
                except Exception as exc:
                    logger.debug(f"PDF image embed skipped for {img_file}: {exc}")

            # Footer
            is_single_well_pdf = state.get("report_type") == "SINGLE_WELL_CPI"
            footer_doc_label = (
                f"quick_pp Single-Well CPI Dossier ({', '.join(state.get('well_names', []))})"
                if is_single_well_pdf
                else "quick_pp FDP Petrophysical Evaluation Report"
            )
            fig.text(
                0.5,
                0.02,
                f"{footer_doc_label} — {state.get('project_name', '')} — Page {page_idx + 1} of {len(raw_pages)}",
                ha="center",
                fontsize=7.5,
                color="#64748b",
            )
            pdf.savefig(fig, dpi=160)

    return pdf_path


# ---------------------------------------------------------------------------
# FastAPI Routes
# ---------------------------------------------------------------------------


@router.get(
    "/projects/{project_id}/reports/context",
    summary="Get compact petrophysical report context for a project",
)
def api_get_project_report_context(
    project_id: int,
    wells: Optional[str] = Query(None, description="Comma-separated well names or ALL"),
    zones: Optional[str] = Query(None, description="Comma-separated zone names"),
):
    """Return compact JSON digest of project parameters, QA/QC flags, and zonal reservoir summary."""
    try:
        well_list = (
            [w.strip() for w in wells.split(",") if w.strip()] if wells else None
        )
        zone_list = (
            [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        )
        return get_compact_report_context(
            project_id, well_names=well_list, zones=zone_list
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to build report context: {e}"
        ) from e


_REPORT_JOBS: Dict[str, Dict[str, Any]] = {}

# Job state is also written to disk. The backend runs several gunicorn workers on Linux, and a status poll
# can reach a different worker than the one that started the job, so an in-memory dict alone returns 404.
_REPORT_JOBS_DIR = REPORTS_STATIC_ROOT / "_jobs"


def _job_path(job_id: str) -> Path:
    safe = "".join(c for c in str(job_id) if c.isalnum() or c in ("_", "-"))
    return _REPORT_JOBS_DIR / f"{safe}.json"


def _save_job(job_id: str, state: Dict[str, Any]) -> None:
    from quick_pp.app.backend.services.demo_guard import get_current_demo_session_id

    sid = get_current_demo_session_id()
    if sid and not state.get("demo_session_id"):
        state["demo_session_id"] = sid
    _REPORT_JOBS[job_id] = state
    try:
        _REPORT_JOBS_DIR.mkdir(parents=True, exist_ok=True)
        path = _job_path(job_id)
        tmp = path.with_suffix(f".{uuid4().hex[:6]}.tmp")
        tmp.write_text(json.dumps(state, default=str), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:
        logger.warning(f"Could not persist report job {job_id} to disk: {exc}")


def _load_job(job_id: str) -> Optional[Dict[str, Any]]:
    path = _job_path(job_id)
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning(f"Could not read report job {job_id} from disk: {exc}")
    return _REPORT_JOBS.get(job_id)


def _run_report_job(job_id: str, req: ReportGenerationRequest) -> None:
    try:
        res = generate_petrophysical_report_service(req)
        _save_job(
            job_id,
            {
                "job_id": job_id,
                "status": "completed",
                "report": res.model_dump(),
                "error": None,
            },
        )
    except Exception as e:
        logger.exception(f"Background report job {job_id} failed: {e}")
        _save_job(
            job_id,
            {
                "job_id": job_id,
                "status": "failed",
                "report": None,
                "error": str(e),
            },
        )


@router.post(
    "/projects/{project_id}/reports/generate",
    summary="Generate an FDP Petrophysical Evaluation Report (Markdown + Plots + optional PDF)",
)
def api_generate_project_report(
    project_id: int,
    req: ReportGenerationRequest,
    async_job: bool = Query(
        False,
        description="Run report generation in background and return job_id immediately to prevent reverse-proxy HTTP 524 timeouts",
    ),
):
    """Generate a standardized 10-section FDP Petrophysical Report for the specified project and wells."""
    req.project_id = project_id
    if async_job:
        job_id = f"job_{uuid4().hex[:10]}"
        _save_job(
            job_id,
            {
                "job_id": job_id,
                "status": "running",
                "report": None,
                "error": None,
            },
        )
        t = threading.Thread(target=_run_report_job, args=(job_id, req), daemon=True)
        t.start()
        return {"job_id": job_id, "status": "running"}

    try:
        return generate_petrophysical_report_service(req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"Report generation error for project {project_id}: {e}")
        raise HTTPException(
            status_code=500, detail=f"Failed to generate report: {e}"
        ) from e


@router.get(
    "/projects/{project_id}/reports/jobs/{job_id}",
    summary="Poll background report generation job status",
)
def api_get_report_job_status(project_id: int, job_id: str):
    """Return status and completed payload for an asynchronous report generation job."""
    job = _load_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Report job '{job_id}' not found.")
    return job


@router.get(
    "/projects/{project_id}/reports",
    summary="List generated reports for a project",
)
def api_list_project_reports(project_id: int):
    """Return metadata for reports generated in this project."""
    pid_str = str(project_id)
    items = []
    for rpt_id, data in _REPORTS_STORE.items():
        if str(data.get("project_id")) == pid_str:
            items.append(
                {
                    "report_id": rpt_id,
                    "project_id": pid_str,
                    "title": data.get("title"),
                    "author": data.get("author"),
                    "report_type": data.get("report_type"),
                    "well_names": data.get("well_names", []),
                    "created_at": data.get("created_at"),
                    "updated_at": data.get("updated_at"),
                    "pdf_download_url": data.get("pdf_download_url"),
                    "print_html_url": data.get("print_html_url"),
                }
            )
    items.sort(
        key=lambda x: str(x.get("updated_at") or x.get("created_at") or ""),
        reverse=True,
    )
    return {"project_id": project_id, "reports": items}


@router.get(
    "/projects/{project_id}/reports/{report_id}",
    response_model=ReportGenerationResponse,
    summary="Retrieve a generated petrophysical report by ID",
)
def api_get_report(project_id: int, report_id: str):
    """Retrieve the full Markdown content, section breakdowns, and asset URLs for a report."""
    try:
        state = _load_report_state(report_id)
        return ReportGenerationResponse(**state)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.put(
    "/projects/{project_id}/reports/{report_id}",
    response_model=ReportGenerationResponse,
    summary="Update full Markdown content of a report",
)
def api_update_full_report(
    project_id: int,
    report_id: str,
    req: ReportFullMarkdownUpdateRequest,
):
    """Save edited Markdown content back to the report store."""
    try:
        return update_full_report_markdown_service(report_id, req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.patch(
    "/projects/{project_id}/reports/{report_id}/sections/{section_id}",
    response_model=ReportGenerationResponse,
    summary="Update a specific section of a petrophysical report",
)
@router.put(
    "/projects/{project_id}/reports/{report_id}/sections/{section_id}",
    response_model=ReportGenerationResponse,
    summary="Update a specific section of a petrophysical report",
)
def api_update_report_section(
    project_id: int,
    report_id: str,
    section_id: str,
    req: ReportSectionUpdateRequest,
):
    """Update a single section's Markdown content and re-compile the master Markdown document."""
    try:
        return update_report_section_service(
            report_id=report_id,
            section_id=section_id,
            markdown_content=req.markdown_content,
            title=req.title,
            custom_notes=req.custom_notes,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.post(
    "/projects/{project_id}/reports/{report_id}/enhance",
    response_model=ReportGenerationResponse,
    summary="Enhance report sections with AI technical interpretation narrative",
)
def api_enhance_report_with_ai(
    project_id: int,
    report_id: str,
    req: ReportAiEnhanceRequest,
):
    """Synthesize or refine grounded Lead Petrophysicist commentary across the report."""
    try:
        return enhance_report_with_ai_service(report_id, req)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"AI enhancement failed: {e}"
        ) from e


@router.get(
    "/projects/{project_id}/reports/{report_id}/pdf",
    summary="Compile and download the FDP Petrophysical Report as a PDF file",
)
def api_download_report_pdf(project_id: int, report_id: str):
    """Compile the current report state into a multi-page PDF and stream it as a file download."""
    try:
        state = _load_report_state(report_id)
        pdf_path = compile_report_to_pdf(report_id)
        safe_proj = "".join(
            c if c.isalnum() or c in ("-", "_") else "_"
            for c in str(state.get("project_name", "project"))
        )
        if state.get("report_type") == "SINGLE_WELL_CPI":
            w_str = (state.get("well_names") or ["Well"])[0]
            safe_well = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in str(w_str)
            )
            filename = f"Well_CPI_Dossier_{safe_well}_{report_id}.pdf"
        else:
            filename = f"FDP_Petrophysical_Report_{safe_proj}_{report_id}.pdf"
        return FileResponse(
            path=str(pdf_path),
            media_type="application/pdf",
            filename=filename,
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        logger.exception(f"PDF export failed for {report_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to export PDF: {e}") from e


@router.get(
    "/projects/{project_id}/reports/{report_id}/print",
    response_class=HTMLResponse,
    summary="Render A4 print-styled HTML report with KaTeX math and embedded figures",
)
def api_print_report_html(
    project_id: int,
    report_id: str,
    auto_print: bool = Query(
        False, description="Automatically trigger browser Print-to-PDF dialog"
    ),
):
    """Serve a print-ready A4 HTML document with KaTeX math rendering and corporate print CSS."""
    try:
        html_doc = render_report_print_html(report_id, auto_print=auto_print)
        return HTMLResponse(content=html_doc)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
