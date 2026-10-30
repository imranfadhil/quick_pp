import base64
import csv
import difflib
import io
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, Body, File, Form, HTTPException, Query, UploadFile
from sqlalchemy import select

from quick_pp.app.backend.schemas.well_metadata import (
    AncillaryColumnMapping,
    AncillaryCommitRequest,
    AncillaryPreviewResponse,
)
from quick_pp.app.backend.services.database import invalidate_well_cache
from quick_pp.app.backend.utils.db import get_db
from quick_pp.database import objects as db_objects

# Project-level router: supports project-based endpoints and accepts optional
# `well_name` as a query parameter or in POST bodies. When a `well_name` is
# provided the handlers delegate to the existing well-based behavior; when
# absent GET endpoints aggregate across all wells in the project.
router = APIRouter(prefix="/database/projects/{project_id}", tags=["Ancillary Data"])
import_ancillary_router = APIRouter(prefix="/import/ancillary", tags=["Ancillary Data"])


# ==============================================================================
# Phase 2: Three-Stage Column Resolution & Multi-Sheet Excel/CSV Parser
# ==============================================================================

ANCILLARY_TARGET_SCHEMAS: Dict[str, Dict[str, List[str]]] = {
    "tops": {
        "WELL_NAME": [
            "well",
            "well_name",
            "wellname",
            "uwi",
            "wellbore",
            "well_id",
            "borehole",
        ],
        "TOP_NAME": [
            "name",
            "top",
            "top_name",
            "formation",
            "surface",
            "zone",
            "horizon",
            "strat",
            "marker",
        ],
        "MD": [
            "depth_md",
            "md",
            "measured_depth",
            "top_md",
            "md_m",
            "md_ft",
            "depth",
            "depth_m",
            "depth_ft",
        ],
        "TVD": [
            "depth_tvdss",
            "tvd",
            "tvdss",
            "true_vertical_depth",
            "tvd_m",
            "tvd_ft",
        ],
    },
    "surveys": {
        "WELL_NAME": ["well", "well_name", "wellname", "uwi", "wellbore", "well_id"],
        "MD": [
            "md",
            "measured_depth",
            "depth_md",
            "depth",
            "md_m",
            "depth_m",
            "md_ft",
            "depth_ft",
            "measured depth",
        ],
        "INC": [
            "inc",
            "incl",
            "inclination",
            "dip",
            "inc_deg",
            "hole_inc",
            "deviation",
        ],
        "AZIM": ["azim", "azi", "azimuth", "dir", "direction", "azim_deg", "azi_deg"],
        "TVD": [
            "tvd",
            "true_vertical_depth",
            "tvd_m",
            "tvd_ft",
            "depth_tvdss",
            "tvdss",
        ],
    },
    "pressures": {
        "WELL_NAME": ["well", "well_name", "wellname", "uwi", "wellbore", "well_id"],
        "DEPTH": [
            "depth",
            "md",
            "depth_md",
            "tvd",
            "measured_depth",
            "test_depth",
            "depth_m",
            "depth_ft",
        ],
        "PRESSURE": [
            "pressure",
            "pres",
            "form_pressure",
            "formation_pressure",
            "p_psi",
            "p_bar",
            "p_kpa",
            "p_mpa",
            "Shut-in Pressure",
        ],
        "PRESSURE_GRADIENT": [
            "gradient",
            "pressure_gradient",
            "grad",
            "fluid_gradient",
        ],
        "MOBILITY": ["mobility", "mob", "drawdown_mobility", "md_cp"],
        "TEST_TYPE": ["test_type", "tool", "type", "rpt", "mdt", "rdt", "sft"],
        "REMARK": ["remark", "remarks", "comment", "comments", "quality", "status"],
    },
    "contacts": {
        "WELL_NAME": ["well", "well_name", "wellname", "uwi", "wellbore", "well_id"],
        "CONTACT_TYPE": [
            "contact_type",
            "type",
            "contact",
            "fluid_contact",
            "name",
            "fluid",
        ],
        "DEPTH": [
            "depth",
            "md",
            "depth_md",
            "tvd",
            "tvdss",
            "contact_depth",
            "depth_m",
            "depth_ft",
        ],
    },
    "rca": {
        "WELL_NAME": ["well", "well_name", "wellname", "uwi", "wellbore", "well_id"],
        "SAMPLE_NAME": [
            "sample",
            "sample_id",
            "sample_name",
            "plug",
            "plug_id",
            "plug_no",
            "sample_no",
            "id",
            "no",
            "number",
        ],
        "DEPTH": [
            "depth",
            "core_depth",
            "md",
            "depth_md",
            "sample_depth",
            "plug_depth",
            "depth_m",
            "depth_ft",
            "top_depth",
        ],
        "CPORE": [
            "cpore",
            "porosity",
            "por_hor",
            "por_sum",
            "por_vert",
            "por",
            "phi",
            "phit",
            "helium_porosity",
            "porosity_he",
            "core_porosity",
            "por_pct",
            "por_frac",
            "he_porosity",
            "ambient_porosity",
            "ncs_porosity",
        ],
        "CPERM": [
            "cperm",
            "permeability",
            "kl_hor",
            "kg_hor",
            "k_hor",
            "perm",
            "k",
            "kh",
            "kl",
            "kair",
            "klinkenberg",
            "horizontal_permeability",
            "perm_h",
            "k_h",
            "air_permeability",
            "perm_md",
            "permeability_horizontal",
            "permeability_klinkenberg",
            "kh_md",
        ],
        "CPERMV": [
            "cpermv",
            "kl_vert",
            "kg_vert",
            "k_vert",
            "kv",
            "vertical_permeability",
            "perm_v",
            "k_v",
            "permeability_vertical",
        ],
        "CGRAIN_DENSITY": [
            "grain_density",
            "rhog",
            "gd",
            "matrix_density",
            "grain_dens",
            "rhoma",
            "density_grain",
        ],
        "CSAT_WATER": [
            "sw",
            "water_saturation",
            "sw_core",
            "dean_stark_sw",
            "sat_water",
        ],
        "CSAT_OIL": ["so", "oil_saturation", "so_core", "sat_oil"],
        "DESCRIPTION": [
            "description",
            "lithological_description",
            "lithology",
            "facies",
            "rock_type",
            "desc",
        ],
        "REMARK": ["remark", "remarks", "comment", "comments", "note"],
    },
    "scal": {
        "WELL_NAME": ["well", "well_name", "wellname", "uwi", "wellbore", "well_id"],
        "SAMPLE_NAME": ["sample", "sample_id", "sample_name", "plug", "plug_id", "id"],
        "DEPTH": ["depth", "core_depth", "md", "depth_md", "sample_depth"],
        "CPORE": [
            "cpore",
            "porosity",
            "por_hor",
            "por",
            "phi",
            "phit",
            "helium_porosity",
            "core_porosity",
            "por_pct",
            "por_frac",
        ],
        "CPERM": [
            "cperm",
            "permeability",
            "kl_hor",
            "kg_hor",
            "k_hor",
            "perm",
            "k",
            "kh",
            "kl",
            "kair",
            "klinkenberg",
            "perm_md",
        ],
        "PC": [
            "pc",
            "capillary_pressure",
            "pressure",
            "pc_psi",
            "pc_bar",
            "pc_kpa",
            "p_c",
        ],
        "SW": [
            "sw",
            "saturation",
            "water_saturation",
            "sw_frac",
            "sw_pct",
            "wetting_phase_saturation",
        ],
        "EXPERIMENT_TYPE": [
            "experiment_type",
            "experiment",
            "method",
            "type",
            "test_type",
            "micp",
            "porous_plate",
            "centrifuge",
        ],
        "CYCLE": ["cycle", "process", "drainage_imbibition", "displacing_phase"],
        "DESCRIPTION": [
            "description",
            "lithological_description",
            "lithology",
            "facies",
            "rock_type",
        ],
    },
    "facies": {
        "WELL_NAME": ["well", "well_name", "wellname", "uwi", "wellbore", "well_id"],
        "DEPTH_TOP": [
            "top",
            "depth_top",
            "from",
            "top_depth",
            "start_depth",
            "top_m",
            "md_top",
            "depth",
        ],
        "DEPTH_BASE": [
            "base",
            "bottom",
            "depth_base",
            "to",
            "base_depth",
            "end_depth",
            "bottom_depth",
            "base_m",
            "md_base",
        ],
        "FACIES": [
            "facies",
            "lithofacies",
            "rock_type",
            "lithology",
            "code",
            "facies_name",
            "class",
            "zone",
        ],
    },
}

UNIT_EXTRACT_PATTERN = re.compile(r"[\(\[\{]\s*([a-zA-Z0-9/%.\-\^]+)\s*[\)\]\}]")
WIDE_PC_COL_PATTERN = re.compile(r"^pc[\._\-\s]*(\d+)$", re.IGNORECASE)
WIDE_SW_COL_PATTERN = re.compile(r"^(?:sw|sat|s_w)[\._\-\s]*(\d+)$", re.IGNORECASE)


def _extract_unit_from_header(header_str: str) -> Optional[str]:
    """Extract embedded unit from column header like 'Porosity (%)' or 'Depth [m]'."""
    m = UNIT_EXTRACT_PATTERN.search(str(header_str))
    if m:
        return m.group(1).strip()
    lower = str(header_str).lower()
    if lower.endswith("_pct") or "%" in lower:
        return "%"
    if lower.endswith("_psi") or "psi" in lower:
        return "psi"
    if lower.endswith("_bar"):
        return "bar"
    if lower.endswith("_kpa"):
        return "kPa"
    if lower.endswith("_ft"):
        return "ft"
    if lower.endswith("_m"):
        return "m"
    return None


def _normalize_header_token(header_str: str) -> str:
    """Strip parenthetical units, punctuation, and extra whitespace for alias matching."""
    cleaned = UNIT_EXTRACT_PATTERN.sub("", str(header_str))
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", cleaned.strip().lower()).strip("_")
    return cleaned


def resolve_ancillary_columns(
    headers: List[str],
    data_type: str,
    unit_row: Optional[Dict[str, str]] = None,
) -> Tuple[List[AncillaryColumnMapping], Dict[str, Optional[str]]]:
    """
    Three-Stage Column Resolution evaluated globally across all headers by match quality:
      1. Exact Match against canonical schema keys across all headers
      2. Exact Alias Lookup in alias-priority order (e.g. 'well' wins over 'well_id', 'depth_md' wins over 'depth_tvdss')
      3. Substring Alias Lookup & Fuzzy Matching (difflib.SequenceMatcher >= 0.75)
    """
    schema = ANCILLARY_TARGET_SCHEMAS.get(
        data_type.lower(), ANCILLARY_TARGET_SCHEMAS["tops"]
    )
    detected: Dict[str, Optional[str]] = {k.lower(): None for k in schema.keys()}
    used_targets: set[str] = set()
    assigned_by_header: Dict[str, Tuple[str, float, str]] = {}

    clean_headers: List[Tuple[str, str]] = []
    for h in headers:
        h_str = str(h).strip()
        if h_str:
            clean_headers.append((h_str, _normalize_header_token(h_str)))

    # Pass 0 (SCAL Wide-Format Series): Detect numbered Pc.1..Pc.N and Sw.1..Sw.N columns
    if data_type.lower() == "scal":
        for h_str, _ in clean_headers:
            if WIDE_PC_COL_PATTERN.match(h_str.strip()):
                used_targets.add("PC")
                if detected.get("pc") is None:
                    detected["pc"] = h_str
                assigned_by_header[h_str] = ("PC", 1.0, "wide_series")
            elif WIDE_SW_COL_PATTERN.match(h_str.strip()):
                used_targets.add("SW")
                if detected.get("sw") is None:
                    detected["sw"] = h_str
                assigned_by_header[h_str] = ("SW", 1.0, "wide_series")

    # Pass 1 (Stage 1): Exact match against canonical target key
    for h_str, norm in clean_headers:
        if h_str in assigned_by_header:
            continue
        for target_key in schema.keys():
            if target_key not in used_targets and norm == target_key.lower():
                used_targets.add(target_key)
                detected[target_key.lower()] = h_str
                assigned_by_header[h_str] = (target_key, 1.0, "exact")
                break

    # Pass 2 (Stage 2a): Exact alias match in priority order of each target's alias list
    for target_key, aliases in schema.items():
        if target_key in used_targets:
            continue
        norm_aliases = [_normalize_header_token(a) for a in aliases]
        matched_h = None
        for alias_token in norm_aliases:
            for h_str, norm in clean_headers:
                if h_str in assigned_by_header:
                    continue
                if norm == alias_token:
                    matched_h = h_str
                    break
            if matched_h:
                break
        if matched_h:
            used_targets.add(target_key)
            detected[target_key.lower()] = matched_h
            assigned_by_header[matched_h] = (target_key, 0.95, "alias")

    # Pass 3 (Stage 2b): Substring alias match for remaining unmatched headers
    for h_str, norm in clean_headers:
        if h_str in assigned_by_header:
            continue
        for target_key, aliases in schema.items():
            if target_key in used_targets:
                continue
            norm_aliases = [_normalize_header_token(a) for a in aliases]
            if any(len(a) >= 4 and a in norm for a in norm_aliases):
                used_targets.add(target_key)
                detected[target_key.lower()] = h_str
                assigned_by_header[h_str] = (target_key, 0.90, "alias")
                break

    # Pass 4 (Stage 3): Fuzzy Matching (difflib >= 0.75) for remaining unmatched headers
    for h_str, norm in clean_headers:
        if h_str in assigned_by_header:
            continue
        best_score = 0.0
        best_target = None
        for target_key, aliases in schema.items():
            if target_key in used_targets:
                continue
            candidates = [target_key.lower()] + [
                _normalize_header_token(a) for a in aliases
            ]
            for cand in candidates:
                score = difflib.SequenceMatcher(None, norm, cand).ratio()
                if score > best_score:
                    best_score = score
                    best_target = target_key
        if best_score >= 0.75 and best_target:
            used_targets.add(best_target)
            detected[best_target.lower()] = h_str
            assigned_by_header[h_str] = (
                best_target,
                round(float(best_score), 3),
                "fuzzy",
            )

    mappings: List[AncillaryColumnMapping] = []
    for h_str, _ in clean_headers:
        unit = (unit_row or {}).get(h_str) or _extract_unit_from_header(h_str)
        target_col, conf, stg = assigned_by_header.get(h_str, (None, 0.0, "unmatched"))
        mappings.append(
            AncillaryColumnMapping(
                source_column=h_str,
                target_column=target_col,
                source_unit=unit,
                confidence=conf,
                match_stage=stg,
            )
        )

    return mappings, detected


def _is_likely_unit_row(row_values: List[Any]) -> bool:
    """Detect if the row immediately following headers is a unit sub-header row (e.g. '(m)', 'mD', '%')."""
    non_empty = [
        str(v).strip()
        for v in row_values
        if v is not None and str(v).strip() not in ("", "nan", "None")
    ]
    if not non_empty:
        return False
    unit_tokens = {
        "m",
        "ft",
        "md",
        "v/v",
        "%",
        "frac",
        "fraction",
        "g/cc",
        "g/cm3",
        "kg/m3",
        "psi",
        "psia",
        "psig",
        "bar",
        "bara",
        "kpa",
        "mpa",
        "deg",
        "degc",
        "degf",
        "no.",
        "no",
        "(m)",
        "(ft)",
        "(md)",
        "(%)",
        "(g/cm3)",
        "(g/cc)",
        "(psi)",
        "(bar)",
    }
    matches = 0
    numeric_count = 0
    for val in non_empty:
        low = val.lower().strip()
        try:
            float(low)
            numeric_count += 1
        except ValueError:
            pass
        if low in unit_tokens or (
            low.startswith("(") and low.endswith(")") and len(low) <= 12
        ):
            matches += 1
    if numeric_count > len(non_empty) * 0.5:
        return False
    return matches >= max(1, int(len(non_empty) * 0.4))


def _detect_header_row_index(raw_rows: List[List[Any]], data_type: str = "rca") -> int:
    """
    Scan first 15 rows of a spreadsheet/CSV to locate the real header row
    (skipping title blocks like 'Well 15/9-19 A' in Excel workbooks).
    """
    schema = ANCILLARY_TARGET_SCHEMAS.get(
        data_type.lower(), ANCILLARY_TARGET_SCHEMAS["rca"]
    )
    all_keywords = set()
    for k, aliases in schema.items():
        all_keywords.add(k.lower())
        for a in aliases:
            all_keywords.add(_normalize_header_token(a))

    best_idx = 0
    best_score = -1

    for idx, row in enumerate(raw_rows[:15]):
        non_empty = [
            str(c).strip()
            for c in row
            if c is not None and str(c).strip() not in ("", "nan", "None")
        ]
        if len(non_empty) < 2:
            continue
        # Count string cells vs pure numbers
        str_cells = 0
        kw_hits = 0
        for c in non_empty:
            try:
                float(c)
            except ValueError:
                str_cells += 1
                norm = _normalize_header_token(c)
                if norm in all_keywords or any(
                    kw in norm for kw in all_keywords if len(kw) >= 3
                ):
                    kw_hits += 1
        score = kw_hits * 3 + str_cells
        if score > best_score and str_cells >= max(2, len(non_empty) // 2):
            best_score = score
            best_idx = idx

    return best_idx


def parse_tabular_ancillary_file(
    file_bytes: bytes,
    filename: str,
    data_type: str = "tops",
    sheet_name: Optional[str] = None,
    header_row: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], Dict[str, str], List[str], int]:
    """
    Parse .csv, .tsv, .txt, .xls, or .xlsx into a cleaned DataFrame, handling:
      - Multi-sheet Excel workbooks
      - Automatic or explicit header_row index (skipping title blocks)
      - Unit sub-header row immediately below header row (e.g. Volve 15-9-19-A_RCA.xls)
    Returns:
      (df, headers, unit_row_map, sheet_names, detected_header_row)
    """
    ext = os.path.splitext(filename or "")[1].lower()
    sheet_names: List[str] = []

    if ext in (".xls", ".xlsx", ".xlsm"):
        excel_file = pd.ExcelFile(io.BytesIO(file_bytes))
        sheet_names = list(excel_file.sheet_names)
        active_sheet = (
            sheet_name if (sheet_name and sheet_name in sheet_names) else sheet_names[0]
        )
        raw_df = pd.read_excel(excel_file, sheet_name=active_sheet, header=None)
        raw_rows = raw_df.values.tolist()
    else:
        text = file_bytes.decode("utf-8", errors="ignore")
        # Detect delimiter
        sample_lines = [line for line in text.splitlines()[:10] if line.strip()]
        delimiter = ","
        if ext == ".tsv" or (
            sample_lines and "\t" in sample_lines[0] and "," not in sample_lines[0]
        ):
            delimiter = "\t"
        elif sample_lines and ";" in sample_lines[0] and "," not in sample_lines[0]:
            delimiter = ";"
        reader = csv.reader(io.StringIO(text), delimiter=delimiter)
        raw_rows = [r for r in reader if any(str(c).strip() for c in r)]

    if not raw_rows:
        return pd.DataFrame(), [], {}, sheet_names, 0

    detected_hdr = (
        header_row
        if header_row is not None
        else _detect_header_row_index(raw_rows, data_type=data_type)
    )
    detected_hdr = max(0, min(detected_hdr, len(raw_rows) - 1))

    raw_headers = raw_rows[detected_hdr]
    headers: List[str] = []
    seen_counts: Dict[str, int] = {}
    for idx, h in enumerate(raw_headers):
        h_clean = (
            re.sub(r"\s+", " ", str(h).strip())
            if (h is not None and str(h).strip() not in ("", "nan", "None"))
            else f"col_{idx}"
        )
        if h_clean in seen_counts:
            seen_counts[h_clean] += 1
            h_clean = f"{h_clean}_{seen_counts[h_clean]}"
        else:
            seen_counts[h_clean] = 0
        headers.append(h_clean)

    data_start_idx = detected_hdr + 1
    unit_row_map: Dict[str, str] = {}
    if data_start_idx < len(raw_rows) and _is_likely_unit_row(raw_rows[data_start_idx]):
        unit_vals = raw_rows[data_start_idx]
        for idx, h in enumerate(headers):
            if idx < len(unit_vals) and unit_vals[idx] is not None:
                u_str = str(unit_vals[idx]).strip().strip("()[]{}")
                if u_str and u_str.lower() not in ("nan", "none"):
                    unit_row_map[h] = u_str
        data_start_idx += 1

    data_rows = []
    for r in raw_rows[data_start_idx:]:
        if not any(
            c is not None and str(c).strip() not in ("", "nan", "None") for c in r
        ):
            continue
        row_dict = {
            headers[i]: (r[i] if i < len(r) and pd.notna(r[i]) else "")
            for i in range(len(headers))
        }
        data_rows.append(row_dict)

    df = pd.DataFrame(data_rows, columns=headers)
    return df, headers, unit_row_map, sheet_names, detected_hdr


# ==============================================================================
# Phase 2.2: Minimum Curvature Trajectory Calculation (TVD, TVDSS, DX, DY, DLS)
# ==============================================================================


def compute_survey_minimum_curvature(
    surveys: List[Dict[str, float]],
    kb_elevation: float = 0.0,
) -> pd.DataFrame:
    """
    Compute full well trajectory via Minimum Curvature from (md, inc, azim) points:
      - TVD (True Vertical Depth)
      - TVDSS = TVD - kb_elevation
      - DX (East/West offset, m)
      - DY (North/South offset, m)
      - DLS (Dogleg Severity, deg/30m)
    """
    if not surveys:
        return pd.DataFrame(
            columns=["md", "inc", "azim", "tvd", "tvdss", "dx", "dy", "dls"]
        )

    df = (
        pd.DataFrame(surveys)
        .sort_values("md")
        .drop_duplicates(subset=["md"], keep="last")
        .reset_index(drop=True)
    )
    mds = df["md"].to_numpy(dtype=np.float64)
    incs = df["inc"].to_numpy(dtype=np.float64)
    azims = df["azim"].to_numpy(dtype=np.float64)

    # Ensure surface tie-in at MD=0 if first survey point > 0
    inserted_zero = False
    if len(mds) == 1 or mds[0] > 0.0:
        mds_calc = np.insert(mds, 0, 0.0)
        incs_calc = np.insert(incs, 0, 0.0)
        azims_calc = np.insert(azims, 0, azims[0] if len(azims) > 0 else 0.0)
        inserted_zero = True
    else:
        mds_calc, incs_calc, azims_calc = mds, incs, azims

    try:
        import wellpathpy as wpp

        dev = wpp.deviation(mds_calc, incs_calc, azims_calc)
        pos = dev.minimum_curvature()
        tvd_calc = np.asarray(pos.depth, dtype=np.float64)
        dy_calc = np.asarray(pos.northing, dtype=np.float64)
        dx_calc = np.asarray(pos.easting, dtype=np.float64)
    except Exception:
        # Pure-numpy Minimum Curvature fallback
        inc_rad = np.deg2rad(incs_calc)
        azi_rad = np.deg2rad(azims_calc)
        dmd = np.diff(mds_calc)
        i1, i2 = inc_rad[:-1], inc_rad[1:]
        a1, a2 = azi_rad[:-1], azi_rad[1:]
        cos_dl = np.clip(
            np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1)),
            -1.0,
            1.0,
        )
        dl = np.arccos(cos_dl)
        rf = np.where(dl > 1e-6, (2.0 / dl) * np.tan(dl / 2.0), 1.0)
        dz = 0.5 * dmd * (np.cos(i1) + np.cos(i2)) * rf
        dn = 0.5 * dmd * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
        de = 0.5 * dmd * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
        tvd_calc = np.concatenate([[0.0], np.cumsum(dz)])
        dy_calc = np.concatenate([[0.0], np.cumsum(dn)])
        dx_calc = np.concatenate([[0.0], np.cumsum(de)])

    # Compute Dogleg Severity (deg / 30m)
    inc_rad = np.deg2rad(incs_calc)
    azi_rad = np.deg2rad(azims_calc)
    dmd = np.diff(mds_calc)
    i1, i2 = inc_rad[:-1], inc_rad[1:]
    a1, a2 = azi_rad[:-1], azi_rad[1:]
    cos_dl = np.clip(
        np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1)), -1.0, 1.0
    )
    dl_deg = np.rad2deg(np.arccos(cos_dl))
    with np.errstate(divide="ignore", invalid="ignore"):
        dls_seg = np.where(dmd > 0, dl_deg * (30.0 / dmd), 0.0)
    dls_calc = np.concatenate([[0.0], dls_seg])

    if inserted_zero:
        tvd_calc = tvd_calc[1:]
        dy_calc = dy_calc[1:]
        dx_calc = dx_calc[1:]
        dls_calc = dls_calc[1:]

    df["tvd"] = tvd_calc
    df["tvdss"] = tvd_calc - float(kb_elevation or 0.0)
    df["dx"] = dx_calc
    df["dy"] = dy_calc
    df["dls"] = dls_calc
    return df


def _apply_survey_minimum_curvature_to_well(
    well: Any,
    surveys: List[Dict[str, float]],
    df_elevation_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Interpolate computed TVD and TVDSS (= TVD - DF_elevation) onto the well's log depth grid and store as curves."""
    from quick_pp.las_handler import extract_well_df_elevation

    w_uom = getattr(well, "depth_uom", None) or "m"
    if df_elevation_override is not None:
        df_elev = float(df_elevation_override)
    else:
        raw_hdr = getattr(well, "header_data", None)
        raw_cfg = getattr(getattr(well, "_orm_well", None), "config_data", None)
        df_elev = float(
            extract_well_df_elevation(raw_hdr, raw_cfg, target_depth_uom=w_uom) or 0.0
        )

    traj_df = compute_survey_minimum_curvature(surveys, kb_elevation=df_elev)
    if traj_df.empty:
        return {"tvd_points": 0, "tvdss_points": 0, "df_elev": df_elev}

    try:
        curve_df = well.get_curve_data(["WELL_NAME"])
        well_depth = (
            curve_df["depth"].to_numpy(dtype=np.float64)
            if "depth" in curve_df.columns and not curve_df.empty
            else np.array([])
        )
    except Exception:
        well_depth = np.array([])

    mds = traj_df["md"].to_numpy(dtype=np.float64)
    tvds = traj_df["tvd"].to_numpy(dtype=np.float64)

    # Prepend 0,0 for interpolation from surface
    if mds[0] > 0:
        mds_interp = np.insert(mds, 0, 0.0)
        tvds_interp = np.insert(tvds, 0, 0.0)
    else:
        mds_interp, tvds_interp = mds, tvds

    if len(well_depth) > 0:
        valid_mask = (well_depth >= mds_interp.min()) & (well_depth <= mds_interp.max())
        target_depths = well_depth[valid_mask]
        if len(target_depths) > 0:
            interp_tvd = np.interp(target_depths, mds_interp, tvds_interp)
            interp_tvdss = interp_tvd - df_elev
            tvd_dict = dict(
                zip(target_depths.tolist(), interp_tvd.tolist(), strict=False)
            )
            tvdss_dict = dict(
                zip(target_depths.tolist(), interp_tvdss.tolist(), strict=False)
            )
            well.add_curve_data("TVD", tvd_dict, unit=w_uom)
            well.add_curve_data("TVDSS", tvdss_dict, unit=w_uom)
            return {
                "tvd_points": len(tvd_dict),
                "tvdss_points": len(tvdss_dict),
                "df_elev": df_elev,
            }

    # Fallback: save both TVD and TVDSS at survey MD stations if well has no log depth grid yet
    tvd_dict = dict(zip(mds.tolist(), tvds.tolist(), strict=False))
    tvdss_dict = dict(zip(mds.tolist(), (tvds - df_elev).tolist(), strict=False))
    well.add_curve_data("TVD", tvd_dict, unit=w_uom)
    well.add_curve_data("TVDSS", tvdss_dict, unit=w_uom)
    return {
        "tvd_points": len(tvd_dict),
        "tvdss_points": len(tvdss_dict),
        "df_elev": df_elev,
    }


# ==============================================================================
# Phase 2.3: Discrete Core Data (RCA / SCAL) Tolerance Alignment & Depth Shift
# ==============================================================================


def align_and_shift_core_depths(
    core_df: pd.DataFrame,
    well_log_depths: Optional[np.ndarray] = None,
    depth_col: str = "depth",
    shift_table: Optional[List[Tuple[float, float]]] = None,
    tolerance_m: float = 0.08,
) -> pd.DataFrame:
    """
    Align discrete core plug depths to the well's continuous log depth grid:
      1. Preserves original unshifted plug depth in `CORE_DEPTH_RAW`.
      2. Applies piecewise linear depth shift table `[(core_d1, log_d1), (core_d2, log_d2)]` if provided.
      3. Uses `pd.merge_asof(..., direction='nearest', tolerance=tolerance_m)` to snap core samples
         to the nearest continuous log depth index without interpolating across gaps.
    """
    if core_df.empty or depth_col not in core_df.columns:
        return core_df

    out = core_df.copy()
    out[depth_col] = pd.to_numeric(out[depth_col], errors="coerce")
    out = out.dropna(subset=[depth_col]).sort_values(depth_col).reset_index(drop=True)
    out["CORE_DEPTH_RAW"] = out[depth_col].astype(np.float64)

    # Apply piecewise linear core-to-log depth shift if provided
    if shift_table and len(shift_table) >= 1:
        pts = sorted(shift_table, key=lambda pair: float(pair[0]))
        core_pts = np.array([float(p[0]) for p in pts], dtype=np.float64)
        log_pts = np.array([float(p[1]) for p in pts], dtype=np.float64)
        shifts = log_pts - core_pts
        interpolated_shifts = np.interp(
            out["CORE_DEPTH_RAW"].to_numpy(dtype=np.float64), core_pts, shifts
        )
        out[depth_col] = out["CORE_DEPTH_RAW"] + interpolated_shifts

    # Snap to nearest well log depth index within tolerance_m using pd.merge_asof
    if (
        well_log_depths is not None
        and len(well_log_depths) > 0
        and tolerance_m
        and tolerance_m > 0
    ):
        log_grid = pd.DataFrame(
            {"_LOG_DEPTH": np.sort(np.asarray(well_log_depths, dtype=np.float64))}
        )
        merged = pd.merge_asof(
            out.sort_values(depth_col),
            log_grid,
            left_on=depth_col,
            right_on="_LOG_DEPTH",
            direction="nearest",
            tolerance=float(tolerance_m),
        )
        merged["SNAPPED_LOG_DEPTH"] = merged["_LOG_DEPTH"]
        merged[depth_col] = merged["_LOG_DEPTH"].combine_first(merged[depth_col])
        out = merged.drop(columns=["_LOG_DEPTH"])

    return out


# ==============================================================================
# Existing Ancillary Endpoints (Preserved + Upgraded)
# ==============================================================================


@router.get(
    "/formation_tops",
    summary="List formation tops for a project (optional well_name)",
    tags=["Formation Tops"],
)
@router.get(
    "/wells/{well_name}/formation_tops",
    include_in_schema=False,
)
def list_formation_tops_project(project_id: int, well_name: Optional[str] = None):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if well_name:
                well = proj.get_well(well_name)
                tops_df = well.get_formation_tops()
                return {"tops": tops_df.to_dict(orient="records")}

            # aggregate across all wells
            all_rows = []
            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    df = well.get_formation_tops()
                    if not df.empty:
                        for r in df.to_dict(orient="records"):
                            r["well_name"] = wn
                            all_rows.append(r)
                except Exception:
                    continue
            return {"tops": all_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list formation tops: {e}"
        ) from e


@router.post(
    "/formation_tops",
    summary="Add or update formation tops for a project (requires well_name)",
    tags=["Formation Tops"],
)
@router.post(
    "/wells/{well_name}/formation_tops",
    include_in_schema=False,
)
def add_formation_tops_project(
    project_id: int,
    well_name: Optional[str] = None,
    payload: Dict[str, Any] = Body(...),
):
    if not isinstance(payload, dict) or "tops" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'tops'."
        )

    tops = payload.get("tops") or []
    replace_mode = bool(payload.get("replace", False))
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )

    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_formation_tops(tops, replace=replace_mode)
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {
            "created": [{"name": t.get("name"), "depth": t.get("depth")} for t in tops]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add formation tops: {e}"
        ) from e


@router.delete(
    "/formation_tops/{top_name}",
    summary="Delete a formation top by name (requires well_name)",
    tags=["Formation Tops"],
)
@router.delete(
    "/wells/{well_name}/formation_tops/{top_name}",
    include_in_schema=False,
)
def delete_formation_top_project(
    project_id: int, top_name: str, well_name: Optional[str] = None
):
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to delete a top",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            orm_top = session.scalar(
                select(db_objects.ORMFormationTop).filter_by(
                    well_id=well.well_id, name=top_name
                )
            )
            if not orm_top:
                raise ValueError(f"Top '{top_name}' not found")
            session.delete(orm_top)
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {"deleted": top_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete formation top: {e}"
        ) from e


@router.post(
    "/formation_tops/preview",
    summary="Upload CSV and return parsed preview for formation tops (optional well_name)",
    tags=["Formation Tops"],
)
@router.post(
    "/wells/{well_name}/formation_tops/preview",
    include_in_schema=False,
)
def formation_tops_preview_project(
    project_id: int,
    well_name: Optional[str] = None,
    file: UploadFile = File(...),
):
    try:
        raw_bytes = file.file.read()
        df, headers, unit_row_map, sheet_names, hdr_idx = parse_tabular_ancillary_file(
            raw_bytes, file.filename or "tops.csv", data_type="tops"
        )
        if df.empty and not headers:
            return {"preview": [], "headers": []}
        mappings, detected_raw = resolve_ancillary_columns(
            headers, "tops", unit_row=unit_row_map
        )
        detected = {
            "name": detected_raw.get("top_name"),
            "depth": detected_raw.get("md") or detected_raw.get("tvd"),
            "well": detected_raw.get("well_name"),
        }
        return {
            "preview": df.head(50).to_dict(orient="records"),
            "headers": headers,
            "detected": detected,
            "column_mappings": [m.model_dump() for m in mappings],
            "sheet_names": sheet_names,
            "detected_header_row": hdr_idx,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to parse CSV preview: {e}"
        ) from e


@router.get(
    "/fluid_contacts",
    summary="List fluid contacts for a project (optional well_name)",
    tags=["Fluid Contacts"],
)
@router.get(
    "/wells/{well_name}/fluid_contacts",
    include_in_schema=False,
)
def list_fluid_contacts_project(project_id: int, well_name: Optional[str] = None):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if well_name:
                well = proj.get_well(well_name)
                df = well.get_fluid_contacts()
                return {"fluid_contacts": df.to_dict(orient="records")}
            all_rows = []
            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    df = well.get_fluid_contacts()
                    for r in df.to_dict(orient="records"):
                        r["well_name"] = wn
                        all_rows.append(r)
                except Exception:
                    continue
            return {"fluid_contacts": all_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list fluid contacts: {e}"
        ) from e


@router.post(
    "/fluid_contacts",
    summary="Add or update fluid contacts for a project (requires well_name)",
    tags=["Fluid Contacts"],
)
@router.post(
    "/wells/{well_name}/fluid_contacts",
    include_in_schema=False,
)
def add_fluid_contacts_project(
    project_id: int,
    well_name: Optional[str] = None,
    payload: Dict[str, Any] = Body(...),
):
    if not isinstance(payload, dict) or "contacts" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'contacts'."
        )
    contacts = payload.get("contacts") or []
    replace_mode = bool(payload.get("replace", False))
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_fluid_contacts(contacts, replace=replace_mode)
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {"created": contacts}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add fluid contacts: {e}"
        ) from e


@router.delete(
    "/fluid_contacts/{contact_name}",
    summary="Delete a fluid contact by name (requires well_name)",
    tags=["Fluid Contacts"],
)
@router.delete(
    "/wells/{well_name}/fluid_contacts/{contact_name}",
    include_in_schema=False,
)
def delete_fluid_contact_project(
    project_id: int, contact_name: str, well_name: Optional[str] = None
):
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to delete a fluid contact",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            orm_contact = session.scalar(
                select(db_objects.ORMFluidContact).filter_by(
                    well_id=well.well_id, name=contact_name
                )
            )
            if not orm_contact:
                raise ValueError(f"Contact '{contact_name}' not found")
            session.delete(orm_contact)
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {"deleted": contact_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete fluid contact: {e}"
        ) from e


@router.get(
    "/pressure_tests",
    summary="List pressure tests for a project (optional well_name)",
    tags=["Pressure Tests"],
)
@router.get(
    "/wells/{well_name}/pressure_tests",
    include_in_schema=False,
)
def list_pressure_tests_project(project_id: int, well_name: Optional[str] = None):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if well_name:
                well = proj.get_well(well_name)
                df = well.get_pressure_tests()
                return {"pressure_tests": df.to_dict(orient="records")}
            all_rows = []
            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    df = well.get_pressure_tests()
                    for r in df.to_dict(orient="records"):
                        r["well_name"] = wn
                        all_rows.append(r)
                except Exception:
                    continue
            return {"pressure_tests": all_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list pressure tests: {e}"
        ) from e


@router.post(
    "/pressure_tests",
    summary="Add or update pressure tests for a project (requires well_name)",
    tags=["Pressure Tests"],
)
@router.post(
    "/wells/{well_name}/pressure_tests",
    include_in_schema=False,
)
def add_pressure_tests_project(
    project_id: int,
    well_name: Optional[str] = None,
    payload: Dict[str, Any] = Body(...),
):
    if not isinstance(payload, dict) or "tests" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'tests'."
        )
    tests = payload.get("tests") or []
    replace_mode = bool(payload.get("replace", False))
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_pressure_tests(tests, replace=replace_mode)
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {"created": tests}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add pressure tests: {e}"
        ) from e


@router.get(
    "/core_samples",
    summary="List core samples for a project (optional well_name)",
    tags=["Core Samples"],
)
@router.get(
    "/wells/{well_name}/core_samples",
    include_in_schema=False,
)
def list_core_samples_project(project_id: int, well_name: Optional[str] = None):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if well_name:
                well = proj.get_well(well_name)
                core_data = well.get_core_data()
                summaries = [
                    {
                        "sample_name": k,
                        "depth": v["depth"],
                        "description": v.get("description"),
                    }
                    for k, v in core_data.items()
                ]
                return {"core_samples": summaries}
            all_rows = []
            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    core_data = well.get_core_data()
                    for k, v in core_data.items():
                        all_rows.append(
                            {
                                "well_name": wn,
                                "sample_name": k,
                                "depth": v.get("depth"),
                                "description": v.get("description"),
                            }
                        )
                except Exception:
                    continue
            return {"core_samples": all_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list core samples: {e}"
        ) from e


@router.post(
    "/core_samples",
    summary="Add or update core sample(s) with RCA measurements and/or SCAL data (requires well_name)",
    tags=["Core Samples"],
)
@router.post(
    "/wells/{well_name}/core_samples",
    include_in_schema=False,
)
def add_core_sample_project(
    project_id: int,
    well_name: Optional[str] = None,
    payload: Dict[str, Any] = Body(...),
):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )

    # Batch RCA update mode (from InspectorAncillary RCA tab)
    if "samples" in payload and isinstance(payload["samples"], list):
        try:
            with get_db() as session:
                proj = db_objects.Project.load(session, project_id=project_id)
                well = proj.get_well(target_well)
                well.replace_rca_samples(payload["samples"])
            invalidate_well_cache(project_id=project_id, well_name=str(target_well))
            return {
                "samples_updated": len(payload["samples"]),
                "status": "replaced_rca",
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to update RCA core samples: {e}"
            ) from e

    required = ["sample_name", "depth"]
    if not all(k in payload for k in required):
        raise HTTPException(
            status_code=400, detail=f"Payload must include {required} or 'samples'"
        )

    try:
        meas_list = payload.get("measurements")
        poro = payload.get("porosity", payload.get("cpore"))
        perm = payload.get("permeability", payload.get("cperm"))
        if poro is not None or perm is not None:
            meas_list = list(meas_list or [])
            if poro is not None and str(poro).strip() != "":
                meas_list.append(
                    {"property_name": "CPORE", "value": float(poro), "unit": "v/v"}
                )
            if perm is not None and str(perm).strip() != "":
                meas_list.append(
                    {"property_name": "CPERM", "value": float(perm), "unit": "mD"}
                )

        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_core_sample_with_measurements(
                sample_name=payload["sample_name"],
                depth=float(payload["depth"]),
                measurements=meas_list,
                description=payload.get("description"),
                remark=payload.get("remark"),
                relperm_data=payload.get("relperm_data"),
                pc_data=payload.get("pc_data"),
            )
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {
            "sample_name": payload["sample_name"],
            "status": "created_or_updated",
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add core sample: {e}"
        ) from e


@router.get(
    "/scal_pc",
    summary="List SCAL capillary pressure pairs for a project or well",
    tags=["SCAL Capillary Pressure"],
)
@router.get(
    "/wells/{well_name}/scal_pc",
    include_in_schema=False,
)
def list_scal_pc_project(project_id: int, well_name: Optional[str] = None):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if well_name:
                well = proj.get_well(well_name)
                return {
                    "scal_pc": well.get_scal_pc_data(),
                    "well_name": well_name,
                }
            all_rows = []
            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    for row in well.get_scal_pc_data():
                        all_rows.append({"well_name": wn, **row})
                except Exception:
                    continue
            return {"scal_pc": all_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list SCAL capillary pressure data: {e}"
        ) from e


@router.post(
    "/scal_pc",
    summary="Add or replace SCAL capillary pressure (Pc, Sw) pairs for a well",
    tags=["SCAL Capillary Pressure"],
)
@router.post(
    "/wells/{well_name}/scal_pc",
    include_in_schema=False,
)
def add_scal_pc_project(
    project_id: int,
    well_name: Optional[str] = None,
    payload: Dict[str, Any] = Body(...),
):
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Payload must be a JSON object")

    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )

    rows = (
        payload.get("pc_data")
        if "pc_data" in payload
        else payload.get("scal_pc")
        if "scal_pc" in payload
        else payload.get("samples")
    )
    if rows is None or not isinstance(rows, list):
        if "sample_name" in payload and "depth" in payload:
            rows = [payload]
        else:
            raise HTTPException(
                status_code=400,
                detail="Payload must include 'pc_data', 'scal_pc', or 'samples' list",
            )

    append_mode = bool(payload.get("append", False))

    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.replace_scal_pc_data(rows, append=append_mode)
            updated_rows = well.get_scal_pc_data()
        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {
            "status": "success",
            "well_name": str(target_well),
            "count": len(updated_rows),
            "scal_pc": updated_rows,
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save SCAL capillary pressure data: {e}"
        ) from e


@router.get(
    "/core_samples/{sample_name}",
    summary="Get core sample details by name (requires well_name)",
    tags=["Core Samples"],
)
@router.get(
    "/wells/{well_name}/core_samples/{sample_name}",
    include_in_schema=False,
)
def get_core_sample_project(
    project_id: int, sample_name: str, well_name: Optional[str] = None
):
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to get a core sample",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            core_data = well.get_core_data()
            if sample_name not in core_data:
                raise ValueError(f"Sample '{sample_name}' not found")
            sample = core_data[sample_name]
            sample_out = {
                "sample_name": sample_name,
                "depth": sample.get("depth"),
                "description": sample.get("description"),
                "measurements": sample.get("measurements").to_dict(orient="records")
                if hasattr(sample.get("measurements"), "to_dict")
                else [],
                "relperm": sample.get("relperm").to_dict(orient="records")
                if hasattr(sample.get("relperm"), "to_dict")
                else [],
                "pc": sample.get("pc").to_dict(orient="records")
                if hasattr(sample.get("pc"), "to_dict")
                else [],
            }
            return {"core_sample": sample_out}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to get core sample: {e}"
        ) from e


@router.get(
    "/well_surveys",
    summary="List well survey points for a project (optional well_name)",
    tags=["Well Surveys"],
)
def list_well_surveys_project(project_id: int, well_name: Optional[str] = Query(None)):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            if well_name:
                well = proj.get_well(well_name)
                surveys_df = well.get_well_surveys()
                if not surveys_df.empty:
                    surveys_df["well_name"] = well_name
                return {
                    "well_surveys": surveys_df.to_dict(orient="records"),
                    "well_name": well_name,
                }

            # aggregate across all wells
            all_rows = []
            for wn in proj.get_well_names():
                well = proj.get_well(wn)
                surveys_df = well.get_well_surveys()
                if not surveys_df.empty:
                    surveys_df["well_name"] = wn
                    all_rows.extend(surveys_df.to_dict(orient="records"))
            return {"well_surveys": all_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list well surveys: {e}"
        ) from e


@router.post(
    "/well_surveys",
    summary="Add or update well survey points for a project (requires well_name)",
    tags=["Well Surveys"],
)
@router.post(
    "/wells/{well_name}/well_surveys",
    include_in_schema=False,
)
def add_well_surveys_project(
    project_id: int,
    well_name: Optional[str] = None,
    payload: Dict[str, Any] = Body(...),
):
    if not isinstance(payload, dict) or "surveys" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'surveys'."
        )
    surveys = payload.get("surveys")
    target_well = well_name or payload.get("well_name")
    calculate_tvd = payload.get("calculate_tvd", True)

    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_well_surveys(surveys)

            if calculate_tvd and surveys:
                _apply_survey_minimum_curvature_to_well(well, surveys)

        invalidate_well_cache(project_id=project_id, well_name=str(target_well))
        return {
            "created": [
                {
                    "md": s.get("md"),
                    "inc": s.get("inc"),
                    "azim": s.get("azim"),
                }
                for s in surveys
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add well surveys: {e}"
        ) from e


@router.delete(
    "/well_surveys/{md}",
    summary="Delete a well survey point by measured depth (requires well_name)",
    tags=["Well Surveys"],
)
def delete_well_survey_project(
    project_id: int, md: float, well_name: Optional[str] = Query(None)
):
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to delete a survey",
        )
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            orm_survey = session.scalar(
                select(db_objects.ORMWellSurvey).filter_by(well_id=well.well_id, md=md)
            )
            if not orm_survey:
                raise ValueError(f"Survey point at MD {md} not found")
            session.delete(orm_survey)
            return {"deleted": f"MD {md}"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete well survey: {e}"
        ) from e


@router.post(
    "/well_surveys/preview",
    summary="Upload CSV/Excel and return parsed preview for well surveys (optional well_name)",
    tags=["Well Surveys"],
)
def well_surveys_preview_project(file: UploadFile = File(...)):
    """Return a preview (first 50 rows) and detected columns from uploaded CSV/Excel file."""
    try:
        raw_bytes = file.file.read()
        df, headers, unit_row_map, sheet_names, hdr_idx = parse_tabular_ancillary_file(
            raw_bytes, file.filename or "surveys.csv", data_type="surveys"
        )
        if df.empty and not headers:
            return {"preview": [], "headers": []}
        mappings, detected_raw = resolve_ancillary_columns(
            headers, "surveys", unit_row=unit_row_map
        )
        detected = {
            "md": detected_raw.get("md"),
            "inc": detected_raw.get("inc"),
            "azim": detected_raw.get("azim"),
            "well": detected_raw.get("well_name"),
        }
        return {
            "preview": df.head(50).to_dict(orient="records"),
            "headers": headers,
            "detected": detected,
            "column_mappings": [m.model_dump() for m in mappings],
            "sheet_names": sheet_names,
            "detected_header_row": hdr_idx,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to preview file: {e}"
        ) from e


@router.post(
    "/well_surveys/upload",
    summary="Upload and parse well survey data from CSV/Excel (requires well_name)",
    tags=["Well Surveys"],
)
def upload_well_surveys_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, Any] = Body(...),
):
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to upload surveys",
        )

    required = ["file_content", "md_column", "inc_column", "azim_column"]
    if not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")

    try:
        file_content = payload["file_content"]
        if isinstance(file_content, str) and file_content.startswith("data:"):
            file_content = file_content.split(",", 1)[1]

        decoded = base64.b64decode(file_content)
        content_str = decoded.decode(errors="ignore")

        reader = csv.reader(io.StringIO(content_str))
        rows = list(reader)
        if len(rows) < 2:
            raise ValueError("No data rows in file")

        headers = [h.strip() for h in rows[0]]
        md_col = payload["md_column"]
        inc_col = payload["inc_column"]
        azim_col = payload["azim_column"]
        tvd_col = payload.get("tvd_column")
        calculate_tvd = payload.get("calculate_tvd", True)

        if md_col not in headers:
            raise ValueError(f"Column '{md_col}' not found in file")
        if inc_col not in headers:
            raise ValueError(f"Column '{inc_col}' not found in file")
        if azim_col not in headers:
            raise ValueError(f"Column '{azim_col}' not found in file")

        md_idx = headers.index(md_col)
        inc_idx = headers.index(inc_col)
        azim_idx = headers.index(azim_col)
        tvd_idx = headers.index(tvd_col) if tvd_col and tvd_col in headers else None

        surveys = []
        tvd_data = {}

        for r in rows[1:]:
            if len(r) <= max(md_idx, inc_idx, azim_idx):
                continue
            try:
                md = float(r[md_idx].strip())
                inc = float(r[inc_idx].strip())
                azim = float(r[azim_idx].strip())
                surveys.append({"md": md, "inc": inc, "azim": azim})

                if tvd_idx is not None and len(r) > tvd_idx:
                    try:
                        tvd = float(r[tvd_idx].strip())
                        tvd_data[md] = tvd
                    except (ValueError, IndexError):
                        pass
            except (ValueError, IndexError):
                continue

        if not surveys:
            raise ValueError("No valid survey points found in file")

        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_well_surveys(surveys)

            tvd_points_stored = 0
            if calculate_tvd:
                res = _apply_survey_minimum_curvature_to_well(well, surveys)
                tvd_points_stored = res.get("tvd_points", 0)
            elif tvd_data:
                from quick_pp.las_handler import extract_well_df_elevation

                w_uom = getattr(well, "depth_uom", None) or "m"
                df_elev = float(
                    extract_well_df_elevation(
                        getattr(well, "header_data", None),
                        getattr(getattr(well, "_orm_well", None), "config_data", None),
                        target_depth_uom=w_uom,
                    )
                    or 0.0
                )
                well.add_curve_data("TVD", tvd_data, unit=w_uom)
                tvdss_data = {d: v - df_elev for d, v in tvd_data.items()}
                well.add_curve_data("TVDSS", tvdss_data, unit=w_uom)
                tvd_points_stored = len(tvd_data)

            invalidate_well_cache(project_id=project_id, well_name=str(target_well))

            return {
                "status": "success",
                "surveys_added": len(surveys),
                "tvd_calculated": calculate_tvd,
                "tvd_points_stored": tvd_points_stored,
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to upload well surveys: {e}"
        ) from e


@router.post(
    "/well_surveys/calculate_tvd",
    summary="Calculate TVD from existing survey data and store as well curve (optional well_name)",
    tags=["Well Surveys"],
)
def calculate_tvd_project(project_id: int, well_name: Optional[str] = Query(None)):
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            if well_name:
                well = proj.get_well(well_name)
                surveys_df = well.get_well_surveys()

                if surveys_df.empty:
                    raise ValueError(f"No survey data found for well '{well_name}'")

                surveys = surveys_df[["md", "inc", "azim"]].to_dict(orient="records")
                res = _apply_survey_minimum_curvature_to_well(well, surveys)

                return {
                    "success": True,
                    "tvd_points_saved": res.get("tvd_points", 0),
                    "wells_processed": 1,
                    "wells": [well_name],
                }

            total_tvd_points = 0
            wells_processed = []

            for wn in proj.get_well_names():
                well = proj.get_well(wn)
                surveys_df = well.get_well_surveys()

                if surveys_df.empty:
                    continue

                try:
                    surveys = surveys_df[["md", "inc", "azim"]].to_dict(
                        orient="records"
                    )
                    res = _apply_survey_minimum_curvature_to_well(well, surveys)
                    total_tvd_points += res.get("tvd_points", 0)
                    wells_processed.append(wn)
                except Exception:
                    continue

            if not wells_processed:
                raise ValueError("No wells with survey data found in project")

            return {
                "success": True,
                "tvd_points_saved": total_tvd_points,
                "wells_processed": len(wells_processed),
                "wells": wells_processed,
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to calculate TVD: {e}"
        ) from e


# ==============================================================================
# Phase 2 & 3: Unified Interactive Ancillary Preview & Commit Endpoints
# ==============================================================================


@import_ancillary_router.post(
    "/preview",
    response_model=AncillaryPreviewResponse,
    summary="Parse CSV/TSV/Excel ancillary file, detect header/unit rows, and resolve column mappings",
)
@router.post(
    "/ancillary/preview",
    response_model=AncillaryPreviewResponse,
    summary="Parse CSV/TSV/Excel ancillary file, detect header/unit rows, and resolve column mappings",
)
async def preview_ancillary_file(
    file: UploadFile = File(...),
    data_type: str = Form("tops"),
    sheet_name: Optional[str] = Form(None),
    header_row: Optional[int] = Form(None),
    project_id: Optional[int] = None,
):
    try:
        raw_bytes = await file.read()
        df, headers, unit_row_map, sheet_names, detected_hdr = (
            parse_tabular_ancillary_file(
                raw_bytes,
                filename=file.filename or "ancillary.csv",
                data_type=data_type,
                sheet_name=sheet_name,
                header_row=header_row,
            )
        )
        mappings, detected = resolve_ancillary_columns(
            headers, data_type, unit_row=unit_row_map
        )
        preview_rows = (
            df.head(50).fillna("").to_dict(orient="records") if not df.empty else []
        )

        return AncillaryPreviewResponse(
            filename=file.filename or "ancillary.csv",
            data_type=data_type,
            sheet_names=sheet_names,
            active_sheet=sheet_name or (sheet_names[0] if sheet_names else None),
            detected_header_row=detected_hdr,
            detected_unit_row=(detected_hdr + 1) if unit_row_map else None,
            headers=headers,
            column_mappings=mappings,
            detected=detected,
            preview=preview_rows,
            total_rows=len(df),
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to preview ancillary file: {e}"
        ) from e


def _convert_ancillary_numeric(
    val: Any, source_unit: Optional[str], target_col: str
) -> Optional[float]:
    """Convert numeric value based on source unit and target canonical property."""
    if val is None or str(val).strip() in (
        "",
        "nan",
        "None",
        "-999.25",
        "-999",
        "-9999",
    ):
        return None
    try:
        clean_str = (
            str(val).strip().replace(",", ".")
            if isinstance(val, str) and "," in str(val) and "." not in str(val)
            else str(val).strip()
        )
        num = float(clean_str)
    except (ValueError, TypeError):
        return None

    u = (source_unit or "").strip().lower()
    t = target_col.upper()

    if t in ("MD", "TVD", "DEPTH", "DEPTH_TOP", "DEPTH_BASE") and u in ("ft", "feet"):
        return num * 0.3048
    if t in ("CPORE", "SW", "CSAT_WATER", "CSAT_OIL"):
        if u in ("%", "pct", "percent") or (not u and num > 1.0 and num <= 100.0):
            return num / 100.0
    if (
        t == "CGRAIN_DENSITY"
        and u in ("kg/m3", "kg/m^3")
        or (t == "CGRAIN_DENSITY" and num > 500)
    ):
        return num / 1000.0
    if t in ("PRESSURE", "PC"):
        if u in ("bar", "bara", "barg"):
            return num * 14.5038
        if u in ("kpa",):
            return num * 0.145038
        if u in ("mpa",):
            return num * 145.038
    return num


def _canonical_well_token(s: str) -> str:
    """Normalize a well name/UWI for flexible matching (e.g. '30/07a- 8' == '30-7a-8')."""
    if not s:
        return ""
    # Strip leading zeros inside numeric groups (07 -> 7) and remove non-alphanumerics
    dezeroed = re.sub(r"(?<![0-9])0+([1-9])", r"\1", str(s).strip().lower())
    return re.sub(r"[^a-z0-9]", "", dezeroed)


def _match_project_well(proj: Any, raw_well_str: str) -> Optional[str]:
    """Match a well identifier from CSV/Excel against project well names and UWIs."""
    well_names = proj.get_well_names()
    if not raw_well_str or not well_names:
        return None
    raw_clean = str(raw_well_str).strip()
    if raw_clean in ("", "nan", "None"):
        return None
    if raw_clean in well_names:
        return raw_clean

    norm = _canonical_well_token(raw_clean)
    if not norm:
        return None

    # Exact canonical token match on well name
    for wn in well_names:
        if _canonical_well_token(wn) == norm:
            return wn

    # Check ORM wells for UWI match as well
    for orm_w in getattr(proj._orm_project, "wells", []) or []:
        if _canonical_well_token(getattr(orm_w, "uwi", "")) == norm:
            return orm_w.name

    # Substring containment match for longer identifiers
    for wn in well_names:
        wn_norm = _canonical_well_token(wn)
        if (
            len(norm) >= 4
            and len(wn_norm) >= 4
            and (norm in wn_norm or wn_norm in norm)
        ):
            return wn

    return None


@import_ancillary_router.post(
    "/commit",
    summary="Commit mapped ancillary rows (or file) with unit conversions, Minimum Curvature TVD, and core depth alignment",
)
@router.post(
    "/ancillary/commit",
    summary="Commit mapped ancillary rows (or file) with unit conversions, Minimum Curvature TVD, and core depth alignment",
)
def commit_ancillary_data(
    request: AncillaryCommitRequest = Body(...),
    project_id: Optional[int] = None,
):
    effective_project_id = project_id if project_id is not None else request.project_id
    dtype = request.data_type.lower()

    # Build lookup from target_column -> (source_column, source_unit)
    col_map: Dict[str, Tuple[str, Optional[str]]] = {}
    for cm in request.column_mappings:
        if cm.target_column:
            col_map[cm.target_column.upper()] = (cm.source_column, cm.source_unit)

    rows = list(request.rows or [])
    if not rows and request.file_content_base64:
        b64 = request.file_content_base64
        if "," in b64 and b64.startswith("data:"):
            b64 = b64.split(",", 1)[1]
        raw_bytes = base64.b64decode(b64)
        df, headers, unit_row_map, _, _ = parse_tabular_ancillary_file(
            raw_bytes,
            filename=request.filename or "upload.csv",
            data_type=dtype,
            sheet_name=request.sheet_name,
            header_row=request.header_row,
        )
        rows = df.to_dict(orient="records")
        if not col_map:
            auto_mappings, _ = resolve_ancillary_columns(
                headers, dtype, unit_row=unit_row_map
            )
            for cm in auto_mappings:
                if cm.target_column:
                    col_map[cm.target_column.upper()] = (
                        cm.source_column,
                        cm.source_unit,
                    )

    if not rows:
        raise HTTPException(status_code=400, detail="No rows provided to commit.")

    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=effective_project_id)
            project_wells = proj.get_well_names()
            if not project_wells:
                raise ValueError(
                    f"Project '{proj.name}' has no wells yet. Please import well logs (.las/.dlis) first."
                )

            well_symbol_col, _ = col_map.get("WELL_NAME", (None, None))
            # Secondary fallback well columns in case CSV has both 'well_id' and 'well'
            fallback_well_cols = [
                k
                for k in (rows[0].keys() if rows else [])
                if k != well_symbol_col
                and _normalize_header_token(k)
                in ("well", "well_name", "wellname", "uwi", "well_id", "wellbore")
            ]

            explicit_target_well = (
                (_match_project_well(proj, request.well_name) or request.well_name)
                if request.well_name
                else None
            )

            grouped_rows: Dict[str, List[Dict[str, Any]]] = {}
            unmatched_csv_wells: set[str] = set()

            for r in rows:
                matched_w: Optional[str] = None
                raw_candidates: List[str] = []
                if well_symbol_col and r.get(well_symbol_col) is not None:
                    raw_candidates.append(str(r.get(well_symbol_col)).strip())
                for fc in fallback_well_cols:
                    if r.get(fc) is not None:
                        raw_candidates.append(str(r.get(fc)).strip())

                for cand in raw_candidates:
                    if cand and cand not in ("", "nan", "None"):
                        m = _match_project_well(proj, cand)
                        if m:
                            matched_w = m
                            break
                        unmatched_csv_wells.add(cand)

                # If user explicitly chose a Target Well in the UI:
                if explicit_target_well:
                    if matched_w == explicit_target_well:
                        grouped_rows.setdefault(explicit_target_well, []).append(r)
                    elif not raw_candidates:
                        grouped_rows.setdefault(explicit_target_well, []).append(r)
                elif matched_w:
                    grouped_rows.setdefault(matched_w, []).append(r)

            # If user selected an explicit Target Well and no rows matched by name in a single-well or custom file,
            # assign all rows to the user's selected Target Well
            if not grouped_rows and explicit_target_well:
                grouped_rows[explicit_target_well] = rows
            # If project has only 1 well and CSV has only 1 unique well identifier, auto-match to that well
            elif (
                not grouped_rows
                and len(project_wells) == 1
                and len(unmatched_csv_wells) <= 1
            ):
                grouped_rows[project_wells[0]] = rows

            if not grouped_rows:
                sample_csv_wells = sorted(unmatched_csv_wells)[:8]
                raise ValueError(
                    f"Could not match CSV well(s) {sample_csv_wells} to project wells {project_wells}. "
                    f"Select a 'Target Well' from the dropdown above to import into a specific well, or import the matching well logs first."
                )

            summary_by_well: Dict[str, int] = {}

            for wn, w_rows in grouped_rows.items():
                well = proj.get_well(wn)

                if dtype == "tops":
                    name_col, _ = col_map.get("TOP_NAME", ("name", None))
                    md_col, md_unit = col_map.get(
                        "MD", col_map.get("TVD", ("depth", None))
                    )
                    tops_payload = []
                    for r in w_rows:
                        t_name = str(r.get(name_col, "")).strip()
                        d_val = _convert_ancillary_numeric(r.get(md_col), md_unit, "MD")
                        if t_name and d_val is not None:
                            tops_payload.append({"name": t_name, "depth": d_val})
                    if tops_payload:
                        well.add_formation_tops(tops_payload, replace=True)
                        summary_by_well[wn] = len(tops_payload)

                elif dtype == "surveys":
                    md_col, md_unit = col_map.get("MD", ("md", None))
                    inc_col, _ = col_map.get("INC", ("inc", None))
                    azi_col, _ = col_map.get("AZIM", ("azim", None))
                    surveys_payload = []
                    for r in w_rows:
                        md_v = _convert_ancillary_numeric(r.get(md_col), md_unit, "MD")
                        inc_v = _convert_ancillary_numeric(r.get(inc_col), None, "INC")
                        azi_v = _convert_ancillary_numeric(r.get(azi_col), None, "AZIM")
                        if md_v is not None and inc_v is not None and azi_v is not None:
                            surveys_payload.append(
                                {"md": md_v, "inc": inc_v, "azim": azi_v}
                            )
                    if surveys_payload:
                        well.add_well_surveys(surveys_payload)
                        if request.calculate_tvd:
                            _apply_survey_minimum_curvature_to_well(
                                well,
                                surveys_payload,
                                df_elevation_override=request.kb_elev_override,
                            )
                        summary_by_well[wn] = len(surveys_payload)

                elif dtype == "pressures":
                    d_col, d_unit = col_map.get("DEPTH", ("depth", None))
                    p_col, p_unit = col_map.get("PRESSURE", ("pressure", None))
                    rem_col, _ = col_map.get("REMARK", (None, None))
                    tests_payload = []
                    for r in w_rows:
                        d_v = _convert_ancillary_numeric(r.get(d_col), d_unit, "DEPTH")
                        p_v = _convert_ancillary_numeric(
                            r.get(p_col), p_unit, "PRESSURE"
                        )
                        if d_v is not None and p_v is not None:
                            tests_payload.append(
                                {
                                    "depth": d_v,
                                    "pressure": p_v,
                                    "pressure_uom": "psi",
                                }
                            )
                    if tests_payload:
                        well.add_pressure_tests(tests_payload, replace=True)
                        summary_by_well[wn] = len(tests_payload)

                elif dtype == "contacts":
                    c_col, _ = col_map.get("CONTACT_TYPE", ("contact_type", None))
                    d_col, d_unit = col_map.get("DEPTH", ("depth", None))
                    contacts_payload = []
                    for r in w_rows:
                        c_type = str(r.get(c_col, "")).strip().upper()
                        d_v = _convert_ancillary_numeric(r.get(d_col), d_unit, "DEPTH")
                        if c_type and d_v is not None:
                            contacts_payload.append({"name": c_type, "depth": d_v})
                    if contacts_payload:
                        well.add_fluid_contacts(contacts_payload, replace=True)
                        summary_by_well[wn] = len(contacts_payload)

                elif dtype == "rca":
                    s_col, _ = col_map.get("SAMPLE_NAME", (None, None))
                    d_col, d_unit = col_map.get("DEPTH", ("depth", None))
                    por_col, por_unit = col_map.get("CPORE", (None, None))
                    perm_col, perm_unit = col_map.get("CPERM", (None, None))
                    permv_col, permv_unit = col_map.get("CPERMV", (None, None))
                    gd_col, gd_unit = col_map.get("CGRAIN_DENSITY", (None, None))
                    sw_col, sw_unit = col_map.get("CSAT_WATER", (None, None))
                    so_col, so_unit = col_map.get("CSAT_OIL", (None, None))
                    desc_col, _ = col_map.get("DESCRIPTION", (None, None))
                    rem_col, _ = col_map.get("REMARK", (None, None))

                    raw_samples = []
                    for idx, r in enumerate(w_rows):
                        d_v = _convert_ancillary_numeric(r.get(d_col), d_unit, "DEPTH")
                        if d_v is None:
                            continue
                        s_name = str(r.get(s_col, "")).strip() if s_col else ""
                        if not s_name or s_name.lower() in ("nan", "none"):
                            s_name = f"S-{idx + 1}"
                        raw_samples.append(
                            {
                                "sample_name": s_name,
                                "depth": d_v,
                                "porosity": _convert_ancillary_numeric(
                                    r.get(por_col), por_unit, "CPORE"
                                )
                                if por_col
                                else None,
                                "permeability": _convert_ancillary_numeric(
                                    r.get(perm_col), perm_unit, "CPERM"
                                )
                                if perm_col
                                else None,
                                "permeability_vertical": _convert_ancillary_numeric(
                                    r.get(permv_col), permv_unit, "CPERMV"
                                )
                                if permv_col
                                else None,
                                "grain_density": _convert_ancillary_numeric(
                                    r.get(gd_col), gd_unit, "CGRAIN_DENSITY"
                                )
                                if gd_col
                                else None,
                                "sw": _convert_ancillary_numeric(
                                    r.get(sw_col), sw_unit, "CSAT_WATER"
                                )
                                if sw_col
                                else None,
                                "so": _convert_ancillary_numeric(
                                    r.get(so_col), so_unit, "CSAT_OIL"
                                )
                                if so_col
                                else None,
                                "description": str(r.get(desc_col, "")).strip()
                                if desc_col and r.get(desc_col) is not None
                                else None,
                                "remark": str(r.get(rem_col, "")).strip()
                                if rem_col and r.get(rem_col) is not None
                                else None,
                            }
                        )

                    if raw_samples:
                        core_df = pd.DataFrame(raw_samples)
                        try:
                            log_df = well.get_curve_data(["WELL_NAME"])
                            log_depths = (
                                log_df["depth"].to_numpy(dtype=np.float64)
                                if "depth" in log_df.columns and not log_df.empty
                                else None
                            )
                        except Exception:
                            log_depths = None

                        aligned_df = align_and_shift_core_depths(
                            core_df,
                            well_log_depths=log_depths,
                            depth_col="depth",
                            shift_table=request.core_depth_shifts,
                            tolerance_m=request.core_snap_tolerance_m,
                        )
                        samples_to_save = []
                        for _, s_row in aligned_df.iterrows():
                            meas = []
                            if pd.notna(s_row.get("porosity")):
                                meas.append(
                                    {
                                        "property_name": "CPORE",
                                        "value": float(s_row["porosity"]),
                                        "unit": "v/v",
                                    }
                                )
                            if pd.notna(s_row.get("permeability")):
                                meas.append(
                                    {
                                        "property_name": "CPERM",
                                        "value": float(s_row["permeability"]),
                                        "unit": "mD",
                                    }
                                )
                            if pd.notna(s_row.get("permeability_vertical")):
                                meas.append(
                                    {
                                        "property_name": "CPERMV",
                                        "value": float(s_row["permeability_vertical"]),
                                        "unit": "mD",
                                    }
                                )
                            if pd.notna(s_row.get("grain_density")):
                                meas.append(
                                    {
                                        "property_name": "CGRAIN_DENSITY",
                                        "value": float(s_row["grain_density"]),
                                        "unit": "g/cc",
                                    }
                                )
                            if pd.notna(s_row.get("sw")):
                                meas.append(
                                    {
                                        "property_name": "CSAT_WATER",
                                        "value": float(s_row["sw"]),
                                        "unit": "v/v",
                                    }
                                )
                            if pd.notna(s_row.get("so")):
                                meas.append(
                                    {
                                        "property_name": "CSAT_OIL",
                                        "value": float(s_row["so"]),
                                        "unit": "v/v",
                                    }
                                )
                            if pd.notna(s_row.get("CORE_DEPTH_RAW")):
                                meas.append(
                                    {
                                        "property_name": "CORE_DEPTH_RAW",
                                        "value": float(s_row["CORE_DEPTH_RAW"]),
                                        "unit": "m",
                                    }
                                )
                            samples_to_save.append(
                                {
                                    "sample_name": str(s_row["sample_name"]),
                                    "depth": float(s_row["depth"]),
                                    "porosity": float(s_row["porosity"])
                                    if pd.notna(s_row.get("porosity"))
                                    else None,
                                    "permeability": float(s_row["permeability"])
                                    if pd.notna(s_row.get("permeability"))
                                    else None,
                                    "description": s_row.get("description")
                                    if pd.notna(s_row.get("description"))
                                    else None,
                                    "remark": s_row.get("remark")
                                    if pd.notna(s_row.get("remark"))
                                    else None,
                                    "measurements": meas,
                                }
                            )
                        well.replace_rca_samples(samples_to_save)
                        summary_by_well[wn] = len(samples_to_save)

                elif dtype == "scal":
                    s_col, _ = col_map.get("SAMPLE_NAME", (None, None))
                    d_col, d_unit = col_map.get("DEPTH", ("depth", None))
                    por_col, por_unit = col_map.get("CPORE", (None, None))
                    perm_col, perm_unit = col_map.get("CPERM", (None, None))
                    exp_col, _ = col_map.get("EXPERIMENT_TYPE", (None, None))
                    cyc_col, _ = col_map.get("CYCLE", (None, None))
                    desc_col, _ = col_map.get("DESCRIPTION", (None, None))

                    # Collect all columns mapped to PC and SW (supports both single-column long format and multi-column wide format)
                    pc_mapped = [
                        (cm.source_column, cm.source_unit)
                        for cm in request.column_mappings
                        if (cm.target_column or "").upper() == "PC"
                    ]
                    sw_mapped = [
                        (cm.source_column, cm.source_unit)
                        for cm in request.column_mappings
                        if (cm.target_column or "").upper() == "SW"
                    ]

                    # Also detect wide series columns (e.g. Pc.1..Pc.20, Sw.1..Sw.20) from row headers if not explicitly mapped
                    sample_keys = list(w_rows[0].keys()) if w_rows else []
                    wide_pc_by_idx: Dict[int, Tuple[str, Optional[str]]] = {}
                    wide_sw_by_idx: Dict[int, Tuple[str, Optional[str]]] = {}
                    unit_lookup = {
                        cm.source_column: cm.source_unit
                        for cm in request.column_mappings
                    }
                    for k in sample_keys:
                        m_pc = WIDE_PC_COL_PATTERN.match(str(k).strip())
                        if m_pc:
                            wide_pc_by_idx[int(m_pc.group(1))] = (
                                k,
                                unit_lookup.get(k) or _extract_unit_from_header(k),
                            )
                        m_sw = WIDE_SW_COL_PATTERN.match(str(k).strip())
                        if m_sw:
                            wide_sw_by_idx[int(m_sw.group(1))] = (
                                k,
                                unit_lookup.get(k) or _extract_unit_from_header(k),
                            )

                    # Build ordered list of ((pc_col, pc_unit), (sw_col, sw_unit)) pairs
                    pc_sw_pairs: List[
                        Tuple[Tuple[str, Optional[str]], Tuple[str, Optional[str]]]
                    ] = []
                    if wide_pc_by_idx and wide_sw_by_idx:
                        for series_idx in sorted(
                            set(wide_pc_by_idx.keys()) & set(wide_sw_by_idx.keys())
                        ):
                            pc_sw_pairs.append(
                                (wide_pc_by_idx[series_idx], wide_sw_by_idx[series_idx])
                            )
                    elif pc_mapped and sw_mapped:
                        for pm, sm in zip(pc_mapped, sw_mapped, strict=False):
                            pc_sw_pairs.append((pm, sm))
                    else:
                        pc_col, pc_unit = col_map.get("PC", ("pc", None))
                        sw_col, sw_unit = col_map.get("SW", ("sw", None))
                        pc_sw_pairs.append(((pc_col, pc_unit), (sw_col, sw_unit)))

                    pc_rows = []
                    for idx, r in enumerate(w_rows):
                        d_v = (
                            _convert_ancillary_numeric(r.get(d_col), d_unit, "DEPTH")
                            if d_col
                            else 0.0
                        )
                        s_name = (
                            str(r.get(s_col, "")).strip()
                            if s_col and r.get(s_col) is not None
                            else ""
                        )
                        if not s_name or s_name.lower() in ("nan", "none"):
                            s_name = f"SCAL-{idx + 1}"
                        por_v = (
                            _convert_ancillary_numeric(r.get(por_col), por_unit, "CPORE")
                            if por_col
                            else None
                        )
                        perm_v = (
                            _convert_ancillary_numeric(r.get(perm_col), perm_unit, "CPERM")
                            if perm_col
                            else None
                        )
                        exp_val = (
                            str(r.get(exp_col, "micp")).strip()
                            if exp_col and r.get(exp_col) is not None
                            else "micp"
                        ) or "micp"
                        cyc_val = (
                            str(r.get(cyc_col, "drainage")).strip()
                            if cyc_col and r.get(cyc_col) is not None
                            else "drainage"
                        ) or "drainage"
                        desc_val = (
                            str(r.get(desc_col, "")).strip()
                            if desc_col and r.get(desc_col) is not None
                            else None
                        )

                        for (p_src, p_u), (s_src, s_u) in pc_sw_pairs:
                            pc_v = _convert_ancillary_numeric(r.get(p_src), p_u, "PC")
                            sw_v = _convert_ancillary_numeric(r.get(s_src), s_u, "SW")
                            if pc_v is None or sw_v is None:
                                continue
                            pc_rows.append(
                                {
                                    "sample_name": s_name,
                                    "depth": d_v if d_v is not None else 0.0,
                                    "porosity": por_v,
                                    "permeability": perm_v,
                                    "pc": pc_v,
                                    "sw": sw_v,
                                    "experiment_type": exp_val,
                                    "cycle": cyc_val,
                                    "description": desc_val,
                                }
                            )
                    if pc_rows:
                        well.replace_scal_pc_data(pc_rows, append=False)
                        summary_by_well[wn] = len(pc_rows)

                invalidate_well_cache(
                    project_id=effective_project_id, well_name=str(wn)
                )

            return {
                "status": "success",
                "data_type": dtype,
                "wells_updated": summary_by_well,
                "total_committed": sum(summary_by_well.values()),
            }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to commit ancillary data: {e}"
        ) from e
