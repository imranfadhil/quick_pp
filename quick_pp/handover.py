"""Petrophysical Handover Package Builder for Techlog, Geolog, IP, and Petrel.

Generates a structured, zero-friction `.zip` archive (or directory) containing:
- Continuous evaluated well logs in LAS 2.0 with embedded `~Parameter` CPI settings
- Multi-well Formation Tops & Fluid Contacts (with interpolated TVD where available)
- Multi-well Deviation Surveys
- Discrete Core RCA point tables (wide CSV, long CSV, and discrete `*_RCA.las` point-sets)
- Formation Pressure Tests (MDT/RFT)
- SCAL Capillary Pressure & Relative Permeability tables + Eclipse/Intersect `SWOF` `.inc` deck
- PRMS/SEC Zonal Reservoir Summary table & Multi-Tab Master Excel Workbook (`.xlsx`)
- Ready-to-run `import_to_techlog.py` and `import_to_geolog.tcl` automation scripts
"""

import datetime
import io
import json
import re
import zipfile
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from quick_pp import __version__ as quick_pp_version
from quick_pp import logger
from quick_pp.config import Config
from quick_pp.las_handler import export_to_las_string
from quick_pp.ressum import calc_reservoir_summary
from quick_pp.scal import generate_swof_table


def _sanitize_name(name: str) -> str:
    """Sanitize a project or well name for safe filesystem paths."""
    cleaned = re.sub(r'[<>:"/\\|?*]+', "_", str(name or "export").strip())
    return cleaned or "export"


def _interpolate_tvd(
    md_values: np.ndarray,
    well_df: Optional[pd.DataFrame],
    surveys_df: Optional[pd.DataFrame],
) -> np.ndarray:
    """Interpolate TVD at given MD depths using the well's TVD curve or survey table."""
    if len(md_values) == 0:
        return np.array([], dtype=float)

    # 1. Try TVD curve from well_df
    if (
        well_df is not None
        and not well_df.empty
        and "DEPTH" in well_df.columns
        and "TVD" in well_df.columns
    ):
        sub = well_df[["DEPTH", "TVD"]].dropna().sort_values("DEPTH")
        if len(sub) >= 2:
            return np.interp(
                md_values,
                sub["DEPTH"].to_numpy(dtype=float),
                sub["TVD"].to_numpy(dtype=float),
            )

    # 2. Try wellpathpy minimum curvature on surveys_df if available
    if (
        surveys_df is not None
        and not surveys_df.empty
        and all(c in surveys_df.columns for c in ("md", "inc", "azim"))
        and len(surveys_df) >= 2
    ):
        try:
            import wellpathpy as wpp

            s_sorted = surveys_df.dropna(subset=["md", "inc", "azim"]).sort_values("md")
            if len(s_sorted) >= 2:
                dev = wpp.deviation(
                    s_sorted["md"].to_numpy(dtype=float),
                    s_sorted["inc"].to_numpy(dtype=float),
                    s_sorted["azim"].to_numpy(dtype=float),
                )
                pos = dev.minimum_curvature()
                return np.interp(
                    md_values,
                    np.asarray(pos.md, dtype=float),
                    np.asarray(pos.depth, dtype=float),
                )
        except Exception:
            pass

    # 3. Default fallback: assume vertical (TVD == MD)
    return np.asarray(md_values, dtype=float).copy()


def _assign_zones_from_tops(
    depths: np.ndarray, tops_df: pd.DataFrame
) -> List[str]:
    """Assign formation top zone labels to a depth array."""
    zones = ["UNZONED"] * len(depths)
    if tops_df is None or tops_df.empty or "depth" not in tops_df.columns:
        return zones

    sorted_tops = tops_df.dropna(subset=["depth"]).sort_values("depth")
    for _, row in sorted_tops.iterrows():
        t_name = str(row.get("name") or "ZONE").strip()
        t_depth = float(row["depth"])
        for idx, d in enumerate(depths):
            if d >= t_depth:
                zones[idx] = t_name
    return zones


def _compute_well_zonal_summary(
    well_name: str,
    well_df: pd.DataFrame,
    tops_df: pd.DataFrame,
    cutoffs: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute zonal reservoir summary for a well if CPI curves are present."""
    if well_df is None or well_df.empty or "DEPTH" not in well_df.columns:
        return pd.DataFrame()

    df = well_df.sort_values("DEPTH").reset_index(drop=True)
    depths = pd.to_numeric(df["DEPTH"], errors="coerce").to_numpy(dtype=float)

    # Resolve VSHALE / VCLAY, PHIT / PHIE, SWT / SWE, PERM
    vsh_col = next(
        (c for c in ("VSHALE", "VCLAY", "VSH_GR") if c in df.columns and df[c].notna().any()),
        None,
    )
    phi_col = next(
        (c for c in ("PHIT", "PHIE") if c in df.columns and df[c].notna().any()),
        None,
    )
    sw_col = next(
        (c for c in ("SWT", "SWE", "SW_WS", "SW_ARCHIE", "SHF") if c in df.columns and df[c].notna().any()),
        None,
    )
    perm_col = next(
        (c for c in ("PERM", "PERM_TRANS") if c in df.columns and df[c].notna().any()),
        None,
    )

    if not phi_col and not vsh_col and not sw_col:
        return pd.DataFrame()

    vsh = (
        pd.to_numeric(df[vsh_col], errors="coerce").to_numpy(dtype=float)
        if vsh_col
        else np.zeros(len(df), dtype=float)
    )
    phit = (
        pd.to_numeric(df[phi_col], errors="coerce").to_numpy(dtype=float)
        if phi_col
        else np.zeros(len(df), dtype=float)
    )
    swt = (
        pd.to_numeric(df[sw_col], errors="coerce").to_numpy(dtype=float)
        if sw_col
        else np.ones(len(df), dtype=float) * 0.5
    )
    perm = (
        pd.to_numeric(df[perm_col], errors="coerce").to_numpy(dtype=float)
        if perm_col
        else np.ones(len(df), dtype=float)
    )

    if "ZONES" in df.columns and df["ZONES"].notna().any():
        zones = (
            df["ZONES"]
            .fillna("UNZONED")
            .astype(str)
            .replace({"nan": "UNZONED", "None": "UNZONED", "": "UNZONED"})
            .to_numpy()
        )
    else:
        zones = np.asarray(_assign_zones_from_tops(depths, tops_df))

    active_cutoffs = dict(Config.RESSUM_CUTOFFS)
    if isinstance(cutoffs, dict):
        for k in ("VSHALE", "PHIT", "SWT"):
            if k in cutoffs and cutoffs[k] is not None:
                active_cutoffs[k] = float(cutoffs[k])

    try:
        res_df = calc_reservoir_summary(
            depth=depths,
            vshale=vsh,
            phit=phit,
            swt=swt,
            perm=perm,
            zones=zones,
            cutoffs=active_cutoffs,
        )
        if not res_df.empty:
            res_df.insert(0, "WELL_NAME", well_name)
        return res_df
    except Exception as e:
        logger.debug(f"Could not compute reservoir summary for well '{well_name}': {e}")
        return pd.DataFrame()


def _generate_techlog_loader_script(project_name: str, well_names: List[str]) -> str:
    """Generate a ready-to-run Techlog Python script for automatic project ingestion."""
    return f'''# -*- coding: utf-8 -*-
"""
Auto-generated Techlog Python Handover Loader for Project: {project_name}
Generated by quick_pp v{quick_pp_version}

Usage in Schlumberger / SLB Techlog:
1. Unzip the handover archive to a local folder.
2. Open Techlog -> Python Editor (or AWI Python console).
3. Run this script (or set HANDOVER_DIR below if running from outside the folder).
"""

import csv
import glob
import os

try:
    import TechlogDatabase as db
except ImportError:
    db = None
    print("Note: TechlogDatabase module not detected. Running in dry-run validation mode.")

HANDOVER_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAS_DIR = os.path.join(HANDOVER_DIR, "01_well_logs_las")
TOPS_CSV = os.path.join(HANDOVER_DIR, "02_tops_and_contacts", "formation_tops.csv")
CONTACTS_CSV = os.path.join(HANDOVER_DIR, "02_tops_and_contacts", "fluid_contacts.csv")
SURVEYS_CSV = os.path.join(HANDOVER_DIR, "03_surveys", "well_surveys.csv")
RCA_LAS_DIR = os.path.join(HANDOVER_DIR, "04_core_and_pressure", "las_point_sets")

# Mapping from quick_pp curve mnemonics to standard Techlog Families & Units
TECHLOG_FAMILY_MAP = {{
    "DEPTH": ("Measured Depth", "m"),
    "TVD": ("True Vertical Depth", "m"),
    "CALI": ("Caliper", "in"),
    "BS": ("Bit Size", "in"),
    "GR": ("Gamma Ray", "GAPI"),
    "RT": ("True Resistivity", "ohm.m"),
    "NPHI": ("Neutron Porosity", "v/v"),
    "RHOB": ("Bulk Density", "g/cc"),
    "DRHO": ("Bulk Density Correction", "g/cc"),
    "PEF": ("Photoelectric Factor", "b/e"),
    "DTC": ("Compressional Slowness", "us/ft"),
    "DTS": ("Shear Slowness", "us/ft"),
    "NPHI_HC": ("Neutron Porosity", "v/v"),
    "RHOB_HC": ("Bulk Density", "g/cc"),
    "VSAND": ("Volume Sand", "v/v"),
    "VSILT": ("Volume Silt", "v/v"),
    "VCLAY": ("Volume Clay", "v/v"),
    "VSHALE": ("Shale Volume", "v/v"),
    "VSH_GR": ("Shale Volume", "v/v"),
    "VCALC": ("Volume Calcite", "v/v"),
    "VDOLO": ("Volume Dolomite", "v/v"),
    "VCOAL": ("Volume Coal", "v/v"),
    "PHIT": ("Total Porosity", "v/v"),
    "PHIE": ("Effective Porosity", "v/v"),
    "SWT": ("Water Saturation", "v/v"),
    "SWE": ("Effective Water Saturation", "v/v"),
    "SHF": ("Water Saturation", "v/v"),
    "PERM": ("Permeability", "mD"),
    "CPORE": ("Core Porosity", "v/v"),
    "CPERM": ("Core Permeability", "mD"),
    "FZI": ("Flow Zone Indicator", "um"),
    "RQI": ("Reservoir Quality Index", "um"),
    "ROCK_FLAG": ("Rock Type", "unitless"),
    "RES_FLAG": ("Net Reservoir Flag", "unitless"),
    "PAY_FLAG": ("Net Pay Flag", "unitless"),
    "BADHOLE": ("Bad Hole Flag", "unitless"),
}}


def apply_techlog_families(well_name, dataset_name):
    """Assign Techlog family names and units to curves in a dataset."""
    if db is None:
        return
    for var_name in db.variableList(well_name, dataset_name):
        fam_info = TECHLOG_FAMILY_MAP.get(var_name.upper())
        if fam_info:
            family_name, default_unit = fam_info
            try:
                db.variableFamilyChange(well_name, dataset_name, var_name, family_name)
                curr_unit = db.variableUnit(well_name, dataset_name, var_name)
                if not curr_unit:
                    db.variableUnitChange(well_name, dataset_name, var_name, default_unit)
            except Exception:
                pass


def import_zonation_csv(csv_path, dataset_name="ZONATION"):
    """Load formation tops or fluid contacts CSV into Techlog Zonation dataset."""
    if not os.path.exists(csv_path):
        return
    by_well = {{}}
    with open(csv_path, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wn = row.get("WELL_NAME", "").strip()
            name = (row.get("TOP_NAME") or row.get("CONTACT_NAME") or "").strip()
            md_str = row.get("MD", "")
            if wn and name and md_str != "":
                by_well.setdefault(wn, []).append((float(md_str), name))

    for wn, items in by_well.items():
        items.sort(key=lambda x: x[0])
        depths = [x[0] for x in items]
        names = [x[1] for x in items]
        print(f"[Techlog] Loading {{len(items)}} markers into {{wn}}.{{dataset_name}}")
        if db is not None:
            if not db.wellExists(wn):
                db.wellCreate(wn)
            db.datasetCreate(wn, dataset_name, "MD", "Measured Depth", "m", depths, "interval")
            db.variableSave(wn, dataset_name, "Zone_Name", "Zone Name", "unitless", names)


def main():
    print("=== Starting quick_pp Handover Import for '{project_name}' ===")
    las_files = sorted(glob.glob(os.path.join(LAS_DIR, "*.las")))
    print(f"Found {{len(las_files)}} evaluated well log LAS files in {{LAS_DIR}}")
    for las_path in las_files:
        well_name = os.path.splitext(os.path.basename(las_path))[0]
        print(f"  -> Importing continuous CPI logs for well: {{well_name}}")
        if db is not None:
            if not db.wellExists(well_name):
                db.wellCreate(well_name)
            # Note: If using Techlog interactive import, drag 01_well_logs_las/*.las into Techlog
            # and run apply_techlog_families(well_name, 'QUICK_PP_CPI')
            for ds in db.datasetList(well_name):
                apply_techlog_families(well_name, ds)

    import_zonation_csv(TOPS_CSV, dataset_name="ZONATION")
    import_zonation_csv(CONTACTS_CSV, dataset_name="FLUID_CONTACTS")
    print("=== Handover Import Completed ===")


if __name__ == "__main__":
    main()
'''


def _generate_geolog_loader_script(project_name: str, well_names: List[str]) -> str:
    """Generate a ready-to-run Emerson/Paradigm Geolog Tcl loader script."""
    wells_tcl_list = " ".join(f'"{w}"' for w in well_names)
    return f"""# Auto-generated Geolog Tcl Handover Loader for Project: {project_name}
# Generated by quick_pp v{quick_pp_version}
#
# Usage in Emerson / AspenTech Geolog:
#   Run from Geolog Command Window: source import_to_geolog.tcl

set handover_dir [file dirname [file dirname [info script]]]
set las_dir [file join $handover_dir "01_well_logs_las"]
set rca_las_dir [file join $handover_dir "04_core_and_pressure" "las_point_sets"]
set tops_csv [file join $handover_dir "02_tops_and_contacts" "formation_tops.csv"]

puts "=== Starting quick_pp Geolog Handover Import for {project_name} ==="
set well_list [list {wells_tcl_list}]

foreach well_name $well_list {{
    set las_file [file join $las_dir "${{well_name}}.las"]
    if {{[file exists $las_file]}} {{
        puts "Loading continuous CPI LAS for well: $well_name"
        catch {{
            tp_in_las -file $las_file -well $well_name -set QUICK_PP_CPI -overwrite yes
        }}
    }}

    set rca_file [file join $rca_las_dir "${{well_name}}_RCA.las"]
    if {{[file exists $rca_file]}} {{
        puts "Loading discrete Core RCA point set for well: $well_name"
        catch {{
            tp_in_las -file $rca_file -well $well_name -set CORE_RCA -overwrite yes
        }}
    }}
}}

puts "=== Completed Geolog LAS & RCA Import. Load $tops_csv into TOPS set via ASCII Importer ==="
"""


def _generate_handover_readme(
    project_name: str,
    well_names: List[str],
    manifest: Dict[str, Any],
) -> str:
    """Generate a markdown README guide explaining how to load the artefacts into Techlog, Geolog, IP, and Petrel."""
    counts = manifest.get("inventory_counts", {})
    return f"""# `{project_name}` — Petrophysical Handover Package

Generated by **quick_pp v{quick_pp_version}** on `{manifest.get("exported_at", "")}`.

## Package Inventory
- **Wells Exported ({len(well_names)})**: {", ".join(f"`{w}`" for w in well_names) if well_names else "None"}
- **Formation Tops**: {counts.get("formation_tops", 0)} markers
- **Fluid Contacts**: {counts.get("fluid_contacts", 0)} contacts
- **Well Deviation Surveys**: {counts.get("well_surveys", 0)} station points
- **Core RCA Plugs**: {counts.get("core_rca_samples", 0)} samples ({counts.get("core_measurements_long", 0)} measurements)
- **SCAL Capillary Pressure ($P_c$)**: {counts.get("scal_pc_points", 0)} pairs
- **SCAL Relative Permeability ($k_r$)**: {counts.get("scal_relperm_points", 0)} points
- **Formation Pressure Tests (MDT/RFT)**: {counts.get("pressure_tests", 0)} points
- **Zonal Reservoir Summary Rows**: {counts.get("reservoir_summary_rows", 0)} rows

---

## Directory Structure

| Folder / File | Contents & Purpose |
| :--- | :--- |
| `01_well_logs_las/*.las` | **Continuous Evaluated Well Logs (LAS 2.0)** — Raw logs + computed CPI logs (`VCLAY`, `VSHALE`, `PHIT`, `PHIE`, `SWT`, `PERM`, `ROCK_FLAG`, `RES_FLAG`, `PAY_FLAG`, `BADHOLE`). Each LAS file embeds the well's petrophysical parameters (`m`, `n`, salinity, mineral endpoints, cutoffs) inside the `~Parameter` (`~P`) block. |
| `02_tops_and_contacts/formation_tops.csv` | **Multi-Well Formation Tops** (`WELL_NAME, UWI, TOP_NAME, MD, TVD, DEPTH_UOM`) ready for batch Zonation import. |
| `02_tops_and_contacts/fluid_contacts.csv` | **Multi-Well Fluid Contacts** (`WELL_NAME, UWI, CONTACT_NAME, MD, TVD, DEPTH_UOM`) such as OWC, GOC, FWL, ODT. |
| `03_surveys/well_surveys.csv` | **Multi-Well Deviation Surveys** (`WELL_NAME, UWI, MD, INC, AZIM, TVD`). |
| `04_core_and_pressure/core_rca_points.csv` | **Discrete Core Plug RCA Table (Wide)** (`WELL_NAME, SAMPLE_NAME, DEPTH, CPORE, CPERM, DESCRIPTION, REMARK`). |
| `04_core_and_pressure/las_point_sets/*_RCA.las` | **Discrete Core Plug LAS 2.0 Point-Sets** (`STEP = 0.0`) — Drag-and-drop directly into Techlog/Geolog to create a separate `CORE_RCA` point dataset without resampling onto the wireline depth grid. |
| `04_core_and_pressure/pressure_tests_mdt.csv` | **Formation Pressure Tests (MDT/RFT)** (`WELL_NAME, DEPTH, PRESSURE, PRESSURE_UOM, MOBILITY, PERMEABILITY, TEMPERATURE, TEMPERATURE_UOM, REMARKS`). |
| `05_scal_and_sim/scal_capillary_pressure.csv` | **SCAL Capillary Pressure Curves** (`WELL_NAME, SAMPLE_NAME, DEPTH, PC, SW, EXPERIMENT_TYPE, CYCLE`). |
| `05_scal_and_sim/scal_relperm_tables.csv` | **SCAL Relative Permeability Curves** (`WELL_NAME, SAMPLE_NAME, DEPTH, SATURATION, ...`). |
| `05_scal_and_sim/eclipse_swof.inc` | **Simulator-Ready `SWOF` Deck** for Eclipse / Intersect / tNavigator. |
| `06_summaries_and_audit/{_sanitize_name(project_name)}_Master_Workbook.xlsx` | **Multi-Tab Master Excel Workbook** containing all tables, zonal Net-to-Gross / Pay summaries, and CPI parameter audit trail. |
| `06_summaries_and_audit/CPI_Parameters_Manifest.json` | **Machine-Readable Audit Manifest** of all model endpoints, Archie/Waxman-Smits parameters, and cutoffs per well. |
| `import_scripts/import_to_techlog.py` | **Techlog Python Auto-Importer** — Assigns curve families/units and loads Zonation markers automatically. |
| `import_scripts/import_to_geolog.tcl` | **Geolog Tcl Batch Loader** — Loads continuous CPI LAS and discrete `CORE_RCA` sets into Geolog. |

---

## Quick-Start Import Guide

### 1. SLB Techlog
1. **Continuous Logs & Core Point Sets**: Select all `.las` files in `01_well_logs_las/` (and `04_core_and_pressure/las_point_sets/`) and drag them into Techlog.
2. **Zonation & Surveys**: Drag `02_tops_and_contacts/formation_tops.csv` and `03_surveys/well_surveys.csv` into Techlog's ASCII Importer (columns `WELL_NAME`, `TOP_NAME`, `MD`, `TVD` are pre-formatted).
3. **Auto-Assign Families & Zonation via Python**: Open **Techlog Python Editor** and run `import_scripts/import_to_techlog.py`.

### 2. Emerson / AspenTech Geolog
1. Open the Geolog Command Window in your project and run:
   ```tcl
   source import_scripts/import_to_geolog.tcl
   ```
2. Or use **Well -> Import -> LAS** to batch-load `01_well_logs_las/*.las` into set `QUICK_PP_CPI` and `04_core_and_pressure/las_point_sets/*_RCA.las` into set `CORE_RCA`.

### 3. Geoactive Interactive Petrophysics (IP) & SLB Petrel
1. Import `01_well_logs_las/*.las` as **Well Logs (LAS 2.0)**.
2. Import `02_tops_and_contacts/formation_tops.csv` as **Well Tops (ASCII)** matching on `WELL_NAME` and `MD`.
3. Import `03_surveys/well_surveys.csv` as **Well Path / Deviation Survey (MD, INC, AZIM)**.
"""


def build_handover_archive_bytes(
    project_name: str,
    project_description: str,
    wells_bundle: List[Dict[str, Any]],
    include_las: bool = True,
    include_excel: bool = True,
) -> bytes:
    """Build an in-memory `.zip` archive bytes object from a list of well bundles.

    Each item in `wells_bundle` is a dict containing:
    - `name`: str (well name)
    - `uwi`: str
    - `depth_uom`: str ('m' or 'ft')
    - `config`: dict (WellConfig parameters)
    - `data`: pd.DataFrame (curve data with 'DEPTH' column)
    - `formation_tops`: pd.DataFrame (columns: ['name', 'depth'])
    - `fluid_contacts`: pd.DataFrame (columns: ['name', 'depth'])
    - `well_surveys`: pd.DataFrame (columns: ['md', 'inc', 'azim'])
    - `pressure_tests`: pd.DataFrame (columns: ['depth', 'pressure', 'pressure_uom', 'mobility', 'permeability', 'temperature', 'temperature_uom', 'remarks'])
    - `core_data`: dict[sample_name -> {'depth', 'description', 'remark', 'measurements', 'relperm', 'pc'}]
    - `scal_pc`: list[dict]
    """
    safe_proj = _sanitize_name(project_name)
    root_prefix = f"{safe_proj}_Handover"

    all_tops_rows: List[Dict[str, Any]] = []
    all_contacts_rows: List[Dict[str, Any]] = []
    all_surveys_rows: List[Dict[str, Any]] = []
    all_pressure_rows: List[Dict[str, Any]] = []
    all_rca_wide_rows: List[Dict[str, Any]] = []
    all_rca_long_rows: List[Dict[str, Any]] = []
    all_scal_pc_rows: List[Dict[str, Any]] = []
    all_scal_relperm_rows: List[Dict[str, Any]] = []
    all_ressum_dfs: List[pd.DataFrame] = []
    wells_overview_rows: List[Dict[str, Any]] = []
    params_manifest_wells: Dict[str, Any] = {}

    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(
        zip_buffer, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as zf:
        well_names_exported: List[str] = []

        for w in wells_bundle:
            w_name = str(w.get("name") or "UNKNOWN_WELL")
            w_uwi = str(w.get("uwi") or w_name)
            w_uom = str(w.get("depth_uom") or "m")
            w_cfg: Dict[str, Any] = dict(w.get("config") or {})
            w_df: pd.DataFrame = (
                w["data"].copy()
                if isinstance(w.get("data"), pd.DataFrame)
                else pd.DataFrame()
            )
            tops_df: pd.DataFrame = (
                w["formation_tops"].copy()
                if isinstance(w.get("formation_tops"), pd.DataFrame)
                else pd.DataFrame()
            )
            contacts_df: pd.DataFrame = (
                w["fluid_contacts"].copy()
                if isinstance(w.get("fluid_contacts"), pd.DataFrame)
                else pd.DataFrame()
            )
            surveys_df: pd.DataFrame = (
                w["well_surveys"].copy()
                if isinstance(w.get("well_surveys"), pd.DataFrame)
                else pd.DataFrame()
            )
            pressure_df: pd.DataFrame = (
                w["pressure_tests"].copy()
                if isinstance(w.get("pressure_tests"), pd.DataFrame)
                else pd.DataFrame()
            )
            core_data: Dict[str, Any] = dict(w.get("core_data") or {})
            scal_pc_list: List[Dict[str, Any]] = list(w.get("scal_pc") or [])

            well_names_exported.append(w_name)
            safe_well = _sanitize_name(w_name)

            # Ensure ZONES column is populated on w_df if formation tops exist
            if not w_df.empty and "DEPTH" in w_df.columns and not tops_df.empty:
                if "ZONES" not in w_df.columns or w_df["ZONES"].isna().all():
                    w_df["ZONES"] = _assign_zones_from_tops(
                        pd.to_numeric(w_df["DEPTH"], errors="coerce").to_numpy(
                            dtype=float
                        ),
                        tops_df,
                    )

            # 1. Export continuous LAS 2.0 file
            if include_las and not w_df.empty and "DEPTH" in w_df.columns:
                try:
                    las_str = export_to_las_string(
                        well_data=w_df,
                        well_name=w_name,
                        uwi=w_uwi,
                        depth_uom=w_uom,
                        parameters=w_cfg,
                    )
                    zf.writestr(
                        f"{root_prefix}/01_well_logs_las/{safe_well}.las",
                        las_str,
                    )
                except Exception as e:
                    logger.warning(f"Failed to generate LAS for well '{w_name}': {e}")

            # 2. Collect Formation Tops (with interpolated TVD)
            if not tops_df.empty and "depth" in tops_df.columns:
                mds = pd.to_numeric(tops_df["depth"], errors="coerce").to_numpy(
                    dtype=float
                )
                tvds = _interpolate_tvd(mds, w_df, surveys_df)
                for (_, row), tvd_val in zip(tops_df.iterrows(), tvds):
                    all_tops_rows.append(
                        {
                            "WELL_NAME": w_name,
                            "UWI": w_uwi,
                            "TOP_NAME": str(row.get("name") or ""),
                            "MD": round(float(row["depth"]), 4),
                            "TVD": round(float(tvd_val), 4)
                            if np.isfinite(tvd_val)
                            else None,
                            "DEPTH_UOM": w_uom,
                        }
                    )

            # 3. Collect Fluid Contacts (with interpolated TVD)
            if not contacts_df.empty and "depth" in contacts_df.columns:
                mds = pd.to_numeric(contacts_df["depth"], errors="coerce").to_numpy(
                    dtype=float
                )
                tvds = _interpolate_tvd(mds, w_df, surveys_df)
                for (_, row), tvd_val in zip(contacts_df.iterrows(), tvds):
                    all_contacts_rows.append(
                        {
                            "WELL_NAME": w_name,
                            "UWI": w_uwi,
                            "CONTACT_NAME": str(row.get("name") or ""),
                            "MD": round(float(row["depth"]), 4),
                            "TVD": round(float(tvd_val), 4)
                            if np.isfinite(tvd_val)
                            else None,
                            "DEPTH_UOM": w_uom,
                        }
                    )

            # 4. Collect Well Surveys
            if not surveys_df.empty and "md" in surveys_df.columns:
                mds = pd.to_numeric(surveys_df["md"], errors="coerce").to_numpy(
                    dtype=float
                )
                tvds = _interpolate_tvd(mds, w_df, surveys_df)
                for (_, row), tvd_val in zip(surveys_df.iterrows(), tvds):
                    all_surveys_rows.append(
                        {
                            "WELL_NAME": w_name,
                            "UWI": w_uwi,
                            "MD": round(float(row["md"]), 4),
                            "INC": round(float(row.get("inc") or 0.0), 4),
                            "AZIM": round(float(row.get("azim") or 0.0), 4),
                            "TVD": round(float(tvd_val), 4)
                            if np.isfinite(tvd_val)
                            else None,
                            "DEPTH_UOM": w_uom,
                        }
                    )

            # 5. Collect Pressure Tests (MDT/RFT)
            if not pressure_df.empty and "depth" in pressure_df.columns:
                mds = pd.to_numeric(pressure_df["depth"], errors="coerce").to_numpy(
                    dtype=float
                )
                tvds = _interpolate_tvd(mds, w_df, surveys_df)

                def _opt_round(v: Any, nd: int = 4) -> Optional[float]:
                    try:
                        f = float(v)
                    except (TypeError, ValueError):
                        return None
                    return round(f, nd) if np.isfinite(f) else None

                for (_, row), tvd_val in zip(pressure_df.iterrows(), tvds):
                    all_pressure_rows.append(
                        {
                            "WELL_NAME": w_name,
                            "UWI": w_uwi,
                            "DEPTH": round(float(row["depth"]), 4),
                            "TVD": round(float(tvd_val), 4)
                            if np.isfinite(tvd_val)
                            else None,
                            "PRESSURE": round(float(row["pressure"]), 4)
                            if pd.notna(row.get("pressure"))
                            else None,
                            "PRESSURE_UOM": str(row.get("pressure_uom") or "psi"),
                            "MOBILITY": _opt_round(row.get("mobility")),
                            "PERMEABILITY": _opt_round(row.get("permeability")),
                            "TEMPERATURE": _opt_round(row.get("temperature"), 2),
                            "TEMPERATURE_UOM": str(row.get("temperature_uom") or "degC"),
                            "REMARKS": str(row.get("remarks") or ""),
                        }
                    )

            # 6. Collect Core RCA & SCAL RelPerm / Pc
            well_rca_wide: List[Dict[str, Any]] = []
            for sample_name, s_info in core_data.items():
                s_depth = s_info.get("depth")
                if s_depth is None:
                    continue
                s_depth_f = round(float(s_depth), 4)
                wide_entry: Dict[str, Any] = {
                    "WELL_NAME": w_name,
                    "UWI": w_uwi,
                    "SAMPLE_NAME": str(sample_name),
                    "DEPTH": s_depth_f,
                    "CPORE": None,
                    "CPERM": None,
                    "DESCRIPTION": s_info.get("description") or "",
                    "REMARK": s_info.get("remark") or "",
                }

                meas_df = s_info.get("measurements")
                has_meas = False
                if isinstance(meas_df, pd.DataFrame) and not meas_df.empty:
                    for _, m_row in meas_df.iterrows():
                        prop_name = str(
                            m_row.get("property")
                            or m_row.get("property_name")
                            or ""
                        ).strip().upper()
                        val = m_row.get("value")
                        unit = m_row.get("unit") or ""
                        if prop_name and pd.notna(val):
                            has_meas = True
                            val_f = float(val)
                            if prop_name in ("POR", "PORO", "CPORE", "POROSITY"):
                                wide_entry["CPORE"] = val_f
                            elif prop_name in ("PERM", "K", "CPERM", "PERMEABILITY"):
                                wide_entry["CPERM"] = val_f
                            else:
                                wide_entry[prop_name] = val_f
                            all_rca_long_rows.append(
                                {
                                    "WELL_NAME": w_name,
                                    "UWI": w_uwi,
                                    "SAMPLE_NAME": str(sample_name),
                                    "DEPTH": s_depth_f,
                                    "PROPERTY_NAME": prop_name,
                                    "VALUE": val_f,
                                    "UNIT": str(unit),
                                }
                            )

                if has_meas or wide_entry["DESCRIPTION"] or wide_entry["REMARK"]:
                    well_rca_wide.append(wide_entry)
                    all_rca_wide_rows.append(wide_entry)

                # Relative permeability table per sample
                rp_df = s_info.get("relperm")
                if isinstance(rp_df, pd.DataFrame) and not rp_df.empty:
                    for _, rp_row in rp_df.iterrows():
                        rp_entry: Dict[str, Any] = {
                            "WELL_NAME": w_name,
                            "UWI": w_uwi,
                            "SAMPLE_NAME": str(sample_name),
                            "DEPTH": s_depth_f,
                        }
                        for col_k, col_v in rp_row.items():
                            rp_entry[str(col_k).upper()] = (
                                float(col_v) if pd.notna(col_v) else None
                            )
                        all_scal_relperm_rows.append(rp_entry)

            # Also if well_rca_wide is empty, check if CPORE/CPERM exist directly as curves on w_df
            if not well_rca_wide and not w_df.empty and "DEPTH" in w_df.columns:
                has_cpore = "CPORE" in w_df.columns and w_df["CPORE"].notna().any()
                has_cperm = "CPERM" in w_df.columns and w_df["CPERM"].notna().any()
                if has_cpore or has_cperm:
                    c_cols = ["DEPTH"] + [
                        c for c in ("CPORE", "CPERM") if c in w_df.columns
                    ]
                    core_sub = w_df[c_cols].dropna(
                        subset=[c for c in ("CPORE", "CPERM") if c in w_df.columns],
                        how="all",
                    )
                    for idx_r, c_row in core_sub.iterrows():
                        sname = f"PLUG-{idx_r + 1}"
                        d_val = round(float(c_row["DEPTH"]), 4)
                        cpore_v = (
                            float(c_row["CPORE"])
                            if "CPORE" in c_row and pd.notna(c_row["CPORE"])
                            else None
                        )
                        cperm_v = (
                            float(c_row["CPERM"])
                            if "CPERM" in c_row and pd.notna(c_row["CPERM"])
                            else None
                        )
                        w_entry = {
                            "WELL_NAME": w_name,
                            "UWI": w_uwi,
                            "SAMPLE_NAME": sname,
                            "DEPTH": d_val,
                            "CPORE": cpore_v,
                            "CPERM": cperm_v,
                            "DESCRIPTION": "",
                            "REMARK": "Extracted from well log core curves",
                        }
                        well_rca_wide.append(w_entry)
                        all_rca_wide_rows.append(w_entry)

            # Export discrete LAS 2.0 point-set for RCA plugs (<well_name>_RCA.las)
            if include_las and well_rca_wide:
                try:
                    rca_las_df = pd.DataFrame(well_rca_wide)
                    rca_las_str = export_to_las_string(
                        well_data=rca_las_df,
                        well_name=w_name,
                        uwi=w_uwi,
                        depth_uom=w_uom,
                        parameters={"DATASET_TYPE": "DISCRETE_CORE_RCA"},
                    )
                    zf.writestr(
                        f"{root_prefix}/04_core_and_pressure/las_point_sets/{safe_well}_RCA.las",
                        rca_las_str,
                    )
                except Exception as e:
                    logger.debug(
                        f"Could not write discrete RCA LAS for well '{w_name}': {e}"
                    )

            # Collect SCAL Capillary Pressure (Pc, Sw) rows
            for pc_row in scal_pc_list:
                all_scal_pc_rows.append(
                    {
                        "WELL_NAME": w_name,
                        "UWI": w_uwi,
                        "SAMPLE_NAME": str(pc_row.get("sample_name") or ""),
                        "DEPTH": round(float(pc_row["depth"]), 4)
                        if pc_row.get("depth") is not None
                        else None,
                        "PC": round(float(pc_row["pc"]), 6)
                        if pc_row.get("pc") is not None
                        else None,
                        "SW": round(float(pc_row["sw"]), 6)
                        if pc_row.get("sw") is not None
                        else None,
                        "EXPERIMENT_TYPE": str(
                            pc_row.get("experiment_type") or "micp"
                        ),
                        "CYCLE": str(pc_row.get("cycle") or "drainage"),
                    }
                )

            # 7. Compute Zonal Reservoir Summary
            ressum_df = _compute_well_zonal_summary(
                well_name=w_name,
                well_df=w_df,
                tops_df=tops_df,
                cutoffs=w_cfg.get("ressum_cutoffs"),
            )
            if not ressum_df.empty:
                all_ressum_dfs.append(ressum_df)

            # 8. Collect Well Overview & Parameters Manifest
            min_d = (
                round(float(w_df["DEPTH"].min()), 2)
                if not w_df.empty and "DEPTH" in w_df.columns
                else None
            )
            max_d = (
                round(float(w_df["DEPTH"].max()), 2)
                if not w_df.empty and "DEPTH" in w_df.columns
                else None
            )
            curve_list = [
                str(c)
                for c in w_df.columns
                if str(c).upper() not in ("WELL_NAME", "UWI")
            ]
            wells_overview_rows.append(
                {
                    "WELL_NAME": w_name,
                    "UWI": w_uwi,
                    "DEPTH_UOM": w_uom,
                    "MIN_DEPTH": min_d,
                    "MAX_DEPTH": max_d,
                    "ROW_COUNT": int(len(w_df)),
                    "CURVE_COUNT": len(curve_list),
                    "FORMATION_TOPS_COUNT": int(len(tops_df)),
                    "FLUID_CONTACTS_COUNT": int(len(contacts_df)),
                    "SURVEY_POINTS_COUNT": int(len(surveys_df)),
                    "PRESSURE_TESTS_COUNT": int(len(pressure_df)),
                    "CORE_RCA_SAMPLES_COUNT": int(len(well_rca_wide)),
                    "SCAL_PC_POINTS_COUNT": int(len(scal_pc_list)),
                    "CURVES": ", ".join(curve_list),
                }
            )
            params_manifest_wells[w_name] = {
                "uwi": w_uwi,
                "depth_uom": w_uom,
                "min_depth": min_d,
                "max_depth": max_d,
                "row_count": int(len(w_df)),
                "curves": curve_list,
                "parameters": w_cfg,
            }

        # Convert collected lists to DataFrames with clean headers even when empty
        tops_out_df = (
            pd.DataFrame(all_tops_rows)
            if all_tops_rows
            else pd.DataFrame(
                columns=["WELL_NAME", "UWI", "TOP_NAME", "MD", "TVD", "DEPTH_UOM"]
            )
        )
        contacts_out_df = (
            pd.DataFrame(all_contacts_rows)
            if all_contacts_rows
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "UWI",
                    "CONTACT_NAME",
                    "MD",
                    "TVD",
                    "DEPTH_UOM",
                ]
            )
        )
        surveys_out_df = (
            pd.DataFrame(all_surveys_rows)
            if all_surveys_rows
            else pd.DataFrame(
                columns=["WELL_NAME", "UWI", "MD", "INC", "AZIM", "TVD", "DEPTH_UOM"]
            )
        )
        rca_wide_out_df = (
            pd.DataFrame(all_rca_wide_rows)
            if all_rca_wide_rows
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "UWI",
                    "SAMPLE_NAME",
                    "DEPTH",
                    "CPORE",
                    "CPERM",
                    "DESCRIPTION",
                    "REMARK",
                ]
            )
        )
        rca_long_out_df = (
            pd.DataFrame(all_rca_long_rows)
            if all_rca_long_rows
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "UWI",
                    "SAMPLE_NAME",
                    "DEPTH",
                    "PROPERTY_NAME",
                    "VALUE",
                    "UNIT",
                ]
            )
        )
        pressure_out_df = (
            pd.DataFrame(all_pressure_rows)
            if all_pressure_rows
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "UWI",
                    "DEPTH",
                    "TVD",
                    "PRESSURE",
                    "PRESSURE_UOM",
                    "MOBILITY",
                    "PERMEABILITY",
                    "TEMPERATURE",
                    "TEMPERATURE_UOM",
                    "REMARKS",
                ]
            )
        )
        scal_pc_out_df = (
            pd.DataFrame(all_scal_pc_rows)
            if all_scal_pc_rows
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "UWI",
                    "SAMPLE_NAME",
                    "DEPTH",
                    "PC",
                    "SW",
                    "EXPERIMENT_TYPE",
                    "CYCLE",
                ]
            )
        )
        scal_rp_out_df = (
            pd.DataFrame(all_scal_relperm_rows)
            if all_scal_relperm_rows
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "UWI",
                    "SAMPLE_NAME",
                    "DEPTH",
                    "SATURATION",
                    "KR_WATER",
                    "KR_OIL",
                ]
            )
        )
        ressum_out_df = (
            pd.concat(all_ressum_dfs, ignore_index=True)
            if all_ressum_dfs
            else pd.DataFrame(
                columns=[
                    "WELL_NAME",
                    "zones",
                    "flag",
                    "top",
                    "bottom",
                    "gross",
                    "net",
                    "ntg",
                    "av_vshale",
                    "av_phit",
                    "av_swt",
                    "av_bvhc",
                    "av_perm_am",
                    "av_perm_gm",
                    "av_perm_hm",
                    "av_perm_pw",
                    "kv_kh_ratio",
                ]
            )
        )
        overview_out_df = pd.DataFrame(wells_overview_rows)

        # Write CSV tables into the archive
        zf.writestr(
            f"{root_prefix}/02_tops_and_contacts/formation_tops.csv",
            tops_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/02_tops_and_contacts/fluid_contacts.csv",
            contacts_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/03_surveys/well_surveys.csv",
            surveys_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/04_core_and_pressure/core_rca_points.csv",
            rca_wide_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/04_core_and_pressure/core_measurements_long.csv",
            rca_long_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/04_core_and_pressure/pressure_tests_mdt.csv",
            pressure_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/05_scal_and_sim/scal_capillary_pressure.csv",
            scal_pc_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/05_scal_and_sim/scal_relperm_tables.csv",
            scal_rp_out_df.to_csv(index=False),
        )

        # Generate default Eclipse SWOF deck
        try:
            swof_res = generate_swof_table(n_points=21, model="corey")
            zf.writestr(
                f"{root_prefix}/05_scal_and_sim/eclipse_swof.inc",
                swof_res["swof_keyword"],
            )
        except Exception as e:
            logger.debug(f"Could not generate SWOF table: {e}")

        # Write Zonal Reservoir Summary CSV
        zf.writestr(
            f"{root_prefix}/06_summaries_and_audit/reservoir_summary_by_zone.csv",
            ressum_out_df.to_csv(index=False),
        )
        zf.writestr(
            f"{root_prefix}/06_summaries_and_audit/wells_overview.csv",
            overview_out_df.to_csv(index=False),
        )

        # Build JSON Manifest
        exported_at = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
        manifest_data = {
            "project_name": project_name,
            "project_description": project_description or "",
            "exported_at": exported_at,
            "quick_pp_version": quick_pp_version,
            "inventory_counts": {
                "wells": len(well_names_exported),
                "formation_tops": int(len(tops_out_df)),
                "fluid_contacts": int(len(contacts_out_df)),
                "well_surveys": int(len(surveys_out_df)),
                "core_rca_samples": int(len(rca_wide_out_df)),
                "core_measurements_long": int(len(rca_long_out_df)),
                "pressure_tests": int(len(pressure_out_df)),
                "scal_pc_points": int(len(scal_pc_out_df)),
                "scal_relperm_points": int(len(scal_rp_out_df)),
                "reservoir_summary_rows": int(len(ressum_out_df)),
            },
            "wells": params_manifest_wells,
        }
        zf.writestr(
            f"{root_prefix}/06_summaries_and_audit/CPI_Parameters_Manifest.json",
            json.dumps(manifest_data, indent=2, default=str),
        )

        # Build Multi-Tab Master Excel Workbook (.xlsx) if openpyxl is available
        if include_excel:
            try:
                excel_buf = io.BytesIO()
                params_rows = []
                for wn, wmeta in params_manifest_wells.items():
                    flat_p = {"WELL_NAME": wn, "UWI": wmeta["uwi"], "DEPTH_UOM": wmeta["depth_uom"]}
                    for pk, pv in (wmeta.get("parameters") or {}).items():
                        flat_p[str(pk).upper()] = (
                            json.dumps(pv) if isinstance(pv, (dict, list, tuple)) else pv
                        )
                    params_rows.append(flat_p)
                params_df = pd.DataFrame(params_rows)

                with pd.ExcelWriter(excel_buf, engine="openpyxl") as writer:
                    overview_out_df.to_excel(
                        writer, sheet_name="Wells_Overview", index=False
                    )
                    ressum_out_df.to_excel(
                        writer, sheet_name="Reservoir_Summary", index=False
                    )
                    tops_out_df.to_excel(
                        writer, sheet_name="Formation_Tops", index=False
                    )
                    contacts_out_df.to_excel(
                        writer, sheet_name="Fluid_Contacts", index=False
                    )
                    rca_wide_out_df.to_excel(
                        writer, sheet_name="Core_RCA", index=False
                    )
                    scal_pc_out_df.to_excel(
                        writer, sheet_name="SCAL_Capillary_Pressure", index=False
                    )
                    scal_rp_out_df.to_excel(
                        writer, sheet_name="SCAL_RelPerm", index=False
                    )
                    pressure_out_df.to_excel(
                        writer, sheet_name="Pressure_Tests", index=False
                    )
                    surveys_out_df.to_excel(
                        writer, sheet_name="Well_Surveys", index=False
                    )
                    params_df.to_excel(
                        writer, sheet_name="CPI_Parameters", index=False
                    )

                zf.writestr(
                    f"{root_prefix}/06_summaries_and_audit/{safe_proj}_Master_Workbook.xlsx",
                    excel_buf.getvalue(),
                )
            except Exception as e:
                logger.debug(
                    f"Skipping Master Excel Workbook generation (openpyxl optional): {e}"
                )

        # Write Techlog & Geolog automation scripts + README
        zf.writestr(
            f"{root_prefix}/import_scripts/import_to_techlog.py",
            _generate_techlog_loader_script(project_name, well_names_exported),
        )
        zf.writestr(
            f"{root_prefix}/import_scripts/import_to_geolog.tcl",
            _generate_geolog_loader_script(project_name, well_names_exported),
        )
        zf.writestr(
            f"{root_prefix}/README_HANDOVER.md",
            _generate_handover_readme(project_name, well_names_exported, manifest_data),
        )

    return zip_buffer.getvalue()


def collect_well_handover_dict(
    well_obj: Any,
    override_df: Optional[pd.DataFrame] = None,
    override_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Extract a well bundle dictionary from a `quick_pp.database.objects.Well` instance."""
    try:
        well_df = (
            override_df.copy()
            if override_df is not None and not override_df.empty
            else well_obj.get_data_optimized()
        )
    except Exception:
        well_df = well_obj.get_data()

    cfg_dict = dict(getattr(well_obj.config, "__dict__", {}) or {})
    if override_config:
        cfg_dict.update(override_config)

    return {
        "name": well_obj.name,
        "uwi": well_obj.uwi or well_obj.name,
        "depth_uom": getattr(well_obj._orm_well, "depth_uom", None) or "m",
        "config": cfg_dict,
        "data": well_df,
        "formation_tops": well_obj.get_formation_tops(),
        "fluid_contacts": well_obj.get_fluid_contacts(),
        "well_surveys": well_obj.get_well_surveys(),
        "pressure_tests": well_obj.get_pressure_tests(),
        "core_data": well_obj.get_core_data(),
        "scal_pc": well_obj.get_scal_pc_data(),
    }

