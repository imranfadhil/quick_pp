"""Unified server-side Workspace Compute API.

Provides compute-by-reference endpoints for petrophysical workflows so the frontend
only sends lightweight model parameters (and optional depth/zone filters + `save` flag)
instead of round-tripping 10,000+ log rows on every parameter adjustment.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from quick_pp.app.backend.services.database import (
    _get_merged_dataframe_sync,
    dataframe_to_columnar,
    dataframe_to_records,
    filter_merged_dataframe,
    get_project_summary,
    invalidate_well_cache,
)
from quick_pp.app.backend.services.rock_typing import _compute_fzi_data_sync
from quick_pp.app.backend.services.database import (
    _ephemeral_wells,
    _merged_data_cache,
    reset_demo_project_state,
    resolve_db_well_name,
)
from quick_pp.app.backend.utils.auth import is_demo_mode
from quick_pp.app.backend.utils.db import get_db
from quick_pp.core_analysis import (
    auto_j_params,
    fit_poroperm_curve,
    leverett_j,
    normalize_sw,
    sw_shf_leverett_j,
)
from quick_pp.config import Config
from quick_pp.database import objects as db_objects
from quick_pp.lithology import gr_index
from quick_pp.lithology.multi_mineral import MultiMineral
from quick_pp.lithology.sand_silt_clay import SandSiltClay
from quick_pp.permeability import (
    choo_permeability,
    coates_permeability,
    kozeny_carman_permeability,
    timur_permeability,
    tixier_permeability,
)
from quick_pp.porosity import (
    density_porosity,
    estimate_shale_porosity,
    neu_den_xplot_poro,
    rho_matrix,
)
from quick_pp.qaqc import (
    apply_environmental_corrections,
    neu_den_xplot_hc_correction,
    stratigraphic_normalization,
)
from quick_pp.ressum import calc_reservoir_summary
from quick_pp.rock_type import calc_fzi, find_cutoffs, rock_typing
from quick_pp.saturation import (
    archie_saturation,
    estimate_b_waxman_smits,
    estimate_qvn,
    estimate_rw_temperature_salinity,
    estimate_temperature_gradient,
    optimize_fwl,
    waxman_smits_saturation,
)
from quick_pp.uncertainty import calc_stochastic_reservoir_summary
from quick_pp.upscaling import expand_blocked_logs, upscale_logs

router = APIRouter(
    prefix="/workspace/projects/{project_id}",
    tags=["Workspace Compute"],
)

_ephemeral_well_configs: Dict[Tuple[int, str], Dict[str, Any]] = {}

ALL_LITHOLOGY_VOL_COLS: Tuple[str, ...] = (
    "VSAND",
    "VSILT",
    "VCLAY",
    "VSHALE",
    "VCALC",
    "VDOLO",
    "VKAOL",
    "VFELD",
    "VANHY",
    "VGYPS",
    "VHALI",
    "VPYRI",
    "VCOAL",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _update_in_memory_well_cache(
    project_id: int, well_name: str, curves_df: pd.DataFrame
) -> None:
    """Update the in-memory merged DataFrame cache so chained live calculations see fresh curves immediately."""
    if curves_df is None or curves_df.empty or "DEPTH" not in curves_df.columns:
        return
    try:
        import time

        base_df = _get_merged_dataframe_sync(project_id, well_name)
        if not base_df.empty and "DEPTH" in base_df.columns:
            upd = curves_df.drop(columns=["WELL_NAME"], errors="ignore").copy()
            upd.columns = [str(c).upper() for c in upd.columns]
            upd = upd.drop_duplicates(subset=["DEPTH"], keep="last")
            overlap_cols = [
                c for c in upd.columns if c != "DEPTH" and c in base_df.columns
            ]
            merged_mem = base_df.drop(columns=overlap_cols, errors="ignore").merge(
                upd, on="DEPTH", how="left"
            )
            for c in overlap_cols:
                if c in base_df.columns:
                    merged_mem[c] = merged_mem[c].combine_first(base_df[c])
            _merged_data_cache[f"{project_id}_{well_name}_0.16"] = (
                time.time(),
                merged_mem,
            )
    except Exception:
        pass


def _save_curves_to_well(project_id: int, well_name: str, curves_df: pd.DataFrame) -> int:
    """Persist a DataFrame containing DEPTH and one or more computed curve columns to the database."""
    if curves_df is None or curves_df.empty or "DEPTH" not in curves_df.columns:
        return 0
    df_to_save = curves_df.copy()
    if "WELL_NAME" not in df_to_save.columns:
        df_to_save["WELL_NAME"] = well_name

    _update_in_memory_well_cache(project_id, well_name, df_to_save)

    if is_demo_mode():
        return len(df_to_save)

    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)
        proj.update_data(df_to_save, group_by="WELL_NAME")
        proj.save()

    invalidate_well_cache(project_id=project_id, well_name=well_name)
    return len(df_to_save)


def _get_filtered_well_df(
    project_id: int,
    well_name: str,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
    tolerance: float = 0.16,
) -> pd.DataFrame:
    """Fetch cached merged well DataFrame and apply optional depth/zone filters."""
    merged = _get_merged_dataframe_sync(project_id, well_name, tolerance=tolerance)
    if merged.empty:
        raise ValueError(f"No data found for well '{well_name}' in project {project_id}")
    filtered = filter_merged_dataframe(
        merged, min_depth=min_depth, max_depth=max_depth, zones=zones
    )
    if filtered.empty:
        raise ValueError(
            f"No data rows remain for well '{well_name}' after applying depth/zone filters"
        )
    return filtered


def _curve_summary_stats(series: pd.Series) -> Dict[str, Optional[float]]:
    """Return compact summary statistics (min, max, mean, p10, p50, p90, valid_count) for a numeric Series."""
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if s.empty:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    return {
        "count": int(len(s)),
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "mean": round(float(s.mean()), 6),
        "p10": round(float(np.percentile(s, 10)), 6),
        "p50": round(float(np.percentile(s, 50)), 6),
        "p90": round(float(np.percentile(s, 90)), 6),
    }


def _select_nphi_rhob(
    df: pd.DataFrame, use_hc_corrected: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """Select NPHI and RHOB series (using NPHI_HC/RHOB_HC if requested and available)."""
    nphi_col = (
        "NPHI_HC"
        if (use_hc_corrected and "NPHI_HC" in df.columns and df["NPHI_HC"].notna().any())
        else "NPHI"
    )
    rhob_col = (
        "RHOB_HC"
        if (use_hc_corrected and "RHOB_HC" in df.columns and df["RHOB_HC"].notna().any())
        else "RHOB"
    )
    if nphi_col not in df.columns or rhob_col not in df.columns:
        raise ValueError(
            f"Required curves NPHI and RHOB not found in well data. Available: {list(df.columns)}"
        )
    return pd.to_numeric(df[nphi_col], errors="coerce"), pd.to_numeric(
        df[rhob_col], errors="coerce"
    )


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class WorkspaceFilterBase(BaseModel):
    min_depth: Optional[float] = Field(None, description="Minimum depth filter")
    max_depth: Optional[float] = Field(None, description="Maximum depth filter")
    zones: Optional[List[str]] = Field(None, description="List of zone names to filter")
    save: bool = Field(
        False, description="Persist computed curves to the database immediately"
    )


class WorkspaceHCCorrectionRequest(WorkspaceFilterBase):
    dry_sand_point: Tuple[float, float] = Field((-0.02, 2.65))
    dry_clay_point: Tuple[float, float] = Field((0.33, 2.7))
    water_point: Tuple[float, float] = Field((1.0, 1.0))
    corr_angle: Optional[float] = Field(
        None,
        description="Explicit correction angle (degrees). If omitted, calculated from rho_water, rho_hc, hi_hc.",
    )
    rho_water: float = Field(1.0, description="Water density (g/cc)")
    rho_hc: float = Field(0.8, description="Hydrocarbon density (g/cc)")
    hi_hc: float = Field(0.9, description="Hydrocarbon hydrogen index")
    buffer: float = Field(0.0, description="Buffer for HC flag detection")


class WorkspaceLithologyRequest(WorkspaceFilterBase):
    model: str = Field(
        "ssc", description="Lithology model: 'ssc', 'multi_mineral', or 'vsh_gr'"
    )
    use_hc_corrected: bool = Field(
        False, description="Use NPHI_HC and RHOB_HC curves if present"
    )
    dry_sand_point: Tuple[float, float] = Field((-0.02, 2.65))
    dry_silt_point: Tuple[Optional[float], Optional[float]] = Field((None, 2.68))
    dry_clay_point: Tuple[Optional[float], Optional[float]] = Field((0.33, 2.7))
    fluid_point: Tuple[float, float] = Field((1.0, 1.0))
    wet_clay_point: Tuple[Optional[float], Optional[float]] = Field((None, None))
    silt_line_angle: float = Field(119.0)
    minerals: List[str] = Field(
        default_factory=lambda: ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"]
    )
    porosity_method: str = Field("density")
    gr_clean: Optional[float] = Field(None, description="Clean sand GR endpoint (API)")
    gr_clay: Optional[float] = Field(None, description="Shale GR endpoint (API)")
    auto_scale: bool = Field(True)


class WorkspacePorosityRequest(WorkspaceFilterBase):
    method: str = Field("neu_den", description="'neu_den' or 'den'")
    lithology_model: Optional[str] = Field(
        None,
        description="Active lithology model ('ssc', 'multi_mineral', 'vsh_gr') so porosity uses matching mineral volumes",
    )
    crossplot_model: str = Field(
        "ssc", description="Crossplot model passed to neu_den_xplot_poro"
    )
    use_hc_corrected: bool = Field(
        False, description="Use NPHI_HC and RHOB_HC curves if present"
    )
    dry_sand_point: Tuple[float, float] = Field((-0.02, 2.65))
    dry_silt_point: Tuple[Optional[float], Optional[float]] = Field((None, 2.68))
    dry_clay_point: Tuple[Optional[float], Optional[float]] = Field((0.33, 2.7))
    fluid_point: Tuple[float, float] = Field((1.0, 1.0))
    wet_clay_point: Tuple[Optional[float], Optional[float]] = Field((None, None))
    silt_line_angle: float = Field(119.0)


class WorkspacePermeabilityRequest(WorkspaceFilterBase):
    method: str = Field(
        "choo",
        description="Permeability model: 'choo', 'timur', 'tixier', 'coates', or 'kozeny_carman'",
    )
    m: float = Field(2.0, description="Cementation/tortuosity exponent")
    A: float = Field(1e6, description="Choo constant A")
    B: float = Field(3.0, description="Choo constant B")
    C: float = Field(2.0, description="Choo constant C")
    s: float = Field(0.01, description="Kozeny-Carman specific surface area factor")
    swirr: float = Field(
        0.2, description="Irreducible water saturation for Timur/Tixier/Coates"
    )


class WorkspaceSaturationRequest(WorkspaceFilterBase):
    method: str = Field(
        "waxman_smits",
        description="Primary saturation curve to save/highlight: 'waxman_smits' or 'archie'",
    )
    meas_system: str = Field("metric", description="'metric' or 'imperial'")
    water_salinity: float = Field(35000.0, description="Formation water salinity (ppm)")
    m: float = Field(2.0, description="Archie cementation exponent m")
    a: float = Field(1.0, description="Archie tortuosity factor a")
    n: float = Field(2.0, description="Archie saturation exponent n")
    m_slope: float = Field(0.1, description="Waxman-Smits m* slope vs VCLAY/PHIT")
    rw: Optional[float] = Field(None, description="Explicit Rw override (ohm.m)")
    b_intercept: float = Field(1.8, description="Waxman-Smits m* intercept")


class WorkspaceReservoirSummaryRequest(WorkspaceFilterBase):
    cut_offs: Dict[str, float] = Field(
        default_factory=lambda: {"VSHALE": 0.4, "PHIT": 0.05, "SWT": 0.8}
    )
    include_upscaling: bool = Field(False)
    cell_size: float = Field(1.0)
    by_zone: bool = Field(True)
    perm_method: str = Field("geometric")
    omega: float = Field(0.33)
    include_uncertainty: bool = Field(False)
    n_samples: int = Field(100)


class WorkspaceRockTypingRequest(BaseModel):
    cutoffs: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.5])
    higher_is_better: bool = Field(True)
    auto_find_cutoffs: bool = Field(False)
    no_of_cutoffs: int = Field(3)
    save: bool = Field(False)


class WorkspacePermTransformRequest(BaseModel):
    fits: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Optional manual (a, b) fits per ROCK_FLAG. If None, auto-fitted from well data.",
    )
    save: bool = Field(False)


class WorkspaceShfRequest(BaseModel):
    cutoffs: List[float] = Field(default_factory=lambda: [0.1, 1.0, 3.0])
    fwl: float = Field(1000.0)
    ift: float = Field(30.0)
    theta: float = Field(30.0)
    gw: float = Field(1.05)
    ghc: float = Field(0.8)
    fits: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Optional J-function fits per ROCK_FLAG. If None, auto-fitted from core SCAL data.",
    )
    save: bool = Field(False)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get(
    "/summary",
    summary="Get project-wide well metadata and summary statistics",
)
def workspace_project_summary(project_id: int) -> Dict[str, Any]:
    """Return lightweight project and per-well summary statistics."""
    return get_project_summary(project_id)


@router.get(
    "/wells/{well_name}/data",
    summary="Get filtered well data in columnar or records format",
)
def workspace_well_data(
    project_id: int,
    well_name: str,
    format: str = Query("columnar", description="'columnar' or 'records'"),
    min_depth: Optional[float] = Query(None),
    max_depth: Optional[float] = Query(None),
    zones: Optional[str] = Query(None, description="Comma-separated zone names"),
    curves: Optional[str] = Query(
        None, description="Optional comma-separated curve mnemonics to include"
    ),
):
    """Return merged well data filtered by depth/zones/curves in compact columnar JSON."""
    try:
        zone_list = [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        df = _get_filtered_well_df(
            project_id, well_name, min_depth=min_depth, max_depth=max_depth, zones=zone_list
        )
        if curves:
            requested = {"DEPTH", "WELL_NAME", "ZONES"} | {
                c.strip().upper() for c in curves.split(",") if c.strip()
            }
            keep_cols = [c for c in df.columns if c.upper() in requested]
            df = df[keep_cols]

        if format.lower() == "records":
            return dataframe_to_records(df)
        return dataframe_to_columnar(df)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve workspace well data: {e}"
        ) from e


@router.get(
    "/wells/{well_name}/bundle",
    summary="Get complete well bundle (columnar curves, stats, tops, contacts, core, pressures, survey) in one request",
)
def workspace_well_bundle(
    project_id: int,
    well_name: str,
    min_depth: Optional[float] = Query(None),
    max_depth: Optional[float] = Query(None),
    zones: Optional[str] = Query(None, description="Comma-separated zone names"),
):
    """Return a single combined payload containing columnar curves, stats, and all ancillary data for a well."""
    try:
        zone_list = [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        df = _get_filtered_well_df(
            project_id, well_name, min_depth=min_depth, max_depth=max_depth, zones=zone_list
        )
        col_payload = dataframe_to_columnar(df)
        curves_map = col_payload.get("data", {})
        if "DEPTH" in curves_map and "DEPT" not in curves_map:
            curves_map["DEPT"] = curves_map["DEPTH"]

        stats: Dict[str, Any] = {}
        for col in df.columns:
            if col in ("WELL_NAME", "ZONES"):
                continue
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                clean = s.dropna()
                stats[col] = {
                    "count": int(len(clean)),
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                    "p10": round(float(np.percentile(clean, 10)), 4),
                    "p50": round(float(np.percentile(clean, 50)), 4),
                    "p90": round(float(np.percentile(clean, 90)), 4),
                }

        ancillary: Dict[str, Any] = {}
        well_config: Dict[str, Any] = {}
        try:
            with get_db() as session:
                proj = db_objects.Project.load(session, project_id=project_id)
                well_obj = proj.get_well(resolve_db_well_name(project_id, well_name))
                ancillary = well_obj.get_ancillary_data() or {}
                well_config = (
                    getattr(well_obj, "config", None).__dict__
                    if getattr(well_obj, "config", None)
                    else {}
                )
                if well_obj._orm_well and isinstance(well_obj._orm_well.config_data, dict):
                    well_config = {**well_config, **well_obj._orm_well.config_data}
        except Exception:
            pass
        eph_cfg = _ephemeral_well_configs.get((project_id, well_name)) or _ephemeral_well_configs.get(
            (project_id, resolve_db_well_name(project_id, well_name))
        )
        if eph_cfg:
            well_config = {**well_config, **eph_cfg}

        def _df_or_list(val: Any) -> List[Dict[str, Any]]:
            if val is None:
                return []
            if isinstance(val, pd.DataFrame):
                return val.to_dict(orient="records") if not val.empty else []
            if isinstance(val, list):
                return val
            return []

        tops_raw = _df_or_list(ancillary.get("formation_tops"))
        tops = [
            {
                "id": f"top-{i}-{t.get('name', i)}",
                "name": str(t.get("name", "")),
                "depth": float(t.get("depth", 0.0)),
            }
            for i, t in enumerate(tops_raw)
            if t.get("depth") is not None
        ]

        contacts_raw = _df_or_list(ancillary.get("fluid_contacts"))
        contacts = [
            {
                "id": f"contact-{i}-{c.get('name', i)}",
                "type": str(c.get("name", "OWC")).upper(),
                "name": str(c.get("name", "OWC")),
                "depth": float(c.get("depth", 0.0)),
            }
            for i, c in enumerate(contacts_raw)
            if c.get("depth") is not None
        ]

        core_samples_list: List[Dict[str, Any]] = []
        core_data_raw = ancillary.get("core_data")
        if isinstance(core_data_raw, dict):
            for sname, sdict in core_data_raw.items():
                meas_df = sdict.get("measurements")
                poro_val = None
                perm_val = None
                if isinstance(meas_df, pd.DataFrame) and not meas_df.empty:
                    for _, mrow in meas_df.iterrows():
                        prop = str(mrow.get("property", "")).upper()
                        if prop in ("POR", "CPORE", "POROSITY"):
                            poro_val = float(mrow["value"])
                        elif prop in ("PERM", "CPERM", "PERMEABILITY"):
                            perm_val = float(mrow["value"])
                core_samples_list.append(
                    {
                        "sample_name": str(sname),
                        "depth": float(sdict.get("depth", 0.0)),
                        "porosity": poro_val,
                        "permeability": perm_val,
                    }
                )

        return {
            "project_id": project_id,
            "well_name": well_name,
            "row_count": col_payload.get("row_count", 0),
            "columns": col_payload.get("columns", []),
            "curves": curves_map,
            "stats": stats,
            "tops": tops,
            "contacts": contacts,
            "core_samples": core_samples_list,
            "pressure_tests": _df_or_list(ancillary.get("pressure_tests")),
            "survey": _df_or_list(
                ancillary.get("well_surveys")
                if "well_surveys" in ancillary
                else ancillary.get("directional_surveys")
            ),
            "config": {k: v for k, v in well_config.items() if not k.startswith("_")},
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to retrieve well bundle: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/hc_correction",
    summary="Compute depth-aligned hydrocarbon correction for NPHI/RHOB on server",
)
def workspace_hc_correction(
    project_id: int, well_name: str, req: WorkspaceHCCorrectionRequest
) -> Dict[str, Any]:
    """Compute NPHI_HC and RHOB_HC aligned with DEPTH and optionally save to the database."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=False)
        valid_mask = nphi.notna() & rhob.notna()
        if not valid_mask.any():
            raise ValueError("No valid NPHI and RHOB pairs found in the selected interval")

        if req.corr_angle is not None:
            corr_angle = float(req.corr_angle)
        else:
            denom = 1.0 - float(req.hi_hc)
            if abs(denom) < 1e-9:
                corr_angle = 50.0
            else:
                rad = math.atan((float(req.rho_water) - float(req.rho_hc)) / denom)
                corr_angle = 90.0 - (rad * 180.0 / math.pi)

        nphi_hc_full = pd.Series(np.nan, index=df.index, dtype=float)
        rhob_hc_full = pd.Series(np.nan, index=df.index, dtype=float)
        hc_flag_full = pd.Series(np.nan, index=df.index, dtype=float)

        nphi_c, rhob_c, hc_flag = neu_den_xplot_hc_correction(
            nphi[valid_mask].to_numpy(dtype=float),
            rhob[valid_mask].to_numpy(dtype=float),
            dry_min1_point=req.dry_sand_point,
            dry_clay_point=req.dry_clay_point,
            water_point=req.water_point,
            corr_angle=corr_angle,
            buffer=req.buffer,
        )

        nphi_hc_full.loc[valid_mask] = np.asarray(nphi_c, dtype=float)
        rhob_hc_full.loc[valid_mask] = np.asarray(rhob_c, dtype=float)
        hc_flag_full.loc[valid_mask] = np.asarray(hc_flag, dtype=float)

        out_df = pd.DataFrame(
            {
                "DEPTH": df["DEPTH"],
                "NPHI": nphi,
                "RHOB": rhob,
                "NPHI_HC": nphi_hc_full,
                "RHOB_HC": rhob_hc_full,
                "HC_FLAG": hc_flag_full,
            }
        )

        saved_rows = 0
        if req.save:
            save_df = out_df.loc[valid_mask, ["DEPTH", "NPHI_HC", "RHOB_HC"]]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[valid_mask, ["DEPTH", "NPHI_HC", "RHOB_HC"]]
            )

        return {
            "well_name": well_name,
            "corr_angle": round(corr_angle, 3),
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": {
                "NPHI_HC": _curve_summary_stats(nphi_hc_full),
                "RHOB_HC": _curve_summary_stats(rhob_hc_full),
            },
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"HC correction failed: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/lithology",
    summary="Compute depth-aligned lithology volumes (SSC, Multi-Mineral, or Vsh-GR) on server",
)
def workspace_lithology(
    project_id: int, well_name: str, req: WorkspaceLithologyRequest
) -> Dict[str, Any]:
    """Compute lithology fractions aligned on DEPTH and optionally save to the database."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        model = req.model.lower()
        out_df = pd.DataFrame({"DEPTH": df["DEPTH"]})
        crossplot_meta: Dict[str, Any] = {}
        save_cols: List[str] = []

        if model == "ssc":
            nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=req.use_hc_corrected)
            valid_mask = nphi.notna() & rhob.notna()
            if not valid_mask.any():
                raise ValueError(
                    "No valid NPHI and RHOB pairs found in the selected interval"
                )

            silt_pt = (
                req.dry_silt_point
                if req.dry_silt_point and req.dry_silt_point[1] is not None
                else (None, 2.68)
            )
            ssc_model = SandSiltClay(
                dry_sand_point=req.dry_sand_point,
                dry_silt_point=silt_pt,
                dry_clay_point=req.dry_clay_point,
                fluid_point=req.fluid_point,
                wet_clay_point=req.wet_clay_point,
                silt_line_angle=req.silt_line_angle,
            )
            vsand, vsilt, vclay, (proj_x, proj_y) = ssc_model.estimate_lithology(
                nphi[valid_mask].reset_index(drop=True),
                rhob[valid_mask].reset_index(drop=True),
            )

            vsand_s = pd.Series(np.nan, index=df.index, dtype=float)
            vsilt_s = pd.Series(np.nan, index=df.index, dtype=float)
            vclay_s = pd.Series(np.nan, index=df.index, dtype=float)
            px_s = pd.Series(np.nan, index=df.index, dtype=float)
            py_s = pd.Series(np.nan, index=df.index, dtype=float)

            vsand_s.loc[valid_mask] = np.clip(np.asarray(vsand, dtype=float), 0.0, 1.0)
            vsilt_s.loc[valid_mask] = np.clip(np.asarray(vsilt, dtype=float), 0.0, 1.0)
            vclay_s.loc[valid_mask] = np.clip(np.asarray(vclay, dtype=float), 0.0, 1.0)
            if proj_x is not None:
                px_s.loc[valid_mask] = np.asarray(proj_x, dtype=float)
            if proj_y is not None:
                py_s.loc[valid_mask] = np.asarray(proj_y, dtype=float)

            out_df["NPHI"] = nphi
            out_df["RHOB"] = rhob
            out_df["VSAND"] = vsand_s
            out_df["VSILT"] = vsilt_s
            out_df["VCLAY"] = vclay_s
            out_df["PROJ_NPHI"] = px_s
            out_df["PROJ_RHOB"] = py_s
            if req.gr_clean is not None and req.gr_clay is not None and "GR" in df.columns:
                gr_s = pd.to_numeric(df["GR"], errors="coerce")
                span = float(req.gr_clay) - float(req.gr_clean)
                if abs(span) > 1e-6 and gr_s.notna().any():
                    vsh_gr_s = ((gr_s - float(req.gr_clean)) / span).clip(0.0, 1.0)
                    out_df["VSH_GR"] = vsh_gr_s
                    out_df["VCLAY"] = (0.65 * vclay_s + 0.35 * vsh_gr_s).clip(0.0, 1.0)
                    rem_m = (1.0 - out_df["VCLAY"]).clip(0.0, 1.0)
                    sand_silt_sum = (vsand_s + vsilt_s).replace(0.0, np.nan)
                    out_df["VSAND"] = np.where(sand_silt_sum.notna(), vsand_s * rem_m / sand_silt_sum, rem_m * 0.7)
                    out_df["VSILT"] = np.where(sand_silt_sum.notna(), vsilt_s * rem_m / sand_silt_sum, rem_m * 0.3)

            # Normalize matrix-relative SSC fractions by (1 - PHIT) so bulk volumes stack with PHIE/PHIT <= 1.0
            rho_ma_ssc = rho_matrix(
                vsand=out_df["VSAND"].to_numpy(dtype=float),
                vsilt=out_df["VSILT"].to_numpy(dtype=float),
                vclay=out_df["VCLAY"].to_numpy(dtype=float),
            )
            phid_ssc = pd.Series(
                np.clip(
                    density_porosity(
                        rhob.to_numpy(dtype=float),
                        rho_ma_ssc,
                        req.fluid_point[1],
                    ),
                    0.0,
                    0.6,
                ),
                index=df.index,
                dtype=float,
            )
            phit_ref = (
                pd.to_numeric(df["PHIT"], errors="coerce")
                .where(lambda s: s.notna() & (s >= 0.0) & (s <= 0.6), phid_ssc)
                .fillna(phid_ssc)
                .fillna(0.0)
                .clip(0.0, 0.6)
            )
            vmatrix_s = (1.0 - phit_ref).clip(0.0, 1.0)
            for mcol in ("VSAND", "VSILT", "VCLAY"):
                out_df[mcol] = (out_df[mcol] * vmatrix_s).clip(0.0, 1.0)

            # Zero out any non-SSC mineral columns so stale carbonate/evaporite volumes do not persist
            for mcol in ALL_LITHOLOGY_VOL_COLS:
                if mcol not in ("VSAND", "VSILT", "VCLAY"):
                    zero_s = pd.Series(np.nan, index=df.index, dtype=float)
                    zero_s.loc[valid_mask] = 0.0
                    out_df[mcol] = zero_s
            save_cols = list(ALL_LITHOLOGY_VOL_COLS)

            crossplot_meta = {
                "dry_sand_point": list(ssc_model.dry_sand_point),
                "dry_silt_point": list(ssc_model.dry_silt_point),
                "dry_clay_point": list(ssc_model.dry_clay_point),
                "fluid_point": list(ssc_model.fluid_point),
            }

        elif model == "multi_mineral":
            nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=req.use_hc_corrected)
            gr = (
                pd.to_numeric(df["GR"], errors="coerce")
                if "GR" in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            )
            pef = (
                pd.to_numeric(df["PEF"], errors="coerce")
                if "PEF" in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            )
            dtc = (
                pd.to_numeric(df["DTC"], errors="coerce")
                if "DTC" in df.columns
                else pd.Series(np.nan, index=df.index, dtype=float)
            )

            valid_mask = nphi.notna() & rhob.notna()
            if not valid_mask.any():
                raise ValueError(
                    "No valid NPHI and RHOB pairs found for multi-mineral inversion"
                )

            selected_minerals = [m.upper() for m in (req.minerals or []) if m]
            if not selected_minerals:
                selected_minerals = ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"]

            mm_model = MultiMineral(
                minerals=selected_minerals,
                porosity_method=req.porosity_method,
                porosity_endpoints={
                    "dry_min1_point": req.dry_sand_point,
                    "dry_clay_point": (
                        req.dry_clay_point
                        if req.dry_clay_point and req.dry_clay_point[0] is not None and req.dry_clay_point[1] is not None
                        else (0.33, 2.7)
                    ),
                    "fluid_point": req.fluid_point,
                },
            )
            # Apply user GR clean / GR clay endpoints to instance-local mineral responses without mutating global Config
            mm_model.responses = {
                k: dict(v) for k, v in Config.MINERALS_LOG_VALUE.items()
            }
            if req.gr_clean is not None:
                for clean_min in (
                    "QUARTZ",
                    "CALCITE",
                    "DOLOMITE",
                    "ANHYDRITE",
                    "GYPSUM",
                    "HALITE",
                ):
                    if clean_min in mm_model.responses:
                        mm_model.responses[clean_min]["GR"] = float(req.gr_clean)
            if req.gr_clay is not None and "SHALE" in mm_model.responses:
                mm_model.responses["SHALE"]["GR"] = float(req.gr_clay)

            res_df = mm_model.estimate_lithology(
                gr=gr[valid_mask].to_numpy(dtype=float),
                nphi=nphi[valid_mask].to_numpy(dtype=float),
                rhob=rhob[valid_mask].to_numpy(dtype=float),
                pef=pef[valid_mask].to_numpy(dtype=float),
                dtc=dtc[valid_mask].to_numpy(dtype=float),
                auto_scale=req.auto_scale,
            )

            for col in res_df.columns:
                col_series = pd.Series(np.nan, index=df.index, dtype=float)
                col_vals = res_df[col]
                if pd.api.types.is_bool_dtype(col_vals):
                    col_vals = col_vals.astype(float)
                col_series.loc[valid_mask] = pd.to_numeric(
                    col_vals, errors="coerce"
                ).to_numpy(dtype=float)
                out_df[col] = col_series
                if col.startswith("V") or col == "PHIT_CONSTRUCTED":
                    save_cols.append(col)

            # Explicitly zero out all unselected mineral volume columns (including VSILT) so deselected minerals clear cleanly
            for mcol in ALL_LITHOLOGY_VOL_COLS:
                if mcol not in out_df.columns:
                    zero_s = pd.Series(np.nan, index=df.index, dtype=float)
                    zero_s.loc[valid_mask] = 0.0
                    out_df[mcol] = zero_s
                if mcol not in save_cols:
                    save_cols.append(mcol)

        elif model == "vsh_gr":
            if "GR" not in df.columns:
                raise ValueError("GR curve not found in well data")
            gr = pd.to_numeric(df["GR"], errors="coerce")
            valid_mask = gr.notna()
            if not valid_mask.any():
                raise ValueError("No valid GR values found in selected interval")
            if req.gr_clean is not None and req.gr_clay is not None and abs(float(req.gr_clay) - float(req.gr_clean)) > 1e-6:
                vsh = ((gr[valid_mask] - float(req.gr_clean)) / (float(req.gr_clay) - float(req.gr_clean))).to_numpy(dtype=float)
            else:
                vsh = gr_index(gr[valid_mask].reset_index(drop=True))
            vclay_s = pd.Series(np.nan, index=df.index, dtype=float)
            vclay_s.loc[valid_mask] = np.clip(
                np.asarray(vsh, dtype=float).ravel(), 0.0, 1.0
            )
            phit_ref = (
                pd.to_numeric(df["PHIT"], errors="coerce")
                if "PHIT" in df.columns
                else pd.Series(0.0, index=df.index, dtype=float)
            ).fillna(0.0).clip(0.0, 0.6)
            vmatrix_s = (1.0 - phit_ref).clip(0.0, 1.0)
            out_df["GR"] = gr
            out_df["VCLAY"] = (vclay_s * vmatrix_s).clip(0.0, 1.0)
            out_df["VSAND"] = ((1.0 - vclay_s).clip(0.0, 1.0) * vmatrix_s).clip(0.0, 1.0)
            for mcol in ALL_LITHOLOGY_VOL_COLS:
                if mcol not in ("VCLAY", "VSAND"):
                    zero_s = pd.Series(np.nan, index=df.index, dtype=float)
                    zero_s.loc[valid_mask] = 0.0
                    out_df[mcol] = zero_s
            save_cols = list(ALL_LITHOLOGY_VOL_COLS)
        else:
            raise ValueError(f"Unsupported lithology model: {req.model}")

        saved_rows = 0
        if save_cols:
            valid_save_mask = out_df[save_cols].notna().any(axis=1)
            save_df = out_df.loc[valid_save_mask, ["DEPTH"] + save_cols]
            if req.save:
                saved_rows = _save_curves_to_well(project_id, well_name, save_df)
            else:
                _update_in_memory_well_cache(project_id, well_name, save_df)

        stats = {col: _curve_summary_stats(out_df[col]) for col in save_cols}
        return {
            "well_name": well_name,
            "model": model,
            "saved": req.save,
            "saved_rows": saved_rows,
            "crossplot": crossplot_meta,
            "stats": stats,
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Lithology calculation failed: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/porosity",
    summary="Compute depth-aligned porosity curves (PHIT, PHIE, PHID) on server",
)
def workspace_porosity(
    project_id: int, well_name: str, req: WorkspacePorosityRequest
) -> Dict[str, Any]:
    """Compute PHIT, PHIE, PHID, and PHIT_SH aligned on DEPTH and compare against core CPORE."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=req.use_hc_corrected)
        valid_mask = nphi.notna() & rhob.notna()
        if not valid_mask.any():
            raise ValueError("No valid NPHI and RHOB pairs found in the selected interval")

        silt_pt = (
            req.dry_silt_point
            if req.dry_silt_point and req.dry_silt_point[1] is not None
            else (None, 2.68)
        )
        ssc_model = SandSiltClay(
            dry_sand_point=req.dry_sand_point,
            dry_silt_point=silt_pt,
            dry_clay_point=req.dry_clay_point,
            fluid_point=req.fluid_point,
            wet_clay_point=req.wet_clay_point,
            silt_line_angle=req.silt_line_angle,
        )
        nphi_s = nphi[valid_mask].reset_index(drop=True)
        rhob_s = rhob[valid_mask].reset_index(drop=True)
        nphi_v = nphi_s.to_numpy(dtype=float)
        rhob_v = rhob_s.to_numpy(dtype=float)

        lith_model = (req.lithology_model or "").lower()
        mineral_vol_cols = [
            c
            for c in (
                "VSAND",
                "VSILT",
                "VCLAY",
                "VCALC",
                "VDOLO",
                "VKAOL",
                "VFELD",
                "VANHY",
                "VGYPS",
                "VHALI",
                "VPYRI",
                "VCOAL",
            )
            if c in df.columns
        ]
        has_mm_cols = any(
            c in df.columns
            and pd.to_numeric(df.loc[valid_mask, c], errors="coerce").fillna(0.0).max() > 1e-4
            for c in ("VCALC", "VDOLO", "VANHY", "VHALI", "VGYPS", "VKAOL", "VFELD", "VPYRI", "VCOAL")
        )
        use_existing_minerals = lith_model in ("multi_mineral", "vsh_gr") or (
            not lith_model and has_mm_cols
        )

        if use_existing_minerals and mineral_vol_cols:
            vol_kwargs: Dict[str, np.ndarray] = {}
            total_min_vol = np.zeros(int(valid_mask.sum()), dtype=float)
            for c in mineral_vol_cols:
                arr = (
                    pd.to_numeric(df.loc[valid_mask, c], errors="coerce")
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                    .to_numpy(dtype=float)
                )
                vol_kwargs[c.lower()] = arr
                total_min_vol += arr

            norm_kwargs = {
                k: np.where(total_min_vol > 1e-6, v / total_min_vol, 0.0)
                for k, v in vol_kwargs.items()
            }
            rho_ma = np.asarray(rho_matrix(**norm_kwargs), dtype=float)
            rho_ma = np.where((rho_ma > 1.2) & np.isfinite(rho_ma), rho_ma, 2.68)
            phid_v = np.asarray(
                density_porosity(rhob_v, rho_ma, req.fluid_point[1]), dtype=float
            )

            if req.method.lower() == "den":
                phit_v = phid_v
            elif (
                lith_model == "multi_mineral"
                and "PHIT_CONSTRUCTED" in df.columns
                and pd.to_numeric(df.loc[valid_mask, "PHIT_CONSTRUCTED"], errors="coerce")
                .notna()
                .any()
            ):
                phit_c = (
                    pd.to_numeric(df.loc[valid_mask, "PHIT_CONSTRUCTED"], errors="coerce")
                    .to_numpy(dtype=float)
                )
                phit_v = np.where(np.isfinite(phit_c), phit_c, phid_v)
            else:
                phit_v = np.clip(0.65 * phid_v + 0.35 * nphi_v, 0.0, 0.6)

            vclay_bulk = vol_kwargs.get(
                "vclay", np.zeros(int(valid_mask.sum()), dtype=float)
            ) + vol_kwargs.get("vkaol", np.zeros(int(valid_mask.sum()), dtype=float))
            vclay_mat = norm_kwargs.get(
                "vclay", np.zeros(int(valid_mask.sum()), dtype=float)
            )
            phit_sh_v = np.asarray(estimate_shale_porosity(nphi_v, phid_v), dtype=float)
            # Use bulk shale volume where mineral volumes are already bulk-normalized
            vclay_eff = np.where(total_min_vol < 0.98, vclay_bulk, vclay_mat)
            phie_v = np.clip(phit_v - (vclay_eff * phit_sh_v), 0.0, phit_v)
        else:
            vsand, vsilt, vclay, _ = ssc_model.estimate_lithology(nphi_s, rhob_s)
            rho_ma = rho_matrix(vsand=vsand, vsilt=vsilt, vclay=vclay)
            phid_v = density_porosity(rhob_v, rho_ma, req.fluid_point[1])

            if req.method.lower() == "den":
                phit_v = np.asarray(phid_v, dtype=float)
            else:
                phit_v = np.asarray(
                    neu_den_xplot_poro(
                        nphi_s,
                        rhob_s,
                        model=req.crossplot_model,
                        dry_min1_point=req.dry_sand_point,
                        dry_silt_point=ssc_model.dry_silt_point,
                        dry_clay_point=ssc_model.dry_clay_point,
                        fluid_point=req.fluid_point,
                    ),
                    dtype=float,
                )

            phit_sh_v = np.asarray(estimate_shale_porosity(nphi_v, phid_v), dtype=float)
            phie_v = phit_v - (np.asarray(vclay, dtype=float) * phit_sh_v)

        phit_s = pd.Series(np.nan, index=df.index, dtype=float)
        phie_s = pd.Series(np.nan, index=df.index, dtype=float)
        phid_s = pd.Series(np.nan, index=df.index, dtype=float)
        phit_sh_s = pd.Series(np.nan, index=df.index, dtype=float)

        phit_s.loc[valid_mask] = np.clip(phit_v, 0.0, 1.0)
        phie_s.loc[valid_mask] = np.clip(phie_v, 0.0, 1.0)
        phid_s.loc[valid_mask] = np.clip(np.asarray(phid_v, dtype=float), 0.0, 1.0)
        phit_sh_s.loc[valid_mask] = np.clip(phit_sh_v, 0.0, 1.0)

        cpore_col = next(
            (c for c in ("CPORE", "KPORE") if c in df.columns), None
        )
        cpore_s = (
            pd.to_numeric(df[cpore_col], errors="coerce")
            if cpore_col
            else pd.Series(np.nan, index=df.index, dtype=float)
        )

        out_df = pd.DataFrame(
            {
                "DEPTH": df["DEPTH"],
                "PHIT": phit_s,
                "PHIE": phie_s,
                "PHID": phid_s,
                "PHIT_SH": phit_sh_s,
                "CPORE": cpore_s,
            }
        )

        # Re-normalize active mineral volumes in out_df so bulk mineral sum == (1 - PHIT)
        active_min_cols = [c for c in ALL_LITHOLOGY_VOL_COLS if c in df.columns and c != "VSHALE"]
        if active_min_cols:
            min_mat = pd.DataFrame(
                {
                    c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(0.0, 1.0)
                    for c in active_min_cols
                },
                index=df.index,
            )
            min_sum = min_mat.sum(axis=1)
            has_pos_min = valid_mask & (min_sum > 1e-6)
            if has_pos_min.any():
                target_vmatrix = (1.0 - phit_s).clip(0.0, 1.0)
                scale_factor = (target_vmatrix / min_sum.replace(0.0, np.nan)).fillna(0.0)
                for c in active_min_cols:
                    scaled_col = pd.Series(np.nan, index=df.index, dtype=float)
                    scaled_col.loc[valid_mask] = np.where(
                        has_pos_min.loc[valid_mask],
                        (min_mat.loc[valid_mask, c] * scale_factor.loc[valid_mask]).clip(0.0, 1.0),
                        0.0,
                    )
                    out_df[c] = scaled_col

        # Core comparison metrics
        core_mask = phit_s.notna() & cpore_s.notna() & (cpore_s > 0)
        core_metrics: Dict[str, Optional[float]] = {
            "count": int(core_mask.sum()),
            "rmse": None,
            "mae": None,
            "r2": None,
        }
        if core_mask.any():
            y_pred = phit_s[core_mask].to_numpy(dtype=float)
            y_true = cpore_s[core_mask].to_numpy(dtype=float)
            diff = y_pred - y_true
            core_metrics["rmse"] = round(float(np.sqrt(np.mean(diff**2))), 5)
            core_metrics["mae"] = round(float(np.mean(np.abs(diff))), 5)
            if len(y_true) > 1:
                ss_res = float(np.sum(diff**2))
                ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
                if ss_tot > 1e-12:
                    core_metrics["r2"] = round(float(1.0 - ss_res / ss_tot), 4)

        save_poro_cols = ["DEPTH", "PHIT", "PHIE", "PHID"] + [
            c for c in active_min_cols if c in out_df.columns
        ]
        saved_rows = 0
        if req.save:
            save_df = out_df.loc[valid_mask, save_poro_cols]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[valid_mask, save_poro_cols]
            )

        return {
            "well_name": well_name,
            "method": req.method,
            "saved": req.save,
            "saved_rows": saved_rows,
            "core_metrics": core_metrics,
            "stats": {
                "PHIT": _curve_summary_stats(phit_s),
                "PHIE": _curve_summary_stats(phie_s),
                "PHID": _curve_summary_stats(phid_s),
            },
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Porosity calculation failed: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/permeability",
    summary="Compute depth-aligned permeability (Choo, Timur, Tixier, Coates, Kozeny-Carman) on server",
)
def workspace_permeability(
    project_id: int, well_name: str, req: WorkspacePermeabilityRequest
) -> Dict[str, Any]:
    """Compute PERM aligned on DEPTH and compare against core CPERM."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        if "PHIT" not in df.columns:
            raise ValueError(
                "PHIT curve not found in well data. Compute and save porosity first."
            )

        phit = pd.to_numeric(df["PHIT"], errors="coerce")
        method = req.method.lower()
        perm_s = pd.Series(np.nan, index=df.index, dtype=float)

        if method == "choo":
            if "VCLAY" not in df.columns or "VSILT" not in df.columns:
                raise ValueError(
                    "VCLAY and VSILT curves are required for Choo permeability. Compute and save lithology first."
                )
            vclay = pd.to_numeric(df["VCLAY"], errors="coerce")
            vsilt = pd.to_numeric(df["VSILT"], errors="coerce")
            valid_mask = phit.notna() & vclay.notna() & vsilt.notna() & (phit > 0)
            if not valid_mask.any():
                raise ValueError(
                    "No valid PHIT, VCLAY, and VSILT rows found in selected interval"
                )
            perm_v = choo_permeability(
                vclay[valid_mask].to_numpy(dtype=float),
                vsilt[valid_mask].to_numpy(dtype=float),
                phit[valid_mask].to_numpy(dtype=float),
                m=req.m,
                A=req.A,
                B=req.B,
                C=req.C,
            )
            perm_s.loc[valid_mask] = np.asarray(perm_v, dtype=float).ravel()

        elif method in ("timur", "tixier", "coates"):
            valid_mask = phit.notna() & (phit > 0)
            if not valid_mask.any():
                raise ValueError("No valid PHIT rows found in selected interval")
            phit_v = phit[valid_mask].to_numpy(dtype=float)
            swirr_v = np.full_like(phit_v, float(req.swirr))
            if method == "timur":
                perm_v = timur_permeability(phit_v, swirr_v)
            elif method == "tixier":
                perm_v = tixier_permeability(phit_v, swirr_v)
            else:
                perm_v = coates_permeability(phit_v, swirr_v)
            perm_s.loc[valid_mask] = np.asarray(perm_v, dtype=float).ravel()

        elif method == "kozeny_carman":
            valid_mask = phit.notna() & (phit > 0)
            if not valid_mask.any():
                raise ValueError("No valid PHIT rows found in selected interval")
            phit_v = phit[valid_mask].to_numpy(dtype=float)
            perm_v = kozeny_carman_permeability(phit_v, S=req.s)
            perm_s.loc[valid_mask] = np.asarray(perm_v, dtype=float).ravel()
        else:
            raise ValueError(f"Unsupported permeability method: {req.method}")

        cperm_col = next(
            (c for c in ("CPERM", "KPERM") if c in df.columns), None
        )
        cperm_s = (
            pd.to_numeric(df[cperm_col], errors="coerce")
            if cperm_col
            else pd.Series(np.nan, index=df.index, dtype=float)
        )

        out_df = pd.DataFrame(
            {
                "DEPTH": df["DEPTH"],
                "PHIT": phit,
                "PERM": perm_s,
                "CPERM": cperm_s,
            }
        )

        core_mask = (perm_s > 0) & (cperm_s > 0)
        core_metrics: Dict[str, Optional[float]] = {
            "count": int(core_mask.sum()),
            "log10_rmse": None,
        }
        if core_mask.any():
            log_diff = np.log10(perm_s[core_mask]) - np.log10(cperm_s[core_mask])
            core_metrics["log10_rmse"] = round(
                float(np.sqrt(np.mean(log_diff**2))), 4
            )

        saved_rows = 0
        if req.save:
            save_df = out_df.loc[perm_s.notna(), ["DEPTH", "PERM"]]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[perm_s.notna(), ["DEPTH", "PERM"]]
            )

        return {
            "well_name": well_name,
            "method": method,
            "saved": req.save,
            "saved_rows": saved_rows,
            "core_metrics": core_metrics,
            "stats": {"PERM": _curve_summary_stats(perm_s)},
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Permeability calculation failed: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/saturation",
    summary="Compute single-pass depth-aligned water saturation (Archie + Waxman-Smits) on server",
)
def workspace_saturation(
    project_id: int, well_name: str, req: WorkspaceSaturationRequest
) -> Dict[str, Any]:
    """Compute TEMP_GRAD, RW, B, QVN, M_STAR, SWT_ARCHIE, SWT_WAXMAN, and crossplot data in one pass."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )

        # Resolve RT and PHIT
        rt_col = next(
            (c for c in ("RT", "ILD", "LLD", "RDEP", "RESD") if c in df.columns),
            None,
        )
        if not rt_col:
            raise ValueError(
                f"Resistivity curve (RT) not found in well data. Available: {list(df.columns)}"
            )
        if "PHIT" not in df.columns:
            raise ValueError(
                "PHIT curve not found in well data. Compute and save porosity first."
            )

        depth = pd.to_numeric(df["DEPTH"], errors="coerce")
        tvdss_col = next(
            (c for c in ("TVDSS", "TVD", "DEPTH") if c in df.columns), "DEPTH"
        )
        tvdss = pd.to_numeric(df[tvdss_col], errors="coerce").fillna(depth)
        rt = pd.to_numeric(df[rt_col], errors="coerce")
        phit = pd.to_numeric(df["PHIT"], errors="coerce")
        vclay = (
            pd.to_numeric(df["VCLAY"], errors="coerce")
            if "VCLAY" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        nphi = (
            pd.to_numeric(df["NPHI"], errors="coerce")
            if "NPHI" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        rhob = (
            pd.to_numeric(df["RHOB"], errors="coerce")
            if "RHOB" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )

        # 1. Temperature gradient & Rw (aligned on every depth row)
        temp_grad_arr = np.asarray(
            estimate_temperature_gradient(tvdss.to_numpy(dtype=float), req.meas_system),
            dtype=float,
        ).ravel()
        rw_arr = np.asarray(
            estimate_rw_temperature_salinity(temp_grad_arr, req.water_salinity),
            dtype=float,
        ).ravel()
        b_arr = np.asarray(
            estimate_b_waxman_smits(temp_grad_arr, rw_arr), dtype=float
        ).ravel()

        temp_grad_s = pd.Series(temp_grad_arr, index=df.index, dtype=float)
        if req.rw is not None and float(req.rw) > 0:
            rw_arr = np.full(len(df), float(req.rw), dtype=float)
            b_arr = np.asarray(estimate_b_waxman_smits(temp_grad_arr, rw_arr), dtype=float).ravel()
        rw_s = pd.Series(rw_arr, index=df.index, dtype=float)
        b_s = pd.Series(b_arr, index=df.index, dtype=float)

        # 2. Shale porosity & Qvn (strictly depth-aligned)
        if "PHIT_SH" in df.columns and df["PHIT_SH"].notna().any():
            phit_sh_s = pd.to_numeric(df["PHIT_SH"], errors="coerce")
        else:
            phit_sh_s = pd.Series(0.15, index=df.index, dtype=float)
            nd_mask = nphi.notna() & rhob.notna()
            if nd_mask.any():
                phid_approx = (2.65 - rhob[nd_mask]) / (2.65 - 1.0)
                phit_sh_s.loc[nd_mask] = np.asarray(
                    estimate_shale_porosity(
                        nphi[nd_mask].to_numpy(dtype=float),
                        phid_approx.to_numpy(dtype=float),
                    ),
                    dtype=float,
                ).ravel()

        qvn_s = pd.Series(np.nan, index=df.index, dtype=float)
        qvn_mask = vclay.notna() & phit.notna() & phit_sh_s.notna() & (phit > 0) & (phit_sh_s > 0)
        if qvn_mask.any():
            qvn_s.loc[qvn_mask] = np.asarray(
                estimate_qvn(
                    vclay[qvn_mask].to_numpy(dtype=float),
                    phit[qvn_mask].to_numpy(dtype=float),
                    phit_sh_s[qvn_mask].to_numpy(dtype=float),
                ),
                dtype=float,
            ).ravel()

        # 3. Archie SWT (strictly depth-aligned)
        swt_archie_s = pd.Series(np.nan, index=df.index, dtype=float)
        archie_mask = rt.notna() & (rt > 0) & rw_s.notna() & (rw_s > 0) & phit.notna() & (phit > 0)
        if archie_mask.any():
            swt_a = archie_saturation(
                rt[archie_mask].to_numpy(dtype=float),
                rw_s[archie_mask].to_numpy(dtype=float),
                phit[archie_mask].to_numpy(dtype=float),
                req.a,
                req.m,
                req.n,
            )
            swt_archie_s.loc[archie_mask] = np.clip(
                np.asarray(swt_a, dtype=float).ravel(), 0.0, 1.0
            )

        # 4. Waxman-Smits SWT (strictly depth-aligned)
        m_star_s = pd.Series(np.nan, index=df.index, dtype=float)
        ws_mask = archie_mask & qvn_s.notna() & b_s.notna() & vclay.notna()
        swt_ws_s = pd.Series(np.nan, index=df.index, dtype=float)
        if ws_mask.any():
            vcl_over_phi = vclay[ws_mask].to_numpy(dtype=float) / phit[
                ws_mask
            ].to_numpy(dtype=float)
            m_star_arr = req.m_slope * vcl_over_phi + req.b_intercept
            m_star_s.loc[ws_mask] = m_star_arr
            swt_w = waxman_smits_saturation(
                rt[ws_mask].to_numpy(dtype=float),
                rw_s[ws_mask].to_numpy(dtype=float),
                phit[ws_mask].to_numpy(dtype=float),
                qvn_s[ws_mask].to_numpy(dtype=float),
                b_s[ws_mask].to_numpy(dtype=float),
                m_star_arr,
            )
            swt_ws_s.loc[ws_mask] = np.clip(
                np.asarray(swt_w, dtype=float).ravel(), 0.0, 1.0
            )

        primary_swt = (
            swt_ws_s if req.method.lower() == "waxman_smits" else swt_archie_s
        )
        bvw_s = phit * primary_swt

        # Crossplot helper columns (CWA vs VCLAY/PHIT)
        vcl_phi_s = pd.Series(np.nan, index=df.index, dtype=float)
        cwa_s = pd.Series(np.nan, index=df.index, dtype=float)
        if ws_mask.any():
            vcl_phi_s.loc[ws_mask] = (
                vclay[ws_mask].to_numpy(dtype=float)
                / phit[ws_mask].to_numpy(dtype=float)
            )
            cwa_s.loc[ws_mask] = (1.0 / rt[ws_mask].to_numpy(dtype=float)) / (
                phit[ws_mask].to_numpy(dtype=float) ** m_star_s[ws_mask].to_numpy(dtype=float)
            )

        out_df = pd.DataFrame(
            {
                "DEPTH": df["DEPTH"],
                "RT": rt,
                "PHIT": phit,
                "VCLAY": vclay,
                "TEMP_GRAD": temp_grad_s,
                "RW": rw_s,
                "B": b_s,
                "QVN": qvn_s,
                "M_STAR": m_star_s,
                "SWT_ARCHIE": swt_archie_s,
                "SWT_WAXMAN": swt_ws_s,
                "SWT": primary_swt,
                "BVW": bvw_s,
                "VCL_OVER_PHI": vcl_phi_s,
                "CWA": cwa_s,
            }
        )

        saved_rows = 0
        if req.save:
            save_df = out_df.loc[primary_swt.notna(), ["DEPTH", "SWT", "BVW"]]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[primary_swt.notna(), ["DEPTH", "SWT", "BVW"]]
            )

        return {
            "well_name": well_name,
            "method": req.method.lower(),
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": {
                "SWT": _curve_summary_stats(primary_swt),
                "SWT_ARCHIE": _curve_summary_stats(swt_archie_s),
                "SWT_WAXMAN": _curve_summary_stats(swt_ws_s),
                "RW": _curve_summary_stats(rw_s),
            },
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Saturation calculation failed: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/reservoir_summary",
    summary="Compute zone-level reservoir summary, optional upscaling, and optional uncertainty directly from DB well data",
)
def workspace_reservoir_summary(
    project_id: int, well_name: str, req: WorkspaceReservoirSummaryRequest
) -> Dict[str, Any]:
    """Calculate reservoir summary table (and optional upscaled blocked logs / Monte Carlo uncertainty) on server."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )

        depth = pd.to_numeric(df["DEPTH"], errors="coerce")
        vclay = (
            pd.to_numeric(df["VCLAY"], errors="coerce")
            if "VCLAY" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        phit = (
            pd.to_numeric(df["PHIT"], errors="coerce")
            if "PHIT" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        swt = (
            pd.to_numeric(df["SWT"], errors="coerce")
            if "SWT" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        perm_col = (
            "PERM_TRANS"
            if ("PERM_TRANS" in df.columns and df["PERM_TRANS"].notna().any())
            else "PERM"
        )
        perm = (
            pd.to_numeric(df[perm_col], errors="coerce")
            if perm_col in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        zones = (
            df["ZONES"].fillna("ALL").astype(str)
            if "ZONES" in df.columns
            else pd.Series("ALL", index=df.index, dtype=str)
        )

        if phit.isna().all() or swt.isna().all():
            raise ValueError(
                "PHIT and SWT curves are required for reservoir summary calculation."
            )

        ressum_df = calc_reservoir_summary(
            depth=depth,
            vshale=vclay,
            phit=phit,
            swt=swt,
            perm=perm,
            zones=zones,
            cutoffs=req.cut_offs,
        )
        ressum_df = ressum_df.dropna(how="all").reset_index(drop=True)
        res_flag = ((vclay < float(req.cut_offs.get("VSHALE", 0.4))) & (phit > float(req.cut_offs.get("PHIT", 0.05)))).astype(int)
        pay_flag = ((res_flag == 1) & (swt < float(req.cut_offs.get("SWT", 0.8)))).astype(int)
        flags_df = pd.DataFrame({"DEPTH": depth, "RES_FLAG": res_flag, "PAY_FLAG": pay_flag, "PAY_FLAGS": pay_flag})

        response: Dict[str, Any] = {
            "well_name": well_name,
            "cut_offs": req.cut_offs,
            "summary": dataframe_to_records(ressum_df),
            "flags": dataframe_to_columnar(flags_df),
        }

        if req.include_upscaling:
            up_input = pd.DataFrame(
                {
                    "DEPTH": depth,
                    "VCLAY": vclay,
                    "PHIT": phit,
                    "SWT": swt,
                    "PERM": perm,
                    "ZONES": zones,
                }
            )
            blocked_df = upscale_logs(
                df=up_input,
                cell_size=req.cell_size,
                by_zone=req.by_zone,
                perm_method=req.perm_method,
                omega=req.omega,
                cutoffs=req.cut_offs,
                return_step_log=False,
            )
            step_df = expand_blocked_logs(up_input, blocked_df, depth_col="DEPTH")
            response["upscaling"] = {
                "n_cells": int(len(blocked_df)),
                "cells": dataframe_to_records(blocked_df.round(4)),
                "blocked_cells": dataframe_to_records(blocked_df.round(4)),
                "step_log": dataframe_to_columnar(step_df.round(4)),
            }

        if req.include_uncertainty:
            unc_res = calc_stochastic_reservoir_summary(
                df=df,
                cutoffs=req.cut_offs,
                n_samples=req.n_samples,
            )
            cpi_cols = [
                c
                for c in [
                    "DEPTH",
                    "SW_P10",
                    "SW_P50",
                    "SW_P90",
                    "PHIT_P10",
                    "PHIT_P50",
                    "PHIT_P90",
                    "PHIE_P10",
                    "PHIE_P50",
                    "PHIE_P90",
                    "VCLAY_P10",
                    "VCLAY_P50",
                    "VCLAY_P90",
                    "BVHC_P10",
                    "BVHC_P50",
                    "BVHC_P90",
                ]
                if c in unc_res["cpi_df"].columns
            ]
            response["uncertainty"] = {
                "summary": dataframe_to_records(unc_res["summary_df"]),
                "volumetrics": unc_res["volumetrics"],
                "cpi": dataframe_to_columnar(unc_res["cpi_df"][cpi_cols].round(4)),
            }

        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Reservoir summary calculation failed: {e}"
        ) from e


@router.post(
    "/rock_typing/compute",
    summary="Compute FZI rock typing across project wells and optionally save ROCK_FLAG",
)
def workspace_rock_typing_compute(
    project_id: int, req: WorkspaceRockTypingRequest
) -> Dict[str, Any]:
    """Compute FZI and ROCK_FLAG across all wells in the project on the server."""
    try:
        fzi_data = _compute_fzi_data_sync(project_id)
        phit_arr = np.asarray(fzi_data["phit"], dtype=float)
        perm_arr = np.asarray(fzi_data["perm"], dtype=float)
        well_names = fzi_data["well_names"]
        depths = fzi_data["depths"]
        zones = fzi_data["zones"]

        valid_mask = (phit_arr > 0) & (phit_arr < 1.0) & (perm_arr > 0)
        fzi_arr = np.full_like(phit_arr, np.nan, dtype=float)
        fzi_arr[valid_mask] = calc_fzi(phit_arr[valid_mask], perm_arr[valid_mask])

        if req.auto_find_cutoffs and valid_mask.sum() >= (req.no_of_cutoffs + 1):
            cutoffs = sorted(
                float(round(c, 4))
                for c in find_cutoffs(fzi_arr[valid_mask], no_of_cutoffs=req.no_of_cutoffs)
            )
        else:
            cutoffs = sorted(float(c) for c in req.cutoffs)

        rf_arr = rock_typing(
            fzi_arr, cut_offs=cutoffs, higher_is_better=req.higher_is_better
        )
        rock_flags = [
            int(val) if not np.isnan(val) else None for val in np.asarray(rf_arr)
        ]
        fzi_list = [
            round(float(val), 5) if not np.isnan(val) else None
            for val in fzi_arr
        ]

        saved_wells = 0
        if req.save:
            by_well: Dict[str, List[Dict[str, Any]]] = {}
            for wn, d, rf in zip(well_names, depths, rock_flags):
                if rf is not None and d is not None:
                    by_well.setdefault(wn, []).append({"DEPTH": d, "ROCK_FLAG": rf})
            for wn, rows in by_well.items():
                _save_curves_to_well(project_id, wn, pd.DataFrame(rows))
            saved_wells = len(by_well)

        # Counts per rock flag
        flag_counts: Dict[str, int] = {}
        for rf in rock_flags:
            if rf is not None:
                key = str(rf)
                flag_counts[key] = flag_counts.get(key, 0) + 1

        return {
            "project_id": project_id,
            "cutoffs": cutoffs,
            "flag_counts": flag_counts,
            "rock_type_counts": flag_counts,
            "saved": req.save,
            "saved_wells": saved_wells,
            "data": {
                "well_names": well_names,
                "depths": depths,
                "zones": zones,
                "phit": fzi_data["phit"],
                "perm": fzi_data["perm"],
                "fzi": fzi_list,
                "rock_flags": rock_flags,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Rock typing computation failed: {e}"
        ) from e


@router.post(
    "/perm_transform/compute",
    summary="Fit poro-perm curves per ROCK_FLAG and compute/save PERM_TRANS across project wells",
)
def workspace_perm_transform_compute(
    project_id: int, req: WorkspacePermTransformRequest
) -> Dict[str, Any]:
    """Fit power-law permeability transforms (a * PHIT^b) per ROCK_FLAG and optionally save PERM_TRANS."""
    try:
        fzi_data = _compute_fzi_data_sync(project_id)
        phit_arr = np.asarray(fzi_data["phit"], dtype=float)
        perm_arr = np.asarray(fzi_data["perm"], dtype=float)
        rock_flags = fzi_data["rock_flags"]
        well_names = fzi_data["well_names"]
        depths = fzi_data["depths"]

        if req.fits:
            fits = {
                str(k): {"a": float(v["a"]), "b": float(v["b"])}
                for k, v in req.fits.items()
            }
        else:
            fits = {}
            unique_flags = sorted({rf for rf in rock_flags if rf is not None})
            if not unique_flags:
                raise ValueError(
                    "No ROCK_FLAG values found in project wells. Run and save Rock Typing first."
                )
            for rf in unique_flags:
                mask = np.array(
                    [
                        (r == rf) and (p > 0) and (k > 0)
                        for r, p, k in zip(rock_flags, phit_arr, perm_arr)
                    ],
                    dtype=bool,
                )
                if mask.sum() >= 2:
                    try:
                        a, b = fit_poroperm_curve(phit_arr[mask], perm_arr[mask])
                        log_k_true = np.log10(np.clip(perm_arr[mask], 1e-6, None))
                        log_k_pred = np.log10(np.clip(float(a) * (phit_arr[mask] ** float(b)), 1e-6, None))
                        ss_res = float(np.sum((log_k_true - log_k_pred) ** 2))
                        ss_tot = float(np.sum((log_k_true - np.mean(log_k_true)) ** 2))
                        r2_val = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 1e-12 else 0.0
                        fits[str(rf)] = {"a": float(a), "b": float(b), "r2": r2_val, "count": int(mask.sum())}
                    except Exception:
                        fits[str(rf)] = {"a": 1.0, "b": 1.0}
                else:
                    fits[str(rf)] = {"a": 1.0, "b": 1.0}

        perm_trans: List[Optional[float]] = []
        for p, rf in zip(phit_arr, rock_flags):
            rf_key = str(rf) if rf is not None else None
            if rf_key and rf_key in fits and p is not None and p > 0:
                val = fits[rf_key]["a"] * (float(p) ** fits[rf_key]["b"])
                perm_trans.append(round(float(val), 6) if math.isfinite(val) else None)
            else:
                perm_trans.append(None)

        saved_wells = 0
        if req.save:
            by_well: Dict[str, List[Dict[str, Any]]] = {}
            for wn, d, pt in zip(well_names, depths, perm_trans):
                if pt is not None and d is not None:
                    by_well.setdefault(wn, []).append({"DEPTH": d, "PERM_TRANS": pt})
            for wn, rows in by_well.items():
                _save_curves_to_well(project_id, wn, pd.DataFrame(rows))
            saved_wells = len(by_well)

        return {
            "project_id": project_id,
            "fits": fits,
            "saved": req.save,
            "saved_wells": saved_wells,
            "data": {
                "well_names": well_names,
                "depths": depths,
                "phit": fzi_data["phit"],
                "perm": fzi_data["perm"],
                "rock_flags": rock_flags,
                "perm_trans": perm_trans,
                "perm_pred": perm_trans,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Permeability transform computation failed: {e}"
        ) from e


@router.post(
    "/shf/compute",
    summary="Compute Leverett-J Saturation Height Function across project wells and optionally save SHF",
)
def workspace_shf_compute(project_id: int, req: WorkspaceShfRequest) -> Dict[str, Any]:
    """Compute Leverett-J SHF curves across all project wells and optionally save SHF to database."""
    try:
        fzi_data = _compute_fzi_data_sync(project_id)
        phit_arr = np.asarray(fzi_data["phit"], dtype=float)
        perm_arr = np.asarray(fzi_data["perm"], dtype=float)
        rock_flags = fzi_data["rock_flags"]
        well_names = fzi_data["well_names"]
        depths = fzi_data["depths"]

        # If ROCK_FLAG not yet saved in wells, compute on the fly from req.cutoffs
        if not any(rf is not None for rf in rock_flags):
            valid_mask = (phit_arr > 0) & (phit_arr < 1.0) & (perm_arr > 0)
            fzi_arr = np.full_like(phit_arr, np.nan, dtype=float)
            fzi_arr[valid_mask] = calc_fzi(phit_arr[valid_mask], perm_arr[valid_mask])
            rf_arr = rock_typing(fzi_arr, cut_offs=req.cutoffs, higher_is_better=True)
            rock_flags = [
                int(v) if not np.isnan(v) else None for v in np.asarray(rf_arr)
            ]

        fits = req.fits or {}
        if not fits:
            # Attempt auto-fit from core SCAL PC data if available
            with get_db() as session:
                proj = db_objects.Project.load(session, project_id=project_id)
                core_rows = []
                for wn in proj.get_well_names():
                    try:
                        anc = proj.get_well_ancillary_data(wn)
                    except Exception:
                        continue
                    core_data = (anc or {}).get("core_data") or {}
                    for sname, sample in core_data.items():
                        meas = sample.get("measurements")
                        pc_df = sample.get("pc")
                        if (
                            meas is None
                            or pc_df is None
                            or meas.empty
                            or pc_df.empty
                        ):
                            continue
                        perm_r = meas[meas["property"].str.upper() == "CPERM"]
                        phit_r = meas[meas["property"].str.upper() == "CPORE"]
                        if perm_r.empty or phit_r.empty:
                            continue
                        cperm = float(perm_r["value"].iloc[0])
                        cpore = float(phit_r["value"].iloc[0])
                        if not (0 < cpore < 1 and cperm > 0):
                            continue
                        rf_val = int(
                            rock_typing(calc_fzi(cpore, cperm), req.cutoffs).item()
                        )
                        pc_col = next(
                            (
                                c
                                for c in pc_df.columns
                                if c.upper() in ("PRESSURE", "PC", "PC_RES")
                            ),
                            None,
                        )
                        sw_col = next(
                            (
                                c
                                for c in pc_df.columns
                                if c.upper() in ("SATURATION", "SW", "SWN")
                            ),
                            None,
                        )
                        if not pc_col or not sw_col:
                            continue
                        swn_s = normalize_sw(pc_df[sw_col].dropna())
                        for pc_v, swn_v in zip(
                            pc_df[pc_col].dropna(), swn_s
                        ):
                            j_v = leverett_j(pc_v, req.ift, req.theta, cperm, cpore)
                            core_rows.append(
                                {
                                    "ROCK_FLAG": rf_val,
                                    "SWN": float(swn_v),
                                    "J": float(j_v),
                                    "Well": wn,
                                    "Sample": sname,
                                }
                            )
                if core_rows:
                    for param in auto_j_params(pd.DataFrame(core_rows)):
                        fits[str(param["ROCK_FLAG"])] = {
                            "a": float(param["a"]),
                            "b": float(param["b"]),
                            "rmse": float(param["rmse"]),
                        }

        valid_flags = [int(rf) for rf in rock_flags if rf is not None]
        highest_flag = max(valid_flags) if valid_flags else None

        shf_values: List[Optional[float]] = []
        for p, k, d, rf in zip(phit_arr, perm_arr, depths, rock_flags):
            rf_key = str(int(rf)) if rf is not None else None
            is_non_res = (
                (rf is not None and highest_flag is not None and int(rf) == highest_flag)
                or rf_key is None
                or rf_key not in fits
            )
            if is_non_res:
                shf_values.append(1.0)
            else:
                a = float(fits[rf_key]["a"])
                b = float(fits[rf_key]["b"])
                shf_v = sw_shf_leverett_j(
                    float(k),
                    float(p),
                    float(d),
                    req.fwl,
                    req.ift,
                    req.theta,
                    req.gw,
                    req.ghc,
                    a,
                    b,
                )
                shf_values.append(
                    round(float(np.clip(shf_v, 0.0, 1.0)), 5)
                    if math.isfinite(shf_v)
                    else None
                )

        saved_wells = 0
        if req.save:
            by_well: Dict[str, List[Dict[str, Any]]] = {}
            for wn, d, shf_v in zip(well_names, depths, shf_values):
                if shf_v is not None and d is not None:
                    by_well.setdefault(wn, []).append({"DEPTH": d, "SHF": shf_v})
            for wn, rows in by_well.items():
                _save_curves_to_well(project_id, wn, pd.DataFrame(rows))
            saved_wells = len(by_well)

        return {
            "project_id": project_id,
            "fits": fits,
            "saved": req.save,
            "saved_wells": saved_wells,
            "data": {
                "well_names": well_names,
                "depths": depths,
                "phit": fzi_data["phit"],
                "perm": fzi_data["perm"],
                "rock_flags": rock_flags,
                "shf": shf_values,
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"SHF computation failed: {e}"
        ) from e


class WorkspaceEnvCorrectionRequest(WorkspaceFilterBase):
    bitsize_default: float = Field(8.5)
    mud_weight: float = Field(9.0)
    surface_temp: float = Field(75.0)
    temp_grad: float = Field(1.5)
    salinity_ppm: float = Field(35000.0)
    rho_mudcake: float = Field(1.5)


class WorkspaceStratNormalizationRequest(BaseModel):
    key_well: Optional[str] = Field(None)
    logs: List[str] = Field(default_factory=lambda: ["GR", "NPHI"])
    method: str = Field("quantile")
    quantiles: Tuple[float, float] = Field((0.05, 0.95))
    save: bool = Field(False)


class WorkspaceOptimizeFwlRequest(BaseModel):
    model: str = Field("leverett_j")
    shm_params: Optional[Dict[str, Any]] = Field(None)
    ift: float = Field(30.0)
    theta: float = Field(30.0)
    gw: float = Field(1.05)
    ghc: float = Field(0.80)
    loss: str = Field("rmse")
    pore_volume_weighted: bool = Field(False)
    equal_well_weights: bool = Field(False)
    mdt_fwl: Optional[float] = Field(None)
    mdt_tolerance: Optional[float] = Field(None)
    mdt_penalty_weight: float = Field(0.0)
    solve_per_well: bool = Field(True)
    optimize_shm_params: bool = Field(False)
    unit: str = Field("metric")
    save: bool = Field(False)


@router.post(
    "/wells/{well_name}/environmental_corrections",
    summary="Apply borehole environmental and invasion corrections to well logs",
)
def workspace_environmental_corrections(
    project_id: int, well_name: str, req: WorkspaceEnvCorrectionRequest
) -> Dict[str, Any]:
    """Apply environmental corrections (GR_CORR, NPHI_CORR, RHOB_CORR, RT_CORR) to a well."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        corr_df = apply_environmental_corrections(
            df,
            bitsize_default=req.bitsize_default,
            mud_weight=req.mud_weight,
            surface_temp=req.surface_temp,
            temp_grad=req.temp_grad,
            salinity_ppm=req.salinity_ppm,
            rho_mudcake=req.rho_mudcake,
        )
        corr_cols = [
            c
            for c in ("GR_CORR", "NPHI_CORR", "RHOB_CORR", "RT_CORR", "BAD_HOLE_FLAG")
            if c in corr_df.columns
        ]
        out_df = corr_df[["DEPTH"] + corr_cols].copy()
        saved_rows = 0
        if req.save and corr_cols:
            saved_rows = _save_curves_to_well(project_id, well_name, out_df)
        stats = {col: _curve_summary_stats(out_df[col]) for col in corr_cols}
        return {
            "well_name": well_name,
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": stats,
            **dataframe_to_columnar(out_df),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Environmental correction failed: {e}"
        ) from e


@router.post(
    "/stratigraphic_normalization",
    summary="Run multi-well stratigraphic (zone-by-zone) log normalization across project wells",
)
def workspace_stratigraphic_normalization(
    project_id: int, req: WorkspaceStratNormalizationRequest
) -> Dict[str, Any]:
    """Normalize log curves across all project wells per geological zone."""
    try:
        proj_summary = get_project_summary(project_id)
        well_names = [w["name"] for w in proj_summary.get("wells", [])]
        if not well_names:
            raise ValueError("No wells found in project.")

        dfs = []
        for wn in well_names:
            wdf = _get_merged_dataframe_sync(project_id, wn)
            if not wdf.empty:
                wdf = wdf.copy()
                wdf["WELL_NAME"] = wn
                if "ZONES" not in wdf.columns:
                    wdf["ZONES"] = "ALL"
                dfs.append(wdf)
        if not dfs:
            raise ValueError("No log data available across project wells.")

        combined = pd.concat(dfs, ignore_index=True)
        norm_df, report_df = stratigraphic_normalization(
            combined,
            key_well=req.key_well,
            zone_col="ZONES",
            quantiles=req.quantiles,
            logs=req.logs,
            well_col="WELL_NAME",
            method=req.method,
            return_report=True,
        )

        norm_cols = [f"{lg}_NORM" for lg in req.logs if f"{lg}_NORM" in norm_df.columns]
        saved_wells = 0
        if req.save and norm_cols:
            for wn, grp in norm_df.groupby("WELL_NAME"):
                _save_curves_to_well(project_id, str(wn), grp[["DEPTH"] + norm_cols])
                saved_wells += 1

        return {
            "project_id": project_id,
            "method": req.method,
            "normalized_curves": norm_cols,
            "normalized_columns": norm_cols,
            "saved": req.save,
            "saved_wells": saved_wells,
            "report": dataframe_to_records(report_df.round(4)),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Stratigraphic normalization failed: {e}"
        ) from e


@router.post(
    "/optimize_fwl",
    summary="Optimize Multi-Well Free Water Level (FWL) & Transition Zone across project wells",
)
def workspace_optimize_fwl(project_id: int, req: WorkspaceOptimizeFwlRequest) -> Dict[str, Any]:
    """Simultaneously optimize Free Water Level across all project wells and reconcile Sw_log vs Sw_SHM."""
    try:
        proj_summary = get_project_summary(project_id)
        well_names = [w["name"] for w in proj_summary.get("wells", [])]
        if not well_names:
            raise ValueError("No wells found in project.")

        dfs = []
        for wn in well_names:
            wdf = _get_merged_dataframe_sync(project_id, wn)
            if not wdf.empty:
                wdf = wdf.copy()
                wdf["WELL_NAME"] = wn
                dfs.append(wdf)
        if not dfs:
            raise ValueError("No log data available across project wells.")

        combined = pd.concat(dfs, ignore_index=True)
        res = optimize_fwl(
            df=combined,
            model=req.model,
            shm_params=req.shm_params,
            ift=req.ift,
            theta=req.theta,
            gw=req.gw,
            ghc=req.ghc,
            loss=req.loss,
            pore_volume_weighted=req.pore_volume_weighted,
            equal_well_weights=req.equal_well_weights,
            mdt_fwl=req.mdt_fwl,
            mdt_tolerance=req.mdt_tolerance,
            mdt_penalty_weight=req.mdt_penalty_weight,
            solve_per_well=req.solve_per_well,
            optimize_shm_params=req.optimize_shm_params,
            unit=req.unit,
        )

        saved_wells = 0
        if req.save and "SW_SHM" in res["df"].columns:
            for wn, grp in res["df"].groupby("WELL_NAME"):
                _save_curves_to_well(
                    project_id,
                    str(wn),
                    grp[["DEPTH", "SW_SHM"]].rename(columns={"SW_SHM": "SHF"}),
                )
                saved_wells += 1

        return {
            "project_id": project_id,
            "fwl": round(float(res["fwl"]), 2),
            "owc": round(float(res["owc"]), 2),
            "ttz": round(float(res["ttz"]), 2),
            "capillary_entry_height": round(float(res["capillary_entry_height"]), 2),
            "transition_zone_thickness": round(
                float(res["transition_zone_thickness"]), 2
            ),
            "rmse": round(float(res["rmse"]), 4),
            "mae": round(float(res["mae"]), 4),
            "r2": round(float(res["r2"]), 4),
            "bvw_rmse": round(float(res["bvw_rmse"]), 4),
            "mdt_fwl": res["mdt_fwl"],
            "fwl_reconciliation_diff": res["fwl_reconciliation_diff"],
            "shm_params": res["shm_params"],
            "well_metrics": dataframe_to_records(res["well_metrics"].round(4)),
            "grid_profile": dataframe_to_records(res["grid_profile"].round(4)),
            "saved": req.save,
            "saved_wells": saved_wells,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"FWL optimization failed: {e}"
        ) from e


class WorkspaceMlInferenceRequest(WorkspaceFilterBase):
    model_config_key: str = Field(
        "clastic",
        alias="model_config",
        description="Pre-trained model suite: 'clastic' or 'carbonate'",
    )
    alpha: float = Field(
        0.20,
        description="Conformal significance level (0.20 -> P10 to P90 interval)",
    )


@router.post(
    "/wells/{well_name}/ml_inference",
    summary="Run Pre-Trained PIML & Conformal Prediction inference (Read-Only Training)",
)
def workspace_ml_inference(
    project_id: int, well_name: str, req: WorkspaceMlInferenceRequest
) -> Dict[str, Any]:
    """Trigger pre-trained PIML & Conformal Prediction inference with P10–P90 uncertainty bands."""
    try:
        from quick_pp.machine_learning.predict_pipeline import predict_well_dataframe

        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        pred_df = predict_well_dataframe(
            df,
            model_config=req.model_config_key,
            alpha=float(req.alpha),
        )

        out_cols = [
            c
            for c in [
                "DEPTH",
                "PHIT",
                "PHIT_P10",
                "PHIT_P90",
                "PHIE",
                "PHIE_P10",
                "PHIE_P90",
                "SW",
                "SW_P10",
                "SW_P90",
                "SWT",
                "SWT_P10",
                "SWT_P90",
                "PERM",
                "PERM_P10",
                "PERM_P90",
                "LOG_PERM",
                "LOG_PERM_P10",
                "LOG_PERM_P90",
                "BVW",
                "BVHC",
                "BVHC_P10",
                "BVHC_P90",
                "VCLAY",
                "VSAND",
                "VSILT",
                "VCALC",
                "VDOLO",
                "OIL_FLAG",
                "GAS_FLAG",
            ]
            if c in pred_df.columns
        ]
        out_df = pred_df[out_cols].copy()

        saved_rows = 0
        if req.save:
            saved_rows = _save_curves_to_well(project_id, well_name, out_df)

        stats = {
            col: _curve_summary_stats(out_df[col])
            for col in out_cols
            if col != "DEPTH"
        }
        return {
            "well_name": well_name,
            "model_config": req.model_config_key,
            "alpha": float(req.alpha),
            "read_only_training": True,
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": stats,
            **dataframe_to_columnar(out_df),
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Pre-trained ML inference failed: {e}"
        ) from e


class WorkspaceWellConfigRequest(BaseModel):
    focus_min_depth: Optional[float] = Field(
        None, description="Persistent working top depth for this well"
    )
    focus_max_depth: Optional[float] = Field(
        None, description="Persistent working bottom depth for this well"
    )
    clear_focus_depth: bool = Field(
        False, description="Clear persistent working depth range back to full well"
    )
    extra: Optional[Dict[str, Any]] = Field(
        None, description="Additional per-well configuration parameters"
    )


@router.post(
    "/wells/{well_name}/config",
    summary="Save per-well persistent configuration such as working top/bottom depth interval",
)
@router.patch(
    "/wells/{well_name}/config",
    include_in_schema=False,
)
def workspace_update_well_config(
    project_id: int, well_name: str, req: WorkspaceWellConfigRequest
) -> Dict[str, Any]:
    """Persist working top and bottom focus depth (and optional well config keys) for a specific well."""
    try:
        from quick_pp.database import models as db_models
        from sqlalchemy import select
        from sqlalchemy.orm.attributes import flag_modified

        eph = dict(_ephemeral_well_configs.get((project_id, well_name), {}))
        if req.clear_focus_depth:
            eph["focus_min_depth"] = None
            eph["focus_max_depth"] = None
        else:
            if req.focus_min_depth is not None or req.focus_max_depth is not None:
                eph["focus_min_depth"] = req.focus_min_depth
                eph["focus_max_depth"] = req.focus_max_depth
        if req.extra:
            eph.update(req.extra)
        _ephemeral_well_configs[(project_id, well_name)] = eph

        saved_cfg = dict(eph)
        if not is_demo_mode():
            with get_db() as session:
                well_orm = session.scalar(
                    select(db_models.Well).filter_by(
                        project_id=project_id, name=well_name
                    )
                )
                if well_orm:
                    current_cfg = (
                        dict(well_orm.config_data)
                        if isinstance(well_orm.config_data, dict)
                        else {}
                    )
                    current_cfg.update(eph)
                    if req.clear_focus_depth:
                        current_cfg.pop("focus_min_depth", None)
                        current_cfg.pop("focus_max_depth", None)
                    well_orm.config_data = current_cfg
                    flag_modified(well_orm, "config_data")
                    session.add(well_orm)
                    session.flush()
                    saved_cfg = current_cfg

        return {
            "project_id": project_id,
            "well_name": well_name,
            "config": saved_cfg,
            "saved": True,
        }
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to update well configuration: {e}"
        ) from e


@router.post(
    "/demo/reset",
    summary="Reset ephemeral demo session edits back to pre-seeded Volve/Costa state",
)
def workspace_demo_reset(project_id: int) -> Dict[str, Any]:
    """Clear in-memory ephemeral session curves and uploaded LAS wells for the project."""
    _ephemeral_wells.pop(project_id, None)
    reset_demo_project_state(project_id)
    for k in [k for k in _ephemeral_well_configs if k[0] == project_id]:
        _ephemeral_well_configs.pop(k, None)
    cleared = invalidate_well_cache(project_id=project_id)
    return {
        "project_id": project_id,
        "status": "reset",
        "entries_cleared": cleared,
        "message": "Ephemeral session reset to pre-seeded Volve & Costa state.",
    }


class WorkspaceCurveEditRequest(WorkspaceFilterBase):
    action: str = Field(
        "transform",
        description="Action: 'transform', 'duplicate', 'rename', 'delete', or 'save_values'",
    )
    curve: str = Field(..., description="Source curve mnemonic")
    target_curve: Optional[str] = Field(
        None, description="Target curve mnemonic (defaults to source curve for in-place transform)"
    )
    operation: Optional[str] = Field(
        None,
        description="Transform operation: 'shift', 'scale', 'linear', 'clip', 'null_outside', 'depth_shift', 'smooth'",
    )
    value: float = Field(0.0, description="Shift offset, scale factor, or depth shift (m)")
    scale: float = Field(1.0, description="Multiplier for 'linear' (Y = scale * X + offset)")
    offset: float = Field(0.0, description="Offset for 'linear' (Y = scale * X + offset)")
    min_val: Optional[float] = Field(None, description="Minimum bound for 'clip' or 'null_outside'")
    max_val: Optional[float] = Field(None, description="Maximum bound for 'clip' or 'null_outside'")
    window: int = Field(5, description="Window size (samples) for 'smooth' median/moving average")
    depths: Optional[List[float]] = Field(
        None, description="Optional explicit depth array when action='save_values'"
    )
    values: Optional[List[Optional[float]]] = Field(
        None, description="Optional explicit curve values array when action='save_values'"
    )


def _update_cached_well_column(
    project_id: int,
    well_name: str,
    action: str,
    curve: str,
    target_curve: Optional[str] = None,
) -> None:
    """Apply rename or delete directly to in-memory cached / ephemeral DataFrames."""
    import time
    from quick_pp.database import models as db_models
    from sqlalchemy import select

    src = curve.strip().upper()
    dst = target_curve.strip().upper() if target_curve else None

    for key, (ts, cdf) in list(_merged_data_cache.items()):
        if key.startswith(f"{project_id}_{well_name}_"):
            updated = cdf.copy()
            if action == "delete" and src in updated.columns:
                updated = updated.drop(columns=[src], errors="ignore")
            elif action == "rename" and dst and src in updated.columns:
                if dst in updated.columns and dst != src:
                    updated = updated.drop(columns=[dst], errors="ignore")
                updated = updated.rename(columns={src: dst})
            _merged_data_cache[key] = (time.time(), updated)

    eph_proj = _ephemeral_wells.get(project_id, {})
    if well_name in eph_proj:
        updated_eph = eph_proj[well_name].copy()
        if action == "delete" and src in updated_eph.columns:
            updated_eph = updated_eph.drop(columns=[src], errors="ignore")
        elif action == "rename" and dst and src in updated_eph.columns:
            if dst in updated_eph.columns and dst != src:
                updated_eph = updated_eph.drop(columns=[dst], errors="ignore")
            updated_eph = updated_eph.rename(columns={src: dst})
        eph_proj[well_name] = updated_eph

    if not is_demo_mode():
        with get_db() as session:
            well_orm = session.scalar(
                select(db_models.Well).filter_by(project_id=project_id, name=well_name)
            )
            if well_orm:
                src_curve = session.scalar(
                    select(db_models.Curve).filter_by(
                        well_id=well_orm.well_id, mnemonic=src
                    )
                )
                if action == "delete" and src_curve:
                    session.delete(src_curve)
                    session.flush()
                elif action == "rename" and dst and src_curve:
                    existing_dst = session.scalar(
                        select(db_models.Curve).filter_by(
                            well_id=well_orm.well_id, mnemonic=dst
                        )
                    )
                    if existing_dst and existing_dst.curve_id != src_curve.curve_id:
                        session.delete(existing_dst)
                        session.flush()
                    src_curve.mnemonic = dst
                    session.flush()
        invalidate_well_cache(project_id=project_id, well_name=well_name)


@router.post(
    "/wells/{well_name}/curve_edit",
    summary="Edit, transform, duplicate, rename, or delete a well curve",
)
def workspace_curve_edit(
    project_id: int, well_name: str, req: WorkspaceCurveEditRequest
) -> Dict[str, Any]:
    """Perform curve arithmetic/conditioning (shift, scale, clip, depth-shift, smooth) or curve management (duplicate, rename, delete)."""
    try:
        act = req.action.strip().lower()
        src = req.curve.strip().upper()
        dst = (req.target_curve.strip().upper() if req.target_curve else src)

        protected = {"DEPTH", "DEPT", "WELL_NAME", "ZONES"}
        if act == "delete":
            if src in protected:
                raise ValueError(f"Cannot delete protected index column '{src}'")
            _update_cached_well_column(project_id, well_name, "delete", src)
            return {
                "well_name": well_name,
                "action": "delete",
                "deleted_curve": src,
                "saved": True,
            }

        if act == "rename":
            if src in protected or dst in protected:
                raise ValueError("Cannot rename to or from a protected index column")
            if not dst:
                raise ValueError("Target curve name is required for rename")
            _update_cached_well_column(project_id, well_name, "rename", src, dst)
            full_df = _get_merged_dataframe_sync(project_id, well_name)
            stats = (
                {dst: _curve_summary_stats(full_df[dst])}
                if dst in full_df.columns
                else {}
            )
            return {
                "well_name": well_name,
                "action": "rename",
                "curve": src,
                "target_curve": dst,
                "saved": True,
                "stats": stats,
            }

        if act == "save_values" and req.depths is not None and req.values is not None:
            if dst in protected:
                raise ValueError(f"Cannot overwrite protected column '{dst}'")
            out_df = pd.DataFrame(
                {
                    "DEPTH": pd.to_numeric(pd.Series(req.depths), errors="coerce"),
                    dst: pd.to_numeric(pd.Series(req.values), errors="coerce"),
                }
            ).dropna(subset=["DEPTH"])
            saved_rows = 0
            if req.save:
                saved_rows = _save_curves_to_well(project_id, well_name, out_df)
            return {
                "well_name": well_name,
                "action": "save_values",
                "target_curve": dst,
                "saved": req.save,
                "saved_rows": saved_rows,
                "stats": {dst: _curve_summary_stats(out_df[dst])},
                **dataframe_to_columnar(out_df),
            }

        full_df = _get_merged_dataframe_sync(project_id, well_name)
        if full_df.empty or src not in full_df.columns:
            raise ValueError(f"Curve '{src}' not found in well '{well_name}'")

        if act == "duplicate":
            if not dst or dst in protected:
                raise ValueError("Valid target curve mnemonic is required for duplicate")
            out_df = pd.DataFrame(
                {
                    "DEPTH": full_df["DEPTH"],
                    dst: pd.to_numeric(full_df[src], errors="coerce"),
                }
            )
            saved_rows = 0
            if req.save:
                saved_rows = _save_curves_to_well(project_id, well_name, out_df)
            return {
                "well_name": well_name,
                "action": "duplicate",
                "curve": src,
                "target_curve": dst,
                "saved": req.save,
                "saved_rows": saved_rows,
                "stats": {dst: _curve_summary_stats(out_df[dst])},
                **dataframe_to_columnar(out_df),
            }

        # Transform operation on full or filtered interval
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        s = pd.to_numeric(df[src], errors="coerce").copy()
        op = (req.operation or "shift").strip().lower()

        if op in ("shift", "add"):
            s = s + float(req.value)
        elif op == "subtract":
            s = s - float(req.value)
        elif op in ("scale", "multiply"):
            s = s * float(req.value)
        elif op == "divide":
            if abs(float(req.value)) < 1e-12:
                raise ValueError("Cannot divide by zero")
            s = s / float(req.value)
        elif op == "linear":
            s = s * float(req.scale) + float(req.offset)
        elif op == "clip":
            lo = float(req.min_val) if req.min_val is not None else None
            hi = float(req.max_val) if req.max_val is not None else None
            s = s.clip(lower=lo, upper=hi)
        elif op == "null_outside":
            if req.min_val is not None:
                s = s.where(s >= float(req.min_val), np.nan)
            if req.max_val is not None:
                s = s.where(s <= float(req.max_val), np.nan)
        elif op == "depth_shift":
            depth_arr = pd.to_numeric(df["DEPTH"], errors="coerce").to_numpy(dtype=float)
            val_arr = s.to_numpy(dtype=float)
            valid = np.isfinite(depth_arr) & np.isfinite(val_arr)
            if valid.sum() >= 2:
                shifted_depths = depth_arr[valid] + float(req.value)
                order = np.argsort(shifted_depths)
                interp_vals = np.interp(
                    depth_arr,
                    shifted_depths[order],
                    val_arr[valid][order],
                    left=np.nan,
                    right=np.nan,
                )
                s = pd.Series(interp_vals, index=df.index, dtype=float)
        elif op == "smooth":
            win = max(3, int(req.window))
            s = s.rolling(window=win, center=True, min_periods=1).mean()
        else:
            raise ValueError(f"Unsupported curve transform operation: {op}")

        out_df = pd.DataFrame({"DEPTH": df["DEPTH"], dst: s})
        saved_rows = 0
        if req.save:
            saved_rows = _save_curves_to_well(project_id, well_name, out_df)

        return {
            "well_name": well_name,
            "action": "transform",
            "operation": op,
            "curve": src,
            "target_curve": dst,
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": {dst: _curve_summary_stats(s)},
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Curve edit failed: {e}"
        ) from e


