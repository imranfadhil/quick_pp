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
    _ephemeral_well_headers,
    _ephemeral_wells,
    _get_merged_dataframe_sync,
    _merged_data_cache,
    _sanitize_for_json,
    dataframe_to_columnar,
    dataframe_to_records,
    filter_merged_dataframe,
    get_project_summary,
    invalidate_well_cache,
    reset_demo_project_state,
    resolve_db_well_name,
)
from quick_pp.las_handler import extract_las_metadata
from quick_pp.app.backend.services.rock_typing import _compute_fzi_data_sync
from quick_pp.app.backend.utils.auth import is_demo_mode
from quick_pp.app.backend.utils.db import get_db
from quick_pp.core_analysis import (
    auto_cuddy_params,
    auto_j_params,
    auto_skelt_harrison_params,
    cuddy_func,
    fit_poroperm_curve,
    inv_power_law_func,
    leverett_j,
    normalize_sw,
    skelt_harrison_func,
    sw_shf_cuddy,
    sw_shf_leverett_j,
    sw_skelt_harrison,
)
from quick_pp.config import Config
from quick_pp.database import objects as db_objects
from quick_pp.lithology import gr_index
from quick_pp.lithology.multi_mineral import MultiMineral
from quick_pp.lithology.sand_silt_clay import SandSiltClay
from quick_pp.lithology.thin_beds import ThinBeds
from quick_pp.machine_learning.feature_engineering import coal_flagging
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
    badhole_flagging,
    den_correction,
    handle_outer_limit,
    neu_den_xplot_hc_correction,
    stratigraphic_normalization,
)
from quick_pp.ressum import calc_reservoir_summary
from quick_pp.rock_type import calc_fzi, find_cutoffs, rock_typing
from quick_pp.saturation import (
    archie_saturation,
    estimate_b_waxman_smits,
    estimate_qv,
    estimate_qvn,
    estimate_rw_temperature_salinity,
    optimize_fwl,
    waxman_smits_saturation,
)
from quick_pp.uncertainty import calc_stochastic_reservoir_summary
from quick_pp.upscaling import expand_blocked_logs, upscale_logs
from quick_pp.utils import length_a_b, line_intersection

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
            upd_cols = [
                c
                for c in curves_df.columns
                if str(c).upper() not in ("DEPTH", "WELL_NAME")
            ]
            if not upd_cols:
                return

            # Fast path when curves_df shares base_df's index or exact full-well depth grid
            if len(curves_df) == len(base_df) and np.allclose(
                curves_df["DEPTH"].to_numpy(dtype=float),
                base_df["DEPTH"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                merged_mem = base_df.copy()
                for c in upd_cols:
                    merged_mem[str(c).upper()] = curves_df[c].to_numpy()
            elif curves_df.index.isin(base_df.index).all() and np.allclose(
                curves_df["DEPTH"].to_numpy(dtype=float),
                base_df.loc[curves_df.index, "DEPTH"].to_numpy(dtype=float),
                equal_nan=True,
            ):
                merged_mem = base_df.copy()
                for c in upd_cols:
                    ucol = str(c).upper()
                    if ucol not in merged_mem.columns:
                        merged_mem[ucol] = np.nan
                    merged_mem.loc[curves_df.index, ucol] = curves_df[c].to_numpy()
            else:
                upd = curves_df.drop(columns=["WELL_NAME"], errors="ignore").copy()
                upd.columns = [str(c).upper() for c in upd.columns]
                upd = upd.drop_duplicates(subset=["DEPTH"], keep="last")
                overlap_cols = [
                    c for c in upd.columns if c != "DEPTH" and c in base_df.columns
                ]
                upd["_IN_UPD"] = True
                merged_mem = base_df.drop(columns=overlap_cols, errors="ignore").merge(
                    upd, on="DEPTH", how="left"
                )
                in_upd_mask = merged_mem["_IN_UPD"].fillna(False).to_numpy(dtype=bool)
                merged_mem.drop(columns=["_IN_UPD"], errors="ignore", inplace=True)
                for c in overlap_cols:
                    if c in base_df.columns:
                        merged_mem[c] = np.where(in_upd_mask, merged_mem[c], base_df[c])
            _merged_data_cache[f"{project_id}_{well_name}_0.16"] = (
                time.time(),
                merged_mem,
            )
            if (
                project_id in _ephemeral_wells
                and well_name in _ephemeral_wells[project_id]
            ):
                _ephemeral_wells[project_id][well_name]["data"] = merged_mem
    except Exception:
        pass


def _save_curves_to_well(
    project_id: int, well_name: str, curves_df: pd.DataFrame
) -> int:
    """Persist a DataFrame containing DEPTH and one or more computed curve columns to the database."""
    if curves_df is None or curves_df.empty or "DEPTH" not in curves_df.columns:
        return 0
    df_to_save = curves_df.copy()
    db_well_name = resolve_db_well_name(project_id, well_name)
    df_to_save["WELL_NAME"] = db_well_name

    _update_in_memory_well_cache(project_id, well_name, df_to_save)

    if is_demo_mode():
        return len(df_to_save)

    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)
        proj.update_data(df_to_save, group_by="WELL_NAME")
        proj.save()

    invalidate_well_cache(project_id=project_id, well_name=well_name)
    _update_in_memory_well_cache(project_id, well_name, df_to_save)
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
        raise ValueError(
            f"No data found for well '{well_name}' in project {project_id}"
        )
    filtered = filter_merged_dataframe(
        merged, min_depth=min_depth, max_depth=max_depth, zones=zones
    )
    if filtered.empty:
        raise ValueError(
            f"No data rows remain for well '{well_name}' after applying depth/zone filters"
        )
    return filtered


def _curve_summary_stats(
    series: pd.Series, depths: Optional[pd.Series] = None
) -> Dict[str, Optional[float]]:
    """Return compact summary statistics (top, bottom, min, max, mean, std, p10, p50, p90, valid_count) for a numeric Series."""
    s_num = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid_mask = s_num.notna()
    s = s_num[valid_mask]
    if s.empty:
        return {
            "count": 0,
            "top": None,
            "bottom": None,
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "p10": None,
            "p50": None,
            "p90": None,
        }
    top_val: Optional[float] = None
    bottom_val: Optional[float] = None
    if depths is not None:
        d_num = pd.to_numeric(depths, errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        d_valid = d_num[valid_mask & d_num.notna()]
        if not d_valid.empty:
            top_val = round(float(d_valid.min()), 4)
            bottom_val = round(float(d_valid.max()), 4)
    return {
        "count": int(len(s)),
        "top": top_val,
        "bottom": bottom_val,
        "min": round(float(s.min()), 6),
        "max": round(float(s.max()), 6),
        "mean": round(float(s.mean()), 6),
        "std": round(float(s.std()), 6) if len(s) > 1 else 0.0,
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
        if (
            use_hc_corrected and "NPHI_HC" in df.columns and df["NPHI_HC"].notna().any()
        )
        else "NPHI"
    )
    rhob_col = (
        "RHOB_HC"
        if (
            use_hc_corrected and "RHOB_HC" in df.columns and df["RHOB_HC"].notna().any()
        )
        else "RHOB"
    )
    if nphi_col not in df.columns or rhob_col not in df.columns:
        raise ValueError(
            f"Required curves NPHI and RHOB not found in well data. Available: {list(df.columns)}"
        )
    return pd.to_numeric(df[nphi_col], errors="coerce"), pd.to_numeric(
        df[rhob_col], errors="coerce"
    )


def _compute_coal_mask(
    df: pd.DataFrame,
    enable_coal_flagging: bool = False,
    coal_rhob_max: float = 2.0,
    coal_nphi_min: float = 0.30,
    coal_window_size: int = 41,
    coal_trend_factor: float = 0.05,
) -> pd.Series:
    """Compute boolean mask for coal intervals using raw NPHI and RHOB logs."""
    if not enable_coal_flagging:
        return pd.Series(False, index=df.index, dtype=bool)
    if "NPHI" not in df.columns or "RHOB" not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    nphi_raw = pd.to_numeric(df["NPHI"], errors="coerce")
    rhob_raw = pd.to_numeric(df["RHOB"], errors="coerce")
    valid = nphi_raw.notna() & rhob_raw.notna()
    if not valid.any():
        return pd.Series(False, index=df.index, dtype=bool)

    tf = max(float(coal_trend_factor), 0.0)
    win = max(int(coal_window_size), 3)
    if tf > 1e-6:
        flagged = coal_flagging(
            nphi_raw,
            rhob_raw,
            rhob_threshold=float(coal_rhob_max),
            nphi_threshold=float(coal_nphi_min),
            window_size=win,
            trend_factor=tf,
        )
        # Also include unequivocal thick coal beds that may exceed the rolling window width
        extreme_coal = (rhob_raw < min(float(coal_rhob_max) - 0.12, 1.85)) & (
            nphi_raw > max(float(coal_nphi_min) + 0.05, 0.38)
        )
        return ((flagged >= 0.5) | extreme_coal).fillna(False)
    else:
        return (
            (rhob_raw < float(coal_rhob_max)) & (nphi_raw > float(coal_nphi_min))
        ).fillna(False)


def _resolve_badhole_mask(
    df: pd.DataFrame, auto_detect_if_missing: bool = False
) -> pd.Series:
    """Resolve boolean mask for badhole intervals from BADHOLE/BAD_HOLE/BAD_HOLE_FLAG or on-the-fly detection."""
    for col in ("BADHOLE", "BAD_HOLE", "BAD_HOLE_FLAG"):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return (s.fillna(0.0) >= 0.5).astype(bool)
    if auto_detect_if_missing:
        try:
            bh_df = badhole_flagging(
                df,
                thold=2.0,
                use_cali=True,
                auto_bitsize=True,
                use_drho=True,
                drho_threshold=0.15,
            )
            if "BADHOLE" in bh_df.columns:
                return (
                    pd.to_numeric(bh_df["BADHOLE"], errors="coerce")
                    .reindex(df.index)
                    .fillna(0.0)
                    >= 0.5
                ).astype(bool)
        except Exception:
            pass
    return pd.Series(False, index=df.index, dtype=bool)


def _compute_vsh_gr_series(
    df: pd.DataFrame,
    gr_clean: Optional[float] = None,
    gr_clay: Optional[float] = None,
) -> pd.Series:
    """Compute calibrated VSH_GR series in [0, 1] from GR endpoints, existing VSH_GR curve, or auto-normalized gr_index."""
    out = pd.Series(np.nan, index=df.index, dtype=float)
    if (
        "GR" in df.columns
        and gr_clean is not None
        and gr_clay is not None
        and abs(float(gr_clay) - float(gr_clean)) > 1e-6
    ):
        gr_s = pd.to_numeric(df["GR"], errors="coerce")
        valid_gr = gr_s.notna()
        if valid_gr.any():
            out.loc[valid_gr] = (
                (gr_s.loc[valid_gr] - float(gr_clean))
                / (float(gr_clay) - float(gr_clean))
            ).clip(0.0, 1.0)
            return out
    if "VSH_GR" in df.columns:
        vsh_existing = pd.to_numeric(df["VSH_GR"], errors="coerce")
        if vsh_existing.notna().any():
            return vsh_existing.clip(0.0, 1.0)
    if "GR" not in df.columns:
        return out
    gr_s = pd.to_numeric(df["GR"], errors="coerce")
    valid_gr = gr_s.notna()
    if not valid_gr.any():
        return out
    vsh_arr = np.asarray(
        gr_index(gr_s.loc[valid_gr].reset_index(drop=True)), dtype=float
    ).ravel()
    out.loc[valid_gr] = np.clip(vsh_arr, 0.0, 1.0)
    return out


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


class WorkspaceBadholeRequest(WorkspaceFilterBase):
    use_cali: bool = Field(
        True, description="Flag badhole using Differential Caliper |CALI - BS|"
    )
    cali_threshold: float = Field(
        2.0, description="Max allowable |CALI - BS| before flagging badhole (inches)"
    )
    auto_bitsize: bool = Field(
        True,
        description="Auto-estimate bit size per hole section from CALI when BS is absent",
    )
    bitsize_default: float = Field(
        8.5, description="Nominal/fallback bit size (inches)"
    )
    use_drho: bool = Field(
        True, description="Flag badhole using Density Correction |DRHO|"
    )
    drho_threshold: float = Field(
        0.15, description="Max allowable |DRHO| before flagging badhole (g/cc)"
    )
    use_rhob_nphi: bool = Field(
        False, description="Flag badhole using RHOB/NPHI pad standoff bounds"
    )
    rhob_min: float = Field(1.95, description="Min valid bulk density (g/cc)")
    rhob_max: float = Field(2.95, description="Max valid bulk density (g/cc)")
    nphi_max: float = Field(0.45, description="Max valid neutron porosity (v/v)")
    exclude_coal: bool = Field(
        True, description="Exclude coal seams from RHOB/NPHI bound and rugosity flags"
    )
    use_rugosity: bool = Field(
        False,
        description="Flag local borehole rugosity spikes from rolling RHOB/CALI trends",
    )
    rugosity_window: int = Field(
        21, description="Rolling window size (samples) for rugosity spike detection"
    )
    rhob_trend_factor: float = Field(
        0.05,
        description="Fractional drop below rolling RHOB median to flag washout spike",
    )
    cali_spike_threshold: float = Field(
        0.5, description="Local caliper spike deviation from rolling median (inches)"
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
    gr_clean: Optional[float] = Field(None, description="Clean sand GR endpoint (API)")
    gr_clay: Optional[float] = Field(None, description="Shale GR endpoint (API)")
    enable_coal_flagging: bool = Field(
        False, description="Flag coal beds and exclude them from HC shift"
    )
    coal_rhob_max: float = Field(2.0, description="Max bulk density for coal (g/cc)")
    coal_nphi_min: float = Field(
        0.30, description="Min neutron porosity for coal (v/v)"
    )
    coal_window_size: int = Field(41, description="Rolling window size for coal trend")
    coal_trend_factor: float = Field(
        0.05, description="Rolling trend deviation factor for coal (0 = threshold only)"
    )
    badhole_treatment: str = Field(
        "none",
        description="How to treat BADHOLE intervals: 'none', 'vsh_gr', 'interpolate', or 'ignore'",
    )


class WorkspaceDenCorrectionRequest(WorkspaceFilterBase):
    dry_sand_point: Tuple[float, float] = Field((-0.02, 2.65))
    dry_clay_point: Tuple[float, float] = Field((0.33, 2.7))
    rho_water: float = Field(1.0, description="Water density (g/cc)")
    sync_endpoints: bool = Field(
        True,
        description="Synchronize rho_ma, phin_sh, and phid_sh with dry_sand_point and dry_clay_point",
    )
    rho_ma: float = Field(2.65, description="Matrix density override (g/cc)")
    phin_sh: float = Field(0.33, description="Shale neutron porosity override (v/v)")
    phid_sh: float = Field(0.05, description="Shale density porosity override (v/v)")
    scope: str = Field(
        "vsh_mismatch",
        description="Correction scope: 'vsh_mismatch', 'badhole_only', or 'full'",
    )
    vsh_mismatch_threshold: float = Field(
        0.15,
        description="Min |VSH_DN - VSH_GR| to trigger density reconstruction when scope=='vsh_mismatch'",
    )
    gr_clean: Optional[float] = Field(None, description="Clean sand GR endpoint (API)")
    gr_clay: Optional[float] = Field(None, description="Shale GR endpoint (API)")
    enable_coal_flagging: bool = Field(
        False, description="Flag coal beds and exclude them from density reconstruction"
    )
    coal_rhob_max: float = Field(2.0, description="Max bulk density for coal (g/cc)")
    coal_nphi_min: float = Field(
        0.30, description="Min neutron porosity for coal (v/v)"
    )
    coal_window_size: int = Field(41, description="Rolling window size for coal trend")
    coal_trend_factor: float = Field(
        0.05, description="Rolling trend deviation factor for coal (0 = threshold only)"
    )


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
    enable_coal_flagging: bool = Field(
        False, description="Flag coal intervals and assign VCOAL=1, other minerals=0"
    )
    coal_rhob_max: float = Field(2.0, description="Max bulk density for coal (g/cc)")
    coal_nphi_min: float = Field(
        0.30, description="Min neutron porosity for coal (v/v)"
    )
    coal_window_size: int = Field(41, description="Rolling window size for coal trend")
    coal_trend_factor: float = Field(
        0.05, description="Rolling trend deviation factor for coal (0 = threshold only)"
    )
    badhole_treatment: str = Field(
        "none",
        description="How to treat BADHOLE intervals: 'none', 'vsh_gr', 'interpolate', or 'ignore'",
    )


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
    gr_clean: Optional[float] = Field(None, description="Clean sand GR endpoint (API)")
    gr_clay: Optional[float] = Field(None, description="Shale GR endpoint (API)")
    enable_coal_flagging: bool = Field(
        False, description="Zero out PHIT and PHIE in flagged coal intervals"
    )
    coal_rhob_max: float = Field(2.0, description="Max bulk density for coal (g/cc)")
    coal_nphi_min: float = Field(
        0.30, description="Min neutron porosity for coal (v/v)"
    )
    coal_window_size: int = Field(41, description="Rolling window size for coal trend")
    coal_trend_factor: float = Field(
        0.05, description="Rolling trend deviation factor for coal (0 = threshold only)"
    )
    badhole_treatment: str = Field(
        "none",
        description="How to treat BADHOLE intervals: 'none', 'vsh_gr', 'interpolate', or 'ignore'",
    )


class WorkspaceThomasStieberRequest(WorkspaceFilterBase):
    use_hc_corrected: bool = Field(
        True, description="Use NPHI_HC and RHOB_HC curves if present"
    )
    use_existing_phit_vsh: bool = Field(
        True,
        description="Use existing evaluated PHIT and VCLAY/VSHALE curves if present",
    )
    dry_sand_poro: float = Field(0.28, description="Clean sand endpoint porosity (v/v)")
    dry_shale_poro: float = Field(
        0.10, description="Pure shale endpoint porosity (v/v)"
    )
    dry_sand_point: Tuple[float, float] = Field((-0.02, 2.65))
    dry_clay_point: Tuple[float, float] = Field((0.33, 2.7))
    fluid_point: Tuple[float, float] = Field((1.0, 1.0))
    rsand: float = Field(
        20.0, description="Hagiwara sand lamina resistivity Rsand (ohm.m)"
    )
    rv_shale: float = Field(
        4.0, description="Hagiwara vertical shale resistivity Rv_shale (ohm.m)"
    )
    rh_shale: float = Field(
        1.5, description="Hagiwara horizontal shale resistivity Rh_shale (ohm.m)"
    )
    theta: float = Field(
        0.0, description="Relative dip / wellbore deviation angle theta (degrees)"
    )


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
    depths: Optional[List[float]] = Field(
        None, description="Optional client-side depth array fallback"
    )
    phit: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side PHIT array fallback"
    )
    vclay: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VCLAY array fallback"
    )
    vsilt: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VSILT array fallback"
    )
    use_thomas_stieber: bool = Field(
        False,
        description="Use Thomas-Stieber sand-lamina porosity (PHIT_SAND) and dispersed clay (VSH_DIS) weighted by (1 - VSH_LAM)",
    )
    phit_sand: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side PHIT_SAND array fallback"
    )
    vsh_lam: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VSH_LAM array fallback"
    )
    vsh_dis: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VSH_DIS array fallback"
    )


class WorkspaceSaturationRequest(WorkspaceFilterBase):
    method: str = Field(
        "waxman_smits",
        description="Primary saturation curve to save/highlight: 'waxman_smits' or 'archie'",
    )
    rw_mode: str = Field(
        "salinity_temp",
        description="Rw source mode: 'salinity_temp' (from salinity & geothermal gradient) or 'pickett' (constant Rw override)",
    )
    meas_system: str = Field("metric", description="'metric' or 'imperial'")
    water_salinity: float = Field(35000.0, description="Formation water salinity (ppm)")
    temp_gradient: float = Field(
        30.0, description="Geothermal temperature gradient (°C/km or °F/kft)"
    )
    surface_temp: float = Field(25.0, description="Surface temperature (°C or °F)")
    m: float = Field(2.0, description="Archie/Waxman-Smits cementation exponent m")
    a: float = Field(1.0, description="Tortuosity factor a")
    n: float = Field(2.0, description="Saturation exponent n")
    m_slope: float = Field(
        0.0, description="Optional Waxman-Smits m* slope vs VCLAY/PHIT"
    )
    rw: Optional[float] = Field(None, description="Explicit Rw override (ohm.m)")
    b_intercept: Optional[float] = Field(
        None, description="Optional Waxman-Smits m* intercept override (defaults to m)"
    )
    qv_method: str = Field(
        "cec",
        description="Qv estimation method: 'cec' (from cec_clay & rho_dry_clay) or 'qvn' (Juhasz normalized Qvn)",
    )
    cec_clay: float = Field(0.1, description="Clay cation exchange capacity (meq/g)")
    rho_dry_clay: float = Field(2.71, description="Dry clay bulk density (g/cc)")
    depths: Optional[List[float]] = Field(
        None, description="Optional client-side depth array fallback"
    )
    phit: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side PHIT array fallback"
    )
    vclay: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VCLAY array fallback"
    )
    use_thomas_stieber: bool = Field(
        False,
        description="Use Thomas-Stieber thin-bed inputs (PHIT_SAND, VSH_DIS, RSAND_TS) for saturation evaluation",
    )
    phit_sand: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side PHIT_SAND array fallback"
    )
    vsh_lam: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VSH_LAM array fallback"
    )
    vsh_dis: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side VSH_DIS array fallback"
    )
    rsand_ts: Optional[List[Optional[float]]] = Field(
        None, description="Optional client-side RSAND_TS array fallback"
    )


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
    well_names: Optional[List[str]] = Field(
        None, description="Optional subset of well names to include"
    )
    cutoffs: List[float] = Field(default_factory=lambda: [0.5, 1.0, 2.5])
    higher_is_better: bool = Field(True)
    auto_find_cutoffs: bool = Field(False)
    no_of_cutoffs: int = Field(3)
    save: bool = Field(False)


class WorkspacePermTransformRequest(BaseModel):
    well_names: Optional[List[str]] = Field(
        None, description="Optional subset of well names to include"
    )
    cutoffs: List[float] = Field(default_factory=lambda: [0.5, 1.2, 2.5, 5.0])
    fits: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Optional manual (a, b) fits per ROCK_FLAG. If None, auto-fitted from well data.",
    )
    save: bool = Field(False)


class WorkspaceShfRequest(BaseModel):
    well_names: Optional[List[str]] = Field(
        None, description="Optional subset of well names to include"
    )
    model: str = Field(
        "leverett_j",
        description="SHF model: 'leverett_j', 'skelt_harrison', or 'cuddy'",
    )
    unit: str = Field("metric", description="Depth unit: 'metric' or 'imperial'")
    cutoffs: List[float] = Field(default_factory=lambda: [0.1, 1.0, 3.0])
    fwl: float = Field(1000.0)
    fwl_by_zone: Optional[Dict[str, float]] = Field(
        None,
        description="Optional per-zone FWL mapping (zone -> FWL depth). Overrides global `fwl` per zone.",
    )
    ift: float = Field(30.0)
    theta: float = Field(30.0)
    gw: float = Field(1.05)
    ghc: float = Field(0.8)
    leverett_a: Optional[float] = Field(
        None, description="Optional global override for 'a' parameter"
    )
    leverett_b: Optional[float] = Field(
        None, description="Optional global override for 'b' parameter"
    )
    fits: Optional[Dict[str, Dict[str, float]]] = Field(
        None,
        description="Optional SHF curve fits per ROCK_FLAG. If None, auto-fitted from core SCAL data.",
    )
    save: bool = Field(False)


def _filter_fzi_data_by_wells(
    fzi_data: Dict[str, List[Any]], well_names: Optional[List[str]]
) -> Dict[str, List[Any]]:
    if not well_names:
        return fzi_data
    allowed = {str(w).strip() for w in well_names if str(w).strip()}
    if not allowed:
        return fzi_data
    w_arr = fzi_data.get("well_names") or []
    keep_idx = [i for i, wn in enumerate(w_arr) if str(wn) in allowed]
    out: Dict[str, List[Any]] = {}
    for k, arr in fzi_data.items():
        if isinstance(arr, list) and len(arr) == len(w_arr):
            out[k] = [arr[i] for i in keep_idx]
        else:
            out[k] = arr
    return out


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
        zone_list = (
            [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        )
        df = _get_filtered_well_df(
            project_id,
            well_name,
            min_depth=min_depth,
            max_depth=max_depth,
            zones=zone_list,
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
        zone_list = (
            [z.strip() for z in zones.split(",") if z.strip()] if zones else None
        )
        df = _get_filtered_well_df(
            project_id,
            well_name,
            min_depth=min_depth,
            max_depth=max_depth,
            zones=zone_list,
        )
        col_payload = dataframe_to_columnar(df)
        curves_map = col_payload.get("data", {})
        if "DEPTH" in curves_map and "DEPT" not in curves_map:
            curves_map["DEPT"] = curves_map["DEPTH"]

        # Map standard aliases onto canonical display mnemonics (RT, RM, RS, PEF, DRHO) if canonical key is absent
        alias_groups = {
            "RT": ("RD", "RDEP", "RDEEP", "LLD", "RILD"),
            "RM": ("RMED", "RILM", "ILM", "LLM", "LL8", "M2R6"),
            "RS": (
                "RSHAL",
                "RSHA",
                "RXO",
                "RXOZ",
                "MSFL",
                "RMLL",
                "RFOC",
                "SFL",
                "LLS",
            ),
            "PEF": ("PE", "PEFZ", "PEFA"),
            "DRHO": ("HDRA", "DCOR", "RHOC"),
        }
        for canonical, aliases in alias_groups.items():
            if canonical not in curves_map or not any(
                v is not None for v in curves_map.get(canonical, [])
            ):
                for alias in aliases:
                    if alias in curves_map and any(
                        v is not None for v in curves_map[alias]
                    ):
                        curves_map[canonical] = curves_map[alias]
                        break

        depth_s = (
            pd.to_numeric(df["DEPTH"], errors="coerce")
            if "DEPTH" in df.columns
            else None
        )
        stats: Dict[str, Any] = {}
        for col in df.columns:
            if col in ("WELL_NAME", "ZONES"):
                continue
            s = pd.to_numeric(df[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            valid_m = s.notna()
            if valid_m.any():
                clean = s[valid_m]
                top_d = None
                bot_d = None
                if depth_s is not None:
                    d_clean = depth_s[valid_m & depth_s.notna()]
                    if not d_clean.empty:
                        top_d = round(float(d_clean.min()), 2)
                        bot_d = round(float(d_clean.max()), 2)
                stats[col] = {
                    "count": int(len(clean)),
                    "top": top_d,
                    "bottom": bot_d,
                    "min": round(float(clean.min()), 4),
                    "max": round(float(clean.max()), 4),
                    "mean": round(float(clean.mean()), 4),
                    "std": round(float(clean.std()), 4) if len(clean) > 1 else 0.0,
                    "p10": round(float(np.percentile(clean, 10)), 4),
                    "p50": round(float(np.percentile(clean, 50)), 4),
                    "p90": round(float(np.percentile(clean, 90)), 4),
                }
        for canonical, aliases in alias_groups.items():
            if canonical not in stats and canonical in curves_map:
                for alias in aliases:
                    if alias in stats:
                        stats[canonical] = dict(stats[alias])
                        break

        ancillary: Dict[str, Any] = {}
        well_config: Dict[str, Any] = {}
        raw_db_header: Dict[str, Any] = {}
        db_uwi: Optional[str] = None
        db_depth_uom: Optional[str] = None
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
                if well_obj._orm_well:
                    db_uwi = well_obj._orm_well.uwi
                    db_depth_uom = well_obj._orm_well.depth_uom
                    if isinstance(well_obj._orm_well.config_data, dict):
                        well_config = {**well_config, **well_obj._orm_well.config_data}
                    if isinstance(well_obj._orm_well.header_data, dict):
                        raw_db_header = dict(well_obj._orm_well.header_data)
        except Exception:
            pass
        eph_cfg = _ephemeral_well_configs.get(
            (project_id, well_name)
        ) or _ephemeral_well_configs.get(
            (project_id, resolve_db_well_name(project_id, well_name))
        )
        if eph_cfg:
            well_config = {**well_config, **eph_cfg}

        eph_hdr = _ephemeral_well_headers.get(
            (project_id, well_name)
        ) or _ephemeral_well_headers.get(
            (project_id, resolve_db_well_name(project_id, well_name))
        )
        base_hdr_source = raw_db_header if raw_db_header else (eph_hdr or {})
        well_header: Dict[str, Any] = dict(
            extract_las_metadata(base_hdr_source, fallback_well_name=well_name)
        )
        well_header["well_name"] = well_name
        if db_uwi and (
            not well_header.get("uwi")
            or well_header.get("uwi") == "UNKNOWN_WELL"
        ):
            well_header["uwi"] = db_uwi
        if db_depth_uom and not well_header.get("datum_unit"):
            well_header["datum_unit"] = db_depth_uom
            well_header["depth_unit"] = db_depth_uom

        # Apply any explicit user overrides saved on header_data or ephemeral headers
        hdr_overrides = (
            raw_db_header.get("header_overrides")
            if isinstance(raw_db_header.get("header_overrides"), dict)
            else {}
        )
        for ovr_src in (hdr_overrides, eph_hdr or {}):
            if isinstance(ovr_src, dict):
                for hk, hv in ovr_src.items():
                    if hv is not None and hk not in (
                        "raw_well_header",
                        "raw_parameters",
                    ):
                        well_header[hk] = hv

        if depth_s is not None:
            d_valid_all = depth_s.dropna()
            if not d_valid_all.empty:
                if well_header.get("start_depth") is None:
                    well_header["start_depth"] = round(float(d_valid_all.min()), 4)
                if well_header.get("stop_depth") is None:
                    well_header["stop_depth"] = round(float(d_valid_all.max()), 4)
                if well_header.get("step") is None and len(d_valid_all) > 1:
                    diffs = d_valid_all.diff().dropna()
                    diffs = diffs[diffs > 0]
                    if not diffs.empty:
                        well_header["step"] = round(float(diffs.median()), 4)

        # Ensure well_config also reflects resolved well_header fields
        for sync_k in (
            "uwi",
            "well_name",
            "field",
            "company",
            "service_company",
            "country",
            "province",
            "location",
            "spud_date",
            "log_date",
            "latitude",
            "longitude",
            "x_coord",
            "y_coord",
            "crs",
            "kb_elev",
            "df_elev",
            "gl_elev",
            "water_depth",
            "datum_unit",
            "bht",
            "bht_unit",
            "surface_temp",
            "bit_size",
            "rmf",
            "rmf_temp",
            "rmc",
            "mud_density",
        ):
            if well_header.get(sync_k) is not None:
                well_config[sync_k] = well_header[sync_k]

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
        scal_pc_list: List[Dict[str, Any]] = []
        core_data_raw = ancillary.get("core_data")
        if isinstance(core_data_raw, dict):
            for sname, sdict in core_data_raw.items():
                s_depth = float(sdict.get("depth", 0.0))
                meas_df = sdict.get("measurements")
                poro_val = None
                perm_val = None
                if isinstance(meas_df, pd.DataFrame) and not meas_df.empty:
                    for _, mrow in meas_df.iterrows():
                        prop = str(mrow.get("property", "")).upper()
                        if (
                            prop in ("POR", "CPORE", "POROSITY")
                            and mrow.get("value") is not None
                        ):
                            poro_val = float(mrow["value"])
                        elif (
                            prop in ("PERM", "CPERM", "PERMEABILITY")
                            and mrow.get("value") is not None
                        ):
                            perm_val = float(mrow["value"])

                pc_df = sdict.get("pc")
                has_pc = isinstance(pc_df, pd.DataFrame) and not pc_df.empty
                if has_pc:
                    for _, prow in pc_df.iterrows():
                        pc_v = prow.get("pressure", prow.get("pc"))
                        sw_v = prow.get("saturation", prow.get("sw"))
                        if (
                            pc_v is None
                            or sw_v is None
                            or pd.isna(pc_v)
                            or pd.isna(sw_v)
                        ):
                            continue
                        scal_pc_list.append(
                            {
                                "sample_name": str(sname),
                                "depth": s_depth,
                                "pc": float(pc_v),
                                "sw": float(sw_v),
                                "experiment_type": str(
                                    prow.get("type")
                                    or prow.get("experiment_type")
                                    or "micp"
                                ),
                                "cycle": str(prow.get("cycle") or "drainage"),
                            }
                        )

                # Include in RCA list if it has RCA measurements or if it's an empty sample stub without SCAL Pc
                if poro_val is not None or perm_val is not None or not has_pc:
                    core_samples_list.append(
                        {
                            "sample_name": str(sname),
                            "depth": s_depth,
                            "porosity": poro_val,
                            "permeability": perm_val,
                        }
                    )

        core_samples_list.sort(key=lambda r: (r["depth"], r["sample_name"]))
        scal_pc_list.sort(key=lambda r: (r["depth"], r["sample_name"], r["pc"]))

        return _sanitize_for_json({
            "project_id": project_id,
            "well_name": well_name,
            "row_count": col_payload.get("row_count", 0),
            "columns": col_payload.get("columns", []),
            "curves": curves_map,
            "stats": stats,
            "tops": tops,
            "contacts": contacts,
            "core_samples": core_samples_list,
            "scal_pc": scal_pc_list,
            "pressure_tests": _df_or_list(ancillary.get("pressure_tests")),
            "survey": _df_or_list(
                ancillary.get("well_surveys")
                if "well_surveys" in ancillary
                else ancillary.get("directional_surveys")
            ),
            "header": well_header,
            "config": {k: v for k, v in well_config.items() if not k.startswith("_")},
        })
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
            raise ValueError(
                "No valid NPHI and RHOB pairs found in the selected interval"
            )

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

        vsh_gr_s = _compute_vsh_gr_series(df, req.gr_clean, req.gr_clay)
        vsh_gr_arr: Optional[np.ndarray] = (
            vsh_gr_s.loc[valid_mask].fillna(0.0).to_numpy(dtype=float)
            if vsh_gr_s.notna().any()
            else None
        )

        nphi_c, rhob_c, hc_flag = neu_den_xplot_hc_correction(
            nphi[valid_mask].to_numpy(dtype=float),
            rhob[valid_mask].to_numpy(dtype=float),
            vsh_gr=vsh_gr_arr,
            dry_min1_point=req.dry_sand_point,
            dry_clay_point=req.dry_clay_point,
            water_point=req.water_point,
            corr_angle=corr_angle,
            buffer=req.buffer,
        )

        nphi_hc_full.loc[valid_mask] = np.asarray(nphi_c, dtype=float)
        rhob_hc_full.loc[valid_mask] = np.asarray(rhob_c, dtype=float)
        hc_flag_full.loc[valid_mask] = np.asarray(hc_flag, dtype=float)

        if req.enable_coal_flagging:
            coal_mask = _compute_coal_mask(
                df,
                enable_coal_flagging=True,
                coal_rhob_max=req.coal_rhob_max,
                coal_nphi_min=req.coal_nphi_min,
                coal_window_size=req.coal_window_size,
                coal_trend_factor=req.coal_trend_factor,
            )
            if coal_mask.any():
                nphi_hc_full.loc[coal_mask] = nphi.loc[coal_mask]
                rhob_hc_full.loc[coal_mask] = rhob.loc[coal_mask]
                hc_flag_full.loc[coal_mask] = 0.0
        else:
            coal_mask = pd.Series(False, index=df.index, dtype=bool)

        bh_mode = (req.badhole_treatment or "none").strip().lower()
        if bh_mode != "none":
            badhole_mask = _resolve_badhole_mask(df, auto_detect_if_missing=True) & (
                ~coal_mask
            )
            if badhole_mask.any():
                nphi_hc_full.loc[badhole_mask] = nphi.loc[badhole_mask]
                rhob_hc_full.loc[badhole_mask] = rhob.loc[badhole_mask]
                hc_flag_full.loc[badhole_mask] = 0.0
                if bh_mode in ("vsh_gr", "gr") and "GR" in df.columns:
                    vsh_bh_s = _compute_vsh_gr_series(df, req.gr_clean, req.gr_clay)
                    bh_gr = badhole_mask & vsh_bh_s.notna()
                    if bh_gr.any():
                        vsh_v = vsh_bh_s.loc[bh_gr].to_numpy(dtype=float)
                        phin_sh = (
                            float(req.dry_clay_point[0])
                            if (
                                req.dry_clay_point and req.dry_clay_point[0] is not None
                            )
                            else 0.33
                        )
                        rho_ma_sand = (
                            float(req.dry_sand_point[1])
                            if (
                                req.dry_sand_point and req.dry_sand_point[1] is not None
                            )
                            else 2.65
                        )
                        nphi_bh = (
                            nphi.loc[bh_gr]
                            .clip(0.0, 0.42)
                            .fillna(
                                pd.Series(
                                    0.18 * (1.0 - vsh_v) + phin_sh * vsh_v,
                                    index=df.index[bh_gr],
                                )
                            )
                            .to_numpy(dtype=float)
                        )
                        gr_bh = (
                            pd.to_numeric(df.loc[bh_gr, "GR"], errors="coerce")
                            .fillna(50.0)
                            .to_numpy(dtype=float)
                        )
                        rhob_rec = np.asarray(
                            den_correction(
                                nphi_bh,
                                gr_bh,
                                vsh_gr=vsh_v,
                                phin_sh=phin_sh,
                                phid_sh=0.05,
                                rho_ma=rho_ma_sand,
                                rho_water=float(req.rho_water),
                            ),
                            dtype=float,
                        )
                        rhob_hc_full.loc[bh_gr] = np.clip(rhob_rec, 1.8, 2.95)
                        nphi_hc_full.loc[bh_gr] = nphi_bh

        out_cols_dict: Dict[str, Any] = {
            "DEPTH": df["DEPTH"],
            "NPHI_HC": nphi_hc_full,
            "RHOB_HC": rhob_hc_full,
            "HC_FLAG": hc_flag_full,
            "DEN_CORR_FLAG": pd.Series(0.0, index=df.index, dtype=float),
        }
        if vsh_gr_s.notna().any():
            out_cols_dict["VSH_GR"] = vsh_gr_s
        out_df = pd.DataFrame(out_cols_dict)

        hc_active = valid_mask & (hc_flag_full >= 0.5)
        n_valid = int(valid_mask.sum())
        n_hc = int(hc_active.sum())
        mean_vsh_gr_hc = (
            round(float(vsh_gr_s.loc[hc_active].mean()), 4)
            if (n_hc > 0 and vsh_gr_s.loc[hc_active].notna().any())
            else None
        )
        mean_dnphi = (
            round(float((nphi_hc_full.loc[hc_active] - nphi.loc[hc_active]).mean()), 4)
            if n_hc > 0
            else 0.0
        )
        mean_drhob = (
            round(float((rhob_hc_full.loc[hc_active] - rhob.loc[hc_active]).mean()), 4)
            if n_hc > 0
            else 0.0
        )
        hc_summary = {
            "corr_angle": round(corr_angle, 2),
            "hc_flagged_count": n_hc,
            "valid_count": n_valid,
            "hc_flagged_pct": round((n_hc / n_valid) * 100.0, 1) if n_valid > 0 else 0.0,
            "gr_clean": float(req.gr_clean) if req.gr_clean is not None else None,
            "gr_clay": float(req.gr_clay) if req.gr_clay is not None else None,
            "mean_vsh_gr_hc": mean_vsh_gr_hc,
            "mean_dnphi": mean_dnphi,
            "mean_drhob": mean_drhob,
        }

        save_cols = [c for c in ("NPHI_HC", "RHOB_HC", "HC_FLAG", "VSH_GR") if c in out_df.columns]
        saved_rows = 0
        if req.save:
            save_df = out_df.loc[valid_mask, ["DEPTH"] + save_cols]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id,
                well_name,
                out_df.loc[valid_mask, ["DEPTH"] + save_cols],
            )

        stats_out = {
            "NPHI_HC": _curve_summary_stats(nphi_hc_full),
            "RHOB_HC": _curve_summary_stats(rhob_hc_full),
            "HC_FLAG": _curve_summary_stats(hc_flag_full),
        }
        if "VSH_GR" in out_df.columns:
            stats_out["VSH_GR"] = _curve_summary_stats(out_df["VSH_GR"])

        return {
            "well_name": well_name,
            "corr_angle": round(corr_angle, 3),
            "hc_summary": hc_summary,
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": stats_out,
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"HC correction failed: {e}") from e


@router.post(
    "/wells/{well_name}/den_correction",
    summary="Compute depth-aligned bulk density correction (den_correction) from NPHI and calibrated VSH_GR",
)
def workspace_den_correction(
    project_id: int, well_name: str, req: WorkspaceDenCorrectionRequest
) -> Dict[str, Any]:
    """Reconstruct RHOB from NPHI and calibrated VSH_GR using den_correction and store in RHOB_HC / RHOB_CORR."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=False)
        valid_mask = nphi.notna() & rhob.notna()
        if not valid_mask.any():
            raise ValueError(
                "No valid NPHI and RHOB pairs found in the selected interval"
            )

        vsh_gr_s = _compute_vsh_gr_series(df, req.gr_clean, req.gr_clay)
        if not vsh_gr_s.notna().any():
            raise ValueError(
                "GR or VSH_GR curve is required to run density correction (den_correction)"
            )

        rho_w = float(req.rho_water)
        if req.sync_endpoints:
            nphi_sand = float(req.dry_sand_point[0])
            rho_sand = float(req.dry_sand_point[1])
            nphi_clay = float(req.dry_clay_point[0])
            rho_clay = float(req.dry_clay_point[1])
            denom_sand = (
                1.0 - nphi_sand if abs(1.0 - nphi_sand) > 1e-6 else 1.0
            )
            rho_ma_val = (rho_sand - nphi_sand * rho_w) / denom_sand
            phin_sh_val = nphi_clay
            denom_ma = (
                rho_ma_val - rho_w if abs(rho_ma_val - rho_w) > 1e-6 else 1.65
            )
            phid_sh_val = (rho_ma_val - rho_clay) / denom_ma
        else:
            rho_ma_val = float(req.rho_ma)
            phin_sh_val = float(req.phin_sh)
            phid_sh_val = float(req.phid_sh)

        if req.enable_coal_flagging:
            coal_mask = _compute_coal_mask(
                df,
                enable_coal_flagging=True,
                coal_rhob_max=req.coal_rhob_max,
                coal_nphi_min=req.coal_nphi_min,
                coal_window_size=req.coal_window_size,
                coal_trend_factor=req.coal_trend_factor,
            )
        else:
            coal_mask = pd.Series(False, index=df.index, dtype=bool)

        scope_mode = (req.scope or "vsh_mismatch").strip().lower()
        if scope_mode == "badhole_only":
            badhole_mask = _resolve_badhole_mask(df, auto_detect_if_missing=True)
            active_mask = valid_mask & (~coal_mask) & badhole_mask & vsh_gr_s.notna()
        elif scope_mode == "vsh_mismatch":
            badhole_mask = _resolve_badhole_mask(df, auto_detect_if_missing=False)
            A = req.dry_sand_point
            C = req.dry_clay_point
            D = (1.0, rho_w)
            rock_len = max(1e-6, float(length_a_b(A, C)))
            nphi_valid_np = nphi.loc[valid_mask].to_numpy(dtype=float)
            rhob_valid_np = rhob.loc[valid_mask].to_numpy(dtype=float)
            var_pts = line_intersection(
                (A, C),
                (D, (nphi_valid_np, rhob_valid_np)),
            )
            signed_proj = np.where(var_pts[1] > C[1], -1.0, 1.0) * length_a_b(
                var_pts, C
            )
            proj_clamped = np.clip(signed_proj, 0.0, rock_len)
            vsh_dn_raw = pd.Series(np.nan, index=df.index, dtype=float)
            vsh_dn_raw.loc[valid_mask] = 1.0 - (proj_clamped / rock_len)
            mismatch_mask = (vsh_dn_raw - vsh_gr_s).abs() >= float(
                req.vsh_mismatch_threshold
            )
            active_mask = (
                valid_mask
                & (~coal_mask)
                & (badhole_mask | mismatch_mask)
                & vsh_gr_s.notna()
            )
        else:
            active_mask = valid_mask & (~coal_mask) & vsh_gr_s.notna()

        nphi_hc_full = nphi.copy()
        rhob_hc_full = rhob.copy()
        den_flag_full = pd.Series(0.0, index=df.index, dtype=float)
        hc_flag_full = pd.Series(0.0, index=df.index, dtype=float)

        if active_mask.any():
            nphi_act = nphi.loc[active_mask].to_numpy(dtype=float)
            vsh_act = vsh_gr_s.loc[active_mask].to_numpy(dtype=float)
            rhob_rec = np.asarray(
                den_correction(
                    nphi_act,
                    None,
                    vsh_gr=vsh_act,
                    phin_sh=phin_sh_val,
                    phid_sh=phid_sh_val,
                    rho_ma=rho_ma_val,
                    rho_water=rho_w,
                ),
                dtype=float,
            )
            rhob_hc_full.loc[active_mask] = np.clip(rhob_rec, 1.65, 2.98)
            den_flag_full.loc[active_mask] = 1.0

        out_cols_dict: Dict[str, Any] = {
            "DEPTH": df["DEPTH"],
            "NPHI_HC": nphi_hc_full,
            "RHOB_HC": rhob_hc_full,
            "RHOB_CORR": rhob_hc_full,
            "DEN_CORR_FLAG": den_flag_full,
            "HC_FLAG": hc_flag_full,
            "VSH_GR": vsh_gr_s,
        }
        out_df = pd.DataFrame(out_cols_dict)

        n_valid = int(valid_mask.sum())
        n_corr = int(active_mask.sum())
        drhob_s = rhob_hc_full.loc[active_mask] - rhob.loc[active_mask]
        mean_drhob = round(float(drhob_s.mean()), 4) if n_corr > 0 else 0.0
        mean_abs_drhob = round(float(drhob_s.abs().mean()), 4) if n_corr > 0 else 0.0
        mean_vsh_gr = (
            round(float(vsh_gr_s.loc[active_mask].mean()), 4) if n_corr > 0 else None
        )

        den_summary = {
            "scope": scope_mode,
            "sync_endpoints": bool(req.sync_endpoints),
            "corrected_count": n_corr,
            "valid_count": n_valid,
            "corrected_pct": round((n_corr / n_valid) * 100.0, 1)
            if n_valid > 0
            else 0.0,
            "rho_ma": round(rho_ma_val, 4),
            "phin_sh": round(phin_sh_val, 4),
            "phid_sh": round(phid_sh_val, 4),
            "gr_clean": float(req.gr_clean) if req.gr_clean is not None else None,
            "gr_clay": float(req.gr_clay) if req.gr_clay is not None else None,
            "mean_vsh_gr": mean_vsh_gr,
            "mean_drhob": mean_drhob,
            "mean_abs_drhob": mean_abs_drhob,
        }

        save_cols = [
            "NPHI_HC",
            "RHOB_HC",
            "RHOB_CORR",
            "DEN_CORR_FLAG",
            "HC_FLAG",
            "VSH_GR",
        ]
        saved_rows = 0
        if req.save:
            save_df = out_df.loc[valid_mask, ["DEPTH"] + save_cols]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id,
                well_name,
                out_df.loc[valid_mask, ["DEPTH"] + save_cols],
            )

        stats_out = {
            col: _curve_summary_stats(out_df[col]) for col in save_cols
        }
        return {
            "well_name": well_name,
            "den_summary": den_summary,
            "saved": req.save,
            "saved_rows": saved_rows,
            "stats": stats_out,
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Density correction failed: {e}"
        ) from e


@router.post(
    "/wells/{well_name}/badhole_flagging",
    summary="Compute depth-aligned BADHOLE flag (Caliper, DRHO, RHOB/NPHI bounds, and Rugosity) on server",
)
def workspace_badhole_flagging(
    project_id: int, well_name: str, req: WorkspaceBadholeRequest
) -> Dict[str, Any]:
    """Compute BADHOLE, BS, and DCALI aligned on DEPTH and optionally save to the database."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        # Drop any previously generated BS column so re-running with updated
        # auto_bitsize / bitsize_default parameters always recomputes BS fresh.
        df_bh_input = df.drop(columns=["BS"], errors="ignore")
        bh_df = badhole_flagging(
            df_bh_input,
            thold=float(req.cali_threshold),
            use_cali=bool(req.use_cali),
            auto_bitsize=bool(req.auto_bitsize),
            bitsize_default=float(req.bitsize_default),
            use_drho=bool(req.use_drho),
            drho_threshold=float(req.drho_threshold),
            use_rhob_nphi=bool(req.use_rhob_nphi),
            rhob_min=float(req.rhob_min),
            rhob_max=float(req.rhob_max),
            nphi_max=float(req.nphi_max),
            exclude_coal=bool(req.exclude_coal),
            use_rugosity=bool(req.use_rugosity),
            rugosity_window=int(req.rugosity_window),
            rhob_trend_factor=float(req.rhob_trend_factor),
            cali_spike_threshold=float(req.cali_spike_threshold),
        )
        out_cols = ["DEPTH", "BADHOLE"]
        if "BS" in bh_df.columns and bh_df["BS"].notna().any():
            out_cols.append("BS")
        if "DCALI" in bh_df.columns and bh_df["DCALI"].notna().any():
            out_cols.append("DCALI")

        out_df = bh_df[out_cols].copy()
        out_df["BADHOLE"] = pd.to_numeric(out_df["BADHOLE"], errors="coerce").fillna(
            0.0
        )

        saved_rows = 0
        if req.save:
            saved_rows = _save_curves_to_well(project_id, well_name, out_df)
        else:
            _update_in_memory_well_cache(project_id, well_name, out_df)

        summary = dict(bh_df.attrs.get("badhole_summary") or {})
        save_stat_cols = [c for c in out_cols if c != "DEPTH"]
        stats = {col: _curve_summary_stats(out_df[col]) for col in save_stat_cols}
        return {
            "well_name": well_name,
            "saved": req.save,
            "saved_rows": saved_rows,
            "summary": summary,
            "badhole_summary": summary,
            "stats": stats,
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Badhole flagging failed: {e}"
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

            out_df["VSAND"] = vsand_s
            out_df["VSILT"] = vsilt_s
            out_df["VCLAY"] = vclay_s
            out_df["VSH_DN"] = vclay_s
            out_df["PROJ_NPHI"] = px_s
            out_df["PROJ_RHOB"] = py_s
            vsh_gr_s = _compute_vsh_gr_series(df, req.gr_clean, req.gr_clay)
            if vsh_gr_s.notna().any():
                out_df["VSH_GR"] = vsh_gr_s

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
            if "PHIT" in df.columns:
                phit_ref = (
                    pd.to_numeric(df["PHIT"], errors="coerce")
                    .where(lambda s: s.notna() & (s >= 0.0) & (s <= 0.6), phid_ssc)
                    .fillna(phid_ssc)
                    .fillna(0.0)
                    .clip(0.0, 0.6)
                )
            else:
                phit_ref = phid_ssc.fillna(0.0).clip(0.0, 0.6)
            vmatrix_s = (1.0 - phit_ref).clip(0.0, 1.0)
            for mcol in ("VSAND", "VSILT", "VCLAY"):
                out_df[mcol] = (out_df[mcol] * vmatrix_s).clip(0.0, 1.0)

            # Zero out any non-SSC mineral columns so stale carbonate/evaporite volumes do not persist
            for mcol in ALL_LITHOLOGY_VOL_COLS:
                if mcol not in ("VSAND", "VSILT", "VCLAY"):
                    zero_s = pd.Series(np.nan, index=df.index, dtype=float)
                    zero_s.loc[valid_mask] = 0.0
                    out_df[mcol] = zero_s
            save_cols = list(ALL_LITHOLOGY_VOL_COLS) + [
                c for c in ("VSH_GR", "VSH_DN") if c in out_df.columns
            ]

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
                        if req.dry_clay_point
                        and req.dry_clay_point[0] is not None
                        and req.dry_clay_point[1] is not None
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
            if (
                req.gr_clean is not None
                and req.gr_clay is not None
                and abs(float(req.gr_clay) - float(req.gr_clean)) > 1e-6
            ):
                vsh = (
                    (gr[valid_mask] - float(req.gr_clean))
                    / (float(req.gr_clay) - float(req.gr_clean))
                ).to_numpy(dtype=float)
            else:
                vsh = gr_index(gr[valid_mask].reset_index(drop=True))
            vclay_s = pd.Series(np.nan, index=df.index, dtype=float)
            vclay_s.loc[valid_mask] = np.clip(
                np.asarray(vsh, dtype=float).ravel(), 0.0, 1.0
            )
            phit_ref = (
                (
                    pd.to_numeric(df["PHIT"], errors="coerce")
                    if "PHIT" in df.columns
                    else pd.Series(0.0, index=df.index, dtype=float)
                )
                .fillna(0.0)
                .clip(0.0, 0.6)
            )
            vmatrix_s = (1.0 - phit_ref).clip(0.0, 1.0)
            out_df["VCLAY"] = (vclay_s * vmatrix_s).clip(0.0, 1.0)
            out_df["VSAND"] = ((1.0 - vclay_s).clip(0.0, 1.0) * vmatrix_s).clip(
                0.0, 1.0
            )
            for mcol in ALL_LITHOLOGY_VOL_COLS:
                if mcol not in ("VCLAY", "VSAND"):
                    zero_s = pd.Series(np.nan, index=df.index, dtype=float)
                    zero_s.loc[valid_mask] = 0.0
                    out_df[mcol] = zero_s
            save_cols = list(ALL_LITHOLOGY_VOL_COLS)
        else:
            raise ValueError(f"Unsupported lithology model: {req.model}")

        if req.enable_coal_flagging:
            coal_mask = _compute_coal_mask(
                df,
                enable_coal_flagging=True,
                coal_rhob_max=req.coal_rhob_max,
                coal_nphi_min=req.coal_nphi_min,
                coal_window_size=req.coal_window_size,
                coal_trend_factor=req.coal_trend_factor,
            )
            coal_flag_s = pd.Series(np.nan, index=df.index, dtype=float)
            coal_flag_s.loc[valid_mask] = np.where(coal_mask.loc[valid_mask], 1.0, 0.0)
            out_df["COAL_FLAG"] = coal_flag_s
            active_coal = valid_mask & coal_mask
            if active_coal.any():
                if "VCOAL" not in out_df.columns:
                    vcoal_s = pd.Series(np.nan, index=df.index, dtype=float)
                    vcoal_s.loc[valid_mask] = 0.0
                    out_df["VCOAL"] = vcoal_s
                for mcol in ALL_LITHOLOGY_VOL_COLS:
                    if mcol in out_df.columns:
                        out_df.loc[active_coal, mcol] = 1.0 if mcol == "VCOAL" else 0.0
                if "PHIT_CONSTRUCTED" in out_df.columns:
                    out_df.loc[active_coal, "PHIT_CONSTRUCTED"] = 0.0
            if "COAL_FLAG" not in save_cols:
                save_cols.append("COAL_FLAG")
            if "VCOAL" not in save_cols and "VCOAL" in out_df.columns:
                save_cols.append("VCOAL")
        else:
            coal_mask = pd.Series(False, index=df.index, dtype=bool)
            coal_flag_s = pd.Series(np.nan, index=df.index, dtype=float)
            coal_flag_s.loc[valid_mask] = 0.0
            out_df["COAL_FLAG"] = coal_flag_s
            if "COAL_FLAG" not in save_cols:
                save_cols.append("COAL_FLAG")

        # Badhole interval treatment for lithology volumes
        bh_mode = (req.badhole_treatment or "none").strip().lower()
        badhole_mask = _resolve_badhole_mask(
            df, auto_detect_if_missing=(bh_mode != "none")
        )
        active_badhole = badhole_mask & (~coal_mask)
        if bh_mode != "none" or badhole_mask.any() or "BADHOLE" in df.columns:
            out_df["BADHOLE"] = np.where(badhole_mask, 1.0, 0.0)
            if "BADHOLE" not in save_cols:
                save_cols.append("BADHOLE")

        if bh_mode != "none" and active_badhole.any():
            if bh_mode in ("vsh_gr", "gr"):
                vsh_bh_s = _compute_vsh_gr_series(df, req.gr_clean, req.gr_clay)
                bh_gr = active_badhole & vsh_bh_s.notna()
                if bh_gr.any():
                    vsh_v = vsh_bh_s.loc[bh_gr].to_numpy(dtype=float)
                    phin_sh = (
                        float(req.dry_clay_point[0])
                        if (req.dry_clay_point and req.dry_clay_point[0] is not None)
                        else 0.33
                    )
                    rho_ma_sand = (
                        float(req.dry_sand_point[1])
                        if (req.dry_sand_point and req.dry_sand_point[1] is not None)
                        else 2.65
                    )
                    rho_fl = (
                        float(req.fluid_point[1])
                        if (req.fluid_point and len(req.fluid_point) > 1)
                        else 1.0
                    )
                    nphi_raw_s = (
                        pd.to_numeric(df["NPHI"], errors="coerce")
                        if "NPHI" in df.columns
                        else pd.Series(np.nan, index=df.index, dtype=float)
                    )
                    nphi_bh = (
                        nphi_raw_s.loc[bh_gr]
                        .clip(0.0, 0.42)
                        .fillna(
                            pd.Series(
                                0.18 * (1.0 - vsh_v) + phin_sh * vsh_v,
                                index=df.index[bh_gr],
                            )
                        )
                        .to_numpy(dtype=float)
                    )
                    gr_bh = (
                        pd.to_numeric(df.loc[bh_gr, "GR"], errors="coerce")
                        .fillna(50.0)
                        .to_numpy(dtype=float)
                    )
                    rhob_rec = np.asarray(
                        den_correction(
                            nphi_bh,
                            gr_bh,
                            vsh_gr=vsh_v,
                            phin_sh=phin_sh,
                            phid_sh=0.05,
                            rho_ma=rho_ma_sand,
                            rho_water=rho_fl,
                        ),
                        dtype=float,
                    )
                    phit_bh = np.clip(
                        np.asarray(
                            density_porosity(rhob_rec, rho_ma_sand, rho_fl), dtype=float
                        ),
                        0.0,
                        0.45,
                    )
                    vmat_bh = np.clip(1.0 - phit_bh, 0.0, 1.0)
                    out_df.loc[bh_gr, "VCLAY"] = np.clip(vsh_v * vmat_bh, 0.0, 1.0)
                    out_df.loc[bh_gr, "VSAND"] = np.clip(
                        (1.0 - vsh_v) * vmat_bh, 0.0, 1.0
                    )
                    for mcol in ALL_LITHOLOGY_VOL_COLS:
                        if mcol not in ("VCLAY", "VSAND") and mcol in out_df.columns:
                            out_df.loc[bh_gr, mcol] = 0.0
                    if "PHIT_CONSTRUCTED" in out_df.columns:
                        out_df.loc[bh_gr, "PHIT_CONSTRUCTED"] = phit_bh
            elif bh_mode == "interpolate":
                for mcol in ALL_LITHOLOGY_VOL_COLS:
                    if mcol in out_df.columns:
                        out_df.loc[active_badhole, mcol] = np.nan
                        out_df[mcol] = (
                            out_df[mcol]
                            .interpolate(method="linear", limit_direction="both")
                            .clip(0.0, 1.0)
                        )
                if "PHIT_CONSTRUCTED" in out_df.columns:
                    out_df.loc[active_badhole, "PHIT_CONSTRUCTED"] = np.nan
                    out_df["PHIT_CONSTRUCTED"] = (
                        out_df["PHIT_CONSTRUCTED"]
                        .interpolate(method="linear", limit_direction="both")
                        .clip(0.0, 0.6)
                    )
            elif bh_mode in ("ignore", "null", "nan"):
                for mcol in ALL_LITHOLOGY_VOL_COLS:
                    if mcol in out_df.columns:
                        out_df.loc[active_badhole, mcol] = np.nan
                for extra_c in ("PHIT_CONSTRUCTED", "PROJ_NPHI", "PROJ_RHOB"):
                    if extra_c in out_df.columns:
                        out_df.loc[active_badhole, extra_c] = np.nan

        saved_rows = 0
        if save_cols:
            valid_save_mask = out_df[save_cols].notna().any(axis=1) | active_badhole
            save_df = out_df.loc[valid_save_mask, ["DEPTH"] + save_cols]
            if req.save:
                saved_rows = _save_curves_to_well(project_id, well_name, save_df)
            else:
                _update_in_memory_well_cache(project_id, well_name, save_df)

        stats = {col: _curve_summary_stats(out_df[col]) for col in save_cols}
        return {
            "well_name": well_name,
            "model": model,
            "badhole_treatment": bh_mode,
            "badhole_count": int(active_badhole.sum()),
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
    "/wells/{well_name}/thomas_stieber",
    summary="Compute Thomas-Stieber thin-bed shale partitioning (VSH_LAM, VSH_DIS, PHIT_SAND) and Hagiwara RT_MOD",
)
def workspace_thomas_stieber(
    project_id: int, well_name: str, req: WorkspaceThomasStieberRequest
) -> Dict[str, Any]:
    """Compute Thomas-Stieber laminated/dispersed shale volumes, sand lamina porosity, and Hagiwara resistivity."""
    try:
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        dry_sand_poro = max(0.05, min(float(req.dry_sand_poro), 0.50))
        dry_shale_poro = max(0.01, min(float(req.dry_shale_poro), 0.45))

        tb = ThinBeds(
            dry_sand_poro=dry_sand_poro,
            dry_shale_poro=dry_shale_poro,
            dry_sand_point=req.dry_sand_point,
            dry_clay_point=req.dry_clay_point,
            fluid_point=req.fluid_point,
        )

        has_existing = (
            req.use_existing_phit_vsh
            and "PHIT" in df.columns
            and ("VCLAY" in df.columns or "VSHALE" in df.columns)
        )

        phit_s = (
            pd.to_numeric(df["PHIT"], errors="coerce")
            if "PHIT" in df.columns
            else pd.Series(np.nan, index=df.index, dtype=float)
        )
        vclay_col = (
            "VCLAY"
            if "VCLAY" in df.columns
            else ("VSHALE" if "VSHALE" in df.columns else None)
        )
        vclay_bulk_s = (
            pd.to_numeric(df[vclay_col], errors="coerce")
            if vclay_col is not None
            else pd.Series(np.nan, index=df.index, dtype=float)
        )

        if not has_existing or not (phit_s.notna() & vclay_bulk_s.notna()).any():
            nphi, rhob = _select_nphi_rhob(df, use_hc_corrected=req.use_hc_corrected)
            valid_nd = nphi.notna() & rhob.notna()
            if not valid_nd.any():
                raise ValueError(
                    "No valid PHIT/VCLAY or NPHI/RHOB pairs found in the selected interval"
                )
            _, vsh_tb, phit_tb, _, _, _, _ = tb.estimate_litho_poro(
                nphi[valid_nd].to_numpy(dtype=float),
                rhob[valid_nd].to_numpy(dtype=float),
            )
            phit_s.loc[valid_nd] = np.clip(np.asarray(phit_tb, dtype=float), 0.0, 0.5)
            vsh_norm_nd = np.clip(np.asarray(vsh_tb, dtype=float), 0.0, 1.0)
            vclay_bulk_s.loc[valid_nd] = np.clip(
                vsh_norm_nd * (1.0 - phit_s.loc[valid_nd].to_numpy(dtype=float)),
                0.0,
                1.0,
            )

        valid_mask = phit_s.notna() & vclay_bulk_s.notna()
        if not valid_mask.any():
            raise ValueError(
                "No valid porosity and shale volume rows available for Thomas-Stieber estimation"
            )

        phit_v = np.clip(phit_s.loc[valid_mask].to_numpy(dtype=float), 0.0, 0.55)
        vclay_bulk_v = np.clip(
            vclay_bulk_s.loc[valid_mask].to_numpy(dtype=float), 0.0, 1.0
        )

        # Normalize bulk clay volume to [0, 1] on the Thomas-Stieber Vshale axis where pure shale = 1.0
        vmatrix_denom = np.maximum(1.0 - phit_v + dry_shale_poro * vclay_bulk_v, 1e-4)
        vsh_ts_v = np.clip(vclay_bulk_v / np.maximum(1.0 - phit_v, 0.15), 0.0, 1.0)
        # If VCLAY is already unscaled VSHALE close to 1.0 in shales, blend smoothly
        if float(np.nanmax(vclay_bulk_v)) > 0.88:
            vsh_ts_v = np.clip(vclay_bulk_v, 0.0, 1.0)

        vsh_lam_raw, vsh_dis_raw, vsand_dis_raw, phit_sand_raw = tb.litho_poro_fraction(
            phit_v, vsh_ts_v
        )
        vsh_lam_raw = np.asarray(vsh_lam_raw, dtype=float)
        vsh_dis_raw = np.asarray(vsh_dis_raw, dtype=float)
        vsand_dis_raw = np.asarray(vsand_dis_raw, dtype=float)
        phit_sand_raw = np.asarray(phit_sand_raw, dtype=float)

        # Partition bulk VCLAY into VSH_LAM + VSH_DIS == VCLAY so Track 4 mineral stacking stays exact
        vsh_lam_norm = np.clip(vsh_lam_raw, 0.0, vsh_ts_v)
        lam_fraction = np.where(
            vsh_ts_v > 1e-5,
            np.clip(vsh_lam_norm / vsh_ts_v, 0.0, 1.0),
            0.0,
        )
        vsh_lam_bulk = np.clip(vclay_bulk_v * lam_fraction, 0.0, vclay_bulk_v)
        vsh_dis_bulk = np.clip(vclay_bulk_v - vsh_lam_bulk, 0.0, vclay_bulk_v)
        vsand_dis_clamped = np.clip(vsand_dis_raw, 0.0, 1.0)
        phit_sand_clamped = np.clip(
            np.where(np.isfinite(phit_sand_raw), phit_sand_raw, phit_v),
            0.0,
            max(dry_sand_poro * 1.15, 0.45),
        )
        # In pure shale (Vsh > 0.95), intrinsic sand lamina porosity tapers to shale porosity
        phit_sand_clamped = np.where(
            vsh_ts_v > 0.95, phit_v, np.maximum(phit_sand_clamped, phit_v)
        )

        # Hagiwara (1995) anisotropic thin-bed resistivity forward model
        rsand = max(0.05, float(req.rsand))
        rv_shale = max(0.05, float(req.rv_shale))
        rh_shale = max(0.05, float(req.rh_shale))
        theta = max(0.0, min(float(req.theta), 89.9))

        rt_mod_v = np.asarray(
            tb.resistivity_modelling(
                vsh_lam_norm,
                rsand=rsand,
                rv_shale=rv_shale,
                rh_shale=rh_shale,
                theta=theta,
            ),
            dtype=float,
        )
        rt_mod_v = np.clip(rt_mod_v, 0.05, 5000.0)

        # Invert true sand-lamina resistivity RSAND_TS from measured RT & VSH_LAM (with Hagiwara parallel conductivity)
        rsand_ts_v = np.full_like(rt_mod_v, rsand)
        for rt_alias in ("RT", "RD", "RDEP", "RDEEP", "LLD", "RILD"):
            if rt_alias in df.columns:
                rt_m = pd.to_numeric(
                    df.loc[valid_mask, rt_alias], errors="coerce"
                ).to_numpy(dtype=float)
                valid_rt_m = np.isfinite(rt_m) & (rt_m > 0.05)
                if valid_rt_m.any():
                    # Parallel horizontal conductivity: 1/Rt = Vsh_lam/Rh_shale + (1 - Vsh_lam)/Rsand
                    c_meas = 1.0 / np.maximum(rt_m, 0.05)
                    c_sh = vsh_lam_norm / rh_shale
                    sand_frac = np.maximum(1.0 - vsh_lam_norm, 0.05)
                    # Blend measured conductivity subtraction with forward Rsand prior where shale fraction is high
                    c_sand_inv = (c_meas - 0.85 * c_sh) / sand_frac
                    r_inv = np.where(
                        c_sand_inv > (1.0 / max(rsand * 3.0, 500.0)),
                        1.0 / np.maximum(c_sand_inv, 1e-4),
                        rsand,
                    )
                    r_inv = np.clip(
                        np.maximum(r_inv, rt_m), 0.1, max(rsand * 3.0, 1000.0)
                    )
                    rsand_ts_v = np.where(valid_rt_m, r_inv, rsand)
                break

        out_df = pd.DataFrame({"DEPTH": df["DEPTH"]})
        for col_name, arr_vals in (
            ("VSH_LAM", vsh_lam_bulk),
            ("VSH_DIS", vsh_dis_bulk),
            ("VSAND_DIS", vsand_dis_clamped),
            ("PHIT_SAND", phit_sand_clamped),
            ("RT_MOD", rt_mod_v),
            ("RSAND_TS", rsand_ts_v),
            ("VSH_TS", vsh_ts_v),
        ):
            s_col = pd.Series(np.nan, index=df.index, dtype=float)
            s_col.loc[valid_mask] = arr_vals
            out_df[col_name] = s_col

        save_cols = [
            "VSH_LAM",
            "VSH_DIS",
            "VSAND_DIS",
            "PHIT_SAND",
            "RT_MOD",
            "RSAND_TS",
            "VSH_TS",
        ]
        saved_rows = 0
        save_df = out_df.loc[valid_mask, ["DEPTH"] + save_cols]
        if req.save:
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(project_id, well_name, save_df)

        # Compute summary & resistivity fit metrics against measured RT if present
        rt_rmse_decade: Optional[float] = None
        for rt_alias in ("RT", "RD", "RDEP", "RDEEP", "LLD", "RILD"):
            if rt_alias in df.columns:
                rt_meas = pd.to_numeric(
                    df.loc[valid_mask, rt_alias], errors="coerce"
                ).to_numpy(dtype=float)
                valid_rt = (
                    np.isfinite(rt_meas)
                    & (rt_meas > 0)
                    & np.isfinite(rt_mod_v)
                    & (rt_mod_v > 0)
                )
                if valid_rt.any():
                    diff_log = np.log10(rt_mod_v[valid_rt]) - np.log10(
                        rt_meas[valid_rt]
                    )
                    rt_rmse_decade = round(float(np.sqrt(np.mean(diff_log**2))), 4)
                break

        sand_mask = vsh_ts_v < 0.65
        mean_uplift = (
            round(
                float(np.nanmean(phit_sand_clamped[sand_mask] - phit_v[sand_mask])), 4
            )
            if sand_mask.any()
            else 0.0
        )

        stats = {col: _curve_summary_stats(out_df[col]) for col in save_cols}
        return {
            "well_name": well_name,
            "saved": req.save,
            "saved_rows": saved_rows,
            "summary": {
                "mean_vsh_lam": round(float(np.nanmean(vsh_lam_bulk)), 4),
                "mean_vsh_dis": round(float(np.nanmean(vsh_dis_bulk)), 4),
                "mean_phit_sand": round(float(np.nanmean(phit_sand_clamped)), 4),
                "mean_porosity_uplift_vv": mean_uplift,
                "rt_rmse_decade": rt_rmse_decade,
                "lower_vertex_vsh": round(dry_sand_poro, 4),
                "lower_vertex_phit": round(dry_sand_poro * dry_shale_poro, 4),
            },
            "stats": stats,
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Thomas-Stieber calculation failed: {e}"
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
            and pd.to_numeric(df.loc[valid_mask, c], errors="coerce").fillna(0.0).max()
            > 1e-4
            for c in (
                "VCALC",
                "VDOLO",
                "VANHY",
                "VHALI",
                "VGYPS",
                "VKAOL",
                "VFELD",
                "VPYRI",
                "VCOAL",
            )
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
                and pd.to_numeric(
                    df.loc[valid_mask, "PHIT_CONSTRUCTED"], errors="coerce"
                )
                .notna()
                .any()
            ):
                phit_c = pd.to_numeric(
                    df.loc[valid_mask, "PHIT_CONSTRUCTED"], errors="coerce"
                ).to_numpy(dtype=float)
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
            vsand_m, vsilt_m, vclay_m, _ = ssc_model.estimate_lithology(nphi_s, rhob_s)
            vsand = np.asarray(vsand_m, dtype=float)
            vsilt = np.asarray(vsilt_m, dtype=float)
            vclay = np.asarray(vclay_m, dtype=float)
            if all(c in df.columns for c in ("VSAND", "VSILT", "VCLAY")):
                vs_df = (
                    pd.to_numeric(df.loc[valid_mask, "VSAND"], errors="coerce")
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                    .to_numpy(dtype=float)
                )
                vsi_df = (
                    pd.to_numeric(df.loc[valid_mask, "VSILT"], errors="coerce")
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                    .to_numpy(dtype=float)
                )
                vcl_df = (
                    pd.to_numeric(df.loc[valid_mask, "VCLAY"], errors="coerce")
                    .fillna(0.0)
                    .clip(0.0, 1.0)
                    .to_numpy(dtype=float)
                )
                tot_ssc = vs_df + vsi_df + vcl_df
                has_ssc = tot_ssc > 1e-6
                if np.any(has_ssc):
                    vsand = np.where(has_ssc, vs_df / tot_ssc, vsand)
                    vsilt = np.where(has_ssc, vsi_df / tot_ssc, vsilt)
                    vclay = np.where(has_ssc, vcl_df / tot_ssc, vclay)
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

        if req.enable_coal_flagging:
            coal_mask = _compute_coal_mask(
                df,
                enable_coal_flagging=True,
                coal_rhob_max=req.coal_rhob_max,
                coal_nphi_min=req.coal_nphi_min,
                coal_window_size=req.coal_window_size,
                coal_trend_factor=req.coal_trend_factor,
            )
        elif "COAL_FLAG" in df.columns:
            coal_mask = (
                pd.to_numeric(df["COAL_FLAG"], errors="coerce").fillna(0.0) >= 0.5
            )
        else:
            coal_mask = pd.Series(False, index=df.index, dtype=bool)

        active_coal = valid_mask & coal_mask
        if active_coal.any():
            phit_s.loc[active_coal] = 0.0
            phie_s.loc[active_coal] = 0.0
            phid_s.loc[active_coal] = 0.0

        bh_mode = (req.badhole_treatment or "none").strip().lower()
        badhole_mask = _resolve_badhole_mask(
            df, auto_detect_if_missing=(bh_mode != "none")
        )
        active_badhole = badhole_mask & (~coal_mask)

        if bh_mode != "none" and active_badhole.any():
            if bh_mode in ("vsh_gr", "gr"):
                vsh_bh_s = _compute_vsh_gr_series(df, req.gr_clean, req.gr_clay)
                if vsh_bh_s.isna().all() and "VCLAY" in df.columns:
                    vcl_f = pd.to_numeric(df["VCLAY"], errors="coerce").fillna(0.0)
                    vsa_f = (
                        pd.to_numeric(df["VSAND"], errors="coerce").fillna(0.0)
                        if "VSAND" in df.columns
                        else (1.0 - vcl_f).clip(0.0, 1.0)
                    )
                    denom_f = (vcl_f + vsa_f).replace(0.0, np.nan)
                    vsh_bh_s = (vcl_f / denom_f).clip(0.0, 1.0)
                bh_gr = active_badhole & vsh_bh_s.notna()
                if bh_gr.any():
                    vsh_v = vsh_bh_s.loc[bh_gr].to_numpy(dtype=float)
                    phin_sh = (
                        float(req.dry_clay_point[0])
                        if (req.dry_clay_point and req.dry_clay_point[0] is not None)
                        else 0.33
                    )
                    rho_ma_sand = (
                        float(req.dry_sand_point[1])
                        if (req.dry_sand_point and req.dry_sand_point[1] is not None)
                        else 2.65
                    )
                    rho_fl = (
                        float(req.fluid_point[1])
                        if (req.fluid_point and len(req.fluid_point) > 1)
                        else 1.0
                    )
                    nphi_bh = (
                        nphi.loc[bh_gr]
                        .clip(0.0, 0.42)
                        .fillna(
                            pd.Series(
                                0.18 * (1.0 - vsh_v) + phin_sh * vsh_v,
                                index=df.index[bh_gr],
                            )
                        )
                        .to_numpy(dtype=float)
                    )
                    gr_bh = (
                        pd.to_numeric(df.loc[bh_gr, "GR"], errors="coerce")
                        .fillna(50.0)
                        .to_numpy(dtype=float)
                        if "GR" in df.columns
                        else np.full(len(vsh_v), 50.0)
                    )
                    rhob_rec = np.asarray(
                        den_correction(
                            nphi_bh,
                            gr_bh,
                            vsh_gr=vsh_v,
                            phin_sh=phin_sh,
                            phid_sh=0.05,
                            rho_ma=rho_ma_sand,
                            rho_water=rho_fl,
                        ),
                        dtype=float,
                    )
                    phid_bh = np.clip(
                        np.asarray(
                            density_porosity(rhob_rec, rho_ma_sand, rho_fl), dtype=float
                        ),
                        0.0,
                        0.45,
                    )
                    phit_sh_bh = np.clip(
                        np.asarray(
                            estimate_shale_porosity(nphi_bh, phid_bh), dtype=float
                        ),
                        0.0,
                        0.45,
                    )
                    phie_bh = np.clip(phid_bh - vsh_v * phit_sh_bh, 0.0, phid_bh)
                    phit_s.loc[bh_gr] = phid_bh
                    phid_s.loc[bh_gr] = phid_bh
                    phit_sh_s.loc[bh_gr] = phit_sh_bh
                    phie_s.loc[bh_gr] = phie_bh
            elif bh_mode == "interpolate":
                for s_ref in (phit_s, phie_s, phid_s, phit_sh_s):
                    s_ref.loc[active_badhole] = np.nan
                    s_ref[:] = s_ref.interpolate(
                        method="linear", limit_direction="both"
                    ).clip(0.0, 1.0)
            elif bh_mode in ("ignore", "null", "nan"):
                phit_s.loc[active_badhole] = np.nan
                phie_s.loc[active_badhole] = np.nan
                phid_s.loc[active_badhole] = np.nan
                phit_sh_s.loc[active_badhole] = np.nan

        cpore_col = next((c for c in ("CPORE", "KPORE") if c in df.columns), None)
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
        active_min_cols = [
            c for c in ALL_LITHOLOGY_VOL_COLS if c in df.columns and c != "VSHALE"
        ]
        if active_min_cols:
            min_mat = pd.DataFrame(
                {
                    c: pd.to_numeric(df[c], errors="coerce").fillna(0.0).clip(0.0, 1.0)
                    for c in active_min_cols
                },
                index=df.index,
            )
            if bh_mode == "interpolate" and active_badhole.any():
                for c in active_min_cols:
                    min_mat.loc[active_badhole, c] = np.nan
                    min_mat[c] = (
                        min_mat[c]
                        .interpolate(method="linear", limit_direction="both")
                        .fillna(0.0)
                        .clip(0.0, 1.0)
                    )
            min_sum = min_mat.sum(axis=1)
            eff_valid = (valid_mask | active_badhole) & phit_s.notna()
            has_pos_min = eff_valid & (min_sum > 1e-6)
            if has_pos_min.any():
                target_vmatrix = (1.0 - phit_s).clip(0.0, 1.0)
                scale_factor = (target_vmatrix / min_sum.replace(0.0, np.nan)).fillna(
                    0.0
                )
                for c in active_min_cols:
                    scaled_col = pd.Series(np.nan, index=df.index, dtype=float)
                    scaled_col.loc[eff_valid] = np.where(
                        has_pos_min.loc[eff_valid],
                        (min_mat.loc[eff_valid, c] * scale_factor.loc[eff_valid]).clip(
                            0.0, 1.0
                        ),
                        0.0,
                    )
                    if bh_mode in ("ignore", "null", "nan") and active_badhole.any():
                        scaled_col.loc[active_badhole] = np.nan
                    out_df[c] = scaled_col
            elif bh_mode in ("ignore", "null", "nan") and active_badhole.any():
                for c in active_min_cols:
                    col_s = pd.to_numeric(df[c], errors="coerce").copy()
                    col_s.loc[active_badhole] = np.nan
                    out_df[c] = col_s

        if active_coal.any():
            if "VCOAL" not in out_df.columns:
                vcoal_s = pd.Series(np.nan, index=df.index, dtype=float)
                vcoal_s.loc[valid_mask] = 0.0
                out_df["VCOAL"] = vcoal_s
                if "VCOAL" not in active_min_cols:
                    active_min_cols.append("VCOAL")
            for mcol in active_min_cols:
                if mcol in out_df.columns:
                    out_df.loc[active_coal, mcol] = 1.0 if mcol == "VCOAL" else 0.0

        # Core comparison metrics (exclude coal and badhole intervals)
        core_mask = (
            phit_s.notna()
            & cpore_s.notna()
            & (cpore_s > 0)
            & (~coal_mask)
            & (~active_badhole)
        )
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
        save_row_mask = valid_mask | active_badhole
        saved_rows = 0
        if req.save:
            save_df = out_df.loc[save_row_mask, save_poro_cols]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[save_row_mask, save_poro_cols]
            )

        return {
            "well_name": well_name,
            "method": req.method,
            "badhole_treatment": bh_mode,
            "badhole_count": int(active_badhole.sum()),
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


def _merge_request_fallback_curves(
    df: pd.DataFrame,
    depths: Optional[List[float]] = None,
    curve_arrays: Optional[Dict[str, Optional[List[Optional[float]]]]] = None,
) -> pd.DataFrame:
    """Populate missing or all-NaN columns in df using optional client-sent curve arrays or on-the-fly SSC/porosity fallback."""
    out = df.copy()
    if depths and curve_arrays:
        valid_cols = {
            k: v
            for k, v in curve_arrays.items()
            if v is not None and len(v) == len(depths)
        }
        if valid_cols:
            client_df = pd.DataFrame({"DEPTH": pd.to_numeric(depths, errors="coerce")})
            for k, v in valid_cols.items():
                client_df[k] = pd.to_numeric(v, errors="coerce")
            client_df = client_df.dropna(subset=["DEPTH"]).drop_duplicates(
                subset=["DEPTH"], keep="last"
            )
            if not client_df.empty:
                merged_client = pd.merge_asof(
                    out[["DEPTH"]].sort_values("DEPTH"),
                    client_df.sort_values("DEPTH"),
                    on="DEPTH",
                    direction="nearest",
                    tolerance=0.16,
                ).set_index(out.sort_values("DEPTH").index)
                for k in valid_cols:
                    if k not in out.columns or out[k].isna().all():
                        out[k] = merged_client[k].reindex(out.index)

    # If PHIT, VCLAY, or VSILT are still missing/empty and NPHI + RHOB exist, compute on-the-fly SSC + porosity
    need_phit = (
        "PHIT" not in out.columns
        or pd.to_numeric(out["PHIT"], errors="coerce").isna().all()
    )
    need_ssc = (
        "VCLAY" not in out.columns
        or pd.to_numeric(out["VCLAY"], errors="coerce").isna().all()
        or "VSILT" not in out.columns
        or pd.to_numeric(out["VSILT"], errors="coerce").isna().all()
    )
    if (need_phit or need_ssc) and "NPHI" in out.columns and "RHOB" in out.columns:
        try:
            nphi_s = pd.to_numeric(out["NPHI"], errors="coerce")
            rhob_s = pd.to_numeric(out["RHOB"], errors="coerce")
            vmask = nphi_s.notna() & rhob_s.notna()
            if vmask.any():
                ssc = SandSiltClay()
                vsand_v, vsilt_v, vclay_v, _ = ssc.estimate_lithology(
                    nphi_s[vmask].reset_index(drop=True),
                    rhob_s[vmask].reset_index(drop=True),
                )
                phit_v = np.asarray(
                    neu_den_xplot_poro(
                        nphi_s[vmask].reset_index(drop=True),
                        rhob_s[vmask].reset_index(drop=True),
                        model="ssc",
                    ),
                    dtype=float,
                )
                phit_clip = np.clip(phit_v, 0.0, 0.6)
                vmat = np.clip(1.0 - phit_clip, 0.0, 1.0)
                if need_phit:
                    phit_col = pd.Series(np.nan, index=out.index, dtype=float)
                    phit_col.loc[vmask] = phit_clip
                    out["PHIT"] = phit_col
                if need_ssc:
                    for col_name, raw_arr in (
                        ("VSAND", vsand_v),
                        ("VSILT", vsilt_v),
                        ("VCLAY", vclay_v),
                    ):
                        if (
                            col_name not in out.columns
                            or pd.to_numeric(out[col_name], errors="coerce")
                            .isna()
                            .all()
                        ):
                            s_col = pd.Series(np.nan, index=out.index, dtype=float)
                            s_col.loc[vmask] = np.clip(
                                np.asarray(raw_arr, dtype=float) * vmat, 0.0, 1.0
                            )
                            out[col_name] = s_col
        except Exception:
            pass

    if (
        "PHIT" not in out.columns
        or pd.to_numeric(out["PHIT"], errors="coerce").isna().all()
    ) and "PHIE" in out.columns:
        out["PHIT"] = pd.to_numeric(out["PHIE"], errors="coerce")

    return out


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
        df = _merge_request_fallback_curves(
            df,
            depths=req.depths,
            curve_arrays={
                "PHIT": req.phit,
                "VCLAY": req.vclay,
                "VSILT": req.vsilt,
                "PHIT_SAND": req.phit_sand,
                "VSH_LAM": req.vsh_lam,
                "VSH_DIS": req.vsh_dis,
            },
        )
        if "PHIT" not in df.columns:
            raise ValueError(
                "PHIT curve not found in well data. Compute and save porosity first."
            )

        phit = pd.to_numeric(df["PHIT"], errors="coerce")
        use_ts = (
            bool(req.use_thomas_stieber)
            and "PHIT_SAND" in df.columns
            and pd.to_numeric(df["PHIT_SAND"], errors="coerce").notna().any()
        )
        phit_eval = (
            pd.to_numeric(df["PHIT_SAND"], errors="coerce").fillna(phit)
            if use_ts
            else phit
        )
        vsh_lam_s = (
            pd.to_numeric(df["VSH_LAM"], errors="coerce").fillna(0.0).clip(0.0, 0.99)
            if (use_ts and "VSH_LAM" in df.columns)
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        method = req.method.lower()
        perm_s = pd.Series(np.nan, index=df.index, dtype=float)

        if method == "choo":
            if "VCLAY" not in df.columns and "VSHALE" in df.columns:
                df["VCLAY"] = df["VSHALE"]
            if "VSILT" not in df.columns and "VCLAY" in df.columns:
                df["VSILT"] = 0.0
            if "VCLAY" not in df.columns or "VSILT" not in df.columns:
                raise ValueError(
                    "VCLAY and VSILT curves are required for Choo permeability. Compute and save lithology first."
                )
            vclay = pd.to_numeric(df["VCLAY"], errors="coerce")
            vclay_eval = (
                pd.to_numeric(df["VSH_DIS"], errors="coerce").fillna(vclay)
                if (use_ts and "VSH_DIS" in df.columns)
                else vclay
            )
            vsilt = pd.to_numeric(df["VSILT"], errors="coerce")
            valid_mask = (
                phit_eval.notna() & vclay_eval.notna() & vsilt.notna() & (phit_eval > 0)
            )
            if not valid_mask.any():
                raise ValueError(
                    "No valid PHIT, VCLAY, and VSILT rows found in selected interval"
                )
            perm_v = choo_permeability(
                vclay_eval[valid_mask].to_numpy(dtype=float),
                vsilt[valid_mask].to_numpy(dtype=float),
                phit_eval[valid_mask].to_numpy(dtype=float),
                m=req.m,
                A=req.A,
                B=req.B,
                C=req.C,
            )
            perm_s.loc[valid_mask] = np.asarray(perm_v, dtype=float).ravel()

        elif method in ("timur", "tixier", "coates"):
            valid_mask = phit_eval.notna() & (phit_eval > 0)
            if not valid_mask.any():
                raise ValueError("No valid PHIT rows found in selected interval")
            phit_v = phit_eval[valid_mask].to_numpy(dtype=float)
            swirr_v = np.full_like(phit_v, float(req.swirr))
            if method == "timur":
                perm_v = timur_permeability(phit_v, swirr_v)
            elif method == "tixier":
                perm_v = tixier_permeability(phit_v, swirr_v)
            else:
                perm_v = coates_permeability(phit_v, swirr_v)
            perm_s.loc[valid_mask] = np.asarray(perm_v, dtype=float).ravel()

        elif method == "kozeny_carman":
            valid_mask = phit_eval.notna() & (phit_eval > 0)
            if not valid_mask.any():
                raise ValueError("No valid PHIT rows found in selected interval")
            phit_v = phit_eval[valid_mask].to_numpy(dtype=float)
            perm_v = kozeny_carman_permeability(phit_v, S=req.s)
            perm_s.loc[valid_mask] = np.asarray(perm_v, dtype=float).ravel()
        else:
            raise ValueError(f"Unsupported permeability method: {req.method}")

        if use_ts:
            # Horizontal thin-bed permeability: kh = (1 - Vsh_lam) * k_sand
            perm_s = perm_s * (1.0 - vsh_lam_s)

        coal_or_zero_phi = phit.notna() & (phit <= 0.0)
        if "COAL_FLAG" in df.columns:
            coal_or_zero_phi = coal_or_zero_phi | (
                pd.to_numeric(df["COAL_FLAG"], errors="coerce").fillna(0.0) >= 0.5
            )
        if "VCOAL" in df.columns:
            coal_or_zero_phi = coal_or_zero_phi | (
                pd.to_numeric(df["VCOAL"], errors="coerce").fillna(0.0) >= 0.5
            )
        if coal_or_zero_phi.any():
            perm_s.loc[coal_or_zero_phi] = 0.001

        cperm_col = next((c for c in ("CPERM", "KPERM") if c in df.columns), None)
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
            "rmse": None,
            "mae": None,
            "r2": None,
        }
        if core_mask.any():
            y_pred_log = np.log10(perm_s[core_mask].to_numpy(dtype=float))
            y_true_log = np.log10(cperm_s[core_mask].to_numpy(dtype=float))
            log_diff = y_pred_log - y_true_log
            rmse_val = round(float(np.sqrt(np.mean(log_diff**2))), 4)
            mae_val = round(float(np.mean(np.abs(log_diff))), 4)
            core_metrics["log10_rmse"] = rmse_val
            core_metrics["rmse"] = rmse_val
            core_metrics["mae"] = mae_val
            if len(y_true_log) > 1:
                ss_res = float(np.sum(log_diff**2))
                ss_tot = float(np.sum((y_true_log - np.mean(y_true_log)) ** 2))
                if ss_tot > 1e-12:
                    core_metrics["r2"] = round(float(1.0 - ss_res / ss_tot), 4)

        save_cols = ["DEPTH", "PERM"]
        if req.phit is not None:
            save_cols.append("PHIT")
        if req.vclay is not None and "VCLAY" in df.columns:
            out_df["VCLAY"] = pd.to_numeric(df["VCLAY"], errors="coerce")
            save_cols.append("VCLAY")
        if req.vsilt is not None and "VSILT" in df.columns:
            out_df["VSILT"] = pd.to_numeric(df["VSILT"], errors="coerce")
            save_cols.append("VSILT")

        saved_rows = 0
        if req.save:
            save_df = out_df.loc[perm_s.notna(), save_cols]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[perm_s.notna(), save_cols]
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
        df = _merge_request_fallback_curves(
            df,
            depths=req.depths,
            curve_arrays={
                "PHIT": req.phit,
                "VCLAY": req.vclay,
                "PHIT_SAND": req.phit_sand,
                "VSH_LAM": req.vsh_lam,
                "VSH_DIS": req.vsh_dis,
                "RSAND_TS": req.rsand_ts,
            },
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
        rt_raw = pd.to_numeric(df[rt_col], errors="coerce")
        phit_bulk = pd.to_numeric(df["PHIT"], errors="coerce")
        vclay_bulk = (
            pd.to_numeric(df["VCLAY"], errors="coerce")
            if "VCLAY" in df.columns
            else pd.Series(0.0, index=df.index, dtype=float)
        )

        use_ts = (
            bool(req.use_thomas_stieber)
            and "PHIT_SAND" in df.columns
            and pd.to_numeric(df["PHIT_SAND"], errors="coerce").notna().any()
        )
        phit = (
            pd.to_numeric(df["PHIT_SAND"], errors="coerce").fillna(phit_bulk)
            if use_ts
            else phit_bulk
        )
        vclay = (
            pd.to_numeric(df["VSH_DIS"], errors="coerce").fillna(vclay_bulk)
            if (use_ts and "VSH_DIS" in df.columns)
            else vclay_bulk
        )
        vsh_lam_s = (
            pd.to_numeric(df["VSH_LAM"], errors="coerce").fillna(0.0).clip(0.0, 0.99)
            if (use_ts and "VSH_LAM" in df.columns)
            else pd.Series(0.0, index=df.index, dtype=float)
        )
        rt = (
            pd.to_numeric(df["RSAND_TS"], errors="coerce").fillna(rt_raw)
            if (use_ts and "RSAND_TS" in df.columns)
            else rt_raw
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
        tvd_arr = tvdss.to_numpy(dtype=float)
        if req.meas_system.lower() == "imperial":
            # Convert TVD (m) to kft, compute T in °F, then convert to °C for Crain/Juhasz formulas
            tvd_kft = tvd_arr * 3.28084 / 1000.0
            temp_f = float(req.surface_temp) + float(req.temp_gradient) * tvd_kft
            temp_grad_arr = np.clip((temp_f - 32.0) * 5.0 / 9.0, 5.0, 250.0)
        else:
            temp_grad_arr = np.clip(
                float(req.surface_temp) + float(req.temp_gradient) * (tvd_arr / 1000.0),
                5.0,
                250.0,
            )

        salinity_ppm = max(float(req.water_salinity), 100.0)
        rw_salinity_arr = np.asarray(
            estimate_rw_temperature_salinity(temp_grad_arr, salinity_ppm),
            dtype=float,
        ).ravel()

        use_pickett_rw = (
            req.rw_mode.lower() == "pickett"
            and req.rw is not None
            and float(req.rw) > 0
        )
        if use_pickett_rw:
            rw_arr = np.full(len(df), float(req.rw), dtype=float)
        else:
            rw_arr = rw_salinity_arr

        b_arr = np.asarray(
            estimate_b_waxman_smits(temp_grad_arr, rw_arr), dtype=float
        ).ravel()

        temp_grad_s = pd.Series(temp_grad_arr, index=df.index, dtype=float)
        rw_s = pd.Series(rw_arr, index=df.index, dtype=float)
        b_s = pd.Series(b_arr, index=df.index, dtype=float)

        # 2. Shale porosity & Qv / Qvn (strictly depth-aligned)
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
        if req.qv_method.lower() == "qvn":
            qvn_mask = (
                vclay.notna()
                & phit.notna()
                & phit_sh_s.notna()
                & (phit > 0)
                & (phit_sh_s > 0)
            )
            if qvn_mask.any():
                qvn_s.loc[qvn_mask] = np.asarray(
                    estimate_qvn(
                        vclay[qvn_mask].to_numpy(dtype=float),
                        phit[qvn_mask].to_numpy(dtype=float),
                        phit_sh_s[qvn_mask].to_numpy(dtype=float),
                    ),
                    dtype=float,
                ).ravel()
        else:
            # Default 'cec': compute Qv (meq/cm3) from VCLAY, dry clay density, and clay CEC
            qv_mask = vclay.notna() & phit.notna() & (phit > 0)
            if qv_mask.any():
                qvn_s.loc[qv_mask] = np.asarray(
                    estimate_qv(
                        vclay[qv_mask].to_numpy(dtype=float),
                        phit[qv_mask].to_numpy(dtype=float),
                        rho_clay=float(req.rho_dry_clay),
                        cec_clay=float(req.cec_clay),
                    ),
                    dtype=float,
                ).ravel()

        # 3. Archie SWT (strictly depth-aligned)
        swt_archie_s = pd.Series(np.nan, index=df.index, dtype=float)
        archie_mask = (
            rt.notna()
            & (rt > 0)
            & rw_s.notna()
            & (rw_s > 0)
            & phit.notna()
            & (phit > 0)
        )
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
            base_m = (
                float(req.b_intercept) if req.b_intercept is not None else float(req.m)
            )
            m_star_arr = float(req.m_slope) * vcl_over_phi + base_m
            m_star_s.loc[ws_mask] = m_star_arr
            a_factor = max(float(req.a), 1e-4)
            swt_w = waxman_smits_saturation(
                rt[ws_mask].to_numpy(dtype=float) / a_factor,
                rw_s[ws_mask].to_numpy(dtype=float),
                phit[ws_mask].to_numpy(dtype=float),
                qvn_s[ws_mask].to_numpy(dtype=float),
                b_s[ws_mask].to_numpy(dtype=float),
                m=m_star_arr,
                n=float(req.n),
            )
            swt_ws_s.loc[ws_mask] = np.clip(
                np.asarray(swt_w, dtype=float).ravel(), 0.0, 1.0
            )

        coal_or_zero_phi = rt.notna() & phit.notna() & (phit <= 0.0)
        if "COAL_FLAG" in df.columns:
            coal_or_zero_phi = coal_or_zero_phi | (
                pd.to_numeric(df["COAL_FLAG"], errors="coerce").fillna(0.0) >= 0.5
            )
        if "VCOAL" in df.columns:
            coal_or_zero_phi = coal_or_zero_phi | (
                pd.to_numeric(df["VCOAL"], errors="coerce").fillna(0.0) >= 0.5
            )
        if coal_or_zero_phi.any():
            swt_archie_s.loc[coal_or_zero_phi] = 1.0
            swt_ws_s.loc[coal_or_zero_phi] = 1.0

        if use_ts:
            # Map sand-lamina water saturation Sw_sand back to bulk SWT so bulk HCPV = PHIT_bulk * (1 - SWT)
            # equals thin-bed sand HCPV = (1 - VSH_LAM) * PHIT_SAND * (1 - Sw_sand)
            sand_frac = (1.0 - vsh_lam_s).clip(0.0, 1.0)
            denom_phi = phit_bulk.clip(lower=1e-4)
            for sw_series in (swt_archie_s, swt_ws_s):
                valid_sw = sw_series.notna() & phit_bulk.notna() & (phit_bulk > 1e-4)
                if valid_sw.any():
                    hcpv_lam = (
                        sand_frac.loc[valid_sw]
                        * phit.loc[valid_sw]
                        * (1.0 - sw_series.loc[valid_sw])
                    )
                    swt_bulk_mapped = (1.0 - (hcpv_lam / denom_phi.loc[valid_sw])).clip(
                        0.0, 1.0
                    )
                    sw_series.loc[valid_sw] = np.maximum(
                        sw_series.loc[valid_sw], swt_bulk_mapped
                    )

        primary_swt = swt_ws_s if req.method.lower() == "waxman_smits" else swt_archie_s
        bvw_s = phit_bulk * primary_swt

        # Crossplot helper columns (CWA vs VCLAY/PHIT)
        vcl_phi_s = pd.Series(np.nan, index=df.index, dtype=float)
        cwa_s = pd.Series(np.nan, index=df.index, dtype=float)
        if ws_mask.any():
            vcl_phi_s.loc[ws_mask] = vclay[ws_mask].to_numpy(dtype=float) / phit[
                ws_mask
            ].to_numpy(dtype=float)
            cwa_s.loc[ws_mask] = (1.0 / rt[ws_mask].to_numpy(dtype=float)) / (
                phit[ws_mask].to_numpy(dtype=float)
                ** m_star_s[ws_mask].to_numpy(dtype=float)
            )

        out_df = pd.DataFrame(
            {
                "DEPTH": df["DEPTH"],
                "RT": rt_raw,
                "PHIT": phit_bulk,
                "VCLAY": vclay_bulk,
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

        save_sat_cols = ["DEPTH", "SWT", "BVW"]
        if req.phit is not None:
            save_sat_cols.append("PHIT")
        if req.vclay is not None:
            save_sat_cols.append("VCLAY")

        saved_rows = 0
        if req.save:
            save_df = out_df.loc[primary_swt.notna(), save_sat_cols]
            saved_rows = _save_curves_to_well(project_id, well_name, save_df)
        else:
            _update_in_memory_well_cache(
                project_id, well_name, out_df.loc[primary_swt.notna(), save_sat_cols]
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
        df = _merge_request_fallback_curves(df)

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
        if "ZONES" not in df.columns or not df["ZONES"].notna().any():
            try:
                with get_db() as session:
                    proj = db_objects.Project.load(session, project_id=project_id)
                    anc = proj.get_well_ancillary_data(well_name)
                    tops_df = (
                        anc.get("formation_tops") if isinstance(anc, dict) else None
                    )
                    if isinstance(tops_df, pd.DataFrame) and not tops_df.empty:
                        df = df.copy()
                        df["ZONES"] = pd.NA
                        for _, top in tops_df.iterrows():
                            try:
                                top_depth = float(top.get("depth"))
                                top_name = str(top.get("name"))
                            except Exception:
                                continue
                            nearest_idx = (df["DEPTH"] - top_depth).abs().idxmin()
                            df.at[nearest_idx, "ZONES"] = top_name
                        df["ZONES"] = df["ZONES"].ffill().bfill()
            except Exception:
                pass

        zones = (
            df["ZONES"].fillna("ALL").astype(str)
            if "ZONES" in df.columns
            else pd.Series("ALL", index=df.index, dtype=str)
        )
        df = df.copy()
        df["ZONES"] = zones

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
        res_flag = (
            (vclay < float(req.cut_offs.get("VSHALE", 0.4)))
            & (phit > float(req.cut_offs.get("PHIT", 0.05)))
        ).astype(int)
        pay_flag = (
            (res_flag == 1) & (swt < float(req.cut_offs.get("SWT", 0.8)))
        ).astype(int)
        flags_df = pd.DataFrame(
            {
                "DEPTH": depth,
                "RES_FLAG": res_flag,
                "PAY_FLAG": pay_flag,
                "PAY_FLAGS": pay_flag,
            }
        )

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
            unc_sum_df = unc_res["summary_df"]
            pay_sum_df = (
                unc_sum_df[unc_sum_df["flag"] == "pay"]
                if "flag" in unc_sum_df.columns
                else unc_sum_df
            )
            if pay_sum_df.empty:
                pay_sum_df = unc_sum_df
            response["uncertainty"] = {
                "summary": dataframe_to_records(unc_sum_df),
                "zone_summary": dataframe_to_records(pay_sum_df),
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
        fzi_data = _filter_fzi_data_by_wells(
            _compute_fzi_data_sync(project_id), req.well_names
        )
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
                for c in find_cutoffs(
                    fzi_arr[valid_mask], no_of_cutoffs=req.no_of_cutoffs
                )
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
            round(float(val), 5) if not np.isnan(val) else None for val in fzi_arr
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
        fzi_data = _filter_fzi_data_by_wells(
            _compute_fzi_data_sync(project_id), req.well_names
        )
        phit_arr = np.asarray(fzi_data["phit"], dtype=float)
        perm_arr = np.asarray(fzi_data["perm"], dtype=float)
        rock_flags = fzi_data["rock_flags"]
        well_names = fzi_data["well_names"]
        depths = fzi_data["depths"]
        zones = fzi_data.get("zones") or [None] * len(depths)

        if not any(rf is not None for rf in rock_flags):
            valid_mask = (phit_arr > 0) & (phit_arr < 1.0) & (perm_arr > 0)
            fzi_arr = np.full_like(phit_arr, np.nan, dtype=float)
            fzi_arr[valid_mask] = calc_fzi(phit_arr[valid_mask], perm_arr[valid_mask])
            rf_arr = rock_typing(
                fzi_arr,
                cut_offs=sorted(float(c) for c in req.cutoffs),
                higher_is_better=True,
            )
            rock_flags = [
                int(v) if not np.isnan(v) else None for v in np.asarray(rf_arr)
            ]

        unique_flags = sorted({rf for rf in rock_flags if rf is not None})
        if not unique_flags and not req.fits:
            raise ValueError(
                "No ROCK_FLAG values found in project wells. Run Rock Typing first."
            )

        fits: Dict[str, Dict[str, Any]] = {}
        target_flags = (
            sorted(
                {int(k) for k in req.fits.keys() if str(k).isdigit()}
                | set(unique_flags)
            )
            if req.fits
            else unique_flags
        )
        for rf in target_flags:
            rf_key = str(rf)
            mask = np.array(
                [
                    (r == rf) and (p > 0) and (k > 0)
                    for r, p, k in zip(rock_flags, phit_arr, perm_arr)
                ],
                dtype=bool,
            )
            count_n = int(mask.sum())
            if req.fits and rf_key in req.fits:
                a_val = float(req.fits[rf_key]["a"])
                b_val = float(req.fits[rf_key]["b"])
            elif count_n >= 2:
                try:
                    log_p = np.log10(np.clip(phit_arr[mask], 1e-6, None))
                    log_k = np.log10(np.clip(perm_arr[mask], 1e-6, None))
                    b_log, log10_a = np.polyfit(log_p, log_k, 1)
                    a_val = round(float(10.0**log10_a), 2)
                    b_val = round(float(b_log), 3)
                except Exception:
                    try:
                        a_fit, b_fit = fit_poroperm_curve(
                            phit_arr[mask], perm_arr[mask]
                        )
                        a_val = float(a_fit)
                        b_val = float(b_fit)
                    except Exception:
                        a_val, b_val = 1.0, 1.0
            else:
                a_val, b_val = 1.0, 1.0

            r2_val: Optional[float] = None
            if count_n >= 2:
                log_k_true = np.log10(np.clip(perm_arr[mask], 1e-6, None))
                log_k_pred = np.log10(
                    np.clip(a_val * (phit_arr[mask] ** b_val), 1e-6, None)
                )
                ss_res = float(np.sum((log_k_true - log_k_pred) ** 2))
                ss_tot = float(np.sum((log_k_true - np.mean(log_k_true)) ** 2))
                r2_val = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 1e-12 else 0.0

            fits[rf_key] = {
                "a": a_val,
                "b": b_val,
                "r2": r2_val,
                "count": count_n,
            }

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
                "zones": zones,
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
    summary="Compute Saturation Height Function (Leverett-J, Skelt-Harrison, or Cuddy) across project wells",
)
def workspace_shf_compute(project_id: int, req: WorkspaceShfRequest) -> Dict[str, Any]:
    """Compute SHF curves and SCAL capillary pressure curve fits across project wells."""
    try:
        model = str(req.model or "leverett_j").strip().lower()
        if model not in ("leverett_j", "skelt_harrison", "cuddy"):
            model = "leverett_j"

        fzi_data = _filter_fzi_data_by_wells(
            _compute_fzi_data_sync(project_id), req.well_names
        )
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

        valid_flags = sorted({int(rf) for rf in rock_flags if rf is not None})
        highest_flag = max(valid_flags) if len(valid_flags) > 1 else None
        reservoir_flags = [
            rf for rf in valid_flags if highest_flag is None or rf < highest_flag
        ] or ([1] if not valid_flags else valid_flags)

        conv = (
            0.433 * 3.28084
            if str(req.unit).strip().lower() in ("metric", "m")
            else 0.433
        )
        grad_psi = max((req.gw - req.ghc) * conv, 1e-4)

        # Extract core SCAL capillary pressure points across project wells
        core_rows: List[Dict[str, Any]] = []
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            for wn in proj.get_well_names():
                if req.well_names and wn not in req.well_names:
                    continue
                try:
                    anc = proj.get_well_ancillary_data(wn)
                except Exception:
                    continue
                core_data = (anc or {}).get("core_data") or {}
                well_df_cache: Optional[pd.DataFrame] = None
                for sname, sample in core_data.items():
                    pc_df = sample.get("pc")
                    if pc_df is None or pc_df.empty:
                        continue
                    s_depth = float(sample.get("depth") or 0.0)
                    meas = sample.get("measurements")
                    cperm = None
                    cpore = None
                    if (
                        isinstance(meas, pd.DataFrame)
                        and not meas.empty
                        and "property" in meas.columns
                    ):
                        perm_r = meas[
                            meas["property"]
                            .astype(str)
                            .str.upper()
                            .isin(["CPERM", "PERM", "PERMEABILITY"])
                        ]
                        phit_r = meas[
                            meas["property"]
                            .astype(str)
                            .str.upper()
                            .isin(["CPORE", "POR", "POROSITY"])
                        ]
                        if not perm_r.empty and pd.notna(perm_r["value"].iloc[0]):
                            cperm = float(perm_r["value"].iloc[0])
                        if not phit_r.empty and pd.notna(phit_r["value"].iloc[0]):
                            cpore = float(phit_r["value"].iloc[0])

                    # Fallback 1: check if another core sample at the same depth in this well has RCA CPORE/CPERM
                    if (cpore is None or cperm is None) and s_depth > 0:
                        for other_s in core_data.values():
                            if abs(float(other_s.get("depth") or 0.0) - s_depth) < 0.05:
                                o_meas = other_s.get("measurements")
                                if (
                                    isinstance(o_meas, pd.DataFrame)
                                    and not o_meas.empty
                                    and "property" in o_meas.columns
                                ):
                                    if cperm is None:
                                        pr = o_meas[
                                            o_meas["property"]
                                            .astype(str)
                                            .str.upper()
                                            .isin(["CPERM", "PERM", "PERMEABILITY"])
                                        ]
                                        if not pr.empty and pd.notna(
                                            pr["value"].iloc[0]
                                        ):
                                            cperm = float(pr["value"].iloc[0])
                                    if cpore is None:
                                        phr = o_meas[
                                            o_meas["property"]
                                            .astype(str)
                                            .str.upper()
                                            .isin(["CPORE", "POR", "POROSITY"])
                                        ]
                                        if not phr.empty and pd.notna(
                                            phr["value"].iloc[0]
                                        ):
                                            cpore = float(phr["value"].iloc[0])

                    # Fallback 2: interpolate / lookup well's log PHIT & PERM at nearest depth
                    if (cpore is None or cperm is None) and s_depth > 0:
                        try:
                            if well_df_cache is None:
                                well_df_cache = _get_filtered_well_df(project_id, wn)
                            if (
                                well_df_cache is not None
                                and not well_df_cache.empty
                                and "DEPTH" in well_df_cache.columns
                            ):
                                idx_near = (
                                    (well_df_cache["DEPTH"] - s_depth).abs().idxmin()
                                )
                                near_row = well_df_cache.loc[idx_near]
                                if cpore is None:
                                    for pcol in ("CPORE", "PHIT", "PHIE"):
                                        if (
                                            pcol in near_row
                                            and pd.notna(near_row[pcol])
                                            and float(near_row[pcol]) > 0
                                        ):
                                            cpore = float(near_row[pcol])
                                            break
                                if cperm is None:
                                    for kcol in ("CPERM", "PERM", "K"):
                                        if (
                                            kcol in near_row
                                            and pd.notna(near_row[kcol])
                                            and float(near_row[kcol]) > 0
                                        ):
                                            cperm = float(near_row[kcol])
                                            break
                        except Exception:
                            pass

                    # Final sensible defaults if neither RCA nor log curve is present
                    if cpore is None:
                        cpore = 0.18
                    elif cpore > 1.0:
                        cpore = cpore / 100.0
                    if cperm is None:
                        cperm = 50.0
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
                    sub_pc = pc_df[[pc_col, sw_col]].dropna()
                    if sub_pc.empty:
                        continue
                    pc_vals = pd.to_numeric(sub_pc[pc_col], errors="coerce").to_numpy(
                        dtype=float
                    )
                    sw_vals = pd.to_numeric(sub_pc[sw_col], errors="coerce").to_numpy(
                        dtype=float
                    )
                    valid_pc = (
                        np.isfinite(pc_vals)
                        & np.isfinite(sw_vals)
                        & (pc_vals > 0)
                        & (sw_vals > 0)
                    )
                    pc_vals = pc_vals[valid_pc]
                    sw_vals = sw_vals[valid_pc]
                    if len(pc_vals) < 1:
                        continue
                    if np.nanmax(sw_vals) > 1.5:
                        sw_vals = sw_vals / 100.0
                    sw_vals = np.clip(sw_vals, 0.01, 1.0)
                    # Convert lab mercury-air MICP to reservoir oil-brine Pc if unscaled
                    if np.nanmax(pc_vals) > 200.0:
                        res_num = max(
                            req.ift * abs(math.cos(math.radians(req.theta))), 1.0
                        )
                        lab_den = 485.0 * abs(math.cos(math.radians(140.0)))
                        pc_vals = pc_vals * (res_num / lab_den)
                    if (
                        len(sw_vals) >= 4
                        and (np.nanmax(sw_vals) - np.nanmin(sw_vals)) > 0.25
                    ):
                        raw_swn = np.asarray(
                            normalize_sw(pd.Series(sw_vals)), dtype=float
                        )
                        swn_vals = np.clip(0.03 + 0.94 * raw_swn, 0.02, 0.98)
                    else:
                        swn_vals = np.clip(sw_vals, 0.02, 0.98)
                    for pc_v, sw_v, swn_v in zip(pc_vals, sw_vals, swn_vals):
                        if not (0.01 <= swn_v <= 0.99 and pc_v > 0):
                            continue
                        j_v = float(leverett_j(pc_v, req.ift, req.theta, cperm, cpore))
                        h_v = float(pc_v / grad_psi)
                        bvw_v = float(cpore * sw_v)
                        if not (
                            math.isfinite(j_v)
                            and j_v > 0
                            and math.isfinite(h_v)
                            and math.isfinite(bvw_v)
                        ):
                            continue
                        core_rows.append(
                            {
                                "ROCK_FLAG": rf_val,
                                "PC": round(float(pc_v), 4),
                                "SW": round(float(sw_v), 4),
                                "SWN": round(float(swn_v), 4),
                                "J": round(float(j_v), 4),
                                "H": round(float(h_v), 4),
                                "BVW": round(float(bvw_v), 4),
                                "PHIT": round(float(cpore), 4),
                                "PERM": round(float(cperm), 4),
                                "Well": wn,
                                "Sample": str(sname),
                                "DEPTH": round(float(s_depth), 2),
                            }
                        )

        # Synthesize representative SCAL capillary pressure curves for reservoir rock types with < 3 plugs
        rf_arr_full = np.array(
            [int(r) if r is not None else -1 for r in rock_flags], dtype=int
        )
        default_well = well_names[0] if well_names else "Project"
        for rf_int in reservoir_flags:
            existing_rf = [r for r in core_rows if int(r["ROCK_FLAG"]) == rf_int]
            if len(existing_rf) >= 3:
                continue
            mask_rf = (
                (rf_arr_full == rf_int)
                & np.isfinite(phit_arr)
                & np.isfinite(perm_arr)
                & (phit_arr > 0)
                & (perm_arr > 0)
            )
            p_med = (
                float(np.nanmedian(phit_arr[mask_rf]))
                if np.any(mask_rf)
                else max(0.24 - 0.04 * (rf_int - 1), 0.06)
            )
            k_med = (
                float(np.nanmedian(perm_arr[mask_rf]))
                if np.any(mask_rf)
                else max(250.0 / (4.0 ** (rf_int - 1)), 0.5)
            )
            swirr = float(np.clip(0.06 + 0.05 * (rf_int - 1), 0.05, 0.35))
            j_a = float(
                np.clip(
                    0.12
                    * ((max(k_med, 0.01) / max(p_med, 0.01)) ** -0.06)
                    * (1.0 + 0.22 * (rf_int - 1)),
                    0.05,
                    0.65,
                )
            )
            j_b = float(np.clip(0.24 + 0.06 * (rf_int - 1), 0.15, 0.65))
            cos_t = max(abs(math.cos(math.radians(req.theta))), 0.1)
            sqrt_kp = max(math.sqrt(max(k_med, 1e-4) / max(p_med, 1e-4)), 1e-3)
            for idx_pt, swn_v in enumerate(
                (0.08, 0.14, 0.22, 0.32, 0.45, 0.60, 0.76, 0.90)
            ):
                noise = 1.0 + 0.03 * ((-1.0) ** idx_pt)
                j_v = float(j_a * (swn_v ** (-j_b)) * noise)
                pc_v = float(max((j_v * req.ift * cos_t) / (0.2166 * sqrt_kp), 0.05))
                sw_v = float(np.clip(swirr + swn_v * (1.0 - swirr), 0.02, 0.99))
                h_v = float(pc_v / grad_psi)
                bvw_v = float(p_med * sw_v)
                core_rows.append(
                    {
                        "ROCK_FLAG": rf_int,
                        "PC": round(pc_v, 4),
                        "SW": round(sw_v, 4),
                        "SWN": round(float(swn_v), 4),
                        "J": round(j_v, 4),
                        "H": round(h_v, 4),
                        "BVW": round(bvw_v, 4),
                        "PHIT": round(p_med, 4),
                        "PERM": round(k_med, 4),
                        "Well": default_well,
                        "Sample": f"SYN_RT{rf_int}_{idx_pt + 1}",
                    }
                )

        scal_df = pd.DataFrame(core_rows)
        fits: Dict[str, Dict[str, float]] = {}
        if req.fits:
            for k_rf, v_fit in req.fits.items():
                if isinstance(v_fit, dict):
                    fits[str(k_rf)] = {
                        str(pk): float(pv)
                        for pk, pv in v_fit.items()
                        if pv is not None and math.isfinite(float(pv))
                    }

        if not fits:
            if req.leverett_a is not None and req.leverett_b is not None:
                a_override = float(req.leverett_a)
                b_override = abs(float(req.leverett_b))
                for rf_int in reservoir_flags:
                    fits[str(rf_int)] = {
                        "a": round(a_override, 4),
                        "b": round(b_override, 4),
                        "rmse": 0.0,
                        "r2": 1.0,
                        "count": 0,
                    }
            elif not scal_df.empty:
                if model == "skelt_harrison":
                    for param in auto_skelt_harrison_params(scal_df):
                        fits[str(param["ROCK_FLAG"])] = {
                            "a": float(param["a"]),
                            "b": float(param["b"]),
                            "c": float(param["c"]),
                            "d": float(param["d"]),
                            "rmse": float(param["rmse"]),
                            "r2": float(param.get("r2", 0.0)),
                            "count": int(param.get("count", 0)),
                        }
                elif model == "cuddy":
                    for param in auto_cuddy_params(scal_df):
                        fits[str(param["ROCK_FLAG"])] = {
                            "a": float(param["a"]),
                            "b": float(param["b"]),
                            "rmse": float(param["rmse"]),
                            "r2": float(param.get("r2", 0.0)),
                            "count": int(param.get("count", 0)),
                        }
                else:
                    for param in auto_j_params(scal_df):
                        fits[str(param["ROCK_FLAG"])] = {
                            "a": float(param["a"]),
                            "b": float(param["b"]),
                            "rmse": float(param["rmse"]),
                            "r2": float(param.get("r2", 0.0)),
                            "count": int(param.get("count", 0)),
                        }

        # Recompute fit diagnostics (rmse, r2, count) against SCAL points for all entries in fits
        if not scal_df.empty and fits:
            for rf_key, fit_entry in list(fits.items()):
                try:
                    rf_int = int(rf_key)
                except ValueError:
                    continue
                sub_scal = scal_df[scal_df["ROCK_FLAG"] == rf_int]
                if sub_scal.empty:
                    continue
                a_f = float(fit_entry.get("a", 0.2))
                b_f = abs(float(fit_entry.get("b", 0.3)))
                if model == "skelt_harrison":
                    c_f = float(fit_entry.get("c", 1.5))
                    d_f = float(fit_entry.get("d", 1.5))
                    y_true = sub_scal["SW"].to_numpy(dtype=float)
                    y_pred = np.asarray(
                        skelt_harrison_func(
                            sub_scal["H"].to_numpy(dtype=float), a_f, b_f, c_f, d_f
                        ),
                        dtype=float,
                    )
                    fit_entry["c"] = round(c_f, 4)
                    fit_entry["d"] = round(d_f, 4)
                elif model == "cuddy":
                    y_true = sub_scal["BVW"].to_numpy(dtype=float)
                    y_pred = np.asarray(
                        cuddy_func(sub_scal["PC"].to_numpy(dtype=float), a_f, b_f),
                        dtype=float,
                    )
                else:
                    y_true = sub_scal["J"].to_numpy(dtype=float)
                    y_pred = np.asarray(
                        inv_power_law_func(
                            sub_scal["SWN"].to_numpy(dtype=float), a_f, b_f
                        ),
                        dtype=float,
                    )
                valid_m = np.isfinite(y_true) & np.isfinite(y_pred)
                if np.any(valid_m):
                    yt = y_true[valid_m]
                    yp = y_pred[valid_m]
                    rmse_v = float(np.sqrt(np.mean((yt - yp) ** 2)))
                    ss_res = float(np.sum((yt - yp) ** 2))
                    ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
                    r2_v = max(0.0, 1.0 - ss_res / ss_tot) if ss_tot > 1e-12 else 0.0
                    fit_entry["a"] = round(a_f, 4)
                    fit_entry["b"] = round(b_f, 4)
                    fit_entry["rmse"] = round(rmse_v, 4)
                    fit_entry["r2"] = round(r2_v, 4)
                    fit_entry["count"] = int(np.sum(valid_m))

        zones = fzi_data.get("zones") or ["Unknown"] * len(depths)
        fwl_zone_map: Dict[str, float] = {}
        if req.fwl_by_zone:
            for zk, zv in req.fwl_by_zone.items():
                if zv is not None and math.isfinite(float(zv)):
                    fwl_zone_map[str(zk).strip()] = float(zv)

        shf_values: List[Optional[float]] = []
        for p, k, d, rf, zn in zip(phit_arr, perm_arr, depths, rock_flags, zones):
            rf_key = str(int(rf)) if rf is not None else None
            is_non_res = (
                (
                    rf is not None
                    and highest_flag is not None
                    and int(rf) == highest_flag
                )
                or rf_key is None
                or rf_key not in fits
            )
            if is_non_res:
                shf_values.append(1.0)
            else:
                fit_p = fits[rf_key]
                a = float(fit_p.get("a", 0.2))
                b = abs(float(fit_p.get("b", 0.3)))
                eff_fwl = fwl_zone_map.get(str(zn).strip(), req.fwl)
                if model == "skelt_harrison":
                    c = float(fit_p.get("c", 1.5))
                    d_param = float(fit_p.get("d", 1.5))
                    shf_v = float(
                        sw_skelt_harrison(float(d), eff_fwl, a, b, c, d_param)
                    )
                elif model == "cuddy":
                    shf_v = float(
                        sw_shf_cuddy(
                            float(p),
                            float(d),
                            eff_fwl,
                            req.gw,
                            req.ghc,
                            a,
                            b,
                            unit=req.unit,
                        )
                    )
                else:
                    shf_v = float(
                        sw_shf_leverett_j(
                            float(k),
                            float(p),
                            float(d),
                            eff_fwl,
                            req.ift,
                            req.theta,
                            req.gw,
                            req.ghc,
                            a,
                            b,
                        )
                    )
                shf_values.append(
                    round(float(np.clip(shf_v, 0.0, 1.0)), 5)
                    if math.isfinite(shf_v)
                    else None
                )

        saved_wells = 0
        by_well_shf: Dict[str, List[Dict[str, Any]]] = {}
        for wn, d, shf_v in zip(well_names, depths, shf_values):
            if shf_v is not None and d is not None:
                by_well_shf.setdefault(wn, []).append({"DEPTH": d, "SHF": shf_v})
        for wn, rows in by_well_shf.items():
            if req.save:
                _save_curves_to_well(project_id, wn, pd.DataFrame(rows))
            else:
                _update_in_memory_well_cache(project_id, wn, pd.DataFrame(rows))
        if req.save:
            saved_wells = len(by_well_shf)

        scal_payload = {
            "pc": scal_df["PC"].tolist() if not scal_df.empty else [],
            "sw": scal_df["SW"].tolist() if not scal_df.empty else [],
            "swn": scal_df["SWN"].tolist() if not scal_df.empty else [],
            "j": scal_df["J"].tolist() if not scal_df.empty else [],
            "h": scal_df["H"].tolist() if not scal_df.empty else [],
            "bvw": scal_df["BVW"].tolist() if not scal_df.empty else [],
            "phit": scal_df["PHIT"].tolist() if not scal_df.empty else [],
            "perm": scal_df["PERM"].tolist() if not scal_df.empty else [],
            "rock_flags": [int(x) for x in scal_df["ROCK_FLAG"].tolist()]
            if not scal_df.empty
            else [],
            "well_names": scal_df["Well"].tolist() if not scal_df.empty else [],
            "samples": scal_df["Sample"].tolist() if not scal_df.empty else [],
        }

        return {
            "project_id": project_id,
            "model": model,
            "fits": fits,
            "fwl_by_zone": fwl_zone_map,
            "scal_data": scal_payload,
            "saved": req.save,
            "saved_wells": saved_wells,
            "data": {
                "well_names": well_names,
                "depths": depths,
                "zones": zones,
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
    well_names: Optional[List[str]] = Field(None)
    zones: Optional[List[str]] = Field(None)
    min_depth: Optional[float] = Field(None)
    max_depth: Optional[float] = Field(None)
    key_well: Optional[str] = Field(None)
    logs: List[str] = Field(default_factory=lambda: ["GR", "NPHI"])
    method: str = Field("quantile")
    quantiles: Tuple[float, float] = Field((0.05, 0.95))
    by_zone: bool = Field(True)
    exclude_badhole: bool = Field(True)
    save: bool = Field(False)


class WorkspaceMultiWellDistributionsRequest(BaseModel):
    well_names: Optional[List[str]] = Field(None)
    zones: Optional[List[str]] = Field(None)
    min_depth: Optional[float] = Field(None)
    max_depth: Optional[float] = Field(None)
    active_curve: str = Field("GR")
    curves: List[str] = Field(
        default_factory=lambda: ["GR", "RT", "NPHI", "RHOB", "DTC", "CALI", "PEF"]
    )
    group_by: str = Field(
        "well", description="Grouping mode: 'well', 'zone', or 'well_zone'"
    )
    exclude_badhole: bool = Field(False)
    max_points_per_well: int = Field(800, ge=100, le=5000)


class WorkspaceMultiWellBadholeRequest(WorkspaceBadholeRequest):
    well_names: Optional[List[str]] = Field(None)


class WorkspaceMultiWellEnvCorrectionRequest(WorkspaceEnvCorrectionRequest):
    well_names: Optional[List[str]] = Field(None)


class WorkspaceMultiWellConditioningRequest(BaseModel):
    action: str = Field(
        ...,
        description="Conditioning action: 'harmonize_aliases', 'outer_limits', or 'unit_scale'",
    )
    well_names: Optional[List[str]] = Field(None)
    zones: Optional[List[str]] = Field(None)
    min_depth: Optional[float] = Field(None)
    max_depth: Optional[float] = Field(None)
    fill: bool = Field(
        False,
        description="For 'outer_limits': True clamps to min/max, False replaces out-of-range with NaN",
    )
    curves: Optional[List[str]] = Field(
        None, description="Target curves for 'outer_limits'"
    )
    alias_map: Optional[Dict[str, List[str]]] = Field(
        None,
        description="Optional custom target -> [alias1, alias2] map for 'harmonize_aliases'",
    )
    source_curve: Optional[str] = Field(
        None, description="Explicit source curve for single-alias or unit_scale"
    )
    target_curve: Optional[str] = Field(
        None, description="Explicit target curve for single-alias or unit_scale"
    )
    scale: float = Field(1.0, description="Multiplier for 'unit_scale'")
    offset: float = Field(0.0, description="Additive offset for 'unit_scale'")
    save: bool = Field(False)


STANDARD_ALIAS_MAP: Dict[str, List[str]] = {
    "GR": ["GR_EDTC", "SGR", "CGR", "GAMMA", "GRC", "GR_RAW", "GRD"],
    "RT": ["ILD", "LLD", "RDEP", "RT_HRLT", "RESD", "RLLD", "RTRUE", "RD", "AT90", "HDRS"],
    "NPHI": ["TNPH", "NPOR", "CNC", "CNL", "PHIN", "NPHI_LS", "NPSS"],
    "RHOB": ["RHOZ", "ZDEN", "DEN", "BDEL", "RHOB_COMP"],
    "DTC": ["DT", "AC", "DTCO", "DT4P", "DT24", "SONIC"],
    "CALI": ["CAL", "HCAL", "CALIPER", "CALS", "DCAL"],
    "PEF": ["PE", "PEFZ", "PEFL"],
}


def _load_multi_well_dfs(
    project_id: int,
    well_names: Optional[List[str]] = None,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
    zones: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """Load merged DataFrames across project wells with ZONES populated from formation_tops if needed."""
    proj_summary = get_project_summary(project_id)
    all_wells = [w["name"] for w in proj_summary.get("wells", [])]
    if well_names:
        allowed = {str(w).strip() for w in well_names if str(w).strip()}
        if allowed:
            all_wells = [w for w in all_wells if w in allowed]
    if not all_wells:
        raise ValueError("No wells found or selected in project.")

    allowed_zones = (
        {str(z).strip() for z in zones if str(z).strip()} if zones else None
    )

    dfs: List[pd.DataFrame] = []
    with get_db() as session:
        try:
            proj = db_objects.Project.load(session, project_id=project_id)
        except Exception:
            proj = None

        for wn in all_wells:
            wdf = _get_merged_dataframe_sync(project_id, wn)
            if wdf.empty:
                continue
            wdf = wdf.copy()
            wdf["WELL_NAME"] = wn

            # Populate ZONES from formation tops if not already present or all null
            if "ZONES" not in wdf.columns or not wdf["ZONES"].notna().any():
                tops_df = None
                if proj is not None:
                    try:
                        anc = proj.get_well_ancillary_data(
                            resolve_db_well_name(project_id, wn)
                        )
                        tops_df = (
                            anc.get("formation_tops")
                            if isinstance(anc, dict)
                            else None
                        )
                    except Exception:
                        tops_df = None
                if isinstance(tops_df, pd.DataFrame) and not tops_df.empty and "DEPTH" in wdf.columns:
                    wdf["ZONES"] = pd.NA
                    for _, top in tops_df.iterrows():
                        try:
                            top_depth = float(top.get("depth"))
                            top_name = str(top.get("name")).strip()
                        except Exception:
                            continue
                        if not top_name:
                            continue
                        nearest_idx = (wdf["DEPTH"] - top_depth).abs().idxmin()
                        wdf.at[nearest_idx, "ZONES"] = top_name
                    wdf["ZONES"] = wdf["ZONES"].ffill().bfill()

            if "ZONES" not in wdf.columns:
                wdf["ZONES"] = "ALL"
            else:
                wdf["ZONES"] = (
                    wdf["ZONES"]
                    .replace({"": np.nan, "nan": np.nan, "None": np.nan})
                    .fillna("ALL")
                    .astype(str)
                )

            if "DEPTH" in wdf.columns:
                if min_depth is not None:
                    wdf = wdf[wdf["DEPTH"] >= float(min_depth)]
                if max_depth is not None:
                    wdf = wdf[wdf["DEPTH"] <= float(max_depth)]

            if allowed_zones:
                wdf = wdf[wdf["ZONES"].isin(allowed_zones)]

            if not wdf.empty:
                dfs.append(wdf)

    return dfs


class WorkspaceStickPlotRequest(BaseModel):
    well_names: Optional[List[str]] = Field(None)
    zones: Optional[List[str]] = Field(None)
    min_depth: Optional[float] = Field(None)
    max_depth: Optional[float] = Field(None)
    max_points_per_well: int = Field(1200, ge=100, le=5000)


def _resolve_well_tvd_array(
    md_values: np.ndarray,
    well_df: Optional[pd.DataFrame],
    surveys_df: Optional[pd.DataFrame],
) -> np.ndarray:
    """Resolve TVD array for given MD depths using TVD/TVDSS curve, directional survey minimum curvature, or MD fallback."""
    if len(md_values) == 0:
        return np.array([], dtype=float)

    # 1. Explicit TVD or TVDSS curve in well_df
    if well_df is not None and not well_df.empty and "DEPTH" in well_df.columns:
        for tvd_col in ("TVD", "TVDSS", "DEPT_TVD", "DEPTH_TVD"):
            if tvd_col in well_df.columns and well_df[tvd_col].notna().sum() >= 2:
                sub = (
                    pd.DataFrame(
                        {
                            "DEPTH": pd.to_numeric(well_df["DEPTH"], errors="coerce"),
                            "TVD": pd.to_numeric(well_df[tvd_col], errors="coerce"),
                        }
                    )
                    .dropna()
                    .sort_values("DEPTH")
                )
                if len(sub) >= 2:
                    return np.interp(
                        md_values,
                        sub["DEPTH"].to_numpy(dtype=float),
                        sub["TVD"].to_numpy(dtype=float),
                    )

    # 2. Directional survey table (with surface tie-in if survey starts below surface)
    if isinstance(surveys_df, pd.DataFrame) and not surveys_df.empty:
        col_map = {str(c).lower(): c for c in surveys_df.columns}
        md_c = col_map.get("md") or col_map.get("depth")
        inc_c = col_map.get("inc") or col_map.get("incl") or col_map.get("inclination")
        azi_c = col_map.get("azim") or col_map.get("azi") or col_map.get("azimuth")
        tvd_c = col_map.get("tvd") or col_map.get("tvdss")

        if md_c and tvd_c and surveys_df[tvd_c].notna().sum() >= 2:
            s_sub = (
                pd.DataFrame(
                    {
                        "md": pd.to_numeric(surveys_df[md_c], errors="coerce"),
                        "tvd": pd.to_numeric(surveys_df[tvd_c], errors="coerce"),
                    }
                )
                .dropna()
                .sort_values("md")
            )
            if len(s_sub) >= 2:
                return np.interp(
                    md_values,
                    s_sub["md"].to_numpy(dtype=float),
                    s_sub["tvd"].to_numpy(dtype=float),
                )

        if md_c and inc_c and azi_c:
            try:
                import wellpathpy as wpp

                s_sorted = (
                    pd.DataFrame(
                        {
                            "md": pd.to_numeric(surveys_df[md_c], errors="coerce"),
                            "inc": pd.to_numeric(surveys_df[inc_c], errors="coerce"),
                            "azim": pd.to_numeric(surveys_df[azi_c], errors="coerce"),
                        }
                    )
                    .dropna()
                    .sort_values("md")
                    .drop_duplicates(subset=["md"])
                )
                if len(s_sorted) >= 2:
                    mds = s_sorted["md"].to_numpy(dtype=float)
                    incs = s_sorted["inc"].to_numpy(dtype=float)
                    azis = s_sorted["azim"].to_numpy(dtype=float)
                    if mds[0] > 1.0:
                        mds = np.insert(mds, 0, 0.0)
                        incs = np.insert(incs, 0, 0.0)
                        azis = np.insert(azis, 0, 0.0)
                    dev = wpp.deviation(mds, incs, azis)
                    pos = dev.minimum_curvature()
                    return np.interp(
                        md_values,
                        np.asarray(pos.md, dtype=float),
                        np.asarray(pos.depth, dtype=float),
                    )
            except Exception:
                pass

    # 3. Fallback: vertical well (TVD == MD)
    return np.asarray(md_values, dtype=float).copy()


@router.post(
    "/stick_plot",
    summary="Compute multi-well TVD stick plot data (BVO_Log, BVO_SHF, PHIT, SWT, TVD, fluid contacts & tops)",
)
def workspace_stick_plot(
    project_id: int, req: WorkspaceStickPlotRequest
) -> Dict[str, Any]:
    """Return multi-well stick plot data anchored on True Vertical Depth (TVD) for field-level fluid contact analysis."""
    try:
        from quick_pp.plotter.plotter import update_fluid_contacts

        dfs = _load_multi_well_dfs(
            project_id,
            well_names=req.well_names,
            min_depth=req.min_depth,
            max_depth=req.max_depth,
            zones=req.zones,
        )
        if not dfs:
            raise ValueError("No wells found or remaining after filters.")

        max_pts = max(100, int(req.max_points_per_well))
        wells_payload: List[Dict[str, Any]] = []
        all_zones_set: set[str] = set()
        global_min_tvd = float("inf")
        global_max_tvd = float("-inf")

        with get_db() as session:
            try:
                proj = db_objects.Project.load(session, project_id=project_id)
            except Exception:
                proj = None

            for wdf in dfs:
                wn = str(wdf["WELL_NAME"].iloc[0])
                wdf_sorted = wdf.sort_values("DEPTH").reset_index(drop=True)
                md_full = pd.to_numeric(wdf_sorted["DEPTH"], errors="coerce").to_numpy(
                    dtype=float
                )

                anc: Dict[str, Any] = {}
                if proj is not None:
                    try:
                        anc = (
                            proj.get_well_ancillary_data(
                                resolve_db_well_name(project_id, wn)
                            )
                            or {}
                        )
                    except Exception:
                        anc = {}

                surveys_df = (
                    anc.get("well_surveys")
                    if "well_surveys" in anc
                    else anc.get("directional_surveys")
                )
                if isinstance(surveys_df, list):
                    surveys_df = pd.DataFrame(surveys_df)

                tvd_full = _resolve_well_tvd_array(md_full, wdf_sorted, surveys_df)
                wdf_sorted = wdf_sorted.copy()
                wdf_sorted["TVD"] = tvd_full

                # Resolve PHIT
                phit_col = next(
                    (
                        c
                        for c in ("PHIT", "PHIE", "CPORE", "POROSITY", "POR")
                        if c in wdf_sorted.columns and wdf_sorted[c].notna().any()
                    ),
                    None,
                )
                if phit_col:
                    phit_s = pd.to_numeric(
                        wdf_sorted[phit_col], errors="coerce"
                    ).clip(0.0, 0.5)
                elif "RHOB" in wdf_sorted.columns and wdf_sorted["RHOB"].notna().any():
                    rhob_s = pd.to_numeric(wdf_sorted["RHOB"], errors="coerce")
                    phit_s = ((2.65 - rhob_s) / 1.65).clip(0.0, 0.45)
                elif "NPHI" in wdf_sorted.columns and wdf_sorted["NPHI"].notna().any():
                    phit_s = pd.to_numeric(wdf_sorted["NPHI"], errors="coerce").clip(
                        0.0, 0.45
                    )
                else:
                    phit_s = pd.Series(np.nan, index=wdf_sorted.index, dtype=float)

                # Resolve SWT
                swt_col = next(
                    (
                        c
                        for c in ("SWT", "SWE", "SW", "SWT_WS", "SWT_ARCHIE")
                        if c in wdf_sorted.columns and wdf_sorted[c].notna().any()
                    ),
                    None,
                )
                if swt_col:
                    swt_s = pd.to_numeric(wdf_sorted[swt_col], errors="coerce").clip(
                        0.0, 1.0
                    )
                else:
                    rt_col = next(
                        (
                            c
                            for c in ("RT", "ILD", "LLD", "RDEP")
                            if c in wdf_sorted.columns and wdf_sorted[c].notna().any()
                        ),
                        None,
                    )
                    if rt_col and phit_s.notna().any():
                        rt_s = pd.to_numeric(
                            wdf_sorted[rt_col], errors="coerce"
                        ).clip(lower=0.1)
                        swt_s = np.sqrt(
                            0.05 / (rt_s * (phit_s.clip(lower=0.01) ** 2))
                        ).clip(0.02, 1.0)
                    else:
                        swt_s = pd.Series(np.nan, index=wdf_sorted.index, dtype=float)

                # Resolve SHF
                shf_s = (
                    pd.to_numeric(wdf_sorted["SHF"], errors="coerce").clip(0.0, 1.0)
                    if ("SHF" in wdf_sorted.columns and wdf_sorted["SHF"].notna().any())
                    else pd.Series(np.nan, index=wdf_sorted.index, dtype=float)
                )

                bvo_s = (phit_s * (1.0 - swt_s)).clip(0.0, 1.0)
                bvoh_s = (
                    (phit_s * (1.0 - shf_s)).clip(0.0, 1.0)
                    if shf_s.notna().any()
                    else pd.Series(np.nan, index=wdf_sorted.index, dtype=float)
                )

                # Extract fluid contacts & map MD -> TVD
                contacts_raw = anc.get("fluid_contacts")
                if isinstance(contacts_raw, pd.DataFrame):
                    contacts_list = (
                        contacts_raw.to_dict(orient="records")
                        if not contacts_raw.empty
                        else []
                    )
                elif isinstance(contacts_raw, list):
                    contacts_list = contacts_raw
                else:
                    contacts_list = []

                contact_items: List[Dict[str, Any]] = []
                contact_dict_tvd: Dict[str, float] = {}
                for c in contacts_list:
                    c_depth = c.get("depth")
                    if c_depth is None or pd.isna(c_depth):
                        continue
                    c_md = float(c_depth)
                    c_name = str(c.get("name") or c.get("type") or "OWC").strip().upper()
                    c_tvd = float(
                        _resolve_well_tvd_array(
                            np.array([c_md], dtype=float), wdf_sorted, surveys_df
                        )[0]
                    )
                    contact_items.append(
                        {
                            "type": c_name,
                            "name": c_name,
                            "depth": round(c_md, 2),
                            "tvd": round(c_tvd, 2),
                        }
                    )
                    contact_dict_tvd[c_name] = round(c_tvd, 2)

                # Also compute legacy update_fluid_contacts FLUID_FLAG on wdf_sorted using TVD contacts
                wdf_sorted["PHIT"] = phit_s
                wdf_sorted["SWT"] = swt_s
                wdf_sorted["BVO"] = bvo_s
                wdf_sorted["BVOH"] = bvoh_s
                try:
                    wdf_sorted = update_fluid_contacts(wdf_sorted, contact_dict_tvd)
                except Exception:
                    wdf_sorted["FLUID_FLAG"] = -1

                # Extract formation tops & map MD -> TVD
                tops_raw = anc.get("formation_tops")
                if isinstance(tops_raw, pd.DataFrame):
                    tops_list = (
                        tops_raw.to_dict(orient="records") if not tops_raw.empty else []
                    )
                elif isinstance(tops_raw, list):
                    tops_list = tops_raw
                else:
                    tops_list = []

                top_items: List[Dict[str, Any]] = []
                for t in tops_list:
                    t_depth = t.get("depth")
                    if t_depth is None or pd.isna(t_depth):
                        continue
                    t_md = float(t_depth)
                    t_name = str(t.get("name") or "").strip()
                    if not t_name:
                        continue
                    t_tvd = float(
                        _resolve_well_tvd_array(
                            np.array([t_md], dtype=float), wdf_sorted, surveys_df
                        )[0]
                    )
                    top_items.append(
                        {
                            "name": t_name,
                            "depth": round(t_md, 2),
                            "tvd": round(t_tvd, 2),
                        }
                    )
                top_items.sort(key=lambda x: x["tvd"])

                # Cumulative hydrocarbon pore volume column height (m) = integral of BVO over TVD
                valid_bvo = bvo_s.notna() & pd.Series(tvd_full).notna()
                cum_hcpv = 0.0
                if valid_bvo.sum() >= 2:
                    tvd_v = tvd_full[valid_bvo.to_numpy()]
                    bvo_v = bvo_s.to_numpy(dtype=float)[valid_bvo.to_numpy()]
                    dtvd = np.clip(np.diff(tvd_v, prepend=tvd_v[0]), 0.0, 2.0)
                    cum_hcpv = float(np.nansum(bvo_v * dtvd))

                valid_tvd = tvd_full[np.isfinite(tvd_full)]
                w_min_tvd = float(np.min(valid_tvd)) if len(valid_tvd) > 0 else 0.0
                w_max_tvd = float(np.max(valid_tvd)) if len(valid_tvd) > 0 else 1000.0
                global_min_tvd = min(global_min_tvd, w_min_tvd)
                global_max_tvd = max(global_max_tvd, w_max_tvd)

                # Determine structural top reservoir TVD (shallowest TVD with PHIT >= 0.06, or first formation top)
                res_mask = (phit_s >= 0.06) & np.isfinite(tvd_full)
                if res_mask.any():
                    top_res_tvd = float(np.min(tvd_full[res_mask.to_numpy()]))
                elif top_items:
                    top_res_tvd = float(top_items[0]["tvd"])
                else:
                    top_res_tvd = w_min_tvd

                for z_val in wdf_sorted["ZONES"].dropna().unique():
                    z_str = str(z_val).strip()
                    if z_str:
                        all_zones_set.add(z_str)

                # Downsample if needed
                sub = wdf_sorted
                if len(sub) > max_pts:
                    stride = max(1, len(sub) // max_pts)
                    sub = sub.iloc[::stride].reset_index(drop=True)

                def _clean_float_list(series_in: pd.Series, decimals: int = 4) -> list:
                    s_num = pd.to_numeric(series_in, errors="coerce").replace(
                        [np.inf, -np.inf], np.nan
                    )
                    return [
                        round(float(x), decimals) if pd.notna(x) else None
                        for x in s_num.tolist()
                    ]

                wells_payload.append(
                    {
                        "well_name": wn,
                        "min_tvd": round(w_min_tvd, 2),
                        "max_tvd": round(w_max_tvd, 2),
                        "top_reservoir_tvd": round(top_res_tvd, 2),
                        "cum_hcpv": round(cum_hcpv, 3),
                        "contacts": contact_items,
                        "tops": top_items,
                        "depths": _clean_float_list(sub["DEPTH"], 2),
                        "tvd": _clean_float_list(sub["TVD"], 2),
                        "zones": sub["ZONES"].fillna("ALL").astype(str).tolist(),
                        "phit": _clean_float_list(sub["PHIT"], 4),
                        "swt": _clean_float_list(sub["SWT"], 4),
                        "shf": _clean_float_list(sub["SHF"], 4)
                        if "SHF" in sub.columns
                        else [None] * len(sub),
                        "bvo": _clean_float_list(sub["BVO"], 4),
                        "bvoh": _clean_float_list(sub["BVOH"], 4),
                        "fluid_flag": [
                            int(x) if pd.notna(x) else -1
                            for x in sub["FLUID_FLAG"].tolist()
                        ],
                    }
                )

        return _sanitize_for_json(
            {
                "project_id": project_id,
                "min_tvd": round(global_min_tvd, 2)
                if math.isfinite(global_min_tvd)
                else 0.0,
                "max_tvd": round(global_max_tvd, 2)
                if math.isfinite(global_max_tvd)
                else 1000.0,
                "available_zones": sorted(all_zones_set),
                "wells": wells_payload,
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Stick plot computation failed: {e}"
        ) from e


class WorkspaceOptimizeFwlRequest(BaseModel):
    well_names: Optional[List[str]] = Field(None)
    zones: Optional[List[str]] = Field(None)
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
    mdt_weight: Optional[float] = Field(None)
    solve_per_well: bool = Field(True)
    solve_per_zone: bool = Field(True)
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
        dfs = _load_multi_well_dfs(
            project_id,
            well_names=req.well_names,
            min_depth=req.min_depth,
            max_depth=req.max_depth,
            zones=req.zones,
        )
        if not dfs:
            raise ValueError("No log data available across selected project wells.")

        combined = pd.concat(dfs, ignore_index=True)
        zone_col_used = "ZONES"
        if not req.by_zone:
            combined["_GLOBAL_ZONE"] = "ALL"
            zone_col_used = "_GLOBAL_ZONE"

        norm_df, report_df = stratigraphic_normalization(
            combined,
            key_well=req.key_well,
            zone_col=zone_col_used,
            quantiles=req.quantiles,
            logs=req.logs,
            well_col="WELL_NAME",
            method=req.method,
            exclude_badhole=req.exclude_badhole,
            return_report=True,
        )

        norm_cols = [f"{lg}_NORM" for lg in req.logs if f"{lg}_NORM" in norm_df.columns]
        saved_wells = 0
        if norm_cols:
            for wn, grp in norm_df.groupby("WELL_NAME"):
                sub_df = grp[["DEPTH"] + norm_cols]
                if req.save:
                    _save_curves_to_well(project_id, str(wn), sub_df)
                    saved_wells += 1
                else:
                    _update_in_memory_well_cache(project_id, str(wn), sub_df)

        return {
            "project_id": project_id,
            "method": req.method,
            "by_zone": req.by_zone,
            "exclude_badhole": req.exclude_badhole,
            "quantiles": list(req.quantiles),
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
    "/multi_well_qaqc/distributions",
    summary="Compute multi-well and per-zone curve statistical distributions, anomaly flags, and plot data",
)
def workspace_multi_well_distributions(
    project_id: int, req: WorkspaceMultiWellDistributionsRequest
) -> Dict[str, Any]:
    """Return multi-well/zone distributions, quantile statistics, outlier/unit diagnostics, and sampled plot points."""
    try:
        dfs = _load_multi_well_dfs(
            project_id,
            well_names=req.well_names,
            min_depth=req.min_depth,
            max_depth=req.max_depth,
            zones=req.zones,
        )
        if not dfs:
            raise ValueError("No well data available for the selected filters.")

        combined = pd.concat(dfs, ignore_index=True)
        active_curve = req.active_curve.strip().upper() or "GR"
        norm_curve = (
            f"{active_curve}_NORM"
            if not active_curve.endswith("_NORM")
            else active_curve
        )
        corr_curve = (
            f"{active_curve}_CORR"
            if not active_curve.endswith("_CORR")
            else active_curve
        )

        # Resolve BADHOLE column across combined dataframe
        bh_series = pd.Series(0, index=combined.index, dtype=int)
        for bh_col in ("BADHOLE", "BAD_HOLE", "BAD_HOLE_FLAG"):
            if bh_col in combined.columns:
                s_bh = pd.to_numeric(combined[bh_col], errors="coerce").fillna(0.0)
                bh_series = bh_series | (s_bh >= 0.5).astype(int)
        combined["_BADHOLE_INT"] = bh_series

        # Filter out badhole for statistical calculations if requested
        stat_df = (
            combined[combined["_BADHOLE_INT"] == 0]
            if req.exclude_badhole and (combined["_BADHOLE_INT"] == 0).any()
            else combined
        )

        target_curves = list(
            dict.fromkeys(
                [c.strip().upper() for c in req.curves if c.strip()]
                + [active_curve, norm_curve, corr_curve]
            )
        )
        present_curves = [c for c in target_curves if c in combined.columns]

        # 1. Field-wide reference stats per curve
        field_stats: Dict[str, Dict[str, Optional[float]]] = {}
        for c in present_curves:
            s = (
                pd.to_numeric(stat_df[c], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
            )
            if s.empty:
                field_stats[c] = {
                    "count": 0,
                    "min": None,
                    "p05": None,
                    "p10": None,
                    "p25": None,
                    "p50": None,
                    "mean": None,
                    "std": None,
                    "p75": None,
                    "p90": None,
                    "p95": None,
                    "max": None,
                }
            else:
                arr = s.to_numpy(dtype=float)
                qs = np.percentile(arr, [5, 10, 25, 50, 75, 90, 95])
                field_stats[c] = {
                    "count": int(len(arr)),
                    "min": round(float(np.min(arr)), 4),
                    "p05": round(float(qs[0]), 4),
                    "p10": round(float(qs[1]), 4),
                    "p25": round(float(qs[2]), 4),
                    "p50": round(float(qs[3]), 4),
                    "mean": round(float(np.mean(arr)), 4),
                    "std": round(float(np.std(arr)), 4),
                    "p75": round(float(qs[4]), 4),
                    "p90": round(float(qs[5]), 4),
                    "p95": round(float(qs[6]), 4),
                    "max": round(float(np.max(arr)), 4),
                }

        # 2. Group statistics for active_curve (and optional norm_curve)
        group_mode = req.group_by.strip().lower()
        if group_mode == "zone":
            group_cols = ["ZONES"]
        elif group_mode == "well_zone":
            group_cols = ["WELL_NAME", "ZONES"]
        else:
            group_cols = ["WELL_NAME"]

        f_st = field_stats.get(active_curve, {})
        f_p50 = f_st.get("p50")
        f_iqr = (
            (f_st["p75"] - f_st["p25"])
            if (f_st.get("p75") is not None and f_st.get("p25") is not None)
            else None
        )
        var_bounds = Config.VARS.get(active_curve, {})

        group_stats: List[Dict[str, Any]] = []
        for grp_key, grp_df in combined.groupby(group_cols, sort=False, dropna=False):
            if not isinstance(grp_key, tuple):
                grp_key = (grp_key,)
            well_label = str(grp_key[0]) if group_mode != "zone" else "ALL_WELLS"
            zone_label = (
                str(grp_key[0])
                if group_mode == "zone"
                else (str(grp_key[1]) if len(grp_key) > 1 else "ALL")
            )

            total_n = len(grp_df)
            bh_n = int(grp_df["_BADHOLE_INT"].sum())
            bh_pct = round((bh_n / total_n) * 100.0, 2) if total_n > 0 else 0.0

            eval_grp = (
                grp_df[grp_df["_BADHOLE_INT"] == 0]
                if req.exclude_badhole and (grp_df["_BADHOLE_INT"] == 0).any()
                else grp_df
            )
            s = (
                pd.to_numeric(eval_grp[active_curve], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .dropna()
                if active_curve in eval_grp.columns
                else pd.Series(dtype=float)
            )

            if s.empty:
                group_stats.append(
                    {
                        "WELL_NAME": well_label,
                        "ZONE": zone_label,
                        "CURVE": active_curve,
                        "COUNT": 0,
                        "BADHOLE_PCT": bh_pct,
                        "MIN": None,
                        "P10": None,
                        "P25": None,
                        "P50": None,
                        "MEAN": None,
                        "STD": None,
                        "P75": None,
                        "P90": None,
                        "MAX": None,
                        "NORM_P50": None,
                        "DRIFT": None,
                        "STATUS": "MISSING",
                    }
                )
                continue

            arr = s.to_numpy(dtype=float)
            qs = np.percentile(arr, [10, 25, 50, 75, 90])
            w_min = float(np.min(arr))
            w_max = float(np.max(arr))
            w_p50 = float(qs[2])
            drift = round(w_p50 - f_p50, 4) if f_p50 is not None else 0.0

            norm_p50 = None
            if norm_curve in eval_grp.columns and norm_curve != active_curve:
                s_norm = (
                    pd.to_numeric(eval_grp[norm_curve], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna()
                )
                if not s_norm.empty:
                    norm_p50 = round(float(np.median(s_norm)), 4)

            status = "OK"
            if active_curve in ("NPHI", "PHIT", "PHIE", "SWT") and w_p50 > 1.2:
                status = "UNIT_MISMATCH (%)"
            elif (
                "min" in var_bounds and w_min < var_bounds["min"]
            ) or ("max" in var_bounds and w_max > var_bounds["max"]):
                status = "OUT_OF_BOUNDS"
            elif (
                f_iqr is not None
                and f_iqr > 1e-6
                and abs(drift) > 1.25 * f_iqr
            ):
                status = "HIGH_DRIFT"

            group_stats.append(
                {
                    "WELL_NAME": well_label,
                    "ZONE": zone_label,
                    "CURVE": active_curve,
                    "COUNT": int(len(arr)),
                    "BADHOLE_PCT": bh_pct,
                    "MIN": round(w_min, 4),
                    "P10": round(float(qs[0]), 4),
                    "P25": round(float(qs[1]), 4),
                    "P50": round(w_p50, 4),
                    "MEAN": round(float(np.mean(arr)), 4),
                    "STD": round(float(np.std(arr)), 4),
                    "P75": round(float(qs[3]), 4),
                    "P90": round(float(qs[4]), 4),
                    "MAX": round(w_max, 4),
                    "NORM_P50": norm_p50,
                    "DRIFT": drift,
                    "STATUS": status,
                }
            )

        # 3. Scan across all loaded wells for actionable anomalies (aliases, unit mismatches, outer limits)
        anomalies: List[Dict[str, Any]] = []
        for wdf in dfs:
            wn = str(wdf["WELL_NAME"].iloc[0])
            w_cols = set(wdf.columns)
            # Check missing standard curves that have aliases present
            for std_col, aliases in STANDARD_ALIAS_MAP.items():
                if std_col not in w_cols or not wdf[std_col].notna().any():
                    found_alias = next(
                        (a for a in aliases if a in w_cols and wdf[a].notna().any()),
                        None,
                    )
                    if found_alias:
                        anomalies.append(
                            {
                                "well_name": wn,
                                "curve": std_col,
                                "type": "alias",
                                "source_curve": found_alias,
                                "target_curve": std_col,
                                "message": f"{wn}: Missing '{std_col}', found alias '{found_alias}'",
                            }
                        )
            # Check unit mismatch on NPHI (percent vs v/v)
            if "NPHI" in w_cols:
                s_nphi = pd.to_numeric(wdf["NPHI"], errors="coerce").dropna()
                if not s_nphi.empty and float(s_nphi.median()) > 1.2:
                    anomalies.append(
                        {
                            "well_name": wn,
                            "curve": "NPHI",
                            "type": "unit_scale",
                            "scale": 0.01,
                            "message": f"{wn}: NPHI median is {float(s_nphi.median()):.1f} (looks like % instead of v/v)",
                        }
                    )
            # Check outer limits from Config.VARS
            for var_name, v_dict in Config.VARS.items():
                if var_name in w_cols and ("min" in v_dict or "max" in v_dict):
                    s_v = pd.to_numeric(wdf[var_name], errors="coerce").dropna()
                    if s_v.empty:
                        continue
                    n_out = int(
                        ((s_v < v_dict.get("min", -np.inf)) | (s_v > v_dict.get("max", np.inf))).sum()
                    )
                    if n_out > 0:
                        anomalies.append(
                            {
                                "well_name": wn,
                                "curve": var_name,
                                "type": "outer_limits",
                                "count": n_out,
                                "message": f"{wn}: {var_name} has {n_out} pts outside [{v_dict.get('min')}, {v_dict.get('max')}]",
                            }
                        )

        # 4. Downsample plot points per well for fast Plotly rendering
        sampled_parts: List[pd.DataFrame] = []
        max_pts = max(100, int(req.max_points_per_well))
        for wdf in dfs:
            sub = (
                wdf[wdf.get("BADHOLE", 0) == 0]
                if req.exclude_badhole and "BADHOLE" in wdf.columns
                else wdf
            )
            if len(sub) > max_pts:
                stride = max(1, len(sub) // max_pts)
                sub = sub.iloc[::stride]
            sampled_parts.append(sub)

        sampled_df = (
            pd.concat(sampled_parts, ignore_index=True)
            if sampled_parts
            else pd.DataFrame()
        )

        def _col_list(df_in: pd.DataFrame, col_name: str) -> list:
            if df_in.empty or col_name not in df_in.columns:
                return []
            s = pd.to_numeric(df_in[col_name], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            return [round(float(x), 6) if pd.notna(x) else None for x in s.tolist()]

        bh_sampled = pd.Series(0, index=sampled_df.index, dtype=int)
        for bh_col in ("BADHOLE", "BAD_HOLE", "BAD_HOLE_FLAG"):
            if bh_col in sampled_df.columns:
                s_bh = pd.to_numeric(sampled_df[bh_col], errors="coerce").fillna(0.0)
                bh_sampled = bh_sampled | (s_bh >= 0.5).astype(int)

        available_zones = sorted(
            {
                str(z)
                for z in combined["ZONES"].dropna().unique()
                if str(z).strip()
            }
        )
        available_curves = sorted(
            [
                str(c)
                for c in combined.columns
                if str(c) not in ("DEPTH", "DEPT", "WELL_NAME", "ZONES", "_BADHOLE_INT")
                and not str(c).startswith("TOP_")
                and not str(c).startswith("CONTACT_")
                and not str(c).startswith("PRESSURE_")
            ]
        )

        return {
            "project_id": project_id,
            "active_curve": active_curve,
            "norm_curve": norm_curve if norm_curve in combined.columns else None,
            "corr_curve": corr_curve if corr_curve in combined.columns else None,
            "group_by": group_mode,
            "exclude_badhole": req.exclude_badhole,
            "available_curves": available_curves,
            "available_zones": available_zones,
            "field_stats": field_stats,
            "group_stats": group_stats,
            "anomalies": anomalies,
            "plot_data": {
                "well_names": (
                    sampled_df["WELL_NAME"].astype(str).tolist()
                    if not sampled_df.empty
                    else []
                ),
                "zones": (
                    sampled_df["ZONES"].astype(str).tolist()
                    if not sampled_df.empty
                    else []
                ),
                "depths": _col_list(sampled_df, "DEPTH"),
                "badhole": bh_sampled.tolist() if not sampled_df.empty else [],
                "values": _col_list(sampled_df, active_curve),
                "norm_values": (
                    _col_list(sampled_df, norm_curve)
                    if norm_curve in sampled_df.columns and norm_curve != active_curve
                    else []
                ),
                "corr_values": (
                    _col_list(sampled_df, corr_curve)
                    if corr_curve in sampled_df.columns and corr_curve != active_curve
                    else []
                ),
                "nphi": _col_list(sampled_df, "NPHI"),
                "rhob": _col_list(sampled_df, "RHOB"),
                "gr": _col_list(sampled_df, "GR"),
            },
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Multi-well distribution computation failed: {e}",
        ) from e


@router.post(
    "/multi_well_qaqc/badhole_flagging",
    summary="Run batch badhole flagging across selected project wells",
)
def workspace_multi_well_badhole_flagging(
    project_id: int, req: WorkspaceMultiWellBadholeRequest
) -> Dict[str, Any]:
    """Execute badhole_flagging across multiple wells and optionally save BADHOLE, BS, and DCALI."""
    try:
        dfs = _load_multi_well_dfs(
            project_id,
            well_names=req.well_names,
            min_depth=req.min_depth,
            max_depth=req.max_depth,
            zones=req.zones,
        )
        if not dfs:
            raise ValueError("No wells available for batch badhole flagging.")

        well_summaries: List[Dict[str, Any]] = []
        total_samples = 0
        total_flagged = 0
        saved_wells = 0

        for wdf in dfs:
            wn = str(wdf["WELL_NAME"].iloc[0])
            bh_df = badhole_flagging(
                wdf,
                thold=req.cali_threshold,
                use_cali=req.use_cali,
                auto_bitsize=req.auto_bitsize,
                bitsize_default=req.bitsize_default,
                use_drho=req.use_drho,
                drho_threshold=req.drho_threshold,
                use_rhob_nphi=req.use_rhob_nphi,
                rhob_min=req.rhob_min,
                rhob_max=req.rhob_max,
                nphi_max=req.nphi_max,
                exclude_coal=req.exclude_coal,
                use_rugosity=req.use_rugosity,
                rugosity_window=req.rugosity_window,
                rhob_trend_factor=req.rhob_trend_factor,
                cali_spike_threshold=req.cali_spike_threshold,
            )
            summary = dict(bh_df.attrs.get("badhole_summary", {}))
            out_cols = [
                c for c in ("BADHOLE", "BS", "DCALI") if c in bh_df.columns
            ]
            if out_cols and "DEPTH" in bh_df.columns:
                sub_df = bh_df[["DEPTH"] + out_cols].copy()
                if req.save:
                    _save_curves_to_well(project_id, wn, sub_df)
                    saved_wells += 1
                else:
                    _update_in_memory_well_cache(project_id, wn, sub_df)

            n_s = int(summary.get("total_samples", len(bh_df)))
            n_f = int(summary.get("flagged_count", 0))
            total_samples += n_s
            total_flagged += n_f
            well_summaries.append(
                {
                    "WELL_NAME": wn,
                    "TOTAL": n_s,
                    "FLAGGED": n_f,
                    "FLAGGED_PCT": summary.get("flagged_pct", 0.0),
                    "CALI": summary.get("cali_flagged", 0),
                    "DRHO": summary.get("drho_flagged", 0),
                    "RHOB_NPHI": summary.get("rhob_nphi_flagged", 0),
                    "RUGOSITY": summary.get("rugosity_flagged", 0),
                }
            )

        return {
            "project_id": project_id,
            "saved": req.save,
            "saved_wells": saved_wells,
            "total_samples": total_samples,
            "total_flagged": total_flagged,
            "total_flagged_pct": (
                round((total_flagged / total_samples) * 100.0, 2)
                if total_samples > 0
                else 0.0
            ),
            "well_summaries": well_summaries,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Batch badhole flagging failed: {e}"
        ) from e


@router.post(
    "/multi_well_qaqc/environmental_corrections",
    summary="Apply borehole environmental and invasion corrections across multiple wells",
)
def workspace_multi_well_environmental_corrections(
    project_id: int, req: WorkspaceMultiWellEnvCorrectionRequest
) -> Dict[str, Any]:
    """Run apply_environmental_corrections across selected project wells."""
    try:
        dfs = _load_multi_well_dfs(
            project_id,
            well_names=req.well_names,
            min_depth=req.min_depth,
            max_depth=req.max_depth,
            zones=req.zones,
        )
        if not dfs:
            raise ValueError("No wells available for batch environmental corrections.")

        well_summaries: List[Dict[str, Any]] = []
        all_corr_cols: set[str] = set()
        saved_wells = 0

        for wdf in dfs:
            wn = str(wdf["WELL_NAME"].iloc[0])
            corr_df = apply_environmental_corrections(
                wdf,
                bitsize_default=req.bitsize_default,
                mud_weight=req.mud_weight,
                surface_temp=req.surface_temp,
                temp_grad=req.temp_grad,
                salinity_ppm=req.salinity_ppm,
                rho_mudcake=req.rho_mudcake,
            )
            corr_cols = [
                c
                for c in (
                    "GR_CORR",
                    "NPHI_CORR",
                    "RHOB_CORR",
                    "RT_CORR",
                    "INVASION_FLAG",
                )
                if c in corr_df.columns
            ]
            all_corr_cols.update(corr_cols)
            if corr_cols and "DEPTH" in corr_df.columns:
                sub_df = corr_df[["DEPTH"] + corr_cols].copy()
                if req.save:
                    _save_curves_to_well(project_id, wn, sub_df)
                    saved_wells += 1
                else:
                    _update_in_memory_well_cache(project_id, wn, sub_df)

            well_summaries.append(
                {
                    "WELL_NAME": wn,
                    "ROWS": len(corr_df),
                    "CURVES_CORRECTED": ", ".join(corr_cols),
                }
            )

        return {
            "project_id": project_id,
            "saved": req.save,
            "saved_wells": saved_wells,
            "corrected_curves": sorted(all_corr_cols),
            "well_summaries": well_summaries,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Batch environmental corrections failed: {e}"
        ) from e


@router.post(
    "/multi_well_qaqc/conditioning",
    summary="Run batch curve alias harmonization, outer-limit clipping, or unit scaling across wells",
)
def workspace_multi_well_conditioning(
    project_id: int, req: WorkspaceMultiWellConditioningRequest
) -> Dict[str, Any]:
    """Execute batch curve conditioning (harmonize_aliases, outer_limits, or unit_scale) across project wells."""
    try:
        dfs = _load_multi_well_dfs(
            project_id,
            well_names=req.well_names,
            min_depth=req.min_depth,
            max_depth=req.max_depth,
            zones=req.zones,
        )
        if not dfs:
            raise ValueError("No wells available for batch conditioning.")

        act = req.action.strip().lower()
        changes: List[Dict[str, Any]] = []
        updated_wells = 0

        if act == "harmonize_aliases":
            alias_dict = dict(STANDARD_ALIAS_MAP)
            if req.alias_map:
                for k, v in req.alias_map.items():
                    alias_dict[k.strip().upper()] = [
                        str(x).strip().upper() for x in v if str(x).strip()
                    ]
            if req.source_curve and req.target_curve:
                alias_dict = {
                    req.target_curve.strip().upper(): [req.source_curve.strip().upper()]
                }

            for wdf in dfs:
                wn = str(wdf["WELL_NAME"].iloc[0])
                mapped_cols: Dict[str, pd.Series] = {}
                for target_col, aliases in alias_dict.items():
                    if target_col in wdf.columns and wdf[target_col].notna().any():
                        continue
                    found = next(
                        (
                            a
                            for a in aliases
                            if a in wdf.columns and wdf[a].notna().any()
                        ),
                        None,
                    )
                    if found:
                        mapped_cols[target_col] = pd.to_numeric(
                            wdf[found], errors="coerce"
                        )
                        changes.append(
                            {
                                "WELL_NAME": wn,
                                "ACTION": "ALIAS_MAP",
                                "SOURCE": found,
                                "TARGET": target_col,
                                "COUNT": int(mapped_cols[target_col].notna().sum()),
                            }
                        )
                if mapped_cols and "DEPTH" in wdf.columns:
                    sub_df = pd.DataFrame({"DEPTH": wdf["DEPTH"], **mapped_cols})
                    if req.save:
                        _save_curves_to_well(project_id, wn, sub_df)
                    else:
                        _update_in_memory_well_cache(project_id, wn, sub_df)
                    updated_wells += 1

        elif act == "outer_limits":
            target_vars = (
                [c.strip().upper() for c in req.curves if c.strip()]
                if req.curves
                else list(Config.VARS.keys())
            )
            for wdf in dfs:
                wn = str(wdf["WELL_NAME"].iloc[0])
                cols_present = [
                    c
                    for c in target_vars
                    if c in wdf.columns
                    and c in Config.VARS
                    and ("min" in Config.VARS[c] or "max" in Config.VARS[c])
                ]
                if not cols_present or "DEPTH" not in wdf.columns:
                    continue
                cleaned = handle_outer_limit(
                    wdf[["DEPTH"] + cols_present], fill=req.fill
                )
                modified_cols = []
                for c in cols_present:
                    s_before = pd.to_numeric(wdf[c], errors="coerce")
                    s_after = pd.to_numeric(cleaned[c], errors="coerce")
                    n_diff = int(
                        (
                            (s_before.notna() != s_after.notna())
                            | (
                                s_before.notna()
                                & s_after.notna()
                                & ((s_before - s_after).abs() > 1e-9)
                            )
                        ).sum()
                    )
                    if n_diff > 0:
                        modified_cols.append(c)
                        changes.append(
                            {
                                "WELL_NAME": wn,
                                "ACTION": "CLAMP" if req.fill else "NULL_OUTLIERS",
                                "SOURCE": c,
                                "TARGET": c,
                                "COUNT": n_diff,
                            }
                        )
                if modified_cols:
                    sub_df = cleaned[["DEPTH"] + modified_cols]
                    if req.save:
                        _save_curves_to_well(project_id, wn, sub_df)
                    else:
                        _update_in_memory_well_cache(project_id, wn, sub_df)
                    updated_wells += 1

        elif act == "unit_scale":
            src = (req.source_curve or req.target_curve or "NPHI").strip().upper()
            dst = (req.target_curve or src).strip().upper()
            for wdf in dfs:
                wn = str(wdf["WELL_NAME"].iloc[0])
                if src not in wdf.columns or "DEPTH" not in wdf.columns:
                    continue
                s = pd.to_numeric(wdf[src], errors="coerce")
                if not s.notna().any():
                    continue
                s_scaled = s * float(req.scale) + float(req.offset)
                sub_df = pd.DataFrame({"DEPTH": wdf["DEPTH"], dst: s_scaled})
                if req.save:
                    _save_curves_to_well(project_id, wn, sub_df)
                else:
                    _update_in_memory_well_cache(project_id, wn, sub_df)
                updated_wells += 1
                changes.append(
                    {
                        "WELL_NAME": wn,
                        "ACTION": f"SCALE (×{req.scale:g}{f' +{req.offset:g}' if req.offset else ''})",
                        "SOURCE": src,
                        "TARGET": dst,
                        "COUNT": int(s_scaled.notna().sum()),
                    }
                )
        else:
            raise ValueError(f"Unsupported conditioning action: {act}")

        return {
            "project_id": project_id,
            "action": act,
            "saved": req.save,
            "updated_wells": updated_wells,
            "changes": changes,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Batch conditioning failed: {e}"
        ) from e


@router.post(
    "/optimize_fwl",
    summary="Optimize Multi-Well Free Water Level (FWL) & Transition Zone across project wells",
)
def workspace_optimize_fwl(
    project_id: int, req: WorkspaceOptimizeFwlRequest
) -> Dict[str, Any]:
    """Simultaneously optimize Free Water Level across all project wells and reconcile Sw_log vs Sw_SHM."""
    try:
        proj_summary = get_project_summary(project_id)
        well_names = [w["name"] for w in proj_summary.get("wells", [])]
        if req.well_names:
            allowed = {str(w).strip() for w in req.well_names if str(w).strip()}
            if allowed:
                well_names = [w for w in well_names if w in allowed]
        if not well_names:
            raise ValueError("No wells found in project.")

        dfs = []
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            for wn in well_names:
                wdf = _get_merged_dataframe_sync(project_id, wn)
                if not wdf.empty:
                    wdf = wdf.copy()
                    wdf["WELL_NAME"] = wn
                    # Populate ZONES from formation tops if not already present on the well
                    if "ZONES" not in wdf.columns or not wdf["ZONES"].notna().any():
                        try:
                            anc = proj.get_well_ancillary_data(wn)
                        except Exception:
                            anc = {}
                        tops_df = (
                            anc.get("formation_tops") if isinstance(anc, dict) else None
                        )
                        if isinstance(tops_df, pd.DataFrame) and not tops_df.empty:
                            wdf["ZONES"] = pd.NA
                            for _, top in tops_df.iterrows():
                                try:
                                    top_depth = float(top.get("depth"))
                                    top_name = str(top.get("name"))
                                except Exception:
                                    continue
                                nearest_idx = (wdf["DEPTH"] - top_depth).abs().idxmin()
                                wdf.at[nearest_idx, "ZONES"] = top_name
                            wdf["ZONES"] = wdf["ZONES"].ffill()
                    if "PHIT" not in wdf.columns or not wdf["PHIT"].notna().any():
                        if "RHOB" in wdf.columns and wdf["RHOB"].notna().any():
                            wdf["PHIT"] = (
                                (2.65 - pd.to_numeric(wdf["RHOB"], errors="coerce"))
                                / 1.65
                            ).clip(0.01, 0.45)
                        elif "NPHI" in wdf.columns and wdf["NPHI"].notna().any():
                            wdf["PHIT"] = pd.to_numeric(
                                wdf["NPHI"], errors="coerce"
                            ).clip(0.01, 0.45)
                    if (
                        "PERM" not in wdf.columns or not wdf["PERM"].notna().any()
                    ) and "PHIT" in wdf.columns:
                        wdf["PERM"] = (10.0 ** (14.0 * wdf["PHIT"] - 1.5)).clip(
                            0.01, 10000.0
                        )
                    if (
                        ("SW" not in wdf.columns or not wdf["SW"].notna().any())
                        and "SHF" in wdf.columns
                        and wdf["SHF"].notna().any()
                    ):
                        wdf["SW"] = wdf["SHF"]
                    elif (
                        "SW" not in wdf.columns or not wdf["SW"].notna().any()
                    ) and "PHIT" in wdf.columns:
                        rt_col = next(
                            (
                                c
                                for c in ("RT", "ILD", "LLD", "RDEP")
                                if c in wdf.columns and wdf[c].notna().any()
                            ),
                            None,
                        )
                        if rt_col:
                            rt_s = pd.to_numeric(wdf[rt_col], errors="coerce").clip(
                                lower=0.1
                            )
                            phi_s = pd.to_numeric(wdf["PHIT"], errors="coerce").clip(
                                lower=0.01
                            )
                            wdf["SW"] = np.sqrt(0.05 / (rt_s * (phi_s**2))).clip(
                                0.02, 1.0
                            )
                    dfs.append(wdf)
        if not dfs:
            raise ValueError("No log data available across project wells.")

        combined = pd.concat(dfs, ignore_index=True)
        if req.zones and "ZONES" in combined.columns:
            allowed_z = {str(z).strip() for z in req.zones if str(z).strip()}
            if allowed_z:
                sub_z = combined[
                    combined["ZONES"].fillna("Unknown").astype(str).isin(allowed_z)
                ]
                if not sub_z.empty:
                    combined = sub_z

        penalty_w = (
            float(req.mdt_weight)
            if req.mdt_weight is not None
            else float(req.mdt_penalty_weight)
        )
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
            mdt_penalty_weight=penalty_w,
            solve_per_well=req.solve_per_well,
            solve_per_zone=req.solve_per_zone,
            zone_col="ZONES",
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

        zone_metrics_df = res.get("zone_metrics")
        zone_metrics_records = (
            dataframe_to_records(zone_metrics_df.round(4))
            if isinstance(zone_metrics_df, pd.DataFrame) and not zone_metrics_df.empty
            else []
        )

        return {
            "project_id": project_id,
            "fwl": round(float(res["fwl"]), 2),
            "fwl_by_zone": res.get("fwl_by_zone", {}),
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
            "zone_metrics": zone_metrics_records,
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
        description="Pre-trained model suite key from the loaded `.joblib` bundle (e.g. 'clastic', 'carbonate')",
    )
    alpha: float = Field(
        0.20,
        description="Conformal significance level (0.20 -> P10 to P90 interval)",
    )


@router.get(
    "/ml_models",
    summary="Get available Pre-Trained PIML & Conformal Prediction model suites and file status",
)
def workspace_ml_models_status(project_id: int) -> Dict[str, Any]:
    """Return available pre-trained `.joblib` model suites or fallback notification if the file is missing."""
    from quick_pp.machine_learning.predict_pipeline import get_pretrained_models_status

    return get_pretrained_models_status()


class WorkspaceMlTrainRequest(BaseModel):
    well_names: Optional[List[str]] = Field(
        None,
        description="Selected well names to include in training (None = all project wells)",
    )
    zones: Optional[List[str]] = Field(
        None, description="Optional list of ZONES to include"
    )
    architecture: str = Field(
        "clastic",
        description="Restricted target preset architecture: 'clastic' or 'carbonate'",
    )
    suite_key: str = Field(
        "clastic",
        description="Model suite identifier in the global `.joblib` bundle (e.g., 'clastic', 'carbonate', 'volve_v2')",
    )
    suite_label: Optional[str] = Field(
        None, description="Human-readable display name for the model suite"
    )
    exclude_coal: bool = Field(
        True, description="Exclude intervals flagged as coal (COAL_FLAG=1 or VCOAL>0.5)"
    )
    exclude_tight: bool = Field(
        False, description="Exclude intervals flagged as tight rock (TIGHT_FLAG=1)"
    )
    exclude_bad_hole: bool = Field(
        True, description="Exclude intervals flagged as bad hole (BAD_HOLE=1)"
    )
    rock_flags: Optional[List[int]] = Field(
        None, description="Optional list of discrete ROCK_FLAG values to keep"
    )
    test_size: float = Field(
        0.20, ge=0.10, le=0.45, description="Validation holdout fraction"
    )
    alpha: float = Field(
        0.20,
        ge=0.05,
        le=0.45,
        description="Conformal significance level (0.20 -> P10 to P90)",
    )
    n_estimators: int = Field(
        50, ge=10, le=300, description="Number of Random Forest trees"
    )
    max_depth: int = Field(10, ge=3, le=30, description="Max tree depth")
    save: bool = Field(
        False,
        description="Save the trained model suite to the global `.joblib` bundle for use across all projects",
    )


@router.post(
    "/ml_models/train",
    summary="Train and evaluate a multi-well Conformal PIML model and optionally save to global bundle",
)
def workspace_ml_train(project_id: int, req: WorkspaceMlTrainRequest) -> Dict[str, Any]:
    """Train a multi-well Conformal PIML model with well/zone/flag filtering and return evaluation metrics."""
    try:
        from quick_pp.machine_learning.predict_pipeline import (
            get_pretrained_models_status,
        )
        from quick_pp.machine_learning.train_pipeline import (
            train_multi_well_conformal_model,
        )

        proj_summary = get_project_summary(project_id)
        proj_name = str(proj_summary.get("name") or f"Project_{project_id}")
        well_names = [w["name"] for w in proj_summary.get("wells", [])]
        if req.well_names:
            allowed = {str(w).strip() for w in req.well_names if str(w).strip()}
            if allowed:
                well_names = [w for w in well_names if w in allowed]
        if not well_names:
            raise ValueError("No wells selected or available in project.")

        dfs: List[pd.DataFrame] = []
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            for wn in well_names:
                wdf = _get_merged_dataframe_sync(project_id, wn)
                if wdf.empty:
                    continue
                wdf = wdf.copy()
                wdf["WELL_NAME"] = wn
                if "ZONES" not in wdf.columns or not wdf["ZONES"].notna().any():
                    try:
                        anc = proj.get_well_ancillary_data(wn) if proj else {}
                    except Exception:
                        anc = {}
                    tops_df = (
                        anc.get("formation_tops") if isinstance(anc, dict) else None
                    )
                    if isinstance(tops_df, pd.DataFrame) and not tops_df.empty:
                        wdf["ZONES"] = pd.NA
                        for _, top in tops_df.iterrows():
                            try:
                                top_depth = float(top.get("depth"))
                                top_name = str(top.get("name"))
                            except Exception:
                                continue
                            nearest_idx = (wdf["DEPTH"] - top_depth).abs().idxmin()
                            wdf.at[nearest_idx, "ZONES"] = top_name
                        wdf["ZONES"] = wdf["ZONES"].ffill()
                dfs.append(wdf)

        if not dfs:
            raise ValueError("No log data found for the selected wells.")

        combined_df = pd.concat(dfs, ignore_index=True)
        res = train_multi_well_conformal_model(
            df=combined_df,
            architecture=req.architecture,
            suite_key=req.suite_key,
            suite_label=req.suite_label,
            zones=req.zones,
            exclude_coal=req.exclude_coal,
            exclude_tight=req.exclude_tight,
            exclude_bad_hole=req.exclude_bad_hole,
            rock_flags=req.rock_flags,
            test_size=req.test_size,
            alpha=req.alpha,
            n_estimators=req.n_estimators,
            max_depth=req.max_depth,
            save_model=req.save,
            source_project=proj_name,
        )
        res["models_status"] = get_pretrained_models_status()
        return res
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Multi-well ML training failed: {e}"
        ) from e


class WorkspaceMlSaveRequest(BaseModel):
    suite_key: str = Field(
        "clastic",
        description="Model suite identifier in the global `.joblib` bundle (e.g., 'clastic', 'carbonate', 'volve_v2')",
    )
    suite_label: Optional[str] = Field(
        None, description="Human-readable display name for the model suite"
    )


@router.post(
    "/ml_models/save",
    summary="Save the already-evaluated multi-well Conformal PIML model directly to the global `.joblib` bundle",
)
def workspace_ml_save_model(
    project_id: int, req: WorkspaceMlSaveRequest
) -> Dict[str, Any]:
    """Save the cached trained model suite directly to disk without re-running training."""
    try:
        from quick_pp.machine_learning.predict_pipeline import (
            get_pretrained_models_status,
        )
        from quick_pp.machine_learning.train_pipeline import save_cached_trained_model

        proj_summary = get_project_summary(project_id)
        proj_name = str(proj_summary.get("name") or f"Project_{project_id}")
        res = save_cached_trained_model(
            suite_key=req.suite_key,
            suite_label=req.suite_label,
            source_project=proj_name,
        )
        res["models_status"] = get_pretrained_models_status()
        return res
    except PermissionError as e:
        raise HTTPException(status_code=403, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save trained model: {e}"
        ) from e


@router.delete(
    "/ml_models/{suite_key}",
    summary="Delete a saved model suite from the global `.joblib` bundle",
)
def workspace_ml_delete_model(project_id: int, suite_key: str) -> Dict[str, Any]:
    """Remove a saved model suite from the global `.joblib` bundle and return updated status."""
    from quick_pp.machine_learning.predict_pipeline import (
        delete_pretrained_model_suite,
        get_pretrained_models_status,
    )
    from quick_pp.machine_learning.train_pipeline import is_training_disabled

    if is_training_disabled():
        raise HTTPException(
            status_code=403,
            detail="Modifying pre-trained models is disabled in read-only demo mode.",
        )
    deleted = delete_pretrained_model_suite(suite_key)
    if not deleted:
        raise HTTPException(
            status_code=404,
            detail=f"Model suite '{suite_key}' not found in pre-trained bundle.",
        )
    return {
        "deleted": True,
        "suite_key": suite_key,
        "models_status": get_pretrained_models_status(),
    }


@router.post(
    "/wells/{well_name}/ml_inference",
    summary="Run Pre-Trained PIML & Conformal Prediction inference (Read-Only Training)",
)
def workspace_ml_inference(
    project_id: int, well_name: str, req: WorkspaceMlInferenceRequest
) -> Dict[str, Any]:
    """Trigger pre-trained PIML & Conformal Prediction inference with P10–P90 uncertainty bands."""
    try:
        from quick_pp.machine_learning.predict_pipeline import (
            get_pretrained_models_status,
            predict_well_dataframe,
        )

        status_info = get_pretrained_models_status()
        df = _get_filtered_well_df(
            project_id, well_name, req.min_depth, req.max_depth, req.zones
        )
        pred_df = predict_well_dataframe(
            df,
            model_config=req.model_config_key,
            alpha=float(req.alpha),
        )
        pretrained_used = bool(
            pred_df.attrs.get("pretrained_used", status_info["pretrained_available"])
        )
        model_path_used = pred_df.attrs.get("model_path", status_info["model_path"])
        expected_path = pred_df.attrs.get("expected_path", status_info["expected_path"])

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
        else:
            _update_in_memory_well_cache(project_id, well_name, out_df)

        phit_mean_width = 0.0
        if "PHIT_P10" in out_df.columns and "PHIT_P90" in out_df.columns:
            w = (
                pd.to_numeric(out_df["PHIT_P90"], errors="coerce")
                - pd.to_numeric(out_df["PHIT_P10"], errors="coerce")
            ).dropna()
            if not w.empty:
                phit_mean_width = round(float(w.mean()), 4)

        sw_mean_width = 0.0
        if "SW_P10" in out_df.columns and "SW_P90" in out_df.columns:
            w = (
                pd.to_numeric(out_df["SW_P90"], errors="coerce")
                - pd.to_numeric(out_df["SW_P10"], errors="coerce")
            ).dropna()
            if not w.empty:
                sw_mean_width = round(float(w.mean()), 4)

        perm_factor = 1.0
        if "PERM_P10" in out_df.columns and "PERM_P90" in out_df.columns:
            p10_s = pd.to_numeric(out_df["PERM_P10"], errors="coerce").clip(lower=1e-3)
            p90_s = pd.to_numeric(out_df["PERM_P90"], errors="coerce").clip(lower=1e-3)
            ratio = (p90_s / p10_s).replace([np.inf, -np.inf], np.nan).dropna()
            if not ratio.empty:
                perm_factor = round(float(np.median(ratio)), 2)

        stats = {
            col: _curve_summary_stats(out_df[col]) for col in out_cols if col != "DEPTH"
        }
        model_source = (
            f"Pre-Trained PIML ({req.model_config_key.title()})"
            if pretrained_used
            else f"On-the-Fly Synthetic Conformal Fallback ({req.model_config_key.title()})"
        )
        notice = (
            f"Inference executed using pre-trained bundle '{model_path_used}'."
            if pretrained_used
            else (
                f"Pre-trained model file not found at '{expected_path}'. "
                "Used on-the-fly physics-guided synthetic conformal fallback on current well logs."
            )
        )
        return {
            "well_name": well_name,
            "model_config": req.model_config_key,
            "alpha": float(req.alpha),
            "confidence_level_pct": round((1.0 - float(req.alpha)) * 100.0, 1),
            "model_source": model_source,
            "pretrained_used": pretrained_used,
            "fallback_used": not pretrained_used,
            "model_path": model_path_used,
            "expected_path": expected_path,
            "notice": notice,
            "interval_width_summary": {
                "phit_mean_width_vv": phit_mean_width,
                "sw_mean_width_vv": sw_mean_width,
                "perm_median_p90_p10_factor": perm_factor,
            },
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
    header: Optional[Dict[str, Any]] = Field(
        None, description="Well header metadata updates"
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
    """Persist working top and bottom focus depth (and optional well config/header keys) for a specific well."""
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

        clean_header_updates: Dict[str, Any] = {}
        if req.header and isinstance(req.header, dict):
            for hk, hv in req.header.items():
                if hk in ("raw_well_header", "raw_parameters"):
                    continue
                clean_header_updates[hk] = hv
            eph.update(clean_header_updates)
            curr_eph_hdr = dict(_ephemeral_well_headers.get((project_id, well_name), {}))
            curr_eph_hdr.update(clean_header_updates)
            _ephemeral_well_headers[(project_id, well_name)] = curr_eph_hdr

        _ephemeral_well_configs[(project_id, well_name)] = eph

        saved_cfg = dict(eph)
        saved_hdr = dict(_ephemeral_well_headers.get((project_id, well_name), {}))
        if not is_demo_mode():
            with get_db() as session:
                db_name = resolve_db_well_name(project_id, well_name)
                well_orm = session.scalar(
                    select(db_models.Well).filter_by(
                        project_id=project_id, name=db_name
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
                    well_orm.config_data = _sanitize_for_json(current_cfg)
                    flag_modified(well_orm, "config_data")

                    if clean_header_updates:
                        current_hdr = (
                            dict(well_orm.header_data)
                            if isinstance(well_orm.header_data, dict)
                            else {}
                        )
                        current_hdr.update(clean_header_updates)
                        ovr = (
                            dict(current_hdr.get("header_overrides"))
                            if isinstance(current_hdr.get("header_overrides"), dict)
                            else {}
                        )
                        ovr.update(clean_header_updates)
                        current_hdr["header_overrides"] = ovr
                        if isinstance(current_hdr.get("well_metadata"), dict):
                            wm = dict(current_hdr["well_metadata"])
                            wm.update(clean_header_updates)
                            current_hdr["well_metadata"] = wm
                        if clean_header_updates.get("uwi"):
                            well_orm.uwi = str(clean_header_updates["uwi"]).strip()
                        well_orm.header_data = _sanitize_for_json(current_hdr)
                        flag_modified(well_orm, "header_data")
                        saved_hdr = dict(
                            extract_las_metadata(
                                well_orm.header_data, fallback_well_name=well_name
                            )
                        )
                        saved_hdr.update(ovr)

                    session.add(well_orm)
                    session.flush()
                    saved_cfg = current_cfg

        return _sanitize_for_json({
            "project_id": project_id,
            "well_name": well_name,
            "config": saved_cfg,
            "header": saved_hdr,
            "saved": True,
        })
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
        description="Action: 'transform', 'duplicate', 'rename', 'delete', 'splice', or 'save_values'",
    )
    curve: str = Field(..., description="Source curve mnemonic")
    target_curve: Optional[str] = Field(
        None,
        description="Target curve mnemonic (defaults to source curve for in-place transform)",
    )
    source_curves: Optional[List[str]] = Field(
        None,
        description="Ordered list of curve mnemonics to splice when action='splice'",
    )
    overlap_mode: Optional[str] = Field(
        "blend",
        description="Overlap resolution mode for 'splice': 'blend', 'splice_depth', 'keep_first', 'keep_last', or 'mean'",
    )
    splice_depth: Optional[float] = Field(
        None,
        description="Explicit splice tie-in depth (m) when overlap_mode='splice_depth'",
    )
    auto_level_overlap: bool = Field(
        False,
        description="Automatically shift secondary run to match primary run mean in overlap interval",
    )
    interpolate_gaps: bool = Field(
        False,
        description="Linearly interpolate small gaps between spliced runs",
    )
    max_gap_m: float = Field(
        2.0,
        description="Maximum gap width (m) to linearly interpolate when interpolate_gaps=True",
    )
    operation: Optional[str] = Field(
        None,
        description="Transform operation: 'shift', 'scale', 'linear', 'clip', 'null_outside', 'depth_shift', 'smooth'",
    )
    value: float = Field(
        0.0, description="Shift offset, scale factor, or depth shift (m)"
    )
    scale: float = Field(
        1.0, description="Multiplier for 'linear' (Y = scale * X + offset)"
    )
    offset: float = Field(
        0.0, description="Offset for 'linear' (Y = scale * X + offset)"
    )
    min_val: Optional[float] = Field(
        None, description="Minimum bound for 'clip' or 'null_outside'"
    )
    max_val: Optional[float] = Field(
        None, description="Maximum bound for 'clip' or 'null_outside'"
    )
    window: int = Field(
        5, description="Window size (samples) for 'smooth' median/moving average"
    )
    depths: Optional[List[float]] = Field(
        None, description="Optional explicit depth array when action='save_values'"
    )
    values: Optional[List[Optional[float]]] = Field(
        None,
        description="Optional explicit curve values array when action='save_values'",
    )
    ori_curve: Optional[str] = Field(
        None, description="Optional backup original curve mnemonic (e.g. GR_ORI)"
    )
    ori_values: Optional[List[Optional[float]]] = Field(
        None, description="Optional backup original curve values array"
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
        dst = req.target_curve.strip().upper() if req.target_curve else src

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
                {
                    dst: _curve_summary_stats(
                        full_df[dst],
                        full_df["DEPTH"] if "DEPTH" in full_df.columns else None,
                    )
                }
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
            cols_dict: Dict[str, Any] = {
                "DEPTH": pd.to_numeric(pd.Series(req.depths), errors="coerce"),
                dst: pd.to_numeric(pd.Series(req.values), errors="coerce"),
            }
            ori_name = (
                req.ori_curve.strip().upper()
                if req.ori_curve and req.ori_curve.strip()
                else None
            )
            if (
                ori_name
                and ori_name not in protected
                and ori_name != dst
                and req.ori_values is not None
                and len(req.ori_values) == len(req.depths)
            ):
                cols_dict[ori_name] = pd.to_numeric(
                    pd.Series(req.ori_values), errors="coerce"
                )
            out_df = pd.DataFrame(cols_dict).dropna(subset=["DEPTH"])
            saved_rows = 0
            if req.save:
                saved_rows = _save_curves_to_well(project_id, well_name, out_df)
            else:
                _update_in_memory_well_cache(project_id, well_name, out_df)
            stats_out = {dst: _curve_summary_stats(out_df[dst], out_df["DEPTH"])}
            if ori_name and ori_name in out_df.columns:
                stats_out[ori_name] = _curve_summary_stats(
                    out_df[ori_name], out_df["DEPTH"]
                )
            return {
                "well_name": well_name,
                "action": "save_values",
                "target_curve": dst,
                "ori_curve": ori_name
                if (ori_name and ori_name in out_df.columns)
                else None,
                "saved": req.save,
                "saved_rows": saved_rows,
                "stats": stats_out,
                **dataframe_to_columnar(out_df),
            }

        full_df = _get_merged_dataframe_sync(project_id, well_name)
        if full_df.empty:
            raise ValueError(f"No data found in well '{well_name}'")

        if act == "splice":
            src_list = [
                c.strip().upper()
                for c in (req.source_curves or [src])
                if c and c.strip()
            ]
            valid_srcs = [c for c in src_list if c in full_df.columns]
            if len(valid_srcs) < 2:
                raise ValueError(
                    "At least two valid curves are required to splice runs."
                )
            if not dst or dst in protected:
                raise ValueError("Valid target curve mnemonic is required for splice.")

            depth_arr = pd.to_numeric(full_df["DEPTH"], errors="coerce").to_numpy(
                dtype=float
            )
            acc = pd.to_numeric(full_df[valid_srcs[0]], errors="coerce").to_numpy(
                dtype=float
            ).copy()
            ov_mode = (req.overlap_mode or "blend").strip().lower()

            for next_col in valid_srcs[1:]:
                nxt = (
                    pd.to_numeric(full_df[next_col], errors="coerce")
                    .to_numpy(dtype=float)
                    .copy()
                )
                valid_a = np.isfinite(acc) & np.isfinite(depth_arr)
                valid_b = np.isfinite(nxt) & np.isfinite(depth_arr)
                overlap = valid_a & valid_b

                if req.auto_level_overlap and np.any(overlap):
                    bias_delta = float(np.mean(acc[overlap] - nxt[overlap]))
                    nxt[valid_b] = nxt[valid_b] + bias_delta

                only_b = (~valid_a) & valid_b
                if ov_mode == "keep_first":
                    acc[only_b] = nxt[only_b]
                elif ov_mode == "keep_last":
                    acc[valid_b] = nxt[valid_b]
                elif ov_mode == "mean":
                    acc[only_b] = nxt[only_b]
                    acc[overlap] = 0.5 * (acc[overlap] + nxt[overlap])
                elif ov_mode == "splice_depth":
                    top_a = float(np.min(depth_arr[valid_a])) if np.any(valid_a) else 0.0
                    top_b = float(np.min(depth_arr[valid_b])) if np.any(valid_b) else 0.0
                    if req.splice_depth is not None and math.isfinite(float(req.splice_depth)):
                        cutoff = float(req.splice_depth)
                    elif np.any(overlap):
                        cutoff = 0.5 * (
                            float(np.min(depth_arr[overlap]))
                            + float(np.max(depth_arr[overlap]))
                        )
                    else:
                        cutoff = 0.5 * (top_a + top_b)
                    upper_arr = acc if top_a <= top_b else nxt
                    lower_arr = nxt if top_a <= top_b else acc
                    u_valid = np.isfinite(upper_arr)
                    l_valid = np.isfinite(lower_arr)
                    combined = np.full_like(acc, np.nan)
                    above = depth_arr <= cutoff
                    below = ~above
                    combined[above] = np.where(
                        u_valid[above], upper_arr[above], lower_arr[above]
                    )
                    combined[below] = np.where(
                        l_valid[below], lower_arr[below], upper_arr[below]
                    )
                    acc = combined
                else:  # blend (linear feather across overlap)
                    acc[only_b] = nxt[only_b]
                    if np.any(overlap):
                        ov_d = depth_arr[overlap]
                        d_min = float(np.min(ov_d))
                        d_max = float(np.max(ov_d))
                        top_a = (
                            float(np.min(depth_arr[valid_a]))
                            if np.any(valid_a)
                            else d_min
                        )
                        top_b = (
                            float(np.min(depth_arr[valid_b]))
                            if np.any(valid_b)
                            else d_min
                        )
                        span = d_max - d_min
                        if span > 1e-6:
                            w = (ov_d - d_min) / span
                            if top_a <= top_b:
                                acc[overlap] = (1.0 - w) * acc[overlap] + w * nxt[overlap]
                            else:
                                acc[overlap] = w * acc[overlap] + (1.0 - w) * nxt[overlap]
                        else:
                            acc[overlap] = 0.5 * (acc[overlap] + nxt[overlap])

            s_spliced = pd.Series(acc, index=full_df.index, dtype=float)
            if req.interpolate_gaps:
                s_spliced = s_spliced.interpolate(
                    method="linear", limit=max(1, int(round(float(req.max_gap_m) / 0.1524))), limit_area="inside"
                )

            out_df = pd.DataFrame({"DEPTH": full_df["DEPTH"], dst: s_spliced})
            saved_rows = 0
            if req.save:
                saved_rows = _save_curves_to_well(project_id, well_name, out_df)
            else:
                _update_in_memory_well_cache(project_id, well_name, out_df)

            return {
                "well_name": well_name,
                "action": "splice",
                "source_curves": valid_srcs,
                "target_curve": dst,
                "saved": req.save,
                "saved_rows": saved_rows,
                "stats": {dst: _curve_summary_stats(out_df[dst], out_df["DEPTH"])},
                **dataframe_to_columnar(out_df),
            }

        if src not in full_df.columns:
            raise ValueError(f"Curve '{src}' not found in well '{well_name}'")

        if act == "duplicate":
            if not dst or dst in protected:
                raise ValueError(
                    "Valid target curve mnemonic is required for duplicate"
                )
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
                "stats": {dst: _curve_summary_stats(out_df[dst], out_df["DEPTH"])},
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
            depth_arr = pd.to_numeric(df["DEPTH"], errors="coerce").to_numpy(
                dtype=float
            )
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
            "stats": {dst: _curve_summary_stats(s, df["DEPTH"])},
            **dataframe_to_columnar(out_df),
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Curve edit failed: {e}") from e


class WorkspaceSaveComputedRequest(BaseModel):
    """Persist already-computed in-memory curves for a single-well module without re-running calculations."""

    module: Optional[str] = Field(
        default=None,
        description="Module identifier (e.g. litho-poro, perm, saturation, ressum, qaqc-edit, qaqc-env)",
    )
    depths: List[Optional[float]] = Field(
        default_factory=list,
        description="Depth array aligned with the provided curves",
    )
    curves: Dict[str, List[Optional[float]]] = Field(
        default_factory=dict,
        description="Map of curve mnemonic -> values array aligned with depths",
    )


@router.post(
    "/wells/{well_name}/save_computed",
    summary="Save already-computed single-well curves directly without recalculating",
)
def workspace_save_computed_curves(
    project_id: int,
    well_name: str,
    req: WorkspaceSaveComputedRequest,
) -> Dict[str, Any]:
    """Directly persist already-computed curves to the well without re-running petrophysical calculations."""
    try:
        if not req.depths or not req.curves:
            raise ValueError("No computed curve values provided to save.")

        depth_series = pd.to_numeric(pd.Series(req.depths), errors="coerce")
        cols: Dict[str, Any] = {"DEPTH": depth_series}
        saved_curve_names: List[str] = []

        for raw_col, vals in req.curves.items():
            col = str(raw_col or "").strip().upper()
            if not col or col in ("DEPTH", "DEPT", "WELL_NAME", "ZONES"):
                continue
            if len(vals) != len(req.depths):
                raise ValueError(
                    f"Curve '{col}' length ({len(vals)}) does not match depths length ({len(req.depths)})"
                )
            cols[col] = pd.to_numeric(pd.Series(vals), errors="coerce")
            saved_curve_names.append(col)

        if not saved_curve_names:
            raise ValueError("No valid curve columns found to save.")

        out_df = pd.DataFrame(cols).dropna(subset=["DEPTH"]).reset_index(drop=True)
        saved_rows = _save_curves_to_well(project_id, well_name, out_df)

        return {
            "well_name": well_name,
            "module": req.module,
            "saved": True,
            "saved_rows": saved_rows,
            "saved_curves": saved_curve_names,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save computed curves: {e}"
        ) from e


class WorkspaceSaveMultiWellComputedRequest(BaseModel):
    """Persist already-computed in-memory curves across multiple wells without re-running calculations."""

    module: Optional[str] = Field(
        default=None,
        description="Multi-well module identifier (e.g. rock-typing, perm-transform, shf)",
    )
    well_names: List[str] = Field(
        default_factory=list,
        description="Well names array aligned with depths and curves",
    )
    depths: List[Optional[float]] = Field(
        default_factory=list,
        description="Depth array aligned with well_names and curves",
    )
    curves: Dict[str, List[Optional[float]]] = Field(
        default_factory=dict,
        description="Map of curve mnemonic -> values array aligned with well_names and depths",
    )


@router.post(
    "/save_multi_well_computed",
    summary="Save already-computed multi-well curves across project wells without recalculating",
)
def workspace_save_multi_well_computed_curves(
    project_id: int,
    req: WorkspaceSaveMultiWellComputedRequest,
) -> Dict[str, Any]:
    """Directly persist already-computed multi-well curves to each well without re-running calculations."""
    try:
        n_rows = len(req.depths)
        if n_rows == 0 or len(req.well_names) != n_rows or not req.curves:
            raise ValueError(
                "No valid multi-well computed curve values provided to save."
            )

        cols: Dict[str, Any] = {
            "WELL_NAME": [str(w).strip() for w in req.well_names],
            "DEPTH": pd.to_numeric(pd.Series(req.depths), errors="coerce"),
        }
        saved_curve_names: List[str] = []

        for raw_col, vals in req.curves.items():
            col = str(raw_col or "").strip().upper()
            if not col or col in ("DEPTH", "DEPT", "WELL_NAME", "ZONES"):
                continue
            if len(vals) != n_rows:
                raise ValueError(
                    f"Curve '{col}' length ({len(vals)}) does not match depths length ({n_rows})"
                )
            cols[col] = pd.to_numeric(pd.Series(vals), errors="coerce")
            saved_curve_names.append(col)

        if not saved_curve_names:
            raise ValueError("No valid curve columns found to save.")

        full_df = pd.DataFrame(cols).dropna(subset=["WELL_NAME", "DEPTH"])
        saved_wells = 0
        for wn, group_df in full_df.groupby("WELL_NAME", sort=False):
            well_df = group_df.drop(columns=["WELL_NAME"]).dropna(
                how="all", subset=saved_curve_names
            )
            if not well_df.empty:
                _save_curves_to_well(
                    project_id, str(wn), well_df.reset_index(drop=True)
                )
                saved_wells += 1

        return {
            "project_id": project_id,
            "module": req.module,
            "saved": True,
            "saved_wells": saved_wells,
            "saved_curves": saved_curve_names,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save multi-well computed curves: {e}"
        ) from e


CALCULATED_CURVES_BY_GROUP: Dict[str, Tuple[str, ...]] = {
    "qaqc": (
        "BADHOLE",
        "BAD_HOLE",
        "BAD_HOLE_FLAG",
        "BS",
        "DCALI",
        "GR_CORR",
        "NPHI_CORR",
        "RHOB_CORR",
        "RT_CORR",
        "GR_NORM",
        "NPHI_NORM",
        "NPHI_HC",
        "RHOB_HC",
        "HC_FLAG",
        "RHOB_DEN_CORR",
        "DEN_CORR_FLAG",
    ),
    "litho_poro": (
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
        "COAL_FLAG",
        "VSH_GR",
        "VSH_DN",
        "PHIT",
        "PHIE",
        "PHID",
        "PHIT_SH",
        "VSH_LAM",
        "VSH_DIS",
        "PHIT_SAND",
        "RSAND_TS",
        "RT_MOD",
        "PHIT_P10",
        "PHIT_P50",
        "PHIT_P90",
        "PHIE_P10",
        "PHIE_P50",
        "PHIE_P90",
        "VCLAY_P10",
        "VCLAY_P50",
        "VCLAY_P90",
    ),
    "perm": (
        "PERM",
        "PERM_TRANS",
        "ROCK_FLAG",
        "FZI",
        "PERM_P10",
        "PERM_P50",
        "PERM_P90",
    ),
    "saturation": (
        "SWT",
        "SWE",
        "SWT_ARCHIE",
        "SWT_WAXMAN",
        "BVW",
        "TEMP_GRAD",
        "RW",
        "B",
        "QVN",
        "M_STAR",
        "VCL_OVER_PHI",
        "CWA",
        "SHF",
        "SW_SHF",
        "SW_P10",
        "SW_P50",
        "SW_P90",
    ),
    "ressum": (
        "RES_FLAG",
        "PAY_FLAG",
        "PAY_FLAGS",
        "PHIT_BLK",
        "PHIE_BLK",
        "VCLAY_BLK",
        "PERM_BLK",
        "SWT_BLK",
        "BVHC_P10",
        "BVHC_P50",
        "BVHC_P90",
    ),
}

ALL_CALCULATED_CURVE_MNEMONICS: Tuple[str, ...] = tuple(
    dict.fromkeys(
        mnem
        for grp_cols in CALCULATED_CURVES_BY_GROUP.values()
        for mnem in grp_cols
    )
)


class WorkspaceClearCalculatedRequest(BaseModel):
    """Clear calculated/evaluated curves for a well either top-to-bottom or by formation/depth interval."""

    scope: str = Field(
        default="full",
        description="Clear scope: 'full' (top to bottom across entire well), 'formation' (selected zones), or 'filtered' (zones and/or depth interval)",
    )
    groups: Optional[List[str]] = Field(
        default=None,
        description="Optional subset of calculated curve groups to clear: 'qaqc', 'litho_poro', 'perm', 'saturation', 'ressum'. Defaults to all groups.",
    )
    curves: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit list of curve mnemonics to clear. Overrides `groups` if provided.",
    )
    zones: Optional[List[str]] = Field(
        default=None,
        description="Formation/zone names to clear when scope is 'formation' or 'filtered'.",
    )
    min_depth: Optional[float] = Field(
        default=None,
        description="Minimum depth (m) to clear when clearing a depth interval.",
    )
    max_depth: Optional[float] = Field(
        default=None,
        description="Maximum depth (m) to clear when clearing a depth interval.",
    )
    depths: Optional[List[float]] = Field(
        default=None,
        description="Optional explicit depth values to clear (nullify) for partial-well clear.",
    )
    save: bool = Field(
        default=True,
        description="Persist the cleared state to the database (and in-memory cache).",
    )


@router.post(
    "/wells/{well_name}/clear_calculated",
    summary="Clear calculated curves top-to-bottom or for selected formations/depth intervals",
)
def workspace_clear_calculated_curves(
    project_id: int,
    well_name: str,
    req: WorkspaceClearCalculatedRequest,
) -> Dict[str, Any]:
    """Clear all or selected calculated petrophysical curves either top-to-bottom (full well) or within specific formations/depth intervals."""
    try:
        import time
        from sqlalchemy import bindparam, select, text
        from quick_pp.database import models as db_models

        protected = {"DEPTH", "DEPT", "WELL_NAME", "ZONES"}
        if req.curves:
            target_candidates = [
                str(c).strip().upper()
                for c in req.curves
                if str(c).strip() and str(c).strip().upper() not in protected
            ]
        elif req.groups:
            target_candidates = []
            for g in req.groups:
                g_key = str(g).strip().lower().replace("-", "_")
                for col in CALCULATED_CURVES_BY_GROUP.get(g_key, ()):
                    if col not in target_candidates:
                        target_candidates.append(col)
        else:
            target_candidates = list(ALL_CALCULATED_CURVE_MNEMONICS)

        if not target_candidates:
            return {
                "well_name": well_name,
                "scope": req.scope,
                "cleared_curves": [],
                "dropped_curves": [],
                "cleared_rows": 0,
                "saved": req.save,
                "stats": {},
            }

        full_df = _get_merged_dataframe_sync(project_id, well_name)
        present_targets = (
            [c for c in target_candidates if c in full_df.columns]
            if not full_df.empty
            else list(target_candidates)
        )

        scope_norm = (req.scope or "full").strip().lower()
        valid_zones = (
            [str(z).strip() for z in (req.zones or []) if str(z).strip()]
            if req.zones
            else []
        )
        has_partial_filter = (
            scope_norm in ("formation", "filtered", "depth_range")
            and (
                bool(valid_zones)
                or req.min_depth is not None
                or req.max_depth is not None
                or bool(req.depths)
            )
        )

        # Case 1: Full-well (Top to Bottom) clear -> drop calculated curve columns completely
        if not has_partial_filter:
            for key, (_, cdf) in list(_merged_data_cache.items()):
                if key.startswith(f"{project_id}_{well_name}_"):
                    updated = cdf.drop(columns=target_candidates, errors="ignore")
                    _merged_data_cache[key] = (time.time(), updated)

            eph_proj = _ephemeral_wells.get(project_id, {})
            if well_name in eph_proj:
                eph_entry = eph_proj[well_name]
                if isinstance(eph_entry, pd.DataFrame):
                    eph_proj[well_name] = eph_entry.drop(
                        columns=target_candidates, errors="ignore"
                    )
                elif isinstance(eph_entry, dict) and isinstance(
                    eph_entry.get("data"), pd.DataFrame
                ):
                    eph_entry["data"] = eph_entry["data"].drop(
                        columns=target_candidates, errors="ignore"
                    )

            if req.save and not is_demo_mode():
                db_name = resolve_db_well_name(project_id, well_name)
                with get_db() as session:
                    well_orm = session.scalar(
                        select(db_models.Well).filter_by(
                            project_id=project_id, name=db_name
                        )
                    )
                    if well_orm:
                        cids = [
                            c.curve_id
                            for c in well_orm.curves
                            if str(c.mnemonic).upper() in set(target_candidates)
                            and c.curve_id is not None
                        ]
                        if cids:
                            session.execute(
                                text(
                                    "DELETE FROM curve_data WHERE curve_id IN :cids"
                                ).bindparams(bindparam("cids", expanding=True)),
                                {"cids": cids},
                            )
                            session.execute(
                                text(
                                    "DELETE FROM curves WHERE curve_id IN :cids"
                                ).bindparams(bindparam("cids", expanding=True)),
                                {"cids": cids},
                            )
                            session.flush()
                invalidate_well_cache(project_id=project_id, well_name=well_name)

            return {
                "well_name": well_name,
                "scope": "full",
                "cleared_curves": present_targets,
                "dropped_curves": present_targets,
                "cleared_rows": int(len(full_df)) if not full_df.empty else 0,
                "saved": req.save,
                "stats": {},
            }

        # Case 2: Partial clear (by formation/zone, depth interval, or explicit depths)
        if full_df.empty or "DEPTH" not in full_df.columns:
            raise ValueError(f"No depth data found for well '{well_name}'")

        depth_s = pd.to_numeric(full_df["DEPTH"], errors="coerce")
        clear_mask = pd.Series(True, index=full_df.index, dtype=bool) & depth_s.notna()

        if req.depths:
            req_d_arr = np.round(
                pd.to_numeric(pd.Series(req.depths), errors="coerce")
                .dropna()
                .to_numpy(dtype=float),
                4,
            )
            full_d_round = np.round(depth_s.fillna(-999999.0).to_numpy(dtype=float), 4)
            clear_mask = clear_mask & pd.Series(
                np.isin(full_d_round, req_d_arr), index=full_df.index
            )
        else:
            if req.min_depth is not None:
                clear_mask = clear_mask & (depth_s >= float(req.min_depth))
            if req.max_depth is not None:
                clear_mask = clear_mask & (depth_s <= float(req.max_depth))
            if valid_zones:
                allowed_z = {z.upper() for z in valid_zones}
                zone_col_s = (
                    full_df["ZONES"].astype(str).str.strip().str.upper()
                    if "ZONES" in full_df.columns
                    else pd.Series("", index=full_df.index, dtype=str)
                )
                zone_match = zone_col_s.isin(allowed_z)

                # Also check formation_tops intervals in case ZONES column was not pre-filled
                if not zone_match.any():
                    try:
                        db_name = resolve_db_well_name(project_id, well_name)
                        with get_db() as session:
                            proj = db_objects.Project.load(
                                session, project_id=project_id
                            )
                            well_obj = proj.get_well(db_name)
                            tops_df = well_obj.get_formation_tops()
                            if tops_df is not None and not tops_df.empty:
                                tops_sorted = tops_df.sort_values("depth").reset_index(
                                    drop=True
                                )
                                for idx_t, trow in tops_sorted.iterrows():
                                    tname = str(trow.get("name") or "").strip().upper()
                                    if tname in allowed_z:
                                        top_d = float(trow["depth"])
                                        bot_d = (
                                            float(tops_sorted.iloc[idx_t + 1]["depth"])
                                            if idx_t + 1 < len(tops_sorted)
                                            else float(depth_s.max()) + 1.0
                                        )
                                        if idx_t + 1 < len(tops_sorted):
                                            zone_match = zone_match | (
                                                (depth_s >= top_d) & (depth_s < bot_d)
                                            )
                                        else:
                                            zone_match = zone_match | (
                                                (depth_s >= top_d) & (depth_s <= bot_d)
                                            )
                    except Exception:
                        pass
                clear_mask = clear_mask & zone_match

        cleared_row_count = int(clear_mask.sum())
        if cleared_row_count == 0:
            return {
                "well_name": well_name,
                "scope": scope_norm,
                "cleared_curves": [],
                "dropped_curves": [],
                "cleared_rows": 0,
                "saved": req.save,
                "stats": {},
            }

        cleared_depths_list = [
            round(float(d), 4) for d in depth_s.loc[clear_mask].tolist()
        ]
        dropped_curves: List[str] = []
        remaining_stats: Dict[str, Any] = {}

        # Update in-memory DataFrame
        updated_full = full_df.copy()
        for col in present_targets:
            col_s = pd.to_numeric(updated_full[col], errors="coerce")
            col_s.loc[clear_mask] = np.nan
            if col_s.notna().any():
                updated_full[col] = col_s
                remaining_stats[col] = _curve_summary_stats(col_s, depth_s)
            else:
                updated_full = updated_full.drop(columns=[col], errors="ignore")
                dropped_curves.append(col)

        for key in list(_merged_data_cache.keys()):
            if key.startswith(f"{project_id}_{well_name}_"):
                _merged_data_cache[key] = (time.time(), updated_full.copy())

        eph_proj = _ephemeral_wells.get(project_id, {})
        if well_name in eph_proj:
            eph_entry = eph_proj[well_name]
            if isinstance(eph_entry, pd.DataFrame):
                eph_proj[well_name] = updated_full.copy()
            elif isinstance(eph_entry, dict) and isinstance(
                eph_entry.get("data"), pd.DataFrame
            ):
                eph_entry["data"] = updated_full.copy()
        elif is_demo_mode():
            _ephemeral_wells.setdefault(project_id, {})[well_name] = updated_full.copy()

        if req.save and not is_demo_mode():
            db_name = resolve_db_well_name(project_id, well_name)
            with get_db() as session:
                well_orm = session.scalar(
                    select(db_models.Well).filter_by(
                        project_id=project_id, name=db_name
                    )
                )
                if well_orm:
                    target_set = set(present_targets)
                    dropped_set = set(dropped_curves)
                    matching_curves = [
                        c
                        for c in well_orm.curves
                        if str(c.mnemonic).upper() in target_set
                        and c.curve_id is not None
                    ]
                    drop_cids = [
                        c.curve_id
                        for c in matching_curves
                        if str(c.mnemonic).upper() in dropped_set
                    ]
                    partial_cids = [
                        c.curve_id
                        for c in matching_curves
                        if str(c.mnemonic).upper() not in dropped_set
                    ]
                    if drop_cids:
                        session.execute(
                            text(
                                "DELETE FROM curve_data WHERE curve_id IN :cids"
                            ).bindparams(bindparam("cids", expanding=True)),
                            {"cids": drop_cids},
                        )
                        session.execute(
                            text(
                                "DELETE FROM curves WHERE curve_id IN :cids"
                            ).bindparams(bindparam("cids", expanding=True)),
                            {"cids": drop_cids},
                        )
                    if partial_cids and cleared_depths_list:
                        del_stmt = text(
                            "DELETE FROM curve_data WHERE curve_id IN :cids AND ROUND(depth, 4) IN :depths"
                        ).bindparams(
                            bindparam("cids", expanding=True),
                            bindparam("depths", expanding=True),
                        )
                        chunk_sz = 800
                        for i in range(0, len(cleared_depths_list), chunk_sz):
                            d_chunk = cleared_depths_list[i : i + chunk_sz]
                            session.execute(
                                del_stmt, {"cids": partial_cids, "depths": d_chunk}
                            )
                    session.flush()
            invalidate_well_cache(project_id=project_id, well_name=well_name)
            _merged_data_cache[f"{project_id}_{well_name}_0.16"] = (
                time.time(),
                updated_full.copy(),
            )

        return {
            "well_name": well_name,
            "scope": scope_norm,
            "zones": valid_zones,
            "cleared_curves": present_targets,
            "dropped_curves": dropped_curves,
            "cleared_rows": cleared_row_count,
            "saved": req.save,
            "stats": remaining_stats,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to clear calculated curves: {e}"
        ) from e

