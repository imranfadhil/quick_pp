"""Formation Pressure Test (FPT) fluid gradient & contact analysis API.

Multi-well, standalone module (not part of the sequential workflow):
  * GET   /fpt/points          -> pressure points across wells with resolved TVDSS & zone
  * POST  /fpt/analyze         -> gradient regression, densities, contacts (P10/P50/P90), excess pressure
  * PATCH /fpt/points          -> persist per-point fluid phase / test quality / zone / TVDSS override
  * POST  /fpt/apply-contacts  -> write estimated contacts (TVDSS -> MD) into each well's fluid contacts

Storage units are psi (pressure) and metres (MD / TVDSS); analysis results are returned in the
requested display units (psi | bar | kPa, m | ft).
"""

from typing import Any, Dict, List, Literal, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from quick_pp import logger
from quick_pp.app.backend.services.database import (
    _get_merged_dataframe_sync,
    _sanitize_for_json,
    invalidate_well_cache,
    resolve_db_well_name,
)
from quick_pp.app.backend.services.workspace import _resolve_well_tvd_array
from quick_pp.app.backend.utils.db import get_db
from quick_pp.database import objects as db_objects
from quick_pp.fluid_type import (
    FLUID_PHASES,
    FormationPressureAnalyzer,
    density_to_gradient,
    gradient_to_density,
)

router = APIRouter(
    prefix="/workspace/projects/{project_id}/fpt",
    tags=["Formation Pressure & Contacts"],
)

# Storage (psi, m) -> display unit factors
PSI_TO: Dict[str, float] = {"psi": 1.0, "bar": 0.0689475729, "kPa": 6.89475729}
M_TO: Dict[str, float] = {"m": 1.0, "ft": 1.0 / 0.3048}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class ManualGradientOverride(BaseModel):
    gradient: Optional[float] = Field(
        None, description="Fixed gradient in display units (pressure_unit / depth_unit)"
    )
    density_gcc: Optional[float] = Field(
        None, description="Alternative to `gradient`: fixed in-situ density (g/cc)"
    )
    anchor_tvdss: Optional[float] = Field(
        None, description="Anchor TVDSS (display depth unit) for a manual / regional line"
    )
    anchor_pressure: Optional[float] = Field(
        None, description="Anchor pressure (display pressure unit) for a manual / regional line"
    )
    std_err_gradient: Optional[float] = Field(0.0, ge=0.0)


class PointOverride(BaseModel):
    fluid_type: Optional[str] = None
    test_quality: Optional[str] = None


class AutoClassifyOptions(BaseModel):
    min_mobility: Optional[float] = Field(None, description="Flag TIGHT below this mobility (mD/cP)")
    ransac_residual_threshold: Optional[float] = Field(
        None,
        description="Supercharge threshold (display pressure unit); 0 = automatic (2.5 x MAD)",
    )
    assign_unassigned: bool = True


class FPTAnalysisRequest(BaseModel):
    well_names: Optional[List[str]] = None
    zones: Optional[List[str]] = None
    active_zone: Optional[str] = None
    group_by_zone: bool = True
    pressure_unit: Literal["psi", "bar", "kPa"] = "psi"
    depth_unit: Literal["m", "ft"] = "m"
    regression_method: Literal["ols", "huber", "ransac"] = "ols"
    manual_overrides: Optional[Dict[str, ManualGradientOverride]] = None
    overrides_by_zone: Optional[Dict[str, Dict[str, ManualGradientOverride]]] = None
    point_overrides: Optional[Dict[str, PointOverride]] = None
    auto_classify: Optional[AutoClassifyOptions] = None
    n_simulations: int = Field(2000, ge=0, le=20000)
    ref_water_gradient: Optional[float] = None
    datum_tvdss: Optional[float] = None
    datum_pressure: Optional[float] = None


class FPTPointUpdate(BaseModel):
    test_id: int
    well_name: str
    fluid_type: Optional[str] = None
    test_quality: Optional[str] = None
    zone: Optional[str] = None
    tvdss: Optional[float] = None
    fields: Optional[List[str]] = Field(
        None,
        description="Fields to update; defaults to every non-null field supplied",
    )


class FPTPointsPatchRequest(BaseModel):
    updates: List[FPTPointUpdate] = Field(default_factory=list)


class FPTApplyContactsRequest(BaseModel):
    contacts: Dict[str, float] = Field(
        ..., description="Contact TVDSS in metres keyed by type (GOC, OWC, GWC)"
    )
    well_names: Optional[List[str]] = None
    owc_name: Literal["FWL", "OWC"] = "FWL"


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def _zone_for_md(md: float, tops: List[Dict[str, Any]]) -> str:
    zone = ""
    for t in tops:
        if t["depth"] <= md + 1e-6:
            zone = t["name"]
        else:
            break
    return zone


def _collect_fpt_points(
    project_id: int, well_names: Optional[List[str]] = None
) -> List[Dict[str, Any]]:
    """Gather pressure tests across wells with TVDSS resolved from override, TVD/TVDSS curve or survey."""
    allowed = {str(w).strip() for w in (well_names or []) if str(w).strip()}
    points: List[Dict[str, Any]] = []
    with get_db() as session:
        proj = db_objects.Project.load(session, project_id=project_id)
        names = proj.get_well_names()
        if allowed:
            names = [w for w in names if w in allowed]
        for wn in names:
            try:
                well = proj.get_well(resolve_db_well_name(project_id, wn))
                ptdf = well.get_pressure_tests()
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(f"FPT: failed loading pressure tests for '{wn}': {exc}")
                continue
            if ptdf is None or ptdf.empty:
                continue
            try:
                surveys_df = well.get_well_surveys()
            except Exception:
                surveys_df = None
            try:
                tops_df = well.get_formation_tops()
                tops = sorted(
                    [
                        {"name": str(r["name"]).strip(), "depth": float(r["depth"])}
                        for _, r in tops_df.iterrows()
                        if pd.notna(r.get("depth")) and str(r.get("name") or "").strip()
                    ],
                    key=lambda t: t["depth"],
                )
            except Exception:
                tops = []

            md_arr = pd.to_numeric(ptdf["depth"], errors="coerce").to_numpy(dtype=float)
            try:
                well_df = _get_merged_dataframe_sync(project_id, wn)
            except Exception:
                well_df = None
            try:
                tvd_arr = _resolve_well_tvd_array(md_arr, well_df, surveys_df)
            except Exception:
                tvd_arr = np.full(len(md_arr), np.nan)

            for i, row in enumerate(ptdf.to_dict(orient="records")):
                override = row.get("tvdss")
                if override is not None and np.isfinite(float(override)):
                    tvdss, src = float(override), "override"
                elif i < len(tvd_arr) and np.isfinite(tvd_arr[i]):
                    tvdss, src = float(tvd_arr[i]), "computed"
                else:
                    tvdss, src = None, "missing"
                md = float(row["depth"])
                points.append(
                    {
                        "id": int(row["test_id"]),
                        "well_name": wn,
                        "md": md,
                        "tvdss": tvdss,
                        "tvdss_source": src,
                        "pressure": float(row["pressure"]),
                        "mobility": row.get("mobility"),
                        "permeability": row.get("permeability"),
                        "temperature": row.get("temperature"),
                        "fluid_type": row.get("fluid_type") or "UNASSIGNED",
                        "test_quality": row.get("test_quality") or "VALID",
                        "zone": (row.get("zone") or "").strip() or _zone_for_md(md, tops),
                        "zone_source": "stored" if (row.get("zone") or "").strip() else "tops",
                        "remarks": row.get("remarks") or "",
                    }
                )
    return points


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@router.get("/points", summary="List formation pressure points across wells with resolved TVDSS")
def fpt_list_points(
    project_id: int,
    well_names: Optional[str] = Query(None, description="Comma-separated well names"),
) -> Dict[str, Any]:
    try:
        wn_list = [w.strip() for w in well_names.split(",")] if well_names else None
        pts = _collect_fpt_points(project_id, wn_list)
        return _sanitize_for_json(
            {
                "points": pts,
                "n_points": len(pts),
                "n_missing_tvdss": sum(1 for p in pts if p["tvdss"] is None),
                "wells": sorted({p["well_name"] for p in pts}),
                "zones": sorted({p["zone"] for p in pts if p["zone"]}),
                "units": {"pressure": "psi", "depth": "m"},
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list FPT points: {e}") from e


def _analyze_subset(
    pts_sub: List[Dict[str, Any]],
    req: FPTAnalysisRequest,
    overrides: Optional[Dict[str, ManualGradientOverride]] = None,
) -> Dict[str, Any]:
    """Fit gradients, contacts and excess pressure for a subset of points (e.g. one zone)."""
    pf, df_ = PSI_TO[req.pressure_unit], M_TO[req.depth_unit]
    df = pd.DataFrame(
        {
            "ID": [p["id"] for p in pts_sub],
            "WELL_NAME": [p["well_name"] for p in pts_sub],
            "TVDSS": [p["tvdss"] * df_ if p["tvdss"] is not None else np.nan for p in pts_sub],
            "PRESSURE": [p["pressure"] * pf for p in pts_sub],
            "FLUID_TYPE": [p["fluid_type"] for p in pts_sub],
            "TEST_QUALITY": [p["test_quality"] for p in pts_sub],
            "MOBILITY": [p["mobility"] for p in pts_sub],
        }
    )
    fpa = FormationPressureAnalyzer(
        df,
        pressure_unit=req.pressure_unit,
        depth_unit=req.depth_unit,
        id_col="ID",
    )

    if req.auto_classify is not None:
        ac = req.auto_classify
        fpa.auto_classify_points(
            min_mobility=ac.min_mobility,
            ransac_residual_threshold=ac.ransac_residual_threshold,
            assign_unassigned=ac.assign_unassigned,
        )

    ov_dict: Dict[str, Dict[str, Any]] = {}
    active_ov = overrides if overrides is not None else req.manual_overrides
    for fl, ov in (active_ov or {}).items():
        if ov is None:
            continue
        grad = ov.gradient
        if grad is None and ov.density_gcc is not None:
            grad = density_to_gradient(ov.density_gcc, req.pressure_unit, req.depth_unit)
        if grad is None:
            continue
        ov_dict[fl.upper()] = {
            "gradient": grad,
            "anchor_tvdss": ov.anchor_tvdss,
            "anchor_pressure": ov.anchor_pressure,
            "std_err_gradient": ov.std_err_gradient or 0.0,
        }
    fpa.fit_all(method=req.regression_method, overrides=ov_dict)
    contacts = fpa.estimate_contacts(n_simulations=req.n_simulations)
    excess = fpa.calculate_excess_pressure(
        req.ref_water_gradient, req.datum_tvdss, req.datum_pressure
    )
    m_ref, z0, p0 = fpa.reference_line(
        req.ref_water_gradient, req.datum_tvdss, req.datum_pressure
    )

    # Trend lines spanning data and contacts (+ 8% padding)
    z_valid = fpa.df["TVDSS"].dropna()
    z_list = list(z_valid.to_numpy()) + [c.tvdss_p50 for c in contacts.values()]
    z_list += [g.tvdss_range[0] for g in fpa.gradients.values()]
    if z_list:
        z_lo, z_hi = float(np.nanmin(z_list)), float(np.nanmax(z_list))
        pad = max((z_hi - z_lo) * 0.08, 1.0)
        plot_lines = fpa.trend_lines(z_lo - pad, z_hi + pad)
    else:
        plot_lines = {}

    gradients = {fl: g.to_dict() for fl, g in fpa.gradients.items()}
    contacts_out = {}
    for k, c in contacts.items():
        cd = c.to_dict()
        cd["tvdss_p50_m"] = c.tvdss_p50 / df_
        cd["tvdss_p10_m"] = c.tvdss_p10 / df_
        cd["tvdss_p90_m"] = c.tvdss_p90 / df_
        contacts_out[k] = cd

    warnings = list(fpa.warnings)
    for g in fpa.gradients.values():
        warnings.extend(g.warnings)
    for c in contacts.values():
        warnings.extend(f"{c.contact_type}: {w}" for w in c.warnings)

    return {
        "fpa": fpa,
        "gradients": gradients,
        "contacts": contacts_out,
        "plot_lines": plot_lines,
        "reference": {
            "gradient": m_ref,
            "density_gcc": gradient_to_density(m_ref, req.pressure_unit, req.depth_unit),
            "datum_tvdss": z0,
            "datum_pressure": p0,
        },
        "warnings": warnings,
        "excess": excess,
        "n_points": len(pts_sub),
    }


@router.post("/analyze", summary="Fit fluid gradients and estimate contacts with P10/P50/P90 uncertainty")
def fpt_analyze(project_id: int, req: FPTAnalysisRequest) -> Dict[str, Any]:
    try:
        pts = _collect_fpt_points(project_id, req.well_names)
        if req.zones:
            allowed_z = {z.strip() for z in req.zones if z and z.strip()}
            pts = [p for p in pts if p["zone"] in allowed_z]
        if not pts:
            raise ValueError("No formation pressure points found for the selected wells / zones.")

        overrides = req.point_overrides or {}
        for p in pts:
            p["fluid_type_stored"] = p["fluid_type"]
            p["test_quality_stored"] = p["test_quality"]
            ov = overrides.get(str(p["id"]))
            if ov is not None:
                if ov.fluid_type is not None:
                    p["fluid_type"] = ov.fluid_type
                if ov.test_quality is not None:
                    p["test_quality"] = ov.test_quality

        by_zone_res: Dict[str, Dict[str, Any]] = {}
        if req.group_by_zone:
            # Group points by zone (or 'Unzoned' if empty)
            zone_pts_map: Dict[str, List[Dict[str, Any]]] = {}
            for p in pts:
                z_name = p["zone"].strip() if (p.get("zone") and str(p["zone"]).strip()) else "Unzoned"
                zone_pts_map.setdefault(z_name, []).append(p)

            for z_name, z_pts in zone_pts_map.items():
                z_ov = (req.overrides_by_zone or {}).get(z_name) or req.manual_overrides
                sub_res = _analyze_subset(z_pts, req, overrides=z_ov)
                by_zone_res[z_name] = {
                    "gradients": sub_res["gradients"],
                    "contacts": sub_res["contacts"],
                    "plot_lines": sub_res["plot_lines"],
                    "reference": sub_res["reference"],
                    "warnings": sub_res["warnings"],
                    "n_points": sub_res["n_points"],
                }
                # Enrich points in this zone
                fpa_df = sub_res["fpa"].df
                fpa_grads = sub_res["fpa"].gradients
                excess = sub_res["excess"]
                for i, p in enumerate(z_pts):
                    row = fpa_df.iloc[i]
                    p["tvdss_display"] = None if pd.isna(row["TVDSS"]) else float(row["TVDSS"])
                    p["pressure_display"] = float(row["PRESSURE"])
                    p["fluid_type"] = row["FLUID_TYPE"]
                    p["test_quality"] = row["TEST_QUALITY"]
                    p["excess_pressure"] = None if pd.isna(excess.iloc[i]) else float(excess.iloc[i])
                    p["used_in_fit"] = any(p["id"] in g.point_ids for g in fpa_grads.values())

            available_zones = sorted(by_zone_res.keys())
            active = req.active_zone if (req.active_zone and req.active_zone in by_zone_res) else ("ALL" if len(available_zones) > 1 else available_zones[0])

            if active in by_zone_res:
                top_gradients = by_zone_res[active]["gradients"]
                top_contacts = by_zone_res[active]["contacts"]
                top_plot_lines = by_zone_res[active]["plot_lines"]
                top_reference = by_zone_res[active]["reference"]
                top_warnings = list(by_zone_res[active]["warnings"])
            else:
                # Active is "ALL" across multiple zones
                top_warnings = []
                for z_name, zr in by_zone_res.items():
                    for w in zr["warnings"]:
                        top_warnings.append(f"[{z_name}] {w}")
                # For top-level fallback, use the zone with the most points
                largest_z = max(by_zone_res.keys(), key=lambda k: by_zone_res[k]["n_points"]) if by_zone_res else None
                top_gradients = by_zone_res[largest_z]["gradients"] if largest_z else {}
                top_contacts = by_zone_res[largest_z]["contacts"] if largest_z else {}
                top_plot_lines = by_zone_res[largest_z]["plot_lines"] if largest_z else {}
                top_reference = by_zone_res[largest_z]["reference"] if largest_z else {
                    "gradient": 0.0, "density_gcc": 1.0, "datum_tvdss": 0.0, "datum_pressure": 0.0
                }
        else:
            # Fit globally across all points
            sub_res = _analyze_subset(pts, req, overrides=req.manual_overrides)
            fpa_df = sub_res["fpa"].df
            fpa_grads = sub_res["fpa"].gradients
            excess = sub_res["excess"]
            for i, p in enumerate(pts):
                row = fpa_df.iloc[i]
                p["tvdss_display"] = None if pd.isna(row["TVDSS"]) else float(row["TVDSS"])
                p["pressure_display"] = float(row["PRESSURE"])
                p["fluid_type"] = row["FLUID_TYPE"]
                p["test_quality"] = row["TEST_QUALITY"]
                p["excess_pressure"] = None if pd.isna(excess.iloc[i]) else float(excess.iloc[i])
                p["used_in_fit"] = any(p["id"] in g.point_ids for g in fpa_grads.values())

            top_gradients = sub_res["gradients"]
            top_contacts = sub_res["contacts"]
            top_plot_lines = sub_res["plot_lines"]
            top_reference = sub_res["reference"]
            top_warnings = list(sub_res["warnings"])
            available_zones = sorted({p["zone"] for p in pts if p.get("zone")})
            active = "ALL"

        n_missing = sum(1 for p in pts if p.get("tvdss") is None)
        if n_missing:
            top_warnings.append(
                f"{n_missing} point(s) have no TVDSS (no TVD/TVDSS curve, survey or override) and were ignored"
            )

        return _sanitize_for_json(
            {
                "points": pts,
                "gradients": top_gradients,
                "contacts": top_contacts,
                "plot_lines": top_plot_lines,
                "reference": top_reference,
                "units": {
                    "pressure": req.pressure_unit,
                    "depth": req.depth_unit,
                    "gradient": f"{req.pressure_unit}/{req.depth_unit}",
                },
                "phases": list(FLUID_PHASES),
                "regression_method": req.regression_method,
                "warnings": top_warnings,
                "n_points": len(pts),
                "n_missing_tvdss": n_missing,
                "by_zone": by_zone_res,
                "active_zone": active,
                "available_zones": available_zones,
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("FPT analysis failed")
        raise HTTPException(status_code=500, detail=f"FPT analysis failed: {e}") from e


@router.patch("/points", summary="Persist fluid phase / test quality / zone assignments for pressure points")
def fpt_patch_points(project_id: int, req: FPTPointsPatchRequest) -> Dict[str, Any]:
    if not req.updates:
        return {"updated": 0}
    try:
        by_well: Dict[str, List[Dict[str, Any]]] = {}
        for u in req.updates:
            fields = u.fields or [
                f for f in ("fluid_type", "test_quality", "zone", "tvdss") if getattr(u, f) is not None
            ]
            payload: Dict[str, Any] = {"test_id": u.test_id}
            for f in fields:
                if f in ("fluid_type", "test_quality", "zone", "tvdss"):
                    payload[f] = getattr(u, f)
            by_well.setdefault(u.well_name, []).append(payload)

        updated = 0
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            for wn, ups in by_well.items():
                well = proj.get_well(resolve_db_well_name(project_id, wn))
                updated += well.update_pressure_test_interpretations(ups)
        for wn in by_well:
            invalidate_well_cache(project_id=project_id, well_name=wn)
        return {"updated": updated, "wells": sorted(by_well.keys())}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update FPT points: {e}") from e


@router.post(
    "/apply-contacts",
    summary="Write estimated contacts (TVDSS) into each well's fluid contacts (converted to MD)",
)
def fpt_apply_contacts(project_id: int, req: FPTApplyContactsRequest) -> Dict[str, Any]:
    contacts = {
        (req.owc_name if k.upper() == "OWC" else k.upper()): float(v)
        for k, v in (req.contacts or {}).items()
        if v is not None and np.isfinite(float(v))
    }
    if not contacts:
        raise HTTPException(status_code=400, detail="No valid contact depths supplied.")
    allowed = {str(w).strip() for w in (req.well_names or []) if str(w).strip()}
    results: List[Dict[str, Any]] = []
    try:
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            names = [w for w in proj.get_well_names() if not allowed or w in allowed]
            for wn in names:
                well = proj.get_well(resolve_db_well_name(project_id, wn))
                try:
                    surveys_df = well.get_well_surveys()
                except Exception:
                    surveys_df = None
                wdf = _get_merged_dataframe_sync(project_id, wn)
                if wdf is None or wdf.empty or "DEPTH" not in wdf.columns:
                    results.append({"well_name": wn, "written": {}, "skipped": "no depth data"})
                    continue
                md = np.sort(pd.to_numeric(wdf["DEPTH"], errors="coerce").dropna().unique())
                tvd = _resolve_well_tvd_array(md, wdf, surveys_df)
                ok = np.isfinite(tvd)
                if ok.sum() < 2:
                    results.append({"well_name": wn, "written": {}, "skipped": "no TVD/TVDSS"})
                    continue
                md_ok, tvd_ok = md[ok], tvd[ok]
                # Keep monotonic TVDSS branch for inversion (ignore horizontal / up-dip sections)
                mono = np.concatenate([[True], np.diff(np.maximum.accumulate(tvd_ok)) > 1e-6])
                md_ok, tvd_ok = md_ok[mono], tvd_ok[mono]
                written: Dict[str, float] = {}
                outside: List[str] = []
                payload = []
                for name, z in contacts.items():
                    if tvd_ok[0] <= z <= tvd_ok[-1]:
                        md_c = float(np.interp(z, tvd_ok, md_ok))
                        payload.append({"name": name, "depth": round(md_c, 3)})
                        written[name] = round(md_c, 3)
                    else:
                        outside.append(name)
                if payload:
                    well.add_fluid_contacts(payload, replace=False)
                results.append(
                    {
                        "well_name": wn,
                        "written": written,
                        "outside_range": outside,
                        "tvdss_range": [float(tvd_ok[0]), float(tvd_ok[-1])],
                    }
                )
        for r in results:
            if r.get("written"):
                invalidate_well_cache(project_id=project_id, well_name=r["well_name"])
        return _sanitize_for_json(
            {
                "contacts_tvdss_m": contacts,
                "wells": results,
                "n_wells_written": sum(1 for r in results if r.get("written")),
            }
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to apply contacts: {e}") from e
