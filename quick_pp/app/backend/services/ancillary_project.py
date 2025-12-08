import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import List, Dict, Any, Optional

from . import database
from quick_pp.database import objects as db_objects
from sqlalchemy import select
from fastapi import Body, UploadFile, File
import csv
import io
from quick_pp.core_analysis import (
    fit_poroperm_curve,
    auto_j_params,
    sw_shf_leverett_j,
    leverett_j,
    normalize_sw,
)
from quick_pp.rock_type import calc_fzi, rock_typing
import numpy as np
import math

# Project-level router: supports project-based endpoints and accepts optional
# `well_name` as a query parameter or in POST bodies. When a `well_name` is
# provided the handlers delegate to the existing well-based behavior; when
# absent GET endpoints aggregate across all wells in the project.
project_router = APIRouter(
    prefix="/database/projects/{project_id}", tags=["Ancillary - Project"]
)


@project_router.get(
    "/formation_tops", summary="List formation tops for a project (optional well_name)"
)
def list_formation_tops_project(
    project_id: int, well_name: Optional[str] = Query(None)
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with database.connector.get_session() as session:
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list formation tops: {e}"
        )


@project_router.post(
    "/formation_tops",
    summary="Add or update formation tops for a project (requires well_name)",
)
def add_formation_tops_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    if not isinstance(payload, dict) or "tops" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'tops'."
        )

    tops = payload.get("tops")
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_formation_tops(tops)
            return {
                "created": [
                    {"name": t.get("name"), "depth": t.get("depth")} for t in tops
                ]
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add formation tops: {e}"
        )


@project_router.delete(
    "/formation_tops/{top_name}",
    summary="Delete a formation top by name (requires well_name)",
)
def delete_formation_top_project(
    project_id: int, top_name: str, well_name: Optional[str] = Query(None)
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to delete a top",
        )
    try:
        with database.connector.get_session() as session:
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
            return {"deleted": top_name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to delete formation top: {e}"
        )


@project_router.post(
    "/formation_tops/preview",
    summary="Upload CSV and return parsed preview for formation tops (optional well_name)",
)
def formation_tops_preview_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    file: UploadFile = File(...),
):
    # this preview is independent of well/project, but we keep the same signature
    try:
        content = file.file.read().decode(errors="ignore")
        reader = csv.reader(io.StringIO(content))
        rows = [r for r in reader]
        if not rows:
            return {"preview": [], "headers": []}
        headers = [h.strip() for h in rows[0]]
        preview = []
        for r in rows[1:51]:
            mapped = {
                headers[i] if i < len(headers) else f"col_{i}": (
                    r[i] if i < len(r) else ""
                )
                for i in range(len(headers))
            }
            preview.append(mapped)
        detected = {
            "name": next(
                (
                    h
                    for h in headers
                    if h.lower() in ("name", "top", "top_name", "formation")
                ),
                None,
            ),
            "depth": next(
                (
                    h
                    for h in headers
                    if h.lower() in ("depth", "md", "tvd", "depth_m", "depth_ft")
                ),
                None,
            ),
        }
        return {"preview": preview, "headers": headers, "detected": detected}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse CSV preview: {e}")


@project_router.get(
    "/fluid_contacts", summary="List fluid contacts for a project (optional well_name)"
)
def list_fluid_contacts_project(
    project_id: int, well_name: Optional[str] = Query(None)
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with database.connector.get_session() as session:
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list fluid contacts: {e}"
        )


@project_router.post(
    "/fluid_contacts",
    summary="Add or update fluid contacts for a project (requires well_name)",
)
def add_fluid_contacts_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    if not isinstance(payload, dict) or "contacts" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'contacts'."
        )
    contacts = payload.get("contacts")
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_fluid_contacts(contacts)
            return {"created": contacts}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add fluid contacts: {e}"
        )


@project_router.get(
    "/pressure_tests", summary="List pressure tests for a project (optional well_name)"
)
def list_pressure_tests_project(
    project_id: int, well_name: Optional[str] = Query(None)
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with database.connector.get_session() as session:
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list pressure tests: {e}"
        )


@project_router.post(
    "/pressure_tests",
    summary="Add or update pressure tests for a project (requires well_name)",
)
def add_pressure_tests_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    if not isinstance(payload, dict) or "tests" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'tests'."
        )
    tests = payload.get("tests")
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_pressure_tests(tests)
            return {"created": tests}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add pressure tests: {e}"
        )


@project_router.get(
    "/core_samples", summary="List core samples for a project (optional well_name)"
)
def list_core_samples_project(project_id: int, well_name: Optional[str] = Query(None)):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with database.connector.get_session() as session:
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list core samples: {e}")


@project_router.post(
    "/core_samples",
    summary="Add or update a core sample with measurements (requires well_name)",
)
def add_core_sample_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, Any] = Body(...),
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    required = ["sample_name", "depth", "measurements"]
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_core_sample_with_measurements(
                sample_name=payload["sample_name"],
                depth=payload["depth"],
                measurements=payload.get("measurements", []),
                description=payload.get("description"),
                remark=payload.get("remark"),
                relperm_data=payload.get("relperm_data"),
                pc_data=payload.get("pc_data"),
            )
            return {
                "sample_name": payload["sample_name"],
                "status": "created_or_updated",
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add core sample: {e}")


@project_router.get(
    "/core_samples/{sample_name}",
    summary="Get core sample details by name (requires well_name)",
)
def get_core_sample_project(
    project_id: int, sample_name: str, well_name: Optional[str] = Query(None)
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    target_well = well_name
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name query parameter is required to get a core sample",
        )
    try:
        with database.connector.get_session() as session:
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
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get core sample: {e}")


@project_router.get(
    "/rca",
    summary="List RCA (core point measurements) for a project (optional well_name)",
)
def list_rca_project(project_id: int, well_name: Optional[str] = Query(None)):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            rows = []
            if well_name:
                well = proj.get_well(well_name)
                core_data = well.get_core_data()
                for sample_name, sd in core_data.items():
                    measurements = sd.get("measurements")
                    if hasattr(measurements, "to_dict"):
                        for r in measurements.to_dict(orient="records"):
                            r.update(
                                {"sample_name": sample_name, "depth": sd.get("depth")}
                            )
                            rows.append(r)
                return {"rca": rows}

            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    core_data = well.get_core_data()
                    for sample_name, sd in core_data.items():
                        measurements = sd.get("measurements")
                        if hasattr(measurements, "to_dict"):
                            for r in measurements.to_dict(orient="records"):
                                r.update(
                                    {
                                        "sample_name": sample_name,
                                        "depth": sd.get("depth"),
                                        "well_name": wn,
                                    }
                                )
                                rows.append(r)
                except Exception:
                    continue
            return {"rca": rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list RCA: {e}")


@project_router.post(
    "/rca",
    summary="Add RCA (core point measurements) for a project (requires well_name)",
)
def add_rca_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, Any] = Body(...),
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    required = ["sample_name", "depth", "measurements"]
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_core_sample_with_measurements(
                sample_name=payload["sample_name"],
                depth=payload["depth"],
                measurements=payload.get("measurements", []),
                description=payload.get("description"),
                remark=payload.get("remark"),
            )
            return {
                "sample_name": payload["sample_name"],
                "status": "measurements_added",
            }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add RCA: {e}")


@project_router.get(
    "/scal",
    summary="List SCAL (relperm & capillary) for a project (optional well_name)",
)
def list_scal_project(project_id: int, well_name: Optional[str] = Query(None)):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            relperm_rows = []
            pc_rows = []
            if well_name:
                well = proj.get_well(well_name)
                core_data = well.get_core_data()
                for sample_name, sd in core_data.items():
                    relperm = sd.get("relperm")
                    if hasattr(relperm, "to_dict"):
                        for r in relperm.to_dict(orient="records"):
                            r.update(
                                {"sample_name": sample_name, "depth": sd.get("depth")}
                            )
                            relperm_rows.append(r)
                    pc = sd.get("pc")
                    if hasattr(pc, "to_dict"):
                        for p in pc.to_dict(orient="records"):
                            p.update(
                                {"sample_name": sample_name, "depth": sd.get("depth")}
                            )
                            pc_rows.append(p)
                return {"relperm": relperm_rows, "pc": pc_rows}

            for wn in proj.get_well_names():
                try:
                    well = proj.get_well(wn)
                    core_data = well.get_core_data()
                    for sample_name, sd in core_data.items():
                        relperm = sd.get("relperm")
                        if hasattr(relperm, "to_dict"):
                            for r in relperm.to_dict(orient="records"):
                                r.update(
                                    {
                                        "sample_name": sample_name,
                                        "depth": sd.get("depth"),
                                        "well_name": wn,
                                    }
                                )
                                relperm_rows.append(r)
                        pc = sd.get("pc")
                        if hasattr(pc, "to_dict"):
                            for p in pc.to_dict(orient="records"):
                                p.update(
                                    {
                                        "sample_name": sample_name,
                                        "depth": sd.get("depth"),
                                        "well_name": wn,
                                    }
                                )
                                pc_rows.append(p)
                except Exception:
                    continue
            return {"relperm": relperm_rows, "pc": pc_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list SCAL: {e}")


@project_router.post(
    "/scal",
    summary="Add SCAL data for a sample (relperm and/or capillary) (requires well_name)",
)
def add_scal_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, Any] = Body(...),
):
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    required = ["sample_name", "depth"]
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")
    target_well = well_name or payload.get("well_name")
    if not target_well:
        raise HTTPException(
            status_code=400,
            detail="well_name must be provided as query parameter or in payload",
        )
    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_core_sample_with_measurements(
                sample_name=payload["sample_name"],
                depth=payload["depth"],
                measurements=payload.get("measurements", []),
                relperm_data=payload.get("relperm_data"),
                pc_data=payload.get("pc_data"),
                description=payload.get("description"),
            )
            return {"sample_name": payload["sample_name"], "status": "scal_added"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add SCAL: {e}")


@project_router.get("/fzi_data", summary="Get FZI data for plotting")
async def get_fzi_data(project_id: int):
    """Return PHIT and PERM data for FZI plotting from all wells in the project.

    Args:
        project_id: Project ID

    Returns:
        JSON with phit and perm arrays
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            all_well_names = proj.get_well_names()
            if not all_well_names:
                raise HTTPException(
                    status_code=404, detail=f"No wells found for project {project_id}"
                )
            well_names = []
            depths_list = []
            cpore_list = []
            cperm_list = []
            zones_list = []
            rock_flags_list = []
            for well_name in all_well_names:
                df = proj.get_well_data_optimized(well_name)

                if df.empty:
                    raise HTTPException(
                        status_code=404, detail=f"No data for project {project_id}"
                    )

                # Load ancillary data (formation tops and core data)
                try:
                    ancillary = proj.get_well_ancillary_data(well_name)
                except Exception:
                    ancillary = {}

                # If formation tops exist, mark their names in a 'ZONES' column
                tops_df = None
                if isinstance(ancillary, dict) and "formation_tops" in ancillary:
                    tops_df = ancillary.get("formation_tops")
                    if (
                        isinstance(tops_df, pd.DataFrame)
                        and not tops_df.empty
                        and not df.empty
                    ):
                        # Ensure ZONES column exists
                        df["ZONES"] = pd.NA
                        # For each top, find the closest depth row in df and annotate
                        for _, top in tops_df.iterrows():
                            try:
                                top_depth = float(top.get("depth"))
                                top_name = str(top.get("name"))
                            except Exception:
                                continue
                            # find index of nearest depth
                            nearest_idx = (df["DEPTH"] - top_depth).abs().idxmin()
                            df.at[nearest_idx, "ZONES"] = top_name
                        df["ZONES"] = df["ZONES"].ffill()

                # Extract PHIT and PERM columns
                phit_col = None
                perm_col = None
                rock_flag_col = None
                for col in df.columns:
                    if col.upper() in ["PHIT", "POROSITY", "POR"]:
                        phit_col = col
                    if col.upper() in ["PERM", "PERMEABILITY", "K"]:
                        perm_col = col
                    if col.upper() == "ROCK_FLAG":
                        rock_flag_col = col

                if not phit_col or not perm_col:
                    continue  # skip this well if columns not found

                cpore = df[phit_col].dropna()
                cperm = df[perm_col].dropna()

                # Align the data
                common_index = cpore.index.intersection(cperm.index)
                cpore = cpore.loc[common_index]
                cperm = cperm.loc[common_index]
                depths = df.loc[common_index, "DEPTH"]

                # Extract zones if available
                zones = []
                if "ZONES" in df.columns:
                    zones = df.loc[common_index, "ZONES"].fillna("Unknown").tolist()
                else:
                    zones = ["Unknown"] * len(cpore)

                # Extract rock_flag if available
                rock_flags = []
                if rock_flag_col:
                    rock_flags = df.loc[common_index, rock_flag_col]
                    rock_flags = rock_flags.where(pd.notna(rock_flags), None).tolist()
                else:
                    rock_flags = [None] * len(cpore)

                if len(cpore) == 0:
                    continue

                well_names.extend([well_name] * len(cpore))
                depths_list.extend(depths.tolist())
                cpore_list.append(cpore)
                cperm_list.append(cperm)
                zones_list.extend(zones)
                rock_flags_list.extend(rock_flags)

            if not cpore_list or not cperm_list:
                raise HTTPException(
                    status_code=404,
                    detail=f"PHIT or PERM columns not found in project data. Available: {list(df.columns)}",
                )

            # Concatenate all data
            phit_all = pd.concat(cpore_list).tolist() if cpore_list else []
            perm_all = pd.concat(cperm_list).tolist() if cperm_list else []

            return {
                "phit": phit_all,
                "perm": perm_all,
                "zones": zones_list,
                "well_names": well_names,
                "depths": depths_list,
                "rock_flags": rock_flags_list,
            }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get FZI data: {e}")


@project_router.get("/poroperm_fits", summary="Get poro-perm fits per rock flag")
async def get_poroperm_fits(project_id: int):
    """Return fitted a and b parameters for poro-perm curves per ROCK_FLAG.

    Args:
        project_id: Project ID

    Returns:
        JSON with fits per rock_flag
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            all_well_names = proj.get_well_names()
            if not all_well_names:
                raise HTTPException(
                    status_code=404, detail=f"No wells found for project {project_id}"
                )

            # Collect all data with rock_flags
            all_phit = []
            all_perm = []
            all_rock_flags = []

            for well_name in all_well_names:
                df = proj.get_well_data_optimized(well_name)

                if df.empty:
                    continue

                # Extract columns
                phit_col = None
                perm_col = None
                rock_flag_col = None
                for col in df.columns:
                    if col.upper() in ["PHIT", "POROSITY", "POR"]:
                        phit_col = col
                    if col.upper() in ["PERM", "PERMEABILITY", "K"]:
                        perm_col = col
                    if col.upper() == "ROCK_FLAG":
                        rock_flag_col = col

                if not phit_col or not perm_col or not rock_flag_col:
                    continue

                cpore = df[phit_col].dropna()
                cperm = df[perm_col].dropna()
                crock = df[rock_flag_col].dropna()

                # Align
                common_index = cpore.index.intersection(cperm.index).intersection(
                    crock.index
                )
                cpore = cpore.loc[common_index]
                cperm = cperm.loc[common_index]
                crock = crock.loc[common_index]

                if len(cpore) == 0:
                    continue

                all_phit.extend(cpore.tolist())
                all_perm.extend(cperm.tolist())
                all_rock_flags.extend(crock.tolist())

            if not all_phit:
                raise HTTPException(
                    status_code=404, detail="No data with ROCK_FLAG found"
                )

            # Group by rock_flag
            from collections import defaultdict

            groups = defaultdict(list)
            for phit, perm, rf in zip(all_phit, all_perm, all_rock_flags):
                groups[rf].append((phit, perm))

            # Fit for each group
            fits = {}
            for rf, data in groups.items():
                poro = [d[0] for d in data]
                perm = [d[1] for d in data]
                try:
                    a, b = fit_poroperm_curve(np.array(poro), np.array(perm))
                    fits[str(rf)] = {"a": a, "b": b}
                except Exception as e:
                    print(f"Failed to fit for ROCK_FLAG {rf}: {e}")
                    fits[str(rf)] = {"a": 1, "b": 1}

            return {"fits": fits}

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get poroperm fits: {e}")


@project_router.post("/save_rock_flags", summary="Save rock flags for project wells")
async def save_rock_flags(project_id: int, payload: dict):
    """Save ROCK_FLAG column to wells based on FZI rock typing.

    Expected payload: { "rock_flag_pairs": [{"well_name": "...", "depth": 123.4, "rock_flag": 1}, ...], "cutoffs": "0.1,1.0,3.0" }

    Each pair contains the well name, depth, and assigned rock flag.
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "rock_flag_pairs" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'rock_flag_pairs' array"
        )

    rock_flag_pairs = payload.get("rock_flag_pairs")
    if not isinstance(rock_flag_pairs, list):
        raise HTTPException(status_code=400, detail="'rock_flag_pairs' must be a list")

    cutoffs_str = payload.get("cutoffs", "")

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Group by well
            well_data = {}
            for pair in rock_flag_pairs:
                if not isinstance(pair, dict) or not all(
                    k in pair for k in ["well_name", "depth", "rock_flag"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Each rock_flag_pair must be a dict with 'well_name', 'depth', and 'rock_flag' keys",
                    )

                well_name = pair["well_name"]
                depth = pair["depth"]
                rock_flag = pair["rock_flag"]

                if well_name not in well_data:
                    well_data[well_name] = []
                well_data[well_name].append({"DEPTH": depth, "ROCK_FLAG": rock_flag})

            # Save to each well
            for well_name, data in well_data.items():
                df = pd.DataFrame(data)
                well_obj = proj.get_well(well_name)
                well_obj.update_data(df)
                well_obj.save()

            proj.save()

        return {
            "message": f"Rock flags saved for {len(well_data)} wells",
            "cutoffs": cutoffs_str,
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save rock flags: {e}")


@project_router.post(
    "/save_perm_trans", summary="Save perm transform for project wells"
)
async def save_perm_trans(project_id: int, payload: dict):
    """Save PERM_TRANS column to wells based on poro-perm fits per ROCK_FLAG.

    Expected payload: { "perm_trans_pairs": [{"well_name": "...", "depth": 123.4, "perm_trans": 1.23}, ...] }

    Each pair contains the well name, depth, and calculated perm_trans.
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "perm_trans_pairs" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'perm_trans_pairs' array"
        )

    perm_trans_pairs = payload.get("perm_trans_pairs")
    if not isinstance(perm_trans_pairs, list):
        raise HTTPException(status_code=400, detail="'perm_trans_pairs' must be a list")

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Group by well
            well_data = {}
            for pair in perm_trans_pairs:
                if not isinstance(pair, dict) or not all(
                    k in pair for k in ["well_name", "depth", "perm_trans"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Each perm_trans_pair must be a dict with 'well_name', 'depth', and 'perm_trans' keys",
                    )

                well_name = pair["well_name"]
                depth = pair["depth"]
                perm_trans = pair["perm_trans"]

                if well_name not in well_data:
                    well_data[well_name] = []
                well_data[well_name].append({"DEPTH": depth, "PERM_TRANS": perm_trans})

            # Save to each well
            for well_name, data in well_data.items():
                df = pd.DataFrame(data)
                well_obj = proj.get_well(well_name)
                well_obj.update_data(df)
                well_obj.save()

            proj.save()

        return {
            "message": f"Perm transforms saved for {len(well_data)} wells",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to save perm transforms: {e}"
        )


@project_router.get("/j_data", summary="Get J data for plotting")
async def get_j_data(project_id: int, cutoffs: str = Query("0.1,1.0,3.0")):
    """Return PC, SW, PERM, PHIT data for J plotting from all wells in the project.

    Args:
        project_id: Project ID
        cutoffs: Optional comma-separated cutoffs for FZI values to define rock flags, e.g., "0.1,1.0,3.0"

    Returns:
        JSON with pc, sw, perm, phit arrays
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            all_well_names = proj.get_well_names()
            if not all_well_names:
                raise HTTPException(
                    status_code=404, detail=f"No wells found for project {project_id}"
                )
            well_names = []
            depths_list = []
            pc_list = []
            sw_list = []
            perm_list = []
            phit_list = []
            zones_list = []
            rock_flags_list = []
            for well_name in all_well_names:
                try:
                    ancillary = proj.get_well_ancillary_data(well_name)
                except Exception:
                    continue

                if not isinstance(ancillary, dict) or "core_data" not in ancillary:
                    continue

                core_data = ancillary["core_data"]
                if not isinstance(core_data, dict):
                    continue

                for sample_name, sample in core_data.items():
                    depth = sample.get("depth")
                    if depth is None:
                        continue

                    measurements = sample.get("measurements")
                    pc_data = sample.get("pc")

                    if (
                        measurements is None
                        or pc_data is None
                        or measurements.empty
                        or pc_data.empty
                    ):
                        continue

                    # Extract perm and phit from measurements (long format)
                    perm_row = measurements[
                        measurements["property"].str.upper() == "CPERM"
                    ]
                    phit_row = measurements[
                        measurements["property"].str.upper() == "CPORE"
                    ]
                    if perm_row.empty or phit_row.empty:
                        continue
                    perm = perm_row["value"].iloc[0]
                    phit = phit_row["value"].iloc[0]

                    # Find columns in pc_data
                    pc_col = None
                    sw_col = None
                    for col in pc_data.columns:
                        if col.upper() in ["PRESSURE", "PC", "PC_RES"]:
                            pc_col = col
                        if col.upper() in ["SATURATION", "SW", "SWN"]:
                            sw_col = col

                    if not pc_col or not sw_col:
                        continue

                    pc_vals = pc_data[pc_col].dropna()
                    sw_vals = pc_data[sw_col].dropna()

                    common_index = pc_vals.index.intersection(sw_vals.index)
                    pc_vals = pc_vals.loc[common_index]
                    sw_vals = sw_vals.loc[common_index]

                    for i in range(len(pc_vals)):
                        pc_list.append(pc_vals.iloc[i])
                        sw_list.append(sw_vals.iloc[i])
                        perm_list.append(perm)
                        phit_list.append(phit)
                        well_names.append(well_name)
                        depths_list.append(depth)
                        zones_list.append("Unknown")
                        rock_flags_list.append(None)

            if cutoffs:
                cutoff_values = [float(x.strip()) for x in cutoffs.split(",")]
                for i in range(len(pc_list)):
                    # Calculate FZI: FZI = sqrt(perm / phit) / (phit / (1 - phit))
                    if phit_list[i] > 0 and phit_list[i] < 1 and perm_list[i] > 0:
                        fzi = calc_fzi(phit_list[i], perm_list[i])
                        rock_flags_list[i] = rock_typing(fzi, cutoff_values).tolist()
                    else:
                        rock_flags_list[i] = None

            if not pc_list:
                raise HTTPException(
                    status_code=404,
                    detail="No RCA/SCAL data found with PC, SW, PERM, PHIT",
                )

            return {
                "pc": pc_list,
                "sw": sw_list,
                "perm": perm_list,
                "phit": phit_list,
                "zones": zones_list,
                "well_names": well_names,
                "depths": depths_list,
                "rock_flags": rock_flags_list,
            }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get J data: {e}")


@project_router.post("/compute_j_fits", summary="Compute J fits per rock flag")
async def compute_j_fits(payload: dict):
    """Compute fitted a and b parameters for J curves per ROCK_FLAG.

    Expected payload: { "data": { "pc": [...], "sw": [...], "perm": [...], "phit": [...], "rock_flags": [...] }, "ift": 30, "theta": 30 }

    Returns:
        JSON with fits per rock_flag
    """
    if not isinstance(payload, dict) or "data" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'data' object"
        )

    data = payload.get("data")
    ift = payload.get("ift", 30)
    theta = payload.get("theta", 30)

    try:
        pc = data.get("pc", [])
        sw = data.get("sw", [])
        perm = data.get("perm", [])
        phit = data.get("phit", [])
        rock_flags = data.get("rock_flags", [])

        if (
            len(pc) != len(sw)
            or len(pc) != len(perm)
            or len(pc) != len(phit)
            or len(pc) != len(rock_flags)
        ):
            raise HTTPException(
                status_code=400, detail="Data arrays must have the same length"
            )

        # Calculate J
        j_values = []
        for i in range(len(pc)):
            j = leverett_j(pc[i], ift, theta, perm[i], phit[i])
            j_values.append(j)

        # Normalize SW to SWN
        swn_values = normalize_sw(pd.Series(sw))

        # Prepare core_data
        core_data_list = []
        for i in range(len(pc)):
            if rock_flags[i] is not None:
                core_data_list.append(
                    {
                        "ROCK_FLAG": rock_flags[i],
                        "SWN": swn_values[i],
                        "J": j_values[i],
                        "Well": f"Well_{i}",
                        "Sample": f"Sample_{i}",
                    }
                )

        core_data = pd.DataFrame(core_data_list)
        j_params = auto_j_params(core_data)

        # Convert to dict
        fits = {}
        for param in j_params:
            rf = str(param["ROCK_FLAG"])
            fits[rf] = {"a": param["a"], "b": param["b"], "rmse": param["rmse"]}

        return fits

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute J fits: {e}")


@project_router.post("/compute_shf", summary="Compute SHF for project data")
async def compute_shf(payload: dict):
    """Compute SHF values using Leverett J function.

    Expected payload: { "data": { "pc": [...], "sw": [...], "perm": [...], "phit": [...], "depths": [...], "rock_flags": [...], "well_names": [...] }, "fits": {...}, "fwl": 1000, "ift": 30, "theta": 30, "gw": 1.05, "ghc": 0.8 }

    Returns:
        JSON with shf_data
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "data" not in payload or "fits" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'data' and 'fits' objects"
        )

    data = payload.get("data")
    fits = payload.get("fits")
    fwl = payload.get("fwl", 1000)
    ift = payload.get("ift", 30)
    theta = payload.get("theta", 30)
    gw = payload.get("gw", 1.05)
    ghc = payload.get("ghc", 0.8)

    try:
        perm = data.get("perm", [])
        phit = data.get("phit", [])
        depths = data.get("depths", [])
        rock_flags = data.get("rock_flags", [])
        well_names = data.get("well_names", [])

        shf_data = []
        for i in range(len(phit)):
            rf = (
                str(int(rock_flags[i]))
                if rock_flags and rock_flags[i] is not None
                else None
            )
            if rf and rf in fits.keys():
                a = fits[rf]["a"]
                b = fits[rf]["b"]
                shf = sw_shf_leverett_j(
                    perm[i], phit[i], depths[i], fwl, ift, theta, gw, ghc, a, b
                )
                if math.isfinite(shf):
                    shf_data.append(
                        {"well": well_names[i], "depth": depths[i], "shf": shf}
                    )
                else:
                    shf_data.append(
                        {"well": well_names[i], "depth": depths[i], "shf": None}
                    )

        return {"shf_data": shf_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to compute SHF: {e}")


@project_router.post("/save_shf", summary="Save SHF for project wells")
async def save_shf(project_id: int, payload: dict):
    """Save SHF column to wells.

    Expected payload: { "shf_data": [{"well": "...", "depth": 123.4, "shf": 0.8}, ...] }

    Each item contains the well name, depth, and SHF value.
    """
    if database.connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "shf_data" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must include 'shf_data' array"
        )

    shf_data = payload.get("shf_data")
    if not isinstance(shf_data, list):
        raise HTTPException(status_code=400, detail="'shf_data' must be a list")

    try:
        with database.connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # Group by well
            well_data = {}
            for item in shf_data:
                if not isinstance(item, dict) or not all(
                    k in item for k in ["well", "depth", "shf"]
                ):
                    raise HTTPException(
                        status_code=400,
                        detail="Each shf_data item must be a dict with 'well', 'depth', and 'shf' keys",
                    )

                well_name = item["well"]
                depth = item["depth"]
                shf = item["shf"]

                if well_name not in well_data:
                    well_data[well_name] = []
                well_data[well_name].append({"DEPTH": depth, "SHF": shf})

            # Save to each well
            for well_name, data in well_data.items():
                df = pd.DataFrame(data)
                well_obj = proj.get_well(well_name)
                well_obj.update_data(df)
                well_obj.save()

            proj.save()

        return {
            "message": f"SHF saved for {len(well_data)} wells",
        }

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save SHF: {e}")
