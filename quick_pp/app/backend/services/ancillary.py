import csv
import io
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import APIRouter, HTTPException, UploadFile, Body, File, Query
from sqlalchemy import select

from quick_pp.database import objects as db_objects

from quick_pp.app.backend.utils.db import get_db


# Project-level router: supports project-based endpoints and accepts optional
# `well_name` as a query parameter or in POST bodies. When a `well_name` is
# provided the handlers delegate to the existing well-based behavior; when
# absent GET endpoints aggregate across all wells in the project.
router = APIRouter(prefix="/database/projects/{project_id}", tags=["Ancillary Data"])


@router.get(
    "/formation_tops",
    summary="List formation tops for a project (optional well_name)",
    tags=["Formation Tops"],
)
def list_formation_tops_project(
    project_id: int, well_name: Optional[str] = Query(None)
):
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
def add_formation_tops_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
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
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_formation_tops(tops)
            return {
                "created": [
                    {"name": t.get("name"), "depth": t.get("depth")} for t in tops
                ]
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
def delete_formation_top_project(
    project_id: int, top_name: str, well_name: Optional[str] = Query(None)
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
def formation_tops_preview_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    file: UploadFile = File(...),
):
    # this preview is independent of well/project, but we keep the same signature
    try:
        content = file.file.read().decode(errors="ignore")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
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
        raise HTTPException(
            status_code=500, detail=f"Failed to parse CSV preview: {e}"
        ) from e


@router.get(
    "/fluid_contacts",
    summary="List fluid contacts for a project (optional well_name)",
    tags=["Fluid Contacts"],
)
def list_fluid_contacts_project(
    project_id: int, well_name: Optional[str] = Query(None)
):
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
def add_fluid_contacts_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
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
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_fluid_contacts(contacts)
            return {"created": contacts}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add fluid contacts: {e}"
        ) from e


@router.get(
    "/pressure_tests",
    summary="List pressure tests for a project (optional well_name)",
    tags=["Pressure Tests"],
)
def list_pressure_tests_project(
    project_id: int, well_name: Optional[str] = Query(None)
):
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
def add_pressure_tests_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
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
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_pressure_tests(tests)
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
def list_core_samples_project(project_id: int, well_name: Optional[str] = Query(None)):
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
    summary="Add or update a core sample with measurements, RCA, and SCAL data (requires well_name)",
    tags=["Core Samples"],
)
def add_core_sample_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, Any] = Body(...),
):
    """
    Add or update a core sample with optional measurements (RCA), relperm, and capillary pressure (SCAL) data.
    This is the unified endpoint for all core sample data types.

    Payload should include:
    - sample_name: Unique identifier for the sample
    - depth: Depth of the sample
    - measurements: List of RCA measurements (optional)
    - relperm_data: Relative permeability data (optional)
    - pc_data: Capillary pressure data (optional)
    - description: Sample description (optional)
    - remark: Additional remarks (optional)
    """

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
        with get_db() as session:
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
        raise HTTPException(status_code=404, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add core sample: {e}"
        ) from e


@router.get(
    "/core_samples/{sample_name}",
    summary="Get core sample details by name (requires well_name)",
    tags=["Core Samples"],
)
def get_core_sample_project(
    project_id: int, sample_name: str, well_name: Optional[str] = Query(None)
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
def add_well_surveys_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if not isinstance(payload, dict) or "surveys" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'surveys'."
        )
    surveys = payload.get("surveys")
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
            well.add_well_surveys(surveys)
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
def well_surveys_preview_project(
    project_id: int,
    well_name: Optional[str] = Query(None),
    file: UploadFile = File(...),
):
    """Return a preview (first 50 rows) and detected columns from uploaded CSV/Excel file."""
    try:
        content = file.file.read().decode(errors="ignore")
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
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
            "md": next(
                (
                    h
                    for h in headers
                    if h.lower() in ("md", "measured depth", "md_m", "depth_m")
                ),
                None,
            ),
            "inc": next(
                (
                    h
                    for h in headers
                    if h.lower() in ("inc", "inclination", "incl", "inc_deg")
                ),
                None,
            ),
            "azim": next(
                (
                    h
                    for h in headers
                    if h.lower() in ("azim", "azimuth", "azi", "azim_deg")
                ),
                None,
            ),
        }
        return {
            "preview": preview,
            "headers": headers,
            "detected": detected,
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
    """
    Upload deviation survey data and optionally calculate TVD.

    Payload example:
    {
      "file_content": "base64_encoded_csv_or_excel",
      "md_column": "MD",
      "inc_column": "INC",
      "azim_column": "AZIM",
      "tvd_column": "TVD",  # optional, if present will store as TVD curve
      "calculate_tvd": true  # optional, if true will calculate TVD from survey
    }
    """

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
        import base64

        try:
            import wellpathpy as wpp
        except ImportError as e:
            raise ImportError(
                "wellpathpy required for TVD calculation. Install: pip install wellpathpy"
            ) from e

        # Decode and read CSV content
        file_content = payload["file_content"]
        if isinstance(file_content, str) and file_content.startswith("data:"):
            # Remove data URL prefix
            file_content = file_content.split(",", 1)[1]

        decoded = base64.b64decode(file_content)
        content_str = decoded.decode(errors="ignore")

        # Parse CSV
        reader = csv.reader(io.StringIO(content_str))
        rows = list(reader)
        if len(rows) < 2:
            raise ValueError("No data rows in file")

        headers = [h.strip() for h in rows[0]]
        md_col = payload["md_column"]
        inc_col = payload["inc_column"]
        azim_col = payload["azim_column"]
        tvd_col = payload.get("tvd_column")
        calculate_tvd = payload.get("calculate_tvd", False)

        if md_col not in headers:
            raise ValueError(f"Column '{md_col}' not found in file")
        if inc_col not in headers:
            raise ValueError(f"Column '{inc_col}' not found in file")
        if azim_col not in headers:
            raise ValueError(f"Column '{azim_col}' not found in file")

        # Extract column indices
        md_idx = headers.index(md_col)
        inc_idx = headers.index(inc_col)
        azim_idx = headers.index(azim_col)
        tvd_idx = headers.index(tvd_col) if tvd_col and tvd_col in headers else None

        # Parse survey points
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

                # Store TVD if column present
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

        # Save to database
        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(target_well)
            well.add_well_surveys(surveys)

            # Add TVD curve data if available or to be calculated
            if calculate_tvd:
                # Calculate TVD using wellpathpy minimum curvature method
                surveys_sorted = sorted(surveys, key=lambda x: x["md"])
                mds = np.array([s["md"] for s in surveys_sorted])
                incs = np.array([s["inc"] for s in surveys_sorted])
                azims = np.array([s["azim"] for s in surveys_sorted])

                # Use wellpathpy to calculate TVD
                dev_survey = wpp.deviation(mds, incs, azims)
                tvds = dev_survey.minimum_curvature().depth

                tvd_data = dict(zip(mds, tvds, strict=True))

            # Store TVD as curve data
            if tvd_data:
                well.add_curve_data("TVD", tvd_data, unit="m")

            return {
                "status": "success",
                "surveys_added": len(surveys),
                "tvd_calculated": calculate_tvd,
                "tvd_points_stored": len(tvd_data),
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
    """
    Calculate TVD from existing well survey data using minimum curvature method.
    If well_name is provided, calculates for that well only.
    If well_name is not provided, calculates for all wells in the project.
    Stores the calculated TVD as a well curve (mnemonic 'TVD').
    """

    try:
        try:
            import wellpathpy as wpp
        except ImportError as e:
            raise ImportError(
                "wellpathpy required for TVD calculation. Install: pip install wellpathpy"
            ) from e

        with get_db() as session:
            proj = db_objects.Project.load(session, project_id=project_id)

            # If well_name provided, calculate for that well only
            if well_name:
                well = proj.get_well(well_name)
                surveys_df = well.get_well_surveys()

                if surveys_df.empty:
                    raise ValueError(f"No survey data found for well '{well_name}'")

                surveys_df = surveys_df.sort_values("md")
                mds = surveys_df["md"].values
                incs = surveys_df["inc"].values
                azims = surveys_df["azim"].values

                deviation = wpp.deviation(mds, incs, azims)
                tvds = deviation.minimum_curvature().depth

                tvd_data = dict(zip(mds, tvds, strict=True))
                well.add_curve_data("TVD", tvd_data, unit="m")

                return {
                    "success": True,
                    "tvd_points_saved": len(tvd_data),
                    "wells_processed": 1,
                    "wells": [well_name],
                }

            # Calculate for all wells in project
            total_tvd_points = 0
            wells_processed = []

            for wn in proj.get_well_names():
                well = proj.get_well(wn)
                surveys_df = well.get_well_surveys()

                if surveys_df.empty:
                    continue

                surveys_df = surveys_df.sort_values("md")
                mds = surveys_df["md"].values
                incs = surveys_df["inc"].values
                azims = surveys_df["azim"].values

                try:
                    deviation = wpp.deviation(mds, incs, azims)
                    tvds = deviation.minimum_curvature().depth

                    tvd_data = dict(zip(mds, tvds, strict=True))
                    well.add_curve_data("TVD", tvd_data, unit="m")

                    total_tvd_points += len(tvd_data)
                    wells_processed.append(wn)
                except Exception as e:
                    # Log error but continue with other wells
                    print(f"Warning: Failed to calculate TVD for well {wn}: {e}")
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
