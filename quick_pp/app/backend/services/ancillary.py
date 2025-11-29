from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List, Dict, Any
import pandas as pd

from .database import connector as db_connector
from quick_pp.database import objects as db_objects
from sqlalchemy import select
from fastapi import Body, UploadFile, File
import csv
import io

router = APIRouter(
    prefix="/database/projects/{project_id}/wells/{well_name}", tags=["Ancillary"]
)


@router.get("/formation_tops", summary="List formation tops for a well")
def list_formation_tops(project_id: int, well_name: str):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            tops_df = well.get_formation_tops()
            return {"tops": tops_df.to_dict(orient="records")}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list formation tops: {e}"
        )


@router.post("/formation_tops", summary="Add or update formation tops for a well")
def add_formation_tops(
    project_id: int,
    well_name: str,
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    if not isinstance(payload, dict) or "tops" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'tops'."
        )

    tops = payload.get("tops")
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            well.add_formation_tops(tops)
            # session commit handled by connector
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


@router.delete("/formation_tops/{top_name}", summary="Delete a formation top by name")
def delete_formation_top(project_id: int, well_name: str, top_name: str):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            # find ORM FormationTop and delete
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


@router.get("/fluid_contacts", summary="List fluid contacts for a well")
def list_fluid_contacts(project_id: int, well_name: str):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            df = well.get_fluid_contacts()
            return {"fluid_contacts": df.to_dict(orient="records")}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list fluid contacts: {e}"
        )


@router.post("/fluid_contacts", summary="Add or update fluid contacts for a well")
def add_fluid_contacts(
    project_id: int,
    well_name: str,
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    if not isinstance(payload, dict) or "contacts" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'contacts'."
        )
    contacts = payload.get("contacts")
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            well.add_fluid_contacts(contacts)
            return {"created": contacts}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add fluid contacts: {e}"
        )


@router.get("/pressure_tests", summary="List pressure tests for a well")
def list_pressure_tests(project_id: int, well_name: str):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            df = well.get_pressure_tests()
            return {"pressure_tests": df.to_dict(orient="records")}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to list pressure tests: {e}"
        )


@router.post("/pressure_tests", summary="Add or update pressure tests for a well")
def add_pressure_tests(
    project_id: int,
    well_name: str,
    payload: Dict[str, List[Dict[str, Any]]] = Body(...),
):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    if not isinstance(payload, dict) or "tests" not in payload:
        raise HTTPException(
            status_code=400, detail="Payload must be a dict with key 'tests'."
        )
    tests = payload.get("tests")
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            well.add_pressure_tests(tests)
            return {"created": tests}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Failed to add pressure tests: {e}"
        )


@router.post(
    "/formation_tops/preview",
    summary="Upload CSV and return parsed preview for formation tops",
)
def formation_tops_preview(
    project_id: int, well_name: str, file: UploadFile = File(...)
):
    """Return a preview (first 50 rows) and detected columns from uploaded CSV file."""
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
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


@router.get("/core_samples", summary="List core samples for a well")
def list_core_samples(project_id: int, well_name: str):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
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
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list core samples: {e}")


@router.get("/data", summary="Get top-to-bottom well data (rows of logs)")
def get_well_top_to_bottom_data(
    project_id: int, well_name: str, include_ancillary: bool = False
):
    """Return the full well data (curve logs) as a JSON array of rows.

    Each row is a dict mapping column names (lowercased) to values. The DEPTH
    column from the DataFrame is renamed to `depth` (lowercase) so clients can
    rely on a consistent key.
    """
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )

    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            # get_well_data returns a pandas DataFrame with DEPTH and curve columns
            df = proj.get_well_data(well_name)
            if df.empty:
                return []

            # normalize column names to lowercase and ensure DEPTH -> depth
            df = df.reset_index(drop=True)
            df.columns = [c.lower() for c in df.columns]

            # convert NaNs to None for JSON serialization
            df = df.where(pd.notnull(df), None)

            records = df.to_dict(orient="records")

            if include_ancillary:
                # collect ancillary data to return alongside rows
                well = proj.get_well(well_name)
                tops_df = well.get_formation_tops()
                contacts_df = well.get_fluid_contacts()
                pressure_df = well.get_pressure_tests()
                core_raw = well.get_core_data()
                # convert frames to records (lists/dicts)
                tops = tops_df.to_dict(orient="records") if not tops_df.empty else []
                contacts = (
                    contacts_df.to_dict(orient="records")
                    if not contacts_df.empty
                    else []
                )
                pressure = (
                    pressure_df.to_dict(orient="records")
                    if not pressure_df.empty
                    else []
                )
                # core_raw is a dict of sample_name -> dict with 'depth','measurements' DataFrame
                core_samples = []
                for name, s in core_raw.items():
                    measurements = []
                    if hasattr(s.get("measurements"), "to_dict"):
                        measurements = s.get("measurements").to_dict(orient="records")
                    core_samples.append(
                        {
                            "sample_name": name,
                            "depth": s.get("depth"),
                            "measurements": measurements,
                        }
                    )

                return {
                    "data": records,
                    "formation_tops": tops,
                    "fluid_contacts": contacts,
                    "pressure_tests": pressure,
                    "core_samples": core_samples,
                }

            return records
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get well data: {e}")


@router.get("/rca", summary="List core point measurements (RCA) for a well")
def list_rca(project_id: int, well_name: str):
    """Return consolidated point measurements across all core samples for a well."""
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            core_data = well.get_core_data()
            rows = []
            for sample_name, sd in core_data.items():
                measurements = sd.get("measurements")
                if hasattr(measurements, "to_dict"):
                    for r in measurements.to_dict(orient="records"):
                        r.update({"sample_name": sample_name, "depth": sd.get("depth")})
                        rows.append(r)
            return {"rca": rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list RCA: {e}")


@router.post("/rca", summary="Add RCA (core point measurements) for a sample")
def add_rca(project_id: int, well_name: str, payload: Dict[str, Any] = Body(...)):
    """Add or update point measurements for a core sample. Requires 'sample_name', 'depth', and 'measurements'."""
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    required = ["sample_name", "depth", "measurements"]
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
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


@router.get("/scal", summary="List SCAL (relperm & capillary) data for a well")
def list_scal(project_id: int, well_name: str):
    """Return consolidated SCAL data (relperm & capillary) across core samples."""
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
            core_data = well.get_core_data()
            relperm_rows = []
            pc_rows = []
            for sample_name, sd in core_data.items():
                relperm = sd.get("relperm")
                if hasattr(relperm, "to_dict"):
                    for r in relperm.to_dict(orient="records"):
                        r.update({"sample_name": sample_name, "depth": sd.get("depth")})
                        relperm_rows.append(r)
                pc = sd.get("pc")
                if hasattr(pc, "to_dict"):
                    for p in pc.to_dict(orient="records"):
                        p.update({"sample_name": sample_name, "depth": sd.get("depth")})
                        pc_rows.append(p)
            return {"relperm": relperm_rows, "pc": pc_rows}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list SCAL: {e}")


@router.post("/scal", summary="Add SCAL data for a sample (relperm and/or capillary)")
def add_scal(project_id: int, well_name: str, payload: Dict[str, Any] = Body(...)):
    """Add relperm and/or capillary pressure data for a core sample. Requires 'sample_name' and 'depth'."""
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    required = ["sample_name", "depth"]
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
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


@router.get("/core_samples/{sample_name}", summary="Get core sample details by name")
def get_core_sample(project_id: int, well_name: str, sample_name: str):
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
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


@router.post("/core_samples", summary="Add or update a core sample with measurements")
def add_core_sample(
    project_id: int, well_name: str, payload: Dict[str, Any] = Body(...)
):
    """Payload example:
    {
      "sample_name": "Plug-1",
      "depth": 2500.5,
      "measurements": [{"property_name":"POR","value":0.21,"unit":"v/v"}],
      "relperm_data": [{"saturation":0.1,"kr":0.01,"phase":"water"}],
      "pc_data": [{"saturation":0.1,"pressure":12.3}]
    }
    """
    if db_connector is None:
        raise HTTPException(
            status_code=400,
            detail="DB connector not initialized. Call /database/init first.",
        )
    required = ["sample_name", "depth", "measurements"]
    if not isinstance(payload, dict) or not all(k in payload for k in required):
        raise HTTPException(status_code=400, detail=f"Payload must include {required}")
    try:
        with db_connector.get_session() as session:
            proj = db_objects.Project.load(session, project_id=project_id)
            well = proj.get_well(well_name)
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
