import io
import json
import os
import zipfile

os.environ["QPP_DATABASE_URL"] = "sqlite:///:memory:"

import lasio
import numpy as np
import pandas as pd
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from quick_pp.database.models import Base
from quick_pp.database.objects import Project
from quick_pp.las_handler import export_to_las, export_to_las_string


@pytest.fixture
def db_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_export_to_las_preserves_df_and_embeds_parameters(tmp_path):
    df = pd.DataFrame(
        {
            "WELL_NAME": ["WELL-1", "WELL-1", "WELL-1"],
            "UWI": ["UWI-001", "UWI-001", "UWI-001"],
            "DEPTH": [2500.0, 2500.5, 2501.0],
            "GR": [45.0, 60.0, 110.0],
            "PHIT": [0.22, 0.18, 0.08],
            "SWT": [0.25, 0.35, 0.90],
            "PERM": [150.0, 45.0, 0.2],
            "ZONES": ["Zone_A", "Zone_A", "Zone_B"],
        }
    )
    orig_cols = list(df.columns)
    las_path = export_to_las(
        well_data=df,
        well_name="WELL-1",
        folder=str(tmp_path),
        uwi="UWI-001",
        depth_uom="m",
        parameters={"sw_m": 2.0, "sw_n": 2.0, "litho_model": "ssc"},
    )

    # Caller's DataFrame must not be mutated
    assert list(df.columns) == orig_cols
    assert "DEPTH" in df.columns

    parsed = lasio.read(las_path)
    assert parsed.well.WELL.value == "WELL-1"
    assert parsed.well.UWI.value == "UWI-001"
    assert float(parsed.well.STEP.value) == pytest.approx(0.5)
    curve_names = [c.mnemonic for c in parsed.curves]
    assert "DEPTH" in curve_names
    assert "GR" in curve_names
    assert "PHIT" in curve_names
    assert "ZONE_ID" in curve_names
    assert "WELL_NAME" not in curve_names

    # Check ~Parameter block
    assert float(parsed.params["SW_M"].value) == pytest.approx(2.0)
    assert parsed.params["LITHO_MODEL"].value == "ssc"
    assert parsed.params["ZONE_1"].value == "Zone_A"
    assert parsed.params["ZONE_2"].value == "Zone_B"

    # Also test in-memory LAS 2.0 string export
    las_str = export_to_las_string(
        well_data=df,
        well_name="WELL-1",
        uwi="UWI-001",
        depth_uom="m",
        parameters={"sw_m": 2.0, "sw_n": 2.0, "litho_model": "ssc"},
    )
    assert "~Params" in las_str
    assert "SW_M" in las_str
    assert "ZONE_ID" in las_str


def test_project_and_well_export_handover_package(db_session, tmp_path):
    proj = Project(
        db_session=db_session,
        name="Volve_Phase1",
        description="Handover unit test project",
    )
    depths = np.linspace(3000.0, 3010.0, 21)
    well_df = pd.DataFrame(
        {
            "WELL_NAME": "15-9-F-12",
            "DEPTH": depths,
            "GR": np.linspace(30.0, 95.0, 21),
            "NPHI": np.linspace(0.18, 0.26, 21),
            "RHOB": np.linspace(2.25, 2.50, 21),
            "RT": np.linspace(25.0, 2.0, 21),
            "VSHALE": np.linspace(0.10, 0.65, 21),
            "PHIT": np.linspace(0.24, 0.10, 21),
            "PHIE": np.linspace(0.22, 0.05, 21),
            "SWT": np.linspace(0.20, 0.90, 21),
            "PERM": np.logspace(2.5, -0.5, 21),
        }
    )
    proj.read_dataframe(well_df)
    proj.update_ancillary_data(
        {
            "15-9-F-12": {
                "formation_tops": [
                    {"name": "Hugin_Top", "depth": 3000.0},
                    {"name": "Sleipner_Top", "depth": 3006.0},
                ],
                "fluid_contacts": [
                    {"name": "OWC", "depth": 3005.5},
                ],
                "pressure_tests": [
                    {"depth": 3002.0, "pressure": 4450.0, "pressure_uom": "psi"},
                    {"depth": 3007.0, "pressure": 4462.0, "pressure_uom": "psi"},
                ],
                "well_surveys": [
                    {"md": 3000.0, "inc": 0.0, "azim": 0.0},
                    {"md": 3010.0, "inc": 5.0, "azim": 120.0},
                ],
                "core_data": [
                    {
                        "sample_name": "PLUG-1",
                        "depth": 3001.5,
                        "description": "Fine sandstone",
                        "remark": "Good plug",
                        "measurements": [
                            {"property_name": "CPORE", "value": 0.23, "unit": "v/v"},
                            {"property_name": "CPERM", "value": 280.0, "unit": "mD"},
                        ],
                        "relperm_data": [
                            {"saturation": 0.20, "kr": 0.0, "phase": "water"},
                            {"saturation": 0.20, "kr": 0.85, "phase": "oil"},
                            {"saturation": 0.80, "kr": 0.35, "phase": "water"},
                            {"saturation": 0.80, "kr": 0.0, "phase": "oil"},
                        ],
                        "pc_data": [
                            {
                                "saturation": 0.85,
                                "pressure": 2.5,
                                "experiment_type": "micp",
                                "cycle": "drainage",
                            },
                            {
                                "saturation": 0.25,
                                "pressure": 45.0,
                                "experiment_type": "micp",
                                "cycle": "drainage",
                            },
                        ],
                    }
                ],
            }
        }
    )
    db_session.commit()

    zip_out = tmp_path / "Volve_Handover.zip"
    written_path = proj.export_handover_package(output_zip_path=str(zip_out))
    assert zip_out.exists()
    assert written_path == str(zip_out)

    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
        prefix = "Volve_Phase1_Handover"
        assert f"{prefix}/01_well_logs_las/15-9-F-12.las" in names
        assert f"{prefix}/02_tops_and_contacts/formation_tops.csv" in names
        assert f"{prefix}/02_tops_and_contacts/fluid_contacts.csv" in names
        assert f"{prefix}/03_surveys/well_surveys.csv" in names
        assert f"{prefix}/04_core_and_pressure/core_rca_points.csv" in names
        assert f"{prefix}/04_core_and_pressure/las_point_sets/15-9-F-12_RCA.las" in names
        assert f"{prefix}/04_core_and_pressure/pressure_tests_mdt.csv" in names
        assert f"{prefix}/05_scal_and_sim/scal_capillary_pressure.csv" in names
        assert f"{prefix}/05_scal_and_sim/scal_relperm_tables.csv" in names
        assert f"{prefix}/05_scal_and_sim/eclipse_swof.inc" in names
        assert f"{prefix}/06_summaries_and_audit/reservoir_summary_by_zone.csv" in names
        assert f"{prefix}/06_summaries_and_audit/CPI_Parameters_Manifest.json" in names
        assert f"{prefix}/import_scripts/import_to_techlog.py" in names
        assert f"{prefix}/import_scripts/import_to_geolog.tcl" in names
        assert f"{prefix}/README_HANDOVER.md" in names

        # Verify manifest counts
        manifest = json.loads(
            zf.read(f"{prefix}/06_summaries_and_audit/CPI_Parameters_Manifest.json")
        )
        counts = manifest["inventory_counts"]
        assert counts["wells"] == 1
        assert counts["formation_tops"] == 2
        assert counts["fluid_contacts"] == 1
        assert counts["well_surveys"] == 2
        assert counts["core_rca_samples"] == 1
        assert counts["pressure_tests"] == 2
        assert counts["scal_pc_points"] == 2
        assert counts["scal_relperm_points"] == 2
        assert counts["reservoir_summary_rows"] > 0

        # Verify tops CSV has TVD column
        tops_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/02_tops_and_contacts/formation_tops.csv"))
        )
        assert list(tops_csv["TOP_NAME"]) == ["Hugin_Top", "Sleipner_Top"]
        assert "TVD" in tops_csv.columns

    # Also test single-well export
    well_obj = proj.get_well("15-9-F-12")
    well_zip = tmp_path / "Well_Handover.zip"
    well_obj.export_handover_package(output_zip_path=str(well_zip))
    assert well_zip.exists()


def test_fastapi_handover_and_las_export_endpoints(db_session, monkeypatch):
    from quick_pp.app.backend.services import database as db_service

    proj = Project(db_session=db_session, name="API_Handover_Test")
    proj.read_dataframe(
        pd.DataFrame(
            {
                "WELL_NAME": ["Well-1", "Well-1"],
                "DEPTH": [2000.0, 2001.0],
                "GR": [40.0, 80.0],
                "PHIT": [0.21, 0.15],
                "SWT": [0.30, 0.55],
                "PERM": [120.0, 25.0],
            }
        )
    )
    proj.update_ancillary_data(
        {"Well-1": {"formation_tops": [{"name": "Top-A", "depth": 2000.0}]}}
    )
    db_session.commit()

    class DummyCtx:
        def __enter__(self):
            return db_session

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db_service, "get_db", lambda: DummyCtx())

    # 1. Test full project handover .zip export
    res_zip = db_service.export_project_handover(
        project_id=proj.project_id,
        well_name=None,
        wells=None,
        include_las=True,
        include_excel=True,
    )
    assert res_zip.status_code == 200
    assert res_zip.media_type == "application/zip"
    with zipfile.ZipFile(io.BytesIO(res_zip.body), "r") as zf:
        assert "API_Handover_Test_Handover/01_well_logs_las/Well-1.las" in zf.namelist()
        assert (
            "API_Handover_Test_Handover/02_tops_and_contacts/formation_tops.csv"
            in zf.namelist()
        )

    # 2. Test single well LAS 2.0 export
    res_las = db_service.export_well_las_endpoint(
        project_id=proj.project_id,
        well_name="Well-1",
    )
    assert res_las.status_code == 200
    las_text = res_las.body.decode("utf-8")
    assert "~Params" in las_text
    assert "PHIT" in las_text
