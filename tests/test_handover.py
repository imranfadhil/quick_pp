import io
import json
import zipfile

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
    proj._orm_project.config_data = {
        "shf": {"fwl_depth": 3005.5, "ift": 30.0},
        "fpt": {"water_gradient": 0.433},
    }
    proj.read_dataframe(well_df)
    w_obj = proj.get_well("15-9-F-12")
    w_obj.header_data = {
        "kb_elev": 54.0,
        "df_elev": 54.0,
        "field": "Volve",
        "company": "Equinor",
        "country": "Norway",
        "x_coord": 435000.0,
        "y_coord": 6478000.0,
    }
    w_obj.save()
    proj.update_ancillary_data(
        {
            "15-9-F-12": {
                "formation_tops": [
                    {"name": "Hugin_Top", "depth": 3000.0},
                    {"name": "Sleipner_Top", "depth": 3006.0},
                ],
                "fluid_contacts": [
                    {
                        "name": "Hugin_Top - OWC",
                        "depth": 3005.5,
                        "zone": "Hugin_Top",
                        "contact_type": "OWC",
                        "fault_block": "FB-North",
                        "source": "MDT+Log",
                        "remarks": "Confirmed by MDT gradient",
                    },
                ],
                "perforations": [
                    {
                        "name": "PERF-1",
                        "top_depth": 3001.0,
                        "bottom_depth": 3004.5,
                        "description": "Upper Hugin interval",
                    }
                ],
                "hole_sections": [
                    {
                        "name": '12.25" Intermediate',
                        "top": 2990.0,
                        "bottom": 3003.0,
                        "bit_size": 12.25,
                        "mud_type": "WBM",
                        "mud_weight": 10.2,
                        "mud_salinity": 28000.0,
                        "rmf": 0.18,
                        "rmf_temp": 25.0,
                    },
                    {
                        "name": '8.5" Reservoir',
                        "top": 3003.0,
                        "bottom": 3015.0,
                        "bit_size": 8.5,
                        "mud_type": "OBM",
                        "mud_weight": 11.4,
                        "mud_salinity": None,
                        "rmf": None,
                        "rmf_temp": None,
                    },
                ],
                "pressure_tests": [
                    {
                        "depth": 3002.0,
                        "pressure": 4450.0,
                        "pressure_uom": "psi",
                        "mobility": 125.0,
                        "permeability": 85.0,
                        "temperature": 98.5,
                        "fluid_type": "OIL",
                        "test_quality": "GOOD",
                        "zone": "Hugin_Top",
                    },
                    {
                        "depth": 3007.0,
                        "pressure": 4462.0,
                        "pressure_uom": "psi",
                        "fluid_type": "WATER",
                        "test_quality": "GOOD",
                        "zone": "Sleipner_Top",
                    },
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
                            {
                                "property_name": "CPORE_AMB",
                                "value": 0.245,
                                "unit": "v/v",
                            },
                            {
                                "property_name": "CPERM_AIR",
                                "value": 310.0,
                                "unit": "mD",
                            },
                            {
                                "property_name": "RAW_DEPTH",
                                "value": 3000.0,
                                "unit": "m",
                            },
                            {"property_name": "SHIFT_M", "value": 1.5, "unit": "m"},
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
        assert f"{prefix}/02_tops_and_contacts/perforations.csv" in names
        assert f"{prefix}/02_tops_and_contacts/hole_sections.csv" in names
        assert f"{prefix}/03_surveys/well_headers_and_datum.csv" in names
        assert f"{prefix}/03_surveys/well_surveys.csv" in names
        assert f"{prefix}/04_core_and_pressure/core_rca_points.csv" in names
        assert (
            f"{prefix}/04_core_and_pressure/las_point_sets/15-9-F-12_RCA.las" in names
        )
        assert (
            f"{prefix}/04_core_and_pressure/las_point_sets/15-9-F-12_MDT.las" in names
        )
        assert f"{prefix}/04_core_and_pressure/pressure_tests_mdt.csv" in names
        assert f"{prefix}/05_scal_and_sim/scal_capillary_pressure.csv" in names
        assert f"{prefix}/05_scal_and_sim/scal_relperm_tables.csv" in names
        assert f"{prefix}/05_scal_and_sim/eclipse_swof.inc" in names
        assert f"{prefix}/06_summaries_and_audit/reservoir_summary_by_zone.csv" in names
        assert f"{prefix}/06_summaries_and_audit/CPI_Parameters_Manifest.json" in names
        assert f"{prefix}/import_scripts/import_to_techlog.py" in names
        assert f"{prefix}/import_scripts/import_to_geolog.tcl" in names
        assert f"{prefix}/README_HANDOVER.md" in names

        # Verify manifest counts & project_parameters
        manifest = json.loads(
            zf.read(f"{prefix}/06_summaries_and_audit/CPI_Parameters_Manifest.json")
        )
        counts = manifest["inventory_counts"]
        assert counts["wells"] == 1
        assert counts["formation_tops"] == 2
        assert counts["fluid_contacts"] == 1
        assert counts["perforations"] == 1
        assert counts["hole_sections"] == 2
        assert counts["well_surveys"] == 2
        assert counts["core_rca_samples"] == 1
        assert counts["pressure_tests"] == 2
        assert counts["scal_pc_points"] == 2
        assert counts["scal_relperm_points"] == 2
        assert counts["reservoir_summary_rows"] > 0
        assert manifest["project_parameters"]["shf"]["fwl_depth"] == pytest.approx(
            3005.5
        )

        # Verify tops CSV has TVD and TVDSS columns
        tops_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/02_tops_and_contacts/formation_tops.csv"))
        )
        assert list(tops_csv["TOP_NAME"]) == ["Hugin_Top", "Sleipner_Top"]
        assert "TVD" in tops_csv.columns
        assert "TVDSS" in tops_csv.columns
        assert float(tops_csv["TVDSS"].iloc[0]) == pytest.approx(3000.0 - 54.0, abs=0.1)

        # Verify perforations & hole sections CSVs
        perfs_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/02_tops_and_contacts/perforations.csv"))
        )
        assert list(perfs_csv["PERF_NAME"]) == ["PERF-1"]
        assert "TOP_TVDSS" in perfs_csv.columns

        hs_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/02_tops_and_contacts/hole_sections.csv"))
        )
        assert len(hs_csv) == 2
        assert list(hs_csv["MUD_TYPE"]) == ["WBM", "OBM"]
        assert float(hs_csv["BIT_SIZE"].iloc[0]) == pytest.approx(12.25)
        assert "TOP_TVDSS" in hs_csv.columns

        contacts_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/02_tops_and_contacts/fluid_contacts.csv"))
        )
        assert contacts_csv["FAULT_BLOCK"].iloc[0] == "FB-North"
        assert contacts_csv["CONTACT_TYPE"].iloc[0] == "OWC"

        # Verify core RCA lineage columns
        rca_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/04_core_and_pressure/core_rca_points.csv"))
        )
        assert float(rca_csv["CPORE_AMB"].iloc[0]) == pytest.approx(0.245)
        assert float(rca_csv["CPERM_AIR"].iloc[0]) == pytest.approx(310.0)
        assert float(rca_csv["RAW_DEPTH"].iloc[0]) == pytest.approx(3000.0)
        assert float(rca_csv["SHIFT_M"].iloc[0]) == pytest.approx(1.5)

        # Verify pressure tests CSV has FPT columns
        mdt_csv = pd.read_csv(
            io.BytesIO(zf.read(f"{prefix}/04_core_and_pressure/pressure_tests_mdt.csv"))
        )
        assert list(mdt_csv["FLUID_TYPE"]) == ["OIL", "WATER"]
        assert "TVDSS" in mdt_csv.columns

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

    # 3. Test inspect_handover_package_endpoint
    from starlette.datastructures import Headers

    class DummyUploadFile:
        def __init__(self, filename, data):
            self.filename = filename
            self.file = io.BytesIO(data)

    upload_f = DummyUploadFile("handover.zip", res_zip.body)
    preview_res = db_service.inspect_handover_package_endpoint(file=upload_f)
    assert preview_res["project_name"] == "API_Handover_Test"
    assert preview_res["wells_count"] == 1
    assert preview_res["well_names"] == ["Well-1"]

    # 4. Test commit_handover_package_endpoint
    class DummyReq:
        headers = Headers({})
        cookies = {}

    commit_upload_f = DummyUploadFile("handover.zip", res_zip.body)
    commit_res = db_service.commit_handover_package_endpoint(
        request=DummyReq(),
        file=commit_upload_f,
        project_name="API_Imported_Project",
        project_id=None,
        overwrite_existing=True,
    )
    assert commit_res["success"] is True
    assert commit_res["project_name"] == "API_Imported_Project"
    assert commit_res["wells_imported"] == 1


def test_roundtrip_inspect_and_import_handover_package(db_session, tmp_path):
    from quick_pp.handover import import_handover_archive, inspect_handover_archive

    # 1. Create a rich project with full ancillary data
    proj = Project(
        db_session=db_session,
        name="Original_Project",
        description="Original project description",
    )
    depths = np.linspace(2500.0, 2510.0, 11)
    df = pd.DataFrame(
        {
            "WELL_NAME": "WELL-ALPHA",
            "DEPTH": depths,
            "GR": np.linspace(40.0, 90.0, 11),
            "RHOB": np.linspace(2.2, 2.5, 11),
            "NPHI": np.linspace(0.25, 0.15, 11),
            "PHIT": np.linspace(0.22, 0.12, 11),
            "SWT": np.linspace(0.20, 0.85, 11),
        }
    )
    proj.read_dataframe(df)
    proj.update_ancillary_data(
        {
            "WELL-ALPHA": {
                "formation_tops": [
                    {"name": "Top_A", "depth": 2500.0},
                    {"name": "Top_B", "depth": 2505.0},
                ],
                "fluid_contacts": [
                    {
                        "name": "Top_A - OWC",
                        "depth": 2506.5,
                        "zone": "Top_A",
                        "contact_type": "OWC",
                        "fault_block": "Block-1",
                        "source": "FPT",
                        "remarks": "Primary contact",
                    },
                ],
                "perforations": [
                    {
                        "name": "PERF-A1",
                        "top": 2501.0,
                        "bottom": 2504.0,
                        "description": "Main pay perf",
                    },
                ],
                "hole_sections": [
                    {
                        "name": '8.5" Production',
                        "top": 2495.0,
                        "bottom": 2515.0,
                        "bit_size": 8.5,
                        "mud_type": "WBM",
                        "mud_weight": 10.5,
                        "mud_salinity": 32000.0,
                        "rmf": 0.15,
                        "rmf_temp": 24.0,
                    },
                ],
                "well_surveys": [
                    {"md": 2500.0, "inc": 0.0, "azim": 0.0},
                    {"md": 2510.0, "inc": 2.5, "azim": 90.0},
                ],
                "pressure_tests": [
                    {
                        "depth": 2502.0,
                        "pressure": 3500.0,
                        "pressure_uom": "psi",
                        "temperature": 95.0,
                        "temperature_uom": "degF",
                        "tvdss": 2448.0,
                        "fluid_type": "OIL",
                        "test_quality": "VALID",
                    },
                    {
                        "depth": 2508.0,
                        "pressure": 3510.0,
                        "pressure_uom": "psi",
                        "fluid_type": "WATER",
                        "test_quality": "VALID",
                    },
                ],
                "core_data": [
                    {
                        "sample_name": "CORE-1",
                        "depth": 2503.0,
                        "measurements": [
                            {"property_name": "CPORE", "value": 0.21, "unit": "v/v"},
                            {"property_name": "CPERM", "value": 150.0, "unit": "mD"},
                            {
                                "property_name": "CGRAIN_DENSITY",
                                "value": 2.65,
                                "unit": "g/cc",
                            },
                            {"property_name": "QV", "value": 0.18, "unit": "meq/mL"},
                        ],
                        "relperm_data": [
                            {"saturation": 0.25, "kr": 0.0, "phase": "water"},
                            {"saturation": 0.25, "kr": 0.82, "phase": "oil"},
                            {"saturation": 0.75, "kr": 0.38, "phase": "water"},
                            {"saturation": 0.75, "kr": 0.0, "phase": "oil"},
                        ],
                        "pc_data": [
                            {
                                "saturation": 0.90,
                                "pressure": 1.8,
                                "experiment_type": "micp",
                                "cycle": "drainage",
                            },
                            {
                                "saturation": 0.28,
                                "pressure": 38.0,
                                "experiment_type": "micp",
                                "cycle": "drainage",
                            },
                        ],
                    },
                ],
            }
        }
    )
    proj._orm_project.config_data = {"shf": {"fwl": 2506.5}}
    db_session.commit()

    # 2. Export handover package
    zip_path = tmp_path / "Exported_Handover.zip"
    proj.export_handover_package(output_zip_path=str(zip_path))
    assert zip_path.exists()

    with open(zip_path, "rb") as f:
        zip_bytes = f.read()

    # 3. Test inspect_handover_archive
    preview = inspect_handover_archive(zip_bytes)
    assert preview["project_name"] == "Original_Project"
    assert preview["wells_count"] == 1
    assert preview["well_names"] == ["WELL-ALPHA"]
    assert preview["inventory_counts"]["formation_tops"] == 2
    assert preview["inventory_counts"]["fluid_contacts"] == 1
    assert preview["inventory_counts"]["perforations"] == 1
    assert preview["inventory_counts"]["hole_sections"] == 1
    assert preview["inventory_counts"]["well_surveys"] == 2
    assert preview["inventory_counts"]["pressure_tests"] == 2
    assert preview["inventory_counts"]["core_rca_samples"] == 1
    assert preview["inventory_counts"]["scal_pc_points"] == 2
    assert preview["inventory_counts"]["scal_relperm_points"] == 2
    assert preview["has_hole_sections"] is True
    assert preview["has_scal"] is True

    # 4. Test import_handover_archive into a new project
    res = import_handover_archive(
        db_session, zip_bytes, target_project_name="Rehydrated_Project"
    )
    assert res["success"] is True
    assert res["wells_imported"] == 1
    assert res["project_name"] == "Rehydrated_Project"

    # 5. Verify rehydrated database entities
    imported_proj = Project.load(db_session, project_id=res["project_id"])
    assert "WELL-ALPHA" in imported_proj.get_well_names()

    rehydrated_well = imported_proj.get_well("WELL-ALPHA")
    assert rehydrated_well.name == "WELL-ALPHA"

    # Check curve data
    w_data = imported_proj.get_well_data("WELL-ALPHA")
    assert not w_data.empty
    assert "DEPTH" in w_data.columns
    assert "PHIT" in w_data.columns
    assert "SWT" in w_data.columns
    assert len(w_data) == 11

    # Check tops
    tops = rehydrated_well.get_formation_tops()
    assert len(tops) == 2
    assert set(tops["name"]) == {"Top_A", "Top_B"}

    # Check contacts (including extended lineage columns)
    contacts = rehydrated_well.get_fluid_contacts()
    assert len(contacts) == 1
    assert contacts.iloc[0]["depth"] == pytest.approx(2506.5)
    assert contacts.iloc[0]["zone"] == "Top_A"
    assert contacts.iloc[0]["contact_type"] == "OWC"
    assert contacts.iloc[0]["fault_block"] == "Block-1"
    assert contacts.iloc[0]["source"] == "FPT"
    assert contacts.iloc[0]["remarks"] == "Primary contact"

    # Check perforations & hole sections
    perfs = rehydrated_well.get_perforations()
    assert len(perfs) == 1
    assert perfs.iloc[0]["name"] == "PERF-A1"
    assert perfs.iloc[0]["top"] == pytest.approx(2501.0)
    assert perfs.iloc[0]["bottom"] == pytest.approx(2504.0)

    hole_secs = rehydrated_well.get_hole_sections()
    assert len(hole_secs) == 1
    assert hole_secs.iloc[0]["name"] == '8.5" Production'
    assert hole_secs.iloc[0]["top"] == pytest.approx(2495.0)
    assert hole_secs.iloc[0]["bottom"] == pytest.approx(2515.0)
    assert hole_secs.iloc[0]["bit_size"] == pytest.approx(8.5)
    assert hole_secs.iloc[0]["mud_type"] == "WBM"
    assert hole_secs.iloc[0]["mud_weight"] == pytest.approx(10.5)
    assert hole_secs.iloc[0]["mud_salinity"] == pytest.approx(32000.0)
    assert hole_secs.iloc[0]["rmf"] == pytest.approx(0.15)
    assert hole_secs.iloc[0]["rmf_temp"] == pytest.approx(24.0)

    # Check surveys
    surveys = rehydrated_well.get_well_surveys()
    assert len(surveys) == 2

    # Check pressure tests (including temperature_uom & tvdss override)
    p_tests = rehydrated_well.get_pressure_tests()
    assert len(p_tests) == 2
    assert p_tests.iloc[0]["temperature_uom"] == "degF"
    assert p_tests.iloc[0]["tvdss"] == pytest.approx(2448.0)

    # Check core samples, extended measurements, SCAL Pc & RelPerm
    core = rehydrated_well.get_core_data()
    assert "CORE-1" in core
    meas_df = core["CORE-1"]["measurements"]
    meas_props = {row["property"]: row["value"] for _, row in meas_df.iterrows()}
    assert meas_props["CGRAIN_DENSITY"] == pytest.approx(2.65)
    assert meas_props["QV"] == pytest.approx(0.18)

    scal_pc = rehydrated_well.get_scal_pc_data()
    assert len(scal_pc) == 2
    assert scal_pc[0]["pc"] == pytest.approx(1.8)

    relperm_df = core["CORE-1"]["relperm"]
    assert len(relperm_df) == 2
    assert "kr_oil" in relperm_df.columns
    assert "kr_water" in relperm_df.columns

    # Check project config data restored
    assert imported_proj._orm_project.config_data.get("shf", {}).get(
        "fwl"
    ) == pytest.approx(2506.5)


def test_handover_import_persists_in_list_projects_for_non_admin_and_preserves_well_config(
    db_session, tmp_path, monkeypatch
):
    import asyncio

    from fastapi import Response
    from starlette.datastructures import Headers

    from quick_pp.app.backend.services import database as db_service
    from quick_pp.app.backend.utils import db as db_utils
    from quick_pp.database import models as db_models

    # 1. Verify PostgreSQL URLs never silently split-brain to local SQLite unless explicitly opted in
    monkeypatch.delenv("QPP_ALLOW_SQLITE_FALLBACK", raising=False)
    assert (
        db_utils._allow_sqlite_fallback(
            "postgresql+psycopg2://user:pw@postgres:5432/quick_pp"
        )
        is False
    )
    assert db_utils._allow_sqlite_fallback("sqlite:///./data/quick_pp.db") is True
    monkeypatch.setenv("QPP_ALLOW_SQLITE_FALLBACK", "true")
    assert (
        db_utils._allow_sqlite_fallback(
            "postgresql+psycopg2://user:pw@postgres:5432/quick_pp"
        )
        is True
    )
    monkeypatch.delenv("QPP_ALLOW_SQLITE_FALLBACK", raising=False)

    # 2. Build a handover package whose well has custom CPI parameters AND datum headers
    source_proj = Project(
        db_session=db_session, name="Guillemot_Source", description="Source pkg"
    )
    source_proj.read_dataframe(
        pd.DataFrame(
            {
                "WELL_NAME": ["G-1069", "G-1069", "G-1069"],
                "DEPTH": [2100.0, 2100.5, 2101.0],
                "GR": [35.0, 55.0, 85.0],
                "PHIT": [0.24, 0.19, 0.11],
                "SWT": [0.22, 0.40, 0.85],
            }
        )
    )
    src_well = source_proj.get_well("G-1069")
    src_well.header_data = {"kb_elev": 42.5, "field": "Guillemot", "well_num": 1069}
    src_well._config_instance.update(
        {
            "sw_m": 1.85,
            "sw_n": 2.15,
            "module_params": {"saturation": {"rw": 0.035}},
        }
    )
    src_well.save()
    db_session.commit()

    zip_path = tmp_path / "Guillemot_Handover.zip"
    source_proj.export_handover_package(output_zip_path=str(zip_path))
    zip_bytes = zip_path.read_bytes()

    # Create a non-admin user and an existing unowned project ("Legacy_Guillemot" with created_by=None)
    user = db_models.User(
        username="petro_user",
        email="petro@example.com",
        hashed_password="x",
        role="user",
        status="CONFIRMED",
        is_active=True,
    )
    db_session.add(user)
    unowned_proj = db_models.Project(
        name="Legacy_Guillemot", description="Legacy", created_by=None
    )
    db_session.add(unowned_proj)
    db_session.commit()

    class DummyCtx:
        def __enter__(self):
            return db_session

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(db_service, "get_db", lambda: DummyCtx())
    monkeypatch.setattr(
        db_service,
        "get_current_user_optional",
        lambda _req: {
            "user_id": user.user_id,
            "username": "petro_user",
            "role": "user",
        },
    )

    class DummyUploadFile:
        def __init__(self, filename, data):
            self.filename = filename
            self.file = io.BytesIO(data)

    class DummyReq:
        headers = Headers({})
        cookies = {}

    # Before import, non-admin user cannot see Legacy_Guillemot in list_projects
    resp = Response()
    initial_projects = asyncio.run(db_service.list_projects(DummyReq(), resp))
    assert resp.headers.get("Cache-Control") == "no-store, no-cache, must-revalidate"
    assert [p["name"] for p in initial_projects] == []

    # Case-insensitive duplicate inspection & collision-free suggested_name
    dup_preview = db_service.inspect_handover_package_endpoint(
        file=DummyUploadFile("pkg.zip", zip_bytes)
    )
    assert dup_preview["is_duplicate_name"] is True
    assert dup_preview["suggested_name"] == "Guillemot_Source_Imported"

    # Import 1: into a brand-new project name for non-admin user
    res_new = db_service.commit_handover_package_endpoint(
        request=DummyReq(),
        file=DummyUploadFile("pkg.zip", zip_bytes),
        project_name="Guillemot_New",
        project_id=None,
        overwrite_existing=True,
    )
    assert res_new["success"] is True
    assert res_new["counts"]["wells"] == 1

    # Import 2: merge into existing unowned project (case-insensitive match) for non-admin user
    res_existing = db_service.commit_handover_package_endpoint(
        request=DummyReq(),
        file=DummyUploadFile("pkg.zip", zip_bytes),
        project_name="legacy_guillemot",
        project_id=None,
        overwrite_existing=True,
    )
    assert res_existing["success"] is True
    assert res_existing["project_id"] == unowned_proj.project_id

    # Both newly created AND merged projects must persist in list_projects for the non-admin user!
    after_projects = asyncio.run(db_service.list_projects(DummyReq(), Response()))
    listed_names = {p["name"] for p in after_projects}
    assert "Guillemot_New" in listed_names
    assert "Legacy_Guillemot" in listed_names

    # Verify well CPI config_data (including nested module_params) survived Step 3 header save
    reloaded_proj = Project.load(db_session, project_id=res_new["project_id"])
    reloaded_well = reloaded_proj.get_well("G-1069")
    assert reloaded_well._orm_well.config_data["sw_m"] == pytest.approx(1.85)
    assert reloaded_well._orm_well.config_data["module_params"]["saturation"][
        "rw"
    ] == pytest.approx(0.035)
    # Verify header values are JSON-serializable Python scalars (no raw numpy scalars)
    json.dumps(reloaded_well.header_data)
