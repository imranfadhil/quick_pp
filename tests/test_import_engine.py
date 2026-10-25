import base64
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from quick_pp.app.backend.router import api_router
from quick_pp.app.backend.services.ancillary import (
    align_and_shift_core_depths,
    compute_survey_minimum_curvature,
    import_ancillary_router,
    parse_tabular_ancillary_file,
    resolve_ancillary_columns,
)
from quick_pp.app.backend.services.las_handler import import_logs_router
from quick_pp.config import Config
from quick_pp.las_handler import (
    compute_curve_qaqc,
    extract_las_metadata,
    fast_read_las,
    inspect_dlis,
    mnemonic_mapper,
    normalize_las3_to_las2_stream,
    read_dlis_frame,
    read_las_file_fast,
    splice_and_regularize_runs,
    standardize_curve_units,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
VOLVE_DIR = REPO_ROOT / "notebooks" / "data" / "01_raw" / "VOLVE"
COSTA_DIR = REPO_ROOT / "notebooks" / "data" / "01_raw" / "COSTA"

app = FastAPI()
app.include_router(api_router)
app.include_router(import_logs_router, prefix="/api/v1", tags=["Log Import Engine v1"])
app.include_router(import_ancillary_router, prefix="/api/v1", tags=["Ancillary Import Engine v1"])

client = TestClient(app)


# ==============================================================================
# Phase 1 Tests: Fast LAS 2.0 Reader, Metadata Extraction, Mnemonic & Unit Engine
# ==============================================================================


def test_fast_read_las_volve():
    las_path = VOLVE_DIR / "15-9-19-A.las"
    assert las_path.exists()
    with open(las_path, "rb") as f:
        header_las, df = fast_read_las(f, filename=las_path.name)

    assert not df.empty
    assert "DEPTH" in df.columns
    assert df["DEPTH"].dtype == np.float64
    for col in df.columns:
        if col != "DEPTH":
            assert df[col].dtype == np.float32

    meta = extract_las_metadata(header_las)
    assert meta.well_name
    assert meta.uwi
    assert meta.depth_unit == "m"
    assert meta.start_depth is not None
    assert meta.stop_depth is not None
    # Verify no numpy scalars leak into raw_well_header or raw_parameters
    for v in meta.raw_well_header.values():
        assert not isinstance(v, np.generic)
    for v in meta.raw_parameters.values():
        assert not isinstance(v, np.generic)


def test_fast_read_las_costa_feet_conversion():
    las_path = COSTA_DIR / "HW-10.las"
    assert las_path.exists()
    with open(las_path, "rb") as f:
        df, meta = read_las_file_fast(f, depth_uom="m")

    assert not df.empty
    assert "DEPTH" in df.columns
    assert meta["depth_unit"] == "m"
    # Original HW-10.las starts at 8440.0 ft (2572.512 m)
    assert np.isclose(float(df["DEPTH"].min()), 2572.512, atol=1e-2)


def test_mnemonic_mapper_and_unit_standardization():
    assert Config.resolve_mnemonic_alias("GR_EDTC") == "GR"
    assert Config.resolve_mnemonic_alias("RDEP") == "RT"
    assert Config.resolve_mnemonic_alias("RHOZ") == "RHOB"
    assert Config.resolve_mnemonic_alias("TNPH") == "NPHI"
    assert Config.resolve_mnemonic_alias("DTCO") == "DTC"

    raw_df = pd.DataFrame(
        {
            "DEPT": [1000.0, 1000.5, 1001.0],
            "GR_ARC": [45.0, 60.0, 75.0],
            "RHOZ": [2350.0, 2400.0, 2450.0],  # kg/m3 -> g/cc
            "TNPH": [18.0, 22.0, 25.0],  # % -> v/v
            "DTCO": [262.467, 295.275, 328.084],  # us/m -> us/ft
        }
    )
    col_map = mnemonic_mapper(raw_df)
    mapped_df = raw_df.rename(columns=col_map)
    assert "DEPTH" in mapped_df.columns
    assert "GR" in mapped_df.columns
    assert "RHOB" in mapped_df.columns
    assert "NPHI" in mapped_df.columns
    assert "DTC" in mapped_df.columns

    units_in = {
        "DEPTH": "ft",
        "GR": "gAPI",
        "RHOB": "kg/m3",
        "NPHI": "%",
        "DTC": "us/m",
    }
    std_df, std_units, conversions = standardize_curve_units(mapped_df, units_in, target_depth_uom="m")
    assert "DEPTH" in conversions
    assert np.isclose(std_df["DEPTH"].iloc[0], 304.8, atol=1e-3)
    assert "RHOB" in conversions
    assert np.isclose(std_df["RHOB"].iloc[0], 2.35, atol=1e-3)
    assert "NPHI" in conversions
    assert np.isclose(std_df["NPHI"].iloc[0], 0.18, atol=1e-3)
    assert "DTC" in conversions
    assert np.isclose(std_df["DTC"].iloc[0], 80.0, atol=1e-2)


# ==============================================================================
# Phase 2 Tests: Excel/CSV Parser, 3-Stage Column Resolver, Minimum Curvature, Core Shift
# ==============================================================================


def test_parse_volve_rca_xls_with_title_and_unit_row():
    xls_path = VOLVE_DIR / "15-9-19-A_RCA.xls"
    assert xls_path.exists()
    raw_bytes = xls_path.read_bytes()

    df, headers, unit_row_map, sheet_names, detected_hdr = parse_tabular_ancillary_file(
        raw_bytes, filename=xls_path.name, data_type="rca"
    )
    assert detected_hdr == 3
    assert len(df) > 100
    assert unit_row_map.get("Depth") == "m"
    mappings, detected = resolve_ancillary_columns(headers, "rca", unit_row=unit_row_map)
    assert detected["depth"] == "Depth"
    assert detected["cpore"] is not None
    assert detected["cperm"] is not None


def test_parse_volve_facies_xlsx():
    xlsx_path = VOLVE_DIR / "15_9-19A_Facies.xlsx"
    assert xlsx_path.exists()
    raw_bytes = xlsx_path.read_bytes()

    df, headers, unit_row_map, sheet_names, detected_hdr = parse_tabular_ancillary_file(
        raw_bytes, filename=xlsx_path.name, data_type="facies"
    )
    assert not df.empty
    assert len(sheet_names) >= 1


def test_minimum_curvature_survey_calculation():
    surveys = [
        {"md": 0.0, "inc": 0.0, "azim": 0.0},
        {"md": 500.0, "inc": 0.0, "azim": 0.0},
        {"md": 1000.0, "inc": 30.0, "azim": 90.0},
        {"md": 1500.0, "inc": 30.0, "azim": 90.0},
    ]
    traj = compute_survey_minimum_curvature(surveys, kb_elevation=25.0)
    assert len(traj) == 4
    assert set(["md", "inc", "azim", "tvd", "tvdss", "dx", "dy", "dls"]).issubset(traj.columns)
    assert np.isclose(traj.loc[1, "tvd"], 500.0, atol=1e-2)
    assert np.isclose(traj.loc[1, "tvdss"], 475.0, atol=1e-2)
    # Inclined section should have TVD < MD and positive Easting (dx > 0)
    assert traj.loc[3, "tvd"] < 1500.0
    assert traj.loc[3, "dx"] > 100.0
    assert traj.loc[2, "dls"] > 0.0


def test_core_depth_shift_and_tolerance_snapping():
    log_depths = np.arange(3800.0, 3810.0, 0.1524)
    core_df = pd.DataFrame(
        {
            "sample_name": ["P1", "P2"],
            "depth": [3801.0, 3805.0],
            "porosity": [0.21, 0.19],
        }
    )
    # Shift core down by +0.50m
    shift_table = [(3800.0, 3800.5), (3810.0, 3810.5)]
    aligned = align_and_shift_core_depths(
        core_df,
        well_log_depths=log_depths,
        depth_col="depth",
        shift_table=shift_table,
        tolerance_m=0.08,
    )
    assert "CORE_DEPTH_RAW" in aligned.columns
    assert np.isclose(aligned["CORE_DEPTH_RAW"].iloc[0], 3801.0)
    # Shifted depth should be near 3801.5 and snapped to nearest log_depths grid point
    nearest_grid_pt = log_depths[np.argmin(np.abs(log_depths - 3801.5))]
    assert np.isclose(aligned["depth"].iloc[0], nearest_grid_pt, atol=1e-4)


# ==============================================================================
# Phase 3 & 4 Tests: Multi-Run Splicing, LAS 3.0 Pre-processor, DLIS & API Endpoints
# ==============================================================================


def test_multi_run_splicing_and_regularization():
    run1 = pd.DataFrame(
        {
            "DEPTH": [100.0, 100.5, 101.0, 101.5],
            "GR": [40.0, 42.0, 44.0, 46.0],
            "RHOB": [2.35, np.nan, np.nan, 2.55],
        }
    )
    run2 = pd.DataFrame(
        {
            "DEPTH": [101.0, 101.5, 102.0, 102.5],
            "GR": [50.0, 52.0, 54.0, 56.0],
        }
    )
    spliced_last = splice_and_regularize_runs([run1, run2], overlap_mode="keep_last", target_step=0.5)
    assert np.isclose(spliced_last["DEPTH"].min(), 100.0)
    assert np.isclose(spliced_last["DEPTH"].max(), 102.5)
    # At depth 101.0, keep_last should take run2 (50.0)
    val_101 = spliced_last.loc[np.isclose(spliced_last["DEPTH"], 101.0), "GR"].iloc[0]
    assert np.isclose(val_101, 50.0, atol=1e-2)
    # Missing data gap in RHOB at 100.5 and 101.0 must remain NaN (not interpolated) by default
    rhob_100_5 = spliced_last.loc[np.isclose(spliced_last["DEPTH"], 100.5), "RHOB"].iloc[0]
    rhob_101_0 = spliced_last.loc[np.isclose(spliced_last["DEPTH"], 101.0), "RHOB"].iloc[0]
    assert np.isnan(rhob_100_5)
    assert np.isnan(rhob_101_0)


def test_las3_normalization_and_parsing():
    las3_text = """~Version
VERS. 3.0 : CWLS LOG ASCII STANDARD - VERSION 3.0
WRAP. NO  : ONE LINE PER DEPTH STEP
DLM . COMMA : DELIMITER CHARACTER
~Well
STRT.M 1000.0 : START DEPTH
STOP.M 1001.0 : STOP DEPTH
STEP.M 0.5    : STEP
NULL.  -999.25: NULL VALUE
WELL.  LAS3-TEST-1 : WELL
UWI .  LAS3-UWI-01 : UWI
~Log_Definition
DEPT.M      : Measured Depth
GR  .GAPI   : Gamma Ray
RHOB.G/C3   : Bulk Density
~Log_Data | Log_Definition
1000.0, 55.2, 2.35
1000.5, 62.1, 2.41
1001.0, 70.4, 2.45
"""
    norm_text = normalize_las3_to_las2_stream(las3_text)
    assert "~Curve" in norm_text
    assert "~Ascii" in norm_text

    df, meta = read_las_file_fast(io.BytesIO(norm_text.encode("utf-8")), depth_uom="m")
    assert len(df) == 3
    assert meta["well_name"] == "LAS3-TEST-1"
    assert np.isclose(df["GR"].iloc[0], 55.2, atol=1e-2)


def test_dlis_inspect_and_read_frame():
    dlis_path = VOLVE_DIR / "seismic" / "input" / "F10-WLC_PETRO_COMPUTED_INPUT_1.DLIS"
    if not dlis_path.exists():
        pytest.skip("Sample Volve DLIS file not present")

    info = inspect_dlis(dlis_path)
    meta = info["metadata"]
    frames = info["frames"]
    assert len(frames) >= 1
    assert meta.well_name
    assert meta.latitude is not None
    assert meta.longitude is not None

    df, frame_meta, units = read_dlis_frame(dlis_path, frame_name=frames[0].frame_name)
    assert not df.empty
    assert "DEPTH" in df.columns
    assert units["DEPTH"] == "m"
    # Verify 0.1 in depth index was converted to meters (~4850 m - 5320 m)
    assert 4500.0 < float(df["DEPTH"].min()) < 5000.0
    assert 5000.0 < float(df["DEPTH"].max()) < 5500.0


def test_api_two_pass_log_inspect_and_commit_with_int64_headers():
    # Create project
    res_proj = client.post("/quick_pp/database/projects", json={"name": "ImportEngineTestProj"})
    assert res_proj.status_code == 200
    pid = res_proj.json()["project_id"]

    # Include integer header items (RUN. 1, API. 1500919001, etc.) that lasio parses as numpy.int64
    las_bytes = """~Version
VERS. 2.0 : CWLS LOG ASCII STANDARD
WRAP. NO  : ONE LINE PER DEPTH STEP
~Well
STRT.M 2000 : START DEPTH
STOP.M 2001 : STOP DEPTH
STEP.M 0.5  : STEP
NULL.  -999.25: NULL VALUE
WELL.  15/9-TEST-A : WELL
UWI .  1500919001 : INTEGER UWI TO TEST NUMPY INT64 SERIALIZATION
API .  1500919001 : API NUMBER
RUN .  1 : RUN NUMBER
EKB .M 25   : KB ELEVATION INTEGER
BHT .DEGC 98.5: BOTTOM HOLE TEMPERATURE
~Parameter
RUN_NO. 1 : INTEGER PARAMETER
BS  .IN 8 : INTEGER BIT SIZE
~Curve
DEPT.M      : Depth
GR_EDTC.GAPI: Gamma Ray
TNPH.%      : Thermal Neutron Porosity
~Ascii
2000.0  45.0  22.0
2000.5  50.0  20.0
2001.0  55.0  18.0
""".encode("utf-8")

    # Pass 1: Inspect via /api/v1/import/logs/inspect
    res_inspect = client.post(
        "/api/v1/import/logs/inspect",
        files=[("files", ("15-9-TEST-A.las", las_bytes, "text/plain"))],
    )
    assert res_inspect.status_code == 200
    inspect_data = res_inspect.json()
    assert len(inspect_data["files"]) == 1
    f_info = inspect_data["files"][0]
    assert f_info["metadata"]["kb_elevation"] == 25.0
    assert f_info["metadata"]["bht"] == 98.5
    tnph_curve = next(c for c in f_info["curves"] if c["raw_mnemonic"] == "TNPH")
    assert tnph_curve["mapped_mnemonic"] == "NPHI"
    assert tnph_curve["unit_conversion_applied"] is True

    # Pass 2: Commit via /api/v1/import/logs/commit (staged file_id JSON body)
    commit_json = {
        "project_id": pid,
        "files": [
            {
                "file_id": f_info["file_id"],
                "filename": "15-9-TEST-A.las",
                "metadata": f_info["metadata"],
                "curve_mappings": {"DEPT": "DEPTH", "GR_EDTC": "GR", "TNPH": "NPHI"},
                "unit_overrides": {"DEPT": "m", "GR_EDTC": "gAPI", "TNPH": "%"},
            }
        ],
        "splice_multi_runs": True,
        "splice_overlap_mode": "keep_last",
    }
    res_commit = client.post("/api/v1/import/logs/commit", json=commit_json)
    assert res_commit.status_code == 200
    commit_out = res_commit.json()
    assert commit_out["status"] == "success"
    assert len(commit_out["wells_imported"]) == 1

    # Also test legacy /quick_pp/database/projects/{pid}/upload-las endpoint for numpy.int64 safety
    res_upload = client.post(
        f"/quick_pp/database/projects/{pid}/upload-las",
        files=[("files", ("15-9-TEST-A.las", las_bytes, "text/plain"))],
    )
    assert res_upload.status_code == 200

    # Now test Ancillary Preview & Commit (RCA CSV on the imported well)
    rca_csv = b"Sample,Depth (m),Porosity (%),Permeability (mD)\nP-1,2000.05,21.5,140.0\nP-2,2000.52,19.0,85.0\n"
    res_anc_prev = client.post(
        "/api/v1/import/ancillary/preview",
        files={"file": ("rca_test.csv", rca_csv, "text/csv")},
        data={"data_type": "rca"},
    )
    assert res_anc_prev.status_code == 200
    prev_json = res_anc_prev.json()
    assert prev_json["detected"]["cpore"] == "Porosity (%)"
    assert prev_json["detected"]["cperm"] == "Permeability (mD)"

    anc_commit_payload = {
        "project_id": pid,
        "well_name": commit_out["wells_imported"][0]["well_name"],
        "data_type": "rca",
        "column_mappings": prev_json["column_mappings"],
        "file_content_base64": base64.b64encode(rca_csv).decode("ascii"),
        "filename": "rca_test.csv",
        "core_snap_tolerance_m": 0.08,
    }
    res_anc_commit = client.post("/api/v1/import/ancillary/commit", json=anc_commit_payload)
    assert res_anc_commit.status_code == 200
    assert res_anc_commit.json()["total_committed"] == 2

    # Test wide-format SCAL (Pc.1..Pc.3, Sw.1..Sw.3) preview & commit on the same well
    scal_wide_csv = (
        b"WELL_NAME,SAMPLE_NAME,DEPTH,Experiment,Pc.1,Pc.2,Pc.3,Sw.1,Sw.2,Sw.3\n"
        b"15-9-TEST-A,1,2000.05,MICP,0.5,2.0,10.0,1.0,0.65,0.25\n"
        b"15-9-TEST-A,2,2000.52,MICP,0.8,4.0,,1.0,0.55,\n"
    )
    res_scal_prev = client.post(
        "/api/v1/import/ancillary/preview",
        files={"file": ("scal_wide.csv", scal_wide_csv, "text/csv")},
        data={"data_type": "scal"},
    )
    assert res_scal_prev.status_code == 200
    scal_prev_json = res_scal_prev.json()
    assert scal_prev_json["detected"]["pc"] == "Pc.1"
    assert scal_prev_json["detected"]["sw"] == "Sw.1"

    scal_commit_payload = {
        "project_id": pid,
        "well_name": commit_out["wells_imported"][0]["well_name"],
        "data_type": "scal",
        "column_mappings": scal_prev_json["column_mappings"],
        "file_content_base64": base64.b64encode(scal_wide_csv).decode("ascii"),
        "filename": "scal_wide.csv",
    }
    res_scal_commit = client.post("/api/v1/import/ancillary/commit", json=scal_commit_payload)
    assert res_scal_commit.status_code == 200
    assert res_scal_commit.json()["total_committed"] == 5


def test_well_header_dms_coords_and_bundle_metadata():
    res_proj = client.post("/quick_pp/database/projects", json={"name": "HeaderMetadataTestProj"})
    assert res_proj.status_code == 200
    pid = res_proj.json()["project_id"]

    las_bytes = """~Version
VERS. 2.0 : CWLS LOG ASCII STANDARD
WRAP. NO  : ONE LINE PER DEPTH STEP
~Well
STRT.FT 100.0 : START DEPTH
STOP.FT 101.0 : STOP DEPTH
STEP.FT 0.5   : STEP
NULL.  -999.25: NULL VALUE
COMP.         : EMPTY COMPANY PLACEHOLDER
OPER.  PHILLIPS PETROLEUM : OPERATOR
WELL.  30/7a-1-TEST : WELL
UWI .  1231-TEST    : UWI
FLD .  JOANNE       : FIELD
LOC .  CENTRAL NORTH SEA : LOCATION
PROV.  UKCS         : PROVINCE
CTRY.  UK           : COUNTRY
CTRY.               : DUPLICATE EMPTY COUNTRY SHOULD NOT OVERWRITE
SRVC.  ATLAS        : SERVICE COMPANY
DATE.  06-JAN-1981  : LOG DATE
LATI.  56 41 57.80 N: LATITUDE SPACE SEPARATED DMS
LONG.  02 15 23.30 E: LONGITUDE SPACE SEPARATED DMS
GDAT.  MSL          : GEODETIC DATUM
EKB .FT 99.0        : KB ELEVATION
EDF .FT 98.0        : DF ELEVATION
EGL .FT -255.0      : GL ELEVATION
~Curve
DEPT.FT : Depth
GR  .GAPI : Gamma Ray
~Ascii
100.0  45.0
100.5  50.0
101.0  55.0
""".encode("utf-8")

    res_inspect = client.post(
        "/api/v1/import/logs/inspect",
        files=[("files", ("30-7a-1-TEST.las", las_bytes, "text/plain"))],
    )
    assert res_inspect.status_code == 200
    f_info = res_inspect.json()["files"][0]
    meta = f_info["metadata"]
    assert meta["company"] == "PHILLIPS PETROLEUM"
    assert meta["country"] == "UK"
    assert meta["field"] == "JOANNE"
    assert meta["service_company"] == "ATLAS"
    assert meta["location"] == "CENTRAL NORTH SEA"
    assert meta["province"] == "UKCS"
    assert meta["log_date"] == "06-JAN-1981"
    assert np.isclose(meta["latitude"], 56.6993889, atol=1e-5)
    assert np.isclose(meta["longitude"], 2.2564722, atol=1e-5)
    assert meta["kb_elev"] == 99.0
    assert meta["df_elev"] == 98.0
    assert meta["gl_elev"] == -255.0

    res_commit = client.post(
        "/api/v1/import/logs/commit",
        json={
            "project_id": pid,
            "files": [
                {
                    "file_id": f_info["file_id"],
                    "filename": "30-7a-1-TEST.las",
                    "metadata": meta,
                }
            ],
        },
    )
    assert res_commit.status_code == 200
    w_name = res_commit.json()["wells"][0]["well_name"]

    res_bundle = client.get(f"/quick_pp/workspace/projects/{pid}/wells/{w_name}/bundle")
    assert res_bundle.status_code == 200
    bundle = res_bundle.json()
    assert "header" in bundle
    hdr = bundle["header"]
    assert hdr["company"] == "PHILLIPS PETROLEUM"
    assert hdr["country"] == "UK"
    assert hdr["field"] == "JOANNE"
    assert hdr["service_company"] == "ATLAS"
    assert hdr["location"] == "CENTRAL NORTH SEA"
    assert hdr["province"] == "UKCS"
    assert np.isclose(hdr["latitude"], 56.6993889, atol=1e-5)
    assert np.isclose(hdr["longitude"], 2.2564722, atol=1e-5)

    # Test updating well header metadata via /config endpoint
    res_cfg = client.post(
        f"/quick_pp/workspace/projects/{pid}/wells/{w_name}/config",
        json={"header": {"company": "Updated Operator", "country": "United Kingdom"}},
    )
    assert res_cfg.status_code == 200
    assert res_cfg.json()["header"]["company"] == "Updated Operator"
    assert res_cfg.json()["header"]["country"] == "United Kingdom"

