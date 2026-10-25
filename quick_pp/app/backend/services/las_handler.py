import asyncio
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from quick_pp.app.backend.schemas.well_metadata import (
    CurveInspectionItem,
    DlisFrameInfo,
    LogCommitFileConfig,
    LogCommitRequest,
    LogInspectFileResult,
    LogInspectResponse,
    WellHeaderMetadata,
    sanitize_json_primitives,
)
from quick_pp.app.backend.services.database import (
    MAX_DEMO_UPLOAD_BYTES,
    _ephemeral_well_headers,
    _ephemeral_wells,
    _merged_data_cache,
    invalidate_well_cache,
)
from quick_pp.app.backend.utils.auth import is_demo_mode
from quick_pp.app.backend.utils.db import get_db
from quick_pp.app.backend.utils.utils import sanitize_filename
from quick_pp.config import Config
from quick_pp.database import objects as db_objects
from quick_pp.las_handler import (
    _las_header_to_dataframe,
    compute_curve_qaqc,
    extract_las_metadata,
    fast_read_las,
    inspect_dlis,
    merge_well_curves_with_dedup,
    mnemonic_mapper,
    read_dlis_frame,
    read_las_file_fast,
    read_las_file_welly,
    splice_and_regularize_runs,
    standardize_curve_units,
)

router = APIRouter(prefix="/las_handler", tags=["File Handler"])
import_logs_router = APIRouter(prefix="/import/logs", tags=["Log Import Engine"])

# Allowed file extensions
ALLOWED_EXTENSIONS = {".LAS", ".las", ".DLIS", ".dlis", ".CSV", ".csv", ".PARQUET", ".parquet"}
STAGING_DIR = Path("uploads/import_staging")

# Registry mapping file_id -> staged file path & metadata
_STAGED_FILES: Dict[str, Dict[str, Any]] = {}


def validate_file_extension(filename: str):
    _, ext = os.path.splitext(filename)
    if ext.lower() not in {e.lower() for e in ALLOWED_EXTENSIONS}:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file extension '{ext}'. Allowed extensions: .las, .dlis, .csv, .parquet",
        )


def save_upload_file(upload_file: UploadFile, destination: Path) -> str:
    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)
    finally:
        upload_file.file.close()
    return destination.name


def unique_id(upload_file: UploadFile) -> str:
    file_bytes = upload_file.file.read()
    upload_file.file.seek(0)
    return sha256(file_bytes).hexdigest()[:8]


def read_las_file(input_path: Path, destination: Path):
    try:
        with open(input_path, "rb") as f:
            try:
                df, _ = read_las_file_fast(f)
            except Exception:
                f.seek(0)
                df, _ = read_las_file_welly(f)
            df.to_parquet(destination, index=False)
        return {"message": f"File {destination.name} processed successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error reading LAS file: {str(e)}"
        ) from e


def _inspect_single_log_file_worker(
    file_id: str, file_path_str: str, original_filename: str
) -> Dict[str, Any]:
    """Top-level worker for inspecting a single `.las` (v2.0/v3.0) or `.dlis` file in <150ms."""
    t0 = time.perf_counter()
    p = Path(file_path_str)
    ext = p.suffix.lower()
    file_size = p.stat().st_size if p.exists() else 0

    if ext == ".dlis":
        dlis_info = inspect_dlis(p)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        res = LogInspectFileResult(
            file_id=file_id,
            filename=original_filename,
            format="DLIS",
            file_size_bytes=file_size,
            metadata=dlis_info["metadata"],
            curves=dlis_info["curves"],
            dlis_frames=dlis_info["frames"],
            has_embedded_core=False,
            has_embedded_tops=False,
            qaqc_warnings=[],
            inspection_ms=elapsed_ms,
        )
        return res.model_dump()

    if ext == ".parquet":
        df_pq = pd.read_parquet(p)
        raw_cols = [str(c) for c in df_pq.columns if str(c).upper() not in ("WELL_NAME", "UWI")]
        col_map = mnemonic_mapper(raw_cols)
        w_name = (
            str(df_pq["WELL_NAME"].dropna().iloc[0])
            if "WELL_NAME" in df_pq.columns and not df_pq["WELL_NAME"].dropna().empty
            else Path(original_filename).stem
        )
        meta = WellHeaderMetadata(well_name=w_name, uwi=w_name)
        curves = [
            CurveInspectionItem(
                raw_mnemonic=c,
                target_mnemonic=col_map.get(c, c.upper()),
                raw_unit="",
                target_unit=Config.VARS.get(col_map.get(c, c.upper()), {}).get("unit", ""),
                include=True,
            )
            for c in raw_cols
        ]
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
        return LogInspectFileResult(
            file_id=file_id,
            filename=original_filename,
            format="Parquet",
            file_size_bytes=file_size,
            metadata=meta,
            curves=curves,
            inspection_ms=elapsed_ms,
        ).model_dump()

    # LAS 2.0 / LAS 3.0 fast header inspection (ignore_data=True)
    las_header, _ = fast_read_las(p, ignore_data=True)
    is_las3 = bool(getattr(las_header, "_is_las3", False))
    core_df = getattr(las_header, "_las3_core_df", pd.DataFrame())
    tops_df = getattr(las_header, "_las3_tops_df", pd.DataFrame())

    meta_dict = extract_las_metadata(
        las_header, fallback_well_name=Path(original_filename).stem
    )
    meta = WellHeaderMetadata(**meta_dict)

    raw_mnemonics = [str(c.mnemonic).strip() for c in las_header.curves]
    mapped_mnemonics = mnemonic_mapper(raw_mnemonics)

    curves_list: List[CurveInspectionItem] = []
    qaqc_warnings: List[str] = []

    for c in las_header.curves:
        raw_m = str(c.mnemonic).strip()
        target_m = mapped_mnemonics.get(raw_m, raw_m.upper())
        raw_u = str(getattr(c, "unit", "") or "").strip()
        descr = str(getattr(c, "descr", "") or "").strip()

        target_u = raw_u
        auto_conv = None
        u_up = raw_u.upper()
        if target_m in ("DEPTH", "TVD", "TVDSS") and u_up in ("FT", "F", "FEET", "0.1 IN", "0.1IN"):
            target_u = "m"
            auto_conv = f"{raw_u} -> m"
        elif target_m in ("NPHI", "PHIT", "PHIE", "CPORE", "CPOR") and u_up in ("%", "PU", "P.U.", "PCT"):
            target_u = "v/v"
            auto_conv = f"{raw_u} -> v/v"
        elif target_m == "RHOB" and u_up in ("KG/M3", "KGM3"):
            target_u = "g/cc"
            auto_conv = f"{raw_u} -> g/cc"
        elif target_m in ("DT", "DTC", "DTS") and u_up in ("US/M", "USEC/M"):
            target_u = "us/ft"
            auto_conv = f"{raw_u} -> us/ft"
        elif not target_u and target_m in Config.VARS:
            target_u = Config.VARS[target_m].get("unit", "")

        curves_list.append(
            CurveInspectionItem(
                raw_mnemonic=raw_m,
                target_mnemonic=target_m,
                raw_unit=raw_u,
                target_unit=target_u,
                description=descr,
                auto_conversion=auto_conv,
                include=True,
            )
        )

    if meta.step is not None and meta.step <= 0:
        qaqc_warnings.append("Non-uniform or zero STEP detected in LAS header; regularization recommended.")

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return LogInspectFileResult(
        file_id=file_id,
        filename=original_filename,
        format="LAS 3.0" if is_las3 else "LAS 2.0",
        file_size_bytes=file_size,
        metadata=meta,
        curves=curves_list,
        dlis_frames=[],
        has_embedded_core=not core_df.empty,
        has_embedded_tops=not tops_df.empty,
        qaqc_warnings=qaqc_warnings,
        inspection_ms=elapsed_ms,
    ).model_dump()


def _parse_single_log_file_worker(task_spec: Dict[str, Any]) -> Dict[str, Any]:
    """Top-level worker function for ProcessPoolExecutor to parse, alias, unit-convert, and QA/QC a log file."""
    file_path = Path(task_spec["file_path"])
    original_filename = task_spec.get("filename") or file_path.name
    target_depth_uom = task_spec.get("target_depth_uom", "m")
    custom_mnemonic_map = task_spec.get("mnemonic_map") or {}
    excluded_curves = {str(c).upper() for c in (task_spec.get("excluded_curves") or [])}
    metadata_overrides = task_spec.get("metadata_overrides") or {}
    selected_frames = task_spec.get("selected_dlis_frames")
    skip_array = bool(task_spec.get("skip_array_channels", True))

    ext = file_path.suffix.lower()
    embedded_core_rows: List[Dict[str, Any]] = []
    embedded_tops_rows: List[Dict[str, Any]] = []

    if ext == ".dlis":
        info = inspect_dlis(file_path)
        frames_to_read = selected_frames or [
            f.frame_name for f in info["frames"] if f.selected
        ]
        if not frames_to_read and info["frames"]:
            frames_to_read = [info["frames"][0].frame_name]

        frame_dfs: List[pd.DataFrame] = []
        meta_dict: Dict[str, Any] = info["metadata"].model_dump()
        std_units: Dict[str, str] = {}
        for fn in frames_to_read:
            f_df, m_dict, s_units = read_dlis_frame(
                file_path,
                logical_file_idx=0,
                frame_name=fn,
                skip_array_channels=skip_array,
                target_depth_uom=target_depth_uom,
                custom_mnemonic_map=custom_mnemonic_map,
            )
            frame_dfs.append(f_df)
            meta_dict.update({k: v for k, v in m_dict.items() if v is not None})
            std_units.update(s_units)

        df = (
            splice_and_regularize_runs(frame_dfs, step=0.1524, conflict_mode="keep_last")
            if len(frame_dfs) > 1
            else (frame_dfs[0] if frame_dfs else pd.DataFrame())
        )
        header_df = pd.DataFrame(
            [
                {"section": "Well", "mnemonic": k.upper(), "unit": "", "value": v, "descr": k}
                for k, v in meta_dict.items()
                if not isinstance(v, (dict, list)) and v is not None
            ]
        )
    elif ext == ".parquet":
        df = pd.read_parquet(file_path)
        df, std_units, _ = standardize_curve_units(
            df,
            curve_units={},
            target_depth_uom=target_depth_uom,
            custom_mnemonic_map=custom_mnemonic_map,
        )
        w_name = (
            str(df["WELL_NAME"].dropna().iloc[0])
            if "WELL_NAME" in df.columns and not df["WELL_NAME"].dropna().empty
            else Path(original_filename).stem
        )
        meta_dict = WellHeaderMetadata(well_name=w_name, uwi=w_name).model_dump()
        header_df = pd.DataFrame(
            [{"section": "Well", "mnemonic": "WELL", "unit": "", "value": w_name, "descr": "WELL"}]
        )
    else:
        # LAS 2.0 / 3.0 fast hybrid parser
        las_header, raw_df = fast_read_las(file_path, ignore_data=False)
        if raw_df is None or raw_df.empty:
            raise ValueError(f"No log data found in {original_filename}")

        meta_dict = extract_las_metadata(
            las_header, fallback_well_name=Path(original_filename).stem
        )
        raw_units = {
            str(c.mnemonic).strip().upper(): str(getattr(c, "unit", "") or "").strip()
            for c in las_header.curves
        }
        # Ensure first curve is mapped to DEPTH if it's the index curve
        first_col = str(raw_df.columns[0]).strip().upper()
        if first_col not in custom_mnemonic_map and Config.resolve_mnemonic_alias(first_col) == "DEPTH":
            custom_mnemonic_map = {"DEPTH": "DEPTH", first_col: "DEPTH", **custom_mnemonic_map}

        df, std_units, _ = standardize_curve_units(
            raw_df,
            curve_units=raw_units,
            target_depth_uom=target_depth_uom,
            custom_mnemonic_map=custom_mnemonic_map,
        )
        header_df = _las_header_to_dataframe(las_header)

        # Extract embedded LAS 3.0 Core_Data and Tops_Data if present
        core_df = getattr(las_header, "_las3_core_df", pd.DataFrame())
        if isinstance(core_df, pd.DataFrame) and not core_df.empty:
            c_cols = {str(c).upper(): c for c in core_df.columns}
            d_col = c_cols.get("DEPTH") or c_cols.get("DEPT") or c_cols.get("MD")
            if d_col:
                for idx_r, r in core_df.iterrows():
                    d_v = pd.to_numeric(r.get(d_col), errors="coerce")
                    if pd.notna(d_v):
                        por_c = c_cols.get("CPOR") or c_cols.get("CPORE") or c_cols.get("POR")
                        perm_c = c_cols.get("CKHG") or c_cols.get("CPERM") or c_cols.get("PERM")
                        por_v = pd.to_numeric(r.get(por_c), errors="coerce") if por_c else np.nan
                        if pd.notna(por_v) and por_v > 1.5:
                            por_v = por_v / 100.0
                        perm_v = pd.to_numeric(r.get(perm_c), errors="coerce") if perm_c else np.nan
                        sname_c = c_cols.get("SAMPLE") or c_cols.get("SAMPLE_ID") or c_cols.get("SAMPLE_NAME")
                        sname = str(r.get(sname_c) if sname_c else f"LAS3-{idx_r + 1}")
                        embedded_core_rows.append(
                            {
                                "sample_name": sname,
                                "depth": float(d_v),
                                "porosity": float(por_v) if pd.notna(por_v) else None,
                                "permeability": float(perm_v) if pd.notna(perm_v) else None,
                            }
                        )

        tops_df = getattr(las_header, "_las3_tops_df", pd.DataFrame())
        if isinstance(tops_df, pd.DataFrame) and not tops_df.empty:
            t_cols = {str(c).upper(): c for c in tops_df.columns}
            n_col = t_cols.get("TOP") or t_cols.get("TOP_NAME") or t_cols.get("NAME") or t_cols.get("ZONE")
            d_col = t_cols.get("DEPTH") or t_cols.get("MD") or t_cols.get("DEPT")
            if n_col and d_col:
                for _, r in tops_df.iterrows():
                    t_name = str(r.get(n_col, "")).strip()
                    d_v = pd.to_numeric(r.get(d_col), errors="coerce")
                    if t_name and pd.notna(d_v):
                        embedded_tops_rows.append({"name": t_name, "depth": float(d_v)})

    # Apply metadata overrides
    for k, v in metadata_overrides.items():
        if v is not None and str(v).strip() != "":
            meta_dict[k] = v

    well_name = (
        str(meta_dict.get("well_name") or Path(original_filename).stem)
        .replace("/", "-")
        .replace(" ", "-")
    )
    uwi = (
        str(meta_dict.get("uwi") or well_name)
        .replace("/", "-")
        .replace(" ", "-")
    )
    meta_dict["well_name"] = well_name
    meta_dict["uwi"] = uwi

    # Drop excluded curves (never drop DEPTH)
    if excluded_curves:
        drop_c = [
            c
            for c in df.columns
            if str(c).upper() in excluded_curves and str(c).upper() != "DEPTH"
        ]
        if drop_c:
            df = df.drop(columns=drop_c, errors="ignore")

    if "WELL_NAME" not in df.columns:
        df.insert(0, "WELL_NAME", well_name)
    else:
        df["WELL_NAME"] = well_name
    if "UWI" not in df.columns:
        df.insert(0, "UWI", uwi)
    else:
        df["UWI"] = uwi

    qaqc_stats = compute_curve_qaqc(df, curve_units=std_units)

    return {
        "file_id": task_spec.get("file_id", ""),
        "filename": original_filename,
        "well_name": well_name,
        "uwi": uwi,
        "metadata": meta_dict,
        "well_data": df,
        "header_df": header_df,
        "units": std_units,
        "qaqc": qaqc_stats,
        "embedded_core": embedded_core_rows,
        "embedded_tops": embedded_tops_rows,
    }


async def _run_tasks_in_pool(worker_fn, task_args_list: List[Any]) -> List[Any]:
    """Execute batch file tasks in parallel using ThreadPoolExecutor.

    Note: Using ThreadPoolExecutor instead of ProcessPoolExecutor avoids forking
    heavy Gunicorn/Uvicorn worker processes (which causes OOM SIGKILL and fork
    deadlocks in Docker) while still achieving multi-core parallelism because
    pandas C-engine (`read_csv`), NumPy, and `dlisio` release the GIL.
    """
    if not task_args_list:
        return []
    max_workers = min(os.cpu_count() or 4, 8, len(task_args_list))
    loop = asyncio.get_running_loop()
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        tasks = [
            loop.run_in_executor(pool, worker_fn, arg) for arg in task_args_list
        ]
        return list(await asyncio.gather(*tasks))


@import_logs_router.post(
    "/inspect",
    response_model=LogInspectResponse,
    summary="Pass 1: Fast header inspection (<150ms) for LAS 2.0/3.0 & DLIS files",
)
async def inspect_log_files(files: List[UploadFile] = File(...)) -> LogInspectResponse:
    """Inspect uploaded `.las` (v2.0/v3.0) and `.dlis` files without parsing full curve arrays.

    Returns extracted `WellHeaderMetadata`, detected curves, suggested standard mnemonics,
    automated unit conversions, QA/QC warnings, and DLIS frames.
    """
    t0 = time.perf_counter()
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    results: List[LogInspectFileResult] = []

    for f in files:
        orig_name = str(f.filename or "upload.las")
        validate_file_extension(orig_name)
        safe_name = sanitize_filename(orig_name)
        if is_demo_mode():
            f.file.seek(0, 2)
            fsize = f.file.tell()
            f.file.seek(0)
            if fsize > MAX_DEMO_UPLOAD_BYTES:
                f.file.close()
                raise HTTPException(
                    status_code=413,
                    detail=f"File '{safe_name}' exceeds 5.0 MB demo upload limit.",
                )

        fid = uuid4().hex[:12]
        staged_path = STAGING_DIR / f"{fid}_{safe_name}"
        save_upload_file(f, staged_path)
        _STAGED_FILES[fid] = {
            "file_id": fid,
            "path": str(staged_path),
            "filename": orig_name,
            "created_at": time.time(),
        }

        try:
            inspected_dict = _inspect_single_log_file_worker(
                fid, str(staged_path), orig_name
            )
            results.append(LogInspectFileResult(**inspected_dict))
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"Failed to inspect '{orig_name}': {e}",
            ) from e

    total_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return LogInspectResponse(files=results, total_inspection_ms=total_ms)


@import_logs_router.post(
    "/commit",
    summary="Pass 2: Parallel C-engine/DLIS ingestion, unit normalization, splicing, and DB/Parquet commit",
)
async def commit_log_files(request: Request) -> Dict[str, Any]:
    """Commit inspected log files with user-confirmed mnemonic/unit mappings & splicing rules.

    Accepts either `application/json` (`LogCommitRequest` referencing staged `file_id`s)
    or `multipart/form-data` (`files` + `commit_request_json`).
    """
    import json

    content_type = (request.headers.get("content-type") or "").lower()
    uploaded_by_name: Dict[str, str] = {}
    if "multipart/form-data" in content_type:
        form = await request.form()
        raw_json = form.get("commit_request_json")
        if not raw_json:
            raise HTTPException(status_code=400, detail="Missing commit_request_json in form data")
        payload = LogCommitRequest(**json.loads(str(raw_json)))
        STAGING_DIR.mkdir(parents=True, exist_ok=True)
        for item in form.getlist("files"):
            if hasattr(item, "filename") and hasattr(item, "file"):
                orig_name = str(item.filename or "upload.las")
                fid = uuid4().hex[:12]
                staged_path = STAGING_DIR / f"{fid}_{sanitize_filename(orig_name)}"
                save_upload_file(item, staged_path)
                _STAGED_FILES[fid] = {
                    "file_id": fid,
                    "path": str(staged_path),
                    "filename": orig_name,
                    "created_at": time.time(),
                }
                uploaded_by_name[orig_name] = fid
    else:
        body_dict = await request.json()
        payload = LogCommitRequest(**body_dict)

    if not payload.files:
        raise HTTPException(status_code=400, detail="No files provided to commit")

    splicing_mode = payload.splice_overlap_mode or payload.splicing_mode or "keep_last"
    task_specs: List[Dict[str, Any]] = []
    for f_cfg in payload.files:
        fid_lookup = f_cfg.file_id or (uploaded_by_name.get(f_cfg.filename or "") if f_cfg.filename else None)
        staged = _STAGED_FILES.get(fid_lookup) if fid_lookup else None
        file_path_str: Optional[str] = None
        orig_filename = f_cfg.filename or f_cfg.file_id or "upload.las"
        if staged and os.path.exists(staged["path"]):
            file_path_str = staged["path"]
            orig_filename = staged.get("filename") or orig_filename
        elif fid_lookup and os.path.exists(fid_lookup):
            file_path_str = fid_lookup
        elif fid_lookup:
            matches = list(STAGING_DIR.glob(f"{fid_lookup}_*"))
            if matches:
                file_path_str = str(matches[0])
        elif f_cfg.filename:
            matches = list(STAGING_DIR.glob(f"*_{sanitize_filename(f_cfg.filename)}"))
            if matches:
                file_path_str = str(matches[-1])

        if not file_path_str or not os.path.exists(file_path_str):
            raise HTTPException(
                status_code=404,
                detail=f"Staged file '{fid_lookup or orig_filename}' not found. Please re-upload and inspect.",
            )

        effective_mnemonic_map = {**f_cfg.curve_mappings, **f_cfg.mnemonic_map}
        effective_unit_map = {**f_cfg.unit_overrides, **f_cfg.unit_map}
        effective_meta = {**(f_cfg.metadata or {}), **(f_cfg.metadata_overrides or {})}
        effective_frames = f_cfg.selected_dlis_frames or (
            [f_cfg.selected_dlis_frame] if f_cfg.selected_dlis_frame else None
        )

        task_specs.append(
            {
                "file_id": fid_lookup or orig_filename,
                "file_path": file_path_str,
                "filename": orig_filename,
                "target_depth_uom": payload.target_depth_uom,
                "mnemonic_map": effective_mnemonic_map,
                "unit_map": effective_unit_map,
                "excluded_curves": f_cfg.excluded_curves,
                "metadata_overrides": effective_meta,
                "selected_dlis_frames": effective_frames,
                "skip_array_channels": f_cfg.skip_array_channels,
                "import_embedded_ancillary": f_cfg.import_embedded_ancillary,
            }
        )

    t0 = time.perf_counter()
    try:
        parsed_results = await _run_tasks_in_pool(_parse_single_log_file_worker, task_specs)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Parallel log ingestion failed: {e}"
        ) from e

    # Group parsed runs by well_name for multi-run splicing & depth grid regularization
    runs_by_well: Dict[str, List[Dict[str, Any]]] = {}
    for res in parsed_results:
        runs_by_well.setdefault(res["well_name"], []).append(res)

    project_id = payload.project_id

    def _commit_wells_to_db_sync() -> List[Dict[str, Any]]:
        committed: List[Dict[str, Any]] = []
        for well_name, well_runs in runs_by_well.items():
            merged_units: Dict[str, str] = {}
            primary_meta: Dict[str, Any] = {}
            for r in well_runs:
                merged_units.update(r.get("units") or {})
                for mk, mv in (r.get("metadata") or {}).items():
                    if mv is not None or mk not in primary_meta:
                        primary_meta[mk] = mv

            batch_appended: List[str] = []
            batch_skipped: List[str] = []
            batch_renamed: Dict[str, str] = {}

            if len(well_runs) == 1:
                first_df = well_runs[0]["well_data"]
                if payload.regularize_grid:
                    spliced_df = splice_and_regularize_runs(
                        [first_df],
                        step=payload.regularize_step,
                        conflict_mode=splicing_mode,
                        interpolate_gaps=payload.interpolate_gaps,
                    )
                else:
                    spliced_df = first_df
                batch_appended = [
                    c
                    for c in spliced_df.columns
                    if str(c).upper() not in ("DEPTH", "DEPT", "WELL_NAME", "UWI")
                ]
            else:
                spliced_df = None
                for idx, r in enumerate(well_runs):
                    run_df = r["well_data"]
                    if payload.regularize_grid:
                        run_df = splice_and_regularize_runs(
                            [run_df],
                            step=payload.regularize_step,
                            conflict_mode=splicing_mode,
                            interpolate_gaps=payload.interpolate_gaps,
                        )
                    spliced_df, step_summary = merge_well_curves_with_dedup(
                        spliced_df, run_df, incoming_units=merged_units
                    )
                    batch_appended.extend(step_summary.get("appended_curves", []))
                    if idx > 0:
                        batch_skipped.extend(step_summary.get("skipped_curves", []))
                        batch_renamed.update(step_summary.get("renamed_curves", {}))

            primary_header_df = well_runs[-1]["header_df"]
            uwi = primary_meta.get("uwi") or well_name
            final_appended = list(batch_appended)
            final_skipped = list(batch_skipped)
            final_renamed = dict(batch_renamed)
            final_curves = [
                c for c in spliced_df.columns if c not in ("WELL_NAME", "UWI")
            ]
            final_row_count = int(len(spliced_df))

            if project_id is not None:
                if is_demo_mode():
                    proj_eph = _ephemeral_wells.setdefault(project_id, {})
                    existing_demo_df = proj_eph.get(well_name)
                    if existing_demo_df is None:
                        try:
                            with get_db() as session:
                                proj = db_objects.Project.load(
                                    session, project_id=project_id
                                )
                                if well_name in set(proj.get_well_names()):
                                    existing_demo_df = proj.get_well_data_optimized(
                                        well_name
                                    )
                        except Exception:
                            existing_demo_df = None

                    if existing_demo_df is not None and not existing_demo_df.empty:
                        spliced_df, demo_summary = merge_well_curves_with_dedup(
                            existing_demo_df,
                            spliced_df,
                            incoming_units=merged_units,
                        )
                        final_appended = list(demo_summary.get("appended_curves", []))
                        final_skipped.extend(demo_summary.get("skipped_curves", []))
                        final_renamed.update(demo_summary.get("renamed_curves", {}))
                        final_curves = [
                            c
                            for c in spliced_df.columns
                            if c not in ("WELL_NAME", "UWI")
                        ]
                        final_row_count = int(len(spliced_df))

                    proj_eph[well_name] = spliced_df.copy()
                    prev_demo_hdr = _ephemeral_well_headers.get(
                        (project_id, well_name), {}
                    )
                    merged_demo_hdr = dict(prev_demo_hdr)
                    for mk, mv in primary_meta.items():
                        if mv is not None or mk not in merged_demo_hdr:
                            merged_demo_hdr[mk] = mv
                    _ephemeral_well_headers[(project_id, well_name)] = merged_demo_hdr
                    _merged_data_cache[f"{project_id}_{well_name}_0.16"] = (
                        time.time(),
                        spliced_df.copy(),
                    )
                else:
                    with get_db() as session:
                        proj = db_objects.Project.load(session, project_id=project_id)
                        existing_names = set(proj.get_well_names())
                        is_existing_well = well_name in existing_names
                        if is_existing_well:
                            well_obj = proj.get_well(well_name)
                        else:
                            well_obj = db_objects.Well(
                                session,
                                project_id=project_id,
                                name=well_name,
                                uwi=uwi,
                                header_data=primary_meta,
                                depth_uom=payload.target_depth_uom,
                            )
                            session.add(well_obj._orm_well)
                            session.flush()

                        db_merge_summary = well_obj.update_data_from_las_parse(
                            spliced_df,
                            primary_header_df,
                            metadata_override=primary_meta,
                        )
                        if is_existing_well and isinstance(db_merge_summary, dict):
                            final_appended = list(
                                db_merge_summary.get("appended_curves", [])
                            )
                            final_skipped.extend(
                                db_merge_summary.get("skipped_curves", [])
                            )
                            final_renamed.update(
                                db_merge_summary.get("renamed_curves", {})
                            )
                            final_curves = ["DEPTH"] + [
                                c.mnemonic for c in well_obj._orm_well.curves
                            ]

                        # Persist any embedded LAS 3.0 core/tops data
                        for r in well_runs:
                            if r.get("embedded_tops"):
                                well_obj.add_formation_tops(r["embedded_tops"])
                            if r.get("embedded_core"):
                                well_obj.replace_rca_samples(r["embedded_core"])

                        session.flush()
                    _ephemeral_well_headers.pop((project_id, well_name), None)
                    invalidate_well_cache(project_id=project_id, well_name=well_name)

            merged_qaqc = compute_curve_qaqc(spliced_df, curve_units=merged_units)

            committed.append(
                {
                    "well_name": well_name,
                    "uwi": uwi,
                    "runs_merged": len(well_runs),
                    "row_count": final_row_count,
                    "curves": final_curves,
                    "appended_curves": final_appended,
                    "skipped_curves": final_skipped,
                    "renamed_curves": final_renamed,
                    "metadata": primary_meta,
                    "qaqc": merged_qaqc,
                }
            )
            # Release large DataFrame references immediately after well commit
            for r in well_runs:
                r["well_data"] = None
            del spliced_df

        return committed

    committed_wells = await asyncio.to_thread(_commit_wells_to_db_sync)

    # Clean up staged files after commit
    for spec in task_specs:
        fp = spec.get("file_path")
        fid = spec.get("file_id")
        if fid and fid in _STAGED_FILES:
            _STAGED_FILES.pop(fid, None)
        if fp and os.path.exists(fp) and "import_staging" in str(fp):
            try:
                os.remove(fp)
            except Exception:
                pass

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return sanitize_json_primitives({
        "status": "success",
        "project_id": project_id,
        "wells_committed": len(committed_wells),
        "wells": committed_wells,
        "wells_imported": committed_wells,
        "elapsed_ms": elapsed_ms,
    })


@router.post(
    "",
    summary="Process LAS Files",
    operation_id="process_las_file_to_parquet",
)
async def process_las_file(files: List[UploadFile] = File(...)):
    """Asynchronously processes a list of uploaded LAS files by saving the raw files and converting them to Parquet."""
    processed_file_paths = []
    for file in files:
        try:
            validate_file_extension(str(file.filename))

            file_name = f"{unique_id(file)}-{file.filename}"
            raw_file_path = Path(f"uploads/las/{file_name}")
            raw_file_path.parent.mkdir(parents=True, exist_ok=True)
            save_upload_file(file, raw_file_path)

            file_stem = file_name.split(".")[0]
            processed_file_path = Path(f"uploads/parquet/{file_stem}.parquet")
            processed_file_path.parent.mkdir(parents=True, exist_ok=True)
            read_las_file(raw_file_path, processed_file_path)
            processed_file_paths.append(processed_file_path)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail="Something went wrong") from e
        finally:
            file.file.close()

    return {
        "message": f"Successfuly converted the LAS files to parquet: {[file.filename for file in files]}",
        "file_paths": processed_file_paths,
    }
