import io
import mmap
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lasio
import numpy as np
import pandas as pd
import welly

from quick_pp import logger
from quick_pp.config import Config



class _LASHeaderDataFrame(pd.DataFrame):
    """DataFrame subclass for LAS headers that also supports dict-like metadata key lookup."""

    _metadata = ["_meta_dict"]

    def __getitem__(self, key):
        if isinstance(key, str) and key not in self.columns:
            meta_d = getattr(self, "_meta_dict", None)
            if isinstance(meta_d, dict) and key in meta_d:
                return meta_d[key]
        return super().__getitem__(key)


def read_las_file_fast(file_object, depth_uom=None):
    """Read a LAS file using the fast two-stage hybrid parser (`fast_read_las`)."""
    source = getattr(file_object, "name", None)
    if source and isinstance(source, str) and os.path.exists(source):
        with open(source, "rb") as raw_f:
            head_sample = raw_f.read(32768).decode("utf-8", errors="ignore")
            if len(re.findall(r"^\s*SET\s*\.", head_sample, re.MULTILINE)) > 1:
                raise ValueError("Multi-set LAS file requires mmap reader")
        las_header, df = fast_read_las(source)
    else:
        las_header, df = fast_read_las(file_object)

    if df is None or df.empty:
        raise ValueError("Empty LAS data block")

    well_header = _las_header_to_dataframe(las_header)
    meta_dict = dict(extract_las_metadata(las_header))
    first_col = df.columns[0]
    if first_col != "DEPTH":
        df = df.rename(columns={first_col: "DEPTH"})

    # Convert depth unit if depth_uom requested
    if depth_uom and len(las_header.curves) > 0:
        raw_d_unit = str(getattr(las_header.curves[0], "unit", "") or "").strip().lower()
        target_d_unit = str(depth_uom).strip().lower()
        if raw_d_unit in ("ft", "f", "feet") and target_d_unit in ("m", "meter", "meters"):
            df["DEPTH"] = df["DEPTH"] * 0.3048
            meta_dict["depth_unit"] = "m"
            meta_dict["datum_unit"] = "m"
        elif raw_d_unit in ("m", "meter", "meters") and target_d_unit in ("ft", "f", "feet"):
            df["DEPTH"] = df["DEPTH"] / 0.3048
            meta_dict["depth_unit"] = "ft"
            meta_dict["datum_unit"] = "ft"
        elif target_d_unit in ("m", "ft"):
            meta_dict["depth_unit"] = target_d_unit

    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce").round(4)

    well_name = get_wellname_from_header(well_header)
    if "WELL_NAME" not in df.columns:
        df.insert(0, "WELL_NAME", well_name)
    if "UWI" in well_header["mnemonic"].values and "UWI" not in df.columns:
        uwi = get_uwi_from_header(well_header)
        df.insert(0, "UWI", uwi)

    wrapped_header = _LASHeaderDataFrame(well_header)
    wrapped_header._meta_dict = meta_dict
    return df, wrapped_header


def read_las_files(las_files, depth_uom=None):
    """Read and merge data and headers from multiple LAS files.

    This function iterates through a list of LAS file objects, reads each one,
    and concatenates their curve data and header information into respective
    pandas DataFrames. It prioritizes the fast hybrid reader (`read_las_file_fast`),
    then `welly`, and uses a memory-mapped fallback for multi-set robustness.

    Args:
        las_files (list): A list of file-like objects, each opened in binary mode.
        depth_uom (str, optional): The unit of measurement for the depth index,
                                   to be used by `welly`. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame containing the merged curve data from all LAS files.
            - A DataFrame containing the merged header information from all LAS files.
    """
    merged_data = pd.DataFrame()
    header_data = pd.DataFrame()

    for f in las_files:
        try:
            df, well_header = read_las_file_fast(f, depth_uom)
        except Exception:
            try:
                if hasattr(f, "seek"):
                    f.seek(0)
                df, well_header = read_las_file_welly(f, depth_uom)
            except Exception as e:
                logger.error(f"[read_las_files] Exception for {getattr(f, 'name', f)} | {e} ")
                if hasattr(f, "seek"):
                    f.seek(0)
                df, well_header, _ = read_las_file_mmap(f)
        merged_data = pd.concat([merged_data, df], ignore_index=True)
        header_data = pd.concat([header_data, well_header], ignore_index=True)

    merged_data.reset_index(inplace=True, drop=True)

    return merged_data, header_data


def read_las_file_mmap(file_object, required_sets=["PEP"]):  # noqa
    """Read a single LAS file using a memory-mapped approach for multi-set data.

    This function parses LAS files, particularly those containing multiple data
    sets (e.g., from different logging runs), which standard parsers might
    mishandle. It employs memory-mapping for efficient file access and allows
    for the selective extraction of specified data sets.

    Args:
        file_object (file): A file-like object opened in binary mode to be read.
        required_sets (list, optional): A list of data set identifiers to extract.
                                        Only data sets matching these identifiers
                                        will be processed. Defaults to ['PEP'].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, welly.well.Well]: A tuple containing:
            - A DataFrame containing the curve data.
            - A DataFrame containing the header information.
            - A `welly.well.Well` object representing the well.
    """
    fileno = file_object.fileno()  # identifier for files
    parameter_line_numbers = []
    with mmap.mmap(fileno, length=0, access=mmap.ACCESS_READ) as mmap_obj:
        all_text = mmap_obj.read()
        set_count = len(re.findall(r"\b(SET)\s+", all_text.decode(), re.MULTILINE))
        if set_count > 1:
            well_count = 0
            dataset_count = 0
            line_number = 1
            parameter_count = 0
            mmap_obj.seek(0)
            pointer = 0
            while pointer < mmap_obj.size():
                text = mmap_obj.readline()
                pointer = mmap_obj.tell()  # show current position of pointer

                if b"~Well" in text:
                    well_count += 1
                if b"~Curve" in text:
                    dataset_count += 1
                if b"~P" in text or b"~Tops_Parameter" in text:
                    # Record parameter info in tuple:(parameter_count, parameter_set, pointer location, line number)
                    parameter_line_numbers.append(
                        (parameter_count, "", pointer, line_number)
                    )
                    parameter_count += 1
                if re.compile(r"^\b(SET)\s+").search(text.decode()):
                    parameter_set = re.split(
                        r"[\s+,.:]", text.decode().replace(" ", "")
                    )[1]
                    temp_list_from_tuple = list(
                        parameter_line_numbers[parameter_count - 1]
                    )
                    temp_list_from_tuple[1] = parameter_set
                    temp_list_from_tuple = tuple(temp_list_from_tuple)
                    parameter_line_numbers[parameter_count - 1] = temp_list_from_tuple
                line_number += 1

            # Record well header numbers in tuple: (0, '', pointer location, line number)
            well_header_line_numbers = [
                (0, "", 0, 1),
                (
                    0,
                    "",
                    parameter_line_numbers[0][2] - 1,
                    parameter_line_numbers[0][3] - 1,
                ),
            ]

            mmap_obj.seek(0)  # Reset the pointer location
            curves_df, header_df, welly_object = concat_datasets(
                file_object=mmap_obj.read(),
                header_line_numbers=well_header_line_numbers,
                parameter_line_numbers=parameter_line_numbers,
                required_sets=required_sets,
            )
        else:
            well_count = 1
            counter = 0
            pointer_list = []
            section_dict = {}
            mmap_obj.seek(0)
            pointer = 0
            while pointer < mmap_obj.size():
                text = mmap_obj.readline()
                pointer = mmap_obj.tell()
                if pointer not in pointer_list:
                    pointer_list.append(pointer)
                    counter += 1
                if b"~" in text:
                    section = (
                        text.decode().replace("~", "").rstrip().split(" ")[0].upper()
                    )
                    rename_set = {
                        "V": "VERSION",
                        "W": "WELL",
                        "P": "PARAMETER",
                        "C": "CURVE",
                        "O": "OTHER",
                        "A": "ASCII",
                    }
                    for initial, word in rename_set.items():
                        if section == initial:
                            section = section.replace(initial, word)
                    section_text = text
                    text = mmap_obj.readline()
                    pointer = mmap_obj.tell()
                    if pointer not in pointer_list:
                        pointer_list.append(pointer)
                        counter += 1
                    while b"~" not in text and len(text) > 0:
                        section_text = section_text + text
                        text = mmap_obj.readline()
                        pointer = mmap_obj.tell()
                        if pointer not in pointer_list:
                            pointer_list.append(pointer)
                            counter += 1
                    section_dict[section] = section_text
                    if len(text) > 0:
                        mmap_obj.seek(pointer_list[counter - 2])

            mmap_obj.seek(0)  # Reset the pointer location
            curves_df, header_df, welly_object = extract_dataset(section_dict)

    return curves_df, header_df, welly_object


def read_las_file_welly(file_object, depth_uom=None):
    """Read a LAS file using the welly library.

    This function reads a LAS file into a `welly` object and then processes it
    to extract curve data and header information into pandas DataFrames.

    Args:
        file_object (file): A file-like object, whose `.name` attribute (file path)
                            is used by `welly`.
        depth_uom (str, optional): The unit of measurement for the depth index,
                                   passed to `welly` for index creation. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame containing the processed curve data.
            - A DataFrame containing the header information.
    """
    welly_dataset = welly.las.from_las(file_object.name)
    welly_object = welly.well.Well.from_datasets(welly_dataset, index_units=depth_uom)
    df, well_header = pre_process(welly_object)
    return df, well_header


def pre_process(welly_object):
    """Pre-process a `welly` object to extract and clean data.

    This function takes a `welly.well.Well` object and performs the following
    pre-processing steps:
    1. Converts the depth index into a 'DEPTH' column.
    2. Replaces the LAS-defined NULL value with `np.nan`.
    3. Inserts 'WELL_NAME' and 'UWI' columns at the beginning of the DataFrame.

    Args:
        welly_object (welly.well.Well): The welly object to process.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame containing the processed curve data.
            - A DataFrame containing the header information.
    """
    header_df = getattr(welly_object, "header", pd.DataFrame())

    # Guard against welly objects with no LAS data
    try:
        data_df = welly_object.las[0]
    except Exception as e:
        logger.error(f"[las_handler] pre_process: no las data in welly_object | {e}")
        return pd.DataFrame(), header_df

    data_df.index.rename("DEPTH", inplace=True)
    data_df = data_df.reset_index(drop=False)
    data_df["DEPTH"] = data_df["DEPTH"].round(4)

    # Safely determine NULL value from header, with sensible default
    try:
        null_rows = header_df[header_df.get("mnemonic", pd.Series()) == "NULL"]["value"]
        nullValue = (
            float(null_rows.values[0])
            if len(null_rows) and pd.notna(null_rows.values[0])
            else -999.25
        )
    except Exception:
        nullValue = -999.25

    data_df = data_df.where(data_df >= nullValue, np.nan)
    # Insert well name
    well_name = get_wellname_from_header(header_df)
    if "WELL_NAME" not in data_df.columns:
        data_df.insert(0, "WELL_NAME", well_name)
    # Insert UWI if available
    if "UWI" in header_df["mnemonic"].values:
        uwi = get_uwi_from_header(header_df)
        data_df.insert(0, "UWI", uwi)

    return data_df, header_df


def get_wellname_from_header(header_df):
    """Extract the well name from the LAS header DataFrame.

    Args:
        header_df (pd.DataFrame): The LAS header data.

    Returns:
        str: The well name, with slashes and spaces replaced by hyphens.
    """
    try:
        descr = header_df.get("descr", pd.Series()).fillna("").str.upper()
        mask = (header_df.get("mnemonic", pd.Series()) == "WELL") | (descr == "WELL")
        if mask.any():
            val = header_df.loc[mask, "value"].values[0]
            if pd.isna(val) or str(val).strip() == "":
                return "UNKNOWN_WELL"
            return str(val).replace("/", "-").replace(" ", "-")
    except Exception as e:
        logger.error(f"[las_handler] get_wellname_from_header error | {e}")
    return "UNKNOWN_WELL"


def get_uwi_from_header(header_df):
    """Extract the Unique Well Identifier (UWI) from the LAS header DataFrame.

    If the UWI is not found, it falls back to using the well name.

    Args:
        header_df (pd.DataFrame): The LAS header data.

    Returns:
        str: The UWI or well name, with slashes and spaces replaced by hyphens.
    """
    try:
        descr = header_df.get("descr", pd.Series()).fillna("").str.upper()
        mask = (header_df.get("mnemonic", pd.Series()) == "UWI") | (
            descr == "UNIQUE WELL ID"
        )
        if mask.any():
            val = header_df.loc[mask, "value"].values[0]
            if pd.isna(val) or str(val).strip() == "":
                return get_wellname_from_header(header_df)
            uwi = str(val).replace("/", "-").replace(" ", "-")
            return uwi
    except Exception as e:
        logger.error(f"[las_handler] get_uwi_from_header error | {e}")
    return get_wellname_from_header(header_df)


def get_unit_from_header(header_df, mnemonic):
    """Extract the unit for a specific curve mnemonic from the LAS header.

    Args:
        header_df (pd.DataFrame): The LAS header data.
        mnemonic (str): The curve mnemonic to look for.

    Returns:
        str or None: The unit of the curve, or None if not found.
    """
    try:
        matches = header_df[
            header_df.get("mnemonic", pd.Series()).str.contains(mnemonic, na=False)
        ]
        return matches["unit"].values[0] if len(matches) else None
    except Exception:
        return None


def get_descr_from_header(header_df, mnemonic):
    """Extract the description for a specific curve mnemonic from the LAS header.

    Args:
        header_df (pd.DataFrame): The LAS header data.
        mnemonic (str): The curve mnemonic to look for.

    Returns:
        str or None: The description of the curve, or None if not found.
    """
    try:
        matches = header_df[
            header_df.get("mnemonic", pd.Series()).str.contains(mnemonic, na=False)
        ]
        return matches["descr"].values[0] if len(matches) else None
    except Exception:
        return None


def extract_dataset(section_dict):
    """Extract a single dataset from a dictionary of LAS file sections.

    This function is designed for LAS files with a single data set. It
    reconstructs the LAS file content from a dictionary of its sections,
    reads it using `lasio` and `welly`, and then applies pre-processing.

    Args:
        section_dict (dict): A dictionary where keys are LAS section names
                             (e.g., 'WELL', 'CURVE') and values are their content.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, welly.well.Well]: A tuple containing the processed curve data, header data, and the `welly` object.
    """
    header_bytes = section_dict["WELL"]
    data_bytes = b""
    for k, v in section_dict.items():
        if k in ["PARAMETER", "CURVE", "ASCII"]:
            data_bytes = data_bytes + v

    file_object = header_bytes.decode() + data_bytes.decode()
    las_object = lasio.read(file_object, read_policy=())

    # Fix las_object
    df = las_object.df()
    df = df.apply(pd.to_numeric, errors="coerce")
    las_object.set_data_from_df(df)

    welly_object = welly.Well.from_lasio(las_object)
    well_df = pre_process(welly_object)
    header_df = welly_object.header

    return well_df, header_df, welly_object


def concat_datasets(
    file_object, header_line_numbers, parameter_line_numbers, required_sets=None
):
    """Extract and concatenate specified datasets from a multi-set LAS file.

    This function iterates through the parameter sections identified in a LAS file.
    For each section that matches the `required_sets`, it reconstructs a temporary
    single-set LAS file in memory, reads it, and concatenates the resulting data.

    Args:
        file_object (bytes): The complete content of the LAS file as a bytes object.
        header_line_numbers (list): A list of tuples defining the start and end pointers
                                    for the main well header section.
        parameter_line_numbers (list): A list of tuples, each containing metadata
                                       about a `~Parameter` section, including its
                                       set identifier and pointer location.
        required_sets (list, optional): A list of data set identifiers to extract and
                                        concatenate. Defaults to ['PEP'].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, welly.well.Well]: A tuple containing the concatenated curve data, the last header data, and the last `welly` object.
    """
    required_sets = required_sets or ["PEP"]
    well_df = pd.DataFrame()
    header_df = pd.DataFrame()
    welly_object = welly.Well()
    for i, (_, param_set, pointer, _) in enumerate(parameter_line_numbers):
        # Currently only extracting one dataset: PEP
        if param_set in required_sets:
            well_info = file_object[
                header_line_numbers[0][2] : header_line_numbers[1][2] + 1
            ].decode()
            if i < len(parameter_line_numbers) - 1:
                temp_file_object = file_object[
                    pointer : parameter_line_numbers[i + 1][2]
                ].decode()
            else:
                temp_file_object = file_object[pointer:].decode()
            temp_file_object = well_info + temp_file_object
            las_object = lasio.read(temp_file_object, read_policy=())

            # Fix las_object
            df = las_object.df()
            df = df.apply(pd.to_numeric, errors="coerce")
            las_object.set_data_from_df(df)

            welly_object = welly.Well.from_lasio(las_object)
            temp_well_df = pre_process(welly_object)
            well_df = pd.concat([well_df, temp_well_df], axis=1)
            header_df = welly_object.header

    return well_df, header_df, welly_object


def check_index_consistent(welly_object):
    """Check if the depth index of a welly object is consistent.

    A consistent index means it is monotonically increasing with a constant step.

    Args:
        welly_object (welly.well.Well): The welly object to check.

    Returns:
        bool: True if the index is consistent, False otherwise.
    """
    try:
        index_diff = np.diff(welly_object.las[0].index)
        if all(index_diff == index_diff[0]) and all(index_diff > 0):
            return True
        else:
            return False
    except Exception as e:
        logger.error(f"[las_handler] `check_index_consistent` Error | {e}")
        return False


def build_las_object(
    well_data: pd.DataFrame,
    well_name: str,
    vars_units: Optional[Dict[str, str]] = None,
    vars_descr: Optional[Dict[str, str]] = None,
    uwi: Optional[str] = None,
    depth_uom: str = "m",
    parameters: Optional[Dict[str, Any]] = None,
) -> lasio.LASFile:
    """Build a LAS 2.0 `lasio.LASFile` object from a DataFrame of well data.

    Preserves only numeric curves for LAS 2.0 compatibility, converts any
    categorical `ZONES` column into a numeric `ZONE_ID` curve (recording the
    zone index table in `~Parameter`), and populates `~Well`, `~Curve`, and
    `~Parameter` metadata for seamless import into Techlog, Geolog, IP, and Petrel.
    """
    from .config import Config

    df = well_data.copy()
    # Normalize depth column name
    if "DEPTH" not in df.columns:
        depth_col = next((c for c in df.columns if str(c).upper() == "DEPTH"), None)
        if depth_col:
            df.rename(columns={depth_col: "DEPTH"}, inplace=True)
        elif df.index.name and str(df.index.name).upper() == "DEPTH":
            df = df.reset_index()
            df.rename(columns={df.columns[0]: "DEPTH"}, inplace=True)
        else:
            raise ValueError("well_data must contain a 'DEPTH' column or index.")

    df["DEPTH"] = pd.to_numeric(df["DEPTH"], errors="coerce")
    df = df.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)

    extra_params: Dict[str, Any] = dict(parameters or {})

    # Extract UWI from column if not explicitly passed
    if not uwi and "UWI" in df.columns:
        valid_uwi = df["UWI"].dropna()
        if not valid_uwi.empty:
            uwi = str(valid_uwi.iloc[0])

    # If ZONES column is present as text, create a numeric ZONE_ID curve and record zone names in ~Parameter
    zone_col = next(
        (c for c in df.columns if str(c).upper() in ("ZONES", "ZONE")), None
    )
    if zone_col and "ZONE_ID" not in df.columns:
        zone_series = df[zone_col].dropna().astype(str).str.strip()
        unique_zones = [
            z for z in zone_series.unique() if z and z.lower() not in ("nan", "none")
        ]
        if unique_zones:
            zone_map = {z: idx + 1 for idx, z in enumerate(unique_zones)}
            df["ZONE_ID"] = (
                df[zone_col].astype(str).str.strip().map(zone_map).astype(float)
            )
            for z_name, z_id in zone_map.items():
                extra_params[f"ZONE_{z_id}"] = z_name

    # Drop non-curve identifier/string metadata columns
    drop_cols = {
        "WELL_NAME",
        "UWI",
        "ZONES",
        "ZONE",
        "TOP_NAME",
        "CONTACT_NAME",
        "CORE_SAMPLE_NAME",
        "SAMPLE_NAME",
        "DESCRIPTION",
        "REMARK",
    }
    keep_cols = ["DEPTH"]
    for c in df.columns:
        c_up = str(c).upper()
        if c == "DEPTH" or c_up in drop_cols:
            continue
        # Attempt numeric conversion; keep if at least one non-null numeric value exists or column is numeric dtype
        num_s = pd.to_numeric(df[c], errors="coerce")
        if pd.api.types.is_numeric_dtype(df[c]) or num_s.notna().any():
            df[c] = num_s
            keep_cols.append(c)

    df = df[keep_cols]
    units_map = {**Config.vars_units(df), **(vars_units or {})}
    descr_map = {**Config.vars_descriptions(df), **(vars_descr or {})}

    las_file = lasio.LASFile()
    las_file.well.WELL = str(well_name)
    las_file.well.UWI = str(uwi or well_name)
    las_file.well.NULL = -999.25

    depths = df["DEPTH"].to_numpy(dtype=float)
    if len(depths) > 0:
        las_file.well.STRT = lasio.HeaderItem(
            "STRT", unit=depth_uom, value=round(float(depths[0]), 4), descr="START DEPTH"
        )
        las_file.well.STOP = lasio.HeaderItem(
            "STOP", unit=depth_uom, value=round(float(depths[-1]), 4), descr="STOP DEPTH"
        )
        if len(depths) > 1:
            diffs = np.diff(depths)
            step_val = (
                round(float(diffs[0]), 4)
                if np.allclose(diffs, diffs[0], atol=1e-3)
                else 0.0
            )
        else:
            step_val = 0.0
        las_file.well.STEP = lasio.HeaderItem(
            "STEP", unit=depth_uom, value=step_val, descr="STEP"
        )

    # Add curves
    las_file.append_curve(
        "DEPTH",
        depths,
        unit=depth_uom,
        descr=descr_map.get("DEPTH", "Measured Depth"),
    )
    for col in keep_cols:
        if col == "DEPTH":
            continue
        col_up = str(col).upper()
        arr = df[col].to_numpy(dtype=float)
        unit_str = units_map.get(col) or units_map.get(col_up) or ""
        descr_str = descr_map.get(col) or descr_map.get(col_up) or col_up
        las_file.append_curve(col_up, arr, unit=unit_str, descr=descr_str)

    # Populate ~Parameter section with CPI parameters & metadata
    for p_key, p_val in extra_params.items():
        if p_val is None:
            continue
        mnemonic = re.sub(r"[^A-Z0-9_]", "_", str(p_key).upper()).strip("_")
        if not mnemonic:
            continue
        if isinstance(p_val, (list, tuple)):
            val_str = ", ".join(str(x) for x in p_val if x is not None)
            las_file.params[mnemonic] = lasio.HeaderItem(
                mnemonic, value=val_str, descr=f"quick_pp parameter {p_key}"
            )
        elif isinstance(p_val, dict):
            for sub_k, sub_v in p_val.items():
                if sub_v is not None:
                    sub_mnem = re.sub(
                        r"[^A-Z0-9_]", "_", f"{mnemonic}_{sub_k}".upper()
                    ).strip("_")
                    las_file.params[sub_mnem] = lasio.HeaderItem(
                        sub_mnem,
                        value=sub_v,
                        descr=f"quick_pp {p_key}.{sub_k}",
                    )
        else:
            las_file.params[mnemonic] = lasio.HeaderItem(
                mnemonic, value=p_val, descr=f"quick_pp parameter {p_key}"
            )

    return las_file


def export_to_las_string(
    well_data: pd.DataFrame,
    well_name: str,
    vars_units: Optional[Dict[str, str]] = None,
    vars_descr: Optional[Dict[str, str]] = None,
    uwi: Optional[str] = None,
    depth_uom: str = "m",
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Serialize a DataFrame of well data to a LAS 2.0 formatted string."""
    las_file = build_las_object(
        well_data=well_data,
        well_name=well_name,
        vars_units=vars_units,
        vars_descr=vars_descr,
        uwi=uwi,
        depth_uom=depth_uom,
        parameters=parameters,
    )
    buf = io.StringIO()
    las_file.write(buf, version=2.0)
    return buf.getvalue()


def export_to_las(
    well_data: pd.DataFrame,
    well_name: str,
    folder: str = "",
    vars_units: Optional[Dict[str, str]] = None,
    vars_descr: Optional[Dict[str, str]] = None,
    uwi: Optional[str] = None,
    depth_uom: str = "m",
    parameters: Optional[Dict[str, Any]] = None,
) -> str:
    """Export a DataFrame of well data to a LAS 2.0 file.

    This function takes a pandas DataFrame containing well log data, sets the
    'DEPTH' column as the first curve, and writes the data and optional
    petrophysical parameters to a LAS 2.0 file without mutating `well_data`.

    Args:
        well_data (pd.DataFrame): The DataFrame containing the well log data.
                                  It must include a 'DEPTH' column.
        well_name (str): The name of the well, used for the output filename.
        folder (str, optional): The directory to save the LAS file in. Defaults to ''.
        vars_units (dict, optional): A dictionary mapping curve mnemonics to their units.
                                     If not provided, units are inferred from the configuration.
        vars_descr (dict, optional): Optional dictionary mapping curve mnemonics to descriptions.
        uwi (str, optional): Optional Unique Well Identifier.
        depth_uom (str, optional): Depth unit of measurement ('m' or 'ft'). Defaults to 'm'.
        parameters (dict, optional): Optional dictionary of well evaluation parameters to write to `~Parameter`.

    Returns:
        str: Path to the written LAS file.
    """
    if folder:
        os.makedirs(folder, exist_ok=True)
    las_file = build_las_object(
        well_data=well_data,
        well_name=well_name,
        vars_units=vars_units,
        vars_descr=vars_descr,
        uwi=uwi,
        depth_uom=depth_uom,
        parameters=parameters,
    )
    well_path = os.path.join(folder, f"{well_name}.las") if folder else f"{well_name}.las"
    las_file.write(well_path, version=2.0)
    return well_path


# ==============================================================================
# Next-Gen Import & Ingestion Engine (Fast Hybrid LAS 2.0/3.0, DLIS, QA/QC)
# ==============================================================================

SENTINEL_NULLS = (-999.25, -9999.0, -999.2500, 999.25, -999.0, -9999.25)


class MetadataDict(dict):
    """Dict subclass supporting both dict key and attribute access (plus alias properties)."""

    _ALIASES = {
        "depth_unit": "datum_unit",
        "kb_elevation": "kb_elev",
        "ground_elevation": "gl_elev",
        "dfe_elevation": "df_elev",
        "field_name": "field",
    }

    def __getattr__(self, name: str) -> Any:
        if name in self:
            return self[name]
        mapped = self._ALIASES.get(name)
        if mapped and mapped in self:
            return self[mapped]
        raise AttributeError(f"'MetadataDict' object has no attribute '{name}'")


def _sanitize_json_value(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays and NaN/Inf to JSON-safe Python primitives."""
    if obj is None:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        fval = float(obj)
        return None if (np.isnan(fval) or np.isinf(fval)) else fval
    if isinstance(obj, np.ndarray):
        return [_sanitize_json_value(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _sanitize_json_value(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_json_value(x) for x in obj]
    return obj


def _parse_float_safe(val: Any) -> Optional[float]:
    """Safely parse a numeric value or return None."""
    if val is None:
        return None
    if isinstance(val, (int, float, np.number)):
        fval = float(val)
        return None if (np.isnan(fval) or np.isinf(fval)) else fval
    s = str(val).strip()
    if not s or s.lower() in ("none", "nan", "null", "n/a", "-"):
        return None
    try:
        return float(s)
    except ValueError:
        m = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return None
    return None


_PLACEHOLDER_HEADER_STRS = {
    "",
    "none",
    "null",
    "nan",
    "n/a",
    "na",
    "unknown",
    "undefined",
    "-",
    "--",
    "---",
    ".",
}


def _is_meaningful_header_val(val: Any) -> bool:
    """Return True if a header value is non-empty and not a placeholder or sentinel null."""
    if val is None:
        return False
    if isinstance(val, (float, np.floating)):
        if np.isnan(val) or np.isinf(val) or float(val) in SENTINEL_NULLS:
            return False
        return True
    s = str(val).strip()
    if s.lower() in _PLACEHOLDER_HEADER_STRS:
        return False
    try:
        if float(s) in SENTINEL_NULLS:
            return False
    except ValueError:
        pass
    return True


def _parse_coordinate_value(val: Any) -> Optional[float]:
    """Parse decimal degree or DMS/DM coordinate string (e.g., `058 26' 29.807" N DMS` or `56 41 57.80 N`) to float."""
    if not _is_meaningful_header_val(val):
        return None
    if isinstance(val, (int, float, np.number)):
        fval = float(val)
        return None if (np.isnan(fval) or np.isinf(fval) or fval in SENTINEL_NULLS) else fval
    s = str(val).strip()
    if not s:
        return None
    # Try simple float first if no DMS indicators
    try:
        fval = float(s)
        return None if (np.isnan(fval) or np.isinf(fval) or fval in SENTINEL_NULLS) else fval
    except ValueError:
        pass
    # Extract numbers for DMS / DM / decimal+hemisphere patterns
    nums = [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", s)]
    if not nums:
        return None
    hemi_match = re.search(
        r"(?:^|[\s\d°'\".,:-])([NSEW])(?:\b|(?=\s*DMS)|$)", s.upper()
    )
    hemi = hemi_match.group(1) if hemi_match else None
    sign = -1.0 if (hemi in ("S", "W") or nums[0] < 0) else 1.0
    if len(nums) >= 3:
        deg = abs(nums[0])
        minutes = abs(nums[1]) if len(nums) > 1 else 0.0
        seconds = abs(nums[2]) if len(nums) > 2 else 0.0
        val_deg = deg + (minutes / 60.0) + (seconds / 3600.0)
        return round(sign * val_deg, 7)
    if len(nums) == 2:
        deg = abs(nums[0])
        minutes = abs(nums[1])
        val_deg = deg + (minutes / 60.0)
        return round(sign * val_deg, 7)
    if nums[0] in SENTINEL_NULLS:
        return None
    return round(sign * abs(nums[0]), 7) if hemi else nums[0]


def _las_header_to_dataframe(las_header: lasio.LASFile) -> pd.DataFrame:
    """Convert `lasio.LASFile` header sections into a welly-compatible header DataFrame."""
    rows = []
    for section_name, section in (
        ("Version", getattr(las_header, "version", [])),
        ("Well", getattr(las_header, "well", [])),
        ("Curves", getattr(las_header, "curves", [])),
        ("Parameter", getattr(las_header, "params", [])),
    ):
        for item in section:
            raw_m = getattr(item, "original_mnemonic", None) or getattr(
                item, "mnemonic", ""
            )
            clean_m = (
                str(raw_m).split(":")[0].strip()
                if section_name != "Curves"
                else str(getattr(item, "mnemonic", "")).strip()
            )
            rows.append(
                {
                    "section": section_name,
                    "mnemonic": clean_m,
                    "unit": str(getattr(item, "unit", "") or "").strip(),
                    "value": getattr(item, "value", None),
                    "descr": str(getattr(item, "descr", "") or "").strip(),
                }
            )
    return pd.DataFrame(
        rows, columns=["section", "mnemonic", "unit", "value", "descr"]
    )


def extract_las_metadata(
    las_source: Any,
    fallback_well_name: str = "UNKNOWN_WELL",
    filename: Optional[str] = None,
) -> MetadataDict:
    """Extract standardized `~Well` and `~Parameter` metadata into a `WellHeaderMetadata`-compatible dict.

    Supports `lasio.LASFile`, `welly` header `pd.DataFrame`, or raw dict.
    """
    from quick_pp.app.backend.schemas.well_metadata import WellHeaderMetadata

    if filename and fallback_well_name == "UNKNOWN_WELL":
        fallback_well_name = Path(filename).stem

    well_items: Dict[str, Dict[str, Any]] = {}
    param_items: Dict[str, Dict[str, Any]] = {}
    direct_meta: Dict[str, Any] = {}

    def _put_item(
        target: Dict[str, Dict[str, Any]],
        key: str,
        val: Any,
        unit: str = "",
        descr: str = "",
    ) -> None:
        if not key:
            return
        k_up = key.strip().upper().split(":")[0].strip()
        if not k_up:
            return
        existing = target.get(k_up)
        if (
            existing is not None
            and _is_meaningful_header_val(existing.get("value"))
            and not _is_meaningful_header_val(val)
        ):
            return
        target[k_up] = {
            "value": val,
            "unit": str(unit or "").strip(),
            "descr": str(descr or "").strip(),
        }

    def _populate_from_df(df_hdr: pd.DataFrame) -> None:
        for _, row in df_hdr.iterrows():
            sec = str(row.get("section", "Well")).strip().lower()
            k = str(row.get("mnemonic", "")).strip().upper()
            if not k:
                continue
            val = row.get("value")
            unit = str(row.get("unit") or "").strip()
            descr = str(row.get("descr") or "").strip()
            if "param" in sec:
                _put_item(param_items, k, val, unit, descr)
            elif "curve" not in sec and "vers" not in sec:
                _put_item(well_items, k, val, unit, descr)

    if isinstance(las_source, lasio.LASFile):
        for item in getattr(las_source, "well", []):
            _put_item(
                well_items,
                str(item.mnemonic),
                item.value,
                str(item.unit or ""),
                str(item.descr or ""),
            )
        for item in getattr(las_source, "params", []):
            _put_item(
                param_items,
                str(item.mnemonic),
                item.value,
                str(item.unit or ""),
                str(item.descr or ""),
            )
    elif isinstance(las_source, pd.DataFrame) and not las_source.empty:
        _populate_from_df(las_source)
    elif isinstance(las_source, dict):
        # 1. Columnar DataFrame dict from header_df.to_dict()
        if (
            "mnemonic" in las_source
            and "value" in las_source
            and isinstance(las_source.get("mnemonic"), dict)
            and isinstance(las_source.get("value"), dict)
        ):
            try:
                df_reconstructed = pd.DataFrame(
                    {
                        col: las_source[col]
                        for col in ("section", "mnemonic", "unit", "value", "descr")
                        if col in las_source and isinstance(las_source[col], dict)
                    }
                )
                _populate_from_df(df_reconstructed)
            except Exception:
                pass

        # 2. Nested well_metadata / raw_well_header / raw_parameters
        nested_meta = las_source.get("well_metadata")
        if isinstance(nested_meta, dict):
            direct_meta.update(nested_meta)
            if isinstance(nested_meta.get("raw_well_header"), dict):
                for rk, rv in nested_meta["raw_well_header"].items():
                    _put_item(well_items, str(rk), rv)
            if isinstance(nested_meta.get("raw_parameters"), dict):
                for rk, rv in nested_meta["raw_parameters"].items():
                    _put_item(param_items, str(rk), rv)

        if isinstance(las_source.get("raw_well_header"), dict):
            for rk, rv in las_source["raw_well_header"].items():
                _put_item(well_items, str(rk), rv)
        if isinstance(las_source.get("raw_parameters"), dict):
            for rk, rv in las_source["raw_parameters"].items():
                _put_item(param_items, str(rk), rv)

        # 3. Direct key-value pairs on las_source
        skip_top_keys = {
            "section",
            "mnemonic",
            "unit",
            "value",
            "descr",
            "well_metadata",
            "raw_well_header",
            "raw_parameters",
        }
        for k, v in las_source.items():
            if str(k) in skip_top_keys:
                continue
            direct_meta[str(k)] = v
            if isinstance(v, dict) and "value" in v:
                _put_item(
                    well_items,
                    str(k),
                    v.get("value"),
                    str(v.get("unit") or ""),
                    str(v.get("descr") or ""),
                )
            elif not isinstance(v, (dict, list)):
                _put_item(well_items, str(k), v)

    combined = {**param_items, **well_items}

    def _find_str(aliases: List[str], disallow_zero: bool = False) -> Optional[str]:
        for a in aliases:
            item = well_items.get(a.upper()) or param_items.get(a.upper())
            if item is not None:
                v = item.get("value")
                if _is_meaningful_header_val(v):
                    s = re.split(r"\s{2,}:\s*", str(v).strip())[0].strip()
                    if not _is_meaningful_header_val(s):
                        continue
                    if disallow_zero and s in ("0", "0.0", "00"):
                        continue
                    return s
            # Also check direct_meta (case-insensitive)
            dm_val = direct_meta.get(a) if a in direct_meta else direct_meta.get(a.lower())
            if _is_meaningful_header_val(dm_val):
                s = re.split(r"\s{2,}:\s*", str(dm_val).strip())[0].strip()
                if not _is_meaningful_header_val(s):
                    continue
                if disallow_zero and s in ("0", "0.0", "00"):
                    continue
                return s
        return None

    def _find_float(aliases: List[str]) -> Tuple[Optional[float], Optional[str]]:
        for a in aliases:
            item = well_items.get(a.upper()) or param_items.get(a.upper())
            if item is not None:
                fval = _parse_float_safe(item.get("value"))
                if fval is not None and fval not in SENTINEL_NULLS:
                    return fval, (item.get("unit") or None)
            dm_val = direct_meta.get(a) if a in direct_meta else direct_meta.get(a.lower())
            if dm_val is not None and not isinstance(dm_val, (dict, list)):
                fval = _parse_float_safe(dm_val)
                if fval is not None and fval not in SENTINEL_NULLS:
                    return fval, None
        return None, None

    def _find_coord(aliases: List[str]) -> Optional[float]:
        for a in aliases:
            item = well_items.get(a.upper()) or param_items.get(a.upper())
            if item is not None:
                cval = _parse_coordinate_value(item.get("value"))
                if cval is not None and cval not in SENTINEL_NULLS:
                    return cval
            dm_val = direct_meta.get(a) if a in direct_meta else direct_meta.get(a.lower())
            if dm_val is not None and not isinstance(dm_val, (dict, list)):
                cval = _parse_coordinate_value(dm_val)
                if cval is not None and cval not in SENTINEL_NULLS:
                    return cval
        return None

    raw_well_name = (
        _find_str(
            [
                "WELL",
                "WN",
                "WELL_NAME",
                "WELL-NAME",
                "WELLNAME",
                "HOLE_NAME",
                "ORIGINALWELLNAME",
                "LGWN",
                "SHWN",
            ]
        )
        or fallback_well_name
    )
    clean_well_name = (
        str(raw_well_name).replace("/", "-").replace(" ", "-").strip()
        or fallback_well_name
    )
    raw_uwi = _find_str(["UWI", "API", "WELL_ID", "UWID", "UWI_NO", "UNIQUE_WELL_ID"])
    clean_uwi = (
        str(raw_uwi).replace("/", "-").replace(" ", "-").strip()
        if raw_uwi
        else clean_well_name
    )
    api_val = _find_str(["API", "APIN", "API_NO"])
    lic_val = _find_str(["LIC", "LICENSE", "LICENCE", "PERMIT"])

    kb_elev, kb_unit = _find_float(
        ["KB", "EK", "EREF", "EKB", "RKB", "KB_ELEV", "KBE", "ELEV_KB", "kb_elevation"]
    )
    df_elev, df_unit = _find_float(
        ["DF", "EDF", "DFE", "DF_ELEV", "ELEV_DF", "dfe_elevation"]
    )
    gl_elev, gl_unit = _find_float(
        ["GL", "EGL", "GLE", "SEABED", "GL_ELEV", "ELEV_GL", "ground_elevation"]
    )
    water_depth, wd_unit = _find_float(
        ["WD", "WDEP", "WATER_DEPTH", "WTRD", "SEA_DEPTH"]
    )
    apd, apd_unit = _find_float(["EPD", "APD", "ELZ"])

    start_d, start_unit = _find_float(["STRT", "START", "START_DEPTH"])
    stop_d, _ = _find_float(["STOP", "STOP_DEPTH"])
    step_d, _ = _find_float(["STEP"])
    null_v, _ = _find_float(["NULL", "NULL_VAL"])
    if null_v is None:
        raw_null = combined.get("NULL", {}).get("value")
        null_v = _parse_float_safe(raw_null) if raw_null is not None else -999.25

    datum_unit = (
        kb_unit
        or df_unit
        or gl_unit
        or start_unit
        or _find_str(["datum_unit", "depth_unit"])
        or "m"
    ).strip().lower()
    if datum_unit in ("ft", "f", "feet", "foot"):
        datum_unit = "ft"
    else:
        datum_unit = "m"

    bht, bht_u = _find_float(["BHT", "BHWT", "BLI", "TEMP", "MAX_BHT", "BHTEMP"])
    bht_unit = bht_u or _find_str(["bht_unit"]) or "degC"
    surface_temp, _ = _find_float(
        ["ST", "SUFT", "MST", "SURF_TEMP", "SURFACE_TEMP", "BHT_SURF"]
    )
    bit_size, _ = _find_float(["BS", "BIT", "BITSIZE", "BIT_SIZE"])
    rmf, _ = _find_float(["RMF", "RMFS"])
    rmf_temp, _ = _find_float(["RMFT", "RMF_TEMP", "RMFST"])
    rmc, _ = _find_float(["RMC", "RMCS"])
    mud_density, _ = _find_float(
        ["MW", "FD", "DFD", "MDEN", "MUD_WEIGHT", "MUD_DENSITY", "MUDW", "MUDD"]
    )

    lat_aliases = ["LATI", "LAT", "LATITUDE", "SURF_LAT", "SLAT", "Y_LAT"]
    lon_aliases = [
        "LONG",
        "LON",
        "LONGITUDE",
        "LNG",
        "SURF_LON",
        "SURF_LONG",
        "SLON",
        "X_LON",
    ]
    lat_val = _find_coord(lat_aliases)
    lon_val = _find_coord(lon_aliases)
    lat_raw = _find_str(lat_aliases)
    lon_raw = _find_str(lon_aliases)

    ns_hemi = _find_str(["NS", "LAT_NS"])
    if lat_val is not None and ns_hemi and ns_hemi.strip().upper().startswith("S"):
        lat_val = -abs(lat_val)
    ew_hemi = _find_str(["EW", "LON_EW", "LONG_EW"])
    if lon_val is not None and ew_hemi and ew_hemi.strip().upper().startswith("W"):
        lon_val = -abs(lon_val)

    county_val = _find_str(["CNTY", "COUNTY"])
    state_val = _find_str(["STAT", "STATE"])
    province_val = (
        _find_str(["PROV", "PROVINCE", "BLOCK", "DISTRICT", "REGION"])
        or state_val
        or county_val
    )

    meta = WellHeaderMetadata(
        uwi=clean_uwi,
        well_name=clean_well_name,
        field=_find_str(["FLD", "FIELD", "FN", "FIELD_NAME", "FIELD-NAME", "FIELDNAME"]),
        company=_find_str(
            ["COMP", "COMPANY", "OPERATOR", "OPER", "CN", "COMPANY_NAME"]
        ),
        service_company=_find_str(
            [
                "SRVC",
                "SERVICE",
                "SERVICE_COMPANY",
                "SRV",
                "LOGGING_COMPANY",
                "PRODUCER-NAME",
                "PRODUCER_NAME",
            ]
        ),
        country=_find_str(["CTRY", "COUNTRY", "NATI", "NATION"]),
        province=province_val,
        county=county_val,
        state=state_val,
        location=_find_str(["LOC", "LOCATION", "SURF_LOC", "SURFACE_LOCATION"]),
        api=api_val,
        license=lic_val,
        spud_date=_find_str(
            ["SPUD", "SPUD_DATE", "SPUD-DATE", "SPDA"], disallow_zero=True
        ),
        log_date=_find_str(
            [
                "DATE",
                "LOG_DATE",
                "LOG-DATE",
                "DATT",
                "DATF",
                "SERVICE_DATE",
                "CREATION-TIME",
                "CREA",
                "RUN_DATE",
            ],
            disallow_zero=True,
        ),
        latitude=lat_val,
        longitude=lon_val,
        latitude_raw=lat_raw,
        longitude_raw=lon_raw,
        x_coord=_find_coord(
            [
                "X",
                "XCOORD",
                "X_COORD",
                "EASTING",
                "EAST",
                "LOC_X",
                "SURF_X",
                "UTMX",
                "XWELL",
            ]
        ),
        y_coord=_find_coord(
            [
                "Y",
                "YCOORD",
                "Y_COORD",
                "NORTHING",
                "NORTH",
                "LOC_Y",
                "SURF_Y",
                "UTMY",
                "YWELL",
            ]
        ),
        crs=_find_str(
            [
                "GDAT",
                "HZCS",
                "CRS",
                "PDAT",
                "PD",
                "DATUM",
                "GEODETIC_DATUM",
                "PROJECTION",
                "UTM_ZONE",
                "ZONE_UTM",
            ]
        ),
        kb_elev=kb_elev,
        df_elev=df_elev,
        gl_elev=gl_elev,
        water_depth=water_depth,
        apd=apd,
        datum_unit=datum_unit,
        start_depth=start_d,
        stop_depth=stop_d,
        step=step_d,
        null_val=null_v if null_v is not None else -999.25,
        bht=bht,
        bht_unit=bht_unit,
        surface_temp=surface_temp,
        bit_size=bit_size,
        bs=bit_size,
        rmf=rmf,
        rmf_temp=rmf_temp,
        rmc=rmc,
        mud_density=mud_density,
        mud_weight=mud_density,
        raw_well_header=_sanitize_json_value(
            {
                k: v.get("value")
                for k, v in well_items.items()
                if _is_meaningful_header_val(v.get("value"))
            }
        ),
        raw_parameters=_sanitize_json_value(
            {
                k: v.get("value")
                for k, v in param_items.items()
                if _is_meaningful_header_val(v.get("value"))
            }
        ),
    )
    out_d = _sanitize_json_value(meta.model_dump())
    out_d["depth_unit"] = out_d.get("datum_unit", "m")
    out_d["kb_elevation"] = out_d.get("kb_elev")
    out_d["ground_elevation"] = out_d.get("gl_elev")
    out_d["dfe_elevation"] = out_d.get("df_elev")
    out_d["field_name"] = out_d.get("field")
    return MetadataDict(out_d)


class LAS3NormalizedStream(str):
    """String subclass representing a normalized LAS 2.0 stream that also exposes extracted LAS 3.0 tables via dict indexing."""

    _info: Dict[str, Any]

    def __new__(cls, full_las2_str: str, info: Dict[str, Any]):
        obj = super().__new__(cls, full_las2_str)
        obj._info = info
        return obj

    def __getitem__(self, key):
        if isinstance(key, str) and hasattr(self, "_info") and key in self._info:
            return self._info[key]
        return super().__getitem__(key)

    def get(self, key: str, default: Any = None) -> Any:
        if hasattr(self, "_info"):
            return self._info.get(key, default)
        return default


def normalize_las3_to_las2_stream(
    text_content: str,
) -> LAS3NormalizedStream:
    """Pre-process a LAS 3.0 file stream into LAS 2.0 compatible header + extracted data sections.

    Handles:
    - `VERS. 3.0`
    - `DELIM` or `DLM` (`COMMA`, `TAB`, `SPACE`)
    - `~Log_Definition` / `~Log_Data`
    - Optional `~Core_Definition` / `~Core_Data` and `~Tops_Definition` / `~Tops_Data`
    """
    delim_match = re.search(
        r"^\s*(?:DELIM|DLM)\s*\.\s*(\S+)", text_content, flags=re.IGNORECASE | re.MULTILINE
    )
    delim_code = delim_match.group(1).split(":")[0].strip().upper() if delim_match else "SPACE"
    sep_map = {"COMMA": ",", "TAB": "\t", "SPACE": r"\s+"}
    sep = sep_map.get(delim_code, r"\s+")

    # Split into ~sections
    sections: List[Tuple[str, str]] = []
    current_title = ""
    current_lines: List[str] = []
    for line in text_content.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("~"):
            if current_title:
                sections.append((current_title, "\n".join(current_lines)))
            current_title = stripped
            current_lines = []
        else:
            current_lines.append(line)
    if current_title:
        sections.append((current_title, "\n".join(current_lines)))

    las2_header_blocks: List[str] = []
    log_data_body: Optional[str] = None
    core_def_lines: List[str] = []
    core_data_body: Optional[str] = None
    tops_def_lines: List[str] = []
    tops_data_body: Optional[str] = None

    for title, body in sections:
        t_up = title.upper()
        if t_up.startswith("~V"):
            # Normalize version to 2.0 for lasio header parsing
            fixed_body = re.sub(
                r"(VERS\s*\.\s*)3\.0", r"\g<1>2.0", body, flags=re.IGNORECASE
            )
            las2_header_blocks.append(f"~Version\n{fixed_body}")
        elif t_up.startswith("~W"):
            las2_header_blocks.append(f"~Well\n{body}")
        elif t_up.startswith(("~P", "~LOG_PARAMETER")):
            las2_header_blocks.append(f"~Parameter\n{body}")
        elif t_up.startswith(("~C", "~LOG_DEFINITION")):
            if "~CORE" in t_up:
                core_def_lines = [
                    ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")
                ]
            else:
                las2_header_blocks.append(f"~Curve\n{body}")
        elif "~CORE_DEFINITION" in t_up:
            core_def_lines = [
                ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")
            ]
        elif "~CORE_DATA" in t_up:
            core_data_body = body
        elif "~TOPS_DEFINITION" in t_up:
            tops_def_lines = [
                ln for ln in body.splitlines() if ln.strip() and not ln.strip().startswith("#")
            ]
        elif "~TOPS_DATA" in t_up:
            tops_data_body = body
        elif t_up.startswith(("~A", "~LOG_DATA")):
            if log_data_body is None:
                log_data_body = body

    def _extract_col_names(def_lines: List[str]) -> List[str]:
        cols = []
        for ln in def_lines:
            m = re.match(r"^\s*([^\s.:]+)", ln)
            if m:
                cols.append(m.group(1).strip().upper())
        return cols

    def _parse_section_table(
        def_lines: List[str], data_body: Optional[str]
    ) -> pd.DataFrame:
        if not data_body or not data_body.strip():
            return pd.DataFrame()
        cols = _extract_col_names(def_lines)
        clean_lines = [
            ln
            for ln in data_body.splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if not clean_lines:
            return pd.DataFrame()
        buf = io.StringIO("\n".join(clean_lines))
        try:
            df_sec = pd.read_csv(
                buf,
                sep=sep,
                names=cols if cols else None,
                header=None if cols else 0,
                na_values=list(SENTINEL_NULLS),
                engine="c" if sep in (",", "\t") else "python",
                on_bad_lines="skip",
            )
            return df_sec
        except Exception:
            return pd.DataFrame()

    core_df = _parse_section_table(core_def_lines, core_data_body)
    tops_df = _parse_section_table(tops_def_lines, tops_data_body)

    ascii_rows = []
    for ln in (log_data_body or "").splitlines():
        if not ln.strip() or ln.lstrip().startswith("#"):
            continue
        if sep in (",", "\t"):
            ascii_rows.append(" ".join(tok.strip() for tok in ln.split(sep)))
        else:
            ascii_rows.append(ln)

    header_str = "\n".join(las2_header_blocks) + "\n~Ascii\n"
    full_las2_str = header_str + "\n".join(ascii_rows) + ("\n" if ascii_rows else "")
    info = {
        "las2_header_str": header_str,
        "log_data_body": log_data_body or "",
        "delimiter": sep,
        "delim_code": delim_code,
        "core_df": core_df,
        "tops_df": tops_df,
    }
    return LAS3NormalizedStream(full_las2_str, info)


def fast_read_las(
    file_path_or_buffer: Union[str, bytes, Path, io.IOBase],
    ignore_data: bool = False,
    filename: Optional[str] = None,
) -> Tuple[lasio.LASFile, Optional[pd.DataFrame]]:
    """Read LAS 2.0 / 3.0 headers via lasio (`ignore_data=True`) in <5ms,
    then parse the `~A` / `~Log_Data` block using pandas C-engine (10x-15x faster)
    when `WRAP == 'NO'`. Falls back to `lasio` for wrapped files.

    Uses `float64` for `DEPTH` (index 0) and `float32` for all petrophysical curves.
    """
    raw_bytes: Optional[bytes] = None
    file_path: Optional[Path] = None

    if isinstance(file_path_or_buffer, (str, Path)):
        p_str = str(file_path_or_buffer)
        if "\n" not in p_str and os.path.exists(p_str):
            file_path = Path(p_str)
            with open(file_path, "rb") as f:
                raw_bytes = f.read()
        else:
            raw_bytes = p_str.encode("utf-8", errors="ignore")
    elif isinstance(file_path_or_buffer, bytes):
        raw_bytes = file_path_or_buffer
    elif hasattr(file_path_or_buffer, "read"):
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        content = file_path_or_buffer.read()
        if hasattr(file_path_or_buffer, "seek"):
            file_path_or_buffer.seek(0)
        raw_bytes = (
            content
            if isinstance(content, bytes)
            else str(content).encode("utf-8", errors="ignore")
        )

    if raw_bytes is None:
        raise ValueError("Unsupported file_path_or_buffer provided to fast_read_las")

    # Quick check for LAS 3.0 header or ~Log_Data section
    header_preview = raw_bytes[:16384].decode("utf-8", errors="ignore")
    is_las3 = bool(
        re.search(r"VERS\s*\.\s*3\.0", header_preview, flags=re.IGNORECASE)
        or re.search(
            r"^\s*~(Log_Data|Log_Definition|Core_Data|Tops_Data)",
            header_preview,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )

    if is_las3:
        full_text = raw_bytes.decode("utf-8", errors="ignore")
        las3_info = normalize_las3_to_las2_stream(full_text)
        las_header = lasio.read(io.StringIO(las3_info["las2_header_str"]), ignore_data=True)
        las_header._las3_core_df = las3_info["core_df"]
        las_header._las3_tops_df = las3_info["tops_df"]
        las_header._is_las3 = True
        if ignore_data:
            return las_header, None

        curve_names = [c.mnemonic for c in las_header.curves]
        null_item = getattr(las_header.well, "NULL", None)
        null_val = _parse_float_safe(getattr(null_item, "value", -999.25)) or -999.25
        dtypes = {
            name: (np.float64 if i == 0 else np.float32)
            for i, name in enumerate(curve_names)
        }
        clean_lines = [
            ln
            for ln in las3_info["log_data_body"].splitlines()
            if ln.strip() and not ln.lstrip().startswith("#")
        ]
        if clean_lines and curve_names:
            sep = las3_info["delimiter"]
            df = pd.read_csv(
                io.StringIO("\n".join(clean_lines)),
                sep=sep,
                names=curve_names,
                dtype=dtypes,
                na_values=list({null_val, *SENTINEL_NULLS}),
                engine="c",
                on_bad_lines="warn",
            )
            return las_header, df
        return las_header, pd.DataFrame(columns=curve_names)

    # Standard LAS 2.0 fast path: slice only header bytes before ~A for ultra-fast header parse (<5ms)
    data_start_offset: Optional[int] = None
    data_line_match = re.search(rb"(?:^|[\r\n])\s*~[Aa][^\r\n]*[\r\n]+", raw_bytes)
    if data_line_match:
        header_only_bytes = raw_bytes[: data_line_match.start()]
        data_start_offset = data_line_match.end()
    else:
        header_only_bytes = raw_bytes

    header_str = header_only_bytes.decode("utf-8", errors="ignore")
    las_header = lasio.read(io.StringIO(header_str), ignore_data=True)
    las_header._is_las3 = False
    if ignore_data:
        return las_header, None

    wrap_item = getattr(las_header.version, "WRAP", None)
    wrap_mode = str(getattr(wrap_item, "value", "NO") or "NO").strip().upper()
    curve_names = [c.mnemonic for c in las_header.curves]
    null_item = getattr(las_header.well, "NULL", None)
    null_val = _parse_float_safe(getattr(null_item, "value", -999.25)) or -999.25

    # Fast C-engine path for unwrapped LAS files (>95% of files)
    if wrap_mode == "NO" and data_start_offset is not None and curve_names:
        try:
            dtypes = {
                name: (np.float64 if i == 0 else np.float32)
                for i, name in enumerate(curve_names)
            }
            data_bio = io.BytesIO(raw_bytes[data_start_offset:])
            df = pd.read_csv(
                data_bio,
                sep=r"\s+",
                names=curve_names,
                dtype=dtypes,
                na_values=list({null_val, *SENTINEL_NULLS}),
                 comment="#",
                engine="c",
                on_bad_lines="warn",
            )
            if not df.empty and len(df.columns) == len(curve_names):
                if df.columns[0].upper() in ("DEPT", "MD", "DEPTH") and df.columns[0] != "DEPTH":
                    df = df.rename(columns={df.columns[0]: "DEPTH"})
                return las_header, df
        except Exception as e:
            logger.debug(f"[fast_read_las] C-engine fallback to lasio due to: {e}")

    # Fallback for wrapped files or non-standard ASCII blocks
    full_str = raw_bytes.decode("utf-8", errors="ignore")
    las_full = lasio.read(io.StringIO(full_str))
    df = las_full.df().reset_index()
    for i, col in enumerate(df.columns):
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].replace(list({null_val, *SENTINEL_NULLS}), np.nan)
            df[col] = df[col].astype(np.float64 if i == 0 else np.float32)
    if not df.empty and df.columns[0].upper() in ("DEPT", "MD", "DEPTH") and df.columns[0] != "DEPTH":
        df = df.rename(columns={df.columns[0]: "DEPTH"})
    return las_full, df


def mnemonic_mapper(
    curves_or_columns: Union[List[str], pd.DataFrame],
    custom_mapping: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """Map raw vendor curve mnemonics to standard `quick_pp` mnemonics.

    Preserves unique mappings and avoids colliding with an already-present target mnemonic.
    """
    if isinstance(curves_or_columns, pd.DataFrame):
        raw_cols = [str(c) for c in curves_or_columns.columns]
    else:
        raw_cols = [str(c) for c in curves_or_columns]

    custom_up = {
        str(k).strip().upper(): str(v).strip().upper()
        for k, v in (custom_mapping or {}).items()
        if k and v
    }
    existing_up = {c.strip().upper() for c in raw_cols}
    assigned_targets: set[str] = set()
    mapping: Dict[str, str] = {}

    # First pass: exact canonical names keep their identity
    for col in raw_cols:
        col_up = col.strip().upper()
        if col_up in custom_up:
            target = custom_up[col_up]
            mapping[col] = target
            assigned_targets.add(target)
        elif col_up in Config.MNEMONIC_ALIASES:
            mapping[col] = col_up
            assigned_targets.add(col_up)

    # Second pass: resolve vendor aliases if target is not already claimed
    for col in raw_cols:
        if col in mapping:
            continue
        col_up = col.strip().upper()
        resolved = Config.resolve_mnemonic_alias(col_up)
        if resolved != col_up and resolved not in assigned_targets and resolved not in existing_up:
            mapping[col] = resolved
            assigned_targets.add(resolved)
        else:
            mapping[col] = col_up
            assigned_targets.add(col_up)

    return mapping


def standardize_curve_units(
    df: pd.DataFrame,
    curve_units: Optional[Dict[str, str]] = None,
    target_depth_uom: str = "m",
    custom_mnemonic_map: Optional[Dict[str, str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, Dict[str, Any]]]:
    """Apply mnemonic aliasing and automated petrophysical unit conversions.

    Standardizes:
    - `DEPTH`, `TVD`, `TVDSS`: `FT`/`F` -> `M` (`* 0.3048`), `0.1 IN` -> `M` (`* 0.00254`), `IN` -> `M` (`* 0.0254`)
    - Porosity (`NPHI`, `PHIT`, `PHIE`, `CPOR`, `CPORE`): `%`, `PU`, `P.U.` or `nanpercentile(x, 95) > 1.5` -> `/ 100.0` (`v/v`)
    - Bulk Density (`RHOB`): `KG/M3`, `KGM3` or `nanmedian(x) > 500` -> `/ 1000.0` (`g/cc`)
    - Sonic (`DT`, `DTC`, `DTS`): `US/M`, `USEC/M` -> `* 0.3048` (`us/ft`)
    """
    if df is None or df.empty:
        return pd.DataFrame(), {}, {}

    out_df = df.copy()
    units_in = {str(k).strip().upper(): str(v or "").strip() for k, v in (curve_units or {}).items()}
    col_map = mnemonic_mapper(out_df, custom_mapping=custom_mnemonic_map)
    out_df.rename(columns=col_map, inplace=True)

    # Map units to renamed columns
    std_units: Dict[str, str] = {}
    for orig_col, new_col in col_map.items():
        std_units[new_col] = units_in.get(orig_col.strip().upper(), units_in.get(new_col, ""))

    conversions: Dict[str, Dict[str, Any]] = {}
    target_uom_clean = (target_depth_uom or "m").strip().lower()

    # Normalize sentinel nulls across all numeric columns first
    for col in out_df.columns:
        if pd.api.types.is_numeric_dtype(out_df[col]):
            out_df[col] = out_df[col].replace(list(SENTINEL_NULLS), np.nan)

    # 1. Depth conversion
    for dcol in ("DEPTH", "TVD", "TVDSS"):
        if dcol not in out_df.columns:
            continue
        raw_u = std_units.get(dcol, "").strip().upper()
        s = pd.to_numeric(out_df[dcol], errors="coerce")
        factor = 1.0
        target_u = "m" if target_uom_clean.startswith("m") else "ft"
        if target_u == "m":
            if raw_u in ("FT", "F", "FEET", "FOOT"):
                factor = 0.3048
            elif raw_u in ("0.1 IN", "0.1IN", "0.1 IN."):
                factor = 0.00254
            elif raw_u in ("IN", "IN.", "INCH", "INCHES"):
                factor = 0.0254
            elif raw_u in ("CM",):
                factor = 0.01
        elif target_u == "ft":
            if raw_u in ("M", "METER", "METERS", "METRE"):
                factor = 1.0 / 0.3048
            elif raw_u in ("0.1 IN", "0.1IN"):
                factor = 1.0 / 120.0

        if abs(factor - 1.0) > 1e-9:
            out_df[dcol] = (s * factor).astype(np.float64)
            conversions[dcol] = {
                "curve": dcol,
                "from_unit": raw_u or "unknown",
                "to_unit": target_u,
                "factor": factor,
                "badge": f"{dcol}: {raw_u} -> {target_u}",
            }
            std_units[dcol] = target_u
        else:
            out_df[dcol] = s.astype(np.float64)
            std_units[dcol] = std_units.get(dcol) or target_u

    # 2. Porosity conversion (% / PU -> v/v)
    for pcol in ("NPHI", "PHIT", "PHIE", "CPOR", "CPORE", "CORE_POR"):
        if pcol not in out_df.columns:
            continue
        raw_u = std_units.get(pcol, "").strip().upper()
        s = pd.to_numeric(out_df[pcol], errors="coerce")
        valid = s.dropna()
        if valid.empty:
            continue
        p95 = float(np.nanpercentile(valid, 95))
        is_pct_unit = raw_u in ("%", "PU", "P.U.", "PERCENT", "PCT")
        if is_pct_unit or p95 > 1.5:
            out_df[pcol] = (s / 100.0).astype(np.float32)
            conversions[pcol] = {
                "curve": pcol,
                "from_unit": raw_u or "%",
                "to_unit": "v/v",
                "factor": 0.01,
                "badge": f"{pcol}: {raw_u or '%'} -> v/v",
            }
            std_units[pcol] = "v/v"
        else:
            out_df[pcol] = s.astype(np.float32)
            std_units[pcol] = "v/v"

    # 3. Bulk Density conversion (kg/m3 -> g/cc)
    if "RHOB" in out_df.columns:
        raw_u = std_units.get("RHOB", "").strip().upper()
        s = pd.to_numeric(out_df["RHOB"], errors="coerce")
        valid = s.dropna()
        if not valid.empty:
            med = float(np.nanmedian(valid))
            is_kgm3 = raw_u in ("KG/M3", "KGM3", "KG/M^3", "KG/CBM")
            if is_kgm3 or med > 500.0:
                out_df["RHOB"] = (s / 1000.0).astype(np.float32)
                conversions["RHOB"] = {
                    "curve": "RHOB",
                    "from_unit": raw_u or "kg/m3",
                    "to_unit": "g/cc",
                    "factor": 0.001,
                    "badge": f"RHOB: {raw_u or 'kg/m3'} -> g/cc",
                }
            else:
                out_df["RHOB"] = s.astype(np.float32)
            std_units["RHOB"] = "g/cc"

    # 4. Sonic conversion (us/m -> us/ft)
    for scol in ("DT", "DTC", "DTS"):
        if scol not in out_df.columns:
            continue
        raw_u = std_units.get(scol, "").strip().upper()
        s = pd.to_numeric(out_df[scol], errors="coerce")
        if raw_u in ("US/M", "USEC/M", "US/M.", "USEC/M."):
            out_df[scol] = (s * 0.3048).astype(np.float32)
            conversions[scol] = {
                "curve": scol,
                "from_unit": raw_u,
                "to_unit": "us/ft",
                "factor": 0.3048,
                "badge": f"{scol}: {raw_u} -> us/ft",
            }
            std_units[scol] = "us/ft"
        else:
            out_df[scol] = s.astype(np.float32)
            std_units[scol] = std_units.get(scol) or "us/ft"

    # Downcast remaining numeric petrophysical curves to float32 (keep DEPTH, TVD, TVDSS as float64)
    for col in out_df.columns:
        if col in ("DEPTH", "TVD", "TVDSS", "WELL_NAME", "UWI"):
            continue
        if pd.api.types.is_numeric_dtype(out_df[col]):
            out_df[col] = out_df[col].astype(np.float32)

    return out_df, std_units, conversions


def compute_curve_qaqc(
    df: pd.DataFrame, curve_units: Optional[Dict[str, str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Run Pre-Import QA/QC Health Check on all curves in a DataFrame.

    Checks:
    - Sentinel null normalization (`-999.25`, `-9999`, `999.25`, `-999`)
    - Flatlined curves (`std(x) == 0` over `>10 m`)
    - Out-of-bounds physics (`RHOB < 1.0` or `> 3.5 g/cc`, `RT <= 0 ohmm`, `GR < 0 gAPI`)
    - Completeness % (non-null percentage across the logged interval)
    """
    if df is None or df.empty:
        return {}

    total_rows = len(df)
    depths = (
        pd.to_numeric(df["DEPTH"], errors="coerce").to_numpy(dtype=float)
        if "DEPTH" in df.columns
        else np.arange(total_rows, dtype=float) * 0.1524
    )
    step_m = 0.1524
    if len(depths) > 1:
        diffs = np.abs(np.diff(depths[np.isfinite(depths)]))
        if len(diffs) > 0 and np.median(diffs) > 0:
            step_m = float(np.median(diffs))
    window_pts = max(int(round(10.0 / max(step_m, 0.01))), 5)

    physics_bounds = {
        "RHOB": (1.0, 3.5, "g/cc"),
        "RT": (1e-6, 1e6, "ohmm"),
        "GR": (0.0, 500.0, "gAPI"),
        "NPHI": (-0.15, 1.0, "v/v"),
        "DT": (20.0, 260.0, "us/ft"),
        "DTC": (20.0, 260.0, "us/ft"),
    }

    diagnostics: Dict[str, Dict[str, Any]] = {}
    for col in df.columns:
        if col in ("WELL_NAME", "UWI"):
            continue
        s = pd.to_numeric(df[col], errors="coerce").replace(list(SENTINEL_NULLS), np.nan)
        valid = s.dropna()
        valid_count = int(len(valid))
        null_count = int(total_rows - valid_count)
        completeness = round((valid_count / total_rows) * 100.0, 2) if total_rows > 0 else 0.0

        warnings: List[str] = []
        min_v = round(float(valid.min()), 4) if valid_count > 0 else None
        max_v = round(float(valid.max()), 4) if valid_count > 0 else None

        # Flatline check (>10m constant value) on continuous logs (skip binary flags)
        col_up = str(col).upper()
        if (
            col_up != "DEPTH"
            and "FLAG" not in col_up
            and valid_count >= window_pts
        ):
            roll_std = s.rolling(window=window_pts, min_periods=window_pts).std()
            flat_count = int((roll_std == 0.0).sum())
            if flat_count > 0 or (valid_count > 10 and float(valid.std()) == 0.0):
                warnings.append(f"Flatlined interval detected (>10m constant value)")

        # Physics bounds check
        if col_up in physics_bounds and valid_count > 0:
            lo, hi, u_lbl = physics_bounds[col_up]
            if col_up == "RT":
                oob = int((valid <= 0.0).sum())
                if oob > 0:
                    warnings.append(f"Out-of-bounds physics: {oob} pts with RT <= 0 {u_lbl}")
            else:
                oob = int(((valid < lo) | (valid > hi)).sum())
                if oob > 0:
                    warnings.append(
                        f"Out-of-bounds physics: {oob} pts outside [{lo}, {hi}] {u_lbl}"
                    )

        diagnostics[col] = {
            "mnemonic": col,
            "unit": (curve_units or {}).get(col, ""),
            "min": min_v,
            "max": max_v,
            "null_count": null_count,
            "completeness_pct": completeness,
            "warnings": warnings,
        }

    return diagnostics


def splice_and_regularize_runs(
    runs: List[pd.DataFrame],
    step: float = 0.1524,
    conflict_mode: str = "keep_last",
    overlap_mode: Optional[str] = None,
    target_step: Optional[float] = None,
    interpolate_gaps: bool = False,
) -> pd.DataFrame:
    """Splice multiple log runs onto a unified regular depth grid (`step = 0.1524 m`).

    Args:
        runs: List of DataFrames, each containing a `DEPTH` column.
        step: Regular depth step in meters (default `0.1524`).
        conflict_mode: Overlap conflict resolution (`keep_last`, `keep_first`, or `mean`).
        interpolate_gaps: Whether to linearly interpolate across missing (`NaN`) curve intervals.
            Defaults to `False` so missing log intervals remain `NaN`.
    """
    if overlap_mode is not None:
        conflict_mode = overlap_mode
    if target_step is not None:
        step = target_step
    valid_runs = [
        r.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)
        for r in runs
        if r is not None and not r.empty and "DEPTH" in r.columns
    ]
    if not valid_runs:
        return pd.DataFrame()
    if len(valid_runs) == 1 and step <= 0:
        return valid_runs[0].copy()

    min_d = min(float(r["DEPTH"].min()) for r in valid_runs)
    max_d = max(float(r["DEPTH"].max()) for r in valid_runs)
    step_val = float(step) if step and step > 0 else 0.1524
    n_steps = int(np.floor((max_d - min_d) / step_val)) + 1
    target_grid = np.round(min_d + np.arange(n_steps, dtype=np.float64) * step_val, 4)

    all_curves: List[str] = []
    meta_Values: Dict[str, Any] = {}
    for r in valid_runs:
        for c in r.columns:
            if c == "DEPTH":
                continue
            if c in ("WELL_NAME", "UWI"):
                valid_m = r[c].dropna()
                if not valid_m.empty:
                    meta_Values[c] = valid_m.iloc[-1]
                continue
            if c not in all_curves:
                all_curves.append(c)

    # Resample each run onto the target depth grid within its logged interval
    interp_runs: List[pd.DataFrame] = []
    for r in valid_runs:
        r_depths = pd.to_numeric(r["DEPTH"], errors="coerce").to_numpy(dtype=np.float64)
        finite_d = r_depths[np.isfinite(r_depths)]
        if len(finite_d) < 2:
            continue
        d_diffs = np.diff(finite_d)
        pos_diffs = d_diffs[d_diffs > 1e-6]
        native_step = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else step_val
        max_valid_span = max(step_val, native_step) * 1.75
        snap_tol = max(step_val, native_step) * 0.55

        r_min, r_max = float(finite_d[0]), float(finite_d[-1])
        in_range = (target_grid >= r_min - 1e-6) & (target_grid <= r_max + 1e-6)
        run_grid_df = pd.DataFrame({"DEPTH": target_grid})
        for c in all_curves:
            if c not in r.columns or not pd.api.types.is_numeric_dtype(r[c]):
                run_grid_df[c] = np.nan
                continue
            vals = pd.to_numeric(r[c], errors="coerce").to_numpy(dtype=np.float64)
            valid_mask = np.isfinite(r_depths) & np.isfinite(vals)
            if valid_mask.sum() < 2:
                run_grid_df[c] = np.nan
                continue
            v_d = r_depths[valid_mask]
            v_y = vals[valid_mask]
            interp_y = np.full(len(target_grid), np.nan, dtype=np.float32)
            local_mask = in_range & (target_grid >= v_d[0]) & (target_grid <= v_d[-1])
            if np.any(local_mask):
                tg_local = target_grid[local_mask]
                y_local = np.interp(tg_local, v_d, v_y).astype(np.float32)
                if not interpolate_gaps:
                    right_idx = np.clip(
                        np.searchsorted(v_d, tg_local, side="right"), 1, len(v_d) - 1
                    )
                    left_idx = right_idx - 1
                    spans = v_d[right_idx] - v_d[left_idx]
                    dist_to_nearest = np.minimum(
                        np.abs(tg_local - v_d[left_idx]),
                        np.abs(v_d[right_idx] - tg_local),
                    )
                    gap_mask = (spans > max_valid_span) & (dist_to_nearest > snap_tol)
                    y_local[gap_mask] = np.nan
                interp_y[local_mask] = y_local
            run_grid_df[c] = interp_y
        interp_runs.append(run_grid_df)

    mode = (conflict_mode or "keep_last").strip().lower()
    out_df = pd.DataFrame({"DEPTH": target_grid})
    for k, v in meta_Values.items():
        out_df[k] = v

    for c in all_curves:
        stacked = np.column_stack([ir[c].to_numpy(dtype=np.float64) for ir in interp_runs])
        if mode == "keep_first":
            merged_c = np.full(len(target_grid), np.nan, dtype=np.float32)
            for col_idx in range(stacked.shape[1]):
                col_vals = stacked[:, col_idx]
                fill_mask = np.isnan(merged_c) & np.isfinite(col_vals)
                merged_c[fill_mask] = col_vals[fill_mask].astype(np.float32)
            out_df[c] = merged_c
        elif mode == "mean":
            with np.errstate(invalid="ignore"):
                out_df[c] = np.nanmean(stacked, axis=1).astype(np.float32)
        else:  # keep_last (default)
            merged_c = np.full(len(target_grid), np.nan, dtype=np.float32)
            for col_idx in range(stacked.shape[1]):
                col_vals = stacked[:, col_idx]
                valid_m = np.isfinite(col_vals)
                merged_c[valid_m] = col_vals[valid_m].astype(np.float32)
            out_df[c] = merged_c

    return out_df


def inspect_dlis(
    file_path: Union[str, Path, bytes], filename: Optional[str] = None
) -> Dict[str, Any]:
    """Open a binary `.dlis` file via `dlisio.dlis.load()`, iterate over Logical Files and Frames,
    and extract `WellHeaderMetadata`, frame index ranges, and 1D scalar vs multi-dimensional array channels.
    """
    import tempfile
    from dlisio import dlis
    from quick_pp.app.backend.schemas.well_metadata import (
        CurveInspectionItem,
        DlisFrameInfo,
        WellHeaderMetadata,
    )

    tmp_path: Optional[Path] = None
    if isinstance(file_path, bytes):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".dlis")
        tf.write(file_path)
        tf.close()
        tmp_path = Path(tf.name)
        p = tmp_path
    else:
        p = Path(file_path)
    frames_info: List[DlisFrameInfo] = []
    curves_info: List[CurveInspectionItem] = []
    seen_curves: set[str] = set()

    well_meta_dict: Dict[str, Any] = {}
    raw_params: Dict[str, Any] = {}
    raw_origin: Dict[str, Any] = {}

    with dlis.load(str(p)) as logical_files:
        for lf_idx, lf in enumerate(logical_files):
            # Extract ORIGIN metadata
            for origin in getattr(lf, "origins", []):
                if getattr(origin, "well_name", None):
                    raw_origin["WELL"] = str(origin.well_name).strip()
                if getattr(origin, "well_id", None):
                    raw_origin["UWI"] = str(origin.well_id).strip()
                if getattr(origin, "field_name", None):
                    raw_origin["FLD"] = str(origin.field_name).strip()
                if getattr(origin, "company", None):
                    raw_origin["COMP"] = str(origin.company).strip()
                if getattr(origin, "producer_name", None):
                    raw_origin["SRVC"] = str(origin.producer_name).strip()
                if getattr(origin, "creation_time", None):
                    raw_origin["DATE"] = str(origin.creation_time).strip()

            # Extract PARAMETER objects
            for param in getattr(lf, "parameters", []):
                p_name = str(getattr(param, "name", "")).strip().upper()
                if not p_name:
                    continue
                vals = getattr(param, "values", None)
                val = vals[0] if (vals is not None and len(vals) > 0) else None
                if val is not None:
                    if isinstance(val, (np.generic, np.ndarray)):
                        val = val.item() if val.ndim == 0 else str(val.tolist())
                    p_unit = str(getattr(param, "units", "") or "").strip()
                    p_descr = str(getattr(param, "long_name", "") or "").strip()
                    raw_params[p_name] = {
                        "value": val,
                        "unit": p_unit,
                        "descr": p_descr,
                    }

            # Iterate over Frames
            for fr in getattr(lf, "frames", []):
                fr_name = str(getattr(fr, "name", "0"))
                idx_type = str(getattr(fr, "index_type", "BOREHOLE-DEPTH") or "BOREHOLE-DEPTH")
                idx_ch_name = str(getattr(fr, "index", "DEPTH") or "DEPTH")

                scalar_chs: List[str] = []
                array_chs: List[str] = []
                depth_raw_unit = "m"

                for ch in getattr(fr, "channels", []):
                    ch_name = str(getattr(ch, "name", "")).strip()
                    ch_unit = str(getattr(ch, "units", "") or "").strip()
                    dim = [int(x) for x in (getattr(ch, "dimension", [1]) or [1])]
                    is_array = len(dim) > 1 or (len(dim) == 1 and dim[0] > 1)
                    if is_array:
                        array_chs.append(ch_name)
                    else:
                        scalar_chs.append(ch_name)
                    if ch_name.upper() == idx_ch_name.upper():
                        depth_raw_unit = ch_unit or "m"

                    if ch_name not in seen_curves:
                        seen_curves.add(ch_name)
                        target_mnem = Config.resolve_mnemonic_alias(ch_name)
                        auto_conv = None
                        t_unit = ch_unit
                        u_up = ch_unit.upper()
                        if target_mnem == "DEPTH" and u_up in ("0.1 IN", "0.1IN", "FT", "F"):
                            t_unit = "m"
                            auto_conv = f"{ch_unit} -> m"
                        elif target_mnem in ("NPHI", "PHIT", "PHIE") and u_up in ("%", "PU", "P.U."):
                            t_unit = "v/v"
                            auto_conv = f"{ch_unit} -> v/v"
                        elif target_mnem == "RHOB" and u_up in ("KG/M3", "KGM3"):
                            t_unit = "g/cc"
                            auto_conv = f"{ch_unit} -> g/cc"

                        curves_info.append(
                            CurveInspectionItem(
                                raw_mnemonic=ch_name,
                                target_mnemonic=target_mnem,
                                raw_unit=ch_unit,
                                target_unit=t_unit,
                                description=str(getattr(ch, "long_name", "") or ""),
                                auto_conversion=auto_conv,
                                is_array_channel=is_array,
                                dimensions=dim,
                                include=not is_array,
                            )
                        )

                # Determine depth range & step in meters
                u_up = depth_raw_unit.upper()
                d_scale = (
                    0.00254
                    if u_up in ("0.1 IN", "0.1IN")
                    else 0.3048
                    if u_up in ("FT", "F")
                    else 0.0254
                    if u_up in ("IN", "INCHES")
                    else 1.0
                )
                start_d = _parse_float_safe(getattr(fr, "index_min", None))
                stop_d = _parse_float_safe(getattr(fr, "index_max", None))
                step_d = _parse_float_safe(getattr(fr, "spacing", None))

                if start_d is None or stop_d is None:
                    try:
                        c_arr = fr.curves()
                        for fld in c_arr.dtype.names or ():
                            if fld.upper() == idx_ch_name.upper():
                                idx_vals = np.asarray(c_arr[fld], dtype=float)
                                if len(idx_vals) > 0:
                                    start_d = float(idx_vals[0])
                                    stop_d = float(idx_vals[-1])
                                    if step_d is None and len(idx_vals) > 1:
                                        step_d = float(idx_vals[1] - idx_vals[0])
                                break
                    except Exception:
                        pass

                frames_info.append(
                    DlisFrameInfo(
                        logical_file_index=lf_idx,
                        frame_name=fr_name,
                        index_type=idx_type,
                        index_channel=idx_ch_name,
                        start_depth=round(start_d * d_scale, 4) if start_d is not None else None,
                        stop_depth=round(stop_d * d_scale, 4) if stop_d is not None else None,
                        step=round(step_d * d_scale, 4) if step_d is not None else None,
                        depth_unit="m" if d_scale != 1.0 else depth_raw_unit,
                        scalar_channels=scalar_chs,
                        array_channels=array_chs,
                        selected=False,
                    )
                )

    # Select longest BOREHOLE-DEPTH frame by default
    if frames_info:
        best_idx = 0
        best_span = -1.0
        for idx, fi in enumerate(frames_info):
            span = (
                abs((fi.stop_depth or 0.0) - (fi.start_depth or 0.0))
                if (fi.start_depth is not None and fi.stop_depth is not None)
                else float(len(fi.scalar_channels))
            )
            if "DEPTH" in fi.index_type.upper():
                span += 10000.0
            if span > best_span:
                best_span = span
                best_idx = idx
        frames_info[best_idx].selected = True

    combined_header = _sanitize_json_value(
        {
            **raw_params,
            **raw_origin,
            "raw_well_header": raw_origin,
            "raw_parameters": {
                k: (v.get("value") if isinstance(v, dict) else v)
                for k, v in raw_params.items()
            },
        }
    )
    fallback_stem = Path(filename).stem if filename else p.stem
    well_meta_dict = extract_las_metadata(combined_header, fallback_well_name=fallback_stem)
    if frames_info:
        sel = next((f for f in frames_info if f.selected), frames_info[0])
        well_meta_dict["start_depth"] = sel.start_depth
        well_meta_dict["stop_depth"] = sel.stop_depth
        well_meta_dict["step"] = sel.step

    if tmp_path is not None and tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    return {
        "metadata": WellHeaderMetadata(**well_meta_dict),
        "frames": frames_info,
        "curves": curves_info,
    }


def read_dlis_frame(
    file_path: Union[str, Path, bytes],
    logical_file_idx: int = 0,
    frame_name: Optional[str] = None,
    skip_array_channels: bool = True,
    target_depth_uom: str = "m",
    custom_mnemonic_map: Optional[Dict[str, str]] = None,
    filename: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any], Dict[str, str]]:
    """Extract a DLIS frame's curves into a standardized `pandas.DataFrame`,
    replace DLIS absent values (`-999.25`) with `NaN`, and run unit/mnemonic normalization.
    """
    import tempfile
    from dlisio import dlis

    tmp_path: Optional[Path] = None
    if isinstance(file_path, bytes):
        tf = tempfile.NamedTemporaryFile(delete=False, suffix=".dlis")
        tf.write(file_path)
        tf.close()
        tmp_path = Path(tf.name)
        p = tmp_path
    else:
        p = Path(file_path)
    inspect_res = inspect_dlis(p, filename=filename)
    meta_dict = inspect_res["metadata"].model_dump()

    with dlis.load(str(p)) as logical_files:
        if logical_file_idx >= len(logical_files):
            logical_file_idx = 0
        lf = logical_files[logical_file_idx]
        frames = list(getattr(lf, "frames", []))
        if not frames:
            raise ValueError(f"No frames found in DLIS file {p.name}")

        target_frame = None
        if frame_name is not None:
            target_frame = next(
                (fr for fr in frames if str(fr.name) == str(frame_name)), None
            )
        if target_frame is None:
            target_frame = frames[0]

        raw_curves = target_frame.curves()
        ch_units: Dict[str, str] = {}
        scalar_cols: Dict[str, np.ndarray] = {}

        for ch in getattr(target_frame, "channels", []):
            ch_name = str(ch.name).strip()
            ch_unit = str(getattr(ch, "units", "") or "").strip()
            dim = list(getattr(ch, "dimension", [1]) or [1])
            is_array = len(dim) > 1 or (len(dim) == 1 and dim[0] > 1)
            if is_array and skip_array_channels:
                continue
            if ch_name in (raw_curves.dtype.names or ()):
                arr = np.asarray(raw_curves[ch_name])
                if arr.ndim == 1:
                    is_idx = ch_name.upper() in ("DEPTH", "DEPT", str(target_frame.index).upper())
                    arr_f = arr.astype(np.float64 if is_idx else np.float32)
                    for nv in SENTINEL_NULLS:
                        arr_f[np.isclose(arr_f, nv, atol=1e-3)] = np.nan
                    scalar_cols[ch_name] = arr_f
                    ch_units[ch_name] = ch_unit

        df = pd.DataFrame(scalar_cols)
        df, std_units, _ = standardize_curve_units(
            df,
            curve_units=ch_units,
            target_depth_uom=target_depth_uom,
            custom_mnemonic_map=custom_mnemonic_map,
        )
        if "WELL_NAME" not in df.columns:
            df.insert(0, "WELL_NAME", meta_dict.get("well_name") or (Path(filename).stem if filename else p.stem))
        if meta_dict.get("uwi") and "UWI" not in df.columns:
            df.insert(0, "UWI", meta_dict["uwi"])

    if tmp_path is not None and tmp_path.exists():
        try:
            tmp_path.unlink()
        except Exception:
            pass

    return df, meta_dict, std_units

