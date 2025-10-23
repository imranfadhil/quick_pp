import re
import lasio
import os
import numpy as np
import pandas as pd
import mmap
import welly

from quick_pp import logger


def read_las_files(las_files, depth_uom=None):
    """Reads multiple LAS files and merges their data and headers.

    This function iterates through a list of LAS file objects, reads each one,
    and concatenates the curve data and header information into two separate
    pandas DataFrames. It attempts to read with `welly` first and falls back
    to a memory-mapped approach if `welly` fails.

    Args:
        las_files (list): A list of file-like objects opened in binary mode.
        depth_uom (str, optional): The unit of measurement for the depth index.
                                   Used by `welly` during reading. Defaults to None.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame containing the merged curve data from all LAS files.
            - A DataFrame containing the merged header information from all LAS files.
    """
    merged_data = pd.DataFrame()
    header_data = pd.DataFrame()

    for f in las_files:
        try:
            df, well_header = read_las_file_welly(f, depth_uom)
        except Exception as e:
            logger.error(f"[read_las_files] Exception for {f.name} | {e} ")
            df, well_header, _ = read_las_file_mmap(f)
        merged_data = pd.concat([merged_data, df], ignore_index=True)
        header_data = pd.concat([header_data, well_header], ignore_index=True)

    merged_data.reset_index(inplace=True, drop=True)

    return merged_data, header_data


def read_las_file_mmap(file_object, required_sets=['PEP']):  # noqa
    """Reads a single LAS file using a memory-mapped approach, handling multiple data sets.

    This function is designed to parse LAS files, especially those with multiple
    data sets (e.g., different logging runs), which might not be handled correctly
    by standard parsers. It uses memory-mapping for efficient file access and can
    selectively extract required data sets.

    Args:
        file_object (file): A file-like object opened in binary mode.
        required_sets (list, optional): A list of data set identifiers to extract.
                                        If the LAS file contains multiple sets, only
                                        those matching the identifiers in this list
                                        will be processed. Defaults to ['PEP'].

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, welly.well.Well]:
            - A DataFrame containing the curve data.
            - A DataFrame containing the header information.
            - A `welly.well.Well` object representing the well.
    """
    fileno = file_object.fileno()  # identifier for files
    parameter_line_numbers = []
    with mmap.mmap(fileno, length=0, access=mmap.ACCESS_READ) as mmap_obj:
        all_text = mmap_obj.read()
        set_count = len(re.findall(r'\b(SET)\s+', all_text.decode(), re.MULTILINE))
        if set_count > 1:
            well_count = 0
            dataset_count = 0
            line_number = 1
            parameter_count = 0
            mmap_obj.seek(0)
            pointer = 0
            while pointer < mmap_obj.size():
                text = mmap_obj.readline()
                pointer = (mmap_obj.tell())  # show current position of pointer

                if b'~Well' in text:
                    well_count += 1
                if b'~Curve' in text:
                    dataset_count += 1
                if b'~P' in text or b'~Tops_Parameter' in text:
                    # Record parameter info in tuple:(parameter_count, parameter_set, pointer location, line number)
                    parameter_line_numbers.append((parameter_count, '', pointer, line_number))
                    parameter_count += 1
                if re.compile(r'^\b(SET)\s+').search(text.decode()):
                    parameter_set = re.split(r'[\s+,.:]', text.decode().replace(' ', ''))[1]
                    temp_list_from_tuple = list(parameter_line_numbers[parameter_count - 1])
                    temp_list_from_tuple[1] = parameter_set
                    temp_list_from_tuple = tuple(temp_list_from_tuple)
                    parameter_line_numbers[parameter_count - 1] = temp_list_from_tuple
                line_number += 1

            # Record well header numbers in tuple: (0, '', pointer location, line number)
            well_header_line_numbers = [(0, '', 0, 1), (0, '', parameter_line_numbers[0][2] - 1,
                                        parameter_line_numbers[0][3] - 1)]

            mmap_obj.seek(0)  # Reset the pointer location
            curves_df, header_df, welly_object = concat_datasets(
                file_object=mmap_obj.read(), header_line_numbers=well_header_line_numbers,
                parameter_line_numbers=parameter_line_numbers, required_sets=required_sets)
        else:
            well_count = 1
            counter = 0
            pointer_list = []
            section_dict = {}
            mmap_obj.seek(0)
            pointer = 0
            while pointer < mmap_obj.size():
                text = mmap_obj.readline()
                pointer = (mmap_obj.tell())
                if pointer not in pointer_list:
                    pointer_list.append(pointer)
                    counter += 1
                if b'~' in text:
                    section = text.decode().replace('~', '').rstrip().split(' ')[0].upper()
                    rename_set = {
                        'V': 'VERSION',
                        'W': 'WELL',
                        'P': 'PARAMETER',
                        'C': 'CURVE',
                        'O': 'OTHER',
                        'A': 'ASCII',
                    }
                    for initial, word in rename_set.items():
                        if section == initial:
                            section = section.replace(initial, word)
                    section_text = text
                    text = mmap_obj.readline()
                    pointer = (mmap_obj.tell())
                    if pointer not in pointer_list:
                        pointer_list.append(pointer)
                        counter += 1
                    while b'~' not in text and len(text) > 0:
                        section_text = section_text + text
                        text = mmap_obj.readline()
                        pointer = (mmap_obj.tell())
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
    """Reads a LAS file using the welly library.

    This function provides a straightforward way to read a LAS file into a
    `welly` object, which is then processed to extract curve data and header
    information into pandas DataFrames.

    Args:
        file_object (file): A file-like object, from which the `.name` attribute
                            (file path) is used by `welly`.
        depth_uom (str, optional): The unit of measurement for the depth index.
                                   Passed to `welly` for index creation. Defaults to None.

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
    """Pre-processes a `welly` object to extract and clean data.

    This function takes a `welly.well.Well` object and performs several
    pre-processing steps:
    1. Converts the depth index to a 'DEPTH' column.
    2. Replaces LAS-defined NULL values with `np.nan`.
    3. Inserts 'WELL_NAME' and 'UWI' columns at the beginning of the DataFrame.

    Args:
        welly_object (welly.well.Well): The welly object to process.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]:
            - A DataFrame containing the processed curve data.
            - A DataFrame containing the header information.
    """
    # Convert index 'DEPTH' as column
    data_df = welly_object.las[0]
    data_df.index.rename('DEPTH', inplace=True)
    data_df = data_df.reset_index(drop=False)
    data_df['DEPTH'] = data_df['DEPTH'].round(4)

    header_df = welly_object.header
    nullValue = header_df[header_df['mnemonic'] == 'NULL']['value'].values[0] if \
        header_df[header_df['mnemonic'] == 'NULL']['value'].values else -999.25
    data_df = data_df.where(data_df >= nullValue, np.nan)
    # Insert well name
    well_name = get_wellname_from_header(header_df)
    if 'WELL_NAME' not in data_df.columns:
        data_df.insert(0, 'WELL_NAME', well_name)
    # Insert UWI if available
    if 'UWI' in header_df['mnemonic'].values:
        uwi = get_uwi_from_header(header_df)
        data_df.insert(0, 'UWI', uwi)

    return data_df, header_df


def get_wellname_from_header(header_df):
    """Extracts the well name from the LAS header DataFrame.

    Args:
        header_df (pd.DataFrame): The LAS header data.

    Returns:
        str: The well name, with slashes and spaces replaced by hyphens.
    """
    return header_df[
        (header_df['mnemonic'] == 'WELL') | (header_df['descr'].str.upper() == 'WELL')
    ]['value'].values[0].replace("/", "-").replace(" ", "-")


def get_uwi_from_header(header_df):
    """Extracts the Unique Well Identifier (UWI) from the LAS header DataFrame.

    If the UWI is not found, it falls back to using the well name.

    Args:
        header_df (pd.DataFrame): The LAS header data.

    Returns:
        str: The UWI or well name, with slashes and spaces replaced by hyphens.
    """
    uwi = header_df[
            (header_df['mnemonic'] == 'UWI') | (header_df['descr'].str.upper() == 'UNIQUE WELL ID')
        ]['value'].values[0].replace("/", "-").replace(" ", "-")
    return uwi if uwi else get_wellname_from_header(header_df)


def get_unit_from_header(header_df, mnemonic):
    """Extracts the unit for a specific curve mnemonic from the LAS header.

    Args:
        header_df (pd.DataFrame): The LAS header data.
        mnemonic (str): The curve mnemonic to look for.

    Returns:
        str or None: The unit of the curve, or None if not found.
    """
    return header_df[header_df['mnemonic'] == mnemonic]['unit'].values[0] if any(
        header_df['mnemonic'].str.contains(mnemonic)) else None


def get_descr_from_header(header_df, mnemonic):
    """Extracts the description for a specific curve mnemonic from the LAS header.

    Args:
        header_df (pd.DataFrame): The LAS header data.
        mnemonic (str): The curve mnemonic to look for.

    Returns:
        str or None: The description of the curve, or None if not found.
    """
    return header_df[header_df['mnemonic'] == mnemonic]['descr'].values[0] if any(
        header_df['mnemonic'].str.contains(mnemonic)) else None


def extract_dataset(section_dict):
    """Extracts a single dataset from a dictionary of LAS file sections.

    This function is intended for LAS files that contain only one data set.
    It reconstructs the LAS file content from the separated sections, reads it
    using `lasio` and `welly`, and then pre-processes the result.

    Args:
        section_dict (dict): Dictionary containing the LAS file section values.

    Returns:
        pd.DataFrame, pd.DataFrame, welly_object: well_df, header_df and welly_object
    """
    header_bytes = section_dict['WELL']
    data_bytes = b''
    for k, v in section_dict.items():
        if k in ['PARAMETER', 'CURVE', 'ASCII']:
            data_bytes = data_bytes + v

    file_object = header_bytes.decode() + data_bytes.decode()
    las_object = lasio.read(file_object, read_policy=())

    # Fix las_object
    df = las_object.df()
    df = df.apply(pd.to_numeric, errors='coerce')
    las_object.set_data_from_df(df)

    welly_object = welly.Well.from_lasio(las_object)
    well_df = pre_process(welly_object)
    header_df = welly_object.header

    return well_df, header_df, welly_object


def concat_datasets(file_object, header_line_numbers, parameter_line_numbers, required_sets=['PEP']):
    """Extracts and concatenates specified datasets from a multi-set LAS file.

    This function iterates through the parameter sections identified in a LAS file.
    For each section that matches the `required_sets`, it reconstructs a temporary
    single-set LAS file in memory, reads it, and concatenates the resulting data.

    Args:
        file_object (bytes): The full content of the LAS file as a bytes object.
        header_line_numbers (list): A list of tuples defining the start and end
                                    pointers for the main well header section.
        parameter_line_numbers (list): A list of tuples, each containing metadata
                                       about a `~Parameter` section, including its
                                       set identifier and pointer location.
        required_sets (list, optional): A list of data set identifiers to extract
                                        and concatenate. Defaults to ['PEP'].

    Returns:
        pd.Dataframe, pd.Dataframe, welly_object: well_df, header_df and welly_object
    """
    well_df = pd.DataFrame()
    header_df = pd.DataFrame()
    welly_object = welly.Well()
    for i, (param_count, param_set, pointer, line_number) in enumerate(parameter_line_numbers):
        # Currently only extracting one dataset: PEP
        if param_set in required_sets:
            well_info = file_object[header_line_numbers[0][2]: header_line_numbers[1][2] + 1].decode()
            if i < len(parameter_line_numbers) - 1:
                temp_file_object = file_object[pointer: parameter_line_numbers[i + 1][2]].decode()
            else:
                temp_file_object = file_object[pointer:].decode()
            temp_file_object = well_info + temp_file_object
            las_object = lasio.read(temp_file_object, read_policy=())

            # Fix las_object
            df = las_object.df()
            df = df.apply(pd.to_numeric, errors='coerce')
            las_object.set_data_from_df(df)

            welly_object = welly.Well.from_lasio(las_object)
            temp_well_df = pre_process(welly_object)
            well_df = pd.concat([well_df, temp_well_df], axis=1)
            header_df = welly_object.header

    return well_df, header_df, welly_object


def check_index_consistent(welly_object):
    """Checks if the depth index of a welly object is consistent.

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


def export_to_las(well_data, well_name, folder='', vars_units=None):
    """Exports a DataFrame of well data to a LAS file.

    The input DataFrame is expected to have a 'DEPTH' column, which will be
    used as the index for the LAS file.

    Args:
        well_data (pd.DataFrame): The DataFrame containing the well log data.
        well_name (str): The name of the well, used for the output filename.
        folder (str, optional): The directory to save the LAS file in. Defaults to ''.
        vars_units (dict, optional): A dictionary mapping curve mnemonics to their
                                     units. If not provided, it will be inferred
                                     from the configuration.
    """
    from .config import Config
    units = vars_units if vars_units else Config.vars_units(well_data)
    well_data.set_index('DEPTH', inplace=True, drop=True)
    w = welly.Well().from_df(well_data, units=units, name=well_name)

    w = w.from_df(well_data, units=units, name=well_name)
    # Convert to lasio to handle index name
    las = w.to_lasio()
    las.curves[0].mnemonic = 'DEPTH'
    # Write to LAS format
    well_path = os.path.join(folder, f"{well_name}.las")
    las.write(well_path)
