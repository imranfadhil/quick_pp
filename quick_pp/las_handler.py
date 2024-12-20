import re
import lasio
import os
import numpy as np
import pandas as pd
import mmap
import welly


def read_las_files(las_files):
    """
    1. Return df and well_header as df from .las files
    2. For every las files, curves_df, header_df, welly_object are returned through function "read_las_file"
    3. curves_df from all files are appended into merged_data
    4. header_df from all files are appended into header_data

    Args:
        las_files (list): list of las files opened as binary.

    Returns:
        pd.DataFrame: Merged data, header data.
    """
    merged_data = pd.DataFrame()
    header_data = pd.DataFrame()

    for f in las_files:
        try:
            df, well_header = read_las_file_welly(f)
        except Exception as e:
            print(f"[read_las_files] Exception for {f.name} | {e} ")
            df, well_header, _ = read_las_file_mmap(f)
        merged_data = pd.concat([merged_data, df], ignore_index=True)
        header_data = pd.concat([header_data, well_header], ignore_index=True)

    header_data_cols = header_data.columns
    header_data = header_data.transpose()
    header_data.rename({0: 'well_name'}, axis=1, inplace=True)
    header_data.rename({x: v for x, v in enumerate(header_data_cols)}, axis=0, inplace=True)

    merged_data.reset_index(inplace=True, drop=True)

    return merged_data, header_data


def read_las_file_mmap(file_object, required_sets=['PEP']):  # noqa
    """Check LAS file and concat datasets if more than one.

    Args:
        file_object (str): las file object.
        required_sets (list, optional): Required param set to be extracted. Defaults to ['PEP'].

    Returns:
        pd.Dataframe: well_df, header_df and welly_object
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


def read_las_file_welly(file_object):  # noqa
    welly_dataset = welly.las.from_las(file_object.name)
    well_header = welly_dataset['Header']
    welly_object = welly.well.Well.from_datasets(welly_dataset)
    print('welly_object:', welly_object)
    df = pre_process(welly_object)
    return df, well_header


def pre_process(welly_object):
    """Pre-process welly_object

    1. Resample depth for welly_object
    2. Replace NULL value with nan
    3. Insert well name at first column
    4. Insert field name at first column

    Args:
        welly_object (well object): Welly object for PEP dataset

    Returns:
        pd.Dataframe: Processed data.
    """
    # Convert index 'DEPTH' as column
    data_df = welly_object.las[0]
    data_df.index.rename('DEPTH', inplace=True)
    data_df = data_df.reset_index(drop=False)

    header_df = welly_object.header
    nullValue = header_df[header_df['mnemonic'] == 'NULL']['value'].values[0] if \
        header_df[header_df['mnemonic'] == 'NULL']['value'].values else -999.25
    data_df = data_df.where(data_df >= nullValue, np.nan)
    # Insert well name
    well_name = header_df[
        (header_df['mnemonic'] == 'WELL') | (header_df['descr'].str.upper() == 'WELL')
    ]['value'].values[0]
    if 'WELL_NAME' not in data_df.columns:
        data_df.insert(0, 'WELL_NAME', well_name)
    # Insert UWI if available
    if 'UWI' in header_df['mnemonic'].values:
        uwi = header_df[
            (header_df['mnemonic'] == 'UWI') | (header_df['descr'].str.upper() == 'UNIQUE WELL ID')
        ]['value'].values[0]
        data_df.insert(0, 'UWI', uwi)

    return data_df


def extract_dataset(section_dict):
    """Extract dataset from section_dict. For only Las file with ONE (1) dataset

    1. Assign well information section as header_bytes
    2. Loop through parameter, curve, and ASCII sections, assign their section values into data_bytes
    3. create file_object by decoding and concat header_bytes and data_bytes
    4. Using lasio.read and welly.well.from_lasio, create welly_object & well_df from file_object
    5. Through function "pre_process", return df with null replaced by nan, addition of column with well and field name

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
    """Concat required datasets in the LAS file.

    1. Iterate through list of parameter_line_numbers, by extracting only param_set of 'PEP'
    2. Through pointer position, subset and then concat well_info and file_object
    3. Through function "pre_process", return df with null replaced by nan, addition of column with well and field name

    Args:
        file_object (object): File .read() object.
        header_line_numbers (list of tuples): Pointer location and line number of the header information.
        parameter_line_numbers (list of tuples): (parameter_count, parameter_set, pointer location, line number) of the
        ~Parameters in the LAS file.
        required_sets (list): Required sets to be concatenated.

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
    """Check if index is consistent in welly_object

    Args:
        welly_object (object): Welly object

    Returns:
        bool: True if index is consistent
    """
    try:
        index_diff = np.diff(welly_object.las[0].index)
        if all(index_diff == index_diff[0]) and all(index_diff > 0):
            return True
        else:
            return False
    except Exception as e:
        print(f"[las_handler] `check_index_consistent` Error | {e}")
        return False


def export_to_las(well_data, well_name, folder=''):
    """Export dataframe to las file. Expecting a DEPTH column in meters unit.

    Args:
        data_df (pd.DataFrame): data input
        well_name (str): well name
    """
    from .config import Config
    units = Config.vars_units(well_data)
    well_data.set_index('DEPTH', inplace=True, drop=True)
    w = welly.Well().from_df(well_data, units=units, name=well_name)

    w = w.from_df(well_data, units=units, name=well_name)
    # Convert to lasio to handle index name
    las = w.to_lasio()
    las.curves[0].mnemonic = 'DEPTH'
    # Write to LAS format
    well_path = os.path.join(folder, f"{well_name}.las")
    las.write(well_path)


if __name__ == '__main__':
    from tkinter import Tk, filedialog

    root = Tk()
    file_objects = filedialog.askopenfiles(title='Choose well Log ASCII Standard (LAS) files to be combined',
                                           filetype=(('LAS Files', '*.LAS *.las'), ('All Files', '*.*')),
                                           multiple=True,
                                           mode='rb')
    root.destroy()
    if file_objects:
        # Test read_las_file function
        merged_df, _ = read_las_files(file_objects)
        fname = 'well_df.csv'
        merged_df.to_csv(fname)
