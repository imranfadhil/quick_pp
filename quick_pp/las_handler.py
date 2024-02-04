import re
import lasio
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
        df, well_header, _ = read_las_file(f)
        merged_data = pd.concat([merged_data, df], ignore_index=True)
        header_data = pd.concat([header_data, well_header], ignore_index=True)

    header_data_cols = header_data.columns
    header_data = header_data.transpose()
    header_data.rename({0: 'well_name'}, axis=1, inplace=True)
    header_data.rename({x: v for x, v in enumerate(header_data_cols)}, axis=0, inplace=True)

    merged_data.reset_index(inplace=True, drop=True)

    return merged_data, header_data


def read_las_file(file_object, required_sets=['PEP']):  # noqa
    """Check LAS file and concat datasets if more than one.

       If more than one set in las file,

       1. Loop through every line,
       2. If ~Well is found, add one to well_count
       3. If ~Curve is found, add one to dataset_count
       4. If ~P or ~Tops_Parameter found, append tuple of (parameter_count, parameter_set, pointer location,
       line number) into a list "parameter_line_numbers".
       5. If SET found, split the line using delimiter and assign only "VALUE" as parameter_set, then add parameter_set
       into tuple in list "parameter_line_numbers"
       6. Record well header numbers in tuple: (0, '', pointer location, line number)
       7. Through function "concat_datasets", return data_df, header_df and welly_object else(to catch las file with
       only 1 dataset),
       8. Loop through every line,
       9. If ~ is found, rename every sets accordingly,
       10. Assign every line in every section (Version/Well/Parameter etc) into a section dictionary
       11. Through function "extract_dataset", return data_df, header_df and welly_object

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
        set_count = len(re.findall(r'\b(SET)\s+', all_text.decode('Windows-1252')))
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
                if re.compile(r'^\b(SET)\s+').search(text.decode('Windows-1252')):
                    parameter_set = re.split(r'[\s+,.:]', text.decode('Windows-1252').replace(' ', ''))[1]
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
                    section = text.decode('Windows-1252').replace('~', '').rstrip().split(' ')[0].upper()
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
    data_df = resample_depth(welly_object)

    header_df = welly_object.header
    nullValue = header_df[header_df['mnemonic'] == 'NULL']['value'].values[0] if \
        header_df[header_df['mnemonic'] == 'NULL']['value'].values else -999.25
    data_df = data_df.where(data_df >= nullValue, np.nan)

    well_name = header_df[
        (header_df['mnemonic'] == 'WELL') | (header_df['descr'].str.upper() == 'WELL')]['value'].values[0]
    data_df.insert(0, 'WELL_NAME', well_name)

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

    file_object = header_bytes.decode('Windows-1252') + data_bytes.decode('Windows-1252')
    las_object = lasio.read(file_object)

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
            well_info = file_object[header_line_numbers[0][2]: header_line_numbers[1][2] + 1].decode('Windows-1252')
            if i < len(parameter_line_numbers) - 1:
                temp_file_object = file_object[pointer: parameter_line_numbers[i + 1][2]].decode('Windows-1252')
            else:
                temp_file_object = file_object[pointer:].decode('Windows-1252')
            temp_file_object = well_info + temp_file_object
            las_object = lasio.read(temp_file_object)
            welly_object = welly.Well.from_lasio(las_object)
            temp_well_df = pre_process(welly_object)
            well_df = pd.concat([well_df, temp_well_df], axis=1)
            header_df = welly_object.header

    return well_df, header_df, welly_object


def resample_depth(welly_object, step_depth=0.5):
    """Convert m to ft and resample DEPTH to feet with 0.5 step

    Args:
        welly_object (object): Welly object
        step_depth (float, optional): Defaults to 0.5

    Returns:
        pd.DataFrame: Resampled dataframe
    """
    return_df = welly_object.df()
    for i, curve_name in enumerate(list(welly_object.data.keys())):
        pre_curve = welly_object.data[curve_name]
        if pre_curve.index_units and pre_curve.index_units.lower() in ['metres', 'm']:
            pre_curve.df.index = welly.well._convert_depth_index_units(pre_curve.index, unit_from='m', unit_to='ft')
            pre_curve.index_units = 'ft'

        # Set new start if not 0.5
        new_start = pre_curve.start
        if np.mod(new_start, 0.5) != 0:
            new_start = np.round((new_start * 2)) / 2

        # Set the new stop of depth.
        new_stop = pre_curve.start + (
            step_depth * np.ceil((pre_curve.stop - pre_curve.start) / step_depth)
        )

        new_curve = pre_curve.to_basis(start=new_start, stop=new_stop, step=step_depth)
        new_curve = pd.DataFrame(new_curve.values, index=new_curve.index.values, columns=[curve_name])
        # Create new dataframe when first loop.
        if i == 0:
            new_clips_df = pd.DataFrame(new_curve.values, index=new_curve.index.values, columns=[curve_name])
        # Merge new dataframe with the previous loop.
        else:
            new_curve_df = pd.DataFrame(new_curve.values, index=new_curve.index.values, columns=[curve_name])
            new_clips_df = pd.merge(new_clips_df, new_curve_df, left_index=True, right_index=True)

    return_df = new_clips_df.copy()
    # Convert index 'DEPTH' as column
    if "DEPTH" in return_df.columns:
        return_df.set_index('DEPTH', inplace=True)
    return_df.index.rename('DEPTH', inplace=True)
    return_df.reset_index(drop=False, inplace=True)

    return return_df


def export_to_las(well_data, well_name):
    """Export dataframe to las file.

    Args:
        data_df (pd.DataFrame): data input
        well_name (str): well name
    """
    from .config import Config
    units = Config.vars_units(well_data)
    well_data['DEPT'] = well_data['DEPTH']
    well_data.set_index('DEPT', inplace=True, drop=True)
    w = welly.Well().from_df(well_data, units=units, name=well_name)
    w.to_las(f'{well_name}.las')


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
