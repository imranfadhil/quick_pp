import pandas as pd
import re
from typing import Dict


def parse_survey_file(file_path: str) -> pd.DataFrame:
    """
    Parse a survey file and extract the SURVEY LIST section into a pandas DataFrame.
    
    Args:
        file_path (str): Path to the survey text file
        
    Returns:
        pd.DataFrame: DataFrame containing the survey data with columns:
                     MD, Inc, Azim, TVD, X-Offset (E/W), Y-Offset (N/S), 
                     UTM E/W, UTM N/S, DLS
    """
    
    with open(file_path, 'r') as file:
        content = file.read()
    
        # Find the SURVEY LIST section
        survey_pattern = r'SURVEY LIST\n([\s\S]*$)'
        survey_match = re.search(survey_pattern, content, re.IGNORECASE)
        
        if not survey_match:
            raise ValueError("SURVEY LIST section not found in the file")
        
        survey_data = survey_match.group(1).strip()
        
        # Split into lines and filter out empty lines
        lines = [line.strip() for line in survey_data.split('\n') if line.strip()]
        
        # Find the header line (contains column names and units)
        header_line = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if 'MD' in line and 'Inc' in line and 'Azim' in line:
                header_line = line
                units_line = lines[i + 1]  # Get the units line which follows the header
                data_start_idx = i + 2
                break
        
        if not header_line:
            raise ValueError("Survey data header not found")
        
        # Parse the header to get column names
        # Header format: "MD        Inc       Azim      TVD       X-Offset (E/W)  Y-Offset (N/S)  UTM E/W         UTM N/S         DLS"
        header_parts = re.split(r'\s{2,}', header_line.strip())
        
        # Clean up column names
        column_names = []
        for part in header_parts:
            if part.strip():
                # Remove units in parentheses and clean up
                clean_name = re.sub(r'\s*\([^)]*\)\s*', '', part.strip())
                column_names.append(clean_name)
        
        # Parse the units line to get column units
        units_parts = re.split(r'\s{2,}', units_line.strip())
        
        # Clean up column units
        column_units = []
        for part in units_parts:
            if part.strip():
                # Remove units in parentheses and clean up
                clean_unit = re.sub(r'\s*\([^)]*\)\s*', '', part.strip())
                column_units.append(clean_unit)
        
        # Parse data lines
        data_rows = []
        for line in lines[data_start_idx:]:
            # Split by multiple spaces and filter out empty strings
            parts = [part.strip() for part in re.split(r'\s+', line.strip())
                    if part.strip()]
            
            if len(parts) >= len(column_names):
                data_rows.append(parts[:len(column_names)])
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=column_names)
        
        # Convert numeric columns
        numeric_columns = ['MD', 'Inc', 'Azim', 'TVD', 'X-Offset (E/W)',
                        'Y-Offset (N/S)', 'UTM E/W', 'UTM N/S', 'DLS']

        # Add units to column names
        column_names = [f"{name} ({units})" for name, units in zip(column_names, column_units)]
        df.columns = column_names
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df


def parse_survey_text(text_content: str) -> pd.DataFrame:
    """
    Parse survey data from text content (string) instead of a file.
    
    Args:
        text_content (str): The survey text content
        
    Returns:
        pd.DataFrame: DataFrame containing the survey data
    """
    
    # Find the SURVEY LIST section
    survey_pattern = r'SURVEY LIST\s*\n(.*?)(?=\n\s*\n|\n\s*[A-Z]|\Z)'
    survey_match = re.search(survey_pattern, text_content, re.DOTALL | re.IGNORECASE)
    
    if not survey_match:
        raise ValueError("SURVEY LIST section not found in the text")
    
    survey_data = survey_match.group(1).strip()
    
    # Split into lines and filter out empty lines
    lines = [line.strip() for line in survey_data.split('\n') if line.strip()]
    
    # Find the header line (contains column names and units)
    header_line = None
    data_start_idx = 0
    
    for i, line in enumerate(lines):
        if 'MD' in line and 'Inc' in line and 'Azim' in line:
            header_line = line
            data_start_idx = i + 1
            break
    
    if not header_line:
        raise ValueError("Survey data header not found")
    
    # Parse the header to get column names
    header_parts = re.split(r'\s{2,}', header_line.strip())
    
    # Clean up column names
    column_names = []
    for part in header_parts:
        if part.strip():
            # Remove units in parentheses and clean up
            clean_name = re.sub(r'\s*\([^)]*\)\s*', '', part.strip())
            column_names.append(clean_name)
    
    # Parse data lines
    data_rows = []
    for line in lines[data_start_idx:]:
        # Split by multiple spaces and filter out empty strings
        parts = [part.strip() for part in re.split(r'\s+', line.strip())
                if part.strip()]
        
        if len(parts) >= len(column_names):
            data_rows.append(parts[:len(column_names)])
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=column_names)
    
    # Convert numeric columns
    numeric_columns = ['MD', 'Inc', 'Azim', 'TVD', 'X-Offset (E/W)',
                      'Y-Offset (N/S)', 'UTM E/W', 'UTM N/S', 'DLS']
    
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    return df


def extract_header_info(text_content: str) -> Dict[str, str]:
    """
    Extract header information from the survey file.
    
    Args:
        text_content (str): The survey text content
        
    Returns:
        Dict[str, str]: Dictionary containing header information
    """
    
    header_info = {}
    
    # Extract key-value pairs from the header section
    lines = text_content.split('\n')
    
    for line in lines:
        line = line.strip()
        if ':' in line and not line.startswith('SURVEY LIST'):
            # Split by first colon
            parts = line.split(':', 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                header_info[key] = value
    
    return header_info


# Example usage
if __name__ == "__main__":
    # Example with the text content you provided
    sample_text = """HEADER INFORMATION (all units in Statoil default) 
----------------------------------------------------------------
COMPANY: Statoil Norway
FIELD: SLEIPNER
WELL NAME: F-7
WELLBORE NAME: F-7
Drilled From: Well Ref. Point
Kick Off Depth: 145.90m
Survey Name: F-7
Survey Date: 06.08.2012
Extraction Date: 26.03.2018
----------------------------------------------------------------
Calculation Method: Minimum Curvature
Datum Name: Rotary Table(54.90m)
----------------------------------------------------------------
Surface EW: 435048.907m
Surface NS: 6478565.478m
Surface Latitude: 58° 26' 29.8694 N
Surface Longitude: 1° 53' 14.8578 E
Bottom Hole MD: 1083.00m
Bottom Hole TVD: 1077.48m
Bottom Hole EW: 434970.272m
Bottom Hole NS: 6478554.576m
----------------------------------------------------------------
Survey Program:  
H  160 - 317 F7 SINGLE SHOT SURVEYS : WELLBORE SURVEYOR, STAT
H  350 - 891 F7 GWD SURVEYS : WELLBORE SURVEYOR, STAT
H  962 - 1083 12 1/4" SURVEYS : MAGNETIC, STD, NON-MAG
----------------------------------------------------------------
Water Depth: 91.00m
KB-WH: 145.90m
KB-MSL: 54.90m
System Datum: Mean Sea Level
----------------------------------------------------------------
Map-Zone: Zone 31N (0 E to 6 E)
North Reference: Grid
Geo Datum: European 1950 - Mean
Vertical Section Direction: 248.17deg
----------------------------------------------------------------
SURVEY LIST
MD        Inc       Azim      TVD       X-Offset (E/W)  Y-Offset (N/S)  UTM E/W         UTM N/S         DLS
m RKB     deg       deg       m RKB     m               m               m               m               deg/30m
145.90    0.00      0.00      145.90    -1.114          1.956           435048.907      6478565.478     0.00      
160.00    0.23      274.41    160.00    -1.142          1.958           435048.879      6478565.480     0.49      
170.00    0.26      272.78    170.00    -1.185          1.961           435048.836      6478565.483     0.09      
180.00    0.30      262.14    180.00    -1.234          1.958           435048.788      6478565.480     0.20      
190.00    0.22      267.67    190.00    -1.279          1.954           435048.743      6478565.476     0.25"""
    
    # Parse the survey data
    survey_df = parse_survey_text(sample_text)
    
    # Extract header information
    header_info = extract_header_info(sample_text)
    
    print("Survey DataFrame:")
    print(survey_df)
    print("\nHeader Information:")
    for key, value in header_info.items():
        print(f"{key}: {value}")
    
    # Save to CSV if needed
    # survey_df.to_csv('survey_data.csv', index=False) 