from datetime import datetime
import getpass
import pandas as pd
import pickle
import os

import quick_pp.las_handler as las
from quick_pp.config import Config
from quick_pp import logger


def get_package_root():
    """Get the absolute path to the package root directory."""
    import quick_pp
    return os.path.dirname(os.path.dirname(os.path.abspath(quick_pp.__file__)))


def get_default_project_path():
    """Get the default project path relative to the package root."""
    package_root = get_package_root()
    return os.path.join(package_root, "data", "04_project")


def get_default_output_path():
    """Get the default output path relative to the package root."""
    package_root = get_package_root()
    return os.path.join(package_root, "data", "04_project", "outputs")


def resolve_path(path, base_path=None):
    """Resolve a path to an absolute path, optionally relative to a base path."""
    if os.path.isabs(path):
        return path
    elif base_path:
        return os.path.abspath(os.path.join(base_path, path))
    else:
        return os.path.abspath(path)


class Project(object):
    def __init__(self, name="", description="", project_path=""):
        self.name = name
        self.description = description
        self.data = {}
        self.history = []

        # Use absolute paths for better robustness
        if project_path:
            self.project_path = resolve_path(project_path)
        else:
            self.project_path = get_default_project_path()

        self.data_path = os.path.join(self.project_path, self.name)
        self.output_path = os.path.join(self.data_path, "outputs")

        # Ensure directories exist
        try:
            os.makedirs(self.data_path, exist_ok=True)
            os.makedirs(self.output_path, exist_ok=True)
            logger.info(f"Project '{name}' initialized with path: {self.data_path}")
        except PermissionError:
            logger.error(f"Permission denied when creating directories: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error creating project directories: {e}")
            raise

    def __getstate__(self):
        """Custom pickle state for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'project_path': self.project_path,
            'data': self.data,
            'history': self.history,
            'data_path': self.data_path,
            'output_path': self.output_path
        }

    def __setstate__(self, state):
        """Custom pickle state restoration."""
        self.name = state['name']
        self.description = state['description']
        self.project_path = state['project_path']
        self.data = state['data']
        self.history = state['history']
        self.data_path = state['data_path']
        self.output_path = state['output_path']

    def read_las(self, path: list):
        logger.info(f"Reading {len(path)} LAS files for project '{self.name}'")
        for file in path:
            logger.debug(f"Processing LAS file: {file}")
            well = Well()
            well.read_las(file)
            self.save_well(well)
        self.update_history(action=f"Read LAS file for project {self.name}")

    def update_data(self, data: pd.DataFrame, group_by: str = "WELL_NAME"):
        logger.info(f"Updating project data with {len(data)} records, grouped by {group_by}")
        for well_name, well_data in data.groupby(group_by):
            logger.debug(f"Processing well: {well_name} with {len(well_data)} records")
            well = Well(well_name)
            well.update_data(well_data)
            self.save_well(well)
        self.update_history(action=f"Added data to project {self.name}")

    def get_all_data(self):
        logger.debug(f"Retrieving all data from {len(self.data)} wells")
        data = pd.DataFrame()
        for well_path in self.data.values():
            well = Well().load(well_path)
            data = pd.concat([data, well.data])
        logger.debug(f"Total data retrieved: {len(data)} records")
        return data

    def get_well_names(self):
        logger.debug(f"Retrieving well names: {list(self.data.keys())}")
        return list(self.data.keys())

    def get_well_data(self, well_name: str):
        logger.debug(f"Retrieving data for well: {well_name}")
        well = Well().load(self.data[well_name])
        return well.data

    def get_well(self, well_name: str):
        logger.debug(f"Loading well object: {well_name}")
        return Well().load(self.data[well_name])

    def export_all_to_parquet(self):
        path = os.path.join(self.output_path, f"{self.name}.parquet")
        logger.info(f"Exporting all project data to parquet: {path}")
        data = self.get_all_data()
        data.to_parquet(path)
        logger.debug(f"Exported {len(data)} records to parquet file")

    def export_all_to_las(self):
        logger.info("Exporting all project data to LAS files")
        data = self.get_all_data()
        for well_name, well_data in data.groupby("WELL_NAME"):
            logger.debug(f"Exporting well {well_name} to LAS format")
            las.export_to_las(well_data=well_data, well_name=well_name, folder=self.output_path)

    def save_well(self, well):
        well_path = os.path.join(self.data_path, f"{well.name}.qppw")
        well.save(well_path)
        self.data.update({well.name: well_path})
        logger.debug(f"Saved well '{well.name}' to {well_path}")
        self.update_history(action=f"Added well {well.name} to project {self.name}")

    def save(self, notes=""):
        path = os.path.join(self.project_path, f"{self.name}.qppp")
        logger.info(f"Saving project to: {path}")
        try:
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.update_history(action=f"Saved project to {path} | {notes}")
        except pickle.PicklingError as e:
            logger.error(f"Pickling error when saving project: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            raise

    def load(self, path: str):
        path = resolve_path(path)
        logger.info(f"Loading project from: {path}")
        try:
            with open(path, "rb") as f:
                project = pickle.load(f)
            return project
        except FileNotFoundError:
            logger.error(f"Project file not found: {path}")
            raise
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error when loading project: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def update_history(self, user=getpass.getuser(), time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action=""):
        self.history.append({"user": user, "time": time, "action": action})
        logger.debug(f"History updated: {action}")

    def update_data_path(self, path: str):
        self.data_path = os.path.abspath(path)
        self.data = {k: os.path.join(self.data_path, f'{k}.qppw') for k, _ in self.data.items()}
        logger.info(f"Updated data path to: {self.data_path}")

    def get_absolute_paths(self):
        """Get absolute paths for all project directories."""
        return {
            'project_path': self.project_path,
            'data_path': self.data_path,
            'output_path': self.output_path
        }

    def validate_paths(self):
        """Validate that all project paths exist and are accessible."""
        paths_to_check = [
            ('project_path', self.project_path),
            ('data_path', self.data_path),
            ('output_path', self.output_path)
        ]

        validation_results = {}
        for name, path in paths_to_check:
            try:
                exists = os.path.exists(path)
                writable = os.access(path, os.W_OK) if exists else False
                validation_results[name] = {
                    'exists': exists,
                    'writable': writable,
                    'path': path
                }
            except Exception as e:
                validation_results[name] = {
                    'exists': False,
                    'writable': False,
                    'error': str(e),
                    'path': path
                }

        return validation_results

    def __str__(self):
        return f"Project: {self.name} - {self.description}"


class WellConfig(object):
    litho_model: str = "ssc"
    dry_sand_point: tuple = Config.SSC_ENDPOINTS["DRY_SAND_POINT"]
    dry_silt_point: tuple = Config.SSC_ENDPOINTS["DRY_SILT_POINT"]
    dry_clay_point: tuple = Config.SSC_ENDPOINTS["DRY_CLAY_POINT"]
    wet_clay_point: tuple = Config.SSC_ENDPOINTS["WET_CLAY_POINT"]
    fluid_point: tuple = Config.SSC_ENDPOINTS["FLUID_POINT"]
    dry_calc_point: tuple = Config.CARB_NEU_DEN_ENDPOINTS["DRY_CALC_POINT"]
    dry_dolo_point: tuple = Config.CARB_NEU_DEN_ENDPOINTS["DRY_DOLO_POINT"]
    silt_line_angle: float = Config.SSC_ENDPOINTS["SILT_LINE_ANGLE"]
    sw_water_salinity: int = 6000
    sw_m: float = 2.0
    sw_n: float = 2.0
    hc_corr_angle: int = 50
    hc_buffer: float = 0.0
    ressum_cutoffs: dict = Config.RESSUM_CUTOFFS

    def update(self, config: dict):
        logger.debug(f"Updating well config with {len(config)} parameters")
        for key, value in config.items():
            setattr(self, key, value)
        logger.debug(f"Updated config parameters: {list(config.keys())}")

    def __setattr__(self, name, value):
        if hasattr(self, name):
            # Allow modification of existing attributes
            super().__setattr__(name, value)
        else:
            # Prevent adding new attributes
            raise AttributeError(f"Cannot add new attribute '{name}' to instances of {self.__class__.__name__}")

    def __getstate__(self):
        """Custom pickle state for serialization."""
        return {
            'litho_model': self.litho_model,
            'dry_sand_point': self.dry_sand_point,
            'dry_silt_point': self.dry_silt_point,
            'dry_clay_point': self.dry_clay_point,
            'wet_clay_point': self.wet_clay_point,
            'fluid_point': self.fluid_point,
            'dry_calc_point': self.dry_calc_point,
            'dry_dolo_point': self.dry_dolo_point,
            'silt_line_angle': self.silt_line_angle,
            'sw_water_salinity': self.sw_water_salinity,
            'sw_m': self.sw_m,
            'sw_n': self.sw_n,
            'hc_corr_angle': self.hc_corr_angle,
            'hc_buffer': self.hc_buffer,
            'ressum_cutoffs': self.ressum_cutoffs
        }

    def __setstate__(self, state):
        """Custom pickle state restoration."""
        self.litho_model = state['litho_model']
        self.dry_sand_point = state['dry_sand_point']
        self.dry_silt_point = state['dry_silt_point']
        self.dry_clay_point = state['dry_clay_point']
        self.wet_clay_point = state['wet_clay_point']
        self.fluid_point = state['fluid_point']
        self.dry_calc_point = state['dry_calc_point']
        self.dry_dolo_point = state['dry_dolo_point']
        self.silt_line_angle = state['silt_line_angle']
        self.sw_water_salinity = state['sw_water_salinity']
        self.sw_m = state['sw_m']
        self.sw_n = state['sw_n']
        self.hc_corr_angle = state['hc_corr_angle']
        self.hc_buffer = state['hc_buffer']
        self.ressum_cutoffs = state['ressum_cutoffs']


class Well(object):
    header: pd.DataFrame = pd.DataFrame()
    data: pd.DataFrame = pd.DataFrame()
    ressum: pd.DataFrame = pd.DataFrame()
    depth_uom: str = ""
    config: WellConfig = WellConfig()
    history: list = []

    def __init__(self, name="", description=""):
        self.name = name
        self.description = description
        logger.debug(f"Well object initialized: {name}")

    def __getstate__(self):
        """Custom pickle state for serialization."""
        return {
            'name': self.name,
            'description': self.description,
            'header': self.header,
            'data': self.data,
            'ressum': self.ressum,
            'depth_uom': self.depth_uom,
            'config': self.config,
            'history': self.history
        }

    def __setstate__(self, state):
        """Custom pickle state restoration."""
        self.name = state['name']
        self.description = state['description']
        self.header = state['header']
        self.data = state['data']
        self.ressum = state['ressum']
        self.depth_uom = state['depth_uom']
        self.config = state['config']
        self.history = state['history']

    def read_las(self, path: str):
        logger.info(f"Reading LAS file: {path}")
        with open(path, "rb") as f:
            data, header = las.read_las_files([f])
        header_df = header.T
        self.name = header_df[header_df['mnemonic'] == 'WELL']['value'].values[0].replace("/", "-")
        self.description = f"Well {self.name}"
        self.depth_uom = header_df[header_df['mnemonic'] == 'STRT']['unit'].values[0]
        self.data = data
        self.header = header
        logger.info(f"LAS file loaded: {self.name} with {len(data)} records, depth unit: {self.depth_uom}")
        self.update_history(action=f"Read LAS file {path}")

    def update_data(self, data: pd.DataFrame):
        logger.debug(f"Updating well data: {len(data)} records")
        self.data = data
        self.update_history(action=f"Updated data for well {self.name}")

    def update_ressum(self, data: pd.DataFrame):
        logger.debug(f"Updating reservoir summary: {len(data)} records")
        self.ressum = data
        self.update_history(action=f"Updated ressum for well {self.name}")

    def update_config(self, config: dict):
        logger.debug(f"Updating well configuration with {len(config)} parameters")
        self.config.update(config)
        self.update_history(action=f"Updated config for well {self.name}")

    def export_to_parquet(self, folder=None):
        if folder is None:
            folder = get_default_output_path()
        else:
            folder = resolve_path(folder)

        try:
            os.makedirs(folder, exist_ok=True)
            path = os.path.join(folder, f"{self.name}.parquet")
            logger.info(f"Exporting well data to parquet: {path}")
            self.data.to_parquet(path)
            logger.debug(f"Exported {len(self.data)} records to parquet")
        except Exception as e:
            logger.error(f"Error exporting to parquet: {e}")
            raise

    def export_to_las(self, folder=None):
        if folder is None:
            folder = get_default_output_path()
        else:
            folder = resolve_path(folder)

        try:
            os.makedirs(folder, exist_ok=True)
            logger.info(f"Exporting well data to LAS format in folder: {folder}")
            las.export_to_las(well_data=self.data, well_name=self.name, folder=folder)
        except Exception as e:
            logger.error(f"Error exporting to LAS: {e}")
            raise

    def save(self, path, notes=""):
        path = resolve_path(path)
        path = path if path.endswith(".qppw") else f"{path}.qppw"

        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(path), exist_ok=True)
            logger.info(f"Saving well to: {path}")
            with open(path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            self.update_history(action=f"Saved well to {path} | {notes}")
        except pickle.PicklingError as e:
            logger.error(f"Pickling error when saving well: {e}")
            raise
        except Exception as e:
            logger.error(f"Error saving well: {e}")
            raise

    def load(self, path: str):
        path = resolve_path(path)
        logger.info(f"Loading well from: {path}")
        try:
            with open(path, "rb") as f:
                well = pickle.load(f)
            return well
        except FileNotFoundError:
            logger.error(f"Well file not found: {path}")
            raise
        except pickle.UnpicklingError as e:
            logger.error(f"Unpickling error when loading well: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading well: {e}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise

    def update_history(self, user=getpass.getuser(), time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action=""):
        self.history.append({"user": user, "time": time, "action": action})
        logger.debug(f"Well history updated: {action}")

    def __str__(self):
        return f"Well: {self.name} - {self.description}"
