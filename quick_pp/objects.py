from datetime import datetime
import getpass
import pandas as pd
import pickle
import os

import quick_pp.las_handler as las
from quick_pp.config import Config
from quick_pp import logger


class Project(object):
    def __init__(self, name="", description="", project_path=""):
        self.name = name
        self.description = description
        self.data = {}
        self.history = []
        self.project_path = project_path or os.path.join("data", "04_project")
        self.data_path = os.path.join(self.project_path, self.name)
        self.output_path = os.path.join(self.data_path, "outputs")
        os.makedirs(self.output_path, exist_ok=True)
        logger.info(f"Project '{name}' initialized with path: {self.data_path}")

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
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.update_history(action=f"Saved project to {path} | {notes}")

    def load(self, path: str):
        logger.info(f"Loading project from: {path}")
        with open(path, "rb") as f:
            project = pickle.load(f)
        return project

    def update_history(self, user=getpass.getuser(), time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action=""):
        self.history.append({"user": user, "time": time, "action": action})
        logger.debug(f"History updated: {action}")

    def update_data_path(self, path: str):
        self.data_path = path
        self.data = {k: os.path.join(path, self.name, f'{k}.qppw') for k, _ in self.data.items()}
        logger.info(f"Updated data path to: {path}")

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
        if not folder:
            folder = self.output_path
        path = os.path.join(folder, f"{self.name}.parquet")
        logger.info(f"Exporting well data to parquet: {path}")
        self.data.to_parquet(path)
        logger.debug(f"Exported {len(self.data)} records to parquet")

    def export_to_las(self, folder=None):
        if not folder:
            folder = self.output_path
        logger.info(f"Exporting well data to LAS format in folder: {folder}")
        las.export_to_las(well_data=self.data, well_name=self.name, folder=folder)

    def save(self, path, notes=""):
        path = path if path.endswith(".qppw") else f"{path}.qppw"
        logger.info(f"Saving well to: {path}")
        with open(path, "wb") as f:
            pickle.dump(self, f)
        self.update_history(action=f"Saved well to {path} | {notes}")

    def load(self, path: str):
        logger.info(f"Loading well from: {path}")
        with open(path, "rb") as f:
            well = pickle.load(f)
        return well

    def update_history(self, user=getpass.getuser(), time=datetime.now().strftime('%Y-%m-%d %H:%M:%S'), action=""):
        self.history.append({"user": user, "time": time, "action": action})
        logger.debug(f"Well history updated: {action}")

    def __str__(self):
        return f"Well: {self.name} - {self.description}"
