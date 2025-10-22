import pandas as pd
import os
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select

import quick_pp.las_handler as las
from quick_pp.config import Config
from quick_pp import logger
from quick_pp.database.models import (
    Project as ORMProject, Well as ORMWell, Curve as ORMCurve,  CurveData as ORMCurveData  # , User as ORMUser
)


class Project(object):
    def __init__(self, db_session: Session,
                 project_id: Optional[int] = None,
                 name: str = "",
                 description: str = "",
                 created_by_user_id: Optional[int] = None):
        self.db_session = db_session
        self._orm_project: Optional[ORMProject] = None

        if project_id:
            self._orm_project = self.db_session.get(ORMProject, project_id)
            if not self._orm_project:
                raise ValueError(f"Project with ID {project_id} not found.")
            logger.info(f"Project '{self.name}' loaded from DB.")
        elif name:
            # Check if a project with this name already exists
            existing_project = self.db_session.scalar(select(ORMProject).filter_by(name=name))
            if existing_project:
                self._orm_project = existing_project
                logger.info(f"Project '{name}' already exists, loaded from DB.")
            else:
                self._orm_project = ORMProject(name=name, description=description, created_by=created_by_user_id)
                self.db_session.add(self._orm_project)
                self.db_session.flush()
                logger.info(f"New project '{name}' initialized and added to session.")
        else:
            raise ValueError("Either project_id or name must be provided to initialize a Project.")

        # Ensure output path exists for exports (not persisted in DB)
        self.output_path = os.path.join("data", "04_project", self.name, "outputs")
        os.makedirs(self.output_path, exist_ok=True)

    @property
    def project_id(self) -> int:
        return self._orm_project.project_id

    @property
    def name(self) -> str:
        return self._orm_project.name

    @name.setter
    def name(self, value: str):
        self._orm_project.name = value

    @property
    def description(self) -> Optional[str]:
        return self._orm_project.description

    @description.setter
    def description(self, value: Optional[str]):
        self._orm_project.description = value

    def save(self):
        """Persists the project and its associated wells/curves to the database."""
        self.db_session.add(self._orm_project)
        # Commit is handled by the session context manager in DBConnector
        logger.info(f"Project '{self.name}' saved to database.")

    @classmethod
    def load(cls, db_session: Session, project_id: int):
        """Loads a project from the database by ID."""
        return cls(db_session=db_session, project_id=project_id)

    def read_las(self, file_paths: List[str], depth_uom: Optional[str] = None):
        logger.info(f"Reading {len(file_paths)} LAS files for project '{self.name}' (ID: {self.project_id})")

        for file in file_paths:
            logger.debug(f"Processing LAS file: {file}")
            with open(file, "rb") as f:
                well_data, header_data = las.read_las_files([f], depth_uom)
            well_name = las.get_wellname_from_header(header_data)
            uwi = las.get_uwi_from_header(header_data)
            header_data_dict = header_data.to_dict()

            # Check if well already exists in this project
            existing_well = self.db_session.scalar(
                select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
            )
            if existing_well:
                logger.warning(f"Well '{well_name}' already exists in project '{self.name}'. Skipping or updating.")
                # Optionally, load and update the existing well
                well_obj = Well(self.db_session, well_id=existing_well.well_id)
                well_obj.update_data_from_las_parse(well_data, header_data_dict, depth_uom)
            else:
                well_obj = Well(
                    self.db_session,
                    project_id=self.project_id,
                    name=well_name,
                    uwi=uwi,
                    header_data=header_data_dict,
                    depth_uom=depth_uom
                )
                well_obj.update_data_from_las_parse(well_data, header_data_dict, depth_uom)
                self.db_session.add(well_obj._orm_well)
                self.db_session.flush()

            logger.debug(f"Processed well '{well_obj.name}' for project '{self.name}'.")
        # Commit is handled by the session context manager

    def update_data(self, data: pd.DataFrame, group_by: str = "WELL_NAME"):
        logger.info(f"Updating project data with {len(data)} records, grouped by {group_by}")
        for well_name, well_data in data.groupby(group_by):
            logger.debug(f"Processing well: {well_name} with {len(well_data)} records")
            orm_well = self.db_session.scalar(
                select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
            )
            if orm_well:
                well_obj = Well(self.db_session, well_id=orm_well.well_id)
                well_obj.update_data(well_data)
            else:
                logger.warning(f"Well '{well_name}' not found in project '{self.name}'. Skipping update.")
        # Commit is handled by the session context manager

    def get_all_data(self) -> pd.DataFrame:
        logger.debug(f"Retrieving all data from project '{self.name}'")
        all_wells_data = []
        for orm_well in self._orm_project.wells:
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            all_wells_data.append(well_obj.get_data())

        if all_wells_data:
            combined_data = pd.concat(all_wells_data, ignore_index=True)
            logger.debug(f"Total data retrieved: {len(combined_data)} records")
            return combined_data
        logger.debug("No data found for this project.")
        return pd.DataFrame()

    def get_well_names(self) -> List[str]:
        well_names = [well.name for well in self._orm_project.wells]
        logger.debug(f"Retrieving well names: {well_names}")
        return well_names

    def get_well_data(self, well_name: str) -> pd.DataFrame:
        logger.debug(f"Retrieving data for well: {well_name} in project '{self.name}'")
        orm_well = self.db_session.scalar(
            select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
        )
        if orm_well:
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            return well_obj.get_data()
        raise ValueError(f"Well '{well_name}' not found in project '{self.name}'.")

    def get_well(self, well_name: str) -> "Well":
        logger.debug(f"Loading well object: {well_name} in project '{self.name}'")
        orm_well = self.db_session.scalar(
            select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
        )
        if orm_well:
            return Well(self.db_session, well_id=orm_well.well_id)
        raise ValueError(f"Well '{well_name}' not found in project '{self.name}'.")

    def export_all_to_parquet(self, folder: Optional[str] = None):
        folder = folder or self.output_path
        path = os.path.join(self.output_path, f"{self.name}.parquet")
        logger.info(f"Exporting all project data to parquet: {path}")
        data = self.get_all_data()
        data.to_parquet(path)
        logger.debug(f"Exported {len(data)} records to parquet file")

    def export_all_to_las(self, folder: Optional[str] = None):
        folder = folder or self.output_path
        logger.info("Exporting all project data to LAS files")
        data = self.get_all_data()
        for well_name, well_data in data.groupby("WELL_NAME"):
            logger.debug(f"Exporting well {well_name} to LAS format")
            las.export_to_las(well_data=well_data, well_name=well_name, folder=folder)

    def __str__(self):
        return f"Project: {self.name} (ID: {self.project_id}) - {self.description}"


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
        # Allow modification of existing attributes and adding new ones if needed for flexibility
        # For persistence, only attributes defined in the ORM model's config_data will be saved.
        super().__setattr__(name, value)


class Well(object):
    def __init__(self, db_session: Session,
                 well_id: Optional[int] = None,
                 project_id: Optional[int] = None,
                 name: str = "",
                 uwi: str = "",
                 header_data: Optional[Dict[str, Any]] = None,
                 depth_uom: Optional[str] = None):
        self.db_session = db_session
        self._orm_well: Optional[ORMWell] = None
        self._config_instance: WellConfig = WellConfig()

        if well_id:
            self._orm_well = self.db_session.get(ORMWell, well_id)
            if not self._orm_well:
                raise ValueError(f"Well with ID {well_id} not found.")
            self._load_config_from_orm()
            logger.info(f"Well '{self.name}' loaded from DB.")
        elif project_id or name or uwi:
            # Check if a well with this UWI already exists (UWI is unique globally)
            existing_well = self.db_session.scalar(select(ORMWell).filter_by(uwi=uwi))
            if existing_well:
                self._orm_well = existing_well
                self._load_config_from_orm()
                logger.info(f"Well '{name}' (UWI: {uwi}) already exists, loaded from DB.")
            else:
                self._orm_well = ORMWell(
                    project_id=project_id,
                    name=name,
                    uwi=uwi,
                    header_data={} if header_data is None else header_data,
                    config_data=self._config_instance.__dict__,
                    depth_uom=depth_uom
                )
                self.db_session.add(self._orm_well)
                self.db_session.flush()
                logger.info(f"New well '{name}' (UWI: {uwi}) initialized and added to session.")
        else:
            raise ValueError("Either well_id or (project_id, name, uwi) must be provided to initialize a Well.")

        # Output path for exports (not persisted in DB)
        self.output_path = os.path.join("data", "04_project", self._orm_well.project.name, "outputs")
        os.makedirs(self.output_path, exist_ok=True)

    @property
    def well_id(self) -> int:
        return self._orm_well.well_id

    @property
    def name(self) -> str:
        return self._orm_well.name

    @name.setter
    def name(self, value: str):
        self._orm_well.name = value

    @property
    def uwi(self) -> str:
        return self._orm_well.uwi

    @uwi.setter
    def uwi(self, value: str):
        self._orm_well.uwi = value

    @property
    def header_data(self) -> Dict[str, Any]:
        return self._orm_well.header_data

    @header_data.setter
    def header_data(self, value: Dict[str, Any]):
        self._orm_well.header_data = value

    @property
    def config(self) -> WellConfig:
        return self._config_instance

    def _load_config_from_orm(self):
        """Loads config from ORM model's JSON into the WellConfig instance."""
        if self._orm_well and self._orm_well.config_data:
            self._config_instance.update(self._orm_well.config_data)

    def save(self):
        """Persists the well and its associated curves to the database."""
        # Update the ORM well's config_data from the in-memory WellConfig instance
        self._orm_well.config_data = self._config_instance.__dict__
        self.db_session.add(self._orm_well)
        # Commit is handled by the session context manager
        logger.info(f"Well '{self.name}' saved to database.")

    @classmethod
    def load(cls, db_session: Session, well_id: int):
        """Loads a well from the database by ID."""
        return cls(db_session=db_session, well_id=well_id)

    def update_data_from_las_parse(self, parsed_data: pd.DataFrame,
                                   parsed_header: Dict[str, Any],
                                   depth_uom: Optional[str] = None):
        """
        Updates well's header and curve data from LAS parsing results.
        `parsed_data` is expected to be a DataFrame where index is depth and columns are curve mnemonics.
        `parsed_header` is a dictionary of header information.
        """
        logger.info(f"Updating well '{self.name}' from LAS parse.")
        self.header_data = parsed_header

        # Get existing curve mnemonics for this well
        existing_mnemonics = {c.mnemonic for c in self._orm_well.curves}
        new_mnemonics = set(parsed_data.columns)

        # Delete curves that are no longer in the new data
        curves_to_delete = [c for c in self._orm_well.curves if c.mnemonic in (existing_mnemonics - new_mnemonics)]
        for curve in curves_to_delete:
            logger.debug(f"Deleting old curve '{curve.mnemonic}' from well '{self.name}'.")
            self.db_session.delete(curve)

        for mnemonic, values in parsed_data.items():
            orm_curve = next((c for c in self._orm_well.curves if c.mnemonic == mnemonic), None)

            # Determine data type from pandas series
            is_numeric = pd.api.types.is_numeric_dtype(values)
            data_type = 'numeric' if is_numeric else 'text'

            if not orm_curve:
                logger.debug(f"Creating new curve '{mnemonic}' for well '{self.name}'.")
                orm_curve = ORMCurve(
                    well_id=self.well_id,
                    mnemonic=mnemonic,
                    unit=parsed_header.get(mnemonic, {}).get('unit'),
                    description=parsed_header.get(mnemonic, {}).get('description'),
                    data_type=data_type
                )
                self._orm_well.curves.append(orm_curve)
            else:
                # Clear existing data points for this curve to replace them
                orm_curve.data.clear()
                orm_curve.data_type = data_type  # Update data type if it changed

            # Create new data points
            if is_numeric:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_numeric=value)
                    for depth, value in zip(parsed_data.index, values) if pd.notna(value)]
            else:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_text=value)
                    for depth, value in zip(parsed_data.index, values) if pd.notna(value)]
            orm_curve.data.extend(curve_data_points)

        logger.info(f"Well '{self.name}' data updated from LAS parse. {len(parsed_data.columns)} curves processed.")

    def update_data(self, data: pd.DataFrame):
        """Updates curve data for the well from a DataFrame."""
        logger.debug(f"Updating well data for '{self.name}': {len(data)} records")
        # Assuming 'data' DataFrame has a depth index and curve columns

        for mnemonic, values in data.items():
            # Determine data type from pandas series
            is_numeric = pd.api.types.is_numeric_dtype(values)
            data_type = 'numeric' if is_numeric else 'text'

            # Find existing curve or create new one
            orm_curve = next((c for c in self._orm_well.curves if c.mnemonic == mnemonic), None)
            if not orm_curve:
                logger.debug(f"Creating new curve '{mnemonic}' for well '{self.name}' during data update.")
                orm_curve = ORMCurve(
                    well_id=self.well_id,
                    mnemonic=mnemonic,
                    data_type=data_type
                )
                self._orm_well.curves.append(orm_curve)
            else:
                # Clear existing data points to replace them
                orm_curve.data.clear()
                orm_curve.data_type = data_type  # Update data type if it changed

            # Create new data points, skipping NaNs
            if is_numeric:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_numeric=value)
                    for depth, value in zip(data.index, values) if pd.notna(value)
                ]
            else:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_text=value)
                    for depth, value in zip(data.index, values) if pd.notna(value)
                ]
            orm_curve.data.extend(curve_data_points)
        logger.debug(f"Updated {len(data.columns)} curves for well '{self.name}'.")

    def update_config(self, config: dict):
        logger.debug(f"Updating well configuration with {len(config)} parameters")
        self.config.update(config)
        # The config will be saved to the ORM model when Well.save() is called
        logger.debug(f"Well config for '{self.name}' updated in memory.")

    def get_data(self) -> pd.DataFrame:
        """Retrieves all curve data for the well as a pandas DataFrame."""
        if not self._orm_well.curves:
            return pd.DataFrame()

        all_curves_data = []
        for curve in self._orm_well.curves:
            if not curve.data:
                continue
            depths = [d.depth for d in curve.data]
            if curve.data_type == 'numeric':
                values = [d.value_numeric for d in curve.data]
            else:  # 'text'
                values = [d.value_text for d in curve.data]
            curve_df = pd.DataFrame({curve.mnemonic: values}, index=pd.Index(depths, name='DEPTH'), dtype=object)
            all_curves_data.append(curve_df)

        if not all_curves_data:
            return pd.DataFrame()

        # Join all curve dataframes on their depth index.
        # This handles curves with different depth ranges or sampling rates.
        df = pd.concat(all_curves_data, axis=1)
        df.index.name = 'DEPTH'
        df['WELL_NAME'] = self.name
        logger.debug(f"Retrieved {len(df)} records for well '{self.name}'.")
        return df

    def export_to_parquet(self, folder: Optional[str] = None):
        folder = folder or self.output_path
        path = os.path.join(folder, f"{self.name}.parquet")
        logger.info(f"Exporting well data to parquet: {path}")
        data_df = self.get_data()
        data_df.to_parquet(path)
        logger.debug(f"Exported {len(data_df)} records to parquet")

    def export_to_las(self, folder: Optional[str] = None):
        folder = folder or self.output_path
        logger.info(f"Exporting well data to LAS format in folder: {folder}")
        data_df = self.get_data()
        las.export_to_las(well_data=data_df, well_name=self.name, folder=folder)

    def __str__(self):
        return f"Well: {self.name} (ID: {self.well_id}, UWI: {self.uwi})"
