import pandas as pd
import os
from typing import Optional, List, Dict, Any

from sqlalchemy.orm import Session
from sqlalchemy import select

import quick_pp.las_handler as las
from quick_pp.config import Config
from quick_pp import logger
from quick_pp.database.models import (
    Project as ORMProject,
    Well as ORMWell,
    Curve as ORMCurve,
    CurveData as ORMCurveData,  # , User as ORMUser
    FormationTop as ORMFormationTop,
    FluidContact as ORMFluidContact,
    PressureTest as ORMPressureTest,
    CoreSample as ORMCoreSample,
    CoreMeasurement as ORMCoreMeasurement,
    RelativePermeability as ORMRelativePermeability,
    CapillaryPressure as ORMCapillaryPressure,
)


class Project(object):
    def __init__(
        self,
        db_session: Session,
        project_id: Optional[int] = None,
        name: str = "",
        description: str = "",
        created_by_user_id: Optional[int] = None,
    ):
        """Initializes or loads a Project.

        If a `project_id` is provided, it loads an existing project from the database.
        If a `name` is provided, it first checks if a project with that name already
        exists and loads it. If not, it creates a new project instance.

        At least one of `project_id` or `name` must be provided.

        Args:
            db_session (Session): The SQLAlchemy session for database interaction.
            project_id (Optional[int]): The ID of an existing project to load.
            name (str): The name of the project to load or create.
            description (str): A description for a new project.
            created_by_user_id (Optional[int]): The ID of the user creating the project.

        Raises:
            ValueError: If the project_id is not found, or if neither project_id nor name is provided.
        """
        self.db_session = db_session
        self._orm_project: Optional[ORMProject] = None

        if project_id:
            self._orm_project = self.db_session.get(ORMProject, project_id)
            if not self._orm_project:
                raise ValueError(f"Project with ID {project_id} not found.")
            logger.info(f"Project '{self.name}' loaded from DB.")
        elif name:
            # Check if a project with this name already exists
            existing_project = self.db_session.scalar(
                select(ORMProject).filter_by(name=name)
            )
            if existing_project:
                self._orm_project = existing_project
                logger.info(f"Project '{name}' already exists, loaded from DB.")
            else:
                self._orm_project = ORMProject(
                    name=name, description=description, created_by=created_by_user_id
                )
                self.db_session.add(self._orm_project)
                self.db_session.flush()
                logger.info(f"New project '{name}' initialized and added to session.")
        else:
            raise ValueError(
                "Either project_id or name must be provided to initialize a Project."
            )

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
        logger.info(f"Project '{self.name}' saved to database.")

    @classmethod
    def load(cls, db_session: Session, project_id: int):
        """Loads a project from the database by its ID.

        Args:
            db_session (Session): The SQLAlchemy session for database interaction.
            project_id (int): The ID of the project to load.

        Returns:
            Project: An instance of the loaded Project.
        """
        return cls(db_session=db_session, project_id=project_id)

    def read_las(self, file_paths: List[str], depth_uom: Optional[str] = None):
        """Reads one or more LAS files and adds or updates wells in the project.

        For each file, it checks if a well with the same name already exists in the
        project. If it does, the existing well's data is updated using the
        `update_data_from_las_parse` method (full replacement). If not, a new well
        is created and added to the project.

        Args:
            file_paths (List[str]): A list of file paths to the LAS files.
            depth_uom (Optional[str]): The unit of measurement for depth to be used if not
                                       specified in the LAS file.
        """
        logger.info(
            f"Reading {len(file_paths)} LAS files for project '{self.name}' (ID: {self.project_id})"
        )

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
                logger.warning(
                    f"Well '{well_name}' already exists in project '{self.name}'. Skipping or updating."
                )
                # Optionally, load and update the existing well
                well_obj = Well(self.db_session, well_id=existing_well.well_id)
                well_obj.update_data_from_las_parse(well_data, header_data)
            else:
                well_obj = Well(
                    self.db_session,
                    project_id=self.project_id,
                    name=well_name,
                    uwi=uwi,
                    header_data=header_data_dict,
                    depth_uom=depth_uom,
                )
                well_obj.update_data_from_las_parse(well_data, header_data)
                self.db_session.add(well_obj._orm_well)
                self.db_session.flush()

            logger.debug(f"Processed well '{well_obj.name}' for project '{self.name}'.")

    def update_data(
        self,
        data: pd.DataFrame,
        group_by: str = "WELL_NAME",
        well_configs: Optional[dict] = None,
    ):
        """Updates data for multiple wells in the project from a single DataFrame.

        The input DataFrame is grouped by the `group_by` column (typically 'WELL_NAME').
        For each group, it finds the corresponding well in the project and updates its
        data using the `Well.update_data` method (upsert logic).

        Args:
            data (pd.DataFrame): A DataFrame containing the data for multiple wells.
            group_by (str): The column name to group the DataFrame by. Defaults to "WELL_NAME".
            well_configs (Optional[dict]): A dictionary where keys are well names and values
                                           are configuration dictionaries to apply.
        """
        logger.info(
            f"Updating project data with {len(data)} records, grouped by {group_by}"
        )
        for well_name, well_data in data.groupby(group_by):
            logger.debug(f"Processing well: {well_name} with {len(well_data)} records")
            orm_well = self.db_session.scalar(
                select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
            )
            if orm_well:
                well_obj = Well(self.db_session, well_id=orm_well.well_id)
                well_obj.update_data(well_data)
                if well_configs is not None:
                    well_obj.update_config(well_configs.get(well_name, {}))
                well_obj.save()  # Persist the updated data and config to the DB
            else:
                logger.warning(
                    f"Well '{well_name}' not found in project '{self.name}'. Skipping update."
                )

    def get_all_data(self) -> pd.DataFrame:
        """Retrieves all data from all wells in the project into a single DataFrame.

        Returns:
            pd.DataFrame: A concatenated DataFrame containing data from all wells,
                          or an empty DataFrame if the project has no data.
        """
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
        """Retrieves a list of all well names in the project.

        Returns:
            List[str]: A list of well names.
        """
        well_names = [well.name for well in self._orm_project.wells]
        logger.debug(f"Retrieving well names: {well_names}")
        return well_names

    def get_well_data(self, well_name: str) -> pd.DataFrame:
        """Retrieves all data for a specific well in the project.

        Args:
            well_name (str): The name of the well to retrieve data for.

        Returns:
            pd.DataFrame: A DataFrame containing the data for the specified well.

        Raises:
            ValueError: If the well is not found in the project.
        """
        logger.debug(f"Retrieving data for well: {well_name} in project '{self.name}'")
        orm_well = self.db_session.scalar(
            select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
        )
        if orm_well:
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            return well_obj.get_data()
        raise ValueError(f"Well '{well_name}' not found in project '{self.name}'.")

    def get_well(self, well_name: str) -> "Well":
        """Retrieves a Well object for a specific well in the project.

        Args:
            well_name (str): The name of the well to retrieve.

        Returns:
            Well: An instance of the Well class for the specified well.

        Raises:
            ValueError: If the well is not found in the project.
        """
        logger.debug(f"Loading well object: {well_name} in project '{self.name}'")
        orm_well = self.db_session.scalar(
            select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
        )
        if orm_well:
            return Well(self.db_session, well_id=orm_well.well_id)
        raise ValueError(f"Well '{well_name}' not found in project '{self.name}'.")

    def export_all_to_parquet(self, folder: Optional[str] = None):
        """Exports all project data to a single Parquet file.

        Args:
            folder (Optional[str]): The directory to save the file in. Defaults to the
                                    project's output path.
        """
        folder = folder or self.output_path
        path = os.path.join(self.output_path, f"{self.name}.parquet")
        logger.info(f"Exporting all project data to parquet: {path}")
        data = self.get_all_data()
        data.to_parquet(path)
        logger.debug(f"Exported {len(data)} records to parquet file")

    def export_all_to_las(self, folder: Optional[str] = None):
        """Exports each well in the project to a separate LAS file.

        Args:
            folder (Optional[str]): The directory to save the files in. Defaults to the
                                    project's output path.
        """
        folder = folder or self.output_path
        logger.info("Exporting all project data to LAS files")
        data = self.get_all_data()
        for well_name, well_data in data.groupby("WELL_NAME"):
            logger.debug(f"Exporting well {well_name} to LAS format")
            las.export_to_las(well_data=well_data, well_name=well_name, folder=folder)

    def __str__(self):
        return f"Project: {self.name} (ID: {self.project_id}) - {self.description}"


class WellConfig(object):
    """
    A configuration object for a well, holding parameters for petrophysical calculations.
    Provides default values which can be updated from a dictionary.
    """

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
        """Updates configuration attributes from a dictionary.

        Args:
            config (dict): A dictionary of configuration parameters to update.
        """
        logger.debug(f"Updating well config with {len(config)} parameters")
        for key, value in config.items():
            setattr(self, key, value)
        logger.debug(f"Updated config parameters: {list(config.keys())}")

    def __setattr__(self, name, value):
        # Allow modification of existing attributes and adding new ones if needed for flexibility
        # For persistence, only attributes defined in the ORM model's config_data will be saved.
        super().__setattr__(name, value)


class Well(object):
    def __init__(
        self,
        db_session: Session,
        well_id: Optional[int] = None,
        project_id: Optional[int] = None,
        name: str = "",
        uwi: str = "",
        header_data: Optional[Dict[str, Any]] = None,
        depth_uom: Optional[str] = None,
    ):
        """Initializes or loads a Well.

        If a `well_id` is provided, it loads an existing well from the database.
        If `project_id`, `name`, and `uwi` are provided, it first checks if a well
        with that UWI (which is globally unique) exists and loads it. If not, it
        creates a new well instance.

        At least one of `well_id` or (`project_id`, `name`, `uwi`) must be provided.

        Args:
            db_session (Session): The SQLAlchemy session for database interaction.
            well_id (Optional[int]): The ID of an existing well to load.
            project_id (Optional[int]): The ID of the project this well belongs to.
            name (str): The name of the well.
            uwi (str): The Unique Well Identifier.
            header_data (Optional[Dict[str, Any]]): A dictionary of header data for a new well.
            depth_uom (Optional[str]): The depth unit of measurement for a new well.

        Raises:
            ValueError: If the well_id is not found, or if required parameters are missing.
        """
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
                logger.info(
                    f"Well '{name}' (UWI: {uwi}) already exists, loaded from DB."
                )
            else:
                self._orm_well = ORMWell(
                    project_id=project_id,
                    name=name,
                    uwi=uwi,
                    header_data={} if header_data is None else header_data,
                    config_data=self._config_instance.__dict__,
                    depth_uom=depth_uom,
                )
                self.db_session.add(self._orm_well)
                self.db_session.flush()
                logger.info(
                    f"New well '{name}' (UWI: {uwi}) initialized and added to session."
                )
        else:
            raise ValueError(
                "Either well_id or (project_id, name, uwi) must be provided to initialize a Well."
            )

        # Output path for exports (not persisted in DB)
        self.output_path = os.path.join(
            "data", "04_project", self._orm_well.project.name, "outputs"
        )
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
        """Loads a well from the database by its ID.

        Args:
            db_session (Session): The SQLAlchemy session for database interaction.
            well_id (int): The ID of the well to load.

        Returns:
            Well: An instance of the loaded Well.
        """
        return cls(db_session=db_session, well_id=well_id)

    def update_data_from_las_parse(
        self, parsed_data: pd.DataFrame, parsed_header: pd.DataFrame
    ):
        """Updates the well's data and metadata from a parsed LAS file.

        This method performs a complete replacement of the well's data based on the
        provided LAS file content. It will:
        1.  Replace the well's header with the new header information.
        2.  Delete any existing curves in the well that are not present in the new LAS data.
        3.  For curves present in both the well and the new LAS data, it deletes all
            existing data points and replaces them with the new data.
        4.  Create new curves for any mnemonics in the LAS data that do not already
            exist in the well.

        Args:
            parsed_data (pd.DataFrame): DataFrame where the index is 'DEPTH' and columns are curve mnemonics.
            parsed_header (pd.DataFrame): DataFrame containing the header information from the LAS file.
        """
        logger.info(f"Updating well '{self.name}' from LAS parse.")
        self.header_data = parsed_header.to_dict()

        # Get existing curve mnemonics for this well
        existing_mnemonics = {c.mnemonic for c in self._orm_well.curves}
        new_mnemonics = set(parsed_data.columns)
        parsed_data = parsed_data.set_index("DEPTH")

        # Delete curves that are no longer in the new data
        curves_to_delete = [
            c
            for c in self._orm_well.curves
            if c.mnemonic in (existing_mnemonics - new_mnemonics)
        ]
        for curve in curves_to_delete:
            logger.debug(
                f"Deleting old curve '{curve.mnemonic}' from well '{self.name}'."
            )
            self.db_session.delete(curve)

        for mnemonic, values in parsed_data.items():
            orm_curve = next(
                (c for c in self._orm_well.curves if c.mnemonic == mnemonic), None
            )

            # Determine data type from pandas series
            is_numeric = pd.api.types.is_numeric_dtype(values)
            data_type = "numeric" if is_numeric else "text"

            if not orm_curve:
                logger.debug(f"Creating new curve '{mnemonic}' for well '{self.name}'.")
                orm_curve = ORMCurve(
                    well_id=self.well_id,
                    mnemonic=mnemonic,
                    unit=las.get_unit_from_header(parsed_header, mnemonic),
                    description=las.get_descr_from_header(parsed_header, mnemonic),
                    data_type=data_type,
                )
                self._orm_well.curves.append(orm_curve)
            else:
                # Delete existing data points for this curve to replace them.
                for data_point in orm_curve.data:
                    self.db_session.delete(data_point)
                self.db_session.flush()
                orm_curve.data_type = data_type

            # Create new data points
            if is_numeric:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_numeric=value)
                    for depth, value in zip(parsed_data.index, values)
                    if pd.notna(value)
                ]
            else:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_text=value)
                    for depth, value in zip(parsed_data.index, values)
                    if pd.notna(value)
                ]
            orm_curve.data.extend(curve_data_points)

        logger.info(
            f"Well '{self.name}' data updated from LAS parse. {len(parsed_data.columns)} curves processed."
        )

    def update_data(self, data: pd.DataFrame):
        """Updates or inserts curve data for the well from a DataFrame ("upsert").

        This method performs an "upsert" (update or insert) operation. For each
        curve in the input DataFrame, it updates the values at the specified depths.
        If a curve does not exist, it will be created.

        Unlike `update_data_from_las_parse`, this method does not delete the entire
        curve's data. It only modifies the data points for the depths present in the
        input DataFrame, making it efficient for partial updates.

        Args:
            data (pd.DataFrame): DataFrame containing the data to update. Must include
                                 a 'DEPTH' column to be used as the index.
        """
        logger.debug(f"Updating well data for '{self.name}': {len(data)} records")
        data = data.set_index("DEPTH")

        for mnemonic, values in data.items():
            # Determine data type from pandas series
            is_numeric = pd.api.types.is_numeric_dtype(values)
            data_type = "numeric" if is_numeric else "text"

            # Find existing curve or create new one
            orm_curve = next(
                (c for c in self._orm_well.curves if c.mnemonic == mnemonic), None
            )
            if not orm_curve:
                logger.debug(
                    f"Creating new curve '{mnemonic}' for well '{self.name}' during data update."
                )
                orm_curve = ORMCurve(
                    well_id=self.well_id, mnemonic=mnemonic, data_type=data_type
                )
                self._orm_well.curves.append(orm_curve)
            else:
                # Delete existing data points only for the depths being updated.
                depths_to_update = data.index[data[mnemonic].notna()].tolist()
                if depths_to_update:
                    # Find existing data points for this curve at the specified depths
                    data_points_to_delete = (
                        self.db_session.query(ORMCurveData)
                        .filter(
                            ORMCurveData.curve_id == orm_curve.curve_id,
                            ORMCurveData.depth.in_(depths_to_update),
                        )
                        .all()
                    )

                    for data_point in data_points_to_delete:
                        self.db_session.delete(data_point)
                    self.db_session.flush()
                orm_curve.data_type = data_type

            # Create new data points, skipping NaNs
            if is_numeric:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_numeric=value)
                    for depth, value in zip(data.index, values)
                    if pd.notna(value)
                ]
            else:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_text=value)
                    for depth, value in zip(data.index, values)
                    if pd.notna(value)
                ]
            orm_curve.data.extend(curve_data_points)
        logger.debug(f"Updated {len(data.columns)} curves for well '{self.name}'.")

    def update_config(self, config: dict):
        """Updates the in-memory well configuration from a dictionary.

        Note: This only updates the configuration on the Python object.
        You must call the `save()` method to persist these changes to the database.

        Args:
            config (dict): A dictionary of configuration parameters to update.
        """
        logger.debug(f"Updating well configuration with {len(config)} parameters")
        self._config_instance.update(config)
        # The config will be saved to the ORM model when Well.save() is called
        logger.debug(f"Well config for '{self.name}' updated in memory.")

    def get_data(self) -> pd.DataFrame:
        """Retrieves all curve data for the well and returns it as a pandas DataFrame.

        The resulting DataFrame will have 'DEPTH' and 'WELL_NAME' columns, with other
        columns representing the curve mnemonics.
        """
        if not self._orm_well.curves:
            return pd.DataFrame()

        all_curves_data = []
        for curve in self._orm_well.curves:
            if not curve.data:
                continue
            depth = [d.depth for d in curve.data]
            if curve.data_type == "numeric":
                values = [d.value_numeric for d in curve.data]
            else:  # 'text'
                values = [d.value_text for d in curve.data]
            curve_df = pd.DataFrame({curve.mnemonic: values}, index=pd.Index(depth))
            all_curves_data.append(curve_df)

        if not all_curves_data:
            return pd.DataFrame()

        # Join all curve dataframes on their depth index.
        df = pd.concat(all_curves_data, axis=1)
        df.insert(0, "DEPTH", df.index)
        df["WELL_NAME"] = self.name
        logger.debug(f"Retrieved {len(df)} records for well '{self.name}'.")
        return df

    def export_to_parquet(self, folder: Optional[str] = None):
        """Exports the well's data to a Parquet file.

        Args:
            folder (Optional[str]): The directory to save the file in. Defaults to the
                                    well's output path.
        """
        folder = folder or self.output_path
        path = os.path.join(folder, f"{self.name}.parquet")
        logger.info(f"Exporting well data to parquet: {path}")
        data_df = self.get_data()
        data_df.to_parquet(path)
        logger.debug(f"Exported {len(data_df)} records to parquet")

    def export_to_las(self, folder: Optional[str] = None):
        """Exports the well's data to a LAS file.

        Args:
            folder (Optional[str]): The directory to save the file in. Defaults to the
                                    well's output path.
        """
        folder = folder or self.output_path
        logger.info(f"Exporting well data to LAS format in folder: {folder}")
        data_df = self.get_data()
        las.export_to_las(well_data=data_df, well_name=self.name, folder=folder)

    def __str__(self):
        return f"Well: {self.name} (ID: {self.well_id}, UWI: {self.uwi})"

    def add_formation_tops(self, tops: List[Dict[str, Any]]):
        """Adds or updates a list of formation tops for the well.

        Args:
            tops (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                         represents a top and must contain 'name' and 'depth'.
        """
        logger.info(
            f"Adding/updating {len(tops)} formation tops for well '{self.name}'."
        )
        for top_data in tops:
            if not all(k in top_data for k in ["name", "depth"]):
                logger.warning(f"Skipping invalid formation top data: {top_data}")
                continue

            # Check if a top with the same name already exists for this well
            orm_top = self.db_session.scalar(
                select(ORMFormationTop).filter_by(
                    well_id=self.well_id, name=top_data["name"]
                )
            )
            if orm_top:
                # Update existing top
                orm_top.depth = top_data["depth"]
                logger.debug(
                    f"Updated formation top '{top_data['name']}' to depth {top_data['depth']}."
                )
            else:
                # Create new top
                orm_top = ORMFormationTop(
                    well_id=self.well_id,
                    name=top_data["name"],
                    depth=top_data["depth"],
                )
                self.db_session.add(orm_top)
                logger.debug(
                    f"Added new formation top '{top_data['name']}' at depth {top_data['depth']}."
                )

    def get_formation_tops(self) -> pd.DataFrame:
        """Retrieves all formation tops for the well as a DataFrame."""
        tops = self._orm_well.formation_tops
        if not tops:
            return pd.DataFrame(columns=["name", "depth"])

        data = [{"name": top.name, "depth": top.depth} for top in tops]
        df = pd.DataFrame(data).sort_values(by="depth").reset_index(drop=True)
        logger.debug(f"Retrieved {len(df)} formation tops for well '{self.name}'.")
        return df

    def add_fluid_contacts(self, contacts: List[Dict[str, Any]]):
        """Adds or updates a list of fluid contacts for the well.

        Args:
            contacts (List[Dict[str, Any]]): A list of dictionaries, where each
                                              dictionary represents a contact and must
                                              contain 'name' and 'depth'.
        """
        logger.info(
            f"Adding/updating {len(contacts)} fluid contacts for well '{self.name}'."
        )
        for contact_data in contacts:
            if not all(k in contact_data for k in ["name", "depth"]):
                logger.warning(f"Skipping invalid fluid contact data: {contact_data}")
                continue

            orm_contact = self.db_session.scalar(
                select(ORMFluidContact).filter_by(
                    well_id=self.well_id, name=contact_data["name"]
                )
            )
            if orm_contact:
                orm_contact.depth = contact_data["depth"]
                logger.debug(
                    f"Updated fluid contact '{contact_data['name']}' to depth {contact_data['depth']}."
                )
            else:
                orm_contact = ORMFluidContact(well_id=self.well_id, **contact_data)
                self.db_session.add(orm_contact)
                logger.debug(
                    f"Added new fluid contact '{contact_data['name']}' at depth {contact_data['depth']}."
                )

    def get_fluid_contacts(self) -> pd.DataFrame:
        """Retrieves all fluid contacts for the well as a DataFrame."""
        contacts = self._orm_well.fluid_contacts
        if not contacts:
            return pd.DataFrame(columns=["name", "depth"])

        data = [{"name": c.name, "depth": c.depth} for c in contacts]
        df = pd.DataFrame(data).sort_values(by="depth").reset_index(drop=True)
        logger.debug(f"Retrieved {len(df)} fluid contacts for well '{self.name}'.")
        return df

    def add_pressure_tests(self, tests: List[Dict[str, Any]]):
        """Adds or updates a list of pressure tests for the well.

        Args:
            tests (List[Dict[str, Any]]): A list of dictionaries, where each
                                          dictionary represents a test and must contain
                                          'depth' and 'pressure'. 'pressure_uom' is optional.
        """
        logger.info(
            f"Adding/updating {len(tests)} pressure tests for well '{self.name}'."
        )
        for test_data in tests:
            if not all(k in test_data for k in ["depth", "pressure"]):
                logger.warning(f"Skipping invalid pressure test data: {test_data}")
                continue

            orm_test = self.db_session.scalar(
                select(ORMPressureTest).filter_by(
                    well_id=self.well_id, depth=test_data["depth"]
                )
            )
            if orm_test:
                orm_test.pressure = test_data["pressure"]
                orm_test.pressure_uom = test_data.get(
                    "pressure_uom", orm_test.pressure_uom
                )
            else:
                orm_test = ORMPressureTest(well_id=self.well_id, **test_data)
                self.db_session.add(orm_test)

    def get_pressure_tests(self) -> pd.DataFrame:
        """Retrieves all pressure tests for the well as a DataFrame."""
        tests = self._orm_well.pressure_tests
        if not tests:
            return pd.DataFrame(columns=["depth", "pressure", "pressure_uom"])

        data = [
            {"depth": t.depth, "pressure": t.pressure, "pressure_uom": t.pressure_uom}
            for t in tests
        ]
        df = pd.DataFrame(data).sort_values(by="depth").reset_index(drop=True)
        logger.debug(f"Retrieved {len(df)} pressure tests for well '{self.name}'.")
        return df

    def add_core_sample_with_measurements(
        self,
        sample_name: str,
        depth: float,
        measurements: List[Dict[str, Any]],
        description: Optional[str] = None,
        remark: Optional[str] = None,
        relperm_data: Optional[List[Dict[str, Any]]] = None,
        pc_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """Adds a core sample and its associated measurements to the well.

        This is a comprehensive method to add a core plug and all its related
        SCAL data (point measurements, rel-perm, and pc curves) in one go.

        Args:
            sample_name (str): The unique name or ID of the core sample.
            depth (float): The depth from which the sample was taken.
            description (Optional[str]): A geological description of the core sample.
            remark (Optional[str]): A remark about data quality, depth shifts, etc.
            measurements (List[Dict[str, Any]]): List of point measurements.
                Each dict must contain 'property_name', 'value', and optionally 'unit'.
                Example: [{'property_name': 'POR', 'value': 0.21, 'unit': 'v/v'}]
            relperm_data (Optional[List[Dict[str, Any]]]): List of relative permeability points.
                Each dict must contain 'saturation', 'kr', and 'phase'.
            pc_data (Optional[List[Dict[str, Any]]]): List of capillary pressure points.
                Each dict must contain 'saturation', 'pressure', and optionally 'experiment_type' and 'cycle'.
        """
        logger.info(
            f"Adding core sample '{sample_name}' at depth {depth} for well '{self.name}'."
        )

        # Create or get the core sample
        orm_sample = self.db_session.scalar(
            select(ORMCoreSample).filter_by(
                well_id=self.well_id, sample_name=sample_name
            )
        )
        if not orm_sample:
            orm_sample = ORMCoreSample(
                well_id=self.well_id,
                sample_name=sample_name,
                depth=depth,
                description=description,
                remark=remark,
            )
            self.db_session.add(orm_sample)
            self.db_session.flush()  # Ensure we get the sample_id
        else:
            orm_sample.description = description
            orm_sample.remark = remark
            # Clear existing child data for a full replacement
            for m in orm_sample.measurements:
                self.db_session.delete(m)
            for rp in orm_sample.relperm_data:
                self.db_session.delete(rp)
            for pc in orm_sample.capillary_pressure_data:
                self.db_session.delete(pc)
            self.db_session.flush()

        # Add point measurements
        for m_data in measurements:
            orm_measurement = ORMCoreMeasurement(
                sample_id=orm_sample.sample_id, **m_data
            )
            orm_sample.measurements.append(orm_measurement)

        # Add relative permeability data
        if relperm_data:
            for rp_data in relperm_data:
                orm_relperm = ORMRelativePermeability(
                    sample_id=orm_sample.sample_id, **rp_data
                )
                orm_sample.relperm_data.append(orm_relperm)

        # Add capillary pressure data
        if pc_data:
            for pc_point in pc_data:
                orm_pc = ORMCapillaryPressure(
                    sample_id=orm_sample.sample_id, **pc_point
                )
                orm_sample.capillary_pressure_data.append(orm_pc)

        logger.debug(
            f"Successfully added/updated data for core sample '{sample_name}'."
        )

    def get_core_data(self) -> Dict[str, pd.DataFrame]:
        """Retrieves all core data for the well, organized by sample."""
        core_data = {}
        for sample in self._orm_well.core_samples:
            sample_data = {
                "depth": sample.depth,
                "description": sample.description,
                "remark": sample.remark,
            }
            measurements = pd.DataFrame(
                [
                    {"property": m.property_name, "value": m.value, "unit": m.unit}
                    for m in sample.measurements
                ]
            )
            # Consolidate relperm data by saturation
            relperm = pd.DataFrame(
                [
                    {"saturation": rp.saturation, f"kr_{rp.phase}": rp.kr}
                    for rp in sample.relperm_data
                ]
            )
            pc = pd.DataFrame(
                [
                    {
                        "saturation": p.saturation,
                        "pressure": p.pressure,
                        "type": p.experiment_type,
                        "cycle": p.cycle,
                    }
                    for p in sample.capillary_pressure_data
                ]
            )

            if not relperm.empty:
                relperm = (
                    relperm.groupby("saturation")
                    .first()
                    .reset_index()
                    .sort_values("saturation")
                )

            sample_data["measurements"] = measurements
            sample_data["relperm"] = relperm
            sample_data["pc"] = pc

            core_data[sample.sample_name] = sample_data

        logger.debug(
            f"Retrieved core data for {len(core_data)} samples in well '{self.name}'."
        )
        return core_data
