import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import select, text
from sqlalchemy.orm import Session, selectinload

import quick_pp.las_handler as las
from quick_pp import logger
from quick_pp.config import Config
from quick_pp.database.models import (
    CapillaryPressure as ORMCapillaryPressure,
    CoreMeasurement as ORMCoreMeasurement,
    CoreSample as ORMCoreSample,
    Curve as ORMCurve,
    CurveData as ORMCurveData,
    FluidContact as ORMFluidContact,
    FormationTop as ORMFormationTop,
    PressureTest as ORMPressureTest,
    Project as ORMProject,
    RelativePermeability as ORMRelativePermeability,
    Well as ORMWell,
    WellSurvey as ORMWellSurvey,
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

    def read_las(
        self,
        file_paths: List[str],
        depth_uom: Optional[str] = None,
        max_workers: int = 4,
    ):
        """Reads one or more LAS files and adds or updates wells in the project.

        Optimized version that:
        - Parses LAS files in parallel using thread pool
        - Batches database lookups to avoid N+1 queries
        - Single flush operation at the end

        For each file, it checks if a well with the same name already exists in the
        project. If it does, the existing well's data is updated using the
        `update_data_from_las_parse` method (full replacement). If not, a new well
        is created and added to the project.

        Args:
            file_paths (List[str]): A list of file paths to the LAS files.
            depth_uom (Optional[str]): The unit of measurement for depth to be used if not
                                       specified in the LAS file.
            max_workers (int): Maximum number of threads for parallel file parsing. Defaults to 4.
        """
        logger.info(
            f"Reading {len(file_paths)} LAS files for project '{self.name}' (ID: {self.project_id})"
        )

        # Optimization 1: Parse LAS files in parallel
        def parse_las_file(file_path: str):
            """Parse a single LAS file and return parsed data."""
            try:
                logger.debug(f"Parsing LAS file: {file_path}")
                with open(file_path, "rb") as f:
                    well_data, header_data = las.read_las_files([f], depth_uom)
                well_name = las.get_wellname_from_header(header_data)
                uwi = las.get_uwi_from_header(header_data)
                header_data_dict = header_data.to_dict()
                return {
                    "file_path": file_path,
                    "well_name": well_name,
                    "uwi": uwi,
                    "well_data": well_data,
                    "header_data": header_data,
                    "header_data_dict": header_data_dict,
                    "success": True,
                }
            except Exception as e:
                logger.error(f"Failed to parse LAS file {file_path}: {e}")
                return {"file_path": file_path, "success": False, "error": str(e)}

        parsed_files = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(parse_las_file, fp): fp for fp in file_paths}
            for future in as_completed(futures):
                result = future.result()
                if result["success"]:
                    parsed_files.append(result)
                else:
                    logger.warning(
                        f"Skipping file {result['file_path']} due to parse error: {result.get('error')}"
                    )

        if not parsed_files:
            logger.warning("No LAS files were successfully parsed.")
            return {
                "processed_files": 0,
                "new_wells": [],
                "updated_wells": [],
            }

        # Optimization 2: Batch fetch all existing wells for this project
        well_names = [pf["well_name"] for pf in parsed_files]
        existing_wells = (
            self.db_session.execute(
                select(ORMWell).filter(
                    ORMWell.project_id == self.project_id, ORMWell.name.in_(well_names)
                )
            )
            .scalars()
            .all()
        )

        # Create lookup dictionary for O(1) access
        wells_by_name = {well.name: well for well in existing_wells}

        # Optimization 3: Process all files and batch database operations
        new_wells = []
        for parsed in parsed_files:
            well_name = parsed["well_name"]
            existing_well = wells_by_name.get(well_name)

            if existing_well:
                logger.warning(
                    f"Well '{well_name}' already exists in project '{self.name}'. Updating data."
                )
                well_obj = Well(self.db_session, well_id=existing_well.well_id)
                well_obj.update_data_from_las_parse(
                    parsed["well_data"], parsed["header_data"]
                )
            else:
                logger.debug(
                    f"Creating new well '{well_name}' in project '{self.name}'"
                )
                well_obj = Well(
                    self.db_session,
                    project_id=self.project_id,
                    name=well_name,
                    uwi=parsed["uwi"],
                    header_data=parsed["header_data_dict"],
                    depth_uom=depth_uom,
                )
                well_obj.update_data_from_las_parse(
                    parsed["well_data"], parsed["header_data"]
                )
                self.db_session.add(well_obj._orm_well)
                new_wells.append(well_name)

            logger.debug(f"Processed well '{well_obj.name}' for project '{self.name}'.")

        # Optimization 4: Single flush for all wells at the end
        processed_count = len(parsed_files)
        if new_wells or parsed_files:
            self.db_session.flush()
            logger.info(
                f"Successfully processed {processed_count} LAS files "
                f"({len(new_wells)} new wells, {processed_count - len(new_wells)} updated)"
            )

        return {
            "processed_files": processed_count,
            "new_wells": new_wells,
            "updated_wells": processed_count - len(new_wells),
        }

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

        # Optimization 1: Batch fetch all wells at once to avoid N+1 queries
        well_names = data[group_by].unique().tolist()
        orm_wells = (
            self.db_session.execute(
                select(ORMWell).filter(
                    ORMWell.project_id == self.project_id, ORMWell.name.in_(well_names)
                )
            )
            .scalars()
            .all()
        )

        # Create a lookup dictionary for O(1) access
        wells_by_name = {well.name: well for well in orm_wells}

        # Optimization 2: Process all wells, but only flush/commit once at the end
        for well_name, well_data in data.groupby(group_by):
            logger.debug(f"Processing well: {well_name} with {len(well_data)} records")
            orm_well = wells_by_name.get(well_name)

            if orm_well:
                well_obj = Well(self.db_session, well_id=orm_well.well_id)
                well_obj.update_data(well_data)
                if well_configs is not None:
                    well_obj.update_config(well_configs.get(well_name, {}))
                # Don't call save() here - will flush once at the end
                self.db_session.add(well_obj._orm_well)
            else:
                logger.warning(
                    f"Well '{well_name}' not found in project '{self.name}'. Skipping update."
                )

        # Optimization 3: Single flush for all wells
        self.db_session.flush()
        logger.info(f"Successfully updated {len(wells_by_name)} wells")

    def update_ancillary_data(self, data: Dict[str, Dict[str, List[Dict[str, Any]]]]):
        """Updates ancillary data for multiple wells in the project.

        This method iterates through a dictionary where each key is a well name.
        The value for each well is another dictionary containing the ancillary data
        to be updated, keyed by data type (e.g., 'formation_tops').

        Args:
            data (Dict[str, Dict[str, List[Dict[str, Any]]]]):
                A dictionary where keys are well names. The values are dictionaries
                where keys are the type of ancillary data ('formation_tops',
                'fluid_contacts', 'pressure_tests', 'core_data') and values are
                the data in the format expected by the corresponding 'add_*'
                methods in the Well class.

        Example:
            project.update_ancillary_data({
                "Well-A": {
                    "formation_tops": [{"name": "Top-1", "depth": 1000}],
                    "pressure_tests": [{"depth": 1050, "pressure": 2500}]
                },
                "Well-B": {
                    "fluid_contacts": [{"name": "GOC", "depth": 2100}]
                }
            })
        """
        logger.info(f"Updating ancillary data for {len(data)} wells.")
        for well_name, ancillary_data in data.items():
            logger.debug(f"Processing ancillary data for well: {well_name}")
            try:
                well_obj = self.get_well(well_name)
            except ValueError:
                logger.warning(
                    f"Well '{well_name}' not found in project '{self.name}'. Skipping ancillary data update."
                )
                continue

            if "formation_tops" in ancillary_data:
                well_obj.add_formation_tops(ancillary_data["formation_tops"])
            if "fluid_contacts" in ancillary_data:
                well_obj.add_fluid_contacts(ancillary_data["fluid_contacts"])
            if "pressure_tests" in ancillary_data:
                well_obj.add_pressure_tests(ancillary_data["pressure_tests"])
            if "well_surveys" in ancillary_data:
                well_obj.add_well_surveys(ancillary_data["well_surveys"])
            if "core_data" in ancillary_data:
                for core_sample_data in ancillary_data["core_data"]:
                    well_obj.add_core_sample_with_measurements(**core_sample_data)

            # Persist the changes for the well
            well_obj.save()

        logger.info("Ancillary data update process completed.")

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

    def get_well_data_optimized(self, well_name: str) -> pd.DataFrame:
        """Optimized version of get_well_data using raw SQL for better performance.

        Args:
            well_name (str): The name of the well to retrieve data for.

        Returns:
            pd.DataFrame: A DataFrame containing the data for the specified well.

        Raises:
            ValueError: If the well is not found in the project.
        """
        logger.debug(
            f"Retrieving data (optimized) for well: {well_name} in project '{self.name}'"
        )
        orm_well = self.db_session.scalar(
            select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
        )
        if orm_well:
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            return well_obj.get_data_optimized()
        raise ValueError(f"Well '{well_name}' not found in project '{self.name}'.")

    def get_well_ancillary_data(self, well_name: str) -> Dict[str, Any]:
        """Retrieves all ancillary (non-curve) data for a specific well.

        Args:
            well_name (str): The name of the well to retrieve ancillary data for.

        Returns:
            Dict[str, Any]: A dictionary containing ancillary data like formation
                            tops, core data, etc.

        Raises:
            ValueError: If the well is not found in the project.
        """
        logger.debug(
            f"Retrieving ancillary data for well: {well_name} in project '{self.name}'"
        )
        orm_well = self.db_session.scalar(
            select(ORMWell).filter_by(project_id=self.project_id, name=well_name)
        )
        if orm_well:
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            return well_obj.get_ancillary_data()
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
            existing_well = None

            # Prefer match by project + name when both provided
            if project_id and name:
                existing_well = self.db_session.scalar(
                    select(ORMWell).filter_by(project_id=project_id, name=name)
                )

            # Next prefer match by project + uwi when provided
            if not existing_well and project_id and uwi:
                existing_well = self.db_session.scalar(
                    select(ORMWell).filter_by(project_id=project_id, uwi=uwi)
                )

            # Lastly, check for global match by UWI only
            if not existing_well and uwi:
                global_match = self.db_session.scalar(
                    select(ORMWell).filter_by(uwi=uwi)
                )
                if global_match:
                    logger.warning(
                        "Found existing well with same UWI in a different project; "
                        "creating a separate well record for this project."
                    )

            if existing_well:
                self._orm_well = existing_well
                self._load_config_from_orm()
                logger.info(f"Well '{name}' (UWI: {uwi}) loaded from DB.")
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
            try:
                self._orm_well.curves.remove(curve)
            except ValueError:
                # If it's already removed from the collection, ignore
                pass
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
                    for depth, value in zip(parsed_data.index, values, strict=True)
                    if pd.notna(value)
                ]
            else:
                curve_data_points = [
                    ORMCurveData(depth=depth, value_text=value)
                    for depth, value in zip(parsed_data.index, values, strict=True)
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

        # Optimization 1: Create a lookup dictionary for existing curves
        curves_by_mnemonic = {c.mnemonic: c for c in self._orm_well.curves}

        # Optimization 2: Batch collect all depths that need deletion across all curves
        all_depths_to_delete = {}
        new_curves = []

        for mnemonic, values in data.items():
            # Determine data type from pandas series
            is_numeric = pd.api.types.is_numeric_dtype(values)
            data_type = "numeric" if is_numeric else "text"

            # Find existing curve or create new one
            orm_curve = curves_by_mnemonic.get(mnemonic)

            if not orm_curve:
                logger.debug(
                    f"Creating new curve '{mnemonic}' for well '{self.name}' during data update."
                )
                orm_curve = ORMCurve(
                    well_id=self.well_id, mnemonic=mnemonic, data_type=data_type
                )
                self._orm_well.curves.append(orm_curve)
                self.db_session.flush()  # Flush to get curve_id
                curves_by_mnemonic[mnemonic] = orm_curve
                new_curves.append(mnemonic)
            else:
                orm_curve.data_type = data_type
                # Collect depths for this curve that need deletion
                depths_to_update = data.index[data[mnemonic].notna()].tolist()
                if depths_to_update:
                    all_depths_to_delete[orm_curve.curve_id] = depths_to_update

        # Optimization 3: Bulk delete old data points using raw SQL for better performance
        if all_depths_to_delete:
            for curve_id, depths in all_depths_to_delete.items():
                # Use bulk delete with raw SQL for much better performance
                self.db_session.execute(
                    text(
                        "DELETE FROM curve_data WHERE curve_id = :curve_id AND depth IN :depths"
                    ).bindparams(curve_id=curve_id),
                    {"curve_id": curve_id, "depths": tuple(depths)},
                )

        # Optimization 4: Batch insert new data points using bulk_insert_mappings
        all_data_points = []
        for mnemonic, values in data.items():
            orm_curve = curves_by_mnemonic[mnemonic]
            is_numeric = pd.api.types.is_numeric_dtype(values)

            # Create data point mappings for bulk insert
            if is_numeric:
                data_points = [
                    {
                        "curve_id": orm_curve.curve_id,
                        "depth": depth,
                        "value_numeric": value,
                        "value_text": None,
                    }
                    for depth, value in zip(data.index, values, strict=True)
                    if pd.notna(value)
                ]
            else:
                data_points = [
                    {
                        "curve_id": orm_curve.curve_id,
                        "depth": depth,
                        "value_numeric": None,
                        "value_text": str(value),
                    }
                    for depth, value in zip(data.index, values, strict=True)
                    if pd.notna(value)
                ]
            all_data_points.extend(data_points)

        # Optimization 5: Single bulk insert for all curves
        if all_data_points:
            self.db_session.bulk_insert_mappings(ORMCurveData, all_data_points)

        logger.debug(
            f"Updated {len(data.columns)} curves for well '{self.name}' "
            f"with {len(all_data_points)} total data points."
        )

    def add_curve_data(
        self,
        mnemonic: str,
        depth_value_map: Dict[float, Any],
        unit: Optional[str] = None,
    ):
        """Adds or updates curve data from a dictionary mapping depths to values.

        This is a convenience method for adding curve data when you have
        a dictionary of depth-value pairs (e.g., {100.5: 2500.3, 101.0: 2501.2})
        rather than a full DataFrame.

        Args:
            mnemonic (str): The curve mnemonic (e.g., 'TVD', 'RES_DEPTH')
            depth_value_map (Dict[float, Any]): Dictionary mapping depths to values
            unit (Optional[str]): The unit for this curve (e.g., 'm', 'ft')
        """
        if not depth_value_map:
            logger.warning(f"No data provided for curve '{mnemonic}'")
            return

        logger.debug(
            f"Adding curve data for '{mnemonic}' with {len(depth_value_map)} points"
        )

        # Determine data type
        sample_value = next(iter(depth_value_map.values()))
        is_numeric = isinstance(sample_value, (int, float)) and not isinstance(
            sample_value, bool
        )
        data_type = "numeric" if is_numeric else "text"

        # Find or create the curve
        orm_curve = next(
            (c for c in self._orm_well.curves if c.mnemonic == mnemonic), None
        )
        if not orm_curve:
            logger.debug(f"Creating new curve '{mnemonic}' for well '{self.name}'")
            orm_curve = ORMCurve(
                well_id=self.well_id, mnemonic=mnemonic, unit=unit, data_type=data_type
            )
            self._orm_well.curves.append(orm_curve)
        else:
            # Update unit if provided
            if unit:
                orm_curve.unit = unit
            orm_curve.data_type = data_type
            # Delete existing data points for these depths
            depths_to_update = list(depth_value_map.keys())
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

        # Add new data points
        if is_numeric:
            curve_data_points = [
                ORMCurveData(depth=depth, value_numeric=value)
                for depth, value in depth_value_map.items()
                if value is not None
            ]
        else:
            curve_data_points = [
                ORMCurveData(depth=depth, value_text=str(value))
                for depth, value in depth_value_map.items()
                if value is not None
            ]
        orm_curve.data.extend(curve_data_points)
        logger.debug(
            f"Added {len(curve_data_points)} data points for curve '{mnemonic}'"
        )

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

    def get_data_optimized(self) -> pd.DataFrame:
        """Optimized data retrieval using raw SQL for better performance.

        This method bypasses the ORM and uses raw SQL to efficiently retrieve
        all curve data for the well. It's significantly faster than the original
        get_data() method for wells with large amounts of data.

        Returns:
            pd.DataFrame: A DataFrame with 'DEPTH' and 'WELL_NAME' columns,
                         plus columns for each curve mnemonic.
        """
        sql = text("""
        SELECT
            cd.depth,
            c.mnemonic,
            COALESCE(CAST(cd.value_numeric AS TEXT), cd.value_text) as value
        FROM curve_data cd
        JOIN curves c ON cd.curve_id = c.curve_id
        WHERE c.well_id = :well_id
        ORDER BY cd.depth, c.mnemonic
        """)

        result = self.db_session.execute(sql, {"well_id": self.well_id})
        rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        # Convert to DataFrame and pivot
        df = pd.DataFrame(rows, columns=["depth", "mnemonic", "value"])
        pivoted = df.pivot(index="depth", columns="mnemonic", values="value")

        # Convert numeric columns back to proper data types
        for col in pivoted.columns:
            pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce")

        # Add standard columns
        pivoted.reset_index(inplace=True)
        pivoted.rename(columns={"depth": "DEPTH"}, inplace=True)
        pivoted["WELL_NAME"] = self.name

        logger.debug(
            f"Retrieved {len(pivoted)} records for well '{self.name}' (optimized)."
        )
        return pivoted

    def get_data_with_eager_loading(self) -> pd.DataFrame:
        """Optimized version of get_data using SQLAlchemy eager loading.

        This method uses selectinload to avoid N+1 queries when loading
        curve data through the ORM.

        Returns:
            pd.DataFrame: A DataFrame with 'DEPTH' and 'WELL_NAME' columns,
                         plus columns for each curve mnemonic.
        """
        # Eager load curves and their data to avoid N+1 queries
        stmt = (
            select(ORMWell)
            .options(selectinload(ORMWell.curves).selectinload(ORMCurve.data))
            .where(ORMWell.well_id == self.well_id)
        )
        orm_well_with_data = self.db_session.execute(stmt).scalar_one()

        if not orm_well_with_data.curves:
            return pd.DataFrame()

        all_curves_data = []
        for curve in orm_well_with_data.curves:
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
        logger.debug(
            f"Retrieved {len(df)} records for well '{self.name}' (eager loaded)."
        )
        return df

    def get_curve_data(self, mnemonics: List[str]) -> pd.DataFrame:
        """Get data for specific curves only - much faster for plotting.

        This method is optimized for cases where you only need a subset of curves,
        such as for plotting operations that only require NPHI and RHOB.

        Args:
            mnemonics (List[str]): List of curve mnemonics to retrieve.

        Returns:
            pd.DataFrame: DataFrame with depth as index and requested curves as columns.
        """
        if not mnemonics:
            return pd.DataFrame()

        # Create placeholders for IN clause
        mnemonic_params = {
            f"mnemonic_{i}": mnemonic for i, mnemonic in enumerate(mnemonics)
        }
        placeholders = ", ".join([f":mnemonic_{i}" for i in range(len(mnemonics))])

        sql = text(f"""
        SELECT
            cd.depth,
            c.mnemonic,
            COALESCE(CAST(cd.value_numeric AS TEXT), cd.value_text) as value
        FROM curve_data cd
        JOIN curves c ON cd.curve_id = c.curve_id
        WHERE c.well_id = :well_id
        AND c.mnemonic IN ({placeholders})
        ORDER BY cd.depth
        """)

        params = {"well_id": self.well_id, **mnemonic_params}
        result = self.db_session.execute(sql, params)

        rows = result.fetchall()
        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=["depth", "mnemonic", "value"])
        pivoted = df.pivot(index="depth", columns="mnemonic", values="value")

        # Convert numeric columns back to proper data types
        for col in pivoted.columns:
            pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce")

        pivoted.reset_index(inplace=True)

        logger.debug(
            f"Retrieved {len(pivoted)} records for {len(mnemonics)} curves in well '{self.name}'."
        )
        return pivoted

    def find_curve_mnemonics(self, search_terms: List[str]) -> Dict[str, str]:
        """Find actual curve mnemonics that match search terms (case-insensitive).

        This helper method searches for curve mnemonics that contain the search terms,
        useful for finding NPHI, RHOB, etc. when you don't know the exact naming.

        Args:
            search_terms (List[str]): List of search terms (e.g., ['nphi', 'rhob']).

        Returns:
            Dict[str, str]: Dictionary mapping search terms to actual mnemonics found.
        """
        found_mnemonics = {}

        for term in search_terms:
            sql = text("""
                SELECT mnemonic FROM curves
                WHERE well_id = :well_id
                AND LOWER(mnemonic) LIKE :pattern
                LIMIT 1
            """)

            result = self.db_session.execute(
                sql, {"well_id": self.well_id, "pattern": f"%{term.lower()}%"}
            )

            mnemonic = result.scalar()
            if mnemonic:
                found_mnemonics[term] = mnemonic

        return found_mnemonics

    def get_data_cached(self) -> pd.DataFrame:
        """Cached version of get_data for repeated access.

        This method implements a simple caching mechanism based on the well's
        last update timestamp to avoid repeated expensive database queries.

        Returns:
            pd.DataFrame: Cached or fresh DataFrame with well data.
        """
        # Create cache key based on well and last update time
        cache_key = f"{self.well_id}_{self._orm_well.updated_at}"

        # Initialize cache if not present
        if not hasattr(self, "_data_cache"):
            self._data_cache = {}

        # Return cached data if available, otherwise fetch and cache
        if cache_key not in self._data_cache:
            self._data_cache[cache_key] = self.get_data_optimized()

        return self._data_cache[cache_key]

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

    def add_well_surveys(self, surveys: List[Dict[str, Any]]):
        """Adds or updates a list of well survey points for the well.

        Args:
            surveys (List[Dict[str, Any]]): A list of dictionaries, where each
                                            dictionary represents a survey point and must contain
                                            'md', 'inc', and 'azim'.
        """
        logger.info(
            f"Adding/updating {len(surveys)} well survey points for well '{self.name}'."
        )
        for survey_data in surveys:
            if not all(k in survey_data for k in ["md", "inc", "azim"]):
                logger.warning(f"Skipping invalid well survey data: {survey_data}")
                continue

            orm_survey = self.db_session.scalar(
                select(ORMWellSurvey).filter_by(
                    well_id=self.well_id, md=survey_data["md"]
                )
            )
            if orm_survey:
                orm_survey.inc = survey_data["inc"]
                orm_survey.azim = survey_data["azim"]
            else:
                orm_survey = ORMWellSurvey(well_id=self.well_id, **survey_data)
                self.db_session.add(orm_survey)

    def get_well_surveys(self) -> pd.DataFrame:
        """Retrieves all well survey points for the well as a DataFrame."""
        surveys = self._orm_well.survey_points
        if not surveys:
            return pd.DataFrame(columns=["md", "inc", "azim"])

        data = [{"md": s.md, "inc": s.inc, "azim": s.azim} for s in surveys]
        df = pd.DataFrame(data).sort_values(by="md").reset_index(drop=True)
        logger.debug(f"Retrieved {len(df)} well survey points for well '{self.name}'.")
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

    @staticmethod
    def transform_wide_core_df_to_long(
        df: pd.DataFrame,
        id_cols: List[str],
        property_cols: List[str],
        unit_map: Dict[str, str],
        sample_name_col: str = "SAMPLE_NAME",
    ) -> pd.DataFrame:
        """Transforms a wide-format core data DataFrame to a long format.

        This is a static helper method to unpivot a DataFrame where each row is a
        unique core sample and measurements (like porosity and permeability) are in
        separate columns. The output is a long-format DataFrame compatible with
        `prepare_core_data_from_df`.

        Example of a wide DataFrame:
           SAMPLE_NAME   DEPTH  CORE_POROSITY  CORE_PERMEABILITY
        0      Plug-1A  2500.5           15.5              120.3
        1      Plug-1B  2501.0            5.2                0.5

        Args:
            df (pd.DataFrame): The wide-format DataFrame.
            id_cols (List[str]): A list of column names to keep as identifier
                variables (e.g., ['SAMPLE_NAME', 'DEPTH', 'DESCRIPTION']).
            property_cols (List[str]): A list of column names to unpivot. These
                are the columns containing the measurement values.
            unit_map (Dict[str, str]): A dictionary mapping the property column
                names to their corresponding units.
                Example: {'CORE_POROSITY': '%', 'CORE_PERMEABILITY': 'mD'}
            sample_name_col (str): The name of the column in `id_cols` that
                uniquely identifies the core sample. Defaults to 'SAMPLE_NAME'.

        Returns:
            pd.DataFrame: A long-format DataFrame with 'PROPERTY_NAME', 'VALUE',
                          and 'UNIT' columns.
        """
        if not all(col in df.columns for col in id_cols + property_cols):
            raise ValueError("One or more specified columns not found in DataFrame.")

        if sample_name_col not in id_cols:
            raise ValueError(
                f"The sample_name_col '{sample_name_col}' must be in id_cols."
            )

        # Rename sample identifier column to the expected 'SAMPLE_NAME' if different
        if sample_name_col != "SAMPLE_NAME":
            df = df.rename(columns={sample_name_col: "SAMPLE_NAME"})
            id_cols[id_cols.index(sample_name_col)] = "SAMPLE_NAME"

        long_df = pd.melt(
            df,
            id_vars=id_cols,
            value_vars=property_cols,
            var_name="PROPERTY_NAME",
            value_name="VALUE",
        )

        long_df["UNIT"] = long_df["PROPERTY_NAME"].map(unit_map)
        return long_df

    @staticmethod
    def prepare_core_data_from_df(
        df: pd.DataFrame,
    ) -> List[Dict[str, Any]]:
        """Transforms a core data DataFrame into the structured format for loading.

        This is a static helper method to convert a flat DataFrame containing various
        types of core data (measurements, relperm, pc) into the nested list of
        dictionaries required by `add_core_sample_with_measurements` or
        `Project.update_ancillary_data`.

        The input DataFrame should be structured with columns like:
        - 'SAMPLE_NAME': Unique identifier for the core sample.
        - 'DEPTH': Depth of the sample.
        - 'DESCRIPTION', 'REMARK': Optional sample-level information.
        - 'PROPERTY_NAME', 'VALUE', 'UNIT': For point core measurements (e.g., POR, PERM).
        - 'RP_SAT', 'RP_KR', 'RP_PHASE': For relative permeability data.
        - 'PC_SAT', 'PC_PRESSURE', 'PC_TYPE', 'PC_CYCLE': For capillary pressure data.

        Rows with the same 'SAMPLE_NAME' will be grouped together.

        Args:
            df (pd.DataFrame): A DataFrame containing the core data.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                  represents a core sample and its associated data,
                                  ready to be used with `add_core_sample_with_measurements`.
        """
        core_data_list = []
        required_cols = ["SAMPLE_NAME", "DEPTH"]
        if not all(col in df.columns for col in required_cols):
            raise ValueError(
                f"Input DataFrame must contain required columns: {required_cols}"
            )

        for sample_name, group in df.groupby("SAMPLE_NAME"):
            first_row = group.iloc[0]
            sample_dict = {
                "sample_name": sample_name,
                "depth": first_row["DEPTH"],
                "description": first_row.get("DESCRIPTION"),
                "remark": first_row.get("REMARK"),
            }

            # Process point measurements
            measurements_df = group[
                group["PROPERTY_NAME"].notna() & group["VALUE"].notna()
            ][["PROPERTY_NAME", "VALUE", "UNIT"]].drop_duplicates()
            sample_dict["measurements"] = [
                {"property_name": r.PROPERTY_NAME, "value": r.VALUE, "unit": r.UNIT}
                for r in measurements_df.itertuples()
            ]

            # Process relative permeability data
            relperm_df = group[group["RP_SAT"].notna() & group["RP_KR"].notna()][
                ["RP_SAT", "RP_KR", "RP_PHASE"]
            ].drop_duplicates()
            sample_dict["relperm_data"] = [
                {"saturation": r.RP_SAT, "kr": r.RP_KR, "phase": r.RP_PHASE}
                for r in relperm_df.itertuples()
            ]

            # Process capillary pressure data
            pc_df = group[group["PC_SAT"].notna() & group["PC_PRESSURE"].notna()][
                ["PC_SAT", "PC_PRESSURE", "PC_TYPE", "PC_CYCLE"]
            ].drop_duplicates()
            sample_dict["pc_data"] = [
                {
                    "saturation": r.PC_SAT,
                    "pressure": r.PC_PRESSURE,
                    "experiment_type": r.PC_TYPE,
                    "cycle": r.PC_CYCLE,
                }
                for r in pc_df.itertuples()
            ]

            core_data_list.append(sample_dict)
        return core_data_list

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

    def get_ancillary_data(self) -> Dict[str, Any]:
        """Retrieves all ancillary (non-curve) data for the well.

        This includes formation tops, fluid contacts, pressure tests, and core data,
        returned in a structured dictionary.

        Returns:
            Dict[str, Any]: A dictionary where keys are the data type (e.g.,
                            'formation_tops', 'core_data') and values are the
                            corresponding data, typically as pandas DataFrames or dicts.
        """
        logger.info(f"Retrieving all ancillary data for well '{self.name}'.")
        ancillary_data = {
            "formation_tops": self.get_formation_tops(),
            "fluid_contacts": self.get_fluid_contacts(),
            "pressure_tests": self.get_pressure_tests(),
            "well_surveys": self.get_well_surveys(),
            "core_data": self.get_core_data(),
        }
        return ancillary_data
