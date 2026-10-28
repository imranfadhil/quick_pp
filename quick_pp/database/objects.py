import io
import numpy as np
import os
import re
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from sqlalchemy import bindparam, select, text
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
                header_data_dict = las._sanitize_json_value(header_data.to_dict())
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
            ordered_futures = [executor.submit(parse_las_file, fp) for fp in file_paths]
            for future in ordered_futures:
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
        well_names = list(dict.fromkeys(pf["well_name"] for pf in parsed_files))
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
                    f"Well '{well_name}' already exists in project '{self.name}'. Merging curves."
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
                self.db_session.add(well_obj._orm_well)
                self.db_session.flush()
                well_obj.update_data_from_las_parse(
                    parsed["well_data"], parsed["header_data"]
                )
                wells_by_name[well_name] = well_obj._orm_well
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

    def read_dataframe(
        self,
        dataframes: Union[pd.DataFrame, List[pd.DataFrame]],
        well_name_column: str = "WELL_NAME",
        depth_column: str = "DEPTH",
        depth_uom: Optional[str] = None,
    ):
        """Reads one or more pandas DataFrames and adds or updates wells in the project.

        This method processes DataFrames similar to `read_las`, with the following behavior:
        - For each well found in the DataFrame(s), it checks if a well with that name
          already exists in the project.
        - If a well exists, its data is updated using the `Well.update_data` method
          (upsert logic - partial replacement).
        - If a well doesn't exist, a new well is created and the data is added.

        Args:
            dataframes (Union[pd.DataFrame, List[pd.DataFrame]]): A single DataFrame or
                list of DataFrames to read.
            well_name_column (str): The column name in the DataFrame(s) that identifies
                the well. Defaults to "WELL_NAME".
            depth_column (str): The column name that represents depth. Defaults to "DEPTH".
            depth_uom (Optional[str]): The unit of measurement for depth to be used if not
                already in the data.

        Returns:
            Dict[str, Any]: A dictionary containing:
                - 'processed_dataframes': Number of DataFrames processed
                - 'new_wells': List of newly created well names
                - 'updated_wells': Number of wells that were updated
        """
        # Normalize input to a list of DataFrames
        if isinstance(dataframes, pd.DataFrame):
            dataframes = [dataframes]

        logger.info(
            f"Reading {len(dataframes)} DataFrame(s) for project '{self.name}' (ID: {self.project_id})"
        )

        # Collect all wells from all dataframes
        all_well_data = {}
        for df_idx, df in enumerate(dataframes):
            logger.debug(f"Processing DataFrame {df_idx + 1}/{len(dataframes)}")

            # Validate required columns
            if well_name_column not in df.columns:
                logger.error(
                    f"DataFrame {df_idx} missing required column '{well_name_column}'"
                )
                continue
            if depth_column not in df.columns:
                logger.error(
                    f"DataFrame {df_idx} missing required column '{depth_column}'"
                )
                continue

            # Group by well name and accumulate data
            for well_name, well_data in df.groupby(well_name_column):
                if well_name not in all_well_data:
                    all_well_data[well_name] = []
                all_well_data[well_name].append(well_data)

        if not all_well_data:
            logger.warning("No valid well data found in provided DataFrames.")
            return {
                "processed_dataframes": len(dataframes),
                "new_wells": [],
                "updated_wells": 0,
            }

        # Batch fetch all existing wells for this project
        well_names = list(all_well_data.keys())
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

        # Process all wells
        new_wells = []
        for well_name, well_dataframes in all_well_data.items():
            logger.debug(
                f"Processing well '{well_name}' with {len(well_dataframes)} DataFrame(s)"
            )

            # Concatenate multiple dataframes for the same well
            combined_well_data = pd.concat(
                well_dataframes, ignore_index=True
            ).drop_duplicates(subset=[depth_column], keep="last")

            existing_well = wells_by_name.get(well_name)

            if existing_well:
                logger.debug(
                    f"Updating existing well '{well_name}' in project '{self.name}'"
                )
                well_obj = Well(self.db_session, well_id=existing_well.well_id)
                well_obj.update_data(combined_well_data)
            else:
                logger.debug(
                    f"Creating new well '{well_name}' in project '{self.name}'"
                )
                well_obj = Well(
                    self.db_session,
                    project_id=self.project_id,
                    name=well_name,
                    uwi=well_name,
                    depth_uom=depth_uom,
                )
                well_obj.update_data(combined_well_data)
                self.db_session.add(well_obj._orm_well)
                new_wells.append(well_name)

            logger.debug(f"Processed well '{well_name}' for project '{self.name}'.")

        # Single flush for all wells at the end
        if new_wells or all_well_data:
            self.db_session.flush()
            logger.info(
                f"Successfully processed {len(dataframes)} DataFrame(s) "
                f"({len(new_wells)} new wells, {len(all_well_data) - len(new_wells)} updated)"
            )

        return {
            "processed_dataframes": len(dataframes),
            "new_wells": new_wells,
            "updated_wells": len(all_well_data) - len(new_wells),
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
        os.makedirs(folder, exist_ok=True)
        logger.info("Exporting all project data to LAS files")
        for orm_well in self._orm_project.wells:
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            well_obj.export_to_las(folder=folder)

    def export_handover_package(
        self,
        output_zip_path: Optional[str] = None,
        well_names: Optional[List[str]] = None,
        include_las: bool = True,
        include_excel: bool = True,
    ) -> str:
        """Exports a complete Techlog/Geolog/IP/Petrel Petrophysical Handover Package (.zip).

        Includes evaluated LAS 2.0 logs (with `~Parameter` CPI settings), multi-well
        formation tops, fluid contacts, deviation surveys, discrete core RCA point
        tables & LAS point-sets, pressure tests, SCAL capillary pressure & relative
        permeability tables, Eclipse SWOF deck, PRMS/SEC zonal reservoir summary,
        master Excel workbook, and Techlog/Geolog auto-import scripts.

        Args:
            output_zip_path (Optional[str]): Path to write the `.zip` archive.
                Defaults to `<project.output_path>/<project.name>_Handover.zip`.
            well_names (Optional[List[str]]): Optional subset of well names to include.
            include_las (bool): Whether to include LAS 2.0 files. Defaults to True.
            include_excel (bool): Whether to include the Master `.xlsx` workbook. Defaults to True.

        Returns:
            str: Path to the generated `.zip` handover package.
        """
        from quick_pp.handover import (
            _sanitize_name,
            build_handover_archive_bytes,
            collect_well_handover_dict,
        )

        allowed = {str(w).strip() for w in well_names} if well_names else None
        wells_bundle = []
        for orm_well in self._orm_project.wells:
            if allowed and orm_well.name not in allowed:
                continue
            well_obj = Well(self.db_session, well_id=orm_well.well_id)
            wells_bundle.append(collect_well_handover_dict(well_obj))

        zip_bytes = build_handover_archive_bytes(
            project_name=self.name,
            project_description=self.description or "",
            wells_bundle=wells_bundle,
            include_las=include_las,
            include_excel=include_excel,
        )
        if not output_zip_path:
            os.makedirs(self.output_path, exist_ok=True)
            output_zip_path = os.path.join(
                self.output_path, f"{_sanitize_name(self.name)}_Handover.zip"
            )
        else:
            parent_dir = os.path.dirname(os.path.abspath(output_zip_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        with open(output_zip_path, "wb") as f:
            f.write(zip_bytes)
        logger.info(
            f"Exported project handover package ({len(wells_bundle)} wells) to {output_zip_path}"
        )
        return output_zip_path

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
        self,
        parsed_data: pd.DataFrame,
        parsed_header: Union[pd.DataFrame, Dict[str, Any]],
        metadata_override: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Updates the well's data and metadata from a parsed LAS/DLIS file.

        When the well already contains curves, this method merges incoming curves
        with existing curves:
        1.  Extract and persist rich `WellHeaderMetadata` (`~Well` and `~Parameter`) into `header_data`
            and auto-populate environmental parameters on `WellConfig` (preserving existing non-null fields).
        2.  Append new curves not yet present on the well.
        3.  For curves with the same name as an existing curve:
            - Skip importing if the incoming curve is identical to the existing curve.
            - Assign a suffix (`_1`, `_2`, ...) and append if the incoming curve differs.
        4.  Write the full merged DataFrame to columnar Parquet (`wells/{uwi}.parquet`).

        Args:
            parsed_data (pd.DataFrame): DataFrame containing 'DEPTH' and curve mnemonics.
            parsed_header (Union[pd.DataFrame, Dict[str, Any]]): Header DataFrame or metadata dict.
            metadata_override (Optional[Dict[str, Any]]): Optional user-reviewed metadata overrides.

        Returns:
            Dict[str, Any]: Merge summary (`appended_curves`, `skipped_curves`, `renamed_curves`).
        """
        logger.info(f"Updating well '{self.name}' from LAS/DLIS parse.")
        rich_meta = las.extract_las_metadata(
            parsed_header, fallback_well_name=self.name
        )
        if metadata_override:
            for k, v in metadata_override.items():
                if v is not None:
                    rich_meta[k] = v

        raw_header_dict = las._sanitize_json_value(
            parsed_header.to_dict()
            if isinstance(parsed_header, pd.DataFrame)
            else dict(parsed_header or {})
        )
        rich_meta_clean = dict(las._sanitize_json_value(dict(rich_meta)))
        prev_header = (
            dict(self.header_data) if isinstance(self.header_data, dict) else {}
        )
        prev_meta = (
            dict(prev_header.get("well_metadata"))
            if isinstance(prev_header.get("well_metadata"), dict)
            else {}
        )
        merged_meta = dict(prev_meta)
        for k, v in rich_meta_clean.items():
            if v is not None or k not in merged_meta:
                merged_meta[k] = v

        if isinstance(prev_header.get("header_overrides"), dict):
            cleaned_overrides = {
                ok: ov
                for ok, ov in prev_header["header_overrides"].items()
                if rich_meta_clean.get(ok) is None
            }
            if cleaned_overrides:
                prev_header["header_overrides"] = cleaned_overrides
            else:
                prev_header.pop("header_overrides", None)

        self.header_data = las._sanitize_json_value(
            {
                **prev_header,
                **raw_header_dict,
                "well_metadata": merged_meta,
                **{
                    k: v
                    for k, v in merged_meta.items()
                    if k not in ("raw_well_header", "raw_parameters") and v is not None
                },
            }
        )

        # Auto-populate header & environmental parameters on WellConfig for downstream modules (saturation, shf, qaqc)
        if merged_meta.get("uwi"):
            self.uwi = str(merged_meta["uwi"])
        env_updates = {}
        for env_key in (
            "uwi",
            "well_name",
            "field",
            "company",
            "service_company",
            "country",
            "province",
            "county",
            "state",
            "location",
            "api",
            "license",
            "spud_date",
            "log_date",
            "latitude",
            "longitude",
            "latitude_raw",
            "longitude_raw",
            "x_coord",
            "y_coord",
            "crs",
            "kb_elev",
            "df_elev",
            "gl_elev",
            "water_depth",
            "apd",
            "datum_unit",
            "start_depth",
            "stop_depth",
            "step",
            "bht",
            "bht_unit",
            "surface_temp",
            "bit_size",
            "rmf",
            "rmf_temp",
            "rmc",
            "mud_density",
        ):
            if merged_meta.get(env_key) is not None:
                env_updates[env_key] = merged_meta[env_key]
        if env_updates:
            clean_env = las._sanitize_json_value(env_updates)
            self._config_instance.update(clean_env)
            self._orm_well.config_data = las._sanitize_json_value(
                dict(self._config_instance.__dict__)
            )

        existing_df: Optional[pd.DataFrame] = None
        if self._orm_well.curves:
            if self.well_id is None:
                self.db_session.flush()
            existing_df = self.get_data_optimized()
            if existing_df.empty:
                existing_df = self.get_data()

        merged_df, merge_summary = las.merge_well_curves_with_dedup(
            existing_df, parsed_data
        )
        if merged_df.empty:
            return merge_summary

        df_work = merged_df.set_index("DEPTH")

        # Save zstd-compressed Parquet to project wells directory for fast columnar access
        try:
            proj_name = (
                self._orm_well.project.name
                if self._orm_well and self._orm_well.project
                else "default"
            )
            wells_parquet_dir = os.path.join("data", "04_project", proj_name, "wells")
            os.makedirs(wells_parquet_dir, exist_ok=True)
            safe_uwi = re.sub(r'[<>:"/\\|?*]+', "_", str(self.uwi or self.name))
            parquet_path = os.path.join(wells_parquet_dir, f"{safe_uwi}.parquet")
            df_work.reset_index().to_parquet(
                parquet_path, compression="zstd", index=False
            )
        except Exception as e:
            logger.debug(f"Could not write well Parquet cache for '{self.name}': {e}")

        # Drop identifier metadata columns before persisting curves to DB
        drop_meta_cols = [
            c for c in df_work.columns if str(c).upper() in ("WELL_NAME", "UWI")
        ]
        if drop_meta_cols:
            df_work = df_work.drop(columns=drop_meta_cols, errors="ignore")

        appended_curves: List[str] = [
            c for c in merge_summary.get("appended_curves", []) if c in df_work.columns
        ]
        if not appended_curves:
            self.db_session.flush()
            logger.info(
                f"Well '{self.name}' LAS parse had no new or modified curves "
                f"(skipped identical: {merge_summary.get('skipped_curves', [])})."
            )
            return merge_summary

        existing_curves_by_mnem = {c.mnemonic: c for c in self._orm_well.curves}
        reverse_renamed = {
            suffixed: orig
            for orig, suffixed in (merge_summary.get("renamed_curves") or {}).items()
        }

        # Delete any pre-existing curve_data rows only for curves in `appended_curves` that already had an ORMCurve record
        reused_curve_ids = [
            existing_curves_by_mnem[m].curve_id
            for m in appended_curves
            if m in existing_curves_by_mnem
            and existing_curves_by_mnem[m].curve_id is not None
        ]
        if reused_curve_ids:
            self.db_session.execute(
                text("DELETE FROM curve_data WHERE curve_id IN :cids").bindparams(
                    bindparam("cids", expanding=True)
                ),
                {"cids": reused_curve_ids},
            )

        # Ensure ORMCurve records exist for all newly appended/suffixed curves and flush once to obtain curve_ids
        curves_map: Dict[str, ORMCurve] = {}
        for mnemonic in appended_curves:
            values = df_work[mnemonic]
            orm_curve = existing_curves_by_mnem.get(mnemonic)
            is_numeric = pd.api.types.is_numeric_dtype(values)
            data_type = "numeric" if is_numeric else "text"
            header_lookup_mnem = reverse_renamed.get(
                mnemonic, re.sub(r"_\d+$", "", mnemonic)
            )
            unit_str = (
                (
                    las.get_unit_from_header(parsed_header, mnemonic)
                    or las.get_unit_from_header(parsed_header, header_lookup_mnem)
                )
                if isinstance(parsed_header, pd.DataFrame)
                else None
            )
            descr_str = (
                (
                    las.get_descr_from_header(parsed_header, mnemonic)
                    or las.get_descr_from_header(parsed_header, header_lookup_mnem)
                )
                if isinstance(parsed_header, pd.DataFrame)
                else None
            )
            if not orm_curve:
                orm_curve = ORMCurve(
                    well_id=self.well_id,
                    mnemonic=mnemonic,
                    unit=unit_str,
                    description=descr_str,
                    data_type=data_type,
                )
                self._orm_well.curves.append(orm_curve)
            else:
                orm_curve.data_type = data_type
                if unit_str:
                    orm_curve.unit = unit_str
            curves_map[mnemonic] = orm_curve

        self.db_session.flush()

        # Vectorized chunked ingestion for newly appended curve_data
        depths_arr = np.asarray(df_work.index, dtype=np.float64)
        # Guard against float32 collisions since PostgreSQL curve_data.depth is REAL (float4)
        depths_f32 = depths_arr.astype(np.float32)
        unique_f32_mask = ~pd.Series(depths_f32).duplicated(keep="last").to_numpy()

        bind = self.db_session.get_bind()
        is_postgres = bool(bind is not None and bind.dialect.name == "postgresql")

        if is_postgres:
            sio = io.StringIO()
            buffered_rows = 0
            COPY_FLUSH_ROWS = 250000
            raw_conn = self.db_session.connection().connection

            def _flush_copy_buffer(buf: io.StringIO) -> None:
                buf.seek(0)
                with raw_conn.cursor() as cur:
                    cur.copy_expert(
                        "COPY curve_data (curve_id, depth, value_numeric, value_text) FROM STDIN",
                        buf,
                    )
                buf.seek(0)
                buf.truncate(0)

            for mnemonic in appended_curves:
                values = df_work[mnemonic]
                orm_curve = curves_map[mnemonic]
                cid = orm_curve.curve_id
                is_numeric = pd.api.types.is_numeric_dtype(values)

                if is_numeric:
                    vals_arr = pd.to_numeric(values, errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    valid_mask = (
                        np.isfinite(depths_arr) & unique_f32_mask & np.isfinite(vals_arr)
                    )
                    if not valid_mask.any():
                        continue
                    v_depths = depths_arr[valid_mask].tolist()
                    v_vals = vals_arr[valid_mask].tolist()
                    sio.writelines(
                        f"{cid}\t{d}\t{v}\t\\N\n" for d, v in zip(v_depths, v_vals)
                    )
                    buffered_rows += len(v_depths)
                else:
                    valid_mask = (
                        values.notna().to_numpy()
                        & np.isfinite(depths_arr)
                        & unique_f32_mask
                    )
                    if not valid_mask.any():
                        continue
                    v_depths = depths_arr[valid_mask].tolist()
                    v_vals = values.iloc[np.flatnonzero(valid_mask)].astype(str).tolist()
                    for d, v in zip(v_depths, v_vals):
                        safe_v = (
                            v.replace("\\", "\\\\")
                            .replace("\t", " ")
                            .replace("\n", " ")
                            .replace("\r", " ")
                        )
                        sio.write(f"{cid}\t{d}\t\\N\t{safe_v}\n")
                    buffered_rows += len(v_depths)

                if buffered_rows >= COPY_FLUSH_ROWS:
                    _flush_copy_buffer(sio)
                    buffered_rows = 0

            if buffered_rows > 0:
                _flush_copy_buffer(sio)
            sio.close()
        else:
            batch_buffer: List[Dict[str, Any]] = []
            BATCH_FLUSH_SIZE = 25000

            for mnemonic in appended_curves:
                values = df_work[mnemonic]
                orm_curve = curves_map[mnemonic]
                cid = orm_curve.curve_id
                is_numeric = pd.api.types.is_numeric_dtype(values)

                if is_numeric:
                    vals_arr = pd.to_numeric(values, errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    valid_mask = (
                        np.isfinite(depths_arr) & unique_f32_mask & np.isfinite(vals_arr)
                    )
                    if not valid_mask.any():
                        continue
                    v_depths = depths_arr[valid_mask].tolist()
                    v_vals = vals_arr[valid_mask].tolist()
                    batch_buffer.extend(
                        {
                            "curve_id": cid,
                            "depth": d,
                            "value_numeric": v,
                            "value_text": None,
                        }
                        for d, v in zip(v_depths, v_vals)
                    )
                else:
                    valid_mask = (
                        values.notna().to_numpy()
                        & np.isfinite(depths_arr)
                        & unique_f32_mask
                    )
                    if not valid_mask.any():
                        continue
                    v_depths = depths_arr[valid_mask].tolist()
                    v_vals = values.iloc[np.flatnonzero(valid_mask)].astype(str).tolist()
                    batch_buffer.extend(
                        {
                            "curve_id": cid,
                            "depth": d,
                            "value_numeric": None,
                            "value_text": v,
                        }
                        for d, v in zip(v_depths, v_vals)
                    )

                while len(batch_buffer) >= BATCH_FLUSH_SIZE:
                    self.db_session.bulk_insert_mappings(
                        ORMCurveData, batch_buffer[:BATCH_FLUSH_SIZE]
                    )
                    del batch_buffer[:BATCH_FLUSH_SIZE]

            if batch_buffer:
                self.db_session.bulk_insert_mappings(ORMCurveData, batch_buffer)
                batch_buffer.clear()

        self.db_session.flush()
        logger.info(
            f"Well '{self.name}' data updated from LAS parse. "
            f"Appended: {len(appended_curves)}, Skipped identical: {len(merge_summary.get('skipped_curves', []))}, "
            f"Renamed: {merge_summary.get('renamed_curves', {})}."
        )
        return merge_summary

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

        # Optimization 3: Bulk delete old data points using primary keys after matching rounded depths
        if all_depths_to_delete:
            sel_stmt = text(
                "SELECT curve_data_id, depth FROM curve_data "
                "WHERE curve_id = :curve_id AND depth >= :min_d AND depth <= :max_d"
            )
            delete_stmt = text(
                "DELETE FROM curve_data WHERE curve_data_id IN :cd_ids"
            ).bindparams(bindparam("cd_ids", expanding=True))
            chunk_size = 900
            for curve_id, depths in all_depths_to_delete.items():
                rounded_set = {round(float(d), 4) for d in depths if pd.notna(d)}
                if not rounded_set:
                    continue
                min_d = min(rounded_set) - 0.01
                max_d = max(rounded_set) + 0.01
                rows = self.db_session.execute(
                    sel_stmt,
                    {"curve_id": curve_id, "min_d": min_d, "max_d": max_d},
                ).fetchall()
                cd_ids = [
                    int(r[0])
                    for r in rows
                    if r[1] is not None and round(float(r[1]), 4) in rounded_set
                ]
                for i in range(0, len(cd_ids), chunk_size):
                    self.db_session.execute(
                        delete_stmt,
                        {"cd_ids": cd_ids[i : i + chunk_size]},
                    )

        # Optimization 4: Batch insert new data points (COPY on PostgreSQL, bulk_insert_mappings otherwise)
        total_inserted = 0
        bind = self.db_session.get_bind()
        is_postgres = bool(bind is not None and bind.dialect.name == "postgresql")
        depths_arr = np.asarray(data.index, dtype=np.float64)
        depths_f32 = depths_arr.astype(np.float32)
        unique_f32_mask = ~pd.Series(depths_f32).duplicated(keep="last").to_numpy()

        if is_postgres:
            sio = io.StringIO()
            raw_conn = self.db_session.connection().connection
            for mnemonic, values in data.items():
                orm_curve = curves_by_mnemonic[mnemonic]
                cid = orm_curve.curve_id
                is_numeric = pd.api.types.is_numeric_dtype(values)

                if is_numeric:
                    vals_arr = pd.to_numeric(values, errors="coerce").to_numpy(
                        dtype=np.float64
                    )
                    valid_mask = (
                        np.isfinite(depths_arr) & unique_f32_mask & np.isfinite(vals_arr)
                    )
                    if not valid_mask.any():
                        continue
                    v_depths = depths_arr[valid_mask].tolist()
                    v_vals = vals_arr[valid_mask].tolist()
                    sio.writelines(
                        f"{cid}\t{d}\t{v}\t\\N\n" for d, v in zip(v_depths, v_vals)
                    )
                    total_inserted += len(v_depths)
                else:
                    valid_mask = (
                        values.notna().to_numpy()
                        & np.isfinite(depths_arr)
                        & unique_f32_mask
                    )
                    if not valid_mask.any():
                        continue
                    v_depths = depths_arr[valid_mask].tolist()
                    v_vals = values.iloc[np.flatnonzero(valid_mask)].astype(str).tolist()
                    for d, v in zip(v_depths, v_vals):
                        safe_v = (
                            v.replace("\\", "\\\\")
                            .replace("\t", " ")
                            .replace("\n", " ")
                            .replace("\r", " ")
                        )
                        sio.write(f"{cid}\t{d}\t\\N\t{safe_v}\n")
                    total_inserted += len(v_depths)

            if total_inserted > 0:
                sio.seek(0)
                with raw_conn.cursor() as cur:
                    cur.copy_expert(
                        "COPY curve_data (curve_id, depth, value_numeric, value_text) FROM STDIN",
                        sio,
                    )
            sio.close()
        else:
            for mnemonic, values in data.items():
                orm_curve = curves_by_mnemonic[mnemonic]
                is_numeric = pd.api.types.is_numeric_dtype(values)

                if is_numeric:
                    data_points = [
                        {
                            "curve_id": orm_curve.curve_id,
                            "depth": depth,
                            "value_numeric": float(value),
                            "value_text": None,
                        }
                        for depth, value, u_ok in zip(
                            data.index, values, unique_f32_mask, strict=True
                        )
                        if u_ok and pd.notna(value)
                    ]
                else:
                    data_points = [
                        {
                            "curve_id": orm_curve.curve_id,
                            "depth": depth,
                            "value_numeric": None,
                            "value_text": str(value),
                        }
                        for depth, value, u_ok in zip(
                            data.index, values, unique_f32_mask, strict=True
                        )
                        if u_ok and pd.notna(value)
                    ]
                if data_points:
                    self.db_session.bulk_insert_mappings(ORMCurveData, data_points)
                    total_inserted += len(data_points)

        logger.debug(
            f"Updated {len(data.columns)} curves for well '{self.name}' "
            f"with {total_inserted} total data points."
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
            cd.value_numeric,
            cd.value_text,
            c.data_type
        FROM curve_data cd
        JOIN curves c ON cd.curve_id = c.curve_id
        WHERE c.well_id = :well_id
        """)

        result = self.db_session.execute(sql, {"well_id": self.well_id})
        rows = result.fetchall()

        if not rows:
            return pd.DataFrame()

        df = pd.DataFrame(
            rows,
            columns=["depth", "mnemonic", "value_numeric", "value_text", "data_type"],
        )

        num_mask = df["data_type"] == "numeric"
        pivoted_parts = []

        if num_mask.any():
            num_df = df.loc[num_mask, ["depth", "mnemonic", "value_numeric"]]
            pivoted_num = num_df.pivot(
                index="depth", columns="mnemonic", values="value_numeric"
            )
            pivoted_parts.append(pivoted_num)

        if (~num_mask).any():
            txt_df = df.loc[~num_mask, ["depth", "mnemonic", "value_text"]]
            pivoted_txt = txt_df.pivot(
                index="depth", columns="mnemonic", values="value_text"
            )
            pivoted_parts.append(pivoted_txt)

        if not pivoted_parts:
            return pd.DataFrame()

        pivoted = (
            pivoted_parts[0]
            if len(pivoted_parts) == 1
            else pd.concat(pivoted_parts, axis=1)
        )

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

    def export_to_las(self, folder: Optional[str] = None) -> str:
        """Exports the well's data and CPI parameters to a LAS 2.0 file.

        Args:
            folder (Optional[str]): The directory to save the file in. Defaults to the
                                    well's output path.

        Returns:
            str: Path to the written LAS file.
        """
        folder = folder or self.output_path
        logger.info(f"Exporting well data to LAS format in folder: {folder}")
        data_df = self.get_data()
        return las.export_to_las(
            well_data=data_df,
            well_name=self.name,
            folder=folder,
            uwi=self.uwi or self.name,
            depth_uom=getattr(self._orm_well, "depth_uom", None) or "m",
            parameters=getattr(self.config, "__dict__", {}),
        )

    def export_handover_package(
        self,
        output_zip_path: Optional[str] = None,
        include_las: bool = True,
        include_excel: bool = True,
    ) -> str:
        """Exports a complete Techlog/Geolog/IP/Petrel Handover Package (.zip) for this well.

        Args:
            output_zip_path (Optional[str]): Path to write the `.zip` archive.
                Defaults to `<well.output_path>/<well.name>_Handover.zip`.
            include_las (bool): Whether to include LAS 2.0 files. Defaults to True.
            include_excel (bool): Whether to include the Master `.xlsx` workbook. Defaults to True.

        Returns:
            str: Path to the generated `.zip` handover package.
        """
        from quick_pp.handover import (
            _sanitize_name,
            build_handover_archive_bytes,
            collect_well_handover_dict,
        )

        proj_name = (
            self._orm_well.project.name
            if self._orm_well and self._orm_well.project
            else self.name
        )
        zip_bytes = build_handover_archive_bytes(
            project_name=f"{proj_name}_{self.name}",
            project_description=f"Single-well handover package for {self.name}",
            wells_bundle=[collect_well_handover_dict(self)],
            include_las=include_las,
            include_excel=include_excel,
        )
        if not output_zip_path:
            os.makedirs(self.output_path, exist_ok=True)
            output_zip_path = os.path.join(
                self.output_path, f"{_sanitize_name(self.name)}_Handover.zip"
            )
        else:
            parent_dir = os.path.dirname(os.path.abspath(output_zip_path))
            if parent_dir:
                os.makedirs(parent_dir, exist_ok=True)

        with open(output_zip_path, "wb") as f:
            f.write(zip_bytes)
        logger.info(
            f"Exported well handover package for '{self.name}' to {output_zip_path}"
        )
        return output_zip_path

    def __str__(self):
        return f"Well: {self.name} (ID: {self.well_id}, UWI: {self.uwi})"

    def add_formation_tops(
        self, tops: List[Dict[str, Any]], replace: bool = False
    ):
        """Adds or updates a list of formation tops for the well.

        Args:
            tops (List[Dict[str, Any]]): A list of dictionaries, where each dictionary
                                         represents a top and must contain 'name' and 'depth'.
            replace (bool): If True, deletes all existing formation tops for this well before inserting.
        """
        logger.info(
            f"Adding/updating {len(tops)} formation tops for well '{self.name}' (replace={replace})."
        )
        dedup_tops: Dict[str, float] = {}
        for top_data in tops or []:
            if not isinstance(top_data, dict) or not all(
                k in top_data for k in ["name", "depth"]
            ):
                logger.warning(f"Skipping invalid formation top data: {top_data}")
                continue
            t_name = str(top_data["name"]).strip()
            if not t_name:
                continue
            try:
                d_val = float(top_data["depth"])
            except (TypeError, ValueError):
                continue
            dedup_tops[t_name] = d_val

        if not dedup_tops and not replace:
            return

        if replace:
            self.db_session.query(ORMFormationTop).filter_by(
                well_id=self.well_id
            ).delete(synchronize_session="fetch")
        else:
            self.db_session.query(ORMFormationTop).filter(
                ORMFormationTop.well_id == self.well_id,
                ORMFormationTop.name.in_(list(dedup_tops.keys())),
            ).delete(synchronize_session="fetch")
        self.db_session.flush()

        for t_name, d_val in dedup_tops.items():
            orm_top = ORMFormationTop(
                well_id=self.well_id,
                name=t_name,
                depth=d_val,
            )
            self.db_session.add(orm_top)
        self.db_session.flush()

    def get_formation_tops(self) -> pd.DataFrame:
        """Retrieves all formation tops for the well as a DataFrame."""
        tops = self._orm_well.formation_tops
        if not tops:
            return pd.DataFrame(columns=["name", "depth"])

        data = [{"name": top.name, "depth": top.depth} for top in tops]
        df = pd.DataFrame(data).sort_values(by="depth").reset_index(drop=True)
        logger.debug(f"Retrieved {len(df)} formation tops for well '{self.name}'.")
        return df

    def add_fluid_contacts(
        self, contacts: List[Dict[str, Any]], replace: bool = False
    ):
        """Adds or updates a list of fluid contacts for the well.

        Args:
            contacts (List[Dict[str, Any]]): A list of dictionaries, where each
                                              dictionary represents a contact and must
                                              contain 'name' (or 'contact_type') and 'depth'.
            replace (bool): If True, deletes all existing fluid contacts for this well before inserting.
        """
        logger.info(
            f"Adding/updating {len(contacts)} fluid contacts for well '{self.name}' (replace={replace})."
        )
        dedup_contacts: Dict[str, float] = {}
        for contact_data in contacts or []:
            if not isinstance(contact_data, dict):
                continue
            c_name = str(
                contact_data.get("name") or contact_data.get("contact_type") or ""
            ).strip()
            d_raw = contact_data.get("depth")
            if not c_name or d_raw is None:
                logger.warning(f"Skipping invalid fluid contact data: {contact_data}")
                continue
            try:
                d_val = float(d_raw)
            except (TypeError, ValueError):
                continue
            dedup_contacts[c_name] = d_val

        if not dedup_contacts and not replace:
            return

        if replace:
            self.db_session.query(ORMFluidContact).filter_by(
                well_id=self.well_id
            ).delete(synchronize_session="fetch")
        else:
            self.db_session.query(ORMFluidContact).filter(
                ORMFluidContact.well_id == self.well_id,
                ORMFluidContact.name.in_(list(dedup_contacts.keys())),
            ).delete(synchronize_session="fetch")
        self.db_session.flush()

        for c_name, d_val in dedup_contacts.items():
            orm_contact = ORMFluidContact(
                well_id=self.well_id, name=c_name, depth=d_val
            )
            self.db_session.add(orm_contact)
        self.db_session.flush()

    def get_fluid_contacts(self) -> pd.DataFrame:
        """Retrieves all fluid contacts for the well as a DataFrame."""
        contacts = self._orm_well.fluid_contacts
        if not contacts:
            return pd.DataFrame(columns=["name", "depth"])

        data = [{"name": c.name, "depth": c.depth} for c in contacts]
        df = pd.DataFrame(data).sort_values(by="depth").reset_index(drop=True)
        logger.debug(f"Retrieved {len(df)} fluid contacts for well '{self.name}'.")
        return df

    def add_pressure_tests(
        self, tests: List[Dict[str, Any]], replace: bool = False
    ):
        """Adds or updates a list of pressure tests for the well.

        Args:
            tests (List[Dict[str, Any]]): A list of dictionaries, where each
                                          dictionary represents a test and must contain
                                          'depth' and 'pressure'. 'pressure_uom' is optional.
            replace (bool): If True, deletes all existing pressure tests for this well before inserting.
        """
        logger.info(
            f"Adding/updating {len(tests)} pressure tests for well '{self.name}' (replace={replace})."
        )
        dedup_tests: Dict[float, Dict[str, Any]] = {}
        for test_data in tests or []:
            if not isinstance(test_data, dict) or not all(
                k in test_data for k in ["depth", "pressure"]
            ):
                logger.warning(f"Skipping invalid pressure test data: {test_data}")
                continue
            try:
                d_val = round(float(test_data["depth"]), 4)
                p_val = float(test_data["pressure"])
            except (TypeError, ValueError):
                continue
            uom = str(
                test_data.get("pressure_uom")
                or test_data.get("pressure_unit")
                or "psi"
            ).strip()
            dedup_tests[d_val] = {
                "depth": d_val,
                "pressure": p_val,
                "pressure_uom": uom or "psi",
            }

        if not dedup_tests and not replace:
            return

        if replace:
            self.db_session.query(ORMPressureTest).filter_by(
                well_id=self.well_id
            ).delete(synchronize_session="fetch")
        else:
            existing_tests = list(
                self.db_session.scalars(
                    select(ORMPressureTest).filter_by(well_id=self.well_id)
                ).all()
            )
            for et in existing_tests:
                if round(float(et.depth), 4) in dedup_tests:
                    self.db_session.delete(et)
        self.db_session.flush()

        for t_clean in dedup_tests.values():
            orm_test = ORMPressureTest(well_id=self.well_id, **t_clean)
            self.db_session.add(orm_test)
        self.db_session.flush()

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
            f"Replacing all survey points for well '{self.name}' with {len(surveys)} new points."
        )

        # Delete all existing surveys for this well to ensure a full replacement
        self.db_session.query(ORMWellSurvey).filter_by(well_id=self.well_id).delete(
            synchronize_session="fetch"
        )
        self.db_session.flush()

        dedup_surveys: Dict[float, Dict[str, float]] = {}
        for survey_data in surveys or []:
            if not isinstance(survey_data, dict) or not all(
                k in survey_data for k in ["md", "inc", "azim"]
            ):
                logger.warning(f"Skipping invalid well survey data: {survey_data}")
                continue
            try:
                md_v = round(float(survey_data["md"]), 4)
                inc_v = float(survey_data["inc"])
                azim_v = float(survey_data["azim"])
            except (TypeError, ValueError):
                continue
            dedup_surveys[md_v] = {"md": md_v, "inc": inc_v, "azim": azim_v}

        for s_clean in dedup_surveys.values():
            orm_survey = ORMWellSurvey(well_id=self.well_id, **s_clean)
            self.db_session.add(orm_survey)
        self.db_session.flush()

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
        measurements: Optional[List[Dict[str, Any]]] = None,
        description: Optional[str] = None,
        remark: Optional[str] = None,
        relperm_data: Optional[List[Dict[str, Any]]] = None,
        pc_data: Optional[List[Dict[str, Any]]] = None,
    ):
        """Adds a core sample and its associated measurements to the well.

        This is a comprehensive method to add a core plug and all its related
        SCAL data (point measurements, rel-perm, and pc curves) in one go.
        Child collections (measurements, relperm_data, pc_data) are only replaced
        when explicitly provided (not None), allowing RCA and SCAL updates to
        coexist on the same core sample without wiping each other.

        Args:
            sample_name (str): The unique name or ID of the core sample.
            depth (float): The depth from which the sample was taken.
            description (Optional[str]): A geological description of the core sample.
            remark (Optional[str]): A remark about data quality, depth shifts, etc.
            measurements (Optional[List[Dict[str, Any]]]): List of point measurements.
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

        # Create or get the core sample by sample_name first, then fallback to approximate depth (handles float32 REAL rounding)
        depth_val = round(float(depth), 4)
        existing_samples = list(
            self.db_session.scalars(
                select(ORMCoreSample).filter_by(well_id=self.well_id)
            ).all()
        )
        orm_sample = next(
            (s for s in existing_samples if s.sample_name == sample_name),
            None,
        )
        # If a sample with the same name exists at a completely different depth (> 1.0m away),
        # disambiguate sample_name so we don't overwrite a different core's plug with the same number
        if orm_sample and abs(float(orm_sample.depth) - depth_val) > 1.0:
            sample_name = f"{sample_name}@{depth_val:.2f}m"
            orm_sample = next(
                (s for s in existing_samples if s.sample_name == sample_name),
                None,
            )

        if not orm_sample:
            orm_sample = next(
                (
                    s
                    for s in existing_samples
                    if abs(float(s.depth) - depth_val) < 1e-3
                    and (not sample_name or s.sample_name == sample_name)
                ),
                None,
            )

        if not orm_sample:
            # Ensure depth_val is unique in 32-bit REAL precision across existing_samples
            existing_f32_depths = {
                float(np.float32(s.depth)) for s in existing_samples
            }
            while float(np.float32(depth_val)) in existing_f32_depths:
                depth_val = round(depth_val + 0.002, 4)

            orm_sample = ORMCoreSample(
                well_id=self.well_id,
                sample_name=sample_name,
                depth=depth_val,
                description=description,
                remark=remark,
            )
            self.db_session.add(orm_sample)
            self.db_session.flush()  # Ensure we get the sample_id
        else:
            # Update depth if no other sample already occupies that 32-bit REAL depth
            if abs(float(orm_sample.depth) - depth_val) >= 1e-4:
                target_f32 = float(np.float32(depth_val))
                conflict_depth = any(
                    s.sample_id != orm_sample.sample_id
                    and float(np.float32(s.depth)) == target_f32
                    for s in existing_samples
                )
                if not conflict_depth:
                    orm_sample.depth = depth_val
            if description is not None:
                orm_sample.description = description
            if remark is not None:
                orm_sample.remark = remark
            # Clear child collections via delete-orphan cascade (safe for both persistent and pending items)
            if measurements is not None:
                orm_sample.measurements.clear()
            if relperm_data is not None:
                orm_sample.relperm_data.clear()
            if pc_data is not None:
                orm_sample.capillary_pressure_data.clear()
            self.db_session.flush()

        # Add point measurements (deduplicate by property_name)
        if measurements:
            dedup_meas: Dict[str, Dict[str, Any]] = {}
            for m_data in measurements:
                prop_name = str(m_data.get("property_name", "")).strip()
                val = m_data.get("value")
                if not prop_name or val is None:
                    continue
                dedup_meas[prop_name.upper()] = {
                    "property_name": prop_name,
                    "value": float(val),
                    "unit": m_data.get("unit"),
                }
            for m_clean in dedup_meas.values():
                orm_measurement = ORMCoreMeasurement(
                    sample_id=orm_sample.sample_id, **m_clean
                )
                orm_sample.measurements.append(orm_measurement)

        # Add relative permeability data
        if relperm_data:
            for rp_data in relperm_data:
                orm_relperm = ORMRelativePermeability(
                    sample_id=orm_sample.sample_id, **rp_data
                )
                orm_sample.relperm_data.append(orm_relperm)

        # Add capillary pressure data (normalize cycle & deduplicate by (saturation, experiment_type, cycle))
        if pc_data:
            dedup_pc: Dict[tuple, Dict[str, Any]] = {}
            for pc_point in pc_data:
                sat_raw = pc_point.get("saturation", pc_point.get("sw"))
                pres_raw = pc_point.get("pressure", pc_point.get("pc"))
                if sat_raw is None or pres_raw is None:
                    continue
                try:
                    sat_val = float(sat_raw)
                    pres_val = float(pres_raw)
                except (TypeError, ValueError):
                    continue
                exp_type = (
                    pc_point.get("experiment_type") or pc_point.get("type") or "micp"
                )
                raw_cycle = str(pc_point.get("cycle") or "drainage").strip().lower()
                cycle_val = "imbibition" if raw_cycle.startswith("imb") else "drainage"
                dedup_pc[(round(sat_val, 6), str(exp_type), cycle_val)] = {
                    "saturation": sat_val,
                    "pressure": pres_val,
                    "experiment_type": str(exp_type),
                    "cycle": cycle_val,
                }
            for pc_clean in dedup_pc.values():
                orm_pc = ORMCapillaryPressure(
                    sample_id=orm_sample.sample_id, **pc_clean
                )
                orm_sample.capillary_pressure_data.append(orm_pc)

        self.db_session.flush()
        logger.debug(
            f"Successfully added/updated data for core sample '{sample_name}'."
        )

    def replace_rca_samples(self, samples: List[Dict[str, Any]]):
        """Replaces all RCA point measurements across the well's core samples while preserving SCAL data."""
        dedup_by_name: Dict[str, Dict[str, Any]] = {}
        used_f32_depths: set[float] = set()

        for idx, s in enumerate(samples or []):
            if not isinstance(s, dict):
                continue
            depth_raw = s.get("depth")
            if depth_raw is None or str(depth_raw).strip() == "":
                continue
            try:
                depth_val = round(float(depth_raw), 4)
            except (TypeError, ValueError):
                continue
            sname = str(s.get("sample_name") or f"RCA-{idx + 1}").strip()
            if not sname:
                sname = f"RCA-{idx + 1}"

            # If another plug in a different core interval shares the same plug number (e.g. "1"),
            # disambiguate by depth so distinct plugs are never overwritten
            if sname in dedup_by_name and abs(dedup_by_name[sname]["depth"] - depth_val) >= 1e-2:
                sname = f"{sname}@{depth_val:.2f}m"
                if sname in dedup_by_name and abs(dedup_by_name[sname]["depth"] - depth_val) >= 1e-2:
                    sname = f"{sname}_{idx + 1}"

            # Avoid uq_well_id_sample_depth collision in 32-bit REAL precision when multiple plugs snap to the same log depth
            if sname not in dedup_by_name:
                while float(np.float32(depth_val)) in used_f32_depths:
                    depth_val = round(depth_val + 0.002, 4)
                used_f32_depths.add(float(np.float32(depth_val)))
            else:
                depth_val = dedup_by_name[sname]["depth"]

            meas_list: List[Dict[str, Any]] = list(s.get("measurements") or [])
            poro = s.get("porosity", s.get("cpore"))
            perm = s.get("permeability", s.get("cperm"))
            if poro is not None and str(poro).strip() != "":
                try:
                    meas_list.append(
                        {"property_name": "CPORE", "value": float(poro), "unit": "v/v"}
                    )
                except (TypeError, ValueError):
                    pass
            if perm is not None and str(perm).strip() != "":
                try:
                    meas_list.append(
                        {"property_name": "CPERM", "value": float(perm), "unit": "mD"}
                    )
                except (TypeError, ValueError):
                    pass
            dedup_by_name[sname] = {
                "sample_name": sname,
                "depth": depth_val,
                "measurements": meas_list,
                "description": s.get("description"),
                "remark": s.get("remark"),
            }

        parsed_samples = list(dedup_by_name.values())

        # Clear all existing RCA measurements for this well, and remove any core sample that has no SCAL data
        for orm_sample in list(self._orm_well.core_samples):
            orm_sample.measurements.clear()
            if (
                not orm_sample.capillary_pressure_data
                and not orm_sample.relperm_data
            ):
                self.db_session.delete(orm_sample)
        self.db_session.flush()

        for item in parsed_samples:
            self.add_core_sample_with_measurements(
                sample_name=item["sample_name"],
                depth=item["depth"],
                measurements=item["measurements"],
                description=item["description"],
                remark=item["remark"],
                relperm_data=None,
                pc_data=None,
            )
        self.db_session.flush()

    def replace_scal_pc_data(self, pc_rows: List[Dict[str, Any]], append: bool = False):
        """Replaces (or appends) SCAL capillary pressure pairs across the well's core samples while preserving RCA measurements."""
        grouped: Dict[str, Dict[str, Any]] = {}
        for idx, row in enumerate(pc_rows or []):
            if not isinstance(row, dict):
                continue
            depth_raw = row.get("depth")
            if depth_raw is None or str(depth_raw).strip() == "":
                depth_val = 0.0
            else:
                try:
                    depth_val = round(float(depth_raw), 4)
                except (TypeError, ValueError):
                    depth_val = 0.0
            sname = str(row.get("sample_name") or f"SCAL-{idx + 1}").strip()
            if not sname:
                sname = f"SCAL-{idx + 1}"

            # If a sample with the same name was already grouped at a different depth (> 1cm), disambiguate
            if sname in grouped and abs(grouped[sname]["depth"] - depth_val) >= 1e-2:
                sname = f"{sname}@{depth_val:.2f}m"

            entry = grouped.setdefault(
                sname,
                {
                    "depth": depth_val,
                    "pc_data": [],
                    "description": row.get("description"),
                },
            )
            # Support both nested `pc_data` and flat `(pc, sw)` row formats
            if isinstance(row.get("pc_data"), list):
                entry["pc_data"].extend(row["pc_data"])
            else:
                pc_val = row.get("pc", row.get("pressure"))
                sw_val = row.get("sw", row.get("saturation"))
                if (
                    pc_val is not None
                    and sw_val is not None
                    and str(pc_val).strip() != ""
                    and str(sw_val).strip() != ""
                ):
                    entry["pc_data"].append(
                        {
                            "pressure": pc_val,
                            "saturation": sw_val,
                            "experiment_type": row.get("experiment_type")
                            or row.get("type")
                            or "micp",
                            "cycle": row.get("cycle") or "drainage",
                        }
                    )

        if not append:
            for orm_sample in list(self._orm_well.core_samples):
                orm_sample.capillary_pressure_data.clear()
                if not orm_sample.measurements and not orm_sample.relperm_data:
                    self.db_session.delete(orm_sample)
            self.db_session.flush()

        for sname, sdata in grouped.items():
            pc_points = sdata["pc_data"]
            if append:
                orm_existing = self.db_session.scalar(
                    select(ORMCoreSample).filter_by(
                        well_id=self.well_id, sample_name=sname
                    )
                )
                if orm_existing and orm_existing.capillary_pressure_data:
                    existing_pts = [
                        {
                            "saturation": p.saturation,
                            "pressure": p.pressure,
                            "experiment_type": p.experiment_type,
                            "cycle": p.cycle,
                        }
                        for p in orm_existing.capillary_pressure_data
                    ]
                    pc_points = existing_pts + pc_points
            self.add_core_sample_with_measurements(
                sample_name=sname,
                depth=sdata["depth"],
                measurements=None,
                description=sdata.get("description"),
                relperm_data=None,
                pc_data=pc_points,
            )
        self.db_session.flush()

    def get_scal_pc_data(self) -> List[Dict[str, Any]]:
        """Retrieves all SCAL capillary pressure pairs for the well as a flat list of dicts."""
        rows: List[Dict[str, Any]] = []
        for sample in self._orm_well.core_samples:
            for p in sample.capillary_pressure_data:
                rows.append(
                    {
                        "sample_name": sample.sample_name,
                        "depth": float(sample.depth),
                        "pc": float(p.pressure),
                        "sw": float(p.saturation),
                        "experiment_type": p.experiment_type or "micp",
                        "cycle": p.cycle or "drainage",
                    }
                )
        rows.sort(key=lambda r: (r["depth"], r["sample_name"], r["pc"]))
        return rows

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
