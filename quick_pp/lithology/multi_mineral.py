from scipy.optimize import minimize
import numpy as np
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import pandas as pd

from quick_pp.config import Config
from quick_pp import logger
from quick_pp.porosity import neu_den_xplot_poro_pt, density_porosity


class MultiMineral:
    """Optimization-based multi-mineral model for lithology estimation with fluid volumes."""

    def __init__(
        self,
        minerals: Optional[List[str]] = None,
        fluid_properties: Optional[Dict] = None,
        porosity_method: str = "density",
        porosity_endpoints: Optional[Dict] = None,
    ):
        """Initialize the MultiMineral optimizer.

        Args:
            minerals (list, optional): A list of minerals for the optimization.
                                       Defaults to ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE'].
            fluid_properties (dict, optional): Properties for oil, gas, and water. Defaults to standard values.
            porosity_method (str, optional): The method for initial porosity estimation
                                             ('neutron_density', 'density', 'sonic'). Defaults to 'density'.
            porosity_endpoints (dict, optional): Endpoints for the 'neutron_density' porosity method.
        """
        self.responses = Config.MINERALS_LOG_VALUE
        available_minerals = list(self.responses.keys())

        # Default minerals if none specified
        if minerals is None:
            minerals = ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"]

        for mineral in minerals:
            if mineral not in available_minerals:
                raise ValueError(
                    f"Mineral '{mineral}' is not defined in Config.MINERALS_LOG_VALUE."
                )

        self.minerals = minerals
        self.porosity_method = porosity_method
        self.porosity_endpoints = porosity_endpoints or {}

        # Default fluid properties if none specified
        if fluid_properties is None:
            fluid_properties = {
                "OIL": {"density": 0.8, "nphi": 0.0, "pef": 0.0, "dtc": 200.0},
                "GAS": {"density": 0.2, "nphi": 0.0, "pef": 0.0, "dtc": 400.0},
                "WATER": {"density": 1.0, "nphi": 1.0, "pef": 0.0, "dtc": 190.0},
            }
        self.fluid_properties = fluid_properties
        self.fluids = ["OIL", "GAS", "WATER"]

        # Define bounds for each mineral type
        mineral_bounds = {mineral: (0, 1) for mineral in available_minerals}

        # Define bounds for fluid volumes (0 to 1 for each fluid)
        fluid_bounds = {"OIL": (0, 1), "GAS": (0, 1), "WATER": (0, 1)}

        # Set bounds based on selected minerals and fluids
        self.mineral_bounds = tuple(
            mineral_bounds[mineral] for mineral in self.minerals
        )
        self.fluid_bounds = tuple(fluid_bounds[fluid] for fluid in self.fluids)

        # Combined bounds: minerals first, then fluids
        self.bounds = self.mineral_bounds + self.fluid_bounds

        # Constraint: sum of mineral volumes + fluid volumes must equal 1
        self.constraints = [{"type": "eq", "fun": self._volume_constraint}]
        self.scaling_factors = {}

    def _volume_constraint(self, volumes: np.ndarray) -> float:
        """Ensure that the sum of all mineral and fluid volumes equals 1."""
        return np.sum(volumes) - 1

    def _calculate_porosity(
        self, nphi: float, rhob: float, dtc: Optional[float] = None
    ) -> float:
        """Calculate an initial porosity estimate based on the selected method.

        Args:
            nphi (float): Neutron porosity value.
            rhob (float): Bulk density value.
            dtc (float, optional): Sonic transit time value. Defaults to None.

        Returns:
            float: The initial porosity estimate.
        """
        if self.porosity_method == "density":
            # Calculate an initial rho_ma based on the selected minerals
            initial_rho_ma = sum(
                self.responses[mineral].get("RHOB", 0)
                for mineral in self.minerals
                if mineral in self.responses
            )
            count_minerals_with_density = sum(
                1
                for mineral in self.minerals
                if mineral in self.responses and "RHOB" in self.responses[mineral]
            )

            if count_minerals_with_density > 0:
                rho_ma = initial_rho_ma / count_minerals_with_density
            else:
                rho_ma = 2.65  # Default matrix density (Quartz)

            # Calculate average fluid density for porosity calculation
            rho_fluid_avg = np.mean(
                [prop["density"] for prop in self.fluid_properties.values()]
            )

            return density_porosity(rhob, rho_ma, rho_fluid_avg)

        elif self.porosity_method == "neutron_density":
            # Use neutron-density crossplot porosity
            dry_min1_point = self.porosity_endpoints.get(
                "dry_min1_point", (-0.02, 2.65)
            )
            dry_clay_point = self.porosity_endpoints.get("dry_clay_point", (0.37, 2.41))
            fluid_point = self.porosity_endpoints.get("fluid_point", (1.0, 1.0))

            return neu_den_xplot_poro_pt(
                nphi,
                rhob,
                model="ss",
                dry_min1_point=dry_min1_point,
                dry_clay_point=dry_clay_point,
                fluid_point=fluid_point,
            )

        elif self.porosity_method == "sonic":
            # Sonic porosity calculation (simplified)
            if dtc is None:
                return 0.0
            dt_ma = 55.0  # Default matrix slowness
            dt_fluid = 190.0  # Default fluid slowness
            return (dtc - dt_ma) / (dt_fluid - dt_ma)

        else:
            raise ValueError(f"Unknown porosity method: {self.porosity_method}")

    def _calculate_matrix_density(self, mineral_volumes: np.ndarray) -> float:
        """Calculate the average matrix density from mineral volumes.

        Args:
            mineral_volumes (np.ndarray): An array of mineral volumes.

        Returns:
            float: The calculated matrix density.
        """
        rho_ma = 0.0
        total_mineral_vol = np.sum(mineral_volumes)
        if total_mineral_vol < 1e-6:
            return 2.65  # Return default if no minerals

        for i, mineral in enumerate(self.minerals):
            if mineral in self.responses and "RHOB" in self.responses[mineral]:
                rho_ma += mineral_volumes[i] * self.responses[mineral]["RHOB"]

        return rho_ma / total_mineral_vol

    def _calculate_fluid_density(self, fluid_volumes: np.ndarray) -> float:
        """Calculate the average fluid density from fluid volumes.

        Args:
            fluid_volumes (np.ndarray): An array of fluid volumes.

        Returns:
            float: The calculated average fluid density.
        """
        rho_fluid = 0.0
        for i, fluid in enumerate(self.fluids):
            rho_fluid += fluid_volumes[i] * self.fluid_properties[fluid]["density"]
        return rho_fluid

    def _calculate_log_response(self, volumes: np.ndarray, log_type: str) -> float:
        """Calculate the reconstructed log response for a given log type.

        Args:
            volumes (np.ndarray): An array of combined mineral and fluid volumes.
            log_type (str): The type of log to reconstruct (e.g., 'NPHI', 'RHOB').

        Returns:
            float: The reconstructed log value.
        """
        response = 0.0

        # Add mineral contributions
        n_minerals = len(self.minerals)
        for i, mineral in enumerate(self.minerals):
            if mineral in self.responses and log_type in self.responses[mineral]:
                response += volumes[i] * self.responses[mineral][log_type]

        # Add fluid contributions
        for i, fluid in enumerate(self.fluids):
            fluid_idx = n_minerals + i
            fluid_vol = volumes[fluid_idx]

            if log_type == "NPHI":
                response += fluid_vol * self.fluid_properties[fluid]["nphi"]
            elif log_type == "RHOB":
                response += fluid_vol * self.fluid_properties[fluid]["density"]
            elif log_type == "DTC":
                response += fluid_vol * self.fluid_properties[fluid]["dtc"]
            elif log_type == "PEF":
                response += fluid_vol * self.fluid_properties[fluid]["pef"]
            # GR doesn't change with fluid type (assuming fresh water)

        return response

    def _calculate_average_error(
        self, volumes: np.ndarray, log_values: Dict[str, float]
    ) -> float:
        """Calculate the average absolute error between measured and reconstructed logs.

        Args:
            volumes (np.ndarray): An array of combined mineral and fluid volumes.
            log_values (dict): A dictionary of measured log values.

        Returns:
            float: The average absolute error.
        """
        total_error = 0.0
        valid_count = 0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                reconstructed_value = self._calculate_log_response(volumes, log_type)
                error = abs(measured_value - reconstructed_value)
                total_error += error
                valid_count += 1

        return total_error / valid_count if valid_count > 0 else np.nan

    def _error_function(
        self, volumes: np.ndarray, log_values: Dict[str, float]
    ) -> float:
        """Calculate the sum of squared errors for the optimization.

        Args:
            volumes (np.ndarray): An array of combined mineral and fluid volumes.
            log_values (dict): A dictionary of measured log values.

        Returns:
            float: The total squared error.
        """
        total_error = 0.0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                # Special handling for RHOB when using density porosity method
                if self.porosity_method == "density" and log_type == "RHOB":
                    mineral_vols = volumes[: len(self.minerals)]
                    fluid_vols = volumes[len(self.minerals) :]
                    porosity = np.sum(fluid_vols)

                    # For density porosity, RHOB is reconstructed from porosity and matrix density
                    rho_ma = self._calculate_matrix_density(mineral_vols)
                    rho_fluid = self._calculate_fluid_density(fluid_vols)
                    reconstructed_value = rho_ma * (1 - porosity) + rho_fluid * porosity
                else:
                    reconstructed_value = self._calculate_log_response(
                        volumes, log_type
                    )

                scaling = self.scaling_factors.get(log_type, 1.0)
                error = (scaling * (measured_value - reconstructed_value)) ** 2
                total_error += error

        return total_error

    def _optimize_volumes(
        self, log_values: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Perform the optimization to find the best-fit mineral and fluid volumes.

        Args:
            log_values (dict): A dictionary of measured log values for a single depth point.

        Returns:
            tuple: A tuple containing the optimized mineral volumes, fluid volumes, and a success flag.
        """
        # Filter out None and NaN values
        valid_logs = {
            k: v for k, v in log_values.items() if v is not None and not np.isnan(v)
        }

        if not valid_logs:
            raise ValueError("No valid log measurements provided")

        # Calculate initial porosity estimate for fluid volume initialization
        nphi = log_values.get("NPHI", 0.0)
        rhob = log_values.get("RHOB", 2.65)
        dtc = log_values.get("DTC", None)

        if nphi is not None and rhob is not None:
            initial_porosity = self._calculate_porosity(nphi, rhob, dtc)
            initial_porosity = max(
                0.0, min(0.4, initial_porosity)
            )  # Constrain to reasonable range
        else:
            initial_porosity = 0.1  # Default porosity

        n_minerals = len(self.minerals)
        n_fluids = len(self.fluids)
        # Distribute volume between minerals and fluids
        mineral_volume = (1.0 - initial_porosity) / n_minerals
        fluid_volume = initial_porosity / n_fluids

        initial_guess = np.concatenate(
            [np.full(n_minerals, mineral_volume), np.full(n_fluids, fluid_volume)]
        )

        # Run optimization
        result = minimize(
            self._error_function,
            initial_guess,
            args=(valid_logs,),
            bounds=self.bounds,
            constraints=self.constraints,
            method="SLSQP",
        )

        # Split results into mineral and fluid volumes
        mineral_volumes = result.x[:n_minerals]
        fluid_volumes = result.x[n_minerals:]

        return mineral_volumes, fluid_volumes, result.success

    def estimate_lithology(
        self,
        gr: np.ndarray,
        nphi: np.ndarray,
        rhob: np.ndarray,
        pef: Optional[np.ndarray] = None,
        dtc: Optional[np.ndarray] = None,
        auto_scale: bool = True,
    ) -> pd.DataFrame:
        """Estimate mineral and fluid volumes for a set of log data points.

        This function iterates through each depth point, performing an optimization
        to determine the volumetric fractions of minerals and fluids that best
        match the measured log responses.

        Args:
            gr (np.ndarray): Gamma Ray log [GAPI].
            nphi (np.ndarray): Neutron Porosity log [v/v].
            rhob (np.ndarray): Bulk Density log [g/cc].
            pef (np.ndarray, optional): Photoelectric Factor log [barns/electron]. Defaults to None.
            dtc (np.ndarray, optional): Compressional slowness log [us/ft]. Defaults to None.
            auto_scale (bool, optional): If True, automatically calculates scaling factors. Defaults to True.

        Returns:
            pd.DataFrame: A DataFrame containing the estimated mineral and fluid volumes,
                          constructed porosity, error metrics, and reconstructed logs for each depth point.
        """
        # Initialize output arrays
        n_samples = len(gr)

        # Initialize mineral volume arrays
        mineral_volumes = {}
        for mineral in self.minerals:
            mineral_volumes[mineral] = np.full(n_samples, np.nan)

        # Initialize fluid volume arrays
        fluid_volumes = {}
        for fluid in self.fluids:
            fluid_volumes[fluid] = np.full(n_samples, np.nan)

        # Initialize porosity array (sum of fluid volumes)
        porosity_constructed = np.full(n_samples, np.nan)

        # Initialize the reconstructed logs
        gr_reconstructed = np.full(n_samples, np.nan)
        nphi_reconstructed = np.full(n_samples, np.nan)
        rhob_reconstructed = np.full(n_samples, np.nan)
        pef_reconstructed = np.full(n_samples, np.nan)
        dtc_reconstructed = np.full(n_samples, np.nan)

        # Initialize error array
        average_errors = np.full(n_samples, np.nan)

        # Initialize QC flags
        success_flags = np.full(n_samples, False, dtype=bool)

        # Dynamic scaling based on log ranges for robustness
        if auto_scale:
            all_logs = {"GR": gr, "NPHI": nphi, "RHOB": rhob, "PEF": pef, "DTC": dtc}
            for log_type, log_data in all_logs.items():
                if log_data is not None:
                    # Use 5th and 95th percentiles to calculate a robust range, ignoring NaNs
                    valid_data = log_data[~np.isnan(log_data)]
                    if len(valid_data) > 1:
                        p05 = np.percentile(valid_data, 5)
                        p95 = np.percentile(valid_data, 95)
                        log_range = p95 - p05
                        if log_range > 1e-6:  # Avoid division by zero for flat logs
                            self.scaling_factors[log_type] = round(1.0 / log_range, 4)
                        else:
                            self.scaling_factors[log_type] = (
                                1.0  # Default if range is zero
                            )

        logger.info(f"Scaling factors: {self.scaling_factors}")
        # Process each depth point
        for i in tqdm(range(n_samples), desc="Processing depth points", unit="points"):
            # Prepare log values for current depth
            log_values = {
                "GR": gr[i],
                "NPHI": nphi[i],
                "RHOB": rhob[i],
                "PEF": pef[i] if pef is not None and not np.isnan(pef[i]) else None,
                "DTC": dtc[i] if dtc is not None and not np.isnan(dtc[i]) else None,
            }

            # Check if we have valid measurements
            valid_measurements = [
                v for v in log_values.values() if v is not None and not np.isnan(v)
            ]

            if len(valid_measurements) >= 3:  # Need at least 3 measurements
                try:
                    # Optimize mineral and fluid volumes
                    mineral_vols, fluid_vols, success = self._optimize_volumes(
                        log_values
                    )

                    # Store results for each mineral
                    for j, mineral in enumerate(self.minerals):
                        mineral_volumes[mineral][i] = mineral_vols[j]

                    # Store results for each fluid
                    for j, fluid in enumerate(self.fluids):
                        fluid_volumes[fluid][i] = fluid_vols[j]

                    # Store total porosity (sum of fluid volumes)
                    porosity_constructed[i] = np.sum(fluid_vols)
                    success_flags[i] = success

                    # Combine volumes for error calculation and log reconstruction
                    combined_volumes = np.concatenate([mineral_vols, fluid_vols])

                    # Calculate and store average error
                    average_errors[i] = self._calculate_average_error(
                        combined_volumes, log_values
                    )

                    # Store the reconstructed logs
                    gr_reconstructed[i] = (
                        self._calculate_log_response(combined_volumes, "GR")
                        if log_values["GR"] is not None
                        else np.nan
                    )
                    nphi_reconstructed[i] = (
                        self._calculate_log_response(combined_volumes, "NPHI")
                        if log_values["NPHI"] is not None
                        else np.nan
                    )
                    rhob_reconstructed[i] = (
                        self._calculate_log_response(combined_volumes, "RHOB")
                        if log_values["RHOB"] is not None
                        else np.nan
                    )
                    pef_reconstructed[i] = (
                        self._calculate_log_response(combined_volumes, "PEF")
                        if log_values["PEF"] is not None
                        else np.nan
                    )
                    dtc_reconstructed[i] = (
                        self._calculate_log_response(combined_volumes, "DTC")
                        if log_values["DTC"] is not None
                        else np.nan
                    )

                except Exception as e:
                    tqdm.write(f"\rError at depth {i}: {e}", end="\r")
                    continue

        # Create output DataFrame
        output_data = {}

        # Add mineral volumes with appropriate column names
        mineral_column_mapping = {
            mineral: Config.MINERALS_NAME_MAPPING.get(mineral, f"V{mineral}")
            for mineral in self.minerals
        }

        for mineral in self.minerals:
            column_name = mineral_column_mapping.get(mineral, f"V{mineral}")
            output_data[column_name] = mineral_volumes[mineral]

        # Add fluid volumes with appropriate column names
        fluid_column_mapping = {"OIL": "VOIL", "GAS": "VGAS", "WATER": "VWATER"}

        for fluid in self.fluids:
            column_name = fluid_column_mapping.get(fluid, f"V{fluid}")
            output_data[column_name] = fluid_volumes[fluid]

        # Add total porosity (sum of fluid volumes)
        output_data["PHIT_CONSTRUCTED"] = porosity_constructed

        # Add error and reconstructed logs
        output_data.update(
            {
                "AVG_ERROR": average_errors,
                "OPTIMIZER_SUCCESS": success_flags,
                "GR_RECONSTRUCTED": gr_reconstructed,
                "NPHI_RECONSTRUCTED": nphi_reconstructed,
                "RHOB_RECONSTRUCTED": rhob_reconstructed,
                "PEF_RECONSTRUCTED": pef_reconstructed,
                "DTC_RECONSTRUCTED": dtc_reconstructed,
            }
        )

        return pd.DataFrame(output_data)
