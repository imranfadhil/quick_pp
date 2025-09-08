from scipy.optimize import minimize
import numpy as np
from typing import Dict, Optional, List
from tqdm import tqdm
import pandas as pd

from quick_pp.config import Config
from quick_pp import logger


class MultiMineralOptimizer:
    """Optimization-based multi-mineral model for lithology estimation."""

    def __init__(self, minerals: Optional[List[str]] = None):
        """
        Initialize the MultiMineralOptimizer.

        Args:
            minerals: List of minerals to include in the optimization.
                     If None, uses default minerals: ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'MUD']
                     Available options: 'QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'GYPSUM', 'HALITE', 'MUD'
        """
        # Default minerals if none specified
        if minerals is None:
            minerals = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'MUD']

        self.minerals = minerals

        # Define bounds for each mineral type
        mineral_bounds = {
            'QUARTZ': (0, 1),
            'CALCITE': (0, 1),
            'DOLOMITE': (0, 0.1),
            'SHALE': (0, 1),
            'ANHYDRITE': (0, 1),
            'GYPSUM': (0, 1),
            'HALITE': (0, 1),
            'MUD': (0, 0.45)
        }

        # Set bounds based on selected minerals
        self.bounds = tuple(mineral_bounds[mineral] for mineral in self.minerals)

        # Constraint: sum of mineral volumes must equal 1
        self.constraints = [{"type": "eq", "fun": self._volume_constraint}]

        # Mineral log responses from Config
        self.responses = Config.MINERALS_LOG_VALUE

        # Scaling factors for different log types to balance their contribution to the error
        self.scaling_factors = {
            'GR': 1.0,
            'NPHI': 300.0,
            'RHOB': 100.0,
            'PEF': 30.0,
            'DTC': 1.0
        }

    def _volume_constraint(self, volumes: np.ndarray) -> float:
        """Constraint function ensuring mineral volumes sum to 1."""
        return np.sum(volumes) - 1

    def _calculate_log_response(self, volumes: np.ndarray, log_type: str) -> float:
        """Calculate reconstructed log response for a given log type."""
        response = 0.0
        for i, mineral in enumerate(self.minerals):
            response_key = f"{log_type}_{mineral}"
            if response_key in self.responses:
                response += volumes[i] * self.responses[response_key]
        return response

    def _calculate_average_error(self, volumes: np.ndarray, log_values: Dict[str, float]) -> float:
        """
        Calculate the average absolute error between measured and reconstructed log values.

        Args:
            volumes: Mineral volumes in the order of self.minerals
            log_values: Dictionary of measured log values

        Returns:
            Average error across all log types
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

    def _error_function(self, volumes: np.ndarray, log_values: Dict[str, float]) -> float:
        """
        Calculate the sum of squared error between measured and reconstructed log values.

        Args:
            volumes: Mineral volumes in the order of self.minerals
            log_values: Dictionary of measured log values

        Returns:
            Total squared error
        """
        total_error = 0.0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                reconstructed_value = self._calculate_log_response(volumes, log_type)
                scaling = self.scaling_factors.get(log_type, 1.0)
                error = (scaling * (measured_value - reconstructed_value)) ** 2
                total_error += error

        return total_error

    def _optimize_mineral_volumes(self, log_values: Dict[str, float]) -> np.ndarray:
        """
        Optimize mineral volumes given available log measurements.

        Args:
            log_values: Dictionary of measured log values

        Returns:
            Optimized mineral volumes in the order of self.minerals
        """
        # Filter out None and NaN values
        valid_logs = {k: v for k, v in log_values.items()
                      if v is not None and not np.isnan(v)}

        if not valid_logs:
            raise ValueError("No valid log measurements provided")

        # Initial guess: equal distribution among minerals
        n_minerals = len(self.minerals)
        initial_guess = np.full(n_minerals, 1.0 / n_minerals)

        # Run optimization
        result = minimize(
            self._error_function,
            initial_guess,
            args=(valid_logs,),
            bounds=self.bounds,
            constraints=self.constraints,
            method='SLSQP'
        )

        return result.x

    def estimate_lithology(self, gr: np.ndarray, nphi: np.ndarray, rhob: np.ndarray, pef: Optional[np.ndarray] = None,
                           dtc: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Estimate mineral volumes using available log measurements.

        Args:
            gr: Gamma Ray log in GAPI
            nphi: Neutron Porosity log in v/v
            rhob: Bulk Density log in g/cc
            pef: Photoelectric Factor log in barns/electron (optional)
            dtc: Compressional slowness log in us/ft (optional)

        Returns:
            DataFrame containing:
            - Mineral volume fractions for each mineral in self.minerals
            - AVERAGE_ERROR: Average error between measured and reconstructed logs
            - *_RECONSTRUCTED: Reconstructed log values for each log type
        """
        # Initialize output arrays
        n_samples = len(gr)

        # Initialize mineral volume arrays
        mineral_volumes = {}
        for mineral in self.minerals:
            mineral_volumes[mineral] = np.full(n_samples, np.nan)

        # Initialize the reconstructed logs
        gr_reconstructed = np.full(n_samples, np.nan)
        nphi_reconstructed = np.full(n_samples, np.nan)
        rhob_reconstructed = np.full(n_samples, np.nan)
        pef_reconstructed = np.full(n_samples, np.nan)
        dtc_reconstructed = np.full(n_samples, np.nan)

        # Initialize error array
        average_errors = np.full(n_samples, np.nan)

        # Process each depth point
        for i in tqdm(range(n_samples), desc="Processing depth points", unit="points"):
            # Prepare log values for current depth
            log_values = {
                'GR': gr[i],
                'NPHI': nphi[i],
                'RHOB': rhob[i],
                'PEF': pef[i] if pef is not None and not np.isnan(pef[i]) else None,
                'DTC': dtc[i] if dtc is not None and not np.isnan(dtc[i]) else None
            }

            # Check if we have valid measurements
            valid_measurements = [v for v in log_values.values()
                                  if v is not None and not np.isnan(v)]

            if len(valid_measurements) >= 3:  # Need at least 3 measurements
                try:
                    # Optimize mineral volumes
                    volumes = self._optimize_mineral_volumes(log_values)

                    # Store results for each mineral
                    for j, mineral in enumerate(self.minerals):
                        mineral_volumes[mineral][i] = volumes[j]

                    # Calculate and store average error
                    average_errors[i] = self._calculate_average_error(volumes, log_values)

                    # Store the reconstructed logs
                    gr_reconstructed[i] = self._calculate_log_response(
                        volumes, 'GR') if log_values['GR'] is not None else np.nan
                    nphi_reconstructed[i] = self._calculate_log_response(
                        volumes, 'NPHI') if log_values['NPHI'] is not None else np.nan
                    rhob_reconstructed[i] = self._calculate_log_response(
                        volumes, 'RHOB') if log_values['RHOB'] is not None else np.nan
                    pef_reconstructed[i] = self._calculate_log_response(
                        volumes, 'PEF') if log_values['PEF'] is not None else np.nan
                    dtc_reconstructed[i] = self._calculate_log_response(
                        volumes, 'DTC') if log_values['DTC'] is not None else np.nan

                except Exception as e:
                    logger.error(f'\rError at depth {i}: {e}')
                    continue
            else:
                logger.error(f'\rInsufficient data at depth {i}')

        # Create output DataFrame
        output_data = {}

        # Add mineral volumes with appropriate column names
        mineral_column_mapping = {
            'QUARTZ': 'VSAND',
            'CALCITE': 'VCALC',
            'DOLOMITE': 'VDOLO',
            'SHALE': 'VCLD',
            'ANHYDRITE': 'VANHY',
            'GYPSUM': 'VGYPS',
            'HALITE': 'VHALI',
            'MUD': 'VMUD'
        }

        for mineral in self.minerals:
            column_name = mineral_column_mapping.get(mineral, f'V{mineral}')
            output_data[column_name] = mineral_volumes[mineral]

        # Add error and reconstructed logs
        output_data.update({
            'AVG_ERROR': average_errors,
            'GR_RECONSTRUCTED': gr_reconstructed,
            'NPHI_RECONSTRUCTED': nphi_reconstructed,
            'RHOB_RECONSTRUCTED': rhob_reconstructed,
            'PEF_RECONSTRUCTED': pef_reconstructed,
            'DTC_RECONSTRUCTED': dtc_reconstructed
        })

        return pd.DataFrame(output_data)


class MultiMineral:
    """Legacy interface for backward compatibility."""

    def __init__(self, minerals: Optional[List[str]] = None):
        """
        Initialize MultiMineral with optional mineral specification.

        Args:
            minerals: List of minerals to include in the optimization.
                     If None, uses default minerals: 
                        ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'MUD']
                     Available options: 'QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'GYPSUM', 'HALITE', 'MUD'
        """
        self.optimizer = MultiMineralOptimizer(minerals)

    def estimate_lithology(self, gr, nphi, rhob, pef=None, dtc=None):
        """
        Estimate lithology using multi-mineral optimization.

        Args:
            gr (np.ndarray): Gamma Ray log in GAPI.
            nphi (np.ndarray): Neutron Porosity log in v/v.
            rhob (np.ndarray): Bulk Density log in g/cc.
            pef (np.ndarray, optional): Photoelectric Factor log in barns/electron.
            dtc (np.ndarray, optional): Compressional slowness log in us/ft.

        Returns:
            pd.DataFrame: DataFrame containing mineral volumes and reconstructed logs
        """
        return self.optimizer.estimate_lithology(gr, nphi, rhob, pef, dtc)
