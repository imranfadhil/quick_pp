from scipy.optimize import minimize
import numpy as np
from typing import Dict, Optional
from tqdm import tqdm
import pandas as pd

from quick_pp.config import Config
from quick_pp import logger


class MultiMineralOptimizer:
    """Optimization-based multi-mineral model for lithology estimation."""

    def __init__(self):
        # Mineral volume bounds (quartz, calcite, dolomite, shale, mud)
        self.bounds = ((0, 1), (0, 1), (0, 0.1), (0, 1), (0, 0.45))

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

        # Mineral names in order
        self.minerals = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'MUD']

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
            volumes: Mineral volumes [quartz, calcite, dolomite, shale, mud]
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
            volumes: Mineral volumes [quartz, calcite, dolomite, shale, mud]
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
            Optimized mineral volumes [quartz, calcite, dolomite, shale, mud]
        """
        # Filter out None and NaN values
        valid_logs = {k: v for k, v in log_values.items()
                      if v is not None and not np.isnan(v)}

        if not valid_logs:
            raise ValueError("No valid log measurements provided")

        # Initial guess: equal distribution among minerals
        initial_guess = np.array([0.3, 0.3, 0.3, 0.3, 0.3])

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
            - VSAND: Quartz volume fraction
            - VCALC: Calcite volume fraction
            - VDOLO: Dolomite volume fraction
            - VCLD: Shale volume fraction
            - VMUD: Mud volume fraction
            - AVERAGE_ERROR: Average error between measured and reconstructed logs
            - *_RECONSTRUCTED: Reconstructed log values for each log type
        """
        # Initialize output arrays
        n_samples = len(gr)
        vol_quartz = np.full(n_samples, np.nan)
        vol_calcite = np.full(n_samples, np.nan)
        vol_dolomite = np.full(n_samples, np.nan)
        vol_shale = np.full(n_samples, np.nan)
        vol_mud = np.full(n_samples, np.nan)

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

                    # Store results
                    vol_quartz[i] = volumes[0]
                    vol_calcite[i] = volumes[1]
                    vol_dolomite[i] = volumes[2]
                    vol_shale[i] = volumes[3]
                    vol_mud[i] = volumes[4]

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

        return pd.DataFrame({
            'VSAND': vol_quartz,
            'VCALC': vol_calcite,
            'VDOLO': vol_dolomite,
            'VCLD': vol_shale,
            'VMUD': vol_mud,
            'AVG_ERROR': average_errors,
            'GR_RECONSTRUCTED': gr_reconstructed,
            'NPHI_RECONSTRUCTED': nphi_reconstructed,
            'RHOB_RECONSTRUCTED': rhob_reconstructed,
            'PEF_RECONSTRUCTED': pef_reconstructed,
            'DTC_RECONSTRUCTED': dtc_reconstructed
        })


class MultiMineral:
    """Legacy interface for backward compatibility."""

    def __init__(self):
        self.optimizer = MultiMineralOptimizer()

    def estimate_lithology(self, gr, nphi, rhob, pef, dtc):
        """
        Legacy method for backward compatibility.

        Args:
            gr (float): Gamma Ray log in GAPI.
            nphi (float): Neutron Porosity log in v/v.
            rhob (float): Bulk Density log in g/cc.
            dtc (float): Compressional slowness log in us/ft.
            pef (float): Photoelectric Factor log in barns/electron.

        Returns:
            (float, float, float, float, float): vol_quartz, vol_calcite, vol_dolomite, vol_shale, vol_mud
        """
        return self.optimizer.estimate_lithology(gr, nphi, rhob, pef, dtc)


# Legacy functions for backward compatibility (deprecated)
def constraint_(x):
    """Legacy constraint function (deprecated)."""
    return x[0] + x[1] + x[2] + x[3] + x[4] - 1


constrains = [{"type": "eq", "fun": constraint_}]
bounds = ((0, 1), (0, 1), (0, 0.1), (0, 1), (0, 0.45))
responses = Config.MINERALS_LOG_VALUE


def minimizer_1(gr, nphi, rhob):
    """Legacy minimizer function (deprecated)."""
    optimizer = MultiMineralOptimizer()
    log_values = {'GR': gr, 'NPHI': nphi, 'RHOB': rhob}
    return type('Result', (), {'x': optimizer._optimize_mineral_volumes(log_values)})()


def minimizer_2(gr, nphi, rhob, pef):
    """Legacy minimizer function (deprecated)."""
    optimizer = MultiMineralOptimizer()
    log_values = {'GR': gr, 'NPHI': nphi, 'RHOB': rhob, 'PEF': pef}
    return type('Result', (), {'x': optimizer._optimize_mineral_volumes(log_values)})()


def minimizer_3(gr, nphi, rhob, dtc):
    """Legacy minimizer function (deprecated)."""
    optimizer = MultiMineralOptimizer()
    log_values = {'GR': gr, 'NPHI': nphi, 'RHOB': rhob, 'DTC': dtc}
    return type('Result', (), {'x': optimizer._optimize_mineral_volumes(log_values)})()


def minimizer_4(gr, nphi, rhob, pef, dtc):
    """Legacy minimizer function (deprecated)."""
    optimizer = MultiMineralOptimizer()
    log_values = {'GR': gr, 'NPHI': nphi, 'RHOB': rhob, 'PEF': pef, 'DTC': dtc}
    return type('Result', (), {'x': optimizer._optimize_mineral_volumes(log_values)})()
