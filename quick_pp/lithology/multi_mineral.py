from scipy.optimize import minimize
import numpy as np
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
import pandas as pd

from quick_pp.config import Config
from quick_pp import logger
from quick_pp.porosity import neu_den_xplot_poro_pt, density_porosity


class MultiMineralOptimizer:
    """Optimization-based multi-mineral model for lithology estimation."""

    def __init__(self, minerals: Optional[List[str]] = None,
                 porosity_method: str = 'neutron_density',
                 porosity_endpoints: Optional[Dict] = None,
                 rho_fluid: float = 1.0):
        """
        Initialize the MultiMineralOptimizer.

        Args:
            minerals: List of minerals to include in the optimization.
                     If None, uses default minerals: ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE']
                     Available options: 'QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'GYPSUM', 'HALITE'
            porosity_method: Method for calculating porosity ('neutron_density', 'density', 'sonic')
            porosity_endpoints: Endpoints for porosity calculation (for neutron_density method)
            rho_fluid: Fluid density for porosity calculations
        """
        # Default minerals if none specified
        if minerals is None:
            minerals = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE']

        self.minerals = minerals
        self.porosity_method = porosity_method
        self.porosity_endpoints = porosity_endpoints or {}
        self.rho_fluid = rho_fluid

        # Define bounds for each mineral type
        mineral_bounds = {
            'QUARTZ': (0, 1),
            'CALCITE': (0, 1),
            'DOLOMITE': (0, 0.1),
            'SHALE': (0, 1),
            'ANHYDRITE': (0, 1),
            'GYPSUM': (0, 1),
            'HALITE': (0, 1)
        }

        # Set bounds based on selected minerals
        self.bounds = tuple(mineral_bounds[mineral] for mineral in self.minerals)

        # Constraint: sum of mineral volumes + porosity must equal 1
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

    def _volume_constraint(self, volumes: np.ndarray, porosity: float = 0.0) -> float:
        """Constraint function ensuring mineral volumes + porosity sum to 1."""
        return np.sum(volumes) + porosity - 1

    def _calculate_porosity(self, nphi: float, rhob: float, dtc: Optional[float] = None) -> float:
        """Calculate porosity using the specified method."""
        if self.porosity_method == 'neutron_density':
            # Use neutron-density crossplot porosity
            model = self.porosity_endpoints.get('model', 'ssc')
            dry_sand_point = self.porosity_endpoints.get('dry_sand_point', (-0.02, 2.65))
            dry_silt_point = self.porosity_endpoints.get('dry_silt_point', (0.0, 2.68))
            dry_clay_point = self.porosity_endpoints.get('dry_clay_point', (0.37, 2.41))
            fluid_point = self.porosity_endpoints.get('fluid_point', (1.0, 1.0))

            return neu_den_xplot_poro_pt(
                nphi, rhob, model=model,
                dry_min1_point=dry_sand_point,
                dry_silt_point=dry_silt_point,
                dry_clay_point=dry_clay_point,
                fluid_point=fluid_point
            )

        elif self.porosity_method == 'density':
            # Calculate matrix density from mineral volumes (will be updated during optimization)
            # For now, use a default matrix density
            rho_ma = 2.65  # Default matrix density
            return density_porosity(rhob, rho_ma, self.rho_fluid)

        elif self.porosity_method == 'sonic':
            # Sonic porosity calculation (simplified)
            if dtc is None:
                return 0.0
            dt_ma = 55.0  # Default matrix slowness
            dt_fluid = 190.0  # Default fluid slowness
            return (dtc - dt_ma) / (dt_fluid - dt_ma)

        else:
            raise ValueError(f"Unknown porosity method: {self.porosity_method}")

    def _calculate_matrix_density(self, volumes: np.ndarray) -> float:
        """Calculate matrix density from mineral volumes."""
        rho_ma = 0.0
        for i, mineral in enumerate(self.minerals):
            rho_key = f"RHOB_{mineral}"
            if rho_key in self.responses:
                rho_ma += volumes[i] * self.responses[rho_key]
        return rho_ma

    def _calculate_log_response(self, volumes: np.ndarray, log_type: str, porosity: float = 0.0) -> float:
        """Calculate reconstructed log response for a given log type."""
        response = 0.0

        # Add mineral contributions
        for i, mineral in enumerate(self.minerals):
            response_key = f"{log_type}_{mineral}"
            if response_key in self.responses:
                response += volumes[i] * self.responses[response_key]

        # Add porosity contribution (fluid-filled space)
        if log_type == 'NPHI':
            response += porosity * 1.0  # Fluid has NPHI = 1.0
        elif log_type == 'RHOB':
            response += porosity * self.rho_fluid  # Fluid density
        elif log_type == 'DTC':
            response += porosity * 190.0  # Fluid slowness (us/ft)
        elif log_type == 'PEF':
            response += porosity * 0.0  # Fluid PEF = 0
        # GR doesn't change with porosity (assuming fresh water)

        return response

    def _calculate_average_error(self, volumes: np.ndarray, log_values: Dict[str, float],
                                 porosity: float = 0.0) -> float:
        """
        Calculate the average absolute error between measured and reconstructed log values.

        Args:
            volumes: Mineral volumes in the order of self.minerals
            log_values: Dictionary of measured log values
            porosity: Porosity value

        Returns:
            Average error across all log types
        """
        total_error = 0.0
        valid_count = 0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                reconstructed_value = self._calculate_log_response(volumes, log_type, porosity)
                error = abs(measured_value - reconstructed_value)
                total_error += error
                valid_count += 1

        return total_error / valid_count if valid_count > 0 else np.nan

    def _error_function(self, volumes: np.ndarray, log_values: Dict[str, float], porosity: float = 0.0) -> float:
        """
        Calculate the sum of squared error between measured and reconstructed log values.

        Args:
            volumes: Mineral volumes in the order of self.minerals
            log_values: Dictionary of measured log values
            porosity: Porosity value

        Returns:
            Total squared error
        """
        total_error = 0.0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                reconstructed_value = self._calculate_log_response(volumes, log_type, porosity)
                scaling = self.scaling_factors.get(log_type, 1.0)
                error = (scaling * (measured_value - reconstructed_value)) ** 2
                total_error += error

        return total_error

    def _optimize_mineral_volumes(self, log_values: Dict[str, float]) -> Tuple[np.ndarray, float]:
        """
        Optimize mineral volumes and porosity given available log measurements.

        Args:
            log_values: Dictionary of measured log values

        Returns:
            Tuple of (optimized mineral volumes, porosity) in the order of self.minerals
        """
        # Filter out None and NaN values
        valid_logs = {k: v for k, v in log_values.items()
                      if v is not None and not np.isnan(v)}

        if not valid_logs:
            raise ValueError("No valid log measurements provided")

        # Calculate initial porosity estimate
        nphi = log_values.get('NPHI', 0.0)
        rhob = log_values.get('RHOB', 2.65)
        dtc = log_values.get('DTC', None)

        if nphi is not None and rhob is not None:
            initial_porosity = self._calculate_porosity(nphi, rhob, dtc)
            initial_porosity = max(0.0, min(0.5, initial_porosity))  # Constrain to reasonable range
        else:
            initial_porosity = 0.1  # Default porosity

        # Initial guess: equal distribution among minerals, adjusted for porosity
        n_minerals = len(self.minerals)
        mineral_volume = (1.0 - initial_porosity) / n_minerals
        initial_guess = np.full(n_minerals, mineral_volume)

        # Create constraint function that includes porosity
        def constraint_func(volumes):
            return self._volume_constraint(volumes, initial_porosity)

        # Update constraints
        constraints = [{"type": "eq", "fun": constraint_func}]

        # Run optimization
        result = minimize(
            self._error_function,
            initial_guess,
            args=(valid_logs, initial_porosity),
            bounds=self.bounds,
            constraints=constraints,
            method='SLSQP'
        )

        return result.x, initial_porosity

    def estimate_lithology(self, gr: np.ndarray, nphi: np.ndarray, rhob: np.ndarray, pef: Optional[np.ndarray] = None,
                           dtc: Optional[np.ndarray] = None) -> pd.DataFrame:
        """
        Estimate mineral volumes and porosity using available log measurements.

        Args:
            gr: Gamma Ray log in GAPI
            nphi: Neutron Porosity log in v/v
            rhob: Bulk Density log in g/cc
            pef: Photoelectric Factor log in barns/electron (optional)
            dtc: Compressional slowness log in us/ft (optional)

        Returns:
            DataFrame containing:
            - Mineral volume fractions for each mineral in self.minerals
            - PHIT_CONSTRUCTED: Total porosity constructed by the optimizer
            - AVERAGE_ERROR: Average error between measured and reconstructed logs
            - *_RECONSTRUCTED: Reconstructed log values for each log type
        """
        # Initialize output arrays
        n_samples = len(gr)

        # Initialize mineral volume arrays
        mineral_volumes = {}
        for mineral in self.minerals:
            mineral_volumes[mineral] = np.full(n_samples, np.nan)

        # Initialize porosity array
        porosity_constructed = np.full(n_samples, np.nan)

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
                    # Optimize mineral volumes and porosity
                    volumes, phit = self._optimize_mineral_volumes(log_values)

                    # Store results for each mineral
                    for j, mineral in enumerate(self.minerals):
                        mineral_volumes[mineral][i] = volumes[j]

                    # Store porosity
                    porosity_constructed[i] = phit

                    # Calculate and store average error
                    average_errors[i] = self._calculate_average_error(volumes, log_values, phit)

                    # Store the reconstructed logs
                    gr_reconstructed[i] = self._calculate_log_response(
                        volumes, 'GR', phit) if log_values['GR'] is not None else np.nan
                    nphi_reconstructed[i] = self._calculate_log_response(
                        volumes, 'NPHI', phit) if log_values['NPHI'] is not None else np.nan
                    rhob_reconstructed[i] = self._calculate_log_response(
                        volumes, 'RHOB', phit) if log_values['RHOB'] is not None else np.nan
                    pef_reconstructed[i] = self._calculate_log_response(
                        volumes, 'PEF', phit) if log_values['PEF'] is not None else np.nan
                    dtc_reconstructed[i] = self._calculate_log_response(
                        volumes, 'DTC', phit) if log_values['DTC'] is not None else np.nan

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
            'HALITE': 'VHALI'
        }

        for mineral in self.minerals:
            column_name = mineral_column_mapping.get(mineral, f'V{mineral}')
            output_data[column_name] = mineral_volumes[mineral]

        # Add porosity
        output_data['PHIT_CONSTRUCTED'] = porosity_constructed

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

    def __init__(self, minerals: Optional[List[str]] = None,
                 porosity_method: str = 'neutron_density',
                 porosity_endpoints: Optional[Dict] = None,
                 rho_fluid: float = 1.0):
        """
        Initialize MultiMineral with optional mineral specification.

        Args:
            minerals: List of minerals to include in the optimization.
                     If None, uses default minerals:
                        ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE']
                     Available options: 'QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE', 'GYPSUM', 'HALITE'
            porosity_method: Method for calculating porosity ('neutron_density', 'density', 'sonic')
            porosity_endpoints: Endpoints for porosity calculation (for neutron_density method)
            rho_fluid: Fluid density for porosity calculations
        """
        self.optimizer = MultiMineralOptimizer(
            minerals=minerals,
            porosity_method=porosity_method,
            porosity_endpoints=porosity_endpoints,
            rho_fluid=rho_fluid
        )

    def estimate_lithology(self, gr, nphi, rhob, pef=None, dtc=None):
        """
        Estimate lithology and porosity using multi-mineral optimization.

        Args:
            gr (np.ndarray): Gamma Ray log in GAPI.
            nphi (np.ndarray): Neutron Porosity log in v/v.
            rhob (np.ndarray): Bulk Density log in g/cc.
            pef (np.ndarray, optional): Photoelectric Factor log in barns/electron.
            dtc (np.ndarray, optional): Compressional slowness log in us/ft.

        Returns:
            pd.DataFrame: DataFrame containing mineral volumes, porosity, and reconstructed logs
        """
        return self.optimizer.estimate_lithology(gr, nphi, rhob, pef, dtc)
