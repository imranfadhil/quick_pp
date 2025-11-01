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

    def __init__(self, minerals: Optional[List[str]] = None,
                 fluid_properties: Optional[Dict] = None,
                 porosity_method: str = 'neutron_density',
                 porosity_endpoints: Optional[Dict] = None):
        """
        Initialize the MultiMineralOptimizer.

        Args:
            minerals: List of minerals to include in the optimization.
                     If None, uses default minerals: ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE']
                     Available options: 'QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE', 'ANHYDRITE',
                                      'GYPSUM', 'HALITE', 'PYRITE', 'FELDSPAR', 'COAL'
            fluid_properties: Dictionary containing fluid properties for oil, gas, and water.
                            If None, uses default properties:
                            {
                                'OIL': {'density': 0.8, 'nphi': 0.0, 'pef': 0.0, 'dtc': 200.0},
                                'GAS': {'density': 0.2, 'nphi': 0.0, 'pef': 0.0, 'dtc': 400.0},
                                'WATER': {'density': 1.0, 'nphi': 1.0, 'pef': 0.0, 'dtc': 190.0}
                            }
            porosity_method: Method for calculating porosity ('neutron_density', 'density', 'sonic')
            porosity_endpoints: Endpoints for porosity calculation (for neutron_density method)
        """
        # Default minerals if none specified
        if minerals is None:
            minerals = ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE']

        self.minerals = minerals
        self.porosity_method = porosity_method
        self.porosity_endpoints = porosity_endpoints or {}

        # Default fluid properties if none specified
        if fluid_properties is None:
            fluid_properties = {
                'OIL': {'density': 0.8, 'nphi': 0.0, 'pef': 0.0, 'dtc': 200.0},
                'GAS': {'density': 0.2, 'nphi': 0.0, 'pef': 0.0, 'dtc': 400.0},
                'WATER': {'density': 1.0, 'nphi': 1.0, 'pef': 0.0, 'dtc': 190.0}
            }

        self.fluid_properties = fluid_properties
        self.rho_fluid = self.fluid_properties['WATER']['density']  # Assume water for fluid density
        self.fluids = ['OIL', 'GAS', 'WATER']

        # Define bounds for each mineral type
        mineral_bounds = {
            'QUARTZ': (0, 1),
            'CALCITE': (0, 1),
            'DOLOMITE': (0, 1),
            'SHALE': (0, 1),
            'KAOLINITE': (0, 1),
            'FELDSPAR': (0, 1),
            'ANHYDRITE': (0, 1),
            'GYPSUM': (0, 1),
            'HALITE': (0, 1),
            'PYRITE': (0, 1),
            'COAL': (0, 1)
        }

        # Define bounds for fluid volumes (0 to 1 for each fluid)
        fluid_bounds = {
            'OIL': (0, 1),
            'GAS': (0, 1),
            'WATER': (0, 1)
        }

        # Set bounds based on selected minerals and fluids
        self.mineral_bounds = tuple(mineral_bounds[mineral] for mineral in self.minerals)
        self.fluid_bounds = tuple(fluid_bounds[fluid] for fluid in self.fluids)

        # Combined bounds: minerals first, then fluids
        self.bounds = self.mineral_bounds + self.fluid_bounds

        # Constraint: sum of mineral volumes + fluid volumes must equal 1
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
        """Constraint function ensuring mineral volumes + fluid volumes sum to 1."""
        return np.sum(volumes) - 1

    def _calculate_porosity(self, nphi: float, rhob: float, dtc: Optional[float] = None,
                            volumes: Optional[np.ndarray] = None) -> float:
        """Calculate porosity using the specified method."""
        if self.porosity_method == 'neutron_density':
            # Use neutron-density crossplot porosity
            dry_sand_point = self.porosity_endpoints.get('dry_sand_point', (-0.02, 2.65))
            dry_clay_point = self.porosity_endpoints.get('dry_clay_point', (0.37, 2.41))
            fluid_point = self.porosity_endpoints.get('fluid_point', (1.0, 1.0))

            return neu_den_xplot_poro_pt(
                nphi, rhob, model='ss',
                dry_min1_point=dry_sand_point,
                dry_clay_point=dry_clay_point,
                fluid_point=fluid_point,
            )

        elif self.porosity_method == 'density':
            if volumes is not None:
                # Calculate matrix density from the current mineral volumes during optimization
                mineral_vols = volumes[:len(self.minerals)]
                matrix_vol = np.sum(mineral_vols)
                if matrix_vol > 1e-3:
                    rho_ma = self._calculate_matrix_density(mineral_vols) / matrix_vol
                else:
                    rho_ma = 2.65  # Fallback if no matrix
            else:
                rho_ma = 2.65  # Default for initial guess if volumes are not provided
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

    def _calculate_log_response(self, volumes: np.ndarray, log_type: str) -> float:
        """Calculate reconstructed log response for a given log type."""
        response = 0.0

        # Add mineral contributions
        n_minerals = len(self.minerals)
        for i, mineral in enumerate(self.minerals):
            response_key = f"{log_type}_{mineral}"
            if response_key in self.responses:
                response += volumes[i] * self.responses[response_key]

        # Add fluid contributions
        for i, fluid in enumerate(self.fluids):
            fluid_idx = n_minerals + i
            fluid_vol = volumes[fluid_idx]

            if log_type == 'NPHI':
                response += fluid_vol * self.fluid_properties[fluid]['nphi']
            elif log_type == 'RHOB':
                response += fluid_vol * self.fluid_properties[fluid]['density']
            elif log_type == 'DTC':
                response += fluid_vol * self.fluid_properties[fluid]['dtc']
            elif log_type == 'PEF':
                response += fluid_vol * self.fluid_properties[fluid]['pef']
            # GR doesn't change with fluid type (assuming fresh water)

        return response

    def _calculate_average_error(self, volumes: np.ndarray, log_values: Dict[str, float]) -> float:
        """
        Calculate the average absolute error between measured and reconstructed log values.

        Args:
            volumes: Combined mineral and fluid volumes in the order of self.minerals + self.fluids
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
            volumes: Combined mineral and fluid volumes in the order of self.minerals + self.fluids
            log_values: Dictionary of measured log values

        Returns:
            Total squared error
        """
        total_error = 0.0

        # Porosity is the sum of fluid volumes
        porosity = np.sum(volumes[len(self.minerals):])

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                if self.porosity_method == 'density' and log_type == 'RHOB':
                    # For density porosity, RHOB is reconstructed from porosity and matrix density
                    mineral_vols = volumes[:len(self.minerals)]
                    rho_ma = self._calculate_matrix_density(mineral_vols)
                    reconstructed_value = rho_ma * (1 - porosity) + self.rho_fluid * porosity
                else:
                    reconstructed_value = self._calculate_log_response(volumes, log_type)

                scaling = self.scaling_factors.get(log_type, 1.0)
                error = (scaling * (measured_value - reconstructed_value)) ** 2
                total_error += error

        return total_error

    def _optimize_volumes(self, log_values: Dict[str, float]) -> Tuple[np.ndarray, np.ndarray, bool]:
        """
        Optimize mineral and fluid volumes given available log measurements.

        Args:
            log_values: Dictionary of measured log values

        Returns:
            Tuple of (optimized mineral volumes, fluid volumes, success_flag)
        """
        # Filter out None and NaN values
        valid_logs = {k: v for k, v in log_values.items()
                      if v is not None and not np.isnan(v)}

        if not valid_logs:
            raise ValueError("No valid log measurements provided")

        # Calculate initial porosity estimate for fluid volume initialization
        nphi = log_values.get('NPHI', 0.0)
        rhob = log_values.get('RHOB', 2.65)
        dtc = log_values.get('DTC', None)

        # Initial guess: equal distribution among minerals and fluids
        n_minerals = len(self.minerals)
        n_fluids = len(self.fluids)

        # Create an initial guess for volumes to pass to porosity calculation
        initial_mineral_guess = np.full(n_minerals, (1.0 - 0.1) / n_minerals)
        initial_fluid_guess = np.full(n_fluids, 0.1 / n_fluids)
        initial_guess_for_poro = np.concatenate([initial_mineral_guess, initial_fluid_guess])

        if nphi is not None and rhob is not None:
            initial_porosity = self._calculate_porosity(nphi, rhob, dtc, volumes=initial_guess_for_poro)
            initial_porosity = max(0.0, min(0.4, initial_porosity))  # Constrain to reasonable range
        else:
            initial_porosity = 0.01  # Default porosity

        # Distribute volume between minerals and fluids
        mineral_volume = (1.0 - initial_porosity) / n_minerals
        fluid_volume = initial_porosity / n_fluids

        initial_guess = np.concatenate([
            np.full(n_minerals, mineral_volume),
            np.full(n_fluids, fluid_volume)
        ])

        # Run optimization
        result = minimize(
            self._error_function,
            initial_guess,
            args=(valid_logs,),
            bounds=self.bounds,
            constraints=self.constraints,
            method='SLSQP'
        )

        # Split results into mineral and fluid volumes
        mineral_volumes = result.x[:n_minerals]
        fluid_volumes = result.x[n_minerals:]

        return mineral_volumes, fluid_volumes, result.success

    def estimate_lithology(self, gr: np.ndarray, nphi: np.ndarray, rhob: np.ndarray, pef: Optional[np.ndarray] = None,
                           dtc: Optional[np.ndarray] = None, auto_scale: bool = True) -> pd.DataFrame:
        """
        Estimate mineral and fluid volumes using available log measurements.

        Args:
            gr: Gamma Ray log in GAPI
            nphi: Neutron Porosity log in v/v
            rhob: Bulk Density log in g/cc
            pef: Photoelectric Factor log in barns/electron (optional)
            dtc: Compressional slowness log in us/ft (optional)
            auto_scale: If True, automatically calculate scaling factors based on log ranges.

        Returns:
            DataFrame containing:
            - Mineral volume fractions for each mineral in self.minerals
            - Fluid volume fractions for each fluid in self.fluids
            - PHIT_CONSTRUCTED: Total porosity (sum of fluid volumes)
            - AVERAGE_ERROR: Average error between measured and reconstructed logs
            - OPTIMIZER_SUCCESS: Boolean flag indicating if the optimization was successful.
            - *_RECONSTRUCTED: Reconstructed log values for each log type
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
            all_logs = {'GR': gr, 'NPHI': nphi, 'RHOB': rhob, 'PEF': pef, 'DTC': dtc}
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
                            self.scaling_factors[log_type] = 1.0  # Default if range is zero

        logger.info(f"Scaling factors: {self.scaling_factors}")
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
                    # Optimize mineral and fluid volumes
                    mineral_vols, fluid_vols, success = self._optimize_volumes(log_values)

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
                    average_errors[i] = self._calculate_average_error(combined_volumes, log_values)

                    # Store the reconstructed logs
                    gr_reconstructed[i] = self._calculate_log_response(
                        combined_volumes, 'GR') if log_values['GR'] is not None else np.nan
                    nphi_reconstructed[i] = self._calculate_log_response(
                        combined_volumes, 'NPHI') if log_values['NPHI'] is not None else np.nan
                    rhob_reconstructed[i] = self._calculate_log_response(
                        combined_volumes, 'RHOB') if log_values['RHOB'] is not None else np.nan
                    pef_reconstructed[i] = self._calculate_log_response(
                        combined_volumes, 'PEF') if log_values['PEF'] is not None else np.nan
                    dtc_reconstructed[i] = self._calculate_log_response(
                        combined_volumes, 'DTC') if log_values['DTC'] is not None else np.nan

                except Exception as e:
                    tqdm.write(f'\rError at depth {i}: {e}', end='\r')
                    continue

        # Create output DataFrame
        output_data = {}

        # Add mineral volumes with appropriate column names
        mineral_column_mapping = {
            'QUARTZ': 'VSAND',
            'CALCITE': 'VCALC',
            'DOLOMITE': 'VDOLO',
            'SHALE': 'VCLAY',
            'KAOLINITE': 'VKAOL',
            'FELDSPAR': 'VFELD',
            'ANHYDRITE': 'VANHY',
            'GYPSUM': 'VGYPS',
            'HALITE': 'VHALI',
            'PYRITE': 'VPYRI',
            'COAL': 'VCOAL'
        }

        for mineral in self.minerals:
            column_name = mineral_column_mapping.get(mineral, f'V{mineral}')
            output_data[column_name] = mineral_volumes[mineral]

        # Add fluid volumes with appropriate column names
        fluid_column_mapping = {
            'OIL': 'VOIL',
            'GAS': 'VGAS',
            'WATER': 'VWATER'
        }

        for fluid in self.fluids:
            column_name = fluid_column_mapping.get(fluid, f'V{fluid}')
            output_data[column_name] = fluid_volumes[fluid]

        # Add total porosity (sum of fluid volumes)
        output_data['PHIT_CONSTRUCTED'] = porosity_constructed

        # Add error and reconstructed logs
        output_data.update({
            'AVG_ERROR': average_errors,
            'OPTIMIZER_SUCCESS': success_flags,
            'GR_RECONSTRUCTED': gr_reconstructed,
            'NPHI_RECONSTRUCTED': nphi_reconstructed,
            'RHOB_RECONSTRUCTED': rhob_reconstructed,
            'PEF_RECONSTRUCTED': pef_reconstructed,
            'DTC_RECONSTRUCTED': dtc_reconstructed
        })

        return pd.DataFrame(output_data)
