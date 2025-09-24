import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve, hilbert
import wellpathpy as wpp
from scipy import signal


def calculate_acoustic_impedance(rhob, dtc):
    """Calculate acoustic impedance from density and sonic logs.

    Acoustic impedance is the product of density and velocity:
    AI = rho * v
    where rho is density in g/cm³ and v is P-wave velocity in m/s

    Args:
        rhob (numpy.ndarray): Bulk density log in g/cm³
        dtc (numpy.ndarray): Compressional slowness log in us/ft

    Returns:
        numpy.ndarray: Acoustic impedance in (g/cm³)*(m/s)
    """
    # Convert slowness from us/ft to velocity in m/s
    velocity = (1 / dtc) * (1e6 / 3.281)

    # Calculate acoustic impedance
    acoustic_impedance = rhob * velocity

    return acoustic_impedance


def auto_identify_layers_from_seismic_trace(df, method='peaks_troughs', min_distance=30):
    """Auto-identify layers from seismic trace using peaks/troughs or zero crossings.

    Args:
        df (pandas.DataFrame): DataFrame with 'TRACE' and 'DEPTH' columns.
        method (str): 'peaks_troughs' or 'zero_crossings' to select boundary detection method.
        min_distance (int): Minimum distance between boundaries in array indices.

    Returns:
        pandas.DataFrame: DataFrame with 'LAYERS' column added.
    """
    seismic_trace = df['TRACE'].values
    depth = df['DEPTH']

    if method == 'peaks_troughs':
        peaks = signal.find_peaks(seismic_trace, distance=min_distance)[0]
        troughs = signal.find_peaks(-seismic_trace, distance=min_distance)[0]
        all_boundaries = np.sort(np.concatenate([peaks, troughs]))
    elif method == 'zero_crossings':
        # Find indices where the sign changes (zero crossings)
        zero_crossings = np.where(np.diff(np.sign(seismic_trace)))[0]
        # Filter zero crossings by minimum distance
        all_boundaries = zero_crossings[np.insert(np.diff(zero_crossings) >= min_distance, 0, True)]
    else:
        raise ValueError("method must be 'peaks_troughs' or 'zero_crossings'")

    # Remove boundaries that are too close together (for peaks/troughs)
    if method == 'peaks_troughs':
        too_close = np.diff(all_boundaries) < min_distance
        filtered_boundaries = all_boundaries[~np.concatenate([too_close, [False]])]
        all_boundaries = filtered_boundaries

    # Create layer labels
    layer_labels = [f'Layer_{i+1}' for i in range(len(all_boundaries))]
    layers = pd.Series(data=layer_labels, index=depth.iloc[all_boundaries], name='LAYERS', dtype='category')
    df['LAYERS'] = layers.reindex(df['DEPTH']).ffill().bfill().values
    return df


def calculate_reflectivity_from_layers(df):
    """Calculate reflection coefficients from acoustic impedance log.

    The reflection coefficient at an interface is calculated as:
    R = (Z2 - Z1)/(Z2 + Z1)
    where Z2 and Z1 are the acoustic impedances of the layers above and below the interface.

    Args:
        df (pandas.DataFrame): Input DataFrame containing well log data; RHOB, DT and LAYERS
    Returns:
        numpy.ndarray: Array of reflection coefficients at each interface
    """
    df = df.copy()
    df['AI'] = calculate_acoustic_impedance(df.RHOB, df.DT.rolling(30).mean())
    df['LAYERS'] = df['LAYERS'].ffill().bfill()  # Ensure LAYERS column is filled

    # Calculate reflectivity between consecutive layers
    df = df.sort_values(by='DEPTH')
    df['REFLECTIVITY'] = np.nan

    # Get unique layers in depth order
    layers = df['LAYERS'].unique()

    # Calculate reflectivity between consecutive layers
    for i in range(len(layers)-1):
        layer1 = layers[i]
        layer2 = layers[i+1]

        Z1 = df[df['LAYERS'] == layer1]['AI'].mean()
        Z2 = df[df['LAYERS'] == layer2]['AI'].mean()

        # Calculate reflection coefficient
        R = (Z2 - Z1)/(Z2 + Z1)

        # Assign average amplitude and reflectivity to points in the upper layer
        layer1_mask = df['LAYERS'] == layer1
        layer1_indices = df.index[layer1_mask]
        df.loc[layer1_indices[:-1], 'REFLECTIVITY'] = 0
        df.loc[layer1_indices[-1], 'REFLECTIVITY'] = R
    df['REFLECTIVITY'] = df['REFLECTIVITY'].fillna(0)
    return df.sort_values(by='DEPTH')


def calculate_reflectivity_each_step(df):
    """Calculate reflection coefficients for each step in the well log.

    Args:
        df (pandas.DataFrame): Input DataFrame containing well log data; LAYERS and AI

    Returns:
        pandas.DataFrame: DataFrame with REFLECTIVITY column added.
    """
    df = df.copy()
    df['AI'] = calculate_acoustic_impedance(df.RHOB, df.DT.rolling(30).mean())
    df['AVG_DIFF_AI'] = df['AI'].rolling(50).mean().diff()
    df['AVG_DIFF_AI_2'] = df['AVG_DIFF_AI'].shift()
    df['REFLECTIVITY'] = (df['AVG_DIFF_AI_2'] - df['AVG_DIFF_AI']) / (df['AVG_DIFF_AI_2'] + df['AVG_DIFF_AI'])
    # Remove outliers from REFLECTIVITY using IQR method
    q1, q3 = df['REFLECTIVITY'].quantile([0.25, 0.75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    df.loc[(df['REFLECTIVITY'] < lower_bound) | (df['REFLECTIVITY'] > upper_bound), 'REFLECTIVITY'] = np.nan
    std = df['REFLECTIVITY'].std()
    df.loc[df['REFLECTIVITY'].between(-std, std), 'REFLECTIVITY'] = np.nan
    df['REFLECTIVITY'] = df['REFLECTIVITY'].fillna(0)
    return df.sort_values(by='DEPTH')


def convert_depth_to_time(depth, velocity):
    """Convert depth domain to time domain using velocity information.

    The two-way-time (TWT) is calculated by integrating interval times over depth:
    1. Calculate interval time: dt = 2/velocity (s/m)
    2. Multiply by depth gradient: dt * gradient(depth)
    3. Cumulative sum to get total TWT: cumsum(dt * gradient(depth))

    Args:
        depth (numpy.ndarray): Depth values in meters
        velocity (numpy.ndarray): P-wave velocity in m/s. Must be same length as depth.

    Returns:
        numpy.ndarray: Two-way-time values in seconds. Array has same length as input depth.

    Note:
        This uses integration rather than simple division to account for varying
        velocities at different depths. The gradient and cumsum ensure proper
        accumulation of travel times through each depth interval.
    """
    # Calculate interval time from velocity
    dt = 2 / velocity  # Two-way interval time in seconds per meter
    # Integrate interval time over depth using cumsum to get total time
    twt = np.cumsum(dt * np.gradient(depth))  # Two-way time in seconds

    return twt


def convert_md_to_tvd(df, well_coords):
    """Convert MD to TVD using well coordinates and resample to 0.5m interval.
    Args:
        df (pandas.DataFrame): Input DataFrame containing well log data; MD and DEPTH
        well_coords (pandas.DataFrame): Well coordinates with columns for MD, INCL and AZIM
    Returns:
        pandas.DataFrame: DataFrame with TVD column added.
    """
    dev_survey = wpp.deviation(md=well_coords['md'], inc=well_coords['incl'], azi=well_coords['azim'])
    df['TVD'] = dev_survey.minimum_curvature().resample(df.DEPTH.values).depth

    # Get list of columns to resample
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    object_cols = df.select_dtypes(include='object').columns.tolist()

    # Create new TVD range with 0.5m increment
    tvd_range = np.arange(df.TVD.min(), df.TVD.max(), 0.5).round(4)

    # Create new dataframe resampled on TVD
    resampled_df = pd.DataFrame({'TVD': tvd_range})

    # Interpolate numeric columns based on TVD
    for col in numeric_cols:
        if col != 'TVD':
            resampled_df[col] = np.interp(tvd_range, df.TVD, df[col]).round(4)

    # Forward fill object columns based on nearest TVD
    for col in object_cols:
        # Create temporary series indexed by TVD
        temp_series = pd.Series(df[col].values, index=df.TVD)
        # Reindex to new TVD values and forward fill
        resampled_df[col] = temp_series.reindex(tvd_range, method='ffill')

    return resampled_df


def extract_seismic_along_well(seismic_cube, well_trajectory):
    """Extract seismic data along a well trajectory from a 3D seismic cube.

    Args:
        seismic_cube (xarray.Dataset): 3D seismic cube with coordinates (inline, xline, twt/depth)
        well_trajectory (pandas.DataFrame): Well trajectory with columns for X,Y coordinates and depth/time
        method (str, optional): Interpolation method - 'nearest' or 'linear'. Defaults to 'nearest'.

    Returns:
        numpy.ndarray: Seismic trace values extracted along the well path

    Notes:
        - The seismic cube and well trajectory must be in the same coordinate reference system
        - For time domain seismic, the well trajectory depth must be converted to time first
        - Uses xarray's interp() method for extraction
        - Handles both inline/xline and CDP coordinate systems
    """
    well_ilxl, z = convert_well_trajectory_to_ilxl(seismic_cube, well_trajectory)

    # Extract seismic trace
    well_seismic_trace = seismic_cube.interp(**well_ilxl, samples=z)

    return well_seismic_trace.data


def convert_well_trajectory_to_ilxl(seismic_cube, well_trajectory):
    """Convert well trajectory to inline/crossline coordinates.

    Args:
        seismic_cube (xarray.Dataset): 3D seismic cube with coordinates (inline, xline, twt/depth)
        well_trajectory (pandas.DataFrame): Well trajectory with columns for X,Y coordinates and depth/time
    """
    import xarray as xr

    # Scale the coordinates if required
    try:
        seismic_cube.segysak.scale_coords()
    except Exception as e:
        print(f"Error scaling coordinates: {e}")

    # Verify required columns exist in well trajectory
    required_cols = ['md', 'incl', 'azim', 'x', 'y', 'z']
    missing_cols = [col for col in required_cols if col not in well_trajectory.columns]
    if missing_cols:
        raise ValueError(f"Well trajectory missing required columns: {missing_cols}")

    # Recalculate well deviation for higher sampling rate
    well_dev_pos = wpp.deviation(
        well_trajectory['md'], well_trajectory['incl'], well_trajectory['azim']
    )
    # depth values in MD that we want to sample the seismic cube at
    new_depths = np.arange(0, well_trajectory['md'].max(), .1)

    # use minimum curvature and resample to 1m interval
    well_dev_pos = well_dev_pos.minimum_curvature().resample(new_depths)

    # adjust position of deviation to local coordinates and TVDSS
    well_dev_pos.to_wellhead(
        well_trajectory['y'].iloc[0],
        well_trajectory['x'].iloc[0],
        inplace=True,
    )

    # Get resampled md based on well_dev_pos
    md_depths = np.linspace(0, well_trajectory['md'].max(), len(well_dev_pos.depth))

    affine = seismic_cube.segysak.get_affine_transform().inverted()
    ilxl = affine.transform(np.dstack([well_dev_pos.easting, well_dev_pos.northing])[0])
    well_ilxl = dict(
        iline=xr.DataArray(ilxl[:, 0], dims="well", coords={"well": md_depths}),
        xline=xr.DataArray(ilxl[:, 1], dims="well", coords={"well": md_depths}),
    )
    z = xr.DataArray(
        1.0 * well_dev_pos.depth,
        dims="well",
        coords={"well": md_depths},
    )
    return well_ilxl, z


def plot_well_trajectory(seismic_cube, well_coords):
    """Plot well trajectory in map view and cross sections.

    Args:
        well_coords: DataFrame with x,y,z coordinates

    Returns:
        fig: Figure object
    """
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 3, 4)
    ax3 = fig.add_subplot(2, 3, 5)
    ax4 = fig.add_subplot(2, 3, 6)

    # Print seismic CDP coordinates
    print("\nSeismic CDP coordinates:")
    print(f"CDP X range: {seismic_cube.cdp_x.min().values:.1f} to {seismic_cube.cdp_x.max().values:.1f}")
    print(f"CDP Y range: {seismic_cube.cdp_y.min().values:.1f} to {seismic_cube.cdp_y.max().values:.1f}")

    # Plot well location and seismic coverage
    ax1.scatter(seismic_cube.cdp_x, seismic_cube.cdp_y, c='lightgray', s=1, alpha=0.1, label='Seismic coverage')
    ax1.scatter(well_coords['x'], well_coords['y'], c='red', s=20, label='Well trajectory')
    ax1.axis('equal')
    ax1.grid(True)
    ax1.set_xlabel('Easting (m)')
    ax1.set_ylabel('Northing (m)')
    ax1.set_title('Well Location vs Seismic Coverage')

    # Plot X/Y view (map view)
    ax2.plot(well_coords.x, well_coords.y, 'r-', label='Well path')
    ax2.scatter(well_coords.x.iloc[0], well_coords.y.iloc[0],
                color='green', s=100, marker='^', label='Well head')
    ax2.scatter(well_coords.x.iloc[-1], well_coords.y.iloc[-1],
                color='red', s=100, marker='v', label='Well bottom')
    ax2.grid(True)
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_title('Map View')
    ax2.legend()

    # Plot X/Z view (cross-section)
    ax3.plot(well_coords.x, well_coords.z, 'r-', label='Well path')
    ax3.scatter(well_coords.x.iloc[0], well_coords.z.iloc[0],
                color='green', s=100, marker='^', label='Well head')
    ax3.scatter(well_coords.x.iloc[-1], well_coords.z.iloc[-1],
                color='red', s=100, marker='v', label='Well bottom')
    ax3.grid(True)
    ax3.set_xlabel('X (m)')
    ax3.set_ylabel('Z (m)')
    ax3.invert_yaxis()  # Invert depth axis
    ax3.set_title('Cross-section View')
    ax3.legend()

    # Plot Y/Z view (cross-section)
    ax4.plot(well_coords.y, well_coords.z, 'r-', label='Well path')
    ax4.scatter(well_coords.y.iloc[0], well_coords.z.iloc[0],
                color='green', s=100, marker='^', label='Well head')
    ax4.scatter(well_coords.y.iloc[-1], well_coords.z.iloc[-1],
                color='red', s=100, marker='v', label='Well bottom')
    ax4.grid(True)
    ax4.set_xlabel('Y (m)')
    ax4.set_ylabel('Z (m)')
    ax4.invert_yaxis()  # Invert depth axis
    ax4.set_title('Cross-section View')
    ax4.legend()

    plt.tight_layout()

    return fig


def plot_seismic_along_well(seismic_cube, well_trajectory):
    """Plot seismic trace along well trajectory.

    Args:
        seismic_cube (xarray.Dataset): 3D seismic cube with coordinates (inline, xline, twt/depth)
        well_trajectory (pandas.DataFrame): Well trajectory with columns for X,Y coordinates and depth/time
    """
    well_ilxl, z = convert_well_trajectory_to_ilxl(seismic_cube, well_trajectory)
    seismic_along_well = seismic_cube.interp(**well_ilxl)
    seismic_traces_along_well = extract_seismic_along_well(seismic_cube, well_trajectory)

    # Plot the extracted trace
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))
    seismic_along_well.data.T.plot(ax=axs[0], yincrease=False)
    z.plot(ax=axs[0], color='k')
    axs[0].grid(True)
    axs[0].set_xlabel('Well')
    axs[0].set_ylabel('Z (m)')
    axs[0].set_title('Seismic Trace Along Well')

    axs[1].plot(seismic_traces_along_well)
    axs[1].grid(True)
    axs[1].set_xlabel('Well')
    axs[1].set_ylabel('Amplitude')
    axs[1].set_title('Seismic Traces Along Well')

    plt.tight_layout()

    return fig


def optimize_wavelet(seismic_trace, reflectivity):
    """Estimate the seismic wavelet by matching synthetic to actual seismic trace.

    This implementation uses a combination of frequency-domain and time-domain methods
    based on the approaches described in:
    - White (1980) "Inverse Problems in Reflection Seismology"
    - Edgar & van der Baan (2011) "How reliable is statistical wavelet estimation?"
    - Rosa (2018) "Seismic Inversion Methods"

    Args:
        seismic_trace (numpy.ndarray): Observed seismic trace
        reflectivity (numpy.ndarray): Reflectivity series
        wavelet_length (int): Length of wavelet in samples (should be odd)
        max_iter (int): Maximum iterations for optimization

    Returns:
        numpy.ndarray: Estimated wavelet
        float: Final error (RMSE) between synthetic and seismic trace
        numpy.ndarray: Synthetic seismic trace generated with the estimated wavelet
    """
    # Initial guess: zero-phase Ricker wavelet with statistical estimation
    def ricker_wavelet(f0=30, phase=0.0):
        """Generate a Ricker wavelet with specified parameters.

        Args:
            length (int): Length of wavelet in samples
            f0 (float): Peak frequency in Hz
            phase (float): Phase rotation in degrees

        Returns:
            numpy.ndarray: Ricker wavelet
        """
        dt = 0.001
        tmin = -0.15
        tmax = 0.15
        t = np.arange(tmin, tmax, dt)
        pi2 = np.pi * np.pi

        # Generate zero-phase Ricker wavelet
        w = (1 - 2*pi2*f0**2*t**2) * np.exp(-pi2*f0**2*t**2)

        # Apply phase rotation
        phase_rad = np.deg2rad(phase)
        w = w * np.cos(phase_rad) + np.imag(hilbert(w)) * np.sin(phase_rad)

        # Apply tapering to reduce edge effects
        taper = np.hanning(len(w))
        return w * taper

    # Objective function using Wiener deconvolution approach
    def objective(params):
        f0, phase = params
        wavelet = ricker_wavelet(f0, phase)
        # Generate synthetic and normalize both traces
        synthetic = convolve(reflectivity, wavelet, mode='same')
        synthetic = (synthetic - np.mean(synthetic)) / np.std(synthetic)
        return np.sqrt(np.mean((synthetic - seismic_trace)**2))

    # Set bounds to maintain stability
    bounds = [
        (10, 100),  # frequency bounds
        (-180, 180)  # phase bounds
    ]

    init_params = np.array([
        30,  # frequency
        0.0,  # phase
    ])

    # Optimize with multiple restarts
    best_result = None
    best_error = np.inf
    for _ in range(3):  # Try multiple initializations
        result = minimize(
            objective,
            init_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 100}
        )

        if result.fun < best_error:
            best_error = result.fun
            best_result = result

    best_params = best_result.x
    final_error = best_result.fun
    print(f"Best parameters: {best_params}")

    # Generate final synthetic trace
    estimated_wavelet = ricker_wavelet(best_params[0], best_params[1])
    synthetic_trace = convolve(reflectivity, estimated_wavelet, mode='same')

    return estimated_wavelet, final_error, synthetic_trace


def create_synthetic_seismogram(wavelet, reflectivity):
    """Create synthetic seismogram by convolving wavelet with reflectivity series.

    Args:
        wavelet (numpy.ndarray): Seismic wavelet
        reflectivity (numpy.ndarray): Reflectivity coefficient series

    Returns:
        numpy.ndarray: Synthetic seismogram trace
    """
    # Convolve wavelet with reflectivity series
    synthetic = convolve(reflectivity, wavelet, mode='same')

    return synthetic


def qc_synthetic_seismogram(synthetic, seismic, title="Synthetic vs Seismic"):
    """Quality control plot comparing synthetic seismogram to actual seismic trace.

    Args:
        synthetic (numpy.ndarray): Synthetic seismogram trace
        seismic (numpy.ndarray): Actual seismic trace
        title (str, optional): Plot title. Defaults to "Synthetic vs Seismic".

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot traces side by side
    samples = np.arange(len(synthetic))
    ax1.plot(synthetic, samples, 'b-', label='Synthetic')
    ax1.plot(seismic, samples, 'r-', label='Seismic')
    ax1.set_ylim(ax1.get_ylim()[::-1])  # Reverse y-axis
    ax1.grid(True)
    ax1.legend()
    ax1.set_xlabel('Amplitude')
    ax1.set_ylabel('Sample')
    ax1.set_title('Trace Comparison')

    # Crossplot
    ax2.scatter(seismic, synthetic, alpha=0.5)
    ax2.plot([-1, 1], [-1, 1], 'r--')  # 1:1 line
    ax2.grid(True)
    ax2.set_xlabel('Seismic Amplitude')
    ax2.set_ylabel('Synthetic Amplitude')
    ax2.set_title('Crossplot')

    # Calculate correlation coefficient
    corr = np.corrcoef(synthetic, seismic)[0, 1]
    ax2.text(0.05, 0.95, f'Correlation: {corr:.3f}',
             transform=ax2.transAxes, verticalalignment='top')

    plt.suptitle(title)
    plt.tight_layout()

    return fig
