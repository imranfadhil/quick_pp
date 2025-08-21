import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve
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
    velocity = (1 / dtc) * (1e6 / 0.3048)

    # Calculate acoustic impedance
    acoustic_impedance = rhob * velocity

    return acoustic_impedance


def auto_identify_layers_from_seismic_trace(df):
    """Auto-identify layers from seismic trace.

    Args:
        seismic_trace (numpy.ndarray): Seismic trace
        depth (numpy.ndarray): Depth values
    """
    seismic_trace = df['TRACE'].values
    depth = df['DEPTH']
    peaks = signal.find_peaks(seismic_trace, distance=30)[0]
    troughs = signal.find_peaks(-seismic_trace, distance=30)[0]

    # Combine peaks and troughs
    all_boundaries = np.sort(np.concatenate([peaks, troughs]))

    # Find indices where boundaries are too close together
    min_distance = 30  # Minimum distance between boundaries in array indices
    too_close = np.diff(all_boundaries) < min_distance

    # Keep only boundaries that are far enough apart
    filtered_boundaries = all_boundaries[~np.concatenate([too_close, [False]])]
    all_boundaries = filtered_boundaries

    # Create layer labels
    layer_labels = [f'Layer_{i+1}' for i in range(len(all_boundaries))]

    # Create categorical pandas Series of layer boundaries
    layers = pd.Series(data=layer_labels, index=depth.iloc[all_boundaries], name='LAYERS', dtype='category')
    df['LAYERS'] = layers.reindex(df['DEPTH']).ffill().bfill().values
    return df


def calculate_reflectivity(df):
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
    df['AI'] = calculate_acoustic_impedance(df.RHOB, df.DT)
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


def convert_depth_to_time(depth, velocity):
    """Convert depth domain to time domain using velocity information.

    The two-way-time (TWT) is calculated as:
    TWT = 2 * depth / velocity
    where depth is in meters and velocity is in m/s

    Args:
        depth (numpy.ndarray): Depth values in meters
        velocity (numpy.ndarray): P-wave velocity in m/s. Must be same length as depth.

    Returns:
        numpy.ndarray: Two-way-time values in seconds
    """
    # Calculate two-way-time
    twt = 2 * depth / velocity

    return twt


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
        well_trajectory['y'][0],
        well_trajectory['x'][0],
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


def optimize_wavelet(seismic_trace, reflectivity, wavelet_length=601, max_iter=100):
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
    # Ensure wavelet length is odd and reasonable (typically 100-300ms)
    if wavelet_length % 2 == 0:
        wavelet_length += 1

    # Initial guess: zero-phase Ricker wavelet with statistical estimation
    def ricker_wavelet(length, f0=30):
        t = np.linspace(-length/2, length/2, length) / 1000  # Convert to seconds
        pi2 = np.pi * np.pi
        w = (1 - 2*pi2*f0*f0*t*t) * np.exp(-pi2*f0*f0*t*t)
        return w / np.max(np.abs(w))  # Normalize

    # Estimate dominant frequency from seismic trace using FFT
    fft = np.fft.fft(seismic_trace)
    freqs = np.fft.fftfreq(len(seismic_trace))
    dom_freq = np.abs(freqs[np.argmax(np.abs(fft))])
    init_wavelet = ricker_wavelet(wavelet_length, f0=dom_freq*100)

    # Objective function using Wiener deconvolution approach
    def objective(wavelet):
        # Apply tapering to reduce edge effects
        taper = np.hanning(len(wavelet))
        tapered_wavelet = wavelet * taper

        # Generate synthetic and normalize both traces
        synthetic = convolve(reflectivity, tapered_wavelet, mode='same')
        synthetic = (synthetic - np.mean(synthetic)) / np.std(synthetic)
        seismic_norm = (seismic_trace - np.mean(seismic_trace)) / np.std(seismic_trace)

        # Compute error in both time and frequency domains
        time_error = np.sqrt(np.mean((synthetic - seismic_norm) ** 2))

        # Add frequency domain constraint
        fft_syn = np.fft.fft(synthetic)
        fft_real = np.fft.fft(seismic_norm)
        freq_error = np.sqrt(np.mean(np.abs(fft_syn - fft_real) ** 2))

        return time_error + 0.5 * freq_error

    # Set bounds to maintain stability
    bounds = [(-1, 1)] * wavelet_length

    # Optimize using L-BFGS-B with multiple restarts
    best_result = None
    best_error = np.inf

    for _ in range(3):  # Try multiple initializations
        result = minimize(
            objective,
            init_wavelet,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter}
        )

        if result.fun < best_error:
            best_error = result.fun
            best_result = result

    estimated_wavelet = best_result.x
    final_error = best_result.fun

    # Apply phase correction to ensure zero-phase
    estimated_wavelet = np.roll(estimated_wavelet, -np.argmax(estimated_wavelet) + len(estimated_wavelet)//2)

    # Generate final synthetic trace
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
