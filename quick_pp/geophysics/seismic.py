import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.signal import convolve
import wellpathpy as wpp
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


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


def auto_identify_layers(gr, rhob, dtc, depth):
    """Auto identify layers using PCA on multiple well logs.

    This function uses Principal Component Analysis (PCA) to identify distinct
    layers by analyzing multiple log curves simultaneously. It looks for
    natural clustering in the transformed space.

    Args:
        gr (numpy.ndarray): Gamma ray log values
        rhob (numpy.ndarray): Bulk density log values
        dtc (numpy.ndarray): Sonic compressional slowness log values

    Returns:
        numpy.ndarray: Array of identified layer boundaries
    """
    # Stack logs into feature matrix
    X = np.column_stack([gr, rhob, dtc])

    # Remove any rows with NaN values
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Use elbow method to determine optimal number of clusters
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X_scaled)
        distortions.append(kmeans.inertia_)

    # Find elbow point (simple method)
    k_optimal = np.argmin(np.diff(distortions)) + 7

    # Final clustering
    kmeans = KMeans(n_clusters=k_optimal, random_state=42)
    labels = kmeans.fit_predict(X_scaled)

    # Find layer boundaries where cluster labels change
    layer_boundaries = np.where(np.diff(labels) != 0)[0]

    # Map back to original indices accounting for removed NaN values
    if not np.all(valid_mask):
        valid_indices = np.where(valid_mask)[0]
        layer_boundaries = valid_indices[layer_boundaries]

    # Create a categorical pandas Series of layer boundaries as layers
    layer_labels = [f'Layer_{i+1}' for i in range(len(layer_boundaries))]
    LAYERS = pd.Series(data=layer_labels, index=depth.iloc[layer_boundaries], name='LAYERS', dtype='category')
    return LAYERS


def calculate_reflectivity(df):
    """Calculate reflection coefficients from acoustic impedance log.

    The reflection coefficient at an interface is calculated as:
    R = (Z2 - Z1)/(Z2 + Z1)
    where Z2 and Z1 are the acoustic impedances of the layers above and below the interface.

    Args:
        acoustic_impedance (numpy.ndarray): Array of acoustic impedance values
        layers (pandas.Series): Series of layer boundaries
    Returns:
        numpy.ndarray: Array of reflection coefficients at each interface
    """
    df = df.copy()
    df['AI'] = calculate_acoustic_impedance(df.RHOB, df.DT)
    layers = auto_identify_layers(df.GR, df.RHOB, df.DT, df.DEPTH)
    df = pd.merge_asof(df, layers, on='DEPTH')
    df['LAYERS'] = df['LAYERS'].ffill().bfill()

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

        # Assign reflectivity to all points in the upper layer
        df.loc[df['LAYERS'] == layer1, ['AVG_AI', 'REFLECTIVITY']] = Z1, R

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
    new_depths = np.arange(0, int(well_trajectory['md'].max()), 1)

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


def optimize_wavelet(seismic_trace, reflectivity, wavelet_length=31, max_iter=100):
    """Estimate the seismic wavelet by matching synthetic to actual seismic trace.

    Returns:
        numpy.ndarray: Estimated wavelet
        float: Final error (RMSE) between synthetic and seismic trace
        numpy.ndarray: Synthetic seismic trace generated with the estimated wavelet
    """
    # Ensure wavelet length is odd
    if wavelet_length % 2 == 0:
        wavelet_length += 1

    # Initial guess: zero-phase Ricker wavelet
    def ricker_wavelet(length, a=2.0):
        t = np.linspace(-length // 2, length // 2, length)
        pi2 = (np.pi ** 2)
        return (1 - 2 * pi2 * (t / a) ** 2) * np.exp(-pi2 * (t / a) ** 2)

    init_wavelet = ricker_wavelet(wavelet_length)

    # Objective: minimize RMSE between synthetic and seismic
    def objective(wavelet):
        synthetic = convolve(reflectivity, wavelet, mode='same')
        scaled_synthetic = StandardScaler().fit_transform(synthetic.reshape(-1, 1)).flatten()
        scaled_seismic = StandardScaler().fit_transform(seismic_trace.values.reshape(-1, 1)).flatten()
        error = np.sqrt(np.mean((scaled_synthetic - scaled_seismic) ** 2))
        return error

    bounds = [(-2, 2)] * wavelet_length

    result = minimize(
        objective,
        init_wavelet,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': max_iter}
    )

    estimated_wavelet = result.x
    final_error = result.fun

    # To produce the synthetic seismic trace with the estimated wavelet:
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
