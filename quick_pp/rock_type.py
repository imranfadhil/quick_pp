import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from quick_pp import logger
from quick_pp.lithology import shale_volume_steiber

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {"axes.labelsize": 10, "xtick.labelsize": 10, "legend.fontsize": "small"}
)


def calc_rqi(pore, perm):
    """Calculate RQI (Rock Quality Index) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        pore (np.ndarray or float): Total porosity in fraction.
        perm (np.ndarray or float): Permeability in mD.

    Returns:
        np.ndarray or float: The Rock Quality Index (RQI).
    """
    return 0.0314 * (perm / pore) ** 0.5


def calc_fzi(pore, perm):
    """Calculate FZI (Flow Zone Indicator) from Kozeny-Carman equation, based on Amaefule et al. (1993)

    Args:
        pore (np.ndarray or float): Total porosity in fraction.
        perm (np.ndarray or float): Permeability in mD.

    Returns:
        np.ndarray or float: The Flow Zone Indicator (FZI).
    """
    return calc_rqi(pore, perm) / (pore / (1 - pore))


def calc_fzi_perm(fzi, pore):
    """Calculate permeability from FZI and porosity, based on Amaefule et al. (1993)

    Args:
        fzi (np.ndarray or float): Flow Zone Indicator.
        pore (np.ndarray or float): Total porosity in fraction.

    Returns:
        np.ndarray or float: Permeability in mD.
    """
    return pore * ((pore * fzi) / (0.0314 * (1 - pore))) ** 2


def calc_r35(pore, perm):
    """Calculate Winland R35 from Kozeny-Carman equation, based on Winland (1979)

    Args:
        pore (np.ndarray or float): Total porosity in fraction.
        perm (np.ndarray or float): Permeability in mD.

    Returns:
        np.ndarray or float: The Winland R35 value.
    """
    return 10 ** (0.732 + 0.588 * np.log10(perm) - 0.864 * np.log10(pore * 100))


def calc_r35_perm(r35, pore):
    """Calculate permeability from Winland R35 and porosity, based on Winland (1979)

    Args:
        r35 (np.ndarray or float): Winland R35 value.
        pore (np.ndarray or float): Total porosity in fraction.

    Returns:
        np.ndarray or float: Permeability in mD.
    """
    return 10 ** ((np.log10(r35) - 0.732 + 0.864 * np.log10(pore * 100)) / 0.588)


def plot_rqi(
    cpore, cperm, cut_offs=None, rock_type=None, title="Rock Quality Index (RQI)"
):
    """Generate a Rock Quality Index (RQI) cross-plot.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        cut_offs (list, optional): List of RQI values to plot as lines.
                                   Defaults to None, which will use [0.1, 0.3, 0.5, 1.0].
        rock_type (array, optional): Array of rock types for coloring points. Defaults to None.
        title (str, optional): Plot title. Defaults to 'Rock Quality Index (RQI)'.
    """
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker=".", c=rock_type, cmap="viridis")
    cut_offs = cut_offs if cut_offs is not None else [0.1, 0.3, 0.5, 1.0]
    pore_points = np.geomspace(0.001, 1, 20)
    for rqi in cut_offs:
        perm_points = pore_points * (rqi / 0.0314) ** 2
        ax.plot(
            pore_points, perm_points, linestyle="dashed", label=f"RQI={round(rqi, 3)}"
        )

    plt.xlabel("Porosity (frac)")
    plt.xlim(-0.05, 0.5)
    plt.ylabel("Permeability (mD)")
    plt.ylim(1e-3, 1e5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    plt.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def plot_fzi(
    cpore, cperm, cut_offs=None, rock_type=None, title="Flow Zone Indicator (FZI)"
):
    """Generate a Flow Zone Indicator (FZI) cross-plot.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        cut_offs (list, optional): List of FZI values to plot as lines. Defaults to `np.arange(0.5, 5)`.
        rock_type (array, optional): Array of rock types for coloring points. Defaults to None.
        title (str, optional): Plot title. Defaults to 'Flow Zone Indicator (FZI)'.
    """
    # Plot the FZI cross plot
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker=".", c=rock_type, cmap="viridis")
    cut_offs = cut_offs if cut_offs is not None else np.arange(0.5, 5)
    phit_points = np.geomspace(0.001, 1, 20)
    prt_num = len(cut_offs)
    ax.annotate(
        f"PRT {prt_num + 1}",
        xy=(0.3, 0.7),
        xytext=(1, 1),
        textcoords="offset points",
        fontsize=10,
        fontweight="bold",
    )
    for i, fzi in enumerate(cut_offs):
        perm_points = (
            phit_points * ((phit_points * fzi) / (0.0314 * (1 - phit_points))) ** 2
        )
        ax.plot(
            phit_points, perm_points, linestyle="dashed", label=f"FZI={round(fzi, 3)}"
        )
        prt_num = len(cut_offs) - i
        ax.annotate(
            f"PRT {prt_num}",
            xy=(0.3, perm_points[-4]),
            xytext=(1, 1),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
        )

    plt.xlabel("Porosity (frac)")
    plt.xlim(-0.05, 0.5)
    plt.ylabel("Permeability (mD)")
    plt.ylim(1e-3, 1e5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )

    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    plt.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def plot_rfn(cpore, cperm, rock_type=None, title="Lucia RFN"):
    """Plot the Rock Fabric Number (RFN) lines on porosity and permeability cross plot. The permeability (mD) is
    calculated based on Lucia-Jenkins, 2003.
    ```
    perm = 10**(9.7892 - 12.0838 * log(RFN) + (8.6711 - 8.2965 * log(RFN)) * log(phi))
    ```

    Args:
        cpore (np.ndarray or float): Core porosity in v/v.
        cperm (np.ndarray or float): Core permeability in mD.
        rock_type (array, optional): Array of rock types for coloring points. Defaults to None.
        title (str, optional): Plot title. Defaults to 'Lucia RFN'.
    """
    # Plot the RFN cross plot
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker=".", c=rock_type, cmap="viridis")
    pore_points = np.linspace(0, 0.6, 20)
    for rfn in np.arange(0.5, 4.5, 0.5):
        perm_points = 10 ** (
            9.7892
            - 12.0838 * np.log10(rfn)
            + (8.6711 - 8.2965 * np.log10(rfn)) * np.log10(pore_points)
        )
        plt.plot(pore_points, perm_points, linestyle="dashed", label=f"RFN={rfn}")

    plt.xlabel("Porosity (frac)")
    plt.xlim(-0.05, 0.5)
    plt.ylabel("Permeability (mD)")
    plt.ylim(1e-3, 1e5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )

    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    plt.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def plot_winland(cpore, cperm, cut_offs=None, rock_type=None, title="Winland R35"):
    """Plot the Winland R35 lines on porosity and permeability cross plot. The permeability (mD) is calculated based on
    Winland, 1972.
    ```
    perm = 10**((log(r35) - 0.732 + 0.864 * log(phi)) / 0.588)
    ```

    Args:
        cpore (np.ndarray or float): Core porosity in v/v.
        cperm (np.ndarray or float): Core permeability in mD.
        cut_offs (list, optional): List of R35 values to plot as lines. Defaults to [.05, .1, .5, 2, 10, 100].
        rock_type (array, optional): Array of rock types for coloring points. Defaults to None.
        title (str, optional): Plot title. Defaults to 'Winland R35'.
    """
    # Plot the Winland R35 cross plot
    _, ax = plt.subplots(figsize=(10, 8))
    plt.title(title)
    plt.scatter(cpore, cperm, marker=".", c=rock_type, cmap="viridis")
    cut_offs = cut_offs if cut_offs is not None else [0.05, 0.1, 0.5, 2, 10, 100]
    pore_points = np.linspace(0.001, 0.6, 20)
    for r35 in cut_offs:
        perm_points = calc_r35_perm(r35, pore_points)
        plt.plot(pore_points, perm_points, linestyle="dashed", label=f"R35={r35}")

    plt.xlabel("Porosity (frac)")
    plt.xlim(-0.05, 0.5)
    plt.ylabel("Permeability (mD)")
    plt.ylim(1e-3, 1e5)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    plt.yscale("log")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda x, pos: (
                "{{:.{:1d}f}}".format(int(np.maximum(-np.log10(x), 0)))
            ).format(x)
        )
    )

    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    plt.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def plot_neuden_vsh(
    nphi,
    rhob,
    vshale_lines=None,
    shale_point=(0.45, 2.55),
    sand_point=(-0.035, 2.65),
    lime_point=(0.0, 2.71),
    dolo_point=(0.02, 2.87),
    fluid_point=(1.0, 1.0),
    rock_flag=None,
):
    """
    Generate a Neutron-Density crossplot with Vshale lines for rock typing.

    This plot helps in identifying lithology and visually estimating shale volume.
    It includes standard matrix lines for sandstone, limestone, and dolomite,
    and overlays lines of constant shale volume.

    Args:
        nphi (pd.Series or np.ndarray): Neutron Porosity log values (v/v).
        rhob (pd.Series or np.ndarray): Bulk Density log values (g/cc).
        shale_point (tuple, optional): (NPHI, RHOB) coordinates of the 100% shale point.
                                       Defaults to (0.45, 2.55).
        sand_point (tuple, optional): (NPHI, RHOB) coordinates of the sandstone matrix.
                                      Defaults to (-0.035, 2.65).
        lime_point (tuple, optional): (NPHI, RHOB) coordinates of the limestone matrix.
                                      Defaults to (0.0, 2.71).
        dolo_point (tuple, optional): (NPHI, RHOB) coordinates of the dolomite matrix.
                                      Defaults to (0.02, 2.87).
        rock_flag (pd.Series or np.ndarray, optional): An array of values (e.g., GR, Vshale, or rock types)
                                                       to color the data points. Defaults to None.
        fluid_point (tuple, optional): (NPHI, RHOB) coordinates of the fluid point (water).
                                       Defaults to (1.0, 1.0).
        vshale_lines (list, optional): List of Vshale fractions (0 to 1) to plot as lines.
                                       Defaults to [0, 0.25, 0.5, 0.75, 1.0].
    """
    if vshale_lines is None:
        vshale_lines = [0, 0.25, 0.5, 0.75, 1.0]

    _, ax = plt.subplots(figsize=(10, 8))
    ax.set_title("Neutron-Density Crossplot with Vshale Lines")

    # Handle plotting based on whether rock_flag is provided
    if rock_flag is None:
        # Plot all data points in a single color
        ax.scatter(nphi, rhob, marker=".", alpha=0.5, label="Log Data")
    else:
        # Plot data points colored by unique rock flags
        unique_flags = sorted(rock_flag.dropna().unique())
        colors = plt.get_cmap("tab10", len(unique_flags))

        for i, flag in enumerate(unique_flags):
            mask = rock_flag == flag
            ax.scatter(
                nphi[mask],
                rhob[mask],
                color=colors(i),
                marker=".",
                alpha=0.7,
                label=f"Flag: {flag}",
            )
        ax.legend(title=rock_flag.name if hasattr(rock_flag, "name") else "Rock Flag")

    # Plot matrix lines
    ax.plot(
        [sand_point[0], fluid_point[0]],
        [sand_point[1], fluid_point[1]],
        "g-",
        label="Sandstone",
    )
    ax.plot(
        [lime_point[0], fluid_point[0]],
        [lime_point[1], fluid_point[1]],
        "b-",
        label="Limestone",
    )
    ax.plot(
        [dolo_point[0], fluid_point[0]],
        [dolo_point[1], fluid_point[1]],
        "r-",
        label="Dolomite",
    )

    # Plot Vshale lines (from clean sand to shale point)
    for vsh in vshale_lines:
        nphi_line = sand_point[0] * (1 - vsh) + shale_point[0] * vsh
        rhob_line = sand_point[1] * (1 - vsh) + shale_point[1] * vsh
        start_point = np.array([nphi_line, rhob_line])
        end_point = np.array(fluid_point)
        ax.plot(
            [start_point[0], end_point[0]],
            [start_point[1], end_point[1]],
            "k--",
            alpha=0.7,
        )

        # Calculate a point slightly along the line for the text, for better positioning
        direction_vector = end_point - start_point
        norm_vector = direction_vector / np.linalg.norm(direction_vector)
        # Adjust the offset_factor to move text closer or further from the matrix line
        offset_factor = 0.03
        text_pos = start_point + norm_vector * offset_factor

        ax.text(
            text_pos[0], text_pos[1], f"{int(vsh * 100)}%", ha="center", va="bottom"
        )

    ax.set_xlabel("Neutron Porosity (v/v)")
    ax.set_ylabel("Bulk Density (g/cc)")
    ax.set_xlim(-0.1, 0.6)
    ax.set_ylim(3.0, 1.8)  # Inverted Y-axis for density
    ax.legend()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def plot_fzi_log_log(
    cpore,
    cperm,
    cut_offs=None,
    rock_type=None,
    title="FZI Log-Log Plot",
    density=True,
):
    """Generate a log-log plot of RQI vs. pore-to-solid volume ratio to identify rock types.

    On this plot, data points belonging to the same rock type (i.e., same FZI) will
    fall along a straight line with a unit slope.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        cut_offs (list, optional): List of FZI values to plot as lines.
                                   Defaults to `np.arange(0.5, 5)`.
        rock_type (array, optional): Array of rock types for coloring points. Defaults to None.
        title (str, optional): Plot title. Defaults to 'FZI Log-Log Plot'.
        density (bool, optional): If True, show data density with a 2D histogram and contours.
                                  Defaults to False.
    """
    rqi = calc_rqi(cpore, cperm)
    phi_z = cpore / (1 - cpore)

    _, ax = plt.subplots(figsize=(10, 8))
    ax.set_title(title)

    if density:
        # Use hist2d to show data density
        valid_data = (phi_z > 0) & (rqi > 0) & np.isfinite(phi_z) & np.isfinite(rqi)
        x = phi_z[valid_data]
        y = rqi[valid_data]

        # Create 2D histogram
        bins = 50
        hist, x_edges, y_edges = np.histogram2d(np.log10(x), np.log10(y), bins=bins)

        # Plot filled contours
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2
        y_centers = (y_edges[:-1] + y_edges[1:]) / 2
        cf = ax.contourf(
            10**x_centers, 10**y_centers, hist.T, levels=10, cmap="viridis", alpha=0.8
        )
        plt.colorbar(cf, ax=ax, label="Data Density")
    else:
        ax.scatter(phi_z, rqi, marker=".", c=rock_type, cmap="viridis")

    cut_offs = cut_offs if cut_offs is not None else np.arange(0.5, 5)
    phi_z_points = np.geomspace(np.nanmin(phi_z[phi_z > 0]), np.nanmax(phi_z), 20)

    for fzi in cut_offs:
        rqi_points = fzi * phi_z_points
        ax.plot(
            phi_z_points,
            rqi_points,
            linestyle="dashed",
            label=f"FZI={round(fzi, 3)}",
        )

    ax.set_xlabel("Pore to Solid Volume Ratio (phi_z)")
    ax.set_ylabel("Rock Quality Index (RQI)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(bbox_to_anchor=(0.1, 1), loc="upper left")
    ax.set_aspect("equal", adjustable="box")

    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def plot_fzi_histogram(
    cpore, cperm, bins="auto", cutoffs=None, title="Log(FZI) Histogram"
):
    """
    Plot a histogram of Log(FZI) to identify modes and define cutoffs.

    Different rock types should appear as distinct modes (peaks) in the histogram.
    The troughs between these modes can be used as objective cutoffs.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        bins (int or str, optional): The number of bins for the histogram. Defaults to "auto".
        cutoffs (list, optional): A list of Log(FZI) cutoff values to display as vertical lines. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Log(FZI) Histogram".
    """
    # Calculate Log(FZI)
    fzi = calc_fzi(cpore, cperm)
    log_fzi = np.log10(fzi)

    # Clean data by removing infinite and NaN values
    log_fzi = log_fzi[np.isfinite(log_fzi)]

    # Generate histogram
    _, ax = plt.subplots(figsize=(10, 6))
    ax.hist(log_fzi, bins=bins, edgecolor="black", alpha=0.7)
    ax.set_title(title)
    ax.set_xlabel("Log(FZI)")
    ax.set_ylabel("Frequency")

    # Plot cutoff lines if provided
    if cutoffs:
        for cutoff in cutoffs:
            ax.axvline(x=cutoff, color="r", linestyle="--", label=f"Cutoff={cutoff}")
        ax.legend()

    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")


def plot_ward_dendogram(X, p=10, title="Ward's Dendogram"):
    """Generate and display a Ward's dendrogram for hierarchical clustering.

    This function is useful for visualizing the hierarchical relationships within a
    dataset to identify natural groupings or clusters.

    Args:
        X (np.ndarray): The input data array for clustering.
        p (int, optional): The number of clusters to display in the truncated dendrogram. Defaults to 10.
        title (str, optional): The title of the plot. Defaults to "Ward's Dendogram".
    """
    from scipy.cluster.hierarchy import dendrogram, ward
    from scipy.spatial.distance import pdist

    # Calculate the pairwise distance matrix
    input_df = X.dropna().sort_values().reset_index(drop=True).values.reshape(-1, 1)
    distance_matrix = pdist(input_df)

    # Perform Ward's linkage
    linkage_matrix = ward(distance_matrix)

    # Plot the dendrogram
    fig_width = min(10, p * 0.5)
    plt.figure(figsize=(fig_width, 10))
    dendrogram(linkage_matrix, truncate_mode="lastp", p=p)
    plt.title("Ward's Dendrogram")
    plt.xlabel("Sample Index")
    plt.ylabel("Distance")
    plt.show()


def plot_cumulative_probability(
    cpore, cperm, cutoffs=None, title="Cumulative Probability Plot"
):
    """Plot cumulative probability to identify possible flow units.

    This plot helps in identifying different flow units by visualizing the
    cumulative probability distribution of the Flow Zone Indicator (FZI).

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        cutoffs (list, optional): A list of cutoff values to display as vertical lines. Defaults to None.
        title (str, optional): The title of the plot. Defaults to "Cumulative Probability Plot".
    """
    if cutoffs is None:
        cutoffs = []
    fzi = calc_fzi(cpore, cperm)
    fzi = fzi[~np.isnan(fzi)]
    sorted_fzi = sorted(fzi)
    log_fzi = np.log10(sorted_fzi)
    zi = np.cumsum(sorted_fzi) / np.sum(sorted_fzi)

    # Generate cumulative probability plot
    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.scatter(log_fzi, zi, marker=".")
    plt.xlabel("log(FZI)")
    plt.ylabel("Cumulative Probability")
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    plt.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")
    plt.xticks(
        np.arange(
            min(log_fzi[log_fzi != -np.inf]), max(log_fzi[log_fzi != np.inf]), 0.2
        )
    )
    for i, c in enumerate(cutoffs):
        plt.axvline(x=c, color="r", linestyle="dashed", label=cutoffs[i])


def plot_lorenz_heterogeneity(cpore, cperm, title="Lorenz's Plot"):
    """Plot Lorenz's plot to estimate heteroginity.

    This plot is used to assess the degree of heterogeneity in a reservoir by
    comparing the cumulative distribution of porosity with that of permeability.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        title (str, optional): The title of the plot. Defaults to "Lorenz's Plot".
    """
    try:
        from sklearn.metrics import auc
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for `plot_lorenz_heterogeneity` (auc). Install scikit-learn."
        ) from e

    sorted_perm, sorted_phit = zip(
        *sorted(zip(cperm, cpore, strict=True), reverse=True), strict=True
    )
    perm_cdf = np.cumsum(sorted_perm) / np.sum(sorted_perm)
    phit_cdf = np.cumsum(sorted_phit) / np.sum(sorted_phit)
    lorenz_coeff = (auc(phit_cdf, perm_cdf) - 0.5) / 0.5
    # Generate Lorenz's plot
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.text(
        0.4,
        0.1,
        f"Lorenz Coefficient: {lorenz_coeff:.2f}",
        fontsize=10,
        transform=plt.gca().transAxes,
    )
    plt.scatter(phit_cdf, perm_cdf, marker=".")
    plt.plot([0, 1], [0, 1], linestyle="dashed", color="gray")
    plt.xlabel("CDF of Porosity")
    plt.ylabel("CDF of Permeability")
    plt.xlim(0, 1)
    plt.ylim(0, 1)


def plot_modified_lorenz(cpore, cperm, title="Modified Lorenz's Plot"):
    """Plot Lorenz's plot to identify possible flow units.

    This is a variation of the Lorenz plot where data is sorted by the Flow Zone
    Indicator (FZI) to better delineate hydraulic flow units.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        title (str, optional): The title of the plot. Defaults to "Modified Lorenz's Plot".
    """
    fzi = calc_fzi(cpore, cperm)
    sorted_fzi, sorted_perm, sorted_pore = zip(
        *sorted(zip(fzi, cperm, cpore, strict=True), reverse=False), strict=True
    )
    perm_cdf = np.cumsum(sorted_perm) / np.sum(sorted_perm)
    pore_cdf = np.cumsum(sorted_pore) / np.sum(sorted_pore)
    # Generate Lorenz's plot
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.scatter(pore_cdf, perm_cdf, marker=".")
    plt.xlabel("CDF of Porosity")
    plt.ylabel("CDF of Permeability")
    plt.minorticks_on()
    plt.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    plt.grid(True, which="minor", linestyle=":", linewidth="0.3", color="gray")


def estimate_pore_throat(pc, ift, theta):
    """Estimate pore throat size from capillary pressure curve based on Washburn 1921.

    Args:
        pc (np.ndarray or float): Capillary pressure in psi.
        ift (float): Interfacial tension in mN/m.
        theta (float): Contact angle in degree.

    Returns:
        np.ndarray or float: Pore throat size in micrometer.
    """
    return 2 * 0.145 * ift * abs(np.cos(np.radians(theta))) / pc


def estimate_vsh_gr(gr, min_gr=None, max_gr=None, alpha=0.1):
    """Estimate volume of shale from gamma ray. If min_gr and max_gr are not provided,
    it will be automatically estimated.

    Args:
        gr (np.ndarray or float): Gamma ray from well log.
        min_gr (float, optional): Minimum gamma ray value. Defaults to None.
        max_gr (float, optional): Maximum gamma ray value. Defaults to None.
        alpha (float, optional): Alpha value for min-max normalization. Defaults to 0.1.

    Returns:
        np.ndarray or float: Volume of shale from gamma ray (VSH_GR).
    """
    from quick_pp.utils import minmax_scale, robust_scale

    # Normalize gamma ray
    if not max_gr or (not min_gr and min_gr != 0):
        # Remove high outliers and forward fill missing values
        # Use IQR for robust outlier detection
        q1, q3 = np.nanquantile(gr, [0.25, 0.75])
        iqr = q3 - q1
        upper_bound = q3 + 1.5 * iqr
        lower_bound = (q1 - 1.5 * iqr).clip(0)
        # Replace outliers with NaN and then forward fill
        gr_series = pd.Series(gr)
        gr = (
            gr_series.where((gr_series >= lower_bound) & (gr_series <= upper_bound))
            .ffill()
            .to_numpy()
        )
        gri = minmax_scale(robust_scale(gr))
    else:
        gri = (gr - min_gr) / (max_gr - min_gr)
        gri = np.where(gri < 1, gri, 1)
    return shale_volume_steiber(gri).flatten()


def estimate_vsh_dn(phin, phid, phin_sh=0.35, phid_sh=0.05):
    """Estimate volume of shale from neutron porosity and density porosity.

    Args:
        phin (np.ndarray or float): Neutron porosity in fraction.
        phid (np.ndarray or float): Density porosity in fraction.
        phin_sh (float, optional): Neutron porosity for shale. Defaults to 0.35.
        phid_sh (float, optional): Density porosity for shale. Defaults to 0.05.

    Returns:
        np.ndarray or float: Volume of shale.
    """
    return (phin - phid) / (phin_sh - phid_sh)


def estimate_rock_flag_neuden(
    nphi,
    rhob,
    vsh_cutoffs=None,
    sand_point=(-0.035, 2.65),
    shale_point=(0.45, 2.55),
    fluid_point=(1.0, 1.0),
):
    """
    Estimates rock type by calculating Vshale from a Neutron-Density crossplot
    and classifying it based on provided cutoffs.

    This method determines the shale volume for each data point by its geometric
    position between a clean sand line and a 100% shale line. The calculated
    Vshale is then used to assign a rock type.

    Args:
        nphi (pd.Series or np.ndarray): Neutron Porosity log values (v/v).
        rhob (pd.Series or np.ndarray): Bulk Density log values (g/cc).
        vsh_cut_offs (list): A list of Vshale cutoff values to define rock types.
        sand_point (tuple, optional): (NPHI, RHOB) coordinates of the clean sand matrix.
                                      Defaults to (-0.035, 2.65).
        shale_point (tuple, optional): (NPHI, RHOB) coordinates of the 100% shale point.
                                       Defaults to (0.45, 2.55).
        fluid_point (tuple, optional): (NPHI, RHOB) coordinates of the fluid point.
                                       Defaults to (1.0, 1.0).

    Returns:
        np.ndarray: An array of integer rock type flags.
    """
    from quick_pp.utils import length_a_b, line_intersection

    if vsh_cutoffs is None:
        vsh_cutoffs = [0.1, 0.25, 0.35, 0.5]
    # Calculate Vshale from the crossplot geometry
    matrix_line = (sand_point, shale_point)
    data_point_line = (fluid_point, (nphi, rhob))
    intersection_pt = line_intersection(matrix_line, data_point_line)
    dist_to_shale = length_a_b(intersection_pt, shale_point)
    total_dist = length_a_b(sand_point, shale_point)
    vsh_neuden = 1 - (dist_to_shale / total_dist)

    # Classify Vshale into rock types using the provided cutoffs
    return rock_typing(vsh_neuden, cut_offs=vsh_cutoffs, higher_is_better=False)


def rock_typing(curve, cut_offs=None, higher_is_better=True):
    """Rock typing based on cutoffs.

    Args:
        curve (np.ndarray or float): The curve to be used for rock typing.
        cut_offs (list, optional): 3 cutoffs to group the curve into 4 rock types. Defaults to [.1, .2, .3, .4].
        higher_is_better (bool, optional): Whether higher value of curve is better quality or not. Defaults to True.

    Returns:
        np.ndarray or float: An array of rock type classifications.
    """
    if cut_offs is None:
        cut_offs = [0.1, 0.2, 0.3, 0.4]
    # Set number of rock types
    rock_type = np.arange(1, len(cut_offs) + 2)
    rock_type = rock_type[::-1] if higher_is_better else rock_type
    # Rock typing based on cutoffs
    conditions = [curve < cut_off for cut_off in cut_offs] + [curve >= cut_offs[-1]]
    choices = rock_type[-len(conditions) :]
    return np.where(np.isnan(curve), np.nan, np.select(conditions, choices))


def find_cutoffs(curve, no_of_cutoffs=3):
    """Find optimal cutoffs for a given curve using KMeans clustering.

    Args:
        curve (pd.Series or np.ndarray): The curve data to find cutoffs for.
        no_of_cutoffs (int, optional): The number of cutoffs to find. This will result in `no_of_cutoffs + 1` clusters.
                                     Defaults to 3.

    Returns:
        np.ndarray: An array of cutoff values.
    """
    from sklearn.cluster import KMeans

    # Reshape data for KMeans and remove NaNs
    if not isinstance(curve, pd.Series):
        curve = pd.Series(curve)

    data = curve.dropna().values.reshape(-1, 1)
    n_clusters = no_of_cutoffs + 1
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(data)
    centers = np.sort(kmeans.cluster_centers_.flatten())
    cutoffs = (centers[:-1] + centers[1:]) / 2
    return cutoffs


def train_classification_model(
    data, input_features: list, target_feature: str, stratifier=None
):
    """Train a classification Random Forest model to predict a binary feature.

    Args:
        data (DataFrame): Dataframe containing input and target features.
        input_features (list): List of input features.
        target_feature (str): The target feature.
        stratifier (array, optional): Stratifier for train-test split. Defaults to None.

    Returns:
        RandomForestClassifier: Trained model.
    """
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import (
            ConfusionMatrixDisplay,
            classification_report,
            confusion_matrix,
        )
        from sklearn.model_selection import RandomizedSearchCV, train_test_split
    except Exception as e:
        raise ImportError(
            "scikit-learn is required to train classification models. Install scikit-learn."
        ) from e

    random_seed = 123
    X = data[input_features]
    y = data[target_feature]
    y = y.astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=stratifier
    )

    # Hyperparameter tuning
    param_dist = {
        "n_estimators": [150, 200],
        "max_depth": [30, None],
        "max_features": [0.5, "sqrt"],
        "min_samples_split": [2, 0.5],
        "min_samples_leaf": [1, 0.2],
        "criterion": ["gini", "entropy"],
    }
    model = RandomizedSearchCV(
        RandomForestClassifier(),
        param_dist,
        cv=7,
        scoring="f1_weighted",
        random_state=random_seed,
    )
    model.fit(X_train, y_train)

    # Feature importance
    best_model = model.best_estimator_
    importances = best_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    rf_importances = pd.Series(importances, index=input_features)

    fig, ax = plt.subplots()
    rf_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # Model evaluation
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    logger.info(f"Score for {target_feature} model")
    logger.info(f"Best parameters found: {model.best_params_}")
    logger.info("### Train Set ###")
    logger.info(
        f"Classification Report:\n {classification_report(y_train, y_pred_train)}"
    )
    cm = confusion_matrix(y_train, y_pred_train, labels=model.classes_)
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

    logger.info("### Test Set ###")
    logger.info(f"Classification Report:\n {classification_report(y_test, y_pred)}")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()

    return model


def train_regression_model(
    data, input_features: list, target_feature: list, stratifier=None
):
    """Train a regression Random Forest model to predict a continuous feature.

    Args:
        data (DataFrame): Dataframe containing input, target and stratifier features.
        input_features (list): List of input features.
        target_feature (str): The target feature.
        stratifier (array, optional): Stratifier for train-test split. Defaults to None.

    Returns:
        RandomForestRegressor: Trained model.
    """
    try:
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.metrics import mean_absolute_error, r2_score
        from sklearn.model_selection import RandomizedSearchCV, train_test_split
    except Exception as e:
        raise ImportError(
            "scikit-learn is required to train regression models. Install scikit-learn."
        ) from e

    random_seed = 123
    X = data[input_features]
    y = data[target_feature]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_seed, stratify=stratifier
    )

    # Hyperparameter tuning
    param_dist = {
        "n_estimators": [150, 200],
        "max_depth": [30, None],
        "max_features": [0.5, "sqrt"],
        "min_samples_split": [2, 0.5],
        "min_samples_leaf": [1, 0.2],
        "criterion": ["squared_error", "absolute_error"],
    }
    model = RandomizedSearchCV(
        RandomForestRegressor(),
        param_dist,
        cv=5,
        scoring="r2",
        random_state=random_seed,
    )
    model.fit(X_train, y_train)

    # Feature importance
    best_model = model.best_estimator_
    importances = best_model.feature_importances_
    std = np.std([tree.feature_importances_ for tree in best_model.estimators_], axis=0)
    rf_importances = pd.Series(importances, index=input_features)

    fig, ax = plt.subplots()
    rf_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    # Model evaluation
    y_pred_train = model.predict(X_train)
    y_pred = model.predict(X_test)
    logger.info(f"Score for {target_feature} model")
    logger.info(f"Best parameters found: {model.best_params_}")
    logger.info("### Train Set ###")
    logger.info(f"R2 Score: {r2_score(y_train, y_pred_train)}")
    logger.info(f"Mean Absolute Error: {mean_absolute_error(y_train, y_pred_train)}")
    logger.info("### Test Set ###")
    logger.info(f"R2 Score: {r2_score(y_test, y_pred)}")
    logger.info(f"Mean Absolute Error: {mean_absolute_error(y_test, y_pred)}")

    # Plot the true vs predicted values
    plt.figure(figsize=(10, 8))
    plt.plot(y_train, y_pred_train, ".", label="Actual", markersize=8)
    plt.plot(y_test, y_pred, ".", label="Predicted", markersize=6)
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.title(f"{target_feature} Prediction")
    plt.legend(bbox_to_anchor=(1, 1), loc="upper left")
    plt.show()

    return model


def cluster_fzi(cpore, cperm, n_clusters=None, max_clusters=10):
    """
    Determine rock types by applying k-Means clustering on the Flow Zone Indicator (FZI).

    This function calculates Log(FZI), then uses k-Means to partition the data
    into a specified number of clusters (rock types). It returns the cluster
    assignments and descriptive statistics for each cluster.

    Args:
        cpore (np.ndarray or float): Core porosity in fraction.
        cperm (np.ndarray or float): Core permeability in mD.
        n_clusters (int, optional): The number of rock types (clusters) to identify.
                                   If None, it will be auto-detected. Defaults to None.
        max_clusters (int, optional): The maximum number of clusters to consider for auto-detection. Defaults to 10.

    Returns:
        tuple: A tuple containing:
            - pd.Series: Cluster assignment for each data point.
            - pd.DataFrame: Mean and standard deviation of Log(FZI) for each cluster.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for `cluster_fzi` (KMeans, silhouette_score). Install scikit-learn."
        ) from e

    # Calculate Log(FZI) and prepare data for clustering
    fzi = calc_fzi(cpore, cperm)
    log_fzi = pd.Series(np.log10(fzi), name="log_fzi").replace(
        [-np.inf, np.inf], np.nan
    )
    data = log_fzi.dropna().values.reshape(-1, 1)

    if n_clusters is None:
        best_score = -1
        optimal_clusters = 2
        # Ensure we have enough samples for clustering
        max_k = min(max_clusters, data.shape[0] - 1)

        for k in range(2, max_k + 1):
            kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto").fit(data)
            score = silhouette_score(data, kmeans.labels_)
            if score > best_score:
                best_score = score
                optimal_clusters = k

        n_clusters = optimal_clusters
        logger.info(
            f"Optimal number of clusters identified: {n_clusters} with silhouette score: {best_score:.2f}"
        )
    elif data.shape[0] < n_clusters:
        raise ValueError(
            f"Number of samples ({data.shape[0]}) must be >= n_clusters ({n_clusters})."
        )

    # Apply k-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit(data)
    labels = kmeans.labels_

    # Map cluster labels back to the original series length
    cluster_series = pd.Series(labels, index=log_fzi.dropna().index)
    cluster_series = cluster_series.reindex(log_fzi.index)

    # Calculate statistics for each cluster
    stats_df = pd.DataFrame({"log_fzi": log_fzi.dropna(), "cluster": labels})
    cluster_stats = stats_df.groupby("cluster")["log_fzi"].agg(["mean", "std"])

    return cluster_series, cluster_stats


def cluster_rock_types_from_logs(
    df,
    features=None,
    n_clusters=None,
    max_clusters=10,
    random_state=42,
):
    """
    Clusters well log data into rock types using k-Means clustering.

    This function takes a DataFrame with well log data, scales the features, and then
    applies k-Means to identify distinct rock types. If 'WELL_NAME' is in the
    DataFrame, scaling is performed on a per-well basis.

    Args:
        df (pd.DataFrame): DataFrame containing the log data. Must include columns
                           specified in `features`, and optionally 'WELL_NAME'.
        features (list, optional): A list of feature columns to use for clustering.
                                   Defaults to ["GR", "NPHI", "RHOB"].
        n_clusters (int, optional): The number of rock types (clusters) to identify.
                                   If None, the optimal number of clusters will be
                                   determined using the Silhouette Score. Defaults to None.

    Returns:
        pd.Series: A Series containing the cluster assignment (rock type) for each data point.
                   NaN values are preserved where input logs had NaNs.
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        from sklearn.preprocessing import StandardScaler
    except Exception as e:
        raise ImportError(
            "scikit-learn is required for `cluster_rock_types_from_logs` (KMeans, StandardScaler, silhouette_score). Install scikit-learn."
        ) from e

    if features is None:
        features = ["GR", "NPHI", "RHOB"]

    # Combine logs into a DataFrame
    log_data = df.copy()[
        features + (["WELL_NAME"] if "WELL_NAME" in df.columns else [])
    ]

    # Drop rows with any NaN values for clustering
    cleaned_data = log_data.dropna(subset=features)

    if cleaned_data.empty:
        logger.warning("No complete log data available for clustering.")
        return pd.Series(np.nan, index=log_data.index)

    # Scale the features
    if "WELL_NAME" in cleaned_data.columns:
        logger.info("Scaling features by WELL_NAME.")
        # Use transform for group-wise scaling to avoid dtype issues with apply
        scaled_features = cleaned_data.groupby("WELL_NAME")[features].transform(
            # Handle std dev of zero by replacing it with 1 to avoid division by zero
            lambda x: (x - x.mean()) / (x.std() if x.std() > 0 else 1)
        )
    else:
        logger.info("Scaling features across all data.")
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(cleaned_data[features])
    scaled_data = pd.DataFrame(
        scaled_features, index=cleaned_data.index, columns=features
    )

    if n_clusters is None:
        best_score = -1
        optimal_clusters = 2
        # Ensure we have enough samples for clustering
        max_k = min(max_clusters, scaled_data.shape[0] - 1)

        if max_k < 2:
            logger.warning(
                "Not enough samples to determine optimal clusters. Defaulting to 1 cluster."
            )
            n_clusters = 1
        else:
            for k in range(2, max_k + 1):
                kmeans = KMeans(
                    n_clusters=k,
                    random_state=random_state,
                    n_init="auto",
                    init="k-means++",
                ).fit(scaled_data)
                score = silhouette_score(scaled_data, kmeans.labels_)
                if score > best_score:
                    best_score = score
                    optimal_clusters = k
            n_clusters = optimal_clusters
            logger.info(
                f"Optimal number of clusters identified: {n_clusters} with silhouette score: {best_score:.2f}"
            )

    # Apply k-Means clustering with the determined or specified number of clusters
    kmeans = KMeans(
        n_clusters=n_clusters, random_state=random_state, n_init="auto"
    ).fit(scaled_data)

    # Map cluster labels back to the original DataFrame index
    cluster_assignments = pd.Series(
        kmeans.labels_, index=cleaned_data.index, name="ROCK_TYPE_LOGS"
    )

    return cluster_assignments.reindex(log_data.index)
