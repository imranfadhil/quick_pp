import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from quick_pp import logger
from quick_pp.lithology import shale_volume_steiber
from quick_pp.porosity import calc_secondary_porosity

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


def calc_rfn(pore, perm):
    """Calculate Lucia Rock Fabric Number (RFN) from porosity and permeability, based on Jennings and Lucia (2003).

    Args:
        pore (np.ndarray or float): Interparticle/total porosity in fraction.
        perm (np.ndarray or float): Permeability in mD.

    Returns:
        np.ndarray or float: The Lucia Rock Fabric Number (RFN).
    """
    pore_eff = np.clip(pore, 0.04, 0.99)
    log_phi = np.log10(pore_eff)
    log_k = np.log10(np.clip(perm, 1e-6, None))
    log_rfn = (9.7892 + 8.6711 * log_phi - log_k) / (12.0838 + 8.2965 * log_phi)
    return np.clip(10.0**log_rfn, 0.1, 10.0)


def calc_rfn_perm(rfn, pore):
    """Calculate permeability from Lucia Rock Fabric Number (RFN) and porosity, based on Jennings and Lucia (2003).

    Args:
        rfn (np.ndarray or float): Lucia Rock Fabric Number (RFN).
        pore (np.ndarray or float): Interparticle/total porosity in fraction.

    Returns:
        np.ndarray or float: Permeability in mD.
    """
    return 10 ** (
        9.7892
        - 12.0838 * np.log10(rfn)
        + (8.6711 - 8.2965 * np.log10(rfn)) * np.log10(pore)
    )


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
        # Replace outliers with NaN and then forward/backward fill
        gr_series = pd.Series(gr)
        gr = (
            gr_series.where((gr_series >= lower_bound) & (gr_series <= upper_bound))
            .ffill()
            .bfill()
            .to_numpy()
        )
        gri = minmax_scale(robust_scale(gr)).to_numpy()
    else:
        gri = (gr - min_gr) / (max_gr - min_gr)
        gri = np.where(gri < 1, gri, 1)
    return np.asarray(shale_volume_steiber(gri)).flatten()


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


def find_cutoffs_gmm(
    curve,
    no_of_cutoffs=None,
    max_clusters=7,
    criterion="combined",
    log_scale=True,
    random_state=42,
):
    """Find optimal rock type group count and boundary cutoffs using Gaussian Mixture Models (GMM)
    with Silhouette and Bayesian Information Criterion (BIC) scoring.

    Args:
        curve (pd.Series or np.ndarray): 1D indicator array (e.g., FZI, Winland R35, or Lucia RFN).
        no_of_cutoffs (int, optional): Fixed number of cutoffs (`k = no_of_cutoffs + 1` groups).
                                       If None, automatically selects the optimal number of rock types `k`
                                       in `2..max_clusters` using `criterion`. Defaults to None.
        max_clusters (int, optional): Maximum number of rock type groups `k` to evaluate. Defaults to 7.
        criterion (str, optional): Model selection criterion: 'combined' (BIC + Silhouette), 'bic', or 'silhouette'.
                                   Defaults to 'combined'.
        log_scale (bool, optional): Fit GMM on `log10(curve)` (recommended for log-normally distributed
                                    indicators such as FZI and R35). Defaults to True.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: Dictionary containing:
            - 'cutoffs': Sorted list of boundary cutoff values in original curve space.
            - 'optimal_k': Automatically detected optimal number of rock types (clusters).
            - 'selected_k': Number of rock types used to generate `cutoffs`.
            - 'k_scores': List of dicts `[{'k', 'bic', 'aic', 'silhouette', 'combined_score'}, ...]`.
            - 'components': List of GMM component dicts `[{'mean', 'std', 'weight', 'mean_raw'}]`.
            - 'distribution': Dict with `'x_grid'`, `'hist_density'`, `'total_pdf'`, `'component_pdfs'` for plotting.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.mixture import GaussianMixture

    raw = np.asarray(curve, dtype=float).ravel()
    if log_scale:
        valid = raw[np.isfinite(raw) & (raw > 0)]
        x = np.log10(valid) if valid.size > 0 else np.array([], dtype=float)
    else:
        valid = raw[np.isfinite(raw)]
        x = valid.copy()

    if x.size < 4 or np.nanstd(x) < 1e-9:
        fallback_c = [0.5, 1.0, 2.5] if log_scale else [1.5, 2.5, 3.5]
        return {
            "cutoffs": fallback_c,
            "optimal_k": len(fallback_c) + 1,
            "selected_k": len(fallback_c) + 1,
            "k_scores": [],
            "components": [],
            "distribution": {},
        }

    X = x.reshape(-1, 1)
    max_k = max(2, min(int(max_clusters), x.size - 1, 10))
    if x.size > 1800:
        sub_idx = np.linspace(0, x.size - 1, 1800, dtype=int)
        x_sorted_idx = np.argsort(x)
        sample_idx = x_sorted_idx[sub_idx]
    else:
        sample_idx = np.arange(x.size)
    X_sub = X[sample_idx]

    models_by_k = {}
    raw_scores = []
    for k in range(2, max_k + 1):
        try:
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=random_state,
                n_init=2,
                max_iter=150,
            ).fit(X)
            bic_val = float(gmm.bic(X))
            aic_val = float(gmm.aic(X))
            labels_sub = gmm.predict(X_sub)
            if np.unique(labels_sub).size > 1:
                sil_val = float(silhouette_score(X_sub, labels_sub))
            else:
                sil_val = -1.0
            models_by_k[k] = gmm
            raw_scores.append(
                {"k": int(k), "bic": bic_val, "aic": aic_val, "silhouette": sil_val}
            )
        except Exception:
            continue

    if not raw_scores:
        fallback_k = (int(no_of_cutoffs) + 1) if no_of_cutoffs else 4
        c_arr = find_cutoffs(valid, no_of_cutoffs=max(1, fallback_k - 1))
        return {
            "cutoffs": [round(float(c), 4) for c in np.sort(c_arr)],
            "optimal_k": fallback_k,
            "selected_k": fallback_k,
            "k_scores": [],
            "components": [],
            "distribution": {},
        }

    bics = np.array([s["bic"] for s in raw_scores], dtype=float)
    sils = np.array([s["silhouette"] for s in raw_scores], dtype=float)
    bic_span = max(float(np.max(bics) - np.min(bics)), 1e-9)
    bic_norm = (np.max(bics) - bics) / bic_span
    sil_span = max(float(np.max(sils) - np.min(sils)), 1e-9)
    sil_norm = (
        (sils - np.min(sils)) / sil_span if sil_span > 1e-6 else np.ones_like(sils)
    )

    crit = str(criterion or "combined").strip().lower()
    k_scores = []
    best_k = raw_scores[0]["k"]
    best_score = -1e18

    for idx, item in enumerate(raw_scores):
        k_val = item["k"]
        b_n = float(bic_norm[idx])
        s_n = float(sil_norm[idx])
        if crit == "bic":
            comb = b_n - 0.03 * max(0, k_val - 3)
        elif crit == "silhouette":
            comb = float(item["silhouette"])
        else:
            two_cluster_penalty = (
                0.08
                if (k_val == 2 and len(raw_scores) >= 2 and bic_norm[1] > 0.35)
                else 0.0
            )
            parsimony_penalty = 0.025 * max(0, k_val - 5)
            comb = 0.52 * b_n + 0.48 * s_n - two_cluster_penalty - parsimony_penalty

        entry = {
            "k": int(k_val),
            "bic": round(float(item["bic"]), 2),
            "aic": round(float(item["aic"]), 2),
            "silhouette": round(float(item["silhouette"]), 4),
            "bic_norm": round(b_n, 4),
            "combined_score": round(float(comb), 4),
        }
        k_scores.append(entry)
        if comb > best_score:
            best_score = comb
            best_k = int(k_val)

    selected_k = (
        max(2, min(int(no_of_cutoffs) + 1, max_k))
        if no_of_cutoffs is not None
        else best_k
    )
    if selected_k not in models_by_k:
        selected_k = best_k
    gmm_sel = models_by_k[selected_k]

    means = gmm_sel.means_.ravel()
    stds = np.sqrt(np.maximum(gmm_sel.covariances_.ravel(), 1e-9))
    weights = gmm_sel.weights_.ravel()
    order = np.argsort(means)
    means_s = means[order]
    stds_s = stds[order]
    weights_s = weights[order]

    cutoffs_x = []
    for i in range(len(means_s) - 1):
        m1, s1, w1 = means_s[i], stds_s[i], weights_s[i]
        m2, s2, w2 = means_s[i + 1], stds_s[i + 1], weights_s[i + 1]
        grid = np.linspace(m1, m2, 200)
        pdf1 = (w1 / (s1 * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((grid - m1) / s1) ** 2
        )
        pdf2 = (w2 / (s2 * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((grid - m2) / s2) ** 2
        )
        diff = pdf2 - pdf1
        cross_idx = np.where(diff >= 0)[0]
        if cross_idx.size > 0 and 0 < cross_idx[0] < grid.size - 1:
            idx0 = int(cross_idx[0])
            d0, d1 = diff[idx0 - 1], diff[idx0]
            frac = -d0 / (d1 - d0) if abs(d1 - d0) > 1e-12 else 0.5
            c_x = float(grid[idx0 - 1] + frac * (grid[idx0] - grid[idx0 - 1]))
        else:
            c_x = float(0.5 * (m1 + m2))
        cutoffs_x.append(c_x)

    cutoffs_raw = [
        round(float(10.0**c if log_scale else c), 4) for c in sorted(cutoffs_x)
    ]

    x_min, x_max = float(np.percentile(x, 0.5)), float(np.percentile(x, 99.5))
    if abs(x_max - x_min) < 1e-6:
        x_min, x_max = float(np.min(x)) - 0.5, float(np.max(x)) + 0.5
    pad = 0.08 * (x_max - x_min)
    x_grid = np.linspace(x_min - pad, x_max + pad, 100)
    hist_counts, bin_edges = np.histogram(
        x, bins=36, range=(x_min - pad, x_max + pad), density=True
    )
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    comp_pdfs = []
    total_pdf = np.zeros_like(x_grid)
    components_meta = []
    for i in range(len(means_s)):
        m_i, s_i, w_i = float(means_s[i]), float(stds_s[i]), float(weights_s[i])
        pdf_i = (w_i / (s_i * np.sqrt(2.0 * np.pi))) * np.exp(
            -0.5 * ((x_grid - m_i) / s_i) ** 2
        )
        total_pdf += pdf_i
        comp_pdfs.append([round(float(v), 4) for v in pdf_i])
        components_meta.append(
            {
                "component": i + 1,
                "mean_log": round(m_i, 4),
                "std_log": round(s_i, 4),
                "weight": round(w_i, 4),
                "mean_raw": round(float(10.0**m_i if log_scale else m_i), 4),
            }
        )

    return {
        "cutoffs": cutoffs_raw,
        "cutoffs_log": [round(float(c), 4) for c in sorted(cutoffs_x)],
        "optimal_k": int(best_k),
        "selected_k": int(selected_k),
        "criterion": crit,
        "log_scale": bool(log_scale),
        "k_scores": k_scores,
        "components": components_meta,
        "distribution": {
            "bin_centers": [round(float(v), 4) for v in bin_centers],
            "hist_density": [round(float(v), 4) for v in hist_counts],
            "x_grid": [round(float(v), 4) for v in x_grid],
            "total_pdf": [round(float(v), 4) for v in total_pdf],
            "component_pdfs": comp_pdfs,
        },
    }


def find_cutoffs_lorenz(
    phit,
    perm,
    indicator=None,
    method="fzi",
    no_of_cutoffs=None,
    max_clusters=7,
):
    """Find optimal flow unit count and boundary cutoffs using Modified Lorenz (Flow vs. Storage Capacity)
    and Cumulative Probability inflection points (slope breaks).

    Uses exact dynamic-programming piecewise-linear segmentation on the resampled Modified Lorenz
    and log-indicator Cumulative Probability curves to locate flow unit inflection knees, scored
    with piecewise-linear BIC and 1D Silhouette score.

    Args:
        phit (np.ndarray or pd.Series): Total/interparticle porosity in fraction (v/v).
        perm (np.ndarray or pd.Series): Permeability in mD.
        indicator (np.ndarray, optional): Precomputed DRT indicator array (FZI, R35, or RFN).
                                          If None, computed from `phit` and `perm` using `method`.
        method (str, optional): DRT method ('fzi', 'winland', or 'lucia'). Defaults to 'fzi'.
        no_of_cutoffs (int, optional): Fixed number of cutoffs (`k = no_of_cutoffs + 1` flow units).
                                       If None, automatically selects optimal `k` in `2..max_clusters`.
        max_clusters (int, optional): Maximum number of flow units `k` to evaluate. Defaults to 7.

    Returns:
        dict: Dictionary containing:
            - 'cutoffs': Sorted list of boundary cutoff values.
            - 'optimal_k': Automatically detected optimal number of rock types (flow units).
            - 'selected_k': Number of rock types used to generate `cutoffs`.
            - 'lorenz_coeff': Lorenz heterogeneity coefficient in [0, 1].
            - 'lorenz_curve': Dict with `'storage_cdf'`, `'flow_cdf'`, `'cum_prob'`, `'log_indicator'`, `'indicator'`.
            - 'inflection_points': List of inflection point dicts on the Lorenz & Cumulative Probability curves.
            - 'k_scores': List of dicts `[{'k', 'bic', 'silhouette', 'r2', 'combined_score'}, ...]`.
    """
    from sklearn.metrics import silhouette_score

    phit_arr = np.asarray(phit, dtype=float).ravel()
    perm_arr = np.asarray(perm, dtype=float).ravel()
    m_norm = str(method or "fzi").strip().lower()
    higher_is_better = m_norm != "lucia"

    if indicator is None:
        if m_norm in ("winland", "r35"):
            ind_arr = np.asarray(calc_r35(phit_arr, perm_arr), dtype=float)
        elif m_norm in ("lucia", "rfn"):
            ind_arr = np.asarray(calc_rfn(phit_arr, perm_arr), dtype=float)
        else:
            ind_arr = np.asarray(calc_fzi(phit_arr, perm_arr), dtype=float)
    else:
        ind_arr = np.asarray(indicator, dtype=float).ravel()

    valid_mask = (
        np.isfinite(phit_arr)
        & np.isfinite(perm_arr)
        & np.isfinite(ind_arr)
        & (phit_arr > 0)
        & (perm_arr > 0)
        & (ind_arr > 0)
    )
    if valid_mask.sum() < 6:
        fallback_c = [1.5, 2.5, 3.5] if not higher_is_better else [0.5, 1.0, 2.5]
        return {
            "cutoffs": fallback_c,
            "optimal_k": len(fallback_c) + 1,
            "selected_k": len(fallback_c) + 1,
            "lorenz_coeff": 0.0,
            "lorenz_curve": {},
            "inflection_points": [],
            "k_scores": [],
        }

    phi_v = phit_arr[valid_mask]
    k_v = perm_arr[valid_mask]
    ind_v = ind_arr[valid_mask]

    order_best = np.argsort(-ind_v) if higher_is_better else np.argsort(ind_v)
    phi_ord = phi_v[order_best]
    k_ord = k_v[order_best]
    ind_ord = ind_v[order_best]
    log_ind_ord = np.log10(ind_ord)

    storage_cdf = np.concatenate(([0.0], np.cumsum(phi_ord) / np.sum(phi_ord)))
    flow_cdf = np.concatenate(([0.0], np.cumsum(k_ord) / np.sum(k_ord)))
    trapezoid_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auc_val = float(trapezoid_fn(flow_cdf, storage_cdf)) if trapezoid_fn else 0.5
    lorenz_coeff = float(np.clip((auc_val - 0.5) / 0.5, 0.0, 1.0))

    M = min(100, len(ind_ord))
    q_idx = np.linspace(0, len(ind_ord) - 1, M, dtype=int)
    c_grid = storage_cdf[1:][q_idx]
    f_grid = flow_cdf[1:][q_idx]
    ind_grid = ind_ord[q_idx]
    log_grid = log_ind_ord[q_idx]
    prob_grid = (q_idx + 0.5) / len(ind_ord)

    log_min, log_max = float(np.min(log_grid)), float(np.max(log_grid))
    log_span = max(log_max - log_min, 1e-9)
    log_norm = (log_grid - log_min) / log_span

    cost = np.full((M, M), np.inf, dtype=float)
    min_seg_len = max(2, M // 16)
    for a in range(M):
        cost[a, a] = 0.0
        for b in range(a + min_seg_len, M):
            dc = c_grid[b] - c_grid[a]
            if abs(dc) > 1e-9:
                t_c = (c_grid[a : b + 1] - c_grid[a]) / dc
            else:
                t_c = np.linspace(0.0, 1.0, b - a + 1)
            f_hat = f_grid[a] + t_c * (f_grid[b] - f_grid[a])
            err_lorenz = float(np.sum((f_grid[a : b + 1] - f_hat) ** 2))

            dp = prob_grid[b] - prob_grid[a]
            t_p = (prob_grid[a : b + 1] - prob_grid[a]) / dp if abs(dp) > 1e-9 else t_c
            y_hat = log_norm[a] + t_p * (log_norm[b] - log_norm[a])
            err_prob = float(np.sum((log_norm[a : b + 1] - y_hat) ** 2))

            cost[a, b] = 0.65 * err_lorenz + 0.35 * err_prob

    total_ss = max(float(cost[0, M - 1]) if np.isfinite(cost[0, M - 1]) else 1.0, 1e-9)
    max_k = max(2, min(int(max_clusters), M // min_seg_len, 8))

    dp_val = np.full((max_k + 1, M), np.inf, dtype=float)
    dp_prev = np.zeros((max_k + 1, M), dtype=int)
    for j in range(min_seg_len, M):
        dp_val[1, j] = cost[0, j]

    for k in range(2, max_k + 1):
        for j in range(k * min_seg_len, M):
            prev_end = np.arange((k - 1) * min_seg_len, j - min_seg_len + 1)
            if prev_end.size == 0:
                continue
            cand = dp_val[k - 1, prev_end] + cost[prev_end, j]
            best_pos = int(np.argmin(cand))
            if np.isfinite(cand[best_pos]):
                dp_val[k, j] = float(cand[best_pos])
                dp_prev[k, j] = int(prev_end[best_pos])

    breakpoints_by_k = {}
    raw_scores = []
    X_log = log_grid.reshape(-1, 1)

    for k in range(2, max_k + 1):
        rss = float(dp_val[k, M - 1])
        if not np.isfinite(rss):
            continue
        bps = []
        curr = M - 1
        for seg in range(k, 1, -1):
            prev_idx = int(dp_prev[seg, curr])
            bps.append(prev_idx)
            curr = prev_idx
        bps = sorted(bps)
        breakpoints_by_k[k] = bps

        labels_grid = np.zeros(M, dtype=int)
        for seg_idx, bp in enumerate(bps):
            labels_grid[bp:] = seg_idx + 1
        sil_val = (
            float(silhouette_score(X_log, labels_grid))
            if np.unique(labels_grid).size > 1
            else -1.0
        )
        r2_val = float(np.clip(1.0 - (rss / total_ss), 0.0, 1.0))
        bic_val = float(M * np.log(max(rss / M, 1e-8)) + 2.5 * k * np.log(M))
        raw_scores.append(
            {
                "k": int(k),
                "rss": rss,
                "r2": r2_val,
                "bic": bic_val,
                "silhouette": sil_val,
            }
        )

    if not raw_scores:
        fallback_k = (int(no_of_cutoffs) + 1) if no_of_cutoffs else 4
        c_arr = find_cutoffs(ind_v, no_of_cutoffs=max(1, fallback_k - 1))
        return {
            "cutoffs": [round(float(c), 4) for c in np.sort(c_arr)],
            "optimal_k": fallback_k,
            "selected_k": fallback_k,
            "lorenz_coeff": round(lorenz_coeff, 4),
            "lorenz_curve": {},
            "inflection_points": [],
            "k_scores": [],
        }

    rss_arr = np.array([s["rss"] for s in raw_scores], dtype=float)
    k_arr = np.array([s["k"] for s in raw_scores], dtype=float)
    rss_min = float(np.min(rss_arr))
    rss_norm = (rss_arr - rss_min) / max(total_ss - rss_min, 1e-9)
    chord_y = 1.0 - (k_arr - 1.0) / max(float(np.max(k_arr)) - 1.0, 1.0)
    elbow_raw = np.maximum(0.0, chord_y - rss_norm)
    bic_norm = elbow_raw / max(float(np.max(elbow_raw)), 1e-9)
    sils = np.array([s["silhouette"] for s in raw_scores], dtype=float)
    sil_span = max(float(np.max(sils) - np.min(sils)), 1e-9)
    sil_norm = (
        (sils - np.min(sils)) / sil_span if sil_span > 1e-6 else np.ones_like(sils)
    )

    k_scores = []
    best_k = raw_scores[0]["k"]
    best_score = -1e18
    for idx, item in enumerate(raw_scores):
        k_val = item["k"]
        b_n = float(bic_norm[idx])
        s_n = float(sil_norm[idx])
        two_penalty = (
            0.08 if (k_val == 2 and item["r2"] < 0.94 and len(raw_scores) > 1) else 0.0
        )
        parsimony_penalty = 0.03 * max(0, k_val - 5)
        comb = 0.55 * b_n + 0.45 * s_n - two_penalty - parsimony_penalty
        k_scores.append(
            {
                "k": int(k_val),
                "bic": round(float(item["bic"]), 2),
                "silhouette": round(float(item["silhouette"]), 4),
                "r2": round(float(item["r2"]), 4),
                "bic_norm": round(b_n, 4),
                "combined_score": round(float(comb), 4),
            }
        )
        if comb > best_score:
            best_score = comb
            best_k = int(k_val)

    selected_k = (
        max(2, min(int(no_of_cutoffs) + 1, max_k))
        if no_of_cutoffs is not None
        else best_k
    )
    if selected_k not in breakpoints_by_k:
        selected_k = best_k
    bps_sel = breakpoints_by_k[selected_k]

    inflection_points = []
    raw_cutoffs = []
    for i, bp in enumerate(bps_sel):
        c_val = float(0.5 * (ind_grid[max(0, bp - 1)] + ind_grid[bp]))
        raw_cutoffs.append(round(c_val, 4))
        inflection_points.append(
            {
                "rank": i + 1,
                "storage_cdf": round(float(c_grid[bp]), 4),
                "flow_cdf": round(float(f_grid[bp]), 4),
                "cum_prob": round(float(prob_grid[bp]), 4),
                "log_indicator": round(float(log_grid[bp]), 4),
                "cutoff": round(c_val, 4),
                "drt_before": i + 1,
                "drt_after": i + 2,
            }
        )

    return {
        "cutoffs": sorted(raw_cutoffs),
        "optimal_k": int(best_k),
        "selected_k": int(selected_k),
        "lorenz_coeff": round(lorenz_coeff, 4),
        "lorenz_curve": {
            "storage_cdf": [0.0] + [round(float(v), 4) for v in c_grid],
            "flow_cdf": [0.0] + [round(float(v), 4) for v in f_grid],
            "cum_prob": [round(float(v), 4) for v in prob_grid],
            "log_indicator": [round(float(v), 4) for v in log_grid],
            "indicator": [round(float(v), 4) for v in ind_grid],
        },
        "inflection_points": inflection_points,
        "k_scores": k_scores,
    }


def apply_lucia_rfn_override(
    phit,
    perm,
    phis=None,
    phi_sec=None,
    primary_rock_flags=None,
    lucia_cutoffs=None,
    phi_sec_cutoff=0.02,
    use_interparticle_porosity=True,
):
    """Compute interparticle-corrected Lucia Rock Fabric Number (RFN) and optionally override
    primary rock type flags in vuggy carbonate intervals where `phi_sec = phi_T - phi_sonic >= phi_sec_cutoff`.

    Args:
        phit (np.ndarray or pd.Series): Total porosity in fraction (v/v).
        perm (np.ndarray or pd.Series): Permeability in mD.
        phis (np.ndarray or pd.Series, optional): Sonic (matrix) porosity in fraction (v/v).
        phi_sec (np.ndarray or pd.Series, optional): Precomputed secondary porosity `phi_T - phi_sonic`.
        primary_rock_flags (np.ndarray, optional): Primary ROCK_FLAG array to override in vuggy intervals.
        lucia_cutoffs (list, optional): Lucia RFN boundary cutoffs. Defaults to `[1.5, 2.5, 3.5]`.
        phi_sec_cutoff (float, optional): Secondary porosity threshold (v/v) to flag vuggy carbonate
                                          and apply Lucia RFN override. Defaults to 0.02 (2 p.u.).
        use_interparticle_porosity (bool, optional): Use interparticle porosity `phi_ip = phi_T - phi_sec`
                                                     when computing Lucia RFN. Defaults to True.

    Returns:
        dict: Dictionary containing:
            - 'rfn': Interparticle-corrected (or total) Lucia RFN array.
            - 'rfn_total': Total-porosity Lucia RFN array.
            - 'phi_sec': Secondary porosity `max(0, phi_T - phi_sonic)` array.
            - 'phi_ip': Interparticle porosity `phi_T - phi_sec` array.
            - 'vuggy_mask': Boolean mask where `phi_sec >= phi_sec_cutoff`.
            - 'lucia_rock_flags': Rock flags classified purely by Lucia RFN (`higher_is_better=False`).
            - 'overridden_rock_flags': `primary_rock_flags` overridden by `lucia_rock_flags` where `vuggy_mask` is True.
            - 'override_count': Number of samples overridden.
            - 'vuggy_count': Number of samples with `phi_sec >= phi_sec_cutoff`.
    """
    if lucia_cutoffs is None:
        lucia_cutoffs = [1.5, 2.5, 3.5]
    sorted_lucia_cutoffs = sorted(float(c) for c in lucia_cutoffs)

    phit_arr = np.asarray(phit, dtype=float)
    perm_arr = np.asarray(perm, dtype=float)

    if phi_sec is not None:
        phi_sec_arr = np.asarray(phi_sec, dtype=float)
        phi_ip_arr = np.where(
            np.isfinite(phit_arr) & np.isfinite(phi_sec_arr),
            np.clip(phit_arr - np.maximum(0.0, phi_sec_arr), 0.001, phit_arr),
            np.clip(phit_arr, 0.001, 0.99),
        )
    elif phis is not None:
        phi_sec_arr, phi_ip_arr = calc_secondary_porosity(phit_arr, phis)
    else:
        phi_sec_arr = np.full_like(phit_arr, np.nan, dtype=float)
        phi_ip_arr = np.clip(phit_arr, 0.001, 0.99)

    valid_mask = (phit_arr > 0) & (phit_arr < 1.0) & (perm_arr > 0)
    rfn_total_arr = np.full_like(phit_arr, np.nan, dtype=float)
    rfn_ip_arr = np.full_like(phit_arr, np.nan, dtype=float)

    if valid_mask.any():
        rfn_total_arr[valid_mask] = calc_rfn(phit_arr[valid_mask], perm_arr[valid_mask])
        pore_for_rfn = (
            phi_ip_arr[valid_mask]
            if use_interparticle_porosity
            else phit_arr[valid_mask]
        )
        rfn_ip_arr[valid_mask] = calc_rfn(pore_for_rfn, perm_arr[valid_mask])

    lucia_rf = rock_typing(
        rfn_ip_arr, cut_offs=sorted_lucia_cutoffs, higher_is_better=False
    )
    vuggy_mask = (
        valid_mask & np.isfinite(phi_sec_arr) & (phi_sec_arr >= float(phi_sec_cutoff))
    )

    if primary_rock_flags is not None:
        prim_arr = np.asarray(primary_rock_flags, dtype=float)
        overridden_rf = np.where(vuggy_mask & np.isfinite(lucia_rf), lucia_rf, prim_arr)
        override_count = int(np.sum(vuggy_mask & np.isfinite(lucia_rf)))
    else:
        overridden_rf = lucia_rf
        override_count = int(np.sum(vuggy_mask & np.isfinite(lucia_rf)))

    return {
        "rfn": rfn_ip_arr,
        "rfn_total": rfn_total_arr,
        "phi_sec": phi_sec_arr,
        "phi_ip": phi_ip_arr,
        "vuggy_mask": vuggy_mask,
        "lucia_rock_flags": lucia_rf,
        "overridden_rock_flags": overridden_rf,
        "override_count": override_count,
        "vuggy_count": int(np.sum(vuggy_mask)),
    }


def train_and_validate_rock_type_classifier(
    df: pd.DataFrame,
    include_pef: bool = True,
    include_vdl: bool = True,
    model_type: str = "random_forest",
    test_size: float = 0.25,
    dt_matrix: float = 47.6,
    dt_fluid: float = 189.0,
    random_state: int = 42,
) -> dict:
    """Train a supervised Rock Type (DRT) classification model on well logs
    (GR, NPHI, RHOB, log10(RT), optional PEF, and optional VDL = Velocity Deviation Log),
    evaluate it with a Contingency Table (Confusion Matrix) validator, and predict/propagate
    rock types across all log depth rows.

    Args:
        df (pd.DataFrame): DataFrame containing log curves and target 'ROCK_FLAG' (binned by DRT cutoffs).
        include_pef (bool, optional): Include Photoelectric Factor (PEF) if present. Defaults to True.
        include_vdl (bool, optional): Compute and include Velocity Deviation Log (VDL). Defaults to True.
        model_type (str, optional): Classifier type: 'random_forest', 'extra_trees', or 'gradient_boosting'.
                                    Defaults to 'random_forest'.
        test_size (float, optional): Validation holdout fraction in (0.1, 0.5). Defaults to 0.25.
        dt_matrix (float, optional): Matrix transit time [us/ft] for VDL calculation. Defaults to 47.6.
        dt_fluid (float, optional): Fluid transit time [us/ft] for VDL calculation. Defaults to 189.0.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        dict: Dictionary containing:
            - 'features_used': List of feature names used in training.
            - 'labels': Sorted list of integer DRT class labels.
            - 'contingency_table': Dict with 'counts' (2D list), 'norm_recall' (2D list), 'labels' (list).
            - 'metrics': Dict with 'accuracy', 'balanced_accuracy', 'f1_weighted', 'f1_macro', 'cohen_kappa',
                         'train_samples', 'val_samples', 'total_predicted'.
            - 'class_metrics': List of per-DRT dicts `[{'drt', 'precision', 'recall', 'f1', 'support'}]`.
            - 'feature_importances': List of `[{'feature', 'importance'}]` sorted descending.
            - 'predicted_rock_flags': List of ML-predicted integer rock flags (or None) for every row of `df`.
            - 'predicted_confidence': List of max class probabilities for every row of `df`.
            - 'vdl': List of computed VDL values [m/s] (or None) for every row of `df`.
    """
    from sklearn.ensemble import (
        ExtraTreesClassifier,
        GradientBoostingClassifier,
        RandomForestClassifier,
    )
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )
    from sklearn.model_selection import train_test_split

    from quick_pp.machine_learning.feature_engineering import velocity_deviation_log

    work = df.copy()
    n_rows = len(work)
    if n_rows == 0:
        raise ValueError("Empty DataFrame provided for rock type classification.")

    def _first_valid_series(candidates):
        for col in candidates:
            if col in work.columns:
                s = pd.to_numeric(work[col], errors="coerce").replace(
                    [np.inf, -np.inf], np.nan
                )
                if s.notna().any():
                    return s
        return pd.Series(np.nan, index=work.index, dtype=float)

    gr_s = _first_valid_series(("GR", "SGR", "CGR", "GAMMA"))
    nphi_s = _first_valid_series(("NPHI_HC", "NPHI", "TNPH", "PHIN"))
    rhob_s = _first_valid_series(("RHOB_HC", "RHOB", "RHOZ", "DEN"))
    rt_s = _first_valid_series(("RT", "RD", "RDEP", "RDEEP", "LLD", "RILD", "RESD"))
    pef_s = _first_valid_series(("PEF", "PE", "PEFZ", "PEFA"))
    dtc_s = _first_valid_series(("DTC", "DT", "AC", "DTCO", "DT4P"))
    phit_s = _first_valid_series(("PHIT", "PHIE", "CPORE", "POROSITY"))

    log_rt_s = pd.Series(np.nan, index=work.index, dtype=float)
    valid_rt = rt_s.notna() & (rt_s > 0)
    if valid_rt.any():
        log_rt_s.loc[valid_rt] = np.log10(rt_s.loc[valid_rt].clip(1e-3, 1e5))

    vdl_s = pd.Series(np.nan, index=work.index, dtype=float)
    if rhob_s.notna().any() or nphi_s.notna().any() or phit_s.notna().any():
        try:
            vdl_calc = velocity_deviation_log(
                rhob=rhob_s if rhob_s.notna().any() else None,
                nphi=nphi_s if nphi_s.notna().any() else None,
                dtc=dtc_s if dtc_s.notna().any() else None,
                phit=phit_s if phit_s.notna().any() else None,
                dt_matrix=dt_matrix,
                dt_fluid=dt_fluid,
            )
            vdl_s = pd.Series(vdl_calc, index=work.index, dtype=float).replace(
                [np.inf, -np.inf], np.nan
            )
        except Exception:
            pass

    feat_df = pd.DataFrame(index=work.index)
    base_candidates = [
        ("GR", gr_s),
        ("NPHI", nphi_s),
        ("RHOB", rhob_s),
        ("LOG_RT", log_rt_s),
    ]
    for fname, fseries in base_candidates:
        if fseries.notna().any():
            feat_df[fname] = fseries

    if include_pef and pef_s.notna().sum() >= max(5, int(0.05 * n_rows)):
        feat_df["PEF"] = pef_s

    if include_vdl and vdl_s.notna().sum() >= 5:
        feat_df["VDL"] = vdl_s

    features_used = list(feat_df.columns)
    if len(features_used) < 2:
        raise ValueError(
            "Insufficient log features found for classification. At least two of GR, NPHI, RHOB, RT are required."
        )

    # Require at least half of the selected features to be non-NaN on a row to predict/train, then median-impute minor gaps
    min_valid_feats = max(2, len(features_used) - 1)
    row_valid_mask = feat_df.notna().sum(axis=1) >= min_valid_feats
    medians = feat_df.median(numeric_only=True)
    feat_imputed = feat_df.fillna(medians).fillna(0.0)

    target_s = pd.to_numeric(work.get("ROCK_FLAG"), errors="coerce")
    train_mask = row_valid_mask & target_s.notna() & (target_s >= 1)
    if train_mask.sum() < 8:
        raise ValueError(
            f"At least 8 cutoff-binned ROCK_FLAG samples with valid logs are required to train classifier (found {int(train_mask.sum())})."
        )

    X_all_labeled = feat_imputed.loc[train_mask, features_used].to_numpy(dtype=float)
    y_all_labeled = target_s.loc[train_mask].to_numpy(dtype=int)
    labels = sorted(int(v) for v in np.unique(y_all_labeled))
    if len(labels) < 2:
        raise ValueError(
            "At least 2 distinct Rock Types (DRTs) are required to train a classification model."
        )

    val_frac = float(np.clip(test_size, 0.10, 0.50))
    _, class_counts = np.unique(y_all_labeled, return_counts=True)
    can_stratify = bool(
        np.min(class_counts) >= 2 and int(len(y_all_labeled) * val_frac) >= len(labels)
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_all_labeled,
        y_all_labeled,
        test_size=val_frac,
        random_state=random_state,
        stratify=y_all_labeled if can_stratify else None,
    )

    m_type = str(model_type or "random_forest").strip().lower()

    def _build_clf():
        if m_type == "extra_trees":
            return ExtraTreesClassifier(
                n_estimators=160,
                max_depth=16,
                class_weight="balanced",
                random_state=random_state,
                n_jobs=-1,
            )
        if m_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=120,
                max_depth=4,
                learning_rate=0.08,
                random_state=random_state,
            )
        return RandomForestClassifier(
            n_estimators=160,
            max_depth=16,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    clf_val = _build_clf()
    clf_val.fit(X_train, y_train)
    y_val_pred = clf_val.predict(X_val)

    cm_counts = confusion_matrix(y_val, y_val_pred, labels=labels)
    row_sums = np.maximum(cm_counts.sum(axis=1, keepdims=True), 1)
    cm_norm = cm_counts.astype(float) / row_sums

    acc = float(accuracy_score(y_val, y_val_pred))
    bal_acc = float(balanced_accuracy_score(y_val, y_val_pred))
    f1_w = float(f1_score(y_val, y_val_pred, average="weighted", zero_division=0))
    f1_m = float(f1_score(y_val, y_val_pred, average="macro", zero_division=0))
    kappa = (
        float(cohen_kappa_score(y_val, y_val_pred))
        if np.unique(y_val).size > 1
        else 0.0
    )

    prec_arr, rec_arr, f1_arr, sup_arr = precision_recall_fscore_support(
        y_val, y_val_pred, labels=labels, zero_division=0
    )
    class_metrics = [
        {
            "drt": int(lbl),
            "precision": round(float(prec_arr[i]), 4),
            "recall": round(float(rec_arr[i]), 4),
            "f1": round(float(f1_arr[i]), 4),
            "support": int(sup_arr[i]),
        }
        for i, lbl in enumerate(labels)
    ]

    # Refit on all labeled data to propagate predictions across the entire multi-well log dataset
    clf_full = _build_clf()
    clf_full.fit(X_all_labeled, y_all_labeled)

    importances_raw = getattr(
        clf_full,
        "feature_importances_",
        np.ones(len(features_used)) / len(features_used),
    )
    feature_importances = sorted(
        [
            {"feature": fname, "importance": round(float(imp), 4)}
            for fname, imp in zip(features_used, importances_raw)
        ],
        key=lambda d: d["importance"],
        reverse=True,
    )

    pred_flags: list = [None] * n_rows
    pred_conf: list = [None] * n_rows
    valid_indices = np.where(row_valid_mask.to_numpy(dtype=bool))[0]
    if valid_indices.size > 0:
        X_pred = feat_imputed.iloc[valid_indices][features_used].to_numpy(dtype=float)
        y_pred_all = clf_full.predict(X_pred)
        proba_all = clf_full.predict_proba(X_pred)
        max_proba = np.max(proba_all, axis=1)
        for pos, row_idx in enumerate(valid_indices):
            pred_flags[int(row_idx)] = int(y_pred_all[pos])
            pred_conf[int(row_idx)] = round(float(max_proba[pos]), 4)

    vdl_list = [
        round(float(v), 2) if pd.notna(v) and np.isfinite(v) else None for v in vdl_s
    ]

    return {
        "model_type": m_type,
        "features_used": features_used,
        "labels": labels,
        "contingency_table": {
            "labels": labels,
            "counts": [[int(c) for c in row] for row in cm_counts],
            "norm_recall": [[round(float(v), 4) for v in row] for row in cm_norm],
        },
        "metrics": {
            "accuracy": round(acc, 4),
            "balanced_accuracy": round(bal_acc, 4),
            "f1_weighted": round(f1_w, 4),
            "f1_macro": round(f1_m, 4),
            "cohen_kappa": round(kappa, 4),
            "train_samples": int(len(y_train)),
            "val_samples": int(len(y_val)),
            "labeled_samples": int(len(y_all_labeled)),
            "total_predicted": int(valid_indices.size),
        },
        "class_metrics": class_metrics,
        "feature_importances": feature_importances,
        "predicted_rock_flags": pred_flags,
        "predicted_confidence": pred_conf,
        "vdl": vdl_list,
    }
