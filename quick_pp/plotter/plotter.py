import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from quick_pp.utils import line_intersection
from quick_pp import logger
import quick_pp.plotter.well_log as plotter_wells
import plotly.graph_objects as go

plotly_log = plotter_wells.plotly_log

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update(
    {"axes.labelsize": 10, "xtick.labelsize": 10, "legend.fontsize": "small"}
)


def update_fluid_contacts(well_data, well_config: dict):
    """Update fluid flags based on fluid contacts.

    Args:
        well_data (pd.DataFrame): DataFrame containing well log data, including a 'TVD' column.
        well_config (dict): A dictionary defining the fluid contacts for a specific zone (e.g., GOC, OWC).

    Returns:
        pd.DataFrame: The input DataFrame with added 'OIL_FLAG', 'GAS_FLAG', 'WATER_FLAG', and 'FLUID_FLAG' columns.
    """
    owc = well_config.get("OWC", np.nan)
    odt = well_config.get("ODT", np.nan)
    out = well_config.get("OUT", np.nan)
    goc = well_config.get("GOC", np.nan)
    gdt = well_config.get("GDT", np.nan)
    gut = well_config.get("GUT", np.nan)
    gwc = well_config.get("GWC", np.nan)
    wut = well_config.get("WUT", np.nan)

    well_data = well_data.copy()
    logger.debug(f"well_data columns before update: {well_data.columns.tolist()}")
    well_data["OIL_FLAG"] = np.where(
        ((well_data["TVD"] > out) | (well_data["TVD"] > goc))
        & ((well_data["TVD"] < odt) | (well_data["TVD"] < owc)),
        1,
        0,
    )

    well_data["GAS_FLAG"] = np.where(
        ((well_data["TVD"] < gdt) | (well_data["TVD"] < gwc) | (well_data["TVD"] < goc))
        & (well_data["TVD"] > gut),
        1,
        0,
    )

    well_data["WATER_FLAG"] = np.where(
        (
            (well_data["TVD"] > wut)
            | (well_data["TVD"] > owc)
            | (well_data["TVD"] > gwc)
        ),
        1,
        0,
    )

    well_data["FLUID_FLAG"] = np.where(
        well_data["OIL_FLAG"] == 1,
        1,
        np.where(
            well_data["GAS_FLAG"] == 1, 2, np.where(well_data["WATER_FLAG"] == 1, 0, -1)
        ),
    )

    return well_data


def generate_zone_config(zones: list = ["ALL"]):
    """Generate zone configuration.

    Args:
        zones (list, optional): List of zone names. Defaults to ['ALL'].

    Returns:
        dict: Zone configuration
    """
    logger.info(f"Generating zone config for zones: {zones}")
    zone_config = {}
    for zone in zones:
        zone_config[zone] = {
            "GUT": 0,
            "GDT": 0,
            "GOC": 0,
            "GWC": 0,
            "OUT": 0,
            "ODT": 0,
            "OWC": 0,
            "WUT": 0,
        }
    return zone_config


def update_zone_config(zone_config: dict, zone: str, fluid_contacts: dict):
    """Update zone configuration with fluid contacts.

    Args:
        zone_config (dict): Dictionary containing zone configuration.
        zone (str): Zone name for which configuration is to be updated.
        fluid_contacts (dict): Fluid contacts for the specified zone.

    Returns:
        dict: Updated zone configuration
    """
    logger.info(
        f"Updating zone config for zone: {zone} with contacts: {fluid_contacts}"
    )
    if zone in zone_config:
        zone_config[zone].update(fluid_contacts)
    else:
        zone_config[zone] = fluid_contacts
    return zone_config


def generate_well_config(well_names: list = ["X"]):
    """Generate well configuration.

    Args:
        well_names (list, optional): List of well names. Defaults to ['X'].

    Returns:
        dict: Well configuration
    """
    logger.info(f"Generating well config for wells: {well_names}")
    well_config = {}
    for i, well in enumerate(well_names):
        well_config[well] = {
            "sorting": i + 1,
            "zones": {
                "ALL": {
                    "GUT": 0,
                    "GDT": 0,
                    "GOC": 0,
                    "GWC": 0,
                    "OUT": 0,
                    "ODT": 0,
                    "OWC": 0,
                    "WUT": 0,
                },
            },
        }
    return well_config


def update_well_config(
    well_config: dict,
    well_name: str,
    zone: str = "",
    fluid_contacts: dict = {},
    sorting: int = 0,
):
    """Update well configuration with fluid contacts.

    Args:
        well_config (dict): Dictionary containing well sorting and fluid contacts.
        well_name (str): Well name for which configuration is to be updated.
        zone (str): Zone name for which configuration is to be updated.
        fluid_contacts (dict): Fluid contacts for the specified zone.
        sorting (int, optional): Sorting of the wells on the stick plot. Defaults to None.

    Returns:
        dict: Updated well configuration
    """
    logger.info(
        f"Updating well config for well: {well_name}, zone: {zone}, sorting: {sorting}"
    )
    if zone in well_config[well_name]["zones"]:
        well_config[well_name]["zones"][zone].update(fluid_contacts)
    elif zone:
        well_config[well_name]["zones"][zone] = fluid_contacts

    if sorting:
        well_config[well_name]["sorting"] = sorting
    return well_config


def assert_well_config_structure(well_config):
    """Assert well configuration structure.

    Args:
        well_config (dict): Dictionary containing well sorting and fluid contacts.
    """
    logger.debug("Asserting well config structure.")
    required_keys = {"sorting", "zones"}
    optional_keys = {"GUT", "GDT", "GOC", "GWC", "OUT", "ODT", "OWC", "WUT"}
    for well, config in well_config.items():
        assert isinstance(config, dict), f"Value for well '{well}' is not a dictionary"
        assert set(config.keys()).intersection(required_keys), (
            f"Well '{well}' does not have the required keys"
        )

        for zone, fluid_contacts in config["zones"].items():
            assert isinstance(fluid_contacts, dict), (
                f"Value for zone '{zone}' in well '{well}' is not a dictionary"
            )
            assert set(fluid_contacts.keys()).intersection(optional_keys), (
                f"zone '{zone}' in well '{well}' has invalid keys"
            )


def stick_plot(data, well_config: dict, zone: str = "ALL"):
    """Generate a multi-well stick plot to visualize fluid distribution.

    This function creates a series of vertical tracks, one for each well,
    displaying the Bulk Volume of Oil (BVO) against True Vertical Depth (TVD).
    It shades the background of each track based on the fluid flags (oil, gas, water)
    derived from the fluid contacts specified in the `well_config`.

    Args:
        data (pd.DataFrame): DataFrame with log data for one or more wells.
                               Must include 'WELL_NAME', 'TVD', 'PHIT', and 'SWT'.
        well_config (dict): A configuration dictionary where keys are well names.
                            Each well's value is a dictionary containing a 'sorting'
                            integer and a 'zones' dictionary. The 'zones' dictionary
                            maps zone names to their respective fluid contacts.
        zone (str, optional): The specific zone to plot. Defaults to 'ALL'.
    """
    logger.info(f"Generating stick plot for zone: {zone}")
    assert "SWT" in data.columns, "SWT column not found in data."
    assert "TVD" in data.columns, "TVD column not found in data."
    assert_well_config_structure(well_config)

    # Create BVO and BVOH columns
    data["BVO"] = data["PHIT"] * (1 - data["SWT"])
    data["BVOH"] = data["PHIT"] * (1 - data["SHF"]) if "SHF" in data.columns else 0

    # Sort well names based on sorting key
    well_names = sorted(
        data["WELL_NAME"].unique(), key=lambda name: well_config[name]["sorting"]
    )

    # Create subplots
    fig, axes = plt.subplots(
        nrows=1, ncols=len(well_names), sharey=True, figsize=(len(well_names) * 2, 15)
    )

    # Plot each well's data
    for ax, well_name in zip(axes, well_names):
        logger.debug(f"Plotting well: {well_name}")
        well_data = data[
            (data["WELL_NAME"] == well_name) & (data["ZONES"] == zone)
        ].copy()
        well_data = update_fluid_contacts(
            well_data, well_config[well_name]["zones"][zone]
        )
        ax.plot(well_data["BVO"], well_data["TVD"], label=r"$BVO_{Log}$")
        if "BVOH" in well_data.columns:
            ax.plot(well_data["BVOH"], well_data["TVD"], label=r"$BVO_{SHF}$")

        # Fill between based on fluid flag
        ax.fill_betweenx(
            well_data["TVD"],
            0,
            1,
            where=well_data["FLUID_FLAG"] == 1,
            color="g",
            alpha=0.3,
            label="Oil",
        )
        ax.fill_betweenx(
            well_data["TVD"],
            0,
            1,
            where=well_data["FLUID_FLAG"] == 2,
            color="r",
            alpha=0.3,
            label="Gas",
        )
        ax.fill_betweenx(
            well_data["TVD"],
            0,
            1,
            where=well_data["FLUID_FLAG"] == 0,
            color="b",
            alpha=0.3,
            label="Water",
        )

        ax.set_title(f"Well: {well_name}")
        ax.set_xlim(0, 0.5)
        ax.legend()

    axes[0].set_ylabel("Depth (TVD)")
    fig.subplots_adjust(wspace=0.3, hspace=0)
    fig.set_facecolor("aliceblue")
    plt.gca().invert_yaxis()
    plt.show()


def neutron_density_xplot(
    nphi,
    rhob,
    dry_min1_point: tuple,
    dry_clay_point: tuple,
    fluid_point: tuple = (1.0, 1.0),
    wet_clay_point: tuple = (),
    dry_silt_point: tuple = (),
    responsive: bool = True,
    **kwargs,
):
    """Neutron-Density crossplot with lithology lines based on specified end points.
    This plot is a standard petrophysical tool used to identify lithology and porosity.

    Args:
        nphi (np.ndarray or float): Neutron porosity log values.
        rhob (np.ndarray or float): Bulk density log values.
        dry_min1_point (tuple): (NPHI, RHOB) coordinates for the primary matrix mineral.
        dry_clay_point (tuple): (NPHI, RHOB) coordinates for dry clay.
        fluid_point (tuple, optional): (NPHI, RHOB) coordinates for the formation fluid. Defaults to (1.0, 1.0).
        wet_clay_point (tuple, optional): (NPHI, RHOB) coordinates for wet clay. Defaults to ().
        dry_silt_point (tuple, optional): (NPHI, RHOB) coordinates for dry silt. Defaults to ().

    Returns:
        matplotlib.figure.Figure: The generated Neutron-Density crossplot figure.
    """
    logger.info("Generating neutron-density crossplot (Plotly).")
    A = dry_min1_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))

    # Compute projected points (intersection of mineral-clay line with fluid->point lines)
    projected_pt = []
    for i in range(len(nphi)):
        pt = line_intersection((A, C), (D, E[i]))
        if pt is not None:
            projected_pt.append(pt)

    # Build Plotly figure
    fig = go.Figure()

    # Data points colored by index (for depth ordering or sequence)
    fig.add_trace(
        go.Scatter(
            x=list(nphi),
            y=list(rhob),
            mode="markers",
            marker=dict(
                color=list(range(len(nphi))),
                colorscale="Rainbow",
                showscale=False,
                size=6,
            ),
            name="Data",
            hoverinfo="x+y+text",
        )
    )

    # Mineral 1 line (D -> A)
    fig.add_trace(
        go.Scatter(
            x=[D[0], A[0]],
            y=[D[1], A[1]],
            mode="lines",
            line=dict(color="blue"),
            name="Mineral 1 Line",
        )
    )

    # Clay line (D -> C)
    fig.add_trace(
        go.Scatter(
            x=[D[0], C[0]],
            y=[D[1], C[1]],
            mode="lines",
            line=dict(color="gray"),
            name="Clay Line",
        )
    )

    # Rock line (A -> C)
    fig.add_trace(
        go.Scatter(
            x=[A[0], C[0]],
            y=[A[1], C[1]],
            mode="lines",
            line=dict(color="black"),
            name="Rock Line",
        )
    )

    # Silt line and marker
    if dry_silt_point:
        B = dry_silt_point
        fig.add_trace(
            go.Scatter(
                x=[D[0], B[0]],
                y=[D[1], B[1]],
                mode="lines",
                line=dict(color="green"),
                name="Silt Line",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[B[0]],
                y=[B[1]],
                mode="markers",
                marker=dict(color="orange", size=8),
                name="Dry Silt Point",
            )
        )

    # Projected points
    if projected_pt:
        xs, ys = zip(*projected_pt)
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="markers",
                marker=dict(color="purple", size=4),
                name="Projected Point",
            )
        )

    # Key markers: mineral, clay, wet clay, fluid
    fig.add_trace(
        go.Scatter(
            x=[A[0]],
            y=[A[1]],
            mode="markers",
            marker=dict(color="yellow", size=9),
            name=f"Mineral Point ({A[0]}, {A[1]})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[C[0]],
            y=[C[1]],
            mode="markers",
            marker=dict(color="black", size=9),
            name=f"Dry Clay ({C[0]}, {C[1]})",
        )
    )
    if wet_clay_point:
        fig.add_trace(
            go.Scatter(
                x=[wet_clay_point[0]],
                y=[wet_clay_point[1]],
                mode="markers",
                marker=dict(color="gray", size=8),
                name="Wet Clay",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[D[0]],
            y=[D[1]],
            mode="markers",
            marker=dict(color="blue", size=9),
            name=f"Fluid ({D[0]}, {D[1]})",
        )
    )

    # Layout: invert y-axis (depth-style), tidy ranges similar to previous implementation
    # Build layout dict so we can toggle fixed pixel size vs responsive autosize
    layout = dict(
        title="NPHI-RHOB Crossplot",
        xaxis=dict(title="NPHI", range=[-0.10, 1]),
        yaxis=dict(title="RHOB", autorange=False, range=[3, 0]),
        legend=dict(orientation="v", yanchor="top", y=0.99, xanchor="left", x=0.01),
        template="plotly_white",
        margin=dict(l=40, r=10, t=40, b=40),
    )

    if responsive:
        # Let plotly.js adapt to container size. The frontend should pass
        # `config = { responsive: true }` when calling `newPlot`/`react`.
        layout["autosize"] = True
        # Do not set `width`/`height` here; the container (CSS) controls width.
    else:
        # Fixed pixel size (square) when not responsive
        layout["width"] = 600
        layout["height"] = 600

    fig.update_layout(**layout)

    return fig
