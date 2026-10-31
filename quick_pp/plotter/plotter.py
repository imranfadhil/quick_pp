import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

import quick_pp.plotter.well_log as plotter_wells
from quick_pp import logger
from quick_pp.utils import line_intersection

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


def generate_zone_config(zones: list = None):
    """Generate zone configuration.

    Args:
        zones (list, optional): List of zone names. Defaults to ['ALL'].

    Returns:
        dict: Zone configuration
    """
    if zones is None:
        zones = ["ALL"]
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


def generate_well_config(well_names: list = None):
    """Generate well configuration.

    Args:
        well_names (list, optional): List of well names. Defaults to ['X'].

    Returns:
        dict: Well configuration
    """
    if well_names is None:
        well_names = ["X"]
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
    fluid_contacts: dict = None,
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
    if fluid_contacts is None:
        fluid_contacts = {}
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
    for ax, well_name in zip(axes, well_names, strict=True):
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
    E = list(zip(nphi, rhob, strict=True))

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
            marker={
                "color": list(range(len(nphi))),
                "colorscale": "Rainbow",
                "showscale": False,
                "size": 6,
            },
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
            line={"color": "blue"},
            name="Mineral 1 Line",
        )
    )

    # Clay line (D -> C)
    fig.add_trace(
        go.Scatter(
            x=[D[0], C[0]],
            y=[D[1], C[1]],
            mode="lines",
            line={"color": "gray"},
            name="Clay Line",
        )
    )

    # Rock line (A -> C)
    fig.add_trace(
        go.Scatter(
            x=[A[0], C[0]],
            y=[A[1], C[1]],
            mode="lines",
            line={"color": "black"},
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
                line={"color": "green"},
                name="Silt Line",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[B[0]],
                y=[B[1]],
                mode="markers",
                marker={"color": "orange", "size": 8},
                name="Dry Silt Point",
            )
        )

    # Projected points
    if projected_pt:
        xs, ys = zip(*projected_pt, strict=True)
        fig.add_trace(
            go.Scatter(
                x=list(xs),
                y=list(ys),
                mode="markers",
                marker={"color": "purple", "size": 4},
                name="Projected Point",
            )
        )

    # Key markers: mineral, clay, wet clay, fluid
    fig.add_trace(
        go.Scatter(
            x=[A[0]],
            y=[A[1]],
            mode="markers",
            marker={"color": "yellow", "size": 9},
            name=f"Mineral Point ({A[0]}, {A[1]})",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[C[0]],
            y=[C[1]],
            mode="markers",
            marker={"color": "black", "size": 9},
            name=f"Dry Clay ({C[0]}, {C[1]})",
        )
    )
    if wet_clay_point:
        fig.add_trace(
            go.Scatter(
                x=[wet_clay_point[0]],
                y=[wet_clay_point[1]],
                mode="markers",
                marker={"color": "gray", "size": 8},
                name="Wet Clay",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[D[0]],
            y=[D[1]],
            mode="markers",
            marker={"color": "blue", "size": 9},
            name=f"Fluid ({D[0]}, {D[1]})",
        )
    )

    # Layout: invert y-axis (depth-style), tidy ranges similar to previous implementation
    # Build layout dict so we can toggle fixed pixel size vs responsive autosize
    layout = {
        "title": "NPHI-RHOB Crossplot",
        "xaxis": {"title": "NPHI", "range": [-0.10, 1]},
        "yaxis": {"title": "RHOB", "autorange": False, "range": [3, 0]},
        "legend": {
            "orientation": "v",
            "yanchor": "top",
            "y": 0.99,
            "xanchor": "left",
            "x": 0.01,
        },
        "template": "plotly_white",
        "margin": {"l": 40, "r": 10, "t": 40, "b": 40},
    }

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


# Petrophysical fluid colour conventions (parity with frontend petrophysicsColors.ts)
FPT_FLUID_COLORS = {
    "GAS": "#EF4444",
    "OIL": "#10B981",
    "WATER": "#3B82F6",
    "UNASSIGNED": "#64748B",
}
FPT_EXCLUDED_COLOR = "#9CA3AF"


def plot_formation_pressure(
    analyzer,
    show_excess: bool = True,
    well_col: str = "WELL_NAME",
    title: str = "Formation Pressure vs. TVDSS",
    figsize: tuple = (11, 7),
    ref_water_gradient: float = None,
):
    """Dual-track Pressure (and Excess Pressure) vs. TVDSS plot from a fitted FormationPressureAnalyzer.

    Left track: measured formation pressure coloured by fluid phase, gradient trend lines extrapolated
    across the depth range, horizontal contact lines with P10-P90 shaded bands and density call-outs.
    Right track (optional): excess pressure relative to the hydrostatic water reference.

    Args:
        analyzer (quick_pp.fluid_type.FormationPressureAnalyzer): Analyzer after `fit_all()` /
            `fit_gradient()` and (optionally) `estimate_contacts()`.
        show_excess (bool): Add the Excess Pressure track. Defaults to True.
        well_col (str): Column used for per-well marker shapes. Defaults to 'WELL_NAME'.
        title (str): Figure title.
        figsize (tuple): Matplotlib figure size.
        ref_water_gradient (float, optional): Reference gradient for excess pressure
            (defaults to the fitted WATER line or 1.0 g/cc).

    Returns:
        matplotlib.figure.Figure: The generated figure.
    """
    df = analyzer.df
    zc, pc, fc, qc = (
        analyzer.tvdss_col,
        analyzer.pressure_col,
        analyzer.fluid_col,
        analyzer.quality_col,
    )
    pu, du = analyzer.pressure_unit, analyzer.depth_unit
    ncols = 2 if show_excess else 1
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=figsize,
        sharey=True,
        gridspec_kw={"width_ratios": [2.2, 1] if show_excess else [1]},
    )
    axes = np.atleast_1d(axes)
    ax = axes[0]

    markers = ["o", "s", "^", "D", "v", "P", "X", "<", ">", "h"]
    wells = list(df[well_col].dropna().unique()) if well_col in df.columns else [None]
    excess = analyzer.calculate_excess_pressure(ref_water_gradient) if show_excess else None
    fin = df[zc].notna() & df[pc].notna()

    for wi, wn in enumerate(wells):
        wmask = fin & ((df[well_col] == wn) if wn is not None else True)
        mk = markers[wi % len(markers)]
        for fl, color in FPT_FLUID_COLORS.items():
            m_valid = wmask & (df[fc] == fl) & (df[qc] == "VALID")
            if m_valid.any():
                lbl = f"{fl.title()}" + (f" · {wn}" if wn is not None and len(wells) > 1 else "")
                ax.scatter(df.loc[m_valid, pc], df.loc[m_valid, zc], c=color, marker=mk, s=28,
                           edgecolors="white", linewidths=0.4, label=lbl, zorder=3)
                if excess is not None:
                    axes[1].scatter(excess[m_valid], df.loc[m_valid, zc], c=color, marker=mk,
                                    s=24, edgecolors="white", linewidths=0.4, zorder=3)
        m_bad = wmask & (df[qc] != "VALID")
        if m_bad.any():
            ax.scatter(df.loc[m_bad, pc], df.loc[m_bad, zc], facecolors="none",
                       edgecolors=FPT_EXCLUDED_COLOR, marker=mk, s=30, linewidths=0.9,
                       label="Excluded / Tight / Supercharged" if wi == 0 else None, zorder=2)
            if excess is not None:
                axes[1].scatter(excess[m_bad], df.loc[m_bad, zc], facecolors="none",
                                edgecolors=FPT_EXCLUDED_COLOR, marker=mk, s=26, linewidths=0.9,
                                zorder=2)

    z_vals = df.loc[fin, zc]
    contact_z = [c.tvdss_p50 for c in analyzer.contacts.values()]
    z_lo = float(min([z_vals.min()] + contact_z)) if len(z_vals) else 0.0
    z_hi = float(max([z_vals.max()] + contact_z)) if len(z_vals) else 1.0
    pad = max((z_hi - z_lo) * 0.08, 1.0)
    z_lo, z_hi = z_lo - pad, z_hi + pad

    for fl, g in analyzer.gradients.items():
        color = FPT_FLUID_COLORS.get(fl, "#334155")
        zz = np.array([z_lo, z_hi])
        ax.plot(g.pressure_at(zz), zz, color=color, lw=1.4,
                ls="-" if g.method != "manual" else "--", alpha=0.85, zorder=2,
                label=f"{fl.title()} {g.gradient:.4f} {pu}/{du} (ρ={g.density_gcc:.3f} g/cc)")

    for label, c in analyzer.contacts.items():
        color = FPT_FLUID_COLORS.get(c.lower_fluid.fluid_type, "#334155")
        for a in axes:
            if c.tvdss_p90 > c.tvdss_p10:
                a.axhspan(c.tvdss_p10, c.tvdss_p90, color=color, alpha=0.12, zorder=1)
            a.axhline(c.tvdss_p50, color=color, ls="--", lw=1.0, zorder=1)
        half = 0.5 * (c.tvdss_p90 - c.tvdss_p10)
        name = "OWC / FWL" if label == "OWC" else label
        ax.annotate(
            f"{name}: {c.tvdss_p50:.1f} {du} TVDSS (±{half:.1f}) @ {c.pressure_at_contact:.0f} {pu} | "
            f"ρ_{c.upper_fluid.fluid_type[0].lower()}={c.upper_fluid.density_gcc:.2f}, "
            f"ρ_{c.lower_fluid.fluid_type[0].lower()}={c.lower_fluid.density_gcc:.2f} g/cc",
            xy=(0.01, c.tvdss_p50), xycoords=("axes fraction", "data"),
            xytext=(0, -3), textcoords="offset points", va="top", fontsize=7.5,
            color="#0F172A",
            bbox={"boxstyle": "round,pad=0.2", "fc": "white", "ec": color, "alpha": 0.85},
        )

    ax.set_ylim(z_hi, z_lo)
    ax.set_xlabel(f"Formation Pressure ({pu})")
    ax.set_ylabel(f"TVDSS ({du})")
    ax.grid(True, ls=":", alpha=0.6)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)
    ax.set_title(title, fontsize=11)
    if show_excess:
        axes[1].axvline(0.0, color="#475569", lw=0.8)
        axes[1].set_xlabel(f"Excess Pressure ({pu})")
        axes[1].grid(True, ls=":", alpha=0.6)
        axes[1].set_title("Excess vs. Hydrostatic", fontsize=10)
    fig.tight_layout()
    return fig
