import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quick_pp.plotter.well_log_config import COLOR_DICT, TRACE_DEFS, XAXIS_DEFS


# --- Helper to add traces from trace_defs ---
def add_defined_traces(fig, df, index, no_of_track, trace_defs, **kwargs):
    """Add predefined traces to a Plotly figure.

    Args:
        fig (plotly.graph_objects.Figure): The Plotly figure to add traces to.
        df (pd.DataFrame): The DataFrame containing the log data.
        index (pd.Series): The depth index for the y-axis.
        no_of_track (int): The total number of tracks in the plot.
        trace_defs (dict): A dictionary defining the traces and their styles.
        **kwargs: Additional keyword arguments to pass to `go.Scatter`.

    Returns:
        plotly.graph_objects.Figure: The figure with the added traces.
    """
    trace_ix = 0
    for col, trace_def in trace_defs.items():
        # Skip special cases handled separately
        if col in ["COAL_FLAG", "ZONES"]:
            continue

        style = trace_def.get("style", {})
        trace_args = dict(
            x=round(df[col], 3) if df[col].dtype == "float64" else df[col],
            y=index,
            name=col,
            **kwargs,
        )
        trace_args.update(style)
        fig.add_trace(
            go.Scatter(**trace_args),
            row=1,
            col=trace_def["track"],
            secondary_y=trace_def.get("secondary_y", False),
        )
        # Required for xaxis visibility for overlapping traces
        if not trace_def.get("hide_xaxis", True) and trace_ix >= no_of_track:
            fig.data[trace_ix].update(xaxis=f"x{trace_ix + 1}")
        trace_ix += 1

    return fig


def add_crossover_traces(df):
    """Calculate and add gas crossover shading traces to the DataFrame.

    If 'NPHI' and 'RHOB' are present, this function calculates the region
    where neutron porosity crosses over bulk density, indicating potential
    gas zones. It adds 'RHOB_ON_NPHI_SCALE', 'GAS_XOVER_TOP', and
    'GAS_XOVER_BOTTOM' columns to the DataFrame for plotting.

    Returns:
        pd.DataFrame: The DataFrame with added crossover-related columns.

    """
    if (
        "NPHI" in df.columns
        and "RHOB" in df.columns
        and "NPHI_XOVER_BOTTOM" not in df.columns
    ):
        rhob_min, rhob_max = 1.95, 2.95
        nphi_min_scale, nphi_max_scale = 0.45, -0.15
        rhob_on_nphi_scale = nphi_min_scale + (df["RHOB"] - rhob_min) * (
            nphi_max_scale - nphi_min_scale
        ) / (rhob_max - rhob_min)
        crossover_condition = df["NPHI"] < rhob_on_nphi_scale
        # Fill first and last crossover_condition values with False to avoid edge effects when plotting
        if len(crossover_condition) > 0:
            crossover_condition.iloc[0] = False
            crossover_condition.iloc[-1] = False
        df["RHOB_ON_NPHI_SCALE"] = rhob_on_nphi_scale
        df["GAS_XOVER_TOP"] = np.where(
            crossover_condition, rhob_on_nphi_scale, nphi_min_scale
        )
        df["GAS_XOVER_BOTTOM"] = np.where(
            crossover_condition, df["NPHI"], nphi_min_scale
        )
    return df


def add_rock_flag_traces(df):
    """Prepare rock flag data for plotting by one-hot encoding.

    If a 'ROCK_FLAG' column exists, this function:
    1. Converts it to integer type.
    2. Assigns a unique color to each flag value in `COLOR_DICT`.
    3. One-hot encodes the 'ROCK_FLAG' column into separate 'ROCK_FLAG_X' columns.

    Returns:
        pd.DataFrame: The DataFrame with one-hot encoded rock flag columns.

    """
    if "ROCK_FLAG" in df.columns:
        # Add ROCK_FLAG colors
        df["ROCK_FLAG"] = df["ROCK_FLAG"].fillna(0).astype(int)
        no_of_rock_flags = df["ROCK_FLAG"].nunique() + 1
        df["ROCK_FLAG"] = df["ROCK_FLAG"].fillna(no_of_rock_flags)
        sorted_rock_flags = sorted(df["ROCK_FLAG"].unique().tolist())
        for i, rock_flag in enumerate(sorted_rock_flags):
            lightness = 100 - (i / no_of_rock_flags * 100)
            fill_color = f"hsl(30, 70%, {lightness}%)"
            COLOR_DICT[f"ROCK_FLAG_{rock_flag}"] = fill_color
            # print(f"i: {i}, rock_flag: {rock_flag}, fill_color: {fill_color}")

        df["ROCK_FLAG"] = df["ROCK_FLAG"].astype("category")
        df = df.drop(columns=[c for c in df.columns if "ROCK_FLAG_" in c])
        df = pd.get_dummies(df, columns=["ROCK_FLAG"], prefix="ROCK_FLAG", dtype=int)
        if "ROCK_FLAG_0" not in df.columns:
            df["ROCK_FLAG_0"] = 0
    return df


def fix_missing_volumetrics(df):
    """Ensure standard volumetric columns exist and fill NaNs.

    Iterates through a predefined list of volumetric curve names. If a column
    exists, it fills missing values with 0. If it doesn't exist, it is not added.

    Returns:
        pd.DataFrame: The DataFrame with NaNs filled in volumetric columns.
    """
    volumetrics = ["VCLAY", "VSILT", "VSAND", "VCALC", "VDOLO", "VGAS", "VOIL", "VHC"]
    for col in volumetrics:
        df[col] = df[col].fillna(0) if col in df.columns else None
    return df


def fix_missing_depths(df: pd.DataFrame) -> pd.DataFrame:
    """Resamples a DataFrame to a regular depth interval.

    This function addresses gaps in depth data by creating a complete,
    evenly-spaced depth range and merging the original data onto it. It uses
    the most common depth interval found in the data as the new step.

    Args:
        df (pd.DataFrame): The input DataFrame, which must contain a 'DEPTH' column.

    Returns:
        pd.DataFrame: A new, resampled DataFrame with consistent depth steps.
                      Returns an empty DataFrame if the input is invalid.
    """
    if df.empty or "DEPTH" not in df.columns or df["DEPTH"].nunique() < 2:
        # Cannot process if DataFrame is empty, no 'DEPTH' column,
        # or not enough data points to determine an interval.
        return pd.DataFrame(columns=df.columns)

    # --- FIX: Sort the right DataFrame (df) by the merge key ('DEPTH') ---
    # This is required by pd.merge_asof.
    df = df.sort_values(by="DEPTH").reset_index(drop=True)

    # Calculate the most common depth interval (step).
    # Using the median of positive differences is more robust against outliers.
    depth_diffs = df["DEPTH"].diff()
    if depth_diffs.iloc[1:].empty:  # Check if diff() resulted in an empty series
        return df  # Not enough data to resample, return sorted original
    depth_step = depth_diffs.iloc[1:].median()

    # If the calculated step is zero or negative, we can't proceed.
    if depth_step <= 0:
        return df  # Return the sorted original DataFrame

    depth_min = df["DEPTH"].min()
    depth_max = df["DEPTH"].max()

    # Create a new, complete depth range.
    # The tolerance ensures the max depth is included in the range.
    complete_depths = np.arange(depth_min, depth_max + depth_step, depth_step)

    # Create a new DataFrame with the complete depth range.
    df_complete = pd.DataFrame({"DEPTH": complete_depths})

    # Merge the original data onto the complete depth range.
    # 'direction="nearest"' finds the closest original depth for each new depth point.
    df_resampled = pd.merge_asof(
        df_complete, df, on="DEPTH", direction="nearest", tolerance=depth_step
    )

    return df_resampled


def plotly_log(
    well_data,
    well_name: str = "",
    depth_uom="",
    trace_defs: dict = None,
    xaxis_defs: dict = None,
    column_widths: list = None,
):
    """Generate a multi-track well log plot using Plotly.

    This function creates a comprehensive, interactive well log plot with multiple
    tracks, supporting various log types, custom styling, and special overlays
    like gas crossover, rock flags, and formation tops.

    Args:
        well_data (pd.DataFrame): The primary DataFrame containing well log data.
                                  It must include a 'DEPTH' column.
        well_name (str, optional): The name of the well, to be displayed as the plot title. Defaults to ''.
        depth_uom (str, optional): The unit of measurement for depth (e.g., 'm', 'ft'). Defaults to "".
        trace_defs (dict, optional): A dictionary to define and style traces. If not provided,
                                     defaults to `TRACE_DEFS` from the config.
        xaxis_defs (dict, optional): A dictionary to configure x-axes. If not provided,
                                     defaults to `XAXIS_DEFS` from the config.
        column_widths (list, optional): A list of relative widths for each track.
                                        If empty, all tracks will have equal width. Defaults to [].

    Returns:
        go.Figure: A Plotly Figure object representing the well log plot.

    Example:
        >>> from quick_pp.plotter import plotly_log
        >>> fig = plotly_log(well_df, well_name='MyWell', depth_uom='m')
        >>> fig.show()
    """
    trace_defs = trace_defs or TRACE_DEFS
    xaxis_defs = xaxis_defs or XAXIS_DEFS
    no_of_track = max([trace["track"] for trace in trace_defs.values()])
    column_widths = column_widths or [1] * no_of_track
    df = well_data.copy()

    # Fix missing depths
    df = fix_missing_depths(df)
    index = df.DEPTH

    # Add yellow shaded crossover if NPHI RHOB present in df
    df = add_crossover_traces(df)

    # Ensure all required columns exist (fill with NaN if missing)
    for k in trace_defs.keys():
        if k not in df.columns:
            df[k] = np.nan

    # One-hot encode ROCK_FLAG if present
    df = add_rock_flag_traces(df)

    # Fix missing volumetrics
    df = fix_missing_volumetrics(df)

    fig = make_subplots(
        rows=1,
        cols=no_of_track,
        shared_yaxes=True,
        horizontal_spacing=0.02,
        column_widths=column_widths,
        specs=[list([{"secondary_y": True}] * no_of_track)],
    )
    fig = add_defined_traces(fig, df, index, no_of_track, trace_defs)

    # --- ROCK_FLAG_X traces (special style, always on track 7, secondary_y=False) ---
    if (
        "ROCK_FLAG_0" in df.columns
        and no_of_track >= 7
        and "ROCK_FLAG_0" in trace_defs.keys()
    ):
        rock_flag_columns = [col for col in df.columns if col.startswith("ROCK_FLAG_")]
        # Remove 'ROCK_FLAG_0' if it already exists as a trace in fig.data
        if any(trace.name == "ROCK_FLAG_0" for trace in fig.data):
            rock_flag_columns.remove("ROCK_FLAG_0")
        fig.update_traces(selector={"name": "ROCK_FLAG_0"}, visible=False)
        for feature in rock_flag_columns:
            fig.add_trace(
                go.Scatter(
                    x=df[feature].replace(0, -999.25),
                    y=index,
                    line_color=COLOR_DICT[feature],
                    name=feature,
                    fill="tozerox",
                    fillcolor=COLOR_DICT.get(feature, "#000000"),
                ),
                row=1,
                col=7,
                secondary_y=False,
            )

    # --- COAL_FLAG traces (special style, always on tracks 4-8, secondary_y=True) ---
    if "COAL_FLAG" in df.columns and no_of_track >= 8:
        COAL_FLAG_HOVER = np.where(
            df["COAL_FLAG"] == 1, "COAL_FLAG<extra></extra>", "<extra></extra>"
        )
        df["COAL_FLAG"] = df["COAL_FLAG"].replace(
            {0: 1e-3, 1: 1e9}
        )  # Cater for plotting on log scale
        for c in [4, 5, 6, 7, 8]:
            fig.add_trace(
                go.Scatter(
                    x=df["COAL_FLAG"],
                    y=index,
                    text=COAL_FLAG_HOVER,
                    line_width=0,
                    fill="tozerox",
                    fillcolor="rgba(0,0,0,1)",
                    hovertemplate="%{text}",
                ),
                row=1,
                col=c,
                secondary_y=True,
            )

    # Add ZONES to hover box in each tracks if available
    if "ZONES" in df.columns:
        for c in range(1, no_of_track + 1):
            fig.add_trace(
                go.Scatter(
                    x=np.zeros(len(df)),
                    y=index,
                    name="ZONES",
                    line={"width": 0},
                    text=df["ZONES"],
                    hovertemplate="<b>ZONES</b>: %{text}<extra></extra>",
                ),
                row=1,
                col=c,
                secondary_y=False,
            )

    # Add TVD to hover box in each tracks if available
    if "TVD" in df.columns:
        for c in range(1, no_of_track + 1):
            fig.add_trace(
                go.Scatter(
                    x=df["TVD"],
                    y=index,
                    name="TVD",
                    line={"width": 0},
                    hovertemplate="<b>TVD</b>: %{x}<extra></extra>",
                ),
                row=1,
                col=c,
                secondary_y=False,
            )

    # Update xaxis and yaxis layout configurations
    visible_xaxis = [
        col
        for col, trace_def in trace_defs.items()
        if not trace_def.get("hide_xaxis", True)
    ]
    layout_dict = {
        f"xaxis{x.xaxis[1:]}": v
        for k, v in xaxis_defs.items()
        for x in fig.data
        if k in visible_xaxis and x.name == k
    }
    layout_dict["yaxis"] = {
        "domain": [0, 0.84],
        "title": {"text": f"<b>DEPTH ({depth_uom.upper()}) MD</b>", "font_size": 12},
    }

    for i in range(1, no_of_track * 2):
        layout_dict[f"yaxis{i}"] = {
            "domain": [0, 0.84],
            "visible": False if i % 2 == 0 and i != no_of_track * 2 else True,
            "showgrid": False,
        }
    fig.update_layout(**layout_dict)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(
        matches="y",
        constrain="domain",
        range=[df["DEPTH"].max() + 10, df["DEPTH"].min() - 10],
    )

    def add_zone_markers(fig, df):
        """Adds horizontal lines and labels for formation tops.

        If a 'ZONES' column is present, this function identifies changes in
        the zone names and adds a horizontal line and a text annotation to the
        plot for each formation top.

        Args:
            fig (plotly.graph_objects.Figure): The figure to add markers to.
            df (pd.DataFrame): The DataFrame containing 'DEPTH' and 'ZONES' columns.

        """
        if "ZONES" in df.columns:
            tops_df = df[["DEPTH", "ZONES"]].dropna().reset_index()
            # Limit the autogenerated ZONE_* to optimize the plotting
            if (
                not tops_df.empty
                and sum([1 for c in tops_df.ZONES.unique() if "ZONE_" in c]) < 30
            ):
                zone_tops_idx = [0] + [
                    idx
                    for idx, (i, j) in enumerate(
                        zip(tops_df["ZONES"], tops_df["ZONES"][1:], strict=True), 1
                    )
                    if i != j
                ]
                zone_tops = tops_df.loc[zone_tops_idx, :]
                for tops in zone_tops.values:
                    fig.add_shape(
                        {
                            "type": "line",
                            "x0": -5,
                            "y0": tops[1],
                            "x1": 1e6,
                            "y1": tops[1],
                        },
                        row=1,
                        col="all",
                        line={"color": "#763F98", "dash": "dot", "width": 1.5},
                    )
                    fig.add_trace(
                        go.Scatter(
                            name=tops[2],
                            x=[1.1, 1.1],
                            y=[tops[1] - 1, tops[1] - 1],
                            hoverinfo="skip",
                            text=tops[2],
                            mode="text",
                            textfont={"color": "#763F98", "size": 12},
                            textposition="top right",
                        ),
                        row=1,
                        col=2,
                    )

    add_zone_markers(fig, df)

    fig.update_layout(
        height=900,
        autosize=True,
        showlegend=False,
        title={
            "text": "<b>%s Well Logs</b>" % well_name,
            "y": 0.99,
            "xanchor": "center",
            "yanchor": "top",
            "font_size": 12,
        },
        hovermode="y unified",
        hoverdistance=1,
        hoverlabel_font_size=10,
        hoverlabel_bgcolor="#F3F3F3",
        dragmode="pan",
        modebar_remove=["lasso", "select", "autoscale"],
        modebar_add=["drawline", "drawcircle", "drawrect", "eraseshape"],
        newshape_line_color="cyan",
        newshape_line_width=3,
        template="none",
        margin={"l": 70, "r": 0, "t": 20, "b": 10},
        paper_bgcolor="#dadada",
    )

    return fig
