import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from quick_pp.plotter.well_log_config import COLOR_DICT, TRACE_DEFS, XAXIS_DEFS


# --- Helper to add traces from trace_defs ---
def add_defined_traces(fig, df, index, no_of_track, trace_defs, **kwargs):
    trace_ix = 0
    for col, trace_def in trace_defs.items():
        # Skip special cases handled separately
        if col in ['COAL_FLAG', 'ZONES']:
            continue

        style = trace_def.get('style', {})
        trace_args = dict(
            x=df[col],
            y=index,
            name=col,
            **kwargs
        )
        trace_args.update(style)
        fig.add_trace(
            go.Scatter(**trace_args),
            row=1,
            col=trace_def['track'],
            secondary_y=trace_def.get('secondary_y', False)
        )
        # Required for xaxis visibility for overlapping traces
        if not trace_def.get('hide_xaxis', True) and trace_ix >= no_of_track:
            fig.data[trace_ix].update(xaxis=f'x{trace_ix + 1}')
        trace_ix += 1

    return fig


def add_crossover_traces(df):
    if 'NPHI' in df.columns and 'RHOB' in df.columns and 'NPHI_XOVER_BOTTOM' not in df.columns:
        rhob_min, rhob_max = 1.95, 2.95
        nphi_min_scale, nphi_max_scale = 0.45, -0.15
        rhob_on_nphi_scale = nphi_min_scale + (
            df['RHOB'] - rhob_min) * (nphi_max_scale - nphi_min_scale) / (rhob_max - rhob_min)
        crossover_condition = df['NPHI'] < rhob_on_nphi_scale
        # Fill first and last crossover_condition values with False to avoid edge effects when plotting
        if len(crossover_condition) > 0:
            crossover_condition.iloc[0] = False
            crossover_condition.iloc[-1] = False
        df['RHOB_ON_NPHI_SCALE'] = rhob_on_nphi_scale
        df['GAS_XOVER_TOP'] = np.where(crossover_condition, rhob_on_nphi_scale, nphi_min_scale)
        df['GAS_XOVER_BOTTOM'] = np.where(crossover_condition, df['NPHI'], nphi_min_scale)
    return df


def add_rock_flag_traces(df):
    if 'ROCK_FLAG' in df.columns:
        # Add ROCK_FLAG colors
        df['ROCK_FLAG'] = df['ROCK_FLAG'].fillna(0).astype(int)
        no_of_rock_flags = df['ROCK_FLAG'].nunique() + 1
        df['ROCK_FLAG'] = df['ROCK_FLAG'].fillna(no_of_rock_flags)
        sorted_rock_flags = sorted(df['ROCK_FLAG'].unique().tolist())
        for i, rock_flag in enumerate(sorted_rock_flags):
            lightness = 100 - (i / no_of_rock_flags * 100)
            fill_color = f'hsl(30, 70%, {lightness}%)'
            COLOR_DICT[f'ROCK_FLAG_{rock_flag}'] = fill_color
            # print(f"i: {i}, rock_flag: {rock_flag}, fill_color: {fill_color}")

        df['ROCK_FLAG'] = df['ROCK_FLAG'].astype('category')
        df = df.drop(columns=[c for c in df.columns if 'ROCK_FLAG_' in c])
        df = pd.get_dummies(df, columns=['ROCK_FLAG'], prefix='ROCK_FLAG', dtype=int)
        if 'ROCK_FLAG_0' not in df.columns:
            df['ROCK_FLAG_0'] = 0
    return df


def fix_missing_volumetrics(df):
    volumetrics = ['VCLD', 'VSILT', 'VSAND', 'VCALC', 'VDOLO', 'VGAS', 'VOIL', 'VHC']
    for col in volumetrics:
        df[col] = df[col].fillna(0) if col in df.columns else None
    return df


def fix_missing_depths(df):
    depth_min = df['DEPTH'].min()
    depth_max = df['DEPTH'].max()
    depth_step = df['DEPTH'].diff().mode()[0].round(4)  # Most common depth step
    complete_depths = np.arange(depth_min, depth_max + depth_step, depth_step)

    # Create new dataframe with complete depths and merge with original data
    df_complete = pd.DataFrame({'DEPTH': np.round(complete_depths, 4)})
    df = pd.merge_asof(df_complete, df, on='DEPTH', direction='nearest', tolerance=depth_step)
    df = df.sort_values(by='DEPTH', ascending=True)
    return df


def plotly_log(well_data, well_name: str = '', depth_uom="", trace_defs: dict = dict(),
               xaxis_defs: dict = {}, column_widths: list = []):
    """
    Generate a multi-track well log plot using Plotly, supporting custom traces, rock/coal flags, and zone markers.
    Parameters
    ----------
    well_data : pandas.DataFrame
        DataFrame containing well log data. Must include a 'DEPTH' column and may include columns referenced in
        `trace_defs`, as well as optional 'ROCK_FLAG', 'COAL_FLAG', and 'ZONES' columns.
    depth_uom : str, optional
        Unit of measurement for depth, displayed on the y-axis label (e.g., 'm', 'ft'). Default is an empty string.
    trace_defs : dict, optional
        Dictionary defining traces to plot. Keys are column names in `well_data`, values are dicts with at least:
            - 'track': int, the subplot column (track) to plot on.
            - 'style': dict, Plotly Scatter style arguments (e.g., line color, dash).
            - 'secondary_y': bool, whether to plot on the secondary y-axis of the track.
            - 'hide_xaxis': bool, whether to hide the x-axis for this trace.
        Default is TRACE_DEFS.
    xaxis_defs : dict, optional
        Dictionary defining x-axis layout options for each trace. Keys are column names, values are dicts of axis layout
        properties (e.g., range, title, tickvals). Default is XAXIS_DEFS.
    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        Plotly Figure object containing the multi-track well log plot.
    Notes
    -----
    - Handles missing columns in `COLOR_DICT` by filling with NaN.
    - One-hot encodes 'ROCK_FLAG' if present, assigning unique colors for each flag.
    - Adds special traces for 'COAL_FLAG' (tracks 4-8, filled black) and 'ROCK_FLAG_X' (track 7, colored by flag).
    - Adds horizontal zone markers and labels if 'ZONES' column is present.
    - Layout is configured for 8 tracks, with shared y-axis (depth), custom margins, and a unified hover mode.
    - The plot is interactive, with pan/zoom and drawing tools enabled, and a custom color theme.
    Examples
    --------
    >>> fig = plotly_log(well_data, depth_uom='m')
    >>> fig.show()
    """
    default_flags = False
    if trace_defs == dict():
        trace_defs = TRACE_DEFS
        default_flags = True

    xaxis_defs = xaxis_defs or XAXIS_DEFS
    no_of_track = max([trace['track'] for trace in trace_defs.values()])
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
        rows=1, cols=no_of_track, shared_yaxes=True, horizontal_spacing=.02,
        column_widths=column_widths, specs=[list([{'secondary_y': True}] * no_of_track)]
    )
    fig = add_defined_traces(fig, df, index, no_of_track, trace_defs)

    # --- COAL_FLAG traces (special style, always on tracks 4-8, secondary_y=True) ---
    if 'COAL_FLAG' in df.columns and no_of_track >= 8 and default_flags:
        df['COAL_FLAG'] = df['COAL_FLAG'].replace({0: 1e-3, 1: 1e9})  # Cater for plotting on log scale
        for c in [4, 5, 6, 7, 8]:
            fig.add_trace(
                go.Scatter(x=df['COAL_FLAG'], y=index, name='', line_color=COLOR_DICT['COAL_FLAG'],
                           fill='tozerox', fillcolor='rgba(0,0,0,1)', opacity=1),
                row=1, col=c, secondary_y=True)

    # --- ROCK_FLAG_X traces (special style, always on track 7, secondary_y=False) ---
    if 'ROCK_FLAG_0' in df.columns and no_of_track >= 7 and default_flags:
        rock_flag_columns = [col for col in df.columns if col.startswith('ROCK_FLAG_')]
        # Remove 'ROCK_FLAG_0' if it already exists as a trace in fig.data
        rock_flag_columns.remove('ROCK_FLAG_0') if any(
            trace.name == 'ROCK_FLAG_0' for trace in fig.data) and 'ROCK_FLAG_0' in rock_flag_columns else None
        for feature in rock_flag_columns:
            fig.add_trace(
                go.Scatter(x=df[feature], y=index, line_color=COLOR_DICT[feature], name=feature,
                           fill='tozerox', fillcolor=COLOR_DICT.get(feature, '#000000')),
                row=1, col=7, secondary_y=False)

    # Update xaxis and yaxis layout configurations
    visible_xaxis = [col for col, trace_def in trace_defs.items() if not trace_def.get('hide_xaxis', True)]
    layout_dict = {f'xaxis{x.xaxis[1:]}': v for k, v in xaxis_defs.items() for x in fig.data
                   if k in visible_xaxis and x.name == k}
    layout_dict['yaxis'] = {'domain': [0, .84], 'title': f'DEPTH ({depth_uom})'}

    for i in range(1, no_of_track * 2 + 1):
        layout_dict[f'yaxis{i}'] = {
            'domain': [0, .84],
            'visible': False if i % 2 == 0 and i != no_of_track * 2 else True,
            'showgrid': False
        }
    fig.update_layout(**layout_dict)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(
        matches='y', constrain='domain', range=[df['DEPTH'].max() + 10, df['DEPTH'].min() - 10]
    )

    def add_zone_markers(fig, df):
        if 'ZONES' in df.columns:
            tops_df = df[['DEPTH', 'ZONES']].dropna().reset_index()
            if not tops_df.empty and sum([1 for c in tops_df.ZONES.unique() if 'ZONE_' in c]) < 30:
                zone_tops_idx = [0] + [idx for idx, (i, j) in enumerate(
                    zip(tops_df['ZONES'], tops_df['ZONES'][1:]), 1) if i != j]
                zone_tops = tops_df.loc[zone_tops_idx, :]
                for tops in zone_tops.values:
                    fig.add_shape(
                        dict(type='line', x0=-5, y0=tops[1], x1=1e6, y1=tops[1]), row=1, col='all',
                        line=dict(color='#763F98', dash='dot', width=1.5)
                    )
                    fig.add_trace(
                        go.Scatter(
                            name=tops[2], x=[1.1, 1.1], y=[tops[1] - 1, tops[1] - 1],
                            text=tops[2], mode='text', textfont=dict(color='#763F98', size=12), textposition='top right'
                        ), row=1, col=2)
    add_zone_markers(fig, df)

    fig.update_layout(
        height=1000,
        autosize=True,
        showlegend=False,
        title={
            'text': '%s Logs' % well_name,
            'y': .99,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': 12
        },
        hovermode='y unified',
        dragmode='pan',
        modebar_remove=['lasso', 'select', 'autoscale'],
        modebar_add=['drawline', 'drawcircle', 'drawrect', 'eraseshape'],
        newshape_line_color='cyan', newshape_line_width=3,
        template='none',
        margin=dict(l=70, r=0, t=20, b=10),
        paper_bgcolor="#dadada",
        hoverlabel_bgcolor='#F3F3F3'
    )

    return fig
