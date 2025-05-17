import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

COLOR_DICT = {
    'BADHOLE': 'rgba(0, 0, 0, .25)',
    'BS': 'brown',
    'CALI': '#618F63',
    'COAL_FLAG': '#262626',
    'GR': '#0000FF',
    'NPHI': '#0000FF',
    'PERFORATED': '#E20000',
    'PERM': '#262626',
    'PHIT': '#262626',
    'PHIE': '#0000FF',
    'BVW': '#ADD8E6',
    'RHOB': '#FF0000',
    'RT': '#FF0000',
    'PEF': '#ba55d3',
    'SWT': '#262626',
    'SWE': '#0000FF',
    'VCLD': '#BFBFBF',
    'VCOAL': '#000000',
    'VGAS': '#FF0000',
    'VOIL': '#00FF00',
    'VHC': 'brown',
    'VSAND': '#F6F674',
    'VSHALE': '#A65628',
    'VSILT': '#FE9800',
    'VCALC': '#b0e0e6',
    'VDOLO': '#ba55d3',
    'CPORE': '#FF0000',
    'CPERM': '#FF0000',
    'CSAT': '#FF0000',
    'SHF': '#618F63',
}

# Centralized trace definitions: (column, display name, color key, track, style dict, secondary_y)
TRACE_DEFS = dict(
    GR=dict(
        track=1,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 1, 'line_color': COLOR_DICT['GR']}
    ),
    RT=dict(
        track=2,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_dash': 'dot', 'line_width': 1, 'line_color': COLOR_DICT['RT']}
    ),
    RHOB=dict(
        track=3,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 1, 'line_color': COLOR_DICT['RHOB']}
    ),
    PHIT=dict(
        track=4,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 1, 'line_color': COLOR_DICT['PHIT']}
    ),
    PERM=dict(
        track=5,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 1, 'line_color': COLOR_DICT['PERM']}
    ),
    SWT=dict(
        track=6,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 1, 'line_color': COLOR_DICT['SWT']}
    ),
    ROCK_FLAG_1=dict(
        track=7,
        secondary_y=False,
        hide_xaxis=False,
        style={'fill': 'tozerox', 'opacity': 1}
    ),
    VCLD=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=False,
        style={
            'line_width': .5, 'line_color': 'black', 'fill': 'tozerox', 'fillpattern_bgcolor': COLOR_DICT['VCLD'],
            'fillpattern_fgcolor': '#000000', 'fillpattern_fillmode': 'replace', 'fillpattern_shape': '-',
            'fillpattern_size': 2, 'fillpattern_solidity': 0.1, 'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    NPHI=dict(
        track=3,
        secondary_y=True,
        hide_xaxis=False,
        style={'line_dash': 'dot', 'line_width': 1, 'line_color': COLOR_DICT['NPHI']}
    ),
    PHIE=dict(
        track=4,
        secondary_y=True,
        hide_xaxis=False,
        style={'line_dash': 'dot', 'line_width': 1, 'line_color': COLOR_DICT['PHIE']}
    ),
    SWE=dict(
        track=6,
        secondary_y=True,
        hide_xaxis=False,
        style={'line_dash': 'dot', 'line_width': 1, 'line_color': COLOR_DICT['SWE']}
    ),
    CALI=dict(
        track=1,
        secondary_y=False,
        hide_xaxis=False,
        style={
            'line_dash': 'dot', 'line_width': 1, 'line_color': COLOR_DICT['CALI'], 'fill': 'tozerox',
            'fillcolor': 'rgba(165, 42, 42, .15)'
        }
    ),
    BS=dict(
        track=1,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_dash': 'dashdot', 'line_width': 1, 'line_color': COLOR_DICT['BS']}
    ),
    BADHOLE=dict(
        track=1,
        secondary_y=False,
        hide_xaxis=False,
        style={'fill': 'tozerox', 'fillcolor': 'rgba(0, 0, 0, .25)'}
    ),
    VSILT=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'line_color': 'black', 'fill': 'tonextx', 'fillpattern_bgcolor': COLOR_DICT['VSILT'],
            'fillpattern_fgcolor': '#000000', 'fillpattern_fillmode': 'replace', 'fillpattern_shape': '.',
            'fillpattern_size': 3, 'fillpattern_solidity': 0.1, 'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    VSAND=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'line_color': 'black', 'fill': 'tonextx', 'fillpattern_bgcolor': COLOR_DICT['VSAND'],
            'fillpattern_fgcolor': '#000000', 'fillpattern_fillmode': 'replace', 'fillpattern_shape': '.',
            'fillpattern_size': 3, 'fillpattern_solidity': 0.1, 'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    VCALC=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'line_color': 'black', 'fill': 'tonextx', 'fillpattern_bgcolor': COLOR_DICT['VCALC'],
            'fillpattern_fgcolor': '#000000', 'fillpattern_fillmode': 'replace', 'fillpattern_shape': '.',
            'fillpattern_size': 3, 'fillpattern_solidity': 0.1, 'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    VDOLO=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'line_color': 'black', 'fill': 'tonextx', 'fillpattern_bgcolor': COLOR_DICT['VDOLO'],
            'fillpattern_fgcolor': '#000000', 'fillpattern_fillmode': 'replace', 'fillpattern_shape': '-',
            'fillpattern_size': 3, 'fillpattern_solidity': 0.3, 'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    VHC=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'fill': 'tonextx', 'fillcolor': COLOR_DICT['VHC'],
            'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    VGAS=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'fill': 'tonextx', 'fillcolor': COLOR_DICT['VGAS'],
            'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    VOIL=dict(
        track=8,
        secondary_y=False,
        hide_xaxis=True,
        style={
            'line_width': .5, 'fill': 'tonextx', 'fillcolor': COLOR_DICT['VOIL'],
            'stackgroup': 'litho', 'orientation': 'h'
        }
    ),
    PEF=dict(
        track=3,
        secondary_y=True,
        hide_xaxis=False,
        style={'line_dash': 'dashdot', 'line_width': .75, 'line_color': COLOR_DICT['PEF']}
    ),
    CPORE=dict(
        track=4,
        secondary_y=True,
        hide_xaxis=False,
        style={'mode': 'markers', 'marker': dict(color=COLOR_DICT['CPORE'], size=3)}
    ),
    CPERM=dict(
        track=5,
        secondary_y=True,
        hide_xaxis=False,
        style={'mode': 'markers', 'marker': dict(color=COLOR_DICT['CPERM'], size=3)}
    ),
    CSAT=dict(
        track=6,
        secondary_y=True,
        hide_xaxis=False,
        style={'mode': 'markers', 'marker': dict(color=COLOR_DICT['CSAT'], size=3)}
    ),
    SHF=dict(
        track=6,
        secondary_y=True,
        hide_xaxis=False,
        style={'line_dash': 'dashdot', 'line_width': 1, 'line_color': COLOR_DICT['SHF']}
    ),
    BVW=dict(
        track=4,
        secondary_y=True,
        hide_xaxis=False,
        style={
            'line_dash': 'dashdot', 'line_width': .5, 'fill': 'tozerox', 'line_color': COLOR_DICT['BVW'],
            'fillcolor': 'rgba(98, 180, 207, .5)'
        }
    ),
    COAL_FLAG=dict(
        track=8,
        secondary_y=True,
        hide_xaxis=False,
        style={
            'line_color': COLOR_DICT['COAL_FLAG'], 'fill': 'tozerox',
            'fillcolor': 'rgba(0,0,0,1)', 'opacity': 1
        }
    ),
    ZONES=dict(
        track=8,
        secondary_y=True,
        hide_xaxis=False,
        style={
            'line_color': COLOR_DICT['COAL_FLAG'], 'fill': 'tozerox',
            'fillcolor': 'rgba(0,0,0,1)', 'opacity': 1
        }
    )
)

# Centralize axis definitions for maintainability
font_size = 8
XAXIS_DEFS = {
    'xaxis1': {
        'title': 'GR',
        'titlefont': {'color': COLOR_DICT['GR'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['GR'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .88,
        'title_standoff': .1, 'dtick': 40, 'range': [0, 200], 'type': 'linear', 'zeroline': False
    },
    'xaxis2': {
        'title': 'RT',
        'titlefont': {'color': COLOR_DICT['RT'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['RT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85,
        'title_standoff': .1, 'range': [np.log10(.2), np.log10(2000)], 'type': 'log',
        'tickmode': 'array', 'tickvals': np.geomspace(0.2, 2000, 5), 'tickangle': -90, 'minor_showgrid': True
    },
    'xaxis3': {
        'title': 'RHOB',
        'titlefont': {'color': COLOR_DICT['RHOB'], 'size': font_size},
        'tickformat': ".2f", 'tick0': 1.95, 'dtick': 0.2, 'tickangle': -90,
        'tickfont': {'color': COLOR_DICT['RHOB'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .89,
        'title_standoff': .1, 'range': [1.95, 2.95], 'type': 'linear'
    },
    'xaxis4': {
        'title': 'PHIT',
        'titlefont': {'color': COLOR_DICT['PHIT'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['PHIT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .88, 'title_standoff': .1,
        'dtick': 0.1, 'range': [0, 0.5], 'type': 'linear', 'zeroline': False
    },
    'xaxis5': {
        'title': 'PERM',
        'titlefont': {'color': COLOR_DICT['PERM'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['PERM'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .91, 'title_standoff': .1,
        'range': [np.log10(0.1), np.log10(10000)], 'type': 'log', 'tickformat': 'd', 'tickangle': -90,
        'minor_showgrid': True
    },
    'xaxis6': {
        'title': 'SWT',
        'titlefont': {'color': COLOR_DICT['SWT'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['SWT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .88, 'title_standoff': .1,
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False
    },
    'xaxis7': {
        'title': '', 'titlefont_size': 1, 'tickfont_size': 1, 'side': 'top', 'anchor': 'free', 'position': .88,
        'title_standoff': .1, 'range': [0.1, 0.2], 'type': 'linear', 'showgrid': False, 'zeroline': False
    },
    'xaxis8': {
        'title': 'LITHOLOGY',
        'titlefont': {'color': 'black', 'size': font_size},
        'tickfont': {'color': 'black', 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1,
        'range': [0, 1], 'type': 'linear', 'zeroline': False
    },
    'xaxis9': {
        'title': 'NPHI',
        'titlefont': {'color': COLOR_DICT['NPHI'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['NPHI'], 'size': font_size}, 'zeroline': False,
        'side': 'top', 'anchor': 'free', 'position': .94, 'title_standoff': .1, 'overlaying': 'x3',
        'tickformat': ".2f", 'tick0': -.15, 'dtick': 0.12, 'range': [.45, -.15], 'type': 'linear', 'tickangle': -90
    },
    'xaxis10': {
        'title': 'PHIE',
        'titlefont': {'color': COLOR_DICT['PHIE'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['PHIE'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1, 'overlaying': 'x4',
        'dtick': 0.1, 'range': [0, 0.5], 'type': 'linear', 'zeroline': False
    },
    'xaxis11': {
        'title': 'SWE',
        'titlefont': {'color': COLOR_DICT['SWE'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['SWE'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1, 'overlaying': 'x6',
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False
    },
    'xaxis12': {
        'title': 'CALI',
        'titlefont': {'color': COLOR_DICT['CALI'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['CALI'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1, 'overlaying': 'x1',
        'dtick': 6, 'range': [6, 24], 'type': 'linear', 'showgrid': False
    },
    'xaxis13': {
        'title': 'BS',
        'titlefont': {'color': COLOR_DICT['BS'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['BS'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1, 'overlaying': 'x1',
        'dtick': 6, 'range': [6, 24], 'type': 'linear', 'showgrid': False
    },
    'xaxis14': {
        'title': 'BADHOLE',
        'titlefont': {'color': COLOR_DICT['BADHOLE'], 'size': font_size},
        'tickfont': {'size': 1},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x1',
        'range': [0.1, 5], 'type': 'linear', 'showgrid': False, 'zeroline': False
    },
    'xaxis22': {
        'title': 'PEF',
        'titlefont': {'color': COLOR_DICT['PEF'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['PEF'], 'size': font_size}, 'zeroline': False,
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x3',
        'range': [-10, 10], 'type': 'linear', 'showgrid': False
    },
    'xaxis23': {
        'title': 'CPORE',
        'titlefont': {'color': COLOR_DICT['CPORE'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['CPORE'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x4',
        'dtick': 0.1, 'range': [0, .5], 'type': 'linear', 'showgrid': False, 'zeroline': False
    },
    'xaxis24': {
        'title': 'CPERM',
        'titlefont': {'color': COLOR_DICT['CPERM'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['CPERM'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x5',
        'range': [np.log10(0.1), np.log10(10000)], 'type': 'log', 'tickformat': 'd', 'tickangle': -90,
        'zeroline': False, 'showgrid': False
    },
    'xaxis25': {
        'title': 'CSAT',
        'titlefont': {'color': COLOR_DICT['CSAT'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['CSAT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x6',
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False, 'showgrid': False
    },
    'xaxis26': {
        'title': 'SHF',
        'titlefont': {'color': COLOR_DICT['SHF'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['SHF'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1, 'overlaying': 'x6',
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False, 'showgrid': False
    },
    'xaxis27': {
        'title': 'BVW',
        'titlefont': {'color': COLOR_DICT['BVW'], 'size': font_size},
        'tickfont': {'color': COLOR_DICT['BVW'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1, 'overlaying': 'x4',
        'dtick': 0.1, 'range': [0, 0.5], 'type': 'linear', 'zeroline': False
    },
    # COAL_FLAG axes
    'xaxis29': {
        'title': '', 'titlefont': {'color': COLOR_DICT['COAL_FLAG'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .97, 'title_standoff': .1, 'overlaying': 'x4',
        'tick0': 0, 'dtick': 1, 'range': [0.1, .2], 'type': 'linear', 'tickfont': {'size': 1}
    },
    'xaxis30': {
        'title': '', 'titlefont': {'color': COLOR_DICT['COAL_FLAG'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .97, 'title_standoff': .1, 'overlaying': 'x5',
        'tick0': 0, 'dtick': 1, 'range': [0.1, .2], 'type': 'linear', 'tickfont': {'size': 1}
    },
    'xaxis31': {
        'title': '', 'titlefont': {'color': COLOR_DICT['COAL_FLAG'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .97, 'title_standoff': .1, 'overlaying': 'x6',
        'tick0': 0, 'dtick': 1, 'range': [0.1, .2], 'type': 'linear', 'tickfont': {'size': 1}
    },
    'xaxis32': {
        'title': '', 'titlefont': {'color': COLOR_DICT['COAL_FLAG'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .97, 'title_standoff': .1, 'overlaying': 'x7',
        'tick0': 0, 'dtick': 1, 'range': [0.1, .2], 'type': 'linear', 'tickfont': {'size': 1}
    },
    'xaxis33': {
        'title': '', 'titlefont': {'color': COLOR_DICT['COAL_FLAG'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .97, 'title_standoff': .1, 'overlaying': 'x8',
        'tick0': 0, 'dtick': 1, 'range': [0.1, .2], 'type': 'linear', 'tickfont': {'size': 1}
    },
}

def plotly_log(well_data, depth_uom="", trace_defs=TRACE_DEFS, xaxis_defs=XAXIS_DEFS):  # noqa
    """Plot well logs using Plotly.

    Args:
        well_data (pandas.Dataframe): Pandas dataframe containing well log data.

    Returns:
        plotly.graph_objects.Figure: Return well plot.
    """
    track = 8
    df = well_data.copy()
    index = df.DEPTH

    # Ensure all required columns exist (fill with NaN if missing)
    for k in COLOR_DICT:
        if k not in df.columns:
            df[k] = np.nan

    # One-hot encode ROCK_FLAG if present
    if 'ROCK_FLAG' in df.columns:
        # Add ROCK_FLAG colors
        no_of_rock_flags = df['ROCK_FLAG'].nunique() + 1
        df['ROCK_FLAG'] = df['ROCK_FLAG'].fillna(no_of_rock_flags)
        for i in df['ROCK_FLAG'].unique():
            lightness = 100 - (int(i) / no_of_rock_flags * 100)
            COLOR_DICT[f'ROCK_FLAG_{i}'] = f'hsl(30, 70%, {lightness}%)'

        df['ROCK_FLAG'] = df['ROCK_FLAG'].astype(int).astype('category')
        df = df.drop(columns=[c for c in df.columns if 'ROCK_FLAG_' in c])
        df = pd.get_dummies(df, columns=['ROCK_FLAG'], prefix='ROCK_FLAG', dtype=int)

    fig = make_subplots(
        rows=1, cols=track, shared_yaxes=True, horizontal_spacing=0.015,
        column_widths=[1, 1, 1, 1, 1, 1, .15, 1],
        specs=[list([{'secondary_y': True}] * track)]
    )

    # --- Helper to add traces from trace_defs ---
    def add_defined_traces(fig, df, index):
        for i, (col, trace_def) in enumerate(trace_defs.items()):
            # Skip special cases handled separately
            if col.startswith('ROCK_FLAG_') or col in ['COAL_FLAG', 'ZONES']:
                continue
            style = trace_def.get('style', {})
            trace_args = dict(
                x=df[col],
                y=index,
                name=col
            )
            trace_args.update(style)
            fig.add_trace(
                go.Scatter(**trace_args),
                row=1,
                col=trace_def['track'],
                secondary_y=trace_def.get('secondary_y', False)
            )
            if not trace_def.get('hide_xaxis', True) and i >= track:
                fig.data[i - 1].update(xaxis=f'x{i + 1}')
    add_defined_traces(fig, df, index)

    # --- COAL_FLAG traces (special style, always on tracks 4-8, secondary_y=True) ---
    for c in [4, 5, 6, 7, 8]:
        fig.add_trace(
            go.Scatter(x=df['COAL_FLAG'], y=index, name='', line_color=COLOR_DICT['COAL_FLAG'],
                       fill='tozerox', fillcolor='rgba(0,0,0,1)', opacity=1),
            row=1, col=c, secondary_y=True)

    # --- ROCK_FLAG_X traces (special style, always on track 7, secondary_y=False) ---
    rock_flag_columns = [col for col in df.columns if col.startswith('ROCK_FLAG_')]
    for feature in rock_flag_columns:
        fig.add_trace(
            go.Scatter(x=df[feature], y=index, line_color=COLOR_DICT[feature], name=feature,
                       fill='tozerox', fillcolor=COLOR_DICT.get(feature, '#000000')),
            row=1, col=7, secondary_y=False)

    # --- Layout dict ---
    layout_dict = {k: v for k, v in xaxis_defs.items()}
    layout_dict['yaxis'] = {'domain': [0, .84], 'title': f'DEPTH ({depth_uom})'}
    for i in range(2, track * 2 + 1):
        layout_dict[f'yaxis{i}'] = {
            'domain': [0, .84],
            'visible': False if i % 2 == 0 else True,
            'showgrid': False
        }
    fig.update_layout(**layout_dict)
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(matches='y', constrain='domain', autorange='reversed')

    # --- Helper for zone markers (ZONES) ---
    def add_zone_markers(fig, df):
        if 'ZONES' in df.columns:
            tops_df = df[['DEPTH', 'ZONES']].dropna().reset_index()
            if not tops_df.empty and sum([1 for c in tops_df.ZONES.unique() if 'SAND_' in c]) < 30:
                zone_tops_idx = [0] + [idx for idx, (i, j) in enumerate(
                    zip(tops_df['ZONES'], tops_df['ZONES'][1:]), 1) if i != j]
                zone_tops = tops_df.loc[zone_tops_idx, :]
                for tops in zone_tops.values:
                    fig.add_shape(
                        dict(type='line', x0=-5, y0=tops[1], x1=150, y1=tops[1]), row=1, col='all',
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
            'text': '%s Logs' % df.WELL_NAME.dropna().unique()[0],
            'y': .99,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': font_size + 4
        },
        hovermode='y unified',
        dragmode='pan',
        modebar_remove=['lasso', 'select', 'autoscale'],
        modebar_add=['drawline', 'drawcircle', 'drawrect', 'eraseshape'],
        newshape_line_color='cyan', newshape_line_width=3,
        template='none',
        margin=dict(l=70, r=0, t=20, b=10),
        paper_bgcolor='#e6f2ff',
        hoverlabel_bgcolor='#F3F3F3'
    )

    return fig
