import numpy as np
from collections import OrderedDict


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
    'BVW': "#56C0E4",
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
    'ROCK_FLAG_0': "hsl(30, 70%, 97%)",
    'DTC': '#FF0000',
    'DTS': '#0000FF',
}

# Define the trace definitions for the well log plot
# Each trace is defined with its properties such as track number, secondary y-axis, and style
TRACE_DEFS = OrderedDict(
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
    GAS_XOVER_BOTTOM=dict(
        track=3,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 0, 'fill': 'tonexty', 'fillcolor': 'rgba(255, 255, 0, 0.4)', 'hoverinfo': 'none'}
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
    ROCK_FLAG_0=dict(
        track=7,
        secondary_y=False,
        hide_xaxis=False,
        style={
            'line_width': 1, 'line_color': COLOR_DICT['ROCK_FLAG_0'], 'fill': 'tozerox',
            'fillcolor': COLOR_DICT['ROCK_FLAG_0']
        }
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
    GAS_XOVER_TOP=dict(
        track=3,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_width': 0, 'fill': 'tonexty', 'fillcolor': 'white', 'hoverinfo': 'none'}
    ),
    NPHI=dict(
        track=3,
        secondary_y=False,
        hide_xaxis=False,
        style={'line_dash': 'dot', 'line_width': 1, 'line_color': COLOR_DICT['NPHI']}
    ),
    RHOB=dict(
        track=3,
        secondary_y=True,
        hide_xaxis=False,
        style={'line_width': 1, 'line_color': COLOR_DICT['RHOB']}
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
        hide_xaxis=True,
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
        hide_xaxis=True,
        style={
            'line_color': COLOR_DICT['COAL_FLAG'], 'fill': 'tozerox',
            'fillcolor': 'rgba(0,0,0,1)', 'opacity': 1
        }
    )
)

# Centralize axis definitions for maintainability
font_size = 8
XAXIS_DEFS = {
    'GR': {
        'title': {'text': 'GR', 'font': {'color': COLOR_DICT['GR'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['GR'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85,
        'title_standoff': .1, 'dtick': 40, 'range': [0, 200], 'type': 'linear', 'zeroline': False
    },
    'RT': {
        'title': {'text': 'RT', 'font': {'color': COLOR_DICT['RT'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['RT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85,
        'title_standoff': .1, 'range': [np.log10(.2), np.log10(2000)], 'type': 'log',
        'tickmode': 'array', 'tickvals': np.geomspace(0.2, 2000, 5), 'tickangle': -90, 'minor_showgrid': True
    },
    'GAS_XOVER_BOTTOM': {
        'title': {'text': '', 'font': {'size': 1}}, 'tickfont': {'size': 1},
        'side': 'top', 'anchor': 'free', 'position': .88,
        'zeroline': False, 'range': [.45, -.15], 'type': 'linear', 'showgrid': False
    },
    'PHIT': {
        'title': {'text': 'PHIT', 'font': {'color': COLOR_DICT['PHIT'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['PHIT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .88, 'title_standoff': .1,
        'dtick': 0.1, 'range': [0, 0.5], 'type': 'linear', 'zeroline': False
    },
    'PERM': {
        'title': {'text': 'PERM', 'font': {'color': COLOR_DICT['PERM'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['PERM'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .91, 'title_standoff': .1,
        'range': [np.log10(0.1), np.log10(10000)], 'type': 'log', 'tickformat': 'd', 'tickangle': -90,
        'minor_showgrid': True
    },
    'SWT': {
        'title': {'text': 'SWT', 'font': {'color': COLOR_DICT['SWT'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['SWT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .88, 'title_standoff': .1,
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False
    },
    'ROCK_FLAG_0': {
        'title': {'text': 'ROCK_FLAG', 'font': {'size': font_size}}, 'tickfont': {'size': 1},
        'side': 'top', 'anchor': 'free', 'position': .85,
        'title_standoff': .1, 'range': [0.1, 0.2], 'type': 'linear', 'showgrid': False, 'zeroline': False
    },
    'VCLD': {
        'title': {'text': 'VOLUMETRIC', 'font': {'color': 'black', 'size': font_size}},
        'tickfont': {'color': 'black', 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1,
        'range': [0, 1], 'type': 'linear', 'zeroline': False
    },
    'GAS_XOVER_TOP': {
        'title': {'text': '', 'font': {'size': 1}}, 'tickfont': {'size': 1},
        'side': 'top', 'anchor': 'free', 'position': .88, 'overlaying': 'x3',
        'zeroline': False, 'range': [.45, -.15], 'type': 'linear', 'showgrid': False
    },
    'NPHI': {
        'title': {'text': 'NPHI', 'font': {'color': COLOR_DICT['NPHI'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['NPHI'], 'size': font_size}, 'zeroline': False,
        'side': 'top', 'anchor': 'free', 'position': .89, 'title_standoff': .1, 'overlaying': 'x3',
        'tickformat': ".2f", 'tick0': -.15, 'dtick': 0.12, 'range': [.45, -.15], 'type': 'linear', 'tickangle': -90
    },
    'RHOB': {
        'title': {'text': 'RHOB', 'font': {'color': COLOR_DICT['RHOB'], 'size': font_size}},
        'tickformat': ".2f", 'tick0': 1.95, 'dtick': 0.2, 'tickangle': -90,
        'tickfont': {'color': COLOR_DICT['RHOB'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'overlaying': 'x3',
        'title_standoff': .1, 'range': [1.95, 2.95], 'type': 'linear'
    },
    'PHIE': {
        'title': {'text': 'PHIE', 'font': {'color': COLOR_DICT['PHIE'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['PHIE'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1, 'overlaying': 'x4',
        'dtick': 0.1, 'range': [0, 0.5], 'type': 'linear', 'zeroline': False
    },
    'SWE': {
        'title': {'text': 'SWE', 'font': {'color': COLOR_DICT['SWE'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['SWE'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1, 'overlaying': 'x6',
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False
    },
    'CALI': {
        'title': {'text': 'CALI', 'font': {'color': COLOR_DICT['CALI'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['CALI'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .89, 'title_standoff': .1, 'overlaying': 'x1',
        'dtick': 6, 'range': [6, 24], 'type': 'linear', 'showgrid': False
    },
    'BS': {
        'title': {'text': 'BS', 'font': {'color': COLOR_DICT['BS'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['BS'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1, 'overlaying': 'x1',
        'dtick': 6, 'range': [6, 24], 'type': 'linear', 'showgrid': False
    },
    'BADHOLE': {
        'title': {'text': 'BADHOLE', 'font': {'color': COLOR_DICT['BADHOLE'], 'size': font_size}},
        'tickfont': {'size': 1},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1, 'overlaying': 'x1',
        'range': [0.1, 5], 'type': 'linear', 'showgrid': False, 'zeroline': False
    },
    'PEF': {
        'title': {'text': 'PEF', 'font': {'color': COLOR_DICT['PEF'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['PEF'], 'size': font_size}, 'zeroline': False,
        'side': 'top', 'anchor': 'free', 'position': .94, 'title_standoff': .1, 'overlaying': 'x3',
        'range': [-10, 10], 'type': 'linear', 'showgrid': False
    },
    'CPORE': {
        'title': {'text': 'CPORE', 'font': {'color': COLOR_DICT['CPORE'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['CPORE'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x4',
        'dtick': 0.1, 'range': [0, .5], 'type': 'linear', 'showgrid': False, 'zeroline': False
    },
    'CPERM': {
        'title': {'text': 'CPERM', 'font': {'color': COLOR_DICT['CPERM'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['CPERM'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x5',
        'range': [np.log10(0.1), np.log10(10000)], 'type': 'log', 'tickformat': 'd', 'tickangle': -90,
        'zeroline': False, 'showgrid': False
    },
    'CSAT': {
        'title': {'text': 'CSAT', 'font': {'color': COLOR_DICT['CSAT'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['CSAT'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .85, 'title_standoff': .1, 'overlaying': 'x6',
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False, 'showgrid': False
    },
    'SHF': {
        'title': {'text': 'SHF', 'font': {'color': COLOR_DICT['SHF'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['SHF'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1, 'overlaying': 'x6',
        'dtick': 0.2, 'range': [0, 1], 'type': 'linear', 'zeroline': False, 'showgrid': False
    },
    'BVW': {
        'title': {'text': 'BVW', 'font': {'color': COLOR_DICT['BVW'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['BVW'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1, 'overlaying': 'x4',
        'dtick': 0.1, 'range': [0, 0.5], 'type': 'linear', 'zeroline': False
    },
    'COAL_FLAG': {
        'range': [0.1, .2], 'type': 'linear', 'showgrid': False, 'zeroline': False,
    },
    'DTC': {
        'title': {'text': 'DTC', 'font': {'color': COLOR_DICT['DTC'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['DTC'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .92, 'title_standoff': .1,
        'dtick': 10, 'range': [0, 200], 'type': 'linear', 'zeroline': False
    },
    'DTS': {
        'title': {'text': 'DTS', 'font': {'color': COLOR_DICT['DTS'], 'size': font_size}},
        'tickfont': {'color': COLOR_DICT['DTS'], 'size': font_size},
        'side': 'top', 'anchor': 'free', 'position': .96, 'title_standoff': .1,
        'dtick': 10, 'range': [0, 200], 'type': 'linear', 'zeroline': False
    }
}
