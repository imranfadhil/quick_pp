import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt

from quick_pp.utils import line_intersection

plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update(
    {
        'axes.labelsize': 10,
        'xtick.labelsize': 10,
        'legend.fontsize': 'small'
    }
)

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
    'VCLW': '#BFBFBF',
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
    'ROCK_FLAG_1': '#F6F674',
    'ROCK_FLAG_2': 'gold',
    'ROCK_FLAG_3': '#FE9800',
    'ROCK_FLAG_4': 'darkgray',
}


def neutron_density_xplot(nphi, rhob,
                          dry_sand_point: tuple,
                          dry_silt_point: tuple,
                          dry_clay_point: tuple,
                          fluid_point: tuple,
                          wet_clay_point: tuple, **kwargs):
    """Neutron-Density crossplot with lithology lines based on specified end points.

    Args:
        nphi (float): Neutron porosity log.
        rhob (float): Bulk density log.
        dry_sand_point (tuple): Neutron porosity and bulk density of dry sand point.
        dry_silt_point (tuple): Neutron porosity and bulk density of dry silt point.
        dry_clay_point (tuple): Neutron porosity and bulk density of dry clay point.
        fluid_point (tuple): Neutron porosity and bulk density of fluid point.
        wet_clay_point (tuple): Neutron porosity and bulk density of wet clay point.

    Returns:
        matplotlib.pyplot.Figure: Neutron porosity and bulk density cross plot.
    """
    A = dry_sand_point
    B = dry_silt_point
    C = dry_clay_point
    D = fluid_point
    E = list(zip(nphi, rhob))
    # Plotting the NPHI-RHOB crossplot
    projected_pt = []
    for i in range(len(nphi)):
        projected_pt.append(line_intersection((A, C), (D, E[i])))
    rockline_from_pt = (A, C)
    sandline_from_pt = (D, A)
    siltline_from_pt = (D, B)
    clayline_from_pt = (D, wet_clay_point)

    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title('NPHI-RHOB Crossplot')
    ax.scatter(nphi, rhob, c=np.arange(0, len(nphi)), cmap='rainbow')
    ax.plot(*zip(*sandline_from_pt), label='Sand Line', color='blue')
    ax.plot(*zip(*siltline_from_pt), label='Silt Line', color='green')
    ax.plot(*zip(*clayline_from_pt), label='Clay Line', color='gray')
    ax.plot(*zip(*rockline_from_pt), label='Rock Line', color='black')
    ax.scatter(*zip(*projected_pt), label='Projected Line', color='purple')
    # ax.scatter(NPHI_reg, RHOB_reg, label='Wet Clay Line', color='gray')
    ax.scatter(dry_sand_point[0], dry_sand_point[1], label='Dry Sand Point', color='yellow')
    ax.scatter(dry_silt_point[0], dry_silt_point[1], label='Dry Silt Point', color='orange')
    ax.scatter(dry_clay_point[0], dry_clay_point[1], label='Dry Clay Point', color='black')
    ax.scatter(wet_clay_point[0], wet_clay_point[1], label='Wet Clay Point', color='gray')
    ax.scatter(fluid_point[0], fluid_point[1], label='Fluid Point', color='blue')
    ax.set_ylim(3, 0)
    ax.set_ylabel('RHOB')
    ax.set_xlim(-0.15, 1)
    ax.set_xlabel('NPHI')
    ax.legend(loc="upper left", prop={'size': 9})
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='gray')
    fig.tight_layout()

    return fig


def sonic_density_xplot():
    """ TODO: Implement sonic density crossplot
    """
    pass


def sonic_neutron_xplot():
    """ TODO: Implement sonic neutron crossplot
    """
    pass


def plotly_log(well_data, depth_uom=""):  # noqa
    """Plot well logs using Plotly.

    Args:
        well_data (pandas.Dataframe): Pandas dataframe containing well log data.

    Returns:
        plotly.graph_objects.Figure: Return well plot.
    """
    track = 8
    df = well_data.copy()
    index = df.DEPTH

    # Create one hot for ROCK_FLAG
    if 'ROCK_FLAG' in df.columns:
        df['ROCK_FLAG'] = df['ROCK_FLAG'].astype('category')
        df = pd.get_dummies(df, columns=['ROCK_FLAG'], prefix='ROCK_FLAG', dtype=int)

    for k, v in COLOR_DICT.items():
        if k not in df.columns:
            df[k] = np.nan

    fig = make_subplots(rows=1, cols=track, shared_yaxes=True, horizontal_spacing=0.015,
                        column_widths=[1, 1, 1, 1, 1, 1, .15, 1],
                        specs=[list([{'secondary_y': True}] * track)])

    i = 0
    # Add GR trace #1.
    fig.add_trace(go.Scatter(x=df['GR'], y=index, name='GR', line_color=COLOR_DICT['GR']),
                  row=1, col=1)

    i += 1
    # Add RESISTIVITY trace #2.
    fig.add_trace(go.Scatter(x=df['RT'], y=index, name='RT', line_color=COLOR_DICT['RT'], line_dash='dot'),
                  row=1, col=2)

    i += 1
    # Add RHOB trace #3.
    fig.add_trace(go.Scatter(x=df['RHOB'], y=index, name='RHOB', line_color=COLOR_DICT['RHOB']),
                  row=1, col=3, secondary_y=False)

    i += 1
    # Add PHIT trace #4.
    fig.add_trace(go.Scatter(x=df['PHIT'], y=index, name='PHIT', line_color=COLOR_DICT['PHIT']),
                  row=1, col=4, secondary_y=False)

    i += 1
    # Add PERM trace #5.
    fig.add_trace(go.Scatter(x=df['PERM'], y=index, name='PERM', line_color=COLOR_DICT['PERM']),
                  row=1, col=5, secondary_y=False)

    i += 1
    # Add SWT trace #6.
    fig.add_trace(go.Scatter(x=df['SWT'], y=index, name='SWT', line_color=COLOR_DICT['SWT']),
                  row=1, col=6, secondary_y=False)

    i += 1
    # Add ROCK_FLAG trace #7.
    fig.add_trace(go.Scatter(x=df['ROCK_FLAG_1'], y=index, name='ROCK_FLAG', line_color=COLOR_DICT['ROCK_FLAG_1'],
                             fill='tozerox', fillcolor=COLOR_DICT['ROCK_FLAG_1'], opacity=1),
                  row=1, col=7, secondary_y=False)

    i += 1
    # Add VCLW trace #8.
    fig.add_trace(go.Scatter(x=df['VCLW'], y=index, name='VCLW', line_color='#000000', line_width=1,
                  fill='tozerox',
                  fillpattern_bgcolor=COLOR_DICT['VCLW'],
                  fillpattern_fgcolor='#000000',
                  fillpattern_fillmode='replace',
                  fillpattern_shape='-',
                  fillpattern_size=2,
                  fillpattern_solidity=0.1,
                  stackgroup='litho',
                  orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add NPHI trace #9.
    fig.add_trace(go.Scatter(x=df['NPHI'], y=index, name='NPHI', line_color=COLOR_DICT['NPHI'], line_dash='dot'),
                  row=1, col=3, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add PHIE
    fig.add_trace(go.Scatter(x=df['PHIE'], y=index, name='PHIE', line_color=COLOR_DICT['PHIE'], line_dash='dot'),
                  row=1, col=4, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add SWE
    fig.add_trace(go.Scatter(x=df['SWE'], y=index, name='SWE', line_color=COLOR_DICT['SWE'], line_dash='dot'),
                  row=1, col=6, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add CALI
    fig.add_trace(go.Scatter(x=df['CALI'], y=index, name='CALI', line_color=COLOR_DICT['CALI'], line_dash='dot',
                  fill='tozerox', fillcolor='rgba(165, 42, 42, .15)'),
                  row=1, col=1, secondary_y=False)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add BITSIZE
    fig.add_trace(go.Scatter(x=df['BS'], y=index, name='BS', line_color=COLOR_DICT['BS'], line_dash='dashdot'),
                  row=1, col=1, secondary_y=False)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add BADHOLE
    fig.add_trace(go.Scatter(x=df['BADHOLE'], y=index, name='BADHOLE', line_color=COLOR_DICT['BADHOLE'],
                  fill='tozerox', fillcolor='rgba(0, 0, 0, .25)'),
                  row=1, col=1, secondary_y=False)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add VSILT
    fig.add_trace(go.Scatter(x=df['VSILT'], y=index, name='VSILT', line_color='#000000', line_width=1,
                  fill='tonextx',
                  fillpattern_bgcolor=COLOR_DICT['VSILT'],
                  fillpattern_fgcolor='#000000',
                  fillpattern_fillmode='replace',
                  fillpattern_shape='.',
                  fillpattern_size=3,
                  fillpattern_solidity=0.1,
                  stackgroup='litho',
                  orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VSAND
    fig.add_trace(go.Scatter(x=df['VSAND'], y=index, name='VSAND', line_color='#000000', line_width=1,
                  fill='tonextx',
                  fillpattern_bgcolor=COLOR_DICT['VSAND'],
                  fillpattern_fgcolor='#000000',
                  fillpattern_fillmode='replace',
                  fillpattern_shape='.',
                  fillpattern_size=3,
                  fillpattern_solidity=0.1,
                  stackgroup='litho',
                  orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VCALC
    fig.add_trace(go.Scatter(x=df['VCALC'], y=index, name='VCALC', line_color='#000000', line_width=1,
                  fill='tonextx',
                  fillpattern_bgcolor=COLOR_DICT['VCALC'],
                  fillpattern_fgcolor='#000000',
                  fillpattern_fillmode='replace',
                  fillpattern_shape='x',
                  fillpattern_size=3,
                  fillpattern_solidity=0.1,
                  stackgroup='litho',
                  orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VDOLO
    fig.add_trace(go.Scatter(x=df['VDOLO'], y=index, name='VDOLO', line_color='#000000', line_width=1,
                  fill='tonextx',
                  fillpattern_bgcolor=COLOR_DICT['VDOLO'],
                  fillpattern_fgcolor='#000000',
                  fillpattern_fillmode='replace',
                  fillpattern_shape='-',
                  fillpattern_size=3,
                  fillpattern_solidity=0.3,
                  stackgroup='litho',
                  orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VHC
    fig.add_trace(go.Scatter(x=df['VHC'], y=index, name='VHC', line_color='#000000', line_width=1,
                  fill='tonextx', fillcolor=COLOR_DICT['VHC'], stackgroup='litho', orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VGAS
    fig.add_trace(go.Scatter(x=df['VGAS'], y=index, name='VGAS', line_color='#000000', line_width=1,
                  fill='tonextx', fillcolor=COLOR_DICT['VGAS'], stackgroup='litho', orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VOIL
    fig.add_trace(go.Scatter(x=df['VOIL'], y=index, name='VOIL', line_color='#000000', line_width=1,
                  fill='tonextx', fillcolor=COLOR_DICT['VOIL'], stackgroup='litho', orientation='h'),
                  row=1, col=8, secondary_y=False)

    i += 1
    # Add VSHALE
    fig.add_trace(go.Scatter(x=df['VSHALE'], y=index, name='VSHALE', line_color=COLOR_DICT['VSHALE']),
                  row=1, col=8, secondary_y=True)

    i += 1
    # Add PEF
    fig.add_trace(go.Scatter(x=df['PEF'], y=index, name='PEF', line_color=COLOR_DICT['PEF'], line_dash='dashdot',
                             line_width=.75),
                  row=1, col=3, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add CPORE
    fig.add_trace(go.Scatter(x=df['CPORE'], y=index, name='CPORE', mode='markers',
                             marker=dict(color=COLOR_DICT['CPORE'], size=7)),
                  row=1, col=4, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add CPERM
    fig.add_trace(go.Scatter(x=df['CPERM'], y=index, name='CPERM', mode='markers',
                             marker=dict(color=COLOR_DICT['CPERM'], size=7)),
                  row=1, col=5, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add CSAT
    fig.add_trace(go.Scatter(x=df['CSAT'], y=index, name='CSAT', mode='markers',
                             marker=dict(color=COLOR_DICT['CSAT'], size=7)),
                  row=1, col=6, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add SHF
    fig.add_trace(go.Scatter(x=df['SHF'], y=index, name='SHF', line_color=COLOR_DICT['SHF'], line_dash='dashdot'),
                  row=1, col=6, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add BVW
    fig.add_trace(go.Scatter(x=df['BVW'], y=index, name='BVW',
                             line_color=COLOR_DICT['BVW'], line_dash='dashdot', line_width=.5,
                             fill='tozerox', fillcolor='rgba(98, 180, 207, .5)'),
                  row=1, col=4, secondary_y=True)
    fig.data[i].update(xaxis=f'x{i + 1}')

    i += 1
    # Add ROCK_FLAG_2
    fig.add_trace(go.Scatter(x=df['ROCK_FLAG_2'], y=index, name='ROCK_FLAG_2', line_color=COLOR_DICT['ROCK_FLAG_2'],
                             line_width=1, fill='tozerox', fillcolor=COLOR_DICT['ROCK_FLAG_2'], opacity=1),
                  row=1, col=7, secondary_y=False)

    i += 1
    # Add ROCK_FLAG_3
    fig.add_trace(go.Scatter(x=df['ROCK_FLAG_3'], y=index, name='ROCK_FLAG_3', line_color=COLOR_DICT['ROCK_FLAG_3'],
                             line_width=1, fill='tozerox', fillcolor=COLOR_DICT['ROCK_FLAG_3'], opacity=1),
                  row=1, col=7, secondary_y=False)

    i += 1
    # Add ROCK_FLAG_4
    fig.add_trace(go.Scatter(x=df['ROCK_FLAG_4'], y=index, name='ROCK_FLAG_4', line_color=COLOR_DICT['ROCK_FLAG_4'],
                             line_width=1, fill='tozerox', fillcolor=COLOR_DICT['ROCK_FLAG_4'], opacity=1),
                  row=1, col=7, secondary_y=False)

    i += 1
    # Add COAL_FLAG trace.
    for c in [4, 5, 6, 7, 8]:
        fig.add_trace(go.Scatter(x=df['COAL_FLAG'], y=index, name='', line_color=COLOR_DICT['COAL_FLAG'],
                      fill='tozerox', fillcolor='rgba(0,0,0,1)', opacity=1),
                      row=1, col=c, secondary_y=True)
        fig.data[i].update(xaxis=f'x{i + 1}')
        i += 1

    font_size = 8
    fig.update_layout(
        xaxis1=dict(title='GR', titlefont=dict(color=COLOR_DICT['GR'], size=font_size),
                    tickfont=dict(color=COLOR_DICT['GR'], size=font_size), side='top', anchor='free', position=.88,
                    title_standoff=.1, dtick=40, range=[0, 200], type='linear', zeroline=False),
        xaxis2=dict(title='RT', titlefont=dict(color=COLOR_DICT['RT'], size=font_size),
                    tickfont=dict(color=COLOR_DICT['RT'], size=font_size), side='top', anchor='free', position=.85,
                    title_standoff=.1, range=[np.log10(.2), np.log10(2000)], type='log',
                    tickmode='array', tickvals=np.geomspace(0.2, 2000, 5), tickangle=-90, minor_showgrid=True),
        xaxis3=dict(title='RHOB', titlefont=dict(color=COLOR_DICT['RHOB'], size=font_size),
                    tickformat=".2f", tick0=1.85, dtick=0.2, tickangle=-90,
                    tickfont=dict(color=COLOR_DICT['RHOB'], size=font_size), side='top', anchor='free', position=.89,
                    title_standoff=.1, range=[1.85, 2.85], type='linear'),
        xaxis4=dict(title='PHIT', titlefont=dict(color=COLOR_DICT['PHIT'], size=font_size),
                    tickfont=dict(color=COLOR_DICT['PHIT'], size=font_size),
                    side='top', anchor='free', position=.88, title_standoff=.1,
                    dtick=0.1, range=[0, 0.5], type='linear', zeroline=False),
        xaxis5=dict(title='PERM', titlefont=dict(color=COLOR_DICT['PERM'], size=font_size),
                    tickfont=dict(color=COLOR_DICT['PERM'], size=font_size),
                    side='top', anchor='free', position=.91, title_standoff=.1,
                    range=[np.log10(0.1), np.log10(10000)], type='log', tickformat='d', tickangle=-90,
                    minor_showgrid=True),
        xaxis6=dict(title='SWT', titlefont=dict(color=COLOR_DICT['SWT'], size=font_size),
                    tickfont=dict(color=COLOR_DICT['SWT'], size=font_size),
                    side='top', anchor='free', position=.88, title_standoff=.1,
                    dtick=0.2, range=[0, 1], type='linear', zeroline=False),
        xaxis7=dict(title='', titlefont_size=1, tickfont_size=1, side='top', anchor='free', position=.88,
                    title_standoff=.1, range=[0.1, 0.2], type='linear', showgrid=False, zeroline=False),
        xaxis8=dict(title='LITHOLOGY', titlefont=dict(color='black', size=font_size),
                    tickfont=dict(color='black', size=font_size),
                    side='top', anchor='free', position=.85, title_standoff=.1,
                    range=[0, 1], type='linear', zeroline=False),
        xaxis9=dict(title='NPHI', titlefont=dict(color=COLOR_DICT['NPHI'], size=font_size),
                    tickfont=dict(color=COLOR_DICT['NPHI'], size=font_size), zeroline=False,
                    side='top', anchor='free', position=.94, title_standoff=.1, overlaying='x3',
                    tickformat=".2f", tick0=-.15, dtick=0.12, range=[.45, -.15], type='linear', tickangle=-90),
        xaxis10=dict(title='PHIE', titlefont=dict(color=COLOR_DICT['PHIE'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['PHIE'], size=font_size),
                     side='top', anchor='free', position=.92, title_standoff=.1, overlaying='x4',
                     dtick=0.1, range=[0, 0.5], type='linear', zeroline=False),
        xaxis11=dict(title='SWE', titlefont=dict(color=COLOR_DICT['SWE'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['SWE'], size=font_size),
                     side='top', anchor='free', position=.92, title_standoff=.1, overlaying='x6',
                     dtick=0.2, range=[0, 1], type='linear', zeroline=False),
        xaxis12=dict(title='CALI', titlefont=dict(color=COLOR_DICT['CALI'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['CALI'], size=font_size),
                     side='top', anchor='free', position=.92, title_standoff=.1, overlaying='x1',
                     dtick=6, range=[6, 24], type='linear', showgrid=False),
        xaxis13=dict(title='BS', titlefont=dict(color=COLOR_DICT['BS'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['BS'], size=font_size),
                     side='top', anchor='free', position=.96, title_standoff=.1, overlaying='x1',
                     dtick=6, range=[6, 24], type='linear', showgrid=False),
        xaxis14=dict(title='BADHOLE', titlefont=dict(color=COLOR_DICT['BADHOLE'], size=font_size),
                     tickfont=dict(size=1),
                     side='top', anchor='free', position=.85, title_standoff=.1, overlaying='x1',
                     range=[0.1, 5], type='linear', showgrid=False, zeroline=False,),
        xaxis23=dict(title='PEF', titlefont=dict(color=COLOR_DICT['PEF'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['PEF'], size=font_size), zeroline=False,
                     side='top', anchor='free', position=.85, title_standoff=.1, overlaying='x3',
                     range=[-10, 10], type='linear', showgrid=False),
        xaxis24=dict(title='CPORE', titlefont=dict(color=COLOR_DICT['CPORE'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['CPORE'], size=font_size),
                     side='top', anchor='free', position=.85, title_standoff=.1, overlaying='x4',
                     dtick=0.1, range=[0, .5], type='linear', showgrid=False, zeroline=False),
        xaxis25=dict(title='CPERM', titlefont=dict(color=COLOR_DICT['CPERM'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['CPERM'], size=font_size),
                     side='top', anchor='free', position=.85, title_standoff=.1, overlaying='x5',
                     range=[np.log10(0.1), np.log10(10000)], type='log', tickformat='d', tickangle=-90,
                     zeroline=False, showgrid=False),
        xaxis26=dict(title='CSAT', titlefont=dict(color=COLOR_DICT['CSAT'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['CSAT'], size=font_size),
                     side='top', anchor='free', position=.85, title_standoff=.1, overlaying='x6',
                     dtick=0.2, range=[0, 1], type='linear', zeroline=False, showgrid=False),
        xaxis27=dict(title='SHF', titlefont=dict(color=COLOR_DICT['SHF'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['SHF'], size=font_size),
                     side='top', anchor='free', position=.96, title_standoff=.1, overlaying='x6',
                     dtick=0.2, range=[0, 1], type='linear', zeroline=False, showgrid=False),
        xaxis28=dict(title='BVW', titlefont=dict(color=COLOR_DICT['BVW'], size=font_size),
                     tickfont=dict(color=COLOR_DICT['BVW'], size=font_size),
                     side='top', anchor='free', position=.96, title_standoff=.1, overlaying='x4',
                     dtick=0.1, range=[0, 0.5], type='linear', zeroline=False),

        # make room to display double x-axes
        yaxis=dict(domain=[0, .84], title=f'DEPTH ({depth_uom})'),
        yaxis2=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis3=dict(domain=[0, .84]),
        yaxis4=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis5=dict(domain=[0, .84]),
        yaxis6=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis7=dict(domain=[0, .84]),
        yaxis8=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis9=dict(domain=[0, .84]),
        yaxis10=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis11=dict(domain=[0, .84]),
        yaxis12=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis13=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis14=dict(domain=[0, .84], visible=False, showgrid=False),
        yaxis15=dict(domain=[0, .84]),
        yaxis16=dict(domain=[0, .84], visible=True, showgrid=False),

        # Update x and y axes for COAL_FLAG
        xaxis32=dict(title='', titlefont=dict(color=COLOR_DICT['COAL_FLAG'], size=font_size),
                     side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x4',
                     tick0=0, dtick=1, range=[0.1, .2], type='linear', tickfont=dict(size=1)),
        yaxis32=dict(domain=[0, .88], visible=False, showgrid=False, zeroline=False),
        xaxis33=dict(title='', titlefont=dict(color=COLOR_DICT['COAL_FLAG'], size=font_size),
                     side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x5',
                     tick0=0, dtick=1, range=[0.1, .2], type='linear', tickfont=dict(size=1)),
        yaxis33=dict(domain=[0, .88], visible=False, showgrid=False, zeroline=False),
        xaxis34=dict(title='', titlefont=dict(color=COLOR_DICT['COAL_FLAG'], size=font_size),
                     side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x6',
                     tick0=0, dtick=1, range=[0.1, .2], type='linear', tickfont=dict(size=1)),
        yaxis34=dict(domain=[0, .88], visible=False, showgrid=False, zeroline=False),
        xaxis35=dict(title='', titlefont=dict(color=COLOR_DICT['COAL_FLAG'], size=font_size),
                     side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x7',
                     tick0=0, dtick=1, range=[0.1, .2], type='linear', tickfont=dict(size=1)),
        yaxis35=dict(domain=[0, .88], visible=False, showgrid=False, zeroline=False),
        xaxis36=dict(title='', titlefont=dict(color=COLOR_DICT['COAL_FLAG'], size=font_size),
                     side='top', anchor='free', position=.97, title_standoff=.1, overlaying='x8',
                     tick0=0, dtick=1, range=[0.1, .2], type='linear', tickfont=dict(size=1)),
        yaxis36=dict(domain=[0, .88], visible=False, showgrid=False, zeroline=False),

        height=900,
        width=900,
        showlegend=False,
        title={
            'text': '%s Logs' % df.WELL_NAME.dropna().unique()[0],
            'y': .99,
            'xanchor': 'center',
            'yanchor': 'top',
            'font_size': font_size + 4
        }
    )

    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(matches='y', constrain='domain', autorange='reversed')

    # Plot horizontal line marker
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

    fig.update_layout(hovermode='y unified',
                      dragmode='pan',
                      modebar_remove=['lasso', 'select', 'autoscale'],
                      modebar_add=['drawline', 'drawcircle', 'drawrect', 'eraseshape'],
                      newshape_line_color='cyan', newshape_line_width=3,
                      template='none',
                      margin=dict(l=70, r=0, t=20, b=10),
                      paper_bgcolor='#e6f2ff',
                      hoverlabel_bgcolor='#F3F3F3')

    return fig
