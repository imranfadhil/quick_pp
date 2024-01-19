import numpy as np
import matplotlib.pyplot as plt

from ..utils import line_intersection

plt.style.use('Solarize_Light2')


def neutron_density_xplot(nphi, rhob,
                          dry_sand_point: tuple,
                          dry_silt_point: tuple,
                          dry_clay_point: tuple,
                          fluid_point: tuple,
                          wet_clay_point: tuple):

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
    fig.tight_layout()

    return fig


def picket_plot(rt, phit):
    """_summary_

    Args:
        rt (_type_): _description_
        phit (_type_): _description_

    Returns:
        _type_: _description_
    """
    fig = plt.Figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    ax.set_title('Pickett Plot')
    ax.scatter(rt, phit)
    ax.set_yscale('log')
    ax.set_ylim(0.01, 1)
    ax.set_ylabel('PHIT (v/v)')
    ax.set_xscale('log')
    ax.set_xlim(0.01, 100)
    ax.set_xlabel('RT (ohm.m)')
    ax.legend()
    fig.tight_layout()

    return fig


def sonic_density_xplot():
    pass


def sonic_neutron_xplot():
    pass
