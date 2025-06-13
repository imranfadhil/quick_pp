import numpy as np
import math
import matplotlib.pyplot as plt

from quick_pp.utils import length_a_b, line_intersection, angle_between_lines
from quick_pp.porosity import neu_den_xplot_poro
from quick_pp.config import Config


class ThinBeds:
    """This binary model only consider a combination of sand-shale components. """

    def __init__(self, dry_sand_poro: tuple = None, dry_shale_poro: tuple = None,
                 dry_sand_point: tuple = None, dry_clay_point: tuple = None, fluid_point: tuple = None, **kwargs):
        # Initialize the endpoints
        self.dry_sand_poro = dry_sand_poro or Config.TS_ENDPOINTS["DRY_SAND_PORO"]
        self.dry_shale_poro = dry_shale_poro or Config.TS_ENDPOINTS["DRY_SHALE_PORO"]
        self.dry_sand_point = dry_sand_point or Config.TS_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_clay_point = dry_clay_point or Config.TS_ENDPOINTS["DRY_SHALE_POINT"]
        self.fluid_point = fluid_point or Config.TS_ENDPOINTS["FLUID_POINT"]

    def estimate_litho_poro(self, nphi, rhob):
        """Estimate laminated and dispersed shale based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vsh_lam, vsh_dis, phit_sand
        """
        # Calculate porosity
        phit = neu_den_xplot_poro(
            nphi, rhob, model='ss',
            dry_min1_point=self.dry_sand_point,
            dry_clay_point=self.dry_clay_point,
        )

        A = self.dry_sand_point
        C = self.dry_clay_point
        D = self.fluid_point
        E = list(zip(nphi, rhob))
        rocklithofrac = length_a_b(A, C)

        vsand = np.empty(0)
        vshale = np.empty(0)
        for i, point in enumerate(E):
            var_pt = line_intersection((A, C), (D, point))
            projlithofrac = length_a_b(var_pt, A)
            vsh_pt = projlithofrac / rocklithofrac
            vsand = np.append(vsand, (1 - vsh_pt))
            vshale = np.append(vshale, vsh_pt)

        vsh_lam, vsh_dis, vsand_dis, phit_sand = self.litho_poro_fraction(phit, vshale)

        return vsand, vshale, phit, vsh_lam, vsh_dis, vsand_dis, phit_sand

    def litho_poro_fraction(self, phit, vshale):
        """Estimate sand and shale based on neutron density cross plot.

        Args:
            nphi (float): Neutron Porosity log in v/v
            rhob (float): Bulk Density log in g/cc

        Returns:
            (float, float): vsand, vcld
        """
        # Initialize the endpoints
        A = (0, self.dry_sand_poro)
        B = (1, self.dry_shale_poro)
        lower_vertex_phit = self.dry_sand_poro * self.dry_shale_poro
        lower_vertex_vshale = lower_vertex_phit / self.dry_shale_poro
        C = (lower_vertex_vshale, lower_vertex_phit)

        theta_vsh_lam = angle_between_lines((A, C), (A, (0, 0)))

        vsh_lam_frac = length_a_b(B, C)
        vsh_dis_frac = length_a_b(A, C)
        vsh_lam = np.empty(0)
        vsh_dis = np.empty(0)
        vsand_dis = np.empty(0)
        phit_sand = np.empty(0)
        D = list(zip(vshale, phit))
        for i, point in enumerate(D):
            vsh_pt = point[0] + length_a_b(point, (point[0], 0)) * math.tan(math.radians(theta_vsh_lam))
            vsh_lam_pt = line_intersection((B, C), ((vsh_pt, 0), point))
            proj_vsh_lam_frac = length_a_b(vsh_lam_pt, C)
            vsh_lam = np.append(vsh_lam, proj_vsh_lam_frac / vsh_lam_frac)

            proj_pt = line_intersection((A, C), (B, point))
            proj_poro_frac = length_a_b(proj_pt, C)
            vsand_dis_pt = proj_poro_frac / vsh_dis_frac
            poro_sand_pt = vsand_dis_pt * (A[1] - C[1])
            phit_sand = np.append(phit_sand, poro_sand_pt)
            vsand_dis = np.append(vsand_dis, vsand_dis_pt)
            vsh_dis = np.append(vsh_dis, (1 - vsand_dis_pt))

        return vsh_lam, vsh_dis, vsand_dis, phit_sand

    def resistivity_modelling(self, vsh_lam, rsand, rv_shale, rh_shale, theta):
        """Calculate the resistivity based on the laminated and dispersed shale based on Hagiwara (1995).

        Args:
            vsh_lam (float): Fraction of laminated shale.
            rsand (float): Resistivity of the sand.
            rh_shale (float): Horizontal resistivity of the shale.
            rv_shale (float): Vertical resistivity of the shale.
            theta (float): Dip angle in degrees.

        Returns:
            float: Resistivity of the formation.
        """
        csd = 1 / rsand
        csh = 1 / rh_shale
        ch = csh * vsh_lam + csd * (1 - vsh_lam)
        rv = rv_shale * vsh_lam + rsand * (1 - vsh_lam)
        return 1 / (ch * (math.cos(math.radians(theta))**2 + ch * rv * math.sin(math.radians(theta))**2))

    def apparent_resistivity(self, rv, rh, theta):
        """Calculate the apparent resistivity based on Hagiwara (1997).

        Args:
            rv (float): Resistivity of the dispersed sand.
            rh (float): Resistivity of the shale.
            theta (float): Dip angle in degrees.

        Returns:
            float: Apparent resistivity.
        """
        return rh / (math.cos(math.radians(theta))**2 + (rh / rv) * math.sin(math.radians(theta))**2)**.5

    def sand_resistivity_macro(self, rv, rh, rshale):
        """Calculate the (macroscopic anistropy) resistivity of the sand based on Hagiwara (1997).

        Args:
            rv (float): Resistivity of the dispersed sand.
            rh (float): Resistivity of the shale.
            rv_shale (float): Vertical resistivity of the shale.
            rh_shale (float): Horizontal resistivity of the shale.

        Returns:
            float: Resistivity of the sand.
        """
        return (rv - rshale) / (rh - rshale)

    def sand_resistivity_micro(self, rv, rh, rv_shale, rh_shale):
        """Calculate the (microscopic anisotropy) resistivity of the sand based on Hagiwara (1997).

        Args:
            rv (float): Resistivity of the dispersed sand.
            rh (float): Resistivity of the shale.
            rv_shale (float): Vertical resistivity of the shale.
            rh_shale (float): Horizontal resistivity of the shale.

        Returns:
            float: Resistivity of the sand.
        """
        alpha = self.sand_resistivity_macro(rv, rh, rh_shale)
        beta = alpha / rh_shale
        return alpha / (1 + .5 * (beta - 1 - ((beta - 1)**2 + 4 * beta * (rh_shale / rv_shale - 1))**.5))


def vsh_phit_xplot(vsh, phit, dry_sand_poro: float, dry_shale_poro: float, **kwargs):
    """Neutron-Density crossplot with lithology lines based on specified end points.

    Args:
        vsh (np.ndarray): Array of shale volume fraction.
        phit (np.ndarray): Array of total porosity.
        dry_sand_poro (float): Dry sand porosity endpoint.
        dry_shale_poro (float): Dry shale porosity endpoint.

    Returns:
        matplotlib.pyplot.Figure: Neutron porosity and bulk density cross plot.
    """
    A = (0, dry_sand_poro)
    B = (1, dry_shale_poro)
    lower_vertex_phit = round(dry_sand_poro * dry_shale_poro, 4)
    lower_vertex_vshale = round(lower_vertex_phit / dry_shale_poro, 4)
    C = (lower_vertex_vshale, lower_vertex_phit)

    # Plotting the NPHI-RHOB crossplot
    vsh_lam_from_pt = (A, B)
    vsh_dis_from_pt = (A, C)
    vsh_lower_envlope_from_pt = ((0, 0), B)

    fig = plt.Figure(figsize=(7, 7))
    ax = fig.add_subplot(111)
    ax.set_title('NPHI-RHOB Crossplot')
    ax.scatter(vsh, phit, c=np.arange(0, len(vsh)), cmap='rainbow', marker='.')

    ax.plot(*zip(*vsh_lam_from_pt), label='Laminated', color='blue')
    ax.plot(*zip(*vsh_dis_from_pt), label='Dispersed (pore filling)', color='green')
    ax.plot(*zip(*vsh_lower_envlope_from_pt), label='Dispersed (grain replacing)', color='black')

    # Add isolines parallel to vsh_lam_from_pt
    num_lines = 9
    for i in range(1, num_lines + 1):
        t = i / (num_lines + 1)
        intermediate_line = [
            (A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])),
            (C[0] + t * (B[0] - C[0]), C[1] + t * (B[1] - C[1]))
        ]
        ax.plot(*zip(*intermediate_line), linestyle='--', color='blue', alpha=0.75)
        ax.text(
            intermediate_line[0][0], intermediate_line[0][1] + .007,
            f'{int(t * 100)}%',
            fontsize=8,
            color='blue',
            ha='center',
            va='center'
        )

    # Add lines with different slopes originating from point B
    num_lines = 9
    for i in range(1, num_lines + 1):
        t = i / (num_lines + 1)
        intermediate_line = [
            (A[0] + t * (C[0] - A[0]), A[1] + t * (C[1] - A[1])), (B[0], B[1])
        ]
        ax.plot(*zip(*intermediate_line), linestyle='--', color='red', alpha=0.75)
        ax.text(
            intermediate_line[0][0] - .03, intermediate_line[0][1],
            f'{int((1 - t) * 100 * (A[1] - C[1]))}%',
            fontsize=8,
            color='red',
            ha='center',
            va='center'
        )

    ax.scatter(A[0], A[1], label=f'Clean Sand: ({A[0]}, {A[1]})', color='orange')
    ax.scatter(B[0], B[1], label=f'Pure Shale: ({B[0]}, {B[1]})', color='black')
    ax.scatter(C[0], C[1], label=f'Lower vertex: ({C[0]}, {C[1]})', color='blue')

    ax.set_ylim(0, 0.5)
    ax.set_ylabel('PHIT (v/v)')
    ax.set_xlim(-.05, 1)
    ax.set_xlabel('Vshale (v/v)')
    ax.legend(loc="upper left", prop={'size': 9})
    ax.minorticks_on()
    ax.grid(True, which='major', linestyle='--', linewidth='0.5', color='gray')
    ax.grid(True, which='minor', linestyle=':', linewidth='0.4', color='gray')
    fig.tight_layout()

    return fig
