from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.porosity import neu_den_xplot_poro
from quick_pp.utils import length_a_b, line_intersection


class ThinBeds:
    """A model for analyzing thin-bedded sand-shale sequences.

    This class implements methods to estimate the volumes of laminated and
    dispersed shale, as well as sand porosity, from neutron-density crossplots
    and the analytical Thomas-Stieber (1975) model. It also includes forward
    and inverse resistivity modeling based on Hagiwara (1995, 1997) to account
    for macroscopic and microscopic electrical anisotropy.
    """

    def __init__(
        self,
        dry_sand_poro: Optional[float] = None,
        dry_shale_poro: Optional[float] = None,
        dry_sand_point: Optional[tuple[float, float]] = None,
        dry_clay_point: Optional[tuple[float, float]] = None,
        fluid_point: Optional[tuple[float, float]] = None,
        **kwargs,
    ):
        """Initializes the ThinBeds model with specified or default endpoints.

        Args:
            dry_sand_poro (float, optional): Porosity of dry sand. Defaults to config values.
            dry_shale_poro (float, optional): Porosity of dry shale. Defaults to config values.
            dry_sand_point (tuple, optional): (NPHI, RHOB) for dry sand. Defaults to config values.
            dry_clay_point (tuple, optional): (NPHI, RHOB) for dry clay. Defaults to config values.
            fluid_point (tuple, optional): (NPHI, RHOB) for the formation fluid. Defaults to config values.
        """
        # Initialize the endpoints
        self.dry_sand_poro = dry_sand_poro or Config.TS_ENDPOINTS["DRY_SAND_PORO"]
        self.dry_shale_poro = dry_shale_poro or Config.TS_ENDPOINTS["DRY_SHALE_PORO"]
        self.dry_sand_point = dry_sand_point or Config.TS_ENDPOINTS["DRY_SAND_POINT"]
        self.dry_clay_point = dry_clay_point or Config.TS_ENDPOINTS["DRY_SHALE_POINT"]
        self.fluid_point = fluid_point or Config.TS_ENDPOINTS["FLUID_POINT"]

        logger.debug(
            f"ThinBeds model initialized with endpoints: sand_poro={self.dry_sand_poro}, "
            f"shale_poro={self.dry_shale_poro}, sand_point={self.dry_sand_point}, "
            f"clay_point={self.dry_clay_point}, fluid_point={self.fluid_point}"
        )

    def estimate_litho_poro(self, nphi, rhob):
        """Estimate laminated and dispersed shale based on neutron density cross plot.

        Args:
            nphi (np.ndarray or float): Neutron Porosity log [v/v].
            rhob (np.ndarray or float): Bulk Density log [g/cc].

        Returns:
            tuple: A tuple containing volumes of sand, shale, total porosity, laminated shale,
                   dispersed shale, dispersed sand, and sand porosity.
        """
        logger.info(
            f"Estimating thin bed lithology and porosity for {len(nphi)} data points"
        )

        nphi_arr = np.asarray(nphi, dtype=float)
        rhob_arr = np.asarray(rhob, dtype=float)
        if nphi_arr.shape != rhob_arr.shape:
            raise ValueError("nphi and rhob must have the same shape")

        # Calculate porosity
        phit = neu_den_xplot_poro(
            nphi_arr,
            rhob_arr,
            model="ss",
            dry_min1_point=self.dry_sand_point,
            dry_clay_point=self.dry_clay_point,
        )

        A = self.dry_sand_point
        C = self.dry_clay_point
        D = self.fluid_point
        rocklithofrac = length_a_b(A, C)

        var_pt = line_intersection((A, C), (D, (nphi_arr, rhob_arr)))
        projlithofrac = length_a_b(var_pt, A)
        vshale = projlithofrac / rocklithofrac
        vsand = 1.0 - vshale

        vsh_lam, vsh_dis, vsand_dis, phit_sand = self.litho_poro_fraction(phit, vshale)

        logger.debug(
            f"Thin bed estimation completed - mean vsand: {np.nanmean(vsand):.3f}, "
            f"vshale: {np.nanmean(vshale):.3f}, vsh_lam: {np.nanmean(vsh_lam):.3f}, "
            f"vsh_dis: {np.nanmean(vsh_dis):.3f}"
        )

        return vsand, vshale, phit, vsh_lam, vsh_dis, vsand_dis, phit_sand

    def litho_poro_fraction(self, phit, vshale):
        """Estimate laminated and dispersed shale and intrinsic sand porosity.

        Follows the analytical Thomas-Stieber (1975) model:
        - Baseline dispersed line: Phi_dis(Vsh) = Phi_sd - Vsh * (1 - Phi_sh)
        - Analytical laminated fraction: Vlam = (Phi - Phi_dis(Vsh)) / (1 - Phi_sd)
        - Dispersed shale fraction: Vsh_dis = Vsh - Vlam
        - Intrinsic sand lamina porosity: Phi_sand = (Phi - Vlam * Phi_sh) / (1 - Vlam)

        Args:
            phit (np.ndarray or float): Total porosity [v/v].
            vshale (np.ndarray or float): Volume of shale [v/v].

        Returns:
            tuple: A tuple containing (vsh_lam, vsh_dis, vsand_dis, phit_sand).
        """
        logger.debug(
            "Calculating lithology-porosity fractions for thin beds using Thomas-Stieber model"
        )

        vsh_arr = np.asarray(vshale, dtype=float)
        phit_arr = np.asarray(phit, dtype=float)
        is_scalar = vsh_arr.ndim == 0 and phit_arr.ndim == 0
        vsh_arr = np.atleast_1d(vsh_arr)
        phit_arr = np.atleast_1d(phit_arr)

        if vsh_arr.shape != phit_arr.shape:
            raise ValueError("phit and vshale must have the same shape")

        phi_sd = float(self.dry_sand_poro)
        phi_sh = float(self.dry_shale_poro)

        # Dispersed shale baseline: all sand pores filled by dispersed shale particles
        phi_dis = phi_sd - vsh_arr * (1.0 - phi_sh)

        # Laminated shale fraction Vlam: analytical solution to Phi = (1 - Vlam)*Phi_sand + Vlam*Phi_sh
        vsh_lam = (phit_arr - phi_dis) / max(1.0 - phi_sd, 1e-4)
        # Bounded between 0 and min(Vshale, 1.0)
        vsh_lam = np.clip(vsh_lam, 0.0, np.clip(vsh_arr, 0.0, 1.0))

        # Dispersed shale fraction (bulk)
        vsh_dis = np.clip(vsh_arr - vsh_lam, 0.0, 1.0)

        # Sand lamina fraction: (1 - Vlam)
        sand_lam_frac = np.maximum(1.0 - vsh_lam, 1e-4)

        # Dispersed shale within sand lamina & dispersed sand fraction
        v_dis_sand = np.clip(vsh_dis / sand_lam_frac, 0.0, 1.0)
        vsand_dis = np.clip(1.0 - v_dis_sand, 0.0, 1.0)

        # Intrinsic sand lamina porosity Phi_sand:
        # In laminated shaly sand: Phi = (1 - Vlam)*Phi_sand + Vlam*Phi_sh
        # Hence: Phi_sand = (Phi - Vlam * Phi_sh) / (1 - Vlam)
        phit_sand = (phit_arr - vsh_lam * phi_sh) / sand_lam_frac

        # In pure shale limit (Vlam -> 1.0 or Vsh -> 1.0), phit_sand smoothly limits to phi_sd
        phit_sand = np.where(vsh_lam >= 0.95, phi_sd, phit_sand)
        # Ensure physical bounds: Phi_sand >= bulk Phi in sands, capped at reasonable max
        phit_sand = np.clip(
            np.maximum(phit_sand, phit_arr),
            0.0,
            max(phi_sd * 1.15, 0.45),
        )

        logger.debug(
            f"Lithology-porosity fraction calculation completed for {len(vsh_lam)} points"
        )
        if is_scalar:
            return (
                float(vsh_lam[0]),
                float(vsh_dis[0]),
                float(vsand_dis[0]),
                float(phit_sand[0]),
            )
        return vsh_lam, vsh_dis, vsand_dis, phit_sand

    def resistivity_modelling(self, vsh_lam, rsand, rv_shale, rh_shale, theta):
        """Calculate the apparent resistivity of laminated sand-shale based on Hagiwara (1995).

        Args:
            vsh_lam (np.ndarray or float): Fraction of laminated shale.
            rsand (np.ndarray or float): Resistivity of the sand layers [Ohm.m].
            rv_shale (np.ndarray or float): Vertical resistivity of the shale layers [Ohm.m].
            rh_shale (np.ndarray or float): Horizontal resistivity of the shale layers [Ohm.m].
            theta (np.ndarray or float): The relative dip angle in degrees.

        Returns:
            np.ndarray or float: The modeled apparent resistivity of the formation [Ohm.m].
        """
        logger.debug(
            f"Calculating resistivity using Hagiwara (1995) model with theta={theta}°"
        )
        vsh_lam_arr = np.asarray(vsh_lam, dtype=float)
        is_scalar = vsh_lam_arr.ndim == 0
        vsh_lam_arr = np.atleast_1d(vsh_lam_arr)

        csd = 1.0 / np.maximum(rsand, 1e-4)
        csh = 1.0 / np.maximum(rh_shale, 1e-4)

        # Parallel horizontal conductivity
        ch = csh * vsh_lam_arr + csd * (1.0 - vsh_lam_arr)
        rh = 1.0 / np.maximum(ch, 1e-6)

        # Series vertical resistivity
        rv = rv_shale * vsh_lam_arr + rsand * (1.0 - vsh_lam_arr)

        # Apparent resistivity at relative dip angle theta (Moran & Gianzero 1979, Hagiwara 1995)
        rad = np.radians(theta)
        cos2 = np.cos(rad) ** 2
        sin2 = np.sin(rad) ** 2
        ratio = rh / np.maximum(rv, 1e-4)
        denom = np.sqrt(np.maximum(cos2 + ratio * sin2, 1e-6))
        ra = rh / denom

        if is_scalar:
            return float(ra[0])
        return ra

    def apparent_resistivity(self, rv, rh, theta):
        """Calculate the apparent resistivity based on Hagiwara (1997) / Moran & Gianzero (1979).

        Args:
            rv (np.ndarray or float): Vertical resistivity of the formation [Ohm.m].
            rh (np.ndarray or float): Horizontal resistivity of the formation [Ohm.m].
            theta (np.ndarray or float): The relative dip angle in degrees.

        Returns:
            np.ndarray or float: The apparent resistivity [Ohm.m].
        """
        logger.debug(
            f"Calculating apparent resistivity using Hagiwara (1997) with theta={theta}°"
        )
        rv_arr = np.asarray(rv, dtype=float)
        rh_arr = np.asarray(rh, dtype=float)
        is_scalar = rv_arr.ndim == 0 and rh_arr.ndim == 0
        rv_arr = np.atleast_1d(rv_arr)
        rh_arr = np.atleast_1d(rh_arr)

        rad = np.radians(theta)
        cos2 = np.cos(rad) ** 2
        sin2 = np.sin(rad) ** 2
        ratio = rh_arr / np.maximum(rv_arr, 1e-4)
        ra = rh_arr / np.sqrt(np.maximum(cos2 + ratio * sin2, 1e-6))

        if is_scalar:
            return float(ra[0])
        return ra

    def sand_resistivity_macro(self, rv, rh, rshale):
        """Calculate the macroscopic resistivity of the sand layers based on Hagiwara (1997).

        In a macroscopically anisotropic laminated medium:
            Rv = (1 - Vlam)*Rsand + Vlam*Rshale
            1/Rh = (1 - Vlam)/Rsand + Vlam/Rshale
        Solving for Rsand:
            Rsand = Rh * (Rv - Rshale) / (Rh - Rshale)

        Args:
            rv (np.ndarray or float): Vertical resistivity of the formation [Ohm.m].
            rh (np.ndarray or float): Horizontal resistivity of the formation [Ohm.m].
            rshale (float): Resistivity of the shale [Ohm.m].

        Returns:
            np.ndarray or float: The macroscopic resistivity of the sand layers [Ohm.m].
        """
        logger.debug("Calculating macroscopic sand resistivity")
        rv_arr = np.asarray(rv, dtype=float)
        rh_arr = np.asarray(rh, dtype=float)
        is_scalar = rv_arr.ndim == 0 and rh_arr.ndim == 0
        rv_arr = np.atleast_1d(rv_arr)
        rh_arr = np.atleast_1d(rh_arr)

        num = rv_arr - rshale
        den = rh_arr - rshale
        valid = np.abs(den) > 1e-4
        rsand = np.where(valid, rh_arr * (num / np.where(valid, den, 1.0)), rh_arr)
        rsand = np.clip(rsand, 0.05, 5000.0)

        if is_scalar:
            return float(rsand[0])
        return rsand

    def sand_resistivity_micro(self, rv, rh, rv_shale, rh_shale):
        """Calculate the microscopic anisotropy resistivity of the sand based on Hagiwara (1997).

        Accounts for intrinsic shale microscopic anisotropy (Rv_shale != Rh_shale).

        Args:
            rv (np.ndarray or float): Vertical resistivity of the formation [Ohm.m].
            rh (np.ndarray or float): Horizontal resistivity of the formation [Ohm.m].
            rv_shale (float): Vertical resistivity of the shale [Ohm.m].
            rh_shale (float): Horizontal resistivity of the shale [Ohm.m].

        Returns:
            np.ndarray or float: The microscopic resistivity of the sand layers [Ohm.m].
        """
        logger.debug("Calculating microscopic sand resistivity")
        alpha = self.sand_resistivity_macro(rv, rh, rh_shale)
        alpha_arr = np.asarray(alpha, dtype=float)
        is_scalar = alpha_arr.ndim == 0
        alpha_arr = np.atleast_1d(alpha_arr)

        beta = alpha_arr / max(rh_shale, 1e-4)
        aniso_sh = rh_shale / max(rv_shale, 1e-4) - 1.0
        disc = np.maximum((beta - 1.0) ** 2 + 4.0 * beta * aniso_sh, 0.0)
        denom = 1.0 + 0.5 * (beta - 1.0 - np.sqrt(disc))
        r_micro = np.where(np.abs(denom) > 1e-4, alpha_arr / denom, alpha_arr)
        r_micro = np.clip(r_micro, 0.05, 5000.0)

        if is_scalar:
            return float(r_micro[0])
        return r_micro


def vsh_phit_xplot(vsh, phit, dry_sand_poro: float, dry_shale_poro: float, **kwargs):
    """Generate a Vshale vs. Porosity crossplot for thin bed analysis.

    This plot helps visualize the distribution of data points relative to
    laminated, dispersed, and structural shale models.

    Args:
        vsh (np.ndarray): An array of shale volume fractions.
        phit (np.ndarray): An array of total porosity values.
        dry_sand_poro (float): The porosity endpoint for clean sand.
        dry_shale_poro (float): The porosity endpoint for pure shale.

    Returns:
        matplotlib.figure.Figure: The generated Vshale-PHIT crossplot figure.
    """
    logger.info(f"Creating VSH-PHIT crossplot for {len(vsh)} data points")

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
    ax.set_title("NPHI-RHOB Crossplot")
    ax.scatter(vsh, phit, c=np.arange(0, len(vsh)), cmap="rainbow", marker=".")

    ax.plot(*zip(*vsh_lam_from_pt, strict=True), label="Laminated", color="blue")
    ax.plot(
        *zip(*vsh_dis_from_pt, strict=True),
        label="Dispersed (pore filling)",
        color="green",
    )
    ax.plot(
        *zip(*vsh_lower_envlope_from_pt, strict=True),
        label="Dispersed (grain replacing)",
        color="black",
    )

    # Add isolines parallel to vsh_lam_from_pt
    num_lines = 9
    for i in range(1, num_lines + 1):
        t = i / (num_lines + 1)
        intermediate_line = [
            (A[0] + t * (B[0] - A[0]), A[1] + t * (B[1] - A[1])),
            (C[0] + t * (B[0] - C[0]), C[1] + t * (B[1] - C[1])),
        ]
        ax.plot(
            *zip(*intermediate_line, strict=True),
            linestyle="--",
            color="blue",
            alpha=0.75,
        )
        ax.text(
            intermediate_line[0][0],
            intermediate_line[0][1] + 0.007,
            f"{int(t * 100)}%",
            fontsize=8,
            color="blue",
            ha="center",
            va="center",
        )

    # Add lines with different slopes originating from point B
    num_lines = 9
    for i in range(1, num_lines + 1):
        t = i / (num_lines + 1)
        intermediate_line = [
            (A[0] + t * (C[0] - A[0]), A[1] + t * (C[1] - A[1])),
            (B[0], B[1]),
        ]
        ax.plot(
            *zip(*intermediate_line, strict=True),
            linestyle="--",
            color="red",
            alpha=0.75,
        )
        ax.text(
            intermediate_line[0][0] - 0.03,
            intermediate_line[0][1],
            f"{int((1 - t) * 100 * (A[1] - C[1]))}%",
            fontsize=8,
            color="red",
            ha="center",
            va="center",
        )

    ax.scatter(A[0], A[1], label=f"Clean Sand: ({A[0]}, {A[1]})", color="orange")
    ax.scatter(B[0], B[1], label=f"Pure Shale: ({B[0]}, {B[1]})", color="black")
    ax.scatter(C[0], C[1], label=f"Lower vertex: ({C[0]}, {C[1]})", color="blue")

    ax.set_ylim(0, 0.5)
    ax.set_ylabel("PHIT (v/v)")
    ax.set_xlim(-0.05, 1)
    ax.set_xlabel("Vshale (v/v)")
    ax.legend(loc="upper left", prop={"size": 9})
    ax.minorticks_on()
    ax.grid(True, which="major", linestyle="--", linewidth="0.5", color="gray")
    ax.grid(True, which="minor", linestyle=":", linewidth="0.4", color="gray")
    fig.tight_layout()

    logger.debug("VSH-PHIT crossplot created successfully")
    return fig
