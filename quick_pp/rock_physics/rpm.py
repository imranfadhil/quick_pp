import matplotlib.pyplot as plt
import numpy as np

from quick_pp.rock_physics.geomechanics import estimate_shear_velocity, estimate_compressional_velocity


def contact_model(K_matrix, G_matrix, K_grain, G_grain, porosity):
    """
    Contact model (Hertz-Mindlin) for dry rock frame moduli.
    """
    phi_c = 0.36  # Critical porosity
    C = (1 - porosity / phi_c)
    K_dry = (C ** 2) * K_grain
    G_dry = (C ** 2) * G_grain
    return K_dry, G_dry


def inclusion_model(K_matrix, G_matrix, K_incl, G_incl, inclusion_fraction):
    """
    Inclusion model (Self-consistent approximation) for effective moduli.
    """
    K_eff = K_matrix + inclusion_fraction * (K_incl - K_matrix)
    G_eff = G_matrix + inclusion_fraction * (G_incl - G_matrix)
    return K_eff, G_eff


def rock_physics_model(K_matrix, G_matrix, K_grain, G_grain, K_fluid, G_fluid, porosity, inclusion_fraction):
    """
    Combines contact and inclusion models to estimate saturated rock moduli.
    """
    # Dry frame moduli from contact model
    K_dry, G_dry = contact_model(K_matrix, G_matrix, K_grain, G_grain, porosity)
    # Effective moduli from inclusion model
    K_eff, G_eff = inclusion_model(K_dry, G_dry, K_fluid, G_fluid, inclusion_fraction)
    return K_eff, G_eff


def qaqc_xplots(rhob, vp=None, vs=None):
    """Create crossplots for rock physics QC.

    Args:
        rhob (numpy.ndarray): Bulk density in g/cm³
        vp (numpy.ndarray): P-wave velocity in m/s
        vs (numpy.ndarray): S-wave velocity in m/s

    Returns:
        matplotlib.figure.Figure: Figure containing the QC crossplots
    """
    # Calculate Vs from Vp if not provided
    if vs is None and vp is not None:
        vs = estimate_shear_velocity(vp)
    # Calculate vp from rhob and vs
    if vp is None:
        vp = estimate_compressional_velocity(rhob)

    # Calculate acoustic impedance
    ai_p = vp * rhob
    ai_s = vs * rhob
    vp_vs = vp/vs

    # Create figure with 6 subplots
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(15, 10))

    # Vp vs Vs crossplot
    ax1.scatter(vp/1000, vs/1000, alpha=0.5, s=20)
    ax1.set_xlabel('Vp (km/s)')
    ax1.set_ylabel('Vs (km/s)')
    ax1.set_title('Vp vs Vs')
    ax1.grid(True)

    # AI_p vs AI_s crossplot
    ax2.scatter(ai_p/1e6, ai_s/1e6, alpha=0.5, s=20)
    ax2.set_xlabel('AI_p (g/cm³ * km/s)')
    ax2.set_ylabel('AI_s (g/cm³ * km/s)')
    ax2.set_title('AI_p vs AI_s')
    ax2.grid(True)

    # Vp/Vs vs Vs crossplot
    ax3.scatter(vp_vs, vs/1000, alpha=0.5, s=20)
    ax3.set_xlabel('Vp/Vs')
    ax3.set_ylabel('Vs (km/s)')
    ax3.set_title('Vp/Vs vs Vs')
    ax3.grid(True)

    # Vp vs Vs crossplot
    ax4.scatter(vs/1000, vp/1000, alpha=0.5, s=20)
    ax4.set_xlabel('Vs (km/s)')
    ax4.set_ylabel('Vp (km/s)')
    ax4.set_title('Vs vs Vp')
    ax4.grid(True)

    # Vp vs density crossplot
    ax5.scatter(rhob, vp/1000, alpha=0.5, s=20)
    ax5.set_xlabel('Density (g/cm³)')
    ax5.set_ylabel('Vp (km/s)')
    ax5.set_title('Density vs Vp')
    ax5.grid(True)

    # AI vs Vp/Vs crossplot
    ax6.scatter(ai_p/1e6, vp_vs, alpha=0.5, s=20)
    ax6.set_xlabel('Vp/Vs')
    ax6.set_ylabel('AIp (g/cm³ * km/s)')
    ax6.set_title('Vp/Vs vs AIp')
    ax6.grid(True)

    plt.tight_layout()
    return fig


def fluid_typing_xplots(rhob, vp=None, vs=None):
    """Create crossplots for fluid typing.
    """
    if vs is None and vp is not None:
        vs = estimate_shear_velocity(vp)
    if vp is None:
        vp = estimate_compressional_velocity(rhob)
    ai = vp * rhob
    vp_vs = vp/vs
    dtc = 1 / vp * (1e6 / 3.281)
    dts = 1 / vs * (1e6 / 3.281)

    x = np.linspace(70, 170, 100)
    lst_vpvs = 82.3 / 43.4
    lst_line = lst_vpvs * x
    print(lst_line)
    dol_vpvs = 72.5 / 39.8
    dol_line = dol_vpvs * x
    sst_vpvs = 78.6 / 48.5
    sst_line = sst_vpvs * x

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.ticklabel_format(style='plain')
    ax2.ticklabel_format(style='plain')
    ax3.ticklabel_format(style='plain')

    ax1.scatter(dts, dtc, alpha=0.5, s=20)
    ax1.plot(x, lst_line, label='LST', color='red')
    ax1.plot(x, dol_line, label='DOL', color='green')
    ax1.plot(x, sst_line, label='SST', color='blue')
    ax1.set_xlim(70, 170)
    ax1.set_ylim(100, 40)
    ax1.set_xlabel('DTs (us/ft)')
    ax1.set_ylabel('Dtc (us/ft)')
    ax1.set_title('Dtc vs Dts')
    ax1.grid(True)
    ax2.scatter(vs/1000, vp/1000, alpha=0.5, s=20)
    ax2.set_xlabel('Vs (km/s)')
    ax2.set_ylabel('Vp (km/s)')
    ax2.set_title('Vp vs Vs')
    ax2.grid(True)
    ax3.scatter(ai, vp_vs, alpha=0.5, s=20)
    ax3.set_xlabel('AI (g/cm³ * m/s)')
    ax3.set_ylabel('Vp/Vs')
    ax3.set_title('AI vs Vp/Vs')
    ax3.grid(True)
    plt.tight_layout()
    return fig


# Example usage
if __name__ == "__main__":
    # Matrix (quartz) properties
    K_matrix = 37.0  # GPa
    G_matrix = 44.0  # GPa
    # Grain properties
    K_grain = 36.6   # GPa
    G_grain = 45.0   # GPa
    # Fluid properties
    K_fluid = 2.2    # GPa
    G_fluid = 0.0    # GPa
    porosity = 0.25
    inclusion_fraction = 0.1

    K_sat, G_sat = rock_physics_model(
        K_matrix, G_matrix, K_grain, G_grain, K_fluid, G_fluid, porosity, inclusion_fraction
    )
    print(f"Saturated Bulk Modulus: {K_sat:.2f} GPa")
    print(f"Saturated Shear Modulus: {G_sat:.2f} GPa")
