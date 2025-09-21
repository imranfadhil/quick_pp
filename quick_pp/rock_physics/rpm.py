import matplotlib.pyplot as plt
import numpy as np
from rockphypy import QI, GM, EM, Fluid
from rockphypy.Emp import Empirical as emp

from quick_pp.rock_physics.geomechanics import (
    estimate_shear_velocity, estimate_compressional_velocity,
    estimate_shear_modulus, estimate_bulk_modulus
)
from quick_pp.rock_type import estimate_vsh_gr
from quick_pp.config import Config


def qaqc_xplots(rhob, vp=None, vs=None):
    """Create a series of crossplots for rock physics quality control (QC).

    This function generates a 2x3 grid of common rock physics crossplots to
    visually inspect the relationships between bulk density, P-wave velocity,
    and S-wave velocity. If Vp or Vs are not provided, they are estimated
    using empirical relationships.

    Args:
        rhob (np.ndarray): Bulk density in g/cm³.
        vp (np.ndarray, optional): P-wave velocity in m/s. If None, it will be
            estimated from `rhob`. Defaults to None.
        vs (np.ndarray, optional): S-wave velocity in m/s. If None, it will be
            estimated from `vp`. Defaults to None.

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

    # AIp vs AIs crossplot
    ax2.scatter(ai_p/1e6, ai_s/1e6, alpha=0.5, s=20)
    ax2.set_xlabel('AIp (g/cm³ * km/s)')
    ax2.set_ylabel('AIs (g/cm³ * km/s)')
    ax2.set_title('AIp vs AIs')
    ax2.grid(True)

    # Vp/Vs vs Vs crossplot
    ax3.scatter(vp_vs, vs/1000, alpha=0.5, s=20)
    ax3.set_xlabel('Vp/Vs')
    ax3.set_ylabel('Vs (km/s)')
    ax3.set_title('Vp/Vs vs Vs')
    ax3.grid(True)

    # Vp vs Vs crossplot
    ax4.scatter(rhob, vs/1000, alpha=0.5, s=20)
    ax4.set_xlabel('Density (g/cm³)')
    ax4.set_ylabel('Vs (km/s)')
    ax4.set_title('Density vs Vs')
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
    """Create crossplots commonly used for fluid typing and lithology discrimination.

    This function generates a 1x3 grid of crossplots including Vp-Vs,
    Lambda-Rho vs. Mu-Rho, and AI vs. Vp/Vs. These plots help in
    identifying fluid effects in reservoir rocks. If Vp or Vs are not
    provided, they are estimated.

    Args:
        rhob (np.ndarray): Bulk density in g/cm³.
        vp (np.ndarray, optional): P-wave velocity in m/s. If None, it will be
            estimated from `rhob`. Defaults to None.
        vs (np.ndarray, optional): S-wave velocity in m/s. If None, it will be
            estimated from `vp`. Defaults to None.

    """
    if vs is None and vp is not None:
        vs = estimate_shear_velocity(vp)
    if vp is None:
        vp = estimate_compressional_velocity(rhob)

    vs = vs/1000
    vp = vp/1000
    ai = vp * rhob
    vp_vs = vp/vs
    mu_rho = (vs * rhob)**2
    lambda_rho = ai**2 - 2 * mu_rho

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    ax1.ticklabel_format(style='plain')
    ax2.ticklabel_format(style='plain')
    ax3.ticklabel_format(style='plain')

    # Vp vs Vs
    ax1.scatter(vs, vp, alpha=0.5, s=20)
    ax1.set_xlabel('Vs (km/s)')
    ax1.set_ylabel('Vp (km/s)')
    ax1.set_title('Vp vs Vs')
    ax1.grid(True)

    # Lambda_rho vs Mu_rho
    ax2.scatter(lambda_rho, mu_rho, alpha=0.5, s=20)
    ax2.set_xlabel('Lambda_rho (g/cm³ * km/s)²')
    ax2.set_ylabel('Mu_rho (g/cm³ * km/s)²')
    ax2.set_title('Lambda_rho vs Mu_rho')
    ax2.grid(True)

    # AI vs Vp/Vs
    ax3.scatter(ai, vp_vs, alpha=0.5, s=20)
    ax3.set_xlabel('AI (g/cm³ * km/s)')
    ax3.set_ylabel('Vp/Vs')
    ax3.set_title('AI vs Vp/Vs')
    ax3.grid(True)
    plt.tight_layout()
    return fig


def QI_screening(gr, rhob, vp):
    """Perform Quantitative Interpretation (QI) screening.

    This function uses the `rockphypy` library to create a Vp vs. Porosity
    screening plot, overlaying the input data on theoretical rock physics model
    lines. It helps to assess where the data lies with respect to different
    lithology and fluid scenarios.

    Args:
        gr (np.ndarray): Gamma Ray log in API units.
        rhob (np.ndarray): Bulk density in g/cm³.
        vp (np.ndarray): P-wave velocity in m/s.

    Returns:
        matplotlib.figure.Figure: Figure containing the QI screening plot.
    """

    # compute the elastic bounds
    Dqz = Config.GEOMECHANICS_VALUE['RHOB_QUARTZ']
    Kqz = Config.GEOMECHANICS_VALUE['K_QUARTZ']
    Gqz = Config.GEOMECHANICS_VALUE['G_QUARTZ']
    Dsh = Config.GEOMECHANICS_VALUE['RHOB_SHALE']
    Ksh = Config.GEOMECHANICS_VALUE['K_SHALE']
    Gsh = Config.GEOMECHANICS_VALUE['G_SHALE']
    Dc = Config.GEOMECHANICS_VALUE['RHOB_QUARTZ']
    Kc = Config.GEOMECHANICS_VALUE['K_QUARTZ']
    Gc = Config.GEOMECHANICS_VALUE['G_QUARTZ']
    Db = Config.GEOMECHANICS_VALUE['RHOB_BRINE']
    Kb = Config.GEOMECHANICS_VALUE['K_BRINE']

    vsh_gr = estimate_vsh_gr(gr)
    phit = (Dqz - rhob) / (Dqz - Db)

    phib_p = 0.3  # Adjusted high porosity limit
    phi_c = 0.4  # Critical porosity
    sigma = 20  # Effective stress
    scheme = 2  # Scheme of cement deposition
    f = 0.5  # Reduced shear factor
    Cn = 8.6  # Coordination number
    phi, vp1, vp2, vp3, vs1, vs2, vs3 = QI.screening(
        Dqz, Kqz, Gqz,
        Dsh, Ksh, Gsh,
        Dc, Kc, Gc,
        Db, Kb,
        phib_p, phi_c, sigma, 0, scheme, f, Cn
    )
    qi = QI(vp, phit, Vsh=vsh_gr)

    # call the screening plot method
    fig = qi.screening_plot(phi, vp1, vp2, vp3)
    plt.ylim([1900, 6100])
    plt.yticks(np.arange(2000, 6200, 1000), [2, 3, 4, 5, 6])
    plt.ylabel('Vp (Km/s)')
    plt.xlim(-0.01, 0.51)
    return fig


def rpt_plot(rhob, vp=None, vs=None, model='soft_sand', fluid_type='gas', sigma=20, phi_c=0.4, Cn=8.6, f=0.0, scheme=2):
    """Generate a Rock Physics Template (RPT) plot.

    This function overlays well log data (AI vs. Vp/Vs) on a theoretical
    Rock Physics Template (RPT) generated using models from the `rockphypy`
    library. The template shows lines of constant porosity and water saturation
    for a given rock and fluid model.

    Args:
        rhob (np.ndarray): Bulk density in g/cm³.
        vp (np.ndarray, optional): P-wave velocity in m/s. Defaults to None.
        vs (np.ndarray, optional): S-wave velocity in m/s. Defaults to None.
        model (str): Model to use for the RPT plot. Defaults to 'soft_sand'.
                     Options are 'soft_sand', 'stiff_sand', 'contact_cement', 'hertz_mindlin'.
        fluid_type (str): Fluid type for the model. Defaults to 'gas'.
                          Options are 'water', 'gas', 'oil'.
        sigma (float): Effective stress in MPa. Defaults to 20.
        phi_c (float): Critical porosity. Defaults to 0.4.
        Cn (float): Coordination number. Defaults to 8.6.
        f (float): Shear modulus correction factor for unconsolidated sands. Defaults to 0.0.
        scheme (int): Cement deposition scheme for contact cement model. Defaults to 2.

    Returns:
        matplotlib.figure.Figure: Figure containing the RPT plot.
    """
    model_options = ['soft_sand', 'stiff_sand', 'contact_cement', 'hertz_mindlin']
    assert model in model_options, f"Model must be one of {model_options}"
    fluid_type_options = ['water', 'gas', 'oil']
    assert fluid_type in ['water', 'gas', 'oil'], f"Fluid type must be one of {fluid_type_options}"

    if vs is None and vp is not None:
        vs = estimate_shear_velocity(vp)
    if vp is None:
        vp = estimate_compressional_velocity(rhob)
    if Cn is None and phi_c is not None:
        Cn = emp.Cp(phi_c)

    ai = vp * rhob
    vp_vs = vp/vs

    # Define the elastic bounds
    Dqz = Config.GEOMECHANICS_VALUE['RHOB_QUARTZ']
    Kqz = Config.GEOMECHANICS_VALUE['K_QUARTZ']
    Gqz = Config.GEOMECHANICS_VALUE['G_QUARTZ']
    Kc = Config.GEOMECHANICS_VALUE['K_CEMENT']
    Gc = Config.GEOMECHANICS_VALUE['G_CEMENT']

    Db = Config.GEOMECHANICS_VALUE['RHOB_BRINE']
    Kb = Config.GEOMECHANICS_VALUE['K_BRINE']
    Dg = Config.GEOMECHANICS_VALUE['RHOB_GAS']
    Kg = Config.GEOMECHANICS_VALUE['K_GAS']
    Do = Config.GEOMECHANICS_VALUE['RHOB_OIL']
    Ko = Config.GEOMECHANICS_VALUE['K_OIL']

    if fluid_type == 'water':
        Df = Db
        Kf = Kb
    elif fluid_type == 'gas':
        Df = Dg
        Kf = Kg
    elif fluid_type == 'oil':
        Df = Do
        Kf = Ko

    phi = np.linspace(0.1, phi_c, 10)  # Define porosity range according to critical porosity
    sw = np.linspace(0, 1, 5)  # Water saturation

    if model == 'soft_sand':
        K, G = GM.softsand(Kqz, Gqz, phi, phi_c, Cn, sigma, f)
    elif model == 'stiff_sand':
        K, G = GM.stiffsand(Kqz, Gqz, phi, phi_c, Cn, sigma, f)
    elif model == 'contact_cement':
        K, G = GM.contactcement(Kqz, Gqz, Kc, Gc, phi, phi_c, Cn, scheme)
    elif model == 'hertz_mindlin':
        K, G = GM.hertzmindlin(Kqz, Gqz, phi_c, Cn, sigma, f)

    # Plot the elastic properties
    fig = QI.plot_rpt(K, G, Kqz, Dqz, Kb, Db, Kf, Df, phi, sw)
    fig.set_size_inches(7, 6)
    plt.scatter(ai, vp_vs)
    plt.title(f'RPT Plot: {model.replace("_", " ").title()} - {fluid_type.title()}')
    plt.xlabel('AI (g/cm³ * m/s)')
    plt.ylabel('Vp/Vs')
    plt.xlim(1000, 20000)
    plt.ylim(1.0, 4.0)

    return fig


def elastic_bounds_plot(rhob, vp=None, vs=None, phi_calc=None):
    """Plot elastic moduli data against theoretical elastic bounds.

    This function plots calculated Bulk (K) and Shear (G) moduli against
    various theoretical bounds (Voigt, Reuss, Hashin-Shtrikman) and the
    critical porosity model. It helps to validate the elastic properties
    of the rock.

    Args:
        rhob (np.ndarray): Bulk density in g/cm³.
        vp (np.ndarray, optional): P-wave velocity in m/s. Defaults to None.
        vs (np.ndarray, optional): S-wave velocity in m/s. Defaults to None.
        phi_calc (np.ndarray, optional): Calculated porosity. If None, it will be
            estimated from `rhob`. Defaults to None.

    Returns:
        matplotlib.figure.Figure: Figure containing the elastic bounds plot
    """
    if vs is None and vp is not None:
        vs = estimate_shear_velocity(vp)
    if vp is None:
        vp = estimate_compressional_velocity(rhob)

    # Estimate bulk modulus and porosity
    K = estimate_bulk_modulus(rhob, vp, vs) * 1e-6
    G = estimate_shear_modulus(rhob, vs) * 1e-6
    if phi_calc is None:
        phi_calc = (Config.GEOMECHANICS_VALUE['RHOB_QUARTZ'] - rhob) / (
            Config.GEOMECHANICS_VALUE['RHOB_QUARTZ'] - Config.GEOMECHANICS_VALUE['RHOB_BRINE'])

    # specify model parameters
    phi = np.linspace(0, 1, 100, endpoint=True)
    K0 = Config.GEOMECHANICS_VALUE['K_QUARTZ']
    G0 = Config.GEOMECHANICS_VALUE['G_QUARTZ']
    Kw = Config.GEOMECHANICS_VALUE['K_BRINE']
    Gw = 0

    # VRH bounds
    volumes = np.vstack((1 - phi, phi)).T
    M = np.array([K0, Kw])
    K_v, K_r, K_h = EM.VRH(volumes, M)
    M = np.array([G0, Gw])
    G_v, G_r, G_h = EM.VRH(volumes, M)
    # Hashin-Strikmann bound
    K_UHS, G_UHS = EM.HS(1 - phi, K0, Kw, G0, Gw, bound='upper')
    # Critical porosity model
    phic = 0.4  # Critical porosity
    phi_ = np.linspace(0.001, phic, 100, endpoint=True)
    K_dry, G_dry = EM.cripor(K0, G0, phi_, phic)  # Compute dry-rock moduli
    Ksat, Gsat = Fluid.Gassmann(K_dry, G_dry, K0, Kw, phi_)  # Saturate rock with water

    # Create subplot with 2 columns and 1 row
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Bulk modulus bounds
    ax1.set_xlabel('Porosity')
    ax1.set_ylabel('Bulk modulus [GPa]')
    ax1.set_title('V, R, VRH, HS bounds')
    ax1.scatter(phi_calc, K, label='K')
    ax1.plot(phi, K_v, label='K Voigt')
    ax1.plot(phi, K_r, label='K Reuss = K HS-')
    ax1.plot(phi, K_h, label='K VRH')
    ax1.plot(phi, K_UHS, label='K HS+')
    ax1.plot(phi_, Ksat, label='K CriPor')
    ax1.legend(loc='best')
    ax1.grid(ls='--')

    # Plot Shear modulus bounds
    ax2.set_xlabel('Porosity')
    ax2.set_ylabel('Shear modulus [GPa]')
    ax2.set_title('V, R, VRH, HS bounds')
    ax2.scatter(phi_calc, G, label='G')
    ax2.plot(phi, G_v, label='G Voigt')
    ax2.plot(phi, G_r, label='G Reuss = G HS-')
    ax2.plot(phi, G_h, label='G VRH')
    ax2.plot(phi, G_UHS, label='G HS+')
    ax2.plot(phi_, Gsat, label='G CriPor')
    ax2.legend(loc='best')
    ax2.grid(ls='--')

    return fig
