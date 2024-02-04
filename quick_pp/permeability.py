def choo_permeability(vclw, vsilt, phit, m=2, B=None, A=20e7, C=2):
    """Estimate permeability using Choo's equation.

    Args:
        vclw (float): Volume of clay in fraction.
        vsilt (float): Volume of silt in fraction.
        phit (float): Total porosity in fraction.
        A (float): Constant based on 0.125 * rg**2 / 10. Defaults to 20e7.
        B (float): Constant, defaults m * ((2 / C) + 1) + 2.
        C (float): Constant, defaults to 2.

    Returns:
        float: Choo's permeability in mD.
    """
    B = B or m * ((2 / C) + 1) + 2
    return A * phit**B / 10**(6 * vclw + 3 * vsilt)


def kozeny_carman_permeability(phit, S=0.01):
    """Estimate permeability using Kozeny-Carman's equation.

    Args:
        phit (float): Total porosity in fraction.
        S (float): Specific surface area of the grains

    Returns:
        float: Permeability in mD.
    """
    return phit**3 / (5 * S**2 * (1 - phit**2))


def timur_permeability(phit, Swirr):
    """Based on Timur (1968) emperical equation established on 155 sandstone samples from 3 different
    oil fields in North America.

    Args:
        phit (float): Porosity in fraction.
        Swirr (float): Irreduceable water saturation in fraction.

    Returns:
        float: Permeability in mD.
    """
    return 8.58 * phit**4.4 / Swirr**2


def coates_permeability(phie, Swirr, a=1):
    """Based on Coates (1974)

    Args:
        phie (float): Effective porosity in fraction.
        Swirr (float): Irreduceable water saturation in fraction.
        a (float): Constant. Defaults to 1.

    Returns:
        float: Coates permeability in mD.
    """
    return (phie**2 / a * (1 - Swirr) / Swirr)**2


def tixier_permeability(phit, Swirr):
    """Based on Tixier (1949)

    Args:
        phit (float): Porosity in fraction.
        Swirr (float): Irreduceable water saturation in fraction.

    Returns:
        float: Permeability in mD
    """
    return 62.5 * phit**6 / Swirr**2


def morris_biggs_permeability(phit, phie, Vbwi):
    """Based on Morris and Biggs (1967)

    Args:
        phit (float): Total porosity in fraction.
        phie (float): Effective porosity in fraction.
        Vbwi (float): Volume of bound water in fraction.

    Returns:
        float: Permeability in mD.
    """
    return (1e2 * phie**2 * (phit - Vbwi) / Vbwi)**2
