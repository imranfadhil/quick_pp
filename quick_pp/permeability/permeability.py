
def choo_permeability(vclw, vsilt, phit, m=2, B=None, A=20e7, C=2):
    """_summary_

    Args:
        vclw (_type_): _description_
        vsilt (_type_): _description_
        phit (_type_): _description_
        A (_type_): 0.125 * rg**2 / 10
        B (_type_): Constant or m * ((2 / C) + 1) + 2
        C (_type_): 2

    Returns:
        _type_: _description_
    """
    B = B or m * ((2 / C) + 1) + 2
    return A * phit**B / 10**(6 * vclw + 3 * vsilt)


def kozeny_carman_permeability(phit, S=0.01):
    """_summary_

    Args:
        phit (v/v): _description_
        S (_type_): Specific surface area of the grains

    Returns:
        _type_: _description_
    """
    return phit**3 / (5 * S**2 * (1 - phit**2))


def permeability_thickness(phit, Swirr, A, B, C):
    """KH = A * phit**B / Swirr**C

    Args:
        phit (_type_): _description_
        Swirr (_type_): _description_
        h (_type_): _description_
        A (_type_): _description_
        B (_type_): _description_
        C (_type_): _description_

    Returns:
        _type_: _description_
    """
    return A * phit**B / Swirr**C


def timur_permeability(phit, Swirr):
    """Based on Timur (1968) emperical equation established on 155 sandstone samples from 3 different
    oil fields in North America.

    Args:
        phit (%): Porosity
        Swirr (%): Irreduceable water saturation

    Returns:
        _type_: Permeability in mD
    """
    return 8.58 * phit**4.4 / Swirr**2


def coates_permeability(phie, Swirr, a=1):
    """Based on Coates (1974)

    Args:
        phie (%): Effective porosity
        Swirr (%): Irreduceable water saturation
        a (float): Constant

    Returns:
        _type_: _description_
    """
    return (phie**2 / a * (1 - Swirr) / Swirr)**2


def tixier_permeability(phit, Swirr):
    """Based on Tixier (1949)

    Args:
        phit (%): Porosity
        Swirr (%): Irreduceable water saturation

    Returns:
        _type_: Permeability in mD
    """
    return 62.5 * phit**6 / Swirr**2


def morris_biggs_permeability(phit, phie, Vbwi):
    """Based on Morris and Biggs (1967)

    Args:
        phit (%): Total porosity
        phie (%): Effective porosity
        Vbwi (%): Volume of bound water

    Returns:
        _type_: _description_
    """
    return (1e2 * phie**2 * (phit - Vbwi) / Vbwi)**2
