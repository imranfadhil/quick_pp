from quick_pp import logger


def choo_permeability(vclw, vsilt, phit, m=2, B=None, A=20e7, C=3):
    """Estimate permeability using Choo's equation.

    Args:
        vclw (float): Volume of clay in fraction.
        vsilt (float): Volume of silt in fraction.
        phit (float): Total porosity in fraction.
        A (float): Constant based on 0.125 * rg**2 / 10. Defaults to 20e7.
        B (float): Constant derived from cementation and compaction factor, defaults to m * ((2 / C) + 1) + 2.
        C (float): Constant, defaults to 2.

    Returns:
        float: Choo's permeability in mD.
    """
    logger.debug(f"Calculating Choo permeability with A={A}, m={m}, C={C}")
    B = B or m * ((2 / C) + 1) + 2
    permeability = A * phit**B / 10**(6 * vclw + 3 * vsilt)
    logger.debug(f"Choo permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def kozeny_carman_permeability(phit, S=0.01):
    """Estimate permeability using Kozeny-Carman's equation.

    Args:
        phit (float): Total porosity in fraction.
        S (float): Specific surface area of the grains

    Returns:
        float: Permeability in mD.
    """
    logger.debug(f"Calculating Kozeny-Carman permeability with S={S}")
    permeability = phit**3 / (5 * S**2 * (1 - phit)**2)
    logger.debug(f"Kozeny-Carman permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def timur_permeability(phit, Swirr):
    """Estimate permeability based on Timur (1968) emperical equation established on 155 sandstone samples from
    3 different oil fields in North America.

    Args:
        phit (float): Porosity in fraction.
        Swirr (float): Irreduceable water saturation in fraction.

    Returns:
        float: Permeability in mD.
    """
    logger.debug("Calculating Timur permeability (1968)")
    permeability = 0.136 * phit**4.4 / Swirr**2
    logger.debug(f"Timur permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def coates_permeability(phie, Swirr, a=1):
    """Estimate permeability based on Coates (1974)

    Args:
        phie (float): Effective porosity in fraction.
        Swirr (float): Irreduceable water saturation in fraction.
        a (float): Constant. Defaults to 1.

    Returns:
        float: Coates permeability in mD.
    """
    logger.debug(f"Calculating Coates permeability with constant a={a}")
    permeability = (phie**2 / a * (1 - Swirr) / Swirr)**2
    logger.debug(f"Coates permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def tixier_permeability(phit, Swirr):
    """Estimate permeability based on Tixier (1949)

    Args:
        phit (float): Porosity in fraction.
        Swirr (float): Irreduceable water saturation in fraction.

    Returns:
        float: Permeability in mD
    """
    logger.debug("Calculating Tixier permeability (1949)")
    permeability = 62.5 * phit**6 / Swirr**2
    logger.debug(f"Tixier permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def morris_biggs_permeability(phit, C, Swirr):
    """Estimate permeability based on Morris and Biggs (1967)

    Args:
        phit (float): Total porosity in fraction.
        C (float): A constant which depends on the density of hydrocarbon in the formation.
                   250 for medium oil. 79 for gas.
        Swirr (float): Irreduceable water saturation in fraction.

    Returns:
        float: Permeability in mD.
    """
    logger.debug(f"Calculating Morris-Biggs permeability with C={C}")
    permeability = (C * phit**3 / Swirr)**2
    logger.debug(f"Morris-Biggs permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def morris_biggs_modified_permeability(phit, phie, Vbwi):
    """Estimate permeability based on Morris and Biggs (1967)

    Args:
        phit (float): Total porosity in fraction.
        phie (float): Effective porosity in fraction.
        Vbwi (float): Volume of bound water in fraction.

    Returns:
        float: Permeability in mD.
    """
    logger.debug("Calculating modified Morris-Biggs permeability")
    permeability = (1e2 * phie**2 * (phit - Vbwi) / Vbwi)**2
    logger.debug(f"Modified Morris-Biggs permeability range: {permeability.min():.3e} - {permeability.max():.3e} mD")
    return permeability


def estimate_krw(swt, swirr):
    """Estimate permeability based on Park Jones (1945)

    Args:
        swt (float): Total water saturation.
        swirr (float): Irreducible water saturation.

    Returns:
        float: Water relative permeability.
    """
    logger.debug("Calculating water relative permeability (Park Jones 1945)")
    krw = ((swt - swirr) / (1 - swirr))**3
    logger.debug(f"Water relative permeability range: {krw.min():.3f} - {krw.max():.3f}")
    return krw


def estimate_kro(swt, swirr):
    """Estimate permeability based on Park Jones (1945)

    Args:
        swt (float): Total water saturation.
        swirr (float): Irreducible water saturation.
    Returns:
        float: Oil relative permeability.
    """
    logger.debug("Calculating oil relative permeability (Park Jones 1945)")
    kro = ((0.9 - swt) / (0.9 - swirr))**2
    logger.debug(f"Oil relative permeability range: {kro.min():.3f} - {kro.max():.3f}")
    return kro


def estimate_wor(krw, kro, A=2):
    """Estimating Water Oil Ratio of surface production. Based on Pirson (1950).

    Args:
        krw (float): Water relative permeability.
        kro (float): Oil relative permeability.
        A (float): Constant. Defaults to 2.
            Light oil: 1 - 5
            Medium oil: 10 - 50
            Heavy oil: > 50
    Returns:
        float: Water Oil Ratio.
    """
    logger.debug(f"Calculating Water Oil Ratio with constant A={A}")
    wor = (krw / kro) * A
    logger.debug(f"Water Oil Ratio range: {wor.min():.3f} - {wor.max():.3f}")
    return wor


def estimate_wc(wor):
    """Estimating Water Cut of surface production.

    Args:
        wor (float): Water Oil Ratio.
    Returns:
        float: Water Cut.
    """
    logger.debug("Calculating Water Cut from Water Oil Ratio")
    wc = wor / (1 + wor)
    logger.debug(f"Water Cut range: {wc.min():.3f} - {wc.max():.3f}")
    return wc
