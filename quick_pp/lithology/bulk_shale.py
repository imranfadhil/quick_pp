"""Bulk-shale density model and explicit spectral gamma response."""

import numpy as np


def spectral_cgr(thorium, potassium, thorium_coefficient, potassium_coefficient):
    """Calculate an estimated uranium-free gamma response from Th (ppm) and K (%)."""
    th = np.asarray(thorium, dtype=float)
    k = np.asarray(potassium, dtype=float)
    if not (
        np.isfinite(thorium_coefficient)
        and np.isfinite(potassium_coefficient)
        and thorium_coefficient > 0
        and potassium_coefficient > 0
    ):
        raise ValueError("CGR coefficients must be finite and positive")
    valid = np.isfinite(th) & np.isfinite(k) & (th >= 0) & (k >= 0)
    return np.where(
        valid, th * thorium_coefficient + k * potassium_coefficient, np.nan
    )


def bulk_shale_density(
    gamma, rhob, clean_gamma, shale_gamma, clean_density, shale_density,
    fluid_density, shale_porosity, ineffective_fraction,
):
    """Return bulk shale, solid volumes and porosities for physically valid mixtures.

    The shale solid volume is a proxy for clay, not a mineralogical clay estimate.
    Inconsistent density mixtures are returned as NaN rather than clipped.
    """
    params = np.asarray(
        [clean_gamma, shale_gamma, clean_density, shale_density, fluid_density,
         shale_porosity, ineffective_fraction], dtype=float
    )
    if not np.all(np.isfinite(params)):
        raise ValueError("Bulk-shale endpoints must be finite")
    if shale_gamma <= clean_gamma:
        raise ValueError("Shale gamma endpoint must be greater than clean endpoint")
    if clean_density <= fluid_density or shale_density <= fluid_density:
        raise ValueError("Solid densities must exceed fluid density")
    if not 0 <= shale_porosity < 1 or not 0 <= ineffective_fraction <= 1:
        raise ValueError("Shale porosity must be in [0, 1) and ineffective fraction in [0, 1]")

    gamma = np.asarray(gamma, dtype=float)
    rhob = np.asarray(rhob, dtype=float)
    vsh = np.clip((gamma - clean_gamma) / (shale_gamma - clean_gamma), 0, 1)
    shale_solid = vsh * (1 - shale_porosity)
    shale_bulk_density = (1 - shale_porosity) * shale_density + shale_porosity * fluid_density
    clean_pore = ((1 - vsh) * clean_density + vsh * shale_bulk_density - rhob) / (
        clean_density - fluid_density
    )
    valid = (
        np.isfinite(gamma) & np.isfinite(rhob)
        & (clean_pore >= -1e-8) & (clean_pore <= 1 - vsh + 1e-8)
    )
    clean_pore = np.minimum(np.maximum(clean_pore, 0), 1 - vsh)
    clean_solid = 1 - vsh - clean_pore
    phit = clean_pore + vsh * shale_porosity
    phie = phit - ineffective_fraction * vsh * shale_porosity
    total_solid = clean_solid + shale_solid
    rhom = np.divide(
        clean_solid * clean_density + shale_solid * shale_density,
        total_solid, out=np.full_like(total_solid, np.nan), where=total_solid > 0,
    )
    return {
        "VSHALE": np.where(valid, vsh, np.nan),
        "VCLAY": np.where(valid, shale_solid, np.nan),
        "VSAND": np.where(valid, clean_solid, np.nan),
        "PHIT": np.where(valid, phit, np.nan),
        "PHIE": np.where(valid, phie, np.nan),
        "PHID": np.where(valid, phit, np.nan),
        "RHOM": np.where(valid, rhom, np.nan),
    }
