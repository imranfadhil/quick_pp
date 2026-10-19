"""Explicit SCAL (Special Core Analysis) & Core Corrections submodule for quick_pp.

Provides petrophysically rigorous core-to-log and lab-to-reservoir reconciliation routines:
  1. Stress / Overburden Porosity & Permeability Corrections:
     - `overburden_porosity_correction`: Exponential pore-volume compressibility or power-law
       compaction of ambient core porosity ($\\phi_{amb}$) to reservoir net confining stress ($NCS$).
     - `overburden_permeability_correction`: Exponential stress-sensitivity (David et al., 1994),
       power-law (Jones & Owens, 1980), or Kozeny-Carman porosity-coupled permeability compaction
       from ambient lab stress to reservoir net confining stress ($NCS$).
     - `overburden_poroperm_correction`: Joint stress correction returning $(\\phi_{res}, k_{res})$.
  2. Klinkenberg Gas Slippage Correction:
     - `klinkenberg_permeability_correction`: Converts laboratory gas/air permeability ($k_{air}$)
       at mean flowing pressure ($P_m$) to equivalent liquid permeability ($k_{\\infty}$) via
       Klinkenberg (1941), supporting explicit slippage factor $b$, Jones & Owens (1980) empirical
       correlation, or Sampath & Keighin (1982) porosity-dependent correlation.
  3. Clay-Bound Water (CBW) Corrections for Dean-Stark vs. Humidity-Dried Cores:
     - `humidity_to_total_porosity`: Reconciles humidity-dried core porosity ($\\phi_e$, retains CBW)
       and convection oven-dried / Dean-Stark core porosity ($\\phi_t$, drives off CBW).
     - `dean_stark_cbw_correction` (alias `dean_stark_saturation_correction`): Reconciles
       Dean-Stark total water saturation ($S_{wt}$) with effective water saturation ($S_{we}$)
       and corrects apparent saturations reported on a humidity-dried pore volume basis.
  4. Capillary Pressure ($P_c$) Lab-to-Reservoir Conversion:
     - `pc_lab_to_reservoir`: Converts laboratory $P_{c,lab}$ (air-brine, mercury-air MICP, or
       lab oil-brine) to reservoir $P_{c,res}$ accounting for interfacial tension ($\\sigma$),
       contact angle ($\\theta$), and overburden stress shifts in $\\sqrt{k / \\phi}$.
  5. DataFrame Pipeline Helper:
     - `correct_core_data`: Applies overburden porosity/permeability, clay-bound water,
       Dean-Stark saturation, Klinkenberg permeability, and $P_c$ corrections to a core DataFrame.
"""

from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

__all__ = [
    "correct_core_data",
    "dean_stark_cbw_correction",
    "dean_stark_saturation_correction",
    "estimate_cbw_from_qv",
    "humidity_to_total_porosity",
    "klinkenberg_permeability_correction",
    "overburden_permeability_correction",
    "overburden_poroperm_correction",
    "overburden_porosity_correction",
    "pc_lab_to_reservoir",
]


def overburden_porosity_correction(
    core_poro: Union[float, np.ndarray, pd.Series],
    net_confining_stress: Union[float, np.ndarray, pd.Series],
    compaction_coeff: float = 4e-6,
    ambient_stress: float = 0.0,
    method: str = "exponential",
) -> Union[float, np.ndarray, pd.Series]:
    """Correct ambient core porosity to reservoir net confining stress (NCS).

    Applies either an exponential pore-volume compressibility decay or a power-law
    stress compaction model to account for pore volume reduction under overburden stress:
      - Exponential: $\\phi_{res} = \\phi_{amb} \\exp(-c_{\\phi} \\Delta \\sigma)$
      - Power-law: $\\phi_{res} = \\phi_{amb} (\\sigma_{NCS} / \\sigma_{ref})^{-c_{\\phi}}$

    Args:
        core_poro (float or array-like): Ambient/lab core porosity (frac).
        net_confining_stress (float or array-like): Reservoir net confining stress (psi).
        compaction_coeff (float, optional): Pore compaction coefficient ($1/\\text{psi}$ for
            'exponential', or dimensionless exponent for 'power_law'). Defaults to 4e-6.
        ambient_stress (float, optional): Laboratory measurement confining stress (psi).
            Defaults to 0.0 (or 400.0 psi reference for 'power_law' if <= 0).
        method (str, optional): Compaction model, either 'exponential' or 'power_law'.
            Defaults to 'exponential'.

    Returns:
        float or np.ndarray: Stress-corrected core porosity at reservoir conditions (frac).
    """
    delta_stress = np.maximum(net_confining_stress - ambient_stress, 0.0)
    if method == "power_law":
        ref_stress = ambient_stress if ambient_stress > 0 else 400.0
        stress_ratio = np.maximum(net_confining_stress / ref_stress, 1.0)
        return core_poro * (stress_ratio ** (-compaction_coeff))
    return core_poro * np.exp(-compaction_coeff * delta_stress)


def overburden_permeability_correction(
    core_perm: Union[float, np.ndarray, pd.Series],
    net_confining_stress: Union[float, np.ndarray, pd.Series],
    gamma: float = 3e-5,
    ambient_stress: float = 0.0,
    method: str = "exponential",
    core_poro: Optional[Union[float, np.ndarray, pd.Series]] = None,
    res_poro: Optional[Union[float, np.ndarray, pd.Series]] = None,
    poro_compaction_coeff: float = 4e-6,
    poro_exponent: float = 3.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Correct ambient/laboratory core permeability to reservoir net confining stress (NCS).

    Supports three industry-standard petrophysical stress-compaction formulations:
      - `'exponential'` (David et al., 1994):
          $$k_{res} = k_{amb} \\exp(-\\gamma \\Delta \\sigma)$$
      - `'power_law'` (Jones & Owens, 1980; Ostensen, 1983):
          $$k_{res} = k_{amb} \\left(\\frac{\\sigma_{NCS}}{\\sigma_{ref}}\\right)^{-\\gamma}$$
      - `'kozeny_carman'` (Porosity-coupled grain/pore throat compaction, Amaefule et al., 1993):
          $$k_{res} = k_{amb} \\left(\\frac{\\phi_{res}}{\\phi_{amb}}\\right)^{c} \\left(\\frac{1 - \\phi_{amb}}{1 - \\phi_{res}}\\right)^2$$

    Args:
        core_perm (float or array-like): Ambient/laboratory permeability (mD).
        net_confining_stress (float or array-like): Reservoir net confining stress (psi).
        gamma (float, optional): Permeability stress-sensitivity coefficient ($1/\\text{psi}$ for
            'exponential', or dimensionless exponent for 'power_law'). Defaults to 3e-5.
        ambient_stress (float, optional): Laboratory confining stress (psi). Defaults to 0.0
            (or 400.0 psi reference for 'power_law' if <= 0).
        method (str, optional): Stress correction model: `'exponential'`, `'power_law'`, or
            `'kozeny_carman'`. Defaults to `'exponential'`.
        core_poro (float or array-like, optional): Ambient core porosity (frac), used when
            `method='kozeny_carman'`. Defaults to None.
        res_poro (float or array-like, optional): Stress-corrected reservoir porosity (frac),
            used when `method='kozeny_carman'`. If None and `core_poro` is provided, computed
            via `overburden_porosity_correction`. Defaults to None.
        poro_compaction_coeff (float, optional): Pore compaction coefficient ($1/\\text{psi}$) if
            `res_poro` is inferred from `core_poro`. Defaults to 4e-6.
        poro_exponent (float, optional): Porosity ratio exponent $c$ in `'kozeny_carman'` mode.
            Defaults to 3.0.

    Returns:
        float or np.ndarray: Stress-corrected permeability at reservoir conditions (mD).
    """
    method_clean = method.strip().lower()
    delta_stress = np.maximum(net_confining_stress - ambient_stress, 0.0)

    if method_clean == "power_law":
        ref_stress = ambient_stress if ambient_stress > 0 else 400.0
        stress_ratio = np.maximum(net_confining_stress / ref_stress, 1.0)
        return np.maximum(core_perm * (stress_ratio ** (-gamma)), 0.0)

    if method_clean == "kozeny_carman":
        if core_poro is None:
            raise ValueError(
                "core_poro is required when method='kozeny_carman' in overburden_permeability_correction."
            )
        if res_poro is None:
            res_poro = overburden_porosity_correction(
                core_poro,
                net_confining_stress,
                compaction_coeff=poro_compaction_coeff,
                ambient_stress=ambient_stress,
                method="exponential",
            )
        phi_amb_safe = np.clip(core_poro, 1e-6, 1.0 - 1e-6)
        phi_res_safe = np.clip(res_poro, 1e-6, 1.0 - 1e-6)
        kc_factor = ((phi_res_safe / phi_amb_safe) ** poro_exponent) * (
            ((1.0 - phi_amb_safe) / (1.0 - phi_res_safe)) ** 2.0
        )
        return np.maximum(core_perm * kc_factor, 0.0)

    if method_clean != "exponential":
        raise ValueError(
            f"Unsupported overburden permeability method '{method}'. "
            "Choose from 'exponential', 'power_law', or 'kozeny_carman'."
        )

    return np.maximum(core_perm * np.exp(-gamma * delta_stress), 0.0)


def overburden_poroperm_correction(
    core_poro: Union[float, np.ndarray, pd.Series],
    core_perm: Union[float, np.ndarray, pd.Series],
    net_confining_stress: Union[float, np.ndarray, pd.Series],
    poro_compaction_coeff: float = 4e-6,
    perm_compaction_coeff: float = 3e-5,
    ambient_stress: float = 0.0,
    method: str = "exponential",
) -> Tuple[Union[float, np.ndarray, pd.Series], Union[float, np.ndarray, pd.Series]]:
    """Jointly correct ambient core porosity and permeability to reservoir net confining stress.

    Args:
        core_poro (float or array-like): Ambient core porosity (frac).
        core_perm (float or array-like): Ambient core permeability (mD).
        net_confining_stress (float or array-like): Reservoir net confining stress (psi).
        poro_compaction_coeff (float, optional): Porosity compaction coefficient. Defaults to 4e-6.
        perm_compaction_coeff (float, optional): Permeability stress-sensitivity coefficient.
            Defaults to 3e-5.
        ambient_stress (float, optional): Laboratory confining stress (psi). Defaults to 0.0.
        method (str, optional): `'exponential'`, `'power_law'`, or `'kozeny_carman'`.
            Defaults to `'exponential'`.

    Returns:
        tuple: `(poro_res, perm_res)` corrected to reservoir net confining stress.
    """
    poro_method = "power_law" if method.strip().lower() == "power_law" else "exponential"
    poro_res = overburden_porosity_correction(
        core_poro,
        net_confining_stress,
        compaction_coeff=poro_compaction_coeff,
        ambient_stress=ambient_stress,
        method=poro_method,
    )
    perm_res = overburden_permeability_correction(
        core_perm,
        net_confining_stress,
        gamma=perm_compaction_coeff,
        ambient_stress=ambient_stress,
        method=method,
        core_poro=core_poro,
        res_poro=poro_res,
        poro_compaction_coeff=poro_compaction_coeff,
    )
    return poro_res, perm_res


def klinkenberg_permeability_correction(
    k_air: Union[float, np.ndarray, pd.Series],
    mean_pore_pressure: Union[float, np.ndarray, pd.Series],
    b_factor: Optional[Union[float, np.ndarray, pd.Series]] = None,
    method: str = "jones_owens",
    poro: Optional[Union[float, np.ndarray, pd.Series]] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Correct laboratory gas/air permeability for gas slippage (Klinkenberg effect).

    Converts measured gas permeability ($k_{air}$) at a given mean flowing pore pressure ($P_m$)
    to equivalent liquid permeability ($k_{liquid}$) using Klinkenberg (1941):
      $$k_{liquid} = \\frac{k_{air}}{1 + b / P_m}$$

    If `b_factor` is not provided, it is estimated via empirical correlation:
      - `'jones_owens'` (Jones & Owens, 1980, default): $b = 12.64 \\cdot k_{air}^{-0.33}$ (psi)
      - `'sampath_keighin'` (Sampath & Keighin, 1982): $b = 13.85 \\cdot (k_{air} / \\phi)^{-0.53}$ (psi)

    Args:
        k_air (float or array-like): Measured gas/air permeability (mD).
        mean_pore_pressure (float or array-like): Mean flowing pore pressure (psi).
        b_factor (float or array-like, optional): Klinkenberg gas slippage factor (psi).
            If None, estimated from `k_air` using `method`. Defaults to None.
        method (str, optional): Empirical correlation for `b_factor` when `b_factor=None`:
            `'jones_owens'` or `'sampath_keighin'`. Defaults to `'jones_owens'`.
        poro (float or array-like, optional): Core porosity (frac), required when
            `method='sampath_keighin'`. Defaults to None.

    Returns:
        float or np.ndarray: Equivalent liquid permeability (mD).
    """
    is_scalar = np.isscalar(k_air) and np.isscalar(mean_pore_pressure) and (
        b_factor is None or np.isscalar(b_factor)
    )
    k_safe = np.maximum(k_air, 1e-6)
    p_safe = np.maximum(mean_pore_pressure, 1e-6)
    if b_factor is None:
        method_clean = method.strip().lower()
        if method_clean == "sampath_keighin":
            if poro is None:
                raise ValueError(
                    "poro is required when method='sampath_keighin' in klinkenberg_permeability_correction."
                )
            phi_safe = np.maximum(poro, 1e-4)
            b_factor = 13.85 * ((k_safe / phi_safe) ** (-0.53))
        else:
            b_factor = 12.64 * (k_safe ** (-0.33))
    res = np.where(k_air > 0, k_air / (1.0 + (b_factor / p_safe)), 0.0)
    return float(res) if is_scalar else res


def estimate_cbw_from_qv(
    qv: Union[float, np.ndarray, pd.Series],
    phit: Union[float, np.ndarray, pd.Series],
    water_salinity: Union[float, np.ndarray, pd.Series] = 10000.0,
) -> Union[float, np.ndarray, pd.Series]:
    """Estimate bulk volume of clay-bound water ($V_{cbw} = \\phi_t S_{wb}$) from $Q_v$ and salinity.

    Uses the Hill, Shirley, and Klein (1979) empirical relationship:
      $$S_{wb} = (0.084 \\cdot C_{sal}^{-0.5} + 0.22) \\cdot Q_v, \\quad V_{cbw} = \\phi_t \\cdot S_{wb}$$

    Args:
        qv (float or array-like): Cation exchange capacity per unit pore volume (meq/cc).
        phit (float or array-like): Total porosity (frac).
        water_salinity (float or array-like, optional): Formation water salinity (ppm).
            Defaults to 10000.0.

    Returns:
        float or np.ndarray: Bulk volume fraction of clay-bound water $V_{cbw}$ (v/v).
    """
    sal_safe = np.maximum(water_salinity, 1.0)
    swb = np.clip((0.084 * (sal_safe ** -0.5) + 0.22) * np.maximum(qv, 0.0), 0.0, 1.0)
    return np.clip(phit * swb, 0.0, np.maximum(phit, 0.0))


def humidity_to_total_porosity(
    core_poro: Union[float, np.ndarray, pd.Series],
    cbw: Union[float, np.ndarray, pd.Series],
    oven_dried: bool = False,
) -> Union[float, np.ndarray, pd.Series]:
    """Reconcile humidity-dried vs. oven-dried / Dean-Stark core plug porosity against PHIT and PHIE.

    Humidity-dried core plugs (e.g., 60°C at 40% RH) retain clay-bound water (CBW) and
    therefore measure effective porosity ($\\phi_e$). Convection oven-dried or Dean-Stark
    extracted core plugs (e.g., 105-115°C) drive off CBW and measure total porosity ($\\phi_t$).

    - When `oven_dried=False` (default): converts humidity-dried core porosity ($\\phi_e$)
      to total porosity ($\\phi_t = \\phi_{core} + V_{cbw}$).
    - When `oven_dried=True`: converts oven-dried / Dean-Stark core porosity ($\\phi_t$)
      to effective porosity ($\\phi_e = \\phi_{core} - V_{cbw}$).

    Args:
        core_poro (float or array-like): Measured core porosity (frac).
        cbw (float or array-like): Bulk volume of clay-bound water (frac, v/v),
            i.e., $V_{sh} \\cdot \\phi_{sh}$ or $\\phi_t - \\phi_e$.
        oven_dried (bool, optional): If True, treats input `core_poro` as oven-dried /
            Dean-Stark ($\\phi_t$) and subtracts `cbw` to return $\\phi_e$. If False, treats
            input as humidity-dried ($\\phi_e$) and adds `cbw` to return $\\phi_t$.
            Defaults to False.

    Returns:
        float or np.ndarray: Reconciled total porosity (or effective porosity if
            `oven_dried=True`), clipped to [0.0, 1.0] (frac).
    """
    reconciled_poro = core_poro - cbw if oven_dried else core_poro + cbw
    return np.clip(reconciled_poro, 0.0, 1.0)


def dean_stark_cbw_correction(
    core_sw: Union[float, np.ndarray, pd.Series],
    swb: Optional[Union[float, np.ndarray, pd.Series]] = None,
    cbw: Optional[Union[float, np.ndarray, pd.Series]] = None,
    phit: Optional[Union[float, np.ndarray, pd.Series]] = None,
    phie: Optional[Union[float, np.ndarray, pd.Series]] = None,
    qv: Optional[Union[float, np.ndarray, pd.Series]] = None,
    water_salinity: float = 10000.0,
    to_effective: bool = True,
    reported_on_humidity_poro: bool = False,
) -> Union[float, np.ndarray, pd.Series]:
    """Correct and reconcile water saturation between Dean-Stark and humidity-dried cores for CBW.

    In Dean-Stark distillation (toluene/xylene extraction at >105°C), extracted water includes
    both free/capillary water and clay-bound water (CBW), yielding total water saturation ($S_{wt}$).
    In contrast, capillary pressure tests on humidity-dried plugs (or effective porosity models)
    represent effective water saturation ($S_{we}$). Furthermore, if Dean-Stark water volume is
    divided by humidity-dried pore volume ($\\phi_e$) rather than total pore volume ($\\phi_t$),
    the reported saturation is artificially inflated by $1 / (1 - S_{wb})$.

    Clay-bound water saturation $S_{wb} \\in [0, 1)$ can be supplied directly via `swb`, or inferred from:
      1. `cbw` and `phit`: $S_{wb} = V_{cbw} / \\phi_t$
      2. `phit` and `phie`: $S_{wb} = (\\phi_t - \\phi_e) / \\phi_t$
      3. `qv` and `water_salinity` (Hill-Shirley-Klein, 1979): $S_{wb} = (0.084 C_{sal}^{-0.5} + 0.22) Q_v$

     Transformations:
      - Optional denominator fix (`reported_on_humidity_poro=True`):
          $$S_{wt} = S_{w,reported} \\cdot (1 - S_{wb})$$
      - Total-to-Effective (`to_effective=True`, default — Dean-Stark $S_{wt} \\to S_{we}$):
          $$S_{we} = \\text{clip}\\left(\\frac{S_{wt} - S_{wb}}{1 - S_{wb}}, 0.0, 1.0\\right)$$
      - Effective-to-Total (`to_effective=False` — humidity-dried/capillary $S_{we} \\to S_{wt}$):
          $$S_{wt} = \\text{clip}\\left(S_{we} (1 - S_{wb}) + S_{wb}, 0.0, 1.0\\right)$$

    Args:
        core_sw (float or array-like): Core water saturation (frac).
        swb (float or array-like, optional): Bound water saturation fraction of total pore volume.
        cbw (float or array-like, optional): Bulk volume of clay-bound water (v/v).
        phit (float or array-like, optional): Total porosity (frac).
        phie (float or array-like, optional): Effective porosity (frac).
        qv (float or array-like, optional): Cation exchange capacity per unit pore volume (meq/cc).
        water_salinity (float, optional): Formation water salinity (ppm) for Hill-Shirley-Klein.
            Defaults to 10000.0.
        to_effective (bool, optional): If True, converts total water saturation ($S_{wt}$) to
            effective saturation ($S_{we}$) by stripping CBW. If False, converts effective
            saturation ($S_{we}$) to total saturation ($S_{wt}$) by adding CBW. Defaults to True.
        reported_on_humidity_poro (bool, optional): Set True if Dean-Stark water volume was
            erroneously normalized by humidity-dried pore volume ($\\phi_e$) instead of $\\phi_t$.
            Defaults to False.

    Returns:
        float or np.ndarray: CBW-corrected core water saturation (frac), clipped to [0.0, 1.0].
    """
    if swb is None:
        if cbw is not None and phit is not None:
            swb = cbw / np.maximum(phit, 1e-6)
        elif phit is not None and phie is not None:
            swb = (phit - phie) / np.maximum(phit, 1e-6)
        elif qv is not None:
            sal_safe = np.maximum(water_salinity, 1.0)
            swb = (0.084 * (sal_safe ** -0.5) + 0.22) * np.maximum(qv, 0.0)
        else:
            raise ValueError(
                "Must provide one of: `swb`, (`cbw` and `phit`), (`phit` and `phie`), or `qv`."
            )

    swb_clipped = np.clip(swb, 0.0, 0.99)
    sw_work = core_sw * (1.0 - swb_clipped) if reported_on_humidity_poro else core_sw

    if to_effective:
        res = (sw_work - swb_clipped) / np.maximum(1.0 - swb_clipped, 1e-6)
    else:
        res = sw_work * (1.0 - swb_clipped) + swb_clipped

    return np.clip(res, 0.0, 1.0)


dean_stark_saturation_correction = dean_stark_cbw_correction


def pc_lab_to_reservoir(
    pc_lab: Union[float, np.ndarray, pd.Series],
    ift_lab: float = 72.0,
    theta_lab: float = 0.0,
    ift_res: float = 30.0,
    theta_res: float = 30.0,
    poro_amb: Optional[Union[float, np.ndarray, pd.Series]] = None,
    perm_amb: Optional[Union[float, np.ndarray, pd.Series]] = None,
    poro_res: Optional[Union[float, np.ndarray, pd.Series]] = None,
    perm_res: Optional[Union[float, np.ndarray, pd.Series]] = None,
) -> Union[float, np.ndarray, pd.Series]:
    """Convert laboratory capillary pressure ($P_{c,lab}$) to reservoir conditions ($P_{c,res}$).

    Scales capillary pressure for fluid interfacial tension ($\\sigma$) and wetting contact angle
    ($\\theta$), and optionally accounts for overburden stress shifts in pore throat geometry via
    Leverett $J$-function invariance:
      $$P_{c,res} = P_{c,lab} \\cdot \\frac{\\sigma_{res} |\\cos\\theta_{res}|}{\\sigma_{lab} |\\cos\\theta_{lab}|} \\cdot \\sqrt{\\frac{k_{amb} / \\phi_{amb}}{k_{res} / \\phi_{res}}}$$

    Common laboratory fluid systems:
      - Air-Brine (porous plate / centrifuge): $\\sigma_{lab} \\approx 72$ dynes/cm, $\\theta_{lab} = 0^\\circ$
      - Mercury-Air (MICP): $\\sigma_{lab} \\approx 485$ dynes/cm, $\\theta_{lab} = 140^\\circ$
      - Lab Oil-Brine: $\\sigma_{lab} \\approx 48$ dynes/cm, $\\theta_{lab} = 30^\\circ$

    Args:
        pc_lab (float or array-like): Laboratory capillary pressure (psi).
        ift_lab (float, optional): Laboratory interfacial tension (dynes/cm). Defaults to 72.0.
        theta_lab (float, optional): Laboratory contact angle (degrees). Defaults to 0.0.
        ift_res (float, optional): Reservoir interfacial tension (dynes/cm). Defaults to 30.0.
        theta_res (float, optional): Reservoir contact angle (degrees). Defaults to 30.0.
        poro_amb (float or array-like, optional): Ambient lab porosity (frac). Defaults to None.
        perm_amb (float or array-like, optional): Ambient lab permeability (mD). Defaults to None.
        poro_res (float or array-like, optional): Reservoir stress-corrected porosity (frac).
            Defaults to None.
        perm_res (float or array-like, optional): Reservoir stress-corrected permeability (mD).
            Defaults to None.

    Returns:
        float or np.ndarray: Reservoir capillary pressure $P_{c,res}$ (psi).
    """
    denom_lab = max(float(ift_lab) * abs(np.cos(np.radians(theta_lab))), 1e-9)
    num_res = float(ift_res) * abs(np.cos(np.radians(theta_res)))
    fluid_scale = num_res / denom_lab

    stress_scale = 1.0
    if (
        poro_amb is not None
        and perm_amb is not None
        and poro_res is not None
        and perm_res is not None
    ):
        rqi_amb = np.maximum(perm_amb, 1e-9) / np.maximum(poro_amb, 1e-6)
        rqi_res = np.maximum(perm_res, 1e-9) / np.maximum(poro_res, 1e-6)
        stress_scale = np.sqrt(rqi_amb / rqi_res)

    return np.maximum(pc_lab * fluid_scale * stress_scale, 0.0)


def correct_core_data(
    df: pd.DataFrame,
    net_confining_stress: Optional[Union[float, str, np.ndarray, pd.Series]] = None,
    mean_pore_pressure: Optional[Union[float, str, np.ndarray, pd.Series]] = None,
    cbw: Optional[Union[bool, float, str, np.ndarray, pd.Series]] = None,
    compaction_coeff: float = 4e-6,
    perm_compaction_coeff: Optional[float] = None,
    ambient_stress: float = 0.0,
    compaction_method: str = "exponential",
    b_factor: Optional[Union[float, str, np.ndarray, pd.Series]] = None,
    klinkenberg_method: str = "jones_owens",
    oven_dried: bool = False,
    correct_dean_stark: bool = False,
    dean_stark_to_effective: bool = True,
    reported_on_humidity_poro: bool = False,
    poro_col: str = "CPORE",
    perm_col: str = "CPERM",
    sw_col: str = "CSW",
    replace_ori: bool = False,
) -> pd.DataFrame:
    """Apply SCAL overburden, clay-bound water, Dean-Stark, and Klinkenberg corrections to a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing core data (`poro_col`, `perm_col`, and/or `sw_col`).
        net_confining_stress (float, str, or array-like, optional): Net confining stress (psi)
            or column name in `df`. If provided, applies `overburden_porosity_correction` (and
            `overburden_permeability_correction` if `perm_compaction_coeff` is specified).
        mean_pore_pressure (float, str, or array-like, optional): Mean flowing pore pressure (psi)
            or column name in `df`. If provided, applies `klinkenberg_permeability_correction`.
        cbw (bool, float, str, or array-like, optional): Clay-bound water volume fraction (v/v),
            column name in `df`, or True to infer `PHIT - PHIE` from `df`. If provided,
            applies `humidity_to_total_porosity`.
        compaction_coeff (float, optional): Pore compaction coefficient for overburden
            porosity correction. Defaults to 4e-6.
        perm_compaction_coeff (float, optional): Permeability stress-sensitivity coefficient $\\gamma$
            for overburden permeability correction when `net_confining_stress` is provided.
            Defaults to None.
        ambient_stress (float, optional): Lab confining stress (psi). Defaults to 0.0.
        compaction_method (str, optional): `'exponential'`, `'power_law'`, or `'kozeny_carman'`.
            Defaults to `'exponential'`.
        b_factor (float, str, or array-like, optional): Klinkenberg slippage factor (psi).
            Defaults to None.
        klinkenberg_method (str, optional): `'jones_owens'` or `'sampath_keighin'`.
            Defaults to `'jones_owens'`.
        oven_dried (bool, optional): Passed to `humidity_to_total_porosity`. Defaults to False.
        correct_dean_stark (bool, optional): If True and `sw_col` is present in `df`, applies
            `dean_stark_cbw_correction` using `cbw` or `PHIT`/`PHIE`. Defaults to False.
        dean_stark_to_effective (bool, optional): Passed as `to_effective` to
            `dean_stark_cbw_correction`. Defaults to True.
        reported_on_humidity_poro (bool, optional): Passed to `dean_stark_cbw_correction`.
            Defaults to False.
        poro_col (str, optional): Core porosity column name. Defaults to `'CPORE'`.
        perm_col (str, optional): Core permeability column name. Defaults to `'CPERM'`.
        sw_col (str, optional): Core water saturation column name. Defaults to `'CSW'`.
        replace_ori (bool, optional): If False, preserves uncorrected columns with `_AMB`,
            `_AIR`, and `_RAW` suffixes. Defaults to False.

    Returns:
        pd.DataFrame: DataFrame with corrected core porosity, permeability, and/or saturation.
    """
    df = df.copy()

    def _resolve_arg(val):
        if isinstance(val, str) and val in df.columns:
            return df[val]
        return val

    cbw_val = None
    if cbw is not None:
        if cbw is True:
            if "PHIT" in df.columns and "PHIE" in df.columns:
                cbw_val = (df["PHIT"] - df["PHIE"]).clip(lower=0.0)
            else:
                raise ValueError(
                    "Cannot infer CBW when cbw=True: 'PHIT' and 'PHIE' columns are required."
                )
        else:
            cbw_val = _resolve_arg(cbw)

    poro_amb = None
    poro_res = None
    if poro_col in df.columns and (net_confining_stress is not None or cbw is not None):
        if not replace_ori and f"{poro_col}_AMB" not in df.columns:
            df[f"{poro_col}_AMB"] = df[poro_col]

        poro = pd.to_numeric(df[poro_col], errors="coerce")
        poro_amb = poro.copy()
        if net_confining_stress is not None:
            ncs = _resolve_arg(net_confining_stress)
            poro_method = (
                "power_law"
                if compaction_method.strip().lower() == "power_law"
                else "exponential"
            )
            poro = overburden_porosity_correction(
                poro,
                ncs,
                compaction_coeff=compaction_coeff,
                ambient_stress=ambient_stress,
                method=poro_method,
            )
        poro_res = poro.copy()
        if cbw_val is not None:
            poro = humidity_to_total_porosity(poro, cbw_val, oven_dried=oven_dried)
        df[poro_col] = poro

    if perm_col in df.columns and (
        mean_pore_pressure is not None
        or (net_confining_stress is not None and perm_compaction_coeff is not None)
    ):
        if not replace_ori and f"{perm_col}_AIR" not in df.columns:
            df[f"{perm_col}_AIR"] = df[perm_col]

        perm = pd.to_numeric(df[perm_col], errors="coerce")
        if mean_pore_pressure is not None:
            mpp = _resolve_arg(mean_pore_pressure)
            b_val = _resolve_arg(b_factor) if b_factor is not None else None
            poro_for_klink = (
                pd.to_numeric(df[poro_col], errors="coerce")
                if poro_col in df.columns
                else None
            )
            perm = pd.Series(
                np.where(
                    perm.notna(),
                    klinkenberg_permeability_correction(
                        perm,
                        mpp,
                        b_factor=b_val,
                        method=klinkenberg_method,
                        poro=poro_for_klink,
                    ),
                    np.nan,
                ),
                index=df.index,
            )
        if net_confining_stress is not None and perm_compaction_coeff is not None:
            ncs = _resolve_arg(net_confining_stress)
            perm = pd.Series(
                np.where(
                    perm.notna(),
                    overburden_permeability_correction(
                        perm,
                        ncs,
                        gamma=perm_compaction_coeff,
                        ambient_stress=ambient_stress,
                        method=compaction_method,
                        core_poro=poro_amb,
                        res_poro=poro_res,
                        poro_compaction_coeff=compaction_coeff,
                    ),
                    np.nan,
                ),
                index=df.index,
            )
        df[perm_col] = perm

    if correct_dean_stark and sw_col in df.columns:
        if not replace_ori and f"{sw_col}_RAW" not in df.columns:
            df[f"{sw_col}_RAW"] = df[sw_col]
        csw = pd.to_numeric(df[sw_col], errors="coerce")
        phit_ref = (
            df["PHIT"]
            if "PHIT" in df.columns
            else (df[poro_col] if poro_col in df.columns else None)
        )
        phie_ref = df["PHIE"] if "PHIE" in df.columns else None
        df[sw_col] = dean_stark_cbw_correction(
            csw,
            cbw=cbw_val,
            phit=phit_ref,
            phie=phie_ref,
            to_effective=dean_stark_to_effective,
            reported_on_humidity_poro=reported_on_humidity_poro,
        )

    return df

