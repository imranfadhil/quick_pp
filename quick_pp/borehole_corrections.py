"""Borehole environmental corrections, per-sample uncertainty and confidence.

Generic (tool-agnostic) physics for wireline (``WL``) and logging-while-drilling
(``LWD``) data in water-based (``WBM``) and oil/synthetic-based (``OBM``/``SBM``)
muds, onshore and offshore.

Design rules
------------
* Tool differences live in :class:`ToolProfile` data, not in code branches.
  Only ``generic`` WL and LWD profiles ship; their constants are documented
  placeholders until vendor chart tables are loaded.
* Raw curves are never modified unless ``inplace=True``.
* Every corrected curve gets a standard deviation (``<LOG>_SIGMA``) built from
  measurement precision, correction model error, uncertainty of the inputs that
  were assumed rather than measured, and hole quality. Confidence is derived
  from that sigma, so it can be propagated or used as a weight downstream.
* Vendor density (RHOB) already contains the spine-and-ribs (DRHO) compensation,
  so no second caliper-based correction is applied when DRHO is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from quick_pp import logger
from quick_pp.environment import (
    PPG_PER_GCC,
    Environment,
    c_to_f,
    environment_from_config,
    mud_weight_to_ppg,
    rw_from_salinity,
)

# --------------------------------------------------------------------------- #
# Tool profiles
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ToolProfile:
    """Generic tool description. Values are placeholders, not vendor charts."""

    name: str
    gr_tool_diameter_in: float
    gr_position: str
    gr_mu: float
    nphi_hole_coeff: float  # v/v per inch of washout (applied as a subtraction)
    hole_span_factor: (
        float  # excess caliper = factor * washout_thold -> hole penalty 1.0
    )
    hole_weight: Dict[str, float]
    sigma_meas: Dict[str, float]  # measurement precision (RT in log10 decades)
    tolerance: Dict[str, float]  # usable tolerance used to scale confidence
    max_trusted_delta: Dict[str, float]  # largest correction still trusted
    res_tool: str  # 'induction' | 'laterolog' | 'propagation'
    notes: str = ""


_COMMON_TOL = {"GR": 15.0, "NPHI": 0.04, "RHOB": 0.06, "RT": 0.25, "DT": 10.0}
_COMMON_MAXD = {"GR": 25.0, "NPHI": 0.06, "RHOB": 0.12, "RT": 0.30, "DT": 15.0}

TOOL_PROFILES: Dict[str, ToolProfile] = {
    "WL": ToolProfile(
        name="WL",
        gr_tool_diameter_in=3.625,
        gr_position="eccentered",
        gr_mu=0.055,
        nphi_hole_coeff=-0.012,
        hole_span_factor=3.0,
        hole_weight={"GR": 0.5, "NPHI": 0.9, "RHOB": 1.0, "RT": 0.2, "DT": 0.8},
        sigma_meas={"GR": 3.0, "NPHI": 0.010, "RHOB": 0.015, "RT": 0.02, "DT": 2.0},
        tolerance=_COMMON_TOL,
        max_trusted_delta=_COMMON_MAXD,
        res_tool="induction",
        notes="Generic wireline profile (pad density, eccentered GR/neutron).",
    ),
    "LWD": ToolProfile(
        name="LWD",
        gr_tool_diameter_in=6.75,
        gr_position="centered",
        gr_mu=0.055,
        nphi_hole_coeff=-0.008,
        hole_span_factor=3.5,
        hole_weight={"GR": 0.4, "NPHI": 0.7, "RHOB": 0.8, "RT": 0.2, "DT": 0.7},
        sigma_meas={"GR": 4.0, "NPHI": 0.012, "RHOB": 0.020, "RT": 0.02, "DT": 3.0},
        tolerance=_COMMON_TOL,
        max_trusted_delta=_COMMON_MAXD,
        res_tool="propagation",
        notes="Generic LWD profile (collar tool, small annulus, rotating).",
    ),
}
_TOOL_ALIASES = {"WL": "WL", "WIRELINE": "WL", "LWD": "LWD", "MWD": "LWD"}

# Generic excess-hole geometric factors (conductivity-space per unit (d^2-ref^2)/ref^2).
_RES_HOLE_G0 = {
    "induction": {"deep": 0.0012, "medium": 0.0025, "shallow": 0.0050},
    "laterolog": {"deep": 0.0008, "medium": 0.0018, "shallow": 0.0040},
    "propagation": {"deep": 0.0010, "medium": 0.0020, "shallow": 0.0045},
}
# Generic radial characteristic radii (in) of the step-profile response.
_RES_RADIUS_IN = {"shallow": 10.0, "medium": 30.0, "deep": 80.0}
_DI_GRID_IN = np.geomspace(6.0, 120.0, 48)
_DI_ASSUMED_IN = 35.0


def get_tool_profile(tool_type: str | ToolProfile = "WL") -> ToolProfile:
    if isinstance(tool_type, ToolProfile):
        return tool_type
    key = _TOOL_ALIASES.get(str(tool_type).strip().upper())
    if key is None:
        raise ValueError("tool_type must be one of 'WL' or 'LWD'.")
    return TOOL_PROFILES[key]


# --------------------------------------------------------------------------- #
# Single-curve physics
# --------------------------------------------------------------------------- #


def _wrap(result, template):
    if isinstance(template, pd.Series):
        return pd.Series(result, index=template.index, name=template.name)
    if np.ndim(template) == 0:
        return float(result)
    return result


def gr_borehole_correction(
    gr,
    cali,
    bitsize=8.5,
    mud_weight=9.0,
    tool_diameter: float = 3.625,
    tool_position: str = "eccentered",
    mu: float = 0.055,
    reference_hole_in=None,
    mud_potassium_pct=0.0,
):
    """Correct gamma ray for mud attenuation over the borehole annulus.

    The attenuation exponent is ``mu * geom * (rho_mud * (d_hole - d_tool) -
    rho_ref * (d_ref - d_tool))`` (cm, g/cc). By default the reference hole is the
    bit size at 9 ppg, so a gauge hole in 9 ppg mud is unchanged; pass
    ``reference_hole_in`` (for example the tool calibration hole) to reference a
    fixed hole instead.

    ``mud_potassium_pct`` (KCl mud) removes the potassium contribution, roughly
    ``16 API per % K`` for a gauge hole, scaled by the fractional annulus volume.
    The factor is clipped to [0.5, 2.5].
    """
    gr_arr = np.asarray(gr, dtype=float)
    cali_arr = np.asarray(cali, dtype=float)
    bs_arr = np.asarray(bitsize, dtype=float)
    mw_arr = np.asarray(mud_weight, dtype=float)

    rho_mud = np.where(mw_arr > 3.0, mw_arr / PPG_PER_GCC, mw_arr)
    rho_mud_ref = 9.0 / PPG_PER_GCC
    d_hole = np.clip(
        np.where(np.isnan(cali_arr), bs_arr, cali_arr), tool_diameter, 36.0
    )
    ref = bs_arr if reference_hole_in is None else np.asarray(reference_hole_in, float)
    d_ref = np.clip(ref, tool_diameter, 36.0)
    geom = 0.65 if str(tool_position).lower().startswith("ecc") else 1.0
    exponent = (
        mu
        * 1.27
        * geom
        * (rho_mud * (d_hole - tool_diameter) - rho_mud_ref * (d_ref - tool_diameter))
    )
    cf = np.clip(np.exp(exponent), 0.5, 2.5)
    corr = gr_arr * cf

    k_pct = np.asarray(mud_potassium_pct, dtype=float)
    if np.any(k_pct > 0):
        annulus = np.clip(
            (d_hole**2 - tool_diameter**2) / np.maximum(d_hole**2, 1e-6), 0.0, 1.0
        )
        corr = corr - 16.0 * k_pct * annulus
    corr = np.maximum(0.0, corr)
    return _wrap(corr, gr)


def nphi_environmental_correction(
    nphi,
    cali=8.5,
    bitsize=8.5,
    temp=75.0,
    pressure=0.0,
    mud_weight=9.0,
    salinity_ppm=0.0,
    standoff: float = 0.0,
    hole_coeff: float = -0.012,
    formation_salinity_ppm=0.0,
):
    """Additive generic environmental correction of thermal neutron porosity.

    Terms (v/v): hole size ``hole_coeff`` per inch of washout (default -0.012),
    temperature +0.004 per 100 degF above 75, pressure -0.0015 per 1000 psi, mud
    weight +0.002 per ppg above 9, borehole and formation salinity +0.001 per
    50 000 ppm each, standoff -0.030 per inch. ``temp`` is degF, ``pressure`` psi.
    These are generic placeholders. The NPHI lithology matrix of the raw curve is
    not converted here.
    """
    nphi_arr = np.asarray(nphi, dtype=float)
    cali_arr = np.asarray(cali, dtype=float)
    bs_arr = np.asarray(bitsize, dtype=float)
    temp_arr = np.asarray(temp, dtype=float)
    pres_arr = np.asarray(pressure, dtype=float)
    sal_arr = np.asarray(salinity_ppm, dtype=float)
    fsal_arr = np.asarray(formation_salinity_ppm, dtype=float)

    d_hole = np.where(np.isnan(cali_arr), bs_arr, cali_arr)
    delta_hole = np.clip(d_hole - bs_arr, -4.0, 16.0)
    mw_ppg = mud_weight_to_ppg(mud_weight)

    corr = (
        hole_coeff * delta_hole
        + 0.004 * ((np.where(np.isnan(temp_arr), 75.0, temp_arr) - 75.0) / 100.0)
        - 0.0015 * (np.where(np.isnan(pres_arr), 0.0, pres_arr) / 1000.0)
        + 0.002 * (np.where(np.isnan(mw_ppg), 9.0, mw_ppg) - 9.0)
        + 0.001 * (np.where(np.isnan(sal_arr), 0.0, sal_arr) / 50000.0)
        + 0.001 * (np.where(np.isnan(fsal_arr), 0.0, fsal_arr) / 50000.0)
        - 0.030 * max(float(standoff), 0.0)
    )
    return _wrap(np.clip(nphi_arr + corr, -0.05, 1.0), nphi)


def density_mudcake_correction(
    rhob,
    cali,
    bitsize=8.5,
    rho_mudcake: float = 1.5,
    drho=None,
    compensated: Optional[bool] = None,
    residual_drho_gain: float = 0.0,
    drho_limit: float = 0.15,
):
    """Residual correction of bulk density for mud between pad and formation.

    Vendor RHOB already includes the DRHO compensation. When ``compensated`` is
    true (default whenever ``drho`` is supplied) no further caliper term is added,
    which avoids correcting the same effect twice; large ``|DRHO|`` only lowers
    confidence. An optional ``residual_drho_gain`` (default 0) adds a fraction of
    the DRHO excess over ``drho_limit``.

    For uncompensated density (no DRHO) a two-layer estimate is used:
    ``delta = 0.06 * wall_gap_in * max(rhob - rho_mudcake, 0)``.
    """
    rhob_arr = np.asarray(rhob, dtype=float)
    cali_arr = np.asarray(cali, dtype=float)
    bs_arr = np.asarray(bitsize, dtype=float)
    if compensated is None:
        compensated = drho is not None
    delta = np.zeros_like(rhob_arr, dtype=float)
    if not compensated:
        gap = np.clip(
            (np.where(np.isnan(cali_arr), bs_arr, cali_arr) - bs_arr) / 2.0, 0.0, 6.0
        )
        delta = delta + 0.06 * gap * np.maximum(rhob_arr - rho_mudcake, 0.0)
    if drho is not None and residual_drho_gain:
        d = np.asarray(drho, dtype=float)
        excess = np.where(
            np.abs(d) > drho_limit,
            residual_drho_gain * (np.abs(d) - drho_limit) * np.sign(d),
            0.0,
        )
        delta = delta + np.where(np.isnan(excess), 0.0, excess)
    return _wrap(np.clip(rhob_arr + delta, 1.0, 3.2), rhob)


def resistivity_borehole_correction(
    r, cali, rm, g0: float, mud_is_oil=False, ref_hole_in=8.5
):
    """Remove excess borehole (washout mud) contribution from apparent resistivity.

    Conductivity-space mixing over the washout annulus above ``ref_hole_in`` (bit size):
    ``Ca = (1-g) Ct + g Cm`` with ``g = g0 * max((d^2 - ref^2) / ref^2, 0)``
    (clipped to 0.15). In-gauge holes (``d <= ref``) and oil/synthetic mud (no
    conductive mud, ``rm`` NaN) return the input unchanged.
    """
    r_arr = np.clip(np.asarray(r, dtype=float), 1e-3, 1e5)
    cali_arr = np.asarray(cali, dtype=float)
    rm_arr = np.asarray(rm, dtype=float)
    ref_arr = np.maximum(np.asarray(ref_hole_in, dtype=float), 1.0)
    oil = np.asarray(mud_is_oil, dtype=bool)
    ok = np.isfinite(rm_arr) & (rm_arr > 0) & np.isfinite(cali_arr) & ~oil
    d_arr = np.where(np.isfinite(cali_arr), cali_arr, ref_arr)
    excess_area = np.clip((d_arr**2 - ref_arr**2) / (ref_arr**2), 0.0, 4.0)
    g = np.clip(g0 * excess_area, 0.0, 0.15)
    ca = 1.0 / r_arr
    cm = 1.0 / np.where(ok, rm_arr, 1.0)
    ct = (ca - g * cm) / (1.0 - g)
    ct = np.maximum(ct, 0.65 * ca)
    out = np.where(ok, 1.0 / ct, r_arr)
    return np.where(np.isfinite(np.asarray(r, dtype=float)), out, np.nan)


def _solve_step_profile(cond: Dict[str, np.ndarray], di_grid: np.ndarray):
    """Least-squares (Ct, Cxo, di) for samples that all have the same curve set."""
    names = list(cond)
    n = len(next(iter(cond.values())))
    g = di_grid[None, :]
    a11 = a12 = a22 = b1 = b2 = 0.0
    for name in names:
        c = cond[name][:, None]
        j = 1.0 - np.exp(-((g / _RES_RADIUS_IN[name]) ** 2))
        u = (1.0 - j) / c
        v = j / c
        a11 = a11 + u * u
        a12 = a12 + u * v
        a22 = a22 + v * v
        b1 = b1 + u
        b2 = b2 + v
    det = a11 * a22 - a12 * a12
    scale = np.maximum(a11 * a22, 1e-300)
    good_det = np.abs(det) > 1e-10 * scale
    safe = np.where(good_det, det, 1.0)
    ct = (a22 * b1 - a12 * b2) / safe
    cxo = (a11 * b2 - a12 * b1) / safe
    res = 0.0
    for name in names:
        c = cond[name][:, None]
        j = 1.0 - np.exp(-((g / _RES_RADIUS_IN[name]) ** 2))
        res = res + (1.0 - ((1.0 - j) * ct + j * cxo) / c) ** 2
    valid = good_det & (ct > 0) & (cxo > 0)
    cost = np.where(valid, res + 1e-6 * np.abs(np.log(g / _DI_ASSUMED_IN)), np.inf)
    best = np.argmin(cost, axis=1)
    rows = np.arange(n)
    ok = np.isfinite(cost[rows, best])
    return (
        np.where(ok, ct[rows, best], np.nan),
        np.where(ok, cxo[rows, best], np.nan),
        np.where(ok, di_grid[best], np.nan),
        np.where(ok, np.sqrt(res[rows, best] / len(names)), np.nan),
    )


def invert_invasion_profile(rt, rm=None, rxo=None) -> Dict[str, np.ndarray]:
    """Invert deep/medium/shallow resistivity for Rt, Rxo and invasion diameter.

    A step-profile model with generic radial responses is used,
    ``C_k = (1 - J_k(di)) Ct + J_k(di) Cxo`` with ``J_k = 1 - exp(-(di/r_k)^2)``.
    With three curves di is searched on a grid; with two curves di is fixed at
    35 in (state 2). Samples that cannot be inverted keep the measured deep value
    (state 3). Returns ``rt``, ``rxo``, ``di_in``, ``misfit`` (RMS relative
    conductivity residual) and ``state`` (0 none, 1 inverted, 2 assumed di, 3 failed).
    """
    rt_arr = np.clip(np.asarray(rt, dtype=float), 1e-3, 1e5)
    n = rt_arr.size
    curves = {"deep": rt_arr}
    if rm is not None:
        curves["medium"] = np.clip(np.asarray(rm, dtype=float), 1e-3, 1e5)
    if rxo is not None:
        curves["shallow"] = np.clip(np.asarray(rxo, dtype=float), 1e-3, 1e5)
    out = {
        "rt": rt_arr.copy(),
        "rxo": np.full(n, np.nan),
        "di_in": np.full(n, np.nan),
        "misfit": np.full(n, np.nan),
        "state": np.zeros(n, dtype=int),
    }
    if len(curves) < 2:
        return out
    avail = np.stack([np.isfinite(v) for v in curves.values()])
    count = avail.sum(axis=0)
    names = list(curves)
    for cnt, grid, ok_state in (
        (3, _DI_GRID_IN, 1),
        (2, np.array([_DI_ASSUMED_IN]), 2),
    ):
        if cnt > len(names):
            continue
        sel_cnt = count == cnt
        if not sel_cnt.any():
            continue
        # group by exact availability pattern
        patterns = {tuple(avail[:, i]) for i in np.flatnonzero(sel_cnt)}
        for pat in patterns:
            if not pat[0]:
                continue  # deep curve required
            idx = np.flatnonzero(sel_cnt & np.all(avail.T == np.array(pat), axis=1))
            if idx.size == 0:
                continue
            cond_all = {
                nm: 1.0 / curves[nm][idx]
                for nm, p in zip(names, pat, strict=False)
                if p
            }
            cd_all = cond_all["deep"]
            max_sep = np.zeros(idx.size, dtype=float)
            for nm, c_arr in cond_all.items():
                if nm != "deep":
                    max_sep = np.maximum(max_sep, np.abs(np.log10(c_arr / cd_all)))
            # Skip step-profile inversion when curves agree within tool precision (~7%)
            need_inv = max_sep >= 0.03
            if not need_inv.any():
                continue
            sub_idx = idx[need_inv]
            cond = {nm: c_arr[need_inv] for nm, c_arr in cond_all.items()}
            ct, cxo, di, mis = _solve_step_profile(cond, grid)
            cd = cond["deep"]
            c_shal = cond["shallow"] if "shallow" in cond else cond["medium"]
            max_ct_dec = 0.75 if cnt == 3 else 0.35
            max_cxo_dec = 0.85 if cnt == 3 else 0.50
            with np.errstate(divide="ignore", invalid="ignore"):
                plausible = (
                    np.isfinite(ct)
                    & np.isfinite(cxo)
                    & (ct > 0)
                    & (cxo > 0)
                    & (np.abs(np.log10(ct / cd)) <= max_ct_dec)
                    & (np.abs(np.log10(cxo / c_shal)) <= max_cxo_dec)
                )
            out["rt"][sub_idx[plausible]] = 1.0 / ct[plausible]
            out["rxo"][sub_idx[plausible]] = 1.0 / cxo[plausible]
            out["di_in"][sub_idx[plausible]] = di[plausible]
            out["misfit"][sub_idx[plausible]] = mis[plausible]
            out["state"][sub_idx[plausible]] = ok_state
            out["state"][sub_idx[~plausible]] = 3
    out["rt"] = np.where(np.isfinite(np.asarray(rt, dtype=float)), out["rt"], np.nan)
    return out


def resistivity_invasion_correction(rt, rm=None, rxo=None):
    """Invasion-corrected deep resistivity and ``INVASION_FLAG``.

    Uses :func:`invert_invasion_profile`. Flag 1: Rxo < Rt (conductive filtrate
    profile), 2: Rxo > Rt (resistive profile), 0: negligible or not invertible.
    The flag is geometric only; use :func:`classify_invasion_profile` for the
    Rmf/Rw interpretation.
    """
    inv = invert_invasion_profile(rt, rm=rm, rxo=rxo)
    ratio = np.log10(inv["rxo"] / inv["rt"])
    flag = np.zeros(len(inv["rt"]), dtype=int)
    done = np.isin(inv["state"], (1, 2))
    flag[done & (ratio < -0.06)] = 1
    flag[done & (ratio > 0.06)] = 2
    rt_corr = inv["rt"]
    if isinstance(rt, pd.Series):
        return (
            pd.Series(rt_corr, index=rt.index, name=rt.name),
            pd.Series(flag, index=rt.index, name="INVASION_FLAG"),
        )
    if np.ndim(rt) == 0:
        return float(rt_corr[0]), int(flag[0])
    return rt_corr, flag


def classify_invasion_profile(rt, rxo, rmf, rw) -> Dict[str, np.ndarray]:
    """Interpret the observed Rxo/Rt against the expected Rmf/Rw contrast.

    In a clean water zone ``Rxo/Rt ~ Rmf/Rw`` (both at formation temperature); a
    hydrocarbon zone gives a lower ratio by ``(Sw/Sxo)^n``. Codes
    (``INVASION_PROFILE``): 0 unknown/uninformative (Rmf ~ Rw or not inverted),
    1 water-like, 2 hydrocarbon-like, 3 anomalous (more contrast than possible:
    check depth shift, shoulder beds, tool), 4 no invasion despite expected
    contrast (tight/impermeable).
    """
    rt = np.asarray(rt, dtype=float)
    rxo = np.asarray(rxo, dtype=float)
    obs = np.log10(rxo / rt)
    exp = np.log10(np.asarray(rmf, dtype=float) / np.asarray(rw, dtype=float))
    exp = exp * np.ones_like(obs)
    code = np.zeros(obs.shape, dtype=int)
    valid = np.isfinite(obs) & np.isfinite(exp) & (np.abs(exp) >= 0.2)
    diff = obs - exp
    code[valid & (diff >= -0.15) & (diff <= 0.18)] = 1
    code[valid & (diff < -0.15)] = 2
    code[valid & (diff > 0.18)] = 3
    code[valid & (np.abs(obs) < 0.06)] = 4
    return {"code": code, "expected_log_ratio": exp, "observed_log_ratio": obs}


# --------------------------------------------------------------------------- #
# Signal-quality flags
# --------------------------------------------------------------------------- #


def detect_spikes(values, tol: float, window: int = 11, k: float = 6.0) -> np.ndarray:
    """Isolated single-sample spikes relative to a rolling median."""
    s = pd.Series(np.asarray(values, dtype=float))
    med = s.rolling(window, center=True, min_periods=3).median()
    dev = (s - med).abs()
    mad = dev.rolling(window, center=True, min_periods=3).median()
    thr = np.maximum(k * 1.4826 * mad, 0.5 * tol)
    out = (dev > thr).fillna(False)
    prev = out.shift(1, fill_value=False)
    nxt = out.shift(-1, fill_value=False)
    return (out & ~prev & ~nxt).to_numpy()


def detect_flatline(values, window: int = 15) -> np.ndarray:
    """Samples inside a run of ``window`` identical values (stuck tool or padded data)."""
    s = pd.Series(np.asarray(values, dtype=float))
    flat_end = (
        (
            s.rolling(window, min_periods=window).max()
            == s.rolling(window, min_periods=window).min()
        )
        & s.notna()
    ).astype(float)
    return (flat_end[::-1].rolling(window, min_periods=1).max()[::-1] > 0).to_numpy()


def detect_sonic_cycle_skip(dt, rhob=None, window: int = 21) -> np.ndarray:
    """Positive DT excursions of >20 % and >15 us/ft over the rolling median.

    When RHOB is available, the excursion must also exceed the Gardner estimate
    (``rho = 0.23 Vp^0.25``, Vp in ft/s) by 25 us/ft.
    """
    s = pd.Series(np.asarray(dt, dtype=float))
    med = s.rolling(window, center=True, min_periods=5).median()
    excess = s - med
    cs = ((excess / med) > 0.20) & (excess > 15.0)
    if rhob is not None:
        rho = np.asarray(rhob, dtype=float)
        with np.errstate(invalid="ignore", divide="ignore"):
            dt_gardner = 1e6 / (np.clip(rho, 1.0, 3.2) / 0.23) ** 4
        known = np.isfinite(dt_gardner)
        cs = cs & (~known | ((s.to_numpy() - dt_gardner) > 25.0))
    return cs.fillna(False).to_numpy()


# --------------------------------------------------------------------------- #
# Confidence framework
# --------------------------------------------------------------------------- #

ENV_QC_STATUS = {-1: "NO_DATA", 0: "UNCHANGED", 1: "CORRECTED", 2: "UNRELIABLE"}
ENV_QC_REASON_BITS = {
    "WASHOUT": 1,  # CALI - BS >= washout_thold
    "DRHO_HIGH": 2,  # |DRHO| > drho limit
    "BADHOLE": 4,  # upstream BADHOLE flag
    "INVASION": 8,  # INVASION_FLAG > 0
    "NO_CALIPER": 16,  # no caliper: hole size unknown, confidence reduced
    "SPIKE": 32,  # isolated spike on any corrected curve
    "FLATLINE": 64,  # stuck / padded values on any corrected curve
    "CYCLE_SKIP": 128,  # suspected sonic cycle skip
    "BS_ASSUMED": 256,  # bit size neither measured nor in the mud program
    "RES_MUD_MISMATCH": 512,  # laterolog in non-conductive mud
}
_HOLE_SEVERITY_EDGES = (0.15, 0.40, 0.75)
_MODEL_ERROR_FRACTION = 0.25  # fraction of |correction| treated as model error


def _perturbation_sigma(
    fn: Callable[..., Any], base: Dict[str, Any], perts: List[Dict[str, Any]]
):
    """Root-sum-square change of ``fn`` for each perturbed input set."""
    ref = np.asarray(fn(**base), dtype=float)
    acc = np.zeros_like(ref)
    for p in perts:
        alt = np.asarray(fn(**{**base, **p}), dtype=float)
        d = np.nan_to_num(alt - ref, nan=0.0)
        acc = acc + d * d
    return np.sqrt(acc)


@dataclass
class _CurveResult:
    corrected: pd.Series
    sigma_corr: pd.Series  # input + model error (curve units; RT decades)
    extra_penalty_sigma: Optional[pd.Series] = None


def _legacy_environment(
    df: pd.DataFrame,
    bitsize_default: float,
    mud_weight,
    surface_temp_f: float,
    temp_grad_f_per_100ft: float,
    salinity,
    mud_type: str,
    depth_unit: str,
    formation_salinity_ppm,
) -> Environment:
    """Environment from the legacy scalar arguments (degF, degF/100 ft)."""
    cfg: Dict[str, Any] = {
        "surface_temp": surface_temp_f,
        "surface_temp_unit": "degF",
        "temp_gradient_c_per_km": temp_grad_f_per_100ft
        * (5.0 / 9.0)
        * (1000.0 / 30.48),
        "mud_type": mud_type,
        "formation_salinity_ppm": formation_salinity_ppm,
    }
    env = environment_from_config(cfg, depth_unit="ft" if depth_unit == "ft" else "m")
    env.default_bit_size_in = float(bitsize_default or 8.5)
    if isinstance(mud_weight, (int, float)) and np.isfinite(mud_weight):
        env.default_mud_weight_ppg = float(mud_weight_to_ppg(mud_weight))
        env.explicit["mud_weight"] = True
    if isinstance(salinity, (int, float)) and np.isfinite(salinity) and salinity > 0:
        env.default_mud_salinity_ppm = float(salinity)
        env.explicit["mud_salinity"] = True
    # legacy path: treat non-default scalar temperature inputs as explicit
    env.explicit["temperature"] = not (
        np.isclose(surface_temp_f, 75.0) and np.isclose(temp_grad_f_per_100ft, 1.5)
    )
    env.assumptions = [
        a
        for a in env.assumptions
        if "gradient provided" not in a and "Surface temperature" not in a
    ]
    return env


def _estimate_bit_size(df: pd.DataFrame, default: float) -> Optional[np.ndarray]:
    """Per-section bit size from CALI via ``badhole_flagging`` (needs sorted depth)."""
    if (
        "CALI" not in df.columns
        or not pd.to_numeric(df["CALI"], errors="coerce").notna().any()
    ):
        return None
    depth = df["DEPTH"] if "DEPTH" in df.columns else None
    if depth is None or not pd.Series(depth).is_monotonic_increasing:
        return None
    try:
        from quick_pp.qaqc import badhole_flagging

        tmp = df.drop(columns=["BS"], errors="ignore").reset_index(drop=True)
        res = badhole_flagging(
            tmp, auto_bitsize=True, bitsize_default=default, use_cali=True
        )
        if len(res) == len(df) and "BS" in res.columns:
            return res["BS"].to_numpy(dtype=float)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug(f"[env_corr] bit size estimation failed: {exc}")
    return None


def _first_col(df: pd.DataFrame, names) -> Optional[str]:
    return next((c for c in names if c in df.columns), None)


def apply_environmental_corrections(
    df: pd.DataFrame,
    bitsize_default: float = 8.5,
    mud_weight: float | str | None = 9.0,
    surface_temp: float = 75.0,
    temp_grad: float = 1.5,
    salinity_ppm: float | str | None = 0.0,
    rho_mudcake: float = 1.5,
    inplace: bool = False,
    suffix: str = "_ENV_CORR",
    depth_unit: str = "auto",
    mud_type: str = "WBM",
    washout_thold: float = 2.0,
    good_conf: float = 0.8,
    unreliable_conf: float = 0.4,
    qc_columns: bool = True,
    *,
    tool_type: str | ToolProfile = "WL",
    environment: Optional[Environment] = None,
    formation_salinity_ppm: Optional[float] = None,
    resistivity_curves: Optional[Dict[str, str]] = None,
    res_tool: Optional[str] = None,
    drho_limit: float = 0.15,
    auto_bitsize: bool = True,
    temp_unit: str = "degF",
    gr_mud_potassium_pct: float = 0.0,
) -> pd.DataFrame:
    """Apply generic borehole environmental and invasion corrections.

    Corrected curves: ``GR``, ``NPHI``, ``RHOB``, ``RT`` (+ ``DT`` with cycle-skip
    samples removed) as ``<LOG><suffix>``. With ``qc_columns=True`` each also gets
    ``<LOG>_CORR_DELTA`` (RT in log10 decades), ``<LOG>_SIGMA`` (1-sigma in curve
    units, RT in decades), ``<LOG>_QC_CONF`` (0-1) and ``<LOG>_QC_STATUS`` (see
    ``ENV_QC_STATUS``), plus ``ENV_QC_REASON`` (bit mask, ``ENV_QC_REASON_BITS``),
    ``HOLE_SEVERITY`` (0-3), ``ENV_TEMP_C``, ``ENV_RM``, ``ENV_RMF``, ``ENV_RW``,
    and, when resistivity is present, ``RXO_EST``, ``DI_EST``, ``INVASION_FLAG``
    and ``INVASION_PROFILE``.

    Environment: pass ``environment`` (see :mod:`quick_pp.environment`) for a mud
    program, offshore water depth and BHT calibration, otherwise one is built from
    the legacy scalar arguments (``surface_temp`` degF, ``temp_grad`` degF/100 ft,
    ``mud_weight`` ppg or column name, ``salinity_ppm`` ppm or column name).
    Assumed inputs are listed in ``result.attrs['env_assumptions']`` and widen
    ``<LOG>_SIGMA``.

    Args:
        tool_type: ``'WL'`` or ``'LWD'`` generic profile.
        mud_type: ``'WBM'``, ``'OBM'`` or ``'SBM'`` default for the whole well; a
            mud program in ``environment`` can vary it by section.
        depth_unit: ``'m'``, ``'ft'`` or the legacy ``'auto'`` guess
            (max depth < 6000 means metres; avoid).
        temp_unit: unit of an existing ``TEMP`` column (``'degF'`` or ``'degC'``).
        resistivity_curves: optional ``{"deep","medium","shallow": column}`` map;
            defaults to ``RT`` and common aliases.
        res_tool: ``'induction'``, ``'laterolog'`` or ``'propagation'`` (defaults to
            the tool profile). Laterolog in OBM is flagged unreliable.
        auto_bitsize: estimate ``BS`` from ``CALI`` when no BS column or mud
            program bit size exists (needs depth sorted ascending).
    """
    profile = get_tool_profile(tool_type)
    mud_type = str(mud_type).upper()
    if mud_type not in ("WBM", "OBM", "SBM"):
        raise ValueError("mud_type must be one of 'WBM', 'OBM', 'SBM'.")
    depth_unit = str(depth_unit).lower()
    if depth_unit not in ("auto", "m", "ft"):
        raise ValueError("depth_unit must be one of 'auto', 'm', 'ft'.")
    temp_unit = str(temp_unit).lower().replace("deg", "")
    if temp_unit not in ("f", "c"):
        raise ValueError("temp_unit must be 'degF' or 'degC'.")
    res_tool = (res_tool or profile.res_tool).lower()
    if res_tool not in _RES_HOLE_G0:
        raise ValueError("res_tool must be 'induction', 'laterolog' or 'propagation'.")

    out_df = df.copy()
    n = len(out_df)
    idx = out_df.index
    notes: List[str] = []

    # ---- depth ---------------------------------------------------------------
    depth_col = (
        "TVD"
        if "TVD" in out_df.columns
        else ("DEPTH" if "DEPTH" in out_df.columns else None)
    )
    d_vals = (
        pd.to_numeric(out_df[depth_col], errors="coerce").to_numpy(dtype=float)
        if depth_col
        else np.zeros(n)
    )
    if depth_unit == "auto":
        eff_unit = "m" if np.nanmax(np.nan_to_num(d_vals, nan=0.0)) < 6000 else "ft"
        notes.append(
            f"Depth unit guessed as '{eff_unit}' from magnitude; set it explicitly."
        )
    else:
        eff_unit = depth_unit
    if depth_col == "DEPTH":
        notes.append(
            "No TVD curve: measured depth was used as TVD (deviated wells will be biased)."
        )
    if depth_col is None:
        notes.append(
            "No depth curve: temperature and pressure fall back to the surface values."
        )

    # ---- environment ---------------------------------------------------------
    sal_scalar = salinity_ppm if not isinstance(salinity_ppm, str) else None
    mw_scalar = mud_weight if not isinstance(mud_weight, str) else None
    if environment is None:
        env = _legacy_environment(
            out_df,
            bitsize_default,
            mw_scalar,
            surface_temp,
            temp_grad,
            sal_scalar,
            mud_type,
            eff_unit,
            formation_salinity_ppm if formation_salinity_ppm is not None else 35000.0,
        )
    else:
        env = environment
        if mud_type != "WBM":
            env.default_mud_type = mud_type
        env.depth_unit = eff_unit
        if formation_salinity_ppm is not None:
            env.formation_salinity_ppm = float(formation_salinity_ppm)
    notes = list(env.assumptions) + notes

    depth_m = env.to_metres(d_vals)
    temp_c = env.temperature_c(depth_m)
    if (
        "TEMP" in out_df.columns
        and pd.to_numeric(out_df["TEMP"], errors="coerce").notna().any()
    ):
        t_col = pd.to_numeric(out_df["TEMP"], errors="coerce").to_numpy(dtype=float)
        t_col_c = (t_col - 32.0) * 5.0 / 9.0 if temp_unit == "f" else t_col
        temp_c = np.where(np.isfinite(t_col_c), t_col_c, temp_c)
        temp_explicit = True
    else:
        temp_explicit = bool(env.explicit.get("temperature"))
    temp_f = c_to_f(temp_c)

    md_vals = (
        pd.to_numeric(out_df["DEPTH"], errors="coerce").to_numpy(dtype=float)
        if "DEPTH" in out_df.columns
        else d_vals
    )
    mud = env.mud_arrays(md_vals, temp_c)
    is_oil = mud["is_oil"]
    mw_ppg = mud["mud_weight_ppg"].copy()
    if isinstance(mud_weight, str) and mud_weight in out_df.columns:
        col = pd.to_numeric(out_df[mud_weight], errors="coerce").to_numpy(dtype=float)
        mw_ppg = np.where(np.isfinite(col), mud_weight_to_ppg(col), mw_ppg)
        mw_explicit = True
    else:
        mw_explicit = bool(env.explicit.get("mud_weight"))
    if not mw_explicit:
        notes.append(f"Mud weight assumed {env.default_mud_weight_ppg:.1f} ppg.")
    sal = mud["salinity_ppm"].copy()
    if isinstance(salinity_ppm, str) and salinity_ppm in out_df.columns:
        col = pd.to_numeric(out_df[salinity_ppm], errors="coerce").to_numpy(dtype=float)
        sal = np.where(np.isfinite(col), col, sal)
        sal_explicit = True
    else:
        sal_explicit = bool(env.explicit.get("mud_salinity"))
    sal = np.where(is_oil, 0.0, sal)
    pressure = (
        pd.to_numeric(out_df["PRESSURE"], errors="coerce").to_numpy(dtype=float)
        if "PRESSURE" in out_df.columns
        else env.borehole_pressure_psi(depth_m, mw_ppg)
    )

    # ---- bit size / caliper -------------------------------------------------
    bs_s = (
        pd.to_numeric(out_df["BS"], errors="coerce").where(lambda s: s > 0)
        if "BS" in out_df.columns
        else pd.Series(np.nan, index=idx)
    )
    prog_bs = pd.Series(
        np.where(mud["bit_size_explicit"], mud["bit_size_in"], np.nan), index=idx
    )
    bs_s = prog_bs.combine_first(bs_s)
    if bs_s.isna().any() and auto_bitsize:
        est = _estimate_bit_size(out_df, float(env.default_bit_size_in))
        if est is not None:
            bs_s = bs_s.fillna(pd.Series(est, index=idx))
    bs_assumed = bs_s.isna() & (not env.explicit.get("bit_size", False))
    bs = bs_s.fillna(float(env.default_bit_size_in))
    if bool(bs_assumed.any()):
        notes.append(
            f"Bit size assumed {env.default_bit_size_in:g} in where BS was unknown."
        )
    cali_raw = (
        pd.to_numeric(out_df["CALI"], errors="coerce")
        if "CALI" in out_df.columns
        else pd.Series(np.nan, index=idx)
    )
    no_cali = cali_raw.isna()
    cali = cali_raw.fillna(bs)
    n_no_cali = int(no_cali.sum())
    if n_no_cali > 0 and n > 0:
        pct_no_cali = 100.0 * n_no_cali / n
        if n_no_cali == n:
            notes.append(
                f"Caliper missing for all samples (100%, {n_no_cali}/{n}): "
                "hole size unknown, confidence reduced."
            )
        else:
            pct_str = "<0.1%" if pct_no_cali < 0.1 else f"{pct_no_cali:.1f}%"
            notes.append(
                f"Caliper missing for {pct_str} of samples ({n_no_cali}/{n}): "
                "hole size unknown, confidence reduced."
            )

    drho_col = _first_col(out_df, ("DRHO", "HDRA", "DCOR"))
    drho = pd.to_numeric(out_df[drho_col], errors="coerce") if drho_col else None

    # ---- hole quality ---------------------------------------------------------
    excess = (cali - bs).clip(lower=0.0).fillna(0.0)
    span = max(profile.hole_span_factor * washout_thold, 1e-6)
    washout = excess >= washout_thold
    drho_abs = drho.abs() if drho is not None else pd.Series(np.nan, index=idx)
    drho_high = (drho_abs > drho_limit).fillna(False)
    drho_pen = ((drho_abs - drho_limit) / 0.20).clip(lower=0.0, upper=1.0).fillna(0.0)
    badhole = (
        pd.to_numeric(out_df["BADHOLE"], errors="coerce").fillna(0).ge(1)
        if "BADHOLE" in out_df.columns
        else pd.Series(False, index=idx)
    )
    hole_pen = np.maximum((excess / span).clip(upper=1.0), drho_pen.to_numpy())
    hole_pen = np.where(badhole, np.maximum(hole_pen, 0.7), hole_pen)
    no_cali_floor = np.where(drho_abs.isna(), 0.35, 0.2)
    hole_pen = np.where(no_cali, np.maximum(hole_pen, no_cali_floor), hole_pen)
    hole_pen = pd.Series(hole_pen, index=idx)
    severity = pd.Series(np.digitize(hole_pen, _HOLE_SEVERITY_EDGES), index=idx)

    # ---- per-curve corrections -----------------------------------------------
    results: Dict[str, _CurveResult] = {}
    mw_unc = 0.5 if mw_explicit else 1.5
    t_unc_f = (5.0 if temp_explicit else 15.0) * 1.8
    bs_unc = np.where(bs_assumed.to_numpy(), 1.0, 0.25)
    sal_unc = 0.2 if sal_explicit else 0.5

    if "GR" in out_df.columns:
        base = {
            "gr": out_df["GR"].to_numpy(dtype=float),
            "cali": cali.to_numpy(),
            "bitsize": bs.to_numpy(),
            "mud_weight": mw_ppg,
            "tool_diameter": profile.gr_tool_diameter_in,
            "tool_position": profile.gr_position,
            "mu": profile.gr_mu,
            "mud_potassium_pct": gr_mud_potassium_pct,
        }
        corr = gr_borehole_correction(**base)
        sig = _perturbation_sigma(
            lambda **kw: gr_borehole_correction(**kw),
            base,
            [{"mud_weight": mw_ppg + mw_unc}, {"bitsize": bs.to_numpy() + bs_unc}],
        )
        results["GR"] = _CurveResult(
            pd.Series(corr, index=idx), pd.Series(sig, index=idx)
        )

    if "NPHI" in out_df.columns:
        base = {
            "nphi": out_df["NPHI"].to_numpy(dtype=float),
            "cali": cali.to_numpy(),
            "bitsize": bs.to_numpy(),
            "temp": temp_f,
            "pressure": pressure,
            "mud_weight": mw_ppg,
            "salinity_ppm": sal,
            "hole_coeff": profile.nphi_hole_coeff,
            "formation_salinity_ppm": env.formation_salinity_ppm,
        }
        corr = nphi_environmental_correction(**base)
        sig = _perturbation_sigma(
            lambda **kw: nphi_environmental_correction(**kw),
            base,
            [
                {"temp": temp_f + t_unc_f},
                {"mud_weight": mw_ppg + mw_unc},
                {"salinity_ppm": sal * (1.0 + sal_unc)},
                {"bitsize": bs.to_numpy() + bs_unc},
            ],
        )
        results["NPHI"] = _CurveResult(
            pd.Series(corr, index=idx), pd.Series(sig, index=idx)
        )

    if "RHOB" in out_df.columns:
        compensated = drho is not None
        base = {
            "rhob": out_df["RHOB"].to_numpy(dtype=float),
            "cali": cali.to_numpy(),
            "bitsize": bs.to_numpy(),
            "rho_mudcake": rho_mudcake,
            "drho": drho.to_numpy(dtype=float) if drho is not None else None,
            "compensated": compensated,
            "drho_limit": drho_limit,
        }
        corr = density_mudcake_correction(**base)
        sig = _perturbation_sigma(
            lambda **kw: density_mudcake_correction(**kw),
            base,
            [{"bitsize": bs.to_numpy() + bs_unc}, {"rho_mudcake": rho_mudcake + 0.2}],
        )
        results["RHOB"] = _CurveResult(
            pd.Series(corr, index=idx), pd.Series(sig, index=idx)
        )

    dt_col = _first_col(out_df, ("DTC", "DT"))
    cycle_skip = np.zeros(n, dtype=bool)
    if dt_col:
        dt_raw = pd.to_numeric(out_df[dt_col], errors="coerce")
        rhob_for_cs = (
            pd.to_numeric(out_df["RHOB"], errors="coerce").to_numpy(dtype=float)
            if "RHOB" in out_df.columns
            else None
        )
        cycle_skip = detect_sonic_cycle_skip(dt_raw.to_numpy(dtype=float), rhob_for_cs)
        dt_corr = dt_raw.where(~pd.Series(cycle_skip, index=idx))
        results["DT"] = _CurveResult(dt_corr, pd.Series(0.0, index=idx))

    # ---- resistivity ----------------------------------------------------------
    rw_t = rw_from_salinity(temp_c, env.formation_salinity_ppm)
    inv_flag = inv_profile = None
    res_mismatch = np.zeros(n, dtype=bool)
    rc = resistivity_curves or {}
    deep_col = rc.get("deep") or ("RT" if "RT" in out_df.columns else None)
    if deep_col and deep_col in out_df.columns:
        med_col = rc.get("medium") or _first_col(out_df, ("RM", "ILM", "RMED", "RLA3"))
        sha_col = rc.get("shallow") or _first_col(
            out_df, ("RXO", "MSFL", "RSHAL", "RS", "RLA1")
        )
        g0 = _RES_HOLE_G0[res_tool]
        rm_t = mud["rm"]
        res_mismatch = np.full(n, res_tool == "laterolog") & is_oil

        def _bh(col, key):
            return resistivity_borehole_correction(
                pd.to_numeric(out_df[col], errors="coerce").to_numpy(dtype=float),
                cali.to_numpy(),
                rm_t,
                g0[key],
                is_oil,
                ref_hole_in=bs.to_numpy(),
            )

        deep_bh = _bh(deep_col, "deep")
        med_bh = _bh(med_col, "medium") if med_col else None
        sha_bh = _bh(sha_col, "shallow") if sha_col else None
        inv = invert_invasion_profile(deep_bh, rm=med_bh, rxo=sha_bh)
        flag_arr = np.zeros(n, dtype=int)
        done = np.isin(inv["state"], (1, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.log10(inv["rxo"] / inv["rt"])
        flag_arr[done & (ratio < -0.06)] = 1
        flag_arr[done & (ratio > 0.06)] = 2
        inv_flag = flag_arr
        rt_solved = np.where((inv["state"] == 2) & (inv_flag == 0), deep_bh, inv["rt"])
        rxo_fallback = sha_bh if sha_bh is not None else np.full(n, np.nan)
        rxo_clean = np.where(inv_flag > 0, inv["rxo"], rxo_fallback)
        di_clean = np.where(inv_flag > 0, inv["di_in"], np.nan)
        prof = classify_invasion_profile(rt_solved, rxo_clean, mud["rmf"], rw_t)
        inv_profile = prof["code"]
        rt_corr = pd.Series(rt_solved, index=idx)
        # sigma in log10 decades: misfit + assumed-di + failed-inversion terms + unknown Rm
        sig_inv = np.zeros(n)
        sig_inv += np.nan_to_num(inv["misfit"], nan=0.0) / np.log(10.0)
        sig_inv = np.sqrt(
            sig_inv**2
            + np.where((inv["state"] == 2) & (inv_flag > 0), 0.08, 0.0) ** 2
            + np.where(inv["state"] == 3, 0.15, 0.0) ** 2
            + np.where(~mud["rm_known"] & ~is_oil, 0.02, 0.0) ** 2
        )
        results["RT"] = _CurveResult(rt_corr, pd.Series(sig_inv, index=idx))
        out_df["RXO_EST"] = rxo_clean
        out_df["DI_EST"] = di_clean
        out_df["INVASION_FLAG"] = inv_flag
        out_df["INVASION_PROFILE"] = inv_profile
        if bool(is_oil.any()):
            notes.append(
                "Oil/synthetic mud: Rmf/Rw invasion interpretation is not applicable (profile code 0)."
            )
        if bool(res_mismatch.any()):
            notes.append("Laterolog resistivity in non-conductive mud is unreliable.")
        if not bool(mud["rmf_known"].any()) and not bool(is_oil.all()):
            notes.append(
                "Rmf not provided; estimated from mud salinity for the invasion profile."
            )
        results["RT"].extra_penalty_sigma = pd.Series(
            np.where(res_mismatch, 2.0 * profile.tolerance["RT"], 0.0), index=idx
        )

    # ---- write corrected curves ------------------------------------------------
    for log, res in results.items():
        if suffix:
            out_df[f"{log}{suffix}"] = res.corrected
        if inplace and log != "DT":
            out_df[deep_col if log == "RT" else log] = res.corrected

    # ---- QC columns ----------------------------------------------------------------
    spike_any = np.zeros(n, dtype=bool)
    flat_any = np.zeros(n, dtype=bool)
    if qc_columns:
        raw_cols = {
            "GR": "GR",
            "NPHI": "NPHI",
            "RHOB": "RHOB",
            "RT": deep_col,
            "DT": dt_col,
        }
        for log, res in results.items():
            src = raw_cols.get(log)
            if not src or src not in df.columns:
                continue
            raw = pd.to_numeric(df[src], errors="coerce")
            corr = pd.to_numeric(res.corrected, errors="coerce")
            if log == "RT":
                delta = np.log10(corr.clip(lower=1e-3)) - np.log10(raw.clip(lower=1e-3))
            else:
                delta = corr - raw
            tol = profile.tolerance[log]
            max_delta = profile.max_trusted_delta[log]
            sig_meas = profile.sigma_meas[log]
            sig_corr = np.sqrt(
                res.sigma_corr.to_numpy() ** 2
                + (_MODEL_ERROR_FRACTION * delta.abs().fillna(0.0).to_numpy()) ** 2
            )
            sig_hole = hole_pen.to_numpy() * profile.hole_weight[log] * max_delta
            spike = detect_spikes(raw.to_numpy(dtype=float), tol)
            flat = detect_flatline(raw.to_numpy(dtype=float))
            spike_any |= spike
            flat_any |= flat
            sig_sig = np.where(spike, 2.0 * tol, 0.0)
            sig_sig = np.sqrt(sig_sig**2 + np.where(flat, 1.5 * tol, 0.0) ** 2)
            sig_cs = np.where(cycle_skip, 2.0 * tol, 0.0) if log == "DT" else 0.0
            sig_extra = (
                res.extra_penalty_sigma.to_numpy()
                if res.extra_penalty_sigma is not None
                else 0.0
            )
            sigma = np.sqrt(
                sig_meas**2
                + sig_corr**2
                + sig_hole**2
                + sig_sig**2
                + sig_cs**2
                + sig_extra**2
            )
            conf = 1.0 / (1.0 + (sigma / tol) ** 2)
            conf = pd.Series(conf, index=idx).where(raw.notna() & corr.notna(), 0.0)
            changed = (delta.abs() / max_delta) > 0.05
            status = pd.Series(1, index=idx, dtype=int)
            status[(conf >= good_conf) & ~changed.fillna(False)] = 0
            status[conf < unreliable_conf] = 2
            status[raw.isna()] = -1
            out_df[f"{log}_CORR_DELTA"] = delta
            out_df[f"{log}_SIGMA"] = pd.Series(sigma, index=idx).round(5)
            out_df[f"{log}_QC_CONF"] = conf.round(3)
            out_df[f"{log}_QC_STATUS"] = status.astype(int)

        invaded = (
            pd.Series(inv_flag, index=idx).gt(0)
            if inv_flag is not None
            else pd.Series(False, index=idx)
        )
        bits = ENV_QC_REASON_BITS
        reason = (
            washout.astype(int) * bits["WASHOUT"]
            + drho_high.astype(int) * bits["DRHO_HIGH"]
            + badhole.astype(int) * bits["BADHOLE"]
            + invaded.astype(int) * bits["INVASION"]
            + no_cali.astype(int) * bits["NO_CALIPER"]
            + pd.Series(spike_any, index=idx).astype(int) * bits["SPIKE"]
            + pd.Series(flat_any, index=idx).astype(int) * bits["FLATLINE"]
            + pd.Series(cycle_skip, index=idx).astype(int) * bits["CYCLE_SKIP"]
            + bs_assumed.astype(int) * bits["BS_ASSUMED"]
            + pd.Series(res_mismatch, index=idx).astype(int) * bits["RES_MUD_MISMATCH"]
        )
        out_df["ENV_QC_REASON"] = reason.astype(int)
        out_df["HOLE_SEVERITY"] = severity.astype(int)
        out_df["ENV_TEMP_C"] = np.round(temp_c, 2)
        out_df["ENV_BS"] = pd.Series(bs, index=idx).round(3)
        out_df["ENV_RM"] = pd.Series(mud["rm"], index=idx).round(5)
        out_df["ENV_RMF"] = pd.Series(mud["rmf"], index=idx).round(5)
        out_df["ENV_RW"] = pd.Series(rw_t, index=idx).round(5)
    out_df.attrs["env_assumptions"] = list(dict.fromkeys(notes))
    out_df.attrs["env_tool_profile"] = profile.name
    return out_df


# --------------------------------------------------------------------------- #
# Confidence-weighted curve for downstream modules
# --------------------------------------------------------------------------- #


def confidence_weighted_curve(
    values,
    conf,
    depth=None,
    good_conf: float = 0.8,
    min_conf: float = 0.2,
    window: int = 41,
) -> pd.Series:
    """Blend a curve with a local trend according to per-sample confidence.

    ``out = w * value + (1 - w) * trend`` where the trend is the rolling median of
    high-confidence samples (confidence >= ``good_conf``) interpolated along depth,
    and ``w`` is the confidence rescaled so ``min_conf`` maps to 0 and
    ``good_conf`` to 1. Samples below ``min_conf`` take the trend entirely. Samples
    without any trend (no high-confidence neighbours) keep their value. This lets
    downstream modules use all data while down-weighting unreliable intervals
    instead of applying a binary badhole mask.
    """
    v = pd.Series(np.asarray(values, dtype=float))
    c = pd.Series(np.asarray(conf, dtype=float)).fillna(0.0).clip(0.0, 1.0)
    if depth is not None:
        d = np.asarray(depth, dtype=float)
        order = np.argsort(d, kind="stable")
    else:
        d = np.arange(len(v), dtype=float)
        order = np.arange(len(v))
    vs, cs = v.iloc[order].reset_index(drop=True), c.iloc[order].reset_index(drop=True)
    good = vs.where(cs >= good_conf)
    trend = good.rolling(window, center=True, min_periods=3).median()
    trend = trend.interpolate(limit_direction="both", limit_area=None)
    w = ((cs - min_conf) / max(good_conf - min_conf, 1e-6)).clip(0.0, 1.0)
    blended = w * vs + (1.0 - w) * trend
    blended = blended.where(trend.notna(), vs)
    blended = blended.where(vs.notna(), np.nan)
    out = pd.Series(np.nan, index=range(len(v)), dtype=float)
    out.iloc[order] = blended.to_numpy()
    out.index = v.index
    return out
