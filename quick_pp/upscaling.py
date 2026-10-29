"""Log Blocking & Upscaling utility for Static Geomodeling & Dynamic Reservoir Simulation Handover.

Provides petrophysically rigorous upscaling (blocking) of fine-scale CPI well logs onto
cellular grid resolutions (fixed cell thickness, explicit depth edges, or geological zones),
supporting:
  - Thickness-weighted arithmetic averaging for porosity ($\\phi$), shale/mineral volumes ($V_{sh}$, $V_{clay}$),
    bulk density ($\\rho_b$), and net-to-gross flags ($NTG$).
  - Pore-volume weighted (saturation-weighted) arithmetic averaging for water saturation ($S_w$, $S_{wt}$, $S_{we}$):
      $$\\overline{S_w} = \\frac{\\sum_i \\phi_i S_{w,i} h_i}{\\sum_i \\phi_i h_i}$$
    which strictly preserves bulk hydrocarbon pore volume ($BVHC = \\overline{\\phi}(1 - \\overline{S_w})$).
  - Directional & Cardwell-Parsons (1945) power-law averaging for permeability ($k$):
      * Arithmetic mean ($\\omega = 1$): horizontal / parallel-to-bedding flow ($k_h$)
      * Geometric mean ($\\omega \\to 0$): effective heterogeneous 2D/3D random flow ($k_{eff}$)
      * Harmonic mean ($\\omega = -1$): vertical / perpendicular-to-bedding flow ($k_v$)
      * Power-law mean ($\\omega \\in [-1, 1]$): $\\overline{k}_{\\omega} = \\left( \\frac{\\sum_i h_i k_i^{\\omega}}{\\sum_i h_i} \\right)^{1/\\omega}$
      * Vertical-to-horizontal anisotropy ratio ($k_v / k_h = \\overline{k}_{harm} / \\overline{k}_{arith}$)
  - Majority-vote (thickness-weighted mode) upscaling for discrete facies / rock types ($RT$, $FZI\\_RT$, etc.).
  - Step-profile expansion (`expand_blocked_logs` / `return_step_log=True`) to map blocked cellular
    properties back onto fine-scale depth grids for QA/QC overlay plots and LAS export.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quick_pp import logger
from quick_pp.ressum import flag_interval

__all__ = [
    "arith_upscale",
    "block_logs",
    "cardwell_parsons_bounds",
    "expand_blocked_logs",
    "geom_upscale",
    "harm_upscale",
    "majority_upscale",
    "plot_upscaled_logs",
    "pore_volume_upscale",
    "power_law_upscale",
    "upscale_logs",
    "upscale_permeability",
]


def _infer_sample_thickness(depth: np.ndarray, h: Optional[Union[np.ndarray, float]] = None) -> np.ndarray:
    """Infer per-sample interval thickness `h` from a 1D depth array if not explicitly provided."""
    depth_arr = np.asarray(depth, dtype=float)
    n = len(depth_arr)
    if n == 0:
        return np.empty(0, dtype=float)
    if h is not None:
        h_arr = np.asarray(h, dtype=float)
        if h_arr.ndim == 0:
            return np.full(n, float(h_arr), dtype=float)
        return h_arr.copy()

    if n == 1:
        return np.ones(1, dtype=float)

    diffs = np.diff(depth_arr)
    pos_diffs = np.abs(diffs[np.isfinite(diffs) & (np.abs(diffs) > 0)])
    median_step = float(np.median(pos_diffs)) if len(pos_diffs) > 0 else 1.0

    # Midpoint cell thickness for each sample
    dh = np.empty(n, dtype=float)
    dh[0] = abs(depth_arr[1] - depth_arr[0]) if np.isfinite(depth_arr[1] - depth_arr[0]) else median_step
    dh[-1] = abs(depth_arr[-1] - depth_arr[-2]) if np.isfinite(depth_arr[-1] - depth_arr[-2]) else median_step
    if n > 2:
        dh[1:-1] = 0.5 * np.abs(depth_arr[2:] - depth_arr[:-2])
    dh = np.where(np.isfinite(dh) & (dh > 0), dh, median_step)
    return dh


def arith_upscale(
    values: Union[np.ndarray, pd.Series, Sequence[float]],
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
) -> float:
    """Compute thickness-weighted arithmetic average (e.g., for $\\phi$, $V_{sh}$, $\\rho_b$, or $k_h$).

    $$\\overline{x}_{arith} = \\frac{\\sum_i h_i x_i}{\\sum_i h_i}$$

    Args:
        values: Fine-scale log values.
        h: Per-sample thickness weights. Defaults to uniform weights if None.

    Returns:
        float: Thickness-weighted arithmetic mean, or `np.nan` if no valid samples exist.
    """
    v = np.asarray(values, dtype=float)
    w = np.ones_like(v, dtype=float) if h is None else np.broadcast_to(np.asarray(h, dtype=float), v.shape)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float(np.nan)
    return float(np.sum(v[mask] * w[mask]) / np.sum(w[mask]))


def pore_volume_upscale(
    sw: Union[np.ndarray, pd.Series, Sequence[float]],
    phi: Union[np.ndarray, pd.Series, Sequence[float]],
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
) -> float:
    """Compute pore-volume (saturation-weighted) arithmetic average for water or hydrocarbon saturation.

    $$\\overline{S_w} = \\frac{\\sum_i \\phi_i S_{w,i} h_i}{\\sum_i \\phi_i h_i}$$

    This guarantees conservation of hydrocarbon pore volume ($BVHC = \\overline{\\phi} \\cdot (1 - \\overline{S_w})$)
    when upscaling from fine-scale CPI logs to static geomodel cells. If total pore volume in the
    interval is zero, falls back to thickness-weighted arithmetic mean of `sw`.

    Args:
        sw: Fine-scale saturation values ($S_w$, $S_{wt}$, or $S_{we}$).
        phi: Corresponding fine-scale porosity values ($\\phi_t$ or $\\phi_e$).
        h: Per-sample thickness weights. Defaults to uniform weights if None.

    Returns:
        float: Pore-volume weighted average saturation, or `np.nan` if no valid samples exist.
    """
    s = np.asarray(sw, dtype=float)
    p = np.asarray(phi, dtype=float)
    w = np.ones_like(s, dtype=float) if h is None else np.broadcast_to(np.asarray(h, dtype=float), s.shape)

    mask = np.isfinite(s) & np.isfinite(p) & np.isfinite(w) & (w > 0) & (p >= 0)
    if not np.any(mask):
        return arith_upscale(s, w)

    pv = p[mask] * w[mask]
    pv_sum = float(np.sum(pv))
    if pv_sum > 0:
        return float(np.sum(pv * s[mask]) / pv_sum)
    return arith_upscale(s[mask], w[mask])


def geom_upscale(
    perm: Union[np.ndarray, pd.Series, Sequence[float]],
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
    min_perm: Optional[float] = None,
) -> float:
    """Compute thickness-weighted geometric average for permeability ($k_{eff}$, $\\omega \\to 0$).

    $$\\overline{k}_{geom} = \\exp\\left( \\frac{\\sum_i h_i \\ln(k_i)}{\\sum_i h_i} \\right)$$

    Args:
        perm: Fine-scale permeability values.
        h: Per-sample thickness weights. Defaults to uniform weights if None.
        min_perm: Optional positive floor applied to `perm > 0` or zero values if specified.
                  If None, only strictly positive ($k > 0$) samples are included, or returns 0.0
                  if valid samples exist but none are positive.

    Returns:
        float: Thickness-weighted geometric mean.
    """
    k = np.asarray(perm, dtype=float)
    w = np.ones_like(k, dtype=float) if h is None else np.broadcast_to(np.asarray(h, dtype=float), k.shape)
    valid = np.isfinite(k) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return float(np.nan)

    if min_perm is not None and min_perm > 0:
        k_use = np.maximum(k[valid], float(min_perm))
        w_use = w[valid]
        return float(np.exp(np.sum(w_use * np.log(k_use)) / np.sum(w_use)))

    pos = valid & (k > 0)
    if not np.any(pos):
        return 0.0
    w_pos = w[pos]
    k_pos = k[pos]
    return float(np.exp(np.sum(w_pos * np.log(k_pos)) / np.sum(w_pos)))


def harm_upscale(
    perm: Union[np.ndarray, pd.Series, Sequence[float]],
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
    min_perm: Optional[float] = None,
) -> float:
    """Compute thickness-weighted harmonic average for permeability ($k_v$, $\\omega = -1$).

    $$\\overline{k}_{harm} = \\frac{\\sum_i h_i}{\\sum_i (h_i / k_i)}$$

    Args:
        perm: Fine-scale permeability values.
        h: Per-sample thickness weights. Defaults to uniform weights if None.
        min_perm: Optional positive floor applied to `perm`. If None, only strictly positive
                  ($k > 0$) samples are included, or returns 0.0 if valid samples exist but none are positive.

    Returns:
        float: Thickness-weighted harmonic mean.
    """
    k = np.asarray(perm, dtype=float)
    w = np.ones_like(k, dtype=float) if h is None else np.broadcast_to(np.asarray(h, dtype=float), k.shape)
    valid = np.isfinite(k) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return float(np.nan)

    if min_perm is not None and min_perm > 0:
        k_use = np.maximum(k[valid], float(min_perm))
        w_use = w[valid]
        return float(np.sum(w_use) / np.sum(w_use / k_use))

    pos = valid & (k > 0)
    if not np.any(pos):
        return 0.0
    w_pos = w[pos]
    k_pos = k[pos]
    return float(np.sum(w_pos) / np.sum(w_pos / k_pos))


def power_law_upscale(
    perm: Union[np.ndarray, pd.Series, Sequence[float]],
    omega: float = 0.33,
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
    min_perm: Optional[float] = None,
) -> float:
    """Compute Cardwell-Parsons (1945) power-law average for permeability with exponent $\\omega \\in [-1, 1]$.

    $$\\overline{k}_{\\omega} = \\left( \\frac{\\sum_i h_i k_i^{\\omega}}{\\sum_i h_i} \\right)^{1/\\omega}$$

    Special cases:
      - $\\omega = 1.0$: Arithmetic average (parallel / horizontal flow $k_h$)
      - $\\omega \\to 0.0$: Geometric average (2D isotropic random heterogeneity)
      - $\\omega \\approx 1/3$ ($0.33$): 3D isotropic random heterogeneity (Matheron, 1967)
      - $\\omega = -1.0$: Harmonic average (series / vertical flow $k_v$)

    By Cardwell & Parsons (1945), the effective permeability of any heterogeneous block is bounded
    by the harmonic and arithmetic means:
    $$\\overline{k}_{harm} \\le \\overline{k}_{\\omega} \\le \\overline{k}_{arith} \\quad \\forall \\omega \\in [-1, 1]$$

    Args:
        perm: Fine-scale permeability values.
        omega: Power-law averaging exponent (typically in `[-1.0, 1.0]`). Defaults to `0.33`.
        h: Per-sample thickness weights. Defaults to uniform weights if None.
        min_perm: Optional positive floor applied to `perm` when $\\omega \\le 0$.

    Returns:
        float: Thickness-weighted power-law average permeability.
    """
    omega_val = float(omega)
    if abs(omega_val) < 1e-6:
        return geom_upscale(perm, h=h, min_perm=min_perm)
    if np.isclose(omega_val, 1.0, atol=1e-6):
        if min_perm is not None and min_perm > 0:
            k_arr = np.asarray(perm, dtype=float)
            k_arr = np.where(np.isfinite(k_arr), np.maximum(k_arr, float(min_perm)), np.nan)
            return arith_upscale(k_arr, h=h)
        return arith_upscale(perm, h=h)
    if np.isclose(omega_val, -1.0, atol=1e-6):
        return harm_upscale(perm, h=h, min_perm=min_perm)

    k = np.asarray(perm, dtype=float)
    w = np.ones_like(k, dtype=float) if h is None else np.broadcast_to(np.asarray(h, dtype=float), k.shape)
    valid = np.isfinite(k) & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return float(np.nan)

    if min_perm is not None and min_perm > 0:
        k_use = np.maximum(k[valid], float(min_perm))
        w_use = w[valid]
    else:
        pos = valid & (k > 0)
        if not np.any(pos):
            return 0.0
        k_use = k[pos]
        w_use = w[pos]

    mean_pow = float(np.sum(w_use * np.power(k_use, omega_val)) / np.sum(w_use))
    if mean_pow <= 0:
        return 0.0
    return float(np.power(mean_pow, 1.0 / omega_val))


def cardwell_parsons_bounds(
    perm: Union[np.ndarray, pd.Series, Sequence[float]],
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
    omega: float = 0.33,
    min_perm: Optional[float] = None,
) -> Dict[str, float]:
    """Compute Cardwell-Parsons (1945) permeability bounds and directional upscaling summary.

    Returns harmonic ($k_v$ lower bound), geometric ($k_{eff}$), power-law ($k_{\\omega}$),
    arithmetic ($k_h$ upper bound), and vertical-to-horizontal anisotropy ratio ($k_v / k_h$).

    Args:
        perm: Fine-scale permeability values.
        h: Per-sample thickness weights.
        omega: Power-law exponent $\\omega \\in [-1, 1]$. Defaults to `0.33`.
        min_perm: Optional positive permeability floor.

    Returns:
        dict: `{"k_harm": float, "k_geom": float, "k_power": float, "k_arith": float, "kv_kh_ratio": float}`
    """
    k_arith = arith_upscale(perm, h=h)
    k_geom = geom_upscale(perm, h=h, min_perm=min_perm)
    k_harm = harm_upscale(perm, h=h, min_perm=min_perm)
    k_power = power_law_upscale(perm, omega=omega, h=h, min_perm=min_perm)
    kv_kh = float(k_harm / k_arith) if (np.isfinite(k_harm) and np.isfinite(k_arith) and k_arith > 0) else np.nan
    return {
        "k_harm": k_harm,
        "k_geom": k_geom,
        "k_power": k_power,
        "k_arith": k_arith,
        "kv_kh_ratio": kv_kh,
    }


def upscale_permeability(
    perm: Union[np.ndarray, pd.Series, Sequence[float]],
    method: str = "geometric",
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
    omega: float = 0.33,
    min_perm: Optional[float] = None,
) -> float:
    """Upscale a permeability array using the specified averaging scheme.

    Args:
        perm: Fine-scale permeability values.
        method: One of `'arithmetic'` (`'arith'`, `'kh'`), `'geometric'` (`'geom'`, `'keff'`),
                `'harmonic'` (`'harm'`, `'kv'`), or `'power'` (`'power_law'`, `'cardwell_parsons'`).
        h: Per-sample thickness weights.
        omega: Power-law exponent used when `method` is `'power'` / `'cardwell_parsons'`.
        min_perm: Optional positive floor for permeability.

    Returns:
        float: Upscaled permeability value.
    """
    m = method.strip().lower()
    if m in ("arithmetic", "arith", "am", "kh"):
        return arith_upscale(perm, h=h)
    if m in ("geometric", "geom", "gm", "keff"):
        return geom_upscale(perm, h=h, min_perm=min_perm)
    if m in ("harmonic", "harm", "hm", "kv"):
        return harm_upscale(perm, h=h, min_perm=min_perm)
    if m in ("power", "power_law", "power-law", "cardwell_parsons", "cardwell-parsons"):
        return power_law_upscale(perm, omega=omega, h=h, min_perm=min_perm)
    raise ValueError(
        f"Unknown permeability upscaling method {method!r}. "
        "Choose from 'arithmetic', 'geometric', 'harmonic', or 'power_law'."
    )


def majority_upscale(
    values: Union[np.ndarray, pd.Series, Sequence[Any]],
    h: Optional[Union[np.ndarray, pd.Series, float]] = None,
) -> Any:
    """Compute thickness-weighted majority vote (mode) for categorical/discrete logs (e.g., Rock Type, Facies, Zone).

    Args:
        values: Fine-scale discrete/categorical values.
        h: Per-sample thickness weights.

    Returns:
        The category/value with the largest cumulative thickness in the block, or `np.nan` if empty.
    """
    s = pd.Series(values)
    w = np.ones(len(s), dtype=float) if h is None else np.asarray(h, dtype=float)
    valid = s.notna().to_numpy() & np.isfinite(w) & (w > 0)
    if not np.any(valid):
        return np.nan

    s_valid = s.iloc[np.where(valid)[0]]
    w_valid = w[valid]
    df_vote = pd.DataFrame({"val": s_valid.to_numpy(), "w": w_valid})
    grouped = df_vote.groupby("val", sort=False)["w"].sum()
    return grouped.idxmax()


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """Case-insensitive lookup of the first matching column name in `df`."""
    upper_map = {c.upper(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.upper() in upper_map:
            return upper_map[cand.upper()]
    return None


def upscale_logs(
    df: pd.DataFrame,
    cell_size: Optional[float] = 1.0,
    grid_edges: Optional[Sequence[float]] = None,
    by_zone: bool = False,
    depth_col: str = "DEPTH",
    zone_col: Optional[str] = "ZONES",
    vsh_col: Optional[str] = None,
    phi_col: Optional[str] = None,
    phie_col: Optional[str] = None,
    sw_col: Optional[str] = None,
    swe_col: Optional[str] = None,
    perm_col: Optional[str] = None,
    facies_cols: Optional[Sequence[str]] = None,
    extra_arith_cols: Optional[Sequence[str]] = None,
    perm_method: str = "geometric",
    omega: float = 0.33,
    min_perm: Optional[float] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    net_only: bool = False,
    net_flag_type: str = "reservoir",
    return_step_log: bool = False,
) -> pd.DataFrame:
    """Upscale (block) fine-scale petrophysical CPI logs to static geomodel / simulator cellular grid resolution.

    Applies petrophysically rigorous averaging schemes per grid cell:
      - **Porosity ($\\phi_t$, $\\phi_e$) & Shale/Mineral Volumes ($V_{sh}$, $V_{clay}$, $V_{sand}$, etc.)**:
        Thickness-weighted arithmetic average:
        $$\\overline{\\phi} = \\frac{\\sum h_i \\phi_i}{\\sum h_i}, \\quad \\overline{V_{sh}} = \\frac{\\sum h_i V_{sh,i}}{\\sum h_i}$$
      - **Water Saturation ($S_{wt}$, $S_w$, $S_{we}$)**:
        Pore-volume weighted (saturation-weighted) arithmetic average:
        $$\\overline{S_{wt}} = \\frac{\\sum \\phi_{t,i} S_{wt,i} h_i}{\\sum \\phi_{t,i} h_i}, \\quad \\overline{S_{we}} = \\frac{\\sum \\phi_{e,i} S_{we,i} h_i}{\\sum \\phi_{e,i} h_i}$$
        preserving cell-level hydrocarbon pore volume ($BVHC = \\overline{\\phi_t}(1 - \\overline{S_{wt}})$).
      - **Permeability ($k$)**:
        Outputs directional & Cardwell-Parsons (1945) upscaled permeabilities simultaneously:
          * `<PERM>_ARITH` ($k_h$ horizontal / parallel flow proxy, $\\omega = 1$)
          * `<PERM>_GEOM` ($k_{eff}$ heterogeneous effective flow proxy, $\\omega \\to 0$)
          * `<PERM>_HARM` ($k_v$ vertical / series flow proxy, $\\omega = -1$)
          * `<PERM>_POWER` (Cardwell-Parsons power-law average with exponent `omega`)
          * `KV_KH_RATIO` ($\\overline{k}_{harm} / \\overline{k}_{arith}$ vertical-to-horizontal anisotropy ratio)
          * `<PERM>` set to the user-selected primary `perm_method` (`'geometric'`, `'arithmetic'`, `'harmonic'`, or `'power_law'`).
      - **Facies / Rock Types**:
        Thickness-weighted majority vote (`majority_upscale`).
      - **Net-to-Gross & Pay Summary Flags**:
        Computes cell-level `GROSS`, `NET_ROCK`, `NET_RES`, `NET_PAY`, `NTG` (reservoir net-to-gross),
        and `NTG_PAY` (pay net-to-gross) from `cutoffs`. When `net_only=True`, upscaled reservoir
        properties ($\\phi$, $S_w$, $k$) are averaged strictly over the net flagged samples in each cell
        (standard static geomodel net-cell property handover).

    Grid specification modes:
      1. `grid_edges` provided: Explicit sequence of depth boundaries `[z_0, z_1, ..., z_M]`.
      2. `cell_size` provided (e.g. `1.0` m or `2.0` ft): Regular cells of thickness `cell_size`.
         If `by_zone=True` and `zone_col` is present, cells are partitioned within each geological zone
         so grid blocks never straddle zone boundaries.
      3. `cell_size=None` and `by_zone=True`: One upscaled block per contiguous geological zone.

    Args:
        df (pd.DataFrame): Fine-scale well log DataFrame.
        cell_size (float, optional): Target grid block thickness in depth units. Defaults to `1.0`.
                                     Set to `None` with `by_zone=True` to upscale one block per zone.
        grid_edges (Sequence[float], optional): Explicit depth edges `[top_0, ..., bot_M]`. Overrides `cell_size`.
        by_zone (bool, optional): If True, respect zone boundaries when constructing grid blocks. Defaults to False.
        depth_col (str, optional): Depth column name (`'DEPTH'`, `'TVD'`, `'TVDSS'`). Defaults to `'DEPTH'`.
        zone_col (str, optional): Zone column name. Defaults to `'ZONES'`.
        vsh_col (str, optional): Shale/clay volume column name. Auto-detected (`VSHALE`, `VCLAY`, `VSH`) if None.
        phi_col (str, optional): Total porosity column name. Auto-detected (`PHIT`, `PHI`) if None.
        phie_col (str, optional): Effective porosity column name. Auto-detected (`PHIE`) if None.
        sw_col (str, optional): Total water saturation column name. Auto-detected (`SWT`, `SW`) if None.
        swe_col (str, optional): Effective water saturation column name. Auto-detected (`SWE`) if None.
        perm_col (str, optional): Permeability column name. Auto-detected (`PERM`, `PERM_TRANS`) if None.
        facies_cols (Sequence[str], optional): Categorical/discrete columns to upscale via majority vote.
        extra_arith_cols (Sequence[str], optional): Additional continuous curves to upscale via arithmetic mean.
        perm_method (str, optional): Primary permeability averaging scheme assigned to `<perm_col>`:
                                     `'geometric'`, `'arithmetic'`, `'harmonic'`, or `'power_law'`. Defaults to `'geometric'`.
        omega (float, optional): Cardwell-Parsons power-law exponent $\\omega \\in [-1, 1]$. Defaults to `0.33`.
        min_perm (float, optional): Optional minimum permeability floor for geometric/harmonic/power averages.
        cutoffs (dict, optional): Cutoff dictionary `{'VSHALE': 0.4, 'PHIT': 0.01, 'SWT': 0.9}` for cell NTG & flags.
        net_only (bool, optional): If True, average $\\phi$, $S_w$, and $k$ only over net samples (`net_flag_type`)
                                   within each cell. Defaults to False.
        net_flag_type (str, optional): Which net flag to filter on when `net_only=True`:
                                       `'reservoir'` (default), `'pay'`, or `'rock'`.
        return_step_log (bool, optional): If True, maps the blocked cellular values back onto the original
                                          fine-scale depth rows (step-profile log) with `_BLK` suffixes. Defaults to False.

    Returns:
        pd.DataFrame: Blocked cellular DataFrame (one row per grid cell) or fine-scale step DataFrame if `return_step_log=True`.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    work = df.copy()
    d_col = _find_col(work, [depth_col, "DEPTH", "TVD", "TVDSS", "depth"])
    if d_col is None:
        raise ValueError(f"Depth column {depth_col!r} not found in DataFrame columns: {list(work.columns)}")

    work = work.sort_values(by=d_col, kind="mergesort").reset_index(drop=True)
    depth_vals = work[d_col].to_numpy(dtype=float)
    work["_h"] = _infer_sample_thickness(depth_vals)

    # Identify petrophysical columns
    z_col = _find_col(work, [zone_col, "ZONES", "ZONE", "zones"]) if zone_col else None
    vsh_c = _find_col(work, [vsh_col] if vsh_col else ["VSHALE", "VCLAY", "VSH", "vshale", "vclay"])
    phi_c = _find_col(work, [phi_col] if phi_col else ["PHIT", "PHI", "phit", "phi"])
    phie_c = _find_col(work, [phie_col] if phie_col else ["PHIE", "phie"])
    sw_c = _find_col(work, [sw_col] if sw_col else ["SWT", "SW", "swt", "sw"])
    swe_c = _find_col(work, [swe_col] if swe_col else ["SWE", "swe"])
    perm_c = _find_col(work, [perm_col] if perm_col else ["PERM", "PERM_TRANS", "perm"])

    # Compute fine-scale rock/reservoir/pay flags if Vshale, PHIT, SWT are available
    cutoffs = cutoffs or {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9}
    has_flag_inputs = (vsh_c is not None) and (phi_c is not None) and (sw_c is not None)
    if has_flag_inputs:
        r_flag, res_flag, p_flag = flag_interval(
            work[vsh_c].to_numpy(dtype=float),
            work[phi_c].to_numpy(dtype=float),
            work[sw_c].to_numpy(dtype=float),
            cutoffs,
        )
        work["_rock_flag"] = r_flag
        work["_res_flag"] = res_flag
        work["_pay_flag"] = p_flag
    else:
        work["_rock_flag"] = 1
        work["_res_flag"] = 1
        work["_pay_flag"] = 1

    # Detect additional volumetric/continuous columns to upscale arithmetically
    default_arith_candidates = [
        "VCLAY",
        "VSHALE",
        "VSAND",
        "VSILT",
        "VCALC",
        "VDOLO",
        "VANHY",
        "VGYPS",
        "VHALI",
        "VPYRI",
        "VCOAL",
        "VFELD",
        "VKAOL",
        "GR",
        "RHOB",
        "NPHI",
        "DTC",
        "DTS",
        "PEF",
    ]
    arith_cols: List[str] = []
    for cand in default_arith_candidates + list(extra_arith_cols or []):
        found = _find_col(work, [cand])
        if found and found not in arith_cols and found not in (d_col, z_col, phi_c, phie_c, sw_c, swe_c, perm_c):
            if pd.api.types.is_numeric_dtype(work[found]):
                arith_cols.append(found)

    # Detect discrete/categorical facies or rock-type columns
    default_facies_candidates = ["ROCK_TYPE", "RT", "FZI_RT", "R35_RT", "LUCIA_RT", "FACIES", "LITHOLOGY"]
    disc_cols: List[str] = []
    for cand in default_facies_candidates + list(facies_cols or []):
        found = _find_col(work, [cand])
        # Avoid mistaking resistivity 'RT' for rock type if RT is float resistivity
        if found and found.upper() == "RT" and (facies_cols is None or cand not in facies_cols):
            continue
        if found and found not in disc_cols and found not in (d_col, z_col):
            disc_cols.append(found)

    # Assign cell IDs and (top, bottom) intervals using integer row indices into `work`
    cell_intervals: List[Tuple[int, float, float, Optional[Any], np.ndarray]] = []
    h_full = work["_h"].to_numpy(dtype=float)
    z_full = work[z_col].to_numpy() if z_col is not None else None

    if grid_edges is not None:
        edges = np.asarray(grid_edges, dtype=float)
        if edges.ndim != 1 or len(edges) < 2 or not np.all(np.diff(edges) > 0):
            raise ValueError("`grid_edges` must be a 1D strictly increasing sequence of at least 2 depth values.")
        cell_idx = np.searchsorted(edges, depth_vals, side="right") - 1
        for cid in range(len(edges) - 1):
            mask = (cell_idx == cid) if cid < len(edges) - 2 else ((depth_vals >= edges[cid]) & (depth_vals <= edges[cid + 1]))
            idx_arr = np.flatnonzero(mask)
            if idx_arr.size > 0:
                z_val = majority_upscale(z_full[idx_arr], h_full[idx_arr]) if z_full is not None else None
                cell_intervals.append((cid + 1, float(edges[cid]), float(edges[cid + 1]), z_val, idx_arr))
    elif by_zone and z_col is not None:
        # Partition into contiguous zone segments first
        zone_series = work[z_col].fillna("UNKNOWN")
        zone_change = (zone_series != zone_series.shift(1)).cumsum()
        cid = 1
        for _, z_group in work.groupby(zone_change, sort=False):
            z_indices = work.index.get_indexer(z_group.index)
            z_name = z_group[z_col].iloc[0]
            z_depths = depth_vals[z_indices]
            z_min = float(np.nanmin(z_depths))
            z_max = float(np.nanmax(z_depths))
            if cell_size is None or cell_size <= 0 or (z_max - z_min) <= cell_size:
                cell_intervals.append((cid, z_min, z_max, z_name, z_indices))
                cid += 1
            else:
                edges = np.arange(z_min, z_max + cell_size, float(cell_size))
                if edges[-1] < z_max:
                    edges = np.append(edges, z_max + float(cell_size))
                c_idx = np.clip(np.floor((z_depths - z_min) / float(cell_size)).astype(int), 0, len(edges) - 2)
                for k_idx in range(len(edges) - 1):
                    rel_idx = np.flatnonzero(c_idx == k_idx)
                    if rel_idx.size > 0:
                        top_d = float(edges[k_idx])
                        bot_d = float(min(edges[k_idx + 1], z_max)) if k_idx == len(edges) - 2 else float(edges[k_idx + 1])
                        cell_intervals.append((cid, top_d, bot_d, z_name, z_indices[rel_idx]))
                        cid += 1
    else:
        if cell_size is None or cell_size <= 0:
            raise ValueError("`cell_size` must be positive when `grid_edges` is None and `by_zone` is False.")
        d_min = float(np.nanmin(depth_vals))
        c_idx = np.floor((depth_vals - d_min) / float(cell_size)).astype(int)
        for k_idx in np.unique(c_idx):
            idx_arr = np.flatnonzero(c_idx == k_idx)
            if idx_arr.size > 0:
                top_d = d_min + k_idx * float(cell_size)
                bot_d = top_d + float(cell_size)
                z_val = majority_upscale(z_full[idx_arr], h_full[idx_arr]) if z_full is not None else None
                cell_intervals.append((int(k_idx) + 1, float(top_d), float(bot_d), z_val, idx_arr))

    # Pre-extract NumPy arrays once before the cell loop to avoid per-cell DataFrame slicing overhead
    flag_col_map = {
        "rock": "_rock_flag",
        "reservoir": "_res_flag",
        "res": "_res_flag",
        "pay": "_pay_flag",
    }
    net_flag_col = flag_col_map.get(net_flag_type.strip().lower(), "_res_flag")

    rock_full = work["_rock_flag"].to_numpy(dtype=int)
    res_full = work["_res_flag"].to_numpy(dtype=int)
    pay_full = work["_pay_flag"].to_numpy(dtype=int)
    net_flag_full = work[net_flag_col].to_numpy(dtype=int)
    well_name_val = work["WELL_NAME"].iloc[0] if "WELL_NAME" in work.columns and not work.empty else None

    vsh_full = pd.to_numeric(work[vsh_c], errors="coerce").to_numpy(dtype=float) if vsh_c is not None else None
    phi_full = pd.to_numeric(work[phi_c], errors="coerce").to_numpy(dtype=float) if phi_c is not None else None
    phie_full = pd.to_numeric(work[phie_c], errors="coerce").to_numpy(dtype=float) if phie_c is not None else None
    sw_full = pd.to_numeric(work[sw_c], errors="coerce").to_numpy(dtype=float) if sw_c is not None else None
    swe_full = pd.to_numeric(work[swe_c], errors="coerce").to_numpy(dtype=float) if swe_c is not None else None
    perm_full = pd.to_numeric(work[perm_c], errors="coerce").to_numpy(dtype=float) if perm_c is not None else None
    arith_arrays = {ac: pd.to_numeric(work[ac], errors="coerce").to_numpy(dtype=float) for ac in arith_cols}
    disc_arrays = {dc: work[dc].to_numpy() for dc in disc_cols}

    records: List[Dict[str, Any]] = []
    step_df = work.copy() if return_step_log else None

    for cid, top_d, bot_d, z_val, idx_arr in cell_intervals:
        h_all = h_full[idx_arr]
        gross_h = float(np.nansum(h_all))
        net_rock_h = float(np.nansum(np.where(rock_full[idx_arr] == 1, h_all, 0.0)))
        net_res_h = float(np.nansum(np.where(res_full[idx_arr] == 1, h_all, 0.0)))
        net_pay_h = float(np.nansum(np.where(pay_full[idx_arr] == 1, h_all, 0.0)))

        ntg_res = float(net_res_h / gross_h) if gross_h > 0 else np.nan
        ntg_pay = float(net_pay_h / gross_h) if gross_h > 0 else np.nan

        # Determine active subset for reservoir property upscaling
        if net_only:
            act_idx = idx_arr[net_flag_full[idx_arr] == 1]
        else:
            act_idx = idx_arr
        has_act = act_idx.size > 0
        h_act = h_full[act_idx] if has_act else np.empty(0, dtype=float)

        mid_d = 0.5 * (top_d + bot_d)
        row: Dict[str, Any] = {
            "CELL_ID": cid,
            "TOP": round(top_d, 4),
            "BOTTOM": round(bot_d, 4),
            d_col: round(mid_d, 4),
            "GROSS": round(gross_h, 4),
            "NET": round(net_res_h, 4),
            "NET_PAY": round(net_pay_h, 4),
            "NTG": round(ntg_res, 4) if np.isfinite(ntg_res) else np.nan,
            "NTG_PAY": round(ntg_pay, 4) if np.isfinite(ntg_pay) else np.nan,
            "N_SAMPLES": int(idx_arr.size),
        }
        if well_name_val is not None:
            row["WELL_NAME"] = well_name_val
        if z_col is not None:
            row[z_col] = z_val

        # 1. Shale volume (arithmetic thickness-weighted)
        if vsh_c is not None and vsh_full is not None:
            row[vsh_c] = arith_upscale(vsh_full[act_idx], h_act) if has_act else np.nan

        # 2. Porosity PHIT & PHIE (arithmetic thickness-weighted)
        phi_blk = np.nan
        if phi_c is not None and phi_full is not None:
            phi_blk = arith_upscale(phi_full[act_idx], h_act) if has_act else 0.0
            row[phi_c] = phi_blk

        phie_blk = np.nan
        if phie_c is not None and phie_full is not None:
            phie_blk = arith_upscale(phie_full[act_idx], h_act) if has_act else 0.0
            row[phie_c] = phie_blk

        # 3. Water saturation SWT & SWE (pore-volume weighted arithmetic)
        sw_blk = np.nan
        if sw_c is not None and sw_full is not None:
            if has_act and phi_full is not None:
                sw_blk = pore_volume_upscale(sw_full[act_idx], phi_full[act_idx], h_act)
            elif has_act:
                sw_blk = arith_upscale(sw_full[act_idx], h_act)
            else:
                sw_blk = 1.0 if net_only else np.nan
            row[sw_c] = sw_blk

        if swe_c is not None and swe_full is not None:
            weight_phi_arr = phie_full if phie_full is not None else phi_full
            if has_act and weight_phi_arr is not None:
                row[swe_c] = pore_volume_upscale(swe_full[act_idx], weight_phi_arr[act_idx], h_act)
            elif has_act:
                row[swe_c] = arith_upscale(swe_full[act_idx], h_act)
            else:
                row[swe_c] = 1.0 if net_only else np.nan

        # 4. Bulk Volume Hydrocarbon (BVHC) & Bulk Volume Water (BVW)
        if phi_c is not None and sw_c is not None:
            if np.isfinite(phi_blk) and np.isfinite(sw_blk):
                row["BVW"] = float(phi_blk * sw_blk)
                row["BVHC"] = float(phi_blk * (1.0 - sw_blk))
            else:
                row["BVW"] = np.nan
                row["BVHC"] = np.nan

        # 5. Permeability upscaling (Harmonic / Geometric / Power-Law Cardwell-Parsons / Arithmetic)
        if perm_c is not None and perm_full is not None:
            if has_act:
                k_act = perm_full[act_idx]
                bounds = cardwell_parsons_bounds(
                    k_act,
                    h=h_act,
                    omega=omega,
                    min_perm=min_perm,
                )
                method_l = perm_method.strip().lower()
                if method_l in ("harmonic", "harm", "kv"):
                    k_primary = bounds["k_harm"]
                elif method_l in ("geometric", "geom", "kh"):
                    k_primary = bounds["k_geom"]
                elif method_l in ("arithmetic", "arith"):
                    k_primary = bounds["k_arith"]
                elif method_l in ("power", "power_law", "cardwell_parsons"):
                    k_primary = bounds["k_power"]
                else:
                    k_primary = upscale_permeability(
                        k_act,
                        method=perm_method,
                        h=h_act,
                        omega=omega,
                        min_perm=min_perm,
                    )
            else:
                bounds = {
                    "k_harm": 0.0 if net_only else np.nan,
                    "k_geom": 0.0 if net_only else np.nan,
                    "k_power": 0.0 if net_only else np.nan,
                    "k_arith": 0.0 if net_only else np.nan,
                    "kv_kh_ratio": np.nan,
                }
                k_primary = 0.0 if net_only else np.nan

            row[perm_c] = k_primary
            row[f"{perm_c}_ARITH"] = bounds["k_arith"]
            row[f"{perm_c}_GEOM"] = bounds["k_geom"]
            row[f"{perm_c}_HARM"] = bounds["k_harm"]
            row[f"{perm_c}_POWER"] = bounds["k_power"]
            row["KV_KH_RATIO"] = bounds["kv_kh_ratio"]

        # 6. Additional continuous curves (arithmetic thickness-weighted)
        for ac, ac_arr in arith_arrays.items():
            if ac not in row:
                row[ac] = arith_upscale(ac_arr[act_idx], h_act) if has_act else np.nan

        # 7. Discrete facies / rock type columns (majority thickness-weighted vote)
        for dc, dc_arr in disc_arrays.items():
            if dc not in row:
                row[dc] = majority_upscale(dc_arr[act_idx] if has_act else dc_arr[idx_arr], h_act if has_act else h_all)

        # Cell-level flags
        row["ROCK_FLAG"] = int(net_rock_h > 0)
        row["RES_FLAG"] = int(net_res_h > 0)
        row["PAY_FLAG"] = int(net_pay_h > 0)

        records.append(row)

        if step_df is not None:
            idx = work.index[idx_arr]
            step_df.loc[idx, "CELL_ID"] = cid
            step_df.loc[idx, "CELL_TOP"] = top_d
            step_df.loc[idx, "CELL_BOTTOM"] = bot_d
            step_df.loc[idx, "NTG_BLK"] = row["NTG"]
            if vsh_c is not None:
                step_df.loc[idx, f"{vsh_c}_BLK"] = row[vsh_c]
            if phi_c is not None:
                step_df.loc[idx, f"{phi_c}_BLK"] = row[phi_c]
            if phie_c is not None:
                step_df.loc[idx, f"{phie_c}_BLK"] = row[phie_c]
            if sw_c is not None:
                step_df.loc[idx, f"{sw_c}_BLK"] = row[sw_c]
            if swe_c is not None:
                step_df.loc[idx, f"{swe_c}_BLK"] = row[swe_c]
            if perm_c is not None:
                step_df.loc[idx, f"{perm_c}_BLK"] = row[perm_c]
                step_df.loc[idx, f"{perm_c}_ARITH_BLK"] = row[f"{perm_c}_ARITH"]
                step_df.loc[idx, f"{perm_c}_GEOM_BLK"] = row[f"{perm_c}_GEOM"]
                step_df.loc[idx, f"{perm_c}_HARM_BLK"] = row[f"{perm_c}_HARM"]
                step_df.loc[idx, f"{perm_c}_POWER_BLK"] = row[f"{perm_c}_POWER"]
                step_df.loc[idx, "KV_KH_RATIO_BLK"] = row["KV_KH_RATIO"]
            if "BVHC" in row:
                step_df.loc[idx, "BVHC_BLK"] = row["BVHC"]

    if step_df is not None:
        drop_temp = [c for c in ("_h", "_rock_flag", "_res_flag", "_pay_flag") if c in step_df.columns]
        return step_df.drop(columns=drop_temp)

    out_df = pd.DataFrame(records)
    logger.debug(
        f"Upscaled {len(work)} fine-scale log samples into {len(out_df)} grid blocks "
        f"(perm_method={perm_method}, omega={omega}, net_only={net_only})"
    )
    return out_df


# Alias for convenience
block_logs = upscale_logs


def expand_blocked_logs(
    fine_df: pd.DataFrame,
    blocked_df: pd.DataFrame,
    depth_col: str = "DEPTH",
    cols: Optional[Sequence[str]] = None,
    suffix: str = "_BLK",
) -> pd.DataFrame:
    """Map blocked cellular log values (`TOP`, `BOTTOM`) back onto a fine-scale log DataFrame as step curves.

    Useful for overlaying upscaled static geomodel cells directly on fine-scale CPI well log plots
    or exporting blocked curves to LAS 2.0 files alongside continuous CPI curves.

    Args:
        fine_df (pd.DataFrame): Fine-scale log DataFrame with `depth_col`.
        blocked_df (pd.DataFrame): Upscaled cellular DataFrame returned by `upscale_logs` (must contain `TOP` and `BOTTOM`).
        depth_col (str, optional): Depth column in `fine_df`. Defaults to `'DEPTH'`.
        cols (Sequence[str], optional): Columns from `blocked_df` to expand onto `fine_df`.
                                        If None, expands all property columns (`PHIT`, `PHIE`, `VSHALE`, `VCLAY`,
                                        `SWT`, `SWE`, `PERM`, `PERM_ARITH`, `PERM_GEOM`, `PERM_HARM`, `PERM_POWER`,
                                        `NTG`, `BVHC`, `KV_KH_RATIO`, etc.).
        suffix (str, optional): Suffix appended to expanded column names in `fine_df`. Defaults to `'_BLK'`.

    Returns:
        pd.DataFrame: Copy of `fine_df` with blocked step-profile columns attached.
    """
    out = fine_df.copy()
    if blocked_df is None or blocked_df.empty:
        return out

    d_col = _find_col(out, [depth_col, "DEPTH", "TVD", "TVDSS", "depth"])
    if d_col is None:
        raise ValueError(f"Depth column {depth_col!r} not found in fine_df.")

    meta_cols = {"CELL_ID", "TOP", "BOTTOM", d_col, "WELL_NAME", "N_SAMPLES"}
    if cols is None:
        target_cols = [c for c in blocked_df.columns if c not in meta_cols]
    else:
        target_cols = [c for c in cols if c in blocked_df.columns]

    depths = pd.to_numeric(out[d_col], errors="coerce").to_numpy(dtype=float)
    b_sorted = blocked_df.sort_values("TOP").reset_index(drop=True)
    tops = pd.to_numeric(b_sorted["TOP"], errors="coerce").to_numpy(dtype=float)
    bots = pd.to_numeric(b_sorted["BOTTOM"], errors="coerce").to_numpy(dtype=float)
    n_blocks = len(b_sorted)

    # Vectorized interval lookup: find latest block whose TOP <= depth
    idx = np.searchsorted(tops, depths, side="right") - 1
    idx_clipped = np.clip(idx, 0, max(n_blocks - 1, 0))
    is_last = idx_clipped == (n_blocks - 1)
    valid = (
        np.isfinite(depths)
        & (idx >= 0)
        & np.where(is_last, depths <= bots[idx_clipped], depths < bots[idx_clipped])
    )

    if "CELL_ID" in b_sorted.columns:
        cell_ids = pd.to_numeric(b_sorted["CELL_ID"], errors="coerce").to_numpy(dtype=float)
        out["CELL_ID"] = np.where(valid, cell_ids[idx_clipped], np.nan)
    else:
        out["CELL_ID"] = np.nan

    for c in target_cols:
        out_col = f"{c}{suffix}" if suffix and not c.endswith(suffix) else c
        col_series = b_sorted[c]
        if pd.api.types.is_numeric_dtype(col_series):
            arr = col_series.to_numpy(dtype=float)
            out[out_col] = np.where(valid, arr[idx_clipped], np.nan)
        else:
            arr_obj = col_series.to_numpy(dtype=object)
            out[out_col] = np.where(valid, arr_obj[idx_clipped], None)

    return out


def plot_upscaled_logs(
    fine_df: pd.DataFrame,
    blocked_df: pd.DataFrame,
    depth_col: str = "DEPTH",
    vsh_col: Optional[str] = None,
    phi_col: Optional[str] = None,
    sw_col: Optional[str] = None,
    perm_col: Optional[str] = None,
    figsize: Tuple[int, int] = (13, 9),
    title: Optional[str] = "Static Geomodeling Handover: Fine-Scale CPI vs. Upscaled (Blocked) Logs",
):
    """Plot a 4-track comparison of fine-scale CPI curves and upscaled (blocked) step curves.

    Tracks:
      1. Shale Volume ($V_{sh}$): Fine-scale vs. thickness-weighted arithmetic block average
      2. Porosity ($\\phi$): Fine-scale vs. thickness-weighted arithmetic block average
      3. Water Saturation ($S_w$): Fine-scale vs. pore-volume weighted block average
      4. Permeability ($k$, log scale): Fine-scale vs. Cardwell-Parsons envelope
         ($k_v$ Harmonic, $k_{eff}$ Geometric, $k_{\\omega}$ Power-Law, $k_h$ Arithmetic)

    Args:
        fine_df (pd.DataFrame): Fine-scale CPI log DataFrame.
        blocked_df (pd.DataFrame): Upscaled cellular DataFrame from `upscale_logs`.
        depth_col (str, optional): Depth column name. Defaults to `'DEPTH'`.
        vsh_col, phi_col, sw_col, perm_col (str, optional): Specific column names if non-standard.
        figsize (tuple, optional): Matplotlib figure size. Defaults to `(13, 9)`.
        title (str, optional): Figure suptitle.

    Returns:
        matplotlib.figure.Figure: The generated comparison figure.
    """
    d_col = _find_col(fine_df, [depth_col, "DEPTH", "TVD", "TVDSS", "depth"])
    vsh_c = _find_col(fine_df, [vsh_col] if vsh_col else ["VSHALE", "VCLAY", "VSH", "vshale", "vclay"])
    phi_c = _find_col(fine_df, [phi_col] if phi_col else ["PHIT", "PHI", "PHIE", "phit", "phi"])
    sw_c = _find_col(fine_df, [sw_col] if sw_col else ["SWT", "SW", "SWE", "swt", "sw"])
    perm_c = _find_col(fine_df, [perm_col] if perm_col else ["PERM", "PERM_TRANS", "perm"])

    fig, axes = plt.subplots(1, 4, figsize=figsize, sharey=True, constrained_layout=True)
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")

    z_fine = fine_df[d_col].to_numpy(dtype=float)

    def _build_step_xy(b_df: pd.DataFrame, col_name: str) -> Tuple[np.ndarray, np.ndarray]:
        xs: List[float] = []
        ys: List[float] = []
        for _, r in b_df.iterrows():
            val = float(r[col_name]) if pd.notna(r[col_name]) else np.nan
            xs.extend([val, val, np.nan])
            ys.extend([float(r["TOP"]), float(r["BOTTOM"]), np.nan])
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    # Track 1: Vshale
    ax = axes[0]
    if vsh_c and vsh_c in fine_df.columns:
        ax.plot(fine_df[vsh_c], z_fine, color="lightgray", lw=0.8, label=f"{vsh_c} (Fine)")
        if vsh_c in blocked_df.columns:
            sx, sy = _build_step_xy(blocked_df, vsh_c)
            ax.plot(sx, sy, color="#8B4513", lw=2.0, label=f"{vsh_c} (Arith Block)")
    ax.set_xlabel("Vshale (v/v)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylabel(f"{d_col}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Track 2: Porosity
    ax = axes[1]
    if phi_c and phi_c in fine_df.columns:
        ax.plot(fine_df[phi_c], z_fine, color="lightgray", lw=0.8, label=f"{phi_c} (Fine)")
        if phi_c in blocked_df.columns:
            sx, sy = _build_step_xy(blocked_df, phi_c)
            ax.plot(sx, sy, color="#1f77b4", lw=2.0, label=f"{phi_c} (Arith Block)")
    ax.set_xlabel("Porosity (v/v)")
    ax.set_xlim(0.0, 0.45)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Track 3: Water Saturation (Pore-Volume Weighted)
    ax = axes[2]
    if sw_c and sw_c in fine_df.columns:
        ax.plot(fine_df[sw_c], z_fine, color="lightgray", lw=0.8, label=f"{sw_c} (Fine)")
        if sw_c in blocked_df.columns:
            sx, sy = _build_step_xy(blocked_df, sw_c)
            ax.plot(sx, sy, color="#d62728", lw=2.0, label=f"{sw_c} (PV-Weighted)")
    ax.set_xlabel("Water Saturation (v/v)")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    # Track 4: Permeability (Harmonic / Geometric / Power / Arithmetic)
    ax = axes[3]
    if perm_c and perm_c in fine_df.columns:
        ax.plot(
            np.where(fine_df[perm_c] > 0, fine_df[perm_c], np.nan),
            z_fine,
            color="lightgray",
            lw=0.8,
            label=f"{perm_c} (Fine)",
        )
        for suffix, color, ls, lbl in [
            ("_HARM", "#9467bd", ":", "k_v (Harm)"),
            ("_GEOM", "#2ca02c", "-", "k_eff (Geom)"),
            ("_POWER", "#ff7f0e", "--", "k_omega (Power)"),
            ("_ARITH", "#17becf", "-.", "k_h (Arith)"),
        ]:
            c_name = f"{perm_c}{suffix}"
            if c_name in blocked_df.columns:
                sx, sy = _build_step_xy(blocked_df, c_name)
                sx = np.where(sx > 0, sx, np.nan)
                ax.plot(sx, sy, color=color, linestyle=ls, lw=1.8, label=lbl)
    ax.set_xscale("log")
    ax.set_xlabel("Permeability (mD)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    axes[0].invert_yaxis()
    return fig

