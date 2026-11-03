"""Headless publication-grade plot exporters for FDP Petrophysical Reports.

Generates deterministic high-DPI PNG figures for all 10 FDP report sections using
thread-safe Matplotlib `FigureCanvasAgg` rendering so FastAPI backend threads
never conflict with interactive GUI backends.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from quick_pp import logger
from quick_pp.utils import line_intersection

matplotlib.use("Agg", force=False)

# Publication color palette
PALETTE = [
    "#1d4ed8",  # royal blue
    "#059669",  # emerald
    "#d97706",  # amber
    "#dc2626",  # crimson
    "#7c3aed",  # violet
    "#0891b2",  # cyan
    "#be185d",  # pink
    "#4b5563",  # slate
]


def _resolve_output_dir(output_dir: Optional[Union[str, Path]] = None) -> Path:
    if output_dir is not None:
        p = Path(output_dir)
    else:
        p = (
            Path(__file__).resolve().parent.parent
            / "app"
            / "backend"
            / "static"
            / "reports"
            / "default"
        )
    p.mkdir(parents=True, exist_ok=True)
    return p


def _format_return_path(file_path: Path, url_prefix: Optional[str] = None) -> str:
    if url_prefix:
        return f"{url_prefix.rstrip('/')}/{file_path.name}"
    return str(file_path)


def _save_fig(
    fig: Figure,
    filename: str,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
    dpi: int = 140,
) -> str:
    out_dir = _resolve_output_dir(output_dir)
    out_path = out_dir / filename
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(
        str(out_path),
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
    )
    return _format_return_path(out_path, url_prefix=url_prefix)


def _trim_to_logged_interval(df: pd.DataFrame) -> pd.DataFrame:
    """Trim leading and trailing unlogged surface casing / post-TD intervals where all key log curves are NaN."""
    if df is None or df.empty or "DEPTH" not in df.columns:
        return df
    probe_cols = [
        c
        for c in (
            "GR",
            "RT",
            "RD",
            "NPHI",
            "RHOB",
            "DTC",
            "PHIT",
            "PHIE",
            "SWT",
            "SW",
            "PERM",
            "PERM_TRANS",
            "SHF",
            "CPORE",
            "CPERM",
        )
        if c in df.columns
    ]
    if not probe_cols:
        return df
    valid_mask = np.zeros(len(df), dtype=bool)
    for c in probe_cols:
        valid_mask |= pd.to_numeric(df[c], errors="coerce").notna().to_numpy()
    if not valid_mask.any():
        return df
    valid_indices = np.flatnonzero(valid_mask)
    start_idx = max(0, int(valid_indices[0]) - 10)
    end_idx = min(len(df), int(valid_indices[-1]) + 11)
    return df.iloc[start_idx:end_idx].copy()


def _ensure_df_well_filter(
    df: pd.DataFrame, well_names: Optional[Sequence[str]] = None
) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    if (
        well_names
        and "ALL" not in [str(w).upper() for w in well_names]
        and "WELL_NAME" in df.columns
    ):
        sub = df[df["WELL_NAME"].astype(str).isin([str(w) for w in well_names])]
        return sub.copy()
    return df.copy()


def _get_numeric_series(
    df: pd.DataFrame, *col_names: str, default: float = np.nan
) -> pd.Series:
    """Return the first matching non-empty numeric column from `df` as a Series aligned to `df.index`."""
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float)
    for col in col_names:
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            if s.notna().any():
                return s.fillna(default) if not np.isnan(default) else s
    for col in col_names:
        if col and col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            return s.fillna(default) if not np.isnan(default) else s
    return pd.Series(default, index=df.index, dtype=float)


GENERIC_PLACEHOLDER_ZONES = {
    "MAIN RESERVOIR",
    "UNKNOWN",
    "ALL",
    "NAN",
    "NONE",
    "",
}


def _resolve_target_depth_window(
    df: pd.DataFrame,
    target_zones: Optional[Sequence[str]] = None,
    max_span: float = 480.0,
) -> tuple[float, float]:
    """Determine auto-zoomed [ymin, ymax] depth window focused on the evaluated reservoir succession (<= 500 m per CHK-06)."""
    if df is None or df.empty or "DEPTH" not in df.columns:
        return (0.0, 500.0)

    d_all = pd.to_numeric(df["DEPTH"], errors="coerce")
    d_valid = d_all.dropna()
    if d_valid.empty:
        return (0.0, 500.0)

    full_min = float(d_valid.min())
    full_max = float(d_valid.max())

    # 1. Identify rows with evaluated petrophysical curves (PHIE/PHIT & SWT/SW)
    phi_s = _get_numeric_series(df, "PHIE", "PHIT")
    sw_s = _get_numeric_series(df, "SWT", "SW")
    eval_mask = d_all.notna() & phi_s.notna() & sw_s.notna()

    # 2. Apply target_zones filter if available, or restrict to named formations over generic catch-all placeholders
    if "ZONES" in df.columns and df["ZONES"].notna().any():
        z_upper = df["ZONES"].astype(str).str.strip().str.upper()
        if target_zones:
            tz_set = {str(z).strip().upper() for z in target_zones if z}
            tz_mask = d_all.notna() & z_upper.isin(tz_set)
            if (tz_mask & eval_mask).any():
                eval_mask = tz_mask & eval_mask
            elif tz_mask.any():
                eval_mask = tz_mask
        else:
            named_zone_mask = ~z_upper.isin(GENERIC_PLACEHOLDER_ZONES)
            if (eval_mask & named_zone_mask).any():
                eval_mask = eval_mask & named_zone_mask
            elif named_zone_mask.any():
                eval_mask = d_all.notna() & named_zone_mask

    if eval_mask.any():
        d_target = d_all[eval_mask]
        t_min = float(d_target.min())
        t_max = float(d_target.max())
    else:
        t_min, t_max = full_min, full_max

    ymin = max(full_min, t_min - 30.0)
    ymax = min(full_max, t_max + 30.0)

    # Enforce CHK-06: vertical span <= max_span (<= 500.0 m)
    if (ymax - ymin) > max_span:
        # Center around highest hydrocarbon / porosity concentration within target window
        hc_weight = (
            (phi_s.fillna(0.0) * (1.0 - sw_s.fillna(1.0)).clip(lower=0.0))
            .where(eval_mask, 0.0)
            .to_numpy(dtype=float)
        )
        if np.nansum(hc_weight) > 0:
            center_d = float(
                np.nansum(d_all.fillna(0.0).to_numpy(dtype=float) * hc_weight)
                / np.nansum(hc_weight)
            )
        elif eval_mask.any():
            center_d = float(np.nanmedian(d_all[eval_mask]))
        else:
            center_d = 0.5 * (t_min + t_max)
        half = 0.5 * max_span
        ymin = max(full_min, center_d - half)
        ymax = ymin + max_span
        if ymax > full_max:
            ymax = full_max
            ymin = max(full_min, ymax - max_span)

    if ymax <= ymin + 5.0:
        ymax = ymin + 50.0
    return (round(ymin, 1), round(ymax, 1))


def generate_inventory_heatmap(
    project_id: Union[int, str],
    wells: List[Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 2.1: Multi-Well Curve & Data Availability Matrix Heatmap.

    Args:
        project_id (Union[int, str]): Project identifier displayed in the figure title.
        wells (List[Dict[str, Any]]): List of well summary dictionaries containing curve and ancillary counts.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_2_1_inventory_heatmap.png`.
    """
    target_curves = [
        "TVD",
        "TVDSS",
        "GR",
        "RT",
        "NPHI",
        "RHOB",
        "VCLAY",
        "PHIT",
        "PERM",
        "SWT",
        "ROCK_FLAG",
        "PERM_TRANS",
        "SHF",
        "CORE",
        "MDT/PRESS",
        "TOPS",
    ]
    canonical_aliases = {
        "TVD": {"TVD", "DEPT_TVD", "DEPTH_TVD", "TVD_SCS", "TVDSCS"},
        "TVDSS": {"TVDSS", "DEPT_TVDSS", "DEPTH_TVDSS", "TVD_SS", "Z"},
        "GR": {"GR", "GAMMA", "GR_ARC", "GR_EDTC"},
        "RT": {"RT", "RD", "RDEP", "RDEEP", "LLD", "RILD"},
        "NPHI": {"NPHI", "TNPH", "NPOR", "CNCF"},
        "RHOB": {"RHOB", "RHOZ", "DEN", "ZDEN"},
        "VCLAY": {"VCLAY", "VSH", "VSHALE", "VCL", "VSH_GR", "VSHA"},
        "PHIT": {"PHIT", "PHIE", "POROSITY", "POR"},
        "PERM": {"PERM", "KLOGH", "PERMEABILITY", "K"},
        "SWT": {"SWT", "SWE", "SW"},
        "ROCK_FLAG": {"ROCK_FLAG", "ROCK_FLAG_ML", "ROCK_FLAG_CUTOFF"},
        "PERM_TRANS": {"PERM_TRANS"},
        "SHF": {"SHF", "SW_SHF"},
    }
    well_labels = [str(w.get("name", f"Well-{i + 1}")) for i, w in enumerate(wells)]
    if not well_labels:
        well_labels = ["No Wells"]

    matrix = np.zeros((len(well_labels), len(target_curves)), dtype=float)
    for r_idx, w in enumerate(wells):
        full_curve_list = list(w.get("available_curves") or []) + list(
            w.get("curves") or []
        )
        avail = {str(c).upper() for c in full_curve_list if c}

        # Canonical alias resolution matching Field Inventory (`get_project_summary`)
        for canon_key, aliases in canonical_aliases.items():
            if avail.intersection(aliases):
                avail.add(canon_key)
        if (
            bool(w.get("has_tvd"))
            or (w.get("survey_station_count") or 0) > 0
            or "TVD" in avail
        ):
            avail.add("TVD")
            avail.add("TVDSS")

        for c_idx, cname in enumerate(target_curves):
            if cname == "CORE":
                has_core = bool(
                    (w.get("core_sample_count") or 0) > 0
                    or w.get("has_core") not in (None, False, "No", "0", 0)
                    or avail.intersection({"CPORE", "CPERM"})
                )
                matrix[r_idx, c_idx] = 1.0 if has_core else 0.0
            elif cname == "MDT/PRESS":
                has_p = bool(
                    (w.get("pressure_test_count") or 0) > 0
                    or w.get("has_pressure") not in (None, False, "No", "0", 0)
                )
                matrix[r_idx, c_idx] = 1.0 if has_p else 0.0
            elif cname == "TOPS":
                has_t = bool(
                    (w.get("formation_top_count") or 0) > 0
                    or len(w.get("zones") or []) > 0
                )
                matrix[r_idx, c_idx] = 1.0 if has_t else 0.0
            else:
                matrix[r_idx, c_idx] = 1.0 if cname in avail else 0.0

    fig_h = max(3.4, 1.8 + 0.42 * len(well_labels))
    fig = Figure(figsize=(10.5, fig_h))
    ax = fig.add_subplot(111)

    cmap = matplotlib.colors.ListedColormap(["#f1f5f9", "#10b981"])
    ax.imshow(matrix, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)

    ax.set_xticks(np.arange(len(target_curves)))
    ax.set_xticklabels(
        target_curves, rotation=35, ha="right", fontsize=8.5, fontweight="bold"
    )
    ax.set_yticks(np.arange(len(well_labels)))
    ax.set_yticklabels(well_labels, fontsize=9, fontweight="bold")

    # Grid lines and checkmarks
    for r in range(len(well_labels)):
        for c in range(len(target_curves)):
            val = matrix[r, c]
            txt = "✓" if val >= 0.5 else "—"
            color = "#ffffff" if val >= 0.5 else "#94a3b8"
            ax.text(
                c,
                r,
                txt,
                ha="center",
                va="center",
                color=color,
                fontsize=9.5,
                fontweight="bold",
            )

    ax.set_xticks(np.arange(-0.5, len(target_curves), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(well_labels), 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linestyle="-", linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_title(
        f"Multi-Well Petrophysical Log, Core & Ancillary Data Availability Matrix (Project #{project_id})",
        fontsize=10.5,
        fontweight="bold",
        pad=10,
        color="#0f172a",
    )

    return _save_fig(fig, "fig_2_1_inventory_heatmap.png", output_dir, url_prefix)


def generate_qaqc_norm_plot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 3.1: Multi-Well Curve Distribution & Bad-Hole QA/QC Summary.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame.
        well_names (Optional[List[str]]): Optional subset of well names to include.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_3_1_qaqc_norm_plot.png`.
    """
    sub = _ensure_df_well_filter(df, well_names)
    fig = Figure(figsize=(10.5, 4.4))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.35, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    if sub.empty:
        ax1.text(
            0.5, 0.5, "No log data available for QA/QC plot", ha="center", va="center"
        )
        ax2.axis("off")
        return _save_fig(fig, "fig_3_1_qaqc_norm_plot.png", output_dir, url_prefix)

    if "WELL_NAME" not in sub.columns:
        sub["WELL_NAME"] = well_names[0] if well_names else "Well-1"

    wells = [str(w) for w in sub["WELL_NAME"].dropna().unique()]
    curve_col = (
        "GR" if "GR" in sub.columns else ("NPHI" if "NPHI" in sub.columns else None)
    )

    if curve_col:
        box_data = []
        valid_wells = []
        for w in wells:
            w_s = pd.to_numeric(
                sub.loc[sub["WELL_NAME"].astype(str) == w, curve_col], errors="coerce"
            ).dropna()
            if not w_s.empty:
                box_data.append(w_s.to_numpy(dtype=float))
                valid_wells.append(w)
        if box_data:
            bp = ax1.boxplot(
                box_data,
                labels=valid_wells,
                patch_artist=True,
                vert=True,
                showfliers=False,
                widths=0.48,
            )
            for idx, patch in enumerate(bp["boxes"]):
                patch.set_facecolor(PALETTE[idx % len(PALETTE)])
                patch.set_alpha(0.65)
                patch.set_edgecolor("#1e293b")
            for med in bp["medians"]:
                med.set_color("#0f172a")
                med.set_linewidth(2.0)
            all_vals = np.concatenate(box_data)
            field_p50 = float(np.nanmedian(all_vals))
            ax1.axhline(
                field_p50,
                color="#dc2626",
                linestyle="--",
                linewidth=1.4,
                label=f"Field P50 Baseline ({field_p50:.2f})",
            )
            ax1.legend(loc="upper right", fontsize=8)
            ax1.tick_params(axis="x", rotation=22, labelsize=8.5)
            ax1.set_ylabel(f"{curve_col} Distribution", fontsize=9, fontweight="bold")
            ax1.set_title(
                f"Multi-Well {curve_col} Distribution & Normalization Baseline",
                fontsize=9.5,
                fontweight="bold",
            )
            ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Bad-hole & HC Flag summary per well
    bh_pcts = []
    hc_pcts = []
    for w in wells:
        wdf = sub[sub["WELL_NAME"].astype(str) == w]
        if "BADHOLE" in wdf.columns:
            bh_s = _get_numeric_series(wdf, "BADHOLE", default=0.0) >= 1.0
        else:
            rhob_s = _get_numeric_series(wdf, "RHOB")
            nphi_s = _get_numeric_series(wdf, "NPHI")
            bh_s = ((rhob_s < 2.0) | (nphi_s > 0.42)).fillna(False)
        bh_pcts.append(100.0 * float(bh_s.mean()) if len(wdf) > 0 else 0.0)

        if "HC_FLAG" in wdf.columns:
            hc_s = pd.to_numeric(wdf["HC_FLAG"], errors="coerce").fillna(0.0) >= 1.0
            hc_pcts.append(100.0 * float(hc_s.mean()) if len(wdf) > 0 else 0.0)
        else:
            hc_pcts.append(0.0)

    x_pos = np.arange(len(wells))
    width = 0.36
    ax2.bar(
        x_pos - width / 2,
        bh_pcts,
        width=width,
        color="#ef4444",
        alpha=0.85,
        label="Bad-Hole / Washout (%)",
    )
    ax2.bar(
        x_pos + width / 2,
        hc_pcts,
        width=width,
        color="#f59e0b",
        alpha=0.85,
        label="HC-Corrected (%)",
    )
    ax2.axhline(
        15.0,
        color="#991b1b",
        linestyle=":",
        linewidth=1.2,
        label="15% QC Alert Threshold",
    )
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(wells, rotation=22, fontsize=8.5)
    ax2.set_ylabel("Flagged Interval (%)", fontsize=9, fontweight="bold")
    ax2.set_title(
        "Borehole Washout & Conditioning Audit", fontsize=9.5, fontweight="bold"
    )
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.5)

    return _save_fig(fig, "fig_3_1_qaqc_norm_plot.png", output_dir, url_prefix)


def generate_nd_crossplot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    color_by: str = "GR",
    target_zones: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
    litho_model: str = "ssc",
    dry_sand_point: Tuple[float, float] = (-0.02, 2.65),
    dry_silt_point: Optional[Tuple[Optional[float], Optional[float]]] = None,
    dry_clay_point: Tuple[float, float] = (0.33, 2.70),
    fluid_point: Tuple[float, float] = (1.0, 1.0),
    silt_line_angle: float = 119.0,
) -> str:
    """Generate Figure 4.1: Neutron-Density Crossplot, Core Porosity Calibration & Matrix Grain Density (RHOMA) QC.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame containing `NPHI`, `RHOB`, `PHIT`, and optional `CPORE`.
        well_names (Optional[List[str]]): Optional subset of well names to filter.
        color_by (str): Curve mnemonic used to color-code crossplot points. Defaults to `'GR'`.
        target_zones (Optional[Sequence[str]]): Optional target reservoir zone names to focus calibration.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.
        litho_model (str): Active lithology model ('ssc', 'multi_mineral', etc.). Defaults to 'ssc'.
        dry_sand_point (Tuple[float, float]): (NPHI, RHOB) coordinates for dry sand.
        dry_silt_point (Optional[Tuple]): (NPHI, RHOB) coordinates for dry silt.
        dry_clay_point (Tuple[float, float]): (NPHI, RHOB) coordinates for dry clay.
        fluid_point (Tuple[float, float]): (NPHI, RHOB) coordinates for formation fluid.
        silt_line_angle (float): Angle of silt line in degrees from horizontal. Defaults to 119.0.

    Returns:
        str: File path or URL to the generated `fig_4_1_nd_crossplot.png`.
    """
    sub = _ensure_df_well_filter(df, well_names)
    fig = Figure(figsize=(13.2, 4.4))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.15, 1.0, 0.95], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Filter to target_zones when available for clean reservoir lithology & core calibration
    cal_sub = sub
    if target_zones and "ZONES" in sub.columns and sub["ZONES"].notna().any():
        tz_set = {str(z).strip().upper() for z in target_zones if z}
        tz_df = sub[sub["ZONES"].astype(str).str.strip().str.upper().isin(tz_set)]
        if not tz_df.empty:
            cal_sub = tz_df

    nphi = _get_numeric_series(cal_sub, "NPHI")
    rhob = _get_numeric_series(cal_sub, "RHOB")
    c_series = _get_numeric_series(cal_sub, color_by, "VCLAY", "GR")

    valid = nphi.notna() & rhob.notna()
    if valid.any():
        x = nphi[valid].to_numpy(dtype=float)
        y = rhob[valid].to_numpy(dtype=float)
        c = (
            c_series[valid].fillna(50.0).to_numpy(dtype=float)
            if c_series.notna().any()
            else np.full_like(x, 50.0)
        )
        # Subsample up to 2500 pts for crisp vector/raster rendering
        if len(x) > 2500:
            idx = np.linspace(0, len(x) - 1, 2500, dtype=int)
            x, y, c = x[idx], y[idx], c[idx]
        sc = ax1.scatter(
            x,
            y,
            c=c,
            cmap="viridis_r",
            s=12,
            alpha=0.65,
            edgecolors="none",
        )
        cbar = fig.colorbar(sc, ax=ax1, pad=0.02, fraction=0.046)
        cbar.set_label(color_by, fontsize=8)

        if str(litho_model).lower() == "ssc":
            # Sand-Silt-Clay (SSC) Ternary Endpoints & Lines
            A = dry_sand_point or (-0.02, 2.65)
            C = dry_clay_point or (0.33, 2.70)
            D = fluid_point or (1.0, 1.0)

            # Compute dry silt point B if not specified
            if (
                dry_silt_point
                and dry_silt_point[0] is not None
                and dry_silt_point[1] is not None
            ):
                B = (float(dry_silt_point[0]), float(dry_silt_point[1]))
            else:
                rad = math.radians(float(silt_line_angle))
                p_silt = (D[0] + math.cos(rad), D[1] + math.sin(rad))
                silt_inter = line_intersection((A, C), (D, p_silt))
                if silt_inter is not None and np.isfinite(silt_inter[0]) and np.isfinite(silt_inter[1]):
                    B = (float(silt_inter[0]), float(silt_inter[1]))
                else:
                    drysilt_NPHI = 1.0 - 1.68 * math.tan(float(silt_line_angle - 90) * math.pi / 180.0)
                    B = (drysilt_NPHI, 2.68)

            # Plot SSC ternary model lines
            ax1.plot([D[0], A[0]], [D[1], A[1]], color="#1e40af", lw=1.3, label=f"Sand Line ({A[1]:.2f} g/cc)")
            ax1.plot([D[0], B[0]], [D[1], B[1]], color="#16a34a", lw=1.3, linestyle="-", label=f"Silt Line ({silt_line_angle:.0f}°)")
            ax1.plot([D[0], C[0]], [D[1], C[1]], color="#9333ea", lw=1.2, linestyle="--", label=f"Clay Line ({C[1]:.2f} g/cc)")
            ax1.plot([A[0], C[0]], [A[1], C[1]], color="#0f172a", lw=1.4, linestyle="-", label="Rock Line")

            # Endpoint markers
            ax1.scatter([A[0]], [A[1]], color="#eab308", marker="s", s=50, zorder=6, label=f"Dry Sand ({A[0]:.2f}, {A[1]:.2f})")
            ax1.scatter([B[0]], [B[1]], color="#16a34a", marker="D", s=45, zorder=6, label=f"Dry Silt ({B[0]:.2f}, {B[1]:.2f})")
            ax1.scatter([C[0]], [C[1]], color="#dc2626", marker="^", s=60, zorder=6, label=f"Dry Clay ({C[0]:.2f}, {C[1]:.2f})")
        else:
            # Fallback Standard Quartz / Calcite / Dolomite lithology lines
            phi_grid = np.linspace(0.0, 0.40, 50)
            ax1.plot(
                phi_grid - 0.02 * (1 - phi_grid),
                2.65 * (1 - phi_grid) + 1.0 * phi_grid,
                "k-",
                lw=1.3,
                label="Quartz (2.65 g/cc)",
            )
            ax1.plot(
                phi_grid,
                2.71 * (1 - phi_grid) + 1.0 * phi_grid,
                "b--",
                lw=1.2,
                label="Calcite (2.71 g/cc)",
            )
            ax1.plot(
                phi_grid + 0.02 * (1 - phi_grid),
                2.87 * (1 - phi_grid) + 1.0 * phi_grid,
                "m:",
                lw=1.2,
                label="Dolomite (2.87 g/cc)",
            )
            ax1.scatter(
                [0.33],
                [2.70],
                color="#dc2626",
                marker="^",
                s=65,
                zorder=5,
                label="Dry Clay Endpoint",
            )
        ax1.set_xlim(-0.05, 0.50)
        ax1.set_ylim(2.95, 1.85)
        ax1.set_xlabel("Neutron Porosity NPHI (v/v)", fontsize=8.5, fontweight="bold")
        ax1.set_ylabel("Bulk Density RHOB (g/cc)", fontsize=8.5, fontweight="bold")
        tz_suffix = f" — {', '.join(target_zones)}" if target_zones else ""
        model_title = "SSC" if str(litho_model).lower() == "ssc" else "Lithology"
        ax1.set_title(
            f"Neutron-Density {model_title} Crossplot{tz_suffix}", fontsize=8.5, fontweight="bold"
        )
        ax1.legend(loc="lower left", fontsize=6.8)
        ax1.grid(True, linestyle=":", alpha=0.5)
    else:
        ax1.text(0.5, 0.5, "NPHI / RHOB curves not available", ha="center", va="center")

    # Panel 2: Core vs Log Porosity Calibration (strictly where both CPORE and PHIT are non-null)
    phit = _get_numeric_series(cal_sub, "PHIT", "PHIE")
    cpore = _get_numeric_series(cal_sub, "CPORE", "CORE_PORO")
    cp_frac = np.where(cpore > 1.0, cpore / 100.0, cpore)
    cmask = (
        np.isfinite(cp_frac) & np.isfinite(phit) & (cp_frac > 0.005) & (cp_frac < 0.48)
    )
    if np.sum(cmask) < 3 and cal_sub is not sub:
        # Fallback to full well/field overlapping CPORE & PHIT if target_zones had <3 overlapping plugs
        phit_all = _get_numeric_series(sub, "PHIT", "PHIE")
        cpore_all = _get_numeric_series(sub, "CPORE", "CORE_PORO")
        cp_frac_all = np.where(cpore_all > 1.0, cpore_all / 100.0, cpore_all)
        cmask_all = (
            np.isfinite(cp_frac_all)
            & np.isfinite(phit_all)
            & (cp_frac_all > 0.005)
            & (cp_frac_all < 0.48)
        )
        if np.sum(cmask_all) >= 3:
            cp_frac, phit, cmask = cp_frac_all, phit_all, cmask_all

    if np.sum(cmask) >= 3:
        cx = cp_frac[cmask]
        ly = phit.to_numpy(dtype=float)[cmask]
        rmse = float(np.sqrt(np.mean((ly - cx) ** 2)))
        ss_res = float(np.sum((ly - cx) ** 2))
        ss_tot = float(np.sum((cx - np.mean(cx)) ** 2))
        r2 = max(0.0, 1.0 - ss_res / max(ss_tot, 1e-9))

        ax2.scatter(
            cx,
            ly,
            color="#059669",
            s=24,
            alpha=0.8,
            edgecolors="#064e3b",
            label=f"Valid Overlapping Plugs (n={len(cx)})",
        )
        ax2.plot([0, 0.4], [0, 0.4], "k--", lw=1.3, label="1:1 Parity Line")
        ax2.set_xlim(0.0, 0.40)
        ax2.set_ylim(0.0, 0.40)
        ax2.set_xlabel(
            "Overburden-Corrected Core Porosity (v/v)",
            fontsize=8.5,
            fontweight="bold",
        )
        ax2.set_ylabel(
            "Log Total Porosity PHIT (v/v)", fontsize=8.5, fontweight="bold"
        )
        ax2.set_title(
            f"Core-to-Log Calibration (R²={r2:.2f}, RMSE={rmse:.3f})",
            fontsize=9.0,
            fontweight="bold",
        )
        ax2.legend(loc="upper left", fontsize=7.5)
        ax2.grid(True, linestyle=":", alpha=0.5)
    else:
        _plot_porosity_histogram(ax2, cal_sub)

    # Panel 3: RCA Matrix Grain Density (RHOMA / CGD / Apparent Matrix Density) QC
    rhoma_s = _get_numeric_series(sub, "RHOMA", "CGD", "GRAIN_DENSITY", "RHO_MA")
    rhoma_vals = rhoma_s[(rhoma_s >= 2.45) & (rhoma_s <= 2.95)].dropna()
    rhoma_label = "RCA Core Grain Density"
    if len(rhoma_vals) < 3:
        # Compute apparent matrix density from RHOB and CPORE/PHIT: rho_ma = (rhob - phi * 1.0) / (1 - phi)
        rb_s = _get_numeric_series(cal_sub, "RHOB")
        phi_ref = _get_numeric_series(cal_sub, "CPORE", "PHIT", "PHIE")
        phi_ref = pd.Series(
            np.where(phi_ref > 1.0, phi_ref / 100.0, phi_ref), index=cal_sub.index
        )
        app_ma = (rb_s - phi_ref * 1.00) / (1.0 - phi_ref).clip(lower=0.4)
        rhoma_vals = app_ma[(app_ma >= 2.50) & (app_ma <= 2.90)].dropna()
        rhoma_label = "Apparent Matrix Density ρ_ma"

    if not rhoma_vals.empty:
        mean_ma = float(rhoma_vals.mean())
        ax3.hist(
            rhoma_vals.to_numpy(dtype=float),
            bins=28,
            color="#f59e0b",
            alpha=0.72,
            edgecolor="#78350f",
            label=f"{rhoma_label} (μ={mean_ma:.2f} g/cc)",
        )
        ax3.axvline(
            2.65,
            color="#0f172a",
            linestyle="--",
            lw=1.4,
            label="Quartz Ref (2.65 g/cc)",
        )
        ax3.axvline(
            2.71,
            color="#2563eb",
            linestyle=":",
            lw=1.3,
            label="Calcite Cement (2.71 g/cc)",
        )
        ax3.set_xlim(2.52, 2.88)
        ax3.set_xlabel("Matrix Grain Density ρ_ma (g/cc)", fontsize=8.5, fontweight="bold")
        ax3.set_ylabel("Sample Frequency", fontsize=8.5, fontweight="bold")
        ax3.set_title("Grain Density (ρ_ma) Matrix QC", fontsize=9.0, fontweight="bold")
        ax3.legend(loc="upper right", fontsize=7.0)
        ax3.grid(True, linestyle=":", alpha=0.5)
    else:
        _plot_porosity_histogram(ax3, cal_sub)

    return _save_fig(fig, "fig_4_1_nd_crossplot.png", output_dir, url_prefix)


def _plot_porosity_histogram(ax: Any, sub: pd.DataFrame) -> None:
    phit = _get_numeric_series(sub, "PHIT").dropna()
    phie = _get_numeric_series(sub, "PHIE").dropna()
    if not phit.empty:
        ax.hist(
            phit.clip(0, 0.45),
            bins=30,
            alpha=0.6,
            color="#2563eb",
            label=f"PHIT (mean={phit.mean():.3f})",
        )
    if not phie.empty:
        ax.hist(
            phie.clip(0, 0.45),
            bins=30,
            alpha=0.6,
            color="#10b981",
            label=f"PHIE (mean={phie.mean():.3f})",
        )
    ax.set_xlabel("Porosity (v/v)", fontsize=9, fontweight="bold")
    ax.set_ylabel("Sample Frequency", fontsize=9, fontweight="bold")
    ax.set_title(
        "Total (PHIT) vs. Effective (PHIE) Porosity Distribution",
        fontsize=9.5,
        fontweight="bold",
    )
    if not phit.empty or not phie.empty:
        ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, linestyle=":", alpha=0.5)


def generate_core_calibration_plot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate standalone Core Calibration & Overburden Compaction Plot.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame containing `PHIT` and `PHIE`.
        well_names (Optional[List[str]]): Optional subset of well names to filter.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_4_2_core_calibration.png`.
    """
    sub = _ensure_df_well_filter(df, well_names)
    fig = Figure(figsize=(9.5, 4.2))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    _plot_porosity_histogram(ax1, sub)
    # Overburden stress compaction curve
    stress_psi = np.linspace(500, 6000, 50)
    compaction_factor = 1.0 - 0.045 * (1.0 - np.exp(-stress_psi / 2200.0))
    ax2.plot(stress_psi, compaction_factor, color="#1d4ed8", lw=2.0)
    ax2.axhline(
        0.96,
        color="#dc2626",
        linestyle="--",
        lw=1.1,
        label="Reservoir NOB Correction (0.96)",
    )
    ax2.set_xlabel("Net Overburden Stress (psi)", fontsize=9, fontweight="bold")
    ax2.set_ylabel(
        "Porosity Compaction Factor (Φ_insitu / Φ_ambient)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.set_title("RCA Overburden Compaction Factor", fontsize=9.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(True, linestyle=":", alpha=0.5)

    return _save_fig(fig, "fig_4_2_core_calibration.png", output_dir, url_prefix)


def generate_rock_typing_perm_plot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    rock_types: Optional[List[Dict[str, Any]]] = None,
    fzi_cutoffs: Optional[List[float]] = None,
    target_zones: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 5.1: Core/Log Porosity-Permeability Crossplot by Rock Type & Lorenz SMLP.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame containing porosity and permeability curves.
        well_names (Optional[List[str]]): Optional subset of well names to filter.
        rock_types (Optional[List[Dict[str, Any]]]): Optional list of rock type definitions.
        fzi_cutoffs (Optional[List[float]]): Configured FZI cutoff boundaries. Defaults to [0.5, 1.5, 3.0].
        target_zones (Optional[Sequence[str]]): Optional target reservoir zone names to focus rock typing.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_5_1_perm_crossplot.png`.
    """
    sub = _ensure_df_well_filter(df, well_names)
    if target_zones and "ZONES" in sub.columns and sub["ZONES"].notna().any():
        tz_set = {str(z).strip().upper() for z in target_zones if z}
        tz_df = sub[sub["ZONES"].astype(str).str.strip().str.upper().isin(tz_set)]
        if not tz_df.empty:
            sub = tz_df

    fig = Figure(figsize=(11.0, 4.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.2, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    phi = _get_numeric_series(sub, "CPORE", "PHIE", "PHIT")
    if phi.notna().any() and (phi > 1.0).any():
        phi = pd.Series(np.where(phi > 1.0, phi / 100.0, phi), index=sub.index)
    perm = _get_numeric_series(sub, "CPERM", "PERM")

    if not (phi.notna() & perm.notna()).any():
        phi = _get_numeric_series(sub, "PHIE", "PHIT")
        perm = _get_numeric_series(sub, "PERM")

    valid = phi.notna() & perm.notna() & (phi > 0.01) & (phi < 0.48) & (perm > 1e-3)

    if valid.any():
        p_arr = phi[valid].to_numpy(dtype=float)
        k_arr = perm[valid].to_numpy(dtype=float)
        rqi = 0.0314 * np.sqrt(k_arr / np.clip(p_arr, 1e-4, 0.99))
        phiz = p_arr / np.clip(1.0 - p_arr, 1e-4, 1.0)
        fzi = rqi / np.clip(phiz, 1e-4, 100.0)

        # Parse and resolve cutoffs
        cutoffs: List[float] = []
        if fzi_cutoffs and len(fzi_cutoffs) > 0:
            cutoffs = sorted(float(c) for c in fzi_cutoffs if float(c) > 0)
        elif rock_types:
            import re
            for rt in rock_types:
                fzi_r = str(rt.get("fzi_range", ""))
                nums = re.findall(r"[-+]?\d*\.\d+|\d+", fzi_r)
                for num in nums:
                    val = float(num)
                    if val > 0 and val not in cutoffs:
                        cutoffs.append(val)
            cutoffs = sorted(cutoffs)
        if not cutoffs:
            cutoffs = [0.5, 1.5, 3.0]

        rt_bins = np.digitize(fzi, bins=cutoffs) + 1
        n_rts = len(cutoffs) + 1

        if len(p_arr) > 2000:
            idx = np.linspace(0, len(p_arr) - 1, 2000, dtype=int)
            p_plot, k_plot, rt_plot = p_arr[idx], k_arr[idx], rt_bins[idx]
        else:
            p_plot, k_plot, rt_plot = p_arr, k_arr, rt_bins

        # Plot data points by partitioned rock type
        phi_line = np.linspace(0.03, 0.38, 60)
        for rt_code in range(1, n_rts + 1):
            mask = rt_plot == rt_code
            color = PALETTE[(rt_code - 1) % len(PALETTE)]
            if rt_code == 1:
                lbl = f"RT1 (FZI < {cutoffs[0]:.2f})"
            elif rt_code == n_rts:
                lbl = f"RT{rt_code} (FZI ≥ {cutoffs[-1]:.2f})"
            else:
                lbl = f"RT{rt_code} ({cutoffs[rt_code-2]:.2f} ≤ FZI < {cutoffs[rt_code-1]:.2f})"

            if np.any(mask):
                ax1.scatter(
                    p_plot[mask],
                    k_plot[mask],
                    color=color,
                    s=14,
                    alpha=0.75,
                    label=lbl,
                )

        # Plot explicit cutoff boundary curves so points and cutoffs align perfectly
        for i, c_val in enumerate(cutoffs):
            k_cutoff = 1014.24 * (c_val**2) * (phi_line**3) / ((1.0 - phi_line) ** 2)
            ax1.plot(
                phi_line,
                k_cutoff,
                color="#334155",
                linestyle="--",
                lw=1.1,
                alpha=0.85,
                label=f"Cutoff {c_val:.2f} µm",
            )

        ax1.set_yscale("log")
        ax1.set_xlim(0.0, 0.40)
        ax1.set_ylim(1e-2, 1e4)
        ax1.set_xlabel("Porosity Φ (v/v)", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Permeability K (mD)", fontsize=9, fontweight="bold")
        tz_title = f" — {', '.join(target_zones)}" if target_zones else ""
        ax1.set_title(
            f"Porosity-Permeability by FZI Rock Type{tz_title}",
            fontsize=9.0,
            fontweight="bold",
        )
        ax1.legend(loc="lower right", fontsize=7.0)
        ax1.grid(True, which="both", linestyle=":", alpha=0.5)

        # Panel 2: Stratigraphic Modified Lorenz Plot (Ordered by stratigraphic depth on key cored/evaluated well)
        well_col = (
            sub["WELL_NAME"]
            if "WELL_NAME" in sub.columns
            else pd.Series("Project", index=sub.index)
        )
        wells_with_pts = well_col[valid].value_counts()
        key_well = str(wells_with_pts.index[0]) if not wells_with_pts.empty else "Key Well"
        well_mask = valid & (well_col == key_well)
        if not well_mask.any():
            well_mask = valid
            key_well = "Field"

        w_sub = sub[well_mask].copy()
        if "DEPTH" in w_sub.columns and w_sub["DEPTH"].notna().any():
            w_sub = w_sub.dropna(subset=["DEPTH"]).sort_values("DEPTH").reset_index(drop=True)
            w_p = _get_numeric_series(w_sub, "CPORE", "PHIE", "PHIT").to_numpy(dtype=float)
            w_p = np.where(w_p > 1.0, w_p / 100.0, w_p)
            w_k = _get_numeric_series(w_sub, "CPERM", "PERM").to_numpy(dtype=float)
            d_vals = pd.to_numeric(w_sub["DEPTH"], errors="coerce").to_numpy(dtype=float)
            dh = np.diff(d_vals, prepend=d_vals[0])
            dh = np.where((dh > 0) & (dh < 5.0), dh, 0.1524)
        else:
            w_p = p_arr
            w_k = k_arr
            dh = np.full_like(w_p, 0.1524)

        flow_cap = np.cumsum(np.clip(w_k, 1e-4, 5e4) * dh)
        store_cap = np.cumsum(np.clip(w_p, 1e-4, 0.45) * dh)
        kh_cum = flow_cap / max(float(flow_cap[-1]), 1e-9)
        phih_cum = store_cap / max(float(store_cap[-1]), 1e-9)

        # Calculate Lorenz coefficient L
        try:
            lorenz_val = max(0.0, min(1.0, float(np.trapz(kh_cum, phih_cum) - 0.5) * 2.0))
        except Exception:
            lorenz_val = 0.50

        ax2.plot([0, 1], [0, 1], "k--", lw=1.1, label="Homogeneous (Lorenz = 0)")
        ax2.plot(
            phih_cum,
            kh_cum,
            color="#1d4ed8",
            lw=2.2,
            label=f"Stratigraphic Flow Profile (L = {abs(lorenz_val):.2f})",
        )
        ax2.fill_between(phih_cum, phih_cum, kh_cum, color="#93c5fd", alpha=0.35)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel(
            "Cumulative Storage Capacity Σ(Φ·h) [frac]", fontsize=9, fontweight="bold"
        )
        ax2.set_ylabel(
            "Cumulative Flow Capacity Σ(K·h) [frac]", fontsize=9, fontweight="bold"
        )
        ax2.set_title(
            f"Stratigraphic Modified Lorenz Plot (SMLP — {key_well})",
            fontsize=9.0,
            fontweight="bold",
        )
        ax2.legend(loc="lower right", fontsize=7.5)
        ax2.grid(True, linestyle=":", alpha=0.5)
    else:
        ax1.text(
            0.5,
            0.5,
            "Porosity / Permeability data unavailable",
            ha="center",
            va="center",
        )
        ax2.axis("off")

    return _save_fig(fig, "fig_5_1_perm_crossplot.png", output_dir, url_prefix)


def generate_pickett_plot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    rw: float = 0.08,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
    fwl: Optional[float] = None,
    fwl_basis: Optional[str] = None,
    target_zones: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 6.1: Log-Log Pickett Plot & Saturation vs Depth / FWL Profile.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame containing `RT`, `PHIT`, `SWT`, and optional `SHF`.
        well_names (Optional[List[str]]): Optional subset of well names to filter.
        rw (float): Formation water resistivity in `ohm.m`. Defaults to 0.08.
        a (float): Tortuosity factor. Defaults to 1.0.
        m (float): Cementation exponent. Defaults to 2.0.
        n (float): Saturation exponent. Defaults to 2.0.
        fwl (Optional[float]): Optional Free Water Level depth (`m`) to overlay on the saturation profile.
        fwl_basis (Optional[str]): Source / basis of the FWL value (e.g. 'Log Transition Zone', 'Fluid Contacts DB').
        target_zones (Optional[Sequence[str]]): Optional target reservoir zone names to focus Pickett evaluation.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_6_1_pickett_shf_plot.png`.
    """
    sub = _ensure_df_well_filter(df, well_names)
    if target_zones and "ZONES" in sub.columns and sub["ZONES"].notna().any():
        tz_set = {str(z).strip().upper() for z in target_zones if z}
        tz_df = sub[sub["ZONES"].astype(str).str.strip().str.upper().isin(tz_set)]
        if not tz_df.empty:
            sub = tz_df

    fig = Figure(figsize=(11.8, 4.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.38)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    rt = _get_numeric_series(sub, "RT", "RD")
    phit = _get_numeric_series(sub, "PHIT", "PHIE")
    vclay = _get_numeric_series(sub, "VCLAY", "GR")

    valid = rt.notna() & phit.notna() & (rt > 0.05) & (phit > 0.02) & (phit < 0.45)

    if valid.any():
        rt_v = rt[valid].to_numpy(dtype=float)
        phi_v = phit[valid].to_numpy(dtype=float)
        c_v = (
            vclay[valid].fillna(0.2).to_numpy(dtype=float)
            if vclay.notna().any()
            else np.full_like(rt_v, 0.2)
        )
        if len(rt_v) > 2200:
            idx = np.linspace(0, len(rt_v) - 1, 2200, dtype=int)
            rt_v, phi_v, c_v = rt_v[idx], phi_v[idx], c_v[idx]

        sc = ax1.scatter(rt_v, phi_v, c=c_v, cmap="YlGnBu_r", s=13, alpha=0.65)
        cbar = fig.colorbar(sc, ax=ax1, pad=0.03, fraction=0.046)
        cbar.set_label("VCLAY (v/v)", fontsize=8.5, fontweight="bold")

        phi_grid = np.logspace(np.log10(0.02), np.log10(0.45), 60)
        for sw_iso, col, ls in [
            (1.0, "#1d4ed8", "-"),
            (0.5, "#059669", "--"),
            (0.2, "#dc2626", ":"),
        ]:
            rt_iso = (a * rw) / ((phi_grid**m) * (sw_iso**n))
            ax1.plot(
                rt_iso,
                phi_grid,
                color=col,
                linestyle=ls,
                lw=1.5,
                label=f"Sw = {int(sw_iso * 100)}%",
            )

        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(0.1, 1000.0)
        ax1.set_ylim(0.02, 0.50)
        ax1.set_xlabel("True Resistivity Rt (Ω·m)", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Porosity Φ (v/v)", fontsize=9, fontweight="bold")
        tz_title = f" — {', '.join(target_zones)}" if target_zones else ""
        ax1.set_title(
            f"Pickett Plot{tz_title} (Rw={rw:.3f} Ω·m, a={a}, m={m}, n={n})",
            fontsize=9.0,
            fontweight="bold",
        )
        ax1.legend(loc="lower left", fontsize=7.5)
        ax1.grid(True, which="both", linestyle=":", alpha=0.5)
    else:
        ax1.text(
            0.5,
            0.5,
            "Rt / Porosity data unavailable for Pickett plot",
            ha="center",
            va="center",
        )

    # Panel 2: Water Saturation vs Depth & Free Water Level (FWL)
    depth = _get_numeric_series(sub, "TVDSS", "TVD", "DEPTH")
    depth_label = "Depth TVDss (m)" if "TVDSS" in sub.columns else ("Depth TVD (m)" if "TVD" in sub.columns else "Depth (m)")
    swt = _get_numeric_series(sub, "SWT", "SW")
    shf = _get_numeric_series(sub, "SHF")
    if (depth.notna() & swt.notna()).any():
        d_mask = depth.notna() & swt.notna()
        d_arr = depth[d_mask].to_numpy(dtype=float)
        sw_arr = swt[d_mask].to_numpy(dtype=float)
        if len(d_arr) > 2000:
            idx = np.linspace(0, len(d_arr) - 1, 2000, dtype=int)
            d_arr, sw_arr = d_arr[idx], sw_arr[idx]
        ax2.scatter(
            sw_arr, d_arr, color="#2563eb", s=8, alpha=0.45, label="Log Sw (SWT)"
        )
        if (depth.notna() & shf.notna()).any():
            s_mask = depth.notna() & shf.notna()
            sd_arr = depth[s_mask].to_numpy(dtype=float)
            shf_arr = shf[s_mask].to_numpy(dtype=float)
            if len(sd_arr) > 1500:
                idx = np.linspace(0, len(sd_arr) - 1, 1500, dtype=int)
                sd_arr, shf_arr = sd_arr[idx], shf_arr[idx]
            ax2.scatter(
                shf_arr, sd_arr, color="#dc2626", s=8, alpha=0.45, label="Capillary SHF"
            )
        if fwl is not None and np.isfinite(fwl):
            fwl_basis_label = f" [{fwl_basis}]" if fwl_basis else ""
            ax2.axhline(
                float(fwl),
                color="#0f172a",
                linestyle="--",
                lw=1.5,
                label=f"FWL: {float(fwl):.1f} m{fwl_basis_label}",
            )
        ax2.invert_yaxis()
        ax2.set_xlim(0.0, 1.05)
        ax2.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
        ax2.set_ylabel(depth_label, fontsize=9, fontweight="bold")
        ax2.set_title(
            "Water Saturation Profile & Contact Reconciliation",
            fontsize=9.0,
            fontweight="bold",
        )
        ax2.legend(loc="lower left", fontsize=7.5)
        ax2.grid(True, linestyle=":", alpha=0.5)
    else:
        ax2.axis("off")

    return _save_fig(fig, "fig_6_1_pickett_shf_plot.png", output_dir, url_prefix)


def generate_shf_j_function_plot(
    df: pd.DataFrame,
    rock_types: Optional[List[Dict[str, Any]]] = None,
    fwl: Optional[float] = None,
    is_analogue_scal: bool = True,
    scal_params: Optional[Dict[str, Any]] = None,
    shf_params: Optional[Dict[str, Any]] = None,
    target_zones: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 7.1: Leverett J-Function by Rock Type & Relative Permeability Curves.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame.
        rock_types (Optional[List[Dict[str, Any]]]): Optional list of rock type J-function definitions.
        fwl (Optional[float]): Optional Free Water Level depth (`m`).
        is_analogue_scal (bool): Whether the SCAL curves represent analogue defaults vs. measured project SCAL.
            Defaults to True.
        scal_params (Optional[Dict[str, Any]]): Project SCAL parameters (Corey/LET exponents, Swirr, Sor, etc.).
        shf_params (Optional[Dict[str, Any]]): Project SHF parameters (IFT, contact angle, fluid densities).
        target_zones (Optional[Sequence[str]]): Optional target reservoir zone names.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_7_1_shf_scal_plot.png`.
    """
    sub = df
    if target_zones and "ZONES" in df.columns and df["ZONES"].notna().any():
        tz_set = {str(z).strip().upper() for z in target_zones if z}
        tz_df = df[df["ZONES"].astype(str).str.strip().str.upper().isin(tz_set)]
        if not tz_df.empty:
            sub = tz_df

    fig = Figure(figsize=(11.2, 4.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.05, 1.0], wspace=0.30)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Panel 1: Leverett J-Function vs Sw by Rock Type
    j_grid = np.linspace(0.05, 8.0, 80)
    rt_fits = [
        ("RT1 (Tight / Nano)", 0.65, -0.32, PALETTE[0]),
        ("RT2 (Meso-Porous)", 0.38, -0.42, PALETTE[1]),
        ("RT3 (Macro-Porous)", 0.22, -0.50, PALETTE[2]),
        ("RT4 (Mega-Porous Sand)", 0.12, -0.60, PALETTE[3]),
    ]
    if rock_types and len(rock_types) >= 2:
        new_fits = []
        for i, rt in enumerate(rock_types):
            r_name = rt.get("name", f"RT{i+1}")
            # Use specific J-fit coefficients if present
            a_p = float(rt.get("j_a", rt_fits[i % len(rt_fits)][1]))
            b_p = float(rt.get("j_b", rt_fits[i % len(rt_fits)][2]))
            new_fits.append((r_name.split("—")[0].strip(), a_p, b_p, PALETTE[i % len(PALETTE)]))
        if new_fits:
            rt_fits = new_fits

    # Plot actual project data points if Sw, Phi, Perm, and Depth exist
    sw_s = _get_numeric_series(sub, "SWT", "SW")
    phi_s = _get_numeric_series(sub, "CPORE", "PHIE", "PHIT")
    phi_s = pd.Series(np.where(phi_s > 1.0, phi_s / 100.0, phi_s), index=sub.index)
    k_s = _get_numeric_series(sub, "CPERM", "PERM")
    d_s = _get_numeric_series(sub, "TVDSS", "TVD", "DEPTH")

    shf_dict = shf_params or {}
    ift_val = float(shf_dict.get("ift", 30.0))
    theta_val = float(shf_dict.get("theta", 30.0))
    gw_val = float(shf_dict.get("gw", 1.05))
    ghc_val = float(shf_dict.get("ghc", 0.80))
    d_rho = max(gw_val - ghc_val, 0.05)
    sigma_cos = max(abs(ift_val * math.cos(math.radians(theta_val))), 1.0)

    if fwl is not None and (sw_s.notna() & phi_s.notna() & k_s.notna() & d_s.notna()).any():
        v_mask = (
            sw_s.notna()
            & (sw_s > 0.02)
            & (sw_s <= 1.0)
            & phi_s.notna()
            & (phi_s > 0.02)
            & k_s.notna()
            & (k_s > 1e-3)
            & d_s.notna()
        )
        if v_mask.sum() >= 10:
            sw_pts = sw_s[v_mask].to_numpy(dtype=float)
            phi_pts = phi_s[v_mask].to_numpy(dtype=float)
            k_pts = k_s[v_mask].to_numpy(dtype=float)
            depth_pts = d_s[v_mask].to_numpy(dtype=float)
            h_pts = np.maximum(float(fwl) - depth_pts, 0.2)
            pc_psi = 0.433 * d_rho * h_pts * 3.28084
            j_pts = 0.21645 * (pc_psi / sigma_cos) * np.sqrt(k_pts / phi_pts)
            valid_j = (j_pts > 0.01) & (j_pts < 12.0)
            if valid_j.sum() >= 5:
                sub_n = min(valid_j.sum(), 400)
                idx_j = np.linspace(0, valid_j.sum() - 1, sub_n, dtype=int)
                ax1.scatter(
                    sw_pts[valid_j][idx_j],
                    j_pts[valid_j][idx_j],
                    color="#94a3b8",
                    s=11,
                    alpha=0.45,
                    edgecolors="none",
                    label=f"Measured Log/Plug J(Sw) (n={sub_n})",
                )

    for label, a_param, b_param, col in rt_fits:
        sw_j = np.clip(a_param * (j_grid**b_param), 0.05, 1.0)
        ax1.plot(
            sw_j,
            j_grid,
            color=col,
            lw=2.0,
            label=f"{label}: Sw={a_param:.2f}·J^({b_param:.2f})",
        )

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 8.0)
    ax1.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
    ax1.set_ylabel(
        "Leverett J-Function J(Sw) [dimensionless]", fontsize=9, fontweight="bold"
    )
    tz_title = f" — {', '.join(target_zones)}" if target_zones else ""
    ax1.set_title(
        f"Capillary Pressure Leverett J-Function{tz_title}",
        fontsize=9.0,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=7.0)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Project SCAL Water-Oil Relative Permeability Curves (Corey or LET)
    scal = scal_params or {}
    model_type = str(scal.get("model", "corey")).lower()
    swirr = float(scal.get("swirr", 0.15))
    sor = float(scal.get("sor", 0.20))
    krw0 = float(scal.get("krw_max", 0.45))
    kro0 = float(scal.get("kro_max", 0.90))
    nw = float(scal.get("nw", 2.5))
    no = float(scal.get("no", 2.5))

    sw = np.linspace(swirr, max(1.0 - sor, swirr + 0.05), 100)
    denom_se = max(1.0 - swirr - sor, 0.01)
    se = np.clip((sw - swirr) / denom_se, 0.0, 1.0)

    if model_type == "let":
        lw = float(scal.get("lw", 2.0))
        ew = float(scal.get("ew", 1.5))
        tw = float(scal.get("tw", 2.0))
        lo = float(scal.get("lo", 2.0))
        eo = float(scal.get("eo", 1.5))
        to = float(scal.get("to", 2.0))
        krw = krw0 * (lw * (se**ew)) / np.maximum(lw * (se**ew) + tw * ((1.0 - se) ** ew), 1e-9)
        kro = kro0 * (lo * ((1.0 - se) ** eo)) / np.maximum(lo * ((1.0 - se) ** eo) + to * (se**eo), 1e-9)
        krw_label = f"Krw LET (L={lw:.1f}, E={ew:.1f}, T={tw:.1f})"
        kro_label = f"Kro LET (L={lo:.1f}, E={eo:.1f}, T={to:.1f})"
    else:
        krw = krw0 * (se**nw)
        kro = kro0 * ((1.0 - se) ** no)
        krw_label = f"Krw Corey (Nw={nw:.1f}, Krw0={krw0:.2f})"
        kro_label = f"Kro Corey (No={no:.1f}, Kro0={kro0:.2f})"

    ax2.plot(sw, krw, color="#2563eb", lw=2.2, label=krw_label)
    ax2.plot(sw, kro, color="#059669", lw=2.2, label=kro_label)
    ax2.axvline(swirr, color="#64748b", linestyle=":", lw=1.1, label=f"Swirr = {swirr:.2f}")
    ax2.axvline(1.0 - sor, color="#dc2626", linestyle=":", lw=1.1, label=f"1 - Sor = {1.0 - sor:.2f}")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Relative Permeability Kr (fraction)", fontsize=9, fontweight="bold")

    scal_source = scal.get("source_label") or (
        "Analogue / Default Corey SCAL RelPerm"
        if is_analogue_scal
        else f"Measured Project SCAL {model_type.upper()} RelPerm"
    )
    ax2.set_title(
        scal_source,
        fontsize=8.5,
        fontweight="bold",
    )
    ax2.legend(loc="center right", fontsize=7.2)
    ax2.grid(True, linestyle=":", alpha=0.5)

    return _save_fig(fig, "fig_7_1_shf_scal_plot.png", output_dir, url_prefix)


def generate_uncertainty_cutoff_plot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    target_zones: Optional[Sequence[str]] = None,
    ressum_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 8.1: Cut-off Sensitivity Carpet Plot & P10-P50-P90 HCPT Uncertainty.

    Args:
        df (pd.DataFrame): Multi-well log DataFrame containing `PHIE`/`PHIT`, `VCLAY`, and `SWT`.
        well_names (Optional[List[str]]): Optional subset of well names to filter.
        cutoffs (Optional[Dict[str, float]]): Cut-off dictionary (`vsh`, `phie`, `sw`).
        target_zones (Optional[Sequence[str]]): Optional target reservoir zone names.
        ressum_df (Optional[pd.DataFrame]): Optional pre-computed reservoir summary DataFrame.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_8_1_uncertainty_plot.png`.
    """
    sub = _ensure_df_well_filter(df, well_names)
    if target_zones and "ZONES" in sub.columns and sub["ZONES"].notna().any():
        tz_set = {str(z).strip().upper() for z in target_zones if z}
        tz_sub = sub[sub["ZONES"].astype(str).str.strip().str.upper().isin(tz_set)]
        if not tz_sub.empty:
            sub = tz_sub

    cut = cutoffs or {"vsh": 0.40, "phie": 0.08, "sw": 0.60}
    vsh_cut = float(cut.get("vsh", cut.get("VSHALE", 0.40)))
    phi_cut = float(cut.get("phie", cut.get("PHIT", 0.08)))
    sw_cut = float(cut.get("sw", cut.get("SWT", 0.60)))

    fig = Figure(figsize=(10.5, 4.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    phie = _get_numeric_series(sub, "PHIE", "PHIT").to_numpy(dtype=float)
    vclay = _get_numeric_series(sub, "VCLAY").to_numpy(dtype=float)
    swt = _get_numeric_series(sub, "SWT", "SW").to_numpy(dtype=float)
    n_wells = max(1, int(sub["WELL_NAME"].nunique()) if "WELL_NAME" in sub.columns else 1)
    valid_m = (phie > 0.0) & np.isfinite(phie) & np.isfinite(vclay) & np.isfinite(swt)
    if valid_m.any():
        phie = phie[valid_m]
        vclay = vclay[valid_m]
        swt = swt[valid_m]
    else:
        phie = np.array([], dtype=float)
        vclay = np.array([], dtype=float)
        swt = np.array([], dtype=float)

    if len(phie) > 0:
        phi_thresholds = np.linspace(0.0, 0.22, 35)
        for v_lim, col in [
            (0.30, PALETTE[0]),
            (vsh_cut, PALETTE[1]),
            (0.55, PALETTE[2]),
        ]:
            hcpt_norm = []
            base_hc = np.sum(np.maximum(0.0, phie * (1.0 - swt)))
            base_hc = max(float(base_hc), 1e-6)
            for p_lim in phi_thresholds:
                mask = (vclay <= v_lim) & (phie >= p_lim) & (swt <= sw_cut)
                hc = float(np.sum(np.where(mask, phie * (1.0 - swt), 0.0)))
                hcpt_norm.append(100.0 * hc / base_hc)
            ax1.plot(
                phi_thresholds, hcpt_norm, color=col, lw=2.0, label=f"Vsh ≤ {v_lim:.2f}"
            )

        ax1.axvline(
            phi_cut,
            color="#dc2626",
            linestyle="--",
            lw=1.4,
            label=f"Selected Φ Cutoff ({phi_cut:.2f})",
        )
        ax1.set_xlim(0.0, 0.22)
        ax1.set_ylim(0.0, 105.0)
        ax1.set_xlabel(
            "Effective Porosity Cut-off Φ_cutoff (v/v)", fontsize=9, fontweight="bold"
        )
        ax1.set_ylabel("Cumulative HCPT Retained (%)", fontsize=9, fontweight="bold")
        ax1.set_title(
            "Net Pay Cut-off Sensitivity ('Carpet' Analysis)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax1.legend(loc="lower left", fontsize=8)
        ax1.grid(True, linestyle=":", alpha=0.5)

        # Panel 2: Empirical Monte Carlo P90 / P50 / P10 HCPT Perturbation Distribution (N=1000)
        rng = np.random.default_rng(42)
        n_reals = 1000
        d_phi = rng.normal(0.0, 0.018, size=(n_reals, 1))
        d_vsh = rng.normal(0.0, 0.035, size=(n_reals, 1))
        sw_mult = rng.lognormal(0.0, 0.11, size=(n_reals, 1))

        # Subsample up to 4000 pts for fast vectorized 1000xN matrix evaluation
        if len(phie) > 4000:
            idx_mc = np.linspace(0, len(phie) - 1, 4000, dtype=int)
            phi_mc = phie[idx_mc][None, :]
            vsh_mc = vclay[idx_mc][None, :]
            sw_mc = swt[idx_mc][None, :]
            step_h = 0.1524 * (len(phie) / 4000.0) / float(n_wells)
        else:
            phi_mc = phie[None, :]
            vsh_mc = vclay[None, :]
            sw_mc = swt[None, :]
            step_h = 0.1524 / float(n_wells)

        phi_r = np.clip(phi_mc + d_phi, 0.0, 0.45)
        vsh_r = np.clip(vsh_mc + d_vsh, 0.0, 1.0)
        sw_r = np.clip(sw_mc * sw_mult, 0.02, 1.0)
        pay_r = (vsh_r <= vsh_cut) & (phi_r >= phi_cut) & (sw_r <= sw_cut)
        sim_hcpt = np.sum(np.where(pay_r, phi_r * (1.0 - sw_r) * step_h, 0.0), axis=1)

        p90_v, p50_v, p10_v = np.nanpercentile(sim_hcpt, [10, 50, 90])

        if p10_v > 0.001:
            ax2.hist(
                sim_hcpt,
                bins=30,
                density=True,
                color="#3b82f6",
                alpha=0.65,
                edgecolor="#1e3a8a",
            )
            ax2.axvline(
                p90_v,
                color="#dc2626",
                linestyle="--",
                lw=1.6,
                label=f"P90 (Low): {p90_v:.2f} m",
            )
            ax2.axvline(
                p50_v,
                color="#0f172a",
                linestyle="-",
                lw=1.8,
                label=f"P50 (Base): {p50_v:.2f} m",
            )
            ax2.axvline(
                p10_v,
                color="#059669",
                linestyle="--",
                lw=1.6,
                label=f"P10 (High): {p10_v:.2f} m",
            )
            ax2.legend(loc="upper right", fontsize=8)
        else:
            ax2.text(
                0.5,
                0.5,
                "Deterministic & Probabilistic Net Pay = 0.00 m\n(100% Water-Bearing across Target Cut-offs)",
                ha="center",
                va="center",
                fontsize=9,
                color="#475569",
            )
        ax2.set_xlabel(
            "Mean Well Hydrocarbon Pore Thickness HCPT (m)", fontsize=9, fontweight="bold"
        )
        ax2.set_ylabel("Probability Density", fontsize=9, fontweight="bold")
        ax2.set_title(
            "Monte Carlo HCPT Uncertainty (N=1000 Realizations)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax2.grid(True, linestyle=":", alpha=0.5)

    return _save_fig(fig, "fig_8_1_uncertainty_plot.png", output_dir, url_prefix)


def generate_upscaling_comparison_plot(
    df: pd.DataFrame,
    well_name: str,
    cell_size: float = 2.0,
    cutoffs: Optional[Dict[str, float]] = None,
    target_zones: Optional[Sequence[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 9.1: Fine-Scale vs. Upscaled Blocked Logs Comparison Track (Target-Zoomed <= 500 m).

    Args:
        df (pd.DataFrame): Log DataFrame containing `DEPTH`, `PHIT`, `SWT`, and `PERM`.
        well_name (str): Name of the reference well to upscale and plot.
        cell_size (float): Vertical grid cell thickness in meters. Defaults to 2.0.
        cutoffs (Optional[Dict[str, float]]): Optional reservoir/pay cut-offs.
        target_zones (Optional[Sequence[str]]): Optional target reservoir zone names for auto-zooming.
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `fig_9_1_upscaling_plot.png`.
    """
    from quick_pp.upscaling import expand_blocked_logs, upscale_logs

    sub = _trim_to_logged_interval(_ensure_df_well_filter(df, [well_name]))
    fig = Figure(figsize=(10.8, 5.2))
    gs = GridSpec(1, 3, figure=fig, wspace=0.18)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax3 = fig.add_subplot(gs[0, 2], sharey=ax1)

    if sub.empty or "DEPTH" not in sub.columns:
        ax1.text(
            0.5,
            0.5,
            "No depth log data for upscaling preview",
            ha="center",
            va="center",
        )
        return _save_fig(fig, "fig_9_1_upscaling_plot.png", output_dir, url_prefix)

    ymin, ymax = _resolve_target_depth_window(sub, target_zones=target_zones, max_span=480.0)
    d_col = pd.to_numeric(sub["DEPTH"], errors="coerce")
    sub_win = sub[(d_col >= ymin) & (d_col <= ymax)].copy()
    if sub_win.empty:
        sub_win = sub.copy()

    # Preserve NaN outside evaluated intervals — never pad with constant 0.15 / 0.45 / 25.0
    work = pd.DataFrame(
        {
            "DEPTH": _get_numeric_series(sub_win, "DEPTH"),
            "VCLAY": _get_numeric_series(sub_win, "VCLAY", "VSHALE"),
            "PHIT": _get_numeric_series(sub_win, "PHIE", "PHIT"),
            "SWT": _get_numeric_series(sub_win, "SWT", "SW", "SHF"),
            "PERM": _get_numeric_series(sub_win, "PERM", "PERM_TRANS").clip(lower=1e-2),
            "ZONES": sub_win["ZONES"].fillna("Target").astype(str)
            if "ZONES" in sub_win.columns
            else "Target",
        }
    ).dropna(subset=["DEPTH", "PHIT", "SWT"], how="any")

    if work.empty:
        work = pd.DataFrame(
            {
                "DEPTH": _get_numeric_series(sub_win, "DEPTH"),
                "VCLAY": _get_numeric_series(sub_win, "VCLAY", "VSHALE"),
                "PHIT": _get_numeric_series(sub_win, "PHIE", "PHIT"),
                "SWT": _get_numeric_series(sub_win, "SWT", "SW", "SHF"),
                "PERM": _get_numeric_series(sub_win, "PERM", "PERM_TRANS").clip(lower=1e-2),
                "ZONES": sub_win["ZONES"].fillna("Target").astype(str)
                if "ZONES" in sub_win.columns
                else "Target",
            }
        ).dropna(subset=["DEPTH"])

    try:
        blocked = upscale_logs(
            work,
            cell_size=cell_size,
            by_zone=True,
            depth_col="DEPTH",
            zone_col="ZONES",
            perm_method="geometric",
        )
        step_df = expand_blocked_logs(work, blocked, depth_col="DEPTH", suffix="_UP")
    except Exception as exc:
        logger.debug(f"Upscaling preview fallback: {exc}")
        step_df = work.copy()

    d = work["DEPTH"].to_numpy(dtype=float)
    ax1.plot(work["PHIT"], d, color="#93c5fd", lw=0.8, label="Fine-Scale Φe")
    if "PHIT_UP" in step_df.columns:
        ax1.plot(
            step_df["PHIT_UP"],
            d,
            color="#1d4ed8",
            lw=1.8,
            label=f"Blocked ({cell_size}m Arith)",
        )
    ax1.set_xlim(0.0, 0.40)
    ax1.set_ylabel("Depth (m MD)", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Effective Porosity Φe (v/v)", fontsize=9, fontweight="bold")
    ax1.set_title("Porosity Blocking", fontsize=9.5, fontweight="bold")
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    ax2.plot(work["SWT"], d, color="#fca5a5", lw=0.8, label="Fine-Scale Sw")
    if "SWT_UP" in step_df.columns:
        ax2.plot(
            step_df["SWT_UP"], d, color="#dc2626", lw=1.8, label="Blocked (PV-Weighted)"
        )
    ax2.set_xlim(0.0, 1.0)
    ax2.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
    ax2.set_title("Saturation Blocking", fontsize=9.5, fontweight="bold")
    ax2.legend(loc="upper right", fontsize=7.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    ax3.plot(work["PERM"], d, color="#bbf7d0", lw=0.8, label="Fine-Scale K")
    if "PERM_GEOM_UP" in step_df.columns:
        ax3.plot(
            step_df["PERM_GEOM_UP"], d, color="#059669", lw=1.8, label="Blocked K_geom"
        )
    if "PERM_HARM_UP" in step_df.columns:
        ax3.plot(
            step_df["PERM_HARM_UP"],
            d,
            color="#7c3aed",
            lw=1.2,
            linestyle="--",
            label="Blocked K_harm (Kv)",
        )
    ax3.set_xscale("log")
    ax3.set_xlim(1e-2, 1e4)
    ax3.set_xlabel("Permeability K (mD)", fontsize=9, fontweight="bold")
    ax3.set_title("Permeability Upscaling", fontsize=9.5, fontweight="bold")
    ax3.legend(loc="upper right", fontsize=7.5)
    ax3.grid(True, which="both", linestyle=":", alpha=0.5)

    # Formation tops overlay across all 3 tracks
    if "ZONES" in work.columns and work["ZONES"].notna().any() and len(d) > 0:
        z_s = work["ZONES"].astype(str)
        chg = [0] + [i for i in range(1, len(z_s)) if z_s.iloc[i] != z_s.iloc[i - 1]]
        for idx in chg[:10]:
            zd = float(d[idx])
            zl = str(z_s.iloc[idx]).strip()
            if not zl or zl.upper() in ("NAN", "ALL", "MAIN RESERVOIR", "UNKNOWN"):
                continue
            for a_i in (ax1, ax2, ax3):
                a_i.axhline(zd, color="k", linestyle="--", lw=0.8, alpha=0.75)
            ax1.text(
                0.01,
                zd + (ymax - ymin) * 0.014,
                zl,
                fontsize=7.0,
                fontweight="bold",
                color="#0f172a",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 1.0},
            )

    ax1.set_ylim(ymax, ymin)
    fig.suptitle(
        f"Well {well_name} — Target Reservoir Fine-Scale CPI vs. {cell_size}m Cellular Grid Upscaled Logs ({ymin:.1f}–{ymax:.1f} m MD)",
        fontsize=10.0,
        fontweight="bold",
        y=1.02,
    )
    return _save_fig(fig, "fig_9_1_upscaling_plot.png", output_dir, url_prefix)


def generate_well_cpi_plot(
    df: pd.DataFrame,
    well_name: str,
    zones: Optional[List[str]] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Appendix Composite Petrophysical Evaluation Log (CPI) 6-Track Plot (Target-Zoomed <= 500 m per CHK-06).

    Args:
        df (pd.DataFrame): Well log DataFrame containing `DEPTH`, `GR`, `RT`, `NPHI`, `RHOB`, `PHIT`, `PERM`, and `SWT`.
        well_name (str): Name of the well to render.
        zones (Optional[List[str]]): Optional target reservoir zone names for depth window auto-zooming.
        cutoffs (Optional[Dict[str, float]]): Optional Net Reservoir / Net Pay cut-offs (`vsh`, `phie`, `sw`).
        output_dir (Optional[Union[str, Path]]): Directory where the rendered PNG will be written.
        url_prefix (Optional[str]): Optional URL prefix to prepend to the returned filename.

    Returns:
        str: File path or URL to the generated `cpi_<well_name>.png`.
    """
    sub_well = _ensure_df_well_filter(df, [well_name])
    safe_well = "".join(
        c if c.isalnum() or c in ("-", "_") else "_" for c in str(well_name)
    )
    filename = f"cpi_{safe_well}.png"

    fig = Figure(figsize=(12.5, 8.2))
    gs = GridSpec(
        1,
        6,
        figure=fig,
        width_ratios=[1.0, 1.0, 1.0, 1.15, 1.05, 1.15],
        wspace=0.08,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(6)]

    if sub_well.empty or "DEPTH" not in sub_well.columns:
        axes[0].text(0.5, 0.5, f"No log data for {well_name}", ha="center", va="center")
        return _save_fig(fig, filename, output_dir, url_prefix)

    # Ensure strictly sorted monotonic depth index to prevent any zig-zag line rendering
    sub_well = (
        sub_well.dropna(subset=["DEPTH"])
        .sort_values(by="DEPTH")
        .reset_index(drop=True)
    )
    sub_well = _trim_to_logged_interval(sub_well)

    # Auto-zoom to target reservoir window (CHK-06: vertical span <= 500 m)
    d_min, d_max = _resolve_target_depth_window(sub_well, target_zones=zones, max_span=480.0)
    d_col = pd.to_numeric(sub_well["DEPTH"], errors="coerce")
    sub_full = sub_well[(d_col >= d_min) & (d_col <= d_max)].copy()
    if sub_full.empty:
        sub_full = sub_well.copy()
    sub_full = (
        sub_full.dropna(subset=["DEPTH"])
        .sort_values(by="DEPTH")
        .reset_index(drop=True)
    )
    d_full = pd.to_numeric(sub_full["DEPTH"], errors="coerce").to_numpy(dtype=float)

    cut = cutoffs or {"vsh": 0.40, "phie": 0.08, "sw": 0.65}
    vsh_cut = float(cut.get("vsh", cut.get("VSHALE", 0.40)))
    phi_cut = float(cut.get("phie", cut.get("PHIT", 0.08)))
    sw_cut = float(cut.get("sw", cut.get("SWT", 0.65)))

    # Downsample continuous log tracks if needed while preserving all discrete CPORE/CPERM rows on sub_full
    if len(sub_full) > 2500:
        step_cpi = max(1, len(sub_full) // 2500)
        sub = sub_full.iloc[::step_cpi]
        d = d_full[::step_cpi]
    else:
        sub = sub_full
        d = d_full

    # Track 1: GR with Clean Sand (Yellow) & Shale (Green/Brown) Lithology Shading
    ax = axes[0]
    gr = _get_numeric_series(sub, "GR").to_numpy(dtype=float)
    if np.isfinite(gr).any():
        gr_clip = np.clip(gr, 0.0, 150.0)
        sand_baseline = 65.0
        ax.plot(gr_clip, d, color="#14532d", lw=1.1, label="GR")
        # Yellow fill left of curve for low GR (clean sand)
        ax.fill_betweenx(
            d,
            0.0,
            gr_clip,
            where=(gr_clip <= sand_baseline),
            color="#fef08a",
            alpha=0.75,
            label="Clean Sand",
        )
        # Green/olive fill for shale (high GR)
        ax.fill_betweenx(
            d,
            0.0,
            gr_clip,
            where=(gr_clip > sand_baseline),
            color="#86efac",
            alpha=0.55,
            label="Shale / Argillaceous",
        )
    ax.set_xlim(0, 150)
    ax.set_xlabel("GR (gAPI)", fontsize=8.5, fontweight="bold", color="#15803d")
    ax.set_ylabel("Depth (m MD)", fontsize=9.5, fontweight="bold")
    ax.set_title("Track 1\nGamma Ray", fontsize=8.5, fontweight="bold")

    # Track 2: Deep Resistivity RT
    ax = axes[1]
    rt = _get_numeric_series(sub, "RT", "RD").clip(lower=0.1).to_numpy(dtype=float)
    if np.isfinite(rt).any():
        ax.plot(rt, d, color="#dc2626", lw=1.1, label="RT")
    ax.set_xscale("log")
    ax.set_xlim(0.2, 200.0)
    ax.set_xlabel("RT (Ω·m)", fontsize=8.5, fontweight="bold", color="#dc2626")
    ax.set_title("Track 2\nResistivity", fontsize=8.5, fontweight="bold")

    # Track 3: RHOB & NPHI with Yellow Clean Sand / HC Crossover Fill
    ax = axes[2]
    nphi = _get_numeric_series(sub, "NPHI").to_numpy(dtype=float)
    rhob = _get_numeric_series(sub, "RHOB").to_numpy(dtype=float)
    if np.isfinite(nphi).any():
        ax.plot(nphi, d, color="#2563eb", lw=1.1, linestyle="--", label="NPHI")
    if np.isfinite(rhob).any():
        # Normalize RHOB (1.95 -> 2.95) onto NPHI (0.45 -> -0.15) scale for crossover display
        nphi_from_rhob = 0.45 - ((rhob - 1.95) / 1.0) * 0.60
        ax.plot(nphi_from_rhob, d, color="#b91c1c", lw=1.1, label="RHOB (scaled)")
        ax.fill_betweenx(
            d,
            nphi,
            nphi_from_rhob,
            where=(nphi < nphi_from_rhob),
            color="#fde047",
            alpha=0.70,
            label="Sand / HC Crossover",
        )
    ax.set_xlim(0.45, -0.15)
    ax.set_xlabel("NPHI (v/v) / RHOB", fontsize=8.5, fontweight="bold", color="#1e40af")
    ax.set_title("Track 3\nDensity-Neutron", fontsize=8.5, fontweight="bold")

    # Track 4: Lithology & Porosity + Core CPORE + Conformal P10/P90
    ax = axes[3]
    phit = _get_numeric_series(sub, "PHIT", "PHIE").to_numpy(dtype=float)
    phie = _get_numeric_series(sub, "PHIE", "PHIT").to_numpy(dtype=float)
    vclay = _get_numeric_series(sub, "VCLAY", "VSHALE").to_numpy(dtype=float)
    if "PHIT_P10" in sub.columns and "PHIT_P90" in sub.columns:
        p10 = _get_numeric_series(sub, "PHIT_P10").to_numpy(dtype=float)
        p90 = _get_numeric_series(sub, "PHIT_P90").to_numpy(dtype=float)
        ax.fill_betweenx(d, p10, p90, color="#93c5fd", alpha=0.45, label="P10-P90")
    if np.isfinite(phit).any():
        ax.plot(phit, d, color="#1d4ed8", lw=1.2, label="PHIT")
    if np.isfinite(phie).any():
        ax.plot(phie, d, color="#0d9488", lw=1.0, linestyle=":", label="PHIE")
    if "CPORE" in sub_full.columns:
        cp = _get_numeric_series(sub_full, "CPORE").to_numpy(dtype=float)
        cp = np.where(cp > 1.0, cp / 100.0, cp)
        cm = np.isfinite(cp) & (cp > 0)
        if np.any(cm):
            ax.scatter(cp[cm], d_full[cm], color="#dc2626", s=18, zorder=5, label="CPORE")
    ax.set_xlim(0.0, 0.50)
    ax.set_xlabel("Porosity Φ (v/v)", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    ax.set_title("Track 4\nPorosity & Core", fontsize=8.5, fontweight="bold")

    # Track 5: Permeability & Core CPERM
    ax = axes[4]
    perm = _get_numeric_series(sub, "PERM", "PERM_TRANS").clip(lower=1e-2).to_numpy(dtype=float)
    if np.isfinite(perm).any():
        ax.plot(perm, d, color="#059669", lw=1.1, label="PERM")
    if "CPERM" in sub_full.columns:
        ck = _get_numeric_series(sub_full, "CPERM").to_numpy(dtype=float)
        km = np.isfinite(ck) & (ck > 0)
        if np.any(km):
            ax.scatter(ck[km], d_full[km], color="#b91c1c", s=18, zorder=5, label="CPERM")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e4)
    ax.set_xlabel("PERM (mD)", fontsize=8.5, fontweight="bold", color="#059669")
    ax.set_title("Track 5\nPermeability", fontsize=8.5, fontweight="bold")

    # Track 6: Water Saturation SWT, SHF, Net Reservoir (Green) & Net Pay (Dark Green / Red) Shading
    ax = axes[5]
    swt = _get_numeric_series(sub, "SWT", "SW").to_numpy(dtype=float)
    shf = _get_numeric_series(sub, "SHF").to_numpy(dtype=float)
    if "SW_P10" in sub.columns and "SW_P90" in sub.columns:
        s10 = _get_numeric_series(sub, "SW_P10").to_numpy(dtype=float)
        s90 = _get_numeric_series(sub, "SW_P90").to_numpy(dtype=float)
        ax.fill_betweenx(d, s10, s90, color="#ddd6fe", alpha=0.55)

    res_flag = (
        np.isfinite(phie)
        & (phie >= phi_cut)
        & (np.where(np.isfinite(vclay), vclay, 0.2) <= vsh_cut)
    )
    pay_flag = res_flag & np.isfinite(swt) & (swt <= sw_cut)

    if np.any(res_flag):
        # Green fill on right margin for Net Reservoir flag
        ax.fill_betweenx(
            d,
            0.92,
            1.00,
            where=res_flag,
            color="#86efac",
            alpha=0.80,
            label="Net Reservoir",
        )
    if np.isfinite(swt).any():
        ax.plot(swt, d, color="#1d4ed8", lw=1.2, label="SWT")
        # Dark green hydrocarbon saturation shading for Net Pay intervals
        ax.fill_betweenx(
            d,
            swt,
            1.0,
            where=pay_flag,
            color="#15803d",
            alpha=0.45,
            label="Net Pay (HC)",
        )
    if np.isfinite(shf).any():
        ax.plot(shf, d, color="#dc2626", lw=1.1, linestyle="--", label="SHF")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Sw / SHF (v/v)", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    ax.set_title("Track 6\nSaturation & Pay", fontsize=8.5, fontweight="bold")

    # Formation top dashed lines across all tracks with Track 1 annotations
    if "ZONES" in sub.columns and sub["ZONES"].notna().any():
        z_series = sub["ZONES"].astype(str)
        change_idx = [0] + [
            i
            for i in range(1, len(z_series))
            if z_series.iloc[i] != z_series.iloc[i - 1]
        ]
        for idx in change_idx[:14]:
            z_depth = float(d[idx])
            z_label = str(z_series.iloc[idx]).strip()
            if not z_label or z_label.upper() in ("NAN", "MAIN RESERVOIR", "UNKNOWN", "ALL"):
                continue
            for a_i in axes:
                a_i.axhline(z_depth, color="k", linestyle="--", lw=0.85, alpha=0.85)
            axes[0].text(
                4.0,
                z_depth + (d_max - d_min) * 0.013,
                z_label,
                fontsize=7.5,
                fontweight="bold",
                color="#0f172a",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.85,
                    "edgecolor": "#cbd5e1",
                    "pad": 1.2,
                },
            )

    for idx, a_i in enumerate(axes):
        a_i.set_ylim(d_max, d_min)
        a_i.grid(True, which="both", linestyle=":", alpha=0.45)
        if idx > 0:
            a_i.tick_params(labelleft=False)

    fig.suptitle(
        f"Composite Petrophysical Evaluation Log (CPI) — Well {well_name} (Target Window: {d_min:.1f}–{d_max:.1f} m MD)",
        fontsize=11,
        fontweight="bold",
        y=0.98,
        color="#0f172a",
    )
    return _save_fig(fig, filename, output_dir, url_prefix)
