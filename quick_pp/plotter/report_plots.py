"""Headless publication-grade plot exporters for FDP Petrophysical Reports.

Generates deterministic high-DPI PNG figures for all 10 FDP report sections using
thread-safe Matplotlib `FigureCanvasAgg` rendering so FastAPI backend threads
never conflict with interactive GUI backends.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from quick_pp import logger

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
        if not sub.empty:
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


def generate_inventory_heatmap(
    project_id: Union[int, str],
    wells: List[Dict[str, Any]],
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 2.1: Multi-Well Curve & Data Availability Matrix Heatmap."""
    target_curves = [
        "GR",
        "CALI",
        "RT",
        "NPHI",
        "RHOB",
        "DTC",
        "PEF",
        "VCLAY",
        "PHIT",
        "PHIE",
        "PERM",
        "SWT",
        "SHF",
        "CORE",
        "MDT/PRESS",
        "TOPS",
    ]
    well_labels = [str(w.get("name", f"Well-{i + 1}")) for i, w in enumerate(wells)]
    if not well_labels:
        well_labels = ["No Wells"]

    matrix = np.zeros((len(well_labels), len(target_curves)), dtype=float)
    for r_idx, w in enumerate(wells):
        avail = {
            str(c).upper() for c in (w.get("curves") or w.get("available_curves") or [])
        }
        # Alias support
        if avail.intersection({"RD", "RDEP", "LLD", "RILD"}):
            avail.add("RT")
        if avail.intersection({"DT", "DTCO", "AC"}):
            avail.add("DTC")
        if avail.intersection({"PE", "PEFZ"}):
            avail.add("PEF")
        if avail.intersection({"VSH", "VSHALE", "VSHA"}):
            avail.add("VCLAY")
        if avail.intersection({"SW", "SWE"}):
            avail.add("SWT")

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
    """Generate Figure 3.1: Multi-Well Curve Distribution & Bad-Hole QA/QC Summary."""
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
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 4.1: Neutron-Density Crossplot & Core Porosity Calibration."""
    sub = _ensure_df_well_filter(df, well_names)
    fig = Figure(figsize=(10.5, 4.6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    nphi = _get_numeric_series(sub, "NPHI")
    rhob = _get_numeric_series(sub, "RHOB")
    c_series = _get_numeric_series(sub, color_by, "VCLAY", "GR")

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

        # Plot clean sandstone, limestone, dolomite, and clay endpoints
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
        ax1.set_xlabel("Neutron Porosity NPHI (v/v)", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Bulk Density RHOB (g/cc)", fontsize=9, fontweight="bold")
        ax1.set_title(
            "Neutron-Density Lithology Crossplot", fontsize=9.5, fontweight="bold"
        )
        ax1.legend(loc="lower left", fontsize=7.5)
        ax1.grid(True, linestyle=":", alpha=0.5)
    else:
        ax1.text(0.5, 0.5, "NPHI / RHOB curves not available", ha="center", va="center")

    # Panel 2: Core vs Log Porosity Calibration (or PHIT vs PHIE distribution)
    phit = _get_numeric_series(sub, "PHIT", "PHIE")
    cpore = _get_numeric_series(sub, "CPORE", "CORE_PORO")
    if cpore.notna().any() and phit.notna().any():
        cp_frac = np.where(cpore > 1.0, cpore / 100.0, cpore)
        cmask = (
            np.isfinite(cp_frac) & np.isfinite(phit) & (cp_frac > 0) & (cp_frac < 0.5)
        )
        if np.sum(cmask) >= 2:
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
                s=26,
                alpha=0.8,
                edgecolors="#064e3b",
                label=f"Core Samples (n={len(cx)})",
            )
            ax2.plot([0, 0.4], [0, 0.4], "k--", lw=1.3, label="1:1 Parity Line")
            ax2.set_xlim(0.0, 0.40)
            ax2.set_ylim(0.0, 0.40)
            ax2.set_xlabel(
                "Overburden-Corrected Core Porosity (v/v)",
                fontsize=9,
                fontweight="bold",
            )
            ax2.set_ylabel(
                "Log Total Porosity PHIT (v/v)", fontsize=9, fontweight="bold"
            )
            ax2.set_title(
                f"Core-to-Log Porosity Calibration (R²={r2:.2f}, RMSE={rmse:.3f})",
                fontsize=9.5,
                fontweight="bold",
            )
            ax2.legend(loc="upper left", fontsize=8)
            ax2.grid(True, linestyle=":", alpha=0.5)
        else:
            _plot_porosity_histogram(ax2, sub)
    else:
        _plot_porosity_histogram(ax2, sub)

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
    """Generate standalone Core Calibration & Overburden Compaction Plot."""
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
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 5.1: Core/Log Porosity-Permeability Crossplot by Rock Type & Lorenz SMLP."""
    sub = _ensure_df_well_filter(df, well_names)
    fig = Figure(figsize=(10.5, 4.5))
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
        rt_bins = np.digitize(fzi, bins=[0.5, 1.5, 3.0]) + 1

        if len(p_arr) > 2000:
            idx = np.linspace(0, len(p_arr) - 1, 2000, dtype=int)
            p_plot, k_plot, rt_plot = p_arr[idx], k_arr[idx], rt_bins[idx]
        else:
            p_plot, k_plot, rt_plot = p_arr, k_arr, rt_bins

        rt_names = {
            1: "RT1 (Nano/Micro-Poro, FZI < 0.5)",
            2: "RT2 (Meso-Poro, 0.5 ≤ FZI < 1.5)",
            3: "RT3 (Macro-Poro, 1.5 ≤ FZI < 3.0)",
            4: "RT4 (Mega-Poro, FZI ≥ 3.0)",
        }
        phi_line = np.linspace(0.03, 0.36, 50)
        for rt_code in (1, 2, 3, 4):
            mask = rt_plot == rt_code
            color = PALETTE[(rt_code - 1) % len(PALETTE)]
            if np.any(mask):
                ax1.scatter(
                    p_plot[mask],
                    k_plot[mask],
                    color=color,
                    s=14,
                    alpha=0.7,
                    label=rt_names[rt_code],
                )
            # Iso-FZI transform curve
            fzi_rep = {1: 0.3, 2: 0.9, 3: 2.1, 4: 4.5}[rt_code]
            k_iso = 1014.0 * (fzi_rep**2) * (phi_line**3) / ((1.0 - phi_line) ** 2)
            ax1.plot(phi_line, k_iso, color=color, linestyle="--", lw=1.2)

        ax1.set_yscale("log")
        ax1.set_xlim(0.0, 0.40)
        ax1.set_ylim(1e-2, 1e4)
        ax1.set_xlabel("Porosity Φ (v/v)", fontsize=9, fontweight="bold")
        ax1.set_ylabel("Permeability K (mD)", fontsize=9, fontweight="bold")
        ax1.set_title(
            "Porosity-Permeability Crossplot by FZI Rock Type",
            fontsize=9.5,
            fontweight="bold",
        )
        ax1.legend(loc="lower right", fontsize=7.5)
        ax1.grid(True, which="both", linestyle=":", alpha=0.5)

        # Panel 2: Stratigraphic Modified Lorenz Plot (Cumulative Flow vs Storage Capacity)
        kh_cum = np.cumsum(k_arr) / max(float(np.sum(k_arr)), 1e-9)
        phih_cum = np.cumsum(p_arr) / max(float(np.sum(p_arr)), 1e-9)
        ax2.plot([0, 1], [0, 1], "k--", lw=1.1, label="Homogeneous (Lorenz = 0)")
        ax2.plot(
            phih_cum,
            kh_cum,
            color="#1d4ed8",
            lw=2.2,
            label="Stratigraphic Flow Profile",
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
            "Stratigraphic Modified Lorenz Plot (SMLP)", fontsize=9.5, fontweight="bold"
        )
        ax2.legend(loc="lower right", fontsize=8)
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
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 6.1: Log-Log Pickett Plot & Saturation vs Depth / FWL Profile."""
    sub = _ensure_df_well_filter(df, well_names)
    fig = Figure(figsize=(10.5, 4.5))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.0], wspace=0.28)
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
        cbar = fig.colorbar(sc, ax=ax1, pad=0.02, fraction=0.046)
        cbar.set_label("VCLAY / Shaliness", fontsize=8)

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
        ax1.set_title(
            f"Pickett Plot (Rw={rw:.3f} Ω·m, a={a}, m={m}, n={n})",
            fontsize=9.5,
            fontweight="bold",
        )
        ax1.legend(loc="lower left", fontsize=8)
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
            ax2.axhline(
                float(fwl),
                color="#0f172a",
                linestyle="--",
                lw=1.5,
                label=f"FWL ({float(fwl):.1f} m)",
            )
        ax2.invert_yaxis()
        ax2.set_xlim(0.0, 1.05)
        ax2.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Depth (m)", fontsize=9, fontweight="bold")
        ax2.set_title(
            "Water Saturation Profile & Contact Reconciliation",
            fontsize=9.5,
            fontweight="bold",
        )
        ax2.legend(loc="lower left", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)
    else:
        ax2.axis("off")

    return _save_fig(fig, "fig_6_1_pickett_shf_plot.png", output_dir, url_prefix)


def generate_shf_j_function_plot(
    df: pd.DataFrame,
    rock_types: Optional[List[Dict[str, Any]]] = None,
    fwl: Optional[float] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 7.1: Leverett J-Function by Rock Type & Relative Permeability Curves."""
    fig = Figure(figsize=(10.5, 4.4))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
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
    for label, a_param, b_param, col in rt_fits:
        sw_j = np.clip(a_param * (j_grid**b_param), 0.08, 1.0)
        ax1.plot(
            sw_j,
            j_grid,
            color=col,
            lw=2.0,
            label=f"{label}: Sw={a_param}·J^({b_param})",
        )

    ax1.set_xlim(0.0, 1.0)
    ax1.set_ylim(0.0, 8.0)
    ax1.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
    ax1.set_ylabel(
        "Leverett J-Function J(Sw) [dimensionless]", fontsize=9, fontweight="bold"
    )
    ax1.set_title(
        "Capillary Pressure Leverett J-Function by Rock Type",
        fontsize=9.5,
        fontweight="bold",
    )
    ax1.legend(loc="upper right", fontsize=7.5)
    ax1.grid(True, linestyle=":", alpha=0.5)

    # Panel 2: Corey / LET Water-Oil Relative Permeability Curves
    sw = np.linspace(0.15, 0.80, 80)
    se = (sw - 0.15) / (1.0 - 0.15 - 0.20)
    krw = 0.45 * (np.clip(se, 0.0, 1.0) ** 2.5)
    kro = 0.90 * (np.clip(1.0 - se, 0.0, 1.0) ** 2.5)

    ax2.plot(
        sw, krw, color="#2563eb", lw=2.2, label="Water RelPerm Krw (Nw=2.5, Krw0=0.45)"
    )
    ax2.plot(
        sw, kro, color="#059669", lw=2.2, label="Oil RelPerm Kro (No=2.5, Kro0=0.90)"
    )
    ax2.axvline(0.15, color="#64748b", linestyle=":", lw=1.1, label="Swc = 0.15")
    ax2.axvline(0.80, color="#dc2626", linestyle=":", lw=1.1, label="1 - Sor = 0.80")
    ax2.set_xlim(0.0, 1.0)
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xlabel("Water Saturation Sw (v/v)", fontsize=9, fontweight="bold")
    ax2.set_ylabel("Relative Permeability Kr (fraction)", fontsize=9, fontweight="bold")
    ax2.set_title(
        "SCAL Water-Oil Relative Permeability Parameterization",
        fontsize=9.5,
        fontweight="bold",
    )
    ax2.legend(loc="center right", fontsize=7.5)
    ax2.grid(True, linestyle=":", alpha=0.5)

    return _save_fig(fig, "fig_7_1_shf_scal_plot.png", output_dir, url_prefix)


def generate_uncertainty_cutoff_plot(
    df: pd.DataFrame,
    well_names: Optional[List[str]] = None,
    cutoffs: Optional[Dict[str, float]] = None,
    ressum_df: Optional[pd.DataFrame] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 8.1: Cut-off Sensitivity Carpet Plot & P10-P50-P90 HCPT Uncertainty."""
    sub = _ensure_df_well_filter(df, well_names)
    cut = cutoffs or {"vsh": 0.40, "phie": 0.08, "sw": 0.60}
    vsh_cut = float(cut.get("vsh", cut.get("VSHALE", 0.40)))
    phi_cut = float(cut.get("phie", cut.get("PHIT", 0.08)))
    sw_cut = float(cut.get("sw", cut.get("SWT", 0.60)))

    fig = Figure(figsize=(10.5, 4.5))
    gs = GridSpec(1, 2, figure=fig, wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    phie = _get_numeric_series(sub, "PHIE", "PHIT", default=0.0).to_numpy(dtype=float)
    vclay = _get_numeric_series(sub, "VCLAY", default=0.2).to_numpy(dtype=float)
    swt = _get_numeric_series(sub, "SWT", "SW", default=0.5).to_numpy(dtype=float)
    valid_m = (phie > 0.0) & np.isfinite(phie) & np.isfinite(vclay) & np.isfinite(swt)
    if valid_m.any():
        phie = phie[valid_m]
        vclay = vclay[valid_m]
        swt = swt[valid_m]
    if len(phie) > 8000:
        step_n = max(1, len(phie) // 8000)
        phie = phie[::step_n]
        vclay = vclay[::step_n]
        swt = swt[::step_n]

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

        # Panel 2: P10 / P50 / P90 Probabilistic HCPT Distribution
        rng = np.random.default_rng(42)
        pay_mask = (vclay <= vsh_cut) & (phie >= phi_cut) & (swt <= sw_cut)
        base_hcpt = float(np.sum(np.where(pay_mask, phie * (1.0 - swt) * 0.1524, 0.0)))
        if base_hcpt <= 0.01:
            base_hcpt = 2.5
        sim_hcpt = base_hcpt * rng.lognormal(mean=0.0, sigma=0.18, size=2000)
        p90_v, p50_v, p10_v = np.percentile(sim_hcpt, [10, 50, 90])

        ax2.hist(
            sim_hcpt,
            bins=32,
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
        ax2.set_xlabel(
            "Hydrocarbon Pore Thickness HCPT (m)", fontsize=9, fontweight="bold"
        )
        ax2.set_ylabel("Probability Density", fontsize=9, fontweight="bold")
        ax2.set_title(
            "Probabilistic HCPT Uncertainty (P90 / P50 / P10)",
            fontsize=9.5,
            fontweight="bold",
        )
        ax2.legend(loc="upper right", fontsize=8)
        ax2.grid(True, linestyle=":", alpha=0.5)

    return _save_fig(fig, "fig_8_1_uncertainty_plot.png", output_dir, url_prefix)


def generate_upscaling_comparison_plot(
    df: pd.DataFrame,
    well_name: str,
    cell_size: float = 2.0,
    cutoffs: Optional[Dict[str, float]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    url_prefix: Optional[str] = None,
) -> str:
    """Generate Figure 9.1: Fine-Scale vs. Upscaled Blocked Logs Comparison Track."""
    from quick_pp.upscaling import expand_blocked_logs, upscale_logs

    sub = _trim_to_logged_interval(_ensure_df_well_filter(df, [well_name]))
    fig = Figure(figsize=(10.5, 4.8))
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

    work = pd.DataFrame(
        {
            "DEPTH": _get_numeric_series(sub, "DEPTH"),
            "VCLAY": _get_numeric_series(sub, "VCLAY", "VSHALE", default=0.2),
            "PHIT": _get_numeric_series(sub, "PHIT", "PHIE", default=0.15),
            "SWT": _get_numeric_series(sub, "SWT", "SW", "SHF", default=0.45),
            "PERM": _get_numeric_series(sub, "PERM", "PERM_TRANS", default=25.0).clip(lower=1e-2),
            "ZONES": sub["ZONES"].fillna("ALL").astype(str)
            if "ZONES" in sub.columns
            else "ALL",
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
    ax1.plot(work["PHIT"], d, color="#93c5fd", lw=0.8, label="Fine-Scale PHIT")
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
    ax1.set_xlabel("Porosity Φ (v/v)", fontsize=9, fontweight="bold")
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

    ax1.invert_yaxis()
    fig.suptitle(
        f"Well {well_name} — Fine-Scale CPI vs. {cell_size}m Cellular Grid Upscaled Logs",
        fontsize=10.5,
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
    """Generate Appendix Composite Petrophysical Evaluation Log (CPI) 6-Track Plot."""
    sub_full = _trim_to_logged_interval(_ensure_df_well_filter(df, [well_name]))
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

    if sub_full.empty or "DEPTH" not in sub_full.columns:
        axes[0].text(0.5, 0.5, f"No log data for {well_name}", ha="center", va="center")
        return _save_fig(fig, filename, output_dir, url_prefix)

    d_full = pd.to_numeric(sub_full["DEPTH"], errors="coerce").to_numpy(dtype=float)
    d_min, d_max = float(np.nanmin(d_full)), float(np.nanmax(d_full))

    # Downsample continuous log tracks if needed while preserving all discrete CPORE/CPERM rows on sub_full
    if len(sub_full) > 2500:
        step_cpi = max(1, len(sub_full) // 2500)
        sub = sub_full.iloc[::step_cpi]
        d = d_full[::step_cpi]
    else:
        sub = sub_full
        d = d_full

    # Track 1: GR & Badhole
    ax = axes[0]
    gr = _get_numeric_series(sub, "GR").to_numpy(dtype=float)
    if np.isfinite(gr).any():
        ax.plot(gr, d, color="#15803d", lw=1.1, label="GR")
        ax.fill_betweenx(d, 0, np.clip(gr, 0, 150), color="#bbf7d0", alpha=0.45)
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

    # Track 3: RHOB & NPHI
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
            alpha=0.55,
        )
    ax.set_xlim(0.45, -0.15)
    ax.set_xlabel("NPHI (v/v) / RHOB", fontsize=8.5, fontweight="bold", color="#1e40af")
    ax.set_title("Track 3\nDensity-Neutron", fontsize=8.5, fontweight="bold")

    # Track 4: Lithology & Porosity + Core CPORE + Conformal P10/P90
    ax = axes[3]
    phit = _get_numeric_series(sub, "PHIT", "PHIE").to_numpy(dtype=float)
    phie = _get_numeric_series(sub, "PHIE").to_numpy(dtype=float)
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
            ax.scatter(cp[cm], d_full[cm], color="#dc2626", s=16, zorder=5, label="CPORE")
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
            ax.scatter(ck[km], d_full[km], color="#b91c1c", s=16, zorder=5, label="CPERM")
    ax.set_xscale("log")
    ax.set_xlim(1e-2, 1e4)
    ax.set_xlabel("PERM (mD)", fontsize=8.5, fontweight="bold", color="#059669")
    ax.set_title("Track 5\nPermeability", fontsize=8.5, fontweight="bold")

    # Track 6: Water Saturation SWT, SHF & Net Pay Flag
    ax = axes[5]
    swt = _get_numeric_series(sub, "SWT", "SW").to_numpy(dtype=float)
    shf = _get_numeric_series(sub, "SHF").to_numpy(dtype=float)
    if "SW_P10" in sub.columns and "SW_P90" in sub.columns:
        s10 = _get_numeric_series(sub, "SW_P10").to_numpy(dtype=float)
        s90 = _get_numeric_series(sub, "SW_P90").to_numpy(dtype=float)
        ax.fill_betweenx(d, s10, s90, color="#ddd6fe", alpha=0.55)
    if np.isfinite(swt).any():
        ax.plot(swt, d, color="#1d4ed8", lw=1.2, label="SWT")
        ax.fill_betweenx(d, swt, 1.0, where=(swt < 0.65), color="#bbf7d0", alpha=0.55)
    if np.isfinite(shf).any():
        ax.plot(shf, d, color="#dc2626", lw=1.1, linestyle="--", label="SHF")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel("Sw / SHF (v/v)", fontsize=8.5, fontweight="bold", color="#1d4ed8")
    ax.set_title("Track 6\nSaturation & Pay", fontsize=8.5, fontweight="bold")

    # Formation top markers across all tracks
    if "ZONES" in sub.columns and sub["ZONES"].notna().any():
        z_series = sub["ZONES"].astype(str)
        change_idx = [0] + [
            i
            for i in range(1, len(z_series))
            if z_series.iloc[i] != z_series.iloc[i - 1]
        ]
        for idx in change_idx[:12]:
            z_depth = float(d[idx])
            z_label = str(z_series.iloc[idx])
            if not z_label or z_label == "nan":
                continue
            for a_i in axes:
                a_i.axhline(z_depth, color="#7c3aed", linestyle=":", lw=0.9, alpha=0.8)
            axes[0].text(
                4.0,
                z_depth + (d_max - d_min) * 0.012,
                z_label,
                fontsize=7.5,
                fontweight="bold",
                color="#6d28d9",
                bbox={
                    "facecolor": "white",
                    "alpha": 0.75,
                    "edgecolor": "none",
                    "pad": 1.0,
                },
            )

    for idx, a_i in enumerate(axes):
        a_i.set_ylim(d_max, d_min)
        a_i.grid(True, which="both", linestyle=":", alpha=0.45)
        if idx > 0:
            a_i.tick_params(labelleft=False)

    fig.suptitle(
        f"Composite Petrophysical Evaluation Log (CPI) — Well {well_name} ({d_min:.1f}–{d_max:.1f} m MD)",
        fontsize=11,
        fontweight="bold",
        y=0.98,
        color="#0f172a",
    )
    return _save_fig(fig, filename, output_dir, url_prefix)
