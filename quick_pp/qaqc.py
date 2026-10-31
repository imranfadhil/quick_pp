import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.ressum import calc_reservoir_summary, flag_interval
from quick_pp.rock_type import estimate_vsh_gr
from quick_pp.utils import length_a_b, nd_triangle_coords

plt.style.use("seaborn-v0_8-paper")


def handle_outer_limit(data, fill=False):
    """Replace values outside of predefined min/max limits with NaN or the limit value.

    This function iterates through the columns of a DataFrame and, for each column
    defined in `Config.VARS` with 'min' or 'max' limits, it replaces out-of-range
    values.

    Args:
        data (pd.DataFrame): The input DataFrame with log data to be processed.
        fill (bool, optional): If True, replaces out-of-range values with the min/max limit.
                               If False, replaces them with NaN. Defaults to False.

    Returns:
        pd.DataFrame: The DataFrame with out-of-range values handled.
    """
    data = data.copy()
    for var, var_dict in Config.VARS.items():
        if ("min" in var_dict.keys()) or ("max" in var_dict.keys()):
            if var in data.columns:
                if not fill:
                    data.loc[:, var] = np.where(
                        data[var] < var_dict["min"], np.nan, data[var]
                    )
                    data.loc[:, var] = np.where(
                        data[var] > var_dict["max"], np.nan, data[var]
                    )
                else:
                    data.loc[:, var] = np.where(
                        data[var] < var_dict["min"], var_dict["min"], data[var]
                    )
                    data.loc[:, var] = np.where(
                        data[var] > var_dict["max"], var_dict["max"], data[var]
                    )
    return data


def badhole_flagging(
    data: pd.DataFrame,
    thold: float = 4.0,
    use_cali: bool = True,
    auto_bitsize: bool = True,
    bitsize_default: float = 8.5,
    use_drho: bool = False,
    drho_threshold: float = 0.15,
    use_rhob_nphi: bool = False,
    rhob_min: float = 1.95,
    rhob_max: float = 2.95,
    nphi_max: float = 0.45,
    exclude_coal: bool = True,
    use_rugosity: bool = False,
    rugosity_window: int = 21,
    rhob_trend_factor: float = 0.05,
    cali_spike_threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate a BADHOLE flag based on caliper, DRHO, density/neutron pad standoff bounds, and rugosity spikes.

    Supports four complementary petrophysical badhole detection criteria:
    1. **Differential Caliper (`use_cali=True`)**: Estimates bit size (`BS`) from `CALI`
       change-point detection (when `auto_bitsize=True` and `BS` is absent) or uses
       `bitsize_default`, and flags intervals where `|CALI - BS| >= thold`.
    2. **Density Correction (`use_drho=True`)**: Flags intervals where `|DRHO| > drho_threshold`
       (e.g., excessive mudcake or washout pad standoff > 0.15 g/cc).
    3. **Bulk Density / Neutron Pad Standoff Bounds (`use_rhob_nphi=True`)**: Flags extreme washout
       readings where `RHOB < rhob_min`, `RHOB > rhob_max`, or `NPHI > nphi_max` (optionally
       excluding flagged coal seams when `exclude_coal=True`).
    4. **Rolling Trend / Rugosity Spike Detection (`use_rugosity=True`)**: Flags sharp borehole
       rugosity washout spikes where `RHOB` drops below its local rolling mean by more than
       `rhob_trend_factor` or where `|CALI - rolling_median(CALI)| > cali_spike_threshold`.

    Args:
        data (pd.DataFrame): Input DataFrame containing well log curves.
        thold (float, optional): Differential caliper threshold `|CALI - BS|` (inches). Defaults to 4.0.
        use_cali (bool, optional): Flag badhole using caliper vs bit size. Defaults to True.
        auto_bitsize (bool, optional): Automatically estimate bit size per hole section from `CALI`
            when `BS` is not provided. Defaults to True.
        bitsize_default (float, optional): Nominal/fallback bit size (inches). Defaults to 8.5.
        use_drho (bool, optional): Flag badhole where `|DRHO| > drho_threshold`. Defaults to False.
        drho_threshold (float, optional): Maximum allowable `|DRHO|` (g/cc). Defaults to 0.15.
        use_rhob_nphi (bool, optional): Flag badhole using RHOB/NPHI washout bounds. Defaults to False.
        rhob_min (float, optional): Minimum valid bulk density before washout flag (g/cc). Defaults to 1.95.
        rhob_max (float, optional): Maximum valid bulk density before flag (g/cc). Defaults to 2.95.
        nphi_max (float, optional): Maximum valid neutron porosity before washout flag (v/v). Defaults to 0.45.
        exclude_coal (bool, optional): Exclude coal intervals from RHOB/NPHI bound & rugosity flags. Defaults to True.
        use_rugosity (bool, optional): Flag local rugosity spikes from rolling RHOB/CALI trends. Defaults to False.
        rugosity_window (int, optional): Rolling window size (samples) for rugosity trend. Defaults to 21.
        rhob_trend_factor (float, optional): Fractional drop below rolling RHOB mean to flag spike. Defaults to 0.05.
        cali_spike_threshold (float, optional): Local caliper deviation from rolling median (inches). Defaults to 0.5.

    Returns:
        pd.DataFrame: DataFrame with added `'BADHOLE'` (0 or 1), `'BS'`, and `'DCALI'` columns.
    """
    import scipy.stats
    from scipy.signal import find_peaks

    try:
        import ruptures as rpt
    except ImportError:
        rpt = None  # type: ignore[assignment]

    df = data.copy()
    if df.empty:
        df["BADHOLE"] = pd.Series(dtype=int)
        return df

    has_well_col = "WELL_NAME" in df.columns
    if not has_well_col:
        df["WELL_NAME"] = "WELL_1"

    has_depth_col = "DEPTH" in df.columns
    if not has_depth_col and "DEPT" in df.columns:
        df["DEPTH"] = df["DEPT"]

    def reject_outliers(series: pd.Series, m: float = 3.0) -> pd.Series:
        s_clean = series.where((series > 4.5) & (series < 24.0))
        med = s_clean.median()
        d = np.abs(s_clean - med)
        mdev = np.nanmedian(d)
        s = (
            d / mdev
            if (mdev and not np.isnan(mdev) and mdev > 1e-6)
            else pd.Series(0.0, index=s_clean.index, dtype=float)
        )
        return s_clean.where(s < m, med)

    bit_sizes = {6.0: 6.0, 8.5: 8.5, 12.25: 12.25, 17.5: 17.5}
    default_bs = float(bitsize_default) if bitsize_default and bitsize_default > 0 else 8.5

    if "BS" not in df.columns:
        df["BS"] = np.nan
    else:
        df["BS"] = pd.to_numeric(df["BS"], errors="coerce")

    processed_wells = []
    for well, well_data in df.groupby("WELL_NAME", sort=False):
        if "DEPTH" in well_data.columns:
            well_data = well_data.sort_values(by="DEPTH").copy()
        else:
            well_data = well_data.copy()

        well_data["ESTIMATED_BITSIZE"] = default_bs
        cali_s = (
            pd.to_numeric(well_data["CALI"], errors="coerce")
            if "CALI" in well_data.columns
            else pd.Series(np.nan, index=well_data.index, dtype=float)
        )

        if auto_bitsize and cali_s.notna().any():
            logger.info(f"[feature_transformer `badhole_flag`] Processing {well}.")
            signal_data = reject_outliers(cali_s)
            clean_signal = signal_data.dropna()

            estimated_ok = False
            if rpt is not None and len(clean_signal) >= 60:
                try:
                    if clean_signal.nunique() > 1:
                        kde_sample = (
                            clean_signal.iloc[:: max(1, len(clean_signal) // 2500)]
                            if len(clean_signal) > 2500
                            else clean_signal
                        )
                        kde = scipy.stats.gaussian_kde(kde_sample)
                        evaluated = kde.evaluate(
                            np.linspace(clean_signal.min(), clean_signal.max(), 100)
                        )
                        peaks, _ = find_peaks(evaluated, height=0.1)
                    else:
                        peaks = []

                    jump = min(60, max(5, len(signal_data) // 4))
                    algo = rpt.BottomUp(model="l2", jump=jump).fit(
                        signal_data.ffill().bfill().to_numpy(dtype=float)
                    )
                    n_bkps = min(len(peaks), max(0, (len(signal_data) // jump) - 1))
                    my_bkps = algo.predict(n_bkps=n_bkps) if n_bkps > 0 else [len(signal_data)]

                    indices = [0] + my_bkps
                    bins = list(zip(indices[:-1], indices[1:], strict=True))
                    cali_np = cali_s.to_numpy(dtype=float)
                    fallback_q05 = float(np.nanpercentile(clean_signal, 5))

                    estimated_bits = []
                    for start_idx, end_idx in bins:
                        seg = cali_np[start_idx:end_idx]
                        valid_seg = seg[np.isfinite(seg)]
                        val = (
                            float(np.nanpercentile(valid_seg, 5))
                            if len(valid_seg) > 0
                            else fallback_q05
                        )
                        bit_ = bit_sizes[min(bit_sizes.keys(), key=lambda k: abs(k - val))]
                        estimated_bits.append(bit_)

                    if len(estimated_bits) >= 2:
                        bits_diff_1 = np.array(estimated_bits[1:] + [estimated_bits[-1]])
                        bits_diff_neg1 = np.array([estimated_bits[0]] + estimated_bits[:-1])
                        estimated_bits_arr = np.array(estimated_bits, dtype=float)
                        estimated_bits_arr = np.where(
                            bits_diff_1 == bits_diff_neg1, bits_diff_1, estimated_bits_arr
                        )
                    else:
                        estimated_bits_arr = np.array(estimated_bits, dtype=float)

                    est_bs_arr = np.full(len(well_data), default_bs, dtype=float)
                    for (start_idx, end_idx), b_val in zip(
                        bins, estimated_bits_arr, strict=True
                    ):
                        est_bs_arr[start_idx:end_idx] = float(b_val)
                    well_data["ESTIMATED_BITSIZE"] = est_bs_arr
                    estimated_ok = True
                except Exception as e:
                    logger.debug(f"[badhole_flagging] Change-point fallback on well {well}: {e}")

            if not estimated_ok and not clean_signal.empty:
                # Fallback for short intervals or when ruptures breakpoints are not applicable:
                # snap 5th percentile of caliper to nearest standard bit size
                p05 = float(np.nanpercentile(clean_signal, 5))
                nearest_bs = bit_sizes[min(bit_sizes.keys(), key=lambda k: abs(k - p05))]
                well_data["ESTIMATED_BITSIZE"] = float(nearest_bs)

        processed_wells.append(well_data)

    return_df = pd.concat(processed_wells, ignore_index=True) if processed_wells else df.copy()

    if auto_bitsize:
        bitsize_arr = np.where(
            return_df["BS"].notna() & (return_df["BS"] > 0),
            return_df["BS"],
            return_df["ESTIMATED_BITSIZE"],
        )
    else:
        bitsize_arr = np.where(
            return_df["BS"].notna() & (return_df["BS"] > 0),
            return_df["BS"],
            default_bs,
        )
    return_df["BS"] = pd.Series(bitsize_arr, index=return_df.index, dtype=float)
    return_df.drop(columns=["ESTIMATED_BITSIZE"], errors="ignore", inplace=True)

    n_rows = len(return_df)
    cali_flag = np.zeros(n_rows, dtype=bool)
    drho_flag = np.zeros(n_rows, dtype=bool)
    bounds_flag = np.zeros(n_rows, dtype=bool)
    rugosity_flag = np.zeros(n_rows, dtype=bool)

    # 1. Differential Caliper criterion
    if "CALI" in return_df.columns:
        cali_num = pd.to_numeric(return_df["CALI"], errors="coerce")
        dcali = cali_num - return_df["BS"]
        return_df["DCALI"] = dcali
        if use_cali:
            absdiff = np.where(cali_num.notna(), np.abs(dcali), 0.0)
            cali_flag = absdiff >= float(thold)

    # Identify coal mask if exclude_coal is enabled so coal seams aren't falsely flagged by low RHOB / high NPHI
    coal_mask = np.zeros(n_rows, dtype=bool)
    if exclude_coal:
        if "COAL_FLAG" in return_df.columns:
            coal_mask |= (
                pd.to_numeric(return_df["COAL_FLAG"], errors="coerce").fillna(0.0).to_numpy()
                >= 0.5
            )
        if "VCOAL" in return_df.columns:
            coal_mask |= (
                pd.to_numeric(return_df["VCOAL"], errors="coerce").fillna(0.0).to_numpy()
                >= 0.5
            )
        if (
            not np.any(coal_mask)
            and "RHOB" in return_df.columns
            and "NPHI" in return_df.columns
        ):
            rhob_c = pd.to_numeric(return_df["RHOB"], errors="coerce").to_numpy()
            nphi_c = pd.to_numeric(return_df["NPHI"], errors="coerce").to_numpy()
            coal_mask = (
                np.isfinite(rhob_c)
                & np.isfinite(nphi_c)
                & (rhob_c < 1.90)
                & (nphi_c > 0.35)
            )

    # 2. Density Correction (DRHO) criterion
    if use_drho:
        drho_col = next(
            (c for c in ("DRHO", "HDRA", "DCOR") if c in return_df.columns), None
        )
        if drho_col is not None:
            drho_num = pd.to_numeric(return_df[drho_col], errors="coerce").to_numpy()
            drho_flag = np.isfinite(drho_num) & (np.abs(drho_num) > float(drho_threshold))

    # 3. Bulk Density / Neutron Pad Standoff Bounds criterion
    if use_rhob_nphi:
        if "RHOB" in return_df.columns:
            rhob_num = pd.to_numeric(return_df["RHOB"], errors="coerce").to_numpy()
            bounds_flag |= np.isfinite(rhob_num) & (
                (rhob_num < float(rhob_min)) | (rhob_num > float(rhob_max))
            )
        if "NPHI" in return_df.columns:
            nphi_num = pd.to_numeric(return_df["NPHI"], errors="coerce").to_numpy()
            bounds_flag |= np.isfinite(nphi_num) & (nphi_num > float(nphi_max))
        if exclude_coal:
            bounds_flag &= ~coal_mask

    # 4. Rolling Trend / Rugosity Spike criterion
    if use_rugosity:
        win = max(3, int(rugosity_window))
        if "RHOB" in return_df.columns and float(rhob_trend_factor) > 0:
            rhob_s = pd.to_numeric(return_df["RHOB"], errors="coerce")
            rhob_trend = rhob_s.rolling(window=win, center=True, min_periods=1).median()
            rhob_drop = (
                rhob_s.notna()
                & rhob_trend.notna()
                & (rhob_s < rhob_trend * (1.0 - float(rhob_trend_factor)))
            ).to_numpy()
            if exclude_coal:
                rhob_drop = rhob_drop & ~coal_mask
            rugosity_flag |= rhob_drop
        if "CALI" in return_df.columns and float(cali_spike_threshold) > 0:
            cali_s = pd.to_numeric(return_df["CALI"], errors="coerce")
            cali_trend = cali_s.rolling(window=win, center=True, min_periods=1).median()
            cali_spike = (
                cali_s.notna()
                & cali_trend.notna()
                & (np.abs(cali_s - cali_trend) > float(cali_spike_threshold))
            ).to_numpy()
            rugosity_flag |= cali_spike

    combined_flag = cali_flag | drho_flag | bounds_flag | rugosity_flag
    return_df["BADHOLE"] = np.where(combined_flag, 1, 0)

    if not has_well_col:
        return_df.drop(columns=["WELL_NAME"], errors="ignore", inplace=True)

    return_df.attrs["badhole_summary"] = {
        "total_rows": int(n_rows),
        "total_samples": int(n_rows),
        "flagged_count": int(np.sum(combined_flag)),
        "flagged_pct": round(float(np.mean(combined_flag) * 100.0), 2) if n_rows > 0 else 0.0,
        "cali_count": int(np.sum(cali_flag)),
        "cali_flagged": int(np.sum(cali_flag)),
        "drho_count": int(np.sum(drho_flag)),
        "drho_flagged": int(np.sum(drho_flag)),
        "bounds_count": int(np.sum(bounds_flag)),
        "rhob_nphi_flagged": int(np.sum(bounds_flag)),
        "rugosity_count": int(np.sum(rugosity_flag)),
        "rugosity_flagged": int(np.sum(rugosity_flag)),
    }
    return return_df


def neu_den_xplot_hc_correction(
    nphi,
    rhob,
    vsh_gr=None,
    dry_min1_point: tuple = (),
    dry_clay_point: tuple = (),
    water_point: tuple = (1.0, 1.0),
    corr_angle: float = 50,
    buffer: float = 0.0,
    target_mode: str = "bulk",
    flag_mode: str = "strict",
    vsh_tol: float = 0.05,
    nphi_max: float = 0.45,
):
    """Correct neutron porosity and bulk density for light hydrocarbon effects.

    Hydrocarbon-affected (NPHI, RHOB) points are shifted along ``corr_angle``
    until the crossplot clay volume reconciles with the gamma-ray shale volume
    ``vsh_gr``. The shift is solved analytically (exact intersection of the
    correction ray with the target iso-Vshale line), so there is no step
    quantization or iteration cap.

    Crossplot coordinates are expressed in the barycentric (porosity, clay)
    frame of the matrix (A), dry clay (C) and water (D) triangle, see
    :func:`quick_pp.utils.nd_triangle_coords`. This makes the result independent
    of whether the dry clay is heavier or lighter than the matrix.

    Flagging:
        * ``flag_mode="strict"`` (default): only points plotting beyond the
          clean-sand (D-A) line are flagged, i.e. where the clay fraction of
          solids projected from the water point is below
          ``buffer / |A - C|``. ``buffer`` (crossplot distance along A-C) only
          widens this flagging region; it does not move the correction target.
        * ``flag_mode="vsh_gr"``: additionally flags points inside the triangle
          whose crossplot clay volume is below ``vsh_gr - vsh_tol``. This
          captures shaly gas sands, but may produce false positives where the
          gamma ray overestimates shale (e.g. radioactive/feldspathic sands).

    Target:
        * ``target_mode="bulk"`` (default): ``vsh_gr`` is treated as bulk clay
          volume; the target is the iso-Vcl line through ``A + vsh_gr (C - A)``
          parallel to D-A. Consistent with :func:`den_correction`.
        * ``target_mode="solid"``: ``vsh_gr`` is treated as clay fraction of the
          solids, i.e. the target is the line from D through
          ``C + (1 - vsh_gr)(A - C)`` (legacy convention).

    Corrected points never move backwards along the ray (flagged points already
    on the shaly side of the target are left unchanged) and never cross the
    zero-porosity (A-C) line.

    Args:
        nphi (np.ndarray or float): Neutron porosity log in fraction (v/v).
        rhob (np.ndarray or float): Bulk density log in g/cc.
        vsh_gr (np.ndarray or float, optional): Vshale from gamma ray (v/v). NaN values
            are skipped (not corrected, not flagged). If None, a clean-sand target
            (0) is used everywhere. Defaults to None.
        dry_min1_point (tuple): (NPHI, RHOB) of the primary matrix mineral (A).
        dry_clay_point (tuple): (NPHI, RHOB) of dry clay (C).
        water_point (tuple, optional): (NPHI, RHOB) of the formation water (D). Defaults to (1.0, 1.0).
        corr_angle (float, optional): Correction angle in degrees counter-clockwise from the
            positive NPHI axis (tan = dRHOB / dNPHI). See
            :func:`neu_den_xplot_hc_correction_angle`. Defaults to 50.
        buffer (float, optional): Flagging tolerance as crossplot distance along A-C. Only widens
            the strict flagging region. Defaults to 0.0.
        target_mode (str, optional): 'bulk' or 'solid'. Defaults to 'bulk'.
        flag_mode (str, optional): 'strict' or 'vsh_gr'. Defaults to 'strict'.
        vsh_tol (float, optional): Tolerance for ``flag_mode='vsh_gr'``. Defaults to 0.05.
        nphi_max (float, optional): Points with NPHI >= nphi_max are never corrected. Defaults to 0.45.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing:
            - Corrected neutron porosity (NPHI).
            - Corrected bulk density (RHOB).
            - Hydrocarbon flag (1.0 where flagged, 0.0 otherwise).
    """
    target_mode = str(target_mode).strip().lower()
    flag_mode = str(flag_mode).strip().lower()
    if target_mode not in ("bulk", "solid"):
        raise ValueError("target_mode must be 'bulk' or 'solid'")
    if flag_mode not in ("strict", "vsh_gr"):
        raise ValueError("flag_mode must be 'strict' or 'vsh_gr'")

    A = (float(dry_min1_point[0]), float(dry_min1_point[1]))
    C = (float(dry_clay_point[0]), float(dry_clay_point[1]))
    D = (float(water_point[0]), float(water_point[1]))
    rock_len = float(length_a_b(A, C))
    if rock_len <= 0:
        raise ValueError("dry_min1_point and dry_clay_point must be distinct")

    nphi = np.asarray(nphi, dtype=float)
    rhob = np.asarray(rhob, dtype=float)
    if vsh_gr is None:
        target_vsh = np.zeros_like(nphi)
    else:
        target_vsh = np.broadcast_to(
            np.clip(np.asarray(vsh_gr, dtype=float), 0.0, 1.0), nphi.shape
        )

    with np.errstate(divide="ignore", invalid="ignore"):
        # Barycentric porosity and bulk clay volume of the raw points
        phi0, vcl0 = nd_triangle_coords(nphi, rhob, A, C, D)
        # Clay fraction of solids == projection from D onto A-C (legacy vsh_dn)
        vsh_dn_solid = vcl0 / (1.0 - phi0)

        valid = (
            np.isfinite(nphi)
            & np.isfinite(rhob)
            & np.isfinite(target_vsh)
            & (nphi < nphi_max)
            & (phi0 < 1.0)
        )

        # 1. Flag hydrocarbon-affected points
        flag = vsh_dn_solid < (float(buffer) / rock_len)
        if flag_mode == "vsh_gr":
            v_mode = vcl0 if target_mode == "bulk" else vsh_dn_solid
            flag = flag | (v_mode < (target_vsh - float(vsh_tol)))
        flag = flag & valid

        # 2. Correction ray direction, expressed in the (phi, vcl) frame
        rad = np.radians(float(corr_angle))
        ux, uy = float(np.cos(rad)), float(np.sin(rad))
        dphi, dvcl = nd_triangle_coords(A[0] + ux, A[1] + uy, A, C, D)
        dphi, dvcl = float(dphi), float(dvcl)

        # 3. Exact ray length to reach the target iso-Vshale line
        if target_mode == "bulk":
            denom = np.full_like(phi0, dvcl)
            numer = target_vsh - vcl0
        else:
            denom = dvcl + target_vsh * dphi
            numer = target_vsh * (1.0 - phi0) - vcl0
        s = np.where(np.abs(denom) > 1e-12, numer / denom, np.nan)

        # Do not cross the zero-porosity (A-C) line
        if dphi < 0:
            s = np.minimum(s, -phi0 / dphi)

        move = flag & np.isfinite(s) & (s > 0)
        nphi_corrected = np.where(move, nphi + s * ux, nphi)
        rhob_corrected = np.where(move, rhob + s * uy, rhob)

    return nphi_corrected, rhob_corrected, flag.astype(float)


def neu_den_xplot_hc_correction_angle(rho_water=1.0, rho_hc=0.3, HI_hc=None):
    """Estimate the hydrocarbon correction angle on the NPHI-RHOB crossplot.

    Replacing water with hydrocarbon lowers NPHI by ``phi * Shc * (1 - HI_hc)`` and
    RHOB by ``phi * Shc * (rho_water - rho_hc)``. The correction vector therefore
    satisfies ``tan(angle) = dRHOB / dNPHI = (rho_water - rho_hc) / (1 - HI_hc)``,
    with the angle measured counter-clockwise from the positive NPHI (horizontal)
    axis, the same convention used by :func:`neu_den_xplot_hc_correction`.

    The neutron excavation effect is not modelled; in gas zones the real NPHI
    depression is larger, so the effective angle may be somewhat shallower.

    Args:
        rho_water (float, optional): Fluid density, typically 0.8-1.2 g/cc. Defaults to 1.0.
        rho_hc (float, optional): Hydrocarbon density (gas: 0.1-0.3, oil: 0.4-0.8 g/cc). Defaults to 0.3.
        HI_hc (float, optional): Hydrogen Index of the hydrocarbon (gas: 0.2-0.7, oil: 0.8-1.0).
            If None, it is estimated with the gas relation ``2.2 * rho_hc`` capped at 1.0
            (liquids). Defaults to None.

    Returns:
        float: The hydrocarbon correction angle in degrees (90 for HI_hc = 1, > 90 for HI_hc > 1).
    """
    if HI_hc is None:
        HI_hc = min(2.2 * float(rho_hc), 1.0)
    return float(
        np.degrees(np.arctan2(float(rho_water) - float(rho_hc), 1.0 - float(HI_hc)))
    )


def den_correction_params_from_endpoints(
    dry_sand_point: tuple, dry_clay_point: tuple, rho_water: float = 1.0
):
    """Derive :func:`den_correction` parameters consistent with crossplot endpoints.

    Maps the dry sand (A), dry clay (C) and water (rho_water) crossplot points to
    ``(rho_ma, phin_sh, phid_sh)`` such that :func:`den_correction` reproduces the
    linear three-component (matrix-clay-water) mixing model exactly, with
    ``vsh_gr`` interpreted as bulk clay volume.

    Args:
        dry_sand_point (tuple): (NPHI, RHOB) of the dry matrix point.
        dry_clay_point (tuple): (NPHI, RHOB) of the dry clay point.
        rho_water (float, optional): Water density in g/cc. Defaults to 1.0.

    Returns:
        tuple[float, float, float]: ``(rho_ma, phin_sh, phid_sh)``.
    """
    nphi_sand, rho_sand = float(dry_sand_point[0]), float(dry_sand_point[1])
    nphi_clay, rho_clay = float(dry_clay_point[0]), float(dry_clay_point[1])
    rho_w = float(rho_water)
    if abs(1.0 - nphi_sand) < 1e-6:
        raise ValueError("Dry sand NPHI must differ from 1.0")
    rho_ma = (rho_sand - nphi_sand * rho_w) / (1.0 - nphi_sand)
    if abs(rho_ma - rho_w) < 1e-6:
        raise ValueError("Matrix density must differ from water density")
    phid_sh = (rho_ma - rho_clay) / (rho_ma - rho_w)
    return rho_ma, nphi_clay, phid_sh


def den_correction(
    nphi,
    gr=None,
    vsh_gr=None,
    phin_sh=0.35,
    phid_sh=0.05,
    rho_ma=2.68,
    rho_water=1.0,
    alpha=0.05,
):
    """Reconstruct bulk density from neutron porosity and gamma-ray shale volume.

    Assumes ``vsh_gr = vsh_dn = (phin - phid) / (phin_sh - phid_sh)`` and solves for
    density porosity, then converts it to bulk density. Intended for bad-hole
    (washout) RHOB reconstruction in **water-bearing** intervals.

    Important:
        * ``vsh_gr`` is interpreted as **bulk** clay volume.
        * ``nphi`` must be water-filled or already **hydrocarbon-corrected**
          (e.g. NPHI_HC). Gas-depressed NPHI yields an artificially high RHOB
          (in clean sand ``phid = nphi``).
        * Use :func:`den_correction_params_from_endpoints` to derive
          ``rho_ma``, ``phin_sh`` and ``phid_sh`` consistent with the
          neutron-density crossplot endpoints.

    Args:
        nphi (np.ndarray or float): Neutron porosity log in fraction (water-filled or HC-corrected).
        gr (np.ndarray or float, optional): Gamma ray log in GAPI. Only used when vsh_gr is None.
        vsh_gr (np.ndarray or float, optional): Pre-calculated bulk Vshale from gamma ray. If None, it is
            estimated from ``gr``. Defaults to None.
        phin_sh (float, optional): Neutron porosity of shale. Defaults to .35.
        phid_sh (float, optional): Density porosity of shale. Defaults to .05.
        rho_ma (float, optional): Matrix density. Defaults to 2.68.
        rho_water (float, optional): Fluid density. Defaults to 1.0.
        alpha (float, optional): Alpha value for gamma ray normalization if vsh_gr is not provided. Defaults to 0.05.

    Returns:
        np.ndarray or float: Reconstructed bulk density.
    """
    # Estimate vsh_gr
    if vsh_gr is None:
        if gr is None:
            raise ValueError("Either gr or vsh_gr must be provided")
        vsh_gr = estimate_vsh_gr(gr, alpha=alpha)
    if not np.isscalar(nphi) or not np.isscalar(vsh_gr):
        nphi = np.asarray(nphi, dtype=float)
        vsh_gr = np.asarray(vsh_gr, dtype=float)
    # Estimate phid assuming vsh_gr = vsh_dn = (phin - phid) / (phin_sh - phid_sh)
    phid = nphi - (phin_sh - phid_sh) * vsh_gr
    return rho_ma - (rho_ma - rho_water) * phid


def quick_qc(well_data, return_fig=False):
    """Perform a quick quality control analysis on processed well data.

    This function generates QC flags, calculates a reservoir summary, and optionally
    creates distribution and depth plots for key petrophysical properties. It is
    designed to process data for a single well.

    Args:
        well_data (pd.DataFrame): The input DataFrame containing processed well log data.
        return_fig (bool, optional): If True, generates and returns matplotlib figures. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The input DataFrame with added 'QC_FLAG' and other flags.
            - A DataFrame summarizing key petrophysical properties.
            - A distribution plot figure (if `return_fig` is True, else None).
            - A depth plot figure (if `return_fig` is True, else None).
    """
    return_df = well_data.copy()
    return_df["ZONES"] = "ALL"
    return_df["QC_FLAG"] = 0
    return_df["PERM"] = return_df["PERM"].clip(lower=1e-3, upper=1e5)

    cutoffs = {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9}
    _, return_df["RES_FLAG"], _ = flag_interval(
        return_df["VCLAY"], return_df["PHIT"], return_df["SWT"], cutoffs
    )
    return_df["ALL_FLAG"] = 1

    # Check average swt in non-reservoir zone
    return_df["QC_FLAG"] = np.where(
        (return_df["RES_FLAG"] == 0) & (return_df["SWT"] < 0.8), 3, return_df["QC_FLAG"]
    )

    # Summarize
    summary_df = calc_reservoir_summary(
        return_df.DEPTH,
        return_df.VCLAY,
        return_df.PHIT,
        return_df.SWT,
        return_df.PERM,
        return_df.ZONES,
    )
    summary_df.columns = [col.upper() for col in summary_df.columns]
    for group in ["all", "reservoir"]:
        index = (
            return_df["ALL_FLAG"] == 1 if group == "all" else return_df["RES_FLAG"] == 1
        )
        data = return_df[index]

        # Calculate average and standard deviation
        temp_df = data.groupby("ZONES").agg(
            {
                "GR": ["mean", "std"],
                "RT": ["mean", "std"],
                "NPHI": ["mean", "std"],
                "RHOB": ["mean", "std"],
            }
        )
        # Rename columns
        cols_rename = [f"{stat}_{col}" for col, stat in temp_df.columns]
        cols_rename = [
            col.replace("mean", "AV").replace("std", "STD") for col in cols_rename
        ]
        temp_df.columns = cols_rename

        summary_df.loc[
            summary_df.FLAG == group, ["AV_GR", "AV_RT", "AV_NPHI", "AV_RHOB"]
        ] = temp_df[["AV_GR", "AV_RT", "AV_NPHI", "AV_RHOB"]].values

        summary_df.loc[
            summary_df.FLAG == group, ["STD_GR", "STD_RT", "STD_NPHI", "STD_RHOB"]
        ] = temp_df[["STD_GR", "STD_RT", "STD_NPHI", "STD_RHOB"]].values

        summary_df.loc[summary_df.FLAG == group, "QC_FLAG_COUNT"] = (
            data.where(data["QC_FLAG"] == 1, 0).groupby("ZONES")["QC_FLAG"].sum().values
        )

        summary_df.loc[summary_df.FLAG == group, "QC_FLAG_mode"] = (
            data.groupby("ZONES")["QC_FLAG"].agg(lambda x: x.mode()).values
        )

    if return_fig:
        # Distribution plot
        dist_fig, axs = plt.subplots(4, 1, figsize=(5, 10))
        for group, data in return_df.groupby("RES_FLAG"):
            label = "RES" if group == 1 else "NON-RES"
            axs[0].hist(data["VCLAY"], bins=100, alpha=0.7, label=label, range=[0, 1])
            axs[1].hist(data["PHIT"], bins=100, alpha=0.7, label=label, range=[0, 0.5])
            axs[2].hist(data["SWT"], bins=100, alpha=0.7, label=label, range=[0, 1])
            axs[3].hist(
                np.log10(data["PERM"]), bins=100, alpha=0.7, label=f"Log PERM_{label}"
            )
        axs[0].set_title("VSHALE Distribution")
        axs[0].legend()

        axs[1].set_title("PHIT Distribution")
        axs[1].legend()

        axs[2].set_title("SWT Distribution")
        axs[2].set_yscale("log")
        axs[2].legend()

        axs[3].set_title("Log PERM Distribution")
        axs[3].set_yscale("log")
        axs[3].legend()

        dist_fig.tight_layout()

        # Depth plot
        depth_fig, axs = plt.subplots(4, 1, figsize=(20, 5), sharex=True)
        axs[0].plot(return_df["DEPTH"], return_df["VCLAY"], label="VSHALE")
        # Compare vsh_gr and vsh_dn
        if "VSH_GR" in return_df.columns:
            return_df["QC_FLAG"] = np.where(
                abs(return_df["VSH_GR"] - return_df["VCLAY"]) > 0.1,
                1,
                return_df["QC_FLAG"],
            )
            axs[0].plot(return_df["DEPTH"], return_df["VSH_GR"], label="VSH_GR")
        axs[0].set_frame_on(False)
        axs[0].set_title("VSHALE")
        axs[0].legend()

        axs[1].plot(return_df["DEPTH"], return_df["PHIT"], label="PHIT")
        # Compare phit_dn and phit_d
        if "PHID" in return_df.columns:
            return_df["QC_FLAG"] = np.where(
                abs(return_df["PHIT"] - return_df["PHID"]) > 0.1,
                2,
                return_df["QC_FLAG"],
            )
            axs[1].plot(return_df["DEPTH"], return_df["PHID"], label="PHID")
        axs[1].set_frame_on(False)
        axs[1].set_title("PHIT")
        axs[1].legend()

        axs[2].plot(return_df["DEPTH"], return_df["SWT"], label="SWT")
        axs[2].plot(return_df["DEPTH"], return_df["RES_FLAG"], label="RES_FLAG")
        axs[2].set_frame_on(False)
        axs[2].set_title("SWT")
        axs[2].legend()

        axs[3].plot(return_df["DEPTH"], return_df["QC_FLAG"], label="QC_FLAG")
        axs[3].set_frame_on(False)
        axs[3].set_title("QC_FLAG")
        axs[3].legend()

        depth_fig.tight_layout()

        return return_df, summary_df, dist_fig, depth_fig

    else:
        return return_df, summary_df, None, None


def quick_compare(field_data, level="WELL", return_fig=False):
    """Perform a quick comparison of petrophysical properties across multiple wells or zones.

    This function iterates through the field data, performs a quick QC on each well,
    and aggregates the summary statistics. It can optionally generate a figure
    with box plots, histograms, and summary tables for key properties.

    Args:
        field_data (pd.DataFrame): A DataFrame containing data for multiple wells.
        level (str, optional): The level of comparison, either 'WELL' or 'ZONE'. Defaults to 'WELL'.
        return_fig (bool, optional): Whether to return figure. Defaults to False.

    Returns:
        tuple[pd.DataFrame, plt.Figure or None]: A tuple containing the comparison DataFrame and the generated figure (or None).
    """
    assert level in ["WELL", "ZONE"], "Level must be either 'WELL' or 'ZONE'"
    field_data["ZONES"] = "ALL" if level == "WELL" else field_data["ZONES"]
    compare_df = pd.DataFrame()
    for well, well_data in field_data.groupby("WELL_NAME"):
        logger.info(f"Processing {well}")
        _, summary_df, _, _ = quick_qc(well_data, return_fig=False)
        summary_df.insert(0, "WELL_NAME", well)
        compare_df = pd.concat([compare_df, summary_df])
    logger.info("Finished processing all wells.")

    if return_fig:
        # Box plot
        curves = [
            "AV_GR",
            "AV_RT",
            "AV_NPHI",
            "AV_RHOB",
            "AV_VSHALE",
            "AV_PHIT",
            "AV_SWT",
            "AV_PERM_GM",
        ]
        no_curves = len(curves)
        plot_df = compare_df[compare_df.FLAG == "all"]
        idx_percentiles = [
            int(x * (len(plot_df) - 1)) for x in [0, 0.1, 0.25, 0.5, 0.75, 0.9, 1]
        ]
        fig, axes = plt.subplots(
            no_curves,
            3,
            figsize=(7, 2.5 * no_curves),
            gridspec_kw={"width_ratios": [1, 3, 5]},
        )
        for i, curve in enumerate(curves):
            # Box plot
            axes[i, 0].set_title(curve)
            axes[i, 0].boxplot(plot_df[curve], labels=[""])

            # Distribution plot
            axes[i, 1].hist(
                plot_df[curve], bins=25, alpha=0.7, orientation="horizontal"
            )

            # Summary table
            df = (
                plot_df[["WELL_NAME", curve]]
                .sort_values(by=curve, ascending=True)
                .reset_index(drop=True)
            )
            df = df.loc[idx_percentiles, :].round(2)
            df["PERCENTILES"] = ["0%", "10%", "25%", "50%", "75%", "90%", "100%"]
            df = df[["PERCENTILES", curve, "WELL_NAME"]]
            axes[i, 2].table(
                cellText=df.values,
                colLabels=df.columns,
                loc="center",
                cellLoc="center",
                colColours=["#f2f2f2"] * 3,
            )
            axes[i, 2].axis("off")

        plt.tight_layout()
        return compare_df.reset_index(drop=True), fig

    else:
        return compare_df.reset_index(drop=True), None


def extract_quick_stats(compare_df, flag="all"):
    """Extract descriptive statistics from a comparison DataFrame.

    This function filters a comparison DataFrame (typically generated by `quick_compare`)
    by a specified flag and calculates summary statistics for key petrophysical properties.

    Args:
        compare_df (pd.DataFrame): The input comparison DataFrame.
        flag (str, optional): The flag to filter the DataFrame by (e.g., 'all', 'reservoir', 'pay'). Defaults to 'all'.

    Returns:
        pd.DataFrame: A DataFrame containing the descriptive statistics.
    """
    compare_df = compare_df[compare_df.FLAG == flag].copy()
    # Extract quick stats
    reqs = ["PHIT", "SWT"]
    stats_df = pd.DataFrame()
    for col in compare_df.columns:
        if any(req in col for req in reqs) or col in ["NET", "NTG"]:
            stats = compare_df[col].describe(percentiles=[0.1, 0.5, 0.9])
            stats_df[col] = stats

    return round(stats_df, 3)


def log_normalization(
    df: pd.DataFrame,
    logs: list[str] | tuple[str, ...] = ("GR", "NPHI"),
    method: str = "minmax",
    quantiles: tuple[float, float] = (0.05, 0.95),
    inplace: bool = False,
    suffix: str = "_NORM",
) -> pd.DataFrame:
    """Perform global log normalization (min-max, quantile, or z-score) across a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing well log curves.
        logs (list[str] | tuple[str, ...], optional): Log columns to normalize. Defaults to ("GR", "NPHI").
        method (str, optional): Normalization method: 'minmax', 'quantile', or 'zscore'. Defaults to 'minmax'.
        quantiles (tuple[float, float], optional): Low and high quantiles when `method='quantile'`. Defaults to (0.05, 0.95).
        inplace (bool, optional): If True, overwrites the original log columns. If False, writes to `<log><suffix>`. Defaults to False.
        suffix (str, optional): Suffix for normalized columns when `inplace=False`. Defaults to '_NORM'.

    Returns:
        pd.DataFrame: DataFrame with normalized log curves.
    """
    method_lower = method.lower()
    if method_lower not in ("minmax", "quantile", "zscore", "z-score"):
        raise ValueError(
            f"Unsupported normalization method '{method}'. Choose from 'minmax', 'quantile', or 'zscore'."
        )

    out_df = df.copy()
    if isinstance(logs, str):
        logs = [logs]

    q_low, q_high = quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            f"quantiles must satisfy 0.0 <= q_low < q_high <= 1.0, got {quantiles}."
        )

    for col in logs:
        if col not in out_df.columns:
            continue
        series = pd.to_numeric(out_df[col], errors="coerce")
        target_col = col if inplace else f"{col}{suffix}"

        if method_lower == "minmax":
            vmin = series.min()
            vmax = series.max()
            denom = vmax - vmin
            if pd.isna(denom) or abs(denom) < 1e-12:
                out_df[target_col] = np.where(series.notna(), 0.0, np.nan)
            else:
                out_df[target_col] = (series - vmin) / denom
        elif method_lower == "quantile":
            vmin = series.quantile(q_low)
            vmax = series.quantile(q_high)
            denom = vmax - vmin
            if pd.isna(denom) or abs(denom) < 1e-12:
                out_df[target_col] = np.where(series.notna(), 0.0, np.nan)
            else:
                out_df[target_col] = (series - vmin) / denom
        else:  # zscore / z-score
            mean_val = series.mean()
            std_val = series.std()
            if pd.isna(std_val) or abs(std_val) < 1e-12:
                out_df[target_col] = np.where(series.notna(), 0.0, np.nan)
            else:
                out_df[target_col] = (series - mean_val) / std_val

    return out_df


def normalize_by_zone(
    df: pd.DataFrame,
    key_well: str | None = None,
    zone_col: str = "ZONES",
    quantiles: tuple[float, float] = (0.05, 0.95),
    logs: list[str] | tuple[str, ...] = ("GR", "NPHI"),
    well_col: str = "WELL_NAME",
    method: str = "quantile",
    n_quantiles: int = 101,
    inplace: bool = False,
    suffix: str = "_NORM",
    return_report: bool = False,
    exclude_badhole: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """Perform stratigraphic (zone-by-zone) multi-well quantile or histogram log normalization.

    Normalizes log curves (e.g., GR, NPHI, RHOB, DT) across 50+ multi-vintage wells
    within each stratigraphic formation/zone independently, calibrating each target
    well's distribution to match a reference `key_well` (or the multi-well field
    median per zone if `key_well` is None or absent in a zone). This removes vintage
    tool calibration drift and borehole environmental shifts between wells without
    smearing true geological/lithological differences across zones.

    Supports two stratigraphic normalization modes via `method`:
    1. `'quantile'` (default): Linear shift-and-stretch matching low and high quantile
       anchors `(P_low, P_high)` (e.g., clean sand line and shale baseline):
       $$X_{\\text{norm}} = P_{\\text{low, ref}}^{(z)} + \\frac{X - P_{\\text{low, well}}^{(z)}}{P_{\\text{high, well}}^{(z)} - P_{\\text{low, well}}^{(z)}} \\left(P_{\\text{high, ref}}^{(z)} - P_{\\text{low, ref}}^{(z)}\\right)$$
    2. `'histogram'` (or `'cdf'`): Non-linear empirical CDF / multi-quantile histogram
       matching across `n_quantiles` evaluation points within each zone, aligning the
       full distribution shape of each well to the reference key well or field median CDF.

    Args:
        df (pd.DataFrame): Input multi-well DataFrame.
        key_well (str | None, optional): Name of the reference key well in `well_col`.
            If None or if `key_well` has no valid data in a given zone, the zone-level
            field median of well quantiles is used as the reference anchor. Defaults to None.
        zone_col (str, optional): Column name identifying stratigraphic zones/formations.
            Defaults to 'ZONES'.
        quantiles (tuple[float, float], optional): Tuple of `(q_low, q_high)` quantiles
            in `(0, 1)` used as low and high anchors. Defaults to (0.05, 0.95).
        logs (list[str] | tuple[str, ...], optional): Log curves to normalize.
            Defaults to ('GR', 'NPHI').
        well_col (str, optional): Column name identifying wells. Defaults to 'WELL_NAME'.
        method (str, optional): Normalization method: `'quantile'` (linear two-point
            percentile scaling) or `'histogram'` / `'cdf'` (piecewise empirical CDF
            histogram matching). Defaults to `'quantile'`.
        n_quantiles (int, optional): Number of quantile grid points for `'histogram'`
            matching. Defaults to 101.
        inplace (bool, optional): If True, overwrites the input log columns in the returned
            DataFrame in addition to writing `<log><suffix>`. Defaults to False.
        suffix (str, optional): Suffix appended to normalized log column names.
            Defaults to '_NORM'.
        return_report (bool, optional): If True, returns `(normalized_df, report_df)`
            containing per-well, per-zone, per-log shift, scale, and quantile diagnostics.
            Defaults to False.

    Returns:
        pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]: A copy of `df` with normalized
        curves (and `report_df` if `return_report=True`).
    """
    q_low, q_high = quantiles
    if not (0.0 <= q_low < q_high <= 1.0):
        raise ValueError(
            f"quantiles must satisfy 0.0 <= q_low < q_high <= 1.0, got {quantiles}."
        )

    method_lower = method.lower()
    if method_lower not in ("quantile", "histogram", "cdf"):
        raise ValueError(
            f"Unsupported method '{method}'. Choose from 'quantile' or 'histogram'."
        )

    if zone_col not in df.columns:
        raise ValueError(f"Zone column '{zone_col}' not found in DataFrame.")
    if well_col not in df.columns:
        raise ValueError(f"Well column '{well_col}' not found in DataFrame.")

    if isinstance(logs, str):
        logs = [logs]

    present_logs = [col for col in logs if col in df.columns]
    if not present_logs:
        raise ValueError(f"None of the specified logs {logs} were found in DataFrame.")

    out_df = df.copy()
    report_rows: list[dict] = []

    # Initialize output columns as float copies of original curves
    for col in present_logs:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce").astype(float)
        if not inplace and suffix:
            out_df[f"{col}{suffix}"] = out_df[col].copy()

    q_grid = np.linspace(0.0, 1.0, max(int(n_quantiles), 11))

    # Group by stratigraphic zone
    for zone_name, zone_df in out_df.groupby(zone_col, dropna=False):
        if pd.isna(zone_name):
            continue

        for col in present_logs:
            target_col = col if (inplace or not suffix) else f"{col}{suffix}"

            # Compute (p_low, p_high) and full quantile grid for every well in this zone
            well_stats: dict[str, tuple[float, float]] = {}
            well_cdfs: dict[str, np.ndarray] = {}
            for well_name, well_zone_df in zone_df.groupby(well_col, dropna=False):
                val_series = well_zone_df[col]
                if exclude_badhole:
                    for bh_col in ("BADHOLE", "BAD_HOLE", "BAD_HOLE_FLAG"):
                        if bh_col in well_zone_df.columns:
                            bh_mask = (
                                pd.to_numeric(well_zone_df[bh_col], errors="coerce")
                                .fillna(0.0)
                                >= 0.5
                            )
                            if (~bh_mask & val_series.notna()).sum() >= 5:
                                val_series = val_series.where(~bh_mask)
                            break
                valid_vals = val_series.dropna().to_numpy(dtype=float)
                if len(valid_vals) < 2:
                    continue
                p_low = float(np.nanquantile(valid_vals, q_low))
                p_high = float(np.nanquantile(valid_vals, q_high))
                well_stats[well_name] = (p_low, p_high)
                if method_lower in ("histogram", "cdf"):
                    well_cdfs[well_name] = np.nanquantile(valid_vals, q_grid)

            if not well_stats:
                continue

            # Determine reference (p_low_ref, p_high_ref) and reference CDF from key_well or field median
            if (
                key_well is not None
                and key_well in well_stats
                and (well_stats[key_well][1] - well_stats[key_well][0]) > 1e-12
            ):
                p_low_ref, p_high_ref = well_stats[key_well]
                ref_cdf = well_cdfs.get(key_well)
                ref_source = str(key_well)
            else:
                if key_well is not None and key_well not in well_stats:
                    logger.warning(
                        f"[normalize_by_zone] key_well '{key_well}' missing or insufficient data "
                        f"in zone '{zone_name}' for '{col}'. Falling back to field median quantiles."
                    )
                p_low_ref = float(np.median([v[0] for v in well_stats.values()]))
                p_high_ref = float(np.median([v[1] for v in well_stats.values()]))
                ref_cdf = (
                    np.median(np.vstack(list(well_cdfs.values())), axis=0)
                    if well_cdfs
                    else None
                )
                ref_source = "FIELD_MEDIAN"

            ref_span = p_high_ref - p_low_ref
            if abs(ref_span) < 1e-12:
                continue

            # Normalize each well in this zone toward the reference
            for well_name, well_zone_df in zone_df.groupby(well_col, dropna=False):
                if well_name not in well_stats:
                    continue
                p_low_w, p_high_w = well_stats[well_name]
                well_span = p_high_w - p_low_w
                idx = well_zone_df.index
                raw_series = out_df.loc[idx, col]

                if abs(well_span) < 1e-12:
                    scale = 1.0
                    shift = p_low_ref - p_low_w
                    norm_vals = raw_series + shift
                elif method_lower in ("histogram", "cdf") and ref_cdf is not None:
                    scale = ref_span / well_span
                    shift = p_low_ref - p_low_w * scale
                    w_cdf = well_cdfs[well_name]
                    # Deduplicate knot points for monotonic piecewise linear interpolation
                    unique_w_cdf, unique_idx = np.unique(w_cdf, return_index=True)
                    unique_ref_cdf = ref_cdf[unique_idx]
                    raw_np = raw_series.to_numpy(dtype=float)
                    valid_mask = ~np.isnan(raw_np)
                    norm_np = np.full_like(raw_np, np.nan, dtype=float)
                    if len(unique_w_cdf) >= 2:
                        # Interpolate within empirical range and linearly extrapolate tails using (p_low, p_high) slope
                        interp_vals = np.interp(
                            raw_np[valid_mask], unique_w_cdf, unique_ref_cdf
                        )
                        low_tail = raw_np[valid_mask] < unique_w_cdf[0]
                        high_tail = raw_np[valid_mask] > unique_w_cdf[-1]
                        if np.any(low_tail):
                            interp_vals[low_tail] = (
                                unique_ref_cdf[0]
                                + (raw_np[valid_mask][low_tail] - unique_w_cdf[0])
                                * scale
                            )
                        if np.any(high_tail):
                            interp_vals[high_tail] = (
                                unique_ref_cdf[-1]
                                + (raw_np[valid_mask][high_tail] - unique_w_cdf[-1])
                                * scale
                            )
                        norm_np[valid_mask] = interp_vals
                    else:
                        norm_np[valid_mask] = (
                            p_low_ref + (raw_np[valid_mask] - p_low_w) * scale
                        )
                    norm_vals = pd.Series(norm_np, index=idx)
                else:
                    scale = ref_span / well_span
                    shift = p_low_ref - p_low_w * scale
                    norm_vals = p_low_ref + (raw_series - p_low_w) * scale

                out_df.loc[idx, target_col] = norm_vals
                if inplace and suffix:
                    out_df.loc[idx, f"{col}{suffix}"] = norm_vals

                report_rows.append(
                    {
                        "WELL_NAME": well_name,
                        "ZONE": zone_name,
                        "LOG": col,
                        "METHOD": method_lower,
                        "REFERENCE": ref_source,
                        "P_LOW_ORIG": p_low_w,
                        "P_HIGH_ORIG": p_high_w,
                        "P_LOW_REF": p_low_ref,
                        "P_HIGH_REF": p_high_ref,
                        "SCALE_FACTOR": float(scale),
                        "DRIFT_SHIFT": float(p_low_ref - p_low_w),
                        "SCALE": float(scale),
                        "SHIFT": float(p_low_ref - p_low_w),
                    }
                )

    if return_report:
        return out_df, pd.DataFrame(report_rows)
    return out_df


stratigraphic_normalization = normalize_by_zone



def gr_borehole_correction(
    gr: np.ndarray | pd.Series | float,
    cali: np.ndarray | pd.Series | float,
    bitsize: np.ndarray | pd.Series | float = 8.5,
    mud_weight: np.ndarray | pd.Series | float = 9.0,
    tool_diameter: float = 3.625,
    tool_position: str = "eccentered",
) -> np.ndarray | pd.Series | float:
    """Correct Gamma Ray (GR) log for borehole size and mud weight attenuation.

    Applies the standard Schlumberger/API exponential borehole absorption model
    where drilling fluid between the gamma-ray detector and the borehole wall
    attenuates natural formation gamma radiation:
    $$\\text{CF}_{\\text{GR}} = \\exp\\left(\\mu \\cdot \\rho_{\\text{mud}} \\cdot \\frac{2.54}{2} \\cdot (d_{\\text{hole}} - d_{\\text{ref}}) \\cdot f_{\\text{geom}}\\right)$$

    Args:
        gr (np.ndarray | pd.Series | float): Measured Gamma Ray log (GAPI).
        cali (np.ndarray | pd.Series | float): Caliper / borehole diameter (inches).
        bitsize (np.ndarray | pd.Series | float, optional): Reference nominal bit size
            (inches) at which the tool is calibrated (typically 8.5 in). Defaults to 8.5.
        mud_weight (np.ndarray | pd.Series | float, optional): Drilling fluid density
            in ppg (pounds per gallon, where 8.345 ppg = 1.0 g/cc) or g/cc (auto-detected
            if values <= 3.0). Defaults to 9.0 ppg.
        tool_diameter (float, optional): Logging tool outer diameter (inches). Defaults to 3.625.
        tool_position (str, optional): Tool geometry in borehole: `'eccentered'`
            (sidewall-pressed, lower mud attenuation) or `'centered'` (stronger mud
            attenuation). Defaults to `'eccentered'`.

    Returns:
        np.ndarray | pd.Series | float: Borehole-corrected Gamma Ray log (GAPI).
    """
    is_series = isinstance(gr, pd.Series)
    gr_idx = gr.index if is_series else None
    gr_arr = np.asarray(gr, dtype=float)
    cali_arr = np.asarray(cali, dtype=float)
    bs_arr = np.asarray(bitsize, dtype=float)
    mw_arr = np.asarray(mud_weight, dtype=float)

    # Convert mud weight to g/cc if provided in ppg (> 3.0)
    rho_mud = np.where(mw_arr > 3.0, mw_arr / 8.3454, mw_arr)
    rho_mud_ref = 9.0 / 8.3454  # 9.0 ppg reference calibration mud weight (~1.08 g/cc)

    # Effective fluid annulus thickness change relative to reference 8.5" borehole & 9 ppg mud
    d_hole = np.clip(np.where(np.isnan(cali_arr), bs_arr, cali_arr), tool_diameter, 36.0)
    d_ref = np.clip(bs_arr, tool_diameter, 36.0)

    geom_factor = 0.65 if tool_position.lower().startswith("ecc") else 1.0
    # Mass attenuation coefficient for ~1.5 MeV gamma rays in drilling mud (~0.055 cm^2/g)
    mu = 0.055
    half_inch_to_cm = 1.27

    exponent = (
        mu
        * half_inch_to_cm
        * geom_factor
        * (
            rho_mud * (d_hole - tool_diameter)
            - rho_mud_ref * (d_ref - tool_diameter)
        )
    )
    cf = np.clip(np.exp(exponent), 0.5, 2.5)
    gr_corr = np.maximum(0.0, gr_arr * cf)

    if is_series:
        return pd.Series(gr_corr, index=gr_idx, name=gr.name)
    if np.ndim(gr) == 0:
        return float(gr_corr)
    return gr_corr


def nphi_environmental_correction(
    nphi: np.ndarray | pd.Series | float,
    cali: np.ndarray | pd.Series | float = 8.5,
    bitsize: np.ndarray | pd.Series | float = 8.5,
    temp: np.ndarray | pd.Series | float = 75.0,
    pressure: np.ndarray | pd.Series | float = 0.0,
    mud_weight: np.ndarray | pd.Series | float = 9.0,
    salinity_ppm: np.ndarray | pd.Series | float = 0.0,
    standoff: float = 0.0,
) -> np.ndarray | pd.Series | float:
    """Apply environmental corrections to thermal Neutron Porosity (NPHI).

    Corrects raw thermal neutron porosity (fraction v/v) for departures from standard
    API calibration conditions (8.5 in borehole, 75 °F, atmospheric pressure, fresh mud):
    - **Borehole size (washout)**: Larger boreholes add hydrogen-rich mud between tool
      and formation, artificially increasing apparent NPHI (~ +0.012 v/v per inch above bit size).
    - **Formation temperature**: Higher temperature reduces fluid density/hydrogen index,
      requiring a positive correction (~ +0.004 v/v per 100 °F above 75 °F).
    - **Hydrostatic pressure**: Higher pressure compresses borehole/pore fluid, slightly
      increasing apparent hydrogen density (~ -0.0015 v/v per 1000 psi).
    - **Mud weight / barite**: High-density mud displaces water with solids, reducing
      borehole hydrogen (~ +0.002 v/v per ppg above 9 ppg).
    - **Salinity**: Chlorine has a high thermal neutron capture cross-section
      (~ +0.001 v/v per 50,000 ppm NaCl).
    - **Tool standoff**: Mud layer between pad and wall increases apparent porosity
      (~ -0.015 v/v per 0.5 in standoff).

    Args:
        nphi (np.ndarray | pd.Series | float): Measured Neutron Porosity (fraction v/v).
        cali (np.ndarray | pd.Series | float, optional): Caliper hole diameter (inches). Defaults to 8.5.
        bitsize (np.ndarray | pd.Series | float, optional): Reference bit size (inches). Defaults to 8.5.
        temp (np.ndarray | pd.Series | float, optional): Formation temperature (°F). Defaults to 75.0.
        pressure (np.ndarray | pd.Series | float, optional): Hydrostatic mud pressure (psi). Defaults to 0.0.
        mud_weight (np.ndarray | pd.Series | float, optional): Mud weight (ppg). Defaults to 9.0.
        salinity_ppm (np.ndarray | pd.Series | float, optional): Formation/mud salinity (ppm NaCl). Defaults to 0.0.
        standoff (float, optional): Tool standoff from borehole wall (inches). Defaults to 0.0.

    Returns:
        np.ndarray | pd.Series | float: Environmentally corrected NPHI (fraction v/v).
    """
    is_series = isinstance(nphi, pd.Series)
    nphi_idx = nphi.index if is_series else None
    nphi_arr = np.asarray(nphi, dtype=float)
    cali_arr = np.asarray(cali, dtype=float)
    bs_arr = np.asarray(bitsize, dtype=float)
    temp_arr = np.asarray(temp, dtype=float)
    pres_arr = np.asarray(pressure, dtype=float)
    mw_arr = np.asarray(mud_weight, dtype=float)
    sal_arr = np.asarray(salinity_ppm, dtype=float)

    d_hole = np.where(np.isnan(cali_arr), bs_arr, cali_arr)
    delta_hole = np.clip(d_hole - bs_arr, -4.0, 16.0)

    # Additive chartbook approximations (in porosity units v/v)
    corr_hole = -0.012 * delta_hole
    corr_temp = 0.004 * (( np.where(np.isnan(temp_arr), 75.0, temp_arr) - 75.0) / 100.0)
    corr_pres = -0.0015 * (np.where(np.isnan(pres_arr), 0.0, pres_arr) / 1000.0)
    mw_ppg = np.where(mw_arr <= 3.0, mw_arr * 8.3454, mw_arr)
    corr_mw = 0.002 * (np.where(np.isnan(mw_ppg), 9.0, mw_ppg) - 9.0)
    corr_sal = 0.001 * (np.where(np.isnan(sal_arr), 0.0, sal_arr) / 50000.0)
    corr_so = -0.030 * max(float(standoff), 0.0)

    nphi_corr = np.clip(
        nphi_arr + corr_hole + corr_temp + corr_pres + corr_mw + corr_sal + corr_so,
        -0.05,
        1.0,
    )

    if is_series:
        return pd.Series(nphi_corr, index=nphi_idx, name=nphi.name)
    if np.ndim(nphi) == 0:
        return float(nphi_corr)
    return nphi_corr


def density_mudcake_correction(
    rhob: np.ndarray | pd.Series | float,
    cali: np.ndarray | pd.Series | float,
    bitsize: np.ndarray | pd.Series | float = 8.5,
    rho_mudcake: float = 1.5,
    drho: np.ndarray | pd.Series | float | None = None,
) -> np.ndarray | pd.Series | float:
    """Correct Bulk Density (RHOB) for borehole rugosity, washout, and mudcake thickness.

    When the caliper indicates washout (`CALI > BS`) or when an uncompensated mudcake/rugosity
    effect is present, a dual-spacing density pad reads a weighted combination of mud/mudcake
    density and true formation bulk density:
    $$\\rho_{b,\\text{corr}} = \\rho_b + f_{\\text{pad}} \\cdot h_{\\text{mc}} \\cdot (\\rho_b - \\rho_{\\text{mc}})$$

    Args:
        rhob (np.ndarray | pd.Series | float): Measured Bulk Density log (g/cc).
        cali (np.ndarray | pd.Series | float): Caliper log (inches).
        bitsize (np.ndarray | pd.Series | float, optional): Bit size (inches). Defaults to 8.5.
        rho_mudcake (float, optional): Density of mudcake / borehole fluid in washout (g/cc).
            Defaults to 1.5.
        drho (np.ndarray | pd.Series | float | None, optional): Density correction curve
            (g/cc). If provided and excessive (`|DRHO| > 0.15`), adds residual washout
            compensation. Defaults to None.

    Returns:
        np.ndarray | pd.Series | float: Corrected Bulk Density (g/cc), clipped to `[1.0, 3.2]`.
    """
    is_series = isinstance(rhob, pd.Series)
    rhob_idx = rhob.index if is_series else None
    rhob_arr = np.asarray(rhob, dtype=float)
    cali_arr = np.asarray(cali, dtype=float)
    bs_arr = np.asarray(bitsize, dtype=float)

    # Effective standoff / washout wall gap (inches)
    gap_in = np.clip((np.where(np.isnan(cali_arr), bs_arr, cali_arr) - bs_arr) / 2.0, 0.0, 6.0)
    # Pad sensitivity factor per inch of wall gap beyond compensation
    pad_factor = 0.06
    delta_rhob = pad_factor * gap_in * np.maximum(rhob_arr - rho_mudcake, 0.0)

    if drho is not None:
        drho_arr = np.asarray(drho, dtype=float)
        excess_drho = np.where(
            np.abs(drho_arr) > 0.15,
            0.25 * (np.abs(drho_arr) - 0.15) * np.sign(drho_arr),
            0.0,
        )
        delta_rhob = delta_rhob + np.where(np.isnan(excess_drho), 0.0, excess_drho)

    rhob_corr = np.clip(rhob_arr + delta_rhob, 1.0, 3.2)

    if is_series:
        return pd.Series(rhob_corr, index=rhob_idx, name=rhob.name)
    if np.ndim(rhob) == 0:
        return float(rhob_corr)
    return rhob_corr


def resistivity_invasion_correction(
    rt: np.ndarray | pd.Series | float,
    rm: np.ndarray | pd.Series | float | None = None,
    rxo: np.ndarray | pd.Series | float | None = None,
) -> tuple[np.ndarray | pd.Series | float, np.ndarray | pd.Series | float]:
    """Correct deep resistivity (`RT`) for mud-filtrate invasion using a 3-curve tornado-chart approximation.

    Uses shallow (`RXO` / `MSFL` / `RS`), medium (`RM` / `ILM` / `RLA3`), and deep (`RT` / `ILD` / `RDEP`)
    resistivity measurements to reconstruct true uninvaded formation resistivity ($R_{t,\\text{corr}}$)
    and produce an invasion profile flag:
    - When both `RM` and `RXO` are available and conductive or resistive invasion is detected
      ($R_m \\neq R_t$ and $(R_t - R_m)(R_m - R_{xo}) > 0$):
      $$\\log_{10}(R_{t,\\text{corr}}) = \\log_{10}(R_t) + c_1 \\left(\\log_{10} R_t - \\log_{10} R_m\\right) + c_2 \\frac{(\\log_{10} R_t - \\log_{10} R_m)^2}{|\\log_{10} R_t - \\log_{10} R_{xo}| + 0.05}$$
    - Invasion flag (`INVASION_FLAG`):
      - `1`: Resistive hydrocarbon zone with conductive mud-filtrate invasion ($R_t > R_m > R_{xo}$)
      - `2`: Conductive water zone with fresh/resistive mud-filtrate invasion ($R_t < R_m < R_{xo}$)
      - `0`: Negligible invasion or single-curve input.

    Args:
        rt (np.ndarray | pd.Series | float): Deep resistivity log (ohm.m).
        rm (np.ndarray | pd.Series | float | None, optional): Medium resistivity log (ohm.m).
            Defaults to None.
        rxo (np.ndarray | pd.Series | float | None, optional): Shallow / flushed-zone
            resistivity log (ohm.m). Defaults to None.

    Returns:
        tuple: `(rt_corrected, invasion_flag)`
    """
    is_series = isinstance(rt, pd.Series)
    rt_idx = rt.index if is_series else None
    rt_arr = np.asarray(rt, dtype=float)
    rt_safe = np.clip(rt_arr, 1e-3, 1e5)

    inv_flag = np.zeros_like(rt_arr, dtype=int)
    log_rt = np.log10(rt_safe)
    log_rt_corr = log_rt.copy()

    if rm is not None and rxo is not None:
        rm_arr = np.clip(np.asarray(rm, dtype=float), 1e-3, 1e5)
        rxo_arr = np.clip(np.asarray(rxo, dtype=float), 1e-3, 1e5)
        log_rm = np.log10(rm_arr)
        log_rxo = np.log10(rxo_arr)

        valid = (~np.isnan(rt_arr)) & (~np.isnan(rm_arr)) & (~np.isnan(rxo_arr))
        # Monotonic separation profiles: Rt > Rm > Rxo (conductive invasion) or Rt < Rm < Rxo (resistive invasion)
        cond_inv = valid & (log_rt > log_rm + 0.03) & (log_rm > log_rxo + 0.03)
        res_inv = valid & (log_rt < log_rm - 0.03) & (log_rm < log_rxo - 0.03)
        active = cond_inv | res_inv

        inv_flag[cond_inv] = 1
        inv_flag[res_inv] = 2

        diff_dm = log_rt - log_rm
        diff_ds = np.abs(log_rt - log_rxo) + 0.05
        # Standard tornado-chart butterfly correction shift in log10 space (bounded to +/- 0.45 decades)
        delta_log = np.clip(
            0.35 * diff_dm + 0.25 * np.sign(diff_dm) * (diff_dm**2) / diff_ds,
            -0.45,
            0.45,
        )
        log_rt_corr = np.where(active, log_rt + delta_log, log_rt)
    elif rm is not None:
        rm_arr = np.clip(np.asarray(rm, dtype=float), 1e-3, 1e5)
        log_rm = np.log10(rm_arr)
        valid = (~np.isnan(rt_arr)) & (~np.isnan(rm_arr))
        cond_inv = valid & (log_rt > log_rm + 0.05)
        res_inv = valid & (log_rt < log_rm - 0.05)
        inv_flag[cond_inv] = 1
        inv_flag[res_inv] = 2
        delta_log = np.clip(0.30 * (log_rt - log_rm), -0.35, 0.35)
        log_rt_corr = np.where(cond_inv | res_inv, log_rt + delta_log, log_rt)

    rt_corr = np.where(np.isnan(rt_arr), np.nan, 10.0**log_rt_corr)

    if is_series:
        return (
            pd.Series(rt_corr, index=rt_idx, name=rt.name),
            pd.Series(inv_flag, index=rt_idx, name="INVASION_FLAG"),
        )
    if np.ndim(rt) == 0:
        return float(rt_corr), int(inv_flag)
    return rt_corr, inv_flag


def apply_environmental_corrections(
    df: pd.DataFrame,
    bitsize_default: float = 8.5,
    mud_weight: float | str = 9.0,
    surface_temp: float = 75.0,
    temp_grad: float = 1.5,
    salinity_ppm: float | str = 0.0,
    rho_mudcake: float = 1.5,
    inplace: bool = False,
    suffix: str = "_CORR",
) -> pd.DataFrame:
    """Apply automated borehole environmental and invasion corrections to a well log DataFrame.

    Orchestrates borehole/washout, temperature/pressure, mudcake, and filtrate invasion
    corrections across standard petrophysical curves (`GR`, `NPHI`, `RHOB`, and `RT`):
    1. Estimates bit size (`BS`) and `BADHOLE` flags via `badhole_flagging` if `CALI` is
       present and `BS` is missing.
    2. Computes formation temperature (`TEMP`) and hydrostatic pressure (`PRESSURE`) from
       `DEPTH` (or `TVD`) when not explicitly present in `df`.
    3. Applies `gr_borehole_correction` to `GR`.
    4. Applies `nphi_environmental_correction` to `NPHI`.
    5. Applies `density_mudcake_correction` to `RHOB` (using `DRHO` if available).
    6. Applies `resistivity_invasion_correction` to `RT` when medium (`RM` / `ILM` / `RMED`)
       and/or shallow (`RXO` / `MSFL` / `RSHAL`) resistivity curves are present.

    Args:
        df (pd.DataFrame): Input well log DataFrame.
        bitsize_default (float, optional): Default bit size (inches) if `BS` and `CALI`
            are absent. Defaults to 8.5.
        mud_weight (float | str, optional): Mud weight in ppg or column name in `df`.
            Defaults to 9.0.
        surface_temp (float, optional): Surface temperature (°F) for geothermal gradient
            estimation when `TEMP` is absent. Defaults to 75.0.
        temp_grad (float, optional): Geothermal gradient (°F per 100 ft or equivalent)
            when `TEMP` is absent. Defaults to 1.5.
        salinity_ppm (float | str, optional): Mud/formation salinity (ppm NaCl) or column
            name in `df`. Defaults to 0.0.
        rho_mudcake (float, optional): Mudcake density (g/cc). Defaults to 1.5.
        inplace (bool, optional): If True, overwrites `GR`, `NPHI`, `RHOB`, `RT` in place
            in addition to writing `<LOG><suffix>`. Defaults to False.
        suffix (str, optional): Suffix for corrected log columns. Defaults to `'_CORR'`.

    Returns:
        pd.DataFrame: DataFrame with environmentally corrected curves and `INVASION_FLAG`.
    """
    out_df = df.copy()

    # Resolve BS and CALI
    if "BS" in out_df.columns and out_df["BS"].notna().any():
        bs = out_df["BS"].fillna(bitsize_default)
    else:
        bs = pd.Series(bitsize_default, index=out_df.index, dtype=float)
    cali = out_df["CALI"].fillna(bs) if "CALI" in out_df.columns else bs

    # Resolve mud weight and salinity
    mw = (
        out_df[mud_weight]
        if isinstance(mud_weight, str) and mud_weight in out_df.columns
        else mud_weight
    )
    sal = (
        out_df[salinity_ppm]
        if isinstance(salinity_ppm, str) and salinity_ppm in out_df.columns
        else salinity_ppm
    )

    # Resolve temperature (°F) and hydrostatic pressure (psi)
    depth_col = "TVD" if "TVD" in out_df.columns else ("DEPTH" if "DEPTH" in out_df.columns else None)
    if "TEMP" in out_df.columns:
        temp = out_df["TEMP"]
    elif depth_col is not None:
        # Assume depth in meters if max depth < 6000, convert to 100 ft units for °F gradient
        d_vals = pd.to_numeric(out_df[depth_col], errors="coerce").fillna(0.0)
        depth_ft = d_vals * 3.28084 if d_vals.max() < 6000 else d_vals
        temp = surface_temp + temp_grad * (depth_ft / 100.0)
    else:
        temp = surface_temp

    if "PRESSURE" in out_df.columns:
        pressure = out_df["PRESSURE"]
    elif depth_col is not None:
        d_vals = pd.to_numeric(out_df[depth_col], errors="coerce").fillna(0.0)
        depth_ft = d_vals * 3.28084 if d_vals.max() < 6000 else d_vals
        mw_val = mw if not isinstance(mw, pd.Series) else mw.fillna(9.0)
        mw_ppg = np.where(np.asarray(mw_val) <= 3.0, np.asarray(mw_val) * 8.3454, np.asarray(mw_val))
        pressure = 0.052 * mw_ppg * depth_ft
    else:
        pressure = 0.0

    # 1. Gamma Ray borehole correction
    if "GR" in out_df.columns:
        gr_corr = gr_borehole_correction(out_df["GR"], cali=cali, bitsize=bs, mud_weight=mw)
        if suffix:
            out_df[f"GR{suffix}"] = gr_corr
        if inplace:
            out_df["GR"] = gr_corr

    # 2. Neutron Porosity environmental correction
    if "NPHI" in out_df.columns:
        nphi_corr = nphi_environmental_correction(
            out_df["NPHI"],
            cali=cali,
            bitsize=bs,
            temp=temp,
            pressure=pressure,
            mud_weight=mw,
            salinity_ppm=sal,
        )
        if suffix:
            out_df[f"NPHI{suffix}"] = nphi_corr
        if inplace:
            out_df["NPHI"] = nphi_corr

    # 3. Bulk Density mudcake / washout correction
    if "RHOB" in out_df.columns:
        drho = out_df["DRHO"] if "DRHO" in out_df.columns else None
        rhob_corr = density_mudcake_correction(
            out_df["RHOB"], cali=cali, bitsize=bs, rho_mudcake=rho_mudcake, drho=drho
        )
        if suffix:
            out_df[f"RHOB{suffix}"] = rhob_corr
        if inplace:
            out_df["RHOB"] = rhob_corr

    # 4. Deep Resistivity invasion correction
    if "RT" in out_df.columns:
        rm_col = next((c for c in ("RM", "ILM", "RMED", "RLA3") if c in out_df.columns), None)
        rxo_col = next((c for c in ("RXO", "MSFL", "RSHAL", "RS", "RLA1") if c in out_df.columns), None)
        rt_corr, inv_flag = resistivity_invasion_correction(
            out_df["RT"],
            rm=out_df[rm_col] if rm_col else None,
            rxo=out_df[rxo_col] if rxo_col else None,
        )
        if suffix:
            out_df[f"RT{suffix}"] = rt_corr
        if inplace:
            out_df["RT"] = rt_corr
        out_df["INVASION_FLAG"] = inv_flag

    return out_df


