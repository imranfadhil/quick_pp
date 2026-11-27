"""Unified logging-environment model shared by QC, saturation and fluid modules.

All internal quantities are SI-like: depth in metres, temperature in degC,
resistivity in ohm.m. Unit conversion happens only at the edges.

The model covers:

* onshore and offshore (seabed temperature, water depth, air gap) wells,
* a formation temperature profile that is calibrated to bottom-hole
  temperature (BHT) records when available,
* borehole (mud column) and formation pressure,
* a mud program (hole section, bit size, mud type, weight, Rm/Rmf/Rmc with the
  temperature they were measured at) that is evaluated per depth sample,
* mud resistivity at formation temperature (Arps).

Every default that had to be assumed is recorded in ``Environment.assumptions``
so the caller (and the UI) can show what was not measured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

import numpy as np

FT_PER_M = 3.28084
M_PER_FT = 0.3048
SEAWATER_PSI_PER_FT = 0.4446  # 1.025 SG
BRINE_PSI_PER_FT = 0.465  # typical formation water hydrostatic gradient
PPG_PER_GCC = 8.3454
MUD_TYPES = ("WBM", "OBM", "SBM")


def f_to_c(temp_f):
    return (np.asarray(temp_f, dtype=float) - 32.0) * 5.0 / 9.0


def c_to_f(temp_c):
    return np.asarray(temp_c, dtype=float) * 9.0 / 5.0 + 32.0


def mud_weight_to_ppg(mw):
    """Convert mud weight to ppg; values <= 3 are assumed to be g/cc."""
    arr = np.asarray(mw, dtype=float)
    return np.where(arr <= 3.0, arr * PPG_PER_GCC, arr)


def arps_resistivity(r1, t1_c, t2_c):
    """Convert a resistivity from temperature ``t1_c`` to ``t2_c`` (Arps, degC form)."""
    return (
        np.asarray(r1, dtype=float)
        * (np.asarray(t1_c, dtype=float) + 21.5)
        / (np.asarray(t2_c, dtype=float) + 21.5)
    )


def rw_from_salinity(temp_c, salinity_ppm):
    """Brine resistivity (ohm.m) from temperature and NaCl salinity (Bateman-Konen)."""
    sal = np.maximum(np.asarray(salinity_ppm, dtype=float), 100.0)
    temp_f = np.asarray(c_to_f(temp_c), dtype=float)
    return (400000.0 / (sal * np.maximum(temp_f, 40.0))) ** 0.88


def rmf_from_rm(rm):
    """Rule-of-thumb filtrate resistivity from mud resistivity (same temperature)."""
    return 0.75 * np.asarray(rm, dtype=float)


@dataclass
class MudInterval:
    """One hole section of the mud program (depths in the log depth unit)."""

    top: float
    bottom: float
    bit_size_in: Optional[float] = None
    mud_type: str = "WBM"
    mud_weight_ppg: Optional[float] = None
    rm: Optional[float] = None
    rm_temp_c: Optional[float] = None
    rmf: Optional[float] = None
    rmf_temp_c: Optional[float] = None
    rmc: Optional[float] = None
    rmc_temp_c: Optional[float] = None
    salinity_ppm: Optional[float] = None

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MudInterval":
        mt = str(d.get("mud_type") or "WBM").upper()
        if mt not in MUD_TYPES:
            raise ValueError(f"mud_type must be one of {MUD_TYPES}, got {mt!r}")

        def _f(*keys):
            for key in keys:
                v = d.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                    if np.isfinite(fv):
                        return fv
                except (TypeError, ValueError):
                    continue
            return None

        return cls(
            top=float(d["top"]),
            bottom=float(d["bottom"]),
            bit_size_in=_f("bit_size_in", "bit_size"),
            mud_type=mt,
            mud_weight_ppg=_f("mud_weight_ppg", "mud_weight", "mud_density"),
            rm=_f("rm"),
            rm_temp_c=_f("rm_temp_c", "rm_temp"),
            rmf=_f("rmf"),
            rmf_temp_c=_f("rmf_temp_c", "rmf_temp"),
            rmc=_f("rmc"),
            rmc_temp_c=_f("rmc_temp_c", "rmc_temp"),
            salinity_ppm=_f("salinity_ppm", "mud_salinity", "mud_salinity_ppm"),
        )


@dataclass
class Environment:
    """Logging environment (temperature, pressure and mud program)."""

    depth_unit: str = "m"  # unit of the log depth axis: 'm' or 'ft'
    surface_temp_c: float = 15.0
    gradient_c_per_km: float = 30.0
    water_depth_m: float = 0.0  # > 0 => offshore
    air_gap_m: float = 0.0  # depth datum (KB/RT) above sea level, offshore only
    seabed_temp_c: float = 4.0
    bht_records: List[Dict[str, float]] = field(default_factory=list)  # depth_m, temp_c
    formation_pressure_gradient_psi_ft: float = BRINE_PSI_PER_FT
    mud_program: List[MudInterval] = field(default_factory=list)
    default_bit_size_in: float = 8.5
    default_mud_type: str = "WBM"
    default_mud_weight_ppg: float = 9.0
    default_mud_salinity_ppm: float = 0.0
    formation_salinity_ppm: float = 35000.0
    assumptions: List[str] = field(default_factory=list)
    # which model inputs were explicit (not defaulted); drives uncertainty
    explicit: Dict[str, bool] = field(default_factory=dict)

    # -- geometry -------------------------------------------------------
    @property
    def offshore(self) -> bool:
        return float(self.water_depth_m or 0.0) > 0.0

    @property
    def seabed_depth_m(self) -> float:
        return float(self.water_depth_m + self.air_gap_m) if self.offshore else 0.0

    @property
    def top_temp_c(self) -> float:
        return float(self.seabed_temp_c if self.offshore else self.surface_temp_c)

    def to_metres(self, depth) -> np.ndarray:
        arr = np.asarray(depth, dtype=float)
        return arr * M_PER_FT if self.depth_unit == "ft" else arr

    # -- temperature ----------------------------------------------------
    def calibrate_to_bht(self) -> None:
        """Fit the gradient through the surface/seabed anchor to the BHT records.

        BHT from wireline runs is normally colder than the undisturbed formation
        (mud circulation). No Horner correction is applied; this is recorded as an
        assumption so the uncertainty can reflect it.
        """
        pts = [
            (float(r["depth_m"]), float(r["temp_c"]))
            for r in self.bht_records
            if r.get("depth_m") is not None
            and r.get("temp_c") is not None
            and np.isfinite(r["depth_m"])
            and np.isfinite(r["temp_c"])
        ]
        pts = [(z, t) for z, t in pts if z > self.seabed_depth_m + 50.0]
        if not pts:
            return
        dz = np.array([z - self.seabed_depth_m for z, _ in pts], dtype=float)
        dt = np.array([t - self.top_temp_c for _, t in pts], dtype=float)
        grad = float(np.sum(dz * dt) / np.sum(dz * dz) * 1000.0)
        if np.isfinite(grad) and 5.0 <= grad <= 100.0:
            self.gradient_c_per_km = grad
            self.explicit["temperature"] = True
            self.assumptions.append(
                f"Temperature gradient {grad:.1f} degC/km calibrated to "
                f"{len(pts)} BHT record(s); BHT not Horner-corrected."
            )
        else:
            self.assumptions.append(
                f"BHT-implied gradient {grad:.1f} degC/km is implausible; default "
                f"{self.gradient_c_per_km:.1f} degC/km kept."
            )

    def temperature_c(self, tvd_m) -> np.ndarray:
        z = np.asarray(tvd_m, dtype=float)
        below = np.clip(z - self.seabed_depth_m, 0.0, None)
        return self.top_temp_c + self.gradient_c_per_km * below / 1000.0

    # -- pressure -------------------------------------------------------
    def borehole_pressure_psi(self, tvd_m, mud_weight_ppg) -> np.ndarray:
        """Static mud-column pressure at depth (no ECD)."""
        return (
            0.052
            * np.asarray(mud_weight_ppg, dtype=float)
            * (np.asarray(tvd_m, dtype=float) * FT_PER_M)
        )

    def formation_pressure_psi(self, tvd_m) -> np.ndarray:
        z = np.asarray(tvd_m, dtype=float)
        if self.offshore:
            wd_ft = self.water_depth_m * FT_PER_M
            below_ft = np.clip(z - self.seabed_depth_m, 0.0, None) * FT_PER_M
            return (
                SEAWATER_PSI_PER_FT * wd_ft
                + self.formation_pressure_gradient_psi_ft * below_ft
            )
        return self.formation_pressure_gradient_psi_ft * z * FT_PER_M

    # -- mud program ----------------------------------------------------
    def mud_arrays(self, depth, temp_c) -> Dict[str, np.ndarray]:
        """Evaluate the mud program per depth sample.

        Returns arrays (same length as ``depth``): ``bit_size_in``,
        ``mud_weight_ppg``, ``is_oil`` (OBM/SBM), ``salinity_ppm``, and
        ``rm``/``rmf``/``rmc`` at formation temperature (NaN when unknown, and for
        OBM/SBM filtrate which is non-conductive), plus ``rm_known``/``rmf_known``.
        """
        d = np.asarray(depth, dtype=float)
        n = d.size
        bit = np.full(n, float(self.default_bit_size_in))
        mw = np.full(n, float(self.default_mud_weight_ppg))
        oil = np.full(n, self.default_mud_type != "WBM", dtype=bool)
        sal = np.full(n, float(self.default_mud_salinity_ppm))
        rm = np.full(n, np.nan)
        rmf = np.full(n, np.nan)
        rmc = np.full(n, np.nan)
        t = np.asarray(temp_c, dtype=float) * np.ones(n)
        bit_set = np.zeros(n, dtype=bool)
        for iv in self.mud_program:
            sel = (d >= iv.top) & (d <= iv.bottom)
            if not sel.any():
                continue
            if iv.bit_size_in:
                bit[sel] = iv.bit_size_in
                bit_set[sel] = True
            if iv.mud_weight_ppg:
                mw[sel] = float(mud_weight_to_ppg(iv.mud_weight_ppg))
            oil[sel] = iv.mud_type != "WBM"
            if iv.salinity_ppm is not None:
                sal[sel] = iv.salinity_ppm
            for arr, val, tmeas in (
                (rm, iv.rm, iv.rm_temp_c),
                (rmf, iv.rmf, iv.rmf_temp_c),
                (rmc, iv.rmc, iv.rmc_temp_c),
            ):
                if val is not None and val > 0:
                    arr[sel] = arps_resistivity(
                        val, 25.0 if tmeas is None else tmeas, t[sel]
                    )
        rm_known = np.isfinite(rm)
        rmf_known = np.isfinite(rmf)
        # Fill gaps with documented estimates (WBM only; oil filtrate is non-conductive)
        fill_rmf = ~rmf_known & rm_known
        rmf[fill_rmf] = rmf_from_rm(rm[fill_rmf])
        need = ~np.isfinite(rmf) & ~oil
        if need.any():
            sal_eff = np.where(sal > 0, sal, 20000.0)
            rmf[need] = rw_from_salinity(t[need], sal_eff[need]) * 1.0
        rm_fill = ~np.isfinite(rm) & ~oil
        rm[rm_fill] = rmf[rm_fill] / 0.75
        rmf[oil] = np.nan
        rmc[oil] = np.nan
        rm[oil] = np.nan
        return {
            "bit_size_in": bit,
            "bit_size_explicit": bit_set,
            "mud_weight_ppg": mw,
            "is_oil": oil,
            "salinity_ppm": np.where(oil, 0.0, sal),
            "rm": rm,
            "rmf": rmf,
            "rmc": rmc,
            "rm_known": rm_known,
            "rmf_known": rmf_known | fill_rmf,
        }


def _num(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def environment_from_config(
    config: Optional[Dict[str, Any]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    depth_unit: str = "m",
) -> Environment:
    """Build an :class:`Environment` from a well config / LAS header dict plus overrides.

    Recognised keys (config and overrides): ``surface_temp`` (+``surface_temp_unit``),
    ``bht`` (+``bht_unit``, ``bht_depth``), ``water_depth``, ``air_gap`` / ``kb_elev``,
    ``seabed_temp_c``, ``temp_gradient_c_per_km``, ``bit_size``, ``mud_density``
    (g/cc or ppg), ``rmf``/``rmf_temp``, ``rmc``/``rmc_temp``, ``rm``/``rm_temp``,
    ``mud_type``, ``mud_salinity_ppm``, ``formation_salinity_ppm``,
    ``mud_program`` (list of dicts), ``bht_records`` (list of dicts).
    Overrides win; ``None`` values are ignored.
    """
    cfg: Dict[str, Any] = {}
    for src in (config or {}, overrides or {}):
        for k, v in src.items():
            if v is not None:
                cfg[k] = v

    env = Environment(depth_unit="ft" if str(depth_unit).lower() == "ft" else "m")
    log_unit_ft = env.depth_unit == "ft"
    notes = env.assumptions

    def _is_ft(u) -> bool:
        return str(u or "").strip().lower() in ("ft", "f", "feet", "foot")

    # --- top temperature / offshore --------------------------------------
    wd = _num(cfg.get("water_depth"))
    if wd and wd > 0:
        wd_from_override = bool(overrides and overrides.get("water_depth") is not None)
        wd_unit = (
            (overrides or {}).get("water_depth_unit")
            or (env.depth_unit if wd_from_override else None)
            or cfg.get("water_depth_unit")
            or cfg.get("datum_unit")
            or env.depth_unit
        )
        env.water_depth_m = wd * M_PER_FT if _is_ft(wd_unit) else wd
        kb = _num(cfg.get("air_gap", cfg.get("kb_elev")))
        if kb and kb > 0:
            kb_unit = cfg.get("datum_unit") or env.depth_unit
            env.air_gap_m = kb * M_PER_FT if _is_ft(kb_unit) else kb
        st = _num(cfg.get("seabed_temp_c"))
        env.seabed_temp_c = st if st is not None else 4.0
        env.explicit["seabed_temp"] = st is not None
        if st is None:
            notes.append("Offshore well: seabed temperature assumed 4 degC.")
    else:
        st = _num(cfg.get("surface_temp"))
        if st is not None:
            unit = str(cfg.get("surface_temp_unit") or "").lower()
            if unit in ("degf", "f", "degf.") or (not unit and st > 60.0):
                st = float(f_to_c(st))
            env.surface_temp_c = st
            env.explicit["surface_temp"] = True
        else:
            notes.append(
                f"Surface temperature not provided; assumed {env.surface_temp_c:.0f} degC."
            )

    # --- gradient / BHT ---------------------------------------------------
    g = _num(cfg.get("temp_gradient_c_per_km"))
    if g is not None:
        env.gradient_c_per_km = g
        env.explicit["temperature"] = True
    records: List[Dict[str, float]] = []
    for r in cfg.get("bht_records") or []:
        dm = _num(r.get("depth_m"))
        tc = _num(r.get("temp_c"))
        if dm is not None and tc is not None:
            records.append({"depth_m": dm, "temp_c": tc})
    bht = _num(cfg.get("bht"))
    if bht is not None and not records:
        unit = str(cfg.get("bht_unit") or "degC").lower()
        bht_c = float(f_to_c(bht)) if unit in ("degf", "f") else bht
        bd = _num(cfg.get("bht_depth"))
        if bd is not None:
            records.append(
                {"depth_m": bd * M_PER_FT if log_unit_ft else bd, "temp_c": bht_c}
            )
        else:
            notes.append(
                "Header BHT has no depth; it was not used to calibrate the gradient."
            )
    env.bht_records = records
    env.calibrate_to_bht()
    if not env.explicit.get("temperature"):
        notes.append(
            f"No BHT / gradient provided; default gradient {env.gradient_c_per_km:.0f} degC/km used."
        )

    # --- fluids / mud -----------------------------------------------------
    mt = str(cfg.get("mud_type") or "WBM").upper()
    if mt not in MUD_TYPES:
        raise ValueError(f"mud_type must be one of {MUD_TYPES}, got {mt!r}")
    env.default_mud_type = mt
    mw = _num(cfg.get("mud_weight_ppg", cfg.get("mud_density")))
    if mw is not None:
        env.default_mud_weight_ppg = float(mud_weight_to_ppg(mw))
        env.explicit["mud_weight"] = True
    bs = _num(cfg.get("bit_size"))
    if bs is not None and bs > 0:
        env.default_bit_size_in = bs
        env.explicit["bit_size"] = True
    ms = _num(cfg.get("mud_salinity_ppm"))
    if ms is not None:
        env.default_mud_salinity_ppm = ms
        env.explicit["mud_salinity"] = True
    fs = _num(cfg.get("formation_salinity_ppm"))
    if fs is not None:
        env.formation_salinity_ppm = fs
    fpg = _num(cfg.get("formation_pressure_gradient_psi_ft"))
    if fpg is not None:
        env.formation_pressure_gradient_psi_ft = fpg

    prog = cfg.get("mud_program") or []
    for item in prog:
        iv = item if isinstance(item, MudInterval) else MudInterval.from_dict(item)
        env.mud_program.append(iv)
        if iv.mud_weight_ppg is not None and iv.mud_weight_ppg > 0:
            env.explicit["mud_weight"] = True
        if iv.bit_size_in is not None and iv.bit_size_in > 0:
            env.explicit["bit_size"] = True
        if iv.salinity_ppm is not None or (iv.rmf is not None and iv.rmf > 0):
            env.explicit["mud_salinity"] = True
    if not env.explicit.get("mud_weight"):
        notes.append(
            f"Mud weight not provided; assumed {env.default_mud_weight_ppg:.1f} ppg."
        )

    def _mud_temp_c(key):
        t = _num(cfg.get(key))
        if t is None:
            return None
        unit = str(cfg.get(f"{key}_unit") or "").lower()
        if unit in ("degf", "f") or (not unit and t > 60.0):
            if not unit:
                notes.append(f"{key} = {t:g} read as degF (no unit given).")
            return float(f_to_c(t))
        return t

    if not env.mud_program and any(
        _num(cfg.get(k)) is not None for k in ("rm", "rmf", "rmc")
    ):
        for k in ("rm", "rmf", "rmc"):
            if _num(cfg.get(k)) is not None and _num(cfg.get(f"{k}_temp")) is None:
                notes.append(
                    f"{k.upper()} measurement temperature unknown; 25 degC assumed."
                )
        env.mud_program.append(
            MudInterval(
                top=-1e9,
                bottom=1e9,
                mud_type=mt,
                rm=_num(cfg.get("rm")),
                rm_temp_c=_mud_temp_c("rm_temp"),
                rmf=_num(cfg.get("rmf")),
                rmf_temp_c=_mud_temp_c("rmf_temp"),
                rmc=_num(cfg.get("rmc")),
                rmc_temp_c=_mud_temp_c("rmc_temp"),
            )
        )
    if mt != "WBM":
        notes.append(
            f"{mt}: non-conductive filtrate; borehole salinity and Rmf terms disabled."
        )
    elif not any(iv.rmf or iv.rm for iv in env.mud_program):
        notes.append(
            "Rm/Rmf not provided; estimated from mud salinity (assumed 20000 ppm if unknown)."
        )
    return env


def collect_assumptions(env: Environment, extra: Optional[Iterable[str]] = None):
    """Return a de-duplicated assumptions list."""
    seen, out = set(), []
    for msg in list(env.assumptions) + list(extra or []):
        if msg not in seen:
            seen.add(msg)
            out.append(msg)
    return out
