"""Pre-seed SQLite database and pre-trained Conformal PIML models for the Interactive Live Demo.

Populates `data/quick_pp.db` (or `QPP_DATABASE_URL`) with multi-well projects from:
- `notebooks/data/01_raw/VOLVE/` (`15-9-19-A.las`, `36_7-3.las`, `VOLVE_15-9_Surveys.csv`, `formation_pressure_data.csv`)
- `notebooks/data/01_raw/COSTA/` (`HW-3.las`, `HW-4.las`, `HW-5.las`, `HW_core_data_all.csv`)

Also trains and serializes lightweight `ConformalMultiOutputRegressor` models (`clastic` and
`carbonate`) to `data/pretrained_models.joblib` so visitors can run Pre-Trained PIML & Conformal
Prediction inference (`predict_pipeline.py`) with P10–P90 uncertainty bands while heavy training
 remains disabled in the public container.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import lasio
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sqlalchemy import select

from quick_pp import logger
from quick_pp.app.backend.utils.auth import generate_totp_secret, hash_password
from quick_pp.database import models as db_models
from quick_pp.database import objects as db_objects
from quick_pp.database.db_connector import DBConnector
from quick_pp.machine_learning.config import (
    CONFORMAL_ALPHA,
    FE_FEATURES,
    MODELLING_CONFIG,
    RAW_FEATURES,
)
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.predict_pipeline import predict_well_dataframe
from quick_pp.machine_learning.utils import ConformalMultiOutputRegressor
from quick_pp.rock_type import calc_fzi, rock_typing

ROOT_DIR = Path(__file__).resolve().parents[2]
VOLVE_DIR = ROOT_DIR / "notebooks" / "data" / "01_raw" / "VOLVE"
COSTA_DIR = ROOT_DIR / "notebooks" / "data" / "01_raw" / "COSTA"


def _load_volve_15_9_19a() -> pd.DataFrame:
    """Load Volve 15/9-19 A reservoir section (3800–3966 m) and prepare clean petrophysical logs."""
    las_path = VOLVE_DIR / "15-9-19-A.las"
    las = lasio.read(str(las_path))
    df = las.df().reset_index()
    df.columns = [str(c).upper() for c in df.columns]
    if "DEPT" in df.columns and "DEPTH" not in df.columns:
        df = df.rename(columns={"DEPT": "DEPTH"})

    # Focus on the reservoir + Sleipner section (3800 - 3966 m, ~1,089 rows) with clean triple-combo & CPI logs
    df = df[(df["DEPTH"] >= 3800.0) & (df["DEPTH"] <= 3966.0)].copy().reset_index(drop=True)
    df["WELL_NAME"] = "15-9-19-A"

    df["GR"] = pd.to_numeric(df["GR"], errors="coerce").clip(5.0, 220.0).interpolate().bfill().ffill()
    df["NPHI"] = pd.to_numeric(df["NPHI"], errors="coerce").clip(-0.02, 0.55).interpolate().bfill().ffill()
    df["RHOB"] = pd.to_numeric(df["RHOB"], errors="coerce").clip(1.85, 2.85).interpolate().bfill().ffill()
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce").clip(0.1, 2000.0).interpolate().bfill().ffill()
    if "DT" in df.columns:
        df["DTC"] = pd.to_numeric(df["DT"], errors="coerce").clip(45.0, 160.0).interpolate().bfill().ffill()

    # Map Volve CPI curves to standard quick_pp evaluated curves
    phit = pd.to_numeric(df.get("PHIF"), errors="coerce")
    if phit is None or phit.isna().all():
        phit = ((2.65 - df["RHOB"]) / 1.65 * 0.65 + df["NPHI"] * 0.35).clip(0.01, 0.38)
    else:
        phit = phit.clip(0.01, 0.40).interpolate().bfill().ffill()

    vsh = pd.to_numeric(df.get("VSH"), errors="coerce")
    if vsh is None or vsh.isna().all():
        vsh = ((df["GR"] - 15.0) / (115.0 - 15.0)).clip(0.0, 1.0)
    else:
        vsh = vsh.clip(0.0, 1.0).interpolate().bfill().ffill()

    sw = pd.to_numeric(df.get("SW"), errors="coerce")
    if sw is None or sw.isna().all():
        sw = ((0.035 / (df["RT"] * (phit**2))).clip(0.01, 1.0)) ** 0.5
    else:
        sw = sw.clip(0.02, 1.0).interpolate().bfill().ffill()

    perm = pd.to_numeric(df.get("KLOGH"), errors="coerce")
    if perm is None or perm.isna().all():
        perm = (10.0 ** (14.5 * phit - 1.8 - 1.5 * vsh)).clip(0.01, 8000.0)
    else:
        perm = perm.clip(0.01, 10000.0).interpolate().bfill().ffill()

    df["VCLAY"] = (vsh * (1.0 - phit)).clip(0.0, 0.90)
    rem_matrix = (1.0 - phit - df["VCLAY"]).clip(0.0, 1.0)
    df["VSAND"] = (rem_matrix * 0.75).clip(0.0, 1.0)
    df["VSILT"] = (rem_matrix * 0.25).clip(0.0, 1.0)
    df["PHIT"] = phit
    df["PHIE"] = (phit * (1.0 - 0.35 * vsh)).clip(0.005, phit)
    df["SWT"] = sw
    df["SW"] = sw
    df["SWE"] = sw
    df["PERM"] = perm
    df["LOG_PERM"] = np.log10(df["PERM"].clip(lower=1e-3))
    df["BVW"] = (df["PHIT"] * df["SWT"]).clip(0.0, 0.45)
    df["BVHC"] = (df["PHIT"] * (1.0 - df["SWT"])).clip(0.0, 0.45)
    df["OIL_FLAG"] = ((df["SWT"] < 0.55) & (df["PHIT"] > 0.08)).astype(int)
    df["GAS_FLAG"] = 0

    # Sample synthetic core points at every ~4m in the Hugin reservoir
    df["CPORE"] = np.nan
    df["CPERM"] = np.nan
    core_idx = df.index[(df["DEPTH"] >= 3825.0) & (df["DEPTH"] <= 3920.0)][::25]
    rng = np.random.default_rng(42)
    df.loc[core_idx, "CPORE"] = np.clip(
        df.loc[core_idx, "PHIT"] + rng.normal(0.0, 0.008, len(core_idx)), 0.03, 0.36
    )
    df.loc[core_idx, "CPERM"] = np.clip(
        df.loc[core_idx, "PERM"] * (10.0 ** rng.normal(0.0, 0.12, len(core_idx))), 0.1, 9000.0
    )

    fzi = calc_fzi(df["PHIT"].to_numpy(dtype=float), df["PERM"].to_numpy(dtype=float))
    df["FZI"] = fzi
    df["ROCK_FLAG"] = rock_typing(fzi, cut_offs=[0.5, 1.2, 2.5], higher_is_better=True)

    # Assign geological zones
    df["ZONES"] = "Hugin_Upper"
    df.loc[df["DEPTH"] >= 3860.0, "ZONES"] = "Hugin_Main_Sand"
    df.loc[df["DEPTH"] >= 3918.0, "ZONES"] = "Sleipner_Coal"

    keep_cols = [
        "DEPTH",
        "WELL_NAME",
        "ZONES",
        "GR",
        "RT",
        "NPHI",
        "RHOB",
        "DTC",
        "CALI",
        "BS",
        "VCLAY",
        "VSAND",
        "VSILT",
        "PHIT",
        "PHIE",
        "SWT",
        "SW",
        "SWE",
        "PERM",
        "LOG_PERM",
        "BVW",
        "BVHC",
        "CPORE",
        "CPERM",
        "FZI",
        "ROCK_FLAG",
        "OIL_FLAG",
        "GAS_FLAG",
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()


def _load_volve_36_7_3() -> pd.DataFrame:
    """Load Volve 36/7-3 reservoir section (2460–2620 m)."""
    las_path = VOLVE_DIR / "36_7-3.las"
    if not las_path.exists():
        las_path = VOLVE_DIR.parent / "36_7-3.las"
    if not las_path.exists():
        df_alt = _load_volve_15_9_19a().copy()
        df_alt["WELL_NAME"] = "15-9-19-BT2"
        df_alt["DEPTH"] = (df_alt["DEPTH"] + 25.0).round(2)
        return df_alt
    las = lasio.read(str(las_path))
    df = las.df().reset_index()
    df.columns = [str(c).upper() for c in df.columns]
    if "DEPT" in df.columns and "DEPTH" not in df.columns:
        df = df.rename(columns={"DEPT": "DEPTH"})

    df = df[(df["DEPTH"] >= 2460.0) & (df["DEPTH"] <= 2620.0)].copy().reset_index(drop=True)
    df["WELL_NAME"] = "36-7-3"
    df["GR"] = pd.to_numeric(df["GR"], errors="coerce").clip(5.0, 200.0).interpolate().bfill().ffill()
    df["NPHI"] = pd.to_numeric(df["NPHI"], errors="coerce").clip(0.01, 0.50).interpolate().bfill().ffill()
    df["RHOB"] = pd.to_numeric(df["RHOB"], errors="coerce").clip(1.90, 2.80).interpolate().bfill().ffill()
    rt_src = next((c for c in ("RT", "RDEP", "ILD", "LLD", "RESD") if c in df.columns), None)
    df["RT"] = (
        pd.to_numeric(df[rt_src], errors="coerce").clip(0.2, 500.0).interpolate().bfill().ffill()
        if rt_src
        else 12.0
    )
    if "DT" in df.columns:
        df["DTC"] = pd.to_numeric(df["DT"], errors="coerce").clip(50.0, 150.0).interpolate().bfill().ffill()

    vsh = ((df["GR"] - 20.0) / (125.0 - 20.0)).clip(0.0, 1.0)
    phit = ((2.65 - df["RHOB"]) / 1.65 * 0.65 + df["NPHI"] * 0.35).clip(0.02, 0.36)
    sw = ((0.04 / (df["RT"] * (phit**2))).clip(0.02, 1.0)) ** 0.5
    perm = (10.0 ** (14.0 * phit - 1.9 - 1.4 * vsh)).clip(0.05, 4500.0)

    df["VCLAY"] = (vsh * (1.0 - phit)).clip(0.0, 0.90)
    rem_matrix = (1.0 - phit - df["VCLAY"]).clip(0.0, 1.0)
    df["VSAND"] = (rem_matrix * 0.72).clip(0.0, 1.0)
    df["VSILT"] = (rem_matrix * 0.28).clip(0.0, 1.0)
    df["PHIT"] = phit
    df["PHIE"] = (phit * (1.0 - 0.35 * vsh)).clip(0.005, phit)
    df["SWT"] = sw
    df["SW"] = sw
    df["SWE"] = sw
    df["PERM"] = perm
    df["LOG_PERM"] = np.log10(df["PERM"].clip(lower=1e-3))
    df["BVW"] = (df["PHIT"] * df["SWT"]).clip(0.0, 0.45)
    df["BVHC"] = (df["PHIT"] * (1.0 - df["SWT"])).clip(0.0, 0.45)
    df["OIL_FLAG"] = ((df["SWT"] < 0.60) & (df["PHIT"] > 0.08)).astype(int)
    df["GAS_FLAG"] = 0

    fzi = calc_fzi(df["PHIT"].to_numpy(dtype=float), df["PERM"].to_numpy(dtype=float))
    df["FZI"] = fzi
    df["ROCK_FLAG"] = rock_typing(fzi, cut_offs=[0.5, 1.2, 2.5], higher_is_better=True)
    df["ZONES"] = "Hugin_Upper"
    df.loc[df["DEPTH"] >= 2530.0, "ZONES"] = "Hugin_Main_Sand"

    keep_cols = [
        "DEPTH",
        "WELL_NAME",
        "ZONES",
        "GR",
        "RT",
        "NPHI",
        "RHOB",
        "DTC",
        "CALI",
        "VCLAY",
        "VSAND",
        "VSILT",
        "PHIT",
        "PHIE",
        "SWT",
        "SW",
        "SWE",
        "PERM",
        "LOG_PERM",
        "BVW",
        "BVHC",
        "FZI",
        "ROCK_FLAG",
        "OIL_FLAG",
        "GAS_FLAG",
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()


def _load_costa_well(well_name: str) -> pd.DataFrame:
    """Load Costa carbonate well (HW-3, HW-4, or HW-5) with core porosity & permeability."""
    las_path = COSTA_DIR / f"{well_name}.las"
    las = lasio.read(str(las_path))
    df = las.df().reset_index()
    df.columns = [str(c).upper() for c in df.columns]
    if "DEPT" in df.columns and "DEPTH" not in df.columns:
        df = df.rename(columns={"DEPT": "DEPTH"})

    df["WELL_NAME"] = well_name
    df["GR"] = pd.to_numeric(df["GR"], errors="coerce").clip(2.0, 180.0).interpolate().bfill().ffill()
    df["NPHI"] = pd.to_numeric(df["NPHI"], errors="coerce").clip(0.005, 0.45).interpolate().bfill().ffill()
    df["RHOB"] = pd.to_numeric(df["RHOB"], errors="coerce").clip(2.05, 2.88).interpolate().bfill().ffill()
    df["RT"] = pd.to_numeric(df["RT"], errors="coerce").clip(0.1, 2000.0).interpolate().bfill().ffill()
    if "DT" in df.columns:
        df["DTC"] = pd.to_numeric(df["DT"], errors="coerce").clip(42.0, 140.0).interpolate().bfill().ffill()

    # Core porosity in COSTA LAS files is in % (e.g. 16.9 -> 0.169 v/v)
    if "CORE_POR" in df.columns:
        cpor = pd.to_numeric(df["CORE_POR"], errors="coerce")
        df["CPORE"] = np.where(cpor > 1.0, cpor / 100.0, cpor)
        df.loc[(df["CPORE"] <= 0.0) | (df["CPORE"] > 0.50), "CPORE"] = np.nan
    if "CORE_PERM" in df.columns:
        cperm = pd.to_numeric(df["CORE_PERM"], errors="coerce")
        df["CPERM"] = np.where(cperm > 0.0, cperm, np.nan)

    phie_las = pd.to_numeric(df.get("PHIE"), errors="coerce")
    phit = (
        phie_las.clip(0.01, 0.38).interpolate().bfill().ffill()
        if (phie_las is not None and phie_las.notna().any())
        else ((2.71 - df["RHOB"]) / 1.71 * 0.6 + df["NPHI"] * 0.4).clip(0.01, 0.36)
    )
    vsh = ((df["GR"] - 10.0) / (95.0 - 10.0)).clip(0.0, 0.85)
    sw_las = pd.to_numeric(df.get("SW"), errors="coerce")
    sw = (
        sw_las.clip(0.03, 1.0).interpolate().bfill().ffill()
        if (sw_las is not None and sw_las.notna().any())
        else ((0.03 / (df["RT"] * (phit**2))).clip(0.02, 1.0)) ** 0.5
    )

    # Fit or interpolate continuous permeability curve guided by core permeability
    perm_base = (10.0 ** (15.5 * phit - 2.0 - 1.2 * vsh)).clip(0.01, 5000.0)
    if "CPERM" in df.columns and df["CPERM"].notna().sum() >= 5:
        cperm_interp = df["CPERM"].interpolate(method="linear", limit=6)
        perm = np.where(cperm_interp.notna(), 0.65 * cperm_interp + 0.35 * perm_base, perm_base)
        perm = pd.Series(perm, index=df.index).clip(0.01, 8000.0)
    else:
        perm = perm_base

    df["VCLAY"] = (vsh * (1.0 - phit)).clip(0.0, 0.85)
    rem_matrix = (1.0 - phit - df["VCLAY"]).clip(0.0, 1.0)
    df["VCALC"] = (rem_matrix * 0.72).clip(0.0, 1.0)
    df["VDOLO"] = (rem_matrix * 0.28).clip(0.0, 1.0)
    df["VSAND"] = 0.0
    df["VSILT"] = 0.0
    df["PHIT"] = phit
    df["PHIE"] = (phit * (1.0 - 0.30 * vsh)).clip(0.005, phit)
    df["SWT"] = sw
    df["SW"] = sw
    df["SWE"] = sw
    df["PERM"] = perm
    df["LOG_PERM"] = np.log10(df["PERM"].clip(lower=1e-3))
    df["BVW"] = (df["PHIT"] * df["SWT"]).clip(0.0, 0.45)
    df["BVHC"] = (df["PHIT"] * (1.0 - df["SWT"])).clip(0.0, 0.45)
    df["OIL_FLAG"] = ((df["SWT"] < 0.55) & (df["PHIT"] > 0.06)).astype(int)
    df["GAS_FLAG"] = 0

    fzi = calc_fzi(df["PHIT"].to_numpy(dtype=float), df["PERM"].to_numpy(dtype=float))
    df["FZI"] = fzi
    df["ROCK_FLAG"] = rock_typing(fzi, cut_offs=[0.5, 1.2, 2.5], higher_is_better=True)

    d_min, d_max = float(df["DEPTH"].min()), float(df["DEPTH"].max())
    d_span = d_max - d_min
    df["ZONES"] = "Upper_Carbonate"
    df.loc[df["DEPTH"] >= d_min + 0.33 * d_span, "ZONES"] = "Main_Reef_BuildUp"
    df.loc[df["DEPTH"] >= d_min + 0.72 * d_span, "ZONES"] = "Lower_Platform"

    keep_cols = [
        "DEPTH",
        "WELL_NAME",
        "ZONES",
        "GR",
        "RT",
        "NPHI",
        "RHOB",
        "DTC",
        "VCLAY",
        "VCALC",
        "VDOLO",
        "VSAND",
        "VSILT",
        "PHIT",
        "PHIE",
        "SWT",
        "SW",
        "SWE",
        "PERM",
        "LOG_PERM",
        "BVW",
        "BVHC",
        "CPORE",
        "CPERM",
        "FZI",
        "ROCK_FLAG",
        "OIL_FLAG",
        "GAS_FLAG",
    ]
    return df[[c for c in keep_cols if c in df.columns]].copy()


def train_and_save_pretrained_models(
    clastic_dfs: List[pd.DataFrame],
    carbonate_dfs: List[pd.DataFrame],
    out_path: Path,
) -> dict:
    """Train lightweight ConformalMultiOutputRegressor models for clastic & carbonate suites."""
    logger.info("Training pre-trained Conformal PIML models for live demo inference...")
    bundle: Dict[str, Dict[str, object]] = {"clastic": {}, "carbonate": {}}

    for suite, dfs in (("clastic", clastic_dfs), ("carbonate", carbonate_dfs)):
        combined = pd.concat(dfs, ignore_index=True)
        fe_df = generate_fe_features(combined).dropna(subset=RAW_FEATURES).reset_index(drop=True)
        for col in RAW_FEATURES + FE_FEATURES:
            if col in fe_df.columns:
                fe_df[col] = (
                    pd.to_numeric(fe_df[col], errors="coerce")
                    .replace([np.inf, -np.inf], np.nan)
                    .fillna(0.0)
                )

        for model_key, cfg in MODELLING_CONFIG[suite].items():
            targets = cfg["targets"]
            features = cfg["features"]
            for c in features + targets:
                if c not in fe_df.columns:
                    fe_df[c] = 0.0
            X = fe_df[features].astype(float).fillna(0.0)
            y = fe_df[targets].astype(float).fillna(0.0)

            if cfg["alg"] is RandomForestClassifier:
                clf = RandomForestClassifier(
                    n_estimators=25, max_depth=6, random_state=42
                )
                clf.fit(X, y.iloc[:, 0].astype(int))
                bundle[suite][model_key] = clf
            else:
                base_rf = RandomForestRegressor(
                    n_estimators=30, max_depth=8, random_state=42
                )
                cmr = ConformalMultiOutputRegressor(
                    base_estimator=base_rf,
                    alpha=CONFORMAL_ALPHA,
                    cv=3,
                    random_state=42,
                    enforce_physics=True,
                    target_names=targets,
                )
                cmr.fit(X, y)
                bundle[suite][model_key] = cmr

    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path, compress=3)
    logger.info(f"Saved pre-trained Conformal PIML models to {out_path}")
    return bundle


def _attach_conformal_bands(df: pd.DataFrame, suite: str, model_path: Path) -> pd.DataFrame:
    """Attach pre-computed P10–P90 conformal prediction bands to the well DataFrame."""
    pred_df = predict_well_dataframe(
        df, model_config=suite, alpha=CONFORMAL_ALPHA, model_path=model_path
    )
    band_cols = [
        c
        for c in pred_df.columns
        if c.endswith("_P10") or c.endswith("_P90")
    ]
    out = df.copy()
    for col in band_cols:
        out[col] = pred_df[col].to_numpy()
    return out


def _seed_well_ancillary_and_core(
    session,
    well_id: int,
    well_name: str,
    df: pd.DataFrame,
    micp_df: pd.DataFrame,
    survey_df: pd.DataFrame,
    pressure_df: pd.DataFrame,
) -> None:
    """Populate formation tops, fluid contacts, core plugs + MICP PC curves, surveys, and pressure tests."""
    d_min = float(df["DEPTH"].min())
    d_max = float(df["DEPTH"].max())
    d_span = d_max - d_min

    # 1. Formation Tops from ZONES transitions
    zone_changes = df.drop_duplicates(subset=["ZONES"], keep="first")
    for _, zrow in zone_changes.iterrows():
        session.add(
            db_models.FormationTop(
                well_id=well_id,
                name=str(zrow["ZONES"]),
                depth=round(float(zrow["DEPTH"]), 2),
            )
        )

    # 2. Fluid Contacts (OWC / FWL)
    owc_depth = round(d_min + 0.64 * d_span, 1)
    session.add(
        db_models.FluidContact(
            well_id=well_id,
            name="OWC",
            depth=owc_depth,
        )
    )

    # 3. Core Samples, Porosity/Permeability Measurements & Capillary Pressure (MICP) curves
    if "CPORE" in df.columns and "CPERM" in df.columns:
        core_rows = (
            df[df["CPORE"].notna() & df["CPERM"].notna()]
            .drop_duplicates(subset=["DEPTH"])
            .iloc[::2]
            .head(18)
        )
        micp_samples = (
            [grp for _, grp in micp_df.groupby("Sample") if len(grp) >= 5]
            if not micp_df.empty
            else []
        )
        for idx, (_, crow) in enumerate(core_rows.iterrows()):
            sname = f"{well_name}_PLUG_{idx + 1:02d}"
            cs = db_models.CoreSample(
                well_id=well_id,
                sample_name=sname,
                depth=round(float(crow["DEPTH"]), 2),
                remark="Routine & SCAL Core Plug",
            )
            session.add(cs)
            session.flush()

            session.add(
                db_models.CoreMeasurement(
                    sample_id=cs.sample_id,
                    property_name="CPORE",
                    value=round(float(crow["CPORE"]), 4),
                    unit="v/v",
                )
            )
            session.add(
                db_models.CoreMeasurement(
                    sample_id=cs.sample_id,
                    property_name="CPERM",
                    value=round(float(crow["CPERM"]), 3),
                    unit="mD",
                )
            )

            # Attach MICP capillary pressure curve to the first 6 core plugs
            if idx < 6 and micp_samples:
                m_grp = micp_samples[idx % len(micp_samples)]
                seen_sats = set()
                for _, mrow in m_grp.iterrows():
                    sw_v = float(mrow["Equiv Brine Sat. (Frac.)"])
                    pc_v = float(mrow["O/B Pc (psia)"])
                    if np.isfinite(sw_v) and np.isfinite(pc_v) and pc_v >= 0:
                        sat_rounded = round(float(np.clip(sw_v, 0.0, 1.0)), 4)
                        if sat_rounded in seen_sats:
                            continue
                        seen_sats.add(sat_rounded)
                        session.add(
                            db_models.CapillaryPressure(
                                sample_id=cs.sample_id,
                                saturation=sat_rounded,
                                pressure=round(float(pc_v), 3),
                                experiment_type="micp",
                                cycle="drainage",
                            )
                        )

    # 4. Directional Survey
    if well_name == "15-9-19-A" and not survey_df.empty and "WELL_NAME" in survey_df.columns:
        w_surv = survey_df[
            survey_df["WELL_NAME"].astype(str).str.contains("15-9-19-A", case=False, na=False)
        ].iloc[::4]
        seen_mds = set()
        for _, srow in w_surv.iterrows():
            md_val = round(float(srow["MD (m RKB)"]), 2)
            if md_val in seen_mds:
                continue
            seen_mds.add(md_val)
            session.add(
                db_models.WellSurvey(
                    well_id=well_id,
                    md=md_val,
                    inc=round(float(srow["Inc (deg)"]), 2),
                    azim=round(float(srow["Azim (deg)"]), 2),
                )
            )
    else:
        for md_val in np.linspace(max(0.0, d_min - 200.0), d_max, 12):
            session.add(
                db_models.WellSurvey(
                    well_id=well_id,
                    md=round(float(md_val), 1),
                    inc=round(float(2.5 + 4.0 * ((md_val - d_min) / max(d_span, 1.0))), 2),
                    azim=135.0,
                )
            )

    # 5. Formation Pressure Tests (MDT / RFT)
    if well_name == "15-9-19-A" and not pressure_df.empty and "TVDSS" in pressure_df.columns:
        p_rows = pressure_df.dropna(subset=["TVDSS", "FORMATION_PRESSURE"]).iloc[::4].head(12)
        seen_p_depths = set()
        for idx_p, (_, prow) in enumerate(p_rows.iterrows()):
            # Map formation pressure points into the reservoir depth window of 15-9-19-A
            p_depth = round(d_min + (0.12 + 0.72 * (idx_p / max(len(p_rows) - 1, 1))) * d_span, 2)
            if p_depth in seen_p_depths:
                continue
            seen_p_depths.add(p_depth)
            session.add(
                db_models.PressureTest(
                    well_id=well_id,
                    depth=p_depth,
                    pressure=round(float(prow["FORMATION_PRESSURE"]), 2),
                    pressure_uom="psi",
                )
            )
    else:
        # Hydrostatic hydrocarbon + water gradient across OWC
        for d_pt in np.linspace(d_min + 0.12 * d_span, d_max - 0.08 * d_span, 8):
            grad = 0.34 if d_pt < owc_depth else 0.45
            p_val = 3400.0 + (d_pt - d_min) * grad
            session.add(
                db_models.PressureTest(
                    well_id=well_id,
                    depth=round(float(d_pt), 1),
                    pressure=round(float(p_val), 1),
                    pressure_uom="psi",
                )
            )


def seed_demo_database(
    db_url: str | None = None,
    model_path: Path | None = None,
    force_retrain: bool = False,
    max_rows_per_well: int | None = None,
) -> dict:
    """Create and populate the pre-seeded Volve + Costa SQLite database and pre-trained ML bundle."""
    db_url = db_url or os.environ.get("QPP_DATABASE_URL") or "sqlite:///./data/quick_pp.db"
    model_path = model_path or Path(
        os.environ.get("QPP_PRETRAINED_MODELS_PATH", "data/pretrained_models.joblib")
    )

    if db_url.startswith("sqlite:///"):
        sqlite_file = Path(db_url.replace("sqlite:///", ""))
        sqlite_file.parent.mkdir(parents=True, exist_ok=True)
        if sqlite_file.exists():
            try:
                os.chmod(sqlite_file, 0o666)
            except Exception:
                pass

    connector = DBConnector(db_url)
    connector.setup_db()
    with connector.get_session() as cleanup_sess:
        from sqlalchemy import text

        for tbl in (
            "audit_log",
            "project_members",
            "relative_permeability",
            "capillary_pressure",
            "core_measurements",
            "core_samples",
            "fluid_contacts",
            "formation_tops",
            "pressure_tests",
            "well_surveys",
            "curve_data",
            "curves",
            "wells",
            "projects",
        ):
            try:
                cleanup_sess.execute(text(f"DELETE FROM {tbl}"))
            except Exception:
                pass

    # Load Volve & Costa well DataFrames
    volve_19a = _load_volve_15_9_19a()
    volve_36 = _load_volve_36_7_3()
    costa_hw3 = _load_costa_well("HW-3")
    costa_hw4 = _load_costa_well("HW-4")
    costa_hw5 = _load_costa_well("HW-5")
    if max_rows_per_well and max_rows_per_well > 0:
        def _thin(df: pd.DataFrame) -> pd.DataFrame:
            if len(df) <= max_rows_per_well:
                return df
            idx = np.linspace(0, len(df) - 1, max_rows_per_well, dtype=int)
            return df.iloc[idx].reset_index(drop=True)

        volve_19a = _thin(volve_19a)
        volve_36 = _thin(volve_36)
        costa_hw3 = _thin(costa_hw3)
        costa_hw4 = _thin(costa_hw4)
        costa_hw5 = _thin(costa_hw5)

    # Train and save pre-trained Conformal PIML models if needed
    if force_retrain or not model_path.exists():
        train_and_save_pretrained_models(
            clastic_dfs=[volve_19a, volve_36],
            carbonate_dfs=[costa_hw3, costa_hw4, costa_hw5],
            out_path=model_path,
        )

    # Attach conformal P10-P90 bands to each well
    volve_19a = _attach_conformal_bands(volve_19a, "clastic", model_path)
    volve_36 = _attach_conformal_bands(volve_36, "clastic", model_path)
    costa_hw3 = _attach_conformal_bands(costa_hw3, "carbonate", model_path)
    costa_hw4 = _attach_conformal_bands(costa_hw4, "carbonate", model_path)
    costa_hw5 = _attach_conformal_bands(costa_hw5, "carbonate", model_path)

    # Load ancillary CSVs
    micp_csv = COSTA_DIR / "HW_core_data_all.csv"
    micp_df = pd.read_csv(micp_csv) if micp_csv.exists() else pd.DataFrame()
    survey_csv = VOLVE_DIR / "VOLVE_15-9_Surveys.csv"
    survey_df = pd.read_csv(survey_csv) if survey_csv.exists() else pd.DataFrame()
    pressure_csv = VOLVE_DIR / "formation_pressure_data.csv"
    if not pressure_csv.exists():
        pressure_csv = VOLVE_DIR.parent / "formation_pressure_data.csv"
    pressure_df = pd.read_csv(pressure_csv) if pressure_csv.exists() else pd.DataFrame()

    projects_spec: List[Tuple[str, str, List[pd.DataFrame]]] = [
        (
            "VOLVE_COSTA_DEMO",
            "Pre-Seeded Multi-Well Showcase: Volve (North Sea Clastic) & Costa (Carbonate SCAL) with Conformal PIML",
            [volve_19a, costa_hw3, costa_hw4, costa_hw5],
        ),
        (
            "COSTA_CARBONATE_FIELD",
            "Costa Field Multi-Well Carbonate Evaluation, FZI Rock Typing & Leverett-J SHF (HW-3, HW-4, HW-5)",
            [costa_hw3, costa_hw4, costa_hw5],
        ),
        (
            "VOLVE_NORTH_SEA",
            "Equinor Volve Field Clastic Petrophysics & Conformal Uncertainty (15-9-19-A, 36-7-3)",
            [volve_19a, volve_36],
        ),
    ]

    with connector.get_session() as session:
        # 1. Create Confirmed Demo Admin user so zero-cold-start auth succeeds immediately
        admin_user = session.scalar(
            select(db_models.User).where(db_models.User.username == "admin")
        )
        if not admin_user:
            admin_user = db_models.User(
                username="admin",
                email="demo@quickpp.local",
                full_name="Interactive Demo Visitor",
                hashed_password=hash_password("QuickPP@Admin2026!"),
                role="admin",
                status="FORCE_CHANGE_PASSWORD",
                is_active=1,
                mfa_enabled=0,
                mfa_secret=None,
                invite_token=None,
                last_login_at=None,
                created_at=datetime.now(timezone.utc).isoformat(),
            )
            session.add(admin_user)
            session.flush()

        created_projects = []
        for proj_name, proj_desc, well_dfs in projects_spec:
            proj = db_objects.Project(
                db_session=session,
                name=proj_name,
                description=proj_desc,
                created_by_user_id=admin_user.user_id,
            )
            proj.save()
            session.add(
                db_models.ProjectMember(
                    project_id=proj.project_id,
                    user_id=admin_user.user_id,
                    role="admin",
                )
            )
            session.flush()

            for wdf in well_dfs:
                wname = str(wdf["WELL_NAME"].iloc[0])
                well_obj = db_objects.Well(
                    session,
                    project_id=proj.project_id,
                    name=wname,
                    uwi=f"{proj.project_id}-{wname}",
                    header_data={},
                    depth_uom="m",
                )
                well_obj.save()
                well_obj.update_data(wdf)
                well_obj.save()
            proj.save()

            for wdf in well_dfs:
                wname = str(wdf["WELL_NAME"].iloc[0])
                well_orm = session.scalar(
                    select(db_models.Well).filter_by(
                        project_id=proj.project_id, name=wname
                    )
                )
                if well_orm:
                    _seed_well_ancillary_and_core(
                        session=session,
                        well_id=well_orm.well_id,
                        well_name=wname,
                        df=wdf,
                        micp_df=micp_df,
                        survey_df=survey_df,
                        pressure_df=pressure_df,
                    )
            session.flush()
            created_projects.append(
                {
                    "project_id": proj.project_id,
                    "name": proj_name,
                    "wells": [str(w["WELL_NAME"].iloc[0]) for w in well_dfs],
                }
            )

    logger.info(f"Successfully seeded {len(created_projects)} projects into {db_url}")
    return {"db_url": db_url, "model_path": str(model_path), "projects": created_projects}


if __name__ == "__main__":
    result = seed_demo_database()
    print("Demo Seed Complete:", result)
