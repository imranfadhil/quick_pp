import os
from pathlib import Path

import mlflow.tracking as mlflow_tracking
import pandas as pd
from mlflow.pyfunc import load_model
from tqdm import tqdm

from quick_pp import logger
from quick_pp.fluid_type import fix_fluid_segregation
from quick_pp.machine_learning.config import (
    CONFORMAL_ALPHA,
    FE_FEATURES,
    MODELLING_CONFIG,
    RAW_FEATURES,
)
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import (
    POST_PROCESSING,
    EnforceMineralSum,
    enforce_physical_constraints,
    get_latest_registered_models,
    predict_with_uncertainty,
    run_mlflow_server,
    unique_id,
)
from quick_pp.plotter.well_log import plotly_log


def load_data(hash: str) -> pd.DataFrame:
    """Load data from the specified directory using a hash to identify the file.

    Args:
        hash (str): A unique hash string contained within the target Parquet filename.

    Raises:
        FileNotFoundError: If no file is found with the specified hash.

    Returns:
        pd.DataFrame: The loaded well log data as a DataFrame.
    """
    data_dir = Path("data/input/")
    matching_files = list(data_dir.glob(f"*{hash}*.parquet"))
    if not matching_files:
        logger.error(f"No file found in {data_dir} containing hash '{hash}'")
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    logger.info(f"Loading data from {path}")
    return pd.read_parquet(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the input DataFrame by generating engineered features.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Raises:
        ValueError: If required columns for feature engineering are missing.

    Returns:
        pd.DataFrame: The DataFrame with added feature-engineered columns.
    """
    df = generate_fe_features(df)
    return df


def postprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Postprocess the DataFrame by enforcing single-column physical bounds, jointly normalizing
    ``['VCLAY', 'VSAND', 'VSILT', 'PHIT']`` via ``EnforceMineralSum``, inverting ``LOG_PERM`` to
    ``PERM`` (including P10–P90 bands), and computing fluid volumes.

    Args:
        df (pd.DataFrame): The DataFrame containing model predictions.

    Returns:
        pd.DataFrame: The postprocessed DataFrame with physical bounds enforced and added
                      'PERM', 'PERM_P10', 'PERM_P90', 'VHC', 'VWATER', 'VOIL', and 'VGAS' columns.
    """
    logger.info("Enforcing petrophysical bounds and volumetric closure constraints")
    df = df.copy()

    # 1. Single-column bound post-processing
    for k, fn in POST_PROCESSING.items():
        if k in df.columns:
            df[k] = fn(df[k])

    # 2. Multi-column joint post-processing for mineral volumes + porosity closure
    joint_mineral_poro_cols = [
        c for c in ["VCLAY", "VSAND", "VSILT", "PHIT"] if c in df.columns
    ]
    if len(joint_mineral_poro_cols) >= 2:
        df[joint_mineral_poro_cols] = EnforceMineralSum(
            mineral_cols=["VCLAY", "VSAND", "VSILT"], porosity_col="PHIT"
        )(df[joint_mineral_poro_cols])

    # 3. Full petrophysical constraints (tight-rock Sw, coal intervals, PERM inversion, P10-P90 ordering)
    df = enforce_physical_constraints(df)

    # Generate fluid volume fractions if porosity and water saturation exist
    poro_col = "PHIT" if "PHIT" in df.columns else ("PHIE" if "PHIE" in df.columns else None)
    sw_col = "SWT" if "SWT" in df.columns else ("SW" if "SW" in df.columns else None)

    if poro_col is not None and sw_col is not None:
        df["VHC"] = ((1.0 - df[sw_col]) * df[poro_col]).clip(lower=0.0, upper=1.0)
        df["VWATER"] = (df[sw_col] * df[poro_col]).clip(lower=0.0, upper=1.0)
    else:
        df["VHC"] = (1.0 - df.get("SWT", 1.0)) * df.get("PHIT", 0.0)

    if "OIL_FLAG" in df.columns and "GAS_FLAG" in df.columns:
        df = fix_fluid_segregation(df)

    return df


def save_predictions(pred_df: pd.DataFrame, output_file_name: str, plot: bool = False):
    """Save the predictions DataFrame to a Parquet file.
    Optionally, generate and save individual well log plots.

    Args:
        pred_df (pd.DataFrame): DataFrame containing predictions.
        output_file_name (str): The base name for the output Parquet and plot files.
        plot (bool, optional): If True, generate and save well log plots. Defaults to False.
    """
    hash = unique_id(pred_df)
    output_dir = Path("data/output/")
    os.makedirs(output_dir, exist_ok=True)
    output_path = Path(f"{output_dir}/{output_file_name}_{hash}.parquet")
    pred_df.to_parquet(output_path, index=False)

    if plot:
        output_dir = Path("data/output/plots")
        os.makedirs(output_dir, exist_ok=True)
        for well_name, well_df in tqdm(
            pred_df.groupby("WELL_NAME"),
            desc="Generating plots",
        ):
            fig = plotly_log(
                well_df,
                well_name=well_name,
                column_widths=[1, 1, 1, 1, 1, 1, 0.3, 1, 1],
            )
            plot_path = Path(f"{output_dir}/{well_name}.html")
            fig.write_html(plot_path, config={"scrollZoom": True})
            tqdm.write(f"Plot for well {well_name} saved to {plot_path}")
    logger.info(f"Predictions saved to {output_path}")


def predict_pipeline(
    model_config: str,
    data_hash: str,
    output_file_name: str,
    env: str = "local",
    plot_predictions: bool = False,
    alpha: float = CONFORMAL_ALPHA,
) -> None:
    """Execute the end-to-end prediction pipeline with P10–P90 conformal intervals and physics constraints.

    This function orchestrates loading data, preprocessing, loading the latest registered
    MLflow models, making point and P10–P90 interval predictions, postprocessing the results,
    and saving the output.

    Args:
        model_config (str): The key for the model configuration (e.g., 'clastic', 'carbonate').
        data_hash (str): The unique hash identifying the input data file.
        output_file_name (str): The base name for the output predictions file.
        env (str, optional): Environment for MLflow server. Defaults to 'local'.
        plot_predictions (bool, optional): If True, generate plots for each well. Defaults to False.
        alpha (float, optional): Significance level for conformal intervals (0.20 -> P10–P90).
                                 Defaults to CONFORMAL_ALPHA.
    """
    logger.info("Starting prediction pipeline")
    # Run MLflow server
    run_mlflow_server(env)

    # Get the information of the latest registered models
    client = mlflow_tracking.MlflowClient()
    latest_rms = get_latest_registered_models(client, model_config, data_hash)

    # Load the data
    data = load_data(data_hash)
    data = preprocess_data(data)

    base_cols = ["DEPTH", "WELL_NAME"] + [
        c for c in RAW_FEATURES + FE_FEATURES if c in data.columns
    ]
    pred_df = data[base_cols].copy()
    pred_df = pred_df.loc[:, ~pred_df.columns.duplicated()]

    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        targets = model_values["targets"]
        features = model_values["features"]
        reg_model_name = f"{model_config}_{model_key}_{data_hash}"
        logger.info(f"Predicting with model: {model_key} | {reg_model_name}")

        # Load the model
        model = load_model(latest_rms[reg_model_name]["model_uri"])

        # Run predictions + P10/P90 uncertainty bands and concat to pred_df
        X_feat = data[features].astype("float")
        preds, preds_p10, preds_p90 = predict_with_uncertainty(
            model, X_feat, targets=targets, alpha=alpha
        )
        temp_df = pd.DataFrame(preds, columns=targets, index=pred_df.index)
        if preds_p10 is not None and preds_p90 is not None:
            for idx, target_name in enumerate(targets):
                temp_df[f"{target_name}_P10"] = preds_p10[:, idx]
                temp_df[f"{target_name}_P90"] = preds_p90[:, idx]

        pred_df = pd.concat([pred_df, temp_df], axis=1)
        # Keep data updated with intermediate predictions (e.g. OIL_FLAG used by GAS model)
        for col in targets:
            data[col] = temp_df[col]

    # Merge specific flag columns from original data missing from pred_df
    merge_cols = ["WELL_NAME", "DEPTH"]
    missing_cols = [
        c
        for c in ["ROCK_FLAG", "COAL_FLAG", "TIGHT_FLAG"]
        if c in data.columns and c not in pred_df.columns
    ]
    if missing_cols:
        pred_df = pd.merge(
            pred_df, data[merge_cols + missing_cols], on=merge_cols, how="left"
        )

    # Postprocess the predictions and save
    pred_df = postprocess_data(pred_df)
    save_predictions(pred_df, output_file_name, plot=plot_predictions)
    logger.info("Prediction pipeline completed successfully")


_PRETRAINED_MODELS_CACHE: dict | None = None
_PRETRAINED_MODELS_CACHE_KEY: tuple[str, float] | None = None


def resolve_pretrained_models_path(
    model_path: str | Path | None = None,
) -> tuple[Path | None, Path]:
    """Resolve the existing pre-trained `.joblib` model path and the primary expected path."""
    env_default = Path(
        os.getenv("QPP_PRETRAINED_MODELS_PATH", "data/pretrained_models.joblib")
    )
    candidates = (
        [Path(model_path)]
        if model_path is not None
        else [
            env_default,
            Path("data/pretrained_models.joblib"),
            Path("/code/data/seed/pretrained_models.joblib"),
        ]
    )
    expected = candidates[0]
    for cand in candidates:
        if cand.exists() and cand.is_file():
            return cand, expected
    return None, expected


def load_pretrained_demo_models(model_path: str | Path | None = None) -> dict:
    """Load pre-trained Conformal PIML models from disk (`data/pretrained_models.joblib`)."""
    global _PRETRAINED_MODELS_CACHE, _PRETRAINED_MODELS_CACHE_KEY

    resolved_path, _ = resolve_pretrained_models_path(model_path)
    if resolved_path is None:
        return {}

    cache_key = (str(resolved_path.resolve()), resolved_path.stat().st_mtime)
    if (
        model_path is None
        and _PRETRAINED_MODELS_CACHE is not None
        and _PRETRAINED_MODELS_CACHE_KEY == cache_key
    ):
        return _PRETRAINED_MODELS_CACHE

    import joblib

    logger.info(f"Loading pre-trained conformal models from {resolved_path}")
    bundle = joblib.load(resolved_path)
    if not isinstance(bundle, dict):
        return {}
    if model_path is None:
        _PRETRAINED_MODELS_CACHE = bundle
        _PRETRAINED_MODELS_CACHE_KEY = cache_key
    return bundle


def _format_suite_label(suite_key: str, pretrained: bool, meta: dict | None = None) -> str:
    if meta and meta.get("label"):
        return str(meta["label"])
    key_lower = suite_key.lower().strip()
    if pretrained:
        if key_lower == "clastic":
            return "Clastic (Pre-Trained PIML)"
        if key_lower == "carbonate":
            return "Carbonate (Pre-Trained PIML)"
        arch = (meta or {}).get("architecture", "")
        suffix = f" ({str(arch).title()} PIML)" if arch else " (Pre-Trained)"
        return f"{suite_key.replace('_', ' ').title()}{suffix}"
    return f"{suite_key.replace('_', ' ').title()} (Synthetic Fallback)"


def save_pretrained_model_suite(
    suite_key: str,
    suite_entry: dict,
    model_path: str | Path | None = None,
) -> Path:
    """Save or update a trained model suite inside the global `.joblib` bundle on disk."""
    global _PRETRAINED_MODELS_CACHE, _PRETRAINED_MODELS_CACHE_KEY
    import joblib

    resolved_path, expected_path = resolve_pretrained_models_path(model_path)
    target_path = resolved_path if resolved_path is not None else expected_path

    bundle: dict = {}
    if resolved_path is not None and resolved_path.exists():
        try:
            existing = joblib.load(resolved_path)
            if isinstance(existing, dict):
                bundle = dict(existing)
        except Exception as e:
            logger.warning(f"Could not read existing bundle at {resolved_path}: {e}")

    # Ensure writable target path (if /code/data/seed is read-only, write to expected_path)
    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        bundle[suite_key] = suite_entry
        joblib.dump(bundle, target_path, compress=3)
    except PermissionError:
        target_path = expected_path
        target_path.parent.mkdir(parents=True, exist_ok=True)
        bundle[suite_key] = suite_entry
        joblib.dump(bundle, target_path, compress=3)

    if model_path is None:
        _PRETRAINED_MODELS_CACHE = bundle
        _PRETRAINED_MODELS_CACHE_KEY = (
            str(target_path.resolve()),
            target_path.stat().st_mtime,
        )
    logger.info(f"Saved pre-trained model suite '{suite_key}' to {target_path}")
    return target_path


def delete_pretrained_model_suite(
    suite_key: str,
    model_path: str | Path | None = None,
) -> bool:
    """Remove a model suite from the global `.joblib` bundle on disk."""
    global _PRETRAINED_MODELS_CACHE, _PRETRAINED_MODELS_CACHE_KEY
    import joblib

    resolved_path, _ = resolve_pretrained_models_path(model_path)
    if resolved_path is None or not resolved_path.exists():
        return False

    bundle = joblib.load(resolved_path)
    if not isinstance(bundle, dict) or suite_key not in bundle:
        return False

    del bundle[suite_key]
    joblib.dump(bundle, resolved_path, compress=3)
    if model_path is None:
        _PRETRAINED_MODELS_CACHE = bundle
        _PRETRAINED_MODELS_CACHE_KEY = (
            str(resolved_path.resolve()),
            resolved_path.stat().st_mtime,
        )
    return True


def get_pretrained_models_status(model_path: str | Path | None = None) -> dict:
    """Inspect disk for the pre-trained `.joblib` model bundle and return available suites + status."""
    resolved_path, expected_path = resolve_pretrained_models_path(model_path)
    bundle = load_pretrained_demo_models(model_path) if resolved_path is not None else {}

    suites: list[dict] = []
    if isinstance(bundle, dict) and bundle:
        for suite_key, suite_val in bundle.items():
            meta = (
                suite_val.get("_meta", {})
                if isinstance(suite_val, dict) and isinstance(suite_val.get("_meta"), dict)
                else {}
            )
            arch = str(meta.get("architecture") or suite_key).lower().strip()
            if arch not in MODELLING_CONFIG:
                arch = "carbonate" if "CARB" in (suite_val.keys() if isinstance(suite_val, dict) else []) else "clastic"

            sub_models = (
                [k for k in suite_val.keys() if not str(k).startswith("_")]
                if isinstance(suite_val, dict)
                else [str(suite_key).upper()]
            )
            targets: list[str] = []
            if isinstance(suite_val, dict):
                for mk, mobj in suite_val.items():
                    if str(mk).startswith("_"):
                        continue
                    t_names = getattr(mobj, "target_names", None)
                    if not t_names and arch in MODELLING_CONFIG:
                        t_names = MODELLING_CONFIG[arch].get(mk, {}).get("targets", [])
                    if t_names:
                        for t in t_names:
                            if t not in targets:
                                targets.append(str(t))
            else:
                t_names = getattr(suite_val, "target_names", None) or []
                targets.extend(str(t) for t in t_names)

            suites.append(
                {
                    "key": str(suite_key),
                    "label": _format_suite_label(str(suite_key), pretrained=True, meta=meta),
                    "architecture": arch,
                    "pretrained": True,
                    "sub_models": sub_models,
                    "targets": targets,
                    "meta": meta,
                }
            )

    if suites and resolved_path is not None:
        return {
            "pretrained_available": True,
            "fallback_active": False,
            "model_path": str(resolved_path),
            "expected_path": str(expected_path),
            "suites": suites,
            "notice": (
                f"Loaded pre-trained model bundle from '{resolved_path.as_posix()}' "
                f"({len(suites)} suite{'s' if len(suites) != 1 else ''}: "
                f"{', '.join(s['key'] for s in suites)})."
            ),
        }

    fallback_suites: list[dict] = []
    for suite_key, suite_cfg in MODELLING_CONFIG.items():
        if suite_key == "mock" or not isinstance(suite_cfg, dict):
            continue
        targets = []
        for mk, mcfg in suite_cfg.items():
            for t in mcfg.get("targets", []):
                if t not in targets:
                    targets.append(str(t))
        fallback_suites.append(
            {
                "key": str(suite_key),
                "label": _format_suite_label(str(suite_key), pretrained=False),
                "pretrained": False,
                "sub_models": list(suite_cfg.keys()),
                "targets": targets,
            }
        )

    return {
        "pretrained_available": False,
        "fallback_active": True,
        "model_path": None,
        "expected_path": str(expected_path.as_posix()),
        "suites": fallback_suites,
        "notice": (
            f"Pre-trained model file not found at '{expected_path.as_posix()}'. "
            "Running inference will use the on-the-fly physics-guided synthetic conformal "
            "fallback calibrated on the current well's logs (GR, RHOB, NPHI, RT)."
        ),
    }


def predict_well_dataframe(
    df: pd.DataFrame,
    model_config: str = "clastic",
    alpha: float = CONFORMAL_ALPHA,
    model_path: str | Path | None = None,
) -> pd.DataFrame:
    """Run pre-trained PIML & Conformal Prediction inference directly on a well DataFrame.

    Computes point predictions and calibrated P10–P90 uncertainty bands (`PHIT_P10/P90`,
    `SW_P10/P90`, `PERM_P10/P90`, `BVHC_P10/P90`) while enforcing petrophysical bounds
    and volumetric closure (`sum(V_mineral) + PHIT = 1.0`).
    """
    import numpy as np

    resolved_path, expected_path = resolve_pretrained_models_path(model_path)
    bundle = load_pretrained_demo_models(model_path=model_path)

    requested_suite = model_config.strip()
    suite = requested_suite.lower()

    # Match requested suite against loaded bundle keys first, then MODELLING_CONFIG
    matched_bundle_key = None
    if isinstance(bundle, dict):
        for k in bundle.keys():
            if str(k).lower().strip() == suite:
                matched_bundle_key = k
                break

    if matched_bundle_key is None and suite not in MODELLING_CONFIG:
        suite = "clastic"

    work = df.copy()
    if "WELL_NAME" not in work.columns:
        work["WELL_NAME"] = "DEMO_WELL"
    if "DEPTH" not in work.columns and "DEPT" in work.columns:
        work["DEPTH"] = work["DEPT"]

    defaults = {"GR": 45.0, "RT": 10.0, "NPHI": 0.15, "RHOB": 2.45}
    for col, default_val in defaults.items():
        if col not in work.columns:
            work[col] = default_val
        else:
            s = pd.to_numeric(work[col], errors="coerce")
            med = float(s.median()) if s.notna().any() else default_val
            work[col] = s.interpolate(limit_direction="both").fillna(med)

    fe_df = preprocess_data(work)
    for col in RAW_FEATURES + FE_FEATURES:
        if col in fe_df.columns:
            fe_df[col] = (
                pd.to_numeric(fe_df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .interpolate(limit_direction="both")
                .fillna(0.0)
            )

    raw_suite_entry = (
        bundle.get(matched_bundle_key)
        if (isinstance(bundle, dict) and matched_bundle_key is not None)
        else {}
    )
    if raw_suite_entry and not isinstance(raw_suite_entry, dict):
        suite_models = {str(matched_bundle_key).upper(): raw_suite_entry}
    elif isinstance(raw_suite_entry, dict):
        suite_models = raw_suite_entry
    else:
        suite_models = {}

    pred_df = fe_df.copy()
    active_sub_models = {
        k: v for k, v in suite_models.items() if not str(k).startswith("_")
    }
    pretrained_used = bool(active_sub_models)

    if active_sub_models:
        meta = (
            suite_models.get("_meta", {})
            if isinstance(suite_models.get("_meta"), dict)
            else {}
        )
        arch = str(meta.get("architecture") or suite).lower().strip()
        if arch not in MODELLING_CONFIG:
            arch = "carbonate" if "CARBONATE" in active_sub_models else "clastic"
        cfg_for_suite = MODELLING_CONFIG.get(arch, {})
        ordered_keys = list(cfg_for_suite.keys()) + [
            k for k in active_sub_models.keys() if k not in cfg_for_suite
        ]
        for model_key in ordered_keys:
            if model_key not in active_sub_models:
                continue
            model = active_sub_models[model_key]
            model_values = cfg_for_suite.get(model_key, {})
            targets = model_values.get("targets") or getattr(model, "target_names", None) or ["PHIT"]
            features = (
                model_values.get("features")
                or list(getattr(model, "feature_names_in_", []))
                or (RAW_FEATURES + FE_FEATURES)
            )
            for fcol in features:
                if fcol not in pred_df.columns:
                    pred_df[fcol] = 0.0
            X_feat = pred_df[features].astype("float").fillna(0.0)
            preds, preds_p10, preds_p90 = predict_with_uncertainty(
                model, X_feat, targets=targets, alpha=alpha
            )
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            for idx, target_name in enumerate(targets):
                pred_df[target_name] = preds[:, idx]
                if preds_p10 is not None and preds_p90 is not None:
                    pred_df[f"{target_name}_P10"] = preds_p10[:, idx]
                    pred_df[f"{target_name}_P90"] = preds_p90[:, idx]
    else:
        from sklearn.ensemble import RandomForestRegressor
        from quick_pp.machine_learning.utils import ConformalMultiOutputRegressor

        reg_key = "CARBONATE" if suite == "carbonate" else "CLASTIC"
        targets = MODELLING_CONFIG[suite][reg_key]["targets"]
        features = MODELLING_CONFIG[suite][reg_key]["features"]
        X_feat = pred_df[features].astype("float").fillna(0.0)

        vsh = np.clip((pred_df["GR"] - 15.0) / 105.0, 0.0, 1.0)
        phit = np.clip((2.65 - pred_df["RHOB"]) / 1.65 * 0.6 + pred_df["NPHI"] * 0.4, 0.01, 0.38)
        swt = np.clip((1.0 / (np.maximum(phit, 0.03) ** 2)) * (0.035 / np.maximum(pred_df["RT"], 0.2)), 0.05, 1.0) ** 0.5
        log_perm = np.clip(14.0 * phit - 2.2 - 1.5 * vsh, -2.5, 4.0)
        rem = np.clip(1.0 - phit - vsh * (1.0 - phit), 0.0, 1.0)
        y_syn = pd.DataFrame(
            {
                "PHIT": phit,
                "SWT": swt,
                "LOG_PERM": log_perm,
                targets[3]: rem * 0.70,
                targets[4]: vsh * (1.0 - phit) if targets[4] == "VCLAY" else rem * 0.30,
                targets[5]: rem * 0.30 if targets[4] == "VCLAY" else vsh * (1.0 - phit),
            },
            index=pred_df.index,
        )[targets]
        cmr = ConformalMultiOutputRegressor(
            base_estimator=RandomForestRegressor(n_estimators=25, max_depth=8, random_state=42),
            alpha=alpha,
            random_state=42,
            enforce_physics=True,
            target_names=targets,
        )
        cmr.fit(X_feat, y_syn)
        preds, preds_p10, preds_p90 = cmr.predict_interval(X_feat, alpha=alpha)
        for idx, target_name in enumerate(targets):
            pred_df[target_name] = preds[:, idx]
            pred_df[f"{target_name}_P10"] = preds_p10[:, idx]
            pred_df[f"{target_name}_P90"] = preds_p90[:, idx]

    pred_df = postprocess_data(pred_df)

    if "SWT" in pred_df.columns:
        pred_df["SW"] = pred_df["SWT"]
        if "SWT_P10" in pred_df.columns:
            pred_df["SW_P10"] = pred_df["SWT_P10"]
        if "SWT_P90" in pred_df.columns:
            pred_df["SW_P90"] = pred_df["SWT_P90"]

    if "PHIT" in pred_df.columns:
        vclay = pred_df["VCLAY"].clip(0.0, 1.0) if "VCLAY" in pred_df.columns else 0.0
        pred_df["PHIE"] = (pred_df["PHIT"] * (1.0 - 0.35 * vclay)).clip(lower=0.0, upper=pred_df["PHIT"])
        if "PHIT_P10" in pred_df.columns:
            pred_df["PHIE_P10"] = (pred_df["PHIT_P10"] * (1.0 - 0.35 * vclay)).clip(
                lower=0.0, upper=pred_df["PHIE"]
            )
        if "PHIT_P90" in pred_df.columns:
            pred_df["PHIE_P90"] = np.maximum(
                pred_df["PHIE"],
                (pred_df["PHIT_P90"] * (1.0 - 0.35 * vclay)).clip(lower=0.0, upper=0.50),
            )
        pred_df["SWE"] = pred_df.get("SW", 1.0)
        pred_df["BVW"] = (pred_df["PHIT"] * pred_df["SW"]).clip(0.0, 0.50)
        pred_df["BVHC"] = (pred_df["PHIT"] * (1.0 - pred_df["SW"])).clip(0.0, 0.50)
        if "PHIT_P10" in pred_df.columns and "SW_P90" in pred_df.columns:
            pred_df["BVHC_P10"] = np.minimum(
                pred_df["BVHC"],
                (pred_df["PHIT_P10"] * (1.0 - pred_df["SW_P90"])).clip(0.0, 0.50),
            )
        if "PHIT_P90" in pred_df.columns and "SW_P10" in pred_df.columns:
            pred_df["BVHC_P90"] = np.maximum(
                pred_df["BVHC"],
                (pred_df["PHIT_P90"] * (1.0 - pred_df["SW_P10"])).clip(0.0, 0.50),
            )

    pred_df.attrs["pretrained_used"] = pretrained_used
    pred_df.attrs["model_path"] = str(resolved_path.as_posix()) if resolved_path else None
    pred_df.attrs["expected_path"] = str(expected_path.as_posix())
    return pred_df

