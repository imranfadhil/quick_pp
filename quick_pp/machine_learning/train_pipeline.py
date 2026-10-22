import importlib.util
import os
from pathlib import Path

import mlflow
import mlflow.sklearn as mlflow_sklearn
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, train_test_split

from quick_pp import logger
from quick_pp.machine_learning.config import CONFORMAL_ALPHA, MODELLING_CONFIG
from quick_pp.machine_learning.feature_engineering import generate_fe_features
from quick_pp.machine_learning.utils import (
    ConformalMultiOutputRegressor,
    physics_informed_loss,
    run_mlflow_server,
)

# Check if config.py exists in the root directory and update MODELLING_CONFIG if found
root_config_path = Path(os.getcwd(), "config.py")
if root_config_path.exists():
    try:
        spec = importlib.util.spec_from_file_location("root_config", root_config_path)
        root_config = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(root_config)
        if hasattr(root_config, "MODELLING_CONFIG"):
            logger.info("Updating MODELLING_CONFIG from root config.py")
            MODELLING_CONFIG.clear()
            MODELLING_CONFIG.update(root_config.MODELLING_CONFIG)
        else:
            logger.warning(
                "MODELLING_CONFIG not found in root config.py, using default MODELLING_CONFIG"
            )
    except Exception as e:
        logger.warning(f"Could not import MODELLING_CONFIG from root config.py: {e}")


def load_data(hash: str):
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
        raise FileNotFoundError(f"No file found in {data_dir} containing hash '{hash}'")
    path = matching_files[0]
    return pd.read_parquet(path)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the DataFrame by generating features and cleaning the data.

    Args:
        df (pd.DataFrame): The raw input DataFrame.

    Returns:
        pd.DataFrame: The preprocessed DataFrame with engineered features and
                      duplicates removed.
    """
    df = generate_fe_features(df)
    # Drop duplicates based on WELL_NAME and DEPTH, including duplicated columns
    df = df.drop_duplicates(subset=["WELL_NAME", "DEPTH"])
    df = df.loc[:, ~df.columns.duplicated()]

    return df


def split_data(
    df: pd.DataFrame,
    target_column: list[str],
    features: list[str],
    test_size: float = 0.2,
    random_state: int = 42,
    group_col: str = "WELL_NAME",
    splitter: str = "group_shuffle",
    n_splits: int = 5,
) -> list:
    """Split the data into training and testing sets using group-aware splitting.

    Uses `GroupShuffleSplit` (or `GroupKFold`) grouped by `group_col` (default `'WELL_NAME'`)
    so entire wells are held out together, preventing depth-adjacent spatial autocorrelation
    leakage between train and test sets. If fewer than 2 unique wells exist in `df`,
    automatically falls back to grouping by `'ZONES'` or contiguous depth blocks.

    Args:
        df (pd.DataFrame): The DataFrame to be split.
        target_column (list[str]): List of target column names.
        features (list[str]): List of feature column names.
        test_size (float, optional): Proportion of groups/data to include in the test split. Defaults to 0.2.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        group_col (str, optional): Column name used to define groups. Defaults to 'WELL_NAME'.
        splitter (str, optional): Splitting strategy: 'group_shuffle' (`GroupShuffleSplit`) or
                                  'group_kfold' (`GroupKFold`). Defaults to 'group_shuffle'.
        n_splits (int, optional): Number of folds when using `GroupKFold` or constructing
                                  contiguous depth blocks. Defaults to 5.

    Returns:
        list: A list containing `[X_train, X_test, y_train, y_test]`.
    """
    # Drop rows with NaN in target or features
    return_df = df.dropna(subset=target_column + features).copy()
    X = return_df[features].astype("float")
    y = return_df[target_column].astype("float")

    # Determine group labels to prevent spatial / well leakage
    if group_col in return_df.columns and return_df[group_col].nunique(dropna=True) >= 2:
        groups = return_df[group_col].astype(str).to_numpy()
        logger.info(
            f"Splitting data by '{group_col}' ({return_df[group_col].nunique()} unique groups) using {splitter}"
        )
    elif "ZONES" in return_df.columns and return_df["ZONES"].nunique(dropna=True) >= 2:
        groups = return_df["ZONES"].astype(str).to_numpy()
        logger.info(
            f"Single well detected; splitting data by 'ZONES' ({return_df['ZONES'].nunique()} zones) using {splitter}"
        )
    else:
        # Construct contiguous depth-interval blocks so adjacent samples do not leak across train/test
        n_blocks = max(2, min(max(n_splits, int(round(1.0 / max(test_size, 0.05)))), len(return_df) // 2))
        if len(return_df) >= 4 and n_blocks >= 2:
            groups = np.floor(np.linspace(0, n_blocks, len(return_df), endpoint=False)).astype(int)
            logger.info(
                f"Single well without multiple zones; splitting by {n_blocks} contiguous depth blocks using {splitter}"
            )
        else:
            return train_test_split(X, y, test_size=test_size, random_state=random_state)

    n_unique_groups = len(np.unique(groups))
    splitter_key = splitter.lower().strip()

    if splitter_key in ("group_kfold", "groupkfold", "gkf") and n_unique_groups >= 2:
        eff_splits = min(max(2, n_splits), n_unique_groups)
        gkf = GroupKFold(n_splits=eff_splits)
        train_idx, test_idx = next(gkf.split(X, y, groups=groups))
    else:
        gss = GroupShuffleSplit(
            n_splits=1, test_size=test_size, random_state=random_state
        )
        train_idx, test_idx = next(gss.split(X, y, groups=groups))

    return [
        X.iloc[train_idx],
        X.iloc[test_idx],
        y.iloc[train_idx],
        y.iloc[test_idx],
    ]


def train_model(
    alg,
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    use_conformal: bool = True,
    alpha: float = CONFORMAL_ALPHA,
):
    """Train the model using the specified algorithm, wrapping regressors in Conformal Prediction.

    Args:
        alg (callable): The scikit-learn model class to instantiate.
        X_train (pd.DataFrame): The training feature DataFrame.
        y_train (pd.DataFrame): The training target DataFrame.
        use_conformal (bool, optional): Whether to wrap regression models in
            `ConformalMultiOutputRegressor` for calibrated P10–P90 intervals. Defaults to True.
        alpha (float, optional): Significance level (0.20 -> P10 to P90). Defaults to CONFORMAL_ALPHA.

    Returns:
        object: The trained model instance.
    """
    logger.info(f"Training model: {getattr(alg, '__name__', str(alg))}")
    base_model = alg(random_state=42)

    if hasattr(base_model, "predict_proba") or not use_conformal:
        y_fit = (
            y_train.iloc[:, 0]
            if isinstance(y_train, pd.DataFrame) and y_train.shape[1] == 1
            else y_train
        )
        base_model.fit(X_train, y_fit)
        logger.debug("Classifier / standard model training complete")
        return base_model

    target_names = list(y_train.columns) if isinstance(y_train, pd.DataFrame) else None
    model = ConformalMultiOutputRegressor(
        base_estimator=base_model,
        alpha=alpha,
        random_state=42,
        enforce_physics=True,
        target_names=target_names,
    )
    model.fit(X_train, y_train)
    logger.debug("Conformal regression model training complete")
    return model


def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    """Evaluate the model using the test data, including physics loss and interval coverage.

    Args:
        model (object): The trained model to evaluate.
        X_test (pd.DataFrame): The testing feature DataFrame.
        y_test (pd.DataFrame): The testing target DataFrame.

    Returns:
        dict: A dictionary of evaluation metrics (e.g., 'f1_score', 'r2_score',
              'physics_loss', 'p10_p90_coverage').
    """
    logger.info(f"Evaluating model: {type(model).__name__}")
    y_pred = model.predict(X_test)

    # Check model type
    if hasattr(model, "predict_proba"):
        logger.debug("Classification metrics calculated")
        return {
            "f1_score": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
        }
    else:
        logger.debug("Regression metrics calculated")
        metrics = {
            "r2_score": r2_score(y_test, y_pred),
            "mean_absolute_error": mean_absolute_error(y_test, y_pred),
            "physics_loss": physics_informed_loss(y_test, y_pred, X=X_test),
        }
        if hasattr(model, "predict_interval"):
            _, y_p10, y_p90 = model.predict_interval(X_test)
            y_true_arr = (
                y_test.to_numpy(dtype=float)
                if isinstance(y_test, pd.DataFrame)
                else np.asarray(y_test, dtype=float)
            )
            if y_true_arr.ndim == 1:
                y_true_arr = y_true_arr.reshape(-1, 1)
            y_p10_arr = np.asarray(y_p10, dtype=float).reshape(y_true_arr.shape)
            y_p90_arr = np.asarray(y_p90, dtype=float).reshape(y_true_arr.shape)
            in_bounds = (y_true_arr >= y_p10_arr) & (y_true_arr <= y_p90_arr)
            metrics["p10_p90_coverage"] = float(np.nanmean(in_bounds))
        return metrics


# 6. Train pipeline
# 6. Train pipeline
def is_training_disabled() -> bool:
    """Return True if heavy model training is disabled (e.g. in the public Hugging Face Spaces demo)."""
    return (
        os.getenv("QPP_DEMO_MODE", "false").lower() in ("1", "true", "yes")
        or os.getenv("QPP_DISABLE_TRAINING", "false").lower() in ("1", "true", "yes")
    )


def train_pipeline(model_config: str, data_hash: str, env: str = "local"):
    """Execute the end-to-end model training pipeline.

    This function automates training, evaluating, and logging multiple models as defined
    in the configuration, leveraging MLflow for experiment tracking and model management.

    Args:
        model_config (str): The key for the model configuration (e.g., 'clastic').
        data_hash (str): The unique hash identifying the data file.
        env (str, optional): The MLflow environment ('local' or 'remote'). Defaults to 'local'.

    Raises:
        TypeError: If the targets or features are not lists of strings.
    """
    logger.info(
        f"Starting train_pipeline with model_config={model_config}, data_hash={data_hash}, env={env}"
    )

    if is_training_disabled():
        msg = (
            "Model training (train_pipeline.py) is disabled in the public interactive demo "
            "(QPP_DEMO_MODE=true). Please use pre-trained PIML & Conformal Prediction inference instead."
        )
        logger.warning(msg)
        raise PermissionError(msg)

    # Check if the model_config exists in MODELLING_CONFIG
    if model_config not in MODELLING_CONFIG:
        error_msg = (
            f"Model configuration '{model_config}' not found in MODELLING_CONFIG"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Run MLflow server
    run_mlflow_server(env)

    for model_key, model_values in MODELLING_CONFIG[model_config].items():
        alg = model_values["alg"]
        targets = model_values["targets"]
        features = model_values["features"]

        logger.info(f"Processing model: {model_key}")
        if not (isinstance(targets, list) and all(isinstance(t, str) for t in targets)):
            logger.error(f"Targets must be a list of strings, got {targets}")
            raise TypeError(f"Targets must be a list of strings, got {targets}")
        if not (
            isinstance(features, list) and all(isinstance(f, str) for f in features)
        ):
            logger.error(f"Features must be a list of strings, got {features}")
            raise TypeError(f"Features must be a list of strings, got {features}")

        df = load_data(data_hash)
        df = preprocess_data(df)
        # Skip if targets or features are not in the DataFrame
        if not all(col in df.columns for col in targets + features):
            missing_cols = [col for col in targets + features if col not in df.columns]
            logger.warning(
                f"Skipping model {model_key} due to missing columns: {missing_cols}"
            )
            continue
        X_train, X_test, y_train, y_test = split_data(df, targets, features)

        mlflow_dir = Path("./mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_experiment(model_config)
        with mlflow.start_run(
            run_name=model_key, description=str(model_values["description"])
        ):
            # Train model
            model = train_model(alg, X_train, y_train)

            # Log metrics
            metrics_dict = evaluate_model(model, X_test, y_test)
            for metric_name, metric_value in metrics_dict.items():
                mlflow.log_metric(metric_name, float(metric_value))
                logger.info(f"Logged metric: {metric_name}={metric_value}")

            # Log model
            reg_model_name = f"{model_config}_{model_key}_{data_hash}"
            signature = infer_signature(X_train, y_train)
            mlflow_sklearn.log_model(
                model,
                "model",
                signature=signature,
                input_example=X_test.sample(5),
                registered_model_name=reg_model_name,
            )
            logger.info(f"Model logged and registered as: {reg_model_name}")


def _prepare_training_targets(df: pd.DataFrame, arch: str) -> pd.DataFrame:
    """Ensure fixed target columns exist for 'clastic' or 'carbonate' training."""
    work = df.copy()

    # 1. Total Porosity (PHIT)
    if "PHIT" not in work.columns or work["PHIT"].isna().all():
        if "PHIE" in work.columns and work["PHIE"].notna().any():
            work["PHIT"] = pd.to_numeric(work["PHIE"], errors="coerce")
        else:
            raise ValueError(
                "Missing evaluated porosity ('PHIT' or 'PHIE') in selected wells. "
                "Run Lithology & Porosity on the training wells first."
            )
    else:
        work["PHIT"] = pd.to_numeric(work["PHIT"], errors="coerce")
        if "PHIE" in work.columns:
            work["PHIT"] = work["PHIT"].fillna(pd.to_numeric(work["PHIE"], errors="coerce"))

    # 2. Water Saturation (SWT)
    if "SWT" not in work.columns or work["SWT"].isna().all():
        if "SW" in work.columns and work["SW"].notna().any():
            work["SWT"] = pd.to_numeric(work["SW"], errors="coerce")
        elif "SWE" in work.columns and work["SWE"].notna().any():
            work["SWT"] = pd.to_numeric(work["SWE"], errors="coerce")
        else:
            raise ValueError(
                "Missing evaluated water saturation ('SWT' or 'SW') in selected wells. "
                "Run Saturation on the training wells first."
            )
    else:
        work["SWT"] = pd.to_numeric(work["SWT"], errors="coerce")
        if "SW" in work.columns:
            work["SWT"] = work["SWT"].fillna(pd.to_numeric(work["SW"], errors="coerce"))

    # 3. Log Permeability (LOG_PERM)
    if "LOG_PERM" not in work.columns or work["LOG_PERM"].isna().all():
        perm_col = next(
            (c for c in ("PERM", "PERM_TRANS", "CPERM") if c in work.columns and work[c].notna().any()),
            None,
        )
        if perm_col is None:
            raise ValueError(
                "Missing evaluated permeability ('PERM' or 'LOG_PERM') in selected wells. "
                "Run Permeability on the training wells first."
            )
        perm_s = pd.to_numeric(work[perm_col], errors="coerce").clip(lower=1e-3)
        work["LOG_PERM"] = np.log10(perm_s)
    else:
        work["LOG_PERM"] = pd.to_numeric(work["LOG_PERM"], errors="coerce")
        if "PERM" in work.columns:
            perm_s = pd.to_numeric(work["PERM"], errors="coerce").clip(lower=1e-3)
            work["LOG_PERM"] = work["LOG_PERM"].fillna(np.log10(perm_s))

    # 4. Lithology volume fractions
    if "VCLAY" not in work.columns or work["VCLAY"].isna().all():
        vcl_alt = next(
            (c for c in ("VSHA", "VSH", "VSHALE") if c in work.columns and work[c].notna().any()),
            None,
        )
        if vcl_alt:
            work["VCLAY"] = pd.to_numeric(work[vcl_alt], errors="coerce").clip(0.0, 1.0)
        else:
            gr_s = pd.to_numeric(work["GR"], errors="coerce")
            work["VCLAY"] = (((gr_s - 15.0) / 100.0).clip(0.0, 1.0) * (1.0 - work["PHIT"])).clip(0.0, 1.0)
    else:
        work["VCLAY"] = pd.to_numeric(work["VCLAY"], errors="coerce").clip(0.0, 1.0)

    rem_matrix = (1.0 - work["PHIT"].clip(0.0, 0.5) - work["VCLAY"].clip(0.0, 1.0)).clip(0.0, 1.0)

    if arch == "carbonate":
        if "VCALC" not in work.columns or work["VCALC"].isna().all():
            if "VCAL" in work.columns and work["VCAL"].notna().any():
                work["VCALC"] = pd.to_numeric(work["VCAL"], errors="coerce").clip(0.0, 1.0)
            else:
                work["VCALC"] = (rem_matrix * 0.72).clip(0.0, 1.0)
        else:
            work["VCALC"] = pd.to_numeric(work["VCALC"], errors="coerce").clip(0.0, 1.0)

        if "VDOLO" not in work.columns or work["VDOLO"].isna().all():
            if "VDOL" in work.columns and work["VDOL"].notna().any():
                work["VDOLO"] = pd.to_numeric(work["VDOL"], errors="coerce").clip(0.0, 1.0)
            else:
                work["VDOLO"] = (rem_matrix - work["VCALC"]).clip(0.0, 1.0)
        else:
            work["VDOLO"] = pd.to_numeric(work["VDOLO"], errors="coerce").clip(0.0, 1.0)
    else:
        if "VSAND" not in work.columns or work["VSAND"].isna().all():
            if "VQTZ" in work.columns and work["VQTZ"].notna().any():
                work["VSAND"] = pd.to_numeric(work["VQTZ"], errors="coerce").clip(0.0, 1.0)
            else:
                work["VSAND"] = (rem_matrix * 0.75).clip(0.0, 1.0)
        else:
            work["VSAND"] = pd.to_numeric(work["VSAND"], errors="coerce").clip(0.0, 1.0)

        if "VSILT" not in work.columns or work["VSILT"].isna().all():
            work["VSILT"] = (rem_matrix - work["VSAND"]).clip(0.0, 1.0)
        else:
            work["VSILT"] = pd.to_numeric(work["VSILT"], errors="coerce").clip(0.0, 1.0)

    # 5. Fluid flags for OIL / GAS classifiers
    if "OIL_FLAG" not in work.columns or work["OIL_FLAG"].isna().all():
        work["OIL_FLAG"] = ((work["SWT"] < 0.60) & (work["PHIT"] > 0.06)).astype(int)
    else:
        work["OIL_FLAG"] = pd.to_numeric(work["OIL_FLAG"], errors="coerce").fillna(0).astype(int)

    if "GAS_FLAG" not in work.columns or work["GAS_FLAG"].isna().all():
        gas_x = pd.to_numeric(work.get("GAS_XOVER", 0.0), errors="coerce").fillna(0.0)
        work["GAS_FLAG"] = ((gas_x > 0.03) & (work["SWT"] < 0.65)).astype(int)
    else:
        work["GAS_FLAG"] = pd.to_numeric(work["GAS_FLAG"], errors="coerce").fillna(0).astype(int)

    return work


def train_multi_well_conformal_model(
    df: pd.DataFrame,
    architecture: str = "clastic",
    suite_key: str = "clastic",
    suite_label: str | None = None,
    zones: list[str] | None = None,
    exclude_coal: bool = True,
    exclude_tight: bool = False,
    exclude_bad_hole: bool = True,
    rock_flags: list[int] | None = None,
    test_size: float = 0.20,
    alpha: float = CONFORMAL_ALPHA,
    n_estimators: int = 50,
    max_depth: int = 10,
    save_model: bool = False,
    source_project: str | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Train and evaluate a multi-well Conformal PIML model suite with restricted inputs & outputs.

    Restricted Inputs:
        - Raw logs: `['GR', 'RT', 'NPHI', 'RHOB']`
        - Engineered features: `['RHOB_INT', 'DPHI', 'GAS_XOVER', 'RHOMA_APP', 'AI', 'M_LITH', 'N_LITH']`
    Restricted Outputs:
        - Clastic: `['PHIT', 'SWT', 'LOG_PERM', 'VSAND', 'VSILT', 'VCLAY']` (+ `OIL_FLAG`, `GAS_FLAG`)
        - Carbonate: `['PHIT', 'SWT', 'LOG_PERM', 'VCALC', 'VCLAY', 'VDOLO']` (+ `OIL_FLAG`, `GAS_FLAG`)
    """
    from datetime import datetime, timezone
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from quick_pp.machine_learning.config import FE_FEATURES, RAW_FEATURES
    from quick_pp.machine_learning.predict_pipeline import save_pretrained_model_suite

    if is_training_disabled():
        msg = (
            "Model training is disabled in read-only demo mode (QPP_DEMO_MODE=true). "
            "Please use Pre-Trained PIML & Conformal Prediction inference instead."
        )
        logger.warning(msg)
        raise PermissionError(msg)

    arch = architecture.lower().strip()
    if arch not in ("clastic", "carbonate"):
        raise ValueError("Architecture must be 'clastic' or 'carbonate'.")

    clean_key = (
        "".join(ch if (ch.isalnum() or ch in ("_", "-")) else "_" for ch in suite_key.strip().lower())
        .strip("_")
        or arch
    )

    work = df.copy()
    if "DEPTH" not in work.columns and "DEPT" in work.columns:
        work["DEPTH"] = work["DEPT"]
    if "WELL_NAME" not in work.columns:
        work["WELL_NAME"] = "WELL_1"

    total_rows_raw = len(work)
    if total_rows_raw == 0:
        raise ValueError("No log rows found in the selected wells.")

    # 1. Zone filter
    if zones:
        if "ZONES" not in work.columns:
            raise ValueError("Selected zone filter cannot be applied because 'ZONES' column is missing.")
        zone_set = {str(z).strip() for z in zones if str(z).strip()}
        if zone_set:
            work = work[work["ZONES"].astype(str).str.strip().isin(zone_set)].copy()

    # 2. Flag filters
    if exclude_coal:
        if "COAL_FLAG" in work.columns:
            work = work[pd.to_numeric(work["COAL_FLAG"], errors="coerce").fillna(0) == 0]
        if "VCOAL" in work.columns:
            work = work[pd.to_numeric(work["VCOAL"], errors="coerce").fillna(0.0) <= 0.5]

    if exclude_tight and "TIGHT_FLAG" in work.columns:
        work = work[pd.to_numeric(work["TIGHT_FLAG"], errors="coerce").fillna(0) == 0]

    if exclude_bad_hole and "BAD_HOLE" in work.columns:
        work = work[pd.to_numeric(work["BAD_HOLE"], errors="coerce").fillna(0) == 0]

    if rock_flags:
        if "ROCK_FLAG" not in work.columns:
            raise ValueError("Rock flag filter specified, but 'ROCK_FLAG' curve is not present.")
        rf_set = {int(r) for r in rock_flags}
        rf_series = pd.to_numeric(work["ROCK_FLAG"], errors="coerce")
        work = work[rf_series.isin(rf_set)].copy()

    if len(work) < 15:
        raise ValueError(
            f"Insufficient valid rows ({len(work)}) after applying well/zone/flag filters (minimum 15 required)."
        )

    # 3. Verify required raw input logs
    missing_raw = [c for c in RAW_FEATURES if c not in work.columns or work[c].isna().all()]
    if missing_raw:
        raise ValueError(
            f"Missing required input log curve(s): {', '.join(missing_raw)}. "
            f"All 4 raw logs ({', '.join(RAW_FEATURES)}) are required."
        )

    for c in RAW_FEATURES:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=RAW_FEATURES).reset_index(drop=True)
    if len(work) < 15:
        raise ValueError(
            f"Insufficient rows ({len(work)}) with complete GR, RT, NPHI, and RHOB measurements."
        )

    # 4. Feature engineering & target preparation
    fe_df = preprocess_data(work)
    for col in RAW_FEATURES + FE_FEATURES:
        if col in fe_df.columns:
            fe_df[col] = (
                pd.to_numeric(fe_df[col], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(0.0)
            )

    fe_df = _prepare_training_targets(fe_df, arch)

    reg_key = "CARBONATE" if arch == "carbonate" else "CLASTIC"
    reg_cfg = MODELLING_CONFIG[arch][reg_key]
    reg_targets: list[str] = list(reg_cfg["targets"])
    reg_features: list[str] = list(reg_cfg["features"])

    fe_df = fe_df.dropna(subset=reg_targets + reg_features).reset_index(drop=True)
    filtered_rows = len(fe_df)
    if filtered_rows < 15:
        raise ValueError(
            f"Insufficient rows ({filtered_rows}) with valid target curves ({', '.join(reg_targets)})."
        )

    wells_used = sorted(fe_df["WELL_NAME"].astype(str).unique().tolist())
    zones_used = (
        sorted(fe_df["ZONES"].dropna().astype(str).unique().tolist())
        if "ZONES" in fe_df.columns
        else []
    )

    # 5. Group-aware Train / Test Split
    eff_test_size = float(np.clip(test_size, 0.10, 0.45))
    X_train, X_test, y_train, y_test = split_data(
        fe_df,
        target_column=reg_targets,
        features=reg_features,
        test_size=eff_test_size,
        random_state=42,
    )

    # 6. Train OIL & GAS classifiers + Conformal Multi-Output Regressor
    suite_models: dict = {}
    classifier_metrics: dict = {}

    for clf_key in ("OIL", "GAS"):
        clf_cfg = MODELLING_CONFIG[arch][clf_key]
        c_targets = clf_cfg["targets"]
        c_features = clf_cfg["features"]
        X_c_train = fe_df.loc[X_train.index, c_features].astype(float).fillna(0.0)
        y_c_train = fe_df.loc[X_train.index, c_targets[0]].astype(int)
        X_c_test = fe_df.loc[X_test.index, c_features].astype(float).fillna(0.0)
        y_c_test = fe_df.loc[X_test.index, c_targets[0]].astype(int)

        clf = RandomForestClassifier(
            n_estimators=max(10, int(n_estimators // 2)),
            max_depth=max(3, int(max_depth)),
            random_state=42,
        )
        clf.fit(X_c_train, y_c_train)
        suite_models[clf_key] = clf
        c_pred = clf.predict(X_c_test)
        classifier_metrics[clf_key] = {
            "target": c_targets[0],
            "accuracy": round(float(accuracy_score(y_c_test, c_pred)), 4),
            "f1_score": round(float(f1_score(y_c_test, c_pred, zero_division=0)), 4),
        }

    base_rf = RandomForestRegressor(
        n_estimators=max(10, int(n_estimators)),
        max_depth=max(3, int(max_depth)),
        random_state=42,
    )
    cmr = ConformalMultiOutputRegressor(
        base_estimator=base_rf,
        alpha=float(alpha),
        cv=3 if len(X_train) >= 18 else 2,
        random_state=42,
        enforce_physics=True,
        target_names=reg_targets,
    )
    cmr.fit(X_train, y_train)
    suite_models[reg_key] = cmr

    # 7. Evaluate overall and per-target metrics on held-out test set
    y_pred, y_p10, y_p90 = cmr.predict_interval(X_test, alpha=float(alpha))
    y_true_arr = y_test.to_numpy(dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float).reshape(y_true_arr.shape)
    y_p10_arr = np.asarray(y_p10, dtype=float).reshape(y_true_arr.shape)
    y_p90_arr = np.asarray(y_p90, dtype=float).reshape(y_true_arr.shape)

    overall_r2 = round(float(r2_score(y_true_arr, y_pred_arr)), 4)
    overall_mae = round(float(mean_absolute_error(y_true_arr, y_pred_arr)), 4)
    overall_rmse = round(float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2))), 4)
    phys_loss = round(float(physics_informed_loss(y_true_arr, y_pred_arr, X=X_test)), 4)
    in_bounds_all = (y_true_arr >= y_p10_arr) & (y_true_arr <= y_p90_arr)
    overall_cov_pct = round(float(np.nanmean(in_bounds_all)) * 100.0, 1)

    target_metrics: list[dict] = []
    for idx, t_name in enumerate(reg_targets):
        yt = y_true_arr[:, idx]
        yp = y_pred_arr[:, idx]
        yp10 = y_p10_arr[:, idx]
        yp90 = y_p90_arr[:, idx]

        ss_tot = float(np.sum((yt - np.mean(yt)) ** 2))
        ss_res = float(np.sum((yt - yp) ** 2))
        t_r2 = round(1.0 - ss_res / ss_tot, 4) if ss_tot > 1e-12 else 1.0
        t_mae = round(float(np.mean(np.abs(yt - yp))), 4)
        t_rmse = round(float(np.sqrt(np.mean((yt - yp) ** 2))), 4)
        t_cov = round(float(np.mean((yt >= yp10) & (yt <= yp90))) * 100.0, 1)
        t_width = round(float(np.mean(yp90 - yp10)), 4)

        target_metrics.append(
            {
                "Target": t_name,
                "R2": t_r2,
                "MAE": t_mae,
                "RMSE": t_rmse,
                "CoveragePct": t_cov,
                "MeanWidth": t_width,
            }
        )

    # Build test-set parity records (up to 250 samples) for visualization & inspection
    test_indices = X_test.index.to_numpy()
    if len(test_indices) > 250:
        sel = np.linspace(0, len(test_indices) - 1, 250, dtype=int)
    else:
        sel = np.arange(len(test_indices))

    parity_rows: list[dict] = []
    for s_idx in sel:
        orig_idx = test_indices[s_idx]
        row_dict: dict = {
            "WELL_NAME": str(fe_df.loc[orig_idx, "WELL_NAME"]),
            "DEPTH": round(float(fe_df.loc[orig_idx, "DEPTH"]), 2),
        }
        for t_i, t_name in enumerate(reg_targets):
            row_dict[f"{t_name}_TRUE"] = round(float(y_true_arr[s_idx, t_i]), 4)
            row_dict[f"{t_name}_PRED"] = round(float(y_pred_arr[s_idx, t_i]), 4)
            row_dict[f"{t_name}_P10"] = round(float(y_p10_arr[s_idx, t_i]), 4)
            row_dict[f"{t_name}_P90"] = round(float(y_p90_arr[s_idx, t_i]), 4)
        parity_rows.append(row_dict)

    display_label = (
        suite_label.strip()
        if suite_label and suite_label.strip()
        else f"{clean_key.replace('_', ' ').title()} ({arch.title()} PIML)"
    )
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    suite_models["_meta"] = {
        "label": display_label,
        "architecture": arch,
        "trained_at": trained_at,
        "source_project": source_project or "Workspace",
        "wells_used": wells_used,
        "zones_used": zones_used,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "alpha": float(alpha),
        "r2_score": overall_r2,
        "mae": overall_mae,
        "rmse": overall_rmse,
        "p10_p90_coverage_pct": overall_cov_pct,
    }

    # Cache the trained suite in memory so "Save Model" can persist it directly without re-training
    _LAST_TRAINED_MODELS_CACHE[str(source_project or "default")] = suite_models
    _LAST_TRAINED_MODELS_CACHE["latest"] = suite_models

    saved_path: str | None = None
    if save_model:
        out_p = save_pretrained_model_suite(clean_key, suite_models, model_path=model_path)
        saved_path = str(out_p.as_posix())

    return {
        "suite_key": clean_key,
        "suite_label": display_label,
        "architecture": arch,
        "saved": bool(save_model),
        "model_path": saved_path,
        "trained_at": trained_at,
        "inputs": {
            "raw_features": RAW_FEATURES,
            "fe_features": FE_FEATURES,
        },
        "outputs": {
            "regression_targets": reg_targets,
            "classification_targets": ["OIL_FLAG", "GAS_FLAG"],
        },
        "data_summary": {
            "total_rows_raw": int(total_rows_raw),
            "filtered_rows": int(filtered_rows),
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "wells_used": wells_used,
            "zones_used": zones_used,
        },
        "overall_metrics": {
            "r2_score": overall_r2,
            "mae": overall_mae,
            "rmse": overall_rmse,
            "physics_loss": phys_loss,
            "p10_p90_coverage_pct": overall_cov_pct,
            "confidence_level_pct": round((1.0 - float(alpha)) * 100.0, 1),
        },
        "target_metrics": target_metrics,
        "classifier_metrics": classifier_metrics,
        "parity_rows": parity_rows,
    }


_LAST_TRAINED_MODELS_CACHE: dict[str, dict] = {}


def save_cached_trained_model(
    suite_key: str,
    suite_label: str | None = None,
    source_project: str | None = None,
    model_path: str | Path | None = None,
) -> dict:
    """Persist the most recently evaluated multi-well model suite directly to disk without re-training."""
    from quick_pp.machine_learning.predict_pipeline import save_pretrained_model_suite

    if is_training_disabled():
        raise PermissionError(
            "Saving custom models is disabled in read-only demo mode (QPP_DEMO_MODE=true)."
        )

    suite_models = (
        _LAST_TRAINED_MODELS_CACHE.get(str(source_project))
        if source_project
        else None
    ) or _LAST_TRAINED_MODELS_CACHE.get("latest")

    if not suite_models or not isinstance(suite_models, dict):
        raise ValueError(
            "No trained model in memory to save. Please click 'Train & Evaluate' first."
        )

    meta = dict(suite_models.get("_meta", {}))
    arch = str(meta.get("architecture") or "clastic").lower().strip()
    clean_key = (
        "".join(
            ch if (ch.isalnum() or ch in ("_", "-")) else "_"
            for ch in suite_key.strip().lower()
        ).strip("_")
        or arch
    )
    display_label = (
        suite_label.strip()
        if suite_label and suite_label.strip()
        else str(
            meta.get("label")
            or f"{clean_key.replace('_', ' ').title()} ({arch.title()} PIML)"
        )
    )
    meta["label"] = display_label

    to_save = dict(suite_models)
    to_save["_meta"] = meta

    out_p = save_pretrained_model_suite(clean_key, to_save, model_path=model_path)
    return {
        "saved": True,
        "suite_key": clean_key,
        "suite_label": display_label,
        "architecture": arch,
        "model_path": str(out_p.as_posix()),
    }
