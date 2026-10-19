import copy
import os
import socket
from hashlib import sha256
from pathlib import Path
from subprocess import Popen
from typing import Optional

import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient
from pandas.util import hash_pandas_object
from sklearn.base import BaseEstimator, RegressorMixin, clone

from quick_pp import logger
from quick_pp.machine_learning.config import (
    CONFORMAL_ALPHA,
    MINERAL_VOL_COLS,
    MLFLOW_CONFIG,
    PHYSICAL_BOUNDS,
)

try:
    from mapie.regression import CrossConformalRegressor as _MapieCrossRegressor
except ImportError:
    _MapieCrossRegressor = None

try:
    from mapie.regression import MapieRegressor as _LegacyMapieRegressor
except ImportError:
    _LegacyMapieRegressor = None


def unique_id(df: pd.DataFrame) -> str:
    """Generate a unique ID for the DataFrame based on its content.

    Args:
        df (pd.DataFrame): DataFrame to hash.
    Returns:
        str: An 8-character unique hexadecimal ID for the DataFrame.
    """
    # Hash the DataFrame content and convert to hex string
    uid = sha256(hash_pandas_object(df, index=True).to_numpy().tobytes()).hexdigest()[
        :8
    ]
    logger.debug(f"Generated unique_id: {uid}")
    return uid


def is_mlflow_server_running(host, port):
    """Check if the MLflow server is running on the specified host and port.

    Args:
        host (str): Hostname or IP address of the MLflow server.
        port (int): Port number of the MLflow server.
    Returns:
        bool: True if the server is running, False otherwise.
    """
    try:
        with socket.create_connection((host, int(port)), timeout=2):
            logger.debug(f"MLflow server is running at {host}:{port}")
            return True
    except Exception as e:
        logger.debug(f"MLflow server not running at {host}:{port}: {e}")
        return False


def run_mlflow_server(env):
    """Start an MLflow tracking server if not already running.

    This function checks for a running MLflow server based on the environment configuration
    and sets the MLflow tracking URI accordingly.

    Args:
        env (str): The environment key to select the MLflow server configuration from MLFLOW_CONFIG.
    Raises:
        KeyError: If the specified environment is not found in MLFLOW_CONFIG.
    """
    mlruns_dir = Path(str(MLFLOW_CONFIG[env]["artifact_location"]))
    os.makedirs(mlruns_dir, exist_ok=True)
    mlflog_config = MLFLOW_CONFIG[env]
    if not is_mlflow_server_running(
        mlflog_config["tracking_host"], mlflog_config["tracking_port"]
    ):
        cmd_mlflow_server = (
            f"mlflow server --backend-store-uri {mlflog_config['backend_store_uri']} "
            f"--default-artifact-root {mlflog_config['artifact_location']} "
            f"--host {mlflog_config['tracking_host']} "
            f"--port {mlflog_config['tracking_port']}"
        )
        logger.warning(
            f"MLflow server is not running. Starting it now... | {cmd_mlflow_server}"
        )
        Popen(cmd_mlflow_server, shell=True)
        logger.info("MLflow server started successfully.")

    mlflow.set_tracking_uri(
        f"http://{MLFLOW_CONFIG[env]['tracking_host']}:{MLFLOW_CONFIG[env]['tracking_port']}"
    )
    logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")


def get_model_info(registered_model):
    """Extract key information from a registered MLflow model version object.

    Args:
        registered_model (list[mlflow.entities.model_registry.ModelVersion]): A list containing
            one or more registered model version objects. This function processes the first one.

    Returns:
        dict: A dictionary containing the model's name, run ID, version, URI, and stage.
    """
    model_info = {}
    for model in registered_model:
        model_info["reg_model_name"] = model.name
        model_info["run_id"] = model.run_id
        model_info["version"] = model.version
        model_info["model_uri"] = model.source
        model_info["stage"] = model.current_stage
    logger.debug(f"Extracted model info: {model_info}")
    return model_info


def get_latest_registered_models(
    client: MlflowClient, experiment_name: str, data_hash: str
) -> dict:
    """Get the latest versions of registered models from MLflow for a given experiment and data hash.

    Args:
        client (MlflowClient): MLflow client to interact with the tracking server.
        experiment_name (str): Name of the experiment to filter registered models.
        data_hash (str): The unique hash of the data used to train the models.

    Returns:
        dict: A dictionary where keys are registered model names and values are dicts of their details.
    """
    latest_rm_models = {}
    filter_str = f"name ILIKE '{experiment_name}%' AND name like '%{data_hash}'"
    logger.info(f"Searching for registered models with filter: {filter_str}")
    for rm in client.search_registered_models(filter_string=filter_str):
        latest_rm_info = get_model_info(rm.latest_versions)
        latest_rm_models[latest_rm_info["reg_model_name"]] = latest_rm_info
        logger.info(
            f"Adding {latest_rm_info['reg_model_name']} to latest_rm_models dictionary"
        )
    return latest_rm_models


def physics_informed_loss(
    y_true,
    y_pred,
    target_names: Optional[list[str]] = None,
    X: Optional[pd.DataFrame] = None,
    lambda_closure: float = 1.0,
    lambda_bounds: float = 1.0,
    lambda_tight_sw: float = 0.5,
) -> float:
    """Calculate a composite physics-informed loss combining prediction error and physical penalties.

    Penalizes:
        1. Volumetric closure violation: (sum(V_mineral) + PHIT - 1.0)^2
        2. Physical bound violations: negative porosity/permeability, Sw outside [0, 1], PHIE > PHIT
        3. Unphysical Sw drops in tight rock (where TIGHT_FLAG == 1 or PHIT < 0.03)

    Args:
        y_true (pd.DataFrame or np.ndarray): Ground-truth target values.
        y_pred (pd.DataFrame or np.ndarray): Predicted target values.
        target_names (list[str], optional): Column names of targets if arrays are passed.
        X (pd.DataFrame, optional): Feature DataFrame (for TIGHT_FLAG / RHOB context).
        lambda_closure (float, optional): Weight for volumetric closure penalty. Defaults to 1.0.
        lambda_bounds (float, optional): Weight for physical bound penalty. Defaults to 1.0.
        lambda_tight_sw (float, optional): Weight for tight-rock Sw spike penalty. Defaults to 0.5.

    Returns:
        float: Scalar composite physics-informed loss.
    """
    if isinstance(y_true, pd.DataFrame):
        if target_names is None:
            target_names = list(y_true.columns)
        y_true_arr = y_true.to_numpy(dtype=float)
    else:
        y_true_arr = np.asarray(y_true, dtype=float)

    if isinstance(y_pred, pd.DataFrame):
        if target_names is None:
            target_names = list(y_pred.columns)
        y_pred_arr = y_pred.to_numpy(dtype=float)
    else:
        y_pred_arr = np.asarray(y_pred, dtype=float)

    if y_true_arr.ndim == 1:
        y_true_arr = y_true_arr.reshape(-1, 1)
    if y_pred_arr.ndim == 1:
        y_pred_arr = y_pred_arr.reshape(-1, 1)

    mse_loss = float(np.nanmean((y_true_arr - y_pred_arr) ** 2))
    if not target_names:
        return mse_loss

    pred_df = pd.DataFrame(y_pred_arr, columns=target_names)
    bound_penalty = 0.0

    # 1. Bound penalties for each target
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in pred_df.columns:
            vals = pred_df[col].to_numpy(dtype=float)
            below = np.maximum(0.0, low - vals)
            above = np.maximum(0.0, vals - high)
            bound_penalty += float(np.nanmean(below**2 + above**2))

    # Porosity hierarchy: PHIE <= PHIT
    if "PHIE" in pred_df.columns and "PHIT" in pred_df.columns:
        phie_excess = np.maximum(
            0.0,
            pred_df["PHIE"].to_numpy(dtype=float)
            - pred_df["PHIT"].to_numpy(dtype=float),
        )
        bound_penalty += float(np.nanmean(phie_excess**2))

    # 2. Volumetric closure penalty: sum(V_mineral) + PHIT = 1.0
    closure_penalty = 0.0
    present_minerals = [c for c in MINERAL_VOL_COLS if c in pred_df.columns]
    for col in present_minerals:
        vals = pred_df[col].to_numpy(dtype=float)
        bound_penalty += float(
            np.nanmean(np.maximum(0.0, -vals) ** 2 + np.maximum(0.0, vals - 1.0) ** 2)
        )

    if present_minerals and "PHIT" in pred_df.columns:
        total_vol = pred_df[present_minerals].sum(axis=1).to_numpy(
            dtype=float
        ) + pred_df["PHIT"].to_numpy(dtype=float)
        closure_penalty = float(np.nanmean((total_vol - 1.0) ** 2))

    # 3. Tight rock Sw spike penalty
    tight_sw_penalty = 0.0
    sw_col = (
        "SWT"
        if "SWT" in pred_df.columns
        else ("SW" if "SW" in pred_df.columns else None)
    )
    if sw_col is not None:
        sw_vals = pred_df[sw_col].to_numpy(dtype=float)
        tight_mask = np.zeros(len(pred_df), dtype=bool)
        if "PHIT" in pred_df.columns:
            tight_mask |= pred_df["PHIT"].to_numpy(dtype=float) < 0.03
        elif "PHIE" in pred_df.columns:
            tight_mask |= pred_df["PHIE"].to_numpy(dtype=float) < 0.02
        if X is not None and isinstance(X, pd.DataFrame):
            if "TIGHT_FLAG" in X.columns and len(X) == len(pred_df):
                tight_mask |= X["TIGHT_FLAG"].to_numpy(dtype=float) >= 1.0
            elif "RHOB" in X.columns and len(X) == len(pred_df):
                tight_mask |= X["RHOB"].to_numpy(dtype=float) > 2.75
        if np.any(tight_mask):
            tight_sw_penalty = float(
                np.nanmean(np.maximum(0.0, 1.0 - sw_vals[tight_mask]) ** 2)
            )

    return (
        mse_loss
        + lambda_closure * closure_penalty
        + lambda_bounds * bound_penalty
        + lambda_tight_sw * tight_sw_penalty
    )


def compute_physics_sample_weights(
    X: pd.DataFrame, y: pd.DataFrame, min_weight: float = 0.25
) -> np.ndarray:
    """Compute sample weights that down-weight physically inconsistent training labels.

    Rows violating sum(V_mineral) + PHIT = 1.0 or physical bounds receive lower weight
    so the ML regressor prioritizes physically consistent petrophysical relationships.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.DataFrame): Training targets.
        min_weight (float, optional): Minimum sample weight floor. Defaults to 0.25.

    Returns:
        np.ndarray: 1D array of sample weights of length len(y).
    """
    n_samples = len(y)
    if not isinstance(y, pd.DataFrame) or n_samples == 0:
        return np.ones(n_samples, dtype=float)

    penalty = np.zeros(n_samples, dtype=float)

    for col, (low, high) in PHYSICAL_BOUNDS.items():
        if col in y.columns:
            vals = y[col].to_numpy(dtype=float)
            span = max(high - low, 1e-6)
            penalty += (
                np.maximum(0.0, low - vals) + np.maximum(0.0, vals - high)
            ) / span

    if "PHIE" in y.columns and "PHIT" in y.columns:
        penalty += np.maximum(
            0.0, y["PHIE"].to_numpy(dtype=float) - y["PHIT"].to_numpy(dtype=float)
        )

    present_minerals = [c for c in MINERAL_VOL_COLS if c in y.columns]
    if present_minerals and "PHIT" in y.columns:
        closure_err = np.abs(
            y[present_minerals].sum(axis=1).to_numpy(dtype=float)
            + y["PHIT"].to_numpy(dtype=float)
            - 1.0
        )
        penalty += closure_err

    weights = np.exp(-2.0 * np.nan_to_num(penalty, nan=0.0))
    return np.clip(weights, min_weight, 1.0)


def enforce_physical_constraints(
    df: pd.DataFrame,
    mineral_cols: Optional[list[str]] = None,
    tight_phit_cutoff: float = 0.03,
    tight_phie_cutoff: float = 0.02,
) -> pd.DataFrame:
    """Enforce petrophysical bounds, tight-rock Sw constraints, and volumetric closure.

    Specifically enforces:
        1. Physical bounds on porosity (0 <= PHIE <= PHIT <= 0.50), water saturation (0 <= Sw <= 1),
           and permeability (LOG_PERM in [-3, 5], PERM in [1e-3, 1e5] mD, strictly positive).
        2. Tight rock Sw spike suppression: where TIGHT_FLAG == 1 or porosity < cutoff,
           clamps Sw -> 1.0 (and OIL_FLAG/GAS_FLAG -> 0) and caps permeability.
        3. Coal interval handling: where COAL_FLAG == 1, clears non-reservoir porosity/permeability/HC flags.
        4. Volumetric closure constraint: sum(V_mineral) + PHIT = 1.0 across bulk-volume minerals.
        5. Uncertainty band ordering: ensures <CURVE>_P10 <= <CURVE> <= <CURVE>_P90 for all curves.

    Args:
        df (pd.DataFrame): DataFrame containing predicted logs and optional P10/P90 columns.
        mineral_cols (list[str], optional): Mineral volume columns to include in closure.
                                            Defaults to MINERAL_VOL_COLS.
        tight_phit_cutoff (float, optional): Total porosity cutoff for tight rock. Defaults to 0.03.
        tight_phie_cutoff (float, optional): Effective porosity cutoff for tight rock. Defaults to 0.02.

    Returns:
        pd.DataFrame: Physically constrained copy of the DataFrame.
    """
    df = df.copy()
    mineral_cols = mineral_cols or MINERAL_VOL_COLS

    # 1. Clamp primary curves and their P10/P90 uncertainty bands to physical bounds
    for col, (low, high) in PHYSICAL_BOUNDS.items():
        for suffix in ("", "_P10", "_P90"):
            cname = f"{col}{suffix}"
            if cname in df.columns:
                df[cname] = df[cname].clip(lower=low, upper=high)

    # 2. Enforce porosity hierarchy: 0 <= PHIE <= PHIT
    if "PHIT" in df.columns and "PHIE" in df.columns:
        df["PHIE"] = np.minimum(df["PHIE"], df["PHIT"])
        if "PHIE_P10" in df.columns and "PHIT_P10" in df.columns:
            df["PHIE_P10"] = np.minimum(df["PHIE_P10"], df["PHIT_P10"])
        if "PHIE_P90" in df.columns and "PHIT_P90" in df.columns:
            df["PHIE_P90"] = np.minimum(df["PHIE_P90"], df["PHIT_P90"])

    # 3. Identify tight streaks and coal intervals
    n_rows = len(df)
    tight_mask = pd.Series(False, index=df.index)
    if "TIGHT_FLAG" in df.columns:
        tight_mask = tight_mask | (df["TIGHT_FLAG"].fillna(0) >= 1.0)
    if "PHIT" in df.columns:
        tight_mask = tight_mask | (df["PHIT"] < tight_phit_cutoff)
    elif "PHIE" in df.columns:
        tight_mask = tight_mask | (df["PHIE"] < tight_phie_cutoff)

    coal_mask = (
        (df["COAL_FLAG"].fillna(0) >= 1.0)
        if "COAL_FLAG" in df.columns
        else pd.Series(False, index=df.index)
    )

    # Suppress unphysical Sw spikes in tight rock
    if tight_mask.any():
        for sw_col in ("SWT", "SW", "SWE"):
            if sw_col in df.columns:
                df.loc[tight_mask, sw_col] = 1.0
            if f"{sw_col}_P10" in df.columns:
                df.loc[tight_mask, f"{sw_col}_P10"] = np.maximum(
                    df.loc[tight_mask, f"{sw_col}_P10"], 0.90
                )
            if f"{sw_col}_P90" in df.columns:
                df.loc[tight_mask, f"{sw_col}_P90"] = 1.0
        for flag_col in ("OIL_FLAG", "GAS_FLAG"):
            if flag_col in df.columns:
                df.loc[tight_mask, flag_col] = 0
        if "LOG_PERM" in df.columns:
            df.loc[tight_mask, "LOG_PERM"] = np.minimum(
                df.loc[tight_mask, "LOG_PERM"], -1.0
            )
            if "LOG_PERM_P10" in df.columns:
                df.loc[tight_mask, "LOG_PERM_P10"] = np.minimum(
                    df.loc[tight_mask, "LOG_PERM_P10"], -1.5
                )
            if "LOG_PERM_P90" in df.columns:
                df.loc[tight_mask, "LOG_PERM_P90"] = np.minimum(
                    df.loc[tight_mask, "LOG_PERM_P90"], -0.5
                )

    # Clear reservoir predictions in coal intervals
    if coal_mask.any():
        for poro_col in ("PHIE", "PHIE_P10", "PHIE_P90"):
            if poro_col in df.columns:
                df.loc[coal_mask, poro_col] = 0.0
        for sw_col in ("SWT", "SW", "SWE", "SWT_P10", "SWT_P90", "SW_P10", "SW_P90", "SWE_P10", "SWE_P90"):
            if sw_col in df.columns:
                df.loc[coal_mask, sw_col] = 1.0
        for flag_col in ("OIL_FLAG", "GAS_FLAG"):
            if flag_col in df.columns:
                df.loc[coal_mask, flag_col] = 0
        for lperm_col in ("LOG_PERM", "LOG_PERM_P10", "LOG_PERM_P90"):
            if lperm_col in df.columns:
                df.loc[coal_mask, lperm_col] = -3.0

    # 4. Enforce Volumetric Closure: sum(V_mineral) + PHIT = 1.0
    present_minerals = [c for c in mineral_cols if c in df.columns]
    if present_minerals:
        for col in present_minerals:
            df[col] = df[col].clip(lower=0.0, upper=1.0)
            for suffix in ("_P10", "_P90"):
                if f"{col}{suffix}" in df.columns:
                    df[f"{col}{suffix}"] = df[f"{col}{suffix}"].clip(
                        lower=0.0, upper=1.0
                    )

        if "PHIT" in df.columns:
            phit_vals = df["PHIT"].clip(lower=0.0, upper=0.50)
            df["PHIT"] = phit_vals
            target_matrix_vol = 1.0 - phit_vals
            mineral_sum = df[present_minerals].sum(axis=1)

            # Where mineral_sum > 0, scale proportionally to (1 - PHIT); otherwise distribute evenly
            valid_sum = mineral_sum > 1e-8
            n_min = len(present_minerals)
            for col in present_minerals:
                df[col] = np.where(
                    valid_sum,
                    df[col] * (target_matrix_vol / np.where(valid_sum, mineral_sum, 1.0)),
                    target_matrix_vol / n_min,
                )

    # 5. Invert LOG_PERM (and P10/P90 bounds) to strictly positive PERM [mD]
    if "LOG_PERM" in df.columns:
        df["PERM"] = (10.0 ** df["LOG_PERM"]).clip(
            lower=PHYSICAL_BOUNDS["PERM"][0], upper=PHYSICAL_BOUNDS["PERM"][1]
        )
        if "LOG_PERM_P10" in df.columns:
            df["PERM_P10"] = (10.0 ** df["LOG_PERM_P10"]).clip(
                lower=PHYSICAL_BOUNDS["PERM"][0], upper=PHYSICAL_BOUNDS["PERM"][1]
            )
        if "LOG_PERM_P90" in df.columns:
            df["PERM_P90"] = (10.0 ** df["LOG_PERM_P90"]).clip(
                lower=PHYSICAL_BOUNDS["PERM"][0], upper=PHYSICAL_BOUNDS["PERM"][1]
            )
    elif "PERM" in df.columns:
        for suffix in ("", "_P10", "_P90"):
            cname = f"PERM{suffix}"
            if cname in df.columns:
                df[cname] = df[cname].clip(
                    lower=PHYSICAL_BOUNDS["PERM"][0], upper=PHYSICAL_BOUNDS["PERM"][1]
                )

    # Alias SW -> SWT if SWT is not present so downstream plotting & volumetrics work seamlessly
    if "SW" in df.columns and "SWT" not in df.columns:
        df["SWT"] = df["SW"]
        if "SW_P10" in df.columns:
            df["SWT_P10"] = df["SW_P10"]
        if "SW_P90" in df.columns:
            df["SWT_P90"] = df["SW_P90"]

    # 6. Guarantee P10 <= Pred <= P90 monotonicity for all curves with intervals
    base_cols = [
        c[:-4]
        for c in df.columns
        if c.endswith("_P10") and c[:-4] in df.columns and f"{c[:-4]}_P90" in df.columns
    ]
    for col in base_cols:
        low_vals = np.minimum(df[f"{col}_P10"], np.minimum(df[col], df[f"{col}_P90"]))
        high_vals = np.maximum(df[f"{col}_P90"], np.maximum(df[col], df[f"{col}_P10"]))
        df[f"{col}_P10"] = low_vals
        df[f"{col}_P90"] = high_vals

    return df


class ConformalMultiOutputRegressor(BaseEstimator, RegressorMixin):
    """Multi-output Conformal Prediction wrapper using MAPIE with physics constraints.

    Fits a conformalized regressor per target column (using MAPIE's CrossConformalRegressor
    when available, with fallback to split-conformal + ensemble tree quantiles) to produce
    calibrated P10–P90 (80%) prediction intervals alongside physically constrained point
    predictions.
    """

    def __init__(
        self,
        base_estimator=None,
        alpha: float = CONFORMAL_ALPHA,
        cv: int = 5,
        random_state: int = 42,
        enforce_physics: bool = True,
        target_names: Optional[list[str]] = None,
    ):
        self.base_estimator = base_estimator
        self.alpha = alpha
        self.cv = cv
        self.random_state = random_state
        self.enforce_physics = enforce_physics
        self.target_names = target_names

    def _make_base_estimator(self):
        if self.base_estimator is None:
            from sklearn.ensemble import RandomForestRegressor

            return RandomForestRegressor(random_state=self.random_state)
        if isinstance(self.base_estimator, type):
            return self.base_estimator(random_state=self.random_state)
        return clone(self.base_estimator)

    def fit(self, X, y, sample_weight=None):
        """Fit conformal regressors for each output target."""
        if isinstance(y, pd.DataFrame):
            self.target_names_ = list(y.columns)
            y_arr = y.to_numpy(dtype=float)
        else:
            y_arr = np.asarray(y, dtype=float)
            if y_arr.ndim == 1:
                self.target_names_ = self.target_names or ["TARGET_0"]
            else:
                self.target_names_ = self.target_names or [
                    f"TARGET_{i}" for i in range(y_arr.shape[1])
                ]

        self.is_1d_ = y_arr.ndim == 1
        if self.is_1d_:
            y_arr = y_arr.reshape(-1, 1)

        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.array(X.columns, dtype=object)
            X_arr = X.to_numpy(dtype=float)
        else:
            X_arr = np.asarray(X, dtype=float)

        if (
            sample_weight is None
            and self.enforce_physics
            and isinstance(y, pd.DataFrame)
        ):
            sample_weight = compute_physics_sample_weights(
                X if isinstance(X, pd.DataFrame) else pd.DataFrame(X_arr), y
            )

        n_samples, self.n_outputs_ = y_arr.shape
        effective_cv = max(2, min(self.cv, n_samples // 2)) if n_samples >= 6 else 2

        self.estimators_ = []
        self.mapie_models_ = []
        self.conformal_q_ = []

        for j in range(self.n_outputs_):
            y_col = y_arr[:, j]
            est_j = self._make_base_estimator()
            try:
                est_j.fit(X_arr, y_col, sample_weight=sample_weight)
            except TypeError:
                est_j.fit(X_arr, y_col)
            self.estimators_.append(est_j)

            # Fit MAPIE conformal regressor if enough samples
            mapie_j = None
            q_val = 0.0
            if n_samples >= 10 and _MapieCrossRegressor is not None:
                try:
                    mapie_j = _MapieCrossRegressor(
                        estimator=self._make_base_estimator(),
                        confidence_level=1.0 - self.alpha,
                        cv=effective_cv,
                        random_state=self.random_state,
                    )
                    mapie_j.fit_conformalize(X_arr, y_col)
                except Exception as e:
                    logger.debug(f"MAPIE CrossConformalRegressor fallback for target {j}: {e}")
                    mapie_j = None
            elif n_samples >= 10 and _LegacyMapieRegressor is not None:
                try:
                    mapie_j = _LegacyMapieRegressor(
                        estimator=self._make_base_estimator(),
                        method="plus",
                        cv=effective_cv,
                        random_state=self.random_state,
                    )
                    mapie_j.fit(X_arr, y_col)
                except Exception as e:
                    logger.debug(f"Legacy MapieRegressor fallback for target {j}: {e}")
                    mapie_j = None

            # Also compute empirical split/OOB residual quantile as fallback / local scaler
            preds_j = est_j.predict(X_arr)
            abs_res = np.abs(y_col - preds_j)
            # If tree ensemble, use tree dispersion to calibrate normalized conformal score
            if hasattr(est_j, "estimators_") and len(est_j.estimators_) > 1:
                tree_preds = np.column_stack(
                    [t.predict(X_arr) for t in est_j.estimators_]
                )
                tree_half_width = 0.5 * (
                    np.percentile(tree_preds, 100 * (1 - self.alpha / 2), axis=1)
                    - np.percentile(tree_preds, 100 * (self.alpha / 2), axis=1)
                )
                q_val = float(
                    np.quantile(
                        np.maximum(abs_res, tree_half_width),
                        min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / max(n_samples, 1))),
                    )
                )
            else:
                q_val = float(
                    np.quantile(
                        abs_res,
                        min(1.0, (1.0 - self.alpha) * (1.0 + 1.0 / max(n_samples, 1))),
                    )
                )

            self.mapie_models_.append(mapie_j)
            self.conformal_q_.append(q_val)

        return self

    def _raw_predict_and_intervals(self, X, alpha: Optional[float] = None):
        alpha_eff = alpha if alpha is not None else self.alpha
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        n_samples = X_arr.shape[0]

        y_pred = np.zeros((n_samples, self.n_outputs_), dtype=float)
        y_low = np.zeros((n_samples, self.n_outputs_), dtype=float)
        y_high = np.zeros((n_samples, self.n_outputs_), dtype=float)

        for j in range(self.n_outputs_):
            est_j = self.estimators_[j]
            mapie_j = self.mapie_models_[j]
            q_j = self.conformal_q_[j]

            # Local heteroscedastic dispersion from tree ensemble if available
            local_scale = np.ones(n_samples, dtype=float)
            if hasattr(est_j, "estimators_") and len(est_j.estimators_) > 1:
                tree_preds = np.column_stack(
                    [t.predict(X_arr) for t in est_j.estimators_]
                )
                tree_std = np.std(tree_preds, axis=1)
                mean_std = float(np.mean(tree_std))
                if mean_std > 1e-8:
                    # Blend constant conformal width (0.5) with normalized local tree std (0.5)
                    local_scale = 0.5 + 0.5 * np.clip(tree_std / mean_std, 0.3, 2.5)

            if mapie_j is not None and hasattr(mapie_j, "predict_interval"):
                pred_j, pis_j = mapie_j.predict_interval(X_arr)
                half_width = 0.5 * (pis_j[:, 1, 0] - pis_j[:, 0, 0]) * local_scale
                y_pred[:, j] = pred_j
                y_low[:, j] = pred_j - half_width
                y_high[:, j] = pred_j + half_width
            elif mapie_j is not None and hasattr(mapie_j, "predict"):
                pred_j, pis_j = mapie_j.predict(X_arr, alpha=alpha_eff)
                half_width = 0.5 * (pis_j[:, 1, 0] - pis_j[:, 0, 0]) * local_scale
                y_pred[:, j] = pred_j
                y_low[:, j] = pred_j - half_width
                y_high[:, j] = pred_j + half_width
            else:
                pred_j = est_j.predict(X_arr)
                half_width = q_j * local_scale
                y_pred[:, j] = pred_j
                y_low[:, j] = pred_j - half_width
                y_high[:, j] = pred_j + half_width

        return y_pred, y_low, y_high

    def predict_interval(self, X, alpha: Optional[float] = None):
        """Return (y_pred, y_p10, y_p90) with physical constraints applied."""
        y_pred, y_low, y_high = self._raw_predict_and_intervals(X, alpha=alpha)

        if self.enforce_physics and getattr(self, "target_names_", None):
            combined = pd.DataFrame(y_pred, columns=self.target_names_)
            for idx, col in enumerate(self.target_names_):
                combined[f"{col}_P10"] = y_low[:, idx]
                combined[f"{col}_P90"] = y_high[:, idx]
            if isinstance(X, pd.DataFrame):
                for ctx_col in ("TIGHT_FLAG", "COAL_FLAG"):
                    if ctx_col in X.columns and ctx_col not in combined.columns:
                        combined[ctx_col] = X[ctx_col].to_numpy()
            constrained = enforce_physical_constraints(combined)
            y_pred = constrained[self.target_names_].to_numpy(dtype=float)
            y_low = constrained[[f"{c}_P10" for c in self.target_names_]].to_numpy(
                dtype=float
            )
            y_high = constrained[[f"{c}_P90" for c in self.target_names_]].to_numpy(
                dtype=float
            )

        if getattr(self, "is_1d_", False):
            return y_pred.ravel(), y_low.ravel(), y_high.ravel()
        return y_pred, y_low, y_high

    def predict(self, X):
        """Return point predictions (with physical constraints if enabled)."""
        y_pred, _, _ = self.predict_interval(X)
        return y_pred


def unwrap_sklearn_model(model):
    """Unwrap an MLflow PyFuncModel to access the underlying scikit-learn / Conformal estimator."""
    impl = getattr(model, "_model_impl", None)
    if impl is not None:
        sk_model = getattr(impl, "sklearn_model", None)
        if sk_model is not None:
            return sk_model
        return impl
    return model


def predict_with_uncertainty(
    model,
    X: pd.DataFrame,
    targets: Optional[list[str]] = None,
    alpha: float = CONFORMAL_ALPHA,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Run predictions and extract P10–P90 uncertainty intervals when supported by the model.

    Supports:
        1. `ConformalMultiOutputRegressor` (directly or wrapped inside MLflow PyFuncModel)
        2. MAPIE regressors (`predict_interval`)
        3. Scikit-learn tree ensembles (`RandomForestRegressor`, `ExtraTreesRegressor`) via tree quantiles
        4. Classifiers (returns `(preds, None, None)`)

    Args:
        model (object): Trained model or MLflow PyFuncModel.
        X (pd.DataFrame): Feature DataFrame.
        targets (list[str], optional): Target column names.
        alpha (float, optional): Significance level (0.20 -> P10 to P90). Defaults to CONFORMAL_ALPHA.

    Returns:
        tuple: `(y_pred, y_p10, y_p90)` where `y_p10` and `y_p90` are None for classifiers.
    """
    raw_model = unwrap_sklearn_model(model)

    # Classifier check: do not generate continuous P10/P90 bands for discrete flags
    if hasattr(raw_model, "predict_proba") or (
        targets and all(t.endswith("_FLAG") for t in targets)
    ):
        preds = model.predict(X)
        return np.asarray(preds), None, None

    if hasattr(raw_model, "predict_interval"):
        y_pred, y_p10, y_p90 = raw_model.predict_interval(X, alpha=alpha)
        y_pred = np.asarray(y_pred, dtype=float)
        y_p10 = np.asarray(y_p10, dtype=float)
        y_p90 = np.asarray(y_p90, dtype=float)
        if y_pred.ndim == 1:
            y_pred = y_pred.reshape(-1, 1)
            y_p10 = y_p10.reshape(-1, 1)
            y_p90 = y_p90.reshape(-1, 1)
        return y_pred, y_p10, y_p90

    # Fallback for legacy RandomForestRegressor models without ConformalMultiOutputRegressor wrapper
    preds = np.asarray(model.predict(X), dtype=float)
    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    if hasattr(raw_model, "estimators_") and len(raw_model.estimators_) > 1:
        X_arr = X.to_numpy(dtype=float) if isinstance(X, pd.DataFrame) else np.asarray(X, dtype=float)
        tree_preds = np.stack([t.predict(X_arr) for t in raw_model.estimators_], axis=0)
        if tree_preds.ndim == 2:
            tree_preds = tree_preds[:, :, np.newaxis]
        y_p10 = np.percentile(tree_preds, 100.0 * (alpha / 2.0), axis=0)
        y_p90 = np.percentile(tree_preds, 100.0 * (1.0 - alpha / 2.0), axis=0)
        y_p10 = np.minimum(y_p10, preds)
        y_p90 = np.maximum(y_p90, preds)
        return preds, y_p10, y_p90

    # Generic fallback if no ensemble/conformal metadata is present
    spread = np.maximum(np.abs(preds) * 0.10, 0.02)
    return preds, preds - spread, preds + spread

