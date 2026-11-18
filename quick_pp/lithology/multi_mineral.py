from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize, nnls
from tqdm import tqdm

from quick_pp import logger
from quick_pp.config import Config

CONDUCTIVITY_MODELS = ("sandsiltclay_waxman_smits", "waxman_smits", "dual_water")
CLAY_MINERAL_NAMES = (
    "SHALE",
    "CLAY",
    "KAOLINITE",
    "ILLITE",
    "SMECTITE",
    "CHLORITE",
)
DEFAULT_LOG_RANGES = {
    "GR": 100.0,
    "DTC": 100.0,
    "NPHI": 0.40,
    "RHOB": 0.80,
    "PEF": 5.0,
    "U": 15.0,
    "POTA": 5.0,
    "THOR": 20.0,
    "URAN": 10.0,
}
DEFAULT_TOOL_UNCERTAINTIES = {
    "RHOB": 0.015,
    "NPHI": 0.015,
    "DTC": 3.0,
    "PEF": 0.15,
    "U": 0.40,
    "GR": 6.0,
    "POTA": 0.25,
    "THOR": 1.5,
    "URAN": 1.0,
    "RT": 0.05,
    "CT": 0.05,
    "PHIT_GUIDE": 0.02,
}
DEFAULT_FLUID_PROPERTIES = {
    "OIL": {
        "density": 0.80,
        "nphi": 0.95,
        "pef": 0.12,
        "dtc": 200.0,
        "u": 0.10,
        "gr": 0.0,
    },
    "GAS": {
        "density": 0.20,
        "nphi": 0.25,
        "pef": 0.05,
        "dtc": 400.0,
        "u": 0.01,
        "gr": 0.0,
    },
    "WATER": {
        "density": 1.0,
        "nphi": 1.0,
        "pef": 0.36,
        "dtc": 189.0,
        "u": 0.36,
        "gr": 0.0,
    },
}
STAGE1_FLUID_PROPS = {
    "density": 1.0,
    "nphi": 1.0,
    "pef": 0.36,
    "dtc": 189.0,
    "u": 0.36,
    "gr": 0.0,
}


def _project_bounded_simplex_batch(
    V: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_sum: float = 1.0,
    n_iter: int = 25,
) -> np.ndarray:
    """Vectorized projection of rows of V onto {v : sum(v) = target_sum, lower <= v <= upper}.

    Uses vectorized bisection on the Lagrange multiplier tau such that
    sum_j clip(V_{i,j} - tau_i, lower_j, upper_j) = target_sum.
    """
    V_arr = np.asarray(V, dtype=float)
    squeeze = V_arr.ndim == 1
    if squeeze:
        V_arr = V_arr[None, :]

    low = np.asarray(lower, dtype=float)[None, :]
    high = np.asarray(upper, dtype=float)[None, :]

    # Bracket tau_i so that tau_lo gives sum >= target_sum and tau_hi gives sum <= target_sum
    tau_lo = np.min(V_arr - high, axis=1, keepdims=True) - 1.0
    tau_hi = np.max(V_arr - low, axis=1, keepdims=True) + 1.0

    for _ in range(n_iter):
        tau_mid = 0.5 * (tau_lo + tau_hi)
        clipped = np.clip(V_arr - tau_mid, low, high)
        row_sum = np.sum(clipped, axis=1, keepdims=True)
        too_high = row_sum > target_sum
        tau_lo = np.where(too_high, tau_mid, tau_lo)
        tau_hi = np.where(too_high, tau_hi, tau_mid)

    tau = 0.5 * (tau_lo + tau_hi)
    projected = np.clip(V_arr - tau, low, high)
    return projected[0] if squeeze else projected


def solve_bounded_least_squares_batch(
    Cw: np.ndarray,
    Yw: np.ndarray,
    V0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    reg: float = 1e-7,
    tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized primal-dual Active-Set Bounded Least Squares solver on the simplex.

    Solves for all N depth samples simultaneously:
        min_V  0.5 * ||Cw @ V - y_i||_2^2 + 0.5 * eps * ||V - V0_i||_2^2
        s.t.   sum(V) = 1.0,  lower <= V <= upper

    Args:
        Cw (np.ndarray): Weighted response matrix of shape (m, p) or (N, m, p).
        Yw (np.ndarray): Weighted measured log matrix of shape (N, m).
        V0 (np.ndarray): Initial volume guesses of shape (N, p).
        lower (np.ndarray): Lower bound vector of shape (p,).
        upper (np.ndarray): Upper bound vector of shape (p,).
        reg (float, optional): Tikhonov regularization for rank-deficient systems. Defaults to 1e-7.
        tol (float, optional): KKT optimality and feasibility tolerance. Defaults to 1e-9.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Optimized volumes (N, p) and boolean success flags (N,).
    """
    N, p = V0.shape
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    if N == 0:
        return np.empty((0, p), dtype=float), np.empty(0, dtype=bool)

    is_3d = Cw.ndim == 3
    if not is_3d:
        ctc = Cw.T @ Cw
        scale = max(float(np.trace(ctc)) / max(p, 1), 1.0)
        eps = reg * scale
        H = ctc + eps * np.eye(p)
        G = Yw @ Cw + eps * V0

        K_base = np.zeros((p + 1, p + 1), dtype=float)
        K_base[:p, :p] = H
        K_base[:p, p] = 1.0
        K_base[p, :p] = 1.0
        K_base[p, p] = -1e-12
    else:
        ctc = np.einsum("nmp,nmq->npq", Cw, Cw)
        traces = np.trace(ctc, axis1=1, axis2=2) / max(p, 1)
        eps = reg * np.maximum(traces, 1.0)
        H_batch = ctc + eps[:, None, None] * np.eye(p)[None, :, :]
        G = np.einsum("nm,nmp->np", Yw, Cw) + eps[:, None] * V0

    V = _project_bounded_simplex_batch(V0, lower, upper, target_sum=1.0)
    state = np.zeros((N, p), dtype=np.int8)
    state[V <= lower[None, :] + tol] = -1
    state[V >= upper[None, :] - tol] = 1

    # Guarantee at least one free variable per sample
    all_clamped = np.all(state != 0, axis=1)
    if np.any(all_clamped):
        state[all_clamped, 0] = 0

    active_idx = np.arange(N)
    max_iter = max(4 * p, 12)

    for _ in range(max_iter):
        M = len(active_idx)
        if M == 0:
            break

        st = state[active_idx]
        free = st == 0
        v_fix = np.where(
            st == -1, lower[None, :], np.where(st == 1, upper[None, :], 0.0)
        )

        if not is_3d:
            g_red = G[active_idx] - v_fix @ H.T
            K = np.broadcast_to(K_base, (M, p + 1, p + 1)).copy()
        else:
            H_act = H_batch[active_idx]
            g_red = G[active_idx] - np.einsum("mp,mpq->mq", v_fix, H_act)
            K = np.zeros((M, p + 1, p + 1), dtype=float)
            K[:, :p, :p] = H_act
            K[:, :p, p] = 1.0
            K[:, p, :p] = 1.0
            K[:, p, p] = -1e-12

        rem_sum = 1.0 - np.sum(v_fix, axis=1)

        for j in range(p):
            cj = ~free[:, j]
            if np.any(cj):
                K[cj, j, :] = 0.0
                K[cj, :, j] = 0.0
                K[cj, j, j] = 1.0

        rhs = np.empty((M, p + 1), dtype=float)
        rhs[:, :p] = np.where(free, g_red, v_fix)
        rhs[:, p] = rem_sum

        try:
            sol = np.linalg.solve(K, rhs)
        except np.linalg.LinAlgError:
            K[:, :p, :p] += 1e-8 * np.eye(p)[None, :, :]
            sol = np.linalg.solve(K, rhs)

        V_cand = sol[:, :p]
        lam = sol[:, p]

        viol_low = free & (V_cand < lower[None, :] - tol)
        viol_high = free & (V_cand > upper[None, :] + tol)
        infeasible = np.any(viol_low | viol_high, axis=1)

        still_active = np.ones(M, dtype=bool)

        if np.any(infeasible):
            inf_rows = np.where(infeasible)[0]
            g_inf = active_idx[inf_rows]
            V_curr_inf = V[g_inf]
            V_cand_inf = V_cand[inf_rows]
            d = V_cand_inf - V_curr_inf
            vl = viol_low[inf_rows]
            vh = viol_high[inf_rows]
            low_b = np.broadcast_to(lower, V_curr_inf.shape)
            high_b = np.broadcast_to(upper, V_curr_inf.shape)

            alpha = np.where(
                vl,
                (low_b - V_curr_inf) / np.where(np.abs(d) > 1e-15, d, -1e-15),
                np.where(
                    vh,
                    (high_b - V_curr_inf) / np.where(np.abs(d) > 1e-15, d, 1e-15),
                    2.0,
                ),
            )
            alpha = np.clip(alpha, 0.0, 2.0)
            min_alpha = np.min(alpha, axis=1, keepdims=True)
            V_new_inf = V_curr_inf + np.minimum(min_alpha, 1.0) * d
            V[g_inf] = np.clip(V_new_inf, lower[None, :], upper[None, :])

            block_j = np.argmin(alpha, axis=1)
            row_arange = np.arange(len(inf_rows))
            is_low_block = vl[row_arange, block_j]
            state[g_inf, block_j] = np.where(is_low_block, -1, 1)

            # Ensure we don't clamp all variables in any row
            all_c = np.all(state[g_inf] != 0, axis=1)
            if np.any(all_c):
                release_rows = g_inf[all_c]
                state[
                    release_rows, np.argmax(V[release_rows] - lower[None, :], axis=1)
                ] = 0

        feas_rows = np.where(~infeasible)[0]
        if len(feas_rows) > 0:
            g_feas = active_idx[feas_rows]
            Vf = np.clip(V_cand[feas_rows], lower[None, :], upper[None, :])
            V[g_feas] = Vf
            if not is_3d:
                mu = Vf @ H.T - G[g_feas] + lam[feas_rows, None]
            else:
                mu = (
                    np.einsum("mp,mpq->mq", Vf, H_batch[g_feas])
                    - G[g_feas]
                    + lam[feas_rows, None]
                )
            st_f = state[g_feas]
            dual_viol = np.zeros_like(mu)
            low_mask = st_f == -1
            high_mask = st_f == 1
            dual_viol[low_mask] = np.maximum(0.0, -mu[low_mask])
            dual_viol[high_mask] = np.maximum(0.0, mu[high_mask])
            max_viol = np.max(dual_viol, axis=1)
            conv = max_viol <= tol
            still_active[feas_rows[conv]] = False
            non_conv = feas_rows[~conv]
            if len(non_conv) > 0:
                rel_j = np.argmax(dual_viol[~conv], axis=1)
                state[active_idx[non_conv], rel_j] = 0

        active_idx = active_idx[still_active]

    V = _project_bounded_simplex_batch(V, lower, upper, target_sum=1.0)
    success = (
        np.all(np.isfinite(V), axis=1)
        & (np.abs(np.sum(V, axis=1) - 1.0) <= 1e-4)
        & np.all((V >= lower[None, :] - 1e-5) & (V <= upper[None, :] + 1e-5), axis=1)
    )
    return V, success


def sandsiltclay_waxman_smits_conductivity(
    phit: Union[float, np.ndarray],
    swt: Union[float, np.ndarray],
    vclay: Union[float, np.ndarray],
    rw: float = 0.05,
    B: Optional[float] = None,
    Qv: Optional[float] = None,
    rho_clay: float = 2.65,
    cec_clay: float = 0.062,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> Union[float, np.ndarray]:
    """Calculate reconstructed formation conductivity Ct [S/m] using the Waxman-Smits model.

    Ct = (phit^m / a) * (swt^n / rw + B * Qv * swt^(n - 1))
    """
    phit_safe = np.clip(phit, 1e-4, 1.0)
    swt_safe = np.clip(swt, 1e-4, 1.0)
    vclay_safe = np.clip(vclay, 0.0, 1.0)
    rw_safe = max(float(rw), 1e-5)

    if B is None:
        B = (1.0 - 0.83 * np.exp(-0.5 / rw_safe)) * 3.83
    if Qv is None:
        Qv_val = vclay_safe * rho_clay * cec_clay / phit_safe
    else:
        Qv_val = Qv

    ct = (phit_safe**m / a) * (
        (swt_safe**n) / rw_safe + B * Qv_val * (swt_safe ** (n - 1.0))
    )
    return ct


def dual_water_conductivity(
    phit: Union[float, np.ndarray],
    swt: Union[float, np.ndarray],
    vclay: Union[float, np.ndarray],
    rw: float = 0.05,
    rwb: float = 0.15,
    nphi_sh: float = 0.30,
    a: float = 1.0,
    m: float = 2.0,
    n: float = 2.0,
) -> Union[float, np.ndarray]:
    """Calculate reconstructed formation conductivity Ct [S/m] using the Dual Water model.

    Ct = (phit^m * swt^n / a) * (1 / rw + (swb / swt) * (1 / rwb - 1 / rw))
    """
    phit_safe = np.clip(phit, 1e-4, 1.0)
    swt_safe = np.clip(swt, 1e-4, 1.0)
    vclay_safe = np.clip(vclay, 0.0, 1.0)
    rw_safe = max(float(rw), 1e-5)
    rwb_safe = max(float(rwb), 1e-5)

    swb = np.clip(vclay_safe * nphi_sh / phit_safe, 0.0, swt_safe)
    ct = (phit_safe**m / a) * (
        (swt_safe**n) / rw_safe
        + swb * (swt_safe ** (n - 1.0)) * (1.0 / rwb_safe - 1.0 / rw_safe)
    )
    return ct


class MultiMineral:
    """Optimization-based multi-mineral model for lithology estimation with fluid volumes.

    Uses a fast vectorized Active-Set Bounded Least Squares solver for the linear
    mineral/fluid system (quadratic objective with linear closure sum(V_i) = 1 and
    box bounds 0 <= V_i <= 1), and reserves SLSQP only when non-linear conductivity
    (sandsiltclay_waxman_smits / dual_water) is enabled or explicitly requested.

    Supports dynamic log channel ingestion (including Spectral GR and geochemical curves),
    exact volumetric photoelectric cross-section linearization (U = PEF * RHOB),
    depth-dependent environmental / bad hole weighting, and custom mineral bounds.
    """

    def __init__(
        self,
        minerals: Optional[List[str]] = None,
        fluid_properties: Optional[Dict] = None,
        conductivity_model: Optional[str] = None,
        conductivity_params: Optional[Dict] = None,
        solver: str = "auto",
        n_jobs: int = 1,
        responses: Optional[Dict[str, Dict[str, float]]] = None,
        mineral_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fluid_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        linearize_pef: bool = False,
        fluids: Optional[List[str]] = None,
        two_stage: bool = True,
        phi_max: float = 0.40,
        tool_uncertainties: Optional[Dict[str, float]] = None,
    ):
        """Initialize the MultiMineral optimizer.

        Args:
            minerals (list, optional): A list of minerals for the optimization.
                                       Defaults to ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE'].
            fluid_properties (dict, optional): Properties for oil, gas, and water. Defaults to standard values.
            conductivity_model (str, optional): Optional non-linear conductivity model
                                                ('sandsiltclay_waxman_smits', 'waxman_smits', 'dual_water', or None).
                                                When enabled, SLSQP is used to solve the non-linear conductivity term.
            conductivity_params (dict, optional): Parameters for the non-linear conductivity model
                                                  (e.g., rw, rwb, B, Qv, a, m, n, nphi_sh).
            solver (str, optional): Optimization solver ('auto', 'active_set', 'bounded_lsq', 'slsqp').
                                    Defaults to 'auto' (vectorized active-set for linear, SLSQP for non-linear).
            n_jobs (int, optional): Number of parallel jobs via joblib.Parallel for batching. Defaults to 1.
            responses (dict, optional): Custom mineral response dictionary overriding or extending
                                        Config.MINERALS_LOG_VALUE.
            mineral_bounds (dict, optional): Custom (min, max) bounds per mineral name (e.g. {'GLAUCONITE': (0, 0.25)}).
            fluid_bounds (dict, optional): Custom (min, max) bounds per fluid name (e.g. {'GAS': (0, 0.4)}).
            linearize_pef (bool, optional): If True, linearizes Photoelectric Factor via volumetric cross-section
                                            U = PEF * RHOB (barns/cm^3) to ensure exact physical volumetric superposition.
                                            Defaults to False.
            fluids (list, optional): List of fluid names. Defaults to ['OIL', 'GAS', 'WATER'].
            two_stage (bool, optional): If True, runs decoupled two-stage solver (Stage 1 linear lithology + total
                                        porosity on the simplex; Stage 2 non-linear fluid saturation & partitioning).
                                        Defaults to True.
            phi_max (float, optional): Maximum physical porosity ceiling for the simplex. Defaults to 0.40.
            tool_uncertainties (dict, optional): Measurement uncertainties (sigma_k) per tool channel for chi2 weighting.
        """
        # Load mineral endpoints (merging defaults with any user-supplied overrides)
        base_responses = {k: dict(v) for k, v in Config.MINERALS_LOG_VALUE.items()}
        if responses is not None:
            for m_name, m_props in responses.items():
                base_responses[m_name] = dict(m_props)
        self.responses = base_responses
        available_minerals = list(self.responses.keys())

        # Default minerals if none specified
        if minerals is None:
            minerals = ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"]

        for mineral in minerals:
            if mineral not in available_minerals:
                raise ValueError(
                    f"Mineral '{mineral}' is not defined in Config.MINERALS_LOG_VALUE or provided responses."
                )

        self.minerals = minerals
        self.conductivity_model = (
            conductivity_model.lower() if conductivity_model is not None else None
        )
        if (
            self.conductivity_model is not None
            and self.conductivity_model not in CONDUCTIVITY_MODELS
        ):
            raise ValueError(
                f"Unknown conductivity_model '{conductivity_model}'. "
                f"Choose from {CONDUCTIVITY_MODELS} or None."
            )
        self.conductivity_params = conductivity_params or {}
        self.solver = solver.lower()
        self.n_jobs = n_jobs
        self.linearize_pef = bool(linearize_pef)
        self.two_stage = bool(two_stage)
        self.phi_max = float(phi_max)

        self.tool_uncertainties = dict(DEFAULT_TOOL_UNCERTAINTIES)
        if tool_uncertainties is not None:
            self.tool_uncertainties.update(tool_uncertainties)

        # Default fluid properties if none specified
        if fluid_properties is None:
            fluid_properties = {k: dict(v) for k, v in DEFAULT_FLUID_PROPERTIES.items()}
        self.fluid_properties = fluid_properties
        self.fluids = list(fluids) if fluids is not None else ["OIL", "GAS", "WATER"]

        # Define bounds for each mineral type (defaulting to (0, 1), overridden by mineral_bounds)
        default_mineral_bounds = dict.fromkeys(available_minerals, (0.0, 1.0))
        if mineral_bounds is not None:
            default_mineral_bounds.update(mineral_bounds)

        # Define bounds for fluid volumes (0 to 1 for each fluid, overridden by fluid_bounds)
        default_fluid_bounds = {
            "OIL": (0.0, 1.0),
            "GAS": (0.0, 1.0),
            "WATER": (0.0, 1.0),
        }
        if fluid_bounds is not None:
            default_fluid_bounds.update(fluid_bounds)

        # Set bounds based on selected minerals and fluids
        self.mineral_bounds = tuple(
            default_mineral_bounds[mineral] for mineral in self.minerals
        )
        self.fluid_bounds = tuple(default_fluid_bounds[fluid] for fluid in self.fluids)

        # Combined bounds: minerals first, then fluids
        self.bounds = self.mineral_bounds + self.fluid_bounds

        # Constraint: sum of mineral volumes + fluid volumes must equal 1
        self.constraints = [{"type": "eq", "fun": self._volume_constraint}]
        self.scaling_factors: Dict[str, float] = {}

    def _has_nonlinear_conductivity(
        self,
        rt: Optional[Union[float, np.ndarray]] = None,
        conductivity_model: Optional[str] = None,
    ) -> bool:
        """Return True if non-linear conductivity modeling (Waxman-Smits / Dual Water) is active."""
        model = (
            conductivity_model.lower()
            if conductivity_model is not None
            else self.conductivity_model
        )
        if model in CONDUCTIVITY_MODELS:
            return True
        return False

    def _volume_constraint(self, volumes: np.ndarray) -> float:
        """Ensure that the sum of all mineral and fluid volumes equals 1."""
        return float(np.sum(volumes) - 1.0)

    def _build_response_matrix(
        self, log_types: List[str], fluids: Optional[List[str]] = None
    ) -> np.ndarray:
        """Construct the linear log response matrix C of shape (n_logs, n_vars).

        Supports volumetric photoelectric cross-section U (barns/cm^3) for exact
        linear superposition, as well as dynamic Spectral GR / geochemical channels
        and total porosity guide (PHIT_GUIDE).
        """
        fluids_active = fluids if fluids is not None else self.fluids
        n_minerals = len(self.minerals)
        n_fluids = len(fluids_active)
        n_vars = n_minerals + n_fluids
        C = np.zeros((len(log_types), n_vars), dtype=float)

        fluid_key_map = {
            "NPHI": "nphi",
            "RHOB": "density",
            "DTC": "dtc",
            "PEF": "pef",
        }

        for r_idx, log_type in enumerate(log_types):
            if log_type == "PHIT_GUIDE":
                # Porosity guide responds 0.0 to matrix minerals and 1.0 to fluid pore space
                C[r_idx, n_minerals:] = 1.0
                continue

            for m_idx, mineral in enumerate(self.minerals):
                if mineral in self.responses:
                    m_dict = self.responses[mineral]
                    if log_type == "U":
                        pef_m = float(m_dict.get("PEF", 0.0))
                        rho_m = float(m_dict.get("RHOB", 2.65))
                        C[r_idx, m_idx] = float(m_dict.get("U", pef_m * rho_m))
                    elif log_type in m_dict:
                        C[r_idx, m_idx] = float(m_dict[log_type])

            if log_type == "U":
                for f_idx, fluid in enumerate(fluids_active):
                    f_dict = self.fluid_properties.get(
                        fluid, self.fluid_properties.get("WATER", STAGE1_FLUID_PROPS)
                    )
                    pef_f = float(f_dict.get("pef", 0.36))
                    rho_f = float(f_dict.get("density", 1.0))
                    C[r_idx, n_minerals + f_idx] = float(f_dict.get("u", pef_f * rho_f))
            else:
                prop_key = fluid_key_map.get(log_type, log_type.lower())
                for f_idx, fluid in enumerate(fluids_active):
                    f_dict = self.fluid_properties.get(
                        fluid, self.fluid_properties.get("WATER", STAGE1_FLUID_PROPS)
                    )
                    if prop_key in f_dict:
                        C[r_idx, n_minerals + f_idx] = float(f_dict[prop_key])
                    elif prop_key == "nphi":
                        C[r_idx, n_minerals + f_idx] = 1.0 if fluid == "WATER" else 0.0
                    elif prop_key == "density":
                        C[r_idx, n_minerals + f_idx] = 1.0 if fluid == "WATER" else 0.8
        return C

    def _calculate_conductivity_response(self, volumes: np.ndarray) -> float:
        """Calculate reconstructed formation conductivity Ct [S/m] from mineral and fluid volumes."""
        n_minerals = len(self.minerals)
        mineral_vols = volumes[:n_minerals]
        fluid_vols = volumes[n_minerals:]

        phit = float(np.clip(np.sum(fluid_vols), 1e-4, 1.0))
        water_idx = self.fluids.index("WATER") if "WATER" in self.fluids else -1
        vwater = float(fluid_vols[water_idx]) if water_idx >= 0 else 0.0
        swt = float(np.clip(vwater / phit, 1e-4, 1.0))

        vclay = sum(
            float(mineral_vols[i])
            for i, m in enumerate(self.minerals)
            if m.upper() in CLAY_MINERAL_NAMES
        )

        params = self.conductivity_params
        model = self.conductivity_model or "sandsiltclay_waxman_smits"
        if model in ("sandsiltclay_waxman_smits", "waxman_smits"):
            return float(
                sandsiltclay_waxman_smits_conductivity(
                    phit=phit,
                    swt=swt,
                    vclay=vclay,
                    rw=params.get("rw", 0.05),
                    B=params.get("B", None),
                    Qv=params.get("Qv", None),
                    rho_clay=params.get("rho_clay", 2.65),
                    cec_clay=params.get("cec_clay", 0.062),
                    a=params.get("a", 1.0),
                    m=params.get("m", 2.0),
                    n=params.get("n", 2.0),
                )
            )
        elif model == "dual_water":
            return float(
                dual_water_conductivity(
                    phit=phit,
                    swt=swt,
                    vclay=vclay,
                    rw=params.get("rw", 0.05),
                    rwb=params.get("rwb", 0.15),
                    nphi_sh=params.get("nphi_sh", 0.30),
                    a=params.get("a", 1.0),
                    m=params.get("m", 2.0),
                    n=params.get("n", 2.0),
                )
            )
        raise ValueError(f"Unsupported conductivity model: {model}")

    def _calculate_log_response(
        self, volumes: np.ndarray, log_type: str, fluids: Optional[List[str]] = None
    ) -> float:
        """Calculate the reconstructed log response for a given log type."""
        if log_type in ("RT", "CT") and self.conductivity_model is not None:
            ct = self._calculate_conductivity_response(volumes)
            return float(1.0 / max(ct, 1e-6)) if log_type == "RT" else float(ct)

        if log_type == "PEF" and self.linearize_pef:
            u_recon = self._calculate_log_response(volumes, "U", fluids=fluids)
            rho_recon = self._calculate_log_response(volumes, "RHOB", fluids=fluids)
            return float(u_recon / rho_recon) if rho_recon > 1e-4 else 0.0

        if log_type == "PHIT_GUIDE":
            n_min = len(self.minerals)
            return float(np.sum(volumes[n_min:])) if len(volumes) > n_min else 0.0

        response = 0.0

        # Add mineral contributions
        n_minerals = len(self.minerals)
        for i, mineral in enumerate(self.minerals):
            if mineral in self.responses:
                m_dict = self.responses[mineral]
                if log_type == "U":
                    pef_m = float(m_dict.get("PEF", 0.0))
                    rho_m = float(m_dict.get("RHOB", 2.65))
                    response += volumes[i] * float(m_dict.get("U", pef_m * rho_m))
                elif log_type in m_dict:
                    response += volumes[i] * float(m_dict[log_type])

        # Add fluid contributions
        active_fluids = (
            fluids
            if fluids is not None
            else (
                self.fluids
                if len(volumes) == n_minerals + len(self.fluids)
                else ["WATER"]
            )
        )
        for i, fluid in enumerate(active_fluids):
            fluid_idx = n_minerals + i
            if fluid_idx >= len(volumes):
                break
            fluid_vol = volumes[fluid_idx]
            props = self.fluid_properties.get(
                fluid, self.fluid_properties.get("WATER", STAGE1_FLUID_PROPS)
            )

            if log_type == "NPHI":
                response += fluid_vol * float(props.get("nphi", 1.0 if fluid == "WATER" else 0.0))
            elif log_type == "RHOB":
                response += fluid_vol * float(props.get("density", 1.0 if fluid == "WATER" else 0.8))
            elif log_type == "DTC":
                response += fluid_vol * float(props.get("dtc", 189.0))
            elif log_type == "PEF":
                response += fluid_vol * float(props.get("pef", 0.36 if fluid == "WATER" else 0.0))
            elif log_type == "U":
                pef_f = float(props.get("pef", 0.36 if fluid == "WATER" else 0.0))
                rho_f = float(props.get("density", 1.0 if fluid == "WATER" else 0.8))
                response += fluid_vol * float(props.get("u", pef_f * rho_f))
            elif log_type.lower() in props:
                response += fluid_vol * float(props[log_type.lower()])

        return float(response)

    def _calculate_average_error(
        self, volumes: np.ndarray, log_values: Dict[str, float]
    ) -> float:
        """Calculate the average absolute error between measured and reconstructed logs."""
        total_error = 0.0
        valid_count = 0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                if log_type in ("RT", "CT") and self.conductivity_model is None:
                    continue
                reconstructed_value = self._calculate_log_response(volumes, log_type)
                error = abs(measured_value - reconstructed_value)
                total_error += error
                valid_count += 1

        return float(total_error / valid_count) if valid_count > 0 else np.nan

    def _error_function(
        self, volumes: np.ndarray, log_values: Dict[str, float]
    ) -> float:
        """Calculate the sum of squared errors for the optimization."""
        total_error = 0.0

        for log_type, measured_value in log_values.items():
            if measured_value is not None and not np.isnan(measured_value):
                if log_type in ("RT", "CT"):
                    if self.conductivity_model is None:
                        continue
                    ct_recon = self._calculate_conductivity_response(volumes)
                    ct_meas = (
                        1.0 / max(float(measured_value), 1e-6)
                        if log_type == "RT"
                        else float(measured_value)
                    )
                    sigma_rt = self.tool_uncertainties.get("RT", 0.05)
                    diff_log = np.log10(max(ct_meas, 1e-6)) - np.log10(max(ct_recon, 1e-6))
                    total_error += (diff_log / sigma_rt) ** 2
                    continue

                reconstructed_value = self._calculate_log_response(volumes, log_type)

                scaling = self.scaling_factors.get(log_type, 1.0)
                error = (scaling * (measured_value - reconstructed_value)) ** 2
                total_error += error

        return float(total_error)

    def _optimize_volumes(
        self,
        log_values: Dict[str, float],
        method: Optional[str] = None,
        initial_guess: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Perform the optimization to find the best-fit mineral and fluid volumes for a single depth point."""
        valid_logs = {
            k: float(v)
            for k, v in log_values.items()
            if v is not None and not np.isnan(v)
        }

        if not valid_logs:
            raise ValueError("No valid log measurements provided")

        n_minerals = len(self.minerals)
        n_fluids = len(self.fluids)

        if initial_guess is None:
            n_vars = n_minerals + n_fluids
            initial_guess = _project_bounded_simplex_batch(
                np.full(n_vars, 1.0 / n_vars),
                np.array([b[0] for b in self.bounds], dtype=float),
                np.array([b[1] for b in self.bounds], dtype=float),
            )

        solver_choice = (method or self.solver or "auto").lower()
        use_nonlinear = solver_choice == "slsqp" or (
            self.conductivity_model in CONDUCTIVITY_MODELS
            and ("RT" in valid_logs or "CT" in valid_logs)
        )

        if not use_nonlinear:
            linear_keys = [k for k in valid_logs.keys() if k not in ("RT", "CT")]
            solve_keys = list(linear_keys)
            solve_vals = [valid_logs[k] for k in linear_keys]

            if self.linearize_pef and "PEF" in linear_keys and "RHOB" in valid_logs:
                pef_idx = solve_keys.index("PEF")
                solve_keys[pef_idx] = "U"
                solve_vals[pef_idx] = valid_logs["PEF"] * valid_logs["RHOB"]

            C = self._build_response_matrix(solve_keys)
            weights = np.array(
                [self.scaling_factors.get(k, 1.0) for k in solve_keys], dtype=float
            )
            Cw = weights[:, None] * C
            Yw = (weights * np.array(solve_vals, dtype=float))[None, :]
            lower = np.array([b[0] for b in self.bounds], dtype=float)
            upper = np.array([b[1] for b in self.bounds], dtype=float)
            V_opt, success_arr = solve_bounded_least_squares_batch(
                Cw, Yw, initial_guess[None, :], lower, upper
            )
            x_opt = V_opt[0]
            return x_opt[:n_minerals], x_opt[n_minerals:], bool(success_arr[0])

        result = minimize(
            self._error_function,
            initial_guess,
            args=(valid_logs,),
            bounds=self.bounds,
            constraints=self.constraints,
            method="SLSQP",
        )

        x_proj = _project_bounded_simplex_batch(
            result.x,
            np.array([b[0] for b in self.bounds], dtype=float),
            np.array([b[1] for b in self.bounds], dtype=float),
        )
        mineral_volumes = x_proj[:n_minerals]
        fluid_volumes = x_proj[n_minerals:]

        return mineral_volumes, fluid_volumes, bool(result.success)

    def optimize(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        bounds: Optional[Tuple[Tuple[float, float], ...]] = None,
        conductivity_model: Optional[str] = None,
        rt: Optional[np.ndarray] = None,
        method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        **kwargs,
    ) -> np.ndarray:
        """Vectorized multi-mineral optimizer for linear systems A * V = b or non-linear conductivity systems."""
        if A is None and hasattr(self, "A"):
            A = self.A
        if b is None and hasattr(self, "b"):
            b = self.b

        if A is None:
            raise ValueError(
                "Matrix A (or measured log array) must be provided to optimize()."
            )

        A_arr = np.asarray(A, dtype=float)
        transposed_output = False

        if b is None:
            log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC"][: A_arr.shape[0]]
            C_mat = self._build_response_matrix(log_names)
            Y_mat = A_arr.T
        else:
            b_arr = np.asarray(b, dtype=float)
            if b_arr.ndim == 1:
                b_arr = b_arr[:, None]

            if A_arr.ndim == 2:
                if b_arr.shape[0] == A_arr.shape[0]:
                    C_mat = A_arr
                    Y_mat = b_arr.T
                elif b_arr.shape[1] == A_arr.shape[0]:
                    C_mat = A_arr
                    Y_mat = b_arr
                    transposed_output = True
                elif A_arr.shape[1] == b_arr.shape[1] and b_arr.ndim == 2:
                    C_mat = b_arr
                    Y_mat = A_arr.T
                else:
                    raise ValueError(
                        f"Incompatible shapes for A {A_arr.shape} and b {b_arr.shape}."
                    )
            elif A_arr.ndim == 3:
                if (
                    b_arr.shape[0] == A_arr.shape[0]
                    and b_arr.shape[1] == A_arr.shape[1]
                ):
                    C_mat = np.transpose(A_arr, (1, 0, 2))
                    Y_mat = b_arr.T
                elif (
                    b_arr.shape[0] == A_arr.shape[0]
                    and b_arr.shape[1] == A_arr.shape[2]
                ):
                    C_mat = np.transpose(A_arr, (2, 0, 1))
                    Y_mat = b_arr.T
                elif (
                    A_arr.shape[0] == b_arr.shape[0]
                    and A_arr.shape[1] == b_arr.shape[1]
                ):
                    C_mat = A_arr
                    Y_mat = b_arr
                    transposed_output = True
                else:
                    raise ValueError(
                        f"Incompatible 3D shape for A {A_arr.shape} and b {b_arr.shape}."
                    )
            else:
                raise ValueError(
                    f"Unsupported number of dimensions for A: {A_arr.ndim}"
                )

        N, m = Y_mat.shape
        p = C_mat.shape[-1]

        if bounds is not None:
            lower = np.array([bd[0] for bd in bounds], dtype=float)
            upper = np.array([bd[1] for bd in bounds], dtype=float)
        elif len(self.bounds) == p:
            lower = np.array([bd[0] for bd in self.bounds], dtype=float)
            upper = np.array([bd[1] for bd in self.bounds], dtype=float)
        else:
            lower = np.zeros(p, dtype=float)
            upper = np.ones(p, dtype=float)

        V0 = _project_bounded_simplex_batch(
            np.full((N, p), 1.0 / p, dtype=float), lower, upper
        )

        cond_model = (
            conductivity_model.lower()
            if conductivity_model is not None
            else self.conductivity_model
        )
        solver_choice = (method or self.solver or "auto").lower()
        workers = n_jobs if n_jobs is not None else self.n_jobs
        use_nonlinear = solver_choice == "slsqp" or cond_model in CONDUCTIVITY_MODELS

        V_out = np.full((N, p), np.nan, dtype=float)
        valid_row = np.sum(np.isfinite(Y_mat), axis=1) >= min(3, m)
        if not np.any(valid_row):
            return V_out if transposed_output else V_out.T

        valid_indices = np.where(valid_row)[0]
        Y_valid = np.where(np.isfinite(Y_mat[valid_indices]), Y_mat[valid_indices], 0.0)
        mask_valid = np.isfinite(Y_mat[valid_indices]).astype(float)

        if C_mat.ndim == 2:
            patterns, inv = np.unique(mask_valid, axis=0, return_inverse=True)

            def _solve_pattern(pat_idx: int) -> Tuple[np.ndarray, np.ndarray]:
                rows_p = np.where(inv == pat_idx)[0]
                col_mask = patterns[pat_idx] > 0.5
                Cw_sub = C_mat[col_mask, :]
                Yw_sub = Y_valid[np.ix_(rows_p, col_mask)]
                V_sub, _ = solve_bounded_least_squares_batch(
                    Cw_sub, Yw_sub, V0[valid_indices[rows_p]], lower, upper
                )
                return valid_indices[rows_p], V_sub

            if workers != 1 and len(patterns) > 1:
                results = Parallel(n_jobs=workers, prefer="threads")(
                    delayed(_solve_pattern)(pi) for pi in range(len(patterns))
                )
            elif workers != 1 and len(valid_indices) > 256:
                chunks = np.array_split(valid_indices, max(2, abs(workers) * 2))
                results = Parallel(n_jobs=workers, prefer="threads")(
                    delayed(
                        lambda idx_c: (
                            idx_c,
                            solve_bounded_least_squares_batch(
                                C_mat,
                                np.where(np.isfinite(Y_mat[idx_c]), Y_mat[idx_c], 0.0),
                                V0[idx_c],
                                lower,
                                upper,
                            )[0],
                        )
                    )(c)
                    for c in chunks
                    if len(c) > 0
                )
            else:
                results = [_solve_pattern(pi) for pi in range(len(patterns))]

            for g_rows, V_sub in results:
                V_out[g_rows] = V_sub
        else:
            Cw_3d = C_mat[valid_indices] * mask_valid[:, :, None]
            V_sub, _ = solve_bounded_least_squares_batch(
                Cw_3d, Y_valid, V0[valid_indices], lower, upper
            )
            V_out[valid_indices] = V_sub

        if use_nonlinear:
            rt_arr = (
                np.asarray(rt, dtype=float)
                if rt is not None
                else np.full(N, np.nan, dtype=float)
            )
            bounds_seq = tuple(zip(lower, upper, strict=False))
            constraints_seq = [{"type": "eq", "fun": lambda v: float(np.sum(v) - 1.0)}]

            def _solve_slsqp_row(idx: int) -> np.ndarray:
                y_i = Y_mat[idx]
                f_mask = np.isfinite(y_i)
                C_i = C_mat[f_mask] if C_mat.ndim == 2 else C_mat[idx, f_mask]
                y_f = y_i[f_mask]
                v_init = V_out[idx] if np.all(np.isfinite(V_out[idx])) else V0[idx]
                rt_val = rt_arr[idx] if idx < len(rt_arr) else np.nan
                has_rt = (
                    cond_model in CONDUCTIVITY_MODELS
                    and np.isfinite(rt_val)
                    and rt_val > 0
                )

                n_min = len(self.minerals)

                def _obj(v: np.ndarray) -> float:
                    diff = C_i @ v - y_f
                    err = float(np.dot(diff, diff))
                    if has_rt:
                        if p > n_min:
                            fluid_vols = v[n_min:]
                            phit_v = max(float(np.sum(fluid_vols)), 1e-4)
                            vwater_v = float(fluid_vols[-1])
                        else:
                            phit_v = max(float(v[-1]), 1e-4)
                            vwater_v = phit_v

                        swt_v = float(np.clip(vwater_v / phit_v, 1e-4, 1.0))
                        clay_indices = [
                            m_i
                            for m_i, m_name in enumerate(self.minerals[:p])
                            if m_name.upper() in CLAY_MINERAL_NAMES
                        ]
                        if clay_indices:
                            vclay_v = float(np.sum(v[clay_indices]))
                        else:
                            vclay_v = float(v[min(3, p - 1)]) if p > 3 else 0.0

                        if cond_model in ("sandsiltclay_waxman_smits", "waxman_smits"):
                            ct_hat = float(
                                sandsiltclay_waxman_smits_conductivity(
                                    phit_v, swt_v, vclay_v, **self.conductivity_params
                                )
                            )
                        else:
                            ct_hat = float(
                                dual_water_conductivity(
                                    phit_v, swt_v, vclay_v, **self.conductivity_params
                                )
                            )
                        sigma_rt = self.tool_uncertainties.get("RT", 0.05)
                        diff_log = np.log10(max(ct_hat, 1e-6)) - np.log10(
                            max(1.0 / rt_val, 1e-6)
                        )
                        err += (diff_log / sigma_rt) ** 2
                    return err

                res = minimize(
                    _obj,
                    v_init,
                    bounds=bounds_seq,
                    constraints=constraints_seq,
                    method="SLSQP",
                )
                return _project_bounded_simplex_batch(res.x, lower, upper)

            if workers != 1:
                slsqp_res = Parallel(n_jobs=workers, prefer="threads")(
                    delayed(_solve_slsqp_row)(int(idx)) for idx in valid_indices
                )
            else:
                slsqp_res = [_solve_slsqp_row(int(idx)) for idx in valid_indices]
            V_out[valid_indices] = np.vstack(slsqp_res)

        return V_out if transposed_output else V_out.T

    def estimate_lithology(
        self,
        gr: Optional[np.ndarray] = None,
        nphi: Optional[np.ndarray] = None,
        rhob: Optional[np.ndarray] = None,
        pef: Optional[np.ndarray] = None,
        dtc: Optional[np.ndarray] = None,
        rt: Optional[np.ndarray] = None,
        auto_scale: Union[bool, str] = True,
        conductivity_model: Optional[str] = None,
        method: Optional[str] = None,
        n_jobs: Optional[int] = None,
        logs: Optional[Dict[str, np.ndarray]] = None,
        bounds: Optional[
            Union[Dict[str, Tuple[float, float]], Tuple[Tuple[float, float], ...]]
        ] = None,
        bad_hole_mask: Optional[np.ndarray] = None,
        cali: Optional[np.ndarray] = None,
        bs: Optional[Union[float, np.ndarray]] = None,
        drho: Optional[np.ndarray] = None,
        log_weights: Optional[Dict[str, Union[float, np.ndarray]]] = None,
        bad_hole_weight: float = 0.05,
        linearize_pef: Optional[bool] = None,
        phit: Optional[np.ndarray] = None,
        phit_uncertainty: float = 0.02,
        phi_max: Optional[float] = None,
        two_stage: Optional[bool] = None,
        **extra_logs: np.ndarray,
    ) -> pd.DataFrame:
        """Estimate mineral and fluid volumes for a set of log data points.

        Features:
        - Fast vectorized Bounded Least Squares / Active-Set simplex solver.
        - Two-Stage Architecture: Decouples linear lithology & total porosity inversion
          (Stage 1) from non-linear fluid saturation & hydrocarbon partitioning (Stage 2),
          eliminating algebraic null spaces and preventing phantom porosity inflation.
        - Dynamic log ingestion: supports any combination of standard logs (GR, NPHI, RHOB, PEF, DTC),
          Spectral GR (POTA, THOR, URAN), geochemical channels, and total porosity priors (PHIT).
        - Exact volumetric photoelectric cross-section linearization: U = PEF * RHOB.
        - Borehole quality & bad-hole weighting: de-weights sensitive tools (RHOB, PEF, U)
          in rugose or washed-out intervals using bad_hole_mask, cali/bs, or drho.
        - Chi-squared tool uncertainty weighting and normalized misfit reporting (CHI2_ERROR).

        Args:
            gr (np.ndarray, optional): Gamma Ray log [GAPI].
            nphi (np.ndarray, optional): Neutron Porosity log [v/v].
            rhob (np.ndarray, optional): Bulk Density log [g/cc].
            pef (np.ndarray, optional): Photoelectric Factor log [barns/electron].
            dtc (np.ndarray, optional): Compressional slowness log [us/ft].
            rt (np.ndarray, optional): True Resistivity log [ohm.m] (used when conductivity_model is enabled).
            auto_scale (bool or str, optional): If True or 'uncertainty', scales equations by tool measurement
                                                uncertainties (1 / sigma_k). If 'percentile', scales by 1 / (P95 - P05).
            conductivity_model (str, optional): Non-linear conductivity model override
                                                ('sandsiltclay_waxman_smits', 'dual_water', or None).
            method (str, optional): Solver method override ('auto', 'active_set', 'bounded_lsq', 'slsqp').
            n_jobs (int, optional): Number of parallel jobs via joblib.Parallel. Defaults to self.n_jobs.
            logs (dict, optional): Dictionary of arbitrary input log arrays (e.g. {'POTA': pota, 'THOR': thor}).
            bounds (dict or tuple, optional): Custom bounds per component or full bounds tuple.
            bad_hole_mask (np.ndarray, optional): Boolean array where True indicates bad borehole / washouts.
            cali (np.ndarray, optional): Caliper log [in] used for automatic washout detection.
            bs (float or np.ndarray, optional): Bit size [in] used for washout detection (cali > bs + 1.5).
            drho (np.ndarray, optional): Density correction [g/cc] used for bad-hole detection (|drho| > 0.05).
            log_weights (dict, optional): Custom weighting multipliers per log curve.
            bad_hole_weight (float, optional): Multiplier for density/PEF curves in bad hole. Defaults to 0.05.
            linearize_pef (bool, optional): Override for PEF linearization (U = PEF * RHOB).
            phit (np.ndarray, optional): Pre-computed or core porosity guide curve used as a Gaussian prior.
            phit_uncertainty (float, optional): Uncertainty on the phit guide curve. Defaults to 0.02.
            phi_max (float, optional): Maximum physical porosity ceiling for the simplex. Defaults to self.phi_max.
            two_stage (bool, optional): If True, decouples linear lithology + total porosity from saturation partitioning.
                                        Defaults to self.two_stage (True).
            **extra_logs (np.ndarray): Additional log arrays passed as keyword arguments (e.g., pota=..., thor=...).

        Returns:
            pd.DataFrame: A DataFrame containing estimated mineral/fluid volumes, constructed porosity,
                          residual errors, and reconstructed logs.
        """

        def _to_float_array(
            arr: Optional[Union[float, np.ndarray]], length: int
        ) -> np.ndarray:
            if arr is None:
                return np.full(length, np.nan, dtype=float)
            try:
                out = np.asarray(arr, dtype=float).ravel()
                if len(out) == 1 and length > 1:
                    return np.full(length, out[0], dtype=float)
                return out
            except (ValueError, TypeError):
                return np.array(
                    [np.nan if v is None else float(v) for v in arr], dtype=float
                )

        # 1. Determine sample length from any provided valid input
        candidate_arrays = [gr, nphi, rhob, pef, dtc, rt, cali, drho]
        if phit is not None:
            candidate_arrays.append(phit)
        if logs is not None:
            candidate_arrays.extend(logs.values())
        candidate_arrays.extend(extra_logs.values())

        valid_lens = [
            len(a) for a in candidate_arrays if a is not None and hasattr(a, "__len__")
        ]
        if not valid_lens:
            raise ValueError("No input log arrays provided to estimate_lithology().")
        n_samples = max(valid_lens)

        # 2. Gather all input logs into a normalized dictionary
        raw_inputs: Dict[str, np.ndarray] = {}
        if gr is not None:
            raw_inputs["GR"] = _to_float_array(gr, n_samples)
        if nphi is not None:
            raw_inputs["NPHI"] = _to_float_array(nphi, n_samples)
        if rhob is not None:
            raw_inputs["RHOB"] = _to_float_array(rhob, n_samples)
        if pef is not None:
            raw_inputs["PEF"] = _to_float_array(pef, n_samples)
        if dtc is not None:
            raw_inputs["DTC"] = _to_float_array(dtc, n_samples)
        if phit is not None:
            raw_inputs["PHIT_GUIDE"] = _to_float_array(phit, n_samples)

        rt_arr = (
            _to_float_array(rt, n_samples)
            if rt is not None
            else np.full(n_samples, np.nan, dtype=float)
        )

        if logs is not None:
            for k, v in logs.items():
                ku = k.upper()
                if ku in ("RT", "CT"):
                    rt_arr = _to_float_array(v, n_samples)
                else:
                    raw_inputs[ku] = _to_float_array(v, n_samples)

        for k, v in extra_logs.items():
            ku = k.upper()
            if ku in ("RT", "CT"):
                rt_arr = _to_float_array(v, n_samples)
            else:
                raw_inputs[ku] = _to_float_array(v, n_samples)

        gr_arr = raw_inputs.get("GR", np.full(n_samples, np.nan, dtype=float))
        nphi_arr = raw_inputs.get("NPHI", np.full(n_samples, np.nan, dtype=float))
        rhob_arr = raw_inputs.get("RHOB", np.full(n_samples, np.nan, dtype=float))
        pef_arr = raw_inputs.get("PEF", np.full(n_samples, np.nan, dtype=float))
        dtc_arr = raw_inputs.get("DTC", np.full(n_samples, np.nan, dtype=float))

        active_cond_model = (
            conductivity_model.lower()
            if conductivity_model is not None
            else self.conductivity_model
        )
        if (
            active_cond_model is not None
            and active_cond_model not in CONDUCTIVITY_MODELS
        ):
            raise ValueError(
                f"Unknown conductivity_model '{active_cond_model}'. "
                f"Choose from {CONDUCTIVITY_MODELS} or None."
            )
        prev_cond_model = self.conductivity_model
        self.conductivity_model = active_cond_model

        try:
            # 3. Handle PEF linearization via U = PEF * RHOB
            do_pef_lin = (
                self.linearize_pef if linearize_pef is None else bool(linearize_pef)
            )
            use_u_channel = False
            solve_inputs = dict(raw_inputs)

            if do_pef_lin and "PEF" in solve_inputs and "RHOB" in solve_inputs:
                u_meas = solve_inputs["PEF"] * solve_inputs["RHOB"]
                # Invert with U instead of raw PEF
                solve_inputs.pop("PEF")
                solve_inputs["U"] = u_meas
                use_u_channel = True

            # 4. Identify linear solve log names (ordered: GR, NPHI, RHOB, U/PEF, DTC, PHIT_GUIDE, then extra)
            canonical_order = [
                "GR",
                "NPHI",
                "RHOB",
                "U" if use_u_channel else "PEF",
                "DTC",
                "PHIT_GUIDE",
            ]
            linear_log_names = [k for k in canonical_order if k in solve_inputs]
            for k in solve_inputs.keys():
                if k not in linear_log_names:
                    # Include if at least one mineral defines this property
                    if any(k in m_props for m_props in self.responses.values()):
                        linear_log_names.append(k)

            n_minerals = len(self.minerals)
            solver_choice = (method or self.solver or "auto").lower()
            workers = n_jobs if n_jobs is not None else self.n_jobs
            phi_max_eff = float(phi_max) if phi_max is not None else self.phi_max

            # Resolve two-stage execution mode
            run_two_stage = (
                self.two_stage if two_stage is None else bool(two_stage)
            )
            if solver_choice == "slsqp":
                run_two_stage = False

            if run_two_stage:
                stage1_fluids = ["WATER"]
                n_vars = n_minerals + 1
            else:
                stage1_fluids = list(self.fluids)
                n_vars = n_minerals + len(self.fluids)

            # Degrees of Freedom (DoF) check
            n_equations = len(linear_log_names) + 1  # logs + closure constraint
            if n_equations < n_vars:
                logger.warning(
                    f"MultiMineral model is under-determined: {len(linear_log_names)} log curves "
                    f"+ 1 closure constraint ({n_equations} equations) < {n_vars} components. "
                    f"Solution will rely on regularization or bounds. Consider providing additional "
                    f"logs (e.g. PEF, DTC, POTA, THOR) or zoning to fewer active minerals."
                )

            # 5. Dynamic scaling factors based on tool uncertainties or percentile ranges
            if auto_scale:
                use_uncertainty = (
                    isinstance(auto_scale, str) and auto_scale.lower() == "uncertainty"
                ) or auto_scale is True
                for log_type, log_array in solve_inputs.items():
                    if use_uncertainty and log_type in self.tool_uncertainties:
                        self.scaling_factors[log_type] = round(
                            1.0 / self.tool_uncertainties[log_type], 4
                        )
                    else:
                        valid_data = log_array[np.isfinite(log_array)]
                        fallback_r = DEFAULT_LOG_RANGES.get(log_type, 1.0)
                        if len(valid_data) > 1:
                            p05 = float(np.percentile(valid_data, 5))
                            p95 = float(np.percentile(valid_data, 95))
                            log_range = p95 - p05
                            if log_range > 1e-4:
                                self.scaling_factors[log_type] = round(1.0 / log_range, 4)
                            else:
                                self.scaling_factors[log_type] = round(1.0 / fallback_r, 4)
                        else:
                            self.scaling_factors[log_type] = round(1.0 / fallback_r, 4)

                if "PHIT_GUIDE" in solve_inputs:
                    self.scaling_factors["PHIT_GUIDE"] = round(1.0 / float(phit_uncertainty), 4)
                self.scaling_factors["CT"] = round(
                    1.0 / self.tool_uncertainties.get("CT", 0.05), 4
                )
                self.scaling_factors["RT"] = round(
                    1.0 / self.tool_uncertainties.get("RT", 0.05), 4
                )

            logger.info(f"Scaling factors: {self.scaling_factors}")

            # 6. Borehole quality / bad hole detection & depth-dependent weighting
            is_bad = np.zeros(n_samples, dtype=bool)
            if bad_hole_mask is not None:
                is_bad |= np.asarray(bad_hole_mask, dtype=bool).ravel()
            if cali is not None and bs is not None:
                cali_arr = _to_float_array(cali, n_samples)
                bs_arr = _to_float_array(bs, n_samples)
                is_bad |= cali_arr > (bs_arr + 1.5)
            if drho is not None:
                drho_arr = _to_float_array(drho, n_samples)
                is_bad |= np.abs(drho_arr) > 0.05

            # Build weight matrix W (n_samples, n_logs)
            custom_weights = log_weights or {}
            base_weights = np.array(
                [
                    self.scaling_factors.get(k, 1.0) * float(custom_weights.get(k, 1.0))
                    for k in linear_log_names
                ],
                dtype=float,
            )
            W_matrix = np.tile(base_weights, (n_samples, 1))

            # Apply washout down-weighting for sensitive tools (RHOB, PEF, U)
            has_depth_varying_weights = False
            if np.any(is_bad):
                has_depth_varying_weights = True
                bad_channels = ("RHOB", "PEF", "U")
                for col_i, l_name in enumerate(linear_log_names):
                    if l_name in bad_channels:
                        W_matrix[is_bad, col_i] *= float(bad_hole_weight)

            for col_i, l_name in enumerate(linear_log_names):
                cw = custom_weights.get(l_name)
                if isinstance(cw, (list, tuple, np.ndarray)):
                    cw_arr = _to_float_array(cw, n_samples)
                    W_matrix[:, col_i] = self.scaling_factors.get(l_name, 1.0) * cw_arr
                    has_depth_varying_weights = True

            # 7. Stack log arrays into measured matrix Y_linear (n_samples, n_logs)
            Y_linear = np.column_stack([solve_inputs[k] for k in linear_log_names])
            finite_mask = np.isfinite(Y_linear)

            has_rt = (
                (np.isfinite(rt_arr) & (rt_arr > 0))
                if active_cond_model is not None
                else np.zeros(n_samples, dtype=bool)
            )
            valid_counts = np.sum(finite_mask, axis=1) + has_rt.astype(int)
            valid_rows = valid_counts >= min(3, len(linear_log_names))

            # Resolve lower and upper bounds for Stage 1
            if run_two_stage:
                cur_lower_min = [b[0] for b in self.mineral_bounds]
                cur_upper_min = [b[1] for b in self.mineral_bounds]
                fl_lower = 0.0
                fl_upper = phi_max_eff

                if bounds is not None:
                    if isinstance(bounds, dict):
                        for j, m in enumerate(self.minerals):
                            if m in bounds:
                                cur_lower_min[j] = bounds[m][0]
                                cur_upper_min[j] = bounds[m][1]
                        for f_key in ("WATER", "FLUID", "PHIT", "PHIT_CONSTRUCTED"):
                            if f_key in bounds:
                                fl_lower = max(0.0, bounds[f_key][0])
                                fl_upper = min(bounds[f_key][1], phi_max_eff)
                    elif len(bounds) >= n_vars:
                        cur_lower_min = [bd[0] for bd in bounds[:n_minerals]]
                        cur_upper_min = [bd[1] for bd in bounds[:n_minerals]]
                        fl_lower = max(0.0, bounds[n_minerals][0])
                        fl_upper = min(bounds[n_minerals][1], phi_max_eff)

                lower = np.array(cur_lower_min + [fl_lower], dtype=float)
                upper = np.array(cur_upper_min + [fl_upper], dtype=float)

                phi_init = min(0.15, fl_upper * 0.5)
                V0 = np.full((n_samples, n_vars), 0.0, dtype=float)
                V0[:, :n_minerals] = (1.0 - phi_init) / max(n_minerals, 1)
                V0[:, -1] = phi_init
                V0 = _project_bounded_simplex_batch(V0, lower, upper, target_sum=1.0)
                C_full = self._build_response_matrix(linear_log_names, fluids=stage1_fluids)
            else:
                if bounds is not None:
                    if isinstance(bounds, dict):
                        cur_lower = [b[0] for b in self.bounds]
                        cur_upper = [b[1] for b in self.bounds]
                        for j, m in enumerate(self.minerals):
                            if m in bounds:
                                cur_lower[j] = bounds[m][0]
                                cur_upper[j] = bounds[m][1]
                        for j, f in enumerate(self.fluids):
                            if f in bounds:
                                cur_lower[n_minerals + j] = bounds[f][0]
                                cur_upper[n_minerals + j] = bounds[f][1]
                        lower = np.array(cur_lower, dtype=float)
                        upper = np.array(cur_upper, dtype=float)
                    else:
                        lower = np.array([bd[0] for bd in bounds], dtype=float)
                        upper = np.array([bd[1] for bd in bounds], dtype=float)
                else:
                    lower = np.array([b[0] for b in self.bounds], dtype=float)
                    upper = np.array([b[1] for b in self.bounds], dtype=float)

                V0 = _project_bounded_simplex_batch(
                    np.full((n_samples, n_vars), 1.0 / n_vars, dtype=float), lower, upper
                )
                C_full = self._build_response_matrix(linear_log_names, fluids=self.fluids)

            V_stage1 = np.full((n_samples, n_vars), np.nan, dtype=float)
            success_flags = np.full(n_samples, False, dtype=bool)

            # 8. Solve bounded least squares across valid depth samples
            if np.any(valid_rows):
                valid_idx = np.where(valid_rows)[0]
                mask_sub = finite_mask[valid_idx]

                if not has_depth_varying_weights:
                    patterns, inv = np.unique(mask_sub, axis=0, return_inverse=True)

                    def _solve_group(
                        pat_i: int,
                    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                        local_rows = np.where(inv == pat_i)[0]
                        global_rows = valid_idx[local_rows]
                        col_mask = patterns[pat_i]
                        if not np.any(col_mask):
                            return (
                                global_rows,
                                V0[global_rows],
                                np.ones(len(global_rows), dtype=bool),
                            )
                        Cw_sub = (base_weights[col_mask, None]) * C_full[col_mask, :]
                        Yw_sub = (
                            Y_linear[np.ix_(global_rows, col_mask)]
                            * base_weights[None, col_mask]
                        )
                        V_sub, ok_sub = solve_bounded_least_squares_batch(
                            Cw_sub, Yw_sub, V0[global_rows], lower, upper
                        )
                        return global_rows, V_sub, ok_sub

                    if workers != 1 and len(patterns) > 1:
                        group_results = Parallel(n_jobs=workers, prefer="threads")(
                            delayed(_solve_group)(pi) for pi in range(len(patterns))
                        )
                    else:
                        group_results = [
                            _solve_group(pi) for pi in range(len(patterns))
                        ]

                    for g_rows, V_sub, ok_sub in group_results:
                        V_stage1[g_rows] = V_sub
                        success_flags[g_rows] = ok_sub
                else:
                    W_valid = W_matrix[valid_idx] * mask_sub
                    Y_clean = np.where(mask_sub, Y_linear[valid_idx], 0.0)
                    Yw_valid = Y_clean * W_valid
                    Cw_3d = W_valid[:, :, None] * C_full[None, :, :]
                    V_sub, ok_sub = solve_bounded_least_squares_batch(
                        Cw_3d, Yw_valid, V0[valid_idx], lower, upper
                    )
                    V_stage1[valid_idx] = V_sub
                    success_flags[valid_idx] = ok_sub

            # 9. Stage 2: Fluid Partitioning or Joint SLSQP
            use_joint_slsqp = solver_choice == "slsqp" or (
                not run_two_stage and active_cond_model in CONDUCTIVITY_MODELS
            )

            if run_two_stage:
                minerals_opt = V_stage1[:, :n_minerals]
                phit_opt = np.clip(V_stage1[:, -1], 0.0, phi_max_eff)

                vwater = np.full(n_samples, np.nan, dtype=float)
                voil = np.full(n_samples, np.nan, dtype=float)
                vgas = np.full(n_samples, np.nan, dtype=float)

                has_rt_and_model = (
                    active_cond_model in CONDUCTIVITY_MODELS
                    and np.any(np.isfinite(rt_arr) & (rt_arr > 0))
                )

                if not has_rt_and_model:
                    vwater[valid_rows] = phit_opt[valid_rows]
                    voil[valid_rows] = 0.0
                    vgas[valid_rows] = 0.0
                else:
                    rt_valid_mask = valid_rows & np.isfinite(rt_arr) & (rt_arr > 0)
                    if np.any(rt_valid_mask):
                        idx_rt = np.where(rt_valid_mask)[0]
                        phit_sub = np.clip(phit_opt[idx_rt], 1e-4, 1.0)
                        rt_sub = np.maximum(rt_arr[idx_rt], 1e-4)
                        ct_sub = 1.0 / rt_sub

                        clay_indices = [
                            i
                            for i, m in enumerate(self.minerals)
                            if m.upper() in CLAY_MINERAL_NAMES
                        ]
                        if clay_indices:
                            vclay_sub = np.sum(minerals_opt[idx_rt][:, clay_indices], axis=1)
                        else:
                            vclay_sub = np.zeros(len(idx_rt), dtype=float)
                        vclay_sub = np.clip(vclay_sub, 0.0, 1.0)

                        params = self.conductivity_params
                        a = float(params.get("a", 1.0))
                        m = float(params.get("m", 2.0))
                        n = float(params.get("n", 2.0))
                        rw = max(float(params.get("rw", 0.05)), 1e-5)

                        if active_cond_model in ("sandsiltclay_waxman_smits", "waxman_smits"):
                            B = float(
                                params.get("B", (1.0 - 0.83 * np.exp(-0.5 / rw)) * 3.83)
                            )
                            rho_clay = float(params.get("rho_clay", 2.65))
                            cec_clay = float(params.get("cec_clay", 0.062))
                            Qv = (
                                float(params.get("Qv", 0.0))
                                if "Qv" in params
                                else (vclay_sub * rho_clay * cec_clay / phit_sub)
                            )

                            if abs(n - 2.0) < 1e-5:
                                alpha = (phit_sub**m) / (a * rw)
                                beta = (phit_sub**m) * B * Qv / a
                                disc = np.maximum(beta**2 + 4.0 * alpha * ct_sub, 0.0)
                                sw_sub = (-beta + np.sqrt(disc)) / (
                                    2.0 * np.maximum(alpha, 1e-8)
                                )
                            else:
                                sw_sub = np.empty(len(idx_rt), dtype=float)
                                for s_i in range(len(idx_rt)):

                                    def _eq(s: float) -> float:
                                        return float(
                                            sandsiltclay_waxman_smits_conductivity(
                                                phit_sub[s_i],
                                                s,
                                                vclay_sub[s_i],
                                                **params,
                                            )
                                            - ct_sub[s_i]
                                        )

                                    try:
                                        from scipy.optimize import brentq

                                        sw_sub[s_i] = brentq(_eq, 1e-4, 1.0)
                                    except Exception:
                                        sw_sub[s_i] = 1.0

                        elif active_cond_model == "dual_water":
                            rwb = max(float(params.get("rwb", 0.15)), 1e-5)
                            nphi_sh = float(params.get("nphi_sh", 0.30))
                            swb = np.clip(vclay_sub * nphi_sh / phit_sub, 0.0, 1.0)

                            if abs(n - 2.0) < 1e-5:
                                alpha = (phit_sub**m) / (a * rw)
                                beta = (phit_sub**m) / a * swb * (1.0 / rwb - 1.0 / rw)
                                disc = np.maximum(beta**2 + 4.0 * alpha * ct_sub, 0.0)
                                sw_sub = (-beta + np.sqrt(disc)) / (
                                    2.0 * np.maximum(alpha, 1e-8)
                                )
                            else:
                                sw_sub = np.empty(len(idx_rt), dtype=float)
                                for s_i in range(len(idx_rt)):

                                    def _eq_dw(s: float) -> float:
                                        return float(
                                            dual_water_conductivity(
                                                phit_sub[s_i],
                                                s,
                                                vclay_sub[s_i],
                                                **params,
                                            )
                                            - ct_sub[s_i]
                                        )

                                    try:
                                        from scipy.optimize import brentq

                                        sw_sub[s_i] = brentq(_eq_dw, 1e-4, 1.0)
                                    except Exception:
                                        sw_sub[s_i] = 1.0
                        else:
                            sw_sub = np.ones(len(idx_rt), dtype=float)

                        sw_sub = np.clip(sw_sub, 0.01, 1.0)
                        vwater_sub = sw_sub * phit_sub
                        vhc_sub = np.maximum(phit_sub - vwater_sub, 0.0)

                        if np.any(np.isfinite(rhob_arr[idx_rt])) and np.any(
                            np.isfinite(nphi_arr[idx_rt])
                        ):
                            rho_matrix = 2.65
                            phi_d = np.clip(
                                (rho_matrix - rhob_arr[idx_rt]) / (rho_matrix - 1.0),
                                0.0,
                                1.0,
                            )
                            gas_xover = phi_d - nphi_arr[idx_rt]
                            has_gas = gas_xover > 0.03
                            vgas_sub = np.where(has_gas, np.clip(gas_xover, 0.0, vhc_sub), 0.0)
                            voil_sub = vhc_sub - vgas_sub
                        else:
                            voil_sub = vhc_sub
                            vgas_sub = np.zeros(len(idx_rt), dtype=float)

                        vwater[idx_rt] = vwater_sub
                        voil[idx_rt] = voil_sub
                        vgas[idx_rt] = vgas_sub

                    no_rt_rows = valid_rows & ~rt_valid_mask
                    if np.any(no_rt_rows):
                        vwater[no_rt_rows] = phit_opt[no_rt_rows]
                        voil[no_rt_rows] = 0.0
                        vgas[no_rt_rows] = 0.0

                if len(self.fluids) == 1:
                    V_all = np.column_stack([minerals_opt, vwater])
                    active_fluids_recon = list(self.fluids)
                else:
                    V_all = np.column_stack([minerals_opt, voil, vgas, vwater])
                    active_fluids_recon = ["OIL", "GAS", "WATER"]
                porosity_constructed = phit_opt
            else:
                V_all = V_stage1
                active_fluids_recon = list(self.fluids)
                if use_joint_slsqp and np.any(valid_rows):
                    valid_idx = np.where(valid_rows)[0]

                    def _run_slsqp_depth(i: int) -> Tuple[int, np.ndarray, bool]:
                        log_values = {
                            l_name: float(raw_inputs[l_name][i])
                            for l_name in raw_inputs.keys()
                            if np.isfinite(raw_inputs[l_name][i])
                        }
                        if (
                            active_cond_model is not None
                            and np.isfinite(rt_arr[i])
                            and rt_arr[i] > 0
                        ):
                            log_values["RT"] = rt_arr[i]
                        warm_v = V_all[i] if np.all(np.isfinite(V_all[i])) else V0[i]
                        m_v, f_v, ok = self._optimize_volumes(
                            log_values, method="slsqp", initial_guess=warm_v
                        )
                        return i, np.concatenate([m_v, f_v]), ok

                    if workers != 1:
                        nl_results = Parallel(n_jobs=workers, prefer="threads")(
                            delayed(_run_slsqp_depth)(int(i)) for i in valid_idx
                        )
                    else:
                        nl_results = [
                            _run_slsqp_depth(int(i))
                            for i in tqdm(
                                valid_idx,
                                desc="Processing non-linear depth points (SLSQP)",
                                unit="points",
                            )
                        ]
                    for idx_i, v_comb, ok in nl_results:
                        V_all[idx_i] = v_comb
                        success_flags[idx_i] = ok

                porosity_constructed = np.where(
                    valid_rows, np.sum(V_all[:, n_minerals:], axis=1), np.nan
                )

            # 10. Vectorized log reconstruction and error metrics
            C_full_all = self._build_response_matrix(linear_log_names, fluids=active_fluids_recon)
            recon_matrix = V_all @ C_full_all.T  # (n_samples, len(linear_log_names))
            recon_dict: Dict[str, np.ndarray] = {}
            for col_i, l_name in enumerate(linear_log_names):
                recon_dict[l_name] = recon_matrix[:, col_i]

            # Reconstruct PEF from U and RHOB when PEF linearization is active
            if use_u_channel:
                u_recon = recon_dict.get("U", np.full(n_samples, np.nan, dtype=float))
                rho_recon = recon_dict.get("RHOB", rhob_arr)
                safe_rho = np.where(
                    np.isfinite(rho_recon) & (rho_recon > 0.5), rho_recon, 2.65
                )
                pef_reconstructed = np.where(
                    valid_rows & np.isfinite(pef_arr),
                    u_recon / safe_rho,
                    np.nan,
                )
            else:
                pef_reconstructed = (
                    np.where(
                        valid_rows & np.isfinite(pef_arr), recon_dict["PEF"], np.nan
                    )
                    if "PEF" in recon_dict
                    else np.full(n_samples, np.nan, dtype=float)
                )

            gr_reconstructed = (
                np.where(valid_rows & np.isfinite(gr_arr), recon_dict["GR"], np.nan)
                if "GR" in recon_dict
                else np.full(n_samples, np.nan, dtype=float)
            )
            nphi_reconstructed = (
                np.where(valid_rows & np.isfinite(nphi_arr), recon_dict["NPHI"], np.nan)
                if "NPHI" in recon_dict
                else np.full(n_samples, np.nan, dtype=float)
            )
            rhob_reconstructed = (
                np.where(valid_rows & np.isfinite(rhob_arr), recon_dict["RHOB"], np.nan)
                if "RHOB" in recon_dict
                else np.full(n_samples, np.nan, dtype=float)
            )
            dtc_reconstructed = (
                np.where(valid_rows & np.isfinite(dtc_arr), recon_dict["DTC"], np.nan)
                if "DTC" in recon_dict
                else np.full(n_samples, np.nan, dtype=float)
            )

            # Compute average absolute error and normalized chi2 error
            all_diffs = []
            chi2_terms = []
            for l_name in linear_log_names:
                if l_name in solve_inputs:
                    meas_val = solve_inputs[l_name]
                    diff_val = np.where(
                        valid_rows & np.isfinite(meas_val),
                        np.abs(meas_val - recon_dict[l_name]),
                        np.nan,
                    )
                    all_diffs.append(diff_val)
                    w = float(self.scaling_factors.get(l_name, 1.0))
                    chi2_diff = np.where(
                        valid_rows & np.isfinite(meas_val),
                        (w * (meas_val - recon_dict[l_name])) ** 2,
                        np.nan,
                    )
                    chi2_terms.append(chi2_diff)

            if all_diffs:
                diff_stack = np.column_stack(all_diffs)
                with np.errstate(invalid="ignore", divide="ignore"):
                    average_errors = np.where(
                        valid_rows,
                        np.nanmean(diff_stack, axis=1),
                        np.nan,
                    )
            else:
                average_errors = np.full(n_samples, np.nan, dtype=float)

            if chi2_terms:
                chi2_stack = np.column_stack(chi2_terms)
                with np.errstate(invalid="ignore", divide="ignore"):
                    chi2_errors = np.where(
                        valid_rows,
                        np.nanmean(chi2_stack, axis=1),
                        np.nan,
                    )
            else:
                chi2_errors = np.full(n_samples, np.nan, dtype=float)

            # 11. Construct output DataFrame
            output_data = {}

            mineral_column_mapping = {
                mineral: Config.MINERALS_NAME_MAPPING.get(mineral, f"V{mineral}")
                for mineral in self.minerals
            }
            for j, mineral in enumerate(self.minerals):
                column_name = mineral_column_mapping.get(mineral, f"V{mineral}")
                output_data[column_name] = V_all[:, j]

            fluid_column_mapping = {"OIL": "VOIL", "GAS": "VGAS", "WATER": "VWATER"}
            for j, fluid in enumerate(active_fluids_recon):
                column_name = fluid_column_mapping.get(fluid, f"V{fluid}")
                output_data[column_name] = V_all[:, n_minerals + j]

            output_data["PHIT_CONSTRUCTED"] = porosity_constructed
            output_data["AVG_ERROR"] = average_errors
            output_data["CHI2_ERROR"] = chi2_errors
            output_data["OPTIMIZER_SUCCESS"] = success_flags

            # Standard reconstructed curves
            output_data["GR_RECONSTRUCTED"] = gr_reconstructed
            output_data["NPHI_RECONSTRUCTED"] = nphi_reconstructed
            output_data["RHOB_RECONSTRUCTED"] = rhob_reconstructed
            output_data["PEF_RECONSTRUCTED"] = pef_reconstructed
            output_data["DTC_RECONSTRUCTED"] = dtc_reconstructed

            # Extra reconstructed curves (e.g. POTA, THOR, URAN)
            for l_name in linear_log_names:
                if l_name not in ("GR", "NPHI", "RHOB", "U", "PEF", "DTC", "PHIT_GUIDE"):
                    col_name = f"{l_name}_RECONSTRUCTED"
                    meas_val = raw_inputs.get(
                        l_name, np.full(n_samples, np.nan, dtype=float)
                    )
                    output_data[col_name] = np.where(
                        valid_rows & np.isfinite(meas_val),
                        recon_dict[l_name],
                        np.nan,
                    )

            return pd.DataFrame(output_data)
        finally:
            self.conductivity_model = prev_cond_model


class Optim(MultiMineral):
    """Multi-mineral optimizer supporting both matrix-vector (A, b) and log-based optimization.

    Provides vectorized Bounded Least Squares / Active-Set optimization for linear mineral
    systems (sum(V_i) = 1, 0 <= V_i <= 1) and reserves SLSQP only when non-linear
    conductivity (sandsiltclay_waxman_smits / dual_water) is enabled.
    """

    def __init__(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        minerals: Optional[List[str]] = None,
        fluid_properties: Optional[Dict] = None,
        conductivity_model: Optional[str] = None,
        conductivity_params: Optional[Dict] = None,
        solver: str = "auto",
        n_jobs: int = 1,
        responses: Optional[Dict[str, Dict[str, float]]] = None,
        mineral_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        fluid_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        linearize_pef: bool = False,
        fluids: Optional[List[str]] = None,
        two_stage: bool = True,
        phi_max: float = 0.40,
        tool_uncertainties: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            minerals=minerals,
            fluid_properties=fluid_properties,
            conductivity_model=conductivity_model,
            conductivity_params=conductivity_params,
            solver=solver,
            n_jobs=n_jobs,
            responses=responses,
            mineral_bounds=mineral_bounds,
            fluid_bounds=fluid_bounds,
            linearize_pef=linearize_pef,
            fluids=fluids,
            two_stage=two_stage,
            phi_max=phi_max,
            tool_uncertainties=tool_uncertainties,
        )
        self.A = A
        self.b = b

    def solve_fast(
        self,
        A: Optional[np.ndarray] = None,
        b: Optional[np.ndarray] = None,
        unity_weight: float = 100.0,
        normalize: bool = True,
    ) -> np.ndarray:
        """Solve non-negative mineral/fluid volumes via scipy.optimize.nnls with weighted unity closure."""
        A_in = A if A is not None else getattr(self, "A", None)
        b_in = b if b is not None else getattr(self, "b", None)

        if A_in is None:
            raise ValueError("Matrix A must be provided to solve_fast().")

        A_arr = np.asarray(A_in, dtype=float)
        lam = float(unity_weight)

        if b_in is None:
            log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC"][: A_arr.shape[0]]
            C_mat = self._build_response_matrix(log_names)
            Y_mat = A_arr.T
            single_sample = False
            transposed_output = False
        else:
            b_arr = np.asarray(b_in, dtype=float)
            single_sample = b_arr.ndim == 1
            transposed_output = False
            if single_sample:
                C_mat = A_arr
                Y_mat = b_arr[None, :]
            elif A_arr.ndim == 2:
                if b_arr.shape[0] == A_arr.shape[0]:
                    C_mat = A_arr
                    Y_mat = b_arr.T
                elif b_arr.shape[1] == A_arr.shape[0]:
                    C_mat = A_arr
                    Y_mat = b_arr
                    transposed_output = True
                elif A_arr.shape[1] == b_arr.shape[1]:
                    C_mat = b_arr
                    Y_mat = A_arr.T
                else:
                    raise ValueError(
                        f"Incompatible shapes for A {A_arr.shape} and b {b_arr.shape}."
                    )
            elif A_arr.ndim == 3:
                if (
                    b_arr.shape[0] == A_arr.shape[0]
                    and b_arr.shape[1] == A_arr.shape[1]
                ):
                    C_mat = np.transpose(A_arr, (1, 0, 2))
                    Y_mat = b_arr.T
                elif (
                    b_arr.shape[0] == A_arr.shape[0]
                    and b_arr.shape[1] == A_arr.shape[2]
                ):
                    C_mat = np.transpose(A_arr, (2, 0, 1))
                    Y_mat = b_arr.T
                elif (
                    A_arr.shape[0] == b_arr.shape[0]
                    and A_arr.shape[1] == b_arr.shape[1]
                ):
                    C_mat = A_arr
                    Y_mat = b_arr
                    transposed_output = True
                else:
                    raise ValueError(
                        f"Incompatible 3D shape for A {A_arr.shape} and b {b_arr.shape}."
                    )
            else:
                raise ValueError(f"Unsupported dimensions for A: {A_arr.ndim}")

        n_samples, n_logs = Y_mat.shape
        n_vars = C_mat.shape[-1]
        V_out = np.full((n_samples, n_vars), np.nan, dtype=float)
        unity_row = np.ones((1, n_vars), dtype=float) * lam
        unity_target = 1.0 * lam

        for i in range(n_samples):
            b_vec = Y_mat[i]
            A_mat = C_mat if C_mat.ndim == 2 else C_mat[i]
            valid_mask = np.isfinite(b_vec) & np.all(np.isfinite(A_mat), axis=1)
            if not np.any(valid_mask):
                continue

            A_valid = A_mat[valid_mask]
            b_valid = b_vec[valid_mask]

            A_aug = np.vstack([A_valid, unity_row])
            b_aug = np.append(b_valid, unity_target)

            x, _ = nnls(A_aug, b_aug)
            if normalize:
                v_sum = float(np.sum(x))
                if v_sum > 1e-12:
                    x = x / v_sum
                else:
                    x = np.full(n_vars, 1.0 / n_vars, dtype=float)
            V_out[i] = x

        if single_sample:
            return V_out[0]
        return V_out if transposed_output else V_out.T
