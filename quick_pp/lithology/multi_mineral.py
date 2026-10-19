from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.optimize import minimize
from tqdm import tqdm

from quick_pp import logger
from quick_pp.config import Config
from quick_pp.porosity import density_porosity, neu_den_xplot_poro_pt

CONDUCTIVITY_MODELS = ("sandsiltclay_waxman_smits", "waxman_smits", "dual_water")
CLAY_MINERAL_NAMES = ("SHALE", "CLAY", "KAOLINITE", "ILLITE", "SMECTITE", "CHLORITE")


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

    # Bracket tau_i so thattau_lo gives sum >= target_sum and tau_hi gives sum <= target_sum
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
                 state[release_rows, np.argmax(V[release_rows] - lower[None, :], axis=1)] = 0

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
    """

    def __init__(
        self,
        minerals: Optional[List[str]] = None,
        fluid_properties: Optional[Dict] = None,
        porosity_method: str = "density",
        porosity_endpoints: Optional[Dict] = None,
        conductivity_model: Optional[str] = None,
        conductivity_params: Optional[Dict] = None,
        solver: str = "auto",
        n_jobs: int = 1,
    ):
        """Initialize the MultiMineral optimizer.

        Args:
            minerals (list, optional): A list of minerals for the optimization.
                                       Defaults to ['QUARTZ', 'CALCITE', 'DOLOMITE', 'SHALE'].
            fluid_properties (dict, optional): Properties for oil, gas, and water. Defaults to standard values.
            porosity_method (str, optional): The method for initial porosity estimation
                                             ('neutron_density', 'density', 'sonic'). Defaults to 'density'.
            porosity_endpoints (dict, optional): Endpoints for the 'neutron_density' porosity method.
            conductivity_model (str, optional): Optional non-linear conductivity model
                                                ('sandsiltclay_waxman_smits', 'waxman_smits', 'dual_water', or None).
                                                When enabled, SLSQP is used to solve the non-linear conductivity term.
            conductivity_params (dict, optional): Parameters for the non-linear conductivity model
                                                  (e.g., rw, rwb, B, Qv, a, m, n, nphi_sh).
            solver (str, optional): Optimization solver ('auto', 'active_set', 'bounded_lsq', 'slsqp').
                                    Defaults to 'auto' (vectorized active-set for linear, SLSQP for non-linear).
            n_jobs (int, optional): Number of parallel jobs via joblib.Parallel for batching. Defaults to 1.
        """
        self.responses = Config.MINERALS_LOG_VALUE
        available_minerals = list(self.responses.keys())

        # Default minerals if none specified
        if minerals is None:
            minerals = ["QUARTZ", "CALCITE", "DOLOMITE", "SHALE"]

        for mineral in minerals:
            if mineral not in available_minerals:
                raise ValueError(
                    f"Mineral '{mineral}' is not defined in Config.MINERALS_LOG_VALUE."
                )

        self.minerals = minerals
        self.porosity_method = porosity_method
        self.porosity_endpoints = porosity_endpoints or {}
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

        # Default fluid properties if none specified
        if fluid_properties is None:
            fluid_properties = {
                "OIL": {"density": 0.8, "nphi": 0.0, "pef": 0.0, "dtc": 200.0},
                "GAS": {"density": 0.2, "nphi": 0.0, "pef": 0.0, "dtc": 400.0},
                "WATER": {"density": 1.0, "nphi": 1.0, "pef": 0.0, "dtc": 190.0},
            }
        self.fluid_properties = fluid_properties
        self.fluids = ["OIL", "GAS", "WATER"]

        # Define bounds for each mineral type
        mineral_bounds = dict.fromkeys(available_minerals, (0, 1))

        # Define bounds for fluid volumes (0 to 1 for each fluid)
        fluid_bounds = {"OIL": (0, 1), "GAS": (0, 1), "WATER": (0, 1)}

        # Set bounds based on selected minerals and fluids
        self.mineral_bounds = tuple(
            mineral_bounds[mineral] for mineral in self.minerals
        )
        self.fluid_bounds = tuple(fluid_bounds[fluid] for fluid in self.fluids)

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

    def _calculate_porosity(
        self, nphi: float, rhob: float, dtc: Optional[float] = None
    ) -> float:
        """Calculate an initial porosity estimate based on the selected method.

        Args:
            nphi (float): Neutron porosity value.
            rhob (float): Bulk density value.
            dtc (float, optional): Sonic transit time value. Defaults to None.

        Returns:
            float: The initial porosity estimate.
        """
        if self.porosity_method == "density":
            initial_rho_ma = sum(
                self.responses[mineral].get("RHOB", 0)
                for mineral in self.minerals
                if mineral in self.responses
            )
            count_minerals_with_density = sum(
                1
                for mineral in self.minerals
                if mineral in self.responses and "RHOB" in self.responses[mineral]
            )

            if count_minerals_with_density > 0:
                rho_ma = initial_rho_ma / count_minerals_with_density
            else:
                rho_ma = 2.65  # Default matrix density (Quartz)

            rho_fluid_avg = float(
                np.mean([prop["density"] for prop in self.fluid_properties.values()])
            )

            return float(density_porosity(rhob, rho_ma, rho_fluid_avg))

        elif self.porosity_method == "neutron_density":
            dry_min1_point = self.porosity_endpoints.get(
                "dry_min1_point", (-0.02, 2.65)
            )
            dry_clay_point = self.porosity_endpoints.get("dry_clay_point", (0.37, 2.41))
            fluid_point = self.porosity_endpoints.get("fluid_point", (1.0, 1.0))

            return float(
                neu_den_xplot_poro_pt(
                    nphi,
                    rhob,
                    model="ss",
                    dry_min1_point=dry_min1_point,
                    dry_clay_point=dry_clay_point,
                    fluid_point=fluid_point,
                )
            )

        elif self.porosity_method == "sonic":
            if dtc is None or np.isnan(dtc):
                return 0.0
            dt_ma = 55.0  # Default matrix slowness
            dt_fluid = 190.0  # Default fluid slowness
            return float((dtc - dt_ma) / (dt_fluid - dt_ma))

        else:
            raise ValueError(f"Unknown porosity method: {self.porosity_method}")

    def _calculate_porosity_vectorized(
        self,
        nphi: np.ndarray,
        rhob: np.ndarray,
        dtc: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Vectorized initial porosity estimation across all depth steps."""
        n_samples = len(rhob)
        both_valid = np.isfinite(nphi) & np.isfinite(rhob)
        init_poro = np.full(n_samples, 0.1, dtype=float)

        if not np.any(both_valid):
            return init_poro

        if self.porosity_method == "density":
            initial_rho_ma = sum(
                self.responses[mineral].get("RHOB", 0)
                for mineral in self.minerals
                if mineral in self.responses
            )
            count_minerals = sum(
                1
                for mineral in self.minerals
                if mineral in self.responses and "RHOB" in self.responses[mineral]
            )
            rho_ma = initial_rho_ma / count_minerals if count_minerals > 0 else 2.65
            rho_fluid_avg = float(
                np.mean([prop["density"] for prop in self.fluid_properties.values()])
            )
            poro_vals = (rho_ma - rhob[both_valid]) / (rho_ma - rho_fluid_avg)
            init_poro[both_valid] = np.clip(poro_vals, 0.0, 0.4)

        elif self.porosity_method == "neutron_density":
            dry_min1_point = self.porosity_endpoints.get(
                "dry_min1_point", (-0.02, 2.65)
            )
            dry_clay_point = self.porosity_endpoints.get("dry_clay_point", (0.37, 2.41))
            fluid_point = self.porosity_endpoints.get("fluid_point", (1.0, 1.0))

            # Vectorized line-intersection porosity equivalent to neu_den_xplot_poro_pt(..., model='ss')
            A = np.asarray(dry_min1_point, dtype=float)
            C = np.asarray(dry_clay_point, dtype=float)
            D = np.asarray(fluid_point, dtype=float)
            m_ac = (A[1] - C[1]) / (A[0] - C[0])
            n_v = nphi[both_valid]
            r_v = rhob[both_valid]
            c_int = r_v - m_ac * n_v

            # Intersection of line ((0, c_int) -> (n_v, r_v)) with line (D -> A)
            # Line 1: y = m_ac * x + c_int
            # Line 2: y - D[1] = m_da * (x - D[0])
            m_da = (A[1] - D[1]) / (A[0] - D[0])
            c_da = D[1] - m_da * D[0]
            x_iso = (c_da - c_int) / (m_ac - m_da)
            y_iso = m_ac * x_iso + c_int
            iso_len = np.hypot(x_iso - A[0], y_iso - A[1])
            poro_len = float(np.hypot(D[0] - A[0], D[1] - A[1]))
            poro_vals = iso_len / poro_len
            init_poro[both_valid] = np.clip(poro_vals, 0.0, 0.4)

        elif self.porosity_method == "sonic":
            if dtc is not None:
                valid_s = both_valid & np.isfinite(dtc)
                poro_vals = (dtc[valid_s] - 55.0) / (190.0 - 55.0)
                init_poro[valid_s] = np.clip(poro_vals, 0.0, 0.4)
                init_poro[both_valid & ~np.isfinite(dtc)] = 0.0
            else:
                init_poro[both_valid] = 0.0
        else:
            raise ValueError(f"Unknown porosity method: {self.porosity_method}")

        return init_poro

    def _calculate_matrix_density(self, mineral_volumes: np.ndarray) -> float:
        """Calculate the average matrix density from mineral volumes.

        Args:
            mineral_volumes (np.ndarray): An array of mineral volumes.

        Returns:
            float: The calculated matrix density.
        """
        rho_ma = 0.0
        total_mineral_vol = np.sum(mineral_volumes)
        if total_mineral_vol < 1e-6:
            return 2.65  # Return default if no minerals

        for i, mineral in enumerate(self.minerals):
            if mineral in self.responses and "RHOB" in self.responses[mineral]:
                rho_ma += mineral_volumes[i] * self.responses[mineral]["RHOB"]

        return float(rho_ma / total_mineral_vol)

    def _calculate_fluid_density(self, fluid_volumes: np.ndarray) -> float:
        """Calculate the average fluid density from fluid volumes.

        Args:
            fluid_volumes (np.ndarray): An array of fluid volumes.

        Returns:
            float: The calculated average fluid density.
        """
        rho_fluid = 0.0
        for i, fluid in enumerate(self.fluids):
            rho_fluid += fluid_volumes[i] * self.fluid_properties[fluid]["density"]
        return float(rho_fluid)

    def _build_response_matrix(self, log_types: List[str]) -> np.ndarray:
        """Construct the linear log response matrix C of shape (n_logs, n_vars)."""
        n_minerals = len(self.minerals)
        n_fluids = len(self.fluids)
        n_vars = n_minerals + n_fluids
        C = np.zeros((len(log_types), n_vars), dtype=float)

        fluid_key_map = {
            "NPHI": "nphi",
            "RHOB": "density",
            "DTC": "dtc",
            "PEF": "pef",
        }

        for r_idx, log_type in enumerate(log_types):
            for m_idx, mineral in enumerate(self.minerals):
                if mineral in self.responses and log_type in self.responses[mineral]:
                    C[r_idx, m_idx] = float(self.responses[mineral][log_type])
            prop_key = fluid_key_map.get(log_type)
            if prop_key is not None:
                for f_idx, fluid in enumerate(self.fluids):
                    C[r_idx, n_minerals + f_idx] = float(
                        self.fluid_properties[fluid].get(prop_key, 0.0)
                    )
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

    def _calculate_log_response(self, volumes: np.ndarray, log_type: str) -> float:
        """Calculate the reconstructed log response for a given log type.

        Args:
            volumes (np.ndarray): An array of combined mineral and fluid volumes.
            log_type (str): The type of log to reconstruct (e.g., 'NPHI', 'RHOB').

        Returns:
            float: The reconstructed log value.
        """
        if log_type in ("RT", "CT") and self.conductivity_model is not None:
            ct = self._calculate_conductivity_response(volumes)
            return float(1.0 / max(ct, 1e-6)) if log_type == "RT" else float(ct)

        response = 0.0

        # Add mineral contributions
        n_minerals = len(self.minerals)
        for i, mineral in enumerate(self.minerals):
            if mineral in self.responses and log_type in self.responses[mineral]:
                response += volumes[i] * self.responses[mineral][log_type]

        # Add fluid contributions
        for i, fluid in enumerate(self.fluids):
            fluid_idx = n_minerals + i
            fluid_vol = volumes[fluid_idx]

            if log_type == "NPHI":
                response += fluid_vol * self.fluid_properties[fluid]["nphi"]
            elif log_type == "RHOB":
                response += fluid_vol * self.fluid_properties[fluid]["density"]
            elif log_type == "DTC":
                response += fluid_vol * self.fluid_properties[fluid]["dtc"]
            elif log_type == "PEF":
                response += fluid_vol * self.fluid_properties[fluid]["pef"]
            # GR doesn't change with fluid type (assuming fresh water)

        return float(response)

    def _calculate_average_error(
        self, volumes: np.ndarray, log_values: Dict[str, float]
    ) -> float:
        """Calculate the average absolute error between measured and reconstructed logs.

        Args:
            volumes (np.ndarray): An array of combined mineral and fluid volumes.
            log_values (dict): A dictionary of measured log values.

        Returns:
            float: The average absolute error.
        """
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
        """Calculate the sum of squared errors for the optimization.

        Args:
            volumes (np.ndarray): An array of combined mineral and fluid volumes.
            log_values (dict): A dictionary of measured log values.

        Returns:
            float: The total squared error.
        """
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
                    scaling = self.scaling_factors.get(
                        log_type, self.scaling_factors.get("CT", 1.0)
                    )
                    total_error += (scaling * (ct_meas - ct_recon)) ** 2
                    continue

                if self.porosity_method == "density" and log_type == "RHOB":
                    mineral_vols = volumes[: len(self.minerals)]
                    fluid_vols = volumes[len(self.minerals) :]
                    porosity = np.sum(fluid_vols)

                    # For density porosity, RHOB is reconstructed from porosity and matrix density
                    rho_ma = self._calculate_matrix_density(mineral_vols)
                    rho_fluid = self._calculate_fluid_density(fluid_vols)
                    reconstructed_value = rho_ma * (1 - porosity) + rho_fluid
                else:
                    reconstructed_value = self._calculate_log_response(
                        volumes, log_type
                    )

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
        """Perform the optimization to find the best-fit mineral and fluid volumes for a single depth point.

        Uses the fast Active-Set Bounded Least Squares solver for the linear system,
        reserving SLSQP for non-linear conductivity models or when method='slsqp'.

        Args:
            log_values (dict): A dictionary of measured log values for a single depth point.
            method (str, optional): Solver method override ('auto', 'active_set', 'slsqp').
            initial_guess (np.ndarray, optional): Warm-start initial volume guess.

        Returns:
            tuple: A tuple containing the optimized mineral volumes, fluid volumes, and a success flag.
        """
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
            nphi = log_values.get("NPHI", 0.0)
            rhob = log_values.get("RHOB", 2.65)
            dtc = log_values.get("DTC", None)

            if (
                nphi is not None
                and rhob is not None
                and not np.isnan(nphi)
                and not np.isnan(rhob)
            ):
                initial_porosity = self._calculate_porosity(nphi, rhob, dtc)
                initial_porosity = max(0.0, min(0.4, initial_porosity))
            else:
                initial_porosity = 0.1

            mineral_volume = (1.0 - initial_porosity) / n_minerals
            fluid_volume = initial_porosity / n_fluids
            initial_guess = np.concatenate(
                [np.full(n_minerals, mineral_volume), np.full(n_fluids, fluid_volume)]
            )

        solver_choice = (method or self.solver or "auto").lower()
        use_nonlinear = solver_choice == "slsqp" or (
            self.conductivity_model in CONDUCTIVITY_MODELS
            and ("RT" in valid_logs or "CT" in valid_logs)
        )

        if not use_nonlinear:
            linear_keys = [
                k for k in valid_logs.keys() if k in ("GR", "NPHI", "RHOB", "PEF", "DTC")
            ]
            C = self._build_response_matrix(linear_keys)
            weights = np.array(
                [self.scaling_factors.get(k, 1.0) for k in linear_keys], dtype=float
            )
            Cw = weights[:, None] * C
            Yw = (
                weights
                * np.array([valid_logs[k] for k in linear_keys], dtype=float)
            )[None, :]
            lower = np.array([b[0] for b in self.bounds], dtype=float)
            upper = np.array([b[1] for b in self.bounds], dtype=float)
            V_opt, success_arr = solve_bounded_least_squares_batch(
                Cw, Yw, initial_guess[None, :], lower, upper
            )
            x_opt = V_opt[0]
            return x_opt[:n_minerals], x_opt[n_minerals:], bool(success_arr[0])

        # Run SLSQP optimization (reserved for non-linear conductivity or explicit method='slsqp')
        result = minimize(
            self._error_function,
            initial_guess,
            args=(valid_logs,),
            bounds=self.bounds,
            constraints=self.constraints,
            method="SLSQP",
        )

        mineral_volumes = result.x[:n_minerals]
        fluid_volumes = result.x[n_minerals:]

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
        """Vectorized multi-mineral optimizer for linear systems A * V = b or non-linear conductivity systems.

        Solves the bounded simplex quadratic program across all depth steps simultaneously using
        a vectorized Active-Set Bounded Least Squares solver when the system is linear, and
        reserves SLSQP (with optional joblib.Parallel batching) only when non-linear conductivity
        (sandsiltclay_waxman_smits / dual_water) is enabled or method='slsqp'.

        Args:
            A (np.ndarray, optional): Response matrix of shape (n_logs, n_vars) or (n_logs, n_vars, n_samples)
                                      or (n_logs, n_samples, n_vars). If b is None, A is treated as the
                                      measured log array of shape (n_logs, n_samples).
            b (np.ndarray, optional): Measured log array of shape (n_logs, n_samples) or (n_samples, n_logs).
            bounds (tuple, optional): Bounds for each component, defaults to (0, 1) per component.
            conductivity_model (str, optional): Non-linear conductivity model
                                                ('sandsiltclay_waxman_smits' or 'dual_water').
            rt (np.ndarray, optional): True resistivity log [ohm.m] for non-linear conductivity inversion.
            method (str, optional): Solver ('auto', 'active_set', 'bounded_lsq', 'slsqp').
            n_jobs (int, optional): Number of parallel workers via joblib.Parallel.

        Returns:
            np.ndarray: Optimized component volume fractions across all depth steps.
        """
        if A is None and hasattr(self, "A"):
            A = getattr(self, "A")
        if b is None and hasattr(self, "b"):
            b = getattr(self, "b")

        if A is None:
            raise ValueError("Matrix A (or measured log array) must be provided to optimize().")

        A_arr = np.asarray(A, dtype=float)
        transposed_output = False

        if b is None:
            # A_arr was provided as log curves (n_logs, n_samples); build response matrix from model
            log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC"][: A_arr.shape[0]]
            C_mat = self._build_response_matrix(log_names)
            Y_mat = A_arr.T  # (n_samples, n_logs)
        else:
            b_arr = np.asarray(b, dtype=float)
            if b_arr.ndim == 1:
                b_arr = b_arr[:, None]

            if A_arr.ndim == 2:
                # Determine if A is (n_logs, n_vars) or if A and b are swapped
                if b_arr.shape[0] == A_arr.shape[0]:
                    C_mat = A_arr  # (m, p)
                    Y_mat = b_arr.T  # (N, m)
                elif b_arr.shape[1] == A_arr.shape[0]:
                    C_mat = A_arr  # (m, p)
                    Y_mat = b_arr  # (N, m)
                    transposed_output = True
                elif A_arr.shape[1] == b_arr.shape[1] and b_arr.ndim == 2:
                    # A was passed as log measurements (m, N) and b as (m, p)
                    C_mat = b_arr
                    Y_mat = A_arr.T
                else:
                    raise ValueError(
                        f"Incompatible shapes for A {A_arr.shape} and b {b_arr.shape}."
                    )
            elif A_arr.ndim == 3:
                # 3D response matrix across depth steps
                if b_arr.shape[0] == A_arr.shape[0] and b_arr.shape[1] == A_arr.shape[1]:
                    # A is (m, N, p), b is (m, N) -> len(A[0]) == N
                    C_mat = np.transpose(A_arr, (1, 0, 2))  # (N, m, p)
                    Y_mat = b_arr.T  # (N, m)
                elif b_arr.shape[0] == A_arr.shape[0] and b_arr.shape[1] == A_arr.shape[2]:
                    # A is (m, p, N), b is (m, N)
                    C_mat = np.transpose(A_arr, (2, 0, 1))  # (N, m, p)
                    Y_mat = b_arr.T  # (N, m)
                elif A_arr.shape[0] == b_arr.shape[0] and A_arr.shape[1] == b_arr.shape[1]:
                    # A is (N, m, p), b is (N, m)
                    C_mat = A_arr
                    Y_mat = b_arr
                    transposed_output = True
                else:
                    raise ValueError(
                        f"Incompatible 3D shape for A {A_arr.shape} and b {b_arr.shape}."
                    )
            else:
                raise ValueError(f"Unsupported number of dimensions for A: {A_arr.ndim}")

        N, m = Y_mat.shape
        p = C_mat.shape[-1]

        if bounds is not None:
            lower = np.array([ bd[0] for bd in bounds ], dtype=float)
            upper = np.array([ bd[1] for bd in bounds ], dtype=float)
        elif len(self.bounds) == p:
            lower = np.array([ bd[0] for bd in self.bounds ], dtype=float)
            upper = np.array([ bd[1] for bd in self.bounds ], dtype=float)
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
        use_nonlinear = (
            solver_choice == "slsqp"
            or cond_model in CONDUCTIVITY_MODELS
        )

        V_out = np.full((N, p), np.nan, dtype=float)
        valid_row = np.sum(np.isfinite(Y_mat), axis=1) >= min(3, m)
        if not np.any(valid_row):
            return V_out if transposed_output else V_out.T

        # Step 1: Fast vectorized Active-Set Bounded Least Squares solution
        # (Serves as final solution for linear system, or warm-start for non-linear SLSQP)
        valid_indices = np.where(valid_row)[0]
        Y_valid = np.where(np.isfinite(Y_mat[valid_indices]), Y_mat[valid_indices], 0.0)
        mask_valid = np.isfinite(Y_mat[valid_indices]).astype(float)

        if C_mat.ndim == 2:
            # Group by missingness pattern for exact weighted active-set solve
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

        # Step 2: If non-linear conductivity (Waxman-Smits / Dual Water) or explicit SLSQP is enabled,
        # refine with SLSQP (warm-started from the vectorized bounded least squares solution)
        if use_nonlinear:
            rt_arr = (
                np.asarray(rt, dtype=float)
                if rt is not None
                else np.full(N, np.nan, dtype=float)
            )
            bounds_seq = tuple(zip(lower, upper))
            constraints_seq = [{"type": "eq", "fun": lambda v: float(np.sum(v) - 1.0)}]

            def _solve_slsqp_row(idx: int) -> np.ndarray:
                y_i = Y_mat[idx]
                f_mask = np.isfinite(y_i)
                C_i = C_mat[f_mask] if C_mat.ndim == 2 else C_mat[idx, f_mask]
                y_f = y_i[f_mask]
                v_init = V_out[idx] if np.all(np.isfinite(V_out[idx])) else V0[idx]
                rt_val = rt_arr[idx] if idx < len(rt_arr) else np.nan
                has_rt = cond_model in CONDUCTIVITY_MODELS and np.isfinite(rt_val) and rt_val > 0

                def _obj(v: np.ndarray) -> float:
                    diff = C_i @ v - y_f
                    err = float(np.dot(diff, diff))
                    if has_rt:
                        phit_v = max(float(v[-1]), 1e-4)
                        vclay_v = float(v[min(3, p - 1)]) if p > 1 else 0.0
                        if cond_model in ("sandsiltclay_waxman_smits", "waxman_smits"):
                            ct_hat = float(
                                sandsiltclay_waxman_smits_conductivity(
                                    phit_v, 1.0, vclay_v, **self.conductivity_params
                                )
                            )
                        else:
                            ct_hat = float(
                                dual_water_conductivity(
                                    phit_v, 1.0, vclay_v, **self.conductivity_params
                                )
                            )
                        err += (ct_hat - 1.0 / rt_val) ** 2
                    return err

                res = minimize(
                    _obj,
                    v_init,
                    bounds=bounds_seq,
                    constraints=constraints_seq,
                    method="SLSQP",
                )
                return res.x

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
        gr: np.ndarray,
        nphi: np.ndarray,
        rhob: np.ndarray,
        pef: Optional[np.ndarray] = None,
        dtc: Optional[np.ndarray] = None,
        rt: Optional[np.ndarray] = None,
        auto_scale: bool = True,
        conductivity_model: Optional[str] = None,
        method: Optional[str] = None,
        n_jobs: Optional[int] = None,
    ) -> pd.DataFrame:
        """Estimate mineral and fluid volumes for a set of log data points.

        Uses the fast vectorized Bounded Least Squares / Active-Set solver across all depth
        points simultaneously for the linear mineral system, reserving SLSQP (with optional
        joblib.Parallel batching) only when non-linear conductivity
        ('sandsiltclay_waxman_smits' or 'dual_water') is enabled or method='slsqp'.

        Args:
            gr (np.ndarray): Gamma Ray log [GAPI].
            nphi (np.ndarray): Neutron Porosity log [v/v].
            rhob (np.ndarray): Bulk Density log [g/cc].
            pef (np.ndarray, optional): Photoelectric Factor log [barns/electron]. Defaults to None.
            dtc (np.ndarray, optional): Compressional slowness log [us/ft]. Defaults to None.
            rt (np.ndarray, optional): True Resistivity log [ohm.m] (used when conductivity_model is enabled).
            auto_scale (bool, optional): If True, automatically calculates scaling factors. Defaults to True.
            conductivity_model (str, optional): Non-linear conductivity model override
                                                ('sandsiltclay_waxman_smits', 'dual_water', or None).
            method (str, optional): Solver method override ('auto', 'active_set', 'bounded_lsq', 'slsqp').
            n_jobs (int, optional): Number of parallel jobs via joblib.Parallel. Defaults to self.n_jobs.

        Returns:
            pd.DataFrame: A DataFrame containing the estimated mineral and fluid volumes,
                          constructed porosity, error metrics, and reconstructed logs for each depth point.
        """
        def _to_float_array(arr: Optional[np.ndarray], length: int) -> np.ndarray:
            if arr is None:
                return np.full(length, np.nan, dtype=float)
            try:
                return np.asarray(arr, dtype=float).ravel()
            except (ValueError, TypeError):
                return np.array(
                    [np.nan if v is None else float(v) for v in arr], dtype=float
                )

        n_samples = len(gr)
        gr_arr = _to_float_array(gr, n_samples)
        nphi_arr = _to_float_array(nphi, n_samples)
        rhob_arr = _to_float_array(rhob, n_samples)
        pef_arr = _to_float_array(pef, n_samples)
        dtc_arr = _to_float_array(dtc, n_samples)
        rt_arr = _to_float_array(rt, n_samples)

        active_cond_model = (
            conductivity_model.lower()
            if conductivity_model is not None
            else self.conductivity_model
        )
        if active_cond_model is not None and active_cond_model not in CONDUCTIVITY_MODELS:
            raise ValueError(
                f"Unknown conductivity_model '{active_cond_model}'. "
                f"Choose from {CONDUCTIVITY_MODELS} or None."
            )
        prev_cond_model = self.conductivity_model
        self.conductivity_model = active_cond_model

        try:
            # Dynamic scaling based on log ranges for robustness
            if auto_scale:
                all_logs = {
                    "GR": gr_arr,
                    "NPHI": nphi_arr,
                    "RHOB": rhob_arr,
                    "PEF": pef_arr if pef is not None else None,
                    "DTC": dtc_arr if dtc is not None else None,
                }
                for log_type, log_array in all_logs.items():
                    if log_array is not None:
                        valid_data = log_array[~np.isnan(log_array)]
                        if len(valid_data) > 1:
                            p05 = np.percentile(valid_data, 5)
                            p95 = np.percentile(valid_data, 95)
                            log_range = p95 - p05
                            if log_range > 1e-6:
                                self.scaling_factors[log_type] = round(
                                    1.0 / log_range, 4
                                )
                            else:
                                self.scaling_factors[log_type] = 1.0

            logger.info(f"Scaling factors: {self.scaling_factors}")

            n_minerals = len(self.minerals)
            n_fluids = len(self.fluids)
            n_vars = n_minerals + n_fluids

            linear_log_names = ["GR", "NPHI", "RHOB", "PEF", "DTC"]
            Y_linear = np.column_stack([gr_arr, nphi_arr, rhob_arr, pef_arr, dtc_arr])
            finite_mask = np.isfinite(Y_linear)

            # Need at least 3 valid measurements (including RT if non-linear conductivity is active)
            has_rt = (
                (np.isfinite(rt_arr) & (rt_arr > 0))
                if active_cond_model is not None
                else np.zeros(n_samples, dtype=bool)
            )
            valid_counts = np.sum(finite_mask, axis=1) + has_rt.astype(int)
            valid_rows = valid_counts >= 3

            # Vectorized initial guesses across all depth steps
            init_poro = self._calculate_porosity_vectorized(
                nphi_arr, rhob_arr, dtc_arr if dtc is not None else None
            )
            mineral_init = ((1.0 - init_poro) / n_minerals)[:, None]
            fluid_init = (init_poro / n_fluids)[:, None]
            V0 = np.hstack(
                [
                    np.repeat(mineral_init, n_minerals, axis=1),
                    np.repeat(fluid_init, n_fluids, axis=1),
                ]
            )

            lower = np.array([b[0] for b in self.bounds], dtype=float)
            upper = np.array([b[1] for b in self.bounds], dtype=float)

            C_full = self._build_response_matrix(linear_log_names)
            weights_full = np.array(
                [self.scaling_factors.get(k, 1.0) for k in linear_log_names],
                dtype=float,
            )

            V_all = np.full((n_samples, n_vars), np.nan, dtype=float)
            success_flags = np.full(n_samples, False, dtype=bool)

            solver_choice = (method or self.solver or "auto").lower()
            workers = n_jobs if n_jobs is not None else self.n_jobs
            use_nonlinear = solver_choice == "slsqp" or (
                active_cond_model in CONDUCTIVITY_MODELS
            )

            if np.any(valid_rows):
                valid_idx = np.where(valid_rows)[0]
                mask_sub = finite_mask[valid_idx]
                patterns, inv = np.unique(mask_sub, axis=0, return_inverse=True)

                def _solve_group(pat_i: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
                    local_rows = np.where(inv == pat_i)[0]
                    global_rows = valid_idx[local_rows]
                    col_mask = patterns[pat_i]
                    if not np.any(col_mask):
                        return (
                            global_rows,
                            V0[global_rows],
                            np.ones(len(global_rows), dtype=bool),
                        )
                    Cw_sub = (weights_full[col_mask, None]) * C_full[col_mask, :]
                    Yw_sub = Y_linear[np.ix_(global_rows, col_mask)] * weights_full[
                        None, col_mask
                    ]
                    V_sub, ok_sub = solve_bounded_least_squares_batch(
                        Cw_sub, Yw_sub, V0[global_rows], lower, upper
                    )
                    return global_rows, V_sub, ok_sub

                if workers != 1 and len(patterns) > 1:
                    group_results = Parallel(n_jobs=workers, prefer="threads")(
                        delayed(_solve_group)(pi) for pi in range(len(patterns))
                    )
                else:
                    group_results = [_solve_group(pi) for pi in range(len(patterns))]

                for g_rows, V_sub, ok_sub in group_results:
                    V_all[g_rows] = V_sub
                    success_flags[g_rows] = ok_sub

                # Reserve SLSQP only when non-linear conductivity is enabled or method='slsqp'
                if use_nonlinear:
                    def _run_slsqp_depth(i: int) -> Tuple[int, np.ndarray, bool]:
                        log_values = {
                            "GR": gr_arr[i] if finite_mask[i, 0] else None,
                            "NPHI": nphi_arr[i] if finite_mask[i, 1] else None,
                            "RHOB": rhob_arr[i] if finite_mask[i, 2] else None,
                            "PEF": pef_arr[i] if finite_mask[i, 3] else None,
                            "DTC": dtc_arr[i] if finite_mask[i, 4] else None,
                        }
                        if active_cond_model is not None and np.isfinite(rt_arr[i]) and rt_arr[i] > 0:
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

            # Vectorized log reconstruction and error calculation
            recon_matrix = V_all @ C_full.T  # (n_samples, 5)
            gr_reconstructed = np.where(finite_mask[:, 0], recon_matrix[:, 0], np.nan)
            nphi_reconstructed = np.where(finite_mask[:, 1], recon_matrix[:, 1], np.nan)
            rhob_reconstructed = np.where(finite_mask[:, 2], recon_matrix[:, 2], np.nan)
            pef_reconstructed = np.where(finite_mask[:, 3], recon_matrix[:, 3], np.nan)
            dtc_reconstructed = np.where(finite_mask[:, 4], recon_matrix[:, 4], np.nan)

            abs_diffs = np.where(
                finite_mask & valid_rows[:, None],
                np.abs(Y_linear - recon_matrix),
                0.0,
            )
            linear_valid_counts = np.sum(finite_mask, axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                average_errors = np.where(
                    valid_rows & (linear_valid_counts > 0),
                    np.sum(abs_diffs, axis=1) / linear_valid_counts,
                    np.nan,
                )

            porosity_constructed = np.where(
                valid_rows, np.sum(V_all[:, n_minerals:], axis=1), np.nan
            )

            # Create output DataFrame
            output_data = {}

            mineral_column_mapping = {
                mineral: Config.MINERALS_NAME_MAPPING.get(mineral, f"V{mineral}")
                for mineral in self.minerals
            }
            for j, mineral in enumerate(self.minerals):
                column_name = mineral_column_mapping.get(mineral, f"V{mineral}")
                output_data[column_name] = V_all[:, j]

            fluid_column_mapping = {"OIL": "VOIL", "GAS": "VGAS", "WATER": "VWATER"}
            for j, fluid in enumerate(self.fluids):
                column_name = fluid_column_mapping.get(fluid, f"V{fluid}")
                output_data[column_name] = V_all[:, n_minerals + j]

            output_data["PHIT_CONSTRUCTED"] = porosity_constructed
            output_data.update(
                {
                    "AVG_ERROR": average_errors,
                    "OPTIMIZER_SUCCESS": success_flags,
                    "GR_RECONSTRUCTED": gr_reconstructed,
                    "NPHI_RECONSTRUCTED": nphi_reconstructed,
                    "RHOB_RECONSTRUCTED": rhob_reconstructed,
                    "PEF_RECONSTRUCTED": pef_reconstructed,
                    "DTC_RECONSTRUCTED": dtc_reconstructed,
                }
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
        porosity_method: str = "density",
        porosity_endpoints: Optional[Dict] = None,
        conductivity_model: Optional[str] = None,
        conductivity_params: Optional[Dict] = None,
        solver: str = "auto",
        n_jobs: int = 1,
    ):
        super().__init__(
            minerals=minerals,
            fluid_properties=fluid_properties,
            porosity_method=porosity_method,
            porosity_endpoints=porosity_endpoints,
            conductivity_model=conductivity_model,
            conductivity_params=conductivity_params,
            solver=solver,
            n_jobs=n_jobs,
        )
        self.A = A
        self.b = b
