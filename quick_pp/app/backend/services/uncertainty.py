"""Probabilistic Uncertainty Analysis API services.

Endpoints for Latin Hypercube Monte Carlo volumetric uncertainty, log-level CPI uncertainty
propagation, and stochastic reservoir summary calculations.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quick_pp.uncertainty import (
    calc_stochastic_reservoir_summary,
    mc_volumetric_uncertainty,
    run_log_uncertainty,
)

router = APIRouter(prefix="/uncertainty", tags=["Uncertainty"])

BoundSpec = Union[Tuple[float, ...], List[float], float]


class VolumetricUncertaintyInput(BaseModel):
    area_bound: List[float] = Field(
        default_factory=lambda: [500.0, 2500.0, 1500.0, 400.0],
        description="Area bound (min, max) or (min, max, mean, std) in acres",
    )
    thickness_bound: List[float] = Field(
        default_factory=lambda: [20.0, 100.0, 50.0, 15.0],
        description="Net pay thickness bound (min, max) or (min, max, mean, std) in ft",
    )
    porosity_bound: List[float] = Field(
        default_factory=lambda: [0.10, 0.28, 0.18, 0.03],
        description="Porosity bound (min, max) or (min, max, mean, std) in fraction",
    )
    water_saturation_bound: List[float] = Field(
        default_factory=lambda: [0.15, 0.55, 0.30, 0.08],
        description="Water saturation bound (min, max) or (min, max, mean, std) in fraction",
    )
    volume_factor_bound: List[float] = Field(
        default_factory=lambda: [1.10, 1.40],
        description="Formation volume factor Bo bound (min, max) or (min, max, mean, std)",
    )
    recovery_factor_bound: Optional[List[float]] = Field(
        default=None,
        description="Optional recovery factor bound (min, max) or (min, max, mean, std)",
    )
    n_try: int = Field(5000, ge=10, le=100000, description="Number of LHS Monte Carlo trials")
    percentile: List[float] = Field(
        default_factory=lambda: [10.0, 50.0, 90.0],
        description="Percentiles to compute",
    )
    seed: Optional[int] = Field(42, description="Random seed for reproducibility")


class LogUncertaintyInput(BaseModel):
    model: str = Field("ssc", description="Lithology model ('ssc' or 'carbonate')")
    saturation_method: str = Field(
        "waxman_smits", description="Saturation equation ('waxman_smits' or 'archie')"
    )
    n_samples: int = Field(
        100, ge=5, le=1000, description="Number of Latin Hypercube realizations"
    )
    param_bounds: Optional[Dict[str, BoundSpec]] = Field(
        default=None,
        description="Optional custom parameter bounds for 'a', 'm', 'n', 'rw', 'rho_ma', 'rho_sh'",
    )
    carbonate_model: str = Field(
        "double", description="'single' or 'double' when model='carbonate'"
    )
    carbonate_type: str = Field(
        "limestone", description="'limestone' or 'dolostone' when carbonate_model='single'"
    )
    water_salinity: Optional[float] = Field(
        default=None, description="Optional water salinity (ppm) for baseline Rw"
    )
    seed: Optional[int] = Field(42, description="Random seed for LHS sampler")
    data: List[dict] = Field(..., description="Well log records (DEPTH, NPHI, RHOB, RT, GR, etc.)")


class StochasticReservoirSummaryInput(BaseModel):
    cutoffs: Dict[str, float] = Field(
        default_factory=lambda: {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9},
        description="Cutoffs for VSHALE, PHIT, and SWT",
    )
    model: str = Field("ssc", description="Lithology model ('ssc' or 'carbonate')")
    saturation_method: str = Field(
        "waxman_smits", description="Saturation equation ('waxman_smits' or 'archie')"
    )
    n_samples: int = Field(
        100, ge=5, le=1000, description="Number of Latin Hypercube realizations"
    )
    param_bounds: Optional[Dict[str, BoundSpec]] = Field(
        default=None,
        description="Optional custom parameter bounds for 'a', 'm', 'n', 'rw', 'rho_ma', 'rho_sh'",
    )
    area_bound: List[float] = Field(
        default_factory=lambda: [500.0, 2500.0, 1500.0, 400.0],
        description="Area bound in acres for STOIIP",
    )
    volume_factor_bound: List[float] = Field(
        default_factory=lambda: [1.1, 1.4],
        description="Formation volume factor bound for STOIIP",
    )
    seed: Optional[int] = Field(42, description="Random seed for LHS sampler")
    data: List[dict] = Field(..., description="Well log records")


@router.post(
    "/volumetric",
    summary="Run Latin Hypercube Volumetric Uncertainty Simulation",
    operation_id="run_volumetric_uncertainty",
)
async def run_volumetric_uncertainty_endpoint(inputs: VolumetricUncertaintyInput):
    """Run vectorized Latin Hypercube Monte Carlo simulation on reservoir volumetric parameters."""
    try:
        res = mc_volumetric_uncertainty(
            area_bound=tuple(inputs.area_bound),
            thickness_bound=tuple(inputs.thickness_bound),
            porosity_bound=tuple(inputs.porosity_bound),
            water_saturation_bound=tuple(inputs.water_saturation_bound),
            volume_factor_bound=tuple(inputs.volume_factor_bound),
            recovery_factor_bound=(
                tuple(inputs.recovery_factor_bound)
                if inputs.recovery_factor_bound is not None
                else None
            ),
            n_try=inputs.n_try,
            percentile=inputs.percentile,
            seed=inputs.seed,
            plot=False,
        )
        return {
            "percentiles": res["percentiles"],
            "mean": res["mean"],
            "std": res["std"],
            "n_try": inputs.n_try,
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error running volumetric uncertainty: {e}"
        ) from e


@router.post(
    "/log",
    summary="Run Log-Level Latin Hypercube Petrophysical Uncertainty Propagation",
    operation_id="run_log_uncertainty_endpoint",
)
async def run_log_uncertainty_endpoint(inputs: LogUncertaintyInput):
    """Propagate parameter uncertainty across well logs to produce P10/P50/P90 CPI curves."""
    try:
        df = pd.DataFrame(inputs.data)
        cpi_df = run_log_uncertainty(
            df,
            model=inputs.model,
            saturation_method=inputs.saturation_method,
            n_samples=inputs.n_samples,
            param_bounds=inputs.param_bounds,
            carbonate_model=inputs.carbonate_model,
            carbonate_type=inputs.carbonate_type,
            water_salinity=inputs.water_salinity,
            seed=inputs.seed,
            return_realizations=False,
        )
        return {"data": cpi_df.replace({np.nan: None}).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error running log uncertainty: {e}"
        ) from e


@router.post(
    "/reservoir_summary",
    summary="Calculate Stochastic P10/P50/P90 Reservoir Summary and Volumetrics",
    operation_id="calculate_stochastic_reservoir_summary",
)
async def calculate_stochastic_reservoir_summary_endpoint(
    inputs: StochasticReservoirSummaryInput,
):
    """Run Latin Hypercube log uncertainty and compute P10/P50/P90 CPI curves, zone summaries, and STOIIP."""
    try:
        df = pd.DataFrame(inputs.data)
        res = calc_stochastic_reservoir_summary(
            df,
            cutoffs=inputs.cutoffs,
            model=inputs.model,
            saturation_method=inputs.saturation_method,
            n_samples=inputs.n_samples,
            param_bounds=inputs.param_bounds,
            area_bound=tuple(inputs.area_bound),
            volume_factor_bound=tuple(inputs.volume_factor_bound),
            seed=inputs.seed,
        )
        cpi_cols = [
            c
            for c in [
                "DEPTH",
                "ZONES",
                "SW_P10",
                "SW_P50",
                "SW_P90",
                "PHIT_P10",
                "PHIT_P50",
                "PHIT_P90",
                "PHIE_P10",
                "PHIE_P50",
                "PHIE_P90",
                "VCLAY_P10",
                "VCLAY_P50",
                "VCLAY_P90",
                "BVHC_P10",
                "BVHC_P50",
                "BVHC_P90",
            ]
            if c in res["cpi_df"].columns
        ]
        return {
            "cpi": res["cpi_df"][cpi_cols].replace({np.nan: None}).to_dict(orient="records"),
            "summary": res["summary_df"].replace({np.nan: None}).to_dict(orient="records"),
            "volumetrics": res["volumetrics"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error calculating stochastic reservoir summary: {e}"
        ) from e

