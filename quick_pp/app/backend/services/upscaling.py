"""Log Blocking & Upscaling API services for Static Geomodeling & Dynamic Simulation Handover.

Endpoints for blocking/upscaling fine-scale CPI well logs onto cellular grids and
evaluating Cardwell-Parsons (1945) directional permeability bounds.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quick_pp.upscaling import (
    cardwell_parsons_bounds,
    expand_blocked_logs,
    upscale_logs,
)

router = APIRouter(prefix="/upscaling", tags=["Upscaling"])


class UpscaleLogsInput(BaseModel):
    cell_size: Optional[float] = Field(
        2.0, description="Target grid block thickness in depth units"
    )
    grid_edges: Optional[List[float]] = Field(
        default=None,
        description="Optional explicit sequence of depth boundaries [top_0, ..., bot_M]",
    )
    by_zone: bool = Field(
        True, description="Respect geological zone boundaries when blocking"
    )
    depth_col: str = Field("DEPTH", description="Depth column name")
    zone_col: Optional[str] = Field("ZONES", description="Zone column name")
    perm_method: str = Field(
        "geometric",
        description="Primary permeability upscaling scheme ('geometric', 'arithmetic', 'harmonic', 'power_law')",
    )
    omega: float = Field(
        0.33, ge=-1.0, le=1.0, description="Cardwell-Parsons power-law exponent"
    )
    min_perm: Optional[float] = Field(
        default=None, description="Optional minimum permeability floor (mD)"
    )
    net_only: bool = Field(
        False, description="Average reservoir properties only over net flagged samples"
    )
    net_flag_type: str = Field(
        "reservoir", description="Net flag filter ('reservoir', 'pay', or 'rock')"
    )
    cutoffs: Dict[str, float] = Field(
        default_factory=lambda: {"VSHALE": 0.4, "PHIT": 0.01, "SWT": 0.9},
        description="Cutoffs for cell NTG and pay flags",
    )
    include_step_log: bool = Field(
        True, description="Also return blocked curves expanded onto fine-scale depth rows"
    )
    data: List[dict] = Field(..., description="Fine-scale well log records")


class PermeabilityBoundsInput(BaseModel):
    perm: List[float] = Field(..., description="Fine-scale permeability values (mD)")
    h: Optional[List[float]] = Field(
        default=None, description="Optional per-sample thickness weights"
    )
    omega: float = Field(
        0.33, ge=-1.0, le=1.0, description="Cardwell-Parsons power-law exponent"
    )
    min_perm: Optional[float] = Field(
        default=None, description="Optional positive permeability floor (mD)"
    )


@router.post(
    "/block_logs",
    summary="Block and Upscale Fine-Scale Well Logs to Cellular Grid Resolution",
    operation_id="block_and_upscale_logs",
)
@router.post(
    "/upscale_logs",
    summary="Upscale Fine-Scale Well Logs to Cellular Grid Resolution",
    operation_id="upscale_well_logs",
)
async def run_log_upscaling(inputs: UpscaleLogsInput):
    """Upscale fine-scale CPI well logs onto cellular geomodel/simulator grid blocks."""
    try:
        df = pd.DataFrame(inputs.data)
        blocked_df = upscale_logs(
            df,
            cell_size=inputs.cell_size,
            grid_edges=inputs.grid_edges,
            by_zone=inputs.by_zone,
            depth_col=inputs.depth_col,
            zone_col=inputs.zone_col,
            perm_method=inputs.perm_method,
            omega=inputs.omega,
            min_perm=inputs.min_perm,
            cutoffs=inputs.cutoffs,
            net_only=inputs.net_only,
            net_flag_type=inputs.net_flag_type,
        )
        response = {
            "blocked": blocked_df.replace({np.nan: None}).to_dict(orient="records"),
        }
        if inputs.include_step_log:
            step_df = expand_blocked_logs(df, blocked_df, depth_col=inputs.depth_col)
            response["step_log"] = step_df.replace({np.nan: None}).to_dict(
                orient="records"
            )
        return response
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error upscaling logs: {e}"
        ) from e


@router.post(
    "/permeability_bounds",
    summary="Calculate Cardwell-Parsons Directional Permeability Bounds",
    operation_id="calculate_permeability_bounds",
)
async def calculate_permeability_bounds_endpoint(inputs: PermeabilityBoundsInput):
    """Compute harmonic (kv), geometric (keff), power-law, arithmetic (kh), and kv/kh anisotropy ratio."""
    try:
        bounds = cardwell_parsons_bounds(
            perm=inputs.perm,
            h=inputs.h,
            omega=inputs.omega,
            min_perm=inputs.min_perm,
        )
        return {
            k: (float(v) if v is not None and np.isfinite(v) else None)
            for k, v in bounds.items()
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error calculating permeability bounds: {e}"
        ) from e

