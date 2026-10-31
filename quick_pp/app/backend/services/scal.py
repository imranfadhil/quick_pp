"""Special Core Analysis (SCAL) & Relative Permeability API services.

Endpoints for two-phase relative permeability evaluation (Corey, Brooks-Corey, LET),
experimental SCAL curve fitting, Eclipse/Intersect SWOF table generation,
laboratory-to-reservoir capillary pressure conversion, and core plug corrections.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from quick_pp.scal import (
    brooks_corey_relperm,
    corey_relperm,
    correct_core_data,
    fit_relperm,
    generate_swof_table,
    let_relperm,
    pc_lab_to_reservoir,
)

router = APIRouter(prefix="/scal", tags=["SCAL"])


class RelPermEvalInput(BaseModel):
    model: str = Field(
        "let",
        description="Relative permeability formulation ('corey', 'brooks_corey', or 'let')",
    )
    sw: Optional[List[float]] = Field(
        default=None,
        description="Water saturation array (v/v). If omitted, generates 25 points from swirr to 1 - sor.",
    )
    swirr: float = Field(0.15, ge=0.0, lt=1.0, description="Irreducible water saturation")
    sor: float = Field(0.20, ge=0.0, lt=1.0, description="Residual hydrocarbon saturation")
    sorw: Optional[float] = Field(
        default=None, description="Optional alias for residual oil saturation (sor)"
    )
    krw_max: float = Field(0.30, gt=0.0, le=1.0, description="Endpoint water rel-perm at 1 - sor")
    kro_max: float = Field(0.90, gt=0.0, le=1.0, description="Endpoint hydrocarbon rel-perm at swirr")
    nw: float = Field(3.0, gt=0.0, description="Corey water exponent")
    no: float = Field(2.5, gt=0.0, description="Corey hydrocarbon exponent")
    lam: float = Field(2.0, gt=0.0, description="Brooks-Corey pore size distribution index")
    lw: float = Field(2.0, gt=0.0, description="LET water lower parameter L_w")
    ew: float = Field(1.5, gt=0.0, description="LET water elevation parameter E_w")
    tw: float = Field(2.0, gt=0.0, description="LET water top parameter T_w")
    lo: float = Field(2.0, gt=0.0, description="LET hydrocarbon lower parameter L_o")
    eo: float = Field(1.5, gt=0.0, description="LET hydrocarbon elevation parameter E_o")
    to: float = Field(2.0, gt=0.0, description="LET hydrocarbon top parameter T_o")


class RelPermFitInput(BaseModel):
    model: str = Field(
        "corey",
        description="Model to fit ('corey' or 'let')",
    )
    sw: List[float] = Field(..., description="Measured water saturation values (v/v)")
    krw: Optional[List[float]] = Field(
        default=None, description="Measured water relative permeability values"
    )
    kro: Optional[List[float]] = Field(
        default=None, description="Measured hydrocarbon relative permeability values"
    )
    swirr: Optional[float] = Field(
        default=None, description="Optional fixed irreducible water saturation"
    )
    sor: Optional[float] = Field(
        default=None, description="Optional fixed residual hydrocarbon saturation"
    )


class SWOFTableInput(BaseModel):
    n_points: int = Field(21, ge=5, le=200, description="Number of saturation table rows")
    model: str = Field(
        "corey",
        description="Relative permeability model ('corey', 'brooks_corey', or 'let')",
    )
    swirr: float = Field(0.15, ge=0.0, lt=1.0, description="Irreducible water saturation")
    sor: float = Field(0.20, ge=0.0, lt=1.0, description="Residual oil saturation")
    krw_max: float = Field(0.30, gt=0.0, le=1.0, description="Endpoint krw at 1 - sor")
    kro_max: float = Field(0.90, gt=0.0, le=1.0, description="Endpoint kro at swirr")
    nw: float = Field(3.0, gt=0.0, description="Corey water exponent")
    no: float = Field(2.5, gt=0.0, description="Corey oil exponent")
    lam: float = Field(2.0, gt=0.0, description="Brooks-Corey lambda exponent")
    lw: float = Field(2.0, gt=0.0, description="LET water L_w")
    ew: float = Field(1.5, gt=0.0, description="LET water E_w")
    tw: float = Field(2.0, gt=0.0, description="LET water T_w")
    lo: float = Field(2.0, gt=0.0, description="LET oil L_o")
    eo: float = Field(1.5, gt=0.0, description="LET oil E_o")
    to: float = Field(2.0, gt=0.0, description="LET oil T_o")
    pc_entry: float = Field(1.5, ge=0.0, description="Capillary entry pressure (psi)")
    pc_exponent: float = Field(1.5, gt=0.0, description="Capillary pressure pore-throat exponent")
    pc_max: float = Field(50.0, gt=0.0, description="Maximum capillary pressure at swirr (psi)")


class CoreCorrectionsInput(BaseModel):
    net_confining_stress: Optional[Union[float, str]] = Field(
        default=2500.0,
        description="Effective reservoir net confining stress (psi) or column name",
    )
    mean_pore_pressure: Optional[Union[float, str]] = Field(
        default=30.0,
        description="Mean flowing gas pressure (psi) for Klinkenberg correction or column name",
    )
    cbw: Optional[Union[bool, float, str]] = Field(
        default=None,
        description="Clay-bound water fraction, column name, or True to infer from PHIT - PHIE",
    )
    compaction_coeff: float = Field(
        4e-6, gt=0.0, description="Pore volume compaction coefficient (1/psi)"
    )
    perm_compaction_coeff: Optional[float] = Field(
        default=None,
        description="Permeability compaction coefficient (1/psi); defaults to 5 * compaction_coeff if omitted",
    )
    ambient_stress: float = Field(
        default=400.0, ge=0.0, description="Ambient laboratory confining stress (psi)"
    )
    compaction_method: str = Field(
        default="exponential",
        description="Overburden compaction model: 'exponential' or 'power_law'",
    )
    klinkenberg_method: str = Field(
        default="jones",
        description="Klinkenberg slippage correlation: 'jones' or 'heithaus'",
    )
    oven_dried: bool = Field(
        False, description="Whether core plugs were oven-dried prior to Dean-Stark"
    )
    replace_ori: bool = Field(
        False, description="Overwrite CPORE/CPERM in-place instead of preserving _AMB/_AIR columns"
    )
    data: List[Dict[str, Any]] = Field(..., description="Core/log records")


class PcLabToReservoirInput(BaseModel):
    pc_lab: List[float] = Field(..., description="Laboratory capillary pressure (psi)")
    ift_lab: float = Field(72.0, gt=0.0, description="Laboratory interfacial tension (dynes/cm)")
    theta_lab: float = Field(0.0, ge=0.0, le=180.0, description="Laboratory contact angle (deg)")
    ift_res: float = Field(30.0, gt=0.0, description="Reservoir interfacial tension (dynes/cm)")
    theta_res: float = Field(30.0, ge=0.0, le=180.0, description="Reservoir contact angle (deg)")
    phi_amb: Optional[Union[float, List[float]]] = Field(
        default=None, description="Optional ambient porosity (v/v) for Leverett-J stress correction"
    )
    phi_res: Optional[Union[float, List[float]]] = Field(
        default=None, description="Optional reservoir porosity (v/v) for Leverett-J stress correction"
    )
    k_amb: Optional[Union[float, List[float]]] = Field(
        default=None, description="Optional ambient permeability (mD) for Leverett-J stress correction"
    )
    k_res: Optional[Union[float, List[float]]] = Field(
        default=None, description="Optional reservoir permeability (mD) for Leverett-J stress correction"
    )


@router.post(
    "/relperm",
    summary="Evaluate Two-Phase Relative Permeability Curves (Corey, Brooks-Corey, LET)",
    operation_id="evaluate_scal_relperm",
)
async def evaluate_relperm_endpoint(inputs: RelPermEvalInput):
    """Compute water and hydrocarbon relative permeability curves using Corey, Brooks-Corey, or LET."""
    try:
        sor_val = inputs.sorw if inputs.sorw is not None else inputs.sor
        if inputs.sw is not None:
            sw_arr = np.asarray(inputs.sw, dtype=float)
        else:
            sw_arr = np.linspace(inputs.swirr, 1.0 - sor_val, 25)

        m = inputs.model.strip().lower()
        if m == "corey":
            res = corey_relperm(
                sw_arr,
                swirr=inputs.swirr,
                sor=sor_val,
                krw_max=inputs.krw_max,
                kro_max=inputs.kro_max,
                nw=inputs.nw,
                no=inputs.no,
            )
        elif m in ("brooks_corey", "brooks-corey", "bc"):
            res = brooks_corey_relperm(
                sw_arr,
                swirr=inputs.swirr,
                sor=sor_val,
                krw_max=inputs.krw_max,
                kro_max=inputs.kro_max,
                lam=inputs.lam,
            )
        elif m == "let":
            res = let_relperm(
                sw_arr,
                swirr=inputs.swirr,
                sor=sor_val,
                krw_max=inputs.krw_max,
                kro_max=inputs.kro_max,
                lw=inputs.lw,
                ew=inputs.ew,
                tw=inputs.tw,
                lo=inputs.lo,
                eo=inputs.eo,
                to=inputs.to,
            )
        else:
            raise ValueError(
                f"Unsupported model '{inputs.model}'. Choose 'corey', 'brooks_corey', or 'let'."
            )

        df_out = pd.DataFrame(
            {
                "sw": np.asarray(res["sw"], dtype=float),
                "swn": np.asarray(res["swn"], dtype=float),
                "krw": np.asarray(res["krw"], dtype=float),
                "kro": np.asarray(res["kro"], dtype=float),
                "fw": np.asarray(res["fw"], dtype=float),
            }
        )
        return {
            "model": m,
            "data": df_out.replace({np.nan: None}).to_dict(orient="records"),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error evaluating relative permeability: {e}"
        ) from e


@router.post(
    "/fit_relperm",
    summary="Fit Corey or LET Relative Permeability Model to SCAL Data",
    operation_id="fit_scal_relperm",
)
async def fit_relperm_endpoint(inputs: RelPermFitInput):
    """Calibrate two-phase relative permeability parameters to experimental SCAL plug data."""
    try:
        res = fit_relperm(
            sw=inputs.sw,
            krw=inputs.krw,
            kro=inputs.kro,
            model=inputs.model,
            swirr=inputs.swirr,
            sor=inputs.sor,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error fitting relative permeability: {e}"
        ) from e


@router.post(
    "/swof",
    summary="Generate Eclipse / Intersect SWOF Saturation Table",
    operation_id="generate_scal_swof_table",
)
async def generate_swof_endpoint(inputs: SWOFTableInput):
    """Generate a two-phase water-oil relative permeability & capillary pressure SWOF table."""
    try:
        res = generate_swof_table(
            n_points=inputs.n_points,
            model=inputs.model,
            swirr=inputs.swirr,
            sor=inputs.sor,
            krw_max=inputs.krw_max,
            kro_max=inputs.kro_max,
            nw=inputs.nw,
            no=inputs.no,
            lam=inputs.lam,
            lw=inputs.lw,
            ew=inputs.ew,
            tw=inputs.tw,
            lo=inputs.lo,
            eo=inputs.eo,
            to=inputs.to,
            pc_entry=inputs.pc_entry,
            pc_exponent=inputs.pc_exponent,
            pc_max=inputs.pc_max,
        )
        return {
            "table": res["table"].replace({np.nan: None}).to_dict(orient="records"),
            "swof_keyword": res["swof_keyword"],
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error generating SWOF table: {e}"
        ) from e


@router.post(
    "/core_corrections",
    summary="Apply Overburden, Klinkenberg, and Clay-Bound Water Corrections to Core Data",
    operation_id="apply_scal_core_corrections",
)
async def apply_core_corrections_endpoint(inputs: CoreCorrectionsInput):
    """Correct ambient laboratory core plug porosity, permeability, and saturation to reservoir conditions."""
    try:
        df = pd.DataFrame(inputs.data)
        eff_perm_coeff = (
            inputs.perm_compaction_coeff
            if inputs.perm_compaction_coeff is not None
            else (float(inputs.compaction_coeff) * 5.0 if inputs.compaction_coeff else 2.0e-5)
        )
        corr_df = correct_core_data(
            df,
            net_confining_stress=inputs.net_confining_stress,
            ambient_stress=inputs.ambient_stress,
            mean_pore_pressure=inputs.mean_pore_pressure,
            cbw=inputs.cbw,
            compaction_coeff=inputs.compaction_coeff,
            perm_compaction_coeff=eff_perm_coeff,
            compaction_method=inputs.compaction_method,
            klinkenberg_method=inputs.klinkenberg_method,
            oven_dried=inputs.oven_dried,
            replace_ori=inputs.replace_ori,
        )
        return {"data": corr_df.replace({np.nan: None}).to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error applying core corrections: {e}"
        ) from e


@router.post(
    "/pc_lab_to_reservoir",
    summary="Convert Laboratory Capillary Pressure to Reservoir Conditions",
    operation_id="convert_pc_lab_to_reservoir",
)
async def convert_pc_lab_to_reservoir_endpoint(inputs: PcLabToReservoirInput):
    """Scale laboratory capillary pressure (MICP / porous plate / centrifuge) to reservoir IFT and wettability."""
    try:
        pc_arr = np.asarray(inputs.pc_lab, dtype=float)
        pc_res = pc_lab_to_reservoir(
            pc_arr,
            ift_lab=inputs.ift_lab,
            theta_lab=inputs.theta_lab,
            ift_res=inputs.ift_res,
            theta_res=inputs.theta_res,
            phi_amb=inputs.phi_amb,
            phi_res=inputs.phi_res,
            k_amb=inputs.k_amb,
            k_res=inputs.k_res,
        )
        unit_res = float(
            np.asarray(
                pc_lab_to_reservoir(
                    1.0,
                    ift_lab=inputs.ift_lab,
                    theta_lab=inputs.theta_lab,
                    ift_res=inputs.ift_res,
                    theta_res=inputs.theta_res,
                    phi_amb=inputs.phi_amb if isinstance(inputs.phi_amb, (int, float)) else None,
                    phi_res=inputs.phi_res if isinstance(inputs.phi_res, (int, float)) else None,
                    k_amb=inputs.k_amb if isinstance(inputs.k_amb, (int, float)) else None,
                    k_res=inputs.k_res if isinstance(inputs.k_res, (int, float)) else None,
                )
            ).flat[0]
        )
        return {
            "pc_res": [float(x) for x in np.asarray(pc_res).flatten()],
            "scale_factor": round(unit_res, 6),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Error converting capillary pressure: {e}"
        ) from e

