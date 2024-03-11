from fastapi import APIRouter, Body
import pandas as pd

from api.schemas.permeability_choo import inputData as perm_ch_inputData, EXAMPLE as PERM_CH_EXAMPLE
from api.schemas.permeability_others import inputData as perm_others_inputData, EXAMPLE as PERM_OTHERS_EXAMPLE
from quick_pp.permeability import (
    choo_permeability, timur_permeability, coates_permeability, kozeny_carman_permeability, tixier_permeability
)

router = APIRouter(prefix="/permeability", tags=["Permeability"])


@router.post("/choo", description="Estimate Choo's permeability.")
async def estimate_perm_choo(inputs: perm_ch_inputData = Body(..., example=PERM_CH_EXAMPLE)):

    input_dict = inputs.model_dump()

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    perm = choo_permeability(input_df['vclw'], input_df['vsilt'], input_df['phit'])
    return_dict = pd.DataFrame({'PERM': perm.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/timur", description="Estimate Timur's permeability.")
async def estimate_perm_timur(inputs: perm_others_inputData = Body(..., example=PERM_OTHERS_EXAMPLE)):

    input_dict = inputs.model_dump()

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    perm = timur_permeability(input_df['phit'], input_df['swirr'])
    return_dict = pd.DataFrame({'PERM': perm.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/tixier", description="Estimate Tixier's permeability.")
async def estimate_perm_tixier(inputs: perm_others_inputData = Body(..., example=PERM_OTHERS_EXAMPLE)):

    input_dict = inputs.model_dump()

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    perm = tixier_permeability(input_df['phit'], input_df['swirr'])
    return_dict = pd.DataFrame({'PERM': perm.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/coates", description="Estimate Coates's permeability.")
async def estimate_perm_coates(inputs: perm_others_inputData = Body(..., example=PERM_OTHERS_EXAMPLE)):

    input_dict = inputs.model_dump()

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    perm = coates_permeability(input_df['phit'], input_df['swirr'])
    return_dict = pd.DataFrame({'PERM': perm.ravel()}).to_dict(orient='records')
    return return_dict


@router.post("/kozeny_carman", description="Estimate Kozeny-Carman's permeability.")
async def estimate_perm_kozeny_carman(inputs: perm_others_inputData = Body(..., example=PERM_OTHERS_EXAMPLE)):

    input_dict = inputs.model_dump()

    # Get the data
    input_df = pd.DataFrame.from_records(input_dict['data'])
    perm = kozeny_carman_permeability(input_df['phit'], input_df['swirr'])
    return_dict = pd.DataFrame({'PERM': perm.ravel()}).to_dict(orient='records')
    return return_dict
