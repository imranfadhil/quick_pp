from fastapi import APIRouter, HTTPException
import pandas as pd
from typing import List, Dict
import logging

from quick_pp.api.schemas.permeability_choo import InputData as PermChInputData
from quick_pp.api.schemas.permeability_others import InputData as PermOthersInputData
from quick_pp.permeability import (
    choo_permeability, timur_permeability, coates_permeability, kozeny_carman_permeability, tixier_permeability
)

router = APIRouter(prefix="/permeability", tags=["Permeability"])
logger = logging.getLogger("api.services.permeability")


@router.post(
    "/choo",
    summary="Estimate Choo permeability",
    description=(
        "Estimate permeability using the Choo empirical model. "
        "Requires volume of clay (vcld), volume of silt (vsilt), and total porosity (phit) measurements.\n"
        "Input model: PermChInputData (see quick_pp.api.schemas.permeability_choo.InputData).\n"
        "Request body must be a JSON object with the following field:\n"
        "- data: list of objects, each with keys 'vcld', 'vsilt', 'phit' (all float, required)\n"
        "Example (truncated): { 'data': [ {'vcld': 0.25, 'vsilt': 0.10, 'phit': 0.18}, ... ] }"
    ),
    operation_id="estimate_choo_permeability",
)
async def estimate_perm_choo(inputs: PermChInputData) -> List[Dict[str, float]]:
    """
    Estimate permeability using the Choo empirical model.

    This endpoint receives input data for multiple samples, including volume of clay (vcld),
    volume of silt (vsilt), and total porosity (phit), and computes the permeability for each
    sample using the Choo permeability correlation.

    Args:
        inputs (PermChInputData): Pydantic model containing a list of input records under the
            'data' key. Each record must include 'vcld', 'vsilt', and 'phit' fields.
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing the estimated permeability
        ('PERM') for the corresponding input sample.
    Technical Details:
        - The input data is converted into a pandas DataFrame for vectorized processing.
        - The `choo_permeability` function is applied to the input columns to compute permeability
          values for all samples in a vectorized manner.
        - The result is reshaped and returned as a list of dictionaries, suitable for JSON
          serialization and API response.
        - This endpoint is asynchronous and designed for use with FastAPI.
        - Any errors during processing are logged and returned as HTTP 400 errors.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict['data'])
        perm = choo_permeability(
            input_df['vcld'], input_df['vsilt'], input_df['phit']
        )
        df_result = pd.DataFrame({'PERM': perm.ravel()})
        # Ensure output is List[Dict[str, float]]
        return [
            {str(k): float(v) for k, v in row.items()}
            for row in df_result.to_dict(orient='records')
        ]
    except Exception as e:
        logger.error(f"Error in estimate_perm_choo: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/timur",
    summary="Estimate Timur permeability",
    description=(
        "Estimate permeability using the Timur empirical model. "
        "Requires porosity (phit) and irreducible water saturation (swirr) measurements.\n"
        "Input model: PermOthersInputData (see quick_pp.api.schemas.permeability_others.InputData).\n"
        "Request body must be a JSON object with the following field:\n"
        "- data: list of objects, each with keys 'phit' (float, required) and 'swirr' (float, required)\n"
        "Example (truncated): { 'data': [ {'phit': 0.18, 'swirr': 0.25}, ... ] }"
    ),
    operation_id="estimate_timur_permeability",
)
async def estimate_perm_timur(inputs: PermOthersInputData) -> List[Dict[str, float]]:
    """
    Estimate permeability using the Timur empirical equation.

    This endpoint receives input data containing porosity (phit) and irreducible water saturation (swirr) values,
    applies the Timur permeability estimation method, and returns the calculated permeability values.

    Args:
        inputs (PermOthersInputData): Input data model containing a list of records with 'phit' and 'swirr'.
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing the estimated permeability value
        under the key 'PERM'.
    Technical Details:
        - The Timur equation is an empirical relationship used in petrophysics to estimate permeability from
          porosity and irreducible water saturation.
        - The function expects the input data to be structured as a list of records, each with 'phit' (porosity)
          and 'swirr' (irreducible water saturation).
        - The permeability calculation is performed using the `timur_permeability` function, which should
          implement the Timur equation in a vectorized fashion for performance.
        - The result is formatted as a list of dictionaries for API response compatibility.
        - Errors are logged and returned as HTTP 400 errors.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict['data'])
        perm = timur_permeability(input_df['phit'], input_df['swirr'])
        df_result = pd.DataFrame({'PERM': perm.ravel()})
        return [
            {str(k): float(v) for k, v in row.items()}
            for row in df_result.to_dict(orient='records')
        ]
    except Exception as e:
        logger.error(f"Error in estimate_perm_timur: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/tixier",
    summary="Estimate Tixier permeability",
    description=(
        "Estimate permeability using the Tixier empirical model. "
        "Requires porosity (phit) and irreducible water saturation (swirr) measurements.\n"
        "Input model: PermOthersInputData (see quick_pp.api.schemas.permeability_others.InputData).\n"
        "Request body must be a JSON object with the following field:\n"
        "- data: list of objects, each with keys 'phit' (float, required) and 'swirr' (float, required)\n"
        "Example (truncated): { 'data': [ {'phit': 0.18, 'swirr': 0.25}, ... ] }"
    ),
    operation_id="estimate_tixier_permeability",
)
async def estimate_perm_tixier(inputs: PermOthersInputData) -> List[Dict[str, float]]:
    """
    Estimate permeability using the Tixier empirical correlation.

    This endpoint receives input data containing porosity (phit) and irreducible water saturation (swirr),
    applies the Tixier permeability estimation method, and returns the calculated permeability values.

    Args:
        inputs (PermOthersInputData): Input data model containing a list of records with 'phit' and 'swirr'.
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing the estimated permeability ('PERM')
        for the corresponding input record.
    Technical Details:
        - The input data is converted to a pandas DataFrame for processing.
        - The `tixier_permeability` function is used to compute permeability based on the Tixier empirical formula.
        - The result is formatted as a list of dictionaries for API response compatibility.
        - Errors are logged and returned as HTTP 400 errors.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict['data'])
        perm = tixier_permeability(input_df['phit'], input_df['swirr'])
        df_result = pd.DataFrame({'PERM': perm.ravel()})
        return [
            {str(k): float(v) for k, v in row.items()}
            for row in df_result.to_dict(orient='records')
        ]
    except Exception as e:
        logger.error(f"Error in estimate_perm_tixier: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/coates",
    summary="Estimate Coates permeability",
    description=(
        "Estimate permeability using the Coates empirical model. "
        "Requires porosity (phit) and irreducible water saturation (swirr) measurements.\n"
        "Input model: PermOthersInputData (see quick_pp.api.schemas.permeability_others.InputData).\n"
        "Request body must be a JSON object with the following field:\n"
        "- data: list of objects, each with keys 'phit' (float, required) and 'swirr' (float, required)\n"
        "Example (truncated): { 'data': [ {'phit': 0.18, 'swirr': 0.25}, ... ] }"
    ),
    operation_id="estimate_coates_permeability",
)
async def estimate_perm_coates(inputs: PermOthersInputData) -> List[Dict[str, float]]:
    """
    Estimate permeability using the Coates method.

    This endpoint receives input data containing porosity (phit) and irreducible water saturation (swirr) values,
    applies the Coates permeability estimation algorithm, and returns the calculated permeability values.

    Args:
        inputs (PermOthersInputData): Input data model containing a list of records with 'phit' and 'swirr'.
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing the estimated permeability ('PERM')
        for the corresponding input record.
    Technical Details:
        - The function extracts the input data, constructs a pandas DataFrame, and computes permeability using the
          `coates_permeability` function, which implements the Coates empirical relationship for permeability
          estimation in a vectorized manner.
        - The results are formatted as a list of dictionaries for API response.
        - Errors are logged and returned as HTTP 400 errors.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict['data'])
        perm = coates_permeability(input_df['phit'], input_df['swirr'])
        df_result = pd.DataFrame({'PERM': perm.ravel()})
        return [
            {str(k): float(v) for k, v in row.items()}
            for row in df_result.to_dict(orient='records')
        ]
    except Exception as e:
        logger.error(f"Error in estimate_perm_coates: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post(
    "/kozeny_carman",
    summary="Estimate Kozeny-Carman permeability",
    description=(
        "Estimate permeability using the Kozeny-Carman equation. "
        "Requires porosity (phit) and irreducible water saturation (swirr) measurements.\n"
        "Input model: PermOthersInputData (see quick_pp.api.schemas.permeability_others.InputData).\n"
        "Request body must be a JSON object with the following field:\n"
        "- data: list of objects, each with keys 'phit' (float, required) and 'swirr' (float, required)\n"
        "Example (truncated): { 'data': [ {'phit': 0.18, 'swirr': 0.25}, ... ] }"
    ),
    operation_id="estimate_kozeny_carman_permeability",
)
async def estimate_perm_kozeny_carman(inputs: PermOthersInputData) -> List[Dict[str, float]]:
    """
    Estimate permeability using the Kozeny-Carman equation.

    This endpoint receives input data containing porosity (phit) and irreducible water saturation (swirr) values,
    applies the Kozeny-Carman permeability model to each record, and returns the estimated permeability values.

    Args:
        inputs (PermOthersInputData): Input data model containing a list of records with 'phit' and 'swirr'.
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing the estimated permeability ('PERM')
        for the corresponding input record.
    Technical Details:
        - The function expects a JSON body matching the PermOthersInputData schema.
        - For each record in the input data, the Kozeny-Carman equation is applied via the
          `kozeny_carman_permeability` function (applied row-wise, not vectorized).
        - The results are returned as a list of dictionaries, suitable for API responses.
        - Errors are logged and returned as HTTP 400 errors.
    """
    try:
        input_dict = inputs.model_dump()
        input_df = pd.DataFrame.from_records(input_dict['data'])
        perm = input_df.apply(lambda row: kozeny_carman_permeability(row['phit'], row['swirr']), axis=1)
        df_result = pd.DataFrame({'PERM': perm.ravel()})
        return [
            {str(k): float(v) for k, v in row.items()}
            for row in df_result.to_dict(orient='records')
        ]
    except Exception as e:
        logger.error(f"Error in estimate_perm_kozeny_carman: {e}")
        raise HTTPException(status_code=400, detail=str(e))
