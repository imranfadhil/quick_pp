from fastapi import APIRouter
from quick_pp.api.schemas.porosity import InputData
from quick_pp.lithology.sand_silt_clay import SandSiltClay
from quick_pp.porosity import neu_den_xplot_poro, density_porosity, rho_matrix
from typing import List, Dict

router = APIRouter(prefix="/porosity", tags=["Porosity"])


def _validate_points(input_dict: dict, required_points: List[str]):
    for k in required_points:
        if input_dict.get(k) is not None and len(input_dict[k]) != 2:
            raise ValueError(
                f"{k} must be a tuple of 2 elements: (neutron porosity, bulk density)"
            )


@router.post(
    "/den",
    summary="Estimate Density Porosity (PHID)",
    description="Estimate Density Porosity (PHID) using the density porosity method.",
)
async def estimate_phit_den(inputs: InputData) -> List[Dict[str, float]]:
    """
    Estimates density porosity (PHID) for a set of input data using a sand-silt-clay (SSC) model.
    This asynchronous function receives input containing neutron porosity (nphi) and bulk density (rhob) measurements,
    along with reference points for dry sand, silt, clay, fluid, and optionally wet clay. It validates the input points,
    constructs an SSC model, estimates lithology fractions (sand, silt, clay), computes matrix density, and finally
    calculates density porosity for each data point.
    Args:
        inputs (InputData): Input data object containing:
            - data: List of measurements, each with 'nphi' and 'rhob' attributes.
            - dry_sand_point, dry_silt_point, dry_clay_point: Reference points for dry sand, silt, and clay (tuples).
            - fluid_point: Reference point for fluid (tuple).
            - wet_clay_point (optional): Reference point for wet clay (tuple or None).
            - silt_line_angle: Angle parameter for the silt line.
            - Other required fields as defined in InputData.
            The request body is validated and an example is provided via the EXAMPLE constant.
    Returns:
        List[Dict[str, float]]: A list of dictionaries, each containing the estimated density porosity value
        for a data point, with the key "PHID".
    Raises:
        ValidationError: If required reference points are missing or invalid.
        Any exceptions raised by the SandSiltClay model or utility functions.
    Technical Details:
        - Uses the SandSiltClay model to estimate lithology fractions (vsand, vsilt, vcld) from input nphi and rhob.
        - Computes matrix density (rho_ma) for each data point using the estimated lithology fractions.
        - Calculates density porosity (PHID) using the measured bulk density (rhob), computed matrix density (rho_ma),
          and the fluid density (from inputs.fluid_point[1]).
        - Returns the results as a list of dictionaries, each with a single key "PHID" and its corresponding value.
    """
    input_dict = inputs.model_dump()
    _validate_points(input_dict, [k for k in input_dict if k.endswith('_point')])

    nphi = [d.nphi for d in inputs.data]
    rhob = [d.rhob for d in inputs.data]

    ssc_model = SandSiltClay(
        dry_sand_point=inputs.dry_sand_point,
        dry_silt_point=inputs.dry_silt_point,
        dry_clay_point=inputs.dry_clay_point,
        fluid_point=inputs.fluid_point,
        wet_clay_point=inputs.wet_clay_point if inputs.wet_clay_point is not None else (None, None),
        silt_line_angle=inputs.silt_line_angle,
    )
    vsand, vsilt, vcld, _ = ssc_model.estimate_lithology(nphi, rhob)
    rho_ma = [rho_matrix(vs, vsi, vc) for vs, vsi, vc in zip(vsand, vsilt, vcld)]
    phid = [density_porosity(rhb, rhma, inputs.fluid_point[1]) for rhb, rhma in zip(rhob, rho_ma)]
    return [{"PHID": float(val)} for val in phid]


@router.post(
    "/neu_den",
    summary="Estimate Total Porosity (PHIT)",
    description="Estimate Total Porosity (PHIT) using neutron-density crossplot analysis.",
)
async def estimate_phit_neu_den(inputs: InputData) -> List[Dict[str, float]]:
    """
    This asynchronous endpoint receives input data containing neutron porosity (NPHI) and bulk density (RHOB)
    measurements, along with reference points for dry sand, silt, clay, and fluid, and applies a crossplot
    porosity estimation method.
    Parameters:
        inputs (InputData):
            The input data object, expected as a request body, containing:
                - data: List of measurement objects, each with 'nphi' (neutron porosity) and 'rhob' (bulk density).
                - method: The crossplot model or method to use for porosity estimation.
                - dry_sand_point: Reference point for dry sand in the crossplot.
                - dry_silt_point: Reference point for dry silt in the crossplot.
                - dry_clay_point: Reference point for dry clay in the crossplot.
                - fluid_point: Reference point for fluid in the crossplot.
    Returns:
        List[Dict[str, float]]:
            A list of dictionaries, each containing the estimated total porosity ('PHIT') value for the corresponding
            input data point.
    Raises:
        ValidationError: If required reference points are missing or invalid in the input data.
    Notes:
        - The function validates that all required crossplot reference points are present.
        - The porosity estimation is performed using the `neu_den_xplot_poro` function, which implements the
          neutron-density crossplot algorithm.
        - The output is formatted as a list of dictionaries for compatibility with API responses.
    """
    input_dict = inputs.model_dump()
    _validate_points(input_dict, [k for k in input_dict if k.endswith('_point')])

    nphi = [d.nphi for d in inputs.data]
    rhob = [d.rhob for d in inputs.data]

    phit = neu_den_xplot_poro(
        nphi,
        rhob,
        model=inputs.method,
        dry_min1_point=inputs.dry_sand_point,
        dry_silt_point=inputs.dry_silt_point,
        dry_clay_point=inputs.dry_clay_point,
        fluid_point=inputs.fluid_point,
    )
    return [{"PHIT": float(val)} for val in phit]
