"""quick_pp: End-to-end petrophysical evaluation, reservoir characterization, and web workspace."""

import logging
import os
import sys
from pathlib import Path

from loguru import logger

# Load .env from the current working directory upwards (like Git).
# This ensures any import of `quick_pp` picks up project-level environment
# variables from the closest .env file. If `python-dotenv` isn't installed, this is a no-op.
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    current_dir = Path(os.getcwd())
    env_file = None
    for parent in [current_dir] + list(current_dir.parents):
        candidate = parent / ".env"
        if candidate.exists():
            env_file = candidate
            break
    if env_file:
        try:
            load_dotenv(dotenv_path=str(env_file))
        except Exception:
            # don't fail package import for dotenv parsing errors
            pass

# Remove default loguru handler
logger.remove()

# Add a new handler with custom formatting and color
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> <level>{level: <8}</level> | <cyan>{name}</cyan> | "
    "<level>{message}</level>",
    level="INFO",
)

# Silence overly verbose loggers from dependencies (if needed, using stdlib logging)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

__author__ = """Imran Fadhil"""
__email__ = "imranfadhil@gmail.com"
__version__ = "0.3.3"  # Need to be updated manually when releasing a new version/ change in pyproject.toml

# Expose core petrophysical submodules and key public functions at top-level package namespace
from quick_pp import (  # noqa: E402
    fluid_type,
    handover,
    qaqc,
    rock_type,
    saturation,
    scal,
    uncertainty,
    upscaling,
)
from quick_pp.fluid_type import (  # noqa: E402
    FluidContactResult,
    FluidGradientResult,
    FormationPressureAnalyzer,
    density_to_gradient,
    gradient_to_density,
)
from quick_pp.handover import (  # noqa: E402
    build_handover_archive_bytes,
    collect_well_handover_dict,
)
from quick_pp.qaqc import (  # noqa: E402
    apply_environmental_corrections,
    badhole_flagging,
    normalize_by_zone,
    stratigraphic_normalization,
)
from quick_pp.rock_type import (  # noqa: E402
    compute_rock_flag_comparison,
    predict_with_saved_rt_classifier,
    train_and_validate_rock_type_classifier,
)
from quick_pp.saturation import (  # noqa: E402
    ContactOptimizer,
    analyze_transition_zone,
    estimate_mdt_fwl,
    fit_swirr_correlation,
    optimize_fwl,
    plot_fwl_reconciliation,
    predict_sw_shm,
    swirr_rqi,
    swirr_semilog_k,
    swirr_timur,
)
from quick_pp.scal import (  # noqa: E402
    CoreyRelPerm,
    brooks_corey_relperm,
    corey_relperm,
    correct_core_data,
    dean_stark_cbw_correction,
    dean_stark_saturation_correction,
    estimate_cbw_from_qv,
    fit_relperm,
    generate_swof_table,
    humidity_to_total_porosity,
    klinkenberg_permeability_correction,
    let_relperm,
    normalize_water_saturation,
    overburden_permeability_correction,
    overburden_poroperm_correction,
    overburden_porosity_correction,
    pc_lab_to_reservoir,
)
from quick_pp.uncertainty import (  # noqa: E402
    calc_stochastic_reservoir_summary,
    mc_volumetric_uncertainty,
    run_cpi_uncertainty,
    run_log_uncertainty,
    sample_lhs_parameters,
)
try:
    from quick_pp.machine_learning.utils import (  # noqa: E402
        ConformalMultiOutputRegressor,
        ConformalRegressor,
        EnforceMineralSum,
        enforce_physical_constraints,
        physics_informed_loss,
    )
except ImportError:
    ConformalMultiOutputRegressor = None  # type: ignore[assignment]
    ConformalRegressor = None  # type: ignore[assignment]
    EnforceMineralSum = None  # type: ignore[assignment]
    enforce_physical_constraints = None  # type: ignore[assignment]
    physics_informed_loss = None  # type: ignore[assignment]
from quick_pp.upscaling import (  # noqa: E402
    arith_upscale,
    block_logs,
    cardwell_parsons_bounds,
    expand_blocked_logs,
    geom_upscale,
    harm_upscale,
    majority_upscale,
    plot_upscaled_logs,
    pore_volume_upscale,
    power_law_upscale,
    upscale_logs,
    upscale_permeability,
)

__all__ = [
    "ConformalMultiOutputRegressor",
    "ConformalRegressor",
    "ContactOptimizer",
    "CoreyRelPerm",
    "EnforceMineralSum",
    "FluidContactResult",
    "FluidGradientResult",
    "FormationPressureAnalyzer",
    "__author__",
    "__email__",
    "__version__",
    "analyze_transition_zone",
    "apply_environmental_corrections",
    "arith_upscale",
    "badhole_flagging",
    "block_logs",
    "brooks_corey_relperm",
    "build_handover_archive_bytes",
    "calc_stochastic_reservoir_summary",
    "cardwell_parsons_bounds",
    "collect_well_handover_dict",
    "compute_rock_flag_comparison",
    "corey_relperm",
    "correct_core_data",
    "dean_stark_cbw_correction",
    "dean_stark_saturation_correction",
    "density_to_gradient",
    "enforce_physical_constraints",
    "estimate_cbw_from_qv",
    "estimate_mdt_fwl",
    "expand_blocked_logs",
    "fit_relperm",
    "fit_swirr_correlation",
    "fluid_type",
    "generate_swof_table",
    "geom_upscale",
    "gradient_to_density",
    "handover",
    "harm_upscale",
    "humidity_to_total_porosity",
    "klinkenberg_permeability_correction",
    "let_relperm",
    "logger",
    "majority_upscale",
    "mc_volumetric_uncertainty",
    "normalize_by_zone",
    "normalize_water_saturation",
    "optimize_fwl",
    "overburden_permeability_correction",
    "overburden_poroperm_correction",
    "overburden_porosity_correction",
    "pc_lab_to_reservoir",
    "physics_informed_loss",
    "plot_fwl_reconciliation",
    "plot_upscaled_logs",
    "pore_volume_upscale",
    "power_law_upscale",
    "predict_sw_shm",
    "predict_with_saved_rt_classifier",
    "qaqc",
    "rock_type",
    "run_cpi_uncertainty",
    "run_log_uncertainty",
    "sample_lhs_parameters",
    "saturation",
    "scal",
    "stratigraphic_normalization",
    "swirr_rqi",
    "swirr_semilog_k",
    "swirr_timur",
    "train_and_validate_rock_type_classifier",
    "uncertainty",
    "upscale_logs",
    "upscale_permeability",
    "upscaling",
]
