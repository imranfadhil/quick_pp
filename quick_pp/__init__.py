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
__version__ = "0.2.106"  # Need to be updated manually when releasing a new version/ change in pyproject.toml

# Expose core petrophysical submodules and key public functions at top-level package namespace
from quick_pp import qaqc, saturation, scal, uncertainty, upscaling  # noqa: E402
from quick_pp.qaqc import (  # noqa: E402
    apply_environmental_corrections,
    normalize_by_zone,
    stratigraphic_normalization,
)
from quick_pp.saturation import (  # noqa: E402
    ContactOptimizer,
    analyze_transition_zone,
    estimate_mdt_fwl,
    optimize_fwl,
    plot_fwl_reconciliation,
    predict_sw_shm,
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
from quick_pp.machine_learning.utils import (  # noqa: E402
    ConformalMultiOutputRegressor,
    ConformalRegressor,
    EnforceMineralSum,
    enforce_physical_constraints,
    physics_informed_loss,
)
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
    "__author__",
    "__email__",
    "__version__",
    "analyze_transition_zone",
    "apply_environmental_corrections",
    "arith_upscale",
    "block_logs",
    "brooks_corey_relperm",
    "calc_stochastic_reservoir_summary",
    "cardwell_parsons_bounds",
    "corey_relperm",
    "correct_core_data",
    "dean_stark_cbw_correction",
    "dean_stark_saturation_correction",
    "enforce_physical_constraints",
    "estimate_cbw_from_qv",
    "estimate_mdt_fwl",
    "expand_blocked_logs",
    "fit_relperm",
    "generate_swof_table",
    "geom_upscale",
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
    "qaqc",
    "run_cpi_uncertainty",
    "run_log_uncertainty",
    "sample_lhs_parameters",
    "saturation",
    "scal",
    "stratigraphic_normalization",
    "uncertainty",
    "upscale_logs",
    "upscale_permeability",
    "upscaling",
]

