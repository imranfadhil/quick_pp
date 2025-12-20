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
__version__ = "0.2.96"  # Need to be updated manually when releasing a new version/ change in pyproject.toml
