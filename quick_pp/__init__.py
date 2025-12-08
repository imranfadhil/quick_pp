from pathlib import Path
from loguru import logger
import logging
import sys

# Load .env from the repository root (single place for the package).
# This ensures any import of `quick_pp` picks up project-level environment
# variables. If `python-dotenv` isn't installed, this is a no-op.
try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    repo_root = Path(__file__).resolve().parent.parent
    candidate = repo_root / ".env"
    if candidate.exists():
        try:
            load_dotenv(dotenv_path=str(candidate))
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
__version__ = "0.2.81"  # Need to be updated manually when releasing a new version/ change in pyproject.toml
