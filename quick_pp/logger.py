from loguru import logger
import logging
import sys

# Remove default loguru handler
logger.remove()

# Add a new handler with custom formatting and color
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>[{time:YYYY-MM-DD HH:mm:ss}]</green> <level>{level: <8}</level> | <cyan>{name}</cyan> | "
    "<level>{message}</level>",
    level="INFO"
)

# Silence overly verbose loggers from dependencies (if needed, using stdlib logging)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
