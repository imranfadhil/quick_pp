import logging
import sys

# Set up a package-wide logger for quick_pp
logger = logging.getLogger("quick_pp")
logger.setLevel(logging.INFO)

# Create handler (console)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)

# Create formatter and add to handler
formatter = logging.Formatter('[%(asctime)s] %(levelname)s | %(name)s | %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger if not already added
if not logger.hasHandlers():
    logger.addHandler(console_handler)

# Optionally, silence overly verbose loggers from dependencies
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)
