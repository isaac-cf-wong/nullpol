from .. import __version__
from . import logger

def log_version_information():
    logger.info(f"Running nullpol: {__version__}")