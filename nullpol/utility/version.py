from .. import __version__
from . import logger


def get_version_information():
    return __version__

def log_version_information():
    logger.info(f"Running nullpol: {__version__}")