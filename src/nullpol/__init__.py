from . import asimov
from . import calibration
from . import clustering
from . import detector
from . import injection
from . import job_creation
from . import likelihood
from . import null_stream
from . import prior
from . import result
from . import source
from . import time_frequency_transform
from . import utils
from .utils import logger


__version__ = '1.0.0'


def get_version_information() -> str:
    """Get version information.

    Returns:
        str: Version information.
    """
    return __version__


def log_version_information():
    """Log version information.
    """
    logger.info(f"Running nullpol: {__version__}")


__all__ = [
    'asimov',
    'calibration',
    'clustering',
    'detector',
    'injection',
    'job_creation',
    'likelihood',
    'null_stream',
    'prior',
    'result',
    'source',
    'time_frequency_transform',
    'utils',
    '__version__',
    'get_version_information',
    'log_version_information',
]
