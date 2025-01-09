from . import asimov
from . import calibration
from . import clustering
from . import detector
from . import injection
from . import job_creation
from . import likelihood
from . import null_stream
from . import prior
from . import psd
from . import result
from . import source
from . import time_frequency_transform
from . import utility
from ._version import __version__
from .utility import logger


def get_version_information():
    return __version__

def log_version_information():
    logger.info(f"Running nullpol: {__version__}")
