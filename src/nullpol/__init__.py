from __future__ import annotations

# Core user-facing modules - these are what most users will need
from . import analysis  # Analysis capabilities (includes prior distributions, clustering, etc.)
from . import detector  # Detector handling (networks)
from . import simulation  # Simulation utilities (injection, source models)
from . import utils  # Common utilities

# Advanced/internal modules - available but not auto-imported for cleaner API
# from . import cli          # Command-line interface (import when needed)
# from . import integrations # External integrations (asimov, htcondor, etc.)

from .utils import logger

__version__ = "0.1.0"


def get_version_information() -> str:
    """Get version information.

    Returns:
        str: Version information.
    """
    return __version__


def log_version_information():
    """Log version information."""
    logger.info(f"Running nullpol: {__version__}")


__all__ = [
    "analysis",
    "detector",
    "simulation",
    "utils",
    "get_version_information",
    "log_version_information",
    "logger",
]
