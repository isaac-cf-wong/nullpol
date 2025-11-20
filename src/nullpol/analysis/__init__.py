"""Analysis package."""

from __future__ import annotations

# Core analysis modules - likelihood is the most important
from . import (
    antenna_patterns,  # Antenna pattern functions (base patterns + conditioning + processor)
    clustering,  # Time-frequency clustering
    data_context,  # Data management and signal processing functions
    lensing,  # Strong lensing analysis modules
    likelihood,  # Core likelihood implementations
    null_stream,  # Null stream projections & calculations
    prior,  # Prior distributions for polarization analysis
    result,  # Result handling (single module)
    tf_transforms,  # Time-frequency transforms (wavelets, STFT)
)

__all__ = [
    "antenna_patterns",
    "clustering",
    "data_context",
    "lensing",
    "likelihood",
    "null_stream",
    "prior",
    "result",
    "tf_transforms",
]
