"""Analysis package."""

from __future__ import annotations

# Individual test modules
# Test subdirectories for different analysis components
from . import (
    antenna_patterns,  # Antenna pattern function tests
    clustering,  # Time-frequency clustering tests
    likelihood,  # Likelihood implementation tests
    null_stream,  # Null stream projection & calculation tests
    prior,  # Prior distribution tests
    test_data_context,  # Data management and signal processing tests
    test_result,  # Result handling tests
    tf_transforms,  # Time-frequency transform tests (wavelets, STFT)
)

__all__ = [
    "antenna_patterns",
    "clustering",
    "likelihood",
    "null_stream",
    "prior",
    "test_data_context",
    "test_result",
    "tf_transforms",
]
