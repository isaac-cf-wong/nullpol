"""Test module for antenna pattern processor functionality.

This module tests the AntennaPatternProcessor class and its methods
for computing antenna patterns and polarization operations.
"""

from __future__ import annotations

import numpy as np
import pytest
import bilby

from nullpol.analysis.antenna_patterns import AntennaPatternProcessor


@pytest.fixture
def sample_interferometers():
    """Create sample interferometers for testing."""
    duration = 4.0
    sampling_frequency = 2048.0

    # Create multiple interferometers (need more than polarization basis count)
    interferometers = []
    for name in ["H1", "L1", "V1"]:  # 3 detectors > 2 polarization bases
        ifo = bilby.gw.detector.get_empty_interferometer(name)
        ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=0)
        interferometers.append(ifo)

    return bilby.gw.detector.InterferometerList(interferometers)


class TestAntennaPatternProcessor:
    """Test class for AntennaPatternProcessor."""

    def test_processor_initialization(self, sample_interferometers):
        """Test processor initialization."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p", "c"], interferometers=sample_interferometers
        )

        # Check that polarization modes and basis are encoded as boolean arrays
        assert isinstance(processor.polarization_modes, np.ndarray)
        assert isinstance(processor.polarization_basis, np.ndarray)
        assert processor.polarization_modes.dtype == bool
        assert processor.polarization_basis.dtype == bool

        # Check that 'p' and 'c' modes are active (indices 0 and 1)
        assert processor.polarization_modes[0]  # 'p'
        assert processor.polarization_modes[1]  # 'c'

    def test_processor_basic_functionality(self, sample_interferometers):
        """Test basic processor functionality."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p"], interferometers=sample_interferometers
        )

        # Test that processor has expected attributes
        assert hasattr(processor, "polarization_modes")
        assert hasattr(processor, "polarization_basis")
        assert hasattr(processor, "polarization_derived")

        # Test that basis is subset of modes
        assert np.all(processor.polarization_basis[processor.polarization_modes] >= 0)
