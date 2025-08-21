"""Test module for null stream calculator functionality.

This module tests the NullStreamCalculator class and its methods
for computing null projections and energies.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.null_stream import NullStreamCalculator


@pytest.fixture
def calculator_instance():
    """Create a calculator instance for testing."""
    return NullStreamCalculator()


@pytest.fixture
def sample_antenna_patterns():
    """Create sample antenna pattern matrices for testing."""
    n_frequencies = 10
    n_detectors = 3
    n_modes = 2

    # Create random but realistic antenna pattern matrices
    patterns = np.random.normal(0, 1, (n_frequencies, n_detectors, n_modes))
    return patterns


@pytest.fixture
def sample_frequency_mask():
    """Create sample frequency mask."""
    n_frequencies = 10
    mask = np.ones(n_frequencies, dtype=bool)
    mask[0] = False  # Typically exclude DC
    mask[-1] = False  # Typically exclude Nyquist
    return mask


class TestNullStreamCalculator:
    """Test class for NullStreamCalculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = NullStreamCalculator()
        # Calculator should initialize without errors
        assert calculator is not None

    def test_compute_gw_projector(self, calculator_instance, sample_antenna_patterns, sample_frequency_mask):
        """Test GW projector computation."""
        gw_projector = calculator_instance.compute_gw_projector(sample_antenna_patterns, sample_frequency_mask)

        # Check output shape
        n_frequencies, n_detectors, _ = sample_antenna_patterns.shape
        expected_shape = (n_frequencies, n_detectors, n_detectors)
        assert gw_projector.shape == expected_shape

        # Check that projectors are hermitian (approximately)
        for freq_idx in range(n_frequencies):
            if sample_frequency_mask[freq_idx]:
                P = gw_projector[freq_idx]
                np.testing.assert_allclose(P, P.T.conj(), atol=1e-10)

    def test_compute_null_projector(self, calculator_instance, sample_antenna_patterns, sample_frequency_mask):
        """Test null projector computation."""
        gw_projector = calculator_instance.compute_gw_projector(sample_antenna_patterns, sample_frequency_mask)
        null_projector = calculator_instance.compute_null_projector(gw_projector)

        # Check output shape
        assert null_projector.shape == gw_projector.shape

        # Check that GW and null projectors are complementary: P_gw + P_null = I
        n_frequencies, n_detectors, _ = gw_projector.shape
        identity = np.eye(n_detectors)

        for freq_idx in range(n_frequencies):
            if sample_frequency_mask[freq_idx]:
                P_gw = gw_projector[freq_idx]
                P_null = null_projector[freq_idx]
                total = P_gw + P_null
                np.testing.assert_allclose(total, identity, atol=1e-10)
