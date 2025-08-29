"""Test module for null stream projector functionality.

This module tests the computation of projection operators used in null stream
construction.
"""

from __future__ import annotations

import unittest

import numpy as np

from nullpol.null_stream import compute_gw_projector_masked


class TestProjector(unittest.TestCase):
    """Test class for null stream projection operator computation.

    This class validates the construction of projection operators used to
    create null streams from multi-detector gravitational wave data,
    ensuring proper mathematical implementation.
    """

    def setUp(self):
        """Set up test environment with deterministic random seed.

        Initializes the random number generator with a fixed seed to ensure
        reproducible test results for projection operator computations.
        """
        seed = 12
        np.random.seed(seed)

    def test_compute_gw_projector_masked(self):
        """Test masked projector computation.

        Validates that the projection operator correctly implements the
        mathematical formula P = F(F†F)^(-1)F† for creating null streams,
        where F is the whitened antenna pattern matrix. Tests proper handling
        of frequency masking to exclude invalid frequency bins.
        """
        whitened_antenna_pattern_matrix = np.random.randn(128, 3, 2) + 1.0j * np.random.randn(128, 3, 2)
        frequency_mask = np.full(128, True)
        frequency_mask[:20] = False
        output = compute_gw_projector_masked(
            whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix, frequency_mask=frequency_mask
        )
        expected_output = np.zeros((128, 3, 3), dtype=whitened_antenna_pattern_matrix.dtype)
        F = whitened_antenna_pattern_matrix[20:]
        F_dagger = np.conj(np.transpose(F, [0, 2, 1]))
        FdF = F_dagger @ F
        expected_output[20:] = F @ np.linalg.inv(FdF) @ F_dagger
        self.assertTrue(np.allclose(output, expected_output))


if __name__ == "__main__":
    unittest.main()
