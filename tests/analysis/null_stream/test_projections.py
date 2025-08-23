"""Test module for null stream functionality.

This module tests null stream projection operators and related computations.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.null_stream.projections import compute_gw_projector, compute_null_projector


# =============================================================================
# TEST UTILITIES
# =============================================================================


def validate_projector_properties(gw_projector, null_projector, tolerance=1e-10):
    """Validate mathematical properties of the projection operators.

    This utility function validates that the GW and null projectors satisfy
    the required mathematical properties for orthogonal projectors.

    Args:
        gw_projector (numpy.ndarray): GW projector matrix.
        null_projector (numpy.ndarray): Null projector matrix.
        tolerance (float): Numerical tolerance for validation checks.

    Returns:
        dict: Dictionary with validation results for each property.
    """
    results = {}

    # Test idempotency: P @ P = P
    gw_idempotent = np.allclose(gw_projector @ gw_projector, gw_projector, atol=tolerance)
    null_idempotent = np.allclose(null_projector @ null_projector, null_projector, atol=tolerance)

    # Test orthogonality: P_gw @ P_null = 0
    orthogonal = np.allclose(gw_projector @ null_projector, 0, atol=tolerance)

    # Test completeness: P_gw + P_null = I
    identity = np.eye(gw_projector.shape[-1])
    complete = np.allclose(gw_projector + null_projector, identity, atol=tolerance)

    results = {
        "gw_idempotent": gw_idempotent,
        "null_idempotent": null_idempotent,
        "orthogonal": orthogonal,
        "complete": complete,
        "all_valid": all([gw_idempotent, null_idempotent, orthogonal, complete]),
    }

    return results


@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set up test environment with deterministic random seed.

    Initializes the random number generator with a fixed seed to ensure
    reproducible test results for projection operator computations.
    """
    seed = 12
    np.random.seed(seed)


# =============================================================================
# PROJECTOR TESTS
# =============================================================================


def test_compute_gw_projector_masked():
    """Test masked projector computation.

    Validates that the projection operator correctly implements the
    mathematical formula P = F(F†F)^(-1)F† for creating null streams,
    where F is the whitened antenna pattern matrix. Tests proper handling
    of frequency masking to exclude invalid frequency bins.
    """
    whitened_antenna_pattern_matrix = np.random.randn(128, 3, 2) + 1.0j * np.random.randn(128, 3, 2)
    frequency_mask = np.full(128, True)
    frequency_mask[:20] = False
    output = compute_gw_projector(
        whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix, frequency_mask=frequency_mask
    )
    expected_output = np.zeros((128, 3, 3), dtype=whitened_antenna_pattern_matrix.dtype)
    F = whitened_antenna_pattern_matrix[20:]
    F_dagger = np.conj(np.transpose(F, [0, 2, 1]))
    FdF = F_dagger @ F
    expected_output[20:] = F @ np.linalg.inv(FdF) @ F_dagger
    assert np.allclose(output, expected_output)


def test_projector_mathematical_properties():
    """Test mathematical properties of GW and null projectors.

    Validates that the computed projectors satisfy the required mathematical
    properties: idempotency, orthogonality, and completeness.
    """
    # Create test data
    whitened_antenna_pattern_matrix = np.random.randn(10, 3, 2) + 1.0j * np.random.randn(10, 3, 2)
    frequency_mask = np.full(10, True)

    # Compute projectors
    gw_projector = compute_gw_projector(whitened_antenna_pattern_matrix, frequency_mask)
    null_projector = compute_null_projector(gw_projector)

    # Validate properties for each frequency bin
    for freq_idx in range(10):
        if frequency_mask[freq_idx]:
            gw_proj_freq = gw_projector[freq_idx]
            null_proj_freq = null_projector[freq_idx]

            validation_results = validate_projector_properties(
                gw_proj_freq[np.newaxis, :, :], null_proj_freq[np.newaxis, :, :], tolerance=1e-10
            )

            assert validation_results["all_valid"], f"Projector properties failed at frequency {freq_idx}"
