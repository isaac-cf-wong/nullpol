"""Test module for null stream functionality.

This module tests null stream projection operators and related computations
using simple, mathematically verifiable examples.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.null_stream.projections import compute_gw_projector, compute_null_projector, compute_null_stream


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
# SIMPLE MATHEMATICAL EXAMPLES
# =============================================================================


class TestProjectorSimpleExamples:
    """Test projectors using simple, hand-calculable examples."""

    def test_identity_matrix_case(self):
        """Test projector with identity antenna pattern (simple case)."""
        # Simple case: 2 detectors, 2 polarizations, identity antenna pattern
        F = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)  # Shape: (1, 2, 2)
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # With identity antenna pattern, GW projector should be identity
        # (all signal space, no null space)
        expected_gw = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)
        expected_null = np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=complex)

        assert np.allclose(gw_projector, expected_gw, atol=1e-10)
        assert np.allclose(null_projector, expected_null, atol=1e-10)

    def test_single_detector_case(self):
        """Test with single detector (2 detectors for valid inverse)."""
        # 2 detectors, 2 polarizations (square, invertible)
        F = np.array([[[1.0, 0.0], [0.5, 1.0]]], dtype=complex)  # Shape: (1, 2, 2)
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # Should have valid projector properties
        validation = validate_projector_properties(gw_projector[0], null_projector[0])
        assert validation["all_valid"], f"Mathematical properties failed: {validation}"

    def test_orthogonal_antenna_patterns(self):
        """Test with orthogonal antenna patterns (well-conditioned case)."""
        # 2 detectors with orthogonal responses
        F = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)  # Shape: (1, 2, 2)
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # Validate mathematical properties
        validation = validate_projector_properties(gw_projector[0], null_projector[0])
        assert validation["all_valid"], f"Mathematical properties failed: {validation}"

        # With orthogonal patterns spanning full space, should have no null space
        expected_gw = np.eye(2, dtype=complex)
        expected_null = np.zeros((2, 2), dtype=complex)

        assert np.allclose(gw_projector[0], expected_gw, atol=1e-10)
        assert np.allclose(null_projector[0], expected_null, atol=1e-10)

    def test_three_detector_two_polarization_case(self):
        """Test 3 detectors, 2 polarizations (creates 1D null space)."""
        # Simple case where we expect a 1-dimensional null space
        F = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]], dtype=complex)  # Shape: (1, 3, 2)
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # Should have valid projector properties
        validation = validate_projector_properties(gw_projector[0], null_projector[0])
        assert validation["all_valid"], f"Mathematical properties failed: {validation}"

        # Rank should be 2 for GW projector (2 polarizations)
        gw_rank = np.linalg.matrix_rank(gw_projector[0])
        null_rank = np.linalg.matrix_rank(null_projector[0])

        assert gw_rank == 2, f"Expected GW projector rank 2, got {gw_rank}"
        assert null_rank == 1, f"Expected null projector rank 1, got {null_rank}"


class TestProjectorMasking:
    """Test frequency masking functionality."""

    def test_frequency_masking_basic(self):
        """Test that frequency masking works correctly."""
        # 2 frequency bins, only second one is valid
        F = np.array(
            [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=complex  # First frequency  # Second frequency
        )
        frequency_mask = np.array([False, True])

        gw_projector = compute_gw_projector(F, frequency_mask)

        # First frequency should be zero (masked)
        assert np.allclose(gw_projector[0], 0.0, atol=1e-10)

        # Second frequency should be computed
        assert not np.allclose(gw_projector[1], 0.0, atol=1e-10)

    def test_all_frequencies_masked(self):
        """Test when all frequencies are masked."""
        F = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)
        frequency_mask = np.array([False])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # GW projector should be zero (masked)
        assert np.allclose(gw_projector, 0.0, atol=1e-10)

        # Null projector should be identity (I - 0 = I)
        expected_identity = np.eye(2, dtype=complex)
        assert np.allclose(null_projector[0], expected_identity, atol=1e-10)

    def test_no_frequencies_masked(self):
        """Test when no frequencies are masked."""
        F = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=complex)
        frequency_mask = np.array([True, True])

        gw_projector = compute_gw_projector(F, frequency_mask)

        # Both frequencies should be computed
        assert not np.allclose(gw_projector[0], 0.0, atol=1e-10)
        assert not np.allclose(gw_projector[1], 0.0, atol=1e-10)


class TestNullStreamProjection:
    """Test the compute_null_stream function."""

    def test_null_stream_simple_case(self):
        """Test null stream computation with simple data."""
        # Simple 2x2 case
        strain_data = np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]])  # Shape: (2, 2)

        # Identity projector (no nulling)
        null_projector = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]], dtype=complex)

        frequency_mask = np.array([True, True])

        result = compute_null_stream(strain_data, null_projector, frequency_mask)

        # With identity null projector, output should equal input
        assert np.allclose(result, strain_data, atol=1e-10)

    def test_null_stream_zero_projector(self):
        """Test null stream with zero projector."""
        strain_data = np.array([[1.0 + 1.0j], [2.0 + 2.0j]], dtype=complex)  # Shape: (2, 1)

        # Zero projector (nulls everything)
        null_projector = np.array([[[0.0, 0.0], [0.0, 0.0]]], dtype=complex)  # Shape: (1, 2, 2)

        frequency_mask = np.array([True])

        result = compute_null_stream(strain_data, null_projector, frequency_mask)

        # Should zero out the strain
        expected = np.array([[0.0 + 0.0j], [0.0 + 0.0j]], dtype=complex)
        assert np.allclose(result, expected, atol=1e-10)

    def test_null_stream_frequency_masking(self):
        """Test null stream respects frequency masking."""
        # Shape convention: strain_data is (n_det, n_freq)
        strain_data = np.array([[1.0, 2.0, 3.0]], dtype=complex)  # Shape: (1, 3) - 1 detector, 3 frequencies

        # Simple diagonal projectors for each frequency
        null_projector = np.array(
            [
                [[0.5]],  # Freq 0 - 1x1 projector
                [[0.8]],  # Freq 1 (masked) - 1x1 projector
                [[0.2]],  # Freq 2 - 1x1 projector
            ],
            dtype=complex,
        )  # Shape: (3, 1, 1)

        frequency_mask = np.array([True, False, True])  # Skip middle frequency

        result = compute_null_stream(strain_data, null_projector, frequency_mask)

        # Check results: processed frequencies get projected, masked ones are zero
        assert abs(result[0, 0] - 0.5 * strain_data[0, 0]) < 1e-10  # Freq 0: processed
        assert abs(result[0, 1] - 0.0) < 1e-10  # Freq 1: masked -> zero
        assert abs(result[0, 2] - 0.2 * strain_data[0, 2]) < 1e-10  # Freq 2: processed

    def test_null_stream_complex_data(self):
        """Test null stream with complex strain data."""
        strain_data = np.array([[1.0 + 2.0j], [0.5 + 1.5j]], dtype=complex)  # Shape: (2, 1)

        # 45-degree rotation projector
        cos45 = 1 / np.sqrt(2)
        rotation_matrix = np.array([[cos45, -cos45], [cos45, cos45]], dtype=complex)
        null_projector = np.array([rotation_matrix], dtype=complex)  # Shape: (1, 2, 2)

        frequency_mask = np.array([True])

        result = compute_null_stream(strain_data, null_projector, frequency_mask)

        # Result should be rotated version of input
        assert result.shape == strain_data.shape
        assert result.dtype == np.complex128

        # Check that transformation was applied (not just copied)
        assert not np.allclose(result, strain_data, atol=1e-10)

        # Verify the actual rotation calculation
        expected = rotation_matrix @ strain_data[:, 0]
        assert np.allclose(result[:, 0], expected, atol=1e-10)


class TestProjectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_frequency_single_detector_square(self):
        """Test minimal valid case: 1 frequency, square matrix."""
        # 2x2 case (minimum for invertible matrix)
        F = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)  # Shape: (1, 2, 2)
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # With identity, projector should be identity
        assert gw_projector.shape == (1, 2, 2)
        assert np.allclose(gw_projector[0], np.eye(2), atol=1e-10)
        assert np.allclose(null_projector[0], np.zeros((2, 2)), atol=1e-10)

    def test_well_conditioned_matrix(self):
        """Test with well-conditioned antenna pattern."""
        # Well-conditioned 2x2 matrix
        F = np.array([[[1.0, 0.0], [0.0, 2.0]]], dtype=complex)  # Different scaling
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # Should still satisfy basic properties
        validation = validate_projector_properties(gw_projector[0], null_projector[0])
        assert validation["all_valid"], f"Properties failed: {validation}"

    def test_different_data_types(self):
        """Test with different complex data types."""
        # Test with complex64 - use well-conditioned matrix
        F_c64 = np.array([[[2.0, 0.0], [0.0, 2.0]]], dtype=np.complex64)  # Well-conditioned
        frequency_mask = np.array([True])

        gw_projector = compute_gw_projector(F_c64, frequency_mask)
        null_projector = compute_null_projector(gw_projector)

        # Should maintain the input dtype
        assert gw_projector.dtype == np.complex64
        assert null_projector.dtype == np.complex64

        # For identity-like matrix, should give identity projector
        expected_gw = np.eye(2, dtype=np.complex64)
        expected_null = np.zeros((2, 2), dtype=np.complex64)

        assert np.allclose(gw_projector[0], expected_gw, atol=1e-6)  # Relaxed tolerance for float32
        assert np.allclose(null_projector[0], expected_null, atol=1e-6)


# =============================================================================
# INTEGRATION TESTS WITH REAL-WORLD SCENARIOS
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
