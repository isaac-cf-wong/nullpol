"""Test module for antenna pattern conditioning functionality.

This module tests the antenna pattern conditioning functions for whitening
and calibration operations with simple examples and known expected results.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.antenna_patterns.conditioning import (
    compute_whitened_antenna_pattern_matrix_masked,
    compute_calibrated_whitened_antenna_pattern_matrix,
)


class TestAntennaPatternConditioning:
    """Test class for antenna pattern conditioning functions."""

    def test_compute_whitened_antenna_pattern_matrix_simple_case(self):
        """Test whitening with simple known values and expected results."""
        # Simple test case with known inputs and hand-calculable outputs
        
        # 2 detectors, 2 polarization modes
        antenna_pattern_matrix = np.array([
            [1.0, 0.5],  # detector 0: F+ = 1.0, Fx = 0.5
            [0.8, 0.6],  # detector 1: F+ = 0.8, Fx = 0.6
        ])
        
        # 3 frequency bins, PSD values chosen for easy calculation
        psd_array = np.array([
            [4.0, 1.0, 9.0],  # detector 0: sqrt(4)=2, sqrt(1)=1, sqrt(9)=3
            [16.0, 4.0, 1.0], # detector 1: sqrt(16)=4, sqrt(4)=2, sqrt(1)=1
        ])
        
        # Mask all frequencies active
        frequency_mask = np.array([True, True, True])
        
        result = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix, psd_array, frequency_mask
        )
        
        # Expected results: F / sqrt(PSD)
        expected = np.zeros((3, 2, 2))  # (frequency, detector, mode)
        
        # Frequency 0
        expected[0, 0, :] = [1.0/2.0, 0.5/2.0]  # [0.5, 0.25]
        expected[0, 1, :] = [0.8/4.0, 0.6/4.0]  # [0.2, 0.15]
        
        # Frequency 1  
        expected[1, 0, :] = [1.0/1.0, 0.5/1.0]  # [1.0, 0.5]
        expected[1, 1, :] = [0.8/2.0, 0.6/2.0]  # [0.4, 0.3]
        
        # Frequency 2
        expected[2, 0, :] = [1.0/3.0, 0.5/3.0]  # [0.333..., 0.166...]
        expected[2, 1, :] = [0.8/1.0, 0.6/1.0]  # [0.8, 0.6]
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
        # Check output shape
        assert result.shape == (3, 2, 2), f"Expected shape (3,2,2), got {result.shape}"

    def test_compute_whitened_antenna_pattern_matrix_masked_frequencies(self):
        """Test whitening with frequency masking - only some frequencies active."""
        # Simple case with specific frequency mask pattern
        antenna_pattern_matrix = np.array([
            [2.0, 1.0],  # Simple values for easy verification
        ])
        
        psd_array = np.array([
            [1.0, 4.0, 9.0],  # sqrt = [1, 2, 3]
        ])
        
        # Only frequency indices 0 and 2 are active (skip index 1)
        frequency_mask = np.array([True, False, True])
        
        result = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix, psd_array, frequency_mask
        )
        
        # Expected results
        expected = np.zeros((3, 1, 2))
        
        # Frequency 0: active, should have values 2.0/1.0=2.0, 1.0/1.0=1.0
        expected[0, 0, :] = [2.0, 1.0]
        
        # Frequency 1: masked out, should remain zero
        expected[1, 0, :] = [0.0, 0.0]
        
        # Frequency 2: active, should have values 2.0/3.0, 1.0/3.0
        expected[2, 0, :] = [2.0/3.0, 1.0/3.0]
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)

    def test_compute_whitened_antenna_pattern_matrix_edge_cases(self):
        """Test whitening with edge cases and boundary conditions."""
        # Test with single detector, single mode, single frequency
        antenna_pattern_matrix = np.array([[0.5]])
        psd_array = np.array([[4.0]])
        frequency_mask = np.array([True])
        
        result = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix, psd_array, frequency_mask
        )
        
        expected = np.array([[[0.5/2.0]]])  # 0.5/sqrt(4.0) = 0.25
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
        # Test with all frequencies masked out
        frequency_mask_all_false = np.array([False])
        result_masked = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix, psd_array, frequency_mask_all_false
        )
        
        expected_zeros = np.zeros((1, 1, 1))
        np.testing.assert_array_equal(result_masked, expected_zeros)

    def test_compute_calibrated_whitened_antenna_pattern_matrix_simple_case(self):
        """Test calibration with simple known values."""
        # Start with simple whitened antenna pattern matrix
        whitened_matrix = np.array([
            [[1.0, 0.5], [0.8, 0.6]],  # frequency 0: 2 detectors, 2 modes
            [[0.0, 0.0], [0.0, 0.0]],  # frequency 1: masked out (zeros)
            [[0.2, 0.4], [0.3, 0.1]],  # frequency 2: 2 detectors, 2 modes  
        ])
        
        # Calibration errors: simple multiplication factors
        calibration_error_matrix = np.array([
            [2.0, 1.0, 0.5],  # detector 0: multiply by [2, 1, 0.5]
            [1.5, 1.0, 2.0],  # detector 1: multiply by [1.5, 1, 2.0]
        ])
        
        # Frequency mask matching the whitened matrix structure
        frequency_mask = np.array([True, False, True])
        
        result = compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask, whitened_matrix, calibration_error_matrix
        )
        
        # Expected results: whitened * calibration_error
        expected = np.zeros_like(whitened_matrix, dtype=calibration_error_matrix.dtype)
        
        # Frequency 0 (active)
        expected[0, 0, :] = [1.0 * 2.0, 0.5 * 2.0]  # [2.0, 1.0]
        expected[0, 1, :] = [0.8 * 1.5, 0.6 * 1.5]  # [1.2, 0.9]
        
        # Frequency 1 (masked) - should remain zero
        expected[1, 0, :] = [0.0, 0.0]
        expected[1, 1, :] = [0.0, 0.0]
        
        # Frequency 2 (active)  
        expected[2, 0, :] = [0.2 * 0.5, 0.4 * 0.5]  # [0.1, 0.2]
        expected[2, 1, :] = [0.3 * 2.0, 0.1 * 2.0]  # [0.6, 0.2]
        
        np.testing.assert_array_almost_equal(result, expected, decimal=6)
        
        # Check output shape matches input
        assert result.shape == whitened_matrix.shape, \
            f"Output shape {result.shape} doesn't match input shape {whitened_matrix.shape}"

    def test_compute_calibrated_whitened_antenna_pattern_matrix_unity_calibration(self):
        """Test calibration with unity calibration errors (no effect)."""
        # Simple whitened matrix
        whitened_matrix = np.array([
            [[1.0, 2.0]],  # 1 frequency, 1 detector, 2 modes
        ])
        
        # Unity calibration errors (should have no effect)
        calibration_error_matrix = np.array([[1.0]])
        
        frequency_mask = np.array([True])
        
        result = compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask, whitened_matrix, calibration_error_matrix
        )
        
        # Result should be identical to input when calibration error = 1.0
        np.testing.assert_array_equal(result, whitened_matrix)

    def test_compute_calibrated_whitened_antenna_pattern_matrix_complex_calibration(self):
        """Test calibration with complex-valued calibration errors."""
        # Simple real whitened matrix
        whitened_matrix = np.array([
            [[1.0, 0.5]],  # 1 frequency, 1 detector, 2 modes  
        ], dtype=np.float64)
        
        # Complex calibration error
        calibration_error_matrix = np.array([[1.0 + 1.0j]], dtype=np.complex128)
        
        frequency_mask = np.array([True])
        
        result = compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask, whitened_matrix, calibration_error_matrix
        )
        
        # Expected: real values multiplied by complex calibration
        expected = np.array([
            [[1.0 * (1.0 + 1.0j), 0.5 * (1.0 + 1.0j)]]
        ], dtype=np.complex128)
        
        np.testing.assert_array_equal(result, expected)
        
        # Verify result is complex type
        assert np.iscomplexobj(result), "Result should be complex when calibration errors are complex"

    def test_conditioning_functions_consistency(self):
        """Test that conditioning functions work together and produce consistent results."""
        # Create realistic test scenario
        antenna_pattern_matrix = np.array([
            [0.7, 0.3],  # H1-like response
            [0.6, 0.8],  # L1-like response  
            [0.4, 0.9],  # V1-like response
        ])
        
        # Realistic PSD values
        psd_array = np.array([
            [1e-46, 2e-46, 5e-46],  # H1 PSD
            [1.5e-46, 1.8e-46, 3e-46],  # L1 PSD
            [2e-46, 2.5e-46, 4e-46],   # V1 PSD
        ])
        
        frequency_mask = np.array([True, True, False])  # 2 active frequencies
        
        # Step 1: Whiten
        whitened = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix, psd_array, frequency_mask
        )
        
        # Step 2: Apply calibration 
        calibration_error_matrix = np.array([
            [0.95 + 0.02j, 0.98 - 0.01j, 1.0 + 0.0j],  # H1 calibration
            [1.02 + 0.01j, 1.01 + 0.03j, 0.99 + 0.0j], # L1 calibration  
            [0.97 - 0.01j, 1.03 + 0.02j, 1.01 + 0.0j], # V1 calibration
        ])
        
        calibrated = compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask, whitened, calibration_error_matrix
        )
        
        # Consistency checks
        assert calibrated.shape == whitened.shape, "Calibration shouldn't change shape"
        assert calibrated.shape == (3, 3, 2), f"Expected (3,3,2), got {calibrated.shape}"
        
        # Check that masked frequencies remain zero
        np.testing.assert_array_equal(calibrated[2, :, :], np.zeros((3, 2)))
        
        # Check that active frequencies have non-zero values where expected
        assert np.any(calibrated[0, :, :] != 0), "First frequency should have non-zero values"
        assert np.any(calibrated[1, :, :] != 0), "Second frequency should have non-zero values"
        
        # Verify all values are finite  
        assert np.all(np.isfinite(calibrated[frequency_mask])), "All active values should be finite"

    def test_conditioning_functions_dtype_preservation(self):
        """Test that conditioning functions preserve appropriate data types."""
        # Test with float32 input
        antenna_pattern_matrix = np.array([[1.0, 0.5]], dtype=np.float32)
        psd_array = np.array([[4.0]], dtype=np.float32)
        frequency_mask = np.array([True])
        
        result_whitened = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix, psd_array, frequency_mask
        )
        
        # Output should match input dtype for whitening
        assert result_whitened.dtype == antenna_pattern_matrix.dtype, \
            f"Whitened output dtype {result_whitened.dtype} doesn't match input {antenna_pattern_matrix.dtype}"
        
        # Test calibration with complex calibration errors
        calibration_error_matrix = np.array([[1.0 + 0.1j]], dtype=np.complex64)
        
        result_calibrated = compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask, result_whitened, calibration_error_matrix
        )
        
        # Output should match calibration error dtype when complex
        assert result_calibrated.dtype == calibration_error_matrix.dtype, \
            f"Calibrated output dtype {result_calibrated.dtype} doesn't match calibration {calibration_error_matrix.dtype}"