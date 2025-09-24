"""Test module for threshold filter functionality.

This module tests the threshold filtering functions with simple examples
and known expected results, focusing on quantile-based filtering.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.clustering.threshold_filter import compute_filter_by_quantile


class TestThresholdFilter:
    """Test class for threshold filtering functions."""

    def test_compute_filter_by_quantile_simple_case(self):
        """Test quantile filtering with simple known data."""
        # Simple data where we can hand-calculate the expected results
        data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

        # Use 50th percentile (median) - value should be 3.5
        # Values > 3.5 should pass: [4, 5, 6]
        filter_result = compute_filter_by_quantile(data, quantile=0.5)

        expected_filter = np.array([[False, False, False], [True, True, True]])

        np.testing.assert_array_equal(filter_result, expected_filter)

    def test_compute_filter_by_quantile_50th_percentile(self):
        """Test 50th percentile filtering with known median."""
        # Data with clear median
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]])

        # 50th percentile of [1,2,3,4,5,6,7,8,9] is 5.0
        # Values > 5.0 should pass: [6,7,8,9]
        filter_result = compute_filter_by_quantile(data, quantile=0.5)

        expected_filter = np.array([[False, False, False, False, False, True, True, True, True]])

        np.testing.assert_array_equal(filter_result, expected_filter)

    def test_compute_filter_by_quantile_with_zeros(self):
        """Test that zeros are excluded from quantile calculation."""
        # Data with zeros and non-zeros
        data = np.array(
            [
                [0.0, 1.0, 0.0],
                [2.0, 0.0, 5.0],
            ]
        )

        # Non-zero values are [1, 2, 5]
        # 50th percentile is 2.0
        # Values > 2.0 should pass: only 5
        filter_result = compute_filter_by_quantile(data, quantile=0.5)

        expected_filter = np.array(
            [
                [False, False, False],
                [False, False, True],  # Only the 5 passes
            ]
        )

        np.testing.assert_array_equal(filter_result, expected_filter)

    def test_compute_filter_by_quantile_extreme_quantiles(self):
        """Test extreme quantile values (very low and very high)."""
        data = np.array([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Test very low quantile (10th percentile)
        filter_low = compute_filter_by_quantile(data, quantile=0.1)
        # 10th percentile ≈ 1.4, so [2,3,4,5] should pass
        assert np.sum(filter_low) == 4, "Low quantile should pass most values"

        # Test very high quantile (99th percentile)
        filter_high = compute_filter_by_quantile(data, quantile=0.99)
        # 99th percentile ≈ 4.96, so only 5 should pass
        assert np.sum(filter_high) == 1, "High quantile should pass few values"

        # Verify the high quantile passes the maximum value
        assert filter_high[0, 4], "Highest quantile should include maximum value"

    def test_compute_filter_by_quantile_edge_cases(self):
        """Test edge cases like single values and uniform data."""
        # Single value
        single_value = np.array([[5.0]])
        filter_single = compute_filter_by_quantile(single_value, quantile=0.5)
        # Single non-zero value should not pass (> itself is False)
        assert not filter_single[0, 0], "Single value should not pass its own threshold"

        # All zeros - this will raise an error since quantile can't be computed on empty array
        all_zeros = np.zeros((2, 2))
        with pytest.raises(IndexError):
            # np.quantile will fail on empty array after filtering out zeros
            _ = compute_filter_by_quantile(all_zeros, quantile=0.9)

        # Uniform non-zero values
        uniform_data = np.full((3, 3), 7.0)
        filter_uniform = compute_filter_by_quantile(uniform_data, quantile=0.5)
        assert not np.any(filter_uniform), "Uniform values should not pass their own threshold"

    def test_compute_filter_by_quantile_kwargs_ignored(self):
        """Test that additional keyword arguments are properly ignored."""
        data = np.array([[1.0, 2.0, 3.0]])

        # Test with additional kwargs
        filter_result = compute_filter_by_quantile(data, quantile=0.5, ignored_param=42, another_ignored="test")

        # Should work the same as without kwargs
        filter_expected = compute_filter_by_quantile(data, quantile=0.5)
        np.testing.assert_array_equal(filter_result, filter_expected)

    def test_compute_filter_by_quantile_2d_shape_preservation(self):
        """Test that 2D shape and properties are preserved."""
        # Rectangular data
        data = np.array(
            [
                [1.0, 8.0, 3.0, 9.0],
                [2.0, 7.0, 4.0, 10.0],
                [5.0, 6.0, 11.0, 12.0],
            ]
        )

        filter_result = compute_filter_by_quantile(data, quantile=0.7)

        # Check shape preservation
        assert filter_result.shape == data.shape, "Filter should preserve input shape"

        # Check dtype
        assert filter_result.dtype == bool, "Filter should be boolean array"

        # Check that some values pass and some don't
        assert np.any(filter_result), "Some values should pass the filter"
        assert not np.all(filter_result), "Not all values should pass the filter"

    def test_compute_filter_by_quantile_numerical_precision(self):
        """Test behavior with values very close to the threshold."""
        # Create data where threshold will be exactly between two values
        data = np.array([[1.0, 1.000001, 2.0, 2.000001]])

        # Use a quantile that should give us a threshold around 1.5
        filter_result = compute_filter_by_quantile(data, quantile=0.5)

        # The threshold should be ~1.5000005, so [2.0, 2.000001] should pass
        expected_sum = 2
        assert np.sum(filter_result) == expected_sum, "Should handle numerical precision correctly"

        # Check specific values
        assert not filter_result[0, 0], "1.0 should not pass"
        assert not filter_result[0, 1], "1.000001 should not pass"
        assert filter_result[0, 2], "2.0 should pass"
        assert filter_result[0, 3], "2.000001 should pass"
