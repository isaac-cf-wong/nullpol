"""Test module for clustering algorithm functionality.

This module tests the clustering algorithm functions with simple examples
and known expected results, focusing on connected component analysis.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.clustering.algorithm import clustering, _get_neighbours, _dfs


class TestClusteringAlgorithm:
    """Test class for clustering algorithm functions."""

    def test_get_neighbours_center_pixel(self):
        """Test neighbor finding for a center pixel with all 8 neighbors."""
        # 5x5 mask with center pixel at (2,2)
        mask = np.ones((5, 5), dtype=bool)

        neighbors = _get_neighbours(2, 2, mask)

        # Should have all 8 neighbors
        expected_neighbors = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2), (3, 3)]  # (2,2) excluded

        assert len(neighbors) == 8, "Center pixel should have 8 neighbors"
        assert set(neighbors) == set(expected_neighbors), "Should include all 8-connected neighbors"

    def test_get_neighbours_corner_pixel(self):
        """Test neighbor finding for corner pixel (fewer neighbors)."""
        mask = np.ones((3, 3), dtype=bool)

        # Top-left corner (0,0)
        neighbors = _get_neighbours(0, 0, mask)
        expected_neighbors = [(0, 1), (1, 0), (1, 1)]

        assert len(neighbors) == 3, "Corner pixel should have 3 neighbors"
        assert set(neighbors) == set(expected_neighbors), "Should include only valid neighbors"

    def test_get_neighbours_edge_pixel(self):
        """Test neighbor finding for edge pixel."""
        mask = np.ones((3, 4), dtype=bool)

        # Top edge, middle (0,2)
        neighbors = _get_neighbours(0, 2, mask)
        expected_neighbors = [(0, 1), (0, 3), (1, 1), (1, 2), (1, 3)]

        assert len(neighbors) == 5, "Edge pixel should have 5 neighbors"
        assert set(neighbors) == set(expected_neighbors), "Should include only valid edge neighbors"

    def test_dfs_simple_connected_cluster(self):
        """Test DFS on a simple connected component."""
        # Simple L-shaped cluster
        mask = np.array(
            [
                [True, True, False],
                [True, False, False],
                [True, True, True],
            ]
        )

        visited = np.zeros_like(mask, dtype=bool)
        cluster = _dfs(0, 0, mask, visited)

        # Should find all True pixels connected to (0,0)
        expected_cluster = [(0, 0), (0, 1), (1, 0), (2, 0), (2, 1), (2, 2)]
        assert len(cluster) == 6, "Should find all 6 connected pixels"
        assert set(cluster) == set(expected_cluster), "Should find the complete L-shaped cluster"

    def test_dfs_disconnected_components(self):
        """Test that DFS only finds one connected component."""
        # Two separate components
        mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, True],
            ]
        )

        visited = np.zeros_like(mask, dtype=bool)

        # Start from top-left component
        cluster1 = _dfs(0, 0, mask, visited)
        assert cluster1 == [(0, 0)], "Should find only the single pixel component"

        # Start from top-right component (not yet visited)
        cluster2 = _dfs(0, 2, mask, visited)
        assert cluster2 == [(0, 2)], "Should find only the other single pixel component"

    def test_clustering_single_cluster_simple(self):
        """Test clustering with a single simple cluster."""
        # Single 2x2 cluster in center
        filter_mask = np.array(
            [
                [False, False, False, False, False],
                [False, True, True, False, False],
                [False, True, True, False, False],
                [False, False, False, False, False],
                [False, False, False, False, False],
            ]
        )

        dt = 0.1  # 0.1 second resolution
        df = 1.0  # 1 Hz resolution
        padding_time = 0.05  # 0.5 time bins padding
        padding_freq = 0.5  # 0.5 freq bins padding

        result = clustering(filter_mask, dt, df, padding_time, padding_freq)

        # Original cluster pixels should be included
        assert result[1, 1] == 1, "Original cluster pixel should be included"
        assert result[1, 2] == 1, "Original cluster pixel should be included"
        assert result[2, 1] == 1, "Original cluster pixel should be included"
        assert result[2, 2] == 1, "Original cluster pixel should be included"

        # Check that result is uint8
        assert result.dtype == np.uint8, "Result should be uint8 type"
        assert result.shape == filter_mask.shape, "Result should have same shape as input"

    def test_clustering_with_padding_calculation(self):
        """Test padding calculation and application with known values."""
        # Single pixel cluster
        filter_mask = np.zeros((7, 7), dtype=bool)
        filter_mask[3, 3] = True  # Center pixel

        dt = 1.0  # 1 second per pixel
        df = 10.0  # 10 Hz per pixel
        padding_time = 1.5  # Should round up to 2 pixels padding
        padding_freq = 25.0  # Should round up to 3 pixels padding

        result = clustering(filter_mask, dt, df, padding_time, padding_freq)

        # Check that padding was applied correctly
        # Time padding: ±2 pixels, Freq padding: ±3 pixels
        # Center is (3,3), so should span [1:6, 0:7]

        # Check boundaries
        assert result[1, 3] == 1, "Time padding should extend to row 1"
        assert result[5, 3] == 1, "Time padding should extend to row 5"
        assert result[3, 0] == 1, "Freq padding should extend to column 0"
        assert result[3, 6] == 1, "Freq padding should extend to column 6"

        # Check corners are included (padding is rectangular)
        assert result[1, 0] == 1, "Padded region should be rectangular"
        assert result[5, 6] == 1, "Padded region should be rectangular"

        # Check outside padding is not included
        assert result[0, 3] == 0, "Should not extend beyond padding boundary"
        assert result[6, 3] == 0, "Should not extend beyond padding boundary"

    def test_clustering_multiple_clusters_selects_largest(self):
        """Test that clustering selects the largest cluster when multiple exist."""
        # Create two clusters: 2-pixel and 4-pixel
        filter_mask = np.array(
            [
                [True, True, False, False, False],  # 2-pixel cluster
                [False, False, False, False, False],
                [False, False, True, True, False],  # 4-pixel cluster (larger)
                [False, False, True, True, False],
                [False, False, False, False, False],
            ]
        )

        dt = 1.0
        df = 1.0
        padding_time = 0.0  # No padding to see cluster selection clearly
        padding_freq = 0.0

        result = clustering(filter_mask, dt, df, padding_time, padding_freq)

        # Should select the larger 4-pixel cluster
        assert result[2, 2] == 1, "Largest cluster pixel should be selected"
        assert result[2, 3] == 1, "Largest cluster pixel should be selected"
        assert result[3, 2] == 1, "Largest cluster pixel should be selected"
        assert result[3, 3] == 1, "Largest cluster pixel should be selected"

        # Should NOT select the smaller 2-pixel cluster
        assert result[0, 0] == 0, "Smaller cluster should not be selected"
        assert result[0, 1] == 0, "Smaller cluster should not be selected"

        # Count should match
        assert np.sum(result) == 4, "Should have exactly 4 pixels from largest cluster"

    def test_clustering_boundary_clipping(self):
        """Test that padding is properly clipped to array boundaries."""
        # Small cluster near edge
        filter_mask = np.zeros((3, 3), dtype=bool)
        filter_mask[0, 0] = True  # Top-left corner

        dt = 1.0
        df = 1.0
        padding_time = 5.0  # Large padding that would exceed boundaries
        padding_freq = 5.0

        result = clustering(filter_mask, dt, df, padding_time, padding_freq)

        # Should be clipped to array boundaries
        expected_result = np.ones((3, 3), dtype=np.uint8)
        np.testing.assert_array_equal(result, expected_result)

    def test_clustering_diagonal_connectivity(self):
        """Test that clustering uses 8-connectivity (includes diagonals)."""
        # Diagonal cluster
        filter_mask = np.array(
            [
                [True, False, False],
                [False, True, False],
                [False, False, True],
            ]
        )

        dt = 1.0
        df = 1.0
        padding_time = 0.0
        padding_freq = 0.0

        result = clustering(filter_mask, dt, df, padding_time, padding_freq)

        # All three pixels should be connected via diagonals
        assert np.sum(result) == 3, "All diagonal pixels should be connected"
        assert result[0, 0] == 1, "Diagonal pixel should be included"
        assert result[1, 1] == 1, "Diagonal pixel should be included"
        assert result[2, 2] == 1, "Diagonal pixel should be included"

    def test_clustering_empty_filter(self):
        """Test clustering behavior with empty filter."""
        filter_mask = np.zeros((5, 5), dtype=bool)

        dt = 1.0
        df = 1.0
        padding_time = 1.0
        padding_freq = 1.0

        # This should handle empty input gracefully
        with pytest.raises(ValueError):
            # max() on empty sequence should raise ValueError
            clustering(filter_mask, dt, df, padding_time, padding_freq)

    def test_clustering_dtype_and_shape_consistency(self):
        """Test that output dtype and shape are consistent."""
        filter_mask = np.array(
            [
                [True, False, True],
                [False, False, False],
                [True, False, False],
            ]
        )

        dt = 0.5
        df = 2.0
        result = clustering(filter_mask, dt, df, padding_time=0.1, padding_freq=1.0)

        # Check output properties
        assert result.dtype == np.uint8, "Output should be uint8"
        assert result.shape == filter_mask.shape, "Output shape should match input"
        assert np.all((result == 0) | (result == 1)), "Output should be binary (0 or 1)"

    def test_clustering_with_different_padding_values(self):
        """Test clustering with various padding parameter combinations."""
        # Small cluster in center
        filter_mask = np.zeros((10, 10), dtype=bool)
        filter_mask[5, 5] = True

        dt = 1.0
        df = 1.0

        # Test different padding combinations
        test_cases = [
            (0.0, 0.0),  # No padding
            (1.0, 0.0),  # Time padding only
            (0.0, 1.0),  # Freq padding only
            (2.0, 3.0),  # Different padding values
        ]

        for padding_time, padding_freq in test_cases:
            result = clustering(filter_mask, dt, df, padding_time, padding_freq)

            # Core pixel should always be included
            assert result[5, 5] == 1, f"Core pixel should be included for padding ({padding_time}, {padding_freq})"

            # Check that padding is applied correctly
            time_pad_pixels = int(np.ceil(padding_time / dt))
            freq_pad_pixels = int(np.ceil(padding_freq / df))

            # Verify extent
            if padding_time > 0:
                expected_time_range = max(0, 5 - time_pad_pixels), min(10, 5 + time_pad_pixels + 1)
                assert result[expected_time_range[0], 5] == 1, "Time padding should be applied"
                assert result[expected_time_range[1] - 1, 5] == 1, "Time padding should be applied"

            if padding_freq > 0:
                expected_freq_range = max(0, 5 - freq_pad_pixels), min(10, 5 + freq_pad_pixels + 1)
                assert result[5, expected_freq_range[0]] == 1, "Freq padding should be applied"
                assert result[5, expected_freq_range[1] - 1] == 1, "Freq padding should be applied"
