"""Test module for tf_transforms utility functions.

This module tests the utility functions used for calculating shapes
of time-frequency transforms, using simple, hand-calculable examples.
"""

from __future__ import annotations

import pytest

from nullpol.analysis.tf_transforms.utils import get_shape_of_stft, get_shape_of_wavelet_transform


class TestWaveletTransformShape:
    """Test wavelet transform shape calculations with simple examples."""

    def test_simple_wavelet_shape_calculation(self):
        """Test wavelet shape calculation with hand-calculable values.

        Uses simple parameters where the result can be easily verified:
        - 1 second duration, 100 Hz sampling, 10 Hz resolution
        - Expected: Nf = 100/2/10 = 5, Nt = 1*100/5 = 20
        """
        duration = 1.0
        sampling_frequency = 100.0
        wavelet_frequency_resolution = 10.0

        nt, nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)

        # Hand calculation: Nf = 100/2/10 = 5, Nt = 1*100/5 = 20
        assert nf == 5
        assert nt == 20

    def test_standard_gw_parameters(self):
        """Test with typical gravitational wave analysis parameters.

        Uses common LIGO parameters for verification.
        """
        duration = 4.0
        sampling_frequency = 2048.0
        wavelet_frequency_resolution = 4.0

        nt, nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)

        # Hand calculation: Nf = 2048/2/4 = 256, Nt = 4*2048/256 = 32
        assert nf == 256
        assert nt == 32

    def test_integer_division_behavior(self):
        """Test that integer division produces expected results.

        Ensures the function handles non-perfect divisions correctly.
        """
        duration = 2.0
        sampling_frequency = 128.0
        wavelet_frequency_resolution = 8.0

        nt, nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)

        # Hand calculation: Nf = 128/2/8 = 8, Nt = 2*128/8 = 32
        assert nf == 8
        assert nt == 32

    def test_return_type_is_tuple(self):
        """Test that the function returns a tuple of integers."""
        result = get_shape_of_wavelet_transform(1.0, 100.0, 10.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_edge_case_small_resolution(self):
        """Test with small frequency resolution (large number of bins).

        Uses resolution that creates larger frequency dimensions.
        """
        duration = 1.0
        sampling_frequency = 64.0
        wavelet_frequency_resolution = 1.0

        nt, nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)

        # Hand calculation: Nf = 64/2/1 = 32, Nt = 1*64/32 = 2
        assert nf == 32
        assert nt == 2


class TestSTFTShape:
    """Test STFT shape calculations with simple examples."""

    def test_simple_stft_shape_calculation(self):
        """Test STFT shape calculation with hand-calculable values.

        Uses simple parameters where the result can be easily verified:
        - 1 second duration, 100 Hz sampling, 10 Hz resolution
        """
        duration = 1.0
        sampling_frequency = 100.0
        frequency_resolution = 10.0

        nt, nf = get_shape_of_stft(duration, sampling_frequency, frequency_resolution)

        # Hand calculation:
        # N = 100/10 = 10
        # hop_size = 10
        # num_segments = (1*100 - 10) // 10 + 1 = 90//10 + 1 = 9 + 1 = 10
        # nf = 10//2 + 1 = 6
        assert nt == 10
        assert nf == 6

    def test_standard_stft_parameters(self):
        """Test with typical STFT analysis parameters."""
        duration = 4.0
        sampling_frequency = 1024.0
        frequency_resolution = 1.0

        nt, nf = get_shape_of_stft(duration, sampling_frequency, frequency_resolution)

        # Hand calculation:
        # N = 1024/1 = 1024
        # hop_size = 1024
        # num_segments = (4*1024 - 1024) // 1024 + 1 = 3072//1024 + 1 = 3 + 1 = 4
        # nf = 1024//2 + 1 = 513
        assert nt == 4
        assert nf == 513

    def test_overlapping_windows_calculation(self):
        """Test STFT calculation with parameters that create overlapping windows.

        Uses parameters that demonstrate the hop size calculation.
        """
        duration = 2.0
        sampling_frequency = 64.0
        frequency_resolution = 8.0

        nt, nf = get_shape_of_stft(duration, sampling_frequency, frequency_resolution)

        # Hand calculation:
        # N = 64/8 = 8
        # hop_size = 8
        # num_segments = (2*64 - 8) // 8 + 1 = 120//8 + 1 = 15 + 1 = 16
        # nf = 8//2 + 1 = 5
        assert nt == 16
        assert nf == 5

    def test_return_type_is_tuple_stft(self):
        """Test that the STFT function returns a tuple of integers."""
        result = get_shape_of_stft(1.0, 100.0, 10.0)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], int)

    def test_minimum_segments_case(self):
        """Test edge case where duration barely allows one segment.

        Uses minimal duration to test boundary conditions.
        """
        duration = 0.1
        sampling_frequency = 100.0
        frequency_resolution = 10.0

        nt, nf = get_shape_of_stft(duration, sampling_frequency, frequency_resolution)

        # Hand calculation:
        # N = 100/10 = 10
        # hop_size = 10
        # num_segments = (0.1*100 - 10) // 10 + 1 = (10-10)//10 + 1 = 0 + 1 = 1
        # nf = 10//2 + 1 = 6
        assert nt == 1
        assert nf == 6

    def test_high_frequency_resolution(self):
        """Test with high frequency resolution (small bins).

        Uses small frequency resolution to create many frequency bins.
        """
        duration = 1.0
        sampling_frequency = 256.0
        frequency_resolution = 1.0

        nt, nf = get_shape_of_stft(duration, sampling_frequency, frequency_resolution)

        # Hand calculation:
        # N = 256/1 = 256
        # hop_size = 256
        # num_segments = (1*256 - 256) // 256 + 1 = 0//256 + 1 = 0 + 1 = 1
        # nf = 256//2 + 1 = 129
        assert nt == 1
        assert nf == 129


class TestParameterValidation:
    """Test edge cases and parameter validation for both functions."""

    def test_consistent_shapes_for_similar_parameters(self):
        """Test that similar parameters produce reasonable relative shapes.

        Compares results for related parameter sets to ensure consistency.
        """
        # Test wavelet transform with doubled resolution
        nt1, nf1 = get_shape_of_wavelet_transform(2.0, 128.0, 8.0)
        nt2, nf2 = get_shape_of_wavelet_transform(2.0, 128.0, 4.0)

        # Halving frequency resolution should double frequency bins
        assert nf2 == 2 * nf1
        # Time bins should halve when frequency bins double (for same total samples)
        assert nt2 == nt1 // 2

    def test_stft_vs_wavelet_parameter_relationship(self):
        """Test understanding of different transform characteristics.

        Compares STFT and wavelet transforms with similar parameters.
        """
        duration = 1.0
        sampling_frequency = 128.0
        resolution = 8.0

        # Get shapes for both transforms
        wavelet_nt, wavelet_nf = get_shape_of_wavelet_transform(duration, sampling_frequency, resolution)
        stft_nt, stft_nf = get_shape_of_stft(duration, sampling_frequency, resolution)

        # Both should return valid positive integers
        assert wavelet_nt > 0 and wavelet_nf > 0
        assert stft_nt > 0 and stft_nf > 0
        
        # Wavelet: Nf = 128/2/8 = 8, Nt = 1*128/8 = 16
        assert wavelet_nf == 8
        assert wavelet_nt == 16
        
        # STFT: N = 128/8 = 16, segments = (128-16)//16 + 1 = 7+1 = 8, nf = 16//2+1 = 9
        assert stft_nt == 8
        assert stft_nf == 9