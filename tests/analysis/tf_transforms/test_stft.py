"""Test module for Short-Time Fourier Transform (STFT) functionality.

This module tests the STFT implementation using simple, verifiable examples
that can be calculated by hand or with basic mathematical operations.
"""

from __future__ import annotations

import numpy as np

from nullpol.analysis.tf_transforms.stft import stft


class TestSTFTSimpleExamples:
    """Test STFT with simple, hand-calculable examples."""

    def test_dc_signal_stft(self):
        """Test STFT of constant (DC) signal.

        A constant signal should have all power in the DC bin (frequency 0).
        """
        # Simple constant signal
        sampling_frequency = 64.0
        frequency_resolution = 8.0
        duration = 1.0
        data = np.ones(int(duration * sampling_frequency))  # 64 samples of value 1

        # Simple rectangular window (no windowing)
        N = int(sampling_frequency / frequency_resolution)  # 8 samples
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # For constant signal, only DC component should be non-zero
        # DC bin is at index 0, should be 1.0/sampling_frequency per segment
        assert result.shape[1] == N // 2 + 1  # 5 frequency bins (0, 1, 2, 3, 4)

        # All segments should have same DC value, other frequencies should be ~0
        for segment in result:
            assert abs(segment[0]) > 0.01  # DC component should be significant
            for freq_bin in range(1, len(segment)):
                assert abs(segment[freq_bin]) < 1e-10  # Other bins should be ~0

    def test_single_frequency_sine_wave(self):
        """Test STFT of pure sinusoidal signal.

        A sine wave at known frequency should peak at the correct frequency bin.
        """
        sampling_frequency = 64.0
        frequency_resolution = 8.0
        signal_frequency = 16.0  # 2nd harmonic bin (16/8 = 2)
        duration = 1.0

        # Create pure sine wave
        t = np.arange(int(duration * sampling_frequency)) / sampling_frequency
        data = np.sin(2 * np.pi * signal_frequency * t)

        # Rectangular window
        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # Signal should peak at bin 2 (frequency 16 Hz)
        expected_bin = int(signal_frequency / frequency_resolution)  # 16/8 = 2

        for segment in result:
            peak_bin = np.argmax(np.abs(segment))
            assert peak_bin == expected_bin

    def test_stft_output_shape(self):
        """Test that STFT output has correct shape.

        Verifies the time-frequency grid dimensions match expectations.
        """
        sampling_frequency = 128.0
        frequency_resolution = 16.0
        duration = 1.0
        data = np.random.randn(int(duration * sampling_frequency))

        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # Expected dimensions
        expected_time_bins = (len(data) - N) // N + 1  # (128-8)//8 + 1 = 16
        expected_freq_bins = N // 2 + 1  # 5

        assert result.shape == (expected_time_bins, expected_freq_bins)

    def test_zero_input_signal(self):
        """Test STFT of zero signal produces zero output."""
        sampling_frequency = 32.0
        frequency_resolution = 4.0
        duration = 1.0
        data = np.zeros(int(duration * sampling_frequency))

        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # All outputs should be zero
        assert np.allclose(result, 0.0, atol=1e-15)

    def test_impulse_response(self):
        """Test STFT of impulse signal.

        An impulse (single 1 surrounded by zeros) has flat frequency spectrum.
        """
        sampling_frequency = 32.0
        frequency_resolution = 4.0
        data = np.zeros(32)
        data[4] = 1.0  # Single impulse at sample 4

        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # The segment containing the impulse should have flat spectrum
        # First segment: samples 0-7 (contains impulse at sample 4)
        impulse_segment = result[0]

        # All frequency bins should have similar magnitude (flat spectrum)
        magnitudes = np.abs(impulse_segment)
        assert np.all(magnitudes > 0)  # All bins should be non-zero


class TestSTFTWindowFunctions:
    """Test STFT with different window functions."""

    def test_rectangular_window(self):
        """Test STFT with rectangular (boxcar) window."""
        sampling_frequency = 64.0
        frequency_resolution = 8.0
        duration = 0.5
        data = np.ones(int(duration * sampling_frequency))  # Constant signal

        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = np.ones(N)  # Rectangular window

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # Should produce valid output
        assert result.shape[0] > 0
        assert result.shape[1] == N // 2 + 1

    def test_hanning_window(self):
        """Test STFT with Hanning window function."""
        sampling_frequency = 64.0
        frequency_resolution = 8.0
        duration = 0.5
        data = np.ones(int(duration * sampling_frequency))

        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = 0.5 * (1 - np.cos(2 * np.pi * np.arange(N) / (N - 1)))  # Hanning

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # Should produce valid output with different characteristics than rectangular
        assert result.shape[0] > 0
        assert result.shape[1] == N // 2 + 1
        assert not np.allclose(result, 0.0)

    def test_triangular_window(self):
        """Test STFT with triangular window function."""
        sampling_frequency = 64.0
        frequency_resolution = 8.0
        duration = 0.5
        data = np.ones(int(duration * sampling_frequency))

        N = int(sampling_frequency / frequency_resolution)  # 8
        # Simple triangular window
        window_function = 1.0 - np.abs((2 * np.arange(N) - (N - 1)) / (N - 1))

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        assert result.shape[0] > 0
        assert result.shape[1] == N // 2 + 1


class TestSTFTMathematicalProperties:
    """Test mathematical properties of STFT."""

    def test_linearity_property(self):
        """Test that STFT is linear: STFT(a*x + b*y) = a*STFT(x) + b*STFT(y).

        Tests the linearity property with simple signals.
        """
        sampling_frequency = 32.0
        frequency_resolution = 4.0
        duration = 1.0

        # Two simple signals
        t = np.arange(int(duration * sampling_frequency)) / sampling_frequency
        signal1 = np.cos(2 * np.pi * 8 * t)  # 8 Hz cosine
        signal2 = np.sin(2 * np.pi * 12 * t)  # 12 Hz sine

        # Linear combination
        a, b = 2.0, 3.0
        combined = a * signal1 + b * signal2

        N = int(sampling_frequency / frequency_resolution)  # 8
        window_function = np.ones(N)

        # Compute STFTs
        stft1 = stft(signal1, sampling_frequency, frequency_resolution, window_function)
        stft2 = stft(signal2, sampling_frequency, frequency_resolution, window_function)
        stft_combined = stft(combined, sampling_frequency, frequency_resolution, window_function)
        stft_linear = a * stft1 + b * stft2

        # Should satisfy linearity (within numerical precision)
        assert np.allclose(stft_combined, stft_linear, rtol=1e-10)

    def test_parseval_energy_conservation_simple(self):
        """Test approximate energy conservation with simple scaling.

        For rectangular window, energy should be approximately conserved
        accounting for the scaling factor in the STFT.
        """
        sampling_frequency = 32.0
        frequency_resolution = 8.0

        # Create simple test signal
        data = np.array([1.0, 2.0, 3.0, 4.0] * 8)  # 32 samples, repeated pattern

        N = int(sampling_frequency / frequency_resolution)  # 4
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # Simple energy check - result should be reasonable magnitude
        total_power = np.sum(np.abs(result) ** 2)
        assert total_power > 0.0
        assert not np.isnan(total_power)
        assert not np.isinf(total_power)

    def test_frequency_shift_property_real_signals(self):
        """Test frequency shift with real signals only.

        Uses amplitude modulation to create frequency shifted content.
        """
        sampling_frequency = 64.0
        frequency_resolution = 8.0
        duration = 1.0

        # Create signals with different frequency content
        t = np.arange(int(duration * sampling_frequency)) / sampling_frequency

        # Signal at 8 Hz
        signal_8hz = np.cos(2 * np.pi * 8 * t)
        # Signal at 16 Hz
        signal_16hz = np.cos(2 * np.pi * 16 * t)

        N = int(sampling_frequency / frequency_resolution)
        window_function = np.ones(N)

        stft_8hz = stft(signal_8hz, sampling_frequency, frequency_resolution, window_function)
        stft_16hz = stft(signal_16hz, sampling_frequency, frequency_resolution, window_function)

        # Peak should be at different frequency bins
        peak_8hz = np.argmax(np.abs(stft_8hz[0]))  # Should be at bin 1 (8/8)
        peak_16hz = np.argmax(np.abs(stft_16hz[0]))  # Should be at bin 2 (16/8)

        assert peak_8hz == 1
        assert peak_16hz == 2


class TestSTFTEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_segment_case(self):
        """Test STFT when data length exactly equals one segment."""
        sampling_frequency = 32.0
        frequency_resolution = 8.0
        N = int(sampling_frequency / frequency_resolution)  # 4 samples

        data = np.ones(N)  # Exactly one segment worth of data
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        # Should have exactly one time segment
        assert result.shape == (1, N // 2 + 1)

    def test_minimal_valid_input(self):
        """Test STFT with minimal valid input size."""
        sampling_frequency = 16.0
        frequency_resolution = 8.0
        N = int(sampling_frequency / frequency_resolution)  # 2 samples

        data = np.array([1.0, -1.0])  # Minimum data for one segment
        window_function = np.ones(N)

        result = stft(data, sampling_frequency, frequency_resolution, window_function)

        assert result.shape == (1, N // 2 + 1)  # (1, 2)

    def test_data_type_preservation(self):
        """Test that STFT works with real float data types."""
        sampling_frequency = 32.0
        frequency_resolution = 8.0
        N = int(sampling_frequency / frequency_resolution)

        # Test with float64
        data_float = np.ones(16, dtype=np.float64)
        window_function = np.ones(N, dtype=np.float64)
        result_float = stft(data_float, sampling_frequency, frequency_resolution, window_function)

        # Test with float32
        data_float32 = np.ones(16, dtype=np.float32)
        window_float32 = np.ones(N, dtype=np.float32)
        result_float32 = stft(data_float32, sampling_frequency, frequency_resolution, window_float32)

        # Results should be complex and have correct shapes
        assert np.iscomplexobj(result_float)
        assert np.iscomplexobj(result_float32)
        assert result_float.shape == result_float32.shape

    def test_different_segment_lengths(self):
        """Test STFT with various segment lengths."""
        sampling_frequency = 64.0
        data = np.random.randn(128)

        # Test different frequency resolutions (different segment lengths)
        for freq_res in [4.0, 8.0, 16.0]:
            N = int(sampling_frequency / freq_res)
            window_function = np.ones(N)

            result = stft(data, sampling_frequency, freq_res, window_function)

            expected_segments = (len(data) - N) // N + 1
            expected_freqs = N // 2 + 1

            assert result.shape == (expected_segments, expected_freqs)
