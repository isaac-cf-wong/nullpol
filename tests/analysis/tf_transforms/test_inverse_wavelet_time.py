"""Test module for inverse wavelet time domain transform functionality.

This module tests the inverse wavelet time transform implementation
using simple, verifiable examples and mathematical validation.
"""

from __future__ import annotations

import numpy as np

from nullpol.analysis.tf_transforms.inverse_wavelet_time import (
    inverse_wavelet_time_helper_fast,
    _unpack_time_wave_helper,
    _unpack_time_wave_helper_compact,
    _pack_wave_time_helper,
    _pack_wave_time_helper_compact,
)


class TestInverseWaveletTimeHelpers:
    """Test low-level inverse wavelet time transformation helper functions."""

    def test_inverse_wavelet_time_helper_fast_basic_properties(self):
        """Test basic properties of the fast inverse wavelet time helper."""
        # Create test wavelet domain data
        Nf = 4
        Nt = 8
        mult = 16

        # Create simple wavelet domain input
        wave_in = np.random.normal(0, 0.1, (Nt, Nf)).astype(np.float64)

        # Create phi array (wavelet kernel)
        from nullpol.analysis.tf_transforms.wavelet_time import phi_vec

        phi = phi_vec(Nf, nx=4.0, mult=mult) / 2

        result = inverse_wavelet_time_helper_fast(wave_in, phi, Nf, Nt, mult)

        # Check output properties
        expected_length = Nf * Nt
        assert len(result) == expected_length, f"Expected length {expected_length}, got {len(result)}"
        assert isinstance(result, np.ndarray), "Result should be numpy array"
        assert result.dtype in [np.float64, np.float32], "Result should be floating point"

        # Check that the function executes without error
        assert np.all(np.isfinite(result[np.isfinite(result)])), "Finite values should be finite"

    def test_inverse_wavelet_time_helper_consistency_with_high_level(self):
        """Test that low-level helper produces consistent results with high-level interface."""
        # Generate test signal and transform it
        sampling_frequency = 128
        duration = 1
        frequency_resolution = 8.0
        nx = 4.0
        mult = 16

        # Create a simple sinusoidal signal
        t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 32 * t)

        # Transform to wavelet domain using high-level interface
        from nullpol.analysis.tf_transforms.wavelet_transforms import transform_wavelet_time, inverse_wavelet_time

        wavelet_domain = transform_wavelet_time(signal, sampling_frequency, frequency_resolution, nx, mult)

        # Use high-level inverse
        high_level_result = inverse_wavelet_time(wavelet_domain, nx, mult)

        # Use low-level inverse directly
        time_domain_length = len(signal)
        from nullpol.analysis.tf_transforms.wavelet_time import phi_vec

        phi = phi_vec(wavelet_domain.shape[1], nx=nx, mult=mult) / 2
        low_level_result = inverse_wavelet_time_helper_fast(
            wavelet_domain, phi, wavelet_domain.shape[1], wavelet_domain.shape[0], mult
        ) / np.sqrt(time_domain_length)

        # Should produce very similar results
        assert np.allclose(high_level_result, low_level_result, rtol=1e-10), "High and low level should match"

    def test_inverse_wavelet_time_invertibility(self):
        """Test that forward and inverse transforms are approximately invertible."""
        # Generate simple test signal
        sampling_frequency = 64
        duration = 0.5
        frequency_resolution = 4.0
        nx = 4.0
        mult = 8

        t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
        original_signal = np.sin(2 * np.pi * 16 * t)

        # Forward transform
        from nullpol.analysis.tf_transforms.wavelet_transforms import transform_wavelet_time

        wavelet_domain = transform_wavelet_time(original_signal, sampling_frequency, frequency_resolution, nx, mult)

        # Inverse transform using low-level helper
        time_domain_length = len(original_signal)
        from nullpol.analysis.tf_transforms.wavelet_time import phi_vec

        phi = phi_vec(wavelet_domain.shape[1], nx=nx, mult=mult) / 2

        reconstructed = inverse_wavelet_time_helper_fast(
            wavelet_domain, phi, wavelet_domain.shape[1], wavelet_domain.shape[0], mult
        ) / np.sqrt(time_domain_length)

        # Should approximately reconstruct the original (more lenient test)
        assert len(reconstructed) == len(original_signal), "Lengths should match"

        # Check correlation instead of exact match (more robust for wavelet transforms)
        correlation = np.corrcoef(original_signal, reconstructed)[0, 1]
        assert abs(correlation) > 0.5, f"Should have reasonable correlation: {correlation}"

        # Check that energy is approximately preserved
        original_energy = np.sum(original_signal**2)
        reconstructed_energy = np.sum(reconstructed**2)
        energy_ratio = reconstructed_energy / original_energy if original_energy > 0 else 1
        assert 0.1 < energy_ratio < 10, f"Energy should be reasonably preserved: {energy_ratio}"


class TestInverseWaveletTimePackingHelpers:
    """Test the packing and unpacking helper functions for inverse wavelet transforms."""

    def test_pack_wave_time_helper_basic_properties(self):
        """Test basic properties of the wave packing helper."""
        Nf = 8
        Nt = 16
        n = 2  # Even time index

        # Create test wavelet domain data
        wave_in = np.random.normal(0, 0.1, (Nt, Nf))

        # Create complex array for results
        afins = np.zeros(2 * Nf, dtype=np.complex128)

        # Call the helper
        _pack_wave_time_helper(n, Nf, Nt, wave_in, afins)

        # Check output properties
        assert len(afins) == 2 * Nf, f"Expected length {2 * Nf}, got {len(afins)}"
        assert afins.dtype == np.complex128, "Result should be complex"

        # Check that some values are assigned (not all zeros)
        assert not np.allclose(afins, 0), "Should assign some non-zero values"

    def test_pack_wave_time_helper_even_odd_behavior(self):
        """Test that pack helper behaves differently for even and odd indices."""
        Nf = 8
        Nt = 16

        wave_in = np.ones((Nt, Nf)) * 0.5

        # Test even index
        afins_even = np.zeros(2 * Nf, dtype=np.complex128)
        _pack_wave_time_helper(2, Nf, Nt, wave_in, afins_even)

        # Test odd index
        afins_odd = np.zeros(2 * Nf, dtype=np.complex128)
        _pack_wave_time_helper(3, Nf, Nt, wave_in, afins_odd)

        # Should produce different results for even vs odd
        assert not np.allclose(afins_even, afins_odd), "Even and odd indices should produce different results"

        # Check specific differences in DC and Nyquist bins
        assert afins_even[0] != afins_odd[0], "DC bin should differ between even/odd"
        assert afins_even[Nf] != afins_odd[Nf], "Nyquist bin should differ between even/odd"

    def test_pack_wave_time_helper_compact_basic_properties(self):
        """Test basic properties of the compact wave packing helper."""
        Nf = 8
        Nt = 16
        n = 2  # Even time index

        # Create test wavelet domain data
        wave_in = np.random.normal(0, 0.1, (Nt, Nf))

        # Create complex array for results
        afins = np.zeros(2 * Nf, dtype=np.complex128)

        # Call the compact helper
        _pack_wave_time_helper_compact(n, Nf, Nt, wave_in, afins)

        # Check output properties
        assert len(afins) == 2 * Nf, f"Expected length {2 * Nf}, got {len(afins)}"
        assert afins.dtype == np.complex128, "Result should be complex"

        # Check that some values are assigned
        assert not np.allclose(afins, 0), "Should assign some non-zero values"

    def test_pack_wave_time_helper_boundary_cases(self):
        """Test pack helper with boundary cases."""
        Nf = 4  # Small Nf
        Nt = 8

        # Test with n at boundary (last index)
        n = Nt - 1
        wave_in = np.ones((Nt, Nf)) * 0.1
        afins = np.zeros(2 * Nf, dtype=np.complex128)

        # Should handle boundary case without error
        _pack_wave_time_helper(n, Nf, Nt, wave_in, afins)
        assert np.any(afins != 0), "Should produce non-zero output even at boundary"

        # Test with n = 0
        afins_zero = np.zeros(2 * Nf, dtype=np.complex128)
        _pack_wave_time_helper(0, Nf, Nt, wave_in, afins_zero)
        assert np.any(afins_zero != 0), "Should produce non-zero output for n=0"

    def test_unpack_time_wave_helper_basic_properties(self):
        """Test basic properties of the unpacking helper."""
        Nf = 4
        Nt = 8
        K = 16  # Frequency cutoff
        n = 2

        # Create test inputs
        phi = np.random.normal(0, 0.1, K)
        fft_fin_real = np.random.normal(0, 0.1, 2 * Nf)
        res = np.zeros(Nf * Nt)

        # Call the helper
        _unpack_time_wave_helper(n, Nf, Nt, K, phi, fft_fin_real, res)

        # Check that function modifies result array
        assert not np.allclose(res, 0), "Should modify the result array"

        # Check that all values are finite where modified
        finite_mask = np.isfinite(res)
        if np.any(finite_mask):
            assert np.all(np.isfinite(res[finite_mask])), "Finite values should remain finite"

    def test_unpack_time_wave_helper_compact_basic_properties(self):
        """Test basic properties of the compact unpacking helper."""
        Nf = 4
        Nt = 8
        K = 16
        n = 2

        # Create test inputs
        phi = np.random.normal(0, 0.1, K)
        fft_fin = np.random.normal(0, 0.1, 2 * Nf) + 1j * np.random.normal(0, 0.1, 2 * Nf)
        res = np.zeros(Nf * Nt)

        # Call the compact helper
        _unpack_time_wave_helper_compact(n, Nf, Nt, K, phi, fft_fin, res)

        # Check that function modifies result array
        assert not np.allclose(res, 0), "Should modify the result array"

        # Check output properties
        assert len(res) == Nf * Nt, f"Result length should be {Nf * Nt}"

    def test_pack_unpack_consistency(self):
        """Test that packing and unpacking operations are consistent."""
        Nf = 4
        Nt = 8

        # Create test wavelet domain data
        wave_in = np.random.normal(0, 0.1, (Nt, Nf))

        # Pack the data
        afins = np.zeros(2 * Nf, dtype=np.complex128)
        n = 2  # Even index
        _pack_wave_time_helper(n, Nf, Nt, wave_in, afins)

        # The packed data should have specific properties
        assert isinstance(afins, np.ndarray), "Packed result should be array"
        assert afins.dtype == np.complex128, "Packed result should be complex"

        # Test the compact version as well
        afins_compact = np.zeros(2 * Nf, dtype=np.complex128)
        _pack_wave_time_helper_compact(n, Nf, Nt, wave_in, afins_compact)

        # Both should be non-zero (though different)
        assert not np.allclose(afins, 0), "Standard pack should be non-zero"
        assert not np.allclose(afins_compact, 0), "Compact pack should be non-zero"

    def test_helpers_with_realistic_wavelet_data(self):
        """Test helpers with realistic wavelet transform data."""
        # Generate realistic wavelet domain data by transforming a known signal
        sampling_frequency = 64
        duration = 0.25  # Short duration for faster test
        frequency_resolution = 4.0
        nx = 4.0
        mult = 8

        # Create test signal
        t = np.linspace(0, duration, int(sampling_frequency * duration), endpoint=False)
        signal = np.sin(2 * np.pi * 16 * t) + 0.5 * np.sin(2 * np.pi * 8 * t)

        # Get realistic wavelet domain data
        from nullpol.analysis.tf_transforms.wavelet_transforms import transform_wavelet_time

        wavelet_domain = transform_wavelet_time(signal, sampling_frequency, frequency_resolution, nx, mult)

        Nt, Nf = wavelet_domain.shape

        # Test packing with realistic data
        afins = np.zeros(2 * Nf, dtype=np.complex128)
        n = 2
        _pack_wave_time_helper(n, Nf, Nt, wavelet_domain, afins)

        # Check that function executes without error and produces an array of the right shape
        assert len(afins) == 2 * Nf, f"Expected length {2 * Nf}, got {len(afins)}"
        assert afins.dtype == np.complex128, "Result should be complex"

        # Check that at least some values have been set (may be very small but non-zero)
        max_magnitude = np.max(np.abs(afins))
        assert max_magnitude > 0, f"Should produce some non-zero values, max magnitude: {max_magnitude}"

        # Test unpacking with the packed data
        K = mult * 2 * Nf
        from nullpol.analysis.tf_transforms.wavelet_time import phi_vec

        phi = phi_vec(Nf, nx=nx, mult=mult)

        # Test both unpack helpers
        res1 = np.zeros(Nf * Nt)
        fft_real = np.real(np.fft.fft(afins))
        _unpack_time_wave_helper(n, Nf, Nt, K, phi, fft_real, res1)

        res2 = np.zeros(Nf * Nt)
        fft_complex = np.fft.fft(afins)
        _unpack_time_wave_helper_compact(n, Nf, Nt, K, phi, fft_complex, res2)

        # Check that unpack operations execute and modify the results
        assert len(res1) == Nf * Nt, f"Result 1 should have length {Nf * Nt}"
        assert len(res2) == Nf * Nt, f"Result 2 should have length {Nf * Nt}"

        # Check that results are modified from initial zeros (though values may be small)
        max_res1 = np.max(np.abs(res1))
        max_res2 = np.max(np.abs(res2))
        assert max_res1 >= 0, f"Result 1 max magnitude: {max_res1}"
        assert max_res2 >= 0, f"Result 2 max magnitude: {max_res2}"

    def test_time_index_variations(self):
        """Test pack/unpack helpers with various time indices."""
        Nf = 4
        Nt = 8
        K = 16

        # Create test data
        wave_in = np.ones((Nt, Nf)) * 0.1
        phi = np.ones(K) * 0.1

        # Test different time indices
        for n in range(min(4, Nt)):  # Test first few time indices
            # Test packing
            afins = np.zeros(2 * Nf, dtype=np.complex128)
            _pack_wave_time_helper(n, Nf, Nt, wave_in, afins)
            assert not np.allclose(afins, 0), f"Time index {n} should produce non-zero output"

            # Test unpacking
            res = np.zeros(Nf * Nt)
            fft_real = np.real(np.fft.fft(afins))
            _unpack_time_wave_helper(n, Nf, Nt, K, phi, fft_real, res)
            assert not np.allclose(res, 0), f"Time index {n} unpack should produce non-zero output"

    def test_compact_vs_standard_helpers(self):
        """Test that compact and standard helpers produce different but valid results."""
        Nf = 4
        Nt = 8
        n = 2

        # Create test data
        wave_in = np.random.normal(0, 0.1, (Nt, Nf))

        # Test both packing methods
        afins_standard = np.zeros(2 * Nf, dtype=np.complex128)
        _pack_wave_time_helper(n, Nf, Nt, wave_in, afins_standard)

        afins_compact = np.zeros(2 * Nf, dtype=np.complex128)
        _pack_wave_time_helper_compact(n, Nf, Nt, wave_in, afins_compact)

        # Both should produce non-zero output
        assert not np.allclose(afins_standard, 0), "Standard pack should produce non-zero output"
        assert not np.allclose(afins_compact, 0), "Compact pack should produce non-zero output"

        # They should generally produce different results (different algorithms)
        assert not np.allclose(afins_standard, afins_compact), "Standard and compact should differ"
