"""Test module for wavelet time-domain transform functionality.

This module tests the time-domain wavelet transform implementation
using simple, verifiable examples and mathematical validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.tf_transforms.wavelet_time import (
    _assign_wdata,
    _pack_wave,
    phi_vec,
    transform_wavelet_time_helper,
)


class TestPhiVec:
    """Test phi_vec wavelet generation with simple examples."""

    def test_phi_vec_basic_properties(self):
        """Test basic properties of phi_vec wavelet.

        Verifies the wavelet has expected shape and normalization properties.
        """
        Nf = 8
        nx = 4.0
        mult = 16

        phi = phi_vec(Nf, nx, mult)

        # Should return array of correct length
        expected_length = mult * 2 * Nf  # 16 * 2 * 8 = 256
        assert len(phi) == expected_length
        assert isinstance(phi, np.ndarray)
        assert phi.dtype in [np.float64, np.float32]

    def test_phi_vec_small_parameters(self):
        """Test phi_vec with small, hand-calculable parameters."""
        Nf = 2
        nx = 4.0
        mult = 4

        phi = phi_vec(Nf, nx, mult)

        # Should return array of length mult * 2 * Nf = 4 * 2 * 2 = 16
        assert len(phi) == 16
        assert not np.any(np.isnan(phi))
        assert not np.any(np.isinf(phi))

    def test_phi_vec_normalization_consistency(self):
        """Test that phi_vec produces consistent normalization.

        For different nx values, the wavelet should maintain reasonable magnitude.
        """
        Nf = 4
        mult = 8

        for nx in [2.0, 4.0, 8.0]:
            phi = phi_vec(Nf, nx, mult)
            
            # Should have reasonable magnitude (not too large or small)
            max_val = np.max(np.abs(phi))
            assert 0.01 < max_val < 100.0
            
            # Should not be all zeros
            assert np.sum(np.abs(phi)) > 1e-10

    def test_phi_vec_different_mult_values(self):
        """Test phi_vec with different multiplier values."""
        Nf = 4
        nx = 4.0

        for mult in [4, 8, 16]:
            phi = phi_vec(Nf, nx, mult)
            expected_length = mult * 2 * Nf
            assert len(phi) == expected_length

    def test_phi_vec_real_output(self):
        """Test that phi_vec returns real-valued output."""
        Nf = 4
        nx = 4.0
        mult = 8

        phi = phi_vec(Nf, nx, mult)

        # Should be real (no imaginary component)
        assert np.all(np.isreal(phi))


class TestAssignWdata:
    """Test _assign_wdata function with simple examples."""

    def test_assign_wdata_simple_case(self):
        """Test _assign_wdata with simple parameters.

        Uses hand-calculable values to verify windowing operation.
        """
        # Simple parameters
        K = 8
        ND = 16
        Nf = 4
        i = 0  # First time index
        
        # Simple data and window
        data_pad = np.arange(ND + K, dtype=float)  # 0, 1, 2, ..., 23
        phi = np.ones(K)  # Unity window
        wdata = np.zeros(K)
        
        _assign_wdata(i, K, ND, Nf, wdata, data_pad, phi)
        
        # For i=0, jj = 0*4 - 8//2 = -4, wrapped to ND-4 = 12
        # So wdata should get data_pad[12:20] * phi
        expected = data_pad[12:20]  # [12, 13, 14, 15, 16, 17, 18, 19]
        
        assert np.allclose(wdata, expected)

    def test_assign_wdata_windowing_effect(self):
        """Test that _assign_wdata applies windowing correctly."""
        K = 4
        ND = 8
        Nf = 2
        i = 1
        
        # Simple data
        data_pad = np.ones(ND + K)
        phi = np.array([0.5, 1.0, 1.5, 2.0])  # Non-unity window
        wdata = np.zeros(K)
        
        _assign_wdata(i, K, ND, Nf, wdata, data_pad, phi)
        
        # Should apply windowing: wdata = data * phi
        # Since data_pad is all ones, result should equal phi
        assert np.allclose(wdata, phi)

    def test_assign_wdata_boundary_wrapping(self):
        """Test _assign_wdata handles boundary wrapping correctly."""
        K = 6
        ND = 8
        Nf = 4
        i = 0  # This creates jj = 0*4 - 6//2 = -3, should wrap
        
        data_pad = np.arange(ND + K, dtype=float)  # 0, 1, ..., 13
        phi = np.ones(K)
        wdata = np.zeros(K)
        
        _assign_wdata(i, K, ND, Nf, wdata, data_pad, phi)
        
        # jj = -3, wrapped to 8-3 = 5
        # Should get data_pad[5:11] = [5, 6, 7, 8, 9, 10]
        expected = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        assert np.allclose(wdata, expected)

    def test_assign_wdata_different_time_indices(self):
        """Test _assign_wdata with different time indices."""
        K = 4
        ND = 8
        Nf = 2
        
        data_pad = np.arange(ND + K, dtype=float)
        phi = np.ones(K)
        
        for i in range(3):  # Test first few time indices
            wdata = np.zeros(K)
            _assign_wdata(i, K, ND, Nf, wdata, data_pad, phi)
            
            # Should produce valid output without errors
            assert len(wdata) == K
            assert not np.any(np.isnan(wdata))


class TestPackWave:
    """Test _pack_wave function with simple examples."""

    def test_pack_wave_basic_functionality(self):
        """Test _pack_wave basic operation without detailed value checking.

        Focus on ensuring the function runs without errors.
        """
        i = 0  # Even index
        mult = 2
        Nf = 3
        
        # Create sufficient frequency data
        wdata_trans = np.array([1.0+0j, 2.0+1j, 3.0+2j, 4.0+3j, 5.0+4j, 6.0+5j], dtype=complex)
        wave = np.zeros((4, Nf))
        
        # Should run without error
        _pack_wave(i, mult, Nf, wdata_trans, wave)
        
        # Check that some values were set (not all zeros)
        assert not np.allclose(wave, 0.0)

    def test_pack_wave_different_indices(self):
        """Test _pack_wave with different time indices."""
        mult = 2
        Nf = 2
        
        wdata_trans = np.array([1.0+0j, 2.0+1j, 3.0+2j, 4.0+3j], dtype=complex)
        wave = np.zeros((6, Nf))
        
        # Test several indices without detailed validation
        for i in range(4):
            _pack_wave(i, mult, Nf, wdata_trans, wave)
        
        # Should complete without errors
        assert wave.shape == (6, Nf)

    def test_pack_wave_output_finite(self):
        """Test that _pack_wave produces finite output."""
        i = 1
        mult = 3
        Nf = 2
        
        wdata_trans = np.array([1.0+1j, 2.0+2j, 3.0+3j, 4.0+4j, 5.0+5j, 6.0+6j], dtype=complex)
        wave = np.zeros((4, Nf))
        
        _pack_wave(i, mult, Nf, wdata_trans, wave)
        
        # All values should be finite
        assert np.all(np.isfinite(wave))
        assert wave.dtype in [np.float64, np.float32]


class TestTransformWaveletTimeHelper:
    """Test transform_wavelet_time_helper with realistic examples."""

    def test_transform_helper_realistic_parameters(self):
        """Test basic functionality with realistic parameters from working examples."""
        # Use parameters from the working wavelet_transforms.py function
        Nf = 8
        Nt = 32
        mult = min(16, Nt // 2)  # 16, limited to Nt//2 = 16
        data = np.ones(Nf * Nt)  # 256 samples
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        # Check output shape
        assert result.shape == (Nt, Nf)
        
        # Should produce finite values
        assert np.all(np.isfinite(result))

    def test_transform_helper_zero_input(self):
        """Test transform_wavelet_time_helper with zero input."""
        Nf = 4
        Nt = 16
        mult = min(8, Nt // 2)  # 8
        data = np.zeros(Nf * Nt)
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        # Zero input should produce zero output
        assert np.allclose(result, 0.0, atol=1e-15)

    def test_transform_helper_mult_limitation(self):
        """Test that mult is appropriately limited to prevent errors."""
        Nf = 4
        Nt = 8
        # Use safe mult that won't cause array size issues
        mult = min(4, Nt // 2)  # 4
        
        data = np.random.randn(Nf * Nt)
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        assert result.shape == (Nt, Nf)
        assert np.all(np.isfinite(result))

    def test_transform_helper_consistent_with_phi_vec(self):
        """Test using phi_vec output with transform_wavelet_time_helper."""
        Nf = 4
        Nt = 16
        mult = min(8, Nt // 2)  # 8
        
        # Generate phi using phi_vec
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        
        # Create test data with enough samples
        data = np.sin(2 * np.pi * np.arange(Nf * Nt) / 16)
        
        # Apply transform
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        assert result.shape == (Nt, Nf)
        assert np.all(np.isfinite(result))
        # Should capture some energy (not all zeros)
        assert np.sum(np.abs(result)) > 1e-10


class TestWaveletTimeIntegration:
    """Integration tests combining multiple functions."""

    def test_realistic_wavelet_transform_example(self):
        """Test realistic example matching the working wavelet_transforms.py usage."""
        # Use parameters similar to the working test
        srate = 128
        seglen = 4
        df = 4
        nx = 4.0
        
        # Calculate realistic Nt, Nf
        duration = seglen
        Nf = int(srate / 2 / df)  # 16
        Nt = int(duration * srate / Nf)  # 32
        
        # Use safe mult value
        mult = min(16, Nt // 2)  # 16
        
        # Generate phi using phi_vec
        phi = phi_vec(Nf, nx, mult)
        
        # Create test data with enough samples
        data = np.sin(2 * np.pi * 32 * np.arange(Nf * Nt) / srate)
        
        # Apply transform
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        assert result.shape == (Nt, Nf)
        assert np.all(np.isfinite(result))
        # Should capture some energy (not all zeros)
        assert np.sum(np.abs(result)) > 1e-10

    def test_linearity_property_realistic(self):
        """Test linearity with realistic parameters."""
        Nf = 8
        Nt = 32
        mult = min(8, Nt // 2)  # 8
        
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        
        # Create data with sufficient length
        data1 = np.random.randn(Nf * Nt)
        data2 = np.random.randn(Nf * Nt)
        a, b = 2.0, 3.0
        
        result1 = transform_wavelet_time_helper(data1, Nf, Nt, phi, mult)
        result2 = transform_wavelet_time_helper(data2, Nf, Nt, phi, mult)
        result_combined = transform_wavelet_time_helper(a * data1 + b * data2, Nf, Nt, phi, mult)
        result_linear = a * result1 + b * result2
        
        # Should satisfy linearity (within numerical precision)
        assert np.allclose(result_combined, result_linear, rtol=1e-10)
        
    def test_energy_properties_realistic(self):
        """Test basic energy properties with realistic parameters."""
        Nf = 4
        Nt = 16
        mult = min(4, Nt // 2)  # 4
        
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        
        # Non-zero input should produce non-zero output
        data = np.ones(Nf * Nt)
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        input_energy = np.sum(data ** 2)
        output_energy = np.sum(np.abs(result) ** 2)
        
        # Both should be non-zero and finite
        assert input_energy > 0
        assert output_energy > 0
        assert np.isfinite(output_energy)

    def test_consistent_output_shapes_realistic(self):
        """Test that all functions produce consistent output shapes with realistic parameters."""
        Nf = 6
        mult = 4
        
        # Generate phi
        phi = phi_vec(Nf, nx=4.0, mult=mult)
        expected_phi_length = mult * 2 * Nf
        assert len(phi) == expected_phi_length
        
        # Use with transform with sufficient data
        Nt = 24
        mult = min(mult, Nt // 2)  # Ensure safe mult
        data = np.random.randn(Nf * Nt)
        phi = phi_vec(Nf, nx=4.0, mult=mult)  # Regenerate with safe mult
        result = transform_wavelet_time_helper(data, Nf, Nt, phi, mult)
        
        assert result.shape == (Nt, Nf)