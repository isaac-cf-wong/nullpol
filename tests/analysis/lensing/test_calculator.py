"""Test module for lensing null stream calculator functionality.

This module tests the LensingNullStreamCalculator class and its
lensing factor application to antenna patterns.
"""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest

from nullpol.analysis.lensing.calculator import LensingNullStreamCalculator


@pytest.fixture
def mock_interferometers():
    """Create mock interferometer sets for testing."""
    mock_ifos_1 = [Mock(), Mock(), Mock()]  # H1, L1, V1
    mock_ifos_2 = [Mock(), Mock()]  # H1, L1
    return [mock_ifos_1, mock_ifos_2]


class TestLensingNullStreamCalculator:
    """Test class for LensingNullStreamCalculator."""

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_initialization(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test calculator initialization with two detector sets."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=None,
        )

        assert calculator is not None
        assert calculator.data_context is not None
        assert calculator.antenna_pattern_processor is not None

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_lensing_factor_computation(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test that lensing factor is correctly computed and applied."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
            polarization_basis="pc",
        )

        # Setup mocks
        n_freqs = 100
        n_detectors = 3
        n_modes = 2

        mock_masked_freq = np.linspace(20, 1000, n_freqs)
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        # Mock interferometers to return the combined list
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(
            return_value=np.ones((n_detectors, n_freqs))
        )
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        # Mock base antenna pattern
        base_pattern = np.ones((n_freqs, n_detectors, n_modes), dtype=complex)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern
        )

        # Test parameters
        amplification = 2.0
        time_delay = 0.1  # 100 ms
        n_morse = 0  # Type I image

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": amplification,
            "time_delay": time_delay,
            "n_morse": n_morse,
        }

        # Compute lensed antenna pattern
        lensed_pattern = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        # Verify shape is preserved
        assert lensed_pattern.shape == base_pattern.shape

        # Verify lensing factor is applied
        # Compute expected lensing factor
        expected_factor = amplification * np.exp(
            1j * np.pi * (2 * time_delay * mock_masked_freq[:, None, None] - n_morse)
        )
        expected_pattern = base_pattern * expected_factor

        np.testing.assert_allclose(lensed_pattern, expected_pattern, rtol=1e-10)

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_lensing_factor_zero_time_delay(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test lensing factor with zero time delay."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        n_freqs = 50
        mock_masked_freq = np.linspace(20, 1000, n_freqs)
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=np.ones((3, n_freqs)))
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        base_pattern = np.ones((n_freqs, 3, 2), dtype=complex)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern
        )

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": 1.5,
            "time_delay": 0.0,  # Zero delay
            "n_morse": 0,
        }

        lensed_pattern = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        # With zero time delay, only amplification and morse phase apply
        # Factor = 1.5 * exp(-i*pi*0) = 1.5
        expected_pattern = base_pattern * 1.5

        np.testing.assert_allclose(lensed_pattern, expected_pattern, rtol=1e-10)

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_morse_phase_effects(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test different Morse phase values."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        n_freqs = 50
        mock_masked_freq = np.linspace(20, 1000, n_freqs)
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=np.ones((3, n_freqs)))
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        base_pattern = np.ones((n_freqs, 3, 2), dtype=complex)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern.copy()
        )

        base_params = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": 1.0,
            "time_delay": 0.0,
        }

        # Test Type I (n_morse=0)
        params_type1 = {**base_params, "n_morse": 0}
        pattern_type1 = calculator._compute_calibrated_whitened_antenna_pattern_matrix(params_type1)

        # Test Type II (n_morse=0.5)
        params_type2 = {**base_params, "n_morse": 0.5}
        pattern_type2 = calculator._compute_calibrated_whitened_antenna_pattern_matrix(params_type2)

        # Test Type III (n_morse=1)
        params_type3 = {**base_params, "n_morse": 1}
        pattern_type3 = calculator._compute_calibrated_whitened_antenna_pattern_matrix(params_type3)

        # Expected factors: exp(-i*pi*0) = 1, exp(-i*pi*0.5) = -i, exp(-i*pi*1) = -1
        expected_type1 = base_pattern * 1.0
        expected_type2 = base_pattern * np.exp(-1j * np.pi * 0.5)
        expected_type3 = base_pattern * np.exp(-1j * np.pi * 1.0)

        np.testing.assert_allclose(pattern_type1, expected_type1, rtol=1e-10)
        np.testing.assert_allclose(pattern_type2, expected_type2, rtol=1e-10)
        np.testing.assert_allclose(pattern_type3, expected_type3, rtol=1e-10)

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_amplification_factor(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test amplification factor scaling."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        n_freqs = 50
        mock_masked_freq = np.linspace(20, 1000, n_freqs)
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=np.ones((3, n_freqs)))
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        base_pattern = np.random.randn(n_freqs, 3, 2) + 1j * np.random.randn(n_freqs, 3, 2)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern.copy()
        )

        # Test magnification (A > 1)
        params_magnified = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": 3.0,
            "time_delay": 0.0,
            "n_morse": 0,
        }
        pattern_magnified = calculator._compute_calibrated_whitened_antenna_pattern_matrix(params_magnified)
        assert np.mean(np.abs(pattern_magnified)) > np.mean(np.abs(base_pattern))

        # Test demagnification (A < 1)
        params_demagnified = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": 0.3,
            "time_delay": 0.0,
            "n_morse": 0,
        }
        pattern_demagnified = calculator._compute_calibrated_whitened_antenna_pattern_matrix(params_demagnified)
        assert np.mean(np.abs(pattern_demagnified)) < np.mean(np.abs(base_pattern))

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_frequency_dependent_phase(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test that phase varies with frequency as expected."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        n_freqs = 100
        mock_masked_freq = np.linspace(20, 1000, n_freqs)
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=np.ones((3, n_freqs)))
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        # Use unit amplitude pattern to isolate phase effects
        base_pattern = np.ones((n_freqs, 3, 2), dtype=complex)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern
        )

        time_delay = 0.1
        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": 1.0,
            "time_delay": time_delay,
            "n_morse": 0,
        }

        lensed_pattern = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        # Extract phases at two different frequencies
        phase_low = np.angle(lensed_pattern[10, 0, 0])
        phase_high = np.angle(lensed_pattern[90, 0, 0])

        # Phases should differ due to frequency-dependent term
        assert not np.isclose(phase_low, phase_high)

        # Verify phase difference matches expected frequency dependence
        freq_low = mock_masked_freq[10]
        freq_high = mock_masked_freq[90]
        expected_phase_diff = 2 * np.pi * time_delay * (freq_high - freq_low)
        actual_phase_diff = phase_high - phase_low

        # Account for 2π wrapping
        expected_phase_diff = np.angle(np.exp(1j * expected_phase_diff))
        actual_phase_diff = np.angle(np.exp(1j * actual_phase_diff))

        np.testing.assert_allclose(actual_phase_diff, expected_phase_diff, atol=1e-10)

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_combined_lensing_effects(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test combined amplification, time delay, and Morse phase."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        n_freqs = 50
        mock_masked_freq = np.linspace(100, 500, n_freqs)
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=np.ones((3, n_freqs)))
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        base_pattern = np.ones((n_freqs, 3, 2), dtype=complex) * (1 + 1j) / np.sqrt(2)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern
        )

        amplification = 2.5
        time_delay = 0.05
        n_morse = 0.5

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": amplification,
            "time_delay": time_delay,
            "n_morse": n_morse,
        }

        lensed_pattern = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        # Compute expected result
        lensing_factor = amplification * np.exp(
            1j * np.pi * (2 * time_delay * mock_masked_freq[:, None, None] - n_morse)
        )
        expected_pattern = base_pattern * lensing_factor

        np.testing.assert_allclose(lensed_pattern, expected_pattern, rtol=1e-10)

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_data_type_preservation(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Test that data types are preserved through lensing factor application."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        n_freqs = 50
        mock_masked_freq = np.linspace(20, 1000, n_freqs)
        combined_ifos = mock_interferometers[0] + mock_interferometers[1]
        type(calculator.data_context).masked_frequency_array = PropertyMock(return_value=mock_masked_freq)
        type(calculator.data_context).interferometers = PropertyMock(return_value=combined_ifos)
        type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=np.ones((3, n_freqs)))
        type(calculator.data_context).frequency_mask = PropertyMock(return_value=np.ones(n_freqs, dtype=bool))

        # Test with complex128
        base_pattern_128 = np.ones((n_freqs, 3, 2), dtype=np.complex128)
        calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
            return_value=base_pattern_128
        )

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1234567890.0,
            "amplification": 2.0,
            "time_delay": 0.1,
            "n_morse": 0,
        }

        lensed_pattern = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        # Should maintain complex type
        assert np.iscomplexobj(lensed_pattern)
        assert lensed_pattern.shape == base_pattern_128.shape
