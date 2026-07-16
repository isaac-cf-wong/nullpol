"""Tests for the strong-lensing null-stream calculator."""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullpol.analysis.lensing.calculator import LensingNullStreamCalculator


@pytest.fixture
def mock_interferometers():
    """Create two image-specific detector networks."""
    return [[Mock(), Mock(), Mock()], [Mock(), Mock()]]


def _configure_data_context(calculator, interferometers, frequencies, start_time_offset):
    """Configure the calculator's mocked data context."""
    n_detectors = sum(len(image_interferometers) for image_interferometers in interferometers)
    context_type = type(calculator.data_context)
    context_type.interferometers_1 = PropertyMock(return_value=interferometers[0])
    context_type.interferometers_2 = PropertyMock(return_value=interferometers[1])
    context_type.inter_image_start_time_offset = PropertyMock(return_value=start_time_offset)
    context_type.frequency_array = PropertyMock(return_value=frequencies)
    context_type.masked_frequency_array = PropertyMock(return_value=frequencies)
    context_type.power_spectral_density_array = PropertyMock(return_value=np.ones((n_detectors, len(frequencies))))
    context_type.frequency_mask = PropertyMock(return_value=np.ones(len(frequencies), dtype=bool))


def _set_image_pattern_side_effect(calculator, interferometers, image_1_pattern, image_2_pattern):
    """Make the antenna-pattern processor return the corresponding image matrix."""

    def compute_pattern(image_interferometers, *_args):
        if image_interferometers is interferometers[0]:
            return image_1_pattern.copy()
        if image_interferometers is interferometers[1]:
            return image_2_pattern.copy()
        pytest.fail("Unexpected interferometer network")

    calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix = Mock(
        side_effect=compute_pattern
    )


def _make_interferometer_network(detectors, start_time):
    """Create zero-noise detector data for one image epoch."""
    interferometers = InterferometerList(detectors)
    for interferometer in interferometers:
        interferometer.minimum_frequency = 20
    interferometers.set_strain_data_from_zero_noise(
        sampling_frequency=256,
        duration=2,
        start_time=start_time,
    )
    return interferometers


class TestLensingNullStreamCalculator:
    """Test lensing-specific antenna-pattern construction."""

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_initialization(self, mock_data_context_class, mock_antenna_class, mock_interferometers):
        """Construct the lensing-specific data context and antenna processor."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
            polarization_basis="pc",
        )

        mock_data_context_class.assert_called_once_with(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            time_frequency_filter=None,
        )
        mock_antenna_class.assert_called_once_with(
            polarization_modes="pc",
            polarization_basis="pc",
            interferometers=mock_interferometers[0] + mock_interferometers[1],
        )
        assert calculator.data_context is not None
        assert calculator.antenna_pattern_processor is not None

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_uses_each_image_time_and_correct_residual_phase(
        self, mock_data_context_class, mock_antenna_class, mock_interferometers
    ):
        """Evaluate image two at its arrival time and phase relative to FFT origins."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )
        frequencies = np.array([20.0, 30.0, 40.0])
        start_time_offset = 0.08
        _configure_data_context(calculator, mock_interferometers, frequencies, start_time_offset)

        image_1_pattern = np.full((len(frequencies), 3, 2), 2 + 1j)
        image_2_pattern = np.full((len(frequencies), 2, 2), 3 - 2j)
        _set_image_pattern_side_effect(calculator, mock_interferometers, image_1_pattern, image_2_pattern)

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1_234_567_890.0,
            "mu_rel": 2.5,
            "delta_t": 0.1,
            "delta_n": 0.5,
        }
        original_parameters = parameters.copy()

        result = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        residual_delay = parameters["delta_t"] - start_time_offset
        lensing_factor = parameters["mu_rel"] * np.exp(
            -1j * np.pi * (2 * residual_delay * frequencies[:, None, None] + parameters["delta_n"])
        )
        expected = np.concatenate([image_1_pattern, image_2_pattern * lensing_factor], axis=1)
        np.testing.assert_allclose(result, expected)
        assert parameters == original_parameters

        calls = calculator.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix.call_args_list
        assert len(calls) == 2
        assert calls[0].args[0] is mock_interferometers[0]
        assert calls[0].args[-1] == parameters
        assert calls[1].args[0] is mock_interferometers[1]
        assert calls[1].args[-1]["geocent_time"] == parameters["geocent_time"] + parameters["delta_t"]

    @patch("nullpol.analysis.lensing.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.lensing.calculator.LensingTimeFrequencyDataContext")
    def test_matching_segment_offset_removes_delay_phase(
        self, mock_data_context_class, mock_antenna_class, mock_interferometers
    ):
        """Do not add a delay phase when the image segments are offset by that delay."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )
        frequencies = np.array([20.0, 30.0])
        delta_t = 10.0
        _configure_data_context(calculator, mock_interferometers, frequencies, start_time_offset=delta_t)

        image_1_pattern = np.ones((len(frequencies), 3, 2), dtype=complex)
        image_2_pattern = np.ones((len(frequencies), 2, 2), dtype=complex)
        _set_image_pattern_side_effect(calculator, mock_interferometers, image_1_pattern, image_2_pattern)

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1_234_567_890.0,
            "mu_rel": 1.5,
            "delta_t": delta_t,
            "delta_n": 0.0,
        }

        result = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)

        np.testing.assert_allclose(result[:, :3, :], image_1_pattern)
        np.testing.assert_allclose(result[:, 3:, :], image_2_pattern * parameters["mu_rel"])

    def test_image_two_uses_its_epoch_antenna_response(self):
        """Use real detector responses at the second image's geocentric time."""
        image_1_interferometers = _make_interferometer_network(["H1", "L1", "V1"], start_time=1_234_567_000)
        image_2_interferometers = _make_interferometer_network(["H1", "L1"], start_time=1_234_567_100)
        calculator = LensingNullStreamCalculator(
            interferometers=[image_1_interferometers, image_2_interferometers],
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )
        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "psi": 0.3,
            "geocent_time": 1_234_567_890.0,
            "mu_rel": 1.0,
            "delta_t": 86_400.0,
            "delta_n": 0.0,
        }

        result = calculator._compute_calibrated_whitened_antenna_pattern_matrix(parameters)
        n_detectors_image_1 = len(image_1_interferometers)
        processor = calculator.antenna_pattern_processor
        image_1_pattern = processor.compute_calibrated_whitened_antenna_pattern_matrix(
            calculator.data_context.interferometers_1,
            calculator.data_context.power_spectral_density_array[:n_detectors_image_1],
            calculator.data_context.masked_frequency_array,
            calculator.data_context.frequency_mask,
            parameters,
        )
        image_2_parameters = {**parameters, "geocent_time": parameters["geocent_time"] + parameters["delta_t"]}
        image_2_pattern = processor.compute_calibrated_whitened_antenna_pattern_matrix(
            calculator.data_context.interferometers_2,
            calculator.data_context.power_spectral_density_array[n_detectors_image_1:],
            calculator.data_context.masked_frequency_array,
            calculator.data_context.frequency_mask,
            image_2_parameters,
        )
        residual_delay = parameters["delta_t"] - calculator.data_context.inter_image_start_time_offset
        lensing_factor = np.exp(
            -1j * np.pi * (2 * residual_delay * calculator.data_context.frequency_array[:, None, None])
        )
        expected = np.concatenate([image_1_pattern, image_2_pattern * lensing_factor], axis=1)
        np.testing.assert_allclose(result, expected)

        unshifted_image_2_pattern = processor.compute_calibrated_whitened_antenna_pattern_matrix(
            calculator.data_context.interferometers_2,
            calculator.data_context.power_spectral_density_array[n_detectors_image_1:],
            calculator.data_context.masked_frequency_array,
            calculator.data_context.frequency_mask,
            parameters,
        )
        assert not np.allclose(image_2_pattern, unshifted_image_2_pattern)
