"""Tests for the strong-lensing null-stream calculator."""

from __future__ import annotations

from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList
from bilby.gw.source import lal_binary_black_hole

from nullpol.analysis.lensing.calculator import LensingNullStreamCalculator
from nullpol.analysis.tf_transforms import get_shape_of_wavelet_transform
from nullpol.simulation.injection import create_injection


@pytest.fixture
def mock_interferometers():
    """Create two image-specific detector networks."""
    return [[Mock(), Mock(), Mock()], [Mock(), Mock()]]


def _configure_data_context(calculator, interferometers, frequencies, frequency_mask=None):
    """Configure the calculator's mocked data context."""
    n_detectors = sum(len(image_interferometers) for image_interferometers in interferometers)
    if frequency_mask is None:
        frequency_mask = np.ones(len(frequencies), dtype=bool)
    context_type = type(calculator.data_context)
    context_type.interferometers_1 = PropertyMock(return_value=interferometers[0])
    context_type.interferometers_2 = PropertyMock(return_value=interferometers[1])
    context_type.frequency_array = PropertyMock(return_value=frequencies)
    context_type.masked_frequency_array = PropertyMock(return_value=frequencies[frequency_mask])
    context_type.power_spectral_density_array = PropertyMock(return_value=np.ones((n_detectors, len(frequencies))))
    context_type.frequency_mask = PropertyMock(return_value=frequency_mask)


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


def _make_lensed_source_model(mu_rel, delta_n):
    """Create a waveform model with the image-two amplitude and Morse phase."""

    def lensed_lal_binary_black_hole(
        frequency_array,
        mass_1,
        mass_2,
        luminosity_distance,
        a_1,
        tilt_1,
        phi_12,
        a_2,
        tilt_2,
        phi_jl,
        theta_jn,
        phase,
        **kwargs,
    ):
        polarizations = lal_binary_black_hole(
            frequency_array,
            mass_1,
            mass_2,
            luminosity_distance,
            a_1,
            tilt_1,
            phi_12,
            a_2,
            tilt_2,
            phi_jl,
            theta_jn,
            phase,
            **kwargs,
        )
        lensing_factor = mu_rel * np.exp(-1j * np.pi * delta_n)
        return {polarization: lensing_factor * waveform for polarization, waveform in polarizations.items()}

    return lensed_lal_binary_black_hole


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
    def test_uses_each_image_time_and_correct_lensing_phase(
        self, mock_data_context_class, mock_antenna_class, mock_interferometers
    ):
        """Evaluate image two at its arrival time and apply its lensing factor."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )
        frequencies = np.array([20.0, 30.0, 40.0])
        _configure_data_context(calculator, mock_interferometers, frequencies)

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

        lensing_factor = parameters["mu_rel"] * np.exp(-1j * np.pi * parameters["delta_n"])
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
    def test_delay_is_not_applied_twice_after_data_alignment(
        self, mock_data_context_class, mock_antenna_class, mock_interferometers
    ):
        """Leave the delay phase to the data context's common-frame alignment."""
        calculator = LensingNullStreamCalculator(
            interferometers=mock_interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )
        frequencies = np.array([20.0, 30.0])
        delta_t = 10.0
        _configure_data_context(calculator, mock_interferometers, frequencies)

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
        lensing_factor = np.exp(-1j * np.pi * parameters["delta_n"])
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


@pytest.mark.integration
def test_two_image_injection_is_null_in_a_common_time_frame():
    """Verify non-aligned image segments use one TF-filter time reference."""
    parameters = {
        "mass_1": 36.0,
        "mass_2": 29.0,
        "a_1": 0.0,
        "a_2": 0.0,
        "tilt_1": 0.0,
        "tilt_2": 0.0,
        "phi_12": 0.0,
        "phi_jl": 0.0,
        "luminosity_distance": 500.0,
        "theta_jn": 0.0,
        "psi": 2.659,
        "phase": 1.3,
        "geocent_time": 1126259642.413,
        "ra": 1.375,
        "dec": -1.2108,
        "mu_rel": 1.3,
        "delta_t": 0.5,
        "delta_n": 0.5,
    }
    duration = 4
    sampling_frequency = 1024
    wavelet_frequency_resolution = 16
    wavelet_nx = 4
    image_1_start_time = parameters["geocent_time"] - duration / 2
    image_2_start_time = image_1_start_time + 0.25
    image_1 = InterferometerList(["H1", "L1"])
    image_2 = InterferometerList(["H1", "L1"])
    for interferometers in (image_1, image_2):
        for interferometer in interferometers:
            interferometer.minimum_frequency = 20

    create_injection(
        interferometers=image_1,
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=image_1_start_time,
        parameters=parameters,
        noise_type="zero_noise",
    )
    create_injection(
        interferometers=image_2,
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=image_2_start_time,
        parameters={**parameters, "geocent_time": parameters["geocent_time"] + parameters["delta_t"]},
        noise_type="zero_noise",
        frequency_domain_source_model=_make_lensed_source_model(parameters["mu_rel"], parameters["delta_n"]),
    )
    tf_nt, tf_nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)
    time_frequency_filter = np.zeros((tf_nt, tf_nf))
    time_frequency_filter[tf_nt // 2 - 10 : tf_nt // 2 + 10, 4 : tf_nf - 8] = 1
    calculator = LensingNullStreamCalculator(
        interferometers=[image_1, image_2],
        wavelet_frequency_resolution=wavelet_frequency_resolution,
        wavelet_nx=wavelet_nx,
        polarization_modes="pc",
        time_frequency_filter=time_frequency_filter,
    )

    null_energy = calculator.compute_null_energy(parameters)
    wrong_morse_phase_energy = calculator.compute_null_energy({**parameters, "delta_n": parameters["delta_n"] + 1})
    wrong_delay_energy = calculator.compute_null_energy({**parameters, "delta_t": parameters["delta_t"] + 0.1})

    assert null_energy < 1e-12
    assert wrong_morse_phase_energy > 1
    assert wrong_delay_energy > 1
