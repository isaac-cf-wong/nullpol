# pylint: disable=duplicate-code  # Calculator classes share initialization and method patterns by design
"""Null stream calculator for strongly lensed signals."""

from __future__ import annotations

import numpy as np

from ..antenna_patterns import AntennaPatternProcessor
from ..null_stream.calculator import NullStreamCalculator
from .data_context import LensingTimeFrequencyDataContext


# pylint: disable=too-few-public-methods
class LensingNullStreamCalculator(NullStreamCalculator):
    """Null stream calculator with strong lensing modifications.

    Extends NullStreamCalculator to apply the relative waveform amplitude and
    Morse phase to image-two antenna patterns. The data context accounts for
    the relative time delay by aligning both images to one FFT time reference.

    This is a library-only API for now; the command-line workflows do not
    construct a two-image likelihood. Each parameter mapping must contain the
    usual sky parameters plus ``mu_rel`` (positive relative waveform amplitude),
    ``delta_t`` (seconds), and ``delta_n`` (Morse phase).

    Args:
        interferometers (list): List of two sublists of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list, optional): List of polarization basis.
        time_frequency_filter (np.ndarray, optional): A fixed time-frequency
            filter in the common image-one reference frame. It should cover
            the signal support and expected timing uncertainty without crossing
            segment boundaries.
    """

    # pylint: disable=super-init-not-called
    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        polarization_basis=None,
        time_frequency_filter=None,
    ):
        """Initialize calculator with lensing-specific data context.

        Note: Does not call super().__init__() because the parent class expects
        a flat list of interferometers, but lensing requires two separate sets
        for the two lensed images. Instead, manually creates LensingTimeFrequencyDataContext.
        """
        if polarization_basis is None:
            polarization_basis = polarization_modes

        self.data_context = LensingTimeFrequencyDataContext(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            time_frequency_filter=time_frequency_filter,
        )

        self.antenna_pattern_processor = AntennaPatternProcessor(
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            interferometers=interferometers[0] + interferometers[1],
        )

    def _compute_calibrated_whitened_antenna_pattern_matrix(self, parameters):
        """Compute antenna pattern matrix with lensing factor applied.

        Evaluates each image's detector response at its own geocentric arrival
        time, then applies the image-two lensing factor
        ``L = mu_rel * exp(-1j * pi * delta_n)``. The data context has already
        removed the frequency-dependent phase caused by different FFT origins.
        Here ``mu_rel`` is the relative waveform-amplitude factor and
        ``delta_n`` is the Morse phase difference.

        Args:
            parameters (dict): Dictionary containing standard GW parameters plus
                'mu_rel', 'delta_t', and 'delta_n'.

        Returns:
            np.ndarray: Calibrated whitened antenna pattern matrix with lensing factor
                applied to second image.
        """
        n_detectors_image_1 = len(self.data_context.interferometers_1)
        calibrated_whitened_antenna_pattern_matrix_image_1 = (
            self.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix(
                self.data_context.interferometers_1,
                self.data_context.power_spectral_density_array[:n_detectors_image_1],
                self.data_context.masked_frequency_array,
                self.data_context.frequency_mask,
                parameters,
            )
        )

        image_2_parameters = {**parameters, "geocent_time": parameters["geocent_time"] + parameters["delta_t"]}
        calibrated_whitened_antenna_pattern_matrix_image_2 = (
            self.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix(
                self.data_context.interferometers_2,
                self.data_context.power_spectral_density_array[n_detectors_image_1:],
                self.data_context.masked_frequency_array,
                self.data_context.frequency_mask,
                image_2_parameters,
            )
        )

        lensing_factor = parameters["mu_rel"] * np.exp(-1j * np.pi * parameters["delta_n"])
        calibrated_whitened_antenna_pattern_matrix_image_2 *= lensing_factor

        return np.concatenate(
            [calibrated_whitened_antenna_pattern_matrix_image_1, calibrated_whitened_antenna_pattern_matrix_image_2],
            axis=1,
        )
