"""Null stream calculator for strongly lensed signals."""

from __future__ import annotations

import numpy as np

from ..null_stream.calculator import NullStreamCalculator
from ..lensing.data_context import LensingTimeFrequencyDataContext
from ..antenna_patterns import AntennaPatternProcessor


class LensingNullStreamCalculator(NullStreamCalculator):
    """Null stream calculator with strong lensing modifications.

    Extends NullStreamCalculator to apply lensing factor (amplification, time delay,
    Morse phase) to antenna patterns.

    Args:
        interferometers (list): List of two sublists of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list, optional): List of polarization basis.
        time_frequency_filter (np.ndarray, optional): The time-frequency filter.
    """

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        polarization_basis=None,
        time_frequency_filter=None,
    ):

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

        Applies lensing factor L(f) = A × exp(i × π × (2 × Δt × f - n_morse))
        where A is amplification, Δt is time delay, and n_morse is Morse phase.

        Args:
            parameters (dict): Dictionary containing standard GW parameters plus
                'amplification', 'time_delay', and 'n_morse'.

        Returns:
            np.ndarray: Calibrated whitened antenna pattern matrix with lensing factor.
        """
        calibrated_whitened_antenna_pattern_matrix = (
            self.antenna_pattern_processor.compute_calibrated_whitened_antenna_pattern_matrix(
                self.data_context.interferometers[0] + self.data_context.interferometers[1],
                self.data_context.power_spectral_density_array,
                self.data_context.masked_frequency_array,
                self.data_context.frequency_mask,
                parameters,
            )
        )
        lensing_factor = parameters["amplification"] * np.exp(
            1j
            * np.pi
            * (
                2 * parameters["time_delay"] * self.data_context.masked_frequency_array[:, None, None]
                - parameters["n_morse"]
            )
        )
        return calibrated_whitened_antenna_pattern_matrix * lensing_factor
