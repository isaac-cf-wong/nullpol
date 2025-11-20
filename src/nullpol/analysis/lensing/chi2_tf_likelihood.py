# pylint: disable=duplicate-code  # Legitimate argument parsing patterns shared across modules
"""Chi-squared time-frequency likelihood for strongly lensed signals."""
from __future__ import annotations

from bilby.core.likelihood import Likelihood

from ..likelihood.chi2_tf_likelihood import Chi2TimeFrequencyLikelihood
from ..lensing.calculator import LensingNullStreamCalculator


class LensingChi2TimeFrequencyLikelihood(Chi2TimeFrequencyLikelihood):
    """Chi-squared likelihood for strongly lensed signals using null stream method.

    Requires two sets of interferometers for two lensed images. Applies lensing
    modifications to antenna patterns when computing null stream energy.

    Args:
        interferometers (list): List of two sublists of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list, optional): List of polarization basis.
        time_frequency_filter (np.ndarray, optional): The time-frequency filter.
    """

    # pylint: disable=super-init-not-called  # Intentionally uses LensingNullStreamCalculator
    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        *args,  # pylint: disable=unused-argument
        polarization_basis=None,
        time_frequency_filter=None,
        **kwargs,  # pylint: disable=unused-argument
    ):
        Likelihood.__init__(self, dict())
        if not (
            isinstance(interferometers, list)
            and len(interferometers) == 2
            and all(isinstance(i, list) for i in interferometers)
        ):
            raise ValueError("interferometers must be a list of two lists of interferometers")

        # Initialize LensingNullStreamCalculator
        self.null_stream_calculator = LensingNullStreamCalculator(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
        )

        # Initialize the normalization constant
        self._noise_log_likelihood_value = None
