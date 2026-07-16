# pylint: disable=duplicate-code  # Legitimate argument parsing patterns shared across modules
"""Chi-squared time-frequency likelihood for strongly lensed signals."""

from __future__ import annotations

from bilby.core.likelihood import Likelihood

from ..lensing.calculator import LensingNullStreamCalculator
from ..likelihood.chi2_tf_likelihood import Chi2TimeFrequencyLikelihood

NUMBER_OF_LENSED_IMAGES = 2


class LensingChi2TimeFrequencyLikelihood(Chi2TimeFrequencyLikelihood):
    """Chi-squared likelihood for strongly lensed signals using null stream method.

    Requires two sets of interferometers for two lensed images. Applies lensing
    modifications to antenna patterns when computing null stream energy.

    This is a library-only API: the command-line workflows do not construct a
    two-image likelihood. Each parameter mapping must provide the usual sky
    parameters plus ``mu_rel`` (positive relative waveform amplitude),
    ``delta_t`` (seconds), and ``delta_n`` (Morse phase).

    Args:
        interferometers (list): List of two sublists of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list, optional): List of polarization basis.
        time_frequency_filter (np.ndarray, optional): The time-frequency filter.
    """

    # pylint: disable=super-init-not-called
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
        """Initialize likelihood with lensing-specific calculator.

        Note: Does not call super().__init__() because it requires LensingNullStreamCalculator
        instead of the standard NullStreamCalculator. The parent initialization is bypassed
        to use the lensing-aware calculator that handles two detector sets.
        """
        Likelihood.__init__(self, {})  # pylint: disable=non-parent-init-called
        if not (
            isinstance(interferometers, list)
            and len(interferometers) == NUMBER_OF_LENSED_IMAGES
            and all(
                isinstance(image_interferometers, list) and image_interferometers
                for image_interferometers in interferometers
            )
        ):
            raise ValueError("interferometers must be a list of two lists of non-empty interferometers")

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
