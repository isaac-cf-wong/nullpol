from __future__ import annotations

import numpy as np

from ..time_frequency_transform import transform_wavelet_freq
from .time_frequency_likelihood import TimeFrequencyLikelihood


class GaussianTimeFrequencyLikelihood(TimeFrequencyLikelihood):
    """A time-frequency likelihood class that calculates the Gaussian likelihood.

    Args:
        interferometers (list): List of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list): List of polarization basis.
        time_frequency_filter (str): The time-frequency filter.
        priors (dict, optional): If given, used in the calibration marginalization.
    """
    def __init__(self,
                 interferometers,
                 wavelet_frequency_resolution,
                 wavelet_nx,
                 polarization_modes,
                 polarization_basis=None,
                 time_frequency_filter=None,
                 priors=None,
                 *args, **kwargs):
        super().__init__(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
            priors=priors,
            *args, **kwargs)
        self._log_normalization_constant = -np.log(2. * np.pi) * 0.5 * (len(self.interferometers) - np.sum(self.polarization_basis)) * np.sum(self.time_frequency_filter)

    def _compute_residual(self):
        s_est = self.estimate_wavelet_domain_signal_at_geocenter()
        d_wavelet = self.compute_cached_wavelet_domain_strain_array_at_geocenter()

        # Subtract the estimated signal from the strain data to obtain the null stream
        d_null = d_wavelet - s_est
        return d_null

    def log_likelihood(self):
        residual = self._compute_residual()
        E_null = np.sum(np.abs(residual * self.time_frequency_filter)**2)
        log_likelihood = -0.5 * E_null + self._log_normalization_constant
        return log_likelihood

    def _calculate_noise_log_likelihood(self):
        """Calculate noise log likelihood.

        Returns:
            float: noise log likelihood.
        """
        wavelet_domain_strain_array = np.array([transform_wavelet_freq(
            data=self.whitened_frequency_domain_strain_array[i],
            sampling_frequency=self.sampling_frequency,
            frequency_resolution=self.wavelet_frequency_resolution,
            nx=self.wavelet_nx) for i in range(len(self.interferometers))]
        )
        E = np.sum(np.abs(wavelet_domain_strain_array *
                          self.time_frequency_filter)**2)
        log_normalization = -np.log(2. * np.pi) * 0.5 * \
            len(self.interferometers) * np.sum(self.time_frequency_filter)
        log_likelihood = -0.5 * E + log_normalization
        return log_likelihood
