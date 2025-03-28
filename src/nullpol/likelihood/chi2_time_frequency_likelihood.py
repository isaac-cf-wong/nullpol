from __future__ import annotations

import numpy as np
import scipy.stats

from ..time_frequency_transform import transform_wavelet_freq
from .time_frequency_likelihood import TimeFrequencyLikelihood


class Chi2TimeFrequencyLikelihood(TimeFrequencyLikelihood):
    """A time-frequency likelihood class that computes the likelihood of the total null energy.

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

    @property
    def DoF(self):
        """Degree of freedom.

        Returns:
            int: Degree of freedom.
        """
        return (len(self.interferometers)-np.sum(self.polarization_basis)) * np.sum(self.time_frequency_filter)

    def _compute_residual_energy(self):
        s_est = self.estimate_wavelet_domain_signal_at_geocenter()
        d_wavelet = self.compute_cached_wavelet_domain_strain_array_at_geocenter()

        # Subtract the estimated signal from the strain data to obtain the null stream
        d_null = d_wavelet - s_est

        # Compute the null energy
        E_null = np.sum(np.abs(d_null * self.time_frequency_filter)**2)
        return E_null

    def log_likelihood(self):
        E_null = self._compute_residual_energy()
        log_likelihood = scipy.stats.chi2.logpdf(E_null, df=self.DoF)
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
        E = np.sum(np.abs(wavelet_domain_strain_array * self.time_frequency_filter)**2)
        log_likelihood = scipy.stats.chi2.logpdf(E, df=len(self.interferometers)*np.sum(self.time_frequency_filter))
        return log_likelihood
