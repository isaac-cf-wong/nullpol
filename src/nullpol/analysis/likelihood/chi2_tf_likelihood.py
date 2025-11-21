# pylint: disable=duplicate-code  # Legitimate argument parsing patterns shared across modules
from __future__ import annotations

import numpy as np
import scipy.stats

from ..tf_transforms import transform_wavelet_freq
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
    """

    def __init__(
        self,
        interferometers,
        wavelet_frequency_resolution,
        wavelet_nx,
        polarization_modes,
        *args,
        polarization_basis=None,
        time_frequency_filter=None,
        **kwargs,
    ):
        super().__init__(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
            *args,
            **kwargs,
        )

    @property
    def DoF(self):
        """Degree of freedom.

        Returns:
            int: Degree of freedom.
        """
        polarization_basis_sum = np.sum(self.antenna_pattern_processor.polarization_basis)

        if self.data_context.time_frequency_filter is None:
            raise ValueError("Time frequency filter is not available")

        time_frequency_filter_sum = np.sum(self.data_context.time_frequency_filter)
        return (len(self.interferometers) - polarization_basis_sum) * time_frequency_filter_sum

    def log_likelihood(self):
        """Compute the log likelihood using the projection approach."""
        # Compute null energy using the parameters
        null_energy = self.null_stream_calculator.compute_null_energy(self.parameters)

        return scipy.stats.chi2.logpdf(null_energy, df=self.DoF)

    def _calculate_noise_log_likelihood(self):
        """Calculate the noise log likelihood.

        Returns:
            float: noise log likelihood.
        """
        if self.data_context.whitened_frequency_domain_strain_array is None:
            raise ValueError("Whitened frequency domain strain array is not available")
        if self.data_context.time_frequency_filter is None:
            raise ValueError("Time frequency filter is not available")

        wavelet_domain_strain_array = np.array(
            [
                transform_wavelet_freq(
                    data=self.data_context.whitened_frequency_domain_strain_array[i],
                    sampling_frequency=self.data_context.sampling_frequency,
                    frequency_resolution=self.data_context.wavelet_frequency_resolution,
                    nx=self.data_context.wavelet_nx,
                )
                for i in range(len(self.interferometers))
            ]
        )
        E = np.sum(np.abs(wavelet_domain_strain_array * self.data_context.time_frequency_filter) ** 2)
        log_likelihood = scipy.stats.chi2.logpdf(
            E, df=len(self.interferometers) * np.sum(self.data_context.time_frequency_filter)
        )
        return log_likelihood
