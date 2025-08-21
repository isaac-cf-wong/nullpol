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
        polarization_basis=None,
        time_frequency_filter=None,
        *args,
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

    def _compute_null_energy(self):
        """Compute null energy using the projection approach.

        This implements the mathematically clean projection approach:
        1. Get whitened strain data at geocenter in frequency domain
        2. Compute whitened antenna patterns in frequency domain
        3. Apply null projection in frequency domain (where dimensions match)
        4. Transform projected data to wavelet domain
        5. Apply time-frequency filter and sum energy

        The key insight is that projection must be done in frequency domain where
        antenna patterns and data have matching dimensions.

        Returns:
            float: Null energy computed via projection.
        """
        if self.parameters is None:
            raise ValueError("Parameters must be set before computing null energy")

        # Step 1: Get whitened strain data at geocenter in FREQUENCY domain
        # Shape: (n_detectors, n_frequencies)
        whitened_freq_strain = self.data_context.compute_whitened_strain_at_geocenter(self.parameters)

        # Step 2: Compute whitened antenna patterns in frequency domain
        # Shape: (n_frequencies, n_detectors, n_modes)
        whitened_antenna_patterns = self.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix(
            self.data_context.interferometers,
            self.data_context.power_spectral_density_array,
            self.data_context.frequency_mask,
            self.parameters,
        )

        # Step 3: Compute projectors in frequency domain
        gw_projector = self.null_stream_calculator.compute_gw_projector(
            whitened_antenna_patterns, self.data_context.frequency_mask
        )
        null_projector = self.null_stream_calculator.compute_null_projector(gw_projector)

        # Step 4: Apply null projection in frequency domain (frequency-by-frequency)
        n_detectors, n_frequencies = whitened_freq_strain.shape
        null_projected_freq_strain = np.zeros_like(whitened_freq_strain, dtype=complex)

        for freq_idx in range(n_frequencies):
            if self.data_context.frequency_mask[freq_idx]:
                # Get detector data vector at this frequency
                d = whitened_freq_strain[:, freq_idx]  # Shape: (n_detectors,)
                P_null = null_projector[freq_idx]  # Shape: (n_detectors, n_detectors)

                # Apply null projection: d_null = P_null @ d
                null_projected_freq_strain[:, freq_idx] = P_null @ d

        # Step 5: Transform null-projected data to wavelet domain
        null_projected_wavelet_strain = self.data_context.transform_to_wavelet_domain(null_projected_freq_strain)

        # Step 6: Apply time-frequency filter and compute energy
        filtered_null_strain = null_projected_wavelet_strain * self.data_context.time_frequency_filter
        null_energy = np.sum(np.abs(filtered_null_strain) ** 2)

        return null_energy

    def log_likelihood(self):
        """Compute the log likelihood using the projection approach."""
        E_null = self._compute_null_energy()
        log_likelihood = scipy.stats.chi2.logpdf(E_null, df=self.DoF)
        return log_likelihood

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
