"""Null stream calculator for energy computations."""

from __future__ import annotations

import numpy as np

from ..tf_transforms import transform_wavelet_freq
from .projections import compute_gw_projector, compute_null_projector, compute_null_stream


class NullStreamCalculator:
    """Modern null stream calculation using direct projection approach.

    This is a pure computational class with no component dependencies, using
    dependency injection for all data inputs.
    """

    def __init__(self):
        """Initialize the null stream calculator.

        No dependencies - this is a pure computational class that receives
        all required data via method parameters (dependency injection pattern).
        """
        pass

    # =========================================================================
    # PRIMARY INTERFACE METHODS
    # =========================================================================

    def compute_null_energy(
        self,
        whitened_antenna_pattern_matrix,
        whitened_frequency_strain_data,
        frequency_mask,
        time_frequency_filter,
        sampling_frequency,
        wavelet_frequency_resolution,
        wavelet_nx,
    ):
        """Compute the total null energy from whitened data and projectors.

        This method projects the whitened frequency-domain strain data onto the null space
        (orthogonal to the GW signal subspace), transforms to the time-frequency domain,
        applies a filter, and sums the squared magnitude to obtain the total null energy.

        Args:
            whitened_antenna_pattern_matrix (np.ndarray): Whitened antenna pattern matrix (n_freq, n_det, n_modes).
            whitened_frequency_strain_data (np.ndarray): Whitened frequency-domain strain data (n_det, n_freq).
            frequency_mask (np.ndarray): Boolean mask for frequency bins (n_freq,).
            time_frequency_filter (np.ndarray): Time-frequency filter to apply (shape matches output of transform).
            sampling_frequency (float): Sampling frequency in Hz.
            wavelet_frequency_resolution (float): Frequency resolution for wavelet transform.
            wavelet_nx (int): Number of points for wavelet transform.

        Returns:
            float: The total null energy after projection and filtering.
        """
        # Step 1: Compute the GW signal projector for each frequency bin (masked)
        gw_projector = self._compute_gw_projector(whitened_antenna_pattern_matrix, frequency_mask)

        # Step 2: Compute the null projector (orthogonal complement to GW projector)
        null_projector = self._compute_null_projector(gw_projector)

        # Step 3: Project the whitened frequency-domain strain onto the null space
        null_stream_freq = self._compute_null_stream(whitened_frequency_strain_data, null_projector, frequency_mask)

        # Step 4: Transform the null stream to the time-frequency domain
        null_stream_time_freq = np.array(
            [
                transform_wavelet_freq(
                    data=null_stream_freq[i],
                    sampling_frequency=sampling_frequency,
                    frequency_resolution=wavelet_frequency_resolution,
                    nx=wavelet_nx,
                )
                for i in range(len(null_stream_freq))
            ]
        )

        # Step 5: Apply the time-frequency filter to the null stream
        filtered_null_strain = null_stream_time_freq * time_frequency_filter

        # Step 6: Sum the squared magnitude to obtain the total null energy
        null_energy = np.sum(np.abs(filtered_null_strain) ** 2)

        return null_energy

    # =========================================================================
    # COMPONENT METHODS (BUILDING BLOCKS)
    # =========================================================================

    def _compute_gw_projector(self, whitened_antenna_pattern_matrix, frequency_mask):
        return compute_gw_projector(whitened_antenna_pattern_matrix, frequency_mask)

    def _compute_null_projector(self, gw_projector):
        return compute_null_projector(gw_projector)

    def _compute_null_stream(self, whitened_freq_strain, null_projector, frequency_mask):
        return compute_null_stream(whitened_freq_strain, null_projector, frequency_mask)
