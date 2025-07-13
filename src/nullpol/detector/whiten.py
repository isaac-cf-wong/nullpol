from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_whitened_frequency_domain_strain_array(
        frequency_mask,
        frequency_resolution,
        frequency_domain_strain_array,
        power_spectral_density_array,
):
    """Compute the whitened frequency domain strain array.

    Args:
        frequency_mask (numpy array): A boolean array of frequency mask.
        frequency_resolution (float): Frequency resolution in Hz.
        frequency_domain_strain_array (numpy array): Frequency domain strain array (detector, frequency).
        power_spectral_density_array (numpy array): Power spectral density array (detector, frequency).

    Returns:
        numpy: Whitened frequency domain strain array.
    """
    output = np.zeros_like(frequency_domain_strain_array)
    output[:, frequency_mask] = frequency_domain_strain_array[:, frequency_mask] \
        / np.sqrt(power_spectral_density_array[:, frequency_mask]/(2*frequency_resolution))
    return output

@njit
def compute_whitened_antenna_pattern_matrix_masked(antenna_pattern_matrix,
                                                   psd_array,
                                                   frequency_mask):
    """Compute the whitened antenna pattern matrix with a frequency mask.

    Args:
        antenna_pattern_matrix (numpy array): Antenna pattern matrix (detector, mode).
        psd_array (numpy array): Power spectral density array (detector, frequency).
        frequency_mask (numpy array): A boolean array (frequency).

    Returns:
        numpy array: Whitened antenna pattern matrix (frequency, detector, mode).
    """
    nfreq = len(frequency_mask)
    ndet, nmode = antenna_pattern_matrix.shape
    output = np.zeros((nfreq, ndet, nmode), dtype=antenna_pattern_matrix.dtype)
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                output[i, j, :] = antenna_pattern_matrix[j, :] / np.sqrt(psd_array[j, i])
    return output
