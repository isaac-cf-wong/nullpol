from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_time_shifted_frequency_domain_strain(
    frequency_array,
    frequency_mask,
    frequency_domain_strain,
    time_delay
):
    """Apply time shift to frequency domain strain data for a single detector.

    Shifts the strain data in time by applying a frequency-dependent phase
    factor in the frequency domain. This is equivalent to a time translation
    in the time domain but computed more efficiently in frequency space.

    The time shift is applied using the formula:
    h_shifted(f) = h(f) × exp(2πif×Δt)

    where h(f) is the original frequency domain strain, f is frequency,
    and Δt is the time delay.

    Args:
        frequency_array (numpy.ndarray): Frequency values with shape (n_frequencies,).
            Array containing the frequency bins in Hz.
        frequency_mask (numpy.ndarray): Boolean mask with shape (n_frequencies,)
            indicating which frequency bins to process. Time shift is applied
            only where the mask is True.
        frequency_domain_strain (numpy.ndarray): Complex frequency domain strain
            with shape (n_frequencies,). Input strain data to be time-shifted.
        time_delay (float): Time delay in seconds. Positive values shift the
            signal forward in time (earlier arrival), negative values shift
            backward (later arrival).

    Returns:
        numpy.ndarray: Time-shifted frequency domain strain with shape
            (n_frequencies,). Complex-valued array with the same shape as input.
            Unmasked frequencies are set to zero.

    Note:
        This function is compiled with Numba for performance. The time shift
        preserves the signal's spectral content while changing its phase
        according to the specified delay.
    """
    output = np.zeros_like(frequency_domain_strain)
    phase_shift = np.exp(1.j*2*np.pi*frequency_array[frequency_mask]*time_delay)
    output[frequency_mask] = frequency_domain_strain[frequency_mask]*phase_shift
    return output


@njit
def compute_time_shifted_frequency_domain_strain_array(
    frequency_array,
    frequency_mask,
    frequency_domain_strain_array,
    time_delay_array
):
    """Apply time shifts to frequency domain strain data for multiple detectors.

    Applies different time shifts to strain data from multiple detectors
    simultaneously. This is commonly used to align multi-detector data to
    a common reference frame (e.g., geocenter) where each detector has a
    different arrival time delay.

    The time shift for each detector is applied using:
    h_shifted[i](f) = h[i](f) × exp(2πif×Δt[i])

    where h[i](f) is the frequency domain strain for detector i, f is frequency,
    and Δt[i] is the time delay for detector i.

    Args:
        frequency_array (numpy.ndarray): Frequency values with shape (n_frequencies,).
            Array containing the frequency bins in Hz.
        frequency_mask (numpy.ndarray): Boolean mask with shape (n_frequencies,)
            indicating which frequency bins to process. Time shifts are applied
            only where the mask is True.
        frequency_domain_strain_array (numpy.ndarray): Complex frequency domain
            strain data with shape (n_detectors, n_frequencies). Input strain
            data from multiple detectors to be time-shifted.
        time_delay_array (numpy.ndarray): Time delays with shape (n_detectors,).
            Array of time delays in seconds, one for each detector. Positive
            values shift signals forward in time, negative values shift backward.

    Returns:
        numpy.ndarray: Time-shifted frequency domain strain array with shape
            (n_detectors, n_frequencies). Complex-valued array where each row
            contains the time-shifted strain for one detector. Unmasked
            frequencies are set to zero.

    Note:
        This function is compiled with Numba for performance. It efficiently
        processes multiple detectors by using numpy's outer product to compute
        all phase shifts simultaneously, then applies them element-wise.
    """
    output = np.zeros_like(frequency_domain_strain_array)
    phase_shift_array = np.exp(np.outer(time_delay_array, 1.j*2*np.pi*frequency_array[frequency_mask]))
    output[:, frequency_mask] = frequency_domain_strain_array[:, frequency_mask] * phase_shift_array
    return output
