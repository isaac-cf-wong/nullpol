import numpy as np
from numba import njit


@njit
def compute_time_shifted_frequency_domain_strain(
    frequency_array,
    frequency_mask,
    frequency_domain_strain,
    time_delay
):
    """Compute the time shifted frequency domain strain.

    Args:
        frequency_array (1D numpy array): Frequency array.
        frequency_mask (1D numpy array): A boolean array of frequency mask.
        frequency_domain_strain (1D numpy array): Frequency domain strain.
        time_delay (float): Time delay in second.

    Returns:
        1D numpy array: Time shifted frequency domain strain array.
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
    """Compute time shifted frequency domain strain array.

    Args:
        frequency_array (1D numpy array): Frequency array.
        frequency_mask (1D numpy array): A boolean array of frequency mask.
        frequency_domain_strain_array (2D numpy array): Frequency domain strain array.
        time_delay_array (1D numpy array): Time delay array in second.

    Returns:
        2D numpy array: Time shifted frequency domain strain array.
    """
    output = np.zeros_like(frequency_domain_strain_array)
    phase_shift_array = np.exp(np.outer(time_delay_array, 1.j*2*np.pi*frequency_array[frequency_mask]))
    output[:, frequency_mask] = frequency_domain_strain_array[:, frequency_mask] * phase_shift_array
    return output
