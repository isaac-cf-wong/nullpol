from __future__ import annotations

from numba import njit

from .signal_estimator import estimate_frequency_domain_signal_at_geocenter
from .time_shift import compute_time_shifted_frequency_domain_strain_array


@njit
def compute_whitened_frequency_domain_fractional_null_stream(
    frequency_array,
    frequency_mask,
    whitened_frequency_domain_strain_array,
    time_delay_array,
    whitened_antenna_pattern_matrix,
    fraction,
):
    """Compute the whitened frequency domain fractional null stream.

    Args:
        frequency_array (numpy.ndarray): 1D numpy array of frequencies.
        frequency_mask (numpy.ndarray): 1D boolean numpy array for frequency mask.
        whitened_frequency_domain_strain_array (numpy.ndarray): 2D numpy array of whitened frequency domain strain.
        time_delay_array (numpy.ndarray): 1D numpy array of time delays.
        whitened_antenna_pattern_matrix (numpy.ndarray): 3D numpy array of whitened antenna pattern matrix.
        fraction (float): Fraction of projection to the signal subspace.

    Returns:
        numpy.ndarray: 2D numpy array representing the fractional null stream.
    """
    # Compute the time delayed frequency domain strain array.
    d_ts = compute_time_shifted_frequency_domain_strain_array(
        frequency_array=frequency_array,
        frequency_mask=frequency_mask,
        frequency_domain_strain_array=whitened_frequency_domain_strain_array,
        time_delay_array=time_delay_array,
    )
    s_est = estimate_frequency_domain_signal_at_geocenter(
        frequency_array=frequency_array,
        frequency_mask=frequency_mask,
        whitened_frequency_domain_strain_array=whitened_frequency_domain_strain_array,
        time_delay_array=time_delay_array,
        whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix
    )
    if fraction != 1.:
        s_est = s_est * fraction
    return d_ts - s_est


@njit
def compute_whitened_frequency_domain_null_stream(
    frequency_array,
    frequency_mask,
    whitened_frequency_domain_strain_array,
    time_delay_array,
    whitened_antenna_pattern_matrix,
):
    """Compute the whitened frequency domain null stream.

    Args:
        frequency_array (numpy.ndarray): 1D numpy array of frequencies.
        frequency_mask (numpy.ndarray): 1D boolean numpy array for frequency mask.
        whitened_frequency_domain_strain_array (numpy.ndarray): 2D numpy array of whitened frequency domain strain.
        time_delay_array (numpy.ndarray): 1D numpy array of time delays.
        whitened_antenna_pattern_matrix (numpy.ndarray): 3D numpy array of whitened antenna pattern matrix.

    Returns:
        numpy.ndarray: 2D numpy array representing the null stream.
    """
    return compute_whitened_frequency_domain_fractional_null_stream(
        frequency_array=frequency_array,
        frequency_mask=frequency_mask,
        whitened_frequency_domain_strain_array=whitened_frequency_domain_strain_array,
        time_delay_array=time_delay_array,
        whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix,
        fraction=1.,
    )
