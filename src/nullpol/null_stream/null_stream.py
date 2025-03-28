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

    Parameters
    ----------
    frequency_array: 1D numpy array
        Frequency array.
    frequency_mask: 1D numpy array
        Boolean frequency mask.
    whitened_frequency_domain_strain_array: 2D numpy array
        Whitened frequency domain strain array.
    time_delay_array: 1D numpy array
        Time delay array.
    whitened_antenna_pattern_matrix: 3D numpy array
        Whitened antenna pattern matrix.
    fraction: float
        Fraction of projection to the signal subspace.

    Returns
    -------
    2D numpy array:
        Fractional null stream.
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
    """Compute the whitened frequency domain fractional null stream.

    Parameters
    ----------
    frequency_array: 1D numpy array
        Frequency array.
    frequency_mask: 1D numpy array
        Boolean frequency mask.
    whitened_frequency_domain_strain_array: 2D numpy array
        Whitened frequency domain strain array.
    time_delay_array: 1D numpy array
        Time delay array.
    whitened_antenna_pattern_matrix: 3D numpy array
        Whitened antenna pattern matrix.

    Returns
    -------
    2D numpy array:
        Null stream.
    """
    return compute_whitened_frequency_domain_fractional_null_stream(
        frequency_array=frequency_array,
        frequency_mask=frequency_mask,
        whitened_frequency_domain_strain_array=whitened_frequency_domain_strain_array,
        time_delay_array=time_delay_array,
        whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix,
        fraction=1.,
    )
