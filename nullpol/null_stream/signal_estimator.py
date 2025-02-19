from numba import njit
import numpy as np
from .projector import compute_gw_projector_masked


@njit
def estimate_frequency_domain_signal_at_geocenter(
    frequency_array,
    frequency_mask,
    whitened_frequency_domain_strain_array_at_geocenter,
    whitened_antenna_pattern_matrix
):
    """Estimate frequency domain signal at geocenter.

    Args:
        frequency_array (numpy array): Frequency array (frequency).
        frequency_mask (numpy array): A boolean array of frequency mask (frequency).
        whitened_frequency_domain_strain_array_at_geocenter (numpy array): Whitened frequency domain strain array at geocenter (detector, frequency).
        whitened_antenna_pattern_matrix (numpy array): Whitened antenna pattern matrix (frequency, detector, mode).

    Returns:
        numpy array: Estimated frequency domain signal at geocenter (detector, frequency).
    """
    # Compute the signal projector.
    gw_projector = compute_gw_projector_masked(
        whitened_antenna_pattern_matrix=whitened_antenna_pattern_matrix,
        frequency_mask=frequency_mask
    )
    # Compute the signal estimator.
    s_est = np.zeros_like(whitened_frequency_domain_strain_array_at_geocenter)
    for i in range(len(frequency_array)):
        if frequency_mask[i]:
            Pgw = np.ascontiguousarray(gw_projector[i, :, :])
            s_est[:, i] = Pgw @ whitened_frequency_domain_strain_array_at_geocenter[:, i]
    return s_est
