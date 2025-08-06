from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_calibrated_whitened_antenna_pattern_matrix(
    frequency_mask,
    whitened_antenna_pattern_matrix,
    calibration_error_matrix,
):
    """Compute the calibrated whitened antenna pattern matrix.

    Args:
        frequency_mask (numpy array): A boolean array of frequency mask.
        whitened_antenna_pattern_matrix (numpy array): Whitened antenna pattern matrix (frequency, detector, mode).
        calibration_error_matrix (numpy array): Calibration error matrix (detector, frequency).

    Returns:
        numpy array: Calibrated and whitened antenna pattern matrix (frequency, detector, mode).
    """
    output = np.zeros_like(whitened_antenna_pattern_matrix, dtype=calibration_error_matrix.dtype)
    nfreq, ndet, nmode = whitened_antenna_pattern_matrix.shape
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                error = calibration_error_matrix[j, i]
                for k in range(nmode):
                    output[i, j, k] = whitened_antenna_pattern_matrix[i, j, k] * error
    return output
