from __future__ import annotations

import numpy as np


def compute_filter_by_quantile(time_freq_transformed: np.ndarray, quantile: float=0.9, **kwargs):
    """Filter the time-frequency transformed data with a threshold.

    Args:
        time_freq_transformed (numpy.ndarray): Time-frequency transformed data in shape (n_time, n_freq).
        quantile (float, optional): The threshold for the filtering. Defaults to 0.9.
        **kwargs: Additional keyword arguments (unused, for compatibility).

    Returns:
        numpy.ndarray: A mask with the largest cluster in shape (n_time, n_freq).
    """
    # threshold the data
    filter = time_freq_transformed > np.quantile(time_freq_transformed[time_freq_transformed>0.], quantile)

    return filter
