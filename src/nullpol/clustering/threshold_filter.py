import numpy as np

def compute_filter_by_quantile(time_freq_transformed, quantile=0.9, **kwargs):
    """
    Filter the time-frequency transformed data with a threshold.

    Parameters
    ----------
    time_freq_transformed : numpy.ndarray
        Time-frequency transformed data in shape (n_time, n_freq).
    quantile : float, optional
        The threshold for the filtering. Default is 0.9.
    dt : float
        The time resolution in seconds.
    df : float
        The frequency resolution in Hz.

    Returns
    -------
    numpy.ndarray
        A mask with the largest cluster in shape (n_time, n_freq).
    """
    # threshold the data
    filter = time_freq_transformed > np.quantile(time_freq_transformed[time_freq_transformed>0.], quantile)

    return filter