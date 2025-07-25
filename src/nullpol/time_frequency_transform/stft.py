from __future__ import annotations

import numpy as np
from numba import njit


@njit
def stft(data, sampling_frequency, frequency_resolution, window_function):
    """Compute the Short-Time Fourier Transform (STFT) of time series data.

    Args:
        data (numpy.ndarray): Input time series data.
        sampling_frequency (float): Sampling frequency in Hz.
        frequency_resolution (float): Desired frequency resolution in Hz.
        window_function (numpy.ndarray): Window function to apply to each segment.

    Returns:
        numpy.ndarray: STFT coefficients with shape (n_time_segments, n_frequencies).
    """
    # Segment length in samples
    N = int(sampling_frequency / frequency_resolution)
    hop_size = N  # No overlap for non-overlapping STFT

    # Number of segments (time bins) we can extract
    num_segments = (len(data) - N) // hop_size + 1
    # Initialize the STFT matrix (complex numbers)
    segments = np.zeros((num_segments, N), dtype=data.dtype)
    # Iterate over each segment
    for t in range(num_segments):
        start_idx = t * hop_size
        segments[t, :] = data[start_idx:start_idx + N]
    return np.fft.rfft(segments*window_function) / sampling_frequency
