from __future__ import annotations


def get_shape_of_wavelet_transform(duration,
                                   sampling_frequency,
                                   wavelet_frequency_resolution):
    """A helper function to get the shape of the wavelet transform.

    Args:
        duration (float): The duration of the data segment.
        sampling_frequency (float): The sampling frequency of the data segment.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.

    Returns:
        tuple[int, int]: The number of time and frequency bins in the wavelet transform (Nt, Nf).
    """
    Nf = int(sampling_frequency / 2 / wavelet_frequency_resolution)
    Nt = int(duration*sampling_frequency / Nf)
    return Nt, Nf


def get_shape_of_stft(duration,
                      sampling_frequency,
                      frequency_resolution):
    N = int(sampling_frequency / frequency_resolution)
    hop_size = N
    num_segments = (duration*sampling_frequency - N) // hop_size + 1
    return num_segments, N//2 + 1
