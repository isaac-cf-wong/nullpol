from __future__ import annotations


def get_shape_of_wavelet_transform(duration: float,
                                   sampling_frequency: float,
                                   wavelet_frequency_resolution: float) -> tuple[int, int]:
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


def get_shape_of_stft(duration: float,
                      sampling_frequency: float,
                      frequency_resolution: float) -> tuple[int, int]:
    """Get the shape of short-time Fourier transform.

    Args:
        duration (float): The duration of the data segment.
        sampling_frequency (float): The sampling frequency of the data segment.
        frequency_resolution (float): The frequency resolution of the short-time Fourier transform.

    Returns:
        Tuple[int, int]: The number of time and frequency bins in the wavelet transform (Nt, Nf).
    """
    N = int(sampling_frequency / frequency_resolution)
    hop_size = N
    num_segments = int(duration*sampling_frequency - N) // hop_size + 1
    return num_segments, N//2 + 1
