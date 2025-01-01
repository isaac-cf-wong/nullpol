def get_shape_of_wavelet_transform(duration,
                                   sampling_frequency,
                                   wavelet_frequency_resolution):
    """A helper function to get the shape of the wavelet transform.
    
    Parameters
    ----------
    duration: float
        The duration of the data segment.
    sampling_frequency: float
        The sampling frequency of the data segment.
    wavelet_frequency_resolution: float
        The frequency resolution of the wavelet transform.
    
    Returns
    -------
    Nt, Nf: int, int
        The number of time and frequency bins in the wavelet transform.
    """
    Nf = int(sampling_frequency / 2 / wavelet_frequency_resolution)
    Nt = int(duration*sampling_frequency / Nf)
    return Nt, Nf
