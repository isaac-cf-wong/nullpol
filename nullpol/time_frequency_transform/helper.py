def get_shape_of_wavelet_transform(duration,
                                   sampling_frequency,
                                   wavelet_frequency_resolution):
    Nf = int(sampling_frequency / 2 / wavelet_frequency_resolution)
    Nt = int(duration*sampling_frequency / Nf)
    return Nt, Nf