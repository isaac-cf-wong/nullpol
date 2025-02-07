import numpy as np
from bilby.core.utils import infft
from ..time_frequency_transform import (transform_wavelet_freq,
                                        get_shape_of_wavelet_transform,
                                        stft)


def simulate_psd_from_bilby_psd(psd,
                                seglen,
                                srate,
                                wavelet_frequency_resolution,
                                nsample,
                                nx=4.):
    Nt, Nf = get_shape_of_wavelet_transform(duration=seglen,
                                            sampling_frequency=srate,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    samples = []
    for i in range(nsample):
        frequency_domain_strain, frequency_array = psd.get_noise_realisation(srate, seglen)
        samples.append(np.abs(transform_wavelet_freq(data=frequency_domain_strain,
                                                     Nf=Nf,
                                                     Nt=Nt,nx=nx))**2)
    return np.mean(np.concatenate(samples), axis=0)


def simulate_stft_psd_from_bilby_psd(psd,
                                     seglen,
                                     srate,
                                     frequency_resolution,
                                     window_function,
                                     nsample):
    samples = []
    for i in range(nsample):
        frequency_domain_strain, frequency_array = psd.get_noise_realisation(srate, seglen)
        time_domain_strain = infft(frequency_domain_strain=frequency_domain_strain,
                                   sampling_frequency=srate)
        samples.append(np.abs(stft(data=time_domain_strain,
                                   sampling_frequency=srate,
                                   frequency_resolution=frequency_resolution,
                                   window_function=window_function))**2)
    return np.mean(np.concatenate(samples), axis=0)
