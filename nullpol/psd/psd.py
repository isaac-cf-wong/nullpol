from pycbc.noise import noise_from_psd
from pycbc.types import FrequencySeries
import numpy as np
from ..time_frequency_transform import (transform_wavelet_freq_time,
                                        transform_wavelet_freq,
                                        get_shape_of_wavelet_transform)


def get_pycbc_psd(psd_array, delta_f):
    psd_array = psd_array.copy()
    psd_array[np.isinf(psd_array)] = 0.
    return FrequencySeries(psd_array, delta_f=delta_f)

def simulate_psd_from_psd(psd, seglen, srate, wavelet_frequency_resolution, nsample, nx=4.):
    Nt, Nf = get_shape_of_wavelet_transform(duration=seglen,
                                            sampling_frequency=srate,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    tlen = int(seglen * srate)
    delta_t = 1. / srate
    return np.mean(np.concatenate([np.abs(transform_wavelet_freq_time(noise_from_psd(tlen, delta_t, psd).numpy(),Nf,Nt,nx))**2 for _ in range(nsample)]), axis=0)

def simulate_psd_from_bilby_psd(psd, seglen, srate, wavelet_frequency_resolution, nsample, nx=4.):
    Nt, Nf = get_shape_of_wavelet_transform(duration=seglen,
                                            sampling_frequency=srate,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)
    tlen = int(seglen * srate)
    delta_t = 1. / srate
    samples = []
    for i in range(nsample):
        frequency_domain_strain, frequency_array = psd.get_noise_realisation(srate, seglen)
        samples.append(np.abs(transform_wavelet_freq(data=frequency_domain_strain,
                                                     Nf=Nf,
                                                     Nt=Nt,nx=nx))**2)
    return np.mean(np.concatenate(samples), axis=0)