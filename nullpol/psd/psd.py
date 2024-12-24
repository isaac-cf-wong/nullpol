from pycbc.noise import noise_from_psd
from pycbc.types import FrequencySeries
import numpy as np
from ..time_frequency_transform import transform_wavelet_freq_time


def get_pycbc_psd(psd_array, delta_f):
    psd_array = psd_array.copy()
    psd_array[np.isinf(psd_array)] = 0.
    return FrequencySeries(psd_array, delta_f=delta_f)

def simulate_psd_from_psd(psd, seglen, srate, wavelet_frequency_resolution, nsample, nx=4., minimum_frequency=None, maximum_frequency=None):
    Nf = int(srate / 2 / wavelet_frequency_resolution)
    tlen = int(seglen * srate)
    Nt = int(tlen / Nf)
    delta_t = 1. / srate
    return np.mean(np.concatenate([np.abs(transform_wavelet_freq_time(noise_from_psd(tlen, delta_t, psd).numpy(),Nf,Nt,nx))**2 for _ in range(nsample)]), axis=0) * 2 / srate