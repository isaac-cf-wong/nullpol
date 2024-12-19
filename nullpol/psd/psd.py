from pycbc.noise import noise_from_psd
import numpy as np
from ..time_frequency_transform import transform_wavelet_freq_time


def simulate_psd_from_psd(psd, seglen, srate, wavelet_frequency_resolution, nsample, nx=4.):
    Nf = int(srate / 2 / wavelet_frequency_resolution)
    tlen = seglen * srate
    Nt = int(tlen / Nf)
    delta_t = 1. / srate
    return np.mean(np.concatenate([transform_wavelet_freq_time(noise_from_psd(tlen, delta_t, psd).numpy(),Nf,Nt,nx)**2 for _ in range(nsample)]), axis=0) * 2 / srate