import unittest
import numpy as np
import bilby
from bilby.gw.detector import PowerSpectralDensity
import scipy.stats
from nullpol.time_frequency_transform import (transform_wavelet_freq,
                                              get_shape_of_wavelet_transform)
from nullpol.psd import simulate_psd_from_bilby_psd
from nullpol.detector import compute_whitened_time_frequency_domain_strain_array


class TestPsd(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)
        bilby.core.utils.random.seed(seed)

    def test_simulate_psd_from_bilby_psd(self):
        seglen = 8
        srate = 2048
        tlen = seglen * srate
        flen = tlen // 2 + 1
        delta_f = 1. / seglen
        minimum_frequency = 20
        wavelet_frequency_resolution = 16
        minimum_k = int(minimum_frequency / delta_f)
        minimum_tf_k = int(np.ceil(minimum_frequency / wavelet_frequency_resolution))
        Nt, Nf = get_shape_of_wavelet_transform(seglen, srate, wavelet_frequency_resolution)        
        nsample = 10
        nx = 4
        psd_array = np.random.randn(flen)**2 + 10
        psd_array[:minimum_k] = 0.
        frequency_array = np.arange(len(psd_array)) / seglen
        psd = PowerSpectralDensity(frequency_array=frequency_array, psd_array=psd_array)        
        wavelet_psd = simulate_psd_from_bilby_psd(psd, seglen, srate, wavelet_frequency_resolution, nsample, nx)
        frequency_domain_strain, frequency_array = psd.get_noise_realisation(srate, seglen)
        noise = transform_wavelet_freq(frequency_domain_strain,Nf,Nt,nx)
        time_frequency_filter = np.full_like(noise, True).astype(bool)
        time_frequency_filter[:,:minimum_tf_k] = False
        # Whiten
        whitened_noise = compute_whitened_time_frequency_domain_strain_array(noise[None,:,:],
                                                                             wavelet_psd[None,:],
                                                                             time_frequency_filter)
        res = scipy.stats.kstest(whitened_noise[:,:,minimum_tf_k:].flatten(), cdf='norm')
        self.assertGreaterEqual(res.pvalue, 0.05)
        
if __name__ == '__main__':
    unittest.main()