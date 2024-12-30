import unittest
import numpy as np
from pycbc.noise import noise_from_psd
import scipy.stats
from nullpol.time_frequency_transform import (transform_wavelet_freq_time,
                                              get_shape_of_wavelet_transform)
from nullpol.psd import (get_pycbc_psd,
                         simulate_psd_from_psd)
from nullpol.null_stream import compute_whitened_time_frequency_domain_strain_array


class TestPsd(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)

    def test_get_pycbc_psd(self):
        psd_array = np.random.randn(1024)**2 + 10
        delta_f = 0.25
        pycbc_psd = get_pycbc_psd(psd_array, delta_f)
        self.assertEqual(pycbc_psd.delta_f, delta_f)
        self.assertTrue(np.allclose(psd_array, pycbc_psd.numpy()))

    def test_simulate_psd_from_psd(self):
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
        pycbc_psd = get_pycbc_psd(psd_array, delta_f)
        wavelet_psd = simulate_psd_from_psd(pycbc_psd, seglen, srate, wavelet_frequency_resolution, nsample, nx)
        noise = transform_wavelet_freq_time(noise_from_psd(tlen, 1. / srate, pycbc_psd).numpy(),Nf,Nt,nx)
        time_frequency_filter = np.full_like(noise, True).astype(bool)
        time_frequency_filter[:,:minimum_tf_k] = False
        # Whiten
        whitened_noise = compute_whitened_time_frequency_domain_strain_array(noise[None,:,:],
                                                                             wavelet_psd[None,:],
                                                                             time_frequency_filter,
                                                                             srate)
        res = scipy.stats.kstest(whitened_noise[:,:,minimum_tf_k:].flatten(), cdf='norm')
        self.assertGreaterEqual(res.pvalue, 0.05)
        
if __name__ == '__main__':
    unittest.main()