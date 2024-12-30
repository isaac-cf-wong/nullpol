import unittest
import bilby
from bilby.gw.detector import InterferometerList
import numpy as np
import scipy.stats
from nullpol.time_frequency_transform import get_shape_of_wavelet_transform
from nullpol.injection import create_injection
from nullpol.detector import (simulate_wavelet_psd,
                              compute_time_frequency_domain_strain_array,
                              compute_whitened_time_frequency_domain_strain_array)


class TestInterferometer(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)
        bilby.core.utils.random.seed(seed)
        self.duration = 8
        self.sampling_frequency = 4096
        self.minimum_frequency = 20
        self.maximum_frequency = self.sampling_frequency / 2
        self.start_time = 0
        self.wavelet_frequency_resolution = 16
        self.nx = 4.
        self.nsample = 100

    def test_whiten_noise(self):
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        # Generate noise
        create_injection(interferometers=interferometers,
                         duration=self.duration,
                         sampling_frequency=self.sampling_frequency,
                         start_time=self.start_time,
                         noise_type='noise')
        # Create the time_frequency_filter
        Nt, Nf = get_shape_of_wavelet_transform(duration=interferometers[0].duration,
                                                sampling_frequency=interferometers[0].sampling_frequency,
                                                wavelet_frequency_resolution=self.wavelet_frequency_resolution)
        time_frequency_filter = np.full((Nt, Nf), True).astype(bool)
        min_k = int(np.ceil(self.minimum_frequency / self.wavelet_frequency_resolution))
        max_k = int(np.floor(self.maximum_frequency / self.wavelet_frequency_resolution))
        time_frequency_filter[:,:min_k] = False
        time_frequency_filter[:,max_k:] = False
        # Simulate the wavelet PSD
        time_frequency_domain_strain_array = []
        psd_array = []
        for interferometer in interferometers:
            wavelet_psd = simulate_wavelet_psd(interferometer=interferometer,
                                               wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                               nx=self.nx,
                                               nsample=self.nsample)
            # Compute the time-frequency domain strain array
            time_frequency_domain_strain = compute_time_frequency_domain_strain_array(interferometer=interferometer,
                                                                                      wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                                      nx=self.nx)
            time_frequency_domain_strain_array.append(time_frequency_domain_strain)
            psd_array.append(wavelet_psd)
        time_frequency_domain_strain_array = np.array(time_frequency_domain_strain_array)
        psd_array = np.array(psd_array)
        # Whiten
        whitened_time_frequency_domain_strain_array = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array=time_frequency_domain_strain_array,
                                                                                                          psd_array=psd_array,
                                                                                                          time_frequency_filter=time_frequency_filter)
        samples = whitened_time_frequency_domain_strain_array[:,time_frequency_filter]
        result = scipy.stats.kstest(samples.flatten(), cdf='norm')
        self.assertGreaterEqual(result.pvalue, 0.05)

if __name__ == '__main__':
    unittest.main()