import bilby
import numpy as np
from nullpol.time_frequency_transform import (
    transform_wavelet_freq,
    transform_wavelet_freq_time,
    transform_wavelet_freq_time_quadrature,
    transform_wavelet_time,
    inverse_wavelet_time,
    inverse_wavelet_freq_time,
    get_shape_of_wavelet_transform
    )
from nullpol.detector import compute_whitened_frequency_domain_strain_array
import unittest


class TestWaveletTransform(unittest.TestCase):
    def setUp(self):
        seed = 12
        np.random.seed(seed)
        bilby.core.utils.random.seed(seed)

    def test_wavelet_transform_of_sine_wave(self):
        srate = 128
        inj_freq = 32
        seglen = 4
        sample_times = np.arange(seglen * srate) / srate
        data = np.sin(2 * np.pi * inj_freq * sample_times)
        df = 4
        Nf = int(srate / 2 / df)
        Nt = int(len(sample_times) / Nf)
        nx = 4.
        data_w = transform_wavelet_freq_time(data, Nf, Nt, nx)
        data_q = transform_wavelet_freq_time_quadrature(data, Nf, Nt, nx)
        data2 = np.abs(data_w)**2 + np.abs(data_q)**2
        inj_freq_idx = int(inj_freq / df)
        # Check whether the output peaks at 32Hz for every time bin
        for i in range(Nt):
            self.assertEqual(np.argmax(np.abs(data2[i])), inj_freq_idx)

    def test_inverse_wavelet_time(self):
        srate = 128
        inj_freq = 32
        seglen = 4
        sample_times = np.arange(seglen * srate) / srate
        data = np.sin(2 * np.pi * inj_freq * sample_times)
        df = 4
        Nf = int(srate / 2 / df)
        Nt = int(len(sample_times) / Nf)
        nx = 4.
        mult = 32
        data_w = transform_wavelet_time(data, Nf, Nt, nx, mult)
        data_rec = inverse_wavelet_time(data_w, Nf, Nt, nx, mult)
        self.assertTrue(np.allclose(data, data_rec))

    def test_inverse_wavelet_freq_time(self):
        srate = 128
        inj_freq = 32
        seglen = 4
        sample_times = np.arange(seglen * srate) / srate
        data = np.sin(2 * np.pi * inj_freq * sample_times)
        df = 4
        Nf = int(srate / 2 / df)
        Nt = int(len(sample_times) / Nf)
        nx = 4.
        data_w = transform_wavelet_freq_time(data, Nf, Nt, nx)
        data_rec = inverse_wavelet_freq_time(data_w, Nf, Nt, nx)
        self.assertTrue(np.allclose(data, data_rec))

    def test_whitened_wavelet_domain_data(self):
        sampling_frequency = 2048
        duration = 16
        minimum_frequency = 20
        wavelet_frequency_resolution = 16.
        wavelet_nx = 4.
        ifo = bilby.gw.detector.InterferometerList(['H1'])[0]
        ifo.minimum_frequency = minimum_frequency
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=sampling_frequency,
            duration=duration,
        )

        # Whiten the data
        whitened_frequency_domain_strain = compute_whitened_frequency_domain_strain_array(
            frequency_mask=ifo.frequency_mask,
            frequency_resolution=1./ifo.duration,
            frequency_domain_strain_array=ifo.frequency_domain_strain[np.newaxis, :],
            power_spectral_density_array=ifo.power_spectral_density_array[np.newaxis, :],
        )
        k_freq_low = int(minimum_frequency*duration)

        # Transform the data to wavelet domain
        Nt, Nf = get_shape_of_wavelet_transform(
            duration=duration,
            sampling_frequency=sampling_frequency,
            wavelet_frequency_resolution=wavelet_frequency_resolution)
        whitened_wavelet_domain_strain = transform_wavelet_freq(
            data=whitened_frequency_domain_strain[0],
            Nf=Nf,
            Nt=Nt,
            nx=wavelet_nx)
        k_wavelet_low = int(np.ceil(minimum_frequency/wavelet_frequency_resolution))
        
        print(np.var(whitened_frequency_domain_strain[0][k_freq_low:-1]))
        print(np.var(whitened_wavelet_domain_strain[:, k_wavelet_low:-1]))
        

if __name__ == '__main__':
    unittest.main()
