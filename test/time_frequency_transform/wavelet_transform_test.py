from pycbc.psd import EinsteinTelescopeP1600143
from pycbc.noise import noise_from_psd
import numpy as np
import scipy.stats
from nullpol.time_frequency_transform import (transform_wavelet_freq_time,
                                              transform_wavelet_freq,
                                              transform_wavelet_freq_time_quadrature,
                                              transform_wavelet_time,
                                              inverse_wavelet_time,
                                              inverse_wavelet_freq_time)
import unittest
import numpy as np

class TestWaveletTransform(unittest.TestCase):
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

    def test_standard_gaussian(self):
        seed = 12
        seglen = 16
        srate = 4096
        tlen = seglen * srate
        delta_f = 1 / seglen
        flen = tlen // 2 + 1
        freq_low = 50
        tf_df = 16
        Nf = int(srate / 2 / tf_df)
        Nt = int(tlen / Nf)
        nx = 4
        freq_low_idx = int(np.ceil(freq_low / tf_df))

        psd = EinsteinTelescopeP1600143(flen, delta_f, freq_low)

        noise = noise_from_psd(tlen, 1. / srate, psd, seed=seed)
        # Whiten the noise frequency series.
        whitened_noise_freq = np.divide((np.fft.rfft(noise.numpy()) / srate), np.sqrt(psd.numpy() / 2 / delta_f), where=psd != 0)
        # Transform to time frequency domain
        whitened_noise_time_freq = transform_wavelet_freq(whitened_noise_freq, Nf, Nt, nx)
        ks_statistic, p_value = scipy.stats.kstest(whitened_noise_time_freq[:,freq_low_idx:-1].flatten(), 'norm')
        self.assertGreater(p_value, 0.05, "The output does not follow a standard Gaussian distribution.")

if __name__ == '__main__':
    unittest.main()
