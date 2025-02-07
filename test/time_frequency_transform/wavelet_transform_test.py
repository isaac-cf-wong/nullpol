from pycbc.psd import EinsteinTelescopeP1600143
from pycbc.noise import noise_from_psd
import numpy as np
from nullpol.time_frequency_transform import (transform_wavelet_freq_time,
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

if __name__ == '__main__':
    unittest.main()
