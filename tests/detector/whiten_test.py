"""Test module for gravitational wave strain whitening functionality.

This module tests the whitening procedures for frequency-domain strain data
from interferometers.
"""

from __future__ import annotations

import unittest

import bilby
import numpy as np
from bilby.gw.detector import InterferometerList
from scipy.stats import kstest

from nullpol.detector.whiten import \
    compute_whitened_frequency_domain_strain_array


class TestWhiten(unittest.TestCase):
    """Test class for strain whitening functionality.

    This class tests the proper implementation of frequency-domain strain
    whitening procedures for multi-detector networks.
    """

    def setUp(self):
        """Set up test interferometer network with synthetic strain data.

        Creates a three-detector network (H1, L1, V1) with simulated strain
        data. This setup mimics realistic detector configurations.
        """
        seed = 12
        bilby.core.utils.random.seed(seed)
        self.sampling_frequency = 2048
        self.duration = 16
        self.minimum_frequency = 20
        self.ifos = InterferometerList(['H1', 'L1', 'V1'])
        for ifo in self.ifos:
            ifo.minimum_frequency = self.minimum_frequency
        self.ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.sampling_frequency,
            duration=self.duration
        )
        self.frequency_mask = np.logical_and.reduce([ifo.frequency_mask for ifo in self.ifos])

    def test_compute_whitened_frequency_domain_strain_array(self):
        """Test whitened strain computation follows expected statistical properties.

        Verifies that the whitening process correctly normalizes the noise to
        unit variance by testing whether the whitened data follows a standard
        normal distribution. This is validated using the Kolmogorov-Smirnov test.
        """
        frequency_domain_strain_array = np.array([
            ifo.frequency_domain_strain for ifo in self.ifos
        ])
        power_spectral_density_array = np.array([
            ifo.power_spectral_density_array for ifo in self.ifos
        ])
        whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain_array(
            frequency_mask=self.frequency_mask,
            frequency_resolution=1./self.duration,
            frequency_domain_strain_array=frequency_domain_strain_array,
            power_spectral_density_array=power_spectral_density_array
        )
        k_low = int(np.ceil(self.minimum_frequency*self.duration))
        truncated_series =  whitened_frequency_domain_strain_array[:, k_low:-1]
        samples = np.concatenate((np.real(truncated_series),
                                  np.imag(truncated_series))).flatten() * np.sqrt(2)
        res = kstest(samples, cdf='norm')
        self.assertGreaterEqual(res.pvalue, 0.05)


if __name__ == '__main__':
    unittest.main()
