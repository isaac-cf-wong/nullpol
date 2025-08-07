"""Test module for signal processing functionality.

This module tests signal conditioning and processing functions used
throughout the analysis pipeline.
"""

from __future__ import annotations

import bilby
import numpy as np
import pytest
from bilby.gw.detector import InterferometerList
from scipy.stats import kstest

from nullpol.analysis.signal_processing import compute_whitened_frequency_domain_strain_array


@pytest.fixture
def realistic_detector_setup():
    """Set up test interferometer network with synthetic strain data.

    Creates a three-detector network (H1, L1, V1) with simulated strain
    data. This setup mimics realistic detector configurations.

    Returns:
        tuple: Contains interferometers, frequency mask, and test parameters
    """
    seed = 12
    bilby.core.utils.random.seed(seed)
    sampling_frequency = 2048
    duration = 16
    minimum_frequency = 20
    ifos = InterferometerList(["H1", "L1", "V1"])
    for ifo in ifos:
        ifo.minimum_frequency = minimum_frequency
    ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)
    frequency_mask = np.logical_and.reduce([ifo.frequency_mask for ifo in ifos])

    return (ifos, frequency_mask, sampling_frequency, duration, minimum_frequency)


def test_compute_whitened_frequency_domain_strain_array_statistical_properties(realistic_detector_setup):
    """Test whitened strain computation follows expected statistical properties.

    Verifies that the whitening process correctly normalizes the noise to
    unit variance by testing whether the whitened data follows a standard
    normal distribution. This is validated using the Kolmogorov-Smirnov test.
    """
    ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = realistic_detector_setup

    frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in ifos])
    power_spectral_density_array = np.array([ifo.power_spectral_density_array for ifo in ifos])

    whitened_frequency_domain_strain_array = compute_whitened_frequency_domain_strain_array(
        frequency_mask=frequency_mask,
        frequency_resolution=1.0 / duration,
        frequency_domain_strain_array=frequency_domain_strain_array,
        power_spectral_density_array=power_spectral_density_array,
    )

    k_low = int(np.ceil(minimum_frequency * duration))
    truncated_series = whitened_frequency_domain_strain_array[:, k_low:-1]
    samples = np.concatenate((np.real(truncated_series), np.imag(truncated_series))).flatten() * np.sqrt(2)
    res = kstest(samples, cdf="norm")

    # Test that whitened noise follows standard normal distribution
    assert res.pvalue >= 0.05, f"Whitened noise does not follow normal distribution (p-value: {res.pvalue})"
