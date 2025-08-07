"""Test module for detector network functionality.

This module tests the detector network strain handling procedures and
detector-specific data array operations.
"""

from __future__ import annotations

import bilby
import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullpol.detector import (
    frequency_domain_strain_array,
    time_domain_strain_array,
    time_frequency_domain_strain_array,
    whitened_frequency_domain_strain_array,
)


@pytest.fixture
def interferometer_setup():
    """Set up test interferometer network with synthetic strain data.

    Creates a three-detector network (H1, L1, V1) with simulated strain
    data. This setup mimics realistic detector configurations.

    Returns:
        tuple: Contains ifos, frequency_mask, and test parameters
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

    return ifos, frequency_mask, duration, minimum_frequency


def test_frequency_domain_strain_array_property(interferometer_setup):
    """Test frequency domain strain array property of detector network.

    Validates that the frequency domain strain array property correctly
    aggregates strain data from all detectors into a single array.
    """
    ifos, frequency_mask, duration, minimum_frequency = interferometer_setup

    # Test that the property exists and is callable
    assert hasattr(ifos, "frequency_domain_strain_array")

    # Get the array
    strain_array = ifos.frequency_domain_strain_array

    # Test shape
    expected_shape = (len(ifos), len(ifos[0].frequency_domain_strain))
    assert strain_array.shape == expected_shape

    # Test that it contains the right data
    for i, ifo in enumerate(ifos):
        assert np.allclose(strain_array[i], ifo.frequency_domain_strain)


def test_time_domain_strain_array_property(interferometer_setup):
    """Test time domain strain array property of detector network.

    Validates that the time domain strain array property correctly
    aggregates strain data from all detectors into a single array.
    """
    ifos, frequency_mask, duration, minimum_frequency = interferometer_setup

    # Test that the property exists and is callable
    assert hasattr(ifos, "time_domain_strain_array")

    # Get the array
    strain_array = ifos.time_domain_strain_array

    # Test shape
    expected_shape = (len(ifos), len(ifos[0].time_domain_strain))
    assert strain_array.shape == expected_shape

    # Test that it contains the right data
    for i, ifo in enumerate(ifos):
        assert np.allclose(strain_array[i], ifo.time_domain_strain)


def test_whitened_frequency_domain_strain_array_property(interferometer_setup):
    """Test whitened frequency domain strain array property.

    Validates that the whitened frequency domain strain array property
    correctly applies whitening to all detector strain data.
    """
    ifos, frequency_mask, duration, minimum_frequency = interferometer_setup

    # Test that the property exists and is callable
    assert hasattr(ifos, "whitened_frequency_domain_strain_array")

    # Get the array
    whitened_strain_array = ifos.whitened_frequency_domain_strain_array

    # Test shape
    expected_shape = (len(ifos), len(ifos[0].frequency_domain_strain))
    assert whitened_strain_array.shape == expected_shape

    # Test that output is complex
    assert np.iscomplexobj(whitened_strain_array)

    # Test that whitening was applied (should differ from original strain)
    original_strain_array = ifos.frequency_domain_strain_array
    assert not np.allclose(whitened_strain_array, original_strain_array)


def test_frequency_domain_strain_array():
    """Test frequency domain strain array function."""
    # Test that the function exists as a property descriptor
    assert hasattr(frequency_domain_strain_array, "__get__")

    # The function is actually a property descriptor, so we test it indirectly
    # through the interferometer setup test above


def test_time_domain_strain_array():
    """Test time domain strain array function."""
    # Test that the function exists as a property descriptor
    assert hasattr(time_domain_strain_array, "__get__")

    # The function is actually a property descriptor, so we test it indirectly
    # through the interferometer setup test above


def test_time_frequency_domain_strain_array():
    """Test time-frequency domain strain array function."""
    # Test that the function exists as a property descriptor
    assert hasattr(time_frequency_domain_strain_array, "__get__")

    # TODO: Add more specific tests when function implementation is stable


def test_whitened_frequency_domain_strain_array():
    """Test whitened frequency domain strain array function."""
    # Test that the function exists as a property descriptor
    assert hasattr(whitened_frequency_domain_strain_array, "__get__")

    # The function is actually a property descriptor, so we test it indirectly
    # through the interferometer setup test above
