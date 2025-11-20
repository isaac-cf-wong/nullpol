"""Test module for lensing data context functionality.

This module tests the LensingTimeFrequencyDataContext class and its
handling of two detector sets for gravitationally lensed signals.
"""

from __future__ import annotations

import bilby
import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullpol.analysis.lensing.data_context import LensingTimeFrequencyDataContext


@pytest.fixture
def two_detector_sets():
    """Set up two sets of interferometers for lensing tests.

    Returns:
        tuple: Two lists of interferometers simulating observation
            of two lensed images using overlapping detectors.
    """
    seed = 42
    bilby.core.utils.random.seed(seed)
    sampling_frequency = 2048
    duration = 8
    minimum_frequency = 20

    # First set of detectors - all three detectors
    ifos_1 = InterferometerList(["H1", "L1", "V1"])
    for ifo in ifos_1:
        ifo.minimum_frequency = minimum_frequency
    ifos_1.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)

    # Second set of detectors - subset of first set
    ifos_2 = InterferometerList(["H1", "L1"])
    for ifo in ifos_2:
        ifo.minimum_frequency = minimum_frequency
    ifos_2.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)

    return [ifos_1, ifos_2]


class TestLensingTimeFrequencyDataContext:
    """Test class for LensingTimeFrequencyDataContext."""

    def test_initialization(self, two_detector_sets):
        """Test that context initializes with two detector sets."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        assert context is not None
        assert len(context.interferometers_1) == 3
        assert len(context.interferometers_2) == 2
        assert len(context.interferometers) == 5  # Combined

    def test_interferometer_properties(self, two_detector_sets):
        """Test that interferometer properties are correctly separated."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        # Check first set
        assert context.interferometers_1[0].name == "H1"
        assert context.interferometers_1[1].name == "L1"
        assert context.interferometers_1[2].name == "V1"

        # Check second set
        assert context.interferometers_2[0].name == "H1"
        assert context.interferometers_2[1].name == "L1"

        # Check combined
        assert context.interferometers[0].name == "H1"
        assert context.interferometers[1].name == "L1"
        assert context.interferometers[2].name == "V1"

    def test_compute_time_delay_array_no_lensing_delay(self, two_detector_sets):
        """Test time delay computation with zero lensing delay."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": 0.0,  # No lensing delay
        }

        time_delays = context.compute_time_delay_array(parameters)

        # Should have 5 time delays (3 from set 1, 2 from set 2)
        assert len(time_delays) == 5

        # All time delays should be finite
        assert np.all(np.isfinite(time_delays))

    def test_compute_time_delay_array_with_lensing_delay(self, two_detector_sets):
        """Test that lensing time delay is applied to second set."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        lensing_delay = 0.1  # 100 ms delay
        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": lensing_delay,
        }

        time_delays = context.compute_time_delay_array(parameters)

        # Compute expected delays manually
        expected_delays_1 = [
            ifo.time_delay_from_geocenter(ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"])
            for ifo in context.interferometers_1
        ]
        expected_delays_2 = [
            ifo.time_delay_from_geocenter(
                ra=parameters["ra"], dec=parameters["dec"], time=parameters["geocent_time"] + lensing_delay
            )
            for ifo in context.interferometers_2
        ]
        expected_delays = np.concatenate([expected_delays_1, expected_delays_2])

        # Check that computed delays match expected
        np.testing.assert_allclose(time_delays, expected_delays, rtol=1e-10)

    def test_time_delay_array_shape(self, two_detector_sets):
        """Test that time delay array has correct shape."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": 0.05,
        }

        time_delays = context.compute_time_delay_array(parameters)

        # Should have total number of detectors
        total_detectors = len(context.interferometers_1) + len(context.interferometers_2)
        assert time_delays.shape == (total_detectors,)

    def test_time_delay_varies_with_sky_position(self, two_detector_sets):
        """Test that time delays change with sky position."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        params1 = {
            "ra": 0.0,
            "dec": 0.0,
            "geocent_time": 1234567890.0,
            "time_delay": 0.1,
        }
        params2 = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": 0.1,
        }

        delays1 = context.compute_time_delay_array(params1)
        delays2 = context.compute_time_delay_array(params2)

        # Different sky positions should give different delays
        assert not np.allclose(delays1, delays2)

    def test_time_delay_varies_with_lensing_delay(self, two_detector_sets):
        """Test that second set time delays change with lensing delay."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        params1 = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": 0.0,
        }
        params2 = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": 0.1,
        }

        delays1 = context.compute_time_delay_array(params1)
        delays2 = context.compute_time_delay_array(params2)

        # First set delays should be the same (first 3 detectors)
        np.testing.assert_allclose(delays1[:3], delays2[:3], rtol=1e-10)

        # Second set delays should differ (last 2 detectors)
        # Since we're using same detectors (H1, L1), the difference is very small
        # but should still be measurable due to the lensing time delay
        # Check that the actual time delay difference matches expectations
        delay_diff = np.abs(delays2[3:] - delays1[3:])
        # The difference should be on the order of the lensing delay (0.1s) effect
        # which manifests as microsecond-level differences in detector arrival times
        assert np.any(delay_diff > 1e-10)  # Should have some difference, even if small

    def test_data_processing_properties(self, two_detector_sets):
        """Test that data processing properties are correctly inherited."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        # Check that properties from parent are accessible
        assert context.sampling_frequency > 0
        assert context.duration > 0
        assert context.frequency_array is not None
        assert context.frequency_mask is not None
        assert context.power_spectral_density_array is not None

    def test_whitened_strain_computation(self, two_detector_sets):
        """Test that whitened strain can be computed for combined detectors."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        whitened_strain = context.whitened_frequency_domain_strain_array

        # Should have strain for all 5 detectors (3 + 2)
        assert whitened_strain.shape[0] == 5
        assert np.all(np.isfinite(whitened_strain))

    def test_time_frequency_filter_application(self, two_detector_sets):
        """Test that time-frequency filter is properly stored."""
        # Create context first to get actual dimensions
        wavelet_frequency_resolution = 4.0
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=256,
        )

        # Calculate expected shape - use tf_Nt and tf_Nf from context
        tf_Nt = context.tf_Nt
        tf_Nf = context.tf_Nf

        # Now create a new context with matching filter
        tf_filter = np.ones((tf_Nt, tf_Nf))
        context_with_filter = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=256,
            time_frequency_filter=tf_filter,
        )

        # Filter should be accessible
        assert context_with_filter.time_frequency_filter is not None
        assert context_with_filter.time_frequency_filter.shape == tf_filter.shape

    def test_edge_case_single_detector_per_set(self):
        """Test with single detector in each set."""
        seed = 42
        bilby.core.utils.random.seed(seed)
        sampling_frequency = 2048
        duration = 8
        minimum_frequency = 20

        ifos_1 = InterferometerList(["H1"])
        ifos_2 = InterferometerList(["L1"])

        # Combine into single list for setup
        combined_list = list(ifos_1) + list(ifos_2)
        all_ifos = InterferometerList(combined_list)
        for ifo in all_ifos:
            ifo.minimum_frequency = minimum_frequency

        all_ifos.set_strain_data_from_power_spectral_densities(sampling_frequency=sampling_frequency, duration=duration)

        # Separate back into two sets
        ifos_1 = InterferometerList([all_ifos[0]])
        ifos_2 = InterferometerList([all_ifos[1]])

        context = LensingTimeFrequencyDataContext(
            interferometers=[ifos_1, ifos_2],
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        parameters = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
            "time_delay": 0.1,
        }

        time_delays = context.compute_time_delay_array(parameters)

        # Should have 2 time delays
        assert len(time_delays) == 2
        assert np.all(np.isfinite(time_delays))

    def test_lensing_delay_independence(self, two_detector_sets):
        """Test that first set is independent of lensing delay value."""
        context = LensingTimeFrequencyDataContext(
            interferometers=two_detector_sets,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
        )

        base_params = {
            "ra": 1.0,
            "dec": 0.5,
            "geocent_time": 1234567890.0,
        }

        # Test with different lensing delays
        delays_small = context.compute_time_delay_array({**base_params, "time_delay": 0.01})
        delays_large = context.compute_time_delay_array({**base_params, "time_delay": 1.0})

        # First set (indices 0, 1, 2) should be identical
        np.testing.assert_allclose(delays_small[:3], delays_large[:3], rtol=1e-12)
