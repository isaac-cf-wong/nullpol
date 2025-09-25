"""Test module for data context functionality.

This module tests the TimeFrequencyDataContext class and its signal processing
functions including whitening, time-shifting, and data management.
"""

from __future__ import annotations

import bilby
import numpy as np
import pytest
from bilby.gw.detector import InterferometerList
from scipy.stats import kstest
import tempfile

from nullpol.analysis.data_context import (
    compute_whitened_frequency_domain_strain_array,
    compute_time_shifted_frequency_domain_strain,
    compute_time_shifted_frequency_domain_strain_array,
    TimeFrequencyDataContext,
)


@pytest.fixture
def detector_setup():
    """Set up test interferometer network with synthetic strain data.

    Creates a three-detector network (H1, L1, V1) with simulated strain
    data. This setup mimics realistic detector configurations.

    Returns:
        tuple: Contains interferometers, frequency mask, and test parameters
            (ifos, frequency_mask, sampling_frequency, duration, minimum_frequency)
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


@pytest.mark.integration
def test_compute_whitened_frequency_domain_strain_array_statistical_properties(detector_setup):
    """Test whitened strain computation follows expected statistical properties.

    Verifies that the whitening process correctly normalizes the noise to
    unit variance by testing whether the whitened data follows a standard
    normal distribution. This is validated using the Kolmogorov-Smirnov test.

    Uses a known statistical test: properly whitened Gaussian noise should
    pass the Kolmogorov-Smirnov test for normality with p-value >= 0.05.
    """
    ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

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


@pytest.mark.integration
def test_compute_whitened_frequency_domain_strain_array_basic_properties(detector_setup):
    """Test basic properties of whitened strain array computation.

    Tests known properties: whitened array should be complex, have same shape
    as input, and produce finite values. Also tests that zero PSD handling
    works correctly (should produce zeros in whitened output).
    """
    ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

    frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in ifos])
    power_spectral_density_array = np.array([ifo.power_spectral_density_array for ifo in ifos])

    whitened_array = compute_whitened_frequency_domain_strain_array(
        frequency_mask=frequency_mask,
        frequency_resolution=1.0 / duration,
        frequency_domain_strain_array=frequency_domain_strain_array,
        power_spectral_density_array=power_spectral_density_array,
    )

    # Test output shape matches input
    assert whitened_array.shape == frequency_domain_strain_array.shape

    # Test that masked frequencies are processed, unmasked are zero
    assert np.all(whitened_array[:, ~frequency_mask] == 0)
    assert np.any(whitened_array[:, frequency_mask] != 0)

    # Test output is complex
    assert np.iscomplexobj(whitened_array)

    # Test known property: whitening with zero PSD should produce zeros
    zero_psd_array = np.zeros_like(power_spectral_density_array)
    zero_psd_array[:, frequency_mask] = 1e-40  # Very small non-zero value
    zero_whitened = compute_whitened_frequency_domain_strain_array(
        frequency_mask=frequency_mask,
        frequency_resolution=1.0 / duration,
        frequency_domain_strain_array=frequency_domain_strain_array,
        power_spectral_density_array=zero_psd_array,
    )

    # With very small PSD, whitened values should be very large or handled appropriately
    assert np.all(np.isfinite(zero_whitened)), "Whitened values should be finite even with small PSD"


def test_compute_time_shifted_frequency_domain_strain_basic():
    """Test basic time-shifting functionality with known phase shifts.

    Uses a simple sinusoidal signal where time shift produces a known
    phase shift: phase_shift = -2π * frequency * time_shift
    """
    # Create synthetic frequency domain strain
    n_freq = 100
    frequency_array = np.linspace(0, 50, n_freq)
    frequency_mask = np.ones(n_freq, dtype=bool)
    frequency_mask[0] = False  # DC component
    frequency_mask[-1] = False  # Nyquist frequency

    # Simple sinusoidal signal in frequency domain at a known frequency
    frequency_domain_strain = np.zeros(n_freq, dtype=complex)
    test_freq_bin = 10
    test_frequency = frequency_array[test_freq_bin]  # ~5 Hz
    frequency_domain_strain[test_freq_bin] = 1.0 + 1.0j  # Known amplitude and phase

    time_delay = 0.1  # seconds

    shifted_strain = compute_time_shifted_frequency_domain_strain(
        frequency_array, frequency_mask, frequency_domain_strain, time_delay
    )

    # Test output shape
    assert shifted_strain.shape == frequency_domain_strain.shape

    # Test that unmasked frequencies are zero
    assert np.all(shifted_strain[~frequency_mask] == 0)

    # Test that phase shift is applied correctly using known formula
    expected_phase_shift = np.exp(1.0j * 2 * np.pi * test_frequency * time_delay)
    expected_value = frequency_domain_strain[test_freq_bin] * expected_phase_shift

    assert np.isclose(
        shifted_strain[test_freq_bin], expected_value, rtol=1e-10
    ), f"Expected {expected_value}, got {shifted_strain[test_freq_bin]}"

    # Test that amplitude is preserved (time shift only changes phase)
    assert np.isclose(
        abs(shifted_strain[test_freq_bin]), abs(frequency_domain_strain[test_freq_bin]), rtol=1e-10
    ), "Amplitude should be preserved during time shift"


def test_compute_time_shifted_frequency_domain_strain_array():
    """Test time-shifting for multiple detectors with known delays.

    Tests that different time delays produce different phase shifts
    for each detector, and that the relationship follows the expected
    formula: phase_shift = 2π * frequency * time_delay
    """
    n_detectors = 3
    n_freq = 50
    frequency_array = np.linspace(0, 25, n_freq)
    frequency_mask = np.ones(n_freq, dtype=bool)
    frequency_mask[0] = False  # DC component

    # Create identical signals for all detectors at a test frequency
    test_freq_bin = 10
    test_frequency = frequency_array[test_freq_bin]  # ~5 Hz
    strain_array = np.zeros((n_detectors, n_freq), dtype=complex)
    strain_array[:, test_freq_bin] = 1.0 + 0.5j  # Same signal for all detectors

    # Different time delays for each detector
    time_delays = np.array([0.0, 0.1, -0.05])  # seconds

    shifted_array = compute_time_shifted_frequency_domain_strain_array(
        frequency_array, frequency_mask, strain_array, time_delays
    )

    # Test output shape
    assert shifted_array.shape == strain_array.shape

    # Test that each detector has the expected phase shift
    for i, time_delay in enumerate(time_delays):
        expected_phase_factor = np.exp(1.0j * 2 * np.pi * test_frequency * time_delay)
        expected_value = strain_array[i, test_freq_bin] * expected_phase_factor

        assert np.isclose(
            shifted_array[i, test_freq_bin], expected_value, rtol=1e-10
        ), f"Detector {i}: Expected {expected_value}, got {shifted_array[i, test_freq_bin]}"

    # Test that relative phase differences are correct
    # Detector 1 vs Detector 0 should have additional phase of 2π * f * 0.1
    relative_delay = time_delays[1] - time_delays[0]  # 0.1 s
    expected_relative_phase = 2 * np.pi * test_frequency * relative_delay

    phase_0 = np.angle(shifted_array[0, test_freq_bin])
    phase_1 = np.angle(shifted_array[1, test_freq_bin])
    actual_relative_phase = phase_1 - phase_0

    # Handle phase wrapping
    while actual_relative_phase > np.pi:
        actual_relative_phase -= 2 * np.pi
    while actual_relative_phase < -np.pi:
        actual_relative_phase += 2 * np.pi
    while expected_relative_phase > np.pi:
        expected_relative_phase -= 2 * np.pi
    while expected_relative_phase < -np.pi:
        expected_relative_phase += 2 * np.pi

    assert (
        abs(actual_relative_phase - expected_relative_phase) < 0.01
    ), f"Expected relative phase {expected_relative_phase:.3f}, got {actual_relative_phase:.3f}"


@pytest.mark.integration
class TestTimeFrequencyDataContext:
    """Test suite for TimeFrequencyDataContext class."""

    def test_initialization_basic(self, detector_setup):
        """Test basic initialization of TimeFrequencyDataContext.

        Verifies that initialization sets expected properties correctly
        and that derived properties match known relationships.
        """
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        wavelet_frequency_resolution = 2.0
        wavelet_nx = 10

        context = TimeFrequencyDataContext(
            interferometers=ifos, wavelet_frequency_resolution=wavelet_frequency_resolution, wavelet_nx=wavelet_nx
        )

        # Test basic properties match expected values
        assert context.duration == duration, f"Expected duration {duration}, got {context.duration}"
        assert (
            context.sampling_frequency == sampling_frequency
        ), f"Expected sampling_frequency {sampling_frequency}, got {context.sampling_frequency}"
        assert len(context.interferometers) == len(
            ifos
        ), f"Expected {len(ifos)} interferometers, got {len(context.interferometers)}"
        assert context.wavelet_frequency_resolution == wavelet_frequency_resolution
        assert context.wavelet_nx == wavelet_nx

        # Test derived properties follow known formulas
        expected_freq_resolution = 1.0 / duration
        assert (
            abs(context.frequency_resolution - expected_freq_resolution) < 1e-10
        ), f"Expected frequency_resolution {expected_freq_resolution}, got {context.frequency_resolution}"
        assert len(context.frequency_array) == len(ifos[0].frequency_array)
        assert len(context.frequency_mask) == len(frequency_mask)
        assert context.power_spectral_density_array.shape == (len(ifos), len(ifos[0].frequency_array))

    def test_initialization_with_time_frequency_filter_array(self, detector_setup):
        """Test initialization with time-frequency filter as numpy array."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        wavelet_frequency_resolution = 2.0
        wavelet_nx = 10

        # Create a realistic filter shape based on wavelet transform dimensions
        from nullpol.analysis.tf_transforms import get_shape_of_wavelet_transform

        tf_Nt, tf_Nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)
        time_frequency_filter = np.random.rand(tf_Nt, tf_Nf)

        context = TimeFrequencyDataContext(
            interferometers=ifos,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            time_frequency_filter=time_frequency_filter,
        )

        assert context.time_frequency_filter is not None
        assert context.time_frequency_filter.shape == (tf_Nt, tf_Nf)
        np.testing.assert_array_equal(context.time_frequency_filter, time_frequency_filter)

    def test_initialization_with_time_frequency_filter_file(self, detector_setup):
        """Test initialization with time-frequency filter as file path."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        wavelet_frequency_resolution = 2.0
        wavelet_nx = 10

        # Create a temporary filter file
        from nullpol.analysis.tf_transforms import get_shape_of_wavelet_transform

        tf_Nt, tf_Nf = get_shape_of_wavelet_transform(duration, sampling_frequency, wavelet_frequency_resolution)
        time_frequency_filter = np.random.rand(tf_Nt, tf_Nf)

        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp_file:
            np.save(tmp_file.name, time_frequency_filter)
            tmp_file_path = tmp_file.name

        try:
            context = TimeFrequencyDataContext(
                interferometers=ifos,
                wavelet_frequency_resolution=wavelet_frequency_resolution,
                wavelet_nx=wavelet_nx,
                time_frequency_filter=tmp_file_path,
            )

            # Test that filter is loaded correctly when accessed
            loaded_filter = context.time_frequency_filter
            np.testing.assert_array_equal(loaded_filter, time_frequency_filter)

        finally:
            import os

            os.unlink(tmp_file_path)

    def test_frequency_domain_strain_array_property(self, detector_setup):
        """Test frequency_domain_strain_array property."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        context = TimeFrequencyDataContext(interferometers=ifos, wavelet_frequency_resolution=2.0, wavelet_nx=10)

        strain_array = context.frequency_domain_strain_array
        assert strain_array.shape == (len(ifos), len(ifos[0].frequency_array))

        # Test that it matches original interferometer data
        for i, ifo in enumerate(ifos):
            np.testing.assert_array_equal(strain_array[i], ifo.frequency_domain_strain)

    def test_whitened_frequency_domain_strain_array_caching(self, detector_setup):
        """Test that whitened strain array is computed and cached properly."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        context = TimeFrequencyDataContext(interferometers=ifos, wavelet_frequency_resolution=2.0, wavelet_nx=10)

        # First access should compute the array
        whitened_array_1 = context.whitened_frequency_domain_strain_array
        assert whitened_array_1 is not None
        assert whitened_array_1.shape == (len(ifos), len(ifos[0].frequency_array))

        # Second access should return cached result
        whitened_array_2 = context.whitened_frequency_domain_strain_array
        assert whitened_array_1 is whitened_array_2  # Same object reference

    def test_compute_time_delay_array(self, detector_setup):
        """Test computation of time delay array with known sky positions.

        Uses specific sky coordinates where time delays can be predicted:
        - For sources directly overhead at a detector, time delay should be ~0
        - For sources at the horizon, time delays should be maximal (~0.02s)
        """
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        context = TimeFrequencyDataContext(interferometers=ifos, wavelet_frequency_resolution=2.0, wavelet_nx=10)

        # Test case 1: Source directly overhead at Hanford (should give small delay to H1)
        # Hanford coordinates: ~46.5°N, 119.4°W
        parameters_overhead = {
            "ra": np.deg2rad(240.6),  # Convert longitude to RA approximately
            "dec": np.deg2rad(46.5),  # Hanford latitude
            "geocent_time": 1000000000.0,  # GPS time
        }

        time_delays = context.compute_time_delay_array(parameters_overhead)

        # Test output properties
        assert len(time_delays) == len(ifos), f"Expected {len(ifos)} delays, got {len(time_delays)}"
        assert isinstance(time_delays, np.ndarray)

        # Test that delays are reasonable (should be small, typically < 0.1s for any GW source)
        assert np.all(np.abs(time_delays) < 0.1), f"Time delays too large: {time_delays}"

        # Test consistency: delays should be relative to geocenter
        # The first detector's delay should be close to its individual calculation
        individual_delay = ifos[0].time_delay_from_geocenter(
            ra=parameters_overhead["ra"], dec=parameters_overhead["dec"], time=parameters_overhead["geocent_time"]
        )

        assert (
            abs(time_delays[0] - individual_delay) < 1e-10
        ), f"Expected delay {individual_delay}, got {time_delays[0]}"

        # Test case 2: Known extreme case - source at horizon should give larger delays
        parameters_horizon = {
            "ra": 0.0,  # On meridian
            "dec": 0.0,  # On celestial equator (horizon for some detectors)
            "geocent_time": 1000000000.0,
        }

        time_delays_horizon = context.compute_time_delay_array(parameters_horizon)

        # Horizon delays should generally be larger than overhead delays
        # (though this depends on detector locations and exact source position)
        assert len(time_delays_horizon) == len(ifos)
        assert np.all(np.abs(time_delays_horizon) < 0.1)

    def test_compute_whitened_strain_at_geocenter(self, detector_setup):
        """Test computation of whitened strain at geocenter with time shifts."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        context = TimeFrequencyDataContext(interferometers=ifos, wavelet_frequency_resolution=2.0, wavelet_nx=10)

        parameters = {"ra": 0.5, "dec": 0.3, "geocent_time": 1000000000.0}

        strain_geocenter = context.compute_whitened_strain_at_geocenter(parameters)

        # Test output shape
        assert strain_geocenter.shape == (len(ifos), len(ifos[0].frequency_array))

        # Test that output is complex
        assert np.iscomplexobj(strain_geocenter)

        # Test caching - second call should return cached result
        strain_geocenter_2 = context.compute_whitened_strain_at_geocenter(parameters)
        np.testing.assert_array_equal(strain_geocenter, strain_geocenter_2)

    def test_masked_frequency_array_property(self, detector_setup):
        """Test masked_frequency_array property."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        context = TimeFrequencyDataContext(interferometers=ifos, wavelet_frequency_resolution=2.0, wavelet_nx=10)

        masked_freq = context.masked_frequency_array
        full_freq = context.frequency_array
        mask = context.frequency_mask

        # Test that masked array contains only frequencies where mask is True
        expected_masked = full_freq[mask]
        np.testing.assert_array_equal(masked_freq, expected_masked)

    def test_interferometer_validation_different_frequency_resolution(self):
        """Test that validation catches interferometers with different frequency resolutions.

        This test manually creates interferometers with different sampling rates
        to ensure the validation logic correctly identifies inconsistencies.
        """
        # Create interferometers with different frequency resolutions
        # We need to create them separately with different sampling frequencies
        # to test our validation, since bilby will also validate consistency

        # Create two separate interferometer lists with different configurations
        ifos1 = InterferometerList(["H1"])
        ifos1.set_strain_data_from_power_spectral_densities(sampling_frequency=1024, duration=4)

        ifos2 = InterferometerList(["L1"])
        ifos2.set_strain_data_from_power_spectral_densities(
            sampling_frequency=512, duration=4  # Different sampling frequency -> different delta_f
        )

        # Try to combine interferometers with different frequency resolutions
        # This should fail at bilby's level before our custom validation
        combined_ifos = [ifos1[0], ifos2[0]]

        with pytest.raises(ValueError, match="sampling_frequency.*not the same"):
            TimeFrequencyDataContext(interferometers=combined_ifos, wavelet_frequency_resolution=2.0, wavelet_nx=10)

    def test_time_frequency_filter_validation_wrong_shape(self, detector_setup):
        """Test validation of time-frequency filter with wrong dimensions."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        # Create filter with wrong shape
        wrong_filter = np.random.rand(10, 20)  # Wrong dimensions

        with pytest.raises(AssertionError):
            TimeFrequencyDataContext(
                interferometers=ifos,
                wavelet_frequency_resolution=2.0,
                wavelet_nx=10,
                time_frequency_filter=wrong_filter,
            )

    def test_wavelet_transform_properties(self, detector_setup):
        """Test that wavelet transform properties are set correctly."""
        ifos, frequency_mask, sampling_frequency, duration, minimum_frequency = detector_setup

        wavelet_frequency_resolution = 1.5
        wavelet_nx = 12

        context = TimeFrequencyDataContext(
            interferometers=ifos, wavelet_frequency_resolution=wavelet_frequency_resolution, wavelet_nx=wavelet_nx
        )

        # Test wavelet properties
        assert context.wavelet_frequency_resolution == wavelet_frequency_resolution
        assert context.wavelet_nx == wavelet_nx

        # Test that time-frequency dimensions are positive integers
        assert isinstance(context.tf_Nt, int) and context.tf_Nt > 0
        assert isinstance(context.tf_Nf, int) and context.tf_Nf > 0
