"""Test module for chi-squared time-frequency likelihood functionality.

This module tests the chi-squared likelihood implementation for time-frequency
domain analysis.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.stats
from tqdm import tqdm
from unittest.mock import Mock

import bilby
from bilby.core.utils import logger
from bilby.gw.detector import InterferometerList
from bilby.gw.source import lal_binary_black_hole

from nullpol.analysis.clustering import run_time_frequency_clustering
from nullpol.simulation.injection import create_injection
from nullpol.analysis.likelihood.chi2_tf_likelihood import Chi2TimeFrequencyLikelihood

# Configure logging after imports
logger.setLevel("CRITICAL")


@pytest.fixture(scope="module")
def configuration() -> dict:
    """Configuration fixture for time-frequency likelihood tests.

    Returns:
        Dict: A dictionary containing all test configuration parameters
            including detector setup, signal parameters, and analysis settings.
    """
    seed = 14
    # Set the seed
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)
    duration = 8
    sampling_frequency = 4096
    geocent_time = 1126259642.413
    start_time = geocent_time - 4
    wavelet_frequency_resolution = 16
    wavelet_nx = 4.0
    minimum_frequency = 20
    maximum_frequency = sampling_frequency / 2
    threshold = 1.0
    time_padding = 0.1
    frequency_padding = 1
    skypoints = 10
    parameters = dict(
        mass_1=36.0,
        mass_2=29.0,
        a_1=0.0,
        a_2=0.0,
        tilt_1=0.0,
        tilt_2=0.0,
        phi_12=0.0,
        phi_jl=0.0,
        luminosity_distance=2000.0,
        theta_jn=0.0,
        psi=2.659,
        phase=1.3,
        geocent_time=geocent_time,
        ra=1.375,
        dec=-1.2108,
    )

    return {
        "seed": seed,
        "duration": duration,
        "sampling_frequency": sampling_frequency,
        "geocent_time": geocent_time,
        "start_time": start_time,
        "wavelet_frequency_resolution": wavelet_frequency_resolution,
        "wavelet_nx": wavelet_nx,
        "minimum_frequency": minimum_frequency,
        "maximum_frequency": maximum_frequency,
        "threshold": threshold,
        "time_padding": time_padding,
        "frequency_padding": frequency_padding,
        "skypoints": skypoints,
        "parameters": parameters,
    }


@pytest.fixture(scope="module")
def time_frequency_filter(configuration: dict) -> np.ndarray:
    """Time-frequency filter fixture for likelihood testing.

    Args:
        configuration (Dict): Test configuration parameters.

    Returns:
        np.ndarray: A time-frequency mask computed from injected signal
            parameters for optimal filtering in the wavelet domain.
    """
    parameters = configuration["parameters"]
    duration = configuration["duration"]
    sampling_frequency = configuration["sampling_frequency"]
    start_time = configuration["start_time"]
    wavelet_frequency_resolution = configuration["wavelet_frequency_resolution"]
    wavelet_nx = configuration["wavelet_nx"]
    threshold = configuration["threshold"]
    time_padding = configuration["time_padding"]
    frequency_padding = configuration["frequency_padding"]
    skypoints = configuration["skypoints"]

    interferometers = InterferometerList(["H1", "L1", "V1"])
    noise_type = "zero_noise"
    frequency_domain_source_model = lal_binary_black_hole
    waveform_arguments = dict(waveform_approximant="IMRPhenomPv2", reference_frequency=50)
    create_injection(
        interferometers=interferometers,
        parameters=parameters,
        duration=duration,
        sampling_frequency=sampling_frequency,
        start_time=start_time,
        noise_type=noise_type,
        frequency_domain_source_model=frequency_domain_source_model,
        waveform_arguments=waveform_arguments,
    )
    frequency_domain_strain_array = np.array([ifo.frequency_domain_strain.copy() for ifo in interferometers])
    time_frequency_filter, _ = run_time_frequency_clustering(
        interferometers=interferometers,
        frequency_domain_strain_array=frequency_domain_strain_array,
        wavelet_frequency_resolution=wavelet_frequency_resolution,
        wavelet_nx=wavelet_nx,
        threshold=threshold,
        time_padding=time_padding,
        frequency_padding=frequency_padding,
        skypoints=skypoints,
        return_sky_maximized_spectrogram=True,
        threshold_type="variance",
    )
    return time_frequency_filter


@pytest.mark.integration
def test_noise_residual_energy(configuration: dict, time_frequency_filter: np.ndarray) -> None:
    """Test noise-only residual energy follows chi-squared distribution.

    Validates that the chi-squared likelihood correctly models noise-only
    data by testing whether residual energies follow the expected chi-squared
    distribution with appropriate degrees of freedom.

    Args:
        configuration (Dict): Test configuration parameters.
        time_frequency_filter (np.ndarray): Time-frequency mask for filtering.
    """
    duration = configuration["duration"]
    sampling_frequency = configuration["sampling_frequency"]
    start_time = configuration["start_time"]
    wavelet_frequency_resolution = configuration["wavelet_frequency_resolution"]
    wavelet_nx = configuration["wavelet_nx"]
    geocent_time = configuration["geocent_time"]

    logL_samples = []
    n_samples = 200

    for i in tqdm(range(n_samples), desc="test_noise_residual_energy"):
        # Create a noise injection
        interferometers = InterferometerList(["H1", "L1", "V1"])
        create_injection(
            interferometers=interferometers,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            noise_type="noise",
        )
        polarization_modes = "pc"
        polarization_basis = "pc"
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
        )
        likelihood.parameters = dict(ra=0, dec=0, psi=0, geocent_time=geocent_time)
        logL = likelihood.log_likelihood()
        logL_samples.append(logL)

    # Simulate chi2 random variables and compute their logpdfs
    rng = np.random.default_rng(1234)
    chi2_samples = rng.chisquare(likelihood.DoF, size=n_samples)
    logpdf_samples = scipy.stats.chi2.logpdf(chi2_samples, likelihood.DoF)

    # KS test: are the log likelihoods distributed as the logpdf values?
    result = scipy.stats.ks_2samp(logL_samples, logpdf_samples)
    print(
        f"logL vs logpdf KS statistic = {result.statistic}, p-value = {result.pvalue}, mean logL = {np.mean(logL_samples):.3f}, mean logpdf = {np.mean(logpdf_samples):.3f}, DoF = {likelihood.DoF}"
    )
    assert result.pvalue >= 0.05, f"Log likelihoods do not match expected logpdf distribution (p = {result.pvalue})"


@pytest.mark.integration
def test_signal_residual_energy(configuration: dict, time_frequency_filter: np.ndarray) -> None:
    """Test signal residual energy with correct parameters follows chi-squared distribution.

    Validates that when analyzing injected signals with correct recovery parameters,
    the likelihood residual energy follows the expected chi-squared distribution,
    confirming proper signal subtraction in the likelihood calculation.

    Args:
        configuration (Dict): Test configuration parameters.
        time_frequency_filter (np.ndarray): Time-frequency mask for filtering.
    """
    duration = configuration["duration"]
    sampling_frequency = configuration["sampling_frequency"]
    start_time = configuration["start_time"]
    wavelet_frequency_resolution = configuration["wavelet_frequency_resolution"]
    wavelet_nx = configuration["wavelet_nx"]
    parameters = configuration["parameters"]
    ra = parameters["ra"]
    dec = parameters["dec"]
    psi = parameters["psi"]
    geocent_time = parameters["geocent_time"]

    logL_samples = []
    n_samples = 200

    for i in tqdm(range(n_samples), desc="test_signal_residual_energy"):
        # Create a noise injection
        interferometers = InterferometerList(["H1", "L1", "V1"])
        create_injection(
            interferometers=interferometers,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            parameters=parameters,
            noise_type="gaussian",
        )
        polarization_modes = "pc"
        polarization_basis = "pc"
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
        )
        likelihood.parameters = dict(ra=ra, dec=dec, psi=psi, geocent_time=geocent_time)
        logL = likelihood.log_likelihood()
        logL_samples.append(logL)

    # Simulate chi2 random variables and compute their logpdfs
    rng = np.random.default_rng(1234)
    chi2_samples = rng.chisquare(likelihood.DoF, size=n_samples)
    logpdf_samples = scipy.stats.chi2.logpdf(chi2_samples, likelihood.DoF)

    # KS test: are the log likelihoods distributed as the logpdf values?
    result = scipy.stats.ks_2samp(logL_samples, logpdf_samples)
    print(
        f"logL vs logpdf KS statistic = {result.statistic}, p-value = {result.pvalue}, mean logL = {np.mean(logL_samples):.3f}, mean logpdf = {np.mean(logpdf_samples):.3f}, DoF = {likelihood.DoF}"
    )
    assert result.pvalue >= 0.05, f"Log likelihoods do not match expected logpdf distribution (p = {result.pvalue})"


@pytest.mark.integration
def test_signal_residual_energy_incorrect_parameters(configuration: dict, time_frequency_filter: np.ndarray) -> None:
    """Test signal residual energy with incorrect parameters deviates from chi-squared.

    Validates that when analyzing injected signals with incorrect recovery
    parameters, the likelihood residual energy deviates from the expected
    chi-squared distribution, confirming the likelihood's sensitivity to
    parameter mismatches.

    Args:
        configuration (Dict): Test configuration parameters.
        time_frequency_filter (np.ndarray): Time-frequency mask for filtering.
    """
    duration = configuration["duration"]
    sampling_frequency = configuration["sampling_frequency"]
    start_time = configuration["start_time"]
    wavelet_frequency_resolution = configuration["wavelet_frequency_resolution"]
    wavelet_nx = configuration["wavelet_nx"]
    parameters = configuration["parameters"]
    ra = parameters["ra"] + 0.5
    dec = parameters["dec"] - 0.5
    psi = parameters["psi"] + 0.5
    geocent_time = parameters["geocent_time"] + 10000

    logL_samples = []
    n_samples = 200

    for i in tqdm(range(n_samples), desc="test_signal_residual_energy_incorrect_parameters"):
        # Create a noise injection
        interferometers = InterferometerList(["H1", "L1", "V1"])
        create_injection(
            interferometers=interferometers,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            parameters=parameters,
            noise_type="gaussian",
        )
        polarization_modes = "pc"
        polarization_basis = "pc"
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
        )
        likelihood.parameters = dict(ra=ra, dec=dec, psi=psi, geocent_time=geocent_time)
        logL = likelihood.log_likelihood()
        logL_samples.append(logL)

    # Simulate chi2 random variables and compute their logpdfs
    rng = np.random.default_rng(1234)
    chi2_samples = rng.chisquare(likelihood.DoF, size=n_samples)
    logpdf_samples = scipy.stats.chi2.logpdf(chi2_samples, likelihood.DoF)

    # KS test: are the log likelihoods distributed as the logpdf values?
    result = scipy.stats.ks_2samp(logL_samples, logpdf_samples)
    print(
        f"logL vs logpdf KS statistic = {result.statistic}, p-value = {result.pvalue}, mean logL = {np.mean(logL_samples):.3f}, mean logpdf = {np.mean(logpdf_samples):.3f}, DoF = {likelihood.DoF}"
    )
    assert (
        result.pvalue < 0.05
    ), f"Log likelihoods should deviate from the expected logpdf distribution (p = {result.pvalue})"


@pytest.mark.integration
def test_signal_pc_c_residual_energy(configuration: dict, time_frequency_filter: np.ndarray) -> None:
    """Test single-mode effective antenna pattern residual energy distribution.

    Validates the likelihood behavior when using single-mode effective antenna
    patterns with relative amplification factors.

    Args:
        configuration (Dict): Test configuration parameters.
        time_frequency_filter (np.ndarray): Time-frequency mask for filtering.
    """
    duration = configuration["duration"]
    sampling_frequency = configuration["sampling_frequency"]
    start_time = configuration["start_time"]
    wavelet_frequency_resolution = configuration["wavelet_frequency_resolution"]
    wavelet_nx = configuration["wavelet_nx"]
    parameters = configuration["parameters"]
    ra = parameters["ra"]
    dec = parameters["dec"]
    psi = parameters["psi"]
    geocent_time = parameters["geocent_time"]
    amplitude_cp = 1
    phase_cp = -np.pi / 2

    logL_samples = []
    n_samples = 200
    frequency_domain_source_model = lal_binary_black_hole
    waveform_arguments = dict(waveform_approximant="IMRPhenomPv2", reference_frequency=50)
    for i in tqdm(range(n_samples), desc="test_signal_pc_p_residual_energy"):
        # Create a noise injection
        interferometers = InterferometerList(["H1", "L1", "V1"])
        create_injection(
            interferometers=interferometers,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            parameters=parameters,
            noise_type="gaussian",
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=waveform_arguments,
        )
        polarization_modes = "pc"
        polarization_basis = "p"
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
        )
        likelihood.parameters = dict(
            ra=ra, dec=dec, psi=psi, geocent_time=geocent_time, amplitude_cp=amplitude_cp, phase_cp=phase_cp
        )
        logL = likelihood.log_likelihood()
        logL_samples.append(logL)

    # Simulate chi2 random variables and compute their logpdfs
    rng = np.random.default_rng(1234)
    chi2_samples = rng.chisquare(likelihood.DoF, size=n_samples)
    logpdf_samples = scipy.stats.chi2.logpdf(chi2_samples, likelihood.DoF)

    # KS test: are the log likelihoods distributed as the logpdf values?
    result = scipy.stats.ks_2samp(logL_samples, logpdf_samples)
    print(
        f"logL vs logpdf KS statistic = {result.statistic}, p-value = {result.pvalue}, mean logL = {np.mean(logL_samples):.3f}, mean logpdf = {np.mean(logpdf_samples):.3f}, DoF = {likelihood.DoF}"
    )
    assert result.pvalue >= 0.05, f"Log likelihoods do not match expected logpdf distribution (p = {result.pvalue})"


@pytest.mark.integration
def test_signal_pc_c_residual_energy_incorrect_parameters(
    configuration: dict, time_frequency_filter: np.ndarray
) -> None:
    """Test single-mode likelihood with incorrect amplification parameters.

    Validates that the likelihood correctly identifies parameter mismatches
    when using incorrect relative amplification factors in single-mode
    effective antenna pattern analyses.

    Args:
        configuration (Dict): Test configuration parameters.
        time_frequency_filter (np.ndarray): Time-frequency mask for filtering.
    """
    duration = configuration["duration"]
    sampling_frequency = configuration["sampling_frequency"]
    start_time = configuration["start_time"]
    wavelet_frequency_resolution = configuration["wavelet_frequency_resolution"]
    wavelet_nx = configuration["wavelet_nx"]
    parameters = configuration["parameters"]
    ra = parameters["ra"]
    dec = parameters["dec"]
    psi = parameters["psi"]
    geocent_time = parameters["geocent_time"]
    amplitude_cp = 1
    phase_cp = np.pi / 2

    logL_samples = []
    n_samples = 200
    frequency_domain_source_model = lal_binary_black_hole
    waveform_arguments = dict(waveform_approximant="IMRPhenomPv2", reference_frequency=50)
    for i in tqdm(range(n_samples), desc="test_signal_pc_p_residual_energy_incorrect_parameters"):
        # Create a noise injection
        interferometers = InterferometerList(["H1", "L1", "V1"])
        create_injection(
            interferometers=interferometers,
            duration=duration,
            sampling_frequency=sampling_frequency,
            start_time=start_time,
            parameters=parameters,
            noise_type="gaussian",
            frequency_domain_source_model=frequency_domain_source_model,
            waveform_arguments=waveform_arguments,
        )
        polarization_modes = "pc"
        polarization_basis = "p"
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter,
        )
        likelihood.parameters = dict(
            ra=ra, dec=dec, psi=psi, geocent_time=geocent_time, amplitude_cp=amplitude_cp, phase_cp=phase_cp
        )
        logL = likelihood.log_likelihood()
        logL_samples.append(logL)

    # Simulate chi2 random variables and compute their logpdfs
    rng = np.random.default_rng(1234)
    chi2_samples = rng.chisquare(likelihood.DoF, size=n_samples)
    logpdf_samples = scipy.stats.chi2.logpdf(chi2_samples, likelihood.DoF)

    # KS test: are the log likelihoods distributed as the logpdf values?
    result = scipy.stats.ks_2samp(logL_samples, logpdf_samples)
    print(
        f"logL vs logpdf KS statistic = {result.statistic}, p-value = {result.pvalue}, mean logL = {np.mean(logL_samples):.3f}, mean logpdf = {np.mean(logpdf_samples):.3f}, DoF = {likelihood.DoF}"
    )
    assert (
        result.pvalue < 0.05
    ), f"Log likelihoods should deviate from the expected logpdf distribution (p = {result.pvalue})"


class TestChi2TimeFrequencyLikelihoodEdgeCases:
    """Test edge cases and error conditions for Chi2TimeFrequencyLikelihood."""

    def test_dof_error_when_filter_is_none(self):
        """Test DoF property raises error when time_frequency_filter is None."""
        # Create a mock likelihood instance to test the DoF property error case
        likelihood = Chi2TimeFrequencyLikelihood.__new__(Chi2TimeFrequencyLikelihood)

        # Mock the components to isolate the DoF property test
        likelihood.antenna_pattern_processor = Mock()
        likelihood.antenna_pattern_processor.polarization_basis = np.array([0, 1])  # Example basis

        likelihood.data_context = Mock()
        likelihood.data_context.time_frequency_filter = None  # This should trigger the error

        # Mock interferometers through data_context since it's a property
        mock_interferometers = Mock()
        mock_interferometers.__len__ = Mock(return_value=2)
        likelihood.data_context.interferometers = mock_interferometers

        # Test that accessing DoF raises ValueError when filter is None
        with pytest.raises(ValueError, match="Time frequency filter is not available"):
            _ = likelihood.DoF

    def test_calculate_noise_log_likelihood_missing_strain_array(self):
        """Test _calculate_noise_log_likelihood raises error when strain array is None."""
        # Create a mock likelihood instance to test error condition
        likelihood = Chi2TimeFrequencyLikelihood.__new__(Chi2TimeFrequencyLikelihood)

        # Mock data_context with missing strain array
        likelihood.data_context = Mock()
        likelihood.data_context.whitened_frequency_domain_strain_array = None  # Missing strain
        likelihood.data_context.time_frequency_filter = np.ones((2, 2))  # Filter exists

        # Test that _calculate_noise_log_likelihood raises ValueError
        with pytest.raises(ValueError, match="Whitened frequency domain strain array is not available"):
            likelihood._calculate_noise_log_likelihood()

    def test_calculate_noise_log_likelihood_missing_filter(self):
        """Test _calculate_noise_log_likelihood raises error when filter is None."""
        # Create a mock likelihood instance to test error condition
        likelihood = Chi2TimeFrequencyLikelihood.__new__(Chi2TimeFrequencyLikelihood)

        # Mock data_context with missing filter
        likelihood.data_context = Mock()
        likelihood.data_context.whitened_frequency_domain_strain_array = np.array([[1 + 1j, 2 + 2j]])  # Strain exists
        likelihood.data_context.time_frequency_filter = None  # Missing filter

        # Test that _calculate_noise_log_likelihood raises ValueError
        with pytest.raises(ValueError, match="Time frequency filter is not available"):
            likelihood._calculate_noise_log_likelihood()
