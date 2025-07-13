from __future__ import annotations

import bilby
from bilby.core.utils import logger
from bilby.gw.detector import InterferometerList
from bilby.gw.source import lal_binary_black_hole

logger.setLevel('CRITICAL')
import numpy as np
import scipy.stats
from tqdm import tqdm

from nullpol.clustering import run_time_frequency_clustering
from nullpol.injection import create_injection
from nullpol.likelihood.chi2_time_frequency_likelihood import \
    Chi2TimeFrequencyLikelihood

seed = 12
# Set the seed
np.random.seed(seed)
bilby.core.utils.random.seed(seed)
duration = 8
sampling_frequency = 4096
geocent_time = 1126259642.413
start_time = geocent_time - 4
wavelet_frequency_resolution = 16
wavelet_nx = 4.
minimum_frequency = 20
maximum_frequency = sampling_frequency / 2
threshold = 1.
time_padding = 0.1
frequency_padding = 1
skypoints = 10
parameters = dict(
    mass_1=36.0,
    mass_2=29.0,
    a_1=0.,
    a_2=0.,
    tilt_1=0.,
    tilt_2=0.,
    phi_12=0.,
    phi_jl=0.,
    luminosity_distance=2000.0,
    theta_jn=0.,
    psi=2.659,
    phase=1.3,
    geocent_time=geocent_time,
    ra=1.375,
    dec=-1.2108,
)


def create_time_frequency_filter():
    interferometers = InterferometerList(['H1', 'L1', 'V1'])
    noise_type = 'zero_noise'
    freuency_domain_source_model = lal_binary_black_hole
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                reference_frequency=50)
    create_injection(interferometers=interferometers,
                        parameters=parameters,
                        duration=duration,
                        sampling_frequency=sampling_frequency,
                        start_time=start_time,
                        noise_type=noise_type,
                        frequency_domain_source_model=freuency_domain_source_model,
                        waveform_arguments=waveform_arguments)
    frequency_domain_strain_array = np.array([ifo.frequency_domain_strain.copy() for ifo in interferometers])
    time_frequency_filter, spectrogram = run_time_frequency_clustering(
        interferometers=interferometers,
        frequency_domain_strain_array=frequency_domain_strain_array,
        wavelet_frequency_resolution=wavelet_frequency_resolution,
        wavelet_nx=wavelet_nx,
        threshold=threshold,
        time_padding=time_padding,
        frequency_padding=frequency_padding,
        skypoints=skypoints,
        return_sky_maximized_spectrogram=True,
        threshold_type='variance')
    import matplotlib.pyplot as plt
    plt.imshow(time_frequency_filter, aspect='auto')
    plt.savefig('TF_filter.png')
    plt.imshow(spectrogram, aspect='auto')
    plt.savefig('spectrogram.png')
    return time_frequency_filter

time_frequency_filter = create_time_frequency_filter()


def test_noise_residual_energy():
    samples = []

    for i in tqdm(range(200), desc='test_noise_residual_energy'):
        # Create a noise injection
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        create_injection(interferometers=interferometers,
                         duration=duration,
                         sampling_frequency=sampling_frequency,
                         start_time=start_time,
                         noise_type='noise')
        polarization_modes = 'pc'
        polarization_basis = 'pc'
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter)
        likelihood.parameters = dict(ra=0,
                                     dec=0,
                                     psi=0,
                                     geocent_time=geocent_time)
        energy = likelihood._compute_residual_energy()
        samples.append(energy)

    result = scipy.stats.kstest(samples, cdf='chi2', args=(likelihood.DoF,))
    print(f"p-value = {result.pvalue}")
    assert result.pvalue >= 0.05


def test_signal_residual_energy():
    samples = []

    for i in tqdm(range(200), desc='test_signal_residual_energy'):
        # Create a noise injection
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        create_injection(interferometers=interferometers,
                            duration=duration,
                            sampling_frequency=sampling_frequency,
                            start_time=start_time,
                            parameters=parameters,
                            noise_type='gaussian')
        polarization_modes = 'pc'
        polarization_basis = 'pc'
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter)
        likelihood.parameters = dict(ra=parameters['ra'],
                                     dec=parameters['dec'],
                                     psi=parameters['psi'],
                                     geocent_time=parameters['geocent_time'])
        energy = likelihood._compute_residual_energy()
        samples.append(energy)
    result = scipy.stats.kstest(samples, cdf='chi2', args=(likelihood.DoF,))
    print(f"p-value = {result.pvalue}")
    assert result.pvalue >= 0.05


def test_signal_residual_energy_incorrect_parameters():
    samples = []

    for i in tqdm(range(200), desc='test_signal_residual_energy_incorrect_parameters'):
        # Create a noise injection
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        create_injection(interferometers=interferometers,
                            duration=duration,
                            sampling_frequency=sampling_frequency,
                            start_time=start_time,
                            parameters=parameters,
                            noise_type='gaussian')
        polarization_modes = 'pc'
        polarization_basis = 'pc'
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter)
        likelihood.parameters = dict(ra=parameters['ra']+0.5,
                                    dec=parameters['dec']-0.5,
                                    psi=parameters['psi']+0.5,
                                    geocent_time=parameters['geocent_time']+10000)
        energy = likelihood._compute_residual_energy()
        samples.append(energy)
    result = scipy.stats.kstest(samples, cdf='chi2', args=(likelihood.DoF,))
    print(f"p-value = {result.pvalue}")
    assert result.pvalue < 0.05


def test_signal_pc_c_residual_energy():
    samples = []
    freuency_domain_source_model = lal_binary_black_hole
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                reference_frequency=50)
    for i in tqdm(range(200), desc='test_signal_pc_p_residual_energy'):
        # Create a noise injection
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        create_injection(interferometers=interferometers,
                            duration=duration,
                            sampling_frequency=sampling_frequency,
                            start_time=start_time,
                            parameters=parameters,
                            noise_type='gaussian',
                            frequency_domain_source_model=freuency_domain_source_model,
                            waveform_arguments=waveform_arguments)
        polarization_modes = 'pc'
        polarization_basis = 'p'
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter)
        likelihood.parameters = dict(ra=parameters['ra'],
                                    dec=parameters['dec'],
                                    psi=parameters['psi'],
                                    geocent_time=parameters['geocent_time'],
                                    amplitude_cp=1.,
                                    phase_cp=-np.pi/2)
        energy = likelihood._compute_residual_energy()
        samples.append(energy)
    result = scipy.stats.kstest(samples, cdf='chi2', args=(likelihood.DoF,))
    print(f"p-value = {result.pvalue}")
    assert result.pvalue >= 0.05


def test_signal_pc_c_residual_energy_incorrect_parameters():
    samples = []
    freuency_domain_source_model = lal_binary_black_hole
    waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                reference_frequency=50)
    for i in tqdm(range(200), desc='test_signal_pc_p_residual_energy_incorrect_parameters'):
        # Create a noise injection
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        create_injection(interferometers=interferometers,
                            duration=duration,
                            sampling_frequency=sampling_frequency,
                            start_time=start_time,
                            parameters=parameters,
                            noise_type='gaussian',
                            frequency_domain_source_model=freuency_domain_source_model,
                            waveform_arguments=waveform_arguments)
        polarization_modes = 'pc'
        polarization_basis = 'p'
        likelihood = Chi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=time_frequency_filter)
        likelihood.parameters = dict(ra=parameters['ra'],
                                        dec=parameters['dec'],
                                        psi=parameters['psi'],
                                        geocent_time=parameters['geocent_time'],
                                        amplitude_cp=1.,
                                        phase_cp=np.pi/2)
        energy = likelihood._compute_residual_energy()
        samples.append(energy)
    result = scipy.stats.kstest(samples, cdf='chi2', args=(likelihood.DoF,))
    print(f"p-value = {result.pvalue}")
    assert result.pvalue < 0.05
