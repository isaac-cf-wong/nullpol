import bilby
from bilby.gw.detector import InterferometerList
from bilby.gw.source import lal_binary_black_hole
import numpy as np
from tqdm import tqdm
import scipy.stats
import unittest
from nullpol.injection import create_injection
from nullpol.clustering import run_time_frequency_clustering
from nullpol.likelihood.chi2_time_frequency_likelihood import Chi2TimeFrequencyLikelihood


class TestChi2TimeFrequencyLikelihood(unittest.TestCase):
    def setUp(self):
        seed = 5
        # Set the seed
        np.random.seed(seed)
        bilby.core.utils.random.seed(seed)      
        self.duration = 8
        self.sampling_frequency = 4096
        self.geocent_time = 1126259642.413
        self.start_time = self.geocent_time - 4
        self.wavelet_frequency_resolution = 16
        self.wavelet_nx = 4.
        self.minimum_frequency = 20
        self.maximum_frequency = self.sampling_frequency / 2
        self.threshold = 0.9
        self.time_padding = 0.1
        self.frequency_padding = 1
        self.skypoints = 10
        self.create_time_frequency_filter()
        priors = bilby.core.prior.PriorDict()
        self.parameters = dict(
            mass_1=36.0,
            mass_2=29.0,
            a_1=0.4,
            a_2=0.3,
            tilt_1=0.5,
            tilt_2=1.0,
            phi_12=1.7,
            phi_jl=0.3,
            luminosity_distance=2000.0,
            theta_jn=0.4,
            psi=2.659,
            phase=1.3,
            geocent_time=self.geocent_time,
            ra=1.375,
            dec=-1.2108,
        )
        # projector_generator = ProjectorGenerator(wave)
        # likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
        #                                          projector_generator=projector_generator)

    def create_time_frequency_filter(self):
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        noise_type = 'zero_noise'
        freuency_domain_source_model = lal_binary_black_hole
        waveform_arguments = dict(waveform_approximant='IMRPhenomPv2',
                                  reference_frequency=50)
        create_injection(interferometers=interferometers,
                         parameters=self.parameters,
                         duration=self.duration,
                         sampling_frequency=self.sampling_frequency,
                         start_time=self.start_time,
                         noise_type=noise_type,
                         frequency_domain_source_model=freuency_domain_source_model,
                         waveform_arguments=waveform_arguments)
        frequency_domain_strain_array = np.array([ifo.frequency_domain_strain.copy() for ifo in interferometers])        
        self.time_frequency_filter = run_time_frequency_clustering(interferometers=interferometers,
                                                                   frequency_domain_strain_array=frequency_domain_strain_array,
                                                                   wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                   wavelet_nx=self.wavelet_nx,
                                                                   minimum_frequency=self.minimum_frequency,
                                                                   maximum_frequency=self.maximum_frequency,
                                                                   threshold=self.threshold,
                                                                   time_padding=self.time_padding,
                                                                   frequency_padding=self.frequency_padding,
                                                                   skypoints=self.skypoints)        

    def test_noise_residual_power(self):
        samples = []
        for i in tqdm(200):
            print(f'Progress: {i+1}/200')
            # Create a noise injection
            interferometers = InterferometerList(['H1', 'L1', 'V1'])
            create_injection(interferometers=interferometers,
                            duration=self.duration,
                            sampling_frequency=self.sampling_frequency,
                            start_time=self.start_time,
                            parameters=self.parameters,
                            noise_type='noise')
            polarization_modes = 'pc'
            polarization_basis = 'pc'
            likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
                                                    wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                    wavelet_nx=self.wavelet_nx,
                                                    polarization_modes=polarization_modes,
                                                    polarization_basis=polarization_basis,
                                                    time_frequency_filter=self.time_frequency_filter,
                                                    simulate_psd_nsample=100)
            likelihood.parameters = dict(ra=0,
                                        dec=0,
                                        psi=0,
                                        geocent_time=self.geocent_time)
            power = likelihood._calculate_residual_power()
            samples.append(power[0])
        result = scipy.stats.kstest(samples, cdf='chi2', args=(np.sum(self.time_frequency_filter),))
        self.assertGreaterEqual(result.pvalue, 0.05)

    def test_signal_residual_power(self):
        samples = []
        for i in tqdm(200):
            print(f'Progress: {i+1}/200')
            # Create a noise injection
            interferometers = InterferometerList(['H1', 'L1', 'V1'])
            create_injection(interferometers=interferometers,
                            duration=self.duration,
                            sampling_frequency=self.sampling_frequency,
                            start_time=self.start_time,
                            noise_type='gaussian')
            polarization_modes = 'pc'
            polarization_basis = 'pc'
            likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
                                                    wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                    wavelet_nx=self.wavelet_nx,
                                                    polarization_modes=polarization_modes,
                                                    polarization_basis=polarization_basis,
                                                    time_frequency_filter=self.time_frequency_filter,
                                                    simulate_psd_nsample=100)
            likelihood.parameters = dict(ra=self.parameters['ra'],
                                        dec=self.parameters['dec'],
                                        psi=self.parameters['psi'],
                                        geocent_time=self.parameters['geocent_time'])
            power = likelihood._calculate_residual_power()
            samples.append(power[0])
        result = scipy.stats.kstest(samples, cdf='chi2', args=(np.sum(self.time_frequency_filter),))
        self.assertGreaterEqual(result.pvalue, 0.05)

    def test_signal_residual_power_incorrect_parameters(self):
        samples = []
        for i in tqdm(200):
            print(f'Progress: {i+1}/200')
            # Create a noise injection
            interferometers = InterferometerList(['H1', 'L1', 'V1'])
            create_injection(interferometers=interferometers,
                            duration=self.duration,
                            sampling_frequency=self.sampling_frequency,
                            start_time=self.start_time,
                            noise_type='gaussian')
            polarization_modes = 'pc'
            polarization_basis = 'pc'
            likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
                                                    wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                    wavelet_nx=self.wavelet_nx,
                                                    polarization_modes=polarization_modes,
                                                    polarization_basis=polarization_basis,
                                                    time_frequency_filter=self.time_frequency_filter,
                                                    simulate_psd_nsample=100)
            likelihood.parameters = dict(ra=self.parameters['ra']+0.5,
                                        dec=self.parameters['dec']-0.5,
                                        psi=self.parameters['psi']+0.5,
                                        geocent_time=self.parameters['geocent_time']+10000)
            power = likelihood._calculate_residual_power()
            samples.append(power[0])
        result = scipy.stats.kstest(samples, cdf='chi2', args=(np.sum(self.time_frequency_filter),))
        self.assertLess(result.pvalue, 0.05)

if __name__ == '__main__':
    unittest.main()