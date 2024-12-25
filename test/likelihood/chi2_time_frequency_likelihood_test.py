import bilby
from bilby.gw.detector import InterferometerList
import numpy as np
import unittest
from nullpol.likelihood.chi2_time_frequency_likelihood import Chi2TimeFrequencyLikelihood
from nullpol.null_stream.projector_generator import ProjectorGenerator

class TestChi2TimeFrequencyLikelihood(unittest.TestCase):
    def setUp(self):
        seed = 20240731
        # Set the seed
        np.random.seed(seed)
        bilby.core.utils.random.seed(seed)        
        self.wavelet_frequency_resolution = 16
        self.wavelet_nx = 4.
        priors = bilby.core.prior.PriorDict()
        # projector_generator = ProjectorGenerator(wave)
        # likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
        #                                          projector_generator=projector_generator)


    def create_time_frequency_filter(self):
        pass

    def test_noise_log_likelihood(self):
        # Create a noise injection
        interferometers = InterferometerList(['H1', 'L1', 'V1'])
        polarization_modes = 'pc'
        polarization_basis = 'pc'
        likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
                                                 wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                 wavelet_nx=self.wavelet_nx,
                                                 polarization_modes=polarization_modes,
                                                 polarization_basis=polarization_basis,
                                                 time_frequency_filter=self.time_frequency_filter)




if __name__ == '__main__':
    unittest.main()