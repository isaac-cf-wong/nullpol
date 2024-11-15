import bilby
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
        ifos = bilby.gw.detector.InterferometerList(['H1', 'L1', 'V1'])
        priors = bilby.core.prior.PriorDict()
        # projector_generator = ProjectorGenerator(wave)
        # likelihood = Chi2TimeFrequencyLikelihood(interferometers=interferometers,
        #                                          projector_generator=projector_generator)


    def test_log_likelihood(self):
        pass




if __name__ == '__main__':
    unittest.main()