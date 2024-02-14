import numpy as np
from bilby.core.likelihood import Likelihood
import scipy.stats
import numpy as np
from .null_projector import get_null_stream, get_null_energy
from .detector.networks import *


class NullStreamLikelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, projector_generator,
                 priors=None, analysis_domain="frequency",
                 reference_frame="sky", time_reference="geocenter", **kwargs):
        self.interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)
        self.projector_generator = projector_generator
        self.priors = priors
        self.analysis_domain = analysis_domain
        self._marginalized_parameters = dict()
        self.parameters = dict()
        dim = len(self.interferometers) - np.sum(self.projector_generator.basis)
        self._DoF = int((self.projector_generator.maximum_frequency - self.projector_generator.minimum_frequency) * self.interferometers[0].duration) * 2 * dim

    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters)
        null_stream = get_null_stream(interferometers=self.interferometers,
                                      null_projector=null_projector,
                                      ra=self.parameters['ra'],
                                      dec=self.parameters['dec'],
                                      gps_time=self.parameters['geocent_time'],
                                      minimum_frequency=self.projector_generator.minimum_frequency,
                                      maximum_frequency=self.projector_generator.maximum_frequency,
                                      )
        null_energy = get_null_energy(null_stream)
        log_likelihood = scipy.stats.chi2.logpdf(2*null_energy, df=self._DoF)

        return log_likelihood
