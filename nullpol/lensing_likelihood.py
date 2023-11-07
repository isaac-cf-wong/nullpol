import numpy as np
import scipy.stats
import numpy as np
from .null_projector import get_null_energy
from .lensing_null_projector import get_lensing_null_stream
from .likelihood import NullStreamLikelihood
from .detector.networks import *


class LensingNullStreamLikelihood(NullStreamLikelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, projector_generator,
                 priors=None, analysis_domain="frequency",
                 reference_frame="sky", time_reference="geocenter"):
        super().__init__(interferometers[0]+interferometers[1], projector_generator, priors, analysis_domain, reference_frame, time_reference)
        self.interferometers_1 = interferometers[0]
        self.interferometers_2 = interferometers[1]
    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters)
        null_stream = get_lensing_null_stream(interferometers_1=self.interferometers_1,
                                              interferometers_2=self.interferometers_2,
                                              null_projector=null_projector,
                                              ra=self.parameters['ra'],
                                              dec=self.parameters['dec'],
                                              gps_time_1=self.parameters['geocent_time_1'],
                                              gps_time_2=self.parameters['geocent_time_2'],
                                              minimum_frequency=self.projector_generator.minimum_frequency,
                                              maximum_frequency=self.projector_generator.maximum_frequency,
                                              )
        null_energy = get_null_energy(null_stream)
        log_likelihood = scipy.stats.chi2.logpdf(2*null_energy, df=self._DoF)

        return log_likelihood
