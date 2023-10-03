import numpy as np
from bilby.core.likelihood import Likelihood

from .null_projector import get_null_projector
from .detector.networks import *


class NullStreamLikelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, projector_generator,
                 priors=None, analysis_domain="frequency",
                 reference_frame="sky", time_reference="geocenter"):
        self.interferometers = bilby.detector.networks.InterferometerList(interferometers)
        self.projector_generator = projector_generator
        self.priors = priors

    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        Pnull = self.projector_generator.projector(self.parameters)
        null_stream = get_null_stream(self.interferometers, Pnull)
        null_energy = get_null_energy(null_stream)
        # This is the Gaussian likelihood
        log_likelihood = -0.5 * null_energy
        # Should we always use the chi2 likelihood?

        return log_likelihood
