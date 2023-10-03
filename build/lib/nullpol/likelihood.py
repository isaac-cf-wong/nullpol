import numpy as np
from bilby.core.likelihood import Likelihood
from bilby.gw.detector import InterferometerList


from .null_projector import get_null_projector


class NullStreamLikelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, projector_generator,
                 priors=None, analysis_domain="frequency",
                 reference_frame="sky", time_reference="geocenter"):
        self.interferometers = InterferometerList(interferometers)
        self.projector_generator = projector_generator
        self.priors = priors
        self._frequency_domain_strain_array = None

    def __repr__(self):
        return None

    @property
    def frequency_domain_strain_array(self):
        """The frequency domain strain array

        Returns
        =======
        array_like: The frequency domain strain array
        """
        if self._frequency_domain_strain_array is None:
            nifo = len(self.interferometers)
            nfreq = len(self.interferometers[0].frequency_array)
            self._frequency_domain_strain_array = np.zeros(nifo, nfreq, dtype=self.interferometers[0].frequency_domain_strain[0].dtype)
            for info in self.interferometers:
                self._frequency_domain_strain_array[info.name] = info.frequency_domain_strain
        return self._frequency_domain_strain_array

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        Pnull = self.projector_generator.projector(self.parameters)
        null_stream = get_null_stream(self.interferometers, Pnull)
        null_energy = get_null_energy(null_stream)
        log_likelihood = -0.5 * null_energy

        return null_energy
