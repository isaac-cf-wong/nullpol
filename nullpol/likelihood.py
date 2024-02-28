import numpy as np
from bilby.core.likelihood import Likelihood
import scipy.stats
import numpy as np
from .null_projector import get_null_stream, get_null_energy
from .detector.networks import *


class NullStreamLikelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, waveform_generator=None, projector_generator=None,
                 priors=None, analysis_domain="frequency",
                 reference_frame="sky", time_reference="geocenter", **kwargs):
        
        self.interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)

        if projector_generator is not None:
            self.projector_generator = projector_generator
        else:
            self.projector_generator = waveform_generator

        self.priors = priors
        self.analysis_domain = analysis_domain
        self._marginalized_parameters = dict()
        self.parameters = dict()

        dim = len(self.interferometers) - np.sum(self.projector_generator.basis)
        self._DoF = int((self.projector_generator.maximum_frequency - self.projector_generator.minimum_frequency) * self.interferometers[0].duration) * 2 * dim

        self.psd_array = np.array([np.interp(interferometers[0].frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers])

        if len(self.interferometers) <= np.sum(self.projector_generator.basis):
            raise ValueError('Number of interferometers must be larger than the number of basis polarization modes.')
        delta_f = self.interferometers[0].frequency_array[1] - self.interferometers[0].frequency_array[0]
        if not all([interferometer.frequency_array[1] - interferometer.frequency_array[0] == delta_f for interferometer in self.interferometers[1:]]):
            raise ValueError('All interferometers must have the same delta_f.')
        self.frequency_array = self.interferometers[0].frequency_array
        if not all([interferometer.frequency_array[0] <= self.projector_generator.minimum_frequency for interferometer in self.interferometers]):
            raise ValueError('minimum_frequency must be greater than or equal to the minimum frequency of all interferometers.')
        if not all([interferometer.frequency_array[-1] >= self.projector_generator.maximum_frequency for interferometer in self.interferometers]):
            raise ValueError('maximum_frequency must be less than or equal to the maximum frequency of all interferometers.')
        # check if maximum_frequency is less than the Nyquist frequency
        if not all([interferometer.sampling_frequency >= 2*self.projector_generator.maximum_frequency for interferometer in self.interferometers]):
            raise ValueError('maximum_frequency must be less than or equal to the Nyquist frequency of all interferometers.')

    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters, self.interferometers, self.frequency_array, self.psd_array)
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
