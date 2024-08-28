import numpy as np
from bilby.core.likelihood import Likelihood
import scipy.stats
import numpy as np
from nullpol.time_shift import time_shift
from nullpol.null_stream import get_null_stream, get_null_energy
from nullpol.detector.networks import *


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

        self.minimum_frequency = np.max([interferometer.minimum_frequency for interferometer in interferometers])
        self.maximum_frequency = np.min([interferometer.maximum_frequency for interferometer in interferometers])

        dim = len(self.interferometers) - np.sum(self.projector_generator.basis)
        self._DoF = int((self.maximum_frequency - self.minimum_frequency) * self.interferometers[0].duration) * 2 * dim

        if len(self.interferometers) <= np.sum(self.projector_generator.basis):
            raise ValueError('Number of interferometers must be larger than the number of basis polarization modes.')
        
        self.frequency_array = self.interferometers[0].frequency_array
        self.frequency_mask = np.array([self.frequency_array >= self.minimum_frequency, self.frequency_array <= self.maximum_frequency]).all(axis=0)
        self.frequency_array = self.frequency_array[self.frequency_mask]

        if not all([interferometer.frequency_array[1] - interferometer.frequency_array[0] == self.interferometers[0].frequency_array[1] - self.interferometers[0].frequency_array[0] for interferometer in self.interferometers[1:]]):
            raise ValueError('All interferometers must have the same delta_f.')
        
        if not all([interferometer.sampling_frequency >= 2*self.maximum_frequency for interferometer in self.interferometers]):
            raise ValueError('maximum_frequency of all interferometers must be less than or equal to the Nyquist frequency.')

        self.psd_array = np.array([np.interp(self.frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers])

    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters, self.interferometers, self.frequency_array, self.psd_array)
        null_stream = get_null_stream(null_projector=null_projector,
                                      time_shifted_strain_data_array=time_shift(interferometers=self.interferometers,
                                                                                ra=self.parameters['ra'],
                                                                                dec=self.parameters['dec'],
                                                                                gps_time=self.parameters['geocent_time'],
                                                                                frequency_array = self.frequency_array,
                                                                                frequency_mask = self.frequency_mask
                                                                                )
                                        )
        null_energy = get_null_energy(null_stream)
        log_likelihood = scipy.stats.chi2.logpdf(2*null_energy, df=self._DoF)

        return log_likelihood
