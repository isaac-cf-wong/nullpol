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
    def __init__(self, interferometers, waveform_generator=None, projector_generator=None,
                 priors=None, analysis_domain="frequency",
                 reference_frame="sky", time_reference="geocenter"):
        
        if projector_generator is not None:
            self.projector_generator = projector_generator
        else:
            self.projector_generator = waveform_generator

        super().__init__(interferometers[0]+interferometers[1], waveform_generator, projector_generator, priors, analysis_domain, reference_frame, time_reference)

        self.interferometers_1 = interferometers[0]
        self.interferometers_2 = interferometers[1]
        if self.interferometers_1.start_time > self.interferometers_2.start_time:
            raise ValueError('The start time of interferometers_1 must be earlier than the start time of interferometers_2.')
        self.psd_array_1 = np.array([np.interp(self.frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers_1])
        self.psd_array_2 = np.array([np.interp(self.frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers_2])

    def __repr__(self):
        return None

    def log_likelihood(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters, self.interferometers_1, self.interferometers_2, self.frequency_array, self.psd_array_1, self.psd_array_2, self.minimum_frequency, self.maximum_frequency)
        null_stream = get_lensing_null_stream(interferometers_1=self.interferometers_1,
                                              interferometers_2=self.interferometers_2,
                                              null_projector=null_projector,
                                              ra=self.parameters['ra'],
                                              dec=self.parameters['dec'],
                                              gps_time_1=self.parameters['geocent_time_1'],
                                              gps_time_2=self.parameters['geocent_time_2'],
                                              minimum_frequency=self.minimum_frequency,
                                              maximum_frequency=self.maximum_frequency,
                                              )
        null_energy = get_null_energy(null_stream)
        log_likelihood = scipy.stats.chi2.logpdf(2*null_energy, df=self._DoF)

        return log_likelihood
