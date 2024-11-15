import numpy as np
import scipy.stats
from .time_frequency_likelihood import TimeFrequencyLikelihood
from .null_stream import (get_null_stream,
                          time_shift)
from .time_frequency_transform import transform_wavelet_freq

class Chi2TimeFrequencyLikelihood(TimeFrequencyLikelihood):
    def __init__(self, interferometers, waveform_generator=None, projector_generator=None,
                 priors=None,                 
                 time_frequency_filter=None,
                 time_frequency_transform_arguments=None,
                 time_frequency_clustering_arguments=None,
                 reference_frame="sky", time_reference="geocenter", *args, **kwargs):
        super().__init__(interferometers=interferometers,
                         waveform_generator=waveform_generator,
                         projector_generator=projector_generator,
                         priors=priors,
                         time_frequency_filter=time_frequency_filter,
                         time_frequency_transform_arguments=time_frequency_transform_arguments,
                         time_frequency_clustering_arguments=time_frequency_clustering_arguments,
                         reference_frame=reference_frame,
                         time_reference=time_reference,
                         *args, **kwargs)
        self._DoF = (len(self.interferometers)-np.sum(self.projector_generator.basis)) * np.sum(self._time_frequency_filter)

    def log_likelihood(self):
        null_projector = self.projector_generator.null_projector(self.parameters, self.interferometers, self.frequency_array, self.psd_array)
        strain_data_array = self.interferometers.whitened_frequency_domain_strain_array[:, self.frequency_mask]
        null_stream = get_null_stream(null_projector=null_projector,
                                      time_shifted_strain_data_array=time_shift(interferometers=self.interferometers,
                                                                                ra=self.parameters['ra'],
                                                                                dec=self.parameters['dec'],
                                                                                gps_time=self.parameters['geocent_time'],
                                                                                frequency_array = self.frequency_array,
                                                                                strain_data_array = strain_data_array
                                                                                )
                                     )
        time_frequency_null_stream = np.array([transform_wavelet_freq(data,
                                                                      self.time_frequency_transform_arguments['Nf'],
                                                                      self.time_frequency_transform_arguments['Nt'],
                                                                      self.time_frequency_transform_arguments['nx']) for data in null_stream])
        # Apply the time-frequency filter
        filtered_projected_time_frequency_strain_data = time_frequency_null_stream[:,self._time_frequency_filter]
        null_energy = np.sum(np.abs(filtered_projected_time_frequency_strain_data) ** 2)
        log_likelihood = scipy.stats.chi2.logpdf(null_energy, df=self._DoF)
        return log_likelihood
    
    def noise_log_likelihood(self):
        strain_data_array = self.interferometers.whitened_frequency_domain_strain_array[:, self.frequency_mask]
        time_frequency_strain_data = np.array([transform_wavelet_freq(data,
                                                                      self.time_frequency_transform_arguments['Nf'],
                                                                      self.time_frequency_transform_arguments['Nt'],
                                                                      self.time_frequency_transform_arguments['nx']) for data in strain_data_array])
        filtered_time_frequency_strain_data = time_frequency_strain_data[:,self._time_frequency_filter]
        null_energy = np.sum(np.abs(filtered_time_frequency_strain_data) ** 2)
        log_likelihood = scipy.stats.chi2.logpdf(null_energy, df=len(self.interferometers)*np.sum(self._time_frequency_filter))
        return log_likelihood
