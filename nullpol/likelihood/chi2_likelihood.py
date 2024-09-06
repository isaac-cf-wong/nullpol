import numpy as np
from bilby.core.likelihood import Likelihood
import scipy.stats
import numpy as np
from tqdm import tqdm
from nullpol.time_shift import time_shift
from nullpol.wdm.wavelet_transform import transform_wavelet_freq
from nullpol.filter import clustering, get_high_pass_filter
from nullpol.null_stream import get_null_stream, get_null_energy
from nullpol.detector.networks import *


class NullStreamLikelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self, interferometers, waveform_generator=None, projector_generator=None,
                 priors=None, analysis_domain="frequency", time_frequency_analysis_arguments={},
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

        if self.analysis_domain == "time_frequency":
            try:
                self.nx = time_frequency_analysis_arguments['nx']
                self.df = time_frequency_analysis_arguments['df']
            except KeyError:
                raise ValueError('nx and df must be provided for time-frequency analysis.')
            
            for ifo in self.interferometers:
                ifo.strain_data.nx = self.nx
                ifo.strain_data.time_frequency_bandwidth = self.df

            # transform_wavelet_freq(strain, Nf, Nt, nx) only takes 2D array as input, so we need to loop over the first two dimensions
            energy_map_max = np.zeros((self.interferometers[0].Nt, self.interferometers[0].Nf))
            for _ in tqdm(range(1000), desc='Generating energy map'):
                strain = time_shift(interferometers=self.interferometers,
                                     ra=self.priors['ra'].sample(),
                                     dec=self.priors['dec'].sample(),
                                     gps_time=self.interferometers[0].start_time+self.interferometers[0].duration, # take the end time of the interferometer as the reference time
                                     frequency_array=self.frequency_array,
                                     frequency_mask=self.frequency_mask
                                     ) # shape (n_interferometers, n_freqs)
                for j in range(len(self.interferometers)):
                    energy_map = transform_wavelet_freq(strain[j], self.interferometers[j].Nf, self.interferometers[j].Nt, nx=self.nx)
                    energy_map_max = np.fmax(energy_map_max, energy_map)
            self.energy_map_max = energy_map_max

            self.dt = self.interferometers[0].duration / self.interferometers[0].Nt
            self.time_frequency_filter = clustering(get_high_pass_filter(energy_map_max, **time_frequency_analysis_arguments), self.dt, **time_frequency_analysis_arguments)
            self._DoF = np.sum(self.time_frequency_filter) * dim

            self.log_likelihood = self.log_likelihood_time_freq

        elif self.analysis_domain == "frequency":
            self._DoF = int((self.maximum_frequency - self.minimum_frequency) * self.interferometers[0].duration) * 2 * dim
            self.log_likelihood = self.log_likelihood_freq

        else:
            raise ValueError('analysis_domain not recognized.')

    def __repr__(self):
        return None

    def log_likelihood_freq(self):
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
    
    def log_likelihood_time_freq(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters, self.interferometers, self.frequency_array, self.psd_array)
        null_stream_freq = get_null_stream(null_projector=null_projector,
                                      time_shifted_strain_data_array=time_shift(interferometers=self.interferometers,
                                                                                ra=self.parameters['ra'],
                                                                                dec=self.parameters['dec'],
                                                                                gps_time=self.parameters['geocent_time'],
                                                                                frequency_array = self.frequency_array,
                                                                                frequency_mask = self.frequency_mask
                                                                                )
                                        )
        null_stream_time_freq = transform_wavelet_freq(null_stream_freq, self.interferometers[0].Nf, self.interferometers[0].Nt, nx=self.nx)
        log_likelihood = scipy.stats.chi2.logpdf(2*np.sum(null_stream_time_freq[self.time_frequency_filter]), df=self._DoF)

        return log_likelihood
