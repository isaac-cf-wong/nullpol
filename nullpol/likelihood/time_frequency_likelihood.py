import numpy as np
from bilby.core.likelihood import Likelihood
from tqdm import tqdm
from ..time_shift import time_shift
from ..time_frequency_transform import (transform_wavelet_freq,
                                        transform_wavelet_freq_quadrature)
from ..detector.networks import *
from ..clustering.threshold_filter import compute_filter_by_quantile
from ..clustering.single import clustering

class TimeFrequencyLikelihood(Likelihood):
    """Likelihood function evaluated in the time-frequency domain."""
    def __init__(self, interferometers, waveform_generator=None, projector_generator=None,
                 priors=None,                 
                 time_frequency_filter=None,
                 time_frequency_transform_arguments=None,
                 time_frequency_clustering_arguments=None,
                 reference_frame="sky", time_reference="geocenter", *args, **kwargs):
        self.interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)

        if projector_generator is not None:
            self.projector_generator = projector_generator
        else:
            self.projector_generator = waveform_generator

        self.priors = priors
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
        self.time_frequency_transform_arguments = time_frequency_transform_arguments
        self.time_frequency_clustering_arguments = time_frequency_clustering_arguments
        # Check if all interferometers have the same frequency array
        if not all([np.array_equal(interferometer.frequency_array, self.interferometers[0].frequency_array) for interferometer in self.interferometers[1:]]):
            raise ValueError('All interferometers must have the same frequency array for time-frequency analysis.')
        
        for ifo in self.interferometers:
            ifo.strain_data.nx = self.time_frequency_transform_arguments['nx']
            ifo.strain_data.time_frequency_bandwidth = self.time_frequency_transform_arguments['df']
        self.time_frequency_transform_arguments['Nf'] = self.interferometers[0].Nf
        self.time_frequency_transform_arguments['Nt'] = self.interferometers[0].Nt

        # Construct the time-frequency filter if it is not provided.
        if time_frequency_filter is None:
            # transform_wavelet_freq(strain, Nf, Nt, nx) only takes 2D array as input, so we need to loop over the first two dimensions
            energy_map_max = np.zeros((self.interferometers[0].Nt, self.interferometers[0].Nf))
            for _ in tqdm(range(1000), desc='Generating energy map'):
                strain_data_array = interferometers.whitened_frequency_domain_strain_array
                strain = time_shift(interferometers=self.interferometers,
                                        ra=self.priors['ra'].sample(),
                                        dec=self.priors['dec'].sample(),
                                        gps_time=self.interferometers[0].start_time+self.interferometers[0].duration, # take the end time of the interferometer as the reference time
                                        frequency_array=interferometers[0].frequency_array, # use the full frequency array
                                        strain_data_array=strain_data_array
                                        ) # shape (n_interferometers, n_freqs)
                for j in range(len(self.interferometers)):
                    energy_map = transform_wavelet_freq(strain[j], self.interferometers[j].Nf, self.interferometers[j].Nt, nx=self.time_frequency_transform_arguments['nx']) ** 2 + transform_wavelet_freq_quadrature(strain[j], self.interferometers[j].Nf, self.interferometers[j].Nt, nx=time_frequency_transform_arguments['nx']) ** 2
                    energy_map_max = np.fmax(energy_map_max, energy_map)
            self.energy_map_max = energy_map_max

            self.dt = self.interferometers[0].duration / self.interferometers[0].Nt
            time_frequency_filter = clustering(compute_filter_by_quantile(energy_map_max, **time_frequency_clustering_arguments), self.dt,
                                               **time_frequency_transform_arguments,
                                               **time_frequency_clustering_arguments)
        # Cleaning the time-frequency filter to remove components beyond the frequency range
        if self.minimum_frequency is not None:
            freq_low_idx = int(np.ceil(self.minimum_frequency / self.df))
            time_frequency_filter[:,:freq_low_idx] = 0.
        if self.maximum_frequency is not None:
            freq_high_idx = int(np.floor(self.maximum_frequency / self.df))
            time_frequency_filter[:,freq_high_idx:] = 0.
        self._time_frequency_filter = time_frequency_filter

    def log_likelihood(self):
        raise NotImplementedError('log_likelihood() should be implemented in the subclass.')