import numpy as np
from bilby.core.likelihood import Likelihood
import scipy.stats
from pathlib import Path
from pycbc.types.frequencyseries import FrequencySeries
from ..null_stream import (encode_polarization,
                           time_shift,
                           get_antenna_pattern_matrix,
                           get_collapsed_antenna_pattern_matrix,
                           relative_amplification_factor_map,
                           relative_amplification_factor_helper,
                           whiten_antenna_pattern_matrix_masked,
                           get_gw_projector_masked,
                           get_null_projector_from_gw_projector,
                           compute_projection_squared)
from ..psd import simulate_psd_from_psd
from ..time_frequency_transform import transform_wavelet_freq
from nullpol.wdm.wavelet_transform import transform_wavelet_freq, transform_wavelet_freq_quadrature
from nullpol.filter import clustering, get_high_pass_filter
from nullpol.null_stream.null_stream import get_null_stream, get_null_energy
from nullpol.detector.networks import *


class NullStreamChi2Likelihood(Likelihood):
    """A null stream likelihood object

    """
    def __init__(self,
                 interferometers,                 
                 wavelet_frequency_resolution,
                 wavelet_nx,
                 polarization_modes,
                 polarization_basis=None,
                 wavelet_psds=None,
                 time_frequency_filter=None,
                 simulate_psd_nsample=100,
                 *args, **kwargs):
        super(NullStreamChi2Likelihood, self).__init__(dict())
        # Load the interferometers.
        self.interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)
        self.frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])        
        # Validate the interferometers
        self._validate_interferometers(self.interferometers)
        self.wavelet_frequency_resolution = wavelet_frequency_resolution
        self.wavelet_nx = wavelet_nx
        self.simulate_psd_nsample = simulate_psd_nsample
        self._wavelet_Nf = int(self.interferometers[0].sampling_frequency / 2 / self.wavelet_frequency_resolution)
        self._wavelet_Nt = int(len(self.interferometers[0].time_array) / self.wavelet_frequency_resolution)
        # Encode the polarization labels
        self.polarization_modes, self.polarization_basis, self.polarization_derived = encode_polarization(polarization_modes, polarization_basis)
        self.relative_amplification_factor_map = relative_amplification_factor_map(self.polarization_basis,
                                                                                   self.polarization_derived)
        # Load the time_frequency_filter.
        self.time_freuency_filter = self._load_time_frequency_filter(time_frequency_filter)
        # Collapse the time_frequency_filter
        self.time_frequency_filter_collapsed = np.any(self.time_frequency_filter, axis=0)
        # Validate the size of the time_frequency_filter.
        self._validate_time_frequency_filter()
        # Get the resolution matching PSDs
        if wavelet_psds is None:
            self.psd_array = self._get_resolution_matching_psd(self.interferometers)
            logger.warning('wavelet_psds is simulated. If the run is submitted to a scheduler that will restart the job periodically, the wavelet_psds will be different in every run if the RNG is not seeded.')
        else:
            self.psd_array = wavelet_psds
        

        # dim = len(self.interferometers) - np.sum(self.projector_generator.basis)

        # if len(self.interferometers) <= np.sum(self.projector_generator.basis):
        #     raise ValueError('Number of interferometers must be larger than the number of basis polarization modes.')

        # self.frequency_array = self.interferometers[0].frequency_array
        # self.frequency_mask = np.array([self.frequency_array >= self.minimum_frequency, self.frequency_array <= self.maximum_frequency]).all(axis=0)
        # self.frequency_array = self.frequency_array[self.frequency_mask]

        # self.psd_array = np.array([np.interp(self.frequency_array, interferometer.power_spectral_density.frequency_array, interferometer.power_spectral_density.psd_array) for interferometer in self.interferometers])

        # if self.analysis_domain == "time_frequency":
        #     try:
        #         self.nx = time_frequency_analysis_arguments['nx']
        #         self.df = time_frequency_analysis_arguments['df']
        #     except KeyError:
        #         raise ValueError('nx and df must be provided for time-frequency analysis.')

        #     # Check if all interferometers have the same frequency array
        #     if not all([np.array_equal(interferometer.frequency_array, self.interferometers[0].frequency_array) for interferometer in self.interferometers[1:]]):
        #         raise ValueError('All interferometers must have the same frequency array for time-frequency analysis.')

        #     for ifo in self.interferometers:
        #         ifo.strain_data.nx = self.nx
        #         ifo.strain_data.time_frequency_bandwidth = self.df

        #     # transform_wavelet_freq(strain, Nf, Nt, nx) only takes 2D array as input, so we need to loop over the first two dimensions
        #     energy_map_max = np.zeros((self.interferometers[0].Nt, self.interferometers[0].Nf))
        #     for _ in tqdm(range(1000), desc='Generating energy map'):
        #         strain_data_array = interferometers.whitened_frequency_domain_strain_array
        #         strain = time_shift(interferometers=self.interferometers,
        #                              ra=self.priors['ra'].sample(),
        #                              dec=self.priors['dec'].sample(),
        #                              gps_time=self.interferometers[0].start_time+self.interferometers[0].duration, # take the end time of the interferometer as the reference time
        #                              frequency_array=interferometers[0].frequency_array, # use the full frequency array
        #                              strain_data_array=strain_data_array
        #                              ) # shape (n_interferometers, n_freqs)
        #         for j in range(len(self.interferometers)):
        #             energy_map = transform_wavelet_freq(strain[j], self.interferometers[j].Nf, self.interferometers[j].Nt, nx=self.nx) ** 2 + transform_wavelet_freq_quadrature(strain[j], self.interferometers[j].Nf, self.interferometers[j].Nt, nx=self.nx) ** 2
        #             energy_map_max = np.fmax(energy_map_max, energy_map)
        #     self.energy_map_max = energy_map_max

        #     self.dt = self.interferometers[0].duration / self.interferometers[0].Nt
        #     self.time_frequency_filter = clustering(get_high_pass_filter(energy_map_max, **time_frequency_analysis_arguments), self.dt, **time_frequency_analysis_arguments)
        #     self._DoF = np.sum(self.time_frequency_filter) * dim

        #     self.log_likelihood = self.log_likelihood_time_freq

        # elif self.analysis_domain == "frequency":
        #     self._DoF = int((self.maximum_frequency - self.minimum_frequency) * self.interferometers[0].duration) * 2 * dim
        #     self.log_likelihood = self.log_likelihood_freq

        # else:
        #     raise ValueError('analysis_domain not recognized.')

    def _validate_interferometers(self, interferometers):
        if not all([interferometer.frequency_array[1] - interferometer.frequency_array[0] == interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0] for interferometer in interferometers[1:]]):
            raise ValueError('All interferometers must have the same delta_f.')

    def _load_time_frequency_filter(self, time_frequency_filter):
        if isinstance(time_frequency_filter, np.ndarray):
            return time_frequency_filter
        elif isinstance(time_frequency_filter, str):
            fname = Path(time_frequency_filter)
            if fname.suffix == ".npy":
                return np.load(fname)
            elif fname.is_dir():
                return np.load(fname / "time_frequency_filter.npy")
            else:
                raise ValueError(f"Unrecognized format of time_frequency_filter: {fname}")
        elif time_frequency_filter is None:
            return np.load(Path.cwd() / "time_frequency_filter.npy")
        else:
            raise ValueError(f"Unrecognized data type of time_freuency_filter: {time_frequency_filter}")

    def _validate_time_frequency_filter(self):
        # Get the shape of the time_frequency_filter
        ntime, nfreq = self.time_frequency_filter.shape
        assert nfreq==self._wavelet_Nf, "The length of frequency axis in the wavelet domain does not match the time frequency filter."
        assert ntime==self._wavelet_Nt, "The length of time axis in the wavelet domain does not match the time frequency filter."

    def _get_resolution_matching_psd(self, interferometers):
        psd_array = []
        for interferometer in interferometers:
            delta_f = interferometer.power_spectral_density.frequency_array[1]-interferometer.power_spectral_density.frequency_array[0]
            psd_pycbc = FrequencySeries(interferometer.power_spectral_density.psd_array, delta_f=delta_f)
            psd_array.append(simulate_psd_from_psd(psd_pycbc,interferometer.duration,interferometer.sampling_frequency,self.wavelet_frequency_resolution,self.simulate_psd_nsample,self.wavelet_nx))
        return np.array(psd_array)

    def log_likelihood(self):
        # Time shift the data
        frequency_domain_strain_array_time_shifted = time_shift(self.interferometers,
                                                                ra=self.parameters['ra'],
                                                                dec=self.parameters['dec'],
                                                                gps_time=self.parameters['geocent_time'],
                                                                strain_data_array=self.frequency_domain_strain_array)
        # Transform the time-shifted data to the time-freuency domain
        time_frequency_domain_strain_array_time_shifted = np.array([transform_wavelet_freq(data,
                                                                                           self._wavelet_Nf,
                                                                                           self._wavelet_Nt,
                                                                                           self.wavelet_nx) for data in frequency_domain_strain_array_time_shifted])
        # Compute the GW projector
        Pgw = self.compute_gw_projector()
        # Compute the null projector
        Pnull = get_null_projector_from_gw_projector(Pgw)
        # Compute the projection squared
        projection_squared = compute_projection_squared(time_frequency_domain_strain_array_time_shifted,
                                                        Pnull,
                                                        self.time_freuency_filter)
        logl = -0.5 * np.sum(projection_squared) + self.normalization_constant
        return logl

    def compute_gw_projector(self):
        # Evaluate the antenna pattern function
        F_matrix = get_antenna_pattern_matrix(self.interferometers,
                                              right_ascension=self.parameters['ra'],
                                              declination=self.parameters['dec'],
                                              polarization_angle=self.parameters['psi'],
                                              gps_time=self.parameters['geocent_time'],
                                              polarization=self.polarization_modes)
        # Evaluate the collapsed antenna pattern function
        # Compute the relative amplification factor
        if self.relative_amplification_factor_map is not None:
            relative_amplification_factor = relative_amplification_factor_helper(self.relative_amplification_factor_map,
                                                                                 self.parameters)
            F_matrix = get_collapsed_antenna_pattern_matrix(F_matrix,
                                                            self.polarization_basis,
                                                            self.polarization_derived,
                                                            relative_amplification_factor)
        # Compute the whitened F_matrix
        whitened_F_matrix = whiten_antenna_pattern_matrix_masked(F_matrix, self.psd_array, self.time_frequency_filter_collapsed)
        # Compute the GW projector
        Pgw = get_gw_projector_masked(whitened_F_matrix, self.time_frequency_filter_collapsed)
        return Pgw

    def log_likelihood_freq(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
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
        null_energy = get_null_energy(null_stream)
        log_likelihood = scipy.stats.chi2.logpdf(2*null_energy, df=self._DoF)

        return log_likelihood

    def reconstruct_polarizations(self):
        pass

    def log_likelihood_time_freq(self):
        """The log likelihood function

        Returns
        -------
        float: The log likelihood value
        """
        null_projector = self.projector_generator.null_projector(self.parameters, self.interferometers, self.frequency_array, self.psd_array)
        strain_data_array = self.interferometers.whitened_frequency_domain_strain_array[:, self.frequency_mask]
        null_stream_freq = get_null_stream(null_projector=null_projector,
                                      time_shifted_strain_data_array=time_shift(interferometers=self.interferometers,
                                                                                ra=self.parameters['ra'],
                                                                                dec=self.parameters['dec'],
                                                                                gps_time=self.parameters['geocent_time'],
                                                                                frequency_array = self.frequency_array,
                                                                                strain_data_array = strain_data_array
                                                                                )
                                        )
        null_stream_time_freq = transform_wavelet_freq(null_stream_freq, self.interferometers[0].Nf, self.interferometers[0].Nt, nx=self.nx)
        log_likelihood = scipy.stats.chi2.logpdf(2*np.sum(null_stream_time_freq[self.time_frequency_filter]), df=self._DoF)

        return log_likelihood
