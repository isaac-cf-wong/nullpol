import numpy as np
import bilby
from bilby.core.likelihood import Likelihood
from ..detector import (compute_whitened_frequency_domain_strain_array,
                        compute_whitened_antenna_pattern_matrix_masked)
from ..null_stream import (encode_polarization,
                           get_antenna_pattern_matrix,
                           get_collapsed_antenna_pattern_matrix,
                           relative_amplification_factor_map,
                           relative_amplification_factor_helper,
                           compute_time_shifted_frequency_domain_strain_array,
                           estimate_frequency_domain_signal_at_geocenter)
from ..calibration import compute_calibrated_whitened_antenna_pattern_matrix
from ..time_frequency_transform import (get_shape_of_wavelet_transform,
                                        transform_wavelet_freq)


class TimeFrequencyLikelihood(Likelihood):
    """A time-frequency likelihood class.

    Args:
        interferometers (list): List of interferometers.
        wavelet_frequency_resolution (float): The frequency resolution of the wavelet transform.
        wavelet_nx (int): The number of points in the wavelet transform.
        polarization_modes (list): List of polarization modes.
        polarization_basis (list): List of polarization basis.
        time_frequency_filter (str): The time-frequency filter.
        priors (dict, optional): If given, used in the calibration marginalization.
    """
    def __init__(self,
                 interferometers,
                 wavelet_frequency_resolution,
                 wavelet_nx,
                 polarization_modes,
                 polarization_basis=None,
                 time_frequency_filter=None,
                 priors=None,
                 *args, **kwargs):
        super(TimeFrequencyLikelihood, self).__init__(dict())

        # Load the interferometers.
        self._interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)

        # Validate the interferometers
        self._validate_interferometers(self.interferometers)

        # Set up other attributes
        self._duration = self.interferometers[0].duration
        self._sampling_frequency = self.interferometers[0].sampling_frequency
        self._frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])
        self._frequency_array = self.interferometers[0].frequency_array.copy()
        self._frequency_mask = np.logical_and.reduce([ifo.frequency_mask for ifo in self.interferometers])
        self._masked_frequency_array = self._frequency_array[self._frequency_mask]
        self._filtered_frequency_mask = None
        self._filtered_masked_frequency_array = None
        self._frequency_resolution = self._frequency_array[1] - self._frequency_array[0]
        self._power_spectral_density_array = np.array([ifo.power_spectral_density_array for ifo in self.interferometers])        
        self._wavelet_frequency_resolution = wavelet_frequency_resolution
        self._wavelet_nx = wavelet_nx
        self._tf_Nt, self._tf_Nf = get_shape_of_wavelet_transform(self.duration,
                                                                  self.sampling_frequency,
                                                                  self.wavelet_frequency_resolution)
        
        # Encode the polarization labels
        self._polarization_modes, self._polarization_basis, self._polarization_derived = \
            encode_polarization(polarization_modes, polarization_basis)

        # Collapse the polarization encoding
        self._polarization_basis_collapsed = np.array([self.polarization_basis[i] for i in range(len(self.polarization_modes)) if self.polarization_modes[i]]).astype(bool)
        self._polarization_derived_collapsed = np.array([self.polarization_derived[i] for i in range(len(self.polarization_modes)) if self.polarization_modes[i]]).astype(bool)
        self._relative_amplification_factor_map = relative_amplification_factor_map(self.polarization_basis,
                                                                                   self.polarization_derived)

        # Load the time_frequency_filter
        self._time_frequency_filter = time_frequency_filter
        self._time_frequency_filter_collapsed = None

        # Validate the size of the time_frequency_filter
        if isinstance(self._time_frequency_filter, np.ndarray):
            self._validate_time_frequency_filter()

        # Compute the normalization constant        
        self._noise_log_likelihood_value = None               

        # Marginalization
        self._marginalized_parameters = []
        self.priors = priors

    @property
    def interferometers(self):
        """A list of interferometers.

        Returns:
            bilby.gw.detector.InterferometerList: A list of interferometers.
        """
        return self._interferometers

    @property
    def duration(self):
        """Duration of strain data in second.

        Returns:
            float: Duration of strain data in second.
        """
        return self._duration

    @property
    def sampling_frequency(self):
        """Sampling frequency of strain data.

        Returns:
            float: Sampling frequency of strain data in Hz.
        """
        return self._sampling_frequency

    @property
    def frequency_domain_strain_array(self):
        """An array of frequency domain strain of interferometers.

        Returns:
            numpy array: Frequency domain strain array (detector, frequency).
        """
        return self._frequency_domain_strain_array

    @property
    def frequency_array(self):
        """Frequency array.

        Returns:
            numpy array: Frequency array.
        """
        return self._frequency_array

    @property
    def frequency_mask(self):
        """Frequency mask.

        Returns:
            numpy array: A boolean array of frequency mask.
        """
        return self._frequency_mask

    @property
    def masked_frequency_array(self):
        """Masked frequency array.

        Returns:
            numpy array: Masked frequency array.
        """
        return self._masked_frequency_array

    @property
    def filtered_frequency_mask(self):
        if self._filtered_frequency_mask is None:
            self._filtered_frequency_mask = np.full_like(self.frequency_mask, False)
            # Check the minimum and maximum frequencies implied
            # by the time frequency filter.
            indices = np.where(self.time_frequency_filter_collapsed)[0]
            minimum_frequency = indices[0] * self.wavelet_frequency_resolution
            maximum_frequency = indices[-1] * self.wavelet_frequency_resolution
            k_low = int(minimum_frequency / self.frequency_resolution)
            k_high = int(maximum_frequency / self.frequency_resolution)
            self._filtered_frequency_mask[k_low:k_high+1] = True
        return self._filtered_frequency_mask

    @property
    def filtered_masked_frequency_array(self):
        if self._filtered_masked_frequency_array is None:
            self._filtered_masked_frequency_array = self.frequency_array[self.frequency_mask]
        return self._filtered_masked_frequency_array

    @property
    def frequency_resolution(self):
        """Frequency resolution.

        Returns:
            float: Frequency resolution in Hz.
        """
        return self._frequency_resolution

    @property
    def power_spectral_density_array(self):
        """Power spectral density array of interferometers.

        Returns:
            numpy array: Power spectral density array of interferometers.
        """
        return self._power_spectral_density_array

    @property
    def wavelet_frequency_resolution(self):
        """Frequency resolution in wavelet domain.

        Returns:
            float: Frequency resolution in Hz in wavelet domain.
        """
        return self._wavelet_frequency_resolution

    @property
    def wavelet_nx(self):
        """Steepness of the filter in wavelet transform.

        Returns:
            float: Steepness of the filter in wavelet domain.
        """
        return self._wavelet_nx

    @property
    def whitened_frequency_domain_strain_array(self):
        """Whitened frequency domain strain array of the interferometers.

        Returns:
            numpy array: Whitened frequency domain strain array (detector, frequency).
        """
        output = getattr(self, '_whitened_frequency_domain_strain_array', None)
        if output is None and \
                self.frequency_domain_strain_array is not None and \
                self.power_spectral_density_array is not None:
            self._whitened_frequency_domain_strain_array = \
                compute_whitened_frequency_domain_strain_array(
                    frequency_mask=self.frequency_mask,
                    frequency_resolution=self.frequency_resolution,
                    frequency_domain_strain_array=self.frequency_domain_strain_array,
                    power_spectral_density_array=self.power_spectral_density_array,
                )
            output = self._whitened_frequency_domain_strain_array
        return output

    @property
    def polarization_modes(self):
        """Polarization modes.

        Returns:
            numpy array: A boolean array that encodes the polarization modes.
        """
        return self._polarization_modes

    @property
    def polarization_basis(self):
        """Polarization basis.

        Returns:
            numpy array: A boolean array that encodes the polarization basis.
        """
        return self._polarization_basis

    @property
    def polarization_derived(self):
        """Derived polarization modes.

        Returns:
            numpy array: A boolean array that encodes the derived polarization modes.
        """
        return self._polarization_derived

    @property
    def polarization_basis_collapsed(self):
        """A collapsed boolean array of the polarization basis.
        The modes not in polarization_modes are removed.

        Returns:
            numpy array: A collapsed boolean array of the polarization basis.
        """
        return self._polarization_basis_collapsed

    @property
    def relative_amplification_factor_map(self):
        """A map to the relative amplification factor.

        Returns:
            numpy array: A map to the relative amplification factor (detector, mode).
        """
        return self._relative_amplification_factor_map

    @property
    def time_frequency_filter(self):
        """Time frequency filter.

        Returns:
            numpy array: Time frequency filter (time, frequency).
        """
        output = self._time_frequency_filter
        if isinstance(output, str):
            self._time_frequency_filter = np.load(output)
            output = self._time_frequency_filter
        return output

    @property
    def time_frequency_filter_collapsed(self):
        """A collapsed time frequency filter.

        Returns:
            numpy array: A collapsed time frequency filter (frequency).
        """
        output = self._time_frequency_filter_collapsed
        if output is None:
            self._time_frequency_filter_collapsed = np.any(self.time_frequency_filter, axis=0)
            output = self._time_frequency_filter_collapsed
        return output

    def _validate_interferometers(self, interferometers):
        """Validate interferometers.

        Args:
            interferometers (bilby.gw.detector.InterferometerList): A list of interferometers.

        Raises:
            ValueError: Throw an error if the interferometers do not have the same frequency resolution.
        """
        if not all([interferometer.frequency_array[1] - interferometer.frequency_array[0] == interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0] for interferometer in interferometers[1:]]):
            raise ValueError('All interferometers must have the same delta_f.')

    def _validate_time_frequency_filter(self):
        """Validate the time frequency filter.
        """
        # Get the shape of the time_frequency_filter        
        ntime, nfreq = self.time_frequency_filter.shape
        assert nfreq==self.tf_Nf, "The length of frequency axis in the wavelet domain does not match the time frequency filter."
        assert ntime==self.tf_Nt, "The length of time axis in the wavelet domain does not match the time frequency filter."

    @property
    def tf_Nt(self):
        """Number of time bins in the wavelet domain.

        Returns:
            int: Number of time bins in the wavelet domain.
        """
        return self._tf_Nt

    @property
    def tf_Nf(self):
        """Number of frequency bins in the wavelet domain.

        Returns:
            int: Number of frequency bins in the wavelet domain.
        """
        return self._tf_Nf

    def compute_time_delay_array(self):
        """Compute an array of time delays.

        Returns:
            numpy array: Time delay array.
        """
        return np.array([ifo.time_delay_from_geocenter(
            ra=self.parameters['ra'],
            dec=self.parameters['dec'],
            time=self.parameters['geocent_time']) for ifo in self.interferometers])

    def compute_antenna_pattern_matrix(self):
        """Compute the antenna pattern matrix.

        Returns:
            numpy array: Antenna pattern matrix.
        """
        # Evaluate the antenna pattern function
        F_matrix = get_antenna_pattern_matrix(
            self.interferometers,
            right_ascension=self.parameters['ra'],
            declination=self.parameters['dec'],
            polarization_angle=self.parameters['psi'],
            gps_time=self.parameters['geocent_time'],
            polarization=self.polarization_modes)
        # Evaluate the collapsed antenna pattern function
        # Compute the relative amplification factor
        if self.relative_amplification_factor_map.size > 0:
            relative_amplification_factor = relative_amplification_factor_helper(
                self.relative_amplification_factor_map,
                self.parameters)
            F_matrix = get_collapsed_antenna_pattern_matrix(
                F_matrix,
                self.polarization_basis_collapsed,
                self.polarization_derived_collapsed,
                relative_amplification_factor)
        return F_matrix

    def compute_calibration_factor(self):
        """Compute the calibration factor.

        Returns:
            numpy array: Calibration factor array.
        """
        output = np.zeros_like(self.frequency_domain_strain_array)

        for i in range(len(self.interferometers)):
            calibration_errors = self.interferometers[i].calibration_model.get_calibration_factor(
                frequency_array=self.masked_frequency_array,
                prefix=f'recalib_{self.interferometers[i].name}_',
                **self.parameters)
            output[i, self.frequency_mask] = calibration_errors
        return output

    def compute_whitened_frequency_domain_strain_array_at_geocenter(self):
        """Compute the whitened frequency domain strain array time shifted to geocenter.

        Returns:
            numpy array: Whitened frequency domain strain array time shifted at geocenter (detector, frequency).
        """
        time_delay_array = self.compute_time_delay_array()
        output = compute_time_shifted_frequency_domain_strain_array(
            frequency_array=self.frequency_array,
            frequency_mask=self.frequency_mask,
            frequency_domain_strain_array=self.whitened_frequency_domain_strain_array,
            time_delay_array=time_delay_array
        )
        self._cached_whitened_frequency_domain_strain_array_at_geocenter = output
        return output

    def compute_cached_wavelet_domain_strain_array_at_geocenter(self):
        """Compute the wavelet domain strain array at geocenter from the cached
        frequency domain array at geocenter.

        Returns:
            numpy array: Wavelet domain strain array at geocenter (detector, time, frequency).
        """
        return np.array([transform_wavelet_freq(
            data=self._cached_whitened_frequency_domain_strain_array_at_geocenter[i],
            sampling_frequency=self.sampling_frequency,
            frequency_resolution=self.wavelet_frequency_resolution,
            nx=self.wavelet_nx) for i in range(len(self.interferometers))]
        )

    def estimate_frequency_domain_signal_at_geocenter(self):
        """Estimate the frequency domain signal at geocenter.

        Returns:
            numpy array: Estimated frequency domain signal at geocenter (detector, frequency).
        """
        # Compute the antenna pattern matrix.
        antenna_pattern_matrix = self.compute_antenna_pattern_matrix()

        # Compute the whitened antenna pattern matrix.
        whitened_antenna_patten_matrix = compute_whitened_antenna_pattern_matrix_masked(
            antenna_pattern_matrix=antenna_pattern_matrix,
            psd_array=self.power_spectral_density_array,
            frequency_mask=self.frequency_mask)
        calibration_factor = self.compute_calibration_factor()
        calibrated_whitened_antenna_pattern_matrix = compute_calibrated_whitened_antenna_pattern_matrix(
            frequency_mask=self.frequency_mask,
            whitened_antenna_pattern_matrix=whitened_antenna_patten_matrix,
            calibration_error_matrix=calibration_factor
        )
        whitened_frequency_domain_strain_array_at_geocenter = self.compute_whitened_frequency_domain_strain_array_at_geocenter()
        return estimate_frequency_domain_signal_at_geocenter(
            frequency_mask=self.frequency_mask,
            whitened_frequency_domain_strain_array_at_geocenter=whitened_frequency_domain_strain_array_at_geocenter,
            whitened_antenna_pattern_matrix=calibrated_whitened_antenna_pattern_matrix)

    def estimate_wavelet_domain_signal_at_geocenter(self):
        """Estimate the wavelet domain signal at geocenter.

        Returns:
            numpy array: Estimated wavelet domain signal at geocenter (detector, time, frequency).
        """
        frequency_domain_signal = self.estimate_frequency_domain_signal_at_geocenter()

        # Perform the wavelet transform
        wavelet_domain_signal = np.array([transform_wavelet_freq(
            data=frequency_domain_signal[i],
            sampling_frequency=self.sampling_frequency,
            frequency_resolution=self.wavelet_frequency_resolution,
            nx=self.wavelet_nx) for i in range(len(self.interferometers))]
        )
        return wavelet_domain_signal

    def log_likelihood(self):
        """Log likelihood.

        Raises:
            NotImplementedError: This should be implemented in a subclass.
        """
        raise NotImplementedError("The log_likelihood method must be implemented in a subclass.")    

    def _calculate_noise_log_likelihood(self):
        return None

    def noise_log_likelihood(self):
        """
        Compute the noise log likelihood.

        Returns:
            float: The noise log likelihood.
        """
        if self._noise_log_likelihood_value is None:
            self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood_value
