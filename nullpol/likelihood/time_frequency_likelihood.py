import numpy as np
import bilby
from bilby.core.likelihood import Likelihood
from bilby.core.prior import DeltaFunction
from bilby.gw.detector import Recalibrate
from ..null_stream import (encode_polarization,
                           get_antenna_pattern_matrix,
                           get_collapsed_antenna_pattern_matrix,
                           relative_amplification_factor_map,
                           relative_amplification_factor_helper,
                           compute_whitened_antenna_pattern_matrix_masked,
                           compute_whitened_time_frequency_domain_strain_array,
                           compute_gw_projector_masked)
from ..time_frequency_transform import (transform_wavelet_freq,
                                        get_shape_of_wavelet_transform)
from ..calibration import build_calibration_lookup
from ..detector import simulate_wavelet_psd
from ..utility import logger


class TimeFrequencyLikelihood(Likelihood):
    """A time-frequency likelihood class.

    Parameters
    ----------
    interferometers: list
        List of interferometers.
    wavelet_frequency_resolution: float
        The frequency resolution of the wavelet transform.
    wavelet_nx: int
        The number of points in the wavelet transform.
    polarization_modes: list
        List of polarization modes.
    polarization_basis: list
        List of polarization basis.
    time_frequency_filter: str
        The time-frequency filter.
    simulate_psd_nsample: int
        The number of samples to simulate the PSDs.
    calibration_marginalization: bool, optional
        If true, marginalize over calibration response curves in the likelihood.
        This is done numerically over a number of calibration response curve realizations.
    calibration_lookup_table: dict, optional
        If a dict, contains the arrays over which to marginalize for each interferometer or the filepaths of the
        calibration files.
        If not provided, but calibration_marginalization is used, then the appropriate file is created to
        contain the curves.
    calibration_psd_lookup_table: dict, optional
        If a dict, contains the arrays over which to marginalize for each interferometer or the filepaths of the
        calibration PSD files.
        If not provided, but calibration_marginalization is used, then the appropriate file is created to
        contain the curves.
    number_of_response_curves: int, optional
        Number of curves from the calibration lookup table to use.
        Default is 1000.
    starting_index: int, optional
        Sets the index for the first realization of the calibration curve to be considered.
        This, coupled with number_of_response_curves, allows for restricting the set of curves used. This can be used
        when dealing with large frequency arrays to split the calculation into sections.
        Defaults to 0.
    priors: dict, optional            
        If given, used in the calibration marginalization.
        Warning: when using marginalisation the dict is overwritten which will change the
        the dict you are passing in. If this behaviour is undesired, pass `priors.copy()`.
    """
    def __init__(self,
                 interferometers,                 
                 wavelet_frequency_resolution,
                 wavelet_nx,
                 polarization_modes,
                 polarization_basis=None,
                 time_frequency_filter=None,
                 simulate_psd_nsample=100,
                 calibration_marginalization=False,
                 calibration_lookup_table=None,
                 calibration_psd_lookup_table=None,
                 number_of_response_curves=1000,
                 starting_index=0,
                 priors=None,
                 *args, **kwargs):
        super(TimeFrequencyLikelihood, self).__init__(dict())
        # Load the interferometers.
        self.interferometers = bilby.gw.detector.networks.InterferometerList(interferometers)
        self.frequency_domain_strain_array = np.array([ifo.frequency_domain_strain for ifo in self.interferometers])        
        # Validate the interferometers
        self._validate_interferometers(self.interferometers)
        self.wavelet_frequency_resolution = wavelet_frequency_resolution
        self.wavelet_nx = wavelet_nx
        self.simulate_psd_nsample = simulate_psd_nsample
        self._wavelet_Nt, self._wavelet_Nf = get_shape_of_wavelet_transform(self.interferometers[0].duration,
                                                                            self.interferometers[0].sampling_frequency,
                                                                            self.wavelet_frequency_resolution)
        # Encode the polarization labels
        self.polarization_modes, self.polarization_basis, self.polarization_derived = encode_polarization(polarization_modes, polarization_basis)
        self.relative_amplification_factor_map = relative_amplification_factor_map(self.polarization_basis,
                                                                                   self.polarization_derived)
        # Load the time_frequency_filter
        self.time_frequency_filter = time_frequency_filter
        self._time_frequency_filter_collapsed = None
        # Validate the size of the time_frequency_filter
        if isinstance(self._time_frequency_filter, np.ndarray):
            self._validate_time_frequency_filter()
        # Compute the normalization constant        
        self._noise_log_likelihood_value = None               
        # Marginalization
        self._marginalized_parameters = []
        self.calibration_marginalization = calibration_marginalization
        self.priors = priors
        if self.calibration_marginalization:
            logger.warning('Calibration marginalization is not tested.')
            self.number_of_response_curves = number_of_response_curves
            self.starting_index = starting_index
            self._setup_calibration_marginalization(calibration_lookup_table, calibration_psd_lookup_table, priors)
            self._marginalized_parameters.append('recalib_index')
            self._sample_calibration_parameters = False
        elif np.all([isinstance(ifo.calibration_model, Recalibrate) for ifo in self.interferometers]):
            # Simulate the TF domain PSDs
            self._sample_calibration_parameters = False
            self._simulate_psd()
        else:
            self._sample_calibration_parameters = True
        
    def _setup_calibration_marginalization(self,
                                           calibration_lookup_table,
                                           calibration_psd_lookup_table,
                                           priors=None):
        self.calibration_draws, self.calibration_parameter_draws, self.psd_draws = build_calibration_lookup(
            interferometers=self.interferometers,
            lookup_files=calibration_lookup_table,
            psd_lookup_files=calibration_psd_lookup_table,
            priors=priors,
            number_of_response_curves=self.number_of_response_curves,
            starting_index=self.starting_index,
        )
        # Reshape the calibration psd draws into an array of shape (detector, nsample, nfreq)
        self.psd_draws = np.array([self.calibration_psd_draws[ifo.name] for ifo in self.interferometers])
        for name, parameters in self.calibration_parameter_draws.items():
            if parameters is not None:
                for key in set(parameters.keys()).intersection(priors.keys()):
                    priors[key] = DeltaFunction(0.0)
        self.calibration_abs_draws = dict()
        for name in self.calibration_draws:
            self.calibration_abs_draws[name] = np.abs(self.calibration_draws[name])**2        

    def _simulate_psd(self):
        """
        Simulate the PSDs from the PSDs of the interferometers.
        """
        self.psd_array = self._get_resolution_matching_psd(self.interferometers)

    def _validate_interferometers(self, interferometers):
        if not all([interferometer.frequency_array[1] - interferometer.frequency_array[0] == interferometers[0].frequency_array[1] - interferometers[0].frequency_array[0] for interferometer in interferometers[1:]]):
            raise ValueError('All interferometers must have the same delta_f.')

    @property
    def time_frequency_filter(self):
        output = getattr(self, '_time_frequency_filter', None)
        if isinstance(output, str):
            self._time_frequency_filter = np.load(self._time_frequency_filter)
            return self._time_frequency_filter
        return output
    
    @time_frequency_filter.setter
    def time_frequency_filter(self, time_frequency_filter):
        self._time_frequency_filter = time_frequency_filter        

    @property
    def time_frequency_filter_collapsed(self):
        output = getattr(self, "_time_frequency_filter_collapsed", None)
        if output is None:
            self._time_frequency_filter_collapsed = np.any(self.time_frequency_filter, axis=0)            
            return self._time_frequency_filter_collapsed
        return output

    def _validate_time_frequency_filter(self):
        # Get the shape of the time_frequency_filter        
        ntime, nfreq = self.time_frequency_filter.shape
        assert nfreq==self._wavelet_Nf, "The length of frequency axis in the wavelet domain does not match the time frequency filter."
        assert ntime==self._wavelet_Nt, "The length of time axis in the wavelet domain does not match the time frequency filter."

    def _get_resolution_matching_psd(self, interferometers):
        psd_array = []
        for interferometer in interferometers:
            psd_array.append(simulate_wavelet_psd(interferometer=interferometer,
                                                  wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                  nx=self.wavelet_nx,
                                                  nsample=self.simulate_psd_nsample))
        return np.array(psd_array)

    def log_likelihood(self):
        """Log likelihood function.

        Raises
        ------
        NotImplementedError:
            The log_likelihood method must be implemented in a subclass.
        """
        raise NotImplementedError("The log_likelihood method must be implemented in a subclass.")

    def _compute_antenna_pattern_matrix(self):
        # Evaluate the antenna pattern function
        F_matrix = get_antenna_pattern_matrix(self.interferometers,
                                              right_ascension=self.parameters['ra'],
                                              declination=self.parameters['dec'],
                                              polarization_angle=self.parameters['psi'],
                                              gps_time=self.parameters['geocent_time'],
                                              polarization=self.polarization_modes)
        # Evaluate the collapsed antenna pattern function
        # Compute the relative amplification factor
        if self.relative_amplification_factor_map.size > 0:
            relative_amplification_factor = relative_amplification_factor_helper(self.relative_amplification_factor_map,
                                                                                 self.parameters)
            F_matrix = get_collapsed_antenna_pattern_matrix(F_matrix,
                                                            self.polarization_basis,
                                                            self.polarization_derived,
                                                            relative_amplification_factor)
        return F_matrix

    def _calculate_noise_log_likelihood(self):
        return None

    def noise_log_likelihood(self):
        """
        Compute the noise log likelihood.

        Returns
        -------
        float:
            The noise log likelihood.
        """
        if self._noise_log_likelihood_value is None:            
            self._calculate_noise_log_likelihood()
        return self._noise_log_likelihood_value