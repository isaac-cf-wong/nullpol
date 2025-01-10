import numpy as np
from numba import njit
from scipy.special import logsumexp
from .time_frequency_likelihood import TimeFrequencyLikelihood
from ..null_stream import (time_shift,
                           compute_whitened_antenna_pattern_matrix_masked,
                           compute_gw_projector_masked,
                           compute_null_projector_from_gw_projector,
                           compute_projection_squared)
from ..time_frequency_transform import transform_wavelet_freq
from ..detector import compute_whitened_time_frequency_domain_strain_array
from ..detector import get_simulated_calibrated_wavelet_psd
from ..utility import NullpolError


class WeightedGaussianTimeFrequencyLikelihood(TimeFrequencyLikelihood):
    """A time-frequency likelihood class that calculates the weighted Gaussian likelihood.

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
                 wavelet_psd_array=None,
                 time_frequency_filter=None,
                 simulate_psd_nsample=1000,
                 calibration_marginalization=False,
                 calibration_lookup_table=None,
                 calibration_psd_lookup_table=None,
                 number_of_response_curves=1000,
                 starting_index=0,
                 priors=None,
                 regularization_constant=0.99,
                 *args, **kwargs):
        super(WeightedGaussianTimeFrequencyLikelihood, self).__init__(interferometers=interferometers,
                                                                      wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                                      wavelet_nx=wavelet_nx,
                                                                      polarization_modes=polarization_modes,
                                                                      polarization_basis=polarization_basis,
                                                                      wavelet_psd_array=wavelet_psd_array,
                                                                      time_frequency_filter=time_frequency_filter,
                                                                      simulate_psd_nsample=simulate_psd_nsample,
                                                                      calibration_marginalization=calibration_marginalization,
                                                                      calibration_lookup_table=calibration_lookup_table,
                                                                      calibration_psd_lookup_table=calibration_psd_lookup_table,
                                                                      number_of_response_curves=number_of_response_curves,
                                                                      starting_index=starting_index,
                                                                      priors=priors,
                                                                      regularization_constant=regularization_constant,
                                                                      *args, **kwargs)
        if not np.issubdtype(time_frequency_filter.dtype, np.floating):
            raise NullpolError(f"Data type of time_frequency_filter = {time_frequency_filter.dtype} is required to be floating point numbers.")
        if np.any(time_frequency_filter < 0.) or np.any(time_frequency_filter > 1.):
            raise NullpolError(f"Some entries of the time_frequency_filter are not in the range [0,1].")
        if regularization_constant < 0. or regularization_constant >= 1.:
            raise NullpolError(f"regularization_constant = {regularization_constant} is required to be in the range [0, 1).")        
        self.regularization_constant = regularization_constant
        self._log_normalization_constant_noise_tf = -np.log(2. * np.pi) * 0.5 * len(self.interferometers)
        self._log_normalization_constant_signal_tf = self._log_normalization_constant_noise_tf + np.log(1. - regularization_constant) * 0.5 * np.sum(self.polarization_basis)
        self._compute_noise_log_likelihood_time_frequency_map()
    
    def log_likelihood(self):
        joint_log_likelihood_array = self._calculate_joint_log_likelihood()
        if len(joint_log_likelihood_array) > 1:
            return logsumexp(joint_log_likelihood_array) - np.log(len(joint_log_likelihood_array))
        else:
            return joint_log_likelihood_array[0]
    
    def _calculate_joint_log_likelihood(self):
        # Time shift the data
        frequency_domain_strain_array_time_shifted = time_shift(interferometers=self.interferometers,
                                                                ra=self.parameters['ra'],
                                                                dec=self.parameters['dec'],
                                                                gps_time=self.parameters['geocent_time'],
                                                                strain_data_array=self.frequency_domain_strain_array)   
        # Transform the time-shifted data to the time-freuency domain
        time_frequency_domain_strain_array_time_shifted = np.array([transform_wavelet_freq(data,
                                                                                           self._wavelet_Nf,
                                                                                           self._wavelet_Nt,
                                                                                           self.wavelet_nx) for data in frequency_domain_strain_array_time_shifted])
        # Compute the F matrix
        F_matrix = self._compute_antenna_pattern_matrix()

        # Compute the null energy
        if self.calibration_marginalization:
            joint_log_likelihood_array = compute_joint_log_likelihood_array(time_frequency_domain_strain_array_time_shifted=time_frequency_domain_strain_array_time_shifted,
                                                                            psd_draw_array=self.psd_draws,
                                                                            F_matrix=F_matrix,
                                                                            time_frequency_filter=self.time_frequency_filter,
                                                                            time_frequency_filter_collapsed=self.time_frequency_filter_collapsed,
                                                                            regularization_constant=self.regularization_constant,
                                                                            noise_log_likelihood_time_frequency_map=self._noise_log_likelihood_time_frequency_map,
                                                                            log_normalization_constant_signal_tf=self._log_normalization_constant_signal_tf)
        elif self._sample_calibration_parameters:
            # Simulate the PSD.
            psd_array = np.array([get_simulated_calibrated_wavelet_psd(interferometer=ifo,
                                                                       parameters=self.parameters,
                                                                       wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                       nx=self.wavelet_nx,
                                                                       nsample=self.simulate_psd_nsample) for ifo in self.interferometers])
            joint_log_likelihood_array = np.array([compute_joint_log_likelihood(time_frequency_domain_strain_array_time_shifted,
                                                                                psd_array=psd_array,
                                                                                F_matrix=F_matrix,
                                                                                time_frequency_filter=self.time_frequency_filter,                                                                       
                                                                                time_frequency_filter_collapsed=self.time_frequency_filter_collapsed,
                                                                                regularization_constant=self.regularization_constant,
                                                                                noise_log_likelihood_time_frequency_map=self._noise_log_likelihood_time_frequency_map,
                                                                                log_normalization_constant_signal_tf=self._log_normalization_constant_signal_tf)])
        else:
            joint_log_likelihood_array = np.array([compute_joint_log_likelihood(time_frequency_domain_strain_array_time_shifted,
                                                                                psd_array=self.wavelet_psd_array,
                                                                                F_matrix=F_matrix,
                                                                                time_frequency_filter=self.time_frequency_filter,                                                                       
                                                                                time_frequency_filter_collapsed=self.time_frequency_filter_collapsed,
                                                                                regularization_constant=self.regularization_constant,
                                                                                noise_log_likelihood_time_frequency_map=self._noise_log_likelihood_time_frequency_map,
                                                                                log_normalization_constant_signal_tf=self._log_normalization_constant_signal_tf)])
        
        return joint_log_likelihood_array

    def _calculate_noise_log_likelihood(self):
        self._noise_log_likelihood_value = np.sum(self._noise_log_likelihood_time_frequency_map)

    def _compute_noise_log_likelihood_time_frequency_map(self):
        time_frequency_domain_strain_array = np.array([transform_wavelet_freq(data,
                                                                              self._wavelet_Nf,
                                                                              self._wavelet_Nt,
                                                                              self.wavelet_nx) for data in self.frequency_domain_strain_array])        
        time_frequency_domain_strain_array_whitened = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array,
                                                                                                          self.wavelet_psd_array,
                                                                                                          self.time_frequency_filter)
        energy = np.sum(np.abs(time_frequency_domain_strain_array_whitened)**2, axis=0)
        self._noise_log_likelihood_time_frequency_map = -0.5 * energy
        ntime, nfreq = energy.shape
        for i in range(ntime):
            for j in range(nfreq):
                if self.time_frequency_filter[i,j]:
                    self._noise_log_likelihood_time_frequency_map[i,j] += self._log_normalization_constant_noise_tf
        

@njit
def compute_joint_log_likelihood(time_frequency_domain_strain_array_time_shifted,
                                 psd_array,
                                 F_matrix,
                                 time_frequency_filter,
                                 time_frequency_filter_collapsed,
                                 regularization_constant,
                                 noise_log_likelihood_time_frequency_map,
                                 log_normalization_constant_signal_tf):
    ndet, ntime, nfreq = time_frequency_domain_strain_array_time_shifted.shape
    # Compute the whitened time-frequency domain strain array
    time_frequency_domain_strain_array_time_shifted_whitened = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array_time_shifted,
                                                                                                                    psd_array,
                                                                                                                    time_frequency_filter)
    # Compute the whitened F_matrix
    whitened_F_matrix = compute_whitened_antenna_pattern_matrix_masked(F_matrix,
                                                                        psd_array,
                                                                        time_frequency_filter_collapsed)
    # Compute the GW projector        
    Pgw = compute_gw_projector_masked(whitened_F_matrix, time_frequency_filter_collapsed) * regularization_constant
    # Compute the null projector
    Pnull = compute_null_projector_from_gw_projector(Pgw)
    # Compute the projection squared
    projection_squared = compute_projection_squared(time_frequency_domain_strain_array_time_shifted_whitened,
                                                    Pnull,
                                                    time_frequency_filter)
    # Compute the log likelihood p(d_{ij} | \theta, I = 1) conditioned on signal added \log p(I = 1)
    joint_log_likelihood = 0.
    for i in range(ntime):
        for j in range(nfreq):
            if time_frequency_filter[i,j]:
                log_likelihood_tf = -0.5 * projection_squared[i,j] + log_normalization_constant_signal_tf + np.log(time_frequency_filter[i,j])
                if time_frequency_filter[i,j] != 1.:
                    log_likelihood_tf_noise = noise_log_likelihood_time_frequency_map[i,j] + np.log(1. - time_frequency_filter[i,j])
                    # Add the noise joint log likelihood to the total log likelihood using the logexpsum trick.
                    if log_likelihood_tf >= log_likelihood_tf_noise:
                        log_likelihood_tf = log_likelihood_tf + np.log(1. + np.exp(log_likelihood_tf_noise - log_likelihood_tf))
                    else:
                        log_likelihood_tf = log_likelihood_tf_noise + np.log(1. + np.exp(log_likelihood_tf - log_likelihood_tf_noise))
                joint_log_likelihood += log_likelihood_tf                    
    return joint_log_likelihood

@njit
def compute_joint_log_likelihood_array(time_frequency_domain_strain_array_time_shifted,
                                       psd_draw_array,
                                       F_matrix,
                                       time_frequency_filter,
                                       time_frequency_filter_collapsed,
                                       regularization_constant,
                                       noise_log_likelihood_time_frequency_map,
                                       log_normalization_constant_signal_tf):
    _, psd_nsample, _ = psd_draw_array.shape
    joint_log_likelihood_array = np.zeros(psd_nsample)
    for i in range(psd_nsample):
        joint_log_likelihood_array[i] = compute_joint_log_likelihood(time_frequency_domain_strain_array_time_shifted=time_frequency_domain_strain_array_time_shifted,
                                                            psd_array=psd_draw_array[:,i,:],
                                                            F_matrix=F_matrix,
                                                            time_frequency_filter=time_frequency_filter,
                                                            time_frequency_filter_collapsed=time_frequency_filter_collapsed,
                                                            regularization_constant=regularization_constant,
                                                            noise_log_likelihood_time_frequency_map=noise_log_likelihood_time_frequency_map,
                                                            log_normalization_constant_signal_tf=log_normalization_constant_signal_tf)
    return joint_log_likelihood_array