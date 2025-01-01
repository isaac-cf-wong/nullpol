import numpy as np
import scipy.stats
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
from ..detector import (get_simulated_calibrated_wavelet_psd,
                        simulate_wavelet_psd)

class Chi2TimeFrequencyLikelihood(TimeFrequencyLikelihood):
    """A time-frequency likelihood class that calculates the chi-squared likelihood.

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
        super(Chi2TimeFrequencyLikelihood, self).__init__(interferometers=interferometers,
                                                          wavelet_frequency_resolution=wavelet_frequency_resolution,
                                                          wavelet_nx=wavelet_nx,
                                                          polarization_modes=polarization_modes,
                                                          polarization_basis=polarization_basis,
                                                          time_frequency_filter=time_frequency_filter,
                                                          simulate_psd_nsample=simulate_psd_nsample,
                                                          calibration_marginalization=calibration_marginalization,
                                                          calibration_lookup_table=calibration_lookup_table,
                                                          calibration_psd_lookup_table=calibration_psd_lookup_table,
                                                          number_of_response_curves=number_of_response_curves,
                                                          starting_index=starting_index,
                                                          priors=priors,
                                                          *args, **kwargs)
    
    @property
    def _DoF(self):
        return (len(self.interferometers)-np.sum(self.polarization_basis)) * np.sum(self.time_frequency_filter)

    def log_likelihood(self):
        null_energy_array = self._calculate_residual_power()
        log_likelihood = scipy.stats.chi2.logpdf(null_energy_array, df=self._DoF)
        if len(log_likelihood) > 1:
            return logsumexp(log_likelihood) - np.log(len(log_likelihood))
        else:
            return log_likelihood[0]
    
    def _calculate_residual_power(self):
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
            null_energy_array = compute_null_energy_array(time_frequency_domain_strain_array_time_shifted,
                                                          self.psd_draws,
                                                          F_matrix,
                                                          self.time_frequency_filter,
                                                          self.time_frequency_filter_collapsed,
                                                          self.interferometers[0].sampling_frequency)
        elif self._sample_calibration_parameters:
            # Simulate the PSD.
            psd_array = np.array([get_simulated_calibrated_wavelet_psd(interferometer=ifo,
                                                                       parameters=self.parameters,
                                                                       wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                                       nx=self.wavelet_nx,
                                                                       nsample=self.simulate_psd_nsample) for ifo in self.interferometers])
            null_energy_array = np.array([compute_null_energy(time_frequency_domain_strain_array_time_shifted,
                                                              psd_array,
                                                              F_matrix,
                                                              self.time_frequency_filter,
                                                              self.time_frequency_filter_collapsed,
                                                              self.interferometers[0].sampling_frequency)])            
        else:
            null_energy_array = np.array([compute_null_energy(time_frequency_domain_strain_array_time_shifted,
                                                              self.psd_array,
                                                              F_matrix,
                                                              self.time_frequency_filter,
                                                              self.time_frequency_filter_collapsed,
                                                              self.interferometers[0].sampling_frequency)])
        
        return null_energy_array

    def _calculate_noise_log_likelihood(self):
        # Transform the time-shifted data to the time-freuency domain
        time_frequency_domain_strain_array = np.array([transform_wavelet_freq(data,
                                                                              self._wavelet_Nf,
                                                                              self._wavelet_Nt,
                                                                              self.wavelet_nx) for data in self.frequency_domain_strain_array])
        # Whiten the strain data
        psd_array = getattr(self, 'psd_array', None)
        if psd_array is None:
            psd_array = np.array([simulate_wavelet_psd(interferometer=ifo,
                                                       wavelet_frequency_resolution=self.wavelet_frequency_resolution,
                                                       nx=self.wavelet_nx,
                                                       nsample=self.simulate_psd_nsample) for ifo in self.interferometers])
        time_frequency_domain_strain_array_whitened = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array,
                                                                                                          self.psd_array,
                                                                                                          self.time_frequency_filter)
        energy = np.sum(np.abs(time_frequency_domain_strain_array_whitened)**2)
        self._noise_log_likelihood_value = scipy.stats.chi2.logpdf(energy, df=len(self.interferometers)*np.sum(self.time_frequency_filter))

@njit
def compute_null_energy(time_frequency_domain_strain_array_time_shifted,
                        psd_array,
                        F_matrix,
                        time_frequency_filter,
                        time_frequency_filter_collapsed,
                        srate):
    # Compute the whitened time-frequency domain strain array
    time_frequency_domain_strain_array_time_shifted_whitened = compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array_time_shifted,
                                                                                                                    psd_array,
                                                                                                                    time_frequency_filter)
    # Compute the whitened F_matrix
    whitened_F_matrix = compute_whitened_antenna_pattern_matrix_masked(F_matrix,
                                                                        psd_array,
                                                                        time_frequency_filter_collapsed,
                                                                        srate)
    # Compute the GW projector        
    Pgw = compute_gw_projector_masked(whitened_F_matrix, time_frequency_filter_collapsed)
    # Compute the null projector
    Pnull = compute_null_projector_from_gw_projector(Pgw)
    # Compute the projection squared
    projection_squared = compute_projection_squared(time_frequency_domain_strain_array_time_shifted_whitened,
                                                    Pnull,
                                                    time_frequency_filter)        
    return np.sum(projection_squared)

@njit
def compute_null_energy_array(time_frequency_domain_strain_array_time_shifted,
                              psd_draw_array,
                              F_matrix,
                              time_frequency_filter,
                              time_frequency_filter_collapsed,
                              srate):
    _, psd_nsample, _ = psd_draw_array.shape
    null_energy_array = np.zeros(psd_nsample)
    for i in range(psd_nsample):
        null_energy_array[i] = compute_null_energy(time_frequency_domain_strain_array_time_shifted=time_frequency_domain_strain_array_time_shifted,
                                                   psd_array=psd_draw_array[:,i,:],
                                                   F_matrix=F_matrix,
                                                   time_frequency_filter=time_frequency_filter,
                                                   time_frequency_filter_collapsed=time_frequency_filter_collapsed,
                                                   srate=srate)
    return null_energy_array