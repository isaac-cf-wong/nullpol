import numpy as np
from numba import njit
from bilby.gw.detector import PowerSpectralDensity
from ..psd import simulate_psd_from_bilby_psd
from ..time_frequency_transform import transform_wavelet_freq, get_shape_of_wavelet_transform


def simulate_wavelet_psd(interferometer, wavelet_frequency_resolution, nx, nsample):
    wavelet_psd = simulate_psd_from_bilby_psd(psd=interferometer.power_spectral_density,
                                              seglen=interferometer.duration,
                                              srate=interferometer.sampling_frequency,
                                              wavelet_frequency_resolution=wavelet_frequency_resolution,
                                              nsample=nsample,
                                              nx=nx)
    return wavelet_psd

def get_calibrated_power_spectral_density(frequency_array,
                                          power_spectral_density_array,
                                          calibration_errors):
    psd_array = power_spectral_density_array / np.abs(calibration_errors)**2
    return PowerSpectralDensity(frequency_array=frequency_array,
                                psd_array=psd_array)

def simulate_calibrated_wavelet_psd(interferometer, calibration_errors, wavelet_frequency_resolution, nx, nsample):
    calibrated_psd = get_calibrated_power_spectral_density(frequency_array=interferometer.frequency_array,
                                                           power_spectral_density_array=interferometer.power_spectral_density_array,
                                                           calibration_errors=calibration_errors)
    wavelet_psd = simulate_psd_from_bilby_psd(psd=calibrated_psd,
                                              seglen=interferometer.duration,
                                              srate=interferometer.sampling_frequency,
                                              wavelet_frequency_resolution=wavelet_frequency_resolution,
                                              nsample=nsample,
                                              nx=nx)
    return wavelet_psd

@njit
def compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array,
                                                        psd_array,
                                                        time_frequency_filter):
    """
    Whiten the time-frequency domain strain array with the given PSD array.
    
    Parameters
    ----------
    time_frequency_domain_strain_array : array_like
        Time-frequency domain strain array with shape (ndet, ntime, nfreq).
    psd_array : array_like
        PSD array with shape (ndet, nfreq).
    time_frequency_filter : array_like
        Time-frequency filter with shape (ntime, nfreq).

    Returns
    -------
    array_like
        Whiten time-frequency domain strain array with shape (ndet, ntime, nfreq
    """
    output = np.zeros_like(time_frequency_domain_strain_array)
    ndet, ntime, nfreq = time_frequency_domain_strain_array.shape
    for i in range(ntime):
        for j in range(nfreq):
            if time_frequency_filter[i,j]:
                for k in range(ndet):
                    output[k,i,j] = time_frequency_domain_strain_array[k,i,j] / np.sqrt(psd_array[k,j])
    return output

def compute_time_frequency_domain_strain_array(interferometer,
                                               wavelet_frequency_resolution,
                                               nx):
    Nt, Nf = get_shape_of_wavelet_transform(duration=interferometer.duration,
                                            sampling_frequency=interferometer.sampling_frequency,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)    
    return transform_wavelet_freq(data=interferometer.frequency_domain_strain,
                                  Nf=Nf,
                                  Nt=Nt,
                                  nx=nx)

def compute_geocentric_time_shifted_time_frequency_domain_strain_array(interferometer,
                                                                       ra,
                                                                       dec,
                                                                       gps_time,
                                                                       wavelet_frequency_resolution,
                                                                       nx):
    time_delay = interferometer.time_delay_from_geocenter(ra, dec, gps_time)
    time_shifted_frequency_domain_strain_array = np.exp(1.j*np.pi*2*interferometer.frequency_array*time_delay)*interferometer.frequency_domain_strain
    # Transform to wavelet domain
    Nt, Nf = get_shape_of_wavelet_transform(duration=interferometer.duration,
                                            sampling_frequency=interferometer.sampling_frequency,
                                            wavelet_frequency_resolution=wavelet_frequency_resolution)    
    return transform_wavelet_freq(data=time_shifted_frequency_domain_strain_array,
                                  Nf=Nf,
                                  Nt=Nt,
                                  nx=nx)

def get_simulated_calibrated_wavelet_psd(interferometer,
                                         parameters,
                                         wavelet_frequency_resolution,
                                         nx,
                                         nsample,
                                         frequencies=None):
    if frequencies is None:
        mask = interferometer.frequency_mask
        frequencies = interferometer.frequency_array[mask]
    else:
        mask = np.ones(len(frequencies), dtype=bool)
    # Compute the calibration errors
    calibration_errors = interferometer.calibration_model.get_calibration_factor(frequency_array=frequencies,
                                                                                 prefix=f'recalib_{interferometer.name}_',
                                                                                 **parameters)                                                                                 
    calibrated_psd_array = interferometer.power_spectral_density_array.copy()
    calibrated_psd_array[mask] /= np.abs(calibration_errors)**2
    calibrated_psd = PowerSpectralDensity(frequency_array=interferometer.frequency_array.copy(),
                                           psd_array=calibrated_psd_array)
    wavelet_psd = simulate_psd_from_bilby_psd(psd=calibrated_psd,
                                              seglen=interferometer.duration,
                                              srate=interferometer.sampling_frequency,
                                              wavelet_frequency_resolution=wavelet_frequency_resolution,
                                              nsample=nsample,
                                              nx=nx)
    return wavelet_psd                                           