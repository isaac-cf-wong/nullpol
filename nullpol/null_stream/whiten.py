import numpy as np
from numba import njit

@njit
def compute_whitened_antenna_pattern_matrix_masked(antenna_pattern_matrix,
                                                   psd_array,
                                                   frequency_mask,
                                                   srate):
    """
    Whiten the antenna pattern matrix with the given PSD array.
    
    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix with shape (ndet, nmode).
    psd_array : array_like
        PSD array with shape (ndet, nfreq).
    frequency_mask : array_like
        Frequency mask with shape (nfreq).
    srate: float
        Sampling rate in Hz.
        
    Returns
    -------
    array_like
        Whiten antenna pattern matrix with shape (nfreq, ndet, nmode).
    """
    nfreq = len(frequency_mask)
    ndet, nmode = antenna_pattern_matrix.shape
    output = np.zeros((nfreq, ndet, nmode), dtype=antenna_pattern_matrix.dtype)
    scaling = srate / 2
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                output[i,j,:] = antenna_pattern_matrix[j,:] / np.sqrt(psd_array[j,i] * scaling)
    return output

@njit
def compute_whitened_time_frequency_domain_strain_array(time_frequency_domain_strain_array,
                                                        psd_array,
                                                        time_frequency_filter,
                                                        srate):
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
    srate: float
        Sampling rate in Hz.

    Returns
    -------
    array_like
        Whiten time-frequency domain strain array with shape (ndet, ntime, nfreq
    """
    output = np.zeros_like(time_frequency_domain_strain_array)
    ndet, ntime, nfreq = time_frequency_domain_strain_array.shape
    scaling = srate / 2
    for i in range(ntime):
        for j in range(nfreq):
            if time_frequency_filter[i,j]:
                for k in range(ndet):
                    output[k,i,j] = time_frequency_domain_strain_array[k,i,j] / np.sqrt(psd_array[k,j]*scaling)
    return output