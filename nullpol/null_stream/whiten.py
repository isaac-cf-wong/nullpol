import numpy as np
from numba import njit

@njit
def compute_whitened_antenna_pattern_matrix_masked(antenna_pattern_matrix,
                                                   psd_array,
                                                   frequency_mask):
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
    for i in range(nfreq):
        if frequency_mask[i]:
            for j in range(ndet):
                output[i,j,:] = antenna_pattern_matrix[j,:] / np.sqrt(psd_array[j,i])
    return output