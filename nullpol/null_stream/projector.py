from numba import njit
import numpy as np

@njit
def compute_gw_projector_masked(whitened_antenna_pattern_matrix,
                                frequency_mask):
    nfreq, ndet, nmode = whitened_antenna_pattern_matrix.shape
    output = np.zeros((nfreq, ndet, ndet), dtype=whitened_antenna_pattern_matrix.dtype)
    for i in range(len(frequency_mask)):
        if frequency_mask[i]:
            F = np.ascontiguousarray(whitened_antenna_pattern_matrix[i,:,:])
            F_dagger = np.ascontiguousarray(np.conj(F).T)
            output[i,:,:] = F @ np.linalg.inv(F_dagger @ F) @ F_dagger
    return output

@njit
def compute_null_projector_from_gw_projector(gw_projector):
    nfreq, ndet, _ = gw_projector.shape    
    output = -gw_projector.copy()
    for i in range(nfreq):
        for j in range(ndet):
            output[i,j,j] += 1.
    return output

@njit
def compute_projection_squared(time_frequency_domain_strain_array,
                               projector,
                               time_freuency_filter):
    # Dimensions
    ## time_frequency_domain_strain_array: (detector, time, frequency)
    ## projector: (freq, detector, detector)
    ## time_freuency_filter: (time, frequency)
    ndet, ntime, nfreq = time_frequency_domain_strain_array.shape
    output = np.zeros((ntime,nfreq), dtype=np.float64)
    for i in range(ntime):
        for j in range(nfreq):
            if time_freuency_filter[i,j]:
                d = np.ascontiguousarray(time_frequency_domain_strain_array[:,i,j].astype(projector.dtype))
                output[i,j] = np.abs(np.conj(d) @ projector[j] @ d)
    return output