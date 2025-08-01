from __future__ import annotations

import numpy as np
from numba import njit


@njit
def compute_gw_projector_masked(whitened_antenna_pattern_matrix,
                                frequency_mask):
    """Compute gravitational wave signal projector with frequency masking.

    Calculates the orthogonal projection operator that projects detector strain
    onto the subspace spanned by the gravitational wave polarizations, for each
    frequency bin in the mask.

    Args:
        whitened_antenna_pattern_matrix (numpy.ndarray): Whitened antenna pattern matrix
            with shape (n_frequencies, n_detectors, n_modes). Contains the detector
            response to each polarization mode after whitening.
        frequency_mask (numpy.ndarray): Boolean mask with shape (n_frequencies,)
            indicating which frequency bins to process.

    Returns:
        numpy.ndarray: GW projector matrix with shape (n_frequencies, n_detectors, n_detectors).
            For masked frequencies, contains P = F(F†F)⁻¹F† where F is the antenna pattern matrix.
            For unmasked frequencies, contains zeros.
    """
    nfreq, ndet, nmode = whitened_antenna_pattern_matrix.shape
    output = np.zeros((nfreq, ndet, ndet), dtype=whitened_antenna_pattern_matrix.dtype)
    for i in range(len(frequency_mask)):
        if frequency_mask[i]:
            F = np.ascontiguousarray(whitened_antenna_pattern_matrix[i,:,:])
            F_dagger = np.ascontiguousarray(np.conj(F).T)
            output[i, :, :] = F @ np.linalg.inv(F_dagger @ F) @ F_dagger
    return output


@njit
def compute_null_projector_from_gw_projector(gw_projector):
    """Compute null stream projector from gravitational wave projector.

    The null projector is the orthogonal complement to the GW projector,
    computed as N = I - P where I is the identity matrix and P is the GW projector.
    This projects detector strain onto the null space orthogonal to the GW signal subspace.

    Args:
        gw_projector (numpy.ndarray): GW projector matrix with shape
            (n_frequencies, n_detectors, n_detectors). Should be computed from
            compute_gw_projector_masked().

    Returns:
        numpy.ndarray: Null projector matrix with shape (n_frequencies, n_detectors, n_detectors).
            Contains N = I - P for each frequency bin.
    """
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
    """Compute squared magnitude of projected strain in time-frequency domain.

    Projects the detector strain data onto a subspace defined by the projector
    and computes the squared magnitude of the projection for each time-frequency pixel.

    Args:
        time_frequency_domain_strain_array (numpy.ndarray): Time-frequency strain data
            with shape (n_detectors, n_time, n_frequencies). Contains the strain data
            for each detector in the time-frequency representation.
        projector (numpy.ndarray): Projection operator with shape
            (n_frequencies, n_detectors, n_detectors). Defines the subspace for projection.
        time_freuency_filter (numpy.ndarray): Boolean filter with shape (n_time, n_frequencies)
            indicating which time-frequency pixels to process.

    Returns:
        numpy.ndarray: Squared projection magnitudes with shape (n_time, n_frequencies).
            Contains |d†Pd|² for each time-frequency pixel where P is the projector
            and d is the detector strain vector.
    """
    # Dimensions
    ## time_frequency_domain_strain_array: (detector, time, frequency)
    ## projector: (freq, detector, detector)
    ## time_frequency_filter: (time, frequency)
    ndet, ntime, nfreq = time_frequency_domain_strain_array.shape
    output = np.zeros((ntime,nfreq), dtype=np.float64)
    for i in range(ntime):
        for j in range(nfreq):
            if time_freuency_filter[i,j]:
                d = np.ascontiguousarray(time_frequency_domain_strain_array[:,i,j].astype(projector.dtype))
                output[i,j] = np.abs(np.conj(d) @ projector[j] @ d)
    return output


@njit
def compute_time_frequency_domain_strain_array_squared(time_frequency_domain_strain_array,
                                                       time_frequency_filter):
    """Compute squared magnitude of strain in time-frequency domain.

    Calculates the squared magnitude of the strain vector at each time-frequency pixel,
    corresponding to the total power in all detectors at that pixel.

    Args:
        time_frequency_domain_strain_array (numpy.ndarray): Time-frequency strain data
            with shape (n_detectors, n_time, n_frequencies). Contains the strain data
            for each detector in the time-frequency representation.
        time_frequency_filter (numpy.ndarray): Boolean filter with shape (n_time, n_frequencies)
            indicating which time-frequency pixels to process.

    Returns:
        numpy.ndarray: Squared strain magnitudes with shape (n_time, n_frequencies).
            Contains |d|² for each time-frequency pixel where d is the detector strain vector.
            Zero for pixels not in the filter.
    """
    # Dimensions
    ## time_frequency_domain_strain_array: (detector, time, frequency)
    ## time_freuency_filter: (time, frequency)
    ndet, ntime, nfreq = time_frequency_domain_strain_array.shape
    output = np.zeros((ntime,nfreq), dtype=np.float64)
    for i in range(ntime):
        for j in range(nfreq):
            if time_frequency_filter[i,j]:
                d = np.ascontiguousarray(time_frequency_domain_strain_array[:,i,j])
                output[i,j] = np.abs(np.conj(d) @ d)
    return output
