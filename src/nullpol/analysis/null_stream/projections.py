"""Core projection functions for null stream analysis."""

from __future__ import annotations

import numpy as np
from numba import njit


# =============================================================================
# CORE PROJECTION FUNCTIONS (NUMBA OPTIMIZED)
# =============================================================================


@njit
def compute_gw_projector(whitened_antenna_pattern_matrix, frequency_mask):
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
    nfreq, ndet, _ = whitened_antenna_pattern_matrix.shape
    output = np.zeros((nfreq, ndet, ndet), dtype=whitened_antenna_pattern_matrix.dtype)

    for i in range(len(frequency_mask)):
        if frequency_mask[i]:
            F = np.ascontiguousarray(whitened_antenna_pattern_matrix[i, :, :])
            F_dagger = np.ascontiguousarray(np.conj(F).T)
            output[i, :, :] = F @ np.linalg.inv(F_dagger @ F) @ F_dagger

    return output


@njit
def compute_null_projector(gw_projector):
    """Compute null stream projector from gravitational wave projector.

    The null projector is the orthogonal complement to the GW projector,
    computed as N = I - P where I is the identity matrix and P is the GW projector.
    This projects detector strain onto the null space orthogonal to the GW signal subspace.

    Args:
        gw_projector (numpy.ndarray): GW projector matrix with shape
            (n_frequencies, n_detectors, n_detectors). Should be computed from
            _compute_gw_projector().

    Returns:
        numpy.ndarray: Null projector matrix with shape (n_frequencies, n_detectors, n_detectors).
            Contains N = I - P for each frequency bin.
    """
    nfreq, ndet, _ = gw_projector.shape

    output = -gw_projector.copy()
    for i in range(nfreq):
        for j in range(ndet):
            output[i, j, j] += 1.0

    return output


@njit
def compute_null_stream(
    whitened_freq_strain: np.ndarray, null_projector: np.ndarray, frequency_mask: np.ndarray
) -> np.ndarray:
    """Project the whitened frequency-domain strain onto the null space.

    For each frequency bin selected by the mask, applies the null projector to the
    detector strain vector, yielding the null stream in frequency domain.

    Args:
        whitened_freq_strain (np.ndarray): Whitened frequency-domain strain (n_det, n_freq).
        null_projector (np.ndarray): Null projector matrices (n_freq, n_det, n_det).
        frequency_mask (np.ndarray): Boolean mask for frequency bins (n_freq,).

    Returns:
        np.ndarray: Null-projected frequency-domain strain (n_det, n_freq).
    """
    n_det, n_freq = whitened_freq_strain.shape
    null_projected_freq_strain = np.zeros_like(whitened_freq_strain, dtype=whitened_freq_strain.dtype)

    for freq_idx in range(n_freq):
        if frequency_mask[freq_idx]:
            d = whitened_freq_strain[:, freq_idx]
            projector = null_projector[freq_idx]
            # Manual matrix-vector multiplication for small n_det
            for i in range(n_det):
                result = 0.0 + 0.0j  # Ensure complex type
                for j in range(n_det):
                    result += projector[i, j] * d[j]
                null_projected_freq_strain[i, freq_idx] = result

    return null_projected_freq_strain
