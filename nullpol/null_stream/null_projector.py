import numpy as np
from numba import njit

def get_null_projector(antenna_pattern_matrix):
    """Null projector.

    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Whitened antenna pattern matrix in new basis with shape (n_interferometers, n_basis, n_freqs).
    
    Returns
    -------
    array_like
        Null projector with shape (n_interferometers, n_interferometers, n_freqs).
    """
    antenna_pattern_matrix_dag = antenna_pattern_matrix.conj().transpose(1, 0, 2) # shape (n_basis, n_interferometers, n_freqs)
    Pgw = np.einsum('ijk, jlk ->ilk',
                    antenna_pattern_matrix,
                    np.einsum('ijk, jlk ->ilk',
                              np.linalg.inv(np.einsum('ijk, jlk ->ilk',
                                                      antenna_pattern_matrix_dag,
                                                      antenna_pattern_matrix).T).T,
                              antenna_pattern_matrix_dag)) # shape (n_interferometers, n_interferometers, n_freqs)

    return np.array([np.eye(Pgw.shape[0])]*Pgw.shape[2]).T - Pgw