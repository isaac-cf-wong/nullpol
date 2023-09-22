import numpy as np

def get_null_projector(antenna_pattern_matrix, frequency_array, psds):
    """Null projector.

    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix with shape (n_interferometers, n_polarization).
    frequency_array : array_like
        Frequency array with shape (n_freqs).
    psds : array_like
        Power spectral density array with shape (n_interferometers, n_freqs).
    
    Returns
    -------
    array_like
        Null projector with shape (n_interferometers, n_interferometers, n_freqs).

    """
    df = frequency_array[1] - frequency_array[0]

    whitening_factor = 1/np.sqrt(psds/(2*df)) # shape (n_interferometers, n_freqs)

    whitened_antenna_pattern_matrix = np.einsum('ij, ik -> ijk', antenna_pattern_matrix, whitening_factor) # shape (n_interferometers, n_polarization, n_freqs)

    whitened_antenna_pattern_matrix_dag = whitened_antenna_pattern_matrix.conj().transpose(1, 0, 2) # shape (n_polarization, n_interferometers, n_freqs)

    Pgw = np.einsum('ijk, jlk ->ilk',
                    whitened_antenna_pattern_matrix,
                    np.einsum('ijk, jlk ->ilk',
                              np.linalg.inv(np.einsum('ijk, jlk ->ilk',
                                                      whitened_antenna_pattern_matrix_dag,
                                                      whitened_antenna_pattern_matrix).T).T,
                              whitened_antenna_pattern_matrix_dag)) # shape (n_interferometers, n_interferometers, n_freqs)

    return np.array([np.eye(Pgw.shape[0])]*Pgw.shape[2]).T - Pgw

def get_null_stream(strain_data_array, null_projector):
    """Null stream from interferometers.

    Parameters
    ----------
    strain_data_array : array_like
        Strain data array with shape (n_interferometers, n_freqs).
    null_projector : array_like
        Null projector with shape (n_interferometers, n_interferometers, n_freqs).

    Returns
    -------
    array_like
        Null stream with shape (n_interferometers, n_freqs).

    """
    return np.einsum('ijk, jk -> ik', null_projector, strain_data_array)
