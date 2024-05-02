import numpy as np

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

def get_null_stream(interferometers, null_projector, ra, dec, gps_time, frequency_array, frequency_mask):
    """Null stream from interferometers.

    Parameters
    ----------
    interferometers : list
        List of bilby.gw.detector.Interferometer.
    null_projector : array_like
        Null projector with shape (n_interferometers, n_interferometers, n_freqs).
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    gps_time : float
        GPS time.
    frequency_array : array_like
        Frequency array with shape (n_freqs).
    frequency_mask : array_like
        Frequency mask with shape (n_freqs).

    Returns
    -------
    array_like
        Null stream with shape (n_interferometers, n_freqs).
    """
    strain_data_array = interferometers.whitened_frequency_domain_strain_array[:, frequency_mask]

    time_shift = np.conj(np.array([np.exp(-1.j*np.pi*2.*frequency_array*interferometer.time_delay_from_geocenter(ra, dec, gps_time)) for interferometer in interferometers]))

    return np.einsum('ijk, jk -> ik', null_projector, strain_data_array*time_shift)

def get_null_energy(null_stream):
    """Null energy.

    Parameters
    ----------
    null_stream : array_like
        Null stream with shape (n_interferometers, n_freqs).

    Returns
    -------
    float
        Null energy.
    """
    return np.sum(np.abs(null_stream)**2)
