import numpy as np

def get_single_freq_null_projector(antenna_pattern_matrix):
    """Null projector.

    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix.

    Returns
    -------
    array_like
        Null projector with shape (n_interferometers, n_polarization).

    """
    antenna_pattern_matrix_dag = antenna_pattern_matrix.conj().T
    Pgw = antenna_pattern_matrix @ np.linalg.inv(antenna_pattern_matrix_dag @ antenna_pattern_matrix) @ antenna_pattern_matrix_dag

    return np.eye(Pgw.shape[0]) - Pgw

def get_null_projector(interferometers, antenna_pattern_matrix):
    """Null projector.

    Parameters
    ----------
    interferometers : array_like
        Array of bilby.gw.detector.interferometer.Interferometer objects with same frequency array.
    antenna_pattern_matrix : array_like
        Antenna pattern matrix with shape (n_interferometers, n_polarization).

    Returns
    -------
    array_like
        Null projector with shape (n_interferometers, n_polarization, n_freqs).

    """

    frequency_array = interferometers[0].frequency_array
    psds = np.array([interferometer.power_spectral_density_array for interferometer in interferometers])

    df = frequency_array[1] - frequency_array[0]

    whitening_factor = np.sqrt(psds/(2*df)) # shape (n_interferometers, n_freqs)
    null_projector = antenna_pattern_matrix[:, :, np.newaxis] / whitening_factor[:, np.newaxis, :] # shape (n_interferometers, n_polarization, n_freqs)

    return null_projector

def get_null_stream(interferometers, null_projector):
    """Null stream from interferometers.

    Parameters
    ----------
    interferometers : array_like
        Array of bilby.gw.detector.interferometer.Interferometer objects.
    null_projector : array_like
        Null projector with shape (n_interferometers, n_polarization, n_freqs).

    Returns
    -------
    array_like
        Null stream with shape (n_interferometers, n_freqs).

    """
    strain_data = np.array([interferometer.frequency_domain_strain for interferometer in interferometers])

    null_stream = np.zeros_like(null_projector)
    for i in range(len(null_projector.shape[2])):
        null_stream[:, :, i] = null_projector[:, :, i] @ strain_data[:, i]
    return null_stream
