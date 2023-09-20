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

def get_null_projector(antenna_pattern_matrix, frequency_array, psd):
    """Null projector.

    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix.
    frequency_array : array_like
        Frequency array of PSD.
    psd : array_like
        PSD.

    Returns
    -------
    array_like
        Null projector with shape (n_interferometers, n_polarization, n_freqs).

    """
    df = frequency_array[1] - frequency_array[0]
    n_freqs = len(frequency_array)

    null_projector = []
    for i in range(n_freqs):
        null_projector.append(get_single_freq_null_projector(antenna_pattern_matrix/np.sqrt(psd[i]/(2*df))))
    null_projector = np.array(null_projector)

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
    return null_projector @ interferometers
