import numpy as np


def get_null_projector(antenna_pattern_matrix):
    """Null projector.

    Parameters
    ----------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix.

    Returns
    -------
    array_like
        Null projector.

    """
    antenna_pattern_matrix_dag = antenna_pattern_matrix.conj().T
    FdagF_inv = np.linalg.inv(antenna_pattern_matrix_dag @ antenna_pattern_matrix)
    Pgw = antenna_pattern_matrix @ FdagF_inv @ antenna_pattern_matrix_dag

    return np.eye(antenna_pattern_matrix.shape[0]) - Pgw

def get_null_stream_from_interferometers(interferometers, null_projector):
    """Null stream from interferometers.

    Parameters
    ----------
    interferometers : array_like
        Interferometers.
    null_projector : array_like
        Null projector.

    Returns
    -------
    array_like
        Null stream.

    """
    return null_projector @ interferometers
