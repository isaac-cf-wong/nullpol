import numpy as np

def get_null_stream(null_projector, time_shifted_strain_data_array):
    """Null stream from interferometers.

    Parameters
    ----------
    null_projector : array_like
        Null projector with shape (n_interferometers, n_interferometers, n_freqs).
    time_shifted_strain_data_array : array_like
        Time shifted strain data array with shape (n_interferometers, n_freqs).

    Returns
    -------
    array_like
        Null stream with shape (n_interferometers, n_freqs).
    """

    return np.einsum('ijk, jk -> ik', null_projector, time_shifted_strain_data_array)

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
