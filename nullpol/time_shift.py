import numpy as np

def time_shift(interferometers, ra, dec, gps_time, frequency_array, strain_data_array):
    """Time shift.

    Parameters
    ----------
    interferometers : list
        List of bilby.gw.detector.Interferometer.
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
    strain_data_array : array_like
        Frequency domain strain array with shape (n_interferometers, n_freqs).
    Returns
    -------
    array_like
        Time-shifted whitened frequency domain strain array with shape (n_interferometers, n_freqs).
    """
    time_shift = np.conj(np.array([np.exp(-1.j*np.pi*2.*frequency_array*interferometer.time_delay_from_geocenter(ra, dec, gps_time)) for interferometer in interferometers]))

    return strain_data_array*time_shift
