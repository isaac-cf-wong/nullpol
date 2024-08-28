import numpy as np

def time_shift(interferometers, ra, dec, gps_time, frequency_array, frequency_mask):
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

    Returns
    -------
    array_like
        Time shifted strain data array with shape (n_interferometers, n_freqs).
        n_freqs is the number of frequency bins after applying the frequency mask.
    """
    strain_data_array = interferometers.whitened_frequency_domain_strain_array[:, frequency_mask]

    time_shift = np.conj(np.array([np.exp(-1.j*np.pi*2.*frequency_array*interferometer.time_delay_from_geocenter(ra, dec, gps_time)) for interferometer in interferometers]))

    return strain_data_array*time_shift
