import numpy as np

def get_lensing_null_stream(interferometers_1, interferometers_2, null_projector, ra, dec, gps_time_1, gps_time_2, minimum_frequency, maximum_frequency):
    """Null stream from interferometers.

    Parameters
    ----------
    interferometers_1 : list
        List of bilby.gw.detector.Interferometer.
    interferometers_2 : list
        List of bilby.gw.detector.Interferometer.
    null_projector : array_like
        Null projector with shape (n_interferometers, n_interferometers, n_freqs).
    ra : float
        Right ascension in radians.
    dec : float
        Declination in radians.
    gps_time_1 : float
        GPS time.
    gps_time_2 : float
        GPS time.
    minimum_frequency : float
        Minimum frequency.
    maximum_frequency : float
        Maximum frequency.

    Returns
    -------
    array_like
        Null stream with shape (n_interferometers, n_freqs).
    """
    frequency_array = interferometers_1[0].frequency_array

    strain_data_array_1 = interferometers_1.whitened_frequency_domain_strain_array[:, (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)]
    strain_data_array_2 = interferometers_2.whitened_frequency_domain_strain_array[:, (frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)]
    frequency_array = frequency_array[(frequency_array >= minimum_frequency) & (frequency_array <= maximum_frequency)]

    time_shift_1 = np.conj(np.array([np.exp(-1.j*np.pi*2.*frequency_array*interferometer.time_delay_from_geocenter(ra, dec, gps_time_1)) for interferometer in interferometers_1]))
    time_shift_2 = np.conj(np.array([np.exp(-1.j*np.pi*2.*frequency_array*interferometer.time_delay_from_geocenter(ra, dec, gps_time_2)) for interferometer in interferometers_2]))

    time_shifted_strain_data_array = np.concatenate((strain_data_array_1*time_shift_1, strain_data_array_2*time_shift_2), axis=0)

    return np.einsum('ijk, jk -> ik', null_projector, time_shifted_strain_data_array)