from .encoding import POLARIZATION_DECODING
import numpy as np

def get_antenna_pattern(interferometer, right_ascension, declination, polarization_angle, gps_time, polarization):
    """
    Get antenna pattern for a given interferometer at a specific sky location and time.

    Parameters
    ----------
    interferometer : bibly.gw.detector.Interferometer
        Interferometer.
    right_ascension : float
        Right ascension in radians.
    declination : float
        Declination in radians.
    polarization_angle : float
        Polarization angle in radians.
    gps_time : float
        GPS time.
    polarization : array_like
        Array of booleans for polarization modes.

    Returns
    -------
    antenna_pattern : array_like
        Antenna pattern for the given sky location and time with shape (n_polarization).
    """
    polarization_name_list = np.array(['plus', 'cross', 'breathing', 'longitudinal', 'x', 'y'])

    return np.array([interferometer.antenna_response(right_ascension, declination, gps_time, polarization_angle, str(polarization_name)) for polarization_name in polarization_name_list[polarization]])

def get_antenna_pattern_matrix(interferometers, right_ascension, declination, polarization_angle, gps_time, polarization):
    """
    Get antenna pattern matrix for a given sky location and time.

    Parameters
    ----------
    interferometers : list
        List of bilby.gw.detector.Interferometer.
    right_ascension : float
        Right ascension in radians.
    declination : float
        Declination in radians.
    polarization_angle : float
        Polarization angle in radians.
    gps_time : float
        GPS time.
    polarization : array_like
        Array of booleans for polarization modes.

    Returns
    -------
    antenna_pattern_matrix : array_like
        Antenna pattern matrix for the given sky location and time with shape (n_interferometers, n_polarization).
    """
    return np.array([get_antenna_pattern(interferometer, right_ascension, declination, polarization_angle, gps_time, polarization) for interferometer in interferometers])

def whiten_antenna_pattern_matrix(antenna_pattern_matrix, frequency_array, psds):
    """
    Whiten antenna pattern matrix.

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
    whitened_antenna_pattern_matrix : array_like
        Whitened antenna pattern matrix with shape (n_interferometers, n_polarization, n_freqs).
    """
    df = frequency_array[1] - frequency_array[0]

    whitening_factor = 1/np.sqrt(psds/(2*df)) # shape (n_interferometers, n_freqs)

    return np.einsum('ij, ik -> ijk', antenna_pattern_matrix, whitening_factor) # shape (n_interferometers, n_polarization, n_freqs)

def change_basis(whitened_antenna_pattern_matrix, basis, amp_phase_factor):
    """
    Change basis of whitened antenna pattern matrix.

    Parameters
    ----------
    whitened_antenna_pattern_matrix : array_like
        Whitened antenna pattern matrix with shape (n_interferometers, n_polarization, n_freqs).
    basis : array_like
        Array of booleans for basis modes.
    amp_phase_factor : array_like
        Array of amplitude and phase factors for basis modes with shape (n_polarization-n_basis, n_basis, 2).

    Returns
    -------
    whitened_antenna_pattern_matrix_new_basis : array_like
        Whitened antenna pattern matrix with shape (n_interferometers, n_basis, n_freqs).
    """
    multiplicative_factor = amp_phase_factor[:, :, 0] * np.exp(1j * amp_phase_factor[:, :, 1]) # shape (n_polarization-n_basis, n_basis)
    additional_terms = np.einsum('ijk, jl -> ilk', whitened_antenna_pattern_matrix[:, np.invert(basis), :], multiplicative_factor) # shape (n_interferometers, n_basis, n_freqs)
    
    return whitened_antenna_pattern_matrix[:, basis, :] + additional_terms # shape (n_interferometers, n_basis, n_freqs)

def relative_amplification_factor_map(polarization_basis,
                                      polarization_derived):
    nbasis = np.sum(polarization_basis)
    nderived = np.sum(polarization_derived)
    if nderived == 0:
        return None
    output = []
    i_counter = 0
    j_counter = 0
    for i in range(len(polarization_derived)):
        if not polarization_derived[i]:
            continue
        row = []
        for j in range(len(polarization_basis)):
            if polarization_basis[j]:
                row.append(f'{POLARIZATION_DECODING[i]}{POLARIZATION_DECODING[j]}')
            j_counter += 1
        output.append(row)
        i_counter += 1
    return np.array(output)

def relative_amplification_factor_helper(parameters_map,
                                         parameters):
    func = lambda x: parameters[f'amplitude_{x}']*np.exp(1.j*parameters[f'phase_{x}'])
    return np.vectorize(func)(parameters_map)

def get_collapsed_antenna_pattern_matrix(antenna_pattern_matrix,
                                         polarization_basis,
                                         polarization_derived,
                                         relative_amplification_factor):
    # Dimensions:
    ## antenna_pattern_matrix: (detector, polarization)
    # Select the columns corresponds to the basis    
    return antenna_pattern_matrix[:, polarization_basis] + antenna_pattern_matrix[:, polarization_derived] @ relative_amplification_factor