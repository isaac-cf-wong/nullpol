import numpy as np
from .encoding import POLARIZATION_DECODING


def get_antenna_pattern(
        interferometer,
        right_ascension,
        declination,
        polarization_angle,
        gps_time,
        polarization):
    """
    Get antenna pattern for a given interferometer at a specific sky location and time.

    Args:
        interferometer (bibly.gw.detector.Interferometer): Interferometer.
        right_ascension (float): Right ascension in radians.
        declination (float): Declination in radians.
        polarization_angle (float): Polarization angle in radians.
        gps_time (float): GPS time.
        polarization (array_like): Array of booleans for polarization modes.

    Returns:
        array_like: Antenna pattern for the given sky location and time with shape (n_polarization).
    """
    polarization_name_list = np.array(['plus', 'cross', 'breathing', 'longitudinal', 'x', 'y'])

    return np.array([interferometer.antenna_response(right_ascension, declination, gps_time, polarization_angle, str(polarization_name)) for polarization_name in polarization_name_list[polarization]])


def get_antenna_pattern_matrix(
        interferometers,
        right_ascension,
        declination,
        polarization_angle,
        gps_time,
        polarization):
    """
    Get antenna pattern matrix for a given sky location and time.

    Args:
        interferometers (list): List of bilby.gw.detector.Interferometer.
        right_ascension (float): Right ascension in radians.
        declination (float): Declination in radians.
        polarization_angle (float): Polarization angle in radians.
        gps_time (float): GPS time.
        polarization (array_like): Array of booleans for polarization modes.

    Returns:
        array_like: Antenna pattern matrix for the given sky location and time with shape (n_interferometers, n_polarization).
    """
    return np.array([get_antenna_pattern(interferometer,
                                         right_ascension,
                                         declination,
                                         polarization_angle,
                                         gps_time,
                                         polarization) for interferometer in interferometers])


def relative_amplification_factor_map(polarization_basis,
                                      polarization_derived):
    """Get a map of the keywords to the relative amplification factors.

    Args:
        polarization_basis (boolean array): A 6-element boolean array indicating the basis modes.
        polarization_derived (boolean array): A 6-element boolean array indicating the derived modes.

    Returns:
        numpy array: A matrix of keyword labels. The first character indicates the derived mode,
        and the second character indicates the basis mode.
    """
    nbasis = np.sum(polarization_basis)
    nderived = np.sum(polarization_derived)
    if nderived == 0:
        return np.array([[] for _ in range(nbasis)])
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
    """A helper function to construct a matrix of relative amplification factors.
    
    Args:
        parameters_map (array-like): A map of keywords.
        parameters (dict): A dictionary of parameters.

    Returns:
        numpy array: A matrix of relative amplification factors.
    """
    func = lambda x: parameters[f'amplitude_{x}']*np.exp(1.j*parameters[f'phase_{x}'])
    return np.vectorize(func)(parameters_map)


def get_collapsed_antenna_pattern_matrix(
        antenna_pattern_matrix,
        polarization_basis,
        polarization_derived,
        relative_amplification_factor):
    """Get the collapsed antenna pattern matrix.

    Args:
        antenna_pattern_matrix (array-like): Antenna pattern matrix.
        polarization_basis (boolean array): A boolean array to indicate polarization basis.
        polarization_derived (boolean array): A boolean array to indicate the derived modes.
        relative_amplification_factor (array-like): The relative amplification factor.

    Returns:
        numpy array: Get a collapsed antenna pattern matrix.
    """
    # Dimensions:
    # antenna_pattern_matrix: (detector, polarization)
    # Select the columns corresponds to the basis    
    return antenna_pattern_matrix[:, polarization_basis] + \
        antenna_pattern_matrix[:, polarization_derived] @ relative_amplification_factor
