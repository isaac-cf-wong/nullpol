from __future__ import annotations

import numpy as np

from .encoding import POLARIZATION_DECODING


def get_antenna_pattern(interferometer, right_ascension, declination, polarization_angle, gps_time, polarization):
    """
    Get antenna pattern for a given interferometer at a specific sky location and time.

    Args:
        interferometer (bilby.gw.detector.Interferometer): Interferometer object.
        right_ascension (float): Right ascension in radians.
        declination (float): Declination in radians.
        polarization_angle (float): Polarization angle in radians.
        gps_time (float): GPS time in seconds.
        polarization (array-like): Boolean array of shape (6,) for polarization modes
            [plus, cross, breathing, longitudinal, x, y].

    Returns:
        numpy.ndarray: Antenna pattern for the given sky location and time with shape (n_polarization,).
    """
    polarization_name_list = np.array(["plus", "cross", "breathing", "longitudinal", "x", "y"])

    return np.array(
        [
            interferometer.antenna_response(
                right_ascension, declination, gps_time, polarization_angle, str(polarization_name)
            )
            for polarization_name in polarization_name_list[polarization]
        ]
    )


def get_antenna_pattern_matrix(
    interferometers, right_ascension, declination, polarization_angle, gps_time, polarization
):
    """
    Get antenna pattern matrix for a given sky location and time.

    Args:
        interferometers (list): List of bilby.gw.detector.Interferometer objects.
        right_ascension (float): Right ascension in radians.
        declination (float): Declination in radians.
        polarization_angle (float): Polarization angle in radians.
        gps_time (float): GPS time in seconds.
        polarization (array-like): Boolean array of shape (6,) for polarization modes
            [plus, cross, breathing, longitudinal, x, y].

    Returns:
        numpy.ndarray: Antenna pattern matrix for the given sky location and time
            with shape (n_interferometers, n_polarization).
    """
    return np.array(
        [
            get_antenna_pattern(
                interferometer, right_ascension, declination, polarization_angle, gps_time, polarization
            )
            for interferometer in interferometers
        ]
    )


def relative_amplification_factor_map(polarization_basis, polarization_derived):
    """
    Generate a mapping matrix of keyword labels for relative amplification factors.

    Creates a matrix where each element represents a keyword combining a derived
    polarization mode with a basis polarization mode, used for constructing
    relative amplification factor matrices.

    Args:
        polarization_basis (array-like): Boolean array of shape (6,) indicating which
            polarization modes serve as basis modes.
        polarization_derived (array-like): Boolean array of shape (6,) indicating which
            polarization modes are derived from the basis modes.

    Returns:
        numpy.ndarray: Matrix of keyword labels with shape (n_derived, n_basis).
            Each element is a string where the first character indicates the derived mode
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
                row.append(f"{POLARIZATION_DECODING[i]}{POLARIZATION_DECODING[j]}")
            j_counter += 1
        output.append(row)
        i_counter += 1
    return np.array(output)


def relative_amplification_factor_helper(parameters_map, parameters):
    """
    Construct a matrix of relative amplification factors from parameter mappings.

    Args:
        parameters_map (array-like): Matrix of keyword strings mapping derived modes
            to basis modes, typically from relative_amplification_factor_map().
        parameters (dict): Dictionary containing amplitude and phase parameters.
            Expected keys: 'amplitude_{keyword}' and 'phase_{keyword}' for each
            keyword in parameters_map.

    Returns:
        numpy.ndarray: Matrix of complex relative amplification factors with the same
            shape as parameters_map.
    """

    def func(x):
        return parameters[f"amplitude_{x}"] * np.exp(1.0j * parameters[f"phase_{x}"])

    return np.vectorize(func)(parameters_map)


def get_collapsed_antenna_pattern_matrix(
    antenna_pattern_matrix, polarization_basis, polarization_derived, relative_amplification_factor
):
    """
    Compute the collapsed antenna pattern matrix by combining basis and derived modes.

    This function creates a reduced antenna pattern matrix by expressing derived
    polarization modes as linear combinations of basis modes, effectively
    "collapsing" the full polarization space into a smaller basis representation.

    Args:
        antenna_pattern_matrix (array-like): Antenna pattern matrix with shape
            (n_detectors, n_polarizations).
        polarization_basis (array-like): Boolean array of shape (n_polarizations,)
            indicating which polarization modes serve as the basis.
        polarization_derived (array-like): Boolean array of shape (n_polarizations,)
            indicating which polarization modes are derived from the basis.
        relative_amplification_factor (array-like): Matrix of complex amplification
            factors with shape (n_derived, n_basis) relating derived modes to basis modes.

    Returns:
        numpy.ndarray: Collapsed antenna pattern matrix with shape (n_detectors, n_basis).
    """
    # Dimensions:
    # antenna_pattern_matrix: (detector, polarization)
    # Select the columns corresponds to the basis
    return (
        antenna_pattern_matrix[:, polarization_basis]
        + antenna_pattern_matrix[:, polarization_derived] @ relative_amplification_factor
    )
