from __future__ import annotations

import numpy as np

# Mapping from polarization short names to array indices
POLARIZATION_ENCODING = dict(p=0, c=1, b=2, l=3, x=4, y=5)

# Array for converting indices back to short names
POLARIZATION_DECODING = np.array(["p", "c", "b", "l", "x", "y"])

# Mapping from short names to descriptive long names
POLARIZATION_LONG_NAMES = dict(p="plus", c="cross", b="breathing", l="longitudinal", x="vector_x", y="vector_y")

# Mapping from long names back to short names
POLARIZATION_SHORT_NAMES = dict(plus="p", cross="c", breathing="b", longitudinal="l", vector_x="x", vector_y="y")


def encode_polarization(polarization_modes, polarization_basis):
    """Encode polarization modes and basis into boolean arrays.

    Converts lists of polarization mode strings into boolean arrays indicating
    which modes are active, which serve as basis modes, and which are derived
    from the basis modes.

    Args:
        polarization_modes (list): List of polarization mode strings to encode.
            Valid values are 'p', 'c', 'b', 'l', 'x', 'y'.
        polarization_basis (list): List of polarization mode strings that serve
            as the basis. Must be a subset of polarization_modes.

    Returns:
        tuple: A tuple containing three numpy.ndarray objects:
            - polarization_modes_array (numpy.ndarray): Boolean array of shape (6,)
              indicating which polarization modes are active.
            - polarization_basis_array (numpy.ndarray): Boolean array of shape (6,)
              indicating which modes serve as the basis.
            - polarization_derived_array (numpy.ndarray): Boolean array of shape (6,)
              indicating which modes are derived (active but not basis).

    Example:
        >>> modes = ['p', 'c', 'b']
        >>> basis = ['p', 'c']
        >>> modes_arr, basis_arr, derived_arr = encode_polarization(modes, basis)
        >>> # modes_arr: [True, True, True, False, False, False]
        >>> # basis_arr: [True, True, False, False, False, False]
        >>> # derived_arr: [False, False, True, False, False, False]
    """
    _polarization_modes = np.full(6, False)
    _polarization_basis = np.full(6, False)
    for pol in polarization_modes:
        _polarization_modes[POLARIZATION_ENCODING[pol]] = True
    for pol in polarization_basis:
        _polarization_basis[POLARIZATION_ENCODING[pol]] = True
    _polarization_derived = _polarization_modes & (~_polarization_basis)
    return _polarization_modes, _polarization_basis, _polarization_derived


def get_long_names(tokens):
    """Convert polarization short names to their corresponding long names.

    Args:
        tokens (list): List of polarization short name strings.
            Valid values are 'p', 'c', 'b', 'l', 'x', 'y'.

    Returns:
        list: List of corresponding long names for the input tokens.

    Example:
        >>> get_long_names(['p', 'c', 'b'])
        ['plus', 'cross', 'breathing']

    Raises:
        KeyError: If any token in the input list is not a valid polarization
            short name.
    """
    return [POLARIZATION_LONG_NAMES[token] for token in tokens]
