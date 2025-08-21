"""Test module for encoding functionality.

This module tests the polarization encoding utilities used for
mode representation and conversion.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.analysis.antenna_patterns import (
    encode_polarization,
    get_long_names,
    POLARIZATION_ENCODING,
    POLARIZATION_DECODING,
    POLARIZATION_LONG_NAMES,
    POLARIZATION_SHORT_NAMES,
)


def test_polarization_encoding_constants():
    """Test that polarization encoding constants are properly defined."""
    # Test encoding dictionary
    expected_encoding = {"p": 0, "c": 1, "b": 2, "l": 3, "x": 4, "y": 5}
    assert POLARIZATION_ENCODING == expected_encoding

    # Test decoding array
    expected_decoding = ["p", "c", "b", "l", "x", "y"]
    assert np.array_equal(POLARIZATION_DECODING, expected_decoding)

    # Test long names
    expected_long_names = {
        "p": "plus",
        "c": "cross",
        "b": "breathing",
        "l": "longitudinal",
        "x": "vector_x",
        "y": "vector_y",
    }
    assert POLARIZATION_LONG_NAMES == expected_long_names

    # Test short names (reverse mapping)
    expected_short_names = {
        "plus": "p",
        "cross": "c",
        "breathing": "b",
        "longitudinal": "l",
        "vector_x": "x",
        "vector_y": "y",
    }
    assert POLARIZATION_SHORT_NAMES == expected_short_names


def test_encode_polarization():
    """Test encoding of polarization modes into boolean arrays."""
    polarization_modes = ["p", "c", "b"]
    polarization_basis = ["p", "c"]

    modes_arr, basis_arr, derived_arr = encode_polarization(polarization_modes, polarization_basis)

    # Test modes array
    expected_modes = np.array([True, True, True, False, False, False])
    assert np.array_equal(modes_arr, expected_modes)

    # Test basis array
    expected_basis = np.array([True, True, False, False, False, False])
    assert np.array_equal(basis_arr, expected_basis)

    # Test derived array
    expected_derived = np.array([False, False, True, False, False, False])
    assert np.array_equal(derived_arr, expected_derived)


def test_get_long_names():
    """Test conversion from short to long polarization names."""
    tokens = ["p", "c", "b"]
    result = get_long_names(tokens)
    expected = ["plus", "cross", "breathing"]
    assert result == expected

    # Test with all modes
    all_tokens = ["p", "c", "b", "l", "x", "y"]
    all_result = get_long_names(all_tokens)
    expected_all = ["plus", "cross", "breathing", "longitudinal", "vector_x", "vector_y"]
    assert all_result == expected_all


def test_get_long_names_error_handling():
    """Test that get_long_names handles invalid tokens appropriately."""
    with pytest.raises(KeyError):
        get_long_names(["invalid_token"])


def test_encoding_consistency():
    """Test consistency between encoding and decoding mappings."""
    # Test that encoding and decoding are consistent
    for short_name, index in POLARIZATION_ENCODING.items():
        assert POLARIZATION_DECODING[index] == short_name

    # Test that long and short name mappings are consistent
    for short_name, long_name in POLARIZATION_LONG_NAMES.items():
        assert POLARIZATION_SHORT_NAMES[long_name] == short_name
