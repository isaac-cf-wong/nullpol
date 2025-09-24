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


def test_encode_polarization_specific_examples():
    """Test polarization encoding with specific known examples.
    
    Uses concrete examples where the expected boolean arrays
    can be easily verified by inspection.
    """
    # Example 1: Only plus and cross modes
    polarization_modes = ["p", "c"]
    polarization_basis = ["p", "c"]  # Both are basis modes

    modes_arr, basis_arr, derived_arr = encode_polarization(polarization_modes, polarization_basis)

    # Expected results for this case
    expected_modes = np.array([True, True, False, False, False, False])      # p=T, c=T, rest=F
    expected_basis = np.array([True, True, False, False, False, False])      # p=T, c=T, rest=F  
    expected_derived = np.array([False, False, False, False, False, False])  # none derived

    assert np.array_equal(modes_arr, expected_modes), f"Modes: expected {expected_modes}, got {modes_arr}"
    assert np.array_equal(basis_arr, expected_basis), f"Basis: expected {expected_basis}, got {basis_arr}"
    assert np.array_equal(derived_arr, expected_derived), f"Derived: expected {expected_derived}, got {derived_arr}"

    # Example 2: Plus as basis, cross as derived 
    polarization_modes = ["p", "c"] 
    polarization_basis = ["p"]  # Only plus is basis

    modes_arr2, basis_arr2, derived_arr2 = encode_polarization(polarization_modes, polarization_basis)

    expected_modes2 = np.array([True, True, False, False, False, False])      # p=T, c=T, rest=F
    expected_basis2 = np.array([True, False, False, False, False, False])     # p=T only
    expected_derived2 = np.array([False, True, False, False, False, False])   # c=T only

    assert np.array_equal(modes_arr2, expected_modes2), f"Modes2: expected {expected_modes2}, got {modes_arr2}"
    assert np.array_equal(basis_arr2, expected_basis2), f"Basis2: expected {expected_basis2}, got {basis_arr2}"
    assert np.array_equal(derived_arr2, expected_derived2), f"Derived2: expected {expected_derived2}, got {derived_arr2}"

    # Example 3: All modes active, plus+cross as basis
    polarization_modes = ["p", "c", "b", "l", "x", "y"]
    polarization_basis = ["p", "c"]

    modes_arr3, basis_arr3, derived_arr3 = encode_polarization(polarization_modes, polarization_basis)

    expected_modes3 = np.array([True, True, True, True, True, True])       # all modes active
    expected_basis3 = np.array([True, True, False, False, False, False])   # p,c as basis
    expected_derived3 = np.array([False, False, True, True, True, True])   # b,l,x,y derived

    assert np.array_equal(modes_arr3, expected_modes3), f"Modes3: expected {expected_modes3}, got {modes_arr3}"
    assert np.array_equal(basis_arr3, expected_basis3), f"Basis3: expected {expected_basis3}, got {basis_arr3}"
    assert np.array_equal(derived_arr3, expected_derived3), f"Derived3: expected {expected_derived3}, got {derived_arr3}"


def test_get_long_names_comprehensive():
    """Test long name conversion with comprehensive examples.
    
    Verifies all polarization modes and their expected long names.
    """
    # Test individual conversions with expected results
    test_cases = [
        (["p"], ["plus"]),
        (["c"], ["cross"]),  
        (["b"], ["breathing"]),
        (["l"], ["longitudinal"]),
        (["x"], ["vector_x"]),
        (["y"], ["vector_y"]),
    ]
    
    for short_names, expected_long in test_cases:
        result = get_long_names(short_names)
        assert result == expected_long, f"For {short_names}: expected {expected_long}, got {result}"
    
    # Test multiple conversions
    mixed_input = ["p", "b", "x"]
    expected_mixed = ["plus", "breathing", "vector_x"]
    result_mixed = get_long_names(mixed_input)
    assert result_mixed == expected_mixed, f"Mixed case: expected {expected_mixed}, got {result_mixed}"
    
    # Test empty input
    empty_result = get_long_names([])
    assert empty_result == [], f"Empty input should return empty list, got {empty_result}"


def test_encoding_constants_values():
    """Test that encoding constants have expected specific values.
    
    Verifies the exact mapping values rather than just consistency.
    """
    # Test specific encoding values
    assert POLARIZATION_ENCODING["p"] == 0, "Plus should map to index 0"
    assert POLARIZATION_ENCODING["c"] == 1, "Cross should map to index 1"
    assert POLARIZATION_ENCODING["b"] == 2, "Breathing should map to index 2"
    assert POLARIZATION_ENCODING["l"] == 3, "Longitudinal should map to index 3"
    assert POLARIZATION_ENCODING["x"] == 4, "Vector-x should map to index 4"
    assert POLARIZATION_ENCODING["y"] == 5, "Vector-y should map to index 5"
    
    # Test specific decoding values
    assert POLARIZATION_DECODING[0] == "p", "Index 0 should decode to 'p'"
    assert POLARIZATION_DECODING[1] == "c", "Index 1 should decode to 'c'"
    assert POLARIZATION_DECODING[2] == "b", "Index 2 should decode to 'b'"
    assert POLARIZATION_DECODING[3] == "l", "Index 3 should decode to 'l'"
    assert POLARIZATION_DECODING[4] == "x", "Index 4 should decode to 'x'"
    assert POLARIZATION_DECODING[5] == "y", "Index 5 should decode to 'y'"
    
    # Test specific long name mappings
    expected_long_mappings = [
        ("p", "plus"),
        ("c", "cross"),
        ("b", "breathing"),
        ("l", "longitudinal"),
        ("x", "vector_x"),
        ("y", "vector_y"),
    ]
    
    for short, expected_long in expected_long_mappings:
        assert POLARIZATION_LONG_NAMES[short] == expected_long, \
            f"Short name '{short}' should map to '{expected_long}'"
        assert POLARIZATION_SHORT_NAMES[expected_long] == short, \
            f"Long name '{expected_long}' should map back to '{short}'"


def test_encoding_edge_cases():
    """Test encoding behavior with edge cases and boundary conditions."""
    # Test with single mode as both modes and basis
    modes_single = ["b"]
    basis_single = ["b"]
    
    modes_arr, basis_arr, derived_arr = encode_polarization(modes_single, basis_single)
    
    expected_modes_single = np.array([False, False, True, False, False, False])  # only breathing
    expected_basis_single = np.array([False, False, True, False, False, False])  # breathing as basis
    expected_derived_single = np.array([False, False, False, False, False, False])  # nothing derived
    
    assert np.array_equal(modes_arr, expected_modes_single), "Single mode case failed"
    assert np.array_equal(basis_arr, expected_basis_single), "Single basis case failed"
    assert np.array_equal(derived_arr, expected_derived_single), "Single derived case failed"
    
    # Test with empty basis (all modes are derived)
    modes_no_basis = ["p", "c"]
    basis_no_basis = []
    
    modes_arr_nb, basis_arr_nb, derived_arr_nb = encode_polarization(modes_no_basis, basis_no_basis)
    
    expected_modes_nb = np.array([True, True, False, False, False, False])    # p,c active
    expected_basis_nb = np.array([False, False, False, False, False, False])  # no basis
    expected_derived_nb = np.array([True, True, False, False, False, False])  # p,c derived
    
    assert np.array_equal(modes_arr_nb, expected_modes_nb), "No basis modes case failed"
    assert np.array_equal(basis_arr_nb, expected_basis_nb), "No basis basis case failed" 
    assert np.array_equal(derived_arr_nb, expected_derived_nb), "No basis derived case failed"
