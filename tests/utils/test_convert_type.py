"""Test module for type conversion utilities.

This module tests the type conversion functions used in the nullpol package.
Note: These functions are currently marked as legacy/unused but are kept
for potential future use or backwards compatibility.
"""

from __future__ import annotations

from nullpol.utils.convert_type import (
    convert_string_to_float,
    convert_string_to_int,
    convert_string_to_bool,
)


def test_convert_string_to_float():
    """Test string to float conversion."""
    assert convert_string_to_float("3.14") == 3.14
    assert convert_string_to_float("0.0") == 0.0
    assert convert_string_to_float("-2.5") == -2.5
    assert convert_string_to_float("42") == 42.0
    assert convert_string_to_float(None) is None


def test_convert_string_to_int():
    """Test string to int conversion."""
    assert convert_string_to_int("42") == 42
    assert convert_string_to_int("0") == 0
    assert convert_string_to_int("-10") == -10
    assert convert_string_to_int(None) is None


def test_convert_string_to_bool():
    """Test string to bool conversion."""
    # Any non-empty string returns True (standard Python bool behavior)
    assert convert_string_to_bool("true") is True
    assert convert_string_to_bool("false") is True  # Non-empty string = True
    assert convert_string_to_bool("") is False  # Empty string = False
    assert convert_string_to_bool(None) is None
