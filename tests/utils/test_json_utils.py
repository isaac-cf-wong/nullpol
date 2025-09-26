"""Test module for JSON utilities.

This module tests the JSON parsing functions used in the nullpol package.
"""

from __future__ import annotations

import json
import pytest

from nullpol.utils.json_utils import json_loads_with_none


def test_json_loads_with_none_basic():
    """Test basic JSON parsing without None values."""
    # Regular JSON should work normally
    result = json_loads_with_none('{"key": "value", "number": 42}')
    expected = {"key": "value", "number": 42}
    assert result == expected


def test_json_loads_with_none_with_none_values():
    """Test JSON parsing with Python None values."""
    # JSON with None values (Python style)
    result = json_loads_with_none('{"key": None, "other": "value"}')
    expected = {"key": None, "other": "value"}
    assert result == expected


def test_json_loads_with_none_mixed_nulls():
    """Test JSON with both null and None values."""
    # Mix of null and None
    result = json_loads_with_none('{"none_key": None, "null_key": null, "value": 42}')
    expected = {"none_key": None, "null_key": None, "value": 42}
    assert result == expected


def test_json_loads_with_none_array():
    """Test JSON array with None values."""
    result = json_loads_with_none("[1, None, 3, None, 5]")
    expected = [1, None, 3, None, 5]
    assert result == expected


def test_json_loads_with_none_nested():
    """Test nested JSON structures with None values."""
    json_str = """
    {
        "outer": {
            "inner": None,
            "list": [1, None, 3],
            "nested": {
                "deep": None
            }
        },
        "top_level": None
    }
    """
    result = json_loads_with_none(json_str)
    expected = {"outer": {"inner": None, "list": [1, None, 3], "nested": {"deep": None}}, "top_level": None}
    assert result == expected


def test_json_loads_with_none_empty_objects():
    """Test empty objects and arrays."""
    # Empty object
    result = json_loads_with_none("{}")
    assert result == {}

    # Empty array
    result = json_loads_with_none("[]")
    assert result == []


def test_json_loads_with_none_invalid_json():
    """Test that invalid JSON still raises JSONDecodeError."""
    with pytest.raises(json.JSONDecodeError):
        json_loads_with_none('{"invalid": json}')

    with pytest.raises(json.JSONDecodeError):
        json_loads_with_none('{"unclosed": "string}')


def test_json_loads_with_none_boolean_values():
    """Test JSON with boolean values."""
    result = json_loads_with_none('{"true_val": true, "false_val": false, "none_val": None}')
    expected = {"true_val": True, "false_val": False, "none_val": None}
    assert result == expected


def test_json_loads_with_none_numeric_values():
    """Test JSON with various numeric values."""
    result = json_loads_with_none('{"int": 42, "float": 3.14, "negative": -10, "none_num": None}')
    expected = {"int": 42, "float": 3.14, "negative": -10, "none_num": None}
    assert result == expected
