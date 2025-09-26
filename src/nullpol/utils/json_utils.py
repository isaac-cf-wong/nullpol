"""Utilities for JSON parsing and handling."""

from __future__ import annotations

import json


def json_loads_with_none(value: str):
    """Parse JSON string that may contain Python None values.

    Args:
        value: JSON string that may contain 'None' instead of 'null'

    Returns:
        Parsed JSON object
    """
    # Replace 'None' with 'null' to make it valid JSON
    value = value.replace("None", "null")
    return json.loads(value)
