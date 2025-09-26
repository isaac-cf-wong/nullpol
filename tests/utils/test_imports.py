"""Test module for dynamic import utilities.

This module tests the dynamic import functions used in the nullpol package.
"""

from __future__ import annotations

import pytest

from nullpol.utils.imports import import_function


def test_import_function_builtin():
    """Test importing a built-in function."""
    # Import a built-in function
    len_func = import_function("builtins.len")
    assert len_func is len
    assert len_func([1, 2, 3]) == 3


def test_import_function_standard_library():
    """Test importing from standard library."""
    # Import from json module
    loads_func = import_function("json.loads")
    import json

    assert loads_func is json.loads
    assert loads_func('{"key": "value"}') == {"key": "value"}


def test_import_function_from_math():
    """Test importing from math module."""
    # Import sqrt from math
    sqrt_func = import_function("math.sqrt")
    import math

    assert sqrt_func is math.sqrt
    assert sqrt_func(16) == 4.0


def test_import_function_from_pathlib():
    """Test importing from pathlib."""
    # Import Path class from pathlib
    path_class = import_function("pathlib.Path")
    from pathlib import Path

    assert path_class is Path
    # Test that we can create an instance
    p = path_class("test.txt")
    assert isinstance(p, Path)


def test_import_function_invalid_module():
    """Test that importing from non-existent module raises ModuleNotFoundError."""
    with pytest.raises(ModuleNotFoundError):
        import_function("nonexistent.module.function")


def test_import_function_invalid_function():
    """Test that importing non-existent function raises AttributeError."""
    with pytest.raises(AttributeError):
        import_function("json.nonexistent_function")


def test_import_function_invalid_format():
    """Test that invalid path format raises ValueError."""
    with pytest.raises(ValueError):
        import_function("no_dots_here")
