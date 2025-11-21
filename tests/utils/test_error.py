"""Test module for error handling utilities.

This module tests the custom exception classes used throughout the
nullpol package for error handling and user feedback.
"""

from __future__ import annotations

from nullpol.utils.error import NullpolError


def test_nullpol_error_creation():
    """Test creating NullpolError with a message."""
    message = "This is a test error message"
    error = NullpolError(message)

    assert isinstance(error, Exception)
    assert isinstance(error, NullpolError)
    assert str(error) == message


def test_nullpol_error_inheritance():
    """Test that NullpolError inherits from Exception."""
    error = NullpolError("test")
    assert issubclass(NullpolError, Exception)
    assert isinstance(error, Exception)


def test_nullpol_error_raising():
    """Test that NullpolError can be raised and caught."""
    try:
        raise NullpolError("test error")
    except NullpolError as e:
        assert str(e) == "test error"
    except Exception:
        assert False, "Should have caught NullpolError specifically"
