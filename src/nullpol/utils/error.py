"""Error module."""

from __future__ import annotations


class NullpolError(Exception):
    """Errors in nullpol.

    Args:
        message (str): Message.
    """

    def __init__(self, message):
        """Initialize the instance."""
        super().__init__(message)
