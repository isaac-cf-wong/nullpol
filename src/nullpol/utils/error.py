from __future__ import annotations


class NullpolError(Exception):
    """Errors in nullpol.

    Args:
        message (str): Message.
    """

    def __init__(self, message):
        super().__init__(message)
