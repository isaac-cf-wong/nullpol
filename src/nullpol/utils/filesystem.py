from __future__ import annotations

from pathlib import Path


def get_file_extension(file_path):
    """Get the file extension.

    Args:
        file_path (str): Path to the file.

    Returns:
        str: Extension of the file.
    """
    return Path(file_path).suffix


def is_file(file_path):
    """Check if the file exists.

    Args:
        file_path (str): Path to the file.

    Returns:
        bool: True if the file exists, False otherwise.
    """
    return Path(file_path).is_file()


# ======================================================================
# UNUSED FUNCTIONS - LEGACY CODE
# ======================================================================
# The following functions are not currently used in the codebase but are
# kept for potential future use or backwards compatibility. They are only
# exported through the utils module but not actively utilized.


def get_absolute_path(file_path):
    """Get absolute path.

    Args:
        file_path (str): Path to the file.

    Returns:
    str: Absolute path.
    """
    return str(Path(file_path).resolve())
