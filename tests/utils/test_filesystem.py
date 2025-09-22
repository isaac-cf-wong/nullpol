"""Test module for filesystem utility functions.

This module tests basic filesystem operations used throughout the
package for file handling and validation.
"""

from __future__ import annotations

import tempfile

from nullpol.utils import get_file_extension, is_file


def test_get_file_extension():
    """Test file extension extraction from filenames.

    Validates that file extensions are correctly extracted from
    filenames.
    """
    filename = "filename.txt"
    extension = get_file_extension(filename)
    assert extension == ".txt"


def test_is_file():
    """Test file existence validation functionality.

    Validates that the file existence check correctly identifies
    existing and non-existing files.
    """
    with tempfile.NamedTemporaryFile() as temp_file:
        temp_file_name = temp_file.name
        assert is_file(temp_file_name) is True
    assert is_file(temp_file_name) is False
