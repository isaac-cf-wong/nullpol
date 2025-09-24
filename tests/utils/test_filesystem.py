"""Test module for filesystem utility functions.

This module tests basic filesystem operations used throughout the
package for file handling and validation.
"""

from __future__ import annotations

import tempfile

from nullpol.utils import get_file_extension, is_file


def test_get_file_extension():
    """Test file extension extraction from filenames."""
    assert get_file_extension("filename.txt") == ".txt"
    assert get_file_extension("document.pdf") == ".pdf"
    assert get_file_extension("archive.tar.gz") == ".gz"
    assert get_file_extension("README") == ""


def test_is_file():
    """Test file existence validation functionality."""
    with tempfile.NamedTemporaryFile() as temp_file:
        assert is_file(temp_file.name) is True

    assert is_file("/nonexistent/path/file.txt") is False

    with tempfile.TemporaryDirectory() as temp_dir:
        assert is_file(temp_dir) is False
