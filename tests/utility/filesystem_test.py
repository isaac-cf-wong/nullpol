"""Test module for filesystem utility functions.

This module tests basic filesystem operations used throughout the
package for file handling and validation.
"""

from __future__ import annotations

import tempfile
import unittest

from nullpol.utils import get_file_extension, is_file


class TestFilesystem(unittest.TestCase):
    """Test class for filesystem utility functions.

    This class validates the implementation of basic file system operations
    including file extension extraction and file existence checking.
    """

    def test_get_file_extension(self):
        """Test file extension extraction from filenames.

        Validates that file extensions are correctly extracted from
        filenames.
        """
        filename = "filename.txt"
        extension = get_file_extension(filename)
        self.assertEqual(extension, ".txt")

    def test_is_file(self):
        """Test file existence validation functionality.

        Validates that the file existence check correctly identifies
        existing and non-existing files.
        """
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file_name = temp_file.name
            self.assertTrue(is_file(temp_file_name))
        self.assertFalse(is_file(temp_file_name))


if __name__ == "__main__":
    unittest.main()
