from __future__ import annotations

import tempfile
import unittest

from nullpol.utils import get_file_extension, is_file


class TestFilesystem(unittest.TestCase):
    def test_get_file_extension(self):
        filename = "filename.txt"
        extension = get_file_extension(filename)
        self.assertEqual(extension, ".txt")

    def test_is_file(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file_name = temp_file.name
            self.assertTrue(is_file(temp_file_name))
        self.assertFalse(is_file(temp_file_name))

if __name__ == '__main__':
    unittest.main()
