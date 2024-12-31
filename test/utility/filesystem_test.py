import unittest
from nullpol.utility import get_file_extension


class TestFilesystem(unittest.TestCase):
    def test_get_file_extension(self):
        filename = "filename.txt"
        extension = get_file_extension(filename)
        self.assertEqual(extension, ".txt")

if __name__ == '__main__':
    unittest.main()