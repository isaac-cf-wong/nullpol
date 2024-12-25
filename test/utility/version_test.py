import unittest
from nullpol.utility import (get_version_information,
                             log_version_information)


class TestVersion(unittest.TestCase):
    def test_get_version_information(self):
        get_version_information()

    def test_log_version_information(self):
        log_version_information()

if __name__ == '__main__':
    unittest.main()