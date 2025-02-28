import unittest
from nullpol.utils import setup_logger


class TestLog(unittest.TestCase):
    def test_setup_logger(self):
        setup_logger()

if __name__ == '__main__':
    unittest.main()