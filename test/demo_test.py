import unittest


class TestDemo(unittest.TestCase):
    def test_number(self):
        self.assertGreater(1, 0)

if __name__ == '__main__':
    unittest.main()
