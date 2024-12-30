import unittest
import numpy as np
from nullpol.null_stream import (encode_polarization,
                                 get_long_names)


class TestEncoding(unittest.TestCase):
    def test_encode_polarization(self):
        polarization_modes = 'pcbxy'
        polarization_basis = 'pc'
        polarization_modes, polarization_basis, polarization_derived = encode_polarization(polarization_modes, polarization_basis)
        expected_polarization_modes = np.array([True, True, True, False, True, True])
        expected_polarization_basis = np.array([True, True, False, False, False, False])
        expected_polarization_derived = np.array([False, False, True, False, True, True])
        self.assertTrue(np.array_equal(polarization_modes, expected_polarization_modes))
        self.assertTrue(np.array_equal(polarization_basis, expected_polarization_basis))
        self.assertTrue(np.array_equal(polarization_derived, expected_polarization_derived))

    def test_get_long_names(self):
        tokens = 'xyblpc'
        output = get_long_names(tokens)
        expected_output = ['vector_x', 'vector_y', 'breathing', 'longitudinal', 'plus', 'cross']
        self.assertTrue(np.array_equal(output, expected_output))

if __name__ == '__main__':
    unittest.main()