"""Test module for polarization encoding functionality.

This module tests the encoding and representation of different polarization
modes.
"""

from __future__ import annotations

import unittest

import numpy as np

from nullpol.null_stream import encode_polarization, get_long_names


class TestEncoding(unittest.TestCase):
    """Test class for polarization mode encoding and naming utilities.

    This class validates the proper encoding of polarization modes into
    binary arrays and the conversion between short and long polarization
    mode names.
    """

    def test_encode_polarization(self):
        """Test encoding of polarization modes into binary arrays.

        Validates that polarization mode strings are correctly encoded into
        boolean arrays distinguishing between basis modes, non-basis modes,
        and all available modes.
        """
        polarization_modes = "pcbxy"
        polarization_basis = "pc"
        polarization_modes, polarization_basis, polarization_derived = encode_polarization(
            polarization_modes, polarization_basis
        )
        expected_polarization_modes = np.array([True, True, True, False, True, True])
        expected_polarization_basis = np.array([True, True, False, False, False, False])
        expected_polarization_derived = np.array([False, False, True, False, True, True])
        self.assertTrue(np.array_equal(polarization_modes, expected_polarization_modes))
        self.assertTrue(np.array_equal(polarization_basis, expected_polarization_basis))
        self.assertTrue(np.array_equal(polarization_derived, expected_polarization_derived))

    def test_get_long_names(self):
        """Test conversion from short to long polarization mode names.

        Validates that single-character polarization mode identifiers are
        correctly expanded to their full descriptive names for improved
        readability in analysis outputs and documentation.
        """
        tokens = "xyblpc"
        output = get_long_names(tokens)
        expected_output = ["vector_x", "vector_y", "breathing", "longitudinal", "plus", "cross"]
        self.assertTrue(np.array_equal(output, expected_output))


if __name__ == "__main__":
    unittest.main()
