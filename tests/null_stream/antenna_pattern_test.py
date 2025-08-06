"""Test module for detector antenna pattern functionality.

This module tests the computation of antenna response patterns for interferometers
across different polarization modes.
"""

from __future__ import annotations

import unittest

import numpy as np
from bilby.gw.detector import InterferometerList

from nullpol.null_stream import (
    get_antenna_pattern,
    get_antenna_pattern_matrix,
    get_collapsed_antenna_pattern_matrix,
    relative_amplification_factor_helper,
    relative_amplification_factor_map,
)


class TestAntennaPattern(unittest.TestCase):
    """Test class for detector antenna pattern computations.

    This class validates the calculation of antenna response patterns for
    multi-detector networks across various polarization modes, including
    polarization modes beyond General Relativity predictions.
    """

    def setUp(self):
        """Set up test interferometer network and antenna pattern calculations.

        Initializes a three-detector network (H1, L1, V1) and pre-computes
        reference antenna patterns for all six polarization modes (plus, cross,
        breathing, longitudinal, vector-x, vector-y) at a fixed sky location.
        """
        seed = 12
        np.random.seed(seed)
        self.interferometers = InterferometerList(["H1", "L1", "V1"])
        self.right_ascension = 0.5
        self.declination = 0.5
        self.polarization_angle = 0.5
        self.gps_time = 5
        self.antenna_pattern_matrix = np.zeros((3, 6))
        for i in range(len(self.interferometers)):
            antenna_pattern_p = self.interferometers[i].antenna_response(
                ra=self.right_ascension,
                dec=self.declination,
                time=self.gps_time,
                psi=self.polarization_angle,
                mode="plus",
            )
            antenna_pattern_c = self.interferometers[i].antenna_response(
                ra=self.right_ascension,
                dec=self.declination,
                time=self.gps_time,
                psi=self.polarization_angle,
                mode="cross",
            )
            antenna_pattern_b = self.interferometers[i].antenna_response(
                ra=self.right_ascension,
                dec=self.declination,
                time=self.gps_time,
                psi=self.polarization_angle,
                mode="breathing",
            )
            antenna_pattern_l = self.interferometers[i].antenna_response(
                ra=self.right_ascension,
                dec=self.declination,
                time=self.gps_time,
                psi=self.polarization_angle,
                mode="longitudinal",
            )
            antenna_pattern_x = self.interferometers[i].antenna_response(
                ra=self.right_ascension, dec=self.declination, time=self.gps_time, psi=self.polarization_angle, mode="x"
            )
            antenna_pattern_y = self.interferometers[i].antenna_response(
                ra=self.right_ascension, dec=self.declination, time=self.gps_time, psi=self.polarization_angle, mode="y"
            )
            self.antenna_pattern_matrix[i, 0] = antenna_pattern_p
            self.antenna_pattern_matrix[i, 1] = antenna_pattern_c
            self.antenna_pattern_matrix[i, 2] = antenna_pattern_b
            self.antenna_pattern_matrix[i, 3] = antenna_pattern_l
            self.antenna_pattern_matrix[i, 4] = antenna_pattern_x
            self.antenna_pattern_matrix[i, 5] = antenna_pattern_y

    def test_get_antenna_pattern(self):
        """Test single detector antenna pattern computation.

        Validates that the antenna pattern function correctly computes
        response patterns for individual detectors across all polarization
        modes, ensuring consistency with reference bilby calculations.
        """
        for i in range(len(self.interferometers)):
            antenna_pattern = get_antenna_pattern(
                self.interferometers[i],
                right_ascension=self.right_ascension,
                declination=self.declination,
                polarization_angle=self.polarization_angle,
                gps_time=self.gps_time,
                polarization=np.array([True, True, True, True, True, True]),
            )
            self.assertTrue(np.allclose(antenna_pattern[0], self.antenna_pattern_matrix[i, 0]))
            self.assertTrue(np.allclose(antenna_pattern[1], self.antenna_pattern_matrix[i, 1]))
            self.assertTrue(np.allclose(antenna_pattern[2], self.antenna_pattern_matrix[i, 2]))
            self.assertTrue(np.allclose(antenna_pattern[3], self.antenna_pattern_matrix[i, 3]))
            self.assertTrue(np.allclose(antenna_pattern[4], self.antenna_pattern_matrix[i, 4]))
            self.assertTrue(np.allclose(antenna_pattern[5], self.antenna_pattern_matrix[i, 5]))

    def test_get_antenna_pattern_matrix(self):
        """Test multi-detector antenna pattern matrix computation.

        Validates that the antenna pattern matrix function correctly computes
        response patterns for the entire detector network, organizing results
        in a matrix format suitable for null stream construction.
        """
        antenna_pattern_matrix = get_antenna_pattern_matrix(
            interferometers=self.interferometers,
            right_ascension=self.right_ascension,
            declination=self.declination,
            polarization_angle=self.polarization_angle,
            gps_time=self.gps_time,
            polarization=[True, True, True, True, True, True],
        )
        self.assertTrue(np.allclose(antenna_pattern_matrix, self.antenna_pattern_matrix))

    def test_relative_amplitification_factor_map(self):
        """Test relative amplification factor parameter mapping.

        Validates the creation of parameter name maps that connect
        polarization modes to their basis mode components.
        """
        polarization_basis = np.array([True, True, False, False, False, False])
        polarization_derived = np.array([False, False, True, True, True, True])
        expected_output = np.array([["bp", "bc"], ["lp", "lc"], ["xp", "xc"], ["yp", "yc"]])
        output = relative_amplification_factor_map(polarization_basis, polarization_derived)
        self.assertTrue(np.array_equal(expected_output, output))

    def test_relative_amplification_factor_helper(self):
        """Test relative amplification factor computation from parameters.

        Validates the conversion of amplitude and phase parameters into
        complex amplification factors for combining polarization modes with
        basis modes.
        """
        polarization_basis = np.array([True, True, False, False, False, False])
        polarization_derived = np.array([False, False, True, True, True, True])
        parameters_map = relative_amplification_factor_map(polarization_basis, polarization_derived)
        parameters = dict(
            amplitude_bp=np.random.randn(),
            phase_bp=np.random.randn(),
            amplitude_bc=np.random.randn(),
            phase_bc=np.random.randn(),
            amplitude_lp=np.random.randn(),
            phase_lp=np.random.randn(),
            amplitude_lc=np.random.randn(),
            phase_lc=np.random.randn(),
            amplitude_xp=np.random.randn(),
            phase_xp=np.random.randn(),
            amplitude_xc=np.random.randn(),
            phase_xc=np.random.randn(),
            amplitude_yp=np.random.randn(),
            phase_yp=np.random.randn(),
            amplitude_yc=np.random.randn(),
            phase_yc=np.random.randn(),
        )
        output = relative_amplification_factor_helper(parameters_map, parameters)
        expected_output = np.array(
            [
                [
                    parameters["amplitude_bp"] * np.exp(1.0j * parameters["phase_bp"]),
                    parameters["amplitude_bc"] * np.exp(1.0j * parameters["phase_bc"]),
                ],
                [
                    parameters["amplitude_lp"] * np.exp(1.0j * parameters["phase_lp"]),
                    parameters["amplitude_lc"] * np.exp(1.0j * parameters["phase_lc"]),
                ],
                [
                    parameters["amplitude_xp"] * np.exp(1.0j * parameters["phase_xp"]),
                    parameters["amplitude_xc"] * np.exp(1.0j * parameters["phase_xc"]),
                ],
                [
                    parameters["amplitude_yp"] * np.exp(1.0j * parameters["phase_yp"]),
                    parameters["amplitude_yc"] * np.exp(1.0j * parameters["phase_yc"]),
                ],
            ]
        )
        self.assertTrue(np.allclose(expected_output, output))

    def test_get_collapsed_antenna_pattern_matrix(self):
        """Test collapsed antenna pattern matrix.

        Validates the construction of effective antenna pattern matrices that
        combine basis polarization modes with other modes using relative
        amplification factors.
        """
        polarization_basis = np.array([True, True, False, False, False, False])
        polarization_derived = np.array([False, False, True, True, True, True])
        antenna_pattern_matrix = get_antenna_pattern_matrix(
            interferometers=self.interferometers,
            right_ascension=self.right_ascension,
            declination=self.declination,
            polarization_angle=self.polarization_angle,
            gps_time=self.gps_time,
            polarization=np.array([True, True, True, True, True, True]),
        )
        parameters_map = relative_amplification_factor_map(polarization_basis, polarization_derived)
        parameters = dict(
            amplitude_bp=np.random.randn(),
            phase_bp=np.random.randn(),
            amplitude_bc=np.random.randn(),
            phase_bc=np.random.randn(),
            amplitude_lp=np.random.randn(),
            phase_lp=np.random.randn(),
            amplitude_lc=np.random.randn(),
            phase_lc=np.random.randn(),
            amplitude_xp=np.random.randn(),
            phase_xp=np.random.randn(),
            amplitude_xc=np.random.randn(),
            phase_xc=np.random.randn(),
            amplitude_yp=np.random.randn(),
            phase_yp=np.random.randn(),
            amplitude_yc=np.random.randn(),
            phase_yc=np.random.randn(),
        )
        relative_amplification_factor = relative_amplification_factor_helper(parameters_map, parameters)
        output = get_collapsed_antenna_pattern_matrix(
            antenna_pattern_matrix=antenna_pattern_matrix,
            polarization_basis=polarization_basis,
            polarization_derived=polarization_derived,
            relative_amplification_factor=relative_amplification_factor,
        )
        expected_output = antenna_pattern_matrix[:, polarization_basis] + np.array(
            [
                [
                    antenna_pattern_matrix[0, 2] * relative_amplification_factor[0, 0]
                    + antenna_pattern_matrix[0, 3] * relative_amplification_factor[1, 0]
                    + antenna_pattern_matrix[0, 4] * relative_amplification_factor[2, 0]
                    + antenna_pattern_matrix[0, 5] * relative_amplification_factor[3, 0],
                    antenna_pattern_matrix[0, 2] * relative_amplification_factor[0, 1]
                    + antenna_pattern_matrix[0, 3] * relative_amplification_factor[1, 1]
                    + antenna_pattern_matrix[0, 4] * relative_amplification_factor[2, 1]
                    + antenna_pattern_matrix[0, 5] * relative_amplification_factor[3, 1],
                ],
                [
                    antenna_pattern_matrix[1, 2] * relative_amplification_factor[0, 0]
                    + antenna_pattern_matrix[1, 3] * relative_amplification_factor[1, 0]
                    + antenna_pattern_matrix[1, 4] * relative_amplification_factor[2, 0]
                    + antenna_pattern_matrix[1, 5] * relative_amplification_factor[3, 0],
                    antenna_pattern_matrix[1, 2] * relative_amplification_factor[0, 1]
                    + antenna_pattern_matrix[1, 3] * relative_amplification_factor[1, 1]
                    + antenna_pattern_matrix[1, 4] * relative_amplification_factor[2, 1]
                    + antenna_pattern_matrix[1, 5] * relative_amplification_factor[3, 1],
                ],
                [
                    antenna_pattern_matrix[2, 2] * relative_amplification_factor[0, 0]
                    + antenna_pattern_matrix[2, 3] * relative_amplification_factor[1, 0]
                    + antenna_pattern_matrix[2, 4] * relative_amplification_factor[2, 0]
                    + antenna_pattern_matrix[2, 5] * relative_amplification_factor[3, 0],
                    antenna_pattern_matrix[2, 2] * relative_amplification_factor[0, 1]
                    + antenna_pattern_matrix[2, 3] * relative_amplification_factor[1, 1]
                    + antenna_pattern_matrix[2, 4] * relative_amplification_factor[2, 1]
                    + antenna_pattern_matrix[2, 5] * relative_amplification_factor[3, 1],
                ],
            ]
        )
        self.assertTrue(np.allclose(expected_output, output))


if __name__ == "__main__":
    unittest.main()
