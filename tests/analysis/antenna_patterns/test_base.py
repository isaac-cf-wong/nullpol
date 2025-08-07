"""Test module for antenna pattern base functionality.

This module tests antenna pattern computations, relative amplification factors,
and collapsed antenna pattern matrix operations.
"""

from __future__ import annotations

import numpy as np
import pytest
from bilby.gw.detector import InterferometerList

from nullpol.analysis.antenna_patterns import (
    get_antenna_pattern,
    get_antenna_pattern_matrix,
    get_collapsed_antenna_pattern_matrix,
    relative_amplification_factor_helper,
    relative_amplification_factor_map,
)


@pytest.fixture
def antenna_pattern_setup():
    """Set up test interferometer network and antenna pattern calculations.

    Initializes a three-detector network (H1, L1, V1) and pre-computes
    reference antenna patterns for all six polarization modes (plus, cross,
    breathing, longitudinal, vector-x, vector-y) at a fixed sky location.

    Returns:
        tuple: Contains interferometers, sky position, and reference patterns
    """
    seed = 12
    np.random.seed(seed)
    interferometers = InterferometerList(["H1", "L1", "V1"])
    right_ascension = 0.5
    declination = 0.5
    polarization_angle = 0.5
    gps_time = 5
    antenna_pattern_matrix = np.zeros((3, 6))
    for i in range(len(interferometers)):
        antenna_pattern_p = interferometers[i].antenna_response(
            ra=right_ascension,
            dec=declination,
            time=gps_time,
            psi=polarization_angle,
            mode="plus",
        )
        antenna_pattern_c = interferometers[i].antenna_response(
            ra=right_ascension,
            dec=declination,
            time=gps_time,
            psi=polarization_angle,
            mode="cross",
        )
        antenna_pattern_b = interferometers[i].antenna_response(
            ra=right_ascension,
            dec=declination,
            time=gps_time,
            psi=polarization_angle,
            mode="breathing",
        )
        antenna_pattern_l = interferometers[i].antenna_response(
            ra=right_ascension,
            dec=declination,
            time=gps_time,
            psi=polarization_angle,
            mode="longitudinal",
        )
        antenna_pattern_x = interferometers[i].antenna_response(
            ra=right_ascension, dec=declination, time=gps_time, psi=polarization_angle, mode="x"
        )
        antenna_pattern_y = interferometers[i].antenna_response(
            ra=right_ascension, dec=declination, time=gps_time, psi=polarization_angle, mode="y"
        )
        antenna_pattern_matrix[i, 0] = antenna_pattern_p
        antenna_pattern_matrix[i, 1] = antenna_pattern_c
        antenna_pattern_matrix[i, 2] = antenna_pattern_b
        antenna_pattern_matrix[i, 3] = antenna_pattern_l
        antenna_pattern_matrix[i, 4] = antenna_pattern_x
        antenna_pattern_matrix[i, 5] = antenna_pattern_y

    return (interferometers, right_ascension, declination, polarization_angle, gps_time, antenna_pattern_matrix)


@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set up test environment with deterministic random seed.

    Initializes the random number generator with a fixed seed to ensure
    reproducible test results for antenna pattern computations.
    """
    seed = 12
    np.random.seed(seed)


# =============================================================================
# ANTENNA PATTERN TESTS
# =============================================================================


def test_get_antenna_pattern(antenna_pattern_setup):
    """Test single detector antenna pattern computation.

    Validates that the antenna pattern function correctly computes
    response patterns for individual detectors across all polarization
    modes, ensuring consistency with reference bilby calculations.
    """
    (interferometers, right_ascension, declination, polarization_angle, gps_time, antenna_pattern_matrix) = (
        antenna_pattern_setup
    )

    for i in range(len(interferometers)):
        antenna_pattern = get_antenna_pattern(
            interferometers[i],
            right_ascension=right_ascension,
            declination=declination,
            polarization_angle=polarization_angle,
            gps_time=gps_time,
            polarization=np.array([True, True, True, True, True, True]),
        )
        assert np.allclose(antenna_pattern[0], antenna_pattern_matrix[i, 0])
        assert np.allclose(antenna_pattern[1], antenna_pattern_matrix[i, 1])
        assert np.allclose(antenna_pattern[2], antenna_pattern_matrix[i, 2])
        assert np.allclose(antenna_pattern[3], antenna_pattern_matrix[i, 3])
        assert np.allclose(antenna_pattern[4], antenna_pattern_matrix[i, 4])
        assert np.allclose(antenna_pattern[5], antenna_pattern_matrix[i, 5])


def test_get_antenna_pattern_matrix(antenna_pattern_setup):
    """Test multi-detector antenna pattern matrix computation.

    Validates that the antenna pattern matrix function correctly computes
    response patterns for the entire detector network, organizing results
    in a matrix format suitable for null stream construction.
    """
    (interferometers, right_ascension, declination, polarization_angle, gps_time, antenna_pattern_matrix) = (
        antenna_pattern_setup
    )

    computed_antenna_pattern_matrix = get_antenna_pattern_matrix(
        interferometers=interferometers,
        right_ascension=right_ascension,
        declination=declination,
        polarization_angle=polarization_angle,
        gps_time=gps_time,
        polarization=[True, True, True, True, True, True],
    )
    assert np.allclose(computed_antenna_pattern_matrix, antenna_pattern_matrix)


def test_get_collapsed_antenna_pattern_matrix(antenna_pattern_setup):
    """Test collapsed antenna pattern matrix.

    Validates the construction of effective antenna pattern matrices that
    combine basis polarization modes with other modes using relative
    amplification factors.
    """
    (interferometers, right_ascension, declination, polarization_angle, gps_time, antenna_pattern_matrix) = (
        antenna_pattern_setup
    )

    polarization_basis = np.array([True, True, False, False, False, False])
    polarization_derived = np.array([False, False, True, True, True, True])
    computed_antenna_pattern_matrix = get_antenna_pattern_matrix(
        interferometers=interferometers,
        right_ascension=right_ascension,
        declination=declination,
        polarization_angle=polarization_angle,
        gps_time=gps_time,
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
        antenna_pattern_matrix=computed_antenna_pattern_matrix,
        polarization_basis=polarization_basis,
        polarization_derived=polarization_derived,
        relative_amplification_factor=relative_amplification_factor,
    )
    expected_output = computed_antenna_pattern_matrix[:, polarization_basis] + np.array(
        [
            [
                computed_antenna_pattern_matrix[0, 2] * relative_amplification_factor[0, 0]
                + computed_antenna_pattern_matrix[0, 3] * relative_amplification_factor[1, 0]
                + computed_antenna_pattern_matrix[0, 4] * relative_amplification_factor[2, 0]
                + computed_antenna_pattern_matrix[0, 5] * relative_amplification_factor[3, 0],
                computed_antenna_pattern_matrix[0, 2] * relative_amplification_factor[0, 1]
                + computed_antenna_pattern_matrix[0, 3] * relative_amplification_factor[1, 1]
                + computed_antenna_pattern_matrix[0, 4] * relative_amplification_factor[2, 1]
                + computed_antenna_pattern_matrix[0, 5] * relative_amplification_factor[3, 1],
            ],
            [
                computed_antenna_pattern_matrix[1, 2] * relative_amplification_factor[0, 0]
                + computed_antenna_pattern_matrix[1, 3] * relative_amplification_factor[1, 0]
                + computed_antenna_pattern_matrix[1, 4] * relative_amplification_factor[2, 0]
                + computed_antenna_pattern_matrix[1, 5] * relative_amplification_factor[3, 0],
                computed_antenna_pattern_matrix[1, 2] * relative_amplification_factor[0, 1]
                + computed_antenna_pattern_matrix[1, 3] * relative_amplification_factor[1, 1]
                + computed_antenna_pattern_matrix[1, 4] * relative_amplification_factor[2, 1]
                + computed_antenna_pattern_matrix[1, 5] * relative_amplification_factor[3, 1],
            ],
            [
                computed_antenna_pattern_matrix[2, 2] * relative_amplification_factor[0, 0]
                + computed_antenna_pattern_matrix[2, 3] * relative_amplification_factor[1, 0]
                + computed_antenna_pattern_matrix[2, 4] * relative_amplification_factor[2, 0]
                + computed_antenna_pattern_matrix[2, 5] * relative_amplification_factor[3, 0],
                computed_antenna_pattern_matrix[2, 2] * relative_amplification_factor[0, 1]
                + computed_antenna_pattern_matrix[2, 3] * relative_amplification_factor[1, 1]
                + computed_antenna_pattern_matrix[2, 4] * relative_amplification_factor[2, 1]
                + computed_antenna_pattern_matrix[2, 5] * relative_amplification_factor[3, 1],
            ],
        ]
    )
    assert np.allclose(expected_output, output)


# =============================================================================
# RELATIVE AMPLIFICATION FACTOR TESTS
# =============================================================================


def test_relative_amplitification_factor_map():
    """Test relative amplification factor parameter mapping.

    Validates the creation of parameter name maps that connect
    polarization modes to their basis mode components.
    """
    polarization_basis = np.array([True, True, False, False, False, False])
    polarization_derived = np.array([False, False, True, True, True, True])
    expected_output = np.array([["bp", "bc"], ["lp", "lc"], ["xp", "xc"], ["yp", "yc"]])
    output = relative_amplification_factor_map(polarization_basis, polarization_derived)
    assert np.array_equal(expected_output, output)


def test_relative_amplification_factor_helper():
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
    assert np.allclose(expected_output, output)
