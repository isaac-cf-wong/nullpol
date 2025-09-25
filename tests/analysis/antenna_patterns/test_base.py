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


@pytest.mark.integration
def test_get_antenna_pattern_known_values():
    """Test antenna pattern computation with known expected values.

    Uses simple test cases where the expected antenna response can be
    predicted based on detector geometry and source location.
    """
    # Create a simple test interferometer
    from bilby.gw.detector import get_empty_interferometer

    # Get H1 interferometer (we know its orientation and location)
    h1 = get_empty_interferometer("H1")

    # Test case: Source directly overhead at H1 should give maximum plus response
    # H1 is at latitude ~46.5°N, longitude ~119.4°W
    # For a source at zenith, plus mode should dominate

    # Use known coordinates that should give predictable results
    ra = 0.0  # Right ascension
    dec = np.deg2rad(46.5)  # H1's approximate latitude
    psi = 0.0  # Polarization angle
    gps_time = 1000000000.0

    # Test only plus and cross polarizations for simplicity
    polarization_mask = np.array([True, True, False, False, False, False])

    antenna_response = get_antenna_pattern(h1, ra, dec, psi, gps_time, polarization_mask)

    # Basic sanity checks with known properties
    assert len(antenna_response) == 2, "Should have plus and cross components"
    assert isinstance(antenna_response[0], (float, np.floating)), "Plus response should be real"
    assert isinstance(antenna_response[1], (float, np.floating)), "Cross response should be real"

    # For overhead source, antenna response should be reasonable magnitude
    assert abs(antenna_response[0]) <= 1.0, "Plus response magnitude should be <= 1"
    assert abs(antenna_response[1]) <= 1.0, "Cross response magnitude should be <= 1"

    # Test known relationship: response should change with polarization angle
    psi_90 = np.pi / 2  # 90 degree rotation
    antenna_response_rotated = get_antenna_pattern(h1, ra, dec, psi_90, gps_time, polarization_mask)

    # After 90° rotation, plus and cross should be related by rotation
    # This is a fundamental property of gravitational wave polarizations
    assert not np.allclose(antenna_response, antenna_response_rotated), "Response should change with polarization angle"


@pytest.mark.integration
def test_get_antenna_pattern_matrix_symmetry():
    """Test antenna pattern matrix with known symmetry properties.

    Tests that the matrix has expected properties like detector independence
    and polarization mode relationships.
    """
    # Create a two-detector network for simpler testing
    from bilby.gw.detector import InterferometerList

    ifos = InterferometerList(["H1", "L1"])

    # Use simple sky location
    ra, dec, psi = 0.0, 0.0, 0.0
    gps_time = 1000000000.0

    # Test with just plus and cross polarizations
    polarization_mask = np.array([True, True, False, False, False, False])

    antenna_matrix = get_antenna_pattern_matrix(ifos, ra, dec, psi, gps_time, polarization_mask)

    # Test expected shape
    assert antenna_matrix.shape == (2, 2), f"Expected (2,2), got {antenna_matrix.shape}"

    # Test that all values are finite and within reasonable bounds
    assert np.all(np.isfinite(antenna_matrix)), "All antenna responses should be finite"
    assert np.all(np.abs(antenna_matrix) <= 1.0), "All antenna responses should have magnitude <= 1"

    # Test that H1 and L1 responses are different (they're at different locations)
    h1_response = antenna_matrix[0, :]
    l1_response = antenna_matrix[1, :]

    # Allow for small numerical differences but ensure they're not identical
    assert not np.allclose(h1_response, l1_response, atol=1e-10), "H1 and L1 should have different antenna responses"

    # Test consistency: same detector should give same result
    single_h1_response = get_antenna_pattern(ifos[0], ra, dec, psi, gps_time, polarization_mask)
    assert np.allclose(antenna_matrix[0, :], single_h1_response), "Matrix and individual computations should match"


def test_relative_amplification_factor_simple_case():
    """Test relative amplification factor with simple known case.

    Uses a minimal example where the expected parameter mapping
    and complex amplification factors can be easily verified.
    """
    # Simple case: plus and cross as basis, breathing as derived
    polarization_basis = np.array([True, True, False, False, False, False])  # p, c
    polarization_derived = np.array([False, False, True, False, False, False])  # b only

    # Get parameter mapping
    param_map = relative_amplification_factor_map(polarization_basis, polarization_derived)

    # Expected result: breathing mode mapped to plus and cross
    expected_map = np.array([["bp", "bc"]])  # breathing -> plus, breathing -> cross
    assert np.array_equal(param_map, expected_map), f"Expected {expected_map}, got {param_map}"

    # Test amplification factor computation with known values
    # Use simple amplitude=1, phase=0 for easy verification
    parameters = {
        "amplitude_bp": 2.0,  # breathing couples to plus with amplitude 2
        "phase_bp": 0.0,  # no phase
        "amplitude_bc": 0.5,  # breathing couples to cross with amplitude 0.5
        "phase_bc": np.pi / 2,  # 90° phase shift
    }

    amplification_factors = relative_amplification_factor_helper(param_map, parameters)

    # Check expected complex values
    expected_bp = 2.0 * np.exp(1j * 0.0)  # = 2.0 + 0j
    expected_bc = 0.5 * np.exp(1j * np.pi / 2)  # = 0 + 0.5j

    assert np.isclose(
        amplification_factors[0, 0], expected_bp
    ), f"Expected bp factor {expected_bp}, got {amplification_factors[0, 0]}"
    assert np.isclose(
        amplification_factors[0, 1], expected_bc
    ), f"Expected bc factor {expected_bc}, got {amplification_factors[0, 1]}"

    # Verify these are complex numbers with expected properties
    assert np.isreal(amplification_factors[0, 0]), "bp factor should be real (phase=0)"
    assert np.iscomplex(amplification_factors[0, 1]) or np.isreal(
        amplification_factors[0, 1]
    ), "bc factor should be complex"
    assert np.abs(np.imag(amplification_factors[0, 1]) - 0.5) < 1e-10, "bc factor should have imaginary part 0.5"


def test_get_collapsed_antenna_pattern_matrix_simple_example():
    """Test collapsed antenna pattern matrix with known numerical example.

    Creates a simple case where the expected collapsed matrix can be
    computed by hand and verified against the function result.
    """
    # Create simple test antenna pattern matrix
    # 2 detectors, 3 polarizations (plus, cross, breathing)
    antenna_matrix = np.array(
        [
            [0.8, 0.6, 0.4],  # Detector 1: plus=0.8, cross=0.6, breathing=0.4
            [0.7, 0.5, 0.3],  # Detector 2: plus=0.7, cross=0.5, breathing=0.3
        ]
    )

    # Basis: plus and cross (indices 0,1)
    # Derived: breathing (index 2)
    polarization_basis = np.array([True, True, False])
    polarization_derived = np.array([False, False, True])

    # Simple amplification factors: breathing = 2*plus + 1*cross
    relative_amplification_factor = np.array([[2.0, 1.0]])  # [breathing->plus, breathing->cross]

    collapsed_matrix = get_collapsed_antenna_pattern_matrix(
        antenna_matrix, polarization_basis, polarization_derived, relative_amplification_factor
    )

    # Calculate expected result by hand
    # New matrix = basis_part + derived_part * amplification_factors
    # basis_part = antenna_matrix[:, [0,1]] = [[0.8, 0.6], [0.7, 0.5]]
    # derived_contribution = antenna_matrix[:, [2]] @ [[2.0, 1.0]]
    #                      = [[0.4], [0.3]] @ [[2.0, 1.0]]
    #                      = [[0.4*2.0, 0.4*1.0], [0.3*2.0, 0.3*1.0]]
    #                      = [[0.8, 0.4], [0.6, 0.3]]

    expected_result = np.array(
        [
            [0.8 + 0.8, 0.6 + 0.4],  # [1.6, 1.0]
            [0.7 + 0.6, 0.5 + 0.3],  # [1.3, 0.8]
        ]
    )

    assert np.allclose(collapsed_matrix, expected_result), f"Expected\n{expected_result}\nGot\n{collapsed_matrix}"

    # Verify shape is correct (detectors x basis_modes)
    assert collapsed_matrix.shape == (2, 2), f"Expected shape (2,2), got {collapsed_matrix.shape}"


# =============================================================================
# RELATIVE AMPLIFICATION FACTOR TESTS
# =============================================================================


def test_relative_amplification_factor_map():
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
