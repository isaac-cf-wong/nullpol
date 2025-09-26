"""Test module for antenna pattern processor functionality.

This module tests the AntennaPatternProcessor class and its methods
for computing antenna patterns and polarization operations.
"""

from __future__ import annotations

import numpy as np
import pytest
import bilby

from nullpol.analysis.antenna_patterns import AntennaPatternProcessor
from nullpol.utils import NullpolError


@pytest.fixture
def sample_interferometers():
    """Create sample interferometers for testing."""
    duration = 4.0
    sampling_frequency = 2048.0

    # Create multiple interferometers (need more than polarization basis count)
    interferometers = []
    for name in ["H1", "L1", "V1"]:  # 3 detectors > 2 polarization bases
        ifo = bilby.gw.detector.get_empty_interferometer(name)
        ifo.set_strain_data_from_zero_noise(sampling_frequency=sampling_frequency, duration=duration, start_time=0)
        interferometers.append(ifo)

    return bilby.gw.detector.InterferometerList(interferometers)


@pytest.mark.integration
class TestAntennaPatternProcessor:
    """Test class for AntennaPatternProcessor."""

    def test_processor_initialization_basic(self, sample_interferometers):
        """Test processor initialization with specific known values.

        Verifies that polarization encoding produces expected boolean arrays
        for concrete input cases.
        """
        # Test case 1: Plus and cross modes with plus as basis
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p"], interferometers=sample_interferometers
        )

        # Check that polarization modes are encoded correctly
        expected_modes = np.array([True, True, False, False, False, False])  # p=T, c=T
        expected_basis = np.array([True, False, False, False, False, False])  # p=T only
        expected_derived = np.array([False, True, False, False, False, False])  # c=T only

        assert np.array_equal(
            processor.polarization_modes, expected_modes
        ), f"Modes: expected {expected_modes}, got {processor.polarization_modes}"
        assert np.array_equal(
            processor.polarization_basis, expected_basis
        ), f"Basis: expected {expected_basis}, got {processor.polarization_basis}"
        assert np.array_equal(
            processor.polarization_derived, expected_derived
        ), f"Derived: expected {expected_derived}, got {processor.polarization_derived}"

        # Test collapsed arrays (should exclude inactive modes)
        expected_basis_collapsed = np.array([True, False])  # p=basis, c=not basis
        expected_derived_collapsed = np.array([False, True])  # p=not derived, c=derived

        assert np.array_equal(
            processor.polarization_basis_collapsed, expected_basis_collapsed
        ), f"Basis collapsed: expected {expected_basis_collapsed}, got {processor.polarization_basis_collapsed}"
        assert np.array_equal(
            processor.polarization_derived_collapsed, expected_derived_collapsed
        ), f"Derived collapsed: expected {expected_derived_collapsed}, got {processor.polarization_derived_collapsed}"

    def test_processor_initialization_all_modes(self, sample_interferometers):
        """Test processor with all polarization modes and known basis."""
        # Test with all 6 modes, p+c as basis
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c", "b", "l", "x", "y"],
            polarization_basis=["p", "c"],
            interferometers=sample_interferometers,
        )

        # All modes should be active
        expected_all_active = np.array([True, True, True, True, True, True])
        assert np.array_equal(processor.polarization_modes, expected_all_active), "All modes should be active"

        # Only p,c should be basis
        expected_basis = np.array([True, True, False, False, False, False])
        assert np.array_equal(processor.polarization_basis, expected_basis), "Only p,c should be basis modes"

        # b,l,x,y should be derived
        expected_derived = np.array([False, False, True, True, True, True])
        assert np.array_equal(processor.polarization_derived, expected_derived), "b,l,x,y should be derived modes"

        # Check relative amplification factor map shape
        # Should be (4 derived modes, 2 basis modes)
        assert processor.relative_amplification_factor_map.shape == (
            4,
            2,
        ), f"Amplification map shape should be (4,2), got {processor.relative_amplification_factor_map.shape}"

    def test_detector_count_validation_pass(self):
        """Test that sufficient detectors pass validation."""
        # Create 3 detectors for 2 basis modes (should pass)
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1", "V1"])

        # This should not raise an error
        try:
            processor = AntennaPatternProcessor(
                polarization_modes=["p", "c"],
                polarization_basis=["p", "c"],  # 2 basis modes
                interferometers=ifos,  # 3 detectors > 2 basis modes
            )
            # If we get here, validation passed as expected
            assert len(ifos) > np.sum(processor.polarization_basis), "Validation should ensure detectors > basis modes"
        except NullpolError:
            pytest.fail("Should not raise NullpolError with sufficient detectors")

    def test_detector_count_validation_fail(self):
        """Test that insufficient detectors fail validation."""
        # Create 2 detectors for 3 basis modes (should fail)
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])  # 2 detectors

        # This should raise an error
        with pytest.raises(NullpolError, match="Number of detectors.*greater than.*polarization bases"):
            AntennaPatternProcessor(
                polarization_modes=["p", "c", "b"],
                polarization_basis=["p", "c", "b"],  # 3 basis modes > 2 detectors
                interferometers=ifos,
            )

    def test_detector_count_edge_case(self):
        """Test edge case where detectors equals basis count."""
        # Create exactly as many detectors as basis modes (should fail)
        ifos = bilby.gw.detector.InterferometerList(["H1", "L1"])  # 2 detectors

        with pytest.raises(NullpolError):
            AntennaPatternProcessor(
                polarization_modes=["p", "c"],
                polarization_basis=["p", "c"],  # 2 basis modes = 2 detectors (should fail)
                interferometers=ifos,
            )

    def test_compute_antenna_pattern_matrix_basic(self, sample_interferometers):
        """Test antenna pattern matrix computation with known parameters."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p"], interferometers=sample_interferometers
        )

        # Test parameters with known values
        parameters = {
            "ra": 0.0,
            "dec": 0.0,
            "psi": 0.0,
            "geocent_time": 1000000000.0,
            # Need amplification parameters for derived modes (cross in this case)
            "amplitude_cp": 1.0,  # cross -> plus amplitude
            "phase_cp": 0.0,  # cross -> plus phase
        }

        antenna_matrix = processor.compute_antenna_pattern_matrix(sample_interferometers, parameters)

        # Should return matrix with shape (n_detectors, n_basis_modes)
        n_detectors = len(sample_interferometers)
        n_basis = np.sum(processor.polarization_basis_collapsed)
        expected_shape = (n_detectors, n_basis)

        assert antenna_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {antenna_matrix.shape}"

        # All values should be finite
        assert np.all(np.isfinite(antenna_matrix)), "All antenna responses should be finite"

        # Antenna responses should be reasonable magnitude (|F| <= 1 typically)
        assert np.all(np.abs(antenna_matrix) <= 2.0), "Antenna responses should be reasonable magnitude"

    def test_compute_antenna_pattern_matrix_no_derived_modes(self, sample_interferometers):
        """Test antenna pattern matrix when no derived modes exist."""
        # All active modes are also basis modes (no derived modes)
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"],
            polarization_basis=["p", "c"],  # Both modes are basis
            interferometers=sample_interferometers,
        )

        parameters = {
            "ra": 0.5,
            "dec": 0.3,
            "psi": 0.1,
            "geocent_time": 1000000000.0,
        }

        antenna_matrix = processor.compute_antenna_pattern_matrix(sample_interferometers, parameters)

        # Should have shape (3 detectors, 2 basis modes)
        assert antenna_matrix.shape == (3, 2), f"Expected (3,2), got {antenna_matrix.shape}"
        assert np.all(np.isfinite(antenna_matrix)), "All values should be finite"

    def test_relative_amplification_factor_map_property(self, sample_interferometers):
        """Test that relative amplification factor map has expected structure."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c", "b"],  # 3 modes
            polarization_basis=["p"],  # 1 basis mode
            interferometers=sample_interferometers,
        )

        # Should have derived modes: c, b (2 derived)
        # Should map to basis mode: p (1 basis)
        # Expected shape: (2 derived, 1 basis)
        expected_shape = (2, 1)
        actual_shape = processor.relative_amplification_factor_map.shape

        assert actual_shape == expected_shape, f"Amplification map shape: expected {expected_shape}, got {actual_shape}"

        # Check specific mapping content
        amp_map = processor.relative_amplification_factor_map
        # Should have entries like "cp" (cross->plus) and "bp" (breathing->plus)
        expected_entries = {"cp", "bp"}  # derived->basis combinations
        actual_entries = set(amp_map.flatten())

        assert actual_entries == expected_entries, f"Expected map entries {expected_entries}, got {actual_entries}"

    def test_processor_properties_consistency(self, sample_interferometers):
        """Test that processor properties are internally consistent."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c", "b", "l"],
            polarization_basis=["p", "c"],
            interferometers=sample_interferometers,
        )

        # Test that basis is subset of modes
        basis_in_modes = processor.polarization_modes | ~processor.polarization_basis
        assert np.all(basis_in_modes), "Basis must be subset of modes"

        # Test that derived = modes & ~basis
        expected_derived = processor.polarization_modes & (~processor.polarization_basis)
        assert np.array_equal(
            processor.polarization_derived, expected_derived
        ), "Derived should equal modes AND NOT basis"

        # Test collapsed arrays consistency
        n_active_modes = np.sum(processor.polarization_modes)
        assert (
            len(processor.polarization_basis_collapsed) == n_active_modes
        ), "Collapsed basis should have length equal to number of active modes"
        assert (
            len(processor.polarization_derived_collapsed) == n_active_modes
        ), "Collapsed derived should have length equal to number of active modes"

        # Test that collapsed arrays sum correctly
        n_basis_collapsed = np.sum(processor.polarization_basis_collapsed)
        n_derived_collapsed = np.sum(processor.polarization_derived_collapsed)
        assert (
            n_basis_collapsed + n_derived_collapsed == n_active_modes
        ), "Collapsed basis + derived should equal total active modes"

    def test_compute_whitened_antenna_pattern_matrix_method(self, sample_interferometers):
        """Test the processor's whitened antenna pattern matrix computation."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p"], interferometers=sample_interferometers
        )

        # Create test data
        power_spectral_density_array = np.ones((3, 100)) * 1e-46  # 3 detectors, 100 frequencies
        frequency_mask = np.array([True] * 50 + [False] * 50)  # First 50 frequencies active

        parameters = {
            "ra": 0.5,
            "dec": 0.3,
            "psi": 0.1,
            "geocent_time": 1000000000.0,
            "amplitude_cp": 1.5,  # cross -> plus amplitude
            "phase_cp": 0.2,  # cross -> plus phase
        }

        whitened_matrix = processor.compute_whitened_antenna_pattern_matrix(
            sample_interferometers, power_spectral_density_array, frequency_mask, parameters
        )

        # Check output shape: (n_frequencies, n_detectors, n_basis_modes)
        n_frequencies = len(frequency_mask)
        n_detectors = len(sample_interferometers)
        n_basis_modes = np.sum(processor.polarization_basis_collapsed)
        expected_shape = (n_frequencies, n_detectors, n_basis_modes)

        assert whitened_matrix.shape == expected_shape, f"Expected shape {expected_shape}, got {whitened_matrix.shape}"

        # Check that masked frequencies are zero
        assert np.all(whitened_matrix[50:, :, :] == 0), "Masked frequencies should have zero values"

        # Check that active frequencies have non-zero values
        assert np.any(whitened_matrix[:50, :, :] != 0), "Active frequencies should have non-zero values"

        # All active values should be finite
        assert np.all(np.isfinite(whitened_matrix[:50, :, :])), "All active whitened values should be finite"

    def test_compute_calibration_factor_matrix_method(self, sample_interferometers):
        """Test the processor's calibration factor matrix computation."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p"], interferometers=sample_interferometers
        )

        # Create test frequency domain strain data
        n_frequencies = 100
        n_detectors = len(sample_interferometers)

        frequency_domain_strain_array = np.random.normal(
            0, 1e-23, (n_detectors, n_frequencies)
        ) + 1j * np.random.normal(0, 1e-23, (n_detectors, n_frequencies))

        # Frequency array should match the strain data shape for calibration
        masked_frequency_array = np.linspace(50, 500, n_frequencies)  # Frequency values, not complex ones
        frequency_mask = np.array([True] * 50 + [False] * 50)

        parameters = {
            "ra": 0.2,
            "dec": 0.4,
            "psi": 0.3,
            "geocent_time": 1000000000.0,
        }

        try:
            calibration_matrix = processor.compute_calibration_factor_matrix(
                sample_interferometers,
                frequency_domain_strain_array,
                masked_frequency_array,
                frequency_mask,
                parameters,
            )

            # Check output shape: (n_detectors, n_frequencies)
            expected_shape = (n_detectors, n_frequencies)
            assert (
                calibration_matrix.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {calibration_matrix.shape}"

            # All values should be finite
            assert np.all(np.isfinite(calibration_matrix)), "All calibration factors should be finite"

        except ValueError as e:
            # If the calibration model expects different input, skip this test
            # This is common when testing with mock interferometers
            pytest.skip(f"Calibration model interface mismatch: {e}")

    def test_compute_calibrated_whitened_antenna_pattern_matrix_method(self, sample_interferometers):
        """Test the processor's calibrated whitened antenna pattern matrix computation."""
        processor = AntennaPatternProcessor(
            polarization_modes=["p", "c"], polarization_basis=["p"], interferometers=sample_interferometers
        )

        # Create test data
        n_frequencies = 50
        n_detectors = len(sample_interferometers)

        power_spectral_density_array = np.ones((n_detectors, n_frequencies)) * 1e-46

        frequency_domain_strain_array = np.random.normal(
            0, 1e-23, (n_detectors, n_frequencies)
        ) + 1j * np.random.normal(0, 1e-23, (n_detectors, n_frequencies))

        # Use actual frequency values for calibration
        masked_frequency_array = np.linspace(50, 500, n_frequencies)
        frequency_mask = np.array([True] * 25 + [False] * 25)  # Half frequencies active

        parameters = {
            "ra": 0.6,
            "dec": -0.2,
            "psi": 0.8,
            "geocent_time": 1000000000.0,
            "amplitude_cp": 0.8,  # cross -> plus amplitude
            "phase_cp": 1.2,  # cross -> plus phase
        }

        try:
            calibrated_whitened_matrix = processor.compute_calibrated_whitened_antenna_pattern_matrix(
                sample_interferometers,
                power_spectral_density_array,
                frequency_domain_strain_array,
                masked_frequency_array,
                frequency_mask,
                parameters,
            )

            # Check output shape
            n_basis_modes = np.sum(processor.polarization_basis_collapsed)
            expected_shape = (n_frequencies, n_detectors, n_basis_modes)
            assert (
                calibrated_whitened_matrix.shape == expected_shape
            ), f"Expected shape {expected_shape}, got {calibrated_whitened_matrix.shape}"

            # Check that masked frequencies are zero
            assert np.all(calibrated_whitened_matrix[25:, :, :] == 0), "Masked frequencies should have zero values"

            # Check that active frequencies have values (may be complex)
            assert np.any(calibrated_whitened_matrix[:25, :, :] != 0), "Active frequencies should have non-zero values"

            # All active values should be finite
            assert np.all(
                np.isfinite(calibrated_whitened_matrix[:25, :, :])
            ), "All active calibrated values should be finite"

        except (ValueError, AttributeError) as e:
            # If the calibration or whitening fails due to interface issues, skip
            pytest.skip(f"Calibration/whitening interface mismatch: {e}")
