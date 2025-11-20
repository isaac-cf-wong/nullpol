"""Test module for null stream calculator functionality.

This module tests the NullStreamCalculator class and its methods
for computing null projections and energies using simple examples.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from nullpol.analysis.null_stream.calculator import NullStreamCalculator


@pytest.fixture
def calculator_instance():
    """Create a calculator instance for testing."""
    return NullStreamCalculator()


@pytest.fixture
def simple_test_data():
    """Create simple test data for calculator testing."""
    # Simple 2-detector, 2-frequency case
    whitened_antenna_pattern = np.array(
        [
            [[1.0, 0.0], [0.0, 1.0]],  # Frequency 0: orthogonal patterns
            [[1.0, 0.5], [0.5, 1.0]],  # Frequency 1: correlated patterns
        ],
        dtype=complex,
    )

    whitened_strain = np.array(
        [[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]], dtype=complex  # Detector 0  # Detector 1
    )

    frequency_mask = np.array([True, True])

    # Simple filter (all ones)
    tf_filter = np.ones((2, 64), dtype=complex)  # Match expected transform output

    return {
        "antenna_pattern": whitened_antenna_pattern,
        "strain_data": whitened_strain,
        "frequency_mask": frequency_mask,
        "tf_filter": tf_filter,
        "sampling_frequency": 4096.0,
        "wavelet_freq_resolution": 1.0,
        "wavelet_nx": 64,
    }


class TestNullStreamCalculator:
    """Test class for NullStreamCalculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = NullStreamCalculator()
        # Calculator should initialize without errors
        assert calculator is not None

    def test_calculator_is_pure_computational(self):
        """Test that calculator has no state or dependencies."""
        calculator = NullStreamCalculator()

        # Should have no instance variables (pure computational class)
        assert len(calculator.__dict__) == 0

        # Should only have the compute_null_energy method
        public_methods = [
            method for method in dir(calculator) if not method.startswith("_") and callable(getattr(calculator, method))
        ]
        assert public_methods == ["compute_null_energy"]

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_compute_null_energy_workflow(self, mock_transform, calculator_instance, simple_test_data):
        """Test the complete null energy computation workflow."""
        # Setup mock for wavelet transform
        mock_transform_output = np.ones((64,), dtype=complex) * (0.1 + 0.1j)
        mock_transform.return_value = mock_transform_output

        # Call the method
        energy = calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=simple_test_data["antenna_pattern"],
            whitened_frequency_strain_data=simple_test_data["strain_data"],
            frequency_mask=simple_test_data["frequency_mask"],
            time_frequency_filter=simple_test_data["tf_filter"],
            sampling_frequency=simple_test_data["sampling_frequency"],
            wavelet_frequency_resolution=simple_test_data["wavelet_freq_resolution"],
            wavelet_nx=simple_test_data["wavelet_nx"],
        )

        # Energy should be a real positive number
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

        # Transform should have been called for each detector
        assert mock_transform.call_count == 2  # 2 detectors

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_compute_null_energy_zero_filter(self, mock_transform, calculator_instance, simple_test_data):
        """Test null energy computation with zero filter (should give zero energy)."""
        mock_transform.return_value = np.ones((64,), dtype=complex)

        # Use zero filter
        zero_filter = np.zeros_like(simple_test_data["tf_filter"])

        energy = calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=simple_test_data["antenna_pattern"],
            whitened_frequency_strain_data=simple_test_data["strain_data"],
            frequency_mask=simple_test_data["frequency_mask"],
            time_frequency_filter=zero_filter,  # Zero filter
            sampling_frequency=simple_test_data["sampling_frequency"],
            wavelet_frequency_resolution=simple_test_data["wavelet_freq_resolution"],
            wavelet_nx=simple_test_data["wavelet_nx"],
        )

        # Energy should be zero (within numerical precision)
        assert abs(energy) < 1e-10

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_compute_null_energy_all_masked(self, mock_transform, calculator_instance, simple_test_data):
        """Test null energy computation when all frequencies are masked."""
        mock_transform.return_value = np.zeros((64,), dtype=complex)

        # Mask all frequencies
        all_masked = np.array([False, False])

        energy = calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=simple_test_data["antenna_pattern"],
            whitened_frequency_strain_data=simple_test_data["strain_data"],
            frequency_mask=all_masked,
            time_frequency_filter=simple_test_data["tf_filter"],
            sampling_frequency=simple_test_data["sampling_frequency"],
            wavelet_frequency_resolution=simple_test_data["wavelet_freq_resolution"],
            wavelet_nx=simple_test_data["wavelet_nx"],
        )

        # Should still return a valid energy value (zero since no signal)
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_data_type_consistency(self, mock_transform, calculator_instance):
        """Test that data types are handled consistently."""
        # Use complex64 input data
        antenna_pattern = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.complex64)
        strain_data = np.array([[1.0, 2.0]], dtype=np.complex64)
        frequency_mask = np.array([True])
        tf_filter = np.ones((1, 64), dtype=np.complex64)

        mock_transform.return_value = np.ones((64,), dtype=np.complex64)

        energy = calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=antenna_pattern,
            whitened_frequency_strain_data=strain_data,
            frequency_mask=frequency_mask,
            time_frequency_filter=tf_filter,
            sampling_frequency=4096.0,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
        )

        # Energy should be computed successfully regardless of input precision
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_parameter_passing_to_transform(self, mock_transform, calculator_instance, simple_test_data):
        """Test that parameters are correctly passed to the transform function."""
        # Match the expected nx_points for transform output
        nx_points = 64  # Keep consistent with test data setup
        mock_transform.return_value = np.zeros((nx_points,), dtype=complex)

        # Call with specific parameters
        sampling_freq = 8192.0
        freq_resolution = 2.0

        calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=simple_test_data["antenna_pattern"],
            whitened_frequency_strain_data=simple_test_data["strain_data"],
            frequency_mask=simple_test_data["frequency_mask"],
            time_frequency_filter=np.ones((2, nx_points), dtype=complex),
            sampling_frequency=sampling_freq,
            wavelet_frequency_resolution=freq_resolution,
            wavelet_nx=nx_points,
        )

        # Check that transform was called with correct parameters
        for call in mock_transform.call_args_list:
            _, kwargs = call
            assert kwargs["sampling_frequency"] == sampling_freq
            assert kwargs["frequency_resolution"] == freq_resolution
            assert kwargs["nx"] == nx_points

    def test_method_signature_completeness(self, calculator_instance):
        """Test that compute_null_energy has the expected signature."""
        import inspect  # pylint: disable=import-outside-toplevel

        sig = inspect.signature(calculator_instance.compute_null_energy)
        param_names = list(sig.parameters.keys())

        expected_params = [
            "whitened_antenna_pattern_matrix",
            "whitened_frequency_strain_data",
            "frequency_mask",
            "time_frequency_filter",
            "sampling_frequency",
            "wavelet_frequency_resolution",
            "wavelet_nx",
        ]

        assert param_names == expected_params, f"Expected {expected_params}, got {param_names}"

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_edge_case_single_detector(self, mock_transform, calculator_instance):
        """Test with single detector (minimal case)."""
        # Single detector case
        antenna_pattern = np.array([[[1.0]]], dtype=complex)  # (1, 1, 1)
        strain_data = np.array([[5.0]], dtype=complex)  # (1, 1)
        frequency_mask = np.array([True])
        tf_filter = np.ones((1, 32), dtype=complex)

        mock_transform.return_value = np.ones((32,), dtype=complex) * 0.5

        energy = calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=antenna_pattern,
            whitened_frequency_strain_data=strain_data,
            frequency_mask=frequency_mask,
            time_frequency_filter=tf_filter,
            sampling_frequency=4096.0,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=32,
        )

        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_computation_steps_integration(self, mock_transform, calculator_instance):
        """Test that all computation steps are properly integrated."""
        # Create controlled test data
        antenna_pattern = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)
        strain_data = np.array([[1.0], [0.0]], dtype=complex)  # Only first detector has signal
        frequency_mask = np.array([True])
        tf_filter = np.ones((2, 16), dtype=complex) * 2.0  # Amplify by factor 2

        # Mock transform to return predictable values
        mock_transform.return_value = np.ones((16,), dtype=complex) * 0.5

        energy = calculator_instance.compute_null_energy(
            whitened_antenna_pattern_matrix=antenna_pattern,
            whitened_frequency_strain_data=strain_data,
            frequency_mask=frequency_mask,
            time_frequency_filter=tf_filter,
            sampling_frequency=4096.0,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=16,
        )

        # With our setup: 2 detectors × 16 points × |0.5 * 2.0|² = 2 × 16 × 1 = 32
        expected_energy = 2 * 16 * abs(0.5 * 2.0) ** 2
        assert abs(energy - expected_energy) < 1e-10
