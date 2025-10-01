"""Test module for null stream calculator functionality.

This module tests the NullStreamCalculator class and its methods
for computing null projections and energies using simple examples.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from nullpol.analysis.null_stream.calculator import NullStreamCalculator


def setup_antenna_pattern_processor_mock(mock_antenna_class):
    """Helper function to setup AntennaPatternProcessor mocks with proper polarization properties.

    Args:
        mock_antenna_class: The mocked AntennaPatternProcessor class.
    """
    from unittest.mock import PropertyMock

    mock_antenna_instance = mock_antenna_class.return_value
    # Mock polarization_basis: 2 basis modes (plus and cross)
    type(mock_antenna_instance).polarization_basis = PropertyMock(
        return_value=np.array([True, True, False, False, False, False])
    )
    # Mock polarization_modes: 2 modes (plus and cross, same as basis)
    type(mock_antenna_instance).polarization_modes = PropertyMock(
        return_value=np.array([True, True, False, False, False, False])
    )


def setup_calculator_mocks(
    calculator,
    frequency_mask=None,
    sampling_frequency=4096.0,
    wavelet_frequency_resolution=1.0,
    wavelet_nx=64,
    time_frequency_filter=None,
    interferometers=None,
    power_spectral_density_array=None,
):
    """Helper function to setup common mocks for calculator tests."""
    from unittest.mock import PropertyMock

    if frequency_mask is None:
        frequency_mask = np.array([True, True])
    if time_frequency_filter is None:
        time_frequency_filter = np.ones((2, wavelet_nx), dtype=complex)
    if power_spectral_density_array is None:
        power_spectral_density_array = np.ones((2, len(frequency_mask)))

    # Mock all the properties that the calculator needs
    type(calculator.data_context).frequency_mask = PropertyMock(return_value=frequency_mask)
    type(calculator.data_context).sampling_frequency = PropertyMock(return_value=sampling_frequency)
    type(calculator.data_context).wavelet_frequency_resolution = PropertyMock(return_value=wavelet_frequency_resolution)
    type(calculator.data_context).wavelet_nx = PropertyMock(return_value=wavelet_nx)
    type(calculator.data_context).time_frequency_filter = PropertyMock(return_value=time_frequency_filter)

    if interferometers is not None:
        type(calculator.data_context).interferometers = PropertyMock(return_value=interferometers)

    type(calculator.data_context).power_spectral_density_array = PropertyMock(return_value=power_spectral_density_array)


@pytest.fixture
def calculator_instance():
    """Create a calculator instance for testing."""
    from unittest.mock import Mock, PropertyMock

    with (
        patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext"),
        patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor") as mock_antenna_class,
    ):
        # Create mock interferometers
        mock_ifos = Mock()

        # Mock the polarization_basis and polarization_modes properties to return proper values
        mock_antenna_instance = mock_antenna_class.return_value
        type(mock_antenna_instance).polarization_basis = PropertyMock(
            return_value=np.array([True, True, False, False, False, False])  # 2 basis modes: plus and cross
        )
        type(mock_antenna_instance).polarization_modes = PropertyMock(
            return_value=np.array([True, True, False, False, False, False])  # 2 modes: plus and cross
        )

        return NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=np.ones((2, 64), dtype=complex),
        )


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

    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_calculator_initialization(self, mock_data_context_class, mock_antenna_class):
        """Test calculator initialization."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=None,
        )
        # Calculator should initialize without errors
        assert calculator is not None
        # Should have data_context and antenna_pattern_processor
        assert calculator.data_context is not None
        assert calculator.antenna_pattern_processor is not None

    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_calculator_has_required_components(self, mock_data_context_class, mock_antenna_class):
        """Test that calculator has required components."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=None,
        )

        # Should have data_context and antenna_pattern_processor
        assert hasattr(calculator, "data_context")
        assert hasattr(calculator, "antenna_pattern_processor")

        # Should have the compute methods
        public_methods = [
            method for method in dir(calculator) if not method.startswith("_") and callable(getattr(calculator, method))
        ]
        assert "compute_null_energy" in public_methods
        assert "compute_principal_null_components" in public_methods

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_compute_null_energy_workflow(self, mock_transform, calculator_instance):
        """Test the complete null energy computation workflow."""
        from unittest.mock import Mock

        # Setup mock for wavelet transform
        mock_transform_output = np.ones((64,), dtype=complex) * (0.1 + 0.1j)
        mock_transform.return_value = mock_transform_output

        # Setup calculator mocks
        setup_calculator_mocks(calculator_instance)

        # Mock the data_context and antenna_pattern_processor methods
        calculator_instance.data_context.compute_whitened_strain_at_geocenter = Mock(
            return_value=np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]], dtype=complex)
        )
        calculator_instance.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=np.array(
                [
                    [[1.0, 0.0], [0.0, 1.0]],
                    [[1.0, 0.5], [0.5, 1.0]],
                ],
                dtype=complex,
            )
        )

        # Call the method with parameters
        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        energy = calculator_instance.compute_null_energy(test_parameters)

        # Energy should be a real positive number
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

        # Transform should have been called for each detector
        assert mock_transform.call_count == 2  # 2 detectors

    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    def test_compute_null_energy_zero_filter(self, mock_transform, mock_antenna_class, mock_data_context_class):
        """Test null energy computation with zero filter (should give zero energy)."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        # Use zero filter
        zero_filter = np.zeros((2, 64), dtype=complex)

        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=zero_filter,
        )

        mock_transform.return_value = np.ones((64,), dtype=complex)

        # Use helper to setup mocks
        setup_calculator_mocks(calculator, time_frequency_filter=zero_filter, interferometers=mock_ifos)

        calculator.data_context.compute_whitened_strain_at_geocenter = Mock(
            return_value=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=complex)
        )
        calculator.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=complex)
        )

        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        energy = calculator.compute_null_energy(test_parameters)

        # Energy should be zero (within numerical precision)
        assert abs(energy) < 1e-10

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_compute_null_energy_all_masked(self, mock_data_context_class, mock_antenna_class, mock_transform):
        """Test null energy computation when all frequencies are masked."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        tf_filter = np.ones((2, 64), dtype=complex)

        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=tf_filter,
        )

        mock_transform.return_value = np.zeros((64,), dtype=complex)

        # Mask all frequencies
        all_masked = np.array([False, False])
        setup_calculator_mocks(
            calculator, frequency_mask=all_masked, time_frequency_filter=tf_filter, interferometers=mock_ifos
        )

        calculator.data_context.compute_whitened_strain_at_geocenter = Mock(
            return_value=np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]], dtype=complex)
        )
        calculator.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=complex)
        )

        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        energy = calculator.compute_null_energy(test_parameters)

        # Should still return a valid energy value (zero since no signal)
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_data_type_consistency(self, mock_data_context_class, mock_antenna_class, mock_transform):
        """Test that data types are handled consistently."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        # Use complex64 input data
        tf_filter = np.ones((1, 64), dtype=np.complex64)

        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=64,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=tf_filter,
        )

        mock_transform.return_value = np.ones((64,), dtype=np.complex64)

        # Setup mocks for single detector
        setup_calculator_mocks(
            calculator,
            frequency_mask=np.array([True]),
            time_frequency_filter=tf_filter,
            interferometers=mock_ifos,
            power_spectral_density_array=np.ones((1, 1)),
        )

        antenna_pattern = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.complex64)
        strain_data = np.array([[1.0, 2.0]], dtype=np.complex64)

        calculator.data_context.compute_whitened_strain_at_geocenter = Mock(return_value=strain_data)
        calculator.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=antenna_pattern
        )

        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        energy = calculator.compute_null_energy(test_parameters)

        # Energy should be computed successfully regardless of input precision
        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_parameter_passing_to_transform(self, mock_data_context_class, mock_antenna_class, mock_transform):
        """Test that parameters are correctly passed to the transform function."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        # Call with specific parameters
        sampling_freq = 8192.0
        freq_resolution = 2.0
        nx_points = 64

        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=freq_resolution,
            wavelet_nx=nx_points,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=np.ones((2, nx_points), dtype=complex),
        )

        mock_transform.return_value = np.zeros((nx_points,), dtype=complex)

        # Setup mocks with custom parameters
        setup_calculator_mocks(
            calculator,
            sampling_frequency=sampling_freq,
            wavelet_frequency_resolution=freq_resolution,
            wavelet_nx=nx_points,
            interferometers=mock_ifos,
        )

        calculator.data_context.compute_whitened_strain_at_geocenter = Mock(
            return_value=np.array([[1.0 + 0.0j, 2.0 + 0.0j], [3.0 + 0.0j, 4.0 + 0.0j]], dtype=complex)
        )
        calculator.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.5], [0.5, 1.0]]], dtype=complex)
        )

        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        calculator.compute_null_energy(test_parameters)

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

        expected_params = ["parameters"]

        assert param_names == expected_params, f"Expected {expected_params}, got {param_names}"

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_edge_case_single_detector(self, mock_data_context_class, mock_antenna_class, mock_transform):
        """Test with single detector (minimal case)."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        tf_filter = np.ones((1, 32), dtype=complex)

        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=32,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=tf_filter,
        )

        mock_transform.return_value = np.ones((32,), dtype=complex) * 0.5

        # Setup mocks for single detector
        setup_calculator_mocks(
            calculator,
            frequency_mask=np.array([True]),
            wavelet_nx=32,
            time_frequency_filter=tf_filter,
            interferometers=mock_ifos,
            power_spectral_density_array=np.ones((1, 1)),
        )

        # Single detector case
        antenna_pattern = np.array([[[1.0]]], dtype=complex)  # (1, 1, 1)
        strain_data = np.array([[5.0]], dtype=complex)  # (1, 1)

        calculator.data_context.compute_whitened_strain_at_geocenter = Mock(return_value=strain_data)
        calculator.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=antenna_pattern
        )

        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        energy = calculator.compute_null_energy(test_parameters)

        assert isinstance(energy, (float, np.floating))
        assert energy >= 0

    @patch("nullpol.analysis.null_stream.calculator.transform_wavelet_freq")
    @patch("nullpol.analysis.null_stream.calculator.AntennaPatternProcessor")
    @patch("nullpol.analysis.null_stream.calculator.TimeFrequencyDataContext")
    def test_computation_steps_integration(self, mock_data_context_class, mock_antenna_class, mock_transform):
        """Test that all computation steps are properly integrated."""
        from unittest.mock import Mock

        setup_antenna_pattern_processor_mock(mock_antenna_class)
        mock_ifos = Mock()
        tf_filter = np.ones((2, 16), dtype=complex) * 2.0  # Amplify by factor 2

        calculator = NullStreamCalculator(
            interferometers=mock_ifos,
            wavelet_frequency_resolution=1.0,
            wavelet_nx=16,
            polarization_modes="pc",
            polarization_basis="pc",
            time_frequency_filter=tf_filter,
        )

        # Mock transform to return predictable values
        mock_transform.return_value = np.ones((16,), dtype=complex) * 0.5

        # Setup mocks
        setup_calculator_mocks(
            calculator,
            frequency_mask=np.array([True]),
            wavelet_nx=16,
            time_frequency_filter=tf_filter,
            interferometers=mock_ifos,
            power_spectral_density_array=np.ones((2, 1)),
        )

        # Create controlled test data
        antenna_pattern = np.array([[[1.0, 0.0], [0.0, 1.0]]], dtype=complex)
        strain_data = np.array([[1.0], [0.0]], dtype=complex)  # Only first detector has signal

        calculator.data_context.compute_whitened_strain_at_geocenter = Mock(return_value=strain_data)
        calculator.antenna_pattern_processor.compute_whitened_antenna_pattern_matrix = Mock(
            return_value=antenna_pattern
        )

        test_parameters = {"ra": 0.0, "dec": 0.0, "psi": 0.0, "geocent_time": 1234567890.0}
        energy = calculator.compute_null_energy(test_parameters)

        # With our setup: 2 detectors × 16 points × |0.5 * 2.0|² = 2 × 16 × 1 = 32
        expected_energy = 2 * 16 * abs(0.5 * 2.0) ** 2
        assert abs(energy - expected_energy) < 1e-10
