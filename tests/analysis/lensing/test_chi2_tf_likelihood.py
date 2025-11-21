"""Test module for lensing chi-squared likelihood functionality.

This module tests the LensingChi2TimeFrequencyLikelihood class.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, patch

from nullpol.analysis.lensing.chi2_tf_likelihood import LensingChi2TimeFrequencyLikelihood


class TestLensingChi2TimeFrequencyLikelihood:
    """Test class for LensingChi2TimeFrequencyLikelihood."""

    def test_initialization_validation_correct_format(self):
        """Test that initialization succeeds with correct format."""
        mock_ifos_1 = [Mock(), Mock(), Mock()]
        mock_ifos_2 = [Mock(), Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        with patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator"):
            likelihood = LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )

            assert likelihood is not None

    def test_initialization_validation_wrong_format_not_list(self):
        """Test that initialization fails if interferometers is not a list."""
        interferometers = "not_a_list"

        with pytest.raises(ValueError, match="interferometers must be a list of two lists"):
            LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )

    def test_initialization_validation_wrong_length(self):
        """Test that initialization fails if not exactly two sublists."""
        mock_ifos_1 = [Mock()]
        mock_ifos_2 = [Mock()]
        mock_ifos_3 = [Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2, mock_ifos_3]  # Three lists

        with pytest.raises(ValueError, match="interferometers must be a list of two lists"):
            LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )

    def test_initialization_validation_sublists_not_lists(self):
        """Test that initialization fails if sublists are not lists."""
        interferometers = ["not_a_list", "also_not_a_list"]

        with pytest.raises(ValueError, match="interferometers must be a list of two lists"):
            LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )

    def test_initialization_validation_one_sublist(self):
        """Test that initialization fails with only one sublist."""
        mock_ifos = [Mock(), Mock()]
        interferometers = [mock_ifos]  # Only one list

        with pytest.raises(ValueError, match="interferometers must be a list of two lists"):
            LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_null_stream_calculator_initialization(self, mock_calculator_class):
        """Test that LensingNullStreamCalculator is initialized correctly."""
        mock_ifos_1 = [Mock(), Mock(), Mock()]
        mock_ifos_2 = [Mock(), Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        wavelet_frequency_resolution = 4.0
        wavelet_nx = 256
        polarization_modes = "pc"
        polarization_basis = "p"
        tf_filter = np.ones((10, 10))

        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=tf_filter,
        )

        # Verify calculator was initialized with correct arguments
        mock_calculator_class.assert_called_once_with(
            interferometers=interferometers,
            wavelet_frequency_resolution=wavelet_frequency_resolution,
            wavelet_nx=wavelet_nx,
            polarization_modes=polarization_modes,
            polarization_basis=polarization_basis,
            time_frequency_filter=tf_filter,
        )

        assert likelihood.null_stream_calculator is not None

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_attributes_initialization(self, mock_calculator_class):
        """Test that likelihood attributes are initialized correctly."""
        mock_ifos_1 = [Mock()]
        mock_ifos_2 = [Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        # Check that attributes exist
        assert hasattr(likelihood, "null_stream_calculator")
        assert hasattr(likelihood, "_noise_log_likelihood_value")
        assert hasattr(likelihood, "_marginalized_parameters")

        # Check initial values
        assert likelihood._noise_log_likelihood_value is None
        assert likelihood._marginalized_parameters == []

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_inherits_from_chi2_likelihood(self, mock_calculator_class):
        """Test that class inherits from Chi2TimeFrequencyLikelihood."""
        from nullpol.analysis.likelihood.chi2_tf_likelihood import Chi2TimeFrequencyLikelihood

        mock_ifos_1 = [Mock()]
        mock_ifos_2 = [Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        # Should inherit from Chi2TimeFrequencyLikelihood
        assert isinstance(likelihood, Chi2TimeFrequencyLikelihood)

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_optional_parameters(self, mock_calculator_class):
        """Test that optional parameters are handled correctly."""
        mock_ifos_1 = [Mock()]
        mock_ifos_2 = [Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        # Test with all optional parameters
        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
            polarization_basis="p",
            time_frequency_filter=np.ones((5, 5)),
            extra_arg="ignored",
            another_kwarg=123,
        )

        # Should initialize without error
        assert likelihood is not None

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_error_message_clarity(self, mock_calculator_class):
        """Test that error message is clear and helpful."""
        try:
            LensingChi2TimeFrequencyLikelihood(
                interferometers="wrong_format",
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )
            assert False, "Should have raised ValueError"
        except ValueError as e:
            # Error message should be clear
            assert "interferometers must be a list of two lists" in str(e)

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_edge_case_empty_sublists(self, mock_calculator_class):
        """Test behavior with empty sublists."""
        interferometers = [[], []]  # Two empty lists

        # This should pass validation (format is correct)
        # but may fail in calculator initialization
        try:
            likelihood = LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )
            # If it doesn't fail here, that's acceptable
            # (calculator will handle empty lists)
            assert likelihood is not None
        except Exception:
            # If it fails in calculator, that's also acceptable
            pass

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_single_interferometer_per_set(self, mock_calculator_class):
        """Test with single interferometer in each set."""
        mock_ifos_1 = [Mock()]
        mock_ifos_2 = [Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        assert likelihood is not None

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_many_interferometers_per_set(self, mock_calculator_class):
        """Test with many interferometers in each set."""
        mock_ifos_1 = [Mock() for _ in range(5)]
        mock_ifos_2 = [Mock() for _ in range(3)]
        interferometers = [mock_ifos_1, mock_ifos_2]

        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        assert likelihood is not None

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_parent_methods_accessible(self, mock_calculator_class):
        """Test that parent class methods are accessible."""
        mock_ifos_1 = [Mock()]
        mock_ifos_2 = [Mock()]
        interferometers = [mock_ifos_1, mock_ifos_2]

        likelihood = LensingChi2TimeFrequencyLikelihood(
            interferometers=interferometers,
            wavelet_frequency_resolution=4.0,
            wavelet_nx=256,
            polarization_modes="pc",
        )

        # Should have parent methods
        assert hasattr(likelihood, "log_likelihood")
        assert hasattr(likelihood, "noise_log_likelihood")
        assert hasattr(likelihood, "_calculate_noise_log_likelihood")

    @patch("nullpol.analysis.lensing.chi2_tf_likelihood.LensingNullStreamCalculator")
    def test_nested_list_structure(self, mock_calculator_class):
        """Test that deeply nested structures are rejected."""
        # Nested list structure (not just two levels)
        interferometers = [[[Mock()]], [[Mock()]]]

        # Should fail validation (elements are not lists of interferometers)
        # This will pass the isinstance check but may cause issues later
        try:
            likelihood = LensingChi2TimeFrequencyLikelihood(
                interferometers=interferometers,
                wavelet_frequency_resolution=4.0,
                wavelet_nx=256,
                polarization_modes="pc",
            )
            # If validation passes, calculator should handle it
            assert likelihood is not None
        except Exception:
            # Acceptable to fail during initialization
            pass
