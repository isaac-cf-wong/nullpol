"""Test module for clustering plotting functionality.

This module tests the plotting functions with simple examples
and mocked matplotlib to ensure plots are created correctly.
"""

from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import Mock, patch

from nullpol.analysis.clustering.plotting import plot_spectrogram, plot_reverse_cumulative_distribution


class TestPlotSpectrogram:
    """Test class for spectrogram plotting function."""

    @pytest.fixture
    def simple_spectrogram_data(self):
        """Create simple spectrogram data for testing."""
        # 4x8 spectrogram (4 time bins, 8 frequency bins)
        return np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
                [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5],
                [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            ]
        )

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_basic_spectrogram_plot(self, mock_plt, mock_spectrogram_class, mock_get_shape, simple_spectrogram_data):
        """Test basic spectrogram plotting functionality."""
        # Mock the wavelet transform shape
        mock_get_shape.return_value = (4, 8)  # 4 time bins, 8 freq bins

        # Mock the Spectrogram object and its methods
        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()

        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Call the function
        plot_spectrogram(
            spectrogram=simple_spectrogram_data,
            duration=2.0,
            sampling_frequency=1024.0,
            wavelet_frequency_resolution=4.0,
            t0=0.5,
            title="Test Spectrogram",
        )

        # Verify get_shape_of_wavelet_transform was called correctly
        mock_get_shape.assert_called_once_with(
            duration=2.0, sampling_frequency=1024.0, wavelet_frequency_resolution=4.0
        )

        # Verify Spectrogram was created with correct parameters
        mock_spectrogram_class.assert_called_once_with(
            simple_spectrogram_data,
            t0=0.5,
            dt=2.0 / 4,  # duration / Nt = 2.0 / 4 = 0.5
            df=4.0,  # wavelet_frequency_resolution
            name="Test Spectrogram",
        )

        # Verify plot methods were called
        mock_spectrogram_instance.plot.assert_called_once()
        mock_plot_obj.gca.assert_called_once()
        mock_ax.set_yscale.assert_called_once_with("log")
        mock_ax.colorbar.assert_called_once()

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_spectrogram_with_frequency_range(
        self, mock_plt, mock_spectrogram_class, mock_get_shape, simple_spectrogram_data
    ):
        """Test spectrogram plotting with frequency range limits."""
        # Mock setup
        mock_get_shape.return_value = (4, 8)
        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()

        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Call with frequency range
        frequency_range = (10.0, 100.0)
        plot_spectrogram(
            spectrogram=simple_spectrogram_data,
            duration=1.0,
            sampling_frequency=512.0,
            wavelet_frequency_resolution=2.0,
            frequency_range=frequency_range,
        )

        # Verify frequency range was set
        mock_ax.set_ylim.assert_called_once_with(*frequency_range)

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_spectrogram_without_frequency_range(
        self, mock_plt, mock_spectrogram_class, mock_get_shape, simple_spectrogram_data
    ):
        """Test spectrogram plotting without frequency range (should not call set_ylim)."""
        # Mock setup
        mock_get_shape.return_value = (3, 6)
        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()

        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Call without frequency range
        plot_spectrogram(
            spectrogram=simple_spectrogram_data,
            duration=1.5,
            sampling_frequency=256.0,
            wavelet_frequency_resolution=1.0,
        )

        # Verify set_ylim was not called
        mock_ax.set_ylim.assert_not_called()

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_spectrogram_save_figure(self, mock_plt, mock_spectrogram_class, mock_get_shape, simple_spectrogram_data):
        """Test spectrogram plotting with figure saving."""
        # Mock setup
        mock_get_shape.return_value = (5, 10)
        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()

        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Call with savefig parameter
        plot_spectrogram(
            spectrogram=simple_spectrogram_data,
            duration=3.0,
            sampling_frequency=2048.0,
            wavelet_frequency_resolution=5.0,
            savefig="test_plot.png",
            dpi=150,
        )

        # Verify savefig was called with correct parameters
        mock_plt.savefig.assert_called_once_with(fname="test_plot.png", dpi=150, bbox_inches="tight")

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_spectrogram_default_parameters(
        self, mock_plt, mock_spectrogram_class, mock_get_shape, simple_spectrogram_data
    ):
        """Test spectrogram plotting with default parameters."""
        # Mock setup
        mock_get_shape.return_value = (6, 12)
        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()

        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Call with minimal required parameters
        plot_spectrogram(
            spectrogram=simple_spectrogram_data,
            duration=4.0,
            sampling_frequency=4096.0,
            wavelet_frequency_resolution=8.0,
        )

        # Verify Spectrogram was created with defaults
        mock_spectrogram_class.assert_called_once_with(
            simple_spectrogram_data, t0=0, dt=4.0 / 6, df=8.0, name=None  # Default t0  # duration / Nt  # Default title
        )

        # Verify no savefig call
        mock_plt.savefig.assert_not_called()

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_spectrogram_dt_calculation(self, mock_plt, mock_spectrogram_class, mock_get_shape):
        """Test that dt (time resolution) is calculated correctly."""
        # Create specific test data
        test_data = np.ones((8, 4))  # 8 time bins, 4 freq bins

        # Mock get_shape to return specific values
        mock_get_shape.return_value = (8, 4)  # Nt=8, Nf=4

        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()
        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Test with known duration
        duration = 2.4  # seconds
        plot_spectrogram(
            spectrogram=test_data, duration=duration, sampling_frequency=1000.0, wavelet_frequency_resolution=5.0
        )

        # Verify dt calculation: duration / Nt = 2.4 / 8 = 0.3
        expected_dt = 0.3
        mock_spectrogram_class.assert_called_once_with(test_data, t0=0, dt=expected_dt, df=5.0, name=None)

    @patch("nullpol.analysis.clustering.plotting.get_shape_of_wavelet_transform")
    @patch("nullpol.analysis.clustering.plotting.Spectrogram")
    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_spectrogram_all_parameters(self, mock_plt, mock_spectrogram_class, mock_get_shape):
        """Test spectrogram plotting with all parameters specified."""
        # Test data
        test_data = np.random.rand(3, 5)

        # Mock setup
        mock_get_shape.return_value = (3, 5)
        mock_spectrogram_instance = Mock()
        mock_plot_obj = Mock()
        mock_ax = Mock()
        mock_spectrogram_class.return_value = mock_spectrogram_instance
        mock_spectrogram_instance.plot.return_value = mock_plot_obj
        mock_plot_obj.gca.return_value = mock_ax

        # Call with all parameters
        plot_spectrogram(
            spectrogram=test_data,
            duration=6.0,
            sampling_frequency=8192.0,
            wavelet_frequency_resolution=12.0,
            frequency_range=(50.0, 500.0),
            t0=1.5,
            title="Complete Test",
            savefig="complete_test.pdf",
            dpi=300,
        )

        # Verify all calls
        mock_get_shape.assert_called_once_with(
            duration=6.0, sampling_frequency=8192.0, wavelet_frequency_resolution=12.0
        )

        mock_spectrogram_class.assert_called_once_with(
            test_data, t0=1.5, dt=6.0 / 3, df=12.0, name="Complete Test"  # duration / Nt = 2.0
        )

        mock_ax.set_ylim.assert_called_once_with(50.0, 500.0)
        mock_ax.set_yscale.assert_called_once_with("log")
        mock_ax.colorbar.assert_called_once()

        mock_plt.savefig.assert_called_once_with(fname="complete_test.pdf", dpi=300, bbox_inches="tight")


class TestPlotReverseCumulativeDistribution:
    """Test class for reverse cumulative distribution plotting function."""

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_basic_reverse_cumulative_plot(self, mock_plt):
        """Test basic reverse cumulative distribution plotting."""
        # Simple test data
        test_data = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        plot_reverse_cumulative_distribution(test_data)

        # Verify plt.hist was called once
        mock_plt.hist.assert_called_once()

        # Check the call arguments
        call_args, call_kwargs = mock_plt.hist.call_args
        expected_flattened = test_data.flatten()  # [1,2,3,4,5,6,7,8,9]

        # Check the data array was flattened correctly
        np.testing.assert_array_equal(call_args[0], expected_flattened)

        # Check the keyword arguments
        assert call_kwargs["bins"] == 25
        assert call_kwargs["density"] is False
        assert call_kwargs["cumulative"] == -1
        assert call_kwargs["histtype"] == "step"

        # Verify default title was set
        mock_plt.title.assert_called_once_with("Reversed cumulative distribution")

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_with_custom_bins(self, mock_plt):
        """Test reverse cumulative distribution with custom bin count."""
        test_data = np.ones((4, 2)) * 2.5  # All values are 2.5

        plot_reverse_cumulative_distribution(test_data, bins=50)

        # Verify plt.hist was called once
        mock_plt.hist.assert_called_once()

        # Check the call arguments
        call_args, call_kwargs = mock_plt.hist.call_args
        expected_flattened = test_data.flatten()

        # Check the data and parameters
        np.testing.assert_array_equal(call_args[0], expected_flattened)
        assert call_kwargs["bins"] == 50
        assert call_kwargs["density"] is False
        assert call_kwargs["cumulative"] == -1
        assert call_kwargs["histtype"] == "step"

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_with_custom_title(self, mock_plt):
        """Test reverse cumulative distribution with custom title."""
        test_data = np.array([[1.0, 2.0], [3.0, 4.0]])
        custom_title = "Custom Distribution Plot"

        plot_reverse_cumulative_distribution(test_data, title=custom_title)

        # Verify custom title was used
        mock_plt.title.assert_called_once_with(custom_title)

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_with_save_figure(self, mock_plt):
        """Test reverse cumulative distribution with figure saving."""
        test_data = np.random.rand(2, 3)

        plot_reverse_cumulative_distribution(
            test_data, bins=15, title="Saved Plot", savefig="distribution.png", dpi=200
        )

        # Verify savefig was called
        mock_plt.savefig.assert_called_once_with(fname="distribution.png", dpi=200, bbox_inches="tight")

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_without_save_figure(self, mock_plt):
        """Test reverse cumulative distribution without saving (default behavior)."""
        test_data = np.array([[5.0]])

        plot_reverse_cumulative_distribution(test_data)

        # Verify savefig was not called
        mock_plt.savefig.assert_not_called()

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_data_flattening(self, mock_plt):
        """Test that input data is properly flattened for histogram."""
        # Create data with known shape and values
        test_data = np.array([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]])  # Shape (2, 2, 2)

        plot_reverse_cumulative_distribution(test_data, bins=8)

        # Verify flattened data was passed to hist
        expected_flattened = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
        call_args = mock_plt.hist.call_args[0]  # Get positional arguments
        np.testing.assert_array_equal(call_args[0], expected_flattened)

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_all_parameters(self, mock_plt):
        """Test reverse cumulative distribution with all parameters specified."""
        test_data = np.linspace(0, 10, 20).reshape(4, 5)

        plot_reverse_cumulative_distribution(
            spectrogram=test_data, bins=30, title="Full Parameter Test", savefig="full_test.jpg", dpi=150
        )

        # Verify all parameters were used correctly
        mock_plt.hist.assert_called_once()

        # Check the call arguments
        call_args, call_kwargs = mock_plt.hist.call_args
        expected_flattened = test_data.flatten()

        # Check the data and parameters
        np.testing.assert_array_equal(call_args[0], expected_flattened)
        assert call_kwargs["bins"] == 30
        assert call_kwargs["density"] is False
        assert call_kwargs["cumulative"] == -1
        assert call_kwargs["histtype"] == "step"

        mock_plt.title.assert_called_once_with("Full Parameter Test")
        mock_plt.savefig.assert_called_once_with(fname="full_test.jpg", dpi=150, bbox_inches="tight")

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_empty_data(self, mock_plt):
        """Test reverse cumulative distribution with empty data."""
        test_data = np.array([]).reshape(0, 0)

        # Should handle empty data gracefully
        plot_reverse_cumulative_distribution(test_data)

        # Verify hist was called with empty array
        expected_flattened = np.array([])
        call_args = mock_plt.hist.call_args[0]
        np.testing.assert_array_equal(call_args[0], expected_flattened)

    @patch("nullpol.analysis.clustering.plotting.plt")
    def test_reverse_cumulative_single_value(self, mock_plt):
        """Test reverse cumulative distribution with single value data."""
        test_data = np.array([[42.0]])

        plot_reverse_cumulative_distribution(test_data, bins=1)

        # Verify single value is handled correctly
        expected_flattened = np.array([42.0])
        call_args = mock_plt.hist.call_args[0]
        np.testing.assert_array_equal(call_args[0], expected_flattened)
