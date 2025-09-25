"""Test module for simulation source functionality.

This module tests the source modeling functionality for gravitational
wave signal generation, including non-GR polarization waveforms.
"""

from __future__ import annotations

import numpy as np
import pytest

from nullpol.simulation.source import lal_binary_black_hole_non_gr_simple_map


class TestLalBinaryBlackHoleNonGRSimpleMap:
    """Test cases for the non-GR binary black hole waveform function."""

    @pytest.fixture
    def basic_parameters(self):
        """Fixture providing basic BBH parameters."""
        return {
            "mass_1": 30.0,
            "mass_2": 25.0,
            "luminosity_distance": 1000.0,
            "a_1": 0.0,
            "tilt_1": 0.0,
            "phi_12": 0.0,
            "a_2": 0.0,
            "tilt_2": 0.0,
            "phi_jl": 0.0,
            "theta_jn": 0.0,
            "phase": 0.0,
        }

    @pytest.fixture
    def frequency_array(self):
        """Fixture providing a typical frequency array."""
        return np.linspace(20, 1024, 100)

    def test_function_exists_and_callable(self):
        """Test that the non-GR BBH function exists and is callable."""
        assert callable(lal_binary_black_hole_non_gr_simple_map)

    def test_default_gr_waveform(self, frequency_array, basic_parameters):
        """Test generation of default GR waveform (amp_p=1, amp_c=1, others=0)."""
        waveform = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, **basic_parameters)

        # Check return type and structure
        assert isinstance(waveform, dict)

        # Check that all expected polarizations are present
        expected_keys = ["plus", "cross", "x", "y", "breathing", "longitudinal"]
        for key in expected_keys:
            assert key in waveform
            assert isinstance(waveform[key], np.ndarray)
            assert waveform[key].shape == frequency_array.shape
            assert np.iscomplexobj(waveform[key])

    def test_non_gr_polarizations_default_zero(self, frequency_array, basic_parameters):
        """Test that non-GR polarizations are zero by default."""
        waveform = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, **basic_parameters)

        # Non-GR modes should be zero by default
        assert np.allclose(waveform["x"], 0.0)
        assert np.allclose(waveform["y"], 0.0)
        assert np.allclose(waveform["breathing"], 0.0)
        assert np.allclose(waveform["longitudinal"], 0.0)

    def test_amplitude_scaling_plus_polarization(self, frequency_array, basic_parameters):
        """Test amplitude scaling for plus polarization."""
        # Generate reference waveform
        wf_ref = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_p=1.0, **basic_parameters)

        # Generate scaled waveform
        scale_factor = 2.5
        wf_scaled = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, amp_p=scale_factor, **basic_parameters
        )

        # Plus polarization should be scaled by the factor
        assert np.allclose(wf_scaled["plus"], wf_ref["plus"] * scale_factor)

    def test_amplitude_scaling_cross_polarization(self, frequency_array, basic_parameters):
        """Test amplitude scaling for cross polarization."""
        # Generate reference waveform
        wf_ref = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_c=1.0, **basic_parameters)

        # Generate scaled waveform
        scale_factor = 1.5
        wf_scaled = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, amp_c=scale_factor, **basic_parameters
        )

        # Cross polarization should be scaled by the factor
        assert np.allclose(wf_scaled["cross"], wf_ref["cross"] * scale_factor)

    def test_vector_x_polarization_scaling(self, frequency_array, basic_parameters):
        """Test that vector x polarization scales with amp_x and relates to plus mode."""
        _ = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_x=0.0, **basic_parameters)

        amp_x = 0.7
        wf_x = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_x=amp_x, **basic_parameters)

        # X mode should be amp_x times the plus mode (with amp_p=1)
        wf_plus_only = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array,
            amp_p=1.0,
            amp_c=0.0,  # Set cross to zero for cleaner comparison
            **basic_parameters,
        )

        expected_x = wf_plus_only["plus"] * amp_x
        assert np.allclose(wf_x["x"], expected_x)

    def test_vector_y_polarization_scaling(self, frequency_array, basic_parameters):
        """Test that vector y polarization scales with amp_y and relates to cross mode."""
        amp_y = 0.3
        wf_y = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_y=amp_y, **basic_parameters)

        # Y mode should be amp_y times the cross mode (with amp_c=1)
        wf_cross_only = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array,
            amp_p=0.0,  # Set plus to zero for cleaner comparison
            amp_c=1.0,
            **basic_parameters,
        )

        expected_y = wf_cross_only["cross"] * amp_y
        assert np.allclose(wf_y["y"], expected_y)

    def test_breathing_polarization_scaling(self, frequency_array, basic_parameters):
        """Test that breathing polarization scales with amp_b and relates to plus mode."""
        amp_b = 0.5
        wf_b = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_b=amp_b, **basic_parameters)

        # Breathing mode should be amp_b times the plus mode
        wf_plus_only = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, amp_p=1.0, amp_c=0.0, **basic_parameters
        )

        expected_breathing = wf_plus_only["plus"] * amp_b
        assert np.allclose(wf_b["breathing"], expected_breathing)

    def test_longitudinal_polarization_scaling(self, frequency_array, basic_parameters):
        """Test that longitudinal polarization scales with amp_l and relates to cross mode."""
        amp_l = 0.8
        wf_l = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, amp_l=amp_l, **basic_parameters)

        # Longitudinal mode should be amp_l times the cross mode
        wf_cross_only = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, amp_p=0.0, amp_c=1.0, **basic_parameters
        )

        expected_longitudinal = wf_cross_only["cross"] * amp_l
        assert np.allclose(wf_l["longitudinal"], expected_longitudinal)

    def test_multiple_non_gr_modes_simultaneously(self, frequency_array, basic_parameters):
        """Test generation with multiple non-GR modes active simultaneously."""
        wf = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array,
            amp_p=1.0,
            amp_c=1.0,
            amp_x=0.3,
            amp_y=0.4,
            amp_b=0.2,
            amp_l=0.5,
            **basic_parameters,
        )

        # Check that the waveform has non-zero amplitudes somewhere (not all zeros)
        # Use max absolute value to check if any significant signal exists
        assert np.max(np.abs(wf["plus"])) > 1e-30 or len(wf["plus"][wf["plus"] != 0]) > 0
        assert np.max(np.abs(wf["cross"])) > 1e-30 or len(wf["cross"][wf["cross"] != 0]) > 0

        # Check that non-GR modes exist and have expected relationships
        if np.max(np.abs(wf["plus"])) > 1e-30:
            # x and breathing should be proportional to plus when plus is non-zero
            plus_mask = np.abs(wf["plus"]) > 1e-30
            if np.any(plus_mask):
                assert np.max(np.abs(wf["x"])) > 0
                assert np.max(np.abs(wf["breathing"])) > 0

        if np.max(np.abs(wf["cross"])) > 1e-30:
            # y and longitudinal should be proportional to cross when cross is non-zero
            cross_mask = np.abs(wf["cross"]) > 1e-30
            if np.any(cross_mask):
                assert np.max(np.abs(wf["y"])) > 0
                assert np.max(np.abs(wf["longitudinal"])) > 0

    def test_zero_amplitude_modes(self, frequency_array, basic_parameters):
        """Test that zero amplitude produces zero waveforms."""
        wf = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array,
            amp_p=0.0,
            amp_c=0.0,
            amp_x=0.0,
            amp_y=0.0,
            amp_b=0.0,
            amp_l=0.0,
            **basic_parameters,
        )

        # All modes should be zero
        for mode in ["plus", "cross", "x", "y", "breathing", "longitudinal"]:
            assert np.allclose(wf[mode], 0.0)

    def test_waveform_kwargs_passthrough(self, frequency_array, basic_parameters):
        """Test that additional waveform kwargs are passed through correctly."""
        custom_kwargs = {"reference_frequency": 100.0, "waveform_approximant": "IMRPhenomPv2"}

        # Should not raise an error with custom kwargs
        wf = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, **basic_parameters, **custom_kwargs
        )

        assert isinstance(wf, dict)
        assert "plus" in wf
        assert "cross" in wf

    def test_parameter_validation_types(self, frequency_array, basic_parameters):
        """Test that function handles different parameter types appropriately."""
        # Test with integer masses
        params_int = basic_parameters.copy()
        params_int["mass_1"] = 30
        params_int["mass_2"] = 25

        wf = lal_binary_black_hole_non_gr_simple_map(frequency_array=frequency_array, **params_int)

        assert isinstance(wf, dict)
        assert "plus" in wf

    def test_edge_case_extreme_amplitudes(self, frequency_array, basic_parameters):
        """Test behavior with extreme amplitude values."""
        # Test with very large amplitude
        wf_large = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, amp_p=1000.0, **basic_parameters
        )

        # Test with very small amplitude
        wf_small = lal_binary_black_hole_non_gr_simple_map(
            frequency_array=frequency_array, amp_p=1e-10, **basic_parameters
        )

        # Both should succeed and have proper structure
        for wf in [wf_large, wf_small]:
            assert isinstance(wf, dict)
            assert "plus" in wf
            assert wf["plus"].shape == frequency_array.shape

    def test_frequency_array_shapes(self, basic_parameters):
        """Test that function works with different frequency array shapes."""
        # Test with different array sizes
        for size in [10, 50, 200]:
            freq_array = np.linspace(20, 1024, size)
            wf = lal_binary_black_hole_non_gr_simple_map(frequency_array=freq_array, **basic_parameters)

            for mode in ["plus", "cross", "x", "y", "breathing", "longitudinal"]:
                assert wf[mode].shape == freq_array.shape


class TestSourceModuleStructure:
    """Test the overall structure of the source module."""

    def test_source_module_imports(self):
        """Test that the source module can be imported and has expected attributes."""
        import nullpol.simulation.source as source_module  # pylint: disable=import-outside-toplevel

        assert source_module is not None
        assert hasattr(source_module, "lal_binary_black_hole_non_gr_simple_map")

    def test_function_documentation(self):
        """Test that the function has proper documentation."""
        assert lal_binary_black_hole_non_gr_simple_map.__doc__ is not None
        assert len(lal_binary_black_hole_non_gr_simple_map.__doc__.strip()) > 0

        # Check for key documentation elements
        doc = lal_binary_black_hole_non_gr_simple_map.__doc__
        assert "Args:" in doc
        assert "Returns:" in doc
        assert "frequency_array" in doc
        assert "polarization" in doc.lower()

    def test_function_signature(self):
        """Test that the function has the expected signature."""
        import inspect  # pylint: disable=import-outside-toplevel

        sig = inspect.signature(lal_binary_black_hole_non_gr_simple_map)
        params = list(sig.parameters.keys())

        # Check for required parameters
        required_params = [
            "frequency_array",
            "mass_1",
            "mass_2",
            "luminosity_distance",
            "a_1",
            "tilt_1",
            "phi_12",
            "a_2",
            "tilt_2",
            "phi_jl",
            "theta_jn",
            "phase",
        ]

        for param in required_params:
            assert param in params

        # Check for optional amplitude parameters
        optional_params = ["amp_p", "amp_c", "amp_x", "amp_y", "amp_b", "amp_l"]
        for param in optional_params:
            assert param in params
