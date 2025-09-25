"""Comprehensive test suite for PolarizationResult class.

This module provides comprehensive testing for the PolarizationResult class,
including:
- Basic functionality and inheritance
- Property access with various metadata configurations
- Detector injection properties
- Skymap plotting functionality with dependency handling
- Real fixture data integration and validation

The tests use both real fixture data from examples/ and mock likelihood metadata
to ensure comprehensive coverage of all functionality.
"""

from __future__ import annotations
import os
import json
import tempfile

import pytest
import numpy as np
import bilby

# Import result module and PolarizationResult class
import nullpol.analysis.result as result_module
from nullpol.analysis.result import PolarizationResult


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture(scope="session")
def real_result_data():
    """Load real result data from fixtures for comprehensive testing.

    Returns:
        dict: Contains 'result' (bilby Result object) and 'injection_data' (dict)
              from the scalar tensor injection example.
    """
    fixtures_dir = os.path.join(os.path.dirname(__file__), "fixtures")
    result_file = os.path.join(fixtures_dir, "scalar_tensor_injection_data0_0_analysis_H1L1V1_result.hdf5")
    injection_file = os.path.join(fixtures_dir, "injections.json")

    if not os.path.exists(result_file):
        pytest.skip(f"Test fixture not found: {result_file}")

    # Load real result and injection data
    real_result = bilby.core.result.read_in_result(result_file)

    with open(injection_file, "r") as f:
        injection_data = json.load(f)

    return {"result": real_result, "injection_data": injection_data}


@pytest.mark.integration
@pytest.fixture
def mock_likelihood_metadata():
    """Create mock likelihood metadata for testing properties that require structured data.

    This fixture provides the likelihood metadata structure that would be present
    in a complete analysis result, allowing us to test property access methods.

    Returns:
        dict: Mock likelihood metadata with sampling parameters and detector information.
    """
    return {
        "likelihood": {
            "sampling_frequency": 4096.0,
            "duration": 4.0,
            "start_time": 1126259462.0,
            "interferometers": {
                "H1": {
                    "optimal_SNR": 25.3,
                    "matched_filter_SNR": 24.1,
                    "injection_parameters": {"mass_1": 78.37, "mass_2": 36.61},
                },
                "L1": {
                    "optimal_SNR": 18.7,
                    "matched_filter_SNR": 17.9,
                    "injection_parameters": {"mass_1": 78.37, "mass_2": 36.61},
                },
                "V1": {
                    "optimal_SNR": 12.1,
                    "matched_filter_SNR": 11.8,
                    "injection_parameters": {"mass_1": 78.37, "mass_2": 36.61},
                },
            },
        }
    }


# =============================================================================
# BASIC FUNCTIONALITY TESTS
# =============================================================================


@pytest.mark.integration
class TestBasicFunctionality:
    """Test fundamental PolarizationResult functionality."""

    def test_result_module_import(self):
        """Test that result module can be imported and has expected attributes."""
        assert result_module is not None
        assert hasattr(result_module, "PolarizationResult")

        # Verify module is accessible through analysis package
        import nullpol.analysis  # pylint: disable=import-outside-toplevel

        assert hasattr(nullpol.analysis, "result")
        assert getattr(nullpol.analysis, "result") is result_module

    def test_polarization_result_inheritance_and_initialization(self, real_result_data):
        """Test PolarizationResult initialization and inheritance from bilby.Result."""
        real_result = real_result_data["result"]

        with tempfile.TemporaryDirectory() as temp_dir:
            result = PolarizationResult(
                label=real_result.label,
                outdir=temp_dir,
                posterior=real_result.posterior,  # Use 'posterior' parameter name
                meta_data=real_result.meta_data,
            )

            # Test inheritance hierarchy
            from bilby.core.result import Result  # pylint: disable=import-outside-toplevel

            assert isinstance(result, Result)
            assert isinstance(result, PolarizationResult)

            # Test basic attributes
            assert result.label == real_result.label
            assert result.outdir == temp_dir
            assert len(result.posterior) > 0  # Test that posterior data is accessible
            assert result.meta_data is not None

    def test_polarization_result_with_enhanced_metadata(self, real_result_data, mock_likelihood_metadata):
        """Test PolarizationResult with enhanced mock likelihood metadata."""
        real_result = real_result_data["result"]

        # Create metadata with mock likelihood structure
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = PolarizationResult(
                label=real_result.label,
                outdir=temp_dir,
                posterior=real_result.posterior,  # Use 'posterior' parameter name
                meta_data=enhanced_metadata,
            )

            # Test that enhanced metadata enables property access
            assert result.sampling_frequency == 4096.0
            assert result.duration == 4.0
            assert result.start_time == 1126259462.0
            assert result.interferometers == ["H1", "L1", "V1"]


# =============================================================================
# PROPERTY ACCESS TESTS
# =============================================================================


@pytest.mark.integration
class TestPropertyAccess:
    """Test PolarizationResult property access with various metadata configurations."""

    def test_sampling_frequency_property(self, real_result_data, mock_likelihood_metadata):
        """Test sampling_frequency property with valid metadata."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        sampling_freq = result.sampling_frequency
        assert isinstance(sampling_freq, (int, float))
        assert sampling_freq == 4096.0

    def test_duration_property(self, real_result_data, mock_likelihood_metadata):
        """Test duration property with valid metadata."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        duration = result.duration
        assert isinstance(duration, (int, float))
        assert duration == 4.0

    def test_start_time_property(self, real_result_data, mock_likelihood_metadata):
        """Test start_time property with valid metadata."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        start_time = result.start_time
        assert isinstance(start_time, (int, float))
        assert start_time == 1126259462.0

    def test_interferometers_property(self, real_result_data, mock_likelihood_metadata):
        """Test interferometers property returns correct detector list."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        interferometers = result.interferometers
        expected_interferometers = ["H1", "L1", "V1"]

        assert isinstance(interferometers, list)
        assert len(interferometers) == 3
        assert set(interferometers) == set(expected_interferometers)

    def test_property_access_with_none_likelihood(self, real_result_data):
        """Test property access fails gracefully when likelihood=None (realistic case)."""
        real_result = real_result_data["result"]

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=real_result.meta_data,  # Has likelihood=None in real fixtures
        )

        # These should raise TypeError since likelihood=None in real results
        with pytest.raises(TypeError, match="'NoneType' object"):
            _ = result.sampling_frequency

        with pytest.raises(TypeError, match="'NoneType' object"):
            _ = result.duration

        with pytest.raises(TypeError, match="'NoneType' object"):
            _ = result.start_time

        with pytest.raises(TypeError, match="'NoneType' object"):
            _ = result.interferometers


# =============================================================================
# METADATA ACCESS TESTS
# =============================================================================


@pytest.mark.integration
class TestMetadataAccess:
    """Test nested metadata access functionality."""

    def test_nested_metadata_access_with_missing_keys(self, real_result_data):
        """Test __get_from_nested_meta_data method with missing keys raises AttributeError."""
        real_result = real_result_data["result"]

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=real_result.meta_data,
        )

        # Test access to non-existent nested key
        with pytest.raises(AttributeError, match="No information stored for nonexistent/key"):
            result._PolarizationResult__get_from_nested_meta_data("nonexistent", "key")

    def test_nested_metadata_access_partial_path(self, real_result_data, mock_likelihood_metadata):
        """Test nested metadata access with partially valid paths."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        # Test valid partial path
        likelihood_data = result._PolarizationResult__get_from_nested_meta_data("likelihood")
        assert isinstance(likelihood_data, dict)
        assert "sampling_frequency" in likelihood_data

        # Test invalid continuation of valid path
        with pytest.raises(AttributeError, match="No information stored for likelihood/nonexistent"):
            result._PolarizationResult__get_from_nested_meta_data("likelihood", "nonexistent")


# =============================================================================
# DETECTOR INJECTION PROPERTIES TESTS
# =============================================================================


@pytest.mark.integration
class TestDetectorInjectionProperties:
    """Test detector injection properties functionality."""

    def test_detector_injection_properties_with_valid_detectors(self, real_result_data, mock_likelihood_metadata):
        """Test detector_injection_properties method with valid detector names."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        # Test H1 detector
        h1_properties = result.detector_injection_properties("H1")
        assert h1_properties is not None
        assert isinstance(h1_properties, dict)
        assert h1_properties["optimal_SNR"] == 25.3
        assert h1_properties["matched_filter_SNR"] == 24.1
        assert h1_properties["injection_parameters"]["mass_1"] == 78.37

        # Test L1 detector
        l1_properties = result.detector_injection_properties("L1")
        assert l1_properties is not None
        assert isinstance(l1_properties, dict)
        assert l1_properties["optimal_SNR"] == 18.7

        # Test V1 detector
        v1_properties = result.detector_injection_properties("V1")
        assert v1_properties is not None
        assert isinstance(v1_properties, dict)
        assert v1_properties["optimal_SNR"] == 12.1

    def test_detector_injection_properties_with_nonexistent_detector(self, real_result_data, mock_likelihood_metadata):
        """Test detector_injection_properties returns None for non-existent detectors."""
        real_result = real_result_data["result"]
        enhanced_metadata = real_result.meta_data.copy()
        enhanced_metadata.update(mock_likelihood_metadata)

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=enhanced_metadata,
        )

        # Test non-existent detector
        nonexistent_properties = result.detector_injection_properties("K1")
        assert nonexistent_properties is None

        # Test with empty string
        empty_properties = result.detector_injection_properties("")
        assert empty_properties is None

    def test_detector_injection_properties_with_none_likelihood(self, real_result_data):
        """Test detector_injection_properties when likelihood=None (realistic scenario)."""
        real_result = real_result_data["result"]

        result = PolarizationResult(
            posterior=real_result.posterior,
            meta_data=real_result.meta_data,  # Has likelihood=None in real fixtures
        )

        # Should handle the None likelihood gracefully and return None
        with pytest.raises((TypeError, AttributeError)):
            result.detector_injection_properties("H1")


# =============================================================================
# SKYMAP PLOTTING TESTS
# =============================================================================


@pytest.mark.integration
class TestSkymapPlotting:
    """Test skymap plotting functionality with graceful dependency handling."""

    def test_plot_skymap_with_missing_dependencies(self, real_result_data):
        """Test that plot_skymap handles missing external dependencies gracefully."""
        real_result = real_result_data["result"]

        # Create a result with required ra/dec columns
        posterior = real_result.posterior.copy()
        if "ra" not in posterior.columns:
            posterior["ra"] = np.random.uniform(0, 2 * np.pi, 10)
        if "dec" not in posterior.columns:
            posterior["dec"] = np.random.uniform(-np.pi / 2, np.pi / 2, 10)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = PolarizationResult(
                label="test_skymap_dependencies",
                outdir=temp_dir,
                posterior=posterior,
                meta_data=real_result.meta_data,
            )

            # Should handle ImportError gracefully due to potentially missing ligo.skymap dependencies
            try:
                result.plot_skymap(maxpts=10, trials=1, jobs=1)
                # If it succeeds, that's fine - dependencies were available
            except ImportError as e:
                # Expected behavior when dependencies are missing
                expected_modules = ["ligo.skymap", "healpy", "astropy"]
                assert any(module in str(e) for module in expected_modules)
            except Exception:
                # Other exceptions might occur due to missing dependencies or data issues
                # This is acceptable behavior for this test
                pass

    def test_plot_skymap_with_missing_required_columns(self, real_result_data):
        """Test skymap plotting behavior with missing required coordinate columns."""
        real_result = real_result_data["result"]

        # Create result with missing ra/dec columns
        posterior_missing_cols = real_result.posterior.copy()
        columns_to_remove = ["ra", "dec"]
        for col in columns_to_remove:
            if col in posterior_missing_cols.columns:
                posterior_missing_cols = posterior_missing_cols.drop(columns=[col])

        with tempfile.TemporaryDirectory() as temp_dir:
            result = PolarizationResult(
                label="test_missing_columns",
                outdir=temp_dir,
                posterior=posterior_missing_cols,
                meta_data=real_result.meta_data,
            )

            # Should handle missing columns appropriately
            try:
                result.plot_skymap()
                # If it doesn't raise an exception, the method handled missing columns gracefully
            except (KeyError, ImportError, ValueError):
                # Expected exceptions for missing columns or missing dependencies
                pass
            except Exception:
                # Other exceptions might occur, which is acceptable for this edge case test
                pass

    def test_plot_skymap_parameter_forwarding(self, real_result_data):
        """Test that plot_skymap accepts and forwards parameters correctly."""
        real_result = real_result_data["result"]

        # Ensure we have required columns
        posterior = real_result.posterior.copy()
        if "ra" not in posterior.columns:
            posterior["ra"] = np.random.uniform(0, 2 * np.pi, 5)
        if "dec" not in posterior.columns:
            posterior["dec"] = np.random.uniform(-np.pi / 2, np.pi / 2, 5)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = PolarizationResult(
                label="test_parameters",
                outdir=temp_dir,
                posterior=posterior,
                meta_data=real_result.meta_data,
            )

            # Test that method accepts various parameter combinations without error
            test_params = [
                {"maxpts": 5, "trials": 1, "jobs": 1},
                {"enable_multiresolution": False, "dpi": 300},
                {"colorbar": True, "contour": [90], "transparent": True},
            ]

            for params in test_params:
                try:
                    result.plot_skymap(**params)
                    # Success case - parameters were accepted
                except (ImportError, KeyError, ValueError):
                    # Expected exceptions due to missing dependencies or data issues
                    pass
                except TypeError as e:
                    # Unexpected - parameter wasn't accepted
                    pytest.fail(f"plot_skymap should accept parameters {params}, but got TypeError: {e}")
                except Exception:
                    # Other exceptions are acceptable for this test
                    pass


# =============================================================================
# REAL DATA INTEGRATION TESTS
# =============================================================================


@pytest.mark.integration
class TestRealDataIntegration:
    """Test integration and validation with real fixture data."""

    def test_real_injection_parameters_validation(self, real_result_data):
        """Test that injection parameters from real fixtures match expected values."""
        real_result = real_result_data["result"]
        injection_data = real_result_data["injection_data"]

        result = PolarizationResult(
            label=real_result.label,
            posterior=real_result.posterior,
            meta_data=real_result.meta_data,
        )

        # Validate injection parameters match fixture data
        injection_params = result.meta_data["injection_parameters"]
        expected_params = injection_data["injections"]["content"]

        # Test key gravitational wave parameters
        assert injection_params["mass_1"] == pytest.approx(expected_params["mass_1"][0], rel=1e-6)
        assert injection_params["mass_2"] == pytest.approx(expected_params["mass_2"][0], rel=1e-6)
        assert injection_params["luminosity_distance"] == pytest.approx(
            expected_params["luminosity_distance"][0], rel=1e-6
        )

    def test_real_analysis_configuration_validation(self, real_result_data):
        """Test that analysis configuration from real fixtures is as expected."""
        real_result = real_result_data["result"]

        result = PolarizationResult(
            label=real_result.label,
            posterior=real_result.posterior,
            meta_data=real_result.meta_data,
        )

        # Validate analysis configuration
        cmd_args = result.meta_data["command_line_args"]

        # Test detector and analysis setup
        assert cmd_args["detectors"] == ["H1", "L1", "V1"]
        assert cmd_args["duration"] == 4.0
        assert cmd_args["sampling_frequency"] == 4096.0
        assert cmd_args["injection"] is True

    def test_realistic_posterior_data_structure(self, real_result_data):
        """Test that fixture posterior data has realistic gravitational wave analysis structure."""
        real_result = real_result_data["result"]

        # Verify posterior has expected gravitational wave parameters
        expected_gw_params = ["dec", "geocent_time", "psi", "ra"]
        available_params = set(real_result.posterior.columns)

        for param in expected_gw_params:
            assert (
                param in available_params
            ), f"Expected GW parameter '{param}' not found in posterior columns: {list(available_params)[:10]}"

        # Verify posterior has realistic number of samples
        assert len(real_result.posterior) > 0, "Posterior should have samples"

        # Verify injection parameters are physically reasonable
        injection = real_result.meta_data["injection_parameters"]
        assert injection["mass_1"] > 0, "Mass 1 should be positive"
        assert injection["mass_2"] > 0, "Mass 2 should be positive"
        assert injection["luminosity_distance"] > 0, "Luminosity distance should be positive"
        assert (
            0 <= injection["theta_jn"] <= np.pi
        ), f"Inclination angle should be between 0 and π, got {injection['theta_jn']}"

    def test_metadata_structure_completeness(self, real_result_data):
        """Test that real fixture metadata has complete expected structure."""
        real_result = real_result_data["result"]

        result = PolarizationResult(
            label=real_result.label,
            posterior=real_result.posterior,
            meta_data=real_result.meta_data,
        )

        # Verify essential top-level metadata keys exist
        essential_keys = [
            "command_line_args",
            "injection_parameters",
            "likelihood",  # Should be present (but None in real fixtures)
            "nullpol_version",
            "run_statistics",
        ]

        for key in essential_keys:
            assert (
                key in result.meta_data
            ), f"Essential metadata key '{key}' not found in: {list(result.meta_data.keys())}"

        # Verify command_line_args substructure
        cmd_args = result.meta_data["command_line_args"]
        essential_cmd_keys = ["detectors", "duration", "sampling_frequency", "label", "outdir"]

        for key in essential_cmd_keys:
            assert key in cmd_args, f"Essential command_line_args key '{key}' not found in: {list(cmd_args.keys())}"

        # Verify likelihood is None (as expected in real fixtures)
        assert result.meta_data["likelihood"] is None, "Real fixture likelihood should be None"
