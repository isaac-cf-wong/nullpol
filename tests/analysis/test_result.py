"""Test module for result handling functionality.

This module tests the result processing and handling functions used
for managing analysis outputs using real example data.
"""

from __future__ import annotations
import os
import json
import tempfile

import pytest
import numpy as np

# Import result module
import nullpol.analysis.result as result_module
from nullpol.analysis.result import PolarizationResult
import bilby


@pytest.fixture(scope="session")
def real_result_data():
    """Load real result data from examples for testing."""
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


def test_result_module_import():
    """Test that result module can be imported."""
    assert result_module is not None
    assert hasattr(result_module, "PolarizationResult")


def test_polarization_result_initialization(real_result_data):
    """Test PolarizationResult initialization with real data."""
    real_result = real_result_data["result"]

    # Create PolarizationResult from real data using temporary directory
    with tempfile.TemporaryDirectory() as temp_dir:
        result = PolarizationResult(
            label=real_result.label, outdir=temp_dir, samples=real_result.posterior, meta_data=real_result.meta_data
        )

        # Test inheritance
        from bilby.core.result import Result

        assert isinstance(result, Result)
        assert isinstance(result, PolarizationResult)
        assert result.label == real_result.label
        assert result.outdir == temp_dir


def test_real_injection_parameters(real_result_data):
    """Test access to real injection parameters."""
    real_result = real_result_data["result"]
    injection_data = real_result_data["injection_data"]

    result = PolarizationResult(label=real_result.label, samples=real_result.posterior, meta_data=real_result.meta_data)

    # Test injection parameters from real data
    injection_params = result.meta_data["injection_parameters"]
    expected_params = injection_data["injections"]["content"]

    # Verify key parameters match
    assert injection_params["mass_1"] == pytest.approx(expected_params["mass_1"][0], rel=1e-6)
    assert injection_params["mass_2"] == pytest.approx(expected_params["mass_2"][0], rel=1e-6)
    assert injection_params["luminosity_distance"] == pytest.approx(expected_params["luminosity_distance"][0], rel=1e-6)


def test_real_analysis_configuration(real_result_data):
    """Test access to real analysis configuration."""
    real_result = real_result_data["result"]

    result = PolarizationResult(label=real_result.label, samples=real_result.posterior, meta_data=real_result.meta_data)

    # Real configuration is in command_line_args
    cmd_args = result.meta_data["command_line_args"]

    # Test detector configuration
    assert cmd_args["detectors"] == ["H1", "L1", "V1"]
    assert cmd_args["duration"] == 4.0
    assert cmd_args["sampling_frequency"] == 4096.0
    assert cmd_args["injection"] is True


def test_properties_with_real_structure(real_result_data):
    """Test properties with real result structure (likelihood=None)."""
    real_result = real_result_data["result"]

    result = PolarizationResult(
        label=real_result.label, samples=real_result.posterior, meta_data=real_result.meta_data  # Has likelihood=None
    )

    # These should fail with TypeError since likelihood=None in real results
    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        _ = result.sampling_frequency

    with pytest.raises(TypeError, match="'NoneType' object is not subscriptable"):
        _ = result.duration


def test_realistic_data_structure(real_result_data):
    """Test that example data has realistic GW analysis structure."""
    real_result = real_result_data["result"]

    # Verify posterior has expected parameters
    expected_params = ["dec", "geocent_time", "psi", "ra"]
    for param in expected_params:
        assert param in real_result.posterior.columns

    # Verify injection parameters are physical
    injection = real_result.meta_data["injection_parameters"]
    assert injection["mass_1"] > 0
    assert injection["mass_2"] > 0
    assert injection["luminosity_distance"] > 0
    assert 0 <= injection["theta_jn"] <= np.pi


def test_result_module_structure():
    """Test basic result module structure."""
    # Test that the result module is accessible through analysis
    import nullpol.analysis

    assert hasattr(nullpol.analysis, "result")
    result_module = getattr(nullpol.analysis, "result")
    assert result_module is not None
