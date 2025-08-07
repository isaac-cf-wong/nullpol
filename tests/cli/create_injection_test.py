"""Test module for injection creation tool.

This module tests the command-line tool for creating signal injections into
detector data.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

from importlib.resources import files
import pytest

from nullpol.cli.create_injection import main


@pytest.fixture(scope="module")
def test_frame_paths():
    """Create temporary directory and define output file paths for testing.

    Sets up a temporary directory structure and defines expected output
    file paths for various test scenarios including different noise
    realizations and detector configurations.

    Yields:
        dict: Dictionary containing all temporary file paths
    """
    test_dir = tempfile.mkdtemp()

    paths = {
        "test_dir": test_dir,
        "H1_TEST_frame_path": os.path.join(test_dir, "H1-TEST-2024-16.gwf"),
        "L1_TEST_frame_path": os.path.join(test_dir, "L1-TEST-2024-16.gwf"),
        "V1_TEST_frame_path": os.path.join(test_dir, "V1-TEST-2024-16.gwf"),
        "H1_TEST_WITH_SIGNAL_FRAME_frame_path": os.path.join(test_dir, "H1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf"),
        "L1_TEST_WITH_SIGNAL_FRAME_frame_path": os.path.join(test_dir, "L1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf"),
        "V1_TEST_WITH_SIGNAL_FRAME_frame_path": os.path.join(test_dir, "V1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf"),
        "H1_TEST_PSD_frame_path": os.path.join(test_dir, "H1-TEST_PSD-2024-16.gwf"),
        "L1_TEST_PSD_frame_path": os.path.join(test_dir, "L1-TEST_PSD-2024-16.gwf"),
        "V1_TEST_PSD_frame_path": os.path.join(test_dir, "V1-TEST_PSD-2024-16.gwf"),
    }

    yield paths

    # Cleanup - remove all temporary files
    for path_name, path_value in paths.items():
        if path_name != "test_dir" and os.path.exists(path_value):
            os.remove(path_value)


def test_generate_config():
    """Test configuration file generation functionality.

    Validates that the tool correctly generates default configuration
    files matching the expected template, ensuring proper initialization
    of injection parameters.
    """
    # Create a temporary config file path
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
        config_file_path = temp_config_file.name

    with mock.patch("sys.argv", ["nullpol_create_injection", "--generate-config", config_file_path]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Check that the config file was generated
    assert os.path.exists(config_file_path)

    # Verify the contents of the generated config file
    with open(config_file_path) as f:
        generated_content = f.read()

    # Load the default config file
    default_config_file_path = str(files("nullpol.cli.templates") / "default_config_create_injection.ini")
    with open(default_config_file_path) as f:
        default_generated_content = f.read()

    # Clean up the temporary config file
    os.remove(config_file_path)

    # Compare the content of both files
    assert generated_content.strip() == default_generated_content.strip()


def test_create_injection(test_frame_paths):
    """Test basic signal injection creation.

    Validates that the tool can successfully create signal frame files
    containing injected binary black hole signals for all specified
    detectors in a multi-detector network.
    """
    # Create a temporary config file path
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
        config_file_path = temp_config_file.name

    example_signal_parameters_create_injection_path = str(
        files("nullpol.cli.templates") / "example_signal_parameters_create_injection.json"
    )

    with mock.patch(
        "sys.argv",
        [
            "nullpol_create_injection",
            "--generate-config",
            config_file_path,
            "--signal-parameters",
            example_signal_parameters_create_injection_path,
            "--outdir",
            test_frame_paths["test_dir"],
            "--label",
            "TEST",
            "--start-time",
            "2024",
            "--duration",
            "16",
            "--detectors",
            "H1, L1, V1",
        ],
    ):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Execute the tool
    with mock.patch("sys.argv", ["nullpol_create_injection", "--config", config_file_path]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Clean up the temporary config file
    os.remove(config_file_path)

    # Check that the output files are created
    assert os.path.exists(test_frame_paths["H1_TEST_frame_path"]), "H1 output file was not created."
    assert os.path.exists(test_frame_paths["L1_TEST_frame_path"]), "L1 output file was not created."
    assert os.path.exists(test_frame_paths["V1_TEST_frame_path"]), "V1 output file was not created."


def test_create_injection_with_signal_frame(test_frame_paths):
    """Test injection creation using existing signal frame files.

    Validates that the tool can create new injections by combining
    existing signal frame files with additional signals, enabling complex
    multi-signal injection scenarios.
    """
    # Create a temporary config file path
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
        config_file_path = temp_config_file.name

    example_signal_parameters_create_injection_path = str(
        files("nullpol.cli.templates") / "example_signal_parameters_create_injection.json"
    )

    with mock.patch(
        "sys.argv",
        [
            "nullpol_create_injection",
            "--generate-config",
            config_file_path,
            "--signal-parameters",
            example_signal_parameters_create_injection_path,
            "--outdir",
            test_frame_paths["test_dir"],
            "--label",
            "TEST_WITH_SIGNAL_FRAME",
            "--start-time",
            "2024",
            "--duration",
            "16",
            "--detectors",
            "H1, L1, V1",
            "--signal-files",
            json.dumps(
                f'{{"H1": "{test_frame_paths["H1_TEST_frame_path"]}", "L1": "{test_frame_paths["L1_TEST_frame_path"]}", "V1": "{test_frame_paths["V1_TEST_frame_path"]}"}}'
            ),
            "--signal-file-channels",
            json.dumps('{"H1": "H1:STRAIN", "L1": "L1:STRAIN", "V1": "V1:STRAIN"}'),
        ],
    ):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Execute the tool
    with mock.patch("sys.argv", ["nullpol_create_injection", "--config", config_file_path]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Clean up the temporary config file
    os.remove(config_file_path)

    # Check that the output files are created
    assert os.path.exists(test_frame_paths["H1_TEST_WITH_SIGNAL_FRAME_frame_path"]), "H1 output file was not created."
    assert os.path.exists(test_frame_paths["L1_TEST_WITH_SIGNAL_FRAME_frame_path"]), "L1 output file was not created."
    assert os.path.exists(test_frame_paths["V1_TEST_WITH_SIGNAL_FRAME_frame_path"]), "V1 output file was not created."


def test_create_injection_with_custom_psds(test_frame_paths):
    """Test injection creation using custom power spectral densities.

    Validates that the tool can generate injections using user-provided
    power spectral density files instead of default detector noise curves,
    enabling studies with realistic or historical detector sensitivities.
    """
    # Create a temporary config file path
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
        config_file_path = temp_config_file.name

    example_signal_parameters_create_injection_path = str(
        files("nullpol.cli.templates") / "example_signal_parameters_create_injection.json"
    )
    current_dir = os.path.dirname(__file__)
    mock_psd_path = os.path.join(current_dir, "fixtures", "mock_psd.txt")

    with mock.patch(
        "sys.argv",
        [
            "nullpol_create_injection",
            "--generate-config",
            config_file_path,
            "--signal-parameters",
            example_signal_parameters_create_injection_path,
            "--outdir",
            test_frame_paths["test_dir"],
            "--label",
            "TEST_PSD",
            "--start-time",
            "2024",
            "--duration",
            "16",
            "--detectors",
            "H1, L1, V1",
            "--psds",
            json.dumps(f'{{"H1": "{mock_psd_path}", "L1": "{mock_psd_path}", "V1": "{mock_psd_path}"}}'),
        ],
    ):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Execute the tool
    with mock.patch("sys.argv", ["nullpol_create_injection", "--config", config_file_path]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0  # Ensure that it exited with code 0

    # Check that the output files are created
    assert os.path.exists(test_frame_paths["H1_TEST_PSD_frame_path"]), "H1 output file was not created."
    assert os.path.exists(test_frame_paths["L1_TEST_PSD_frame_path"]), "L1 output file was not created."
    assert os.path.exists(test_frame_paths["V1_TEST_PSD_frame_path"]), "V1 output file was not created."
