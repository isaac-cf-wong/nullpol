"""Test module for time-frequency filter creation tool from sample parameters.

This module tests the command-line tool for creating time-frequency filters
based on signal parameters.
"""

from __future__ import annotations

import json
import os
import tempfile
from unittest import mock

from importlib.resources import files

from nullpol.cli.create_time_frequency_filter_from_sample import main


def test_generate_config():
    """Test configuration file generation for filter creation.

    Validates that the tool correctly generates default configuration
    files for time-frequency filter creation.
    """
    # Create a temporary config file path
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
        config_file_path = temp_config_file.name

    with mock.patch(
        "sys.argv", ["nullpol_create_time_frequency_filter_from_sample", "--generate-config", config_file_path]
    ):
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
    default_config_file_path = str(
        files("nullpol.cli.templates") / "default_config_create_time_frequency_filter_from_sample.ini"
    )
    with open(default_config_file_path) as f:
        default_generated_content = f.read()

    # Clean up the temporary config file
    os.remove(config_file_path)

    # Compare the content of both files
    assert generated_content.strip() == default_generated_content.strip()


def test_create_time_frequency_filter():
    """Test time-frequency filter creation from signal parameters.

    Validates that the tool can successfully generate time-frequency
    filters based on signal parameters and custom power spectral densities
    for multi-detector networks.
    """
    # Create a temporary config file path
    with tempfile.NamedTemporaryFile(suffix=".ini", delete=True) as temp_config_file:
        config_file_path = temp_config_file.name
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as temp_time_frequency_filter_file:
            time_frequency_filter_file_path = temp_time_frequency_filter_file.name
            example_signal_parameters_create_injection_path = str(
                files("nullpol.cli.templates") / "example_signal_parameters_create_injection.json"
            )
            current_dir = os.path.dirname(__file__)
            mock_psd_path = os.path.join(current_dir, "fixtures", "mock_psd.txt")
            with mock.patch(
                "sys.argv",
                [
                    "nullpol_create_time_frequency_filter_from_sample",
                    "--generate-config",
                    config_file_path,
                    "--output",
                    time_frequency_filter_file_path,
                    "--detectors",
                    "H1,L1,V1",
                    "--psds",
                    json.dumps(f'{{"H1": "{mock_psd_path}", "L1": "{mock_psd_path}", "V1": "{mock_psd_path}"}}'),
                    "--signal-parameters",
                    example_signal_parameters_create_injection_path,
                    "--nside",
                    "2",
                ],
            ):
                try:
                    main()
                except SystemExit as e:
                    assert e.code == 0  # Ensure that it exited with code 0

                # Execute the tool
                with mock.patch(
                    "sys.argv", ["nullpol_create_time_frequency_filter_from_sample", "--config", config_file_path]
                ):
                    try:
                        main()
                    except SystemExit as e:
                        assert e.code == 0  # Ensure that it exited with code 0

            # Check that the output file is created
            assert os.path.exists(time_frequency_filter_file_path)
