"""Test module for injection creation tool.

This module tests the command-line tool for creating signal injections into
detector data.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from unittest import mock

import pkg_resources

from nullpol.tools.create_injection import main


class TestCreateInjection(unittest.TestCase):
    """Test class for the injection creation command-line tool.

    This class validates the functionality of the nullpol_create_injection
    tool across various configuration scenarios including noise-only data,
    signal injections, and custom power spectral density configurations.
    """

    @classmethod
    def setUpClass(cls):
        """Create temporary directory and define output file paths for testing.

        Sets up a temporary directory structure and defines expected output
        file paths for various test scenarios including different noise
        realizations and detector configurations.
        """
        cls.test_dir = tempfile.mkdtemp()
        cls.H1_TEST_frame_path = os.path.join(cls.test_dir, "H1-TEST-2024-16.gwf")
        cls.L1_TEST_frame_path = os.path.join(cls.test_dir, "L1-TEST-2024-16.gwf")
        cls.V1_TEST_frame_path = os.path.join(cls.test_dir, "V1-TEST-2024-16.gwf")
        cls.H1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, "H1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf")
        cls.L1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, "L1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf")
        cls.V1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, "V1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf")
        cls.H1_TEST_PSD_frame_path = os.path.join(cls.test_dir, "H1-TEST_PSD-2024-16.gwf")
        cls.L1_TEST_PSD_frame_path = os.path.join(cls.test_dir, "L1-TEST_PSD-2024-16.gwf")
        cls.V1_TEST_PSD_frame_path = os.path.join(cls.test_dir, "V1-TEST_PSD-2024-16.gwf")

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary files created during testing.

        Removes all temporary frame files created during the test execution
        to maintain a clean testing environment.
        """
        if os.path.exists(cls.H1_TEST_frame_path):
            os.remove(cls.H1_TEST_frame_path)
        if os.path.exists(cls.L1_TEST_frame_path):
            os.remove(cls.L1_TEST_frame_path)
        if os.path.exists(cls.V1_TEST_frame_path):
            os.remove(cls.V1_TEST_frame_path)
        if os.path.exists(cls.H1_TEST_WITH_SIGNAL_FRAME_frame_path):
            os.remove(cls.H1_TEST_WITH_SIGNAL_FRAME_frame_path)
        if os.path.exists(cls.L1_TEST_WITH_SIGNAL_FRAME_frame_path):
            os.remove(cls.L1_TEST_WITH_SIGNAL_FRAME_frame_path)
        if os.path.exists(cls.V1_TEST_WITH_SIGNAL_FRAME_frame_path):
            os.remove(cls.V1_TEST_WITH_SIGNAL_FRAME_frame_path)
        if os.path.exists(cls.H1_TEST_PSD_frame_path):
            os.remove(cls.H1_TEST_PSD_frame_path)
        if os.path.exists(cls.L1_TEST_PSD_frame_path):
            os.remove(cls.L1_TEST_PSD_frame_path)
        if os.path.exists(cls.V1_TEST_PSD_frame_path):
            os.remove(cls.V1_TEST_PSD_frame_path)

    def test_generate_config(self):
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
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Check that the config file was generated
        self.assertTrue(os.path.exists(config_file_path))

        # Verify the contents of the generated config file
        with open(config_file_path) as f:
            generated_content = f.read()

        # Load the default config file
        default_config_file_path = pkg_resources.resource_filename(
            "nullpol.tools", "default_config_create_injection.ini"
        )
        with open(default_config_file_path) as f:
            default_generated_content = f.read()

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Compare the content of both files
        self.assertEqual(generated_content.strip(), default_generated_content.strip())

    def test_create_injection(self):
        """Test basic signal injection creation.

        Validates that the tool can successfully create signal frame files
        containing injected binary black hole signals for all specified
        detectors in a multi-detector network.
        """
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename(
            "nullpol.tools", "example_signal_parameters_create_injection.json"
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
                self.test_dir,
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
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch("sys.argv", ["nullpol_create_injection", "--config", config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Check that the output file is created
        self.assertTrue(os.path.exists(self.H1_TEST_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.L1_TEST_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.V1_TEST_frame_path), "Output file was not created.")

    def test_create_injection_with_signal_frame(self):
        """Test injection creation using existing signal frame files.

        Validates that the tool can create new injections by combining
        existing signal frame files with additional signals, enabling complex
        multi-signal injection scenarios.
        """
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename(
            "nullpol.tools", "example_signal_parameters_create_injection.json"
        )
        print("testing", self.H1_TEST_frame_path)
        with mock.patch(
            "sys.argv",
            [
                "nullpol_create_injection",
                "--generate-config",
                config_file_path,
                "--signal-parameters",
                example_signal_parameters_create_injection_path,
                "--outdir",
                self.test_dir,
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
                    f'{{"H1": "{self.H1_TEST_frame_path}", "L1": "{self.L1_TEST_frame_path}", "V1": "{self.V1_TEST_frame_path}"}}'
                ),
                "--signal-file-channels",
                json.dumps('{"H1": "H1:STRAIN", "L1": "L1:STRAIN", "V1": "V1:STRAIN"}'),
            ],
        ):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch("sys.argv", ["nullpol_create_injection", "--config", config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Check that the output file is created
        self.assertTrue(os.path.exists(self.H1_TEST_WITH_SIGNAL_FRAME_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.L1_TEST_WITH_SIGNAL_FRAME_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.V1_TEST_WITH_SIGNAL_FRAME_frame_path), "Output file was not created.")

    def test_create_injection_with_custom_psds(self):
        """Test injection creation using custom power spectral densities.

        Validates that the tool can generate injections using user-provided
        power spectral density files instead of default detector noise curves,
        enabling studies with realistic or historical detector sensitivities.
        """
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix=".ini", delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename(
            "nullpol.tools", "example_signal_parameters_create_injection.json"
        )
        current_dir = os.path.dirname(__file__)
        mock_psd_path = os.path.join(current_dir, "mock_psd.txt")
        with mock.patch(
            "sys.argv",
            [
                "nullpol_create_injection",
                "--generate-config",
                config_file_path,
                "--signal-parameters",
                example_signal_parameters_create_injection_path,
                "--outdir",
                self.test_dir,
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
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch("sys.argv", ["nullpol_create_injection", "--config", config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        # os.remove(config_file_path)

        # Check that the output file is created
        self.assertTrue(os.path.exists(self.H1_TEST_PSD_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.L1_TEST_PSD_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.V1_TEST_PSD_frame_path), "Output file was not created.")


if __name__ == "__main__":
    unittest.main()
