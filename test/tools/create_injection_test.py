import unittest
from unittest import mock
import tempfile
import os
import pkg_resources
import json
from nullpol.tools.create_injection import main

class TestCreateInjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Create a temporary directory for testing."""
        cls.test_dir = tempfile.mkdtemp()
        cls.H1_TEST_frame_path = os.path.join(cls.test_dir, 'H1-TEST-2024-16.gwf')
        cls.L1_TEST_frame_path = os.path.join(cls.test_dir, 'L1-TEST-2024-16.gwf')
        cls.V1_TEST_frame_path = os.path.join(cls.test_dir, 'V1-TEST-2024-16.gwf')
        cls.H1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, 'H1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf')
        cls.L1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, 'L1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf')
        cls.V1_TEST_WITH_SIGNAL_FRAME_frame_path = os.path.join(cls.test_dir, 'V1-TEST_WITH_SIGNAL_FRAME-2024-16.gwf')
        cls.H1_TEST_PSD_frame_path = os.path.join(cls.test_dir, 'H1-TEST_PSD-2024-16.gwf')
        cls.L1_TEST_PSD_frame_path = os.path.join(cls.test_dir, 'L1-TEST_PSD-2024-16.gwf')
        cls.V1_TEST_PSD_frame_path = os.path.join(cls.test_dir, 'V1-TEST_PSD-2024-16.gwf')

    @classmethod
    def tearDownClass(cls):
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
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        with mock.patch('sys.argv', ['nullpol-create-injection', '--generate-config', config_file_path]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Check that the config file was generated
        self.assertTrue(os.path.exists(config_file_path))

        # Verify the contents of the generated config file
        with open(config_file_path, 'r') as f:
            generated_content = f.read()

        # Load the default config file
        default_config_file_path = pkg_resources.resource_filename('nullpol.tools', 'default_config_create_injection.ini')
        with open(default_config_file_path, 'r') as f:
            default_generated_content = f.read()

        # Clean up the temporary config file
        os.remove(config_file_path)

        # Compare the content of both files
        self.assertEqual(generated_content.strip(), default_generated_content.strip())

    def test_create_injection(self):
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename('nullpol.tools', 'example_signal_parameters_create_injection.json')
        with mock.patch('sys.argv', ['nullpol-create-injection',
                                     '--generate-config', config_file_path,
                                     '--signal-parameters', example_signal_parameters_create_injection_path,
                                     '--outdir', self.test_dir,
                                     '--label', 'TEST',
                                     '--start-time', '2024',
                                     '--duration', '16',
                                     '--detectors', 'H1, L1, V1']):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch('sys.argv', ['nullpol-create-injection', '--config', config_file_path]):
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
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename('nullpol.tools', 'example_signal_parameters_create_injection.json')
        print('testing', self.H1_TEST_frame_path)
        with mock.patch('sys.argv', ['nullpol-create-injection',
                                     '--generate-config', config_file_path,
                                     '--signal-parameters', example_signal_parameters_create_injection_path,
                                     '--outdir', self.test_dir,
                                     '--label', 'TEST_WITH_SIGNAL_FRAME',
                                     '--start-time', '2024',
                                     '--duration', '16',
                                     '--detectors', 'H1, L1, V1',
                                     '--signal-files', json.dumps(f'{{"H1": "{self.H1_TEST_frame_path}", "L1": "{self.L1_TEST_frame_path}", "V1": "{self.V1_TEST_frame_path}"}}'),
                                     '--signal-file-channels', json.dumps('{"H1": "H1:STRAIN", "L1": "L1:STRAIN", "V1": "V1:STRAIN"}')]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch('sys.argv', ['nullpol-create-injection', '--config', config_file_path]):
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
        # Create a temporary config file path
        with tempfile.NamedTemporaryFile(suffix='.ini', delete=False) as temp_config_file:
            config_file_path = temp_config_file.name
        example_signal_parameters_create_injection_path = pkg_resources.resource_filename('nullpol.tools', 'example_signal_parameters_create_injection.json')
        current_dir = os.path.dirname(__file__)
        mock_psd_path = os.path.join(current_dir, 'mock_psd.txt')
        with mock.patch('sys.argv', ['nullpol-create-injection',
                                     '--generate-config', config_file_path,
                                     '--signal-parameters', example_signal_parameters_create_injection_path,
                                     '--outdir', self.test_dir,
                                     '--label', 'TEST_PSD',
                                     '--start-time', '2024',
                                     '--duration', '16',
                                     '--detectors', 'H1, L1, V1',
                                     '--psds', json.dumps(f'{{"H1": "{mock_psd_path}", "L1": "{mock_psd_path}", "V1": "{mock_psd_path}"}}')]):
            try:
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0
        # Execute the tool
        with mock.patch('sys.argv', ['nullpol-create-injection', '--config', config_file_path]):
            try:                
                main()
            except SystemExit as e:
                self.assertEqual(e.code, 0)  # Ensure that it exited with code 0

        # Clean up the temporary config file
        #os.remove(config_file_path)

        # Check that the output file is created        
        self.assertTrue(os.path.exists(self.H1_TEST_PSD_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.L1_TEST_PSD_frame_path), "Output file was not created.")
        self.assertTrue(os.path.exists(self.V1_TEST_PSD_frame_path), "Output file was not created.")        

if __name__ == '__main__':
    unittest.main()