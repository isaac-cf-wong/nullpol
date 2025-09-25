from __future__ import annotations

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pytest
from bilby.core.prior import Cosine, Uniform

from nullpol.analysis.prior.default import DEFAULT_PRIOR_DIR, PolarizationPriorDict


@pytest.mark.integration
class TestDefaultPriorDir:
    """Test the DEFAULT_PRIOR_DIR constant."""

    def test_default_prior_dir_exists(self):
        """Test that the default prior directory exists."""
        assert os.path.exists(DEFAULT_PRIOR_DIR)
        assert os.path.isdir(DEFAULT_PRIOR_DIR)

    def test_default_prior_dir_path(self):
        """Test that the default prior directory has the expected path structure."""
        assert DEFAULT_PRIOR_DIR.endswith("prior_files")
        assert "nullpol" in DEFAULT_PRIOR_DIR
        assert "analysis" in DEFAULT_PRIOR_DIR
        assert "prior" in DEFAULT_PRIOR_DIR

    def test_default_polarization_prior_file_exists(self):
        """Test that the default polarization prior file exists."""
        polarization_prior_file = os.path.join(DEFAULT_PRIOR_DIR, "polarization.prior")
        assert os.path.exists(polarization_prior_file)
        assert os.path.isfile(polarization_prior_file)


@pytest.mark.integration
class TestPolarizationPriorDict:
    """Test the PolarizationPriorDict class."""

    def test_init_with_default_priors(self):
        """Test initialization with default polarization priors."""
        priors = PolarizationPriorDict()

        # Should have the three default parameters
        assert "dec" in priors
        assert "ra" in priors
        assert "psi" in priors

        # Check that they are the expected prior types
        assert isinstance(priors["dec"], Cosine)
        assert isinstance(priors["ra"], Uniform)
        assert isinstance(priors["psi"], Uniform)

    def test_init_with_custom_dictionary(self):
        """Test initialization with a custom prior dictionary."""
        custom_priors = {
            "amplitude_pp": Uniform(0, 1, name="amplitude_pp"),
            "phase": Uniform(0, 2 * np.pi, name="phase"),
        }
        priors = PolarizationPriorDict(dictionary=custom_priors)

        assert "amplitude_pp" in priors
        assert "phase" in priors
        assert len(priors) == 2
        assert isinstance(priors["amplitude_pp"], Uniform)
        assert isinstance(priors["phase"], Uniform)

    def test_init_with_custom_filename(self):
        """Test initialization with a custom prior filename."""
        # Create a temporary prior file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".prior", delete=False) as f:
            f.write("test_param = Uniform(name='test_param', minimum=0, maximum=1)\n")
            temp_filename = f.name

        try:
            priors = PolarizationPriorDict(filename=temp_filename)
            assert "test_param" in priors
            assert isinstance(priors["test_param"], Uniform)
        finally:
            os.unlink(temp_filename)

    def test_init_with_filename_in_default_dir(self):
        """Test initialization with a filename that exists in the default directory."""
        priors = PolarizationPriorDict(filename="polarization.prior")

        # Should load the default polarization priors
        assert "dec" in priors
        assert "ra" in priors
        assert "psi" in priors

    def test_init_with_nonexistent_filename_searches_default_dir(self):
        """Test that nonexistent filenames are searched in the default directory."""
        # This should try to find 'nonexistent.prior' in the default directory
        with pytest.raises(FileNotFoundError):
            PolarizationPriorDict(filename="nonexistent.prior")

    def test_default_polarization_prior_parameters(self):
        """Test that default polarization priors have correct parameters."""
        priors = PolarizationPriorDict()

        # Test declination (cosine distribution)
        dec_prior = priors["dec"]
        assert dec_prior.name == "dec"

        # Test right ascension (uniform, periodic)
        ra_prior = priors["ra"]
        assert ra_prior.name == "ra"
        assert ra_prior.minimum == 0
        assert ra_prior.maximum == 2 * np.pi
        assert ra_prior.boundary == "periodic"

        # Test polarization angle (uniform, periodic)
        psi_prior = priors["psi"]
        assert psi_prior.name == "psi"
        assert psi_prior.minimum == 0
        assert psi_prior.maximum == np.pi
        assert psi_prior.boundary == "periodic"

    @patch("nullpol.analysis.prior.default.logger")
    def test_logging_when_using_default_priors(self, mock_logger):
        """Test that appropriate logging occurs when using default priors."""
        PolarizationPriorDict()
        mock_logger.info.assert_called_once()
        call_args = mock_logger.info.call_args[0][0]
        assert "No prior given, using default polarization priors" in call_args

    def test_validate_prior_returns_true(self):
        """Test that validate_prior always returns True."""
        priors = PolarizationPriorDict()
        assert priors.validate_prior() is True

    def test_validate_prior_with_kwargs(self):
        """Test that validate_prior accepts arbitrary keyword arguments."""
        priors = PolarizationPriorDict()
        assert priors.validate_prior(some_param=True, another_param="test") is True

    def test_inheritance_from_prior_dict(self):
        """Test that PolarizationPriorDict properly inherits from PriorDict."""
        from bilby.core.prior.dict import PriorDict  # pylint: disable=import-outside-toplevel

        priors = PolarizationPriorDict()
        assert isinstance(priors, PriorDict)

        # Should have all the standard PriorDict methods
        assert hasattr(priors, "sample")
        assert hasattr(priors, "ln_prob")
        assert hasattr(priors, "prob")

    def test_sample_from_default_priors(self):
        """Test sampling from the default polarization priors."""
        priors = PolarizationPriorDict()
        sample = priors.sample()

        # Should contain all three parameters
        assert "dec" in sample
        assert "ra" in sample
        assert "psi" in sample

        # Check parameter ranges
        assert -1 <= np.cos(sample["dec"]) <= 1  # Cosine distribution constraint
        assert 0 <= sample["ra"] <= 2 * np.pi
        assert 0 <= sample["psi"] <= np.pi

    def test_ln_prob_calculation(self):
        """Test log probability calculation."""
        priors = PolarizationPriorDict()

        # Valid sample
        valid_sample = {
            "dec": 0.5,  # Valid declination
            "ra": 1.0,  # Valid right ascension
            "psi": 0.5,  # Valid polarization angle
        }
        ln_prob = priors.ln_prob(valid_sample)
        assert np.isfinite(ln_prob)

        # Invalid sample (outside bounds)
        invalid_sample = {"dec": 0.5, "ra": -1.0, "psi": 0.5}  # Invalid (below minimum)
        ln_prob = priors.ln_prob(invalid_sample)
        assert ln_prob == -np.inf

    def test_prior_dict_functionality_preserved(self):
        """Test that all PriorDict functionality is preserved."""
        # Test with custom priors to ensure inheritance works correctly
        custom_priors = {"param1": Uniform(0, 1, name="param1"), "param2": Uniform(0, 10, name="param2")}
        priors = PolarizationPriorDict(dictionary=custom_priors)

        # Test basic dictionary operations
        assert len(priors) == 2
        assert list(priors.keys()) == ["param1", "param2"]

        # Test sampling
        sample = priors.sample()
        assert 0 <= sample["param1"] <= 1
        assert 0 <= sample["param2"] <= 10

        # Test probability calculation
        test_sample = {"param1": 0.5, "param2": 5.0}
        assert np.isfinite(priors.ln_prob(test_sample))

    def test_empty_dictionary_initialization(self):
        """Test initialization with an empty dictionary."""
        priors = PolarizationPriorDict(dictionary={})
        assert len(priors) == 0

    def test_mixed_prior_types(self):
        """Test initialization with different types of priors."""
        from bilby.core.prior import DeltaFunction, LogUniform  # pylint: disable=import-outside-toplevel

        mixed_priors = {
            "uniform_param": Uniform(0, 1, name="uniform_param"),
            "log_uniform_param": LogUniform(1e-3, 1, name="log_uniform_param"),
            "delta_param": DeltaFunction(1.5, name="delta_param"),
        }

        priors = PolarizationPriorDict(dictionary=mixed_priors)

        assert len(priors) == 3
        assert isinstance(priors["uniform_param"], Uniform)
        assert isinstance(priors["log_uniform_param"], LogUniform)
        assert isinstance(priors["delta_param"], DeltaFunction)

    @patch("nullpol.analysis.prior.default.os.path.isfile")
    def test_filename_path_resolution(self, mock_isfile):
        """Test the filename path resolution logic."""
        # Mock that the file doesn't exist at the original path
        mock_isfile.return_value = False

        # This should try the original path, then construct the default directory path
        # The FileNotFoundError will be raised by bilby's PriorDict.__init__ when
        # it tries to load the non-existent file
        with pytest.raises(FileNotFoundError):
            PolarizationPriorDict(filename="custom.prior")

        # Should have called isfile once to check if "custom.prior" exists
        mock_isfile.assert_called_once_with("custom.prior")

    @patch("nullpol.analysis.prior.default.os.path.isfile")
    @patch("bilby.core.prior.dict.PriorDict.__init__")
    def test_filename_default_directory_construction(self, mock_super_init, mock_isfile):
        """Test that filenames are correctly constructed for the default directory."""
        # Mock that the file doesn't exist at the original path
        mock_isfile.return_value = False
        # Mock the super().__init__ to avoid FileNotFoundError
        mock_super_init.return_value = None

        PolarizationPriorDict(filename="custom.prior")

        # Check that super().__init__ was called with the default directory path
        expected_path = os.path.join(DEFAULT_PRIOR_DIR, "custom.prior")
        mock_super_init.assert_called_once_with(dictionary=None, filename=expected_path)
