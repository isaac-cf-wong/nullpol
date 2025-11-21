"""Tests for time_frequency_likelihood module."""

from unittest.mock import patch

import pytest

from nullpol.analysis.likelihood.time_frequency_likelihood import TimeFrequencyLikelihood


class TestTimeFrequencyLikelihoodSimple:
    """Simple unit tests for TimeFrequencyLikelihood base class."""

    def test_log_likelihood_not_implemented_error(self):
        """Test that log_likelihood raises NotImplementedError in base class."""
        # Create mock likelihood instance directly without going through __init__
        likelihood = TimeFrequencyLikelihood.__new__(TimeFrequencyLikelihood)

        # Test that log_likelihood raises NotImplementedError
        with pytest.raises(NotImplementedError):
            likelihood.log_likelihood()

    def test_calculate_noise_log_likelihood_not_implemented_error(self):
        """Test that _calculate_noise_log_likelihood raises NotImplementedError in base class."""
        # Create mock likelihood instance directly without going through __init__
        likelihood = TimeFrequencyLikelihood.__new__(TimeFrequencyLikelihood)

        # Test that _calculate_noise_log_likelihood raises NotImplementedError
        with pytest.raises(NotImplementedError):
            likelihood._calculate_noise_log_likelihood()

    def test_noise_log_likelihood_caching_and_calculation(self):
        """Test noise_log_likelihood property caches and calculates correctly."""
        # Create mock likelihood instance directly without going through __init__
        likelihood = TimeFrequencyLikelihood.__new__(TimeFrequencyLikelihood)

        # Test case 1: Cache is None initially, should call _calculate_noise_log_likelihood
        likelihood._noise_log_likelihood_value = None

        # Mock _calculate_noise_log_likelihood to return a specific value
        with patch.object(likelihood, "_calculate_noise_log_likelihood", return_value=-2.5):
            result = likelihood.noise_log_likelihood()
            assert result == -2.5
            assert likelihood._noise_log_likelihood_value == -2.5  # Should be cached

        # Test case 2: Cache is already set, should return cached value without calling _calculate_noise_log_likelihood
        likelihood._noise_log_likelihood_value = -1.8  # Pre-set cache

        with patch.object(likelihood, "_calculate_noise_log_likelihood") as mock_calc:
            result = likelihood.noise_log_likelihood()
            assert result == -1.8
            mock_calc.assert_not_called()  # Should not call calculation when cached
