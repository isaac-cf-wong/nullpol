"""Test module for null stream calculator functionality.

This module tests the NullStreamCalculator class and its methods
for computing null projections and energies.
"""

from __future__ import annotations

import pytest

from nullpol.analysis.null_stream.calculator import NullStreamCalculator


@pytest.fixture
def calculator_instance():
    """Create a calculator instance for testing."""
    return NullStreamCalculator()


class TestNullStreamCalculator:
    """Test class for NullStreamCalculator."""

    def test_calculator_initialization(self):
        """Test calculator initialization."""
        calculator = NullStreamCalculator()
        # Calculator should initialize without errors
        assert calculator is not None
