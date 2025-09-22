"""Test module for logging utility functions.

This module tests the logging configuration utilities used throughout the
package.
"""

from __future__ import annotations

from nullpol.utils import setup_logger


def test_setup_logger():
    """Test logger setup and configuration.

    Validates that the logger setup function executes successfully
    and configures the logging system appropriately for analysis
    pipeline monitoring and debugging.
    """
    setup_logger()
