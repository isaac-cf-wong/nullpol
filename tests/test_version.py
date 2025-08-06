"""Test module for nullpol version information functionality.

This module tests the version reporting capabilities of the package,
ensuring that version information can be retrieved and logged correctly.
This is critical for reproducibility.
"""

from __future__ import annotations

from nullpol import get_version_information, log_version_information


def test_get_version_information():
    """Test that version information can be retrieved without errors.

    Verifies that the get_version_information function executes successfully
    and returns version details for dependency tracking.
    """
    get_version_information()


def test_log_version_information():
    """Test that version information can be logged without errors.

    Verifies that the log_version_information function executes successfully
    and logs version details to the appropriate output stream.
    """
    log_version_information()
