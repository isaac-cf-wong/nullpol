"""Test module for logging utility functions.

This module tests the logging configuration utilities used throughout the
package.
"""

from __future__ import annotations

import unittest

from nullpol.utils import setup_logger


class TestLog(unittest.TestCase):
    """Test class for logging utility functions.

    This class validates the logging configuration setup to ensure
    proper initialization of logging systems.
    """

    def test_setup_logger(self):
        """Test logger setup and configuration.

        Validates that the logger setup function executes successfully
        and configures the logging system appropriately for analysis
        pipeline monitoring and debugging.
        """
        setup_logger()


if __name__ == "__main__":
    unittest.main()
