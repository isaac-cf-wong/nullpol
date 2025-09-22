"""Test module for logging utility functions.

This module tests the logging configuration utilities used throughout the
package for analysis pipeline monitoring and debugging.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from nullpol.utils import setup_logger


def test_setup_logger_basic():
    """Test basic logger setup without parameters."""
    setup_logger()
    logger = logging.getLogger("nullpol")
    
    # Should have at least one handler and be configured
    assert len(logger.handlers) >= 1
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    # Clean up
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def test_setup_logger_with_log_level():
    """Test logger setup with different log levels."""
    setup_logger(log_level="DEBUG")
    logger = logging.getLogger("nullpol")
    assert logger.level == logging.DEBUG
    
    setup_logger(log_level="WARNING")
    assert logger.level == logging.WARNING
    
    # Clean up
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)


def test_setup_logger_with_file_output():
    """Test logger setup with file output."""
    with tempfile.TemporaryDirectory() as temp_dir:
        setup_logger(outdir=temp_dir, label="test_log")
        logger = logging.getLogger("nullpol")
        
        # Should have both stream and file handlers
        assert len(logger.handlers) >= 2
        assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
        assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
        
        # Check that log file was created
        log_file = Path(temp_dir) / "test_log.log"
        assert log_file.exists()
        
        # Clean up
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)