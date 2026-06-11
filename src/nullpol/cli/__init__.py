"""Cli package."""

from __future__ import annotations

# Command-line interface tools
# These correspond to the entry points in pyproject.toml
# Main pipeline tools
# Utility tools (always available)
from . import (
    create_injection,  # nullpol_create_injection
    create_time_frequency_filter_from_sample,  # nullpol_create_time_frequency_filter_from_sample
    data_analysis,  # nullpol_pipe_analysis
    data_generation,  # nullpol_pipe_generation
    main,  # nullpol_pipe
    parser,  # nullpol_pipe_write_default_ini
)
from . import input as cli_input  # Input handling shared across CLI tools

# Conditional asimov-dependent imports
try:
    from . import get_asimov_yaml  # nullpol_get_asimov_yaml

    _ASIMOV_CLI_AVAILABLE = True
except ImportError:
    get_asimov_yaml = None
    _ASIMOV_CLI_AVAILABLE = False

__all__ = [
    "cli_input",
    "create_injection",
    "create_time_frequency_filter_from_sample",
    "data_analysis",
    "data_generation",
    "main",
    "parser",
]

# Only export asimov tools if available
if _ASIMOV_CLI_AVAILABLE:
    __all__.append("get_asimov_yaml")
