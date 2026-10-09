"""Integrations package."""

from __future__ import annotations

# Scheduler integrations
from . import htcondor, slurm

# Conditional asimov import
try:
    from . import asimov  # Asimov pipeline integration

    _ASIMOV_AVAILABLE = True
except ImportError:
    asimov = None
    _ASIMOV_AVAILABLE = False

__all__ = [
    "htcondor",
    "slurm",
]

# Only export asimov if it's available
if _ASIMOV_AVAILABLE:
    __all__.append("asimov")
