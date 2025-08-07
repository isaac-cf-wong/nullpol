from __future__ import annotations

# External integrations
from . import htcondor  # HTCondor job creation (formerly job_creation)

# Conditional asimov import
try:
    from . import asimov  # Asimov pipeline integration

    _ASIMOV_AVAILABLE = True
except ImportError:
    asimov = None
    _ASIMOV_AVAILABLE = False

__all__ = [
    "htcondor",
]

# Only export asimov if it's available
if _ASIMOV_AVAILABLE:
    __all__.append("asimov")
