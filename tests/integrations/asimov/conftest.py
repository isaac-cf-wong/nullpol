"""Shared fixtures for the asimov integration tests.

The ``asimov`` and ``cbcflow`` packages are *optional* dependencies (see the
``asimov`` extra in ``pyproject.toml``) and are not installed in the default
test environment. Importing :mod:`nullpol.integrations.asimov` normally
triggers ``asimov.py`` / ``pesummary.py`` which require those packages (and
``htcondor``) at import time.

To allow the pure-Python logic in :mod:`nullpol.integrations.asimov.tgrflow`
and :mod:`nullpol.integrations.asimov.utility` to be exercised without the
optional deps, these fixtures:

1. Inject lightweight :class:`~unittest.mock.MagicMock` stub modules for
   ``asimov``, ``asimov.event`` and ``cbcflow`` into :data:`sys.modules`.
2. Replace the real ``nullpol.integrations.asimov`` package object with an
   empty package module whose ``__path__`` points at the real source directory,
   so the submodules ``tgrflow`` and ``utility`` can be loaded by the import
   machinery without running the package ``__init__`` (and therefore without
   importing the real ``asimov`` / ``htcondor``).

All stubs are scoped to the test via :class:`pytest.MonkeyPatch` and are
removed again afterwards.
"""

from __future__ import annotations

import os
import sys
import types
from unittest import mock

import pytest

_STUB_MODULE_PATHS = ("asimov", "asimov.event", "cbcflow")


def _make_stub(name: str) -> mock.MagicMock:
    """Build a stub module that auto-creates any accessed attribute."""
    mod = mock.MagicMock(name=name)
    mod.__name__ = name
    mod.__path__ = []
    return mod


@pytest.fixture
def asimov_libs(monkeypatch):
    """Import the asimov integration modules with optional deps stubbed out.

    Returns a :class:`types.SimpleNamespace` exposing the loaded ``tgrflow``
    and ``utility`` modules together with the most commonly used symbols.
    """
    for path in _STUB_MODULE_PATHS:
        monkeypatch.setitem(sys.modules, path, _make_stub(path))

    # Replace the real asimov package __init__ with an empty package module so
    # that importing its submodules does not trigger the heavy asimov.py /
    # pesummary.py imports (which require the real asimov + htcondor packages).
    import nullpol

    asimov_dir = os.path.join(os.path.dirname(nullpol.__file__), "integrations", "asimov")
    fake_pkg = types.ModuleType("nullpol.integrations.asimov")
    fake_pkg.__path__ = [asimov_dir]
    fake_pkg.__package__ = "nullpol.integrations.asimov"
    monkeypatch.setitem(sys.modules, "nullpol.integrations.asimov", fake_pkg)

    from nullpol.integrations.asimov import tgrflow, utility

    return types.SimpleNamespace(
        tgrflow=tgrflow,
        utility=utility,
        validate_gr_pe_result=tgrflow.validate_gr_pe_result,
        identify_basis_production=tgrflow.identify_basis_production,
        Applicator=tgrflow.Applicator,
        Collector=tgrflow.Collector,
        convert_string_to_dict=utility._convert_string_to_dict,
        bilby_config_to_asimov=utility.bilby_config_to_asimov,
        deep_update=utility.deep_update,
    )
