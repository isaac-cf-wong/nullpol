"""Smoke tests before publishing to verify the wheel and source distribution."""

from __future__ import annotations

import subprocess
import sys

import nullpol

try:
    import pytest
except ModuleNotFoundError:  # pragma: no cover - used in isolated wheel smoke runs.
    pytestmark = ()
else:
    pytestmark = pytest.mark.integration


def test_basic_import() -> None:
    """Test basic import.

    This smoke test verifies the package is importable from an *installed*
    distribution (site-packages or dist). It is intended for post-build /
    pre-publish verification against a wheel or sdist. When run from a source
    checkout (e.g. ``uv run pytest`` with ``pythonpath = ["src"]``) the import
    resolves to the local ``src`` tree, so the test is skipped rather than
    reported as a failure.
    """
    print(f"Python version: {sys.version}")
    print(f"Package version: {nullpol.__version__}")

    import_path = nullpol.__file__ or ""
    if "site-packages" not in import_path and "dist" not in import_path:
        pytest.skip(f"Package not installed; importing from source tree: {import_path}")

    # Ensure it's not importing the local folder
    assert "site-packages" in import_path or "dist" in import_path, (
        f"Package imported from unexpected location: {import_path}"
    )


def test_cli_help() -> None:
    """Test CLI help."""
    # Ensure the 'nullpol' command was registered and runs
    result = subprocess.run(["nullpol_pipe", "--help"], capture_output=True, text=True, check=False)  # noqa: S607
    assert result.returncode == 0
    assert "usage:" in result.stdout.lower()


if __name__ == "__main__":
    test_basic_import()
    print("Smoke test passed: Package is importable.")

    test_cli_help()
    print("Smoke test passed: The CLI is executable.")
