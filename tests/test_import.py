"""Test module for package import functionality and CLI tools.

This module validates that all core package modules can be imported correctly
and that CLI command-line interfaces work as expected in the installed environment.
"""

# pylint: disable=import-outside-toplevel  # Testing import functionality requires imports inside functions

import subprocess

import pytest


@pytest.mark.integration
def test_nullpol_import():
    """Test that the main nullpol package can be imported without errors."""
    import nullpol  # noqa: F401  # pylint: disable=unused-import


def test_core_module_imports():
    """Test that all core modules (auto-imported) can be accessed."""
    import nullpol

    # Core modules that should be auto-imported
    core_modules = ["analysis", "simulation", "utils"]

    for module_name in core_modules:
        assert hasattr(nullpol, module_name), f"Missing core module: {module_name}"
        module = getattr(nullpol, module_name)
        assert module is not None, f"Module {module_name} is None"


@pytest.mark.integration
def test_simulation_import():
    """Test that simulation modules can be imported without errors."""
    import nullpol.simulation  # noqa: F401  # pylint: disable=unused-import


@pytest.mark.integration
def test_analysis_import():
    """Test that analysis modules can be imported without errors."""
    import nullpol.analysis  # noqa: F401  # pylint: disable=unused-import


@pytest.mark.integration
def test_utils_import():
    """Test that utility modules can be imported without errors."""
    import nullpol.utils  # noqa: F401  # pylint: disable=unused-import


@pytest.mark.integration
def test_cli_module_import():
    """Test that CLI modules can be imported without errors."""
    import nullpol.cli  # noqa: F401  # pylint: disable=unused-import


@pytest.mark.integration
def test_integrations_module_import():
    """Test that integrations module can be imported (with optional dependencies)."""
    import nullpol.integrations

    # Verify the module is properly loaded
    assert nullpol.integrations is not None

    # HTCondor integration should always be available
    assert hasattr(nullpol.integrations, "htcondor")

    # Test that we can get the list of available integrations
    available_modules = getattr(nullpol.integrations, "__all__", [])
    assert isinstance(available_modules, list)
    assert "htcondor" in available_modules

    # Test that we can access each available module
    for module_name in available_modules:
        assert hasattr(nullpol.integrations, module_name)
        module = getattr(nullpol.integrations, module_name)
        assert module is not None, f"Module {module_name} is None"


@pytest.mark.integration
def test_package_api_design():
    """Test the package API design - core modules auto-imported, advanced modules on-demand."""
    # Start a fresh Python process to test clean imports
    import sys

    # Test core modules are auto-imported
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import nullpol
core_modules = ['analysis', 'simulation', 'utils']
main_attrs = [attr for attr in dir(nullpol) if not attr.startswith('_')]
for module_name in core_modules:
    assert module_name in main_attrs, f'Core module {module_name} not auto-imported'

# Advanced modules should NOT be auto-imported
advanced_modules = ['cli', 'integrations']
for module_name in advanced_modules:
    assert module_name not in main_attrs, f'Advanced module {module_name} should not be auto-imported, but found in: {main_attrs}'

print('API design test passed')
""",
        ],
        capture_output=True,
        text=True,
        cwd=None,  # Use current working directory
    )

    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        assert False, f"API design test failed: {result.stderr}"

    assert "API design test passed" in result.stdout

    # But advanced modules should be importable explicitly
    import nullpol.cli
    import nullpol.integrations

    assert nullpol.cli is not None
    assert nullpol.integrations is not None


@pytest.mark.parametrize(
    "module_path",
    [
        "nullpol.simulation.injection",
        "nullpol.analysis.likelihood",
        "nullpol.analysis.null_stream",
        "nullpol.analysis.tf_transforms",
        "nullpol.cli.create_injection",
        "nullpol.integrations.htcondor",
        "nullpol.utils.filesystem",
    ],
)
def test_specific_submodule_imports(module_path):
    """Test that specific important submodules can be imported."""
    # Import the module dynamically
    import importlib

    module = importlib.import_module(module_path)

    # Verify it was imported successfully
    assert module is not None, f"Failed to import {module_path}"
