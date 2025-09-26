"""Utilities for dynamic module importing."""

from __future__ import annotations

import importlib


def import_function(path: str):
    """Import a function from a module path string.

    Args:
        path: String in format 'module.path.function_name'

    Returns:
        The imported function
    """
    module_path, func_name = path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    func = getattr(module, func_name)
    return func
