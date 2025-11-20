#!/usr/bin/env python3
"""
Auto-generate __init__.py files for nullpol package.

This script automatically generates __init__.py files for leaf packages (directories
that contain only .py files and no subdirectories). It imports all public functions,
classes, and constants (anything not starting with '_') from all Python modules in
the directory.

Higher-level __init__.py files in packages that contain subdirectories are left
untouched and should be manually maintained.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set


class PublicApiExtractor(ast.NodeVisitor):
    """Extract public API elements from a Python module."""

    def __init__(self):
        self.public_names: Set[str] = set()
        self.has_all: bool = False
        self.all_items: List[str] = []
        self.depth: int = 0  # Track nesting depth

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visit function definitions."""
        # Only add module-level functions (depth 0)
        if self.depth == 0 and not node.name.startswith("_"):
            self.public_names.add(node.name)

        # Increase depth for nested content
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visit async function definitions."""
        # Only add module-level functions (depth 0)
        if self.depth == 0 and not node.name.startswith("_"):
            self.public_names.add(node.name)

        # Increase depth for nested content
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visit class definitions."""
        # Only add module-level classes (depth 0)
        if self.depth == 0 and not node.name.startswith("_"):
            self.public_names.add(node.name)

        # Increase depth for nested content
        self.depth += 1
        self.generic_visit(node)
        self.depth -= 1

    def visit_Assign(self, node: ast.Assign) -> None:
        """Visit variable assignments."""
        # Only process module-level assignments (depth 0)
        if self.depth != 0:
            self.generic_visit(node)
            return

        # Handle __all__ specially
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                self.has_all = True
                if isinstance(node.value, ast.List):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            self.all_items.append(elt.value)
                return

        # Handle regular variable assignments (only module-level constants)
        for target in node.targets:
            if isinstance(target, ast.Name) and not target.id.startswith("_"):
                # Only include constants (all caps) or explicitly documented variables
                if target.id.isupper() or target.id in ["logger"]:
                    self.public_names.add(target.id)
        self.generic_visit(node)


def extract_public_api(file_path: Path) -> Set[str]:
    """Extract public API from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)
        extractor = PublicApiExtractor()
        extractor.visit(tree)

        # If __all__ is defined, use it; otherwise use extracted public names
        if extractor.has_all:
            return set(extractor.all_items)
        else:
            return extractor.public_names

    except (SyntaxError, UnicodeDecodeError, FileNotFoundError):
        # If we can't parse the file, return empty set
        return set()


def is_leaf_package(directory: Path) -> bool:
    """
    Check if a directory is a leaf package for auto-generation purposes.

    A leaf package is one where we want to auto-generate __init__.py files.
    This includes directories that:
    1. Contain .py files (excluding __init__.py)
    2. Have no code subdirectories (ignoring config/template/cache directories)

    Non-code subdirectories that don't disqualify a directory from being a leaf:
    - templates/ (configuration templates)
    - configs/ (configuration files)
    - data/ (static data files, when not containing Python modules)
    - assets/ (static assets)
    - __pycache__/ (Python cache)
    - .* (hidden directories)
    """
    has_python_files = False
    has_code_subdirectories = False

    # Directories that are considered "non-code" and don't disqualify leaf status
    non_code_subdirs = {
        "templates",
        "configs",
        "config",
        "assets",
        "static",
        "__pycache__",
        "prior_files",  # nullpol-specific: prior file storage
        "fixtures",  # test fixtures
        "data",  # data files (when not containing Python modules)
        "resources",  # resource files
    }

    for item in directory.iterdir():
        if item.is_file() and item.suffix == ".py" and item.name != "__init__.py":
            has_python_files = True
        elif item.is_dir() and not item.name.startswith("."):
            # Check if this is a code subdirectory (contains Python modules)
            if item.name not in non_code_subdirs:
                # If subdirectory contains .py files or __init__.py, it's a code directory
                if any(f.suffix == ".py" for f in item.iterdir() if f.is_file()):
                    has_code_subdirectories = True

    return has_python_files and not has_code_subdirectories


def should_auto_generate(directory: Path, excluded_dirs: Set[str], root_dir: Path) -> bool:
    """Check if we should auto-generate __init__.py for this directory."""
    # Get relative path from the scan root directory
    try:
        rel_path = str(directory.relative_to(root_dir))
    except ValueError:
        # If relative path calculation fails, use directory name
        rel_path = directory.name

    if rel_path in excluded_dirs or directory.name in excluded_dirs:
        return False

    # Only auto-generate for leaf packages
    return is_leaf_package(directory)


def generate_init_content(directory: Path) -> str:
    """Generate __init__.py content for a directory."""
    python_files = [f for f in directory.iterdir() if f.is_file() and f.suffix == ".py" and f.name != "__init__.py"]

    if not python_files:
        return "from __future__ import annotations\n\n"

    # Extract public API from all Python files
    all_exports: Dict[str, Set[str]] = {}
    for py_file in python_files:
        module_name = py_file.stem
        public_api = extract_public_api(py_file)
        if public_api:
            all_exports[module_name] = public_api

    if not all_exports:
        return "from __future__ import annotations\n\n"

    # Generate import statements
    lines = ["from __future__ import annotations", ""]

    # Sort modules for consistent output
    for module_name in sorted(all_exports.keys()):
        exports = all_exports[module_name]
        if exports:
            sorted_exports = sorted(exports)

            # Format imports based on length - use black-compatible formatting
            imports_line = f'from .{module_name} import {", ".join(sorted_exports)}'

            # If the line is too long (>88 chars, black's default), use multi-line format
            if len(imports_line) > 88:
                lines.append(f"from .{module_name} import (")
                for i, export in enumerate(sorted_exports):
                    if i == len(sorted_exports) - 1:
                        lines.append(f"    {export},")
                    else:
                        lines.append(f"    {export},")
                lines.append(")")
            else:
                lines.append(imports_line)

    # Generate __all__ list
    all_items = []
    for exports in all_exports.values():
        all_items.extend(exports)

    if all_items:
        lines.append("")
        lines.append("__all__ = [")
        for item in sorted(set(all_items)):
            lines.append(f'    "{item}",')
        lines.append("]")

    lines.append("")  # End with newline
    return "\n".join(lines)


def update_init_file(directory: Path, content: str) -> bool:
    """Update or create __init__.py file if content has changed."""
    init_file = directory / "__init__.py"

    # Check if content has changed
    existing_content = ""
    if init_file.exists():
        try:
            with open(init_file, "r", encoding="utf-8") as f:
                existing_content = f.read()
        except UnicodeDecodeError:
            pass

    if existing_content.strip() != content.strip():
        with open(init_file, "w", encoding="utf-8") as f:
            f.write(content)
        return True

    return False


def find_python_packages(root_dir: Path) -> List[Path]:
    """Find all Python packages (directories with __init__.py or that could be packages) under root_dir."""
    packages = []

    for item in root_dir.rglob("*"):
        if item.is_dir():
            # Include existing packages
            if (item / "__init__.py").exists():
                packages.append(item)
            # Also include directories with .py files (potential packages)
            elif any(f.suffix == ".py" for f in item.iterdir() if f.is_file() and f.name != "__init__.py"):
                packages.append(item)

    return packages


def auto_generate_init_files(root_dir: Path, excluded_dirs: Set[str] = None, dry_run: bool = False) -> Dict[str, str]:
    """
    Auto-generate __init__.py files for leaf packages.

    Args:
        root_dir: Root directory to scan
        excluded_dirs: Set of directory paths/names to exclude from auto-generation
        dry_run: If True, don't write files, just return what would be changed

    Returns:
        Dictionary mapping directory paths to the action taken
    """
    if excluded_dirs is None:
        excluded_dirs = set()

    # Add default exclusions
    default_exclusions = {
        "__pycache__",
        # High-level packages that should be manually maintained
        "nullpol",  # Main package entry point with curated imports
        "nullpol/analysis",  # High-level analysis organization (has subdirectories)
        "nullpol/cli",  # Complex CLI with conditional imports and templates/
        "nullpol/integrations",  # External integrations organization (has subdirectories)
    }
    excluded_dirs.update(default_exclusions)

    results = {}
    packages = find_python_packages(root_dir)

    for package_dir in packages:
        rel_path = str(package_dir.relative_to(root_dir))

        if should_auto_generate(package_dir, excluded_dirs, root_dir):
            content = generate_init_content(package_dir)

            if dry_run:
                # Check if content would actually change
                init_file = package_dir / "__init__.py"
                existing_content = ""
                if init_file.exists():
                    try:
                        with open(init_file, "r", encoding="utf-8") as f:
                            existing_content = f.read()
                    except UnicodeDecodeError:
                        pass

                if existing_content.strip() != content.strip():
                    results[rel_path] = f"Would update with {len(content)} characters"
                else:
                    results[rel_path] = "No change needed"
            else:
                init_file = package_dir / "__init__.py"
                was_created = not init_file.exists()

                if update_init_file(package_dir, content):
                    results[rel_path] = "Created" if was_created else "Updated"
                else:
                    results[rel_path] = "No change needed"
        else:
            results[rel_path] = "Excluded (manual maintenance)"

    return results


def main():
    """Main entry point."""
    import argparse  # pylint: disable=import-outside-toplevel

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,  # Will be set to src/ in auto_generate_init_files
        help="Root directory to scan (default: src/ directory)",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], help="Additional directories to exclude (can be used multiple times)"
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be changed without making changes")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show detailed output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Minimal output (only show if changes are needed)")

    args = parser.parse_args()

    # Determine the root directory
    root_dir = args.root_dir
    if root_dir is None:
        # Default to src/ directory relative to this script's location
        script_dir = Path(__file__).parent
        root_dir = script_dir.parent / "src"

    excluded_dirs = set(args.exclude)
    results = auto_generate_init_files(root_dir=root_dir, excluded_dirs=excluded_dirs, dry_run=args.dry_run)

    # Count changes first
    updated_count = sum(1 for action in results.values() if "Updated" in action or "Would update" in action)

    # Output logic based on flags
    if args.quiet:
        # In quiet mode, only show output if there are changes
        if updated_count > 0:
            for path, action in sorted(results.items()):
                if "Updated" in action or "Would update" in action:
                    print(f"{path}: {action}")
    elif args.verbose or args.dry_run:
        print(f"Scanning {root_dir}")
        print("Results:")
        for path, action in sorted(results.items()):
            print(f"  {path}: {action}")

    if args.dry_run:
        if not args.quiet or updated_count > 0:
            print(f"\nDry run complete. Would update {updated_count} files.")
    else:
        if not args.quiet or updated_count > 0:
            print(f"Updated {updated_count} __init__.py files.")

    # Exit with non-zero code if changes were made (useful for pre-commit)
    if updated_count > 0 and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
