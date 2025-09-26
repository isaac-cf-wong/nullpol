#!/usr/bin/env python3
"""
Script to validate that all tests are covered by the CI test matrix.

This script ensures that:
1. All test files in the repository are included in at least one CI test batch
2. All test classes/methods are covered by the CI configuration
3. No tests are accidentally missed when new ones are added

Usage:
    python scripts/validate_ci_test_coverage.py
"""

import os
import sys
import ast
import glob
import yaml
from typing import Set, Dict, List, Tuple


def find_all_test_files() -> Set[str]:
    """Find all Python test files in the repository."""
    test_files = set()

    # Find all test_*.py files
    for pattern in ["tests/**/test_*.py", "tests/test_*.py"]:
        for file_path in glob.glob(pattern, recursive=True):
            # Convert to relative path for consistency
            rel_path = os.path.relpath(file_path)
            test_files.add(rel_path)

    return test_files


def extract_test_classes_and_methods(file_path: str) -> Dict[str, List[str]]:
    """Extract test classes and their methods from a Python file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except (SyntaxError, UnicodeDecodeError) as e:
        print(f"Warning: Could not parse {file_path}: {e}")
        return {}

    classes_and_methods = {}

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name.startswith("Test"):
            class_name = node.name
            methods = []

            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                    methods.append(item.name)

            if methods:  # Only include classes with test methods
                classes_and_methods[class_name] = methods

    return classes_and_methods


def parse_ci_config() -> List[Dict]:
    """Parse the CI configuration to extract test batches."""
    ci_file = ".github/workflows/CI.yml"

    try:
        with open(ci_file, "r", encoding="utf-8") as f:
            ci_config = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Error: {ci_file} not found")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing {ci_file}: {e}")
        sys.exit(1)

    # Extract test batches from the matrix
    test_batches = []
    try:
        matrix = ci_config["jobs"]["test"]["strategy"]["matrix"]
        for batch in matrix["test-batch"]:
            test_batches.append({"name": batch["name"], "paths": batch["paths"]})
    except KeyError as e:
        print(f"Error: Could not find test matrix in CI config: {e}")
        sys.exit(1)

    return test_batches


def normalize_path(path: str) -> str:
    """Normalize a path for comparison."""
    return os.path.normpath(path.replace("\\", "/"))


def parse_test_paths(paths_string: str) -> Set[Tuple[str, str, str]]:
    """
    Parse test paths from CI configuration.

    Returns set of tuples: (file_path, class_name, method_name)
    where class_name and method_name can be None for file-level or directory-level inclusion
    """
    parsed_paths = set()

    # Split by spaces to get individual path specifications
    path_specs = paths_string.split()

    for spec in path_specs:
        spec = spec.strip()
        if not spec:
            continue

        if "::" in spec:
            # Specific class or method specification
            parts = spec.split("::")
            file_path = normalize_path(parts[0])

            if len(parts) == 2:
                # Class specification: file.py::ClassName
                class_name = parts[1]
                parsed_paths.add((file_path, class_name, None))
            elif len(parts) == 3:
                # Method specification: file.py::ClassName::method_name
                class_name = parts[1]
                method_name = parts[2]
                parsed_paths.add((file_path, class_name, method_name))
        else:
            # File or directory specification
            norm_spec = normalize_path(spec)

            if norm_spec.endswith("/"):
                # Directory specification - find all test files in directory
                if os.path.isdir(norm_spec.rstrip("/")):
                    for file_path in glob.glob(f"{norm_spec.rstrip('/')}/**/test_*.py", recursive=True):
                        parsed_paths.add((normalize_path(file_path), None, None))
            elif norm_spec.endswith(".py"):
                # Single file specification
                parsed_paths.add((norm_spec, None, None))
            else:
                # Treat as directory without trailing slash
                if os.path.isdir(norm_spec):
                    for file_path in glob.glob(f"{norm_spec}/**/test_*.py", recursive=True):
                        parsed_paths.add((normalize_path(file_path), None, None))

    return parsed_paths


def validate_coverage() -> bool:
    """Validate that all tests are covered by CI configuration."""
    print("Validating CI test coverage...")

    # Find all test files
    all_test_files = find_all_test_files()
    print(f"Found {len(all_test_files)} test files")

    # Parse CI configuration
    test_batches = parse_ci_config()
    print(f"Found {len(test_batches)} test batches in CI")

    # Track coverage
    covered_files = set()
    covered_classes = set()  # (file_path, class_name)
    covered_methods = set()  # (file_path, class_name, method_name)

    # Analyze each test batch
    for batch in test_batches:
        print(f"\nAnalyzing batch: {batch['name']}")
        batch_paths = parse_test_paths(batch["paths"])

        for file_path, class_name, method_name in batch_paths:
            if class_name is None:
                # Full file coverage
                covered_files.add(file_path)
                print(f"  [OK] File: {file_path}")
            elif method_name is None:
                # Class coverage
                covered_classes.add((file_path, class_name))
                print(f"  [OK] Class: {file_path}::{class_name}")
            else:
                # Method coverage
                covered_methods.add((file_path, class_name, method_name))
                print(f"  [OK] Method: {file_path}::{class_name}::{method_name}")

    # Check for uncovered files
    uncovered_files = set()
    validation_errors = []

    for test_file in all_test_files:
        norm_file = normalize_path(test_file)

        # Check if file is covered at file level
        if norm_file in covered_files:
            continue

        # Check if all classes in file are covered
        file_classes = extract_test_classes_and_methods(test_file)

        if not file_classes:
            # File has no test classes, check if it's covered at file level
            if norm_file not in covered_files:
                uncovered_files.add(norm_file)
            continue

        file_fully_covered = True
        uncovered_classes_in_file = []

        for class_name, methods in file_classes.items():
            class_covered = False

            # Check if class is covered at class level
            if (norm_file, class_name) in covered_classes:
                class_covered = True
            else:
                # Check if all methods in class are covered
                all_methods_covered = True
                for method_name in methods:
                    if (norm_file, class_name, method_name) not in covered_methods:
                        all_methods_covered = False
                        break

                if all_methods_covered and methods:
                    class_covered = True

            if not class_covered:
                file_fully_covered = False
                uncovered_classes_in_file.append(class_name)

        if not file_fully_covered:
            uncovered_files.add(norm_file)
            validation_errors.append(f"[FAIL] {norm_file}: Uncovered classes: {uncovered_classes_in_file}")

    # Report results
    print("\nCoverage Summary:")
    print(f"Total test files: {len(all_test_files)}")
    print(f"Covered files: {len(all_test_files) - len(uncovered_files)}")
    print(f"Uncovered files: {len(uncovered_files)}")

    if uncovered_files:
        print("\n[FAIL] Validation FAILED - Uncovered test files:")
        for error in validation_errors:
            print(f"  {error}")
        return False
    else:
        print("\n[PASS] Validation PASSED - All test files are covered by CI!")
        return True


if __name__ == "__main__":
    success = validate_coverage()
    sys.exit(0 if success else 1)
