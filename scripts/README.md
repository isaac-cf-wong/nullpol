# Auto-generated `__init__.py` System

This directory contains tooling to automatically generate `__init__.py` files for leaf packages in the nullpol codebase.

## Overview

The nullpol package uses a hybrid approach for `__init__.py` management:

- **High-level packages**: Manually maintained (e.g., `src/nullpol/`, `src/nullpol/analysis/`, `src/nullpol/integrations/`)
- **Leaf packages**: Auto-generated (e.g., `src/nullpol/simulation/`, `src/nullpol/utils/`, `src/nullpol/analysis/likelihood/`)

This eliminates star imports (`from .module import *`) in favor of explicit imports while reducing developer maintenance burden.

## How it Works

The `auto_generate_init.py` script:

1. Scans the codebase for Python packages (directories with `__init__.py`)
2. Identifies "leaf packages" (directories containing only `.py` files, no subdirectories)
3. Extracts public API from each module using AST parsing:
   - Functions and classes not starting with `_`
   - Module-level constants (ALL_CAPS variables)
   - Respects existing `__all__` definitions when present
4. Generates clean `__init__.py` files with explicit imports and `__all__` lists

## Usage

### Manual Generation

```bash
# Generate all auto-managed __init__.py files
python scripts/auto_generate_init.py

# Dry run to see what would change
python scripts/auto_generate_init.py --dry-run --verbose

# Quiet mode for CI (only show files that need updates)
python scripts/auto_generate_init.py --dry-run --quiet

# Show help
python scripts/auto_generate_init.py --help
```

### Automatic Generation (Pre-commit)

The script runs automatically as a pre-commit hook when Python files in `src/` are modified:

```bash
# Install pre-commit hooks (one-time setup)
pre-commit install

# The hook runs automatically on commit, or manually:
pre-commit run auto-generate-init --all-files
```

### GitLab CI Integration

The script also runs in GitLab CI to ensure auto-generated files stay up to date:

- **`init-files-check` job**: Validates that all auto-generated `__init__.py` files are current
- Runs before code style checks (`black`, `ruff`) in the `code-quality` stage
- Uses `--quiet` mode for clean CI output
- Fails the pipeline if files are out of sync, with clear instructions to run the script locally

## Configuration

### Excluded Directories

High-level packages are excluded from auto-generation and should be manually maintained:

- `src/nullpol/` - Main package entry point
- `src/nullpol/cli/` - Command-line interface (complex conditional imports)
- `src/nullpol/analysis/` - Analysis module organization
- `src/nullpol/integrations/` - External integrations organization

**Auto-generated leaf packages** (eliminated star imports):
- `src/nullpol/detector/` (only .py files)
- `src/nullpol/simulation/` (only .py files)
- `src/nullpol/utils/` (only .py files)
- `src/nullpol/analysis/antenna_patterns/` (only .py files)
- `src/nullpol/analysis/clustering/` (only .py files)
- `src/nullpol/analysis/likelihood/` (only .py files)
- `src/nullpol/analysis/prior/` (has `prior_files/` non-code subdir)
- `src/nullpol/analysis/tf_transforms/` (only .py files)
- `src/nullpol/integrations/asimov/` (has `templates/` non-code subdir)
- `src/nullpol/integrations/htcondor/` (only .py files)

**Template directories** are automatically excluded since they contain no `.py` files.

These maintain curated imports and documentation for their public API.

### Leaf Package Detection

A directory is considered a leaf package if:
- It contains `__init__.py`
- It contains one or more `.py` files (excluding `__init__.py`)
- It contains NO code subdirectories (Python modules)

**Non-code subdirectories are ignored** when determining leaf status:
- `templates/` - Configuration templates
- `configs/`, `config/` - Configuration files
- `assets/`, `static/` - Static assets
- `fixtures/` - Test fixtures and sample data
- `data/`, `resources/` - Data/resource files (when not containing Python modules)
- `prior_files/` - Data storage directories (nullpol-specific)
- `__pycache__/` - Python cache
- `.*` - Hidden directories

**Examples:**
- ✅ `src/nullpol/simulation/` → Leaf (only `.py` files: injection.py, source.py)
- ✅ `src/nullpol/utils/` → Leaf (only `.py` files: config.py, error.py, etc.)
- ✅ `src/nullpol/analysis/prior/` → Leaf (has `.py` files + `prior_files/` non-code subdir)
- ❌ `src/nullpol/analysis/` → Not leaf (has code subdirs like `likelihood/`, `clustering/`)

## Command-Line Options

```bash
python scripts/auto_generate_init.py [options]

Options:
  --root-dir PATH     Root directory to scan (default: src/ directory)
  --exclude DIR       Additional directories to exclude (can be used multiple times)
  --dry-run          Show what would be changed without making changes
  --verbose, -v      Show detailed output including all scanned packages
  --quiet, -q        Minimal output (only show files that need updates)
  --help, -h         Show help message
```

**Flag combinations:**
- Default: Shows summary of updated files
- `--verbose`: Shows all packages scanned and their status
- `--quiet`: Only shows output if changes are needed (ideal for CI)
- `--dry-run`: Always shows what would change (combines well with `--verbose` or `--quiet`)

## Example Transformation

**Before** (star import):
```python
from __future__ import annotations

from .default import *
```

**After** (explicit import):
```python
from __future__ import annotations

from .default import DEFAULT_PRIOR_DIR, PolarizationPriorDict

__all__ = [
    "DEFAULT_PRIOR_DIR",
    "PolarizationPriorDict",
]
```

## Benefits

1. **No star imports**: Eliminates `from .module import *` patterns
2. **Explicit API**: Clear visibility of what each package exports
3. **IDE support**: Better autocomplete and static analysis
4. **Reduced maintenance**: Developers don't need to manually update leaf `__init__.py` files
5. **Consistency**: Uniform format across all auto-generated files

## Adding New Modules

When you add a new `.py` file to a leaf package:

1. The pre-commit hook automatically updates the `__init__.py`
2. All public functions/classes are automatically exported
3. No manual `__init__.py` editing required

For high-level packages, manually edit the `__init__.py` to control the public API.

## Troubleshooting

### GitLab CI Failures

If the `init-files-check` job fails in GitLab CI:

1. **Run the script locally**:
   ```bash
   python scripts/auto_generate_init.py
   ```

2. **Check what changed**:
   ```bash
   git diff src/
   ```

3. **Commit the changes**:
   ```bash
   git add src/
   git commit -m "Auto-update __init__.py files"
   ```

### Pre-commit Hook Issues

If the pre-commit hook is too slow or causing issues:

```bash
# Skip the hook for a specific commit
git commit --no-verify -m "Your commit message"

# Disable the hook temporarily
pre-commit uninstall

# Re-enable when ready
pre-commit install
```

### Manual Package Control

To prevent a leaf package from being auto-generated, add it to exclusions:

```bash
python scripts/auto_generate_init.py --exclude src/your/package/path
```

Or edit the script's `default_exclusions` set for permanent exclusion.
