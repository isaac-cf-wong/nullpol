# nullpol

[![PyPI version](https://badge.fury.io/py/nullpol.svg)](https://pypi.org/project/nullpol/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Build](https://git.ligo.org/bayesian-null-stream/nullpol/badges/main/pipeline.svg)](https://git.ligo.org/bayesian-null-stream/nullpol/-/pipelines)
[![codecov](https://codecov.io/gh/username/package_name/branch/main/graph/badge.svg)](https://codecov.io/gh/username/package_name)
[![Python Version](https://img.shields.io/pypi/pyversions/nullpol)](https://pypi.org/project/nullpol/)
[![Security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![Documentation Status](https://img.shields.io/badge/documentation-online-brightgreen)](https://git.ligo.org/bayesian-null-stream/nullpol/docs)
[![DOI](https://zenodo.org/badge/ID.svg)](https://doi.org/DOI)

**A package to perform model-independent polarization test of gravitational-wave signals.**

`nullpol` is a Python package designed to facilitate model-agnostic tests of gravitational-wave polarizations.
Built upon the robust parameter estimation capabilities of the [bilby](https://git.ligo.org/lscsoft/bilby) framework, nullpol seamlessly integrates with [bilby_pipe](https://git.ligo.org/lscsoft/bilby_pipe) to automate the workflow, enabling efficient and reproducible analyses.

## Installation

### From PyPI (recommended)

```bash
pip install nullpol
```

### From source

```bash
git clone https://git.ligo.org/bayesian-null-stream/nullpol.git
cd nullpol
pip install .
```

### Development installation

```bash
git clone https://git.ligo.org/bayesian-null-stream/nullpol.git
cd nullpol
pip install -e .[test]
```

## Requirements

- Python 3.10 or higher
- Key dependencies: bilby, bilby_pipe, gwpy, numpy, scipy, matplotlib
- Full list available in `pyproject.toml`

## Quick Start

### Command Line Interface

The package provides several command-line tools:

```bash
# Create injection files
nullpol_create_injection --help

# Create time-frequency filters
nullpol_create_time_frequency_filter_from_sample --help

# Main pipeline interface
nullpol_pipe --help

# Data analysis pipeline
nullpol_pipe_analysis --help

# Data generation pipeline
nullpol_pipe_generation --help

# Generate default configuration
nullpol_pipe_write_default_ini --help

# Asimov integration
nullpol_get_asimov_yaml --help
```

### Python API

```python
import nullpol

# Example usage will be added based on the API
```

## Documentation

- **Documentation**: [https://git.ligo.org/bayesian-null-stream/nullpol/-/tree/main/docs](https://git.ligo.org/bayesian-null-stream/nullpol/-/tree/main/docs)
- **Source Code**: [https://git.ligo.org/bayesian-null-stream/nullpol](https://git.ligo.org/bayesian-null-stream/nullpol)
- **Issue Tracker**: [https://git.ligo.org/bayesian-null-stream/nullpol/-/issues](https://git.ligo.org/bayesian-null-stream/nullpol/-/issues)

## Examples

The `examples/` directory contains various usage examples:

- **Injection studies**: Scalar-tensor polarization tests
- **Time-frequency filtering**: Sample analysis workflows
- **Configuration files**: Template setups for different scenarios

## Optional Dependencies

Install additional features:

```bash
# For Asimov integration
pip install nullpol[asimov]

# For Spark support
pip install nullpol[spark]

# For development and testing
pip install nullpol[test]
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) and [Code of Conduct](CODE_OF_CONDUCT.md).

### Development Setup

1. Fork the repository
2. Clone your fork
3. Install development dependencies: `pip install -e .[test]`
4. Install pre-commit hooks: `pre-commit install`
5. Make your changes
6. Run tests: `pytest`
7. Submit a pull request

### Code Quality

We use:

- **Black** for code formatting (line length: 120)
- **Ruff** for linting
- **pytest** for testing
- **pre-commit** for automated checks

## Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest -m "not integration"  # Skip integration tests
pytest -m "unit"             # Run only unit tests
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Authors

- **Isaac C.F. Wong** - chunfung.wong@kuleuven.be
- **Thomas Ng**
- **Balázs Cirok**

## Citation

If you use nullpol in your research, please cite:

```bibtex
@software{nullpol,
  title={nullpol: Model-independent polarization test of gravitational-wave signals},
  author={Wong, Isaac C.F. and Ng, Thomas and Cirok, Balázs},
  url={https://git.ligo.org/bayesian-null-stream/nullpol},
  year={2025}
}
```

## Acknowledgments

- Built on the [bilby](https://git.ligo.org/lscsoft/bilby) framework
- Integrates with [bilby_pipe](https://git.ligo.org/lscsoft/bilby_pipe) for workflow automation
- Part of the LIGO Scientific Collaboration software ecosystem

---

For more information, visit our [documentation](https://git.ligo.org/bayesian-null-stream/nullpol/-/tree/main/docs) or open an [issue](https://git.ligo.org/bayesian-null-stream/nullpol/-/issues).
