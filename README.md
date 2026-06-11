# nullpol

[![CI](https://github.com/isaac-cf-wong/nullpol/actions/workflows/ci.yml/badge.svg)](https://github.com/isaac-cf-wong/nullpol/actions/workflows/ci.yml)
[![Documentation Status](https://github.com/isaac-cf-wong/nullpol/actions/workflows/documentation.yml/badge.svg)](https://isaac-cf-wong.github.io/nullpol/)
[![codecov](https://codecov.io/gh/isaac-cf-wong/nullpol/graph/badge.svg?token=8QEBHUQXH8)](https://codecov.io/gh/isaac-cf-wong/nullpol)
[![PyPI Version](https://img.shields.io/pypi/v/nullpol)](https://pypi.org/project/nullpol/)
[![Python Versions](https://img.shields.io/pypi/pyversions/nullpol)](https://pypi.org/project/nullpol/)
[![License: GPL-3.0-or-later](https://img.shields.io/badge/License-GPL--3.0--or--later-blue.svg)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)

A Python package for model-independent polarization tests of gravitational-wave
(GW) signals. Built on [bilby](https://lscsoft.docs.ligo.org/bilby/) and
[bilby_pipe](https://lscsoft.docs.ligo.org/bilby_pipe/master/index.html),
nullpol automates reproducible parameter-estimation workflows for testing
alternative GW polarizations.

## Features

- **Model-agnostic polarization tests**: Framework for scalar-tensor and related
  polarization hypotheses
- **bilby integration**: Extends bilby and bilby_pipe for likelihood-based
  inference
- **Injection workflows**: Tools for generating and studying simulated signals
- **Time-frequency filtering**: Sample-based filter construction for analysis
- **HTCondor support**: DAG generation for batch submission on compute clusters
- **Asimov integration**: Optional pipeline hooks for LIGO workflow automation
- **CLI**: Command-line tools for injections, filtering, and end-to-end
  pipelines

## Installation

We recommend using `uv` to manage virtual environments for installing nullpol.

If you don't have `uv` installed, you can install it with pip. See the project
pages for more details:

- Install via pip: `pip install --upgrade pip && pip install uv`
- Project pages: [uv on PyPI](https://pypi.org/project/uv/) |
  [uv on GitHub](https://github.com/astral-sh/uv)
- Full documentation and usage guide: [uv docs](https://docs.astral.sh/uv/)

**Note:** The package is built and tested against Python 3.12–3.14. When
creating a virtual environment with `uv`, specify the Python version to ensure
compatibility: `uv venv --python 3.12`.

### From PyPI

```bash
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install nullpol
```

Optional [Asimov](https://asimov.docs.ligo.org/) integration:

```bash
uv pip install nullpol[asimov]
```

### From Source

```bash
git clone https://github.com/isaac-cf-wong/nullpol.git
cd nullpol
# Create a virtual environment (recommended with uv)
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync
```

For development, install all dependency groups:

```bash
uv sync --all-groups
```

## Quick Start

### Command Line

Write an ini configuration file and run the main pipeline:

```bash
# Generate a default configuration template
nullpol_pipe_write_default_ini --outdir ./config

# Build (and optionally submit) the analysis DAG
nullpol_pipe config.ini
nullpol_pipe config.ini --submit
```

Other command-line tools:

```bash
nullpol_create_injection --help
nullpol_create_time_frequency_filter_from_sample --help
nullpol_pipe_analysis --help
nullpol_pipe_generation --help
nullpol_get_asimov_yaml --help
```

See the `examples/` directory for scalar-tensor injection studies and template
configurations.

## Configuration

nullpol uses INI configuration files in the bilby_pipe style. A typical workflow
starts with `nullpol_pipe_write_default_ini`, then edits the generated template
before passing it to `nullpol_pipe`.

Key workflow stages:

| Stage         | Tool                       | Purpose                                           |
| ------------- | -------------------------- | ------------------------------------------------- |
| Injection     | `nullpol_create_injection` | Generate simulated signals for polarization tests |
| Generation    | `nullpol_pipe_generation`  | Produce strain data for analysis                  |
| Analysis      | `nullpol_pipe_analysis`    | Run parameter estimation on generated data        |
| Orchestration | `nullpol_pipe`             | Build and submit the full HTCondor DAG            |

See the [documentation](https://isaac-cf-wong.github.io/nullpol/) for injection
setups, priors, and polarization-specific options.

## Documentation

Full documentation is available at
[https://isaac-cf-wong.github.io/nullpol/](https://isaac-cf-wong.github.io/nullpol/).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

1. Fork the repository
2. Create a feature branch
3. Make your changes and add tests
4. Run `uv run pytest`
5. Submit a pull request

## Testing

Run the test suite:

```bash
uv run pytest
```

## License

This project is licensed under **GPL-3.0-or-later**. See the [LICENSE](LICENSE)
file for the full license text.

## Citation

If you use nullpol in your research, please cite:

```bibtex
@software{nullpol,
  title={nullpol: Model-independent polarization test of gravitational-wave signals},
  author={Wong, Isaac C.F. and Ng, Thomas and Cirok, Balázs},
  url={https://github.com/isaac-cf-wong/nullpol},
  year={2025}
}
```

## Support

For questions or issues, please open an issue on
[GitHub](https://github.com/isaac-cf-wong/nullpol/issues) or contact the
maintainers.
