# Installation

## From PyPI

```bash
pip install nullpol
```

Optional [Asimov](https://asimov.docs.ligo.org/) integration:

```bash
pip install nullpol[asimov]
```

## From source

Clone the repository and install with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/isaac-cf-wong/nullpol.git
cd nullpol
uv sync
uv pip install -e .
```

Or with pip:

```bash
git clone https://github.com/isaac-cf-wong/nullpol.git
cd nullpol
pip install .
```

## Development installation

```bash
git clone https://github.com/isaac-cf-wong/nullpol.git
cd nullpol
uv sync --all-groups
```

This installs the package in editable mode together with development, testing,
and documentation dependencies.

## Requirements

- Python 3.12 or higher
- Key dependencies: bilby, bilby_pipe, gwpy, numpy, scipy, matplotlib
- Full list available in `pyproject.toml`
