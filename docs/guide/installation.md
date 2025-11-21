# Installation

This project is distributed as a Python package and built with Poetry. You can install it in two common ways.

## Prerequisites
- Python 3.11–3.12 (see `pyproject.toml` for the exact supported range)
- Windows, macOS, or Linux
- Optional: Poetry (recommended for dev workflow)

## Users (pip)
```bash
# Clone the repository (if using the source)
# git clone https://github.com/ClubComputacionCuantica/sudoku-nisq-benchmark.git
# cd sudoku-nisq-benchmark

# Install in editable mode
pip install -e .
```

## Developers (Poetry)
```bash
# Install Poetry if not installed
# https://python-poetry.org/docs/#installation

# Install dev environment
poetry install

# Run tests and docs locally
poetry run pytest
poetry run sphinx-build -b html docs docs/_build/html
```

## Optional provider SDKs
Some features require vendor SDKs which are mocked during docs build:
- IBM: `qiskit`, `qiskit-ibm-runtime`
- Quantinuum (TKET): `pytket`, `pytket-quantinuum`
- AWS Braket: `amazon-braket-sdk`

Install only what you need for your workflow.

```{note}
Docs builds mock these packages (`autodoc_mock_imports` in `conf.py`) so you don’t need them to build documentation.
```
