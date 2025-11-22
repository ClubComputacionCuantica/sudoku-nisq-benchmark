# Sudoku NISQ Benchmark

> Quantum Sudoku as an approachable benchmarking problem for near-term quantum devices.

## Overview
This project provides:
- Encoding of Sudoku as an exact cover problem.
- Multiple circuit backends (PyTKET primary, Qiskit native, Braket native).
- Provider abstraction with IBM tested; Quantinuum in progress.
- Solver orchestration with resource estimation utilities.

## Quickstart
See the dedicated page: {doc}`guide/quickstart`.

## Installation
See {doc}`guide/installation`.

## Documentation Structure
- `guide/` high-level usage guides.
- `api/` auto-generated API reference (Sphinx autodoc + autosummary).
 - Examples are embedded in guides to keep things simple.

```{toctree}
:maxdepth: 2
:caption: Contents

guide/installation
guide/quickstart
guide/getting-started
guide/examples
guide/contributing
guide/current-state
releases/index
api/index
```

## Roadmap (TODO)
- [ ] Add Braket circuit implementation
- [ ] Add classical baseline performance comparisons
- [ ] Add benchmarking harness for multiple puzzle sizes
- [ ] Stabilize public API and remove deprecated shims

## Contributing
Please open issues for enhancements or inaccuracies. Ensure tests + type checks pass before PR.

## License
Apache 2.0
