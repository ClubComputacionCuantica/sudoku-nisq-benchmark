# Examples

This directory contains example scripts demonstrating various features of the Sudoku NISQ library.

## Quick Start Examples

These examples are fast-running and good for getting started:

- **[exact.py](exact.py)** - Minimal exact cover example (fastest to run)
- **[exact_cover_demo.py](exact_cover_demo.py)** - Basic quantum exact cover demo  
- **[example_gate_counting.py](example_gate_counting.py)** - Demonstrates automatic gate counting features

## Feature Demonstrations

Core feature examples that showcase specific capabilities:

- **[canonical_encoding_demo.py](canonical_encoding_demo.py)** - Canonical encoding framework, isomorphism detection, and Sudoku encodings
- **[exact_cover_benchmark.py](exact_cover_benchmark.py)** - Compares generic exact cover vs Sudoku quantum resource requirements
- **[example_gate_counting_options.py](example_gate_counting_options.py)** - CnZ decomposition options for consistent cross-SDK gate counting
- **[example_memory_tracking.py](example_memory_tracking.py)** - Advanced memory profiling during circuit construction (dev feature)
- **[example_usage.py](example_usage.py)** - BackendManager and provider pattern demonstration
- **[test_exact_cover.py](test_exact_cover.py)** - Test suite for exact cover functionality

## Advanced/Experimental Examples

These examples may require additional dependencies or are works in progress:

### Error Mitigation (Requires Mitiq)

- **[error_mitigation_comparison.py](error_mitigation_comparison.py)**
  - Demonstrates Zero Noise Extrapolation (ZNE) with Aer simulator
  - **Dependencies:** `pip install mitiq qiskit-aer`
  - **Note:** May take several minutes due to ZNE overhead
  - Shows comparison between mitigated and unmitigated execution

### Phase 4 Metrics (Work in Progress)

- **[phase4_metrics_integration.py](phase4_metrics_integration.py)**
  - Demonstrates automatic Stage 6-7 metrics recording
  - **Status:** Experimental - some features still in development
  - Shows validation context setup and metrics collection workflow

## Notebook Examples

Interactive Jupyter notebooks for exploration:

- **[ibm_backend_benchmark.ipynb](ibm_backend_benchmark.ipynb)** - IBM Quantum backend benchmarking
- **[metrics_walkthrough.ipynb](metrics_walkthrough.ipynb)** - Interactive metrics system walkthrough

## Running Examples

All examples can be run directly with Python:

```bash
# Quick start
python examples/exact.py

# Feature demonstrations
python examples/canonical_encoding_demo.py
python examples/exact_cover_benchmark.py

# Advanced examples (may need extra dependencies)
pip install mitiq qiskit-aer
python examples/error_mitigation_comparison.py
```

## Dependencies

- **Core examples:** Only require base library (`poetry install`)
- **Error mitigation:** Requires `mitiq` package
- **Noisy simulation:** Requires `qiskit-aer` package
- **IBM/Quantinuum backends:** Require provider-specific authentication

## Notes

- Examples marked "Advanced/Experimental" may require additional setup
- Some examples demonstrate features still under active development
- See main [documentation](../docs/) for full API reference
- Check individual file docstrings for detailed usage information

## Getting Help

If you encounter issues:

1. Check that all dependencies are installed: `poetry install`
2. Install optional dependencies as needed: `pip install mitiq qiskit-aer`
3. See the main [README](../README.md) for troubleshooting
4. Review the [API documentation](../docs/api/) for detailed usage
