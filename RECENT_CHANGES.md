# Recent Changes

## December 2025 - Major Project Restructuring

### Code Organization & Cleanup
- **Moved examples**: Relocated all example files from root to `examples/` directory for better organization
- **Moved scripts**: Created `scripts/` directory for utility scripts
- **Cleaned root directory**: Removed numerous temporary and draft files:
  - Development drafts: `aws-draft.py`, `debug_canonical.py`, `debug_test.py`
  - Test files: `test_canonical_encoding.py`, `test_count_solutions.py`, `test_gate_counting.py`, `test_integration.py`
  - Implementation docs: `implementation_summary.py`, `verify_implementation.py`
  - Various markdown documentation files moved to appropriate locations

### Module Refactoring
- **Solvers reorganization**: 
  - Moved `backtracking_solver.py` to `solvers/backtracking_solver.py`
  - Moved `graph_coloring_solver.py` to `solvers/graph_coloring_solver.py`
  - Updated `solvers/__init__.py` for cleaner imports
- **Provider improvements**:
  - Renamed `aws_pending.py` → `aws.py` (promoted from pending)
  - Renamed `quantinuum_pending.py` → `quantinuum.py` (promoted from pending)
  - Enhanced Aer provider implementation
- **Benchmark system**: Consolidated benchmark modules, removed old comparison/results files

### Documentation Overhaul
- **New guide pages**:
  - `canonical_encoding.md` - Detailed encoding documentation
  - `features.md` - Feature overview
  - `providers.md` - Provider usage guide
- **Removed outdated guides**:
  - `architecture.md` and `current-state.md` (consolidated elsewhere)
- **Updated API documentation**: Regenerated all autosummary files to reflect new structure
- **Enhanced existing guides**: Updated quickstart, installation, examples, and error mitigation guides
- **Added internal documentation**: Created `docs/internal/` for development notes

### Configuration & Tooling
- Added `.pre-commit-config.yaml` for code quality automation
- Updated `pyproject.toml` with new module structure and dependencies
- Updated `pytest.ini` for test discovery in new locations
- Modified `.gitignore` to exclude cache directories

### Feature Additions
- Exact cover implementation enhancements
- Benchmark capability system
- Memory tracking improvements
- Gate counting examples and walkthrough notebook
- Decoding helpers in ExactCoverQuantumSolver and QSudoku
- Qiskit Aer as default dependency for easy local simulation

### Cache Management
- Cleaned up `.benchmark_cache/` removing outdated benchmark results
- Cache now properly excluded via `.gitignore`

### Technical Improvements
- Updated import structure in `__init__.py` files across modules
- Enhanced metadata management
- Improved executor implementations for error mitigation
- Better separation of concerns between circuit implementations (Qiskit, Braket, Pytket)

## December 2025 - CI, Docs, Providers, and Examples Refresh

### CI & Tooling
- Updated CI workflow at `.github/workflows/ci.yml` to run lint (ruff), type-check (mypy), and tests.
- `poetry.lock` corrected to remove duplicate dependency blocks; `poetry install` now succeeds.

### Documentation Updates
- Refreshed guides: getting-started, quickstart, providers, error mitigation, Aer features.
- Added internal architecture and metrics references under `docs/internal/`.
- New `docs/guide/metrics_reference.md` added.
- Updated Sphinx config `docs/conf.py`.

### Examples & Notebooks
- Updated examples for exact cover benchmarking and demos.
- Added error mitigation comparison example.
- Tweaked notebooks including `2x2_aer.ipynb`; added `error_mitigation_demo.md`.

### Providers & Backends
- Iterated on providers `aer.py`, `aws.py`, `quantinuum.py` and `providers/__init__.py`.
- Backend utilities refined in `src/sudoku_nisq/backends.py`.

### Circuits & Solvers
- Adjustments across circuit implementations for exact cover (Braket, Pytket, Qiskit).
- Solver and transpilation improvements in `solvers/exact_cover_solver.py`.

### Core Modules & Utilities
- Exact cover problem and QSolve/QSudoku refinements.
- Memory tracking and mitigation executors improved.

### Tests
- Updated and expanded tests: Aer features, IBM native, metadata manager, mitigation, SDK selection, transpilation, and QSudoku.

