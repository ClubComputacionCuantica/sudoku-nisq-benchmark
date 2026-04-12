# Copilot Instructions

## Project Overview
Framework for benchmarking Sudoku puzzles and generic exact cover problems on NISQ quantum hardware using Grover-based algorithms. Supports multi-SDK circuit construction (PyTKET/Qiskit/Braket), provider-agnostic backend management, and modular metrics/error mitigation. Start small with 2×2 or 4×4 puzzles—larger sizes (9×9+) require prohibitive qubit counts for current hardware.

## Architecture & Entry Points

### Public API Surface
Core exports in [src/sudoku_nisq/__init__.py](src/sudoku_nisq/__init__.py):
- `QSudoku`: High-level Sudoku-solving interface with visualization and backend management
- `QExactCover`: Lightweight interface for generic exact cover problems
- `ExactCoverQuantumSolver`: Grover-based solver supporting Sudoku and generic modes
- `ExactCoverEncoding`: Transforms Sudoku into exact cover constraints
- `ExactCoverProblem`: Canonical representation with hashing and indexing
- `BackendManager`: Singleton for cross-provider backend orchestration (**always use `BackendManager.inst()`**, never call constructor directly)

### Dual Workflow: Sudoku vs Generic Exact Cover

**Sudoku workflow** ([QSudoku](src/sudoku_nisq/q_sudoku.py)):
1. Create/load puzzle: `QSudoku.generate(size=2|4, num_missing_cells=2)` or `QSudoku.from_board(...)`
   - For 2×2: Use `size=2, subgrid_size=1` (no real subgrids, skips subgrid constraints)
   - For 4×4: Use `size=4, subgrid_size=2` (proper 2×2 subgrids)
2. Attach solver: `puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple"|"pattern", decompose_cnz=True)`
3. Build circuit: `circuit = puzzle.build_circuit(sdk="pytket"|"qiskit"|"braket")` — caches to `.quantum_solver_cache/{puzzle_hash}/{solver}/{encoding}`
4. Run: `result = puzzle.run_aer(shots=1024)` or `puzzle.run(backend_alias, opt_level=1, shots=1000)`; `run_aer_with_noise(...)` auto-builds noise models from fake IBM backends when none provided
5. Decode & analyze: `puzzle.format_result(result)` for assignments/boards, `puzzle.counts_plot(result)`, `puzzle.report_resources()`

**Generic exact cover workflow** ([QExactCover](src/sudoku_nisq/q_exact_cover.py)):
1. Define problem: `problem = ExactCoverProblem(universe, subsets, num_solutions=1)`
2. Wrap with quantum interface: `qec = QExactCover(problem)`
3. Build & run: `circuit = qec.build_circuit(sdk=...)`, `result = qec.run_aer(shots=512)` (simple encoding only; default SDK is Qiskit)
4. Resources: `qec.report_resources()` — designed for problems smaller than minimal Sudoku (2×2)

### Encoding Layer
[ExactCoverEncoding](src/sudoku_nisq/encodings/exact_cover_encoding.py) generates constraint universes and subset representations:
- **Simple encoding**: Direct cell-digit mapping with row/column/subgrid constraints
- **Pattern encoding**: Row-pattern reduction (Weiß 2022 thesis) for smaller subset spaces
- **2×2 special case**: Uses `universe2x2` (omits subgrid constraints due to row/col overlap); generates `simple_subsets2x2()` and `pattern_subsets2x2()`

### Solver Architecture
[ExactCoverQuantumSolver](src/sudoku_nisq/solvers/exact_cover_solver.py) extends abstract [QuantumSolver](src/sudoku_nisq/quantum_solver.py):
- **Modes**: Sudoku (uses `ExactCoverEncoding`) vs generic (accepts `ExactCoverProblem`, `universe`, `subsets`)
- **Options**: `decompose_cnz=True` (default for consistent gate counts), `track_memory=False` (psutil snapshots, dev-only)
- **Circuit construction**: Implements abstract `_build_sdk_circuit(sdk_type)` for PyTKET/Qiskit/Braket; base class handles SDK detection and caching
- **Resources**: `resource_estimation()` returns `{n_qubits, n_gates, depth}` dict; gate counts stored on instance after build

### Circuit & Caching System
- **Auto-detect SDK**: `build_circuit()` with no params → PyTKET default (reads cache if available)
- **Explicit SDK**: `build_circuit(sdk="qiskit"|"braket"|"pytket")` → always rebuilds (ignores cache), returns native SDK format, and stores a PyTKET copy for caching. Pass `sdk` parameter each time you need native SDK format; omit to reuse cached PyTKET circuit.
- **Cache path**: `.quantum_solver_cache/{puzzle_hash}/{solver_name}/{encoding}/main_circuit.json`
- **Gate-building backends**: [src/sudoku_nisq/circuits/exact_cover/](src/sudoku_nisq/circuits/exact_cover/) contains SDK-specific builders (`qiskit_impl.py`, `pytket_impl.py`, `braket_impl.py`) returning `(circuit, gate_counts)`
- **Force overwrite**: Pass `force_overwrite=True` to `build_circuit()` only when you need to refresh the cached PyTKET copy
- **Cache base override**: `SUDOKU_NISQ_CACHE_DIR` environment variable overrides default `.quantum_solver_cache`

### Transpilation Paths
- **Qiskit**: `_transpile_qiskit` uses `generate_preset_pass_manager` if backend has `Target`, else falls back to `qiskit.compiler.transpile`
- **PyTKET**: `_transpile_pytket` calls backend's `get_compiled_circuit()`
- **Braket**: No client-side transpilation—AWS service handles optimization; don't expect transpiled cache entries

### Backend & Provider System
**Architecture**: [BackendManager](src/sudoku_nisq/backends.py) is a singleton registry; **always use `BackendManager.inst()`** (never call constructor directly). Delegates to provider implementations from [src/sudoku_nisq/providers/](src/sudoku_nisq/providers/).

**Provider interface** ([providers/base.py](src/sudoku_nisq/providers/base.py)):
- `QuantumProvider` abstract base with `authenticate()`, `add_backend()`, `get_backend()`, `submit_job()`
- Concrete providers: `AerProvider` (always available), `IBMProvider` (qiskit-ibm-runtime), `QuantinuumProvider` (PyTKET extensions), `AWSProvider` (Braket SDK)
- Graceful degradation: Optional providers may be `None` if packages missing; wrap imports in try/except for feature detection

**Aer initialization**: `puzzle.init_aer(method="statevector"|"density_matrix", noise_model=..., coupling_map=..., basis_gates=..., device_type=..., precision=...)` registers backend with alias

**IBM/Quantinuum shortcuts**: `puzzle.init_ibm(api_token, instance, device)` → registers backend with alias; `puzzle.init_quantinuum(device, token_store)` → similar pattern

### Metadata & Persistence
Stage-aware metadata architecture in [src/sudoku_nisq/metadata/](src/sudoku_nisq/metadata/):
- **7-stage pipeline**: Stage 1 (Instance), 2a (Logical IR), 2b (IR Policy), 3 (Compilation), 4 (Executable), 5 (Execution), 6-7 (Metrics)
- **BenchmarkSession**: Orchestration layer for multi-run experiments with automatic stage recording
- **Cache override**: `SUDOKU_NISQ_CACHE_DIR` environment variable overrides default `.quantum_solver_cache`

### Puzzle Representation
[SudokuPuzzle](src/sudoku_nisq/sudoku_puzzle.py):
- Accepts `size` (2, 4, 9, 16...) or `subgrid_size` (1, 2, 3...)
- `canonicalize=True` → relabels digits to shortlex form (affects puzzle hash and reproducibility)
- `open_tuples`: List of `(row, col, digit)` for empty cells
- `pre_tuples`: Givens/clues
- `get_hash()`: Deterministic hash drives cache paths; includes board state and constraints

## Metrics & Benchmarking
Modular system in [src/sudoku_nisq/metrics/](src/sudoku_nisq/metrics/):
- **Data models** ([data_models.py](src/sudoku_nisq/metrics/data_models.py)): `ExecutionResult`, `HardwareMetadata`, `BenchmarkResult`
- **Calculators**: Compute success probability, fidelity, gate efficiency, circuit volume
- **Collectors**: Provider-specific metadata extraction (IBM calibration, Quantinuum specs)
- **Aggregators**: Multi-run statistics, comparative analysis
- **Reporters**: Export to CSV/JSON, visualization generation
- **Design docs**: See [docs/internal/architecture/](docs/internal/architecture/) for technical specs (not published)

## Error Mitigation
[src/sudoku_nisq/mitigation/](src/sudoku_nisq/mitigation/) wraps Mitiq library:
- **Executors** ([executors.py](src/sudoku_nisq/mitigation/executors.py)): Zero-noise extrapolation, PEC, CDR
- **Expectation wrapper** ([expectation_wrapper.py](src/sudoku_nisq/mitigation/expectation_wrapper.py)): Integrates with `QSudoku.run()` workflows
- Usage: Pass `mitigation_strategy="zne"|"pec"` to run methods (experimental)

## Testing & Development

### Commands
- Install: `poetry install` (Python 3.11–3.12 required)
- Tests: `poetry run pytest` (excludes heavy tests by default)
- Heavy tests: `poetry run pytest --run-heavy` or set `PYTEST_RUN_HEAVY=1`
- Docs: `poetry run sphinx-build docs docs/_build` (public HTML to `docs/_build/html`)
- Linting: `poetry run ruff check src tests`
- Type checking: `poetry run mypy src`
- Pre-commit: Hooks enforce no stray markdown in root (move to `docs/internal/` or `docs/guide/`)

### Test Markers
- `slow`: Heavy simulations (deselect with `-m "not slow"`)
- `resource_only`: Stage 0/1 analysis without execution
- `heavy`: RAM-dependent tests (opt-in with `--run-heavy` or env var `PYTEST_RUN_HEAVY=1`)
- `integration`: Exercises SDK/backends or external services
- `unit`: Fast logic-only coverage
- `out_of_scope`: Explicitly skipped per project scope
- **Note**: [tests/test_backends.py](tests/test_backends.py) currently skipped during BackendManager refactor; IBM/Quantinuum paths heavily mocked (no live credentials needed)

### Dependency Notes
- **Core deps**: pytket, pytket-qiskit, qiskit-aer (for simulation), matplotlib, py-sudoku, sudoku-py, pandas, tqdm, seaborn, boto3, mitiq, linkify-it-py, qnexus, scipy, ply
- **Optional extras**: qiskit-ibm-runtime (IBM), pytket-quantinuum (Quantinuum), amazon-braket-sdk (AWS)
- **Dev deps**: pytest, pytest-cov, ruff, mypy, psutil, sphinx, myst-parser, furo, sphinx-autodoc-typehints, sphinx-autobuild, sphinxcontrib-katex, insegel
- **Key change**: qiskit-aer is now a core dependency (not optional) for easy local simulation

## Examples & Guides
- **Examples**: [examples/](examples/) contains end-to-end scripts (`exact_cover_benchmark.py`, `canonical_encoding_demo.py`, `error_mitigation_comparison.py`, `example_gate_counting.py`)
- **Notebooks**: [notebooks/](notebooks/) for interactive demos (`2x2_aer.ipynb`, `metrics_walkthrough.ipynb`)
- **User guides**: [docs/guide/](docs/guide/) — quickstart, installation, providers, metrics reference
- **Internal docs**: [docs/internal/](docs/internal/) — architecture specs, implementation roadmaps, TODO trackers (excluded from published site via `conf.py`)

## Recent Architectural Changes (December 2025)
See [RECENT_CHANGES.md](RECENT_CHANGES.md) for full changelog:
- **Code reorganization**: Examples → `examples/`, scripts → `scripts/`, cleaned root directory
- **Module refactoring**: Solvers/providers consolidated; `aws_pending.py` → `aws.py`, `quantinuum_pending.py` → `quantinuum.py`
- **Metadata architecture**: Implemented stage-aware metadata system with `BenchmarkSession` for multi-run orchestration
- **CI/tooling**: Added `.pre-commit-config.yaml`, updated `pyproject.toml` dependencies (qiskit-aer now core dependency)
- **Docs overhaul**: New guides (canonical_encoding.md, features.md, providers.md); internal architecture docs moved to `docs/internal/`
- **Test infrastructure**: Enhanced markers (`heavy`, `integration`, `unit`); test_backends.py temporarily skipped during BackendManager refactor

## Common Pitfalls & Conventions

1. **Cache invalidation**: Cache is read only when `sdk` is omitted; passing `sdk` always rebuilds and writes a PyTKET copy. Pass `sdk` whenever you need a native circuit format; use `force_overwrite=True` only to refresh the cached PyTKET copy.
2. **Provider imports**: Optional providers may be `None` if packages missing; guard with try/except or feature flags before calling
3. **2×2 puzzles**: Skip subgrid constraints (use `universe2x2` paths); pattern encoding still applies
4. **Braket transpilation**: AWS service-side only; no client transpilation, so transpiled cache entries won't exist
5. **BackendManager singleton**: Always use `BackendManager.inst()` — constructor initializes new instance without provider registrations
6. **BenchmarkSession**: Use for multi-run experiments with automatic stage recording and metrics aggregation
7. **Canonicalization**: `canonicalize=True` changes puzzle hash; reuse same flag for consistent caching across experiments
8. **Environment variables**:
   - `SUDOKU_NISQ_CACHE_DIR`: Override default cache directory (`.quantum_solver_cache`)
   - `PYTEST_RUN_HEAVY=1`: Enable heavy/RAM-intensive tests

## Code Style & Patterns

- **Docstrings**: Use reStructuredText for Sphinx (`:param:`, `:returns:`, `:raises:`)
- **Type hints**: Prefer explicit types; use `Any` for SDK-agnostic circuit objects
- **Error handling**: `ValueError` for invalid inputs, `RuntimeError` for unexpected failures, `NotImplementedError` for unsupported features
- **Naming**: Follow PyTKET/Qiskit conventions; use `_` prefix for internal helpers
- **Public API stability**: Avoid reaching into solver internals; prefer public methods and properties
