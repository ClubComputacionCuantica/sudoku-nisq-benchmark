# Metrics, Metadata, and Benchmarking Workflow

This doc ties together circuit builds, metadata stages, and metrics so contributors can follow the full pipeline end to end.

## Big Picture
- Problem space: Sudoku (via `QSudoku`) and generic exact cover (via `ExactCoverProblem` + `QExactCover`), solved with Grover-style `ExactCoverQuantumSolver`.
- Caching root: `.quantum_solver_cache/{puzzle_hash}/{solver}/{encoding}`. Circuit caches default to PyTKET format; passing `sdk=` forces rebuild and writes a PyTKET copy for cache coherence.
- Metadata modes:
  - **Legacy**: `MetadataManager` writes `metadata.json` (still used by `QuantumSolver` helpers and resource reports).
  - **Stage-aware (preferred)**: `BenchmarkSession` + stage managers under `src/sudoku_nisq/metadata/` write Stage 1–7 artifacts when `SUDOKU_NISQ_NEW_METADATA=1`.
- Metrics engine: Data contracts in `src/sudoku_nisq/metrics/data_models.py`; calculator implementations in `src/sudoku_nisq/metrics/calculators/` (many are TODO beyond success/odds/retention scaffolds). Stage 6–7 persistence handled by `MetricsMetadataManager`.

Notes on current behavior (Dec 2025 codebase):
- Stage file *writing* (Stages 2a/2b/3/5/6–7) is primarily driven by `QuantumSolver` when `SUDOKU_NISQ_NEW_METADATA=1`.
- `SUDOKU_NISQ_CACHE_DIR` is supported by `MetadataConfig.get_cache_base()`, but `QSudoku`/`QExactCover` constructors default to `Path(".quantum_solver_cache")` without consulting the config. To honor the environment variable, you must pass `cache_base=MetadataConfig.get_cache_base()` explicitly to the constructor.

⚠️ **Known Issue - BenchmarkSession.execute_run() Incompatibility**: `BenchmarkSession.execute_run()` checks for `puzzle.quantum_solver` attribute, but `QSudoku` stores the solver as the private attribute `_solver` with no public `quantum_solver` property. **All calls to `execute_run()` with `QSudoku` instances will currently fail.** Until this interface is aligned, prefer calling `puzzle.run()` directly and use `BenchmarkSession` only for querying/aggregating stage artifacts via its manager properties.

## Stage Map (new architecture)
- Stage 1 (Instance): Puzzle registry (`instances/registry.json`) via `InstanceMetadataManager`.
- Stage 2a (Logical IR): Circuit resources + hash per solver/encoding via `LogicalIRMetadataManager` → `stage_2a_logical_ir.json`.
- Stage 2b (IR Policy): Policy knobs (decompose flags, etc.) via `IRPolicyMetadataManager` → `stage_2b_ir_policy.json`.
- Stage 3 (Compilation): Append-only transpilation provenance via `CompilationMetadataManager` → `stage_3_compilation.jsonl` (links to Stage 2a `circuit_hash`).
- Stage 4 (Executable): Pulse-level placeholder (`ExecutableMetadataManager`).
- Stage 5 (Execution): Append-only run log via `ExecutionMetadataManager` → `stage_5_executions.jsonl` (links to Stage 3 `compilation_id`).
- Stages 6–7 (Evaluation/Normalization): Metrics via `MetricsMetadataManager` → `stage_6_7_metrics.json` (keyed by `run_id`). Multi-run aggregation is stored as `aggregated_*` entries.

## Two Ways to Run
- **Legacy quick path (works without env flags)**
  1) Build puzzle: `QSudoku.generate(...)` or `from_board`.
  2) Attach solver: `set_solver(ExactCoverQuantumSolver, encoding="simple"|"pattern", decompose_cnz=True, track_memory=False)`.
  3) Attach backend alias via `init_aer|init_ibm|init_quantinuum` (or `BackendManager.init_*`), or omit to keep PyTKET default.
  4) Build circuit with cache-aware behavior:
     - `build_circuit()` with no params → loads cached PyTKET circuit if exists, else builds PyTKET and caches it
     - `build_circuit(sdk="qiskit")` → always rebuilds in Qiskit format, caches a PyTKET copy
     - `build_circuit(force_overwrite=True)` → rebuilds PyTKET circuit and replaces cache
  5) `run(backend_alias, opt_level, shots)` → returns provider result; `report_resources()` reads legacy metadata. Validation/metrics is manual (see next section).
- **Stage-aware path (enable `SUDOKU_NISQ_NEW_METADATA=1`)**
  1) Set `SUDOKU_NISQ_NEW_METADATA=1`.
  2) Create `BenchmarkSession(puzzle_hash, cache_base?)` *if you want a convenient API for Stage queries/aggregation*.
  3) `register_puzzle(puzzle)` (Stage 1).
  4) Build + run using the normal `QSudoku` API (`build_circuit()`, then `run(...)`). Stage 2a/2b/3/5/6–7 are recorded automatically by the solver when the env flag is enabled.
  5) Aggregate metrics via `session.stage6_7.compute_aggregated(run_ids=...)` or `session.get_metrics_summary(run_ids=...)`.

  ⚠️ **Critical Implementation Issue**: `BenchmarkSession.execute_run()` checks for `puzzle.quantum_solver` attribute (line 205 of benchmark_session.py), but `QSudoku` stores the solver as `_solver` (private) with no public `quantum_solver` property. **This means `execute_run()` will always fail with `QSudoku` instances** until the interface is fixed. Use `puzzle.run()` directly and leverage `BenchmarkSession` only for stage queries and aggregation via its manager properties (`stage1`, `stage2a`, `stage3`, `stage5`, `stage6_7`).

## Validation & Metrics Requirements

**Validation Context**:
- Validation context is **required** for both Stage 6 evaluation metrics and Stage 7 normalization metrics. Without it, only Stage 5 execution metadata is recorded; Stages 6–7 are skipped.
- For `QSudoku`, use the public API: `puzzle.set_validation_context(valid_solutions=[...])` where `valid_solutions` are valid solution **bitstrings** (e.g., `["0110", "1001"]`).
- `QSudoku` constructs a `ValidationContext` internally with `valid_solutions`, `total_valid_count`, and `solution_validator` function, then passes it to the solver.

**Circuit Metrics for Normalization**:
- `two_qubit_gates` and `circuit_volume` are optional inputs for Stage 7 retention/log-loss normalization.
- Typically supplied via compiled-circuit inspection or Stage 5 `circuit_metrics` parameter in `ExecutionMetadataManager.record(circuit_metrics=...)`.
- Top-k defaults: k ∈ {1,3,5,10}.

**Legacy Metrics**:
- Legacy `eta_*` metrics (eta_gate, eta_volume, eta_shot) are deprecated.
- Stage 6–7 implementation aggregates legacy keys from older cached files for backward compatibility.
- Current Stage 7 computation prefers: retention-based metrics, log-loss metrics, and shot budget metrics.

## Provider & Backend Notes
- Backend aliases are registered through `BackendManager` (singleton: `BackendManager.inst()`). Aer always available; IBM/Quantinuum/AWS are optional imports and should be guarded.
- SDK detection in `QuantumSolver._detect_backend_sdk` handles native `qiskit_ibm_runtime` backends vs PyTKET IBMQ wrappers; override by passing `sdk="qiskit"|"pytket"|"braket"` to `build_circuit`.
- Qiskit transpilation prefers `generate_preset_pass_manager` when `backend.target` is present; otherwise falls back to `qiskit.compiler.transpile`. PyTKET uses `get_compiled_circuit`. Braket client-side transpilation is not supported (Stage 3 can still log metadata if provided).

## Files to Consult
- Core pipeline: `src/sudoku_nisq/quantum_solver.py`, `src/sudoku_nisq/solvers/exact_cover_solver.py`, `src/sudoku_nisq/q_sudoku.py`, `src/sudoku_nisq/backends.py`.
- Metadata stages: files in `src/sudoku_nisq/metadata/` (notably `benchmark_session.py`, `logical_ir.py`, `compilation.py`, `execution.py`, `metrics.py`, `config.py`).
- Metrics contracts: `src/sudoku_nisq/metrics/data_models.py`; calculator stubs in `src/sudoku_nisq/metrics/calculators/`.
- Architectural docs: `docs/internal/architecture/metrics_architecture_summary.md` and companions; migration guidance in `docs/guide/upgrading_from_metadata_manager.md` (referenced from legacy examples).

## Common Pitfalls

**Missing Environment Flag**:
- Without `SUDOKU_NISQ_NEW_METADATA=1`, stage files are not written; only legacy `metadata.json` is updated.
- With the flag enabled, dual-write occurs: both legacy and new stage-aware files are populated.

**Cache Behavior Misunderstanding**:
- `build_circuit()` with no parameters → loads cached PyTKET circuit if available; builds PyTKET and caches if not.
- `build_circuit(sdk="qiskit")` → **always rebuilds** in Qiskit format; **ignores cache** but writes PyTKET copy for future cache reads.
- `build_circuit(force_overwrite=True)` → rebuilds and **replaces** the cached PyTKET circuit.
- Cache is always stored as PyTKET JSON for SDK-agnostic compatibility.

**Validation Context Missing**:
- Without validation context, only Stage 5 execution metadata is recorded.
- Stages 6–7 metrics computation is **completely skipped**.
- Always call `puzzle.set_validation_context([...])` before runs if you need metrics.

**Backend Not Attached**:
- `QSudoku.run()` requires a backend alias that has been attached via `init_aer()`, `init_ibm()`, `init_quantinuum()`, or `attach_backend()`.
- Aer initialization (`puzzle.init_aer()`) is the safest default for local testing and development.

**BenchmarkSession.execute_run() Incompatibility**:
- `execute_run()` will always fail with `QSudoku` due to `quantum_solver` attribute mismatch (see Big Picture section).
- Use `puzzle.run()` directly until interface is fixed.

**Cache Base Override Behavior**:
- The `SUDOKU_NISQ_CACHE_DIR` environment variable is read by `MetadataConfig.get_cache_base()` but **not automatically used** by `QSudoku`/`QExactCover` constructors.
- `QSudoku.__init__()` and `QExactCover.__init__()` default to `Path(".quantum_solver_cache")` without consulting the config.
- To honor the environment variable, you must explicitly pass: `QSudoku(..., cache_base=MetadataConfig.get_cache_base())`
- This design allows per-instance cache control while supporting global defaults via environment variables.

## Quick Commands
- Install: `poetry install` (Python 3.11–3.12).
- Run examples: `python examples/exact.py` (fast), `python examples/exact_cover_benchmark.py`, `python examples/canonical_encoding_demo.py`.
- Tests: `poetry run pytest` (integration IBM paths auto-skip if deps absent). Use `PYTEST_RUN_HEAVY=1` for heavy cases.
- Migrate legacy metadata: `python scripts/migrate_metadata.py --dry-run` then rerun without `--dry-run` if clean.
