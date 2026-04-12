# CLI Design (Typer) — Sudoku NISQ Benchmark

## Goals

- Provide a single, coherent command-line interface for the existing functionality in this repository.
- Keep the CLI a thin layer over today’s APIs:
  - Sudoku: `QSudoku` and its solver/backends helpers
  - Generic exact cover: `QExactCover` / `ExactCoverProblem`
  - Solver: `ExactCoverQuantumSolver`
  - Backends: `BackendManager.inst()`
  - Metadata: legacy `MetadataManager` + new stage-aware system (feature-flagged)
  - Metrics: existing calculators/collectors and `QSudoku.set_validation_context()` / solver hooks
  - Mitigation: Mitiq-based ZNE/PEC integration
- Use **Typer** for the CLI (help text, subcommands, defaults, shell completion).
- Be explicit about platform constraints: **Windows + Aer uses `qasm_simulator` fallback** (documented behavior).

## Non-goals

- No new algorithms or “smart” puzzle solving beyond what the Python APIs already do.
- No interactive TUI.
- No reworking of caches, metadata formats, or result schemas.
- No attempt to serialize/deserialize arbitrary Python objects for mitigation (e.g., custom noise models) unless a stable JSON schema already exists.

## Proposed Binary & Package Entry Point

- One executable: `sudoku-nisq`
- Python module entry: `python -m sudoku_nisq.cli` (optional convenience)
- Poetry/pyproject entrypoint (design intent): expose `sudoku-nisq = sudoku_nisq.cli:app` (Typer app)

## Global UX Conventions

### Output formats

All commands support:
- `--format text|json` (default `text`)
- `--quiet` (reduces non-essential output)

Guideline:
- Commands that generate artifacts (puzzles, circuits, metadata exports) should print the artifact path in text mode and the full payload/path in JSON mode.

### Cache / metadata environment

The CLI should respect the existing environment variables:
- `SUDOKU_NISQ_CACHE_DIR` — overrides cache base (default `.quantum_solver_cache`)
- `SUDOKU_NISQ_NEW_METADATA=1` — enables stage-aware metadata and `BenchmarkSession`
- `SUDOKU_NISQ_SUPPRESS_DEPRECATION=1` — suppress deprecation warnings (legacy metadata)

CLI-level overrides:
- `--cache-dir PATH` (sets/overrides cache base and should also set `SUDOKU_NISQ_CACHE_DIR` for subprocesses)
- `--new-metadata / --legacy-metadata` (maps to `SUDOKU_NISQ_NEW_METADATA`)

### Artifact layout

Design intent (no new layout invented):
- Use the project’s existing caching conventions:
  - `.quantum_solver_cache/{puzzle_hash}/{solver}/{encoding}/...`
  - Stage-aware metadata under the same cache root when enabled

### Puzzle “handle” concept

Many commands need a way to refer to “the current puzzle” without implementing a full project workspace manager.

Design: support **explicit puzzle references** in every command:
- `--puzzle-hash HASH` (operate on existing cached puzzle)
- `--puzzle-file PATH` (load a serialized puzzle description)
- `--board-file PATH` (load board JSON)
- `--board-string JSON` (inline board JSON)
- `--generate ...` flags (create new puzzle for the command)

The CLI should never rely on implicit “current puzzle” state.

## Input/Output Schemas (Design)

### Sudoku board JSON

A Sudoku board is represented as a JSON array-of-arrays of integers (0 means empty):

```json
[[1,0,0,4],[0,0,1,0],[0,4,0,0],[2,0,0,3]]
```

### Exact cover problem JSON

A generic exact cover problem file:

```json
{
  "universe": [0, 1, 2, 3],
  "subsets": {
    "S_0": [0, 3],
    "S_1": [1, 2],
    "S_2": [0, 1, 2]
  },
  "num_solutions": 1
}
```

### Result JSON

For `--format json`, the CLI should return a stable envelope:

```json
{
  "ok": true,
  "command": "sudoku run-aer",
  "puzzle_hash": "...",
  "artifacts": {"cache_dir": "..."},
  "data": {"counts": {"0101": 123}}
}
```

(Exact fields are design-level; the CLI should primarily surface the raw dicts returned by `QSudoku`/`QExactCover` with minimal reshaping.)

## Windows + Aer Policy (Required)

The CLI must document and surface the existing behavior:
- On Windows, Aer simulation may fall back to Qiskit’s `qasm_simulator` for stability.
- In this fallback mode, **noise models and advanced Aer methods are not available**.

Design additions:
- `sudoku run-aer` prints (text mode) or returns (json mode) an explicit `"simulator": "aer"|"qasm_simulator"` field.
- `sudoku run-aer-noisy` should:
  - error clearly on Windows fallback if noise simulation is requested but not supported
  - recommend setting `SUDOKU_NISQ_USE_QISKIT_AER=1` (if supported by current code) and installing required packages

## Full Command Tree (Draft)

Top-level:

- `sudoku-nisq --help`
- `sudoku-nisq completion [bash|zsh|fish|powershell]`

Subcommands:

### 1) `sudoku`

Purpose: Sudoku-focused workflow using `QSudoku`.

Commands:

- `sudoku generate`
  - Creates a new puzzle and optionally writes it to a file.
  - Maps to `QSudoku.generate(size=..., subgrid_size=..., num_missing_cells=..., canonicalize=...)`.

  Flags:
  - `--size 2|4|9|16...` (preferred)
  - `--subgrid-size K` (backward compatible)
  - `--missing-cells N`
  - `--canonicalize / --no-canonicalize`
  - `--out PATH` (writes a puzzle/board JSON)

- `sudoku from-board`
  - Load a puzzle from a board JSON.
  - Maps to `QSudoku.from_board(board, ...)` (or equivalent constructor).

  Flags:
  - `--board-file PATH | --board-string JSON`
  - `--canonicalize / --no-canonicalize`
  - `--out PATH` (optional)

- `sudoku show`
  - Prints puzzle board and key properties.
  - Optionally prints puzzle hash.

  Flags:
  - puzzle reference flags (`--puzzle-hash`, `--board-file`, etc.)
  - `--show-hash`

- `sudoku set-solver`
  - Design note: CLI is stateless, so “set solver” should be expressed as flags on commands that need it.
  - Provide this command only if it writes a small config artifact for reuse (optional).

  Minimal design: **skip this command** and instead require solver flags in `build-circuit` and `run*`.

- `sudoku build-circuit`
  - Builds the logical circuit, writes to cache.
  - Maps to `puzzle.set_solver(...); puzzle.build_circuit(sdk=..., force_overwrite=...)`.

  Flags:
  - puzzle reference flags
  - `--solver exact-cover` (default; future: backtracking/graph-coloring)
  - `--encoding simple|pattern` (default `simple`)
  - `--sdk pytket|qiskit|braket` (omit to allow cached pytket reuse)
  - `--decompose-cnz / --no-decompose-cnz` (default true)
  - `--track-memory / --no-track-memory` (advanced; default false)
  - `--force-overwrite / --no-force-overwrite` (refresh cached pytket copy)

- `sudoku run-aer`
  - Run locally on Aer (ideal simulation).
  - Maps to `puzzle.run_aer(shots=..., **kwargs)`.

  Flags:
  - puzzle reference flags
  - solver flags (same as `build-circuit`)
  - `--shots N` (default 1024)
  - `--opt-level 0..3` (if supported; default consistent with API)
  - `--memory / --no-memory` (if supported by underlying runner)

- `sudoku run-aer-noisy`
  - Run with noise model support.
  - Maps to `puzzle.run_aer_with_noise(shots=..., device_name=..., method=..., optimization_level=..., **aer_options)`.

  Flags:
  - puzzle reference flags
  - solver flags
  - `--shots N`
  - `--device-name NAME` (auto-generate noise model from IBM fake backend)
  - `--method density_matrix|statevector|...` (default per API)
  - `--optimization-level 0..3`
  - `--aer-option KEY=VALUE` (repeatable; passed through)

  Notes:
  - If a “custom noise model” is needed, it is not CLI-friendly today; keep scope to `--device-name` only.

- `sudoku run`
  - Run on a registered backend.
  - Maps to `puzzle.run(backend_alias, opt_level=..., shots=..., **kwargs)`.

  Flags:
  - puzzle reference flags
  - solver flags
  - `--backend-alias ALIAS` (required)
  - `--shots N` (required)
  - `--opt-level 0..3` (required)
  - `--job-arg KEY=VALUE` (repeatable; passed through)

- `sudoku format-result`
  - Decode a previously obtained result.
  - Maps to `puzzle.format_result(result)`.

  Flags:
  - puzzle reference flags
  - `--result-file PATH` (JSON with counts/result payload)
  - `--top-n N`

- `sudoku counts-plot`
  - Generate counts plot.
  - Maps to `puzzle.counts_plot(...)`.

  Flags:
  - puzzle reference flags
  - `--result-file PATH` or `--counts-file PATH`
  - `--top-n N` (default 20)
  - `--show-valid-only / --no-show-valid-only`
  - `--out PATH` (save figure)

- `sudoku report-resources`
  - Show qubits/gates/depth for main circuit.
  - Maps to `puzzle.report_resources()`.

  Flags:
  - puzzle reference flags
  - solver flags (if resource estimation depends on solver build)

- `sudoku hash`
  - Print deterministic puzzle hash.
  - Maps to `puzzle.get_hash()`.

- `sudoku set-validation-context`
  - Enables metrics collection hooks by providing known valid solutions.
  - Maps to `puzzle.set_validation_context(valid_solutions=[...])`.

  Flags:
  - puzzle reference flags
  - `--valid-solutions-file PATH` (JSON array of bitstrings)
  - `--valid-solution BITSTRING` (repeatable)

### 2) `exact-cover`

Purpose: generic exact cover workflow using `ExactCoverProblem` and `QExactCover`.

Commands:

- `exact-cover run-aer`
  - Load problem JSON and run on Aer.
  - Maps to `problem = ExactCoverProblem(...); qec = QExactCover(problem); qec.run_aer(...)`.

  Flags:
  - `--problem-file PATH` (required)
  - `--sdk qiskit|pytket` (default qiskit)
  - `--shots N` (default 1024)
  - `--opt-level 0..3` (default 0)
  - `--memory / --no-memory` (if supported)

- `exact-cover build-circuit`
  - Build circuit only.

  Flags:
  - `--problem-file PATH`
  - `--sdk qiskit|pytket`
  - `--out PATH` (optional circuit serialization if supported)

- `exact-cover report-resources`
  - Maps to `qec.report_resources()`.

- `exact-cover enumerate-solutions`
  - Classical backtracking enumeration (for small instances).
  - Maps to `ExactCoverProblem.enumerate_solutions(max_solutions=...)`.

  Flags:
  - `--problem-file PATH`
  - `--max-solutions N` (default 100)

### 3) `backend`

Purpose: manage provider authentication and backend registration via `BackendManager.inst()`.

Commands:

- `backend list`
  - Show registered aliases.

- `backend describe --alias ALIAS`
  - Show provider type, target device info if available.

- `backend init-aer`
  - Registers an Aer backend alias.
  - Maps to `QSudoku.init_aer(...)` or `BackendManager.inst().register(...)` (depending on current API).

  Flags:
  - `--alias aer` (default)
  - `--method statevector|density_matrix|...`
  - `--device-type CPU|GPU` (if supported)
  - `--precision ...` (if supported)

- `backend init-ibm`
  - Authenticates and registers device.
  - Maps to `QSudoku.init_ibm(api_token, instance, device, alias=...)`.

  Flags:
  - `--api-token TOKEN` (or `--api-token-file PATH`)
  - `--instance CRN`
  - `--device NAME`
  - `--alias ALIAS` (default to device)

- `backend init-quantinuum`
  - Registers Quantinuum backend.
  - Maps to `QSudoku.init_quantinuum(device, alias=..., token_store=..., provider=...)`.

  Flags:
  - `--device NAME` (e.g., `H1-1`)
  - `--alias ALIAS`
  - `--provider PROVIDER_ID` (optional)
  - Token storage is tricky in CLI; design offers:
    - `--token-store disk|env|none` (maps to supported storage types if available)

- `backend init-aws`
  - Registers AWS Braket backend alias.
  - If current code does not offer a stable helper, the CLI should either:
    - omit this command, or
    - provide “best-effort” registration with clear “optional/experimental” labeling.

### 4) `benchmark`

Purpose: orchestrated multi-run experiments and comparisons.

Design split (because code has both new and legacy):

- `benchmark run-session`
  - Uses `BenchmarkSession`.
  - **Requires** `SUDOKU_NISQ_NEW_METADATA=1`.

  Flags (high-level):
  - puzzle reference flags
  - solver flags
  - `--backends ALIAS` (repeatable)
  - `--shots N`
  - `--opt-level 0..3` (repeatable)
  - `--runs N` (repeatable runs)
  - `--different-seeds / --same-seed` (if supported)

- `benchmark run-matrix`
  - Wraps an existing example script conceptually (not re-implementing logic).
  - Intended to mirror workflows in `examples/metrics_multi_run.py` and experiment notebooks.

### 5) `metrics`

Purpose: compute, summarize, and export metrics from recorded runs.

Commands:

- `metrics summarize`
  - Reads stage-aware metrics (Stage 6–7) if available; otherwise errors or prints “no metrics available”.
  - Requires either:
    - `--puzzle-hash HASH` and run filters, or
    - explicit run IDs.

  Flags:
  - `--puzzle-hash HASH`
  - `--run-id ID` (repeatable)
  - `--backend-alias ALIAS` (optional filter)
  - `--opt-level N` (optional filter)

- `metrics export`
  - Uses existing reporters (e.g., JSON/CSV where available).

  Flags:
  - `--puzzle-hash HASH`
  - `--out PATH`
  - `--format json|csv`

Design note:
- Metrics correctness depends on a validation context (known valid solutions). The CLI should highlight this and offer `sudoku set-validation-context`.

### 6) `mitigation`

Purpose: run workflows with error mitigation (ZNE/PEC) using the existing Mitiq integration.

Key constraints:
- ZNE/PEC require careful configuration. The CLI design should expose only stable, high-level toggles and default strategies.

Commands:

- `mitigation run`
  - Run a Sudoku circuit with mitigation enabled.
  - Conceptually maps to solver run with mitigation strategy and required conversions.

  Flags:
  - puzzle reference flags
  - solver flags
  - `--backend-alias ALIAS` (or `--aer` shortcut)
  - `--shots N`
  - `--strategy zne|pec`
  - `--zne-scale-factors 1,3,5` (optional)
  - `--zne-method richardson|linear|...` (optional)
  - `--pec-samples N` (optional)

Design note:
- If mitigation integration is currently only reachable via Python-level calls, the CLI should still define this surface but mark it “experimental” until the internal API is confirmed stable.

### 7) `cache`

Purpose: inspect and manage `.quantum_solver_cache`.

Commands:

- `cache ls --puzzle-hash HASH`
- `cache path --puzzle-hash HASH` (prints directory)
- `cache purge --puzzle-hash HASH` (dangerous; confirmation flag)

### 8) `metadata`

Purpose: inspect metadata (legacy + stage-aware).

Commands:

- `metadata show --puzzle-hash HASH`
  - Prints available metadata files and key summary fields.

- `metadata stages --puzzle-hash HASH`
  - Stage-aware only: show which stage files exist.

### 9) `migrate`

Purpose: migration tools, including legacy → stage-aware.

Commands:

- `migrate metadata`
  - Wraps behavior of `scripts/migrate_metadata.py`.

  Flags:
  - `--cache-dir PATH`
  - `--dry-run / --no-dry-run`
  - `--verbose`

## Cross-Cutting Flag Sets (Standardized)

### Puzzle reference flags (shared)

- `--puzzle-hash HASH`
- `--board-file PATH`
- `--board-string JSON`
- `--size N --missing-cells M [--subgrid-size K]`
- `--canonicalize / --no-canonicalize`

Validation:
- Exactly one source of puzzle input should be required; if multiple are provided, error.

### Solver flags (shared)

- `--solver exact-cover` (default)
- `--encoding simple|pattern`
- `--decompose-cnz / --no-decompose-cnz`
- `--track-memory / --no-track-memory` (advanced)

### Backend/run flags (shared)

- `--backend-alias ALIAS`
- `--shots N`
- `--opt-level 0..3`

## Error Handling & Diagnostics (Design)

- All user errors should be `Typer.BadParameter`-style messages.
- When an optional provider dependency is missing:
  - error message should include which extra is needed (e.g., `qiskit-ibm-runtime`, `pytket-quantinuum`, `amazon-braket-sdk`).
- For Windows Aer fallback:
  - clearly state whether `qasm_simulator` was used
  - clearly state why noisy simulation is unavailable in fallback mode

## Minimal End-to-End User Stories

### Story A: Quick local solve (2×2)

1. `sudoku-nisq sudoku generate --size 2 --missing-cells 2 --encoding simple`
2. `sudoku-nisq sudoku run-aer --size 2 --missing-cells 2 --encoding simple --shots 512`
3. `sudoku-nisq sudoku format-result --... --result-file result.json`

### Story B: Build circuit for Qiskit (force rebuild)

- `sudoku-nisq sudoku build-circuit --board-file puzzle.json --encoding pattern --sdk qiskit --force-overwrite`

### Story C: Noisy Aer run from fake IBM device model

- `sudoku-nisq sudoku run-aer-noisy --board-file puzzle.json --device-name ibm_brisbane --shots 1024 --method density_matrix`

### Story D: Multi-run benchmark session (new metadata)

- `SUDOKU_NISQ_NEW_METADATA=1 sudoku-nisq benchmark run-session --board-file puzzle.json --backends aer --runs 5 --shots 2048 --opt-level 0 --opt-level 1`

### Story E: Migrate legacy metadata

- `sudoku-nisq migrate metadata --cache-dir .quantum_solver_cache --dry-run`

## Implementation Notes (Still Design)

- Typer structure:
  - root `app = typer.Typer()`
  - sub-apps: `sudoku_app`, `exact_cover_app`, `backend_app`, `benchmark_app`, `metrics_app`, `mitigation_app`, `cache_app`, `metadata_app`, `migrate_app`
- Keep the CLI layer pure orchestration:
  - parse inputs
  - call existing API
  - write outputs
- Avoid adding new dependencies beyond Typer unless necessary.

---

If you want, the next design artifact can be a “command-by-command contract” table listing exact return fields for `--format json`, but this document already contains the full command tree and flag set.
