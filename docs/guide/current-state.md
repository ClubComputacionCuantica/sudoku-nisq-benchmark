# Current State of the Framework

This page summarizes what’s implemented today, what’s stable, and what’s planned.

## Components at a glance

- Core
  - `QSudoku`: high-level puzzle orchestration (attach/init backends, set solver, run).
  - `ExperimentRunner`: utilities for running/comparing configurations (WIP).
- Encodings (what to solve)
  - `ExactCoverEncoding`: emits `.universe`, `.simple_subsets`, `.pattern_subsets` from a Sudoku puzzle.
- Solvers (orchestrate)
  - `ExactCoverQuantumSolver`: Grover-style exact cover solver; delegates to circuit builders per SDK.
  - Resource estimation available via `solver.resource_estimation()`.
- Circuits (how to solve)
  - Exact cover circuit builders:
    - PyTKET: `sudoku_nisq.circuits.exact_cover.pytket_impl` (primary)
    - Qiskit: `sudoku_nisq.circuits.exact_cover.qiskit_impl` (native implementation)
    - Braket: TODO (native); currently falls back to PyTKET conversion in some paths.
- Providers/Backends (where to run)
  - Provider abstraction under `sudoku_nisq.providers`.
  - IBM and Quantinuum implementations present; auth + device registration handled there.
  - `BackendManager`: registry that composes multiple providers and exposes a unified API.

## Public API (import surface)

Preferred canonical imports:

```python
from sudoku_nisq import QSudoku, ExperimentRunner
from sudoku_nisq import ExactCoverEncoding, ExactCoverQuantumSolver
# Optional: BackendManager (public surface may evolve)
from sudoku_nisq import BackendManager
```

Deprecated duplicate modules have been removed:
- `sudoku_nisq.exact_cover_encoding` → use `sudoku_nisq.encodings.exact_cover_encoding`
- `sudoku_nisq.exact_cover_solver` → use `sudoku_nisq.solvers.exact_cover_solver`

## Layering & dependencies

- Encodings → Circuits → Providers; Solvers orchestrate.
- No provider imports inside encodings or circuits.
- Providers normalize run results to a common dict shape.

See the [Architecture](architecture.md) guide for details.

## Supported backends and SDKs

Current production‑ready execution is limited to IBM via PyTKET. Other entries are exploratory and may change:

- SDKs
  - PyTKET: supported (primary build/execution path)
  - Qiskit: supported natively for exact cover circuits
  - Braket: TODO (native) – falls back or is unimplemented
- Providers
  - IBM: tested and functional
  - Quantinuum: placeholder; authentication/circuit paths under active development
  - AWS Braket: initial scaffolding only; native circuits and provider integration pending

Depth values in analytical resource estimation may be `None` when not computed (e.g. exact cover estimation). Documentation examples reflect this.

## Known issues / work in progress

- Tests: Some failures around `BackendManager.clear()` being called without an instance; a classmethod wrapper is planned.
- Notebooks: Some cells still import from removed root paths; update to `from sudoku_nisq import ExactCoverQuantumSolver`.
- Braket: Native exact cover circuit builder to be implemented.
- Docs: Linkcode (source links) and Intersphinx to be enabled with repo/tag mapping.

## Roadmap (short)

- [ ] Add Braket-native circuit builder
- [ ] Finalize provider Protocol and document result normalization
- [ ] Update notebooks to public import paths; add examples gallery
- [ ] Enable docs linkcode + intersphinx; add CI docs build (warnings-as-errors)
- [ ] Stabilize public API and remove any leftover compat shims
