# Changelog

## Latest highlights

- **Gate Counting and Memory Tracking Features (November 2025)**: Added automated tracking for exact cover quantum circuits.
  - **Gate Counting** (always enabled): Tracks fundamental gates (H, X, CX/CCX/C3X, CZ/CCZ, Measure) by type and control count.
  - **Memory Tracking** (optional, for advanced/dev use): Lightweight RAM usage monitoring during circuit construction.
  - Implementations for PyTKET (primary), Qiskit (native), and Braket (native Grover builder). Braket still performs transpilation server-side.
  - Consistent gate counts across all SDKs with `decompose_cnz` parameter.
  - Memory profiling with initial, peak, current, and delta metrics (enable via `track_memory=True`).
  - Accessible via `solver.get_gate_counts()` and `solver.get_memory_usage()`.
  - Gate counts always included in circuit metadata; memory usage included when tracking enabled.
  - See `GATE_COUNTING_IMPLEMENTATION.md` for details.

- Removed deprecated duplicate modules at `sudoku_nisq/exact_cover_encoding.py` and `sudoku_nisq/exact_cover_solver.py`.
	- Canonical imports are now: `from sudoku_nisq.encodings.exact_cover_encoding import ExactCoverEncoding` and `from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver`.
	- Public API convenience imports are available: `from sudoku_nisq import ExactCoverEncoding, ExactCoverQuantumSolver`.
- Added Sphinx documentation scaffold with MyST (Markdown), autosummary, and napoleon.
- Added Architecture guide capturing the recommended layering (encodings → circuits → providers; solvers orchestrate).
- Added Releases section that includes `RECENT_CHANGES.md` automatically.
- Extended `.gitignore` for coverage, docs build, notebooks cache, and tool caches.

- New: Native Qiskit exact cover circuit builder (`sudoku_nisq.circuits.exact_cover.qiskit_impl`).
  - Previously, Qiskit circuits were obtained by converting from PyTKET; now built directly in Qiskit.
  - Matches PyTKET logic (counter, oracle, diffuser, Grover loop) and maintains qubit ordering parity.

> Note: Some test failures currently relate to `BackendManager.clear()` being used without an instance in tests; a classmethod wrapper is planned. Notebooks still use old import paths and will be updated.

---

```{include} ../../RECENT_CHANGES.md
:relative-docs: ./
:relative-images:
```

<!--
TODO (automation options):
- Use Towncrier to collect news fragments per PR and generate this file at release.
- Or use Reno (OpenStack) with its Sphinx extension.
- On GitHub Actions release, generate and commit/publish the updated changelog.
-->
