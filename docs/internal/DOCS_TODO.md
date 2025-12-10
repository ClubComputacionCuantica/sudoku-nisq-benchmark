# Documentation & API Improvement TODOs

This file tracks planned enhancements to documentation clarity and API discoverability. Tasks are grouped by area. When a task is completed, replace its checkbox and briefly note the PR or commit.

## Status Legend
- [ ] Not started
- [~] In progress
- [x] Done

---
## 1. Provider Placeholder Status Notes
- [ ] Add explicit module docstrings to `src/sudoku_nisq/providers/aws_pending.py` and `src/sudoku_nisq/providers/quantinuum_pending.py`:
  - Purpose: clarify they are experimental/placeholders; API subject to change.
  - Include: brief roadmap sentence + link to `docs/guide/current-state.md` section (add a subsection "Provider Status" if missing).
  - Acceptance: Sphinx API pages show the status paragraph at top.

## 2. Package-Level Module Docstrings
Add concise purpose statements to package `__init__.py` files so Sphinx renders informative intros instead of empty pages.
Targets:
- [ ] `src/sudoku_nisq/circuits/__init__.py` – overview of circuit families (exact cover, graph coloring)
- [ ] `src/sudoku_nisq/circuits/exact_cover/__init__.py` – mapping from Sudoku constraints to exact-cover quantum circuits
- [ ] `src/sudoku_nisq/circuits/graph_coloring/__init__.py` – placeholder / future extension (if currently empty, note roadmap)
- [ ] `src/sudoku_nisq/providers/__init__.py` – abstraction layer + supported backends
- [ ] `src/sudoku_nisq/solvers/__init__.py` – solver architecture & strategy (classical vs quantum wrappers)
- [ ] `src/sudoku_nisq/encodings/__init__.py` – encoding transforms (exact cover representation, etc.)
Acceptance: Each page in generated API docs starts with a 2–5 line summary.

## 3. Docstring Coverage Audit
- [ ] Write `scripts/docstring_audit.py` to scan `src/sudoku_nisq` for public (non-leading underscore) classes/functions missing:
  - First line summary
  - `Parameters` section (for functions/methods with params)
  - Returns / Raises where meaningful
- [ ] Output a Markdown table (path, object, missing elements) into `docs/_reports/docstring_coverage.md` (create folder).
- [ ] Optionally wire into CI (GitHub Actions) as a documentation quality check (non-blocking).
Acceptance: Report committed; reduction of missing docstrings tracked over time.

## 4. Concept Workflow Index
Add a navigable conceptual map linking high-level workflow steps to modules.
- [ ] Decide placement: new `docs/guide/concept-index.md` OR section at end of `docs/guide/architecture.md`.
- [ ] Include table columns: Concept | Description | Key Classes/Modules | Example Entry Point.
Concepts to cover:
  - Pattern Generation → `PatternGeneration` (`sudoku_pattern_generation`)
  - Puzzle Abstraction → `SudokuPuzzle`
  - Constraint Encoding → `ExactCoverEncoding`
  - Circuit Construction → `circuits.exact_cover.*`
  - Backend Selection → `backends`, `providers.*`
  - Quantum Solving → `ExactCoverQuantumSolver` / `quantum_solver`
- [ ] Cross-link to existing guide pages for deeper dives.
Acceptance: Table renders; internal links validated; referenced modules resolve.

## 5. Optional Future Tasks
- [ ] Add "Provider Status" subsection to `docs/guide/current-state.md` with a matrix (Provider | Status | Notes).
- [ ] Introduce `docs/guide/contributing.md#documentation-style` examples (parameter, returns, raises formatting).
- [ ] Pre-commit hook to run `docstring_audit.py` in advisory mode.

---
## Process Guidelines
1. Keep docstring changes narrowly scoped (avoid refactors while documenting).
2. Prefer imperative mood for summaries ("Generate pattern dictionary" not "Generates...").
3. When APIs are provisional, mark with "(Experimental)" in summary.
4. Cross-reference using Sphinx roles (e.g., ``:mod:`sudoku_nisq.solvers` `` or ``:class:`sudoku_nisq.solvers.exact_cover_solver.ExactCoverQuantumSolver` ``).
5. Avoid TODO comments in code; use this file + issues for tracking.

## Tracking & Linking
For each completed task add a trailing reference, e.g. `(PR #123)`.

---
_Last updated: 2025-11-20_