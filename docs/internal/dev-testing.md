# Development Testing Guide

This project separates fast logic coverage from SDK/backends and long-running simulations. Use the markers below to pick the right slice for local runs and CI jobs.

## Markers
- `unit`: Pure logic tests with no external SDK requirements.
- `integration`: Exercises SDK/backends (PyTKET, Qiskit, IBM runtime, Aer) or broader flows.
- `heavy`: RAM- or time-intensive simulations (opt-in only).
- `slow`, `resource_only`, `out_of_scope`: Legacy markers retained for completeness.

## Recommended command sets
- Quick unit pass: `poetry run pytest -m "unit"`
- Integration (no heavy): `poetry run pytest -m "integration and not heavy"`
- Full matrix locally: `poetry run pytest` (includes coverage for `src` and `tests`)
- Skip SDK-dependent suites if optional deps are missing: import guards will auto-skip when `pytket`, `qiskit`, or `qiskit_ibm_runtime` are unavailable.

## Notes
- Coverage now targets both `src` and `tests`; HTML output remains in `htmlcov/`.
- BackendManager refactor tests remain skipped intentionally; Aer/Qiskit heavy runs stay behind `heavy` and optional dependency guards.
- Use the markers to configure CI (e.g., default fast job: `-m "unit"`; nightly/extended: `-m "integration"`, plus `--run-heavy` if desired).
