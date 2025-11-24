"""Public API surface for ``sudoku_nisq``.

This module re-exports core entry points. Keep the list minimal and stable.

TODO:
- Add a high-level ``solve(puzzle, encoding=\"simple\", backend_alias=None, **kwargs)`` helper.
- Evaluate exposing resource estimation utilities directly.
- Remove deprecated root shim modules in next minor release.
"""

# Core puzzle & orchestration
from .q_sudoku import QSudoku  # noqa: F401
from .experiment_runner import ExperimentRunner  # noqa: F401

# Generic exact cover interface
from .exact_cover_problem import ExactCoverProblem  # noqa: F401
from .q_exact_cover import QExactCover  # noqa: F401

# Benchmarking (new high-level interface)
from .benchmark import Benchmark  # noqa: F401

# Encodings & solvers (canonical import paths)
from .encodings.exact_cover_encoding import ExactCoverEncoding  # noqa: F401
from .solvers.exact_cover_solver import ExactCoverQuantumSolver  # noqa: F401

# Backend manager singleton accessor (optional public surface)
try:
	from .backends import BackendManager  # noqa: F401
except Exception:  # pragma: no cover
	BackendManager = None  # type: ignore

__all__ = [
	"QSudoku",
	"ExperimentRunner",
	"Benchmark",
	"ExactCoverEncoding",
	"ExactCoverQuantumSolver",
	"BackendManager",
	"ExactCoverProblem",
	"QExactCover",
]

# Deprecation guidance (import-time warning only if needed)
# TODO: Emit a single consolidated deprecation warning if env var SUDOKU_NISQ_WARN_DEPRECATED=1