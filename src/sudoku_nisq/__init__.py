"""Public API surface for ``sudoku_nisq``.

This module re-exports core entry points. Keep the list minimal and stable.

TODO:
- Add a high-level ``solve(puzzle, encoding=\"simple\", backend_alias=None, **kwargs)`` helper.
- Evaluate exposing resource estimation utilities directly.
- Remove deprecated root shim modules in next minor release.
"""

# Core puzzle & orchestration
from .q_sudoku import QSudoku  # noqa: F401

# Generic exact cover interface
from .exact_cover_problem import ExactCoverProblem  # noqa: F401
from .q_exact_cover import QExactCover  # noqa: F401

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
	"ExactCoverEncoding",
	"ExactCoverQuantumSolver",
	"BackendManager",
	"ExactCoverProblem",
	"QExactCover",
]

# Expose selected subpackages for Sphinx autosummary (read-only namespace proxies)
# This allows references like `sudoku_nisq.solvers` or `sudoku_nisq.providers`
# without encouraging direct attribute access to internals.
try:
	from . import solvers as solvers  # noqa: F401
	from . import circuits as circuits  # noqa: F401
	from . import providers as providers  # noqa: F401
	from . import mitigation as mitigation  # noqa: F401
	from . import utils as utils  # noqa: F401
except Exception:
	# Optional; during docs builds heavy deps may be mocked
	pass

# Deprecation guidance (import-time warning only if needed)
# TODO: Emit a single consolidated deprecation warning if env var SUDOKU_NISQ_WARN_DEPRECATED=1