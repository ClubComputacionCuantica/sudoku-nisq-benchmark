"""Solvers package public API.

Expose solver classes and ensure submodules are importable via
`sudoku_nisq.solvers.<module>` for Sphinx autodoc/autosummary.
"""

# Import submodules so attributes exist on the package for autodoc
from . import backtracking_solver as backtracking_solver  # noqa: F401
from . import graph_coloring_solver as graph_coloring_solver  # noqa: F401
from . import exact_cover_solver as exact_cover_solver  # noqa: F401

# Re-export key classes for convenient access
from .exact_cover_solver import ExactCoverQuantumSolver  # noqa: F401
from .backtracking_solver import BacktrackingQuantumSolver  # noqa: F401
from .graph_coloring_solver import GraphColoringQuantumSolver  # noqa: F401

__all__ = [
	'ExactCoverQuantumSolver',
	'BacktrackingQuantumSolver',
	'GraphColoringQuantumSolver',
	'backtracking_solver',
	'graph_coloring_solver',
	'exact_cover_solver',
]