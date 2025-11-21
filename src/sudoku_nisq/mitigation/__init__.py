"""Error mitigation utilities for quantum circuits.

This module provides Zero Noise Extrapolation (ZNE) and Probabilistic Error
Cancellation (PEC) integration for quantum Sudoku solvers via Mitiq.
"""

from sudoku_nisq.mitigation.expectation_wrapper import compute_success_expectation
from sudoku_nisq.mitigation.executors import create_zne_executor, create_pec_executor

__all__ = [
    'compute_success_expectation',
    'create_zne_executor',
    'create_pec_executor',
]
