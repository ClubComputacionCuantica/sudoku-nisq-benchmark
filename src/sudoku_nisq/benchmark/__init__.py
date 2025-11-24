"""Benchmark module for staged quantum hardware comparison.

This module provides a high-level interface for benchmarking quantum Sudoku solvers
across different backends with a staged pipeline:

- Stage 0: Logical circuit analysis (pre-benchmark)
- Stage 1: Transpilation and hardware feasibility checks
- Stage 2: Hardware execution (opt-in with confirmation)
- Stage 3: Analysis, comparison, and reporting

Example:
    >>> from sudoku_nisq.benchmark import Benchmark
    >>> 
    >>> # Quick comparison
    >>> bench = Benchmark(size=4, missing_cells=6)
    >>> bench.add_backend("aer")
    >>> 
    >>> # Stage 0: Analyze logical circuit
    >>> logical = bench.analyze_logical_circuits()
    >>> print(f"Circuit needs {logical.n_qubits} qubits")
    >>> 
    >>> # Stage 1: Check feasibility
    >>> transpiled = bench.transpile_all(opt_level=2)
    >>> transpiled.print_summary()
    >>> 
    >>> # Stage 2: Run on hardware (if feasible)
    >>> results = bench.run_hardware(shots=1024)
"""

from sudoku_nisq.benchmark.comparison import Benchmark
from sudoku_nisq.benchmark.results import (
    LogicalAnalysis,
    BackendFeasibility,
    TranspilationReport,
    ExecutionResults,
    FeasibilityStatus,
)

__all__ = [
    "Benchmark",
    "LogicalAnalysis",
    "BackendFeasibility",
    "TranspilationReport",
    "ExecutionResults",
    "FeasibilityStatus",
]
