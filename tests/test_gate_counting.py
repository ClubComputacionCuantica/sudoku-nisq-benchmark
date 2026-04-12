"""Test gate counting feature for exact cover circuits."""

import pytest

pytest.importorskip("pytket")
pytest.importorskip("qiskit")

pytestmark = pytest.mark.integration

from sudoku_nisq.q_sudoku import QSudoku  # noqa: E402
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver  # noqa: E402

def test_gate_counting_pytket():
    """Test gate counting with PyTKET implementation."""
    # Create a simple 2x2 Sudoku puzzle
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    # Build circuit with PyTKET (default)
    circuit = puzzle.build_circuit(sdk="pytket")
    
    # Get gate counts from the solver
    gate_counts = puzzle._solver.gate_counts
    
    # Basic assertions
    assert circuit.n_qubits > 0
    assert circuit.n_gates > 0
    assert isinstance(gate_counts, dict)

def test_gate_counting_qiskit():
    """Test gate counting with Qiskit implementation."""
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    circuit = puzzle.build_circuit(sdk="qiskit")
    gate_counts = puzzle._solver.gate_counts
    assert circuit.num_qubits > 0
    assert isinstance(gate_counts, dict)

def test_gate_counts_metadata():
    """Test that gate counts are included in metadata."""
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    puzzle.build_circuit()
    gate_counts = puzzle._solver.get_gate_counts()
    assert isinstance(gate_counts, dict)