import pytest
from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.exact_cover_solver import ExactCoverQuantumSolver
from pytket import Circuit

@pytest.fixture
def sudoku_game():
    """Fixture for a small 2x2 Sudoku to keep circuits small."""
    # A fixed board ensures the generated circuit is consistent
    board = [
        [1, 2, 3, 0],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 2, 1]
    ]
    return QSudoku(board=board, subgrid_size=2)

def test_exact_cover_solver_initialization(sudoku_game):
    """Test initialization of the ExactCoverQuantumSolver."""
    solver = ExactCoverQuantumSolver(sudoku_game, encoding="simple")
    assert solver is not None
    assert solver.encoding == "simple"
    assert solver.solver_name == "ExactCoverQuantumSolver"

def test_build_circuit_simple_encoding(sudoku_game):
    """Test the _build_circuit method with 'simple' encoding."""
    solver = ExactCoverQuantumSolver(sudoku_game, encoding="simple")
    circuit = solver._build_circuit()
    assert isinstance(circuit, Circuit)
    # A simple check to ensure the circuit is not empty
    assert circuit.n_qubits > 0
    assert circuit.n_gates > 0

def test_build_circuit_pattern_encoding(sudoku_game):
    """Test the _build_circuit method with 'pattern' encoding."""
    solver = ExactCoverQuantumSolver(sudoku_game, encoding="pattern")
    circuit = solver._build_circuit()
    assert isinstance(circuit, Circuit)
    assert circuit.n_qubits > 0
    # The pattern encoding should also produce a non-empty circuit
    assert circuit.n_gates > 0

def test_invalid_encoding_raises_error(sudoku_game):
    """Test that providing an unknown encoding raises a ValueError."""
    with pytest.raises(ValueError, match="Unknown encoding 'invalid_encoding'"):
        ExactCoverQuantumSolver(sudoku_game, encoding="invalid_encoding")

def test_resource_estimation(sudoku_game):
    """Test the resource_estimation method."""
    solver = ExactCoverQuantumSolver(sudoku_game, encoding="simple")
    resources = solver.resource_estimation()
    assert isinstance(resources, dict)
    assert "n_qubits" in resources
    assert "depth" in resources
    assert resources["n_qubits"] > 0
