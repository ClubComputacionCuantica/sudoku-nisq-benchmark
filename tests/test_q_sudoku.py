import pytest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.quantum_solver import QuantumSolver
from sudoku_nisq.backends import BackendManager

# A concrete dummy solver for testing purposes
class DummySolver(QuantumSolver):
    def _build_circuit(self):
        # A minimal valid circuit
        from pytket import Circuit
        return Circuit(1)

    def resource_estimation(self):
        return {"qubits": 1, "depth": 1}

@pytest.fixture
def sudoku_game():
    """Fixture for a standard 4x4 QSudoku game."""
    # Using a fixed board to ensure test consistency
    board = [
        [1, 2, 3, 4],
        [3, 4, 1, 2],
        [2, 1, 4, 3],
        [4, 3, 0, 0]
    ]
    return QSudoku(board=board, subgrid_size=2)

def test_q_sudoku_initialization(sudoku_game):
    """Test basic attributes of QSudoku initialization."""
    assert sudoku_game.board_size == 4
    assert sudoku_game.subgrid_size == 2
    assert sudoku_game.num_missing_cells == 2
    assert sudoku_game._solver is None

def test_set_solver(sudoku_game):
    """Test the set_solver method."""
    solver = sudoku_game.set_solver(DummySolver, encoding="test")
    assert isinstance(sudoku_game._solver, DummySolver)
    assert sudoku_game._solver == solver
    assert sudoku_game._solver.encoding == "test"

def test_build_circuit_without_solver(sudoku_game):
    """Test that building a circuit without a solver raises an error."""
    with pytest.raises(ValueError, match="No solver set"):
        sudoku_game.build_circuit()

def test_build_circuit_with_solver(sudoku_game):
    """Test that building a circuit delegates to the active solver."""
    sudoku_game.set_solver(DummySolver)
    # Mock the solver's build method to check if it's called
    sudoku_game._solver.build_main_circuit = MagicMock()
    sudoku_game.build_circuit()
    sudoku_game._solver.build_main_circuit.assert_called_once()

@patch('sudoku_nisq.q_sudoku.BackendManager')
def test_attach_backend(mock_backend_manager, sudoku_game):
    """Test attaching a backend."""
    mock_backend = MagicMock()
    mock_backend_manager.get.return_value = mock_backend
    
    sudoku_game.attach_backend("mock_backend")
    
    mock_backend_manager.get.assert_called_once_with("mock_backend")
    assert "mock_backend" in sudoku_game._attached_backends
    assert sudoku_game._attached_backends["mock_backend"] == mock_backend

def test_transpile_without_attached_backend(sudoku_game):
    """Test transpiling without an attached backend raises an error."""
    sudoku_game.set_solver(DummySolver)
    with pytest.raises(ValueError, match="not attached"):
        sudoku_game.transpile("some_backend", opt_levels=[1])

@patch('sudoku_nisq.q_sudoku.BackendManager')
def test_transpile_with_attached_backend(mock_backend_manager, sudoku_game):
    """Test that transpile calls the solver's transpile_and_analyze."""
    # Setup
    sudoku_game.set_solver(DummySolver)
    mock_backend = MagicMock()
    mock_backend_manager.get.return_value = mock_backend
    sudoku_game.attach_backend("mock_backend")
    
    # Mock the solver's method
    sudoku_game._solver.transpile_and_analyze = MagicMock(return_value={})
    
    # Action
    sudoku_game.transpile("mock_backend", opt_levels=[0, 1])
    
    # Assert
    sudoku_game._solver.transpile_and_analyze.assert_called_once_with(
        "mock_backend", [0, 1]
    )

@patch('pytket.extensions.qiskit.AerBackend')
def test_run_aer(mock_aer_backend, sudoku_game):
    """Test the run_aer method with a mock AerBackend."""
    # Setup
    sudoku_game.set_solver(DummySolver)
    
    # Mock the AerBackend instance and its methods
    mock_aer_instance = MagicMock()
    mock_aer_backend.return_value = mock_aer_instance
    
    mock_handle = MagicMock()
    mock_aer_instance.process_circuit.return_value = mock_handle
    
    mock_result = MagicMock()
    mock_aer_instance.get_result.return_value = mock_result

    # Action
    result = sudoku_game.run_aer(shots=100)

    # Assert
    assert result == mock_result
    # The run_aer method calls process_circuit directly with main_circuit, not get_compiled_circuit
    mock_aer_instance.process_circuit.assert_called_once()
    mock_aer_instance.get_result.assert_called_once_with(mock_handle)

def test_get_hash(sudoku_game):
    """Test the puzzle hashing function."""
    h = sudoku_game.get_hash()
    assert isinstance(h, str)
    assert len(h) == 64  # SHA-256

def test_solution_counting(sudoku_game):
    """Test the backtracking solution counter."""
    board = [
        [1, 0, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 0, 1]
    ]
    game = QSudoku(board=board, subgrid_size=2)
    # This is a known puzzle with 18 solutions.
    assert game.count_solutions() == 18
