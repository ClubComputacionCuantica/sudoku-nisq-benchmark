import pytest
from unittest.mock import MagicMock, patch, mock_open
from pathlib import Path
import json

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.quantum_solver import QuantumSolver
from pytket import Circuit, OpType


# A concrete dummy solver for testing purposes
class DummySolver(QuantumSolver):
    def _build_circuit(self) -> Circuit:
        circ = Circuit(2, name="dummy_circuit")
        circ.H(0)
        circ.CX(0, 1)
        circ.add_gate(OpType.CnX, [0, 1])  # for mcx count
        return circ

    def resource_estimation(self):
        return {"qubits": 2, "depth": 2}


@pytest.fixture
def sudoku_game():
    """Fixture for a standard 4x4 QSudoku game."""
    return QSudoku(subgrid_size=2, num_missing_cells=4)


@pytest.fixture
def quantum_solver(sudoku_game):
    """Fixture for a DummySolver instance."""
    # Using a real MetadataManager but with a temporary cache directory
    with patch('sudoku_nisq.metadata_manager.tempfile.NamedTemporaryFile'), \
         patch('sudoku_nisq.metadata_manager.os.fsync'), \
         patch('sudoku_nisq.metadata_manager.os.replace'):

        solver = DummySolver(sudoku_game, encoding="test_encoding", store_transpiled=True)
        # Mock the metadata manager's methods to avoid file system writes in most tests
        solver._metadata.save = MagicMock()
        solver._metadata.set_backend_resources = MagicMock()
        return solver


def test_quantum_solver_initialization(quantum_solver, sudoku_game):
    """Test basic attributes of QuantumSolver initialization."""
    assert quantum_solver.sudoku == sudoku_game
    assert quantum_solver.encoding == "test_encoding"
    assert quantum_solver.store_transpiled is True
    assert quantum_solver.main_circuit is None
    assert quantum_solver.solver_name == "DummySolver"


def test_build_main_circuit_no_cache(quantum_solver):
    """Test building the main circuit when it's not cached."""
    quantum_solver._metadata.set_main_circuit_resources = MagicMock()
    quantum_solver._metadata.save = MagicMock()
    with patch.object(Path, 'exists') as mock_exists, \
         patch.object(quantum_solver, 'save_circuit') as mock_save:

        mock_exists.return_value = False

        circuit = quantum_solver.build_main_circuit()

        assert circuit is not None
        assert circuit.n_qubits == 2
        assert quantum_solver.main_circuit == circuit
        mock_save.assert_called_once_with(circuit, quantum_solver.main_circuit_path)
        quantum_solver._metadata.set_main_circuit_resources.assert_called_once()
        quantum_solver._metadata.save.assert_called_once()


def test_build_main_circuit_with_cache(quantum_solver):
    """Test loading the main circuit from cache."""
    with patch.object(Path, 'exists') as mock_exists, \
         patch.object(quantum_solver, 'load_circuit') as mock_load:

        mock_exists.return_value = True
        mock_circuit = Circuit(2)
        mock_load.return_value = mock_circuit

        circuit = quantum_solver.build_main_circuit()

        assert circuit == mock_circuit
        mock_load.assert_called_once_with(quantum_solver.main_circuit_path)
        assert quantum_solver.main_circuit == mock_circuit


def test_save_and_load_circuit(quantum_solver, tmp_path):
    """Test saving and loading a circuit to/from a file."""
    circuit_to_save = quantum_solver._build_circuit()
    file_path = tmp_path / "test_circuit.json"

    # Save
    quantum_solver.save_circuit(circuit_to_save, file_path)

    # Verify file was written
    assert file_path.exists()

    # Load
    loaded_circuit = quantum_solver.load_circuit(file_path)

    assert loaded_circuit.n_qubits == circuit_to_save.n_qubits
    assert loaded_circuit.n_gates == circuit_to_save.n_gates
    assert loaded_circuit.get_commands() == circuit_to_save.get_commands()


@patch('sudoku_nisq.quantum_solver.BackendManager')
def test_transpile_and_analyze(mock_backend_manager, quantum_solver):
    """Test the transpile_and_analyze method."""
    # Setup backend mock
    mock_backend = MagicMock()
    mock_compiled_circ = Circuit(2)
    mock_backend.get_compiled_circuit.return_value = mock_compiled_circ
    mock_backend_manager.get.return_value = mock_backend

    # Attach backend to sudoku instance
    quantum_solver.sudoku._attached_backends['mock_be'] = mock_backend

    with patch.object(quantum_solver, 'save_circuit') as mock_save:
        # Mock the main circuit to avoid building it during transpile_and_analyze
        quantum_solver.main_circuit = Circuit(2)
        
        results = quantum_solver.transpile_and_analyze(
            backend_alias="mock_be",
            opt_levels=[1, 2]
        )

        assert 1 in results
        assert 2 in results
        assert "error" not in results[1]
        assert results[1]["n_qubits"] == mock_compiled_circ.n_qubits

        # Check that compilation was called for each level
        assert mock_backend.get_compiled_circuit.call_count == 2

        # Check that results were saved (should be 2 calls for transpiled circuits)
        assert mock_save.call_count == 2

        # Check metadata was updated
        assert quantum_solver._metadata.set_backend_resources.call_count == 2
        quantum_solver._metadata.save.assert_called()


@patch('sudoku_nisq.quantum_solver.BackendManager')
def test_transpile_and_analyze_compilation_error(mock_backend_manager, quantum_solver):
    """Test transpile_and_analyze with a compilation failure."""
    quantum_solver._metadata.set_backend_resources = MagicMock()
    mock_backend = MagicMock()
    # Simulate a failure for opt level 2
    mock_backend.get_compiled_circuit.side_effect = [Circuit(2), RuntimeError("Compilation failed")]
    mock_backend_manager.get.return_value = mock_backend
    quantum_solver.sudoku._attached_backends['mock_be'] = mock_backend

    results = quantum_solver.transpile_and_analyze("mock_be", opt_levels=[1, 2])

    assert 1 in results
    assert "error" not in results[1]
    assert 2 in results
    assert "error" in results[2]
    assert "Compilation failed" in results[2]["error"]

    # Metadata should be saved for both success and failure cases
    assert quantum_solver._metadata.set_backend_resources.call_count == 2


@patch('sudoku_nisq.quantum_solver.BackendManager')
def test_run_method(mock_backend_manager, quantum_solver):
    """Test the run method for executing a circuit."""
    # Setup backend mock
    mock_backend = MagicMock()
    mock_compiled_circ = Circuit(2)
    mock_handle = "job-123"
    mock_result = "some_result"
    mock_backend.get_compiled_circuit.return_value = mock_compiled_circ
    mock_backend.process_circuit.return_value = mock_handle
    mock_backend.get_result.return_value = mock_result
    mock_backend_manager.get.return_value = mock_backend
    quantum_solver.sudoku._attached_backends['mock_be'] = mock_backend

    result = quantum_solver.run("mock_be", shots=100, optimisation_level=1)

    assert result == mock_result
    mock_backend.get_compiled_circuit.assert_called_once()
    mock_backend.process_circuit.assert_called_once_with(mock_compiled_circ, n_shots=100)
    mock_backend.get_result.assert_called_once_with(mock_handle)


def test_count_mcx_gates(quantum_solver):
    """Test the MCX gate counting utility."""
    circuit = quantum_solver._build_circuit()  # This dummy circuit has one CnX
    assert quantum_solver.count_mcx_gates(circuit) == 1

    # Test with a circuit having no MCX gates
    simple_circ = Circuit(2)
    simple_circ.H(0)
    assert quantum_solver.count_mcx_gates(simple_circ) == 0