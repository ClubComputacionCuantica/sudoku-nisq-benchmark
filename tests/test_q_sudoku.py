import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import List

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.sudoku_puzzle import SudokuPuzzle


class TestQSudoku:
    """Test suite for QSudoku class functionality."""

    @pytest.fixture
    def sample_4x4_board(self) -> List[List[int]]:
        """Provide a sample 4x4 Sudoku board for testing."""
        return [
            [1, 0, 0, 4],
            [0, 2, 0, 0],
            [0, 0, 3, 0],
            [4, 0, 0, 1]
        ]

    @pytest.fixture
    def complete_4x4_board(self) -> List[List[int]]:
        """Provide a complete 4x4 Sudoku board for testing."""
        return [
            [1, 3, 2, 4],
            [3, 2, 4, 1],
            [2, 4, 3, 1],
            [4, 1, 2, 3]
        ]

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def q_sudoku_4x4(self, sample_4x4_board, temp_cache_dir) -> QSudoku:
        """Create a QSudoku instance with 4x4 board for testing."""
        puzzle = SudokuPuzzle.from_board(sample_4x4_board)
        return QSudoku(puzzle=puzzle, cache_base=temp_cache_dir)

    def test_init_with_puzzle(self, sample_4x4_board, temp_cache_dir):
        """Test QSudoku initialization with a SudokuPuzzle instance."""
        puzzle = SudokuPuzzle.from_board(sample_4x4_board)
        q_sudoku = QSudoku(puzzle=puzzle, cache_base=temp_cache_dir)
        
        assert q_sudoku.puzzle == puzzle
        assert q_sudoku.board == sample_4x4_board
        assert q_sudoku.board_size == 4
        assert q_sudoku.subgrid_size == 2
        assert q_sudoku.num_missing_cells == 10
        assert q_sudoku._solver is None
        assert len(q_sudoku._attached_backends) == 0

    def test_init_default_cache(self, sample_4x4_board):
        """Test QSudoku initialization with default cache directory."""
        puzzle = SudokuPuzzle.from_board(sample_4x4_board)
        q_sudoku = QSudoku(puzzle=puzzle)
        
        assert q_sudoku._metadata.cache_base == Path(".quantum_solver_cache")

    def test_generate_factory_method(self, temp_cache_dir):
        """Test QSudoku.generate() factory method."""
        q_sudoku = QSudoku.generate(
            subgrid_size=2, 
            num_missing_cells=8, 
            canonicalize=False,
            cache_base=temp_cache_dir
        )
        
        assert q_sudoku.board_size == 4
        assert q_sudoku.subgrid_size == 2
        assert q_sudoku.num_missing_cells == 8
        assert q_sudoku._solver is None

    def test_from_board_factory_method(self, sample_4x4_board, temp_cache_dir):
        """Test QSudoku.from_board() factory method."""
        q_sudoku = QSudoku.from_board(
            board=sample_4x4_board,
            canonicalize=False,
            cache_base=temp_cache_dir
        )
        
        assert q_sudoku.board == sample_4x4_board
        assert q_sudoku.board_size == 4
        assert q_sudoku.subgrid_size == 2
        assert q_sudoku.num_missing_cells == 10

    def test_properties(self, q_sudoku_4x4, sample_4x4_board):
        """Test QSudoku property accessors."""
        assert q_sudoku_4x4.board == sample_4x4_board
        assert q_sudoku_4x4.board_size == 4
        assert q_sudoku_4x4.subgrid_size == 2
        assert q_sudoku_4x4.num_missing_cells == 10

    @patch('sudoku_nisq.q_sudoku.gc.collect')
    def test_set_solver(self, mock_gc_collect, q_sudoku_4x4):
        """Test setting a quantum solver."""
        # Mock solver class
        mock_solver_class = Mock()
        mock_solver_instance = Mock()
        mock_solver_class.return_value = mock_solver_instance
        
        # Set solver
        result = q_sudoku_4x4.set_solver(
            solver_class=mock_solver_class,
            encoding="test_encoding",
            test_param="test_value"
        )
        
        # Verify solver was created correctly
        mock_solver_class.assert_called_once_with(
            puzzle=q_sudoku_4x4.puzzle,
            metadata_manager=q_sudoku_4x4._metadata,
            encoding="test_encoding",
            test_param="test_value"
        )
        
        assert q_sudoku_4x4._solver == mock_solver_instance
        assert result == mock_solver_instance

    @patch('sudoku_nisq.q_sudoku.gc.collect')
    def test_set_solver_replaces_existing(self, mock_gc_collect, q_sudoku_4x4):
        """Test that setting a new solver replaces the existing one."""
        # Set first solver
        mock_solver1 = Mock()
        q_sudoku_4x4._solver = mock_solver1
        
        # Set second solver
        mock_solver_class = Mock()
        mock_solver2 = Mock()
        mock_solver_class.return_value = mock_solver2
        
        q_sudoku_4x4.set_solver(mock_solver_class)
        
        # Verify old solver was cleaned up and new one set
        assert q_sudoku_4x4._solver == mock_solver2
        mock_gc_collect.assert_called()

    @patch('sudoku_nisq.q_sudoku.gc.collect')
    def test_drop_solver(self, mock_gc_collect, q_sudoku_4x4):
        """Test explicitly dropping the solver."""
        # Set a solver first
        q_sudoku_4x4._solver = Mock()
        
        # Drop the solver
        q_sudoku_4x4.drop_solver()
        
        assert q_sudoku_4x4._solver is None
        mock_gc_collect.assert_called_once()

    def test_drop_solver_when_none(self, q_sudoku_4x4):
        """Test dropping solver when none is set."""
        assert q_sudoku_4x4._solver is None
        # Should not raise an error
        q_sudoku_4x4.drop_solver()
        assert q_sudoku_4x4._solver is None

    def test_build_circuit_no_solver(self, q_sudoku_4x4):
        """Test building circuit without a solver raises error."""
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.build_circuit()

    def test_build_circuit_with_solver(self, q_sudoku_4x4):
        """Test building circuit with active solver."""
        mock_solver = Mock()
        mock_circuit = Mock()
        mock_solver.build_main_circuit.return_value = mock_circuit
        q_sudoku_4x4._solver = mock_solver
        
        result = q_sudoku_4x4.build_circuit()
        
        mock_solver.build_main_circuit.assert_called_once()
        assert result == mock_circuit

    def test_draw_circuit_no_solver(self, q_sudoku_4x4):
        """Test drawing circuit without a solver raises error."""
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.draw_circuit()

    def test_draw_circuit_with_default_circuit(self, q_sudoku_4x4):
        """Test drawing circuit using default circuit from solver."""
        mock_solver = Mock()
        mock_circuit = Mock()
        mock_solver.build_main_circuit.return_value = mock_circuit
        q_sudoku_4x4._solver = mock_solver
        
        q_sudoku_4x4.draw_circuit()
        
        mock_solver.build_main_circuit.assert_called_once()
        mock_solver.draw_circuit.assert_called_once_with(mock_circuit)

    def test_draw_circuit_with_provided_circuit(self, q_sudoku_4x4):
        """Test drawing circuit with explicitly provided circuit."""
        mock_solver = Mock()
        mock_circuit = Mock()
        q_sudoku_4x4._solver = mock_solver
        
        q_sudoku_4x4.draw_circuit(mock_circuit)
        
        mock_solver.build_main_circuit.assert_not_called()
        mock_solver.draw_circuit.assert_called_once_with(mock_circuit)

    @patch('sudoku_nisq.q_sudoku.BackendManager')
    def test_attach_backend(self, mock_backend_manager, q_sudoku_4x4):
        """Test attaching a backend."""
        mock_backend = Mock()
        mock_backend_manager.get.return_value = mock_backend
        
        q_sudoku_4x4.attach_backend("test_backend")
        
        mock_backend_manager.get.assert_called_once_with("test_backend")
        assert q_sudoku_4x4._attached_backends["test_backend"] == mock_backend

    @patch('sudoku_nisq.q_sudoku.BackendManager')
    def test_attach_backend_not_found(self, mock_backend_manager, q_sudoku_4x4):
        """Test attaching a backend that doesn't exist."""
        mock_backend_manager.get.side_effect = ValueError("Backend not found")
        
        with pytest.raises(ValueError, match="Backend not found"):
            q_sudoku_4x4.attach_backend("nonexistent_backend")

    @patch('sudoku_nisq.q_sudoku.BackendManager')
    def test_init_ibm(self, mock_backend_manager, q_sudoku_4x4):
        """Test initializing IBM backend."""
        mock_backend_manager.init_ibm.return_value = "ibm_test"
        mock_backend = Mock()
        mock_backend_manager.get.return_value = mock_backend
        
        result = q_sudoku_4x4.init_ibm("token", "instance", "device", "custom_alias")
        
        mock_backend_manager.init_ibm.assert_called_once_with(
            "token", "instance", "device", "custom_alias"
        )
        assert result == "ibm_test"
        assert q_sudoku_4x4._attached_backends["ibm_test"] == mock_backend

    @patch('sudoku_nisq.q_sudoku.BackendManager')
    def test_init_quantinuum(self, mock_backend_manager, q_sudoku_4x4):
        """Test initializing Quantinuum backend."""
        mock_backend_manager.init_quantinuum.return_value = "quantinuum_test"
        mock_backend = Mock()
        mock_backend_manager.get.return_value = mock_backend
        
        result = q_sudoku_4x4.init_quantinuum("device", "custom_alias")
        
        mock_backend_manager.init_quantinuum.assert_called_once_with(
            "device", "custom_alias", None, None
        )
        assert result == "quantinuum_test"
        assert q_sudoku_4x4._attached_backends["quantinuum_test"] == mock_backend

    def test_transpile_no_backend(self, q_sudoku_4x4):
        """Test transpiling without attached backend raises error."""
        with pytest.raises(ValueError, match="Backend 'test_backend' not attached"):
            q_sudoku_4x4.transpile("test_backend", 1)

    def test_transpile_no_solver(self, q_sudoku_4x4):
        """Test transpiling without solver raises error."""
        q_sudoku_4x4._attached_backends["test_backend"] = Mock()
        
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.transpile("test_backend", 1)

    def test_transpile_success(self, q_sudoku_4x4):
        """Test successful transpilation."""
        mock_backend = Mock()
        mock_solver = Mock()
        mock_result = Mock()
        
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        q_sudoku_4x4._solver = mock_solver
        mock_solver.transpile_and_analyze.return_value = mock_result
        
        result = q_sudoku_4x4.transpile("test_backend", 2, test_param="value")
        
        mock_solver.transpile_and_analyze.assert_called_once_with(
            mock_backend, "test_backend", 2, test_param="value"
        )
        assert result == mock_result

    def test_run_no_backend(self, q_sudoku_4x4):
        """Test running without attached backend raises error."""
        with pytest.raises(ValueError, match="Backend 'test_backend' not attached"):
            q_sudoku_4x4.run("test_backend", 1, 100)

    def test_run_no_solver(self, q_sudoku_4x4):
        """Test running without solver raises error."""
        q_sudoku_4x4._attached_backends["test_backend"] = Mock()
        
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.run("test_backend", 1, 100)

    def test_run_success(self, q_sudoku_4x4):
        """Test successful quantum execution."""
        mock_backend = Mock()
        mock_solver = Mock()
        mock_result = Mock()
        
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        q_sudoku_4x4._solver = mock_solver
        mock_solver.run.return_value = mock_result
        
        result = q_sudoku_4x4.run("test_backend", 2, 1000, test_param="value")
        
        mock_solver.run.assert_called_once_with(
            mock_backend, "test_backend", 1000, 
            force_run=False, optimisation_level=2, test_param="value"
        )
        assert result == mock_result

    def test_run_aer_no_solver(self, q_sudoku_4x4):
        """Test running on Aer without solver raises error."""
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.run_aer()

    def test_run_aer_success(self, q_sudoku_4x4):
        """Test successful Aer simulation."""
        mock_solver = Mock()
        mock_result = Mock()
        
        q_sudoku_4x4._solver = mock_solver
        mock_solver.run_aer.return_value = mock_result
        
        result = q_sudoku_4x4.run_aer(shots=2048, test_param="value")
        
        mock_solver.run_aer.assert_called_once_with(2048, test_param="value")
        assert result == mock_result

    def test_run_aer_default_shots(self, q_sudoku_4x4):
        """Test Aer simulation with default shots."""
        mock_solver = Mock()
        q_sudoku_4x4._solver = mock_solver
        
        q_sudoku_4x4.run_aer()
        
        mock_solver.run_aer.assert_called_once_with(1024)

    def test_counts_plot_no_solver(self, q_sudoku_4x4):
        """Test counts plot without solver raises error."""
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.counts_plot()

    def test_counts_plot_success(self, q_sudoku_4x4):
        """Test successful counts plot generation."""
        mock_solver = Mock()
        mock_counts = {"000": 50, "111": 50}
        
        q_sudoku_4x4._solver = mock_solver
        
        q_sudoku_4x4.counts_plot(
            counts=mock_counts,
            backend_alias="Test Backend",
            shots=100,
            top_n=10,
            show_valid_only=True,
            figsize=(10, 8),
            show_summary=False
        )
        
        mock_solver.counts_plot.assert_called_once_with(
            counts=mock_counts,
            backend_alias="Test Backend",
            shots=100,
            top_n=10,
            show_valid_only=True,
            figsize=(10, 8),
            show_summary=False
        )

    def test_counts_plot_default_backend_alias(self, q_sudoku_4x4):
        """Test counts plot with default backend alias."""
        mock_solver = Mock()
        q_sudoku_4x4._solver = mock_solver
        
        q_sudoku_4x4.counts_plot()
        
        # Check that default backend alias was used
        call_args = mock_solver.counts_plot.call_args
        assert call_args[1]["backend_alias"] == "Unknown Backend"

    def test_report_resources(self, q_sudoku_4x4):
        """Test resource reporting."""
        mock_resource_summary = {"gates": 100, "depth": 50}
        
        # Mock the entire metadata manager temporarily
        original_metadata = q_sudoku_4x4._metadata
        mock_metadata = Mock()
        mock_metadata.get_resource_summary.return_value = mock_resource_summary
        q_sudoku_4x4._metadata = mock_metadata
        
        try:
            result = q_sudoku_4x4.report_resources()
            
            mock_metadata.get_resource_summary.assert_called_once()
            assert result == mock_resource_summary
        finally:
            # Restore original metadata
            q_sudoku_4x4._metadata = original_metadata

    def test_get_hash(self, q_sudoku_4x4):
        """Test getting puzzle hash."""
        expected_hash = "test_hash_value"
        
        # Mock the entire puzzle temporarily
        original_puzzle = q_sudoku_4x4.puzzle
        mock_puzzle = Mock()
        mock_puzzle.get_hash.return_value = expected_hash
        q_sudoku_4x4.puzzle = mock_puzzle
        
        try:
            result = q_sudoku_4x4.get_hash()
            
            mock_puzzle.get_hash.assert_called_once()
            assert result == expected_hash
        finally:
            # Restore original puzzle
            q_sudoku_4x4.puzzle = original_puzzle

    def test_multiple_backends_attachment(self, q_sudoku_4x4):
        """Test attaching multiple backends."""
        with patch('sudoku_nisq.q_sudoku.BackendManager') as mock_backend_manager:
            mock_backend1 = Mock()
            mock_backend2 = Mock()
            mock_backend_manager.get.side_effect = [mock_backend1, mock_backend2]
            
            q_sudoku_4x4.attach_backend("backend1")
            q_sudoku_4x4.attach_backend("backend2")
            
            assert q_sudoku_4x4._attached_backends["backend1"] == mock_backend1
            assert q_sudoku_4x4._attached_backends["backend2"] == mock_backend2
            assert len(q_sudoku_4x4._attached_backends) == 2

    def test_solver_workflow_integration(self, q_sudoku_4x4):
        """Test complete workflow: set solver, build circuit, run simulation."""
        # Mock solver
        mock_solver_class = Mock()
        mock_solver = Mock()
        mock_circuit = Mock()
        mock_result = Mock()
        
        mock_solver_class.return_value = mock_solver
        mock_solver.build_main_circuit.return_value = mock_circuit
        mock_solver.run_aer.return_value = mock_result
        
        # Execute workflow
        q_sudoku_4x4.set_solver(mock_solver_class, encoding="test")
        circuit = q_sudoku_4x4.build_circuit()
        result = q_sudoku_4x4.run_aer(shots=500)
        
        # Verify complete workflow
        assert circuit == mock_circuit
        assert result == mock_result
        mock_solver.build_main_circuit.assert_called_once()
        mock_solver.run_aer.assert_called_once_with(500)
