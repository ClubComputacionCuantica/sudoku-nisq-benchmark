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
        """Test QSudoku.generate() factory method with subgrid_size."""
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

    def test_generate_with_size_parameter(self, temp_cache_dir):
        """Test QSudoku.generate() factory method with size parameter."""
        # Test with size=4 (2x2 subgrids)
        q_sudoku = QSudoku.generate(
            size=4,
            num_missing_cells=8,
            canonicalize=False,
            cache_base=temp_cache_dir
        )
        
        assert q_sudoku.board_size == 4
        assert q_sudoku.subgrid_size == 2
        assert q_sudoku.num_missing_cells == 8

    def test_generate_with_size_9(self, temp_cache_dir):
        """Test QSudoku.generate() factory method with size=9."""
        q_sudoku = QSudoku.generate(
            size=9,
            num_missing_cells=20,
            cache_base=temp_cache_dir
        )
        
        assert q_sudoku.board_size == 9
        assert q_sudoku.subgrid_size == 3
        assert q_sudoku.num_missing_cells == 20

    def test_generate_with_size_2_special_case(self, temp_cache_dir):
        """Test QSudoku.generate() with size=2 special case (no real subgrids)."""
        q_sudoku = QSudoku.generate(
            size=2,
            num_missing_cells=1,
            cache_base=temp_cache_dir
        )
        
        assert q_sudoku.board_size == 2
        assert q_sudoku.subgrid_size == 1

    def test_generate_with_invalid_size(self, temp_cache_dir):
        """Test QSudoku.generate() with invalid size raises error."""
        with pytest.raises(ValueError, match="size must be 2 or a perfect square"):
            QSudoku.generate(size=5, num_missing_cells=10, cache_base=temp_cache_dir)

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
        
        mock_solver.build_main_circuit.assert_called_once_with(sdk=None)
        assert result == mock_circuit

    def test_build_circuit_with_sdk_parameter(self, q_sudoku_4x4):
        """Test building circuit with explicit SDK selection."""
        mock_solver = Mock()
        mock_circuit = Mock()
        mock_solver.build_main_circuit.return_value = mock_circuit
        q_sudoku_4x4._solver = mock_solver
        
        result = q_sudoku_4x4.build_circuit(sdk="qiskit")
        
        mock_solver.build_main_circuit.assert_called_once_with(sdk="qiskit")
        assert result == mock_circuit

    def test_build_circuit_pytket_sdk(self, q_sudoku_4x4):
        """Test building circuit with pytket SDK."""
        mock_solver = Mock()
        mock_circuit = Mock()
        mock_solver.build_main_circuit.return_value = mock_circuit
        q_sudoku_4x4._solver = mock_solver
        
        result = q_sudoku_4x4.build_circuit(sdk="pytket")
        
        mock_solver.build_main_circuit.assert_called_once_with(sdk="pytket")
        assert result == mock_circuit

    def test_build_circuit_braket_sdk(self, q_sudoku_4x4):
        """Test building circuit with braket SDK."""
        mock_solver = Mock()
        mock_circuit = Mock()
        mock_solver.build_main_circuit.return_value = mock_circuit
        q_sudoku_4x4._solver = mock_solver
        
        result = q_sudoku_4x4.build_circuit(sdk="braket")
        
        mock_solver.build_main_circuit.assert_called_once_with(sdk="braket")
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

    @patch('sudoku_nisq.q_sudoku.BackendManager.inst')
    def test_attach_backend(self, mock_inst, q_sudoku_4x4):
        """Test attaching a backend."""
        mock_manager = Mock()
        mock_backend = Mock()
        mock_manager.get.return_value = mock_backend
        mock_inst.return_value = mock_manager

        q_sudoku_4x4.attach_backend("test_backend")

        mock_manager.get.assert_called_once_with("test_backend")
        assert q_sudoku_4x4._attached_backends["test_backend"] == mock_backend

    @patch('sudoku_nisq.q_sudoku.BackendManager.inst')
    def test_attach_backend_not_found(self, mock_inst, q_sudoku_4x4):
        """Test attaching a backend that doesn't exist."""
        mock_manager = Mock()
        mock_manager.get.side_effect = ValueError("Backend not found")
        mock_inst.return_value = mock_manager

        with pytest.raises(ValueError, match="Backend not found"):
            q_sudoku_4x4.attach_backend("nonexistent_backend")

    @patch('sudoku_nisq.q_sudoku.BackendManager.inst')
    def test_init_ibm(self, mock_inst, q_sudoku_4x4):
        """Test initializing IBM backend."""
        mock_manager = Mock()
        mock_manager.init_ibm.return_value = "ibm_test"
        mock_backend = Mock()
        mock_manager.get.return_value = mock_backend
        mock_inst.return_value = mock_manager

        result = q_sudoku_4x4.init_ibm("token", "instance", "device", "custom_alias")

        mock_manager.init_ibm.assert_called_once_with(device="device", alias="custom_alias", api_token="token", instance="instance")
        assert result == "ibm_test"
        assert q_sudoku_4x4._attached_backends["ibm_test"] == mock_backend

    @patch('sudoku_nisq.q_sudoku.BackendManager.inst')
    def test_init_quantinuum(self, mock_inst, q_sudoku_4x4):
        """Test initializing Quantinuum backend."""
        mock_manager = Mock()
        mock_manager.init_quantinuum.return_value = "quantinuum_test"
        mock_backend = Mock()
        mock_manager.get.return_value = mock_backend
        mock_inst.return_value = mock_manager

        result = q_sudoku_4x4.init_quantinuum("device", "custom_alias")

        mock_manager.init_quantinuum.assert_called_once_with(device="device", alias="custom_alias", token_store=None, provider=None)
        assert result == "quantinuum_test"
        assert q_sudoku_4x4._attached_backends["quantinuum_test"] == mock_backend

    @patch('sudoku_nisq.q_sudoku.BackendManager.inst')
    def test_init_aer_default(self, mock_inst, q_sudoku_4x4):
        """Test initializing Aer backend with defaults."""
        mock_manager = Mock()
        mock_manager.init_aer.return_value = "aer_automatic"
        mock_backend = Mock()
        mock_manager.get.return_value = mock_backend
        mock_inst.return_value = mock_manager

        result = q_sudoku_4x4.init_aer()

        mock_manager.init_aer.assert_called_once()
        assert result == "aer_automatic"
        assert q_sudoku_4x4._attached_backends["aer_automatic"] == mock_backend

    @patch('sudoku_nisq.q_sudoku.BackendManager.inst')
    def test_init_aer_with_options(self, mock_inst, q_sudoku_4x4):
        """Test initializing Aer backend with custom options."""
        mock_manager = Mock()
        mock_manager.init_aer.return_value = "aer_custom"
        mock_backend = Mock()
        mock_manager.get.return_value = mock_backend
        mock_inst.return_value = mock_manager

        result = q_sudoku_4x4.init_aer(
            method="statevector",
            device="GPU",
            precision="single",
            alias="custom_aer"
        )

        mock_manager.init_aer.assert_called_once()
        call_kwargs = mock_manager.init_aer.call_args[1]
        assert call_kwargs["device"] == "statevector"
        assert call_kwargs["method"] == "statevector"
        assert call_kwargs["device_type"] == "GPU"
        assert call_kwargs["precision"] == "single"
        assert call_kwargs["alias"] == "custom_aer"
        assert result == "aer_custom"
        assert q_sudoku_4x4._attached_backends["aer_custom"] == mock_backend

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
        from unittest.mock import Mock, patch
        
        mock_backend = Mock()
        mock_solver = Mock()
        mock_result = {"status": "success", "backend_alias": "test_backend"}
        mock_transpiled_circuit = Mock()
        
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        q_sudoku_4x4._solver = mock_solver
        mock_solver.transpile_and_analyze.return_value = mock_result
        
        # Mock get_transpiled_circuit to return the expected circuit
        with patch.object(q_sudoku_4x4, 'get_transpiled_circuit', return_value=mock_transpiled_circuit):
            result = q_sudoku_4x4.transpile("test_backend", 2, test_param="value")
        
        mock_solver.transpile_and_analyze.assert_called_once_with(
            mock_backend, "test_backend", 2, test_param="value"
        )
        assert result == mock_transpiled_circuit

    def test_transpile_error_handling(self, q_sudoku_4x4):
        """Test transpilation error handling."""
        mock_backend = Mock()
        mock_solver = Mock()
        mock_result = {"error": "Transpilation failed due to invalid circuit"}
        
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        q_sudoku_4x4._solver = mock_solver
        mock_solver.transpile_and_analyze.return_value = mock_result
        
        with pytest.raises(RuntimeError, match="Transpilation failed"):
            q_sudoku_4x4.transpile("test_backend", 2)

    def test_get_transpiled_circuit_no_solver(self, q_sudoku_4x4):
        """Test get_transpiled_circuit without solver raises error."""
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.get_transpiled_circuit("test_backend", 1)

    def test_get_transpiled_circuit_not_found(self, q_sudoku_4x4):
        """Test get_transpiled_circuit when circuit doesn't exist."""
        mock_solver = Mock()
        mock_backend = Mock()
        mock_path = Mock()
        mock_path.exists.return_value = False
        
        q_sudoku_4x4._solver = mock_solver
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        mock_solver._detect_backend_sdk.return_value = "pytket"
        mock_solver.transpiled_circuit_path.return_value = mock_path
        
        with pytest.raises(FileNotFoundError, match="No transpiled circuit found"):
            q_sudoku_4x4.get_transpiled_circuit("test_backend", 1)

    def test_get_transpiled_circuit_success_pytket(self, q_sudoku_4x4):
        """Test successful get_transpiled_circuit for pytket backend."""
        mock_solver = Mock()
        mock_backend = Mock()
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_circuit = Mock()
        
        q_sudoku_4x4._solver = mock_solver
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        mock_solver._detect_backend_sdk.return_value = "pytket"
        mock_solver.transpiled_circuit_path.return_value = mock_path
        mock_solver.load_circuit.return_value = mock_circuit
        
        result = q_sudoku_4x4.get_transpiled_circuit("test_backend", 1)
        
        mock_solver.load_circuit.assert_called_once_with(mock_path)
        assert result == mock_circuit

    def test_get_transpiled_circuit_success_qiskit(self, q_sudoku_4x4):
        """Test successful get_transpiled_circuit for qiskit backend."""
        mock_solver = Mock()
        mock_backend = Mock()
        mock_path = Mock()
        mock_path.exists.return_value = True
        mock_circuit = Mock()
        
        q_sudoku_4x4._solver = mock_solver
        q_sudoku_4x4._attached_backends["test_backend"] = mock_backend
        mock_solver._detect_backend_sdk.return_value = "qiskit"
        mock_solver.transpiled_circuit_path.return_value = mock_path
        mock_solver._load_qiskit_circuit.return_value = mock_circuit
        
        result = q_sudoku_4x4.get_transpiled_circuit("test_backend", 1)
        
        mock_solver._load_qiskit_circuit.assert_called_once_with(mock_path)
        assert result == mock_circuit

    def test_run_no_backend(self, q_sudoku_4x4):
        """Test running without attached backend raises error."""
        # Provide a mock solver so backend lookup is reached
        mock_solver = Mock()
        q_sudoku_4x4._solver = mock_solver
        with pytest.raises(ValueError, match=r"Backend 'test_backend' not found."):
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

    def test_run_aer_with_noise_no_solver(self, q_sudoku_4x4):
        """Test run_aer_with_noise without solver raises error."""
        with pytest.raises(ValueError, match="No solver set. Call set_solver"):
            q_sudoku_4x4.run_aer_with_noise()

    def test_run_aer_with_noise_custom_noise_model(self, q_sudoku_4x4):
        """Test run_aer_with_noise with custom noise model."""
        mock_solver = Mock()
        mock_result = {"counts": {"00": 500, "11": 500}}
        mock_solver.run_aer.return_value = mock_result
        q_sudoku_4x4._solver = mock_solver
        
        mock_noise_model = Mock()
        result = q_sudoku_4x4.run_aer_with_noise(
            shots=1000,
            noise_model=mock_noise_model,
            method="density_matrix",
            optimization_level=2
        )
        
        mock_solver.run_aer.assert_called_once()
        call_kwargs = mock_solver.run_aer.call_args[1]
        assert call_kwargs["shots"] == 1000
        assert call_kwargs["noise_model"] == mock_noise_model
        assert call_kwargs["method"] == "density_matrix"
        assert call_kwargs["optimization_level"] == 2
        assert result == mock_result

    def test_run_aer_with_noise_no_model_or_device(self, q_sudoku_4x4):
        """Test run_aer_with_noise without noise model or device name raises error."""
        mock_solver = Mock()
        q_sudoku_4x4._solver = mock_solver
        
        with pytest.raises(ValueError, match="Either noise_model or device_name must be provided"):
            q_sudoku_4x4.run_aer_with_noise(shots=1024)

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
        with patch('sudoku_nisq.q_sudoku.BackendManager.inst') as mock_inst:
            mock_manager = Mock()
            mock_backend1 = Mock()
            mock_backend2 = Mock()
            mock_manager.get.side_effect = [mock_backend1, mock_backend2]
            mock_inst.return_value = mock_manager

            q_sudoku_4x4.attach_backend("backend1")
            q_sudoku_4x4.attach_backend("backend2")

            assert q_sudoku_4x4._attached_backends["backend1"] == mock_backend1
            assert q_sudoku_4x4._attached_backends["backend2"] == mock_backend2
            assert len(q_sudoku_4x4._attached_backends) == 2

    def test_plot_puzzle(self, q_sudoku_4x4):
        """Test plot_puzzle method delegates to puzzle.plot()."""
        mock_figure = Mock()
        
        # Mock the puzzle's plot method
        original_puzzle = q_sudoku_4x4.puzzle
        mock_puzzle = Mock()
        mock_puzzle.plot.return_value = mock_figure
        q_sudoku_4x4.puzzle = mock_puzzle
        
        try:
            result = q_sudoku_4x4.plot_puzzle()
            
            mock_puzzle.plot.assert_called_once()
            assert result == mock_figure
        finally:
            # Restore original puzzle
            q_sudoku_4x4.puzzle = original_puzzle

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

    def test_run_with_global_backend_manager(self, q_sudoku_4x4):
        """Test run() method fetches backend from global BackendManager."""
        with patch('sudoku_nisq.q_sudoku.BackendManager.inst') as mock_inst:
            mock_manager = Mock()
            mock_backend = Mock()
            mock_solver = Mock()
            mock_result = {"counts": {"00": 100}}
            
            mock_manager.get.return_value = mock_backend
            mock_inst.return_value = mock_manager
            mock_solver.run.return_value = mock_result
            
            q_sudoku_4x4._solver = mock_solver
            
            result = q_sudoku_4x4.run("test_backend", opt_level=1, shots=100)
            
            mock_manager.get.assert_called_once_with("test_backend")
            mock_solver.run.assert_called_once()
            assert result == mock_result
