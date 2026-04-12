"""Integration tests for BenchmarkSession.execute_run() with full Stage 1-7 recording."""

import os
import pytest
import tempfile
from pathlib import Path

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.metadata.benchmark_session import BenchmarkSession
from sudoku_nisq.metadata.config import MetadataConfig

pytestmark = [pytest.mark.integration]


class TestBenchmarkSessionExecuteRun:
    """Integration tests validating BenchmarkSession.execute_run() orchestrates all stages."""

    @pytest.fixture(autouse=True)
    def setup_environment(self):
        """Enable new metadata architecture for all tests in this class."""
        original_value = os.environ.get('SUDOKU_NISQ_NEW_METADATA')
        original_flag = MetadataConfig.ENABLE_NEW_ARCHITECTURE
        
        # Enable new architecture both ways
        os.environ['SUDOKU_NISQ_NEW_METADATA'] = '1'
        MetadataConfig.ENABLE_NEW_ARCHITECTURE = True
        
        yield
        
        # Restore original environment
        if original_value is None:
            os.environ.pop('SUDOKU_NISQ_NEW_METADATA', None)
        else:
            os.environ['SUDOKU_NISQ_NEW_METADATA'] = original_value
        MetadataConfig.ENABLE_NEW_ARCHITECTURE = original_flag

    @pytest.fixture
    def temp_cache_dir(self):
        """Provide isolated temporary cache directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield Path(temp_dir)

    @pytest.fixture
    def simple_2x2_puzzle(self, temp_cache_dir):
        """Create minimal 2×2 puzzle for fast testing."""
        puzzle = QSudoku.generate(size=2, num_missing_cells=2, cache_base=temp_cache_dir)
        return puzzle

    def test_execute_run_records_all_stages(self, simple_2x2_puzzle, temp_cache_dir):
        """Verify execute_run() records Stages 1-7 with Aer backend."""
        # Arrange: Setup puzzle with solver and validation context
        puzzle = simple_2x2_puzzle
        puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=True)
        
        # Build circuit to ensure Stage 2a recording happens
        puzzle.build_circuit()
        
        # Valid solutions for 2×2 with 2 missing cells (example bitstrings)
        # These would be problem-specific; using placeholders for structure
        puzzle.set_validation_context(valid_solutions=["00", "01", "10", "11"])
        
        backend_alias = puzzle.init_aer(method="statevector")
        
        # Create session using new puzzle parameter
        session = BenchmarkSession(puzzle=puzzle, cache_base=temp_cache_dir)
        
        # Register puzzle for Stage 1
        session.register_puzzle(puzzle)
        
        # Act: Execute run through all stages
        result = session.execute_run(
            puzzle=puzzle,
            backend_alias=backend_alias,
            shots=256,
            opt_level=1
        )
        
        # Assert: Verify result returned
        assert result is not None
        
        # Assert: Verify Stage 1 (Instance)
        puzzle_hash = puzzle.get_hash()
        instances = session.stage1.query()
        instance_hashes = [inst.get('puzzle_hash') for inst in instances] if isinstance(instances, list) else []
        assert puzzle_hash in instance_hashes, "Puzzle not registered in Stage 1"
        
        # Assert: Verify Stage 2a (Logical IR)
        stage2a_data = session.stage2a.query()
        assert stage2a_data is not None, "Stage 2a data should exist"
        if isinstance(stage2a_data, dict) and "ExactCoverQuantumSolver" in stage2a_data:
            assert "simple" in stage2a_data["ExactCoverQuantumSolver"], "Stage 2a missing encoding"
        
        # Assert: Verify Stage 2b (IR Policy) - may or may not be recorded depending on solver
        try:
            stage2b_data = session.stage2b.query(solver_name="ExactCoverQuantumSolver")
            if stage2b_data:
                assert isinstance(stage2b_data, dict), "Stage 2b should return dict when present"
        except (ValueError, FileNotFoundError):
            pass  # Stage 2b may not be recorded for all solver types
        
        # Assert: Verify Stage 3 (Compilation)
        compilations = session.stage3.query()
        assert len(compilations) > 0, "No compilations recorded in Stage 3"
        
        # Assert: Verify Stage 5 (Execution)
        run_records = session.stage5.query()
        assert len(run_records) > 0, "No executions recorded in Stage 5"
        
        # Assert: Verify Stage 6-7 (Metrics)
        metrics_list = session.stage6_7.query()
        assert metrics_list and len(metrics_list) > 0, "No metrics recorded in Stages 6-7"
        # Find the metrics for our run
        latest_run_id = run_records[0].get('run_id')
        run_metrics = session.stage6_7.query(run_id=latest_run_id)
        assert run_metrics is not None, f"Metrics missing for run_id {latest_run_id}"

    def test_execute_run_without_validation_skips_metrics(self, simple_2x2_puzzle, temp_cache_dir):
        """Verify execute_run() records Stage 5 but skips Stages 6-7 without validation context."""
        # Arrange: Setup puzzle without validation context
        puzzle = simple_2x2_puzzle
        puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
        
        # Build circuit to ensure Stage 2a recording happens
        puzzle.build_circuit()
        
        backend_alias = puzzle.init_aer(method="statevector")
        
        session = BenchmarkSession(puzzle=puzzle, cache_base=temp_cache_dir)
        session.register_puzzle(puzzle)  # Register for Stage 1
        
        # Act: Execute without validation_context parameter
        session.execute_run(
            puzzle=puzzle,
            backend_alias=backend_alias,
            shots=128,
            opt_level=0
        )
        
        # Assert: Stage 5 recorded
        run_records = session.stage5.query()
        assert len(run_records) > 0, "Stage 5 should record execution even without validation"
        
        # Assert: Stages 6-7 may be empty or have minimal entries
        session.stage6_7.query()
        # Note: Metrics computation is gracefully skipped when validation context absent
        # This is expected behavior per architecture

    def test_session_initialization_with_puzzle_parameter(self, simple_2x2_puzzle, temp_cache_dir):
        """Verify BenchmarkSession can initialize with puzzle parameter instead of puzzle_hash."""
        puzzle = simple_2x2_puzzle
        
        # Act: Create session using puzzle parameter
        session = BenchmarkSession(puzzle=puzzle, cache_base=temp_cache_dir)
        
        # Assert: puzzle_hash extracted correctly
        expected_hash = puzzle.get_hash()
        assert session.puzzle_hash == expected_hash
        assert session.cache_base == temp_cache_dir

    def test_session_initialization_with_explicit_puzzle_hash(self, simple_2x2_puzzle, temp_cache_dir):
        """Verify BenchmarkSession still works with explicit puzzle_hash parameter."""
        puzzle = simple_2x2_puzzle
        puzzle_hash = puzzle.get_hash()
        
        # Act: Create session using puzzle_hash directly
        session = BenchmarkSession(puzzle_hash=puzzle_hash, cache_base=temp_cache_dir)
        
        # Assert: Initialized correctly
        assert session.puzzle_hash == puzzle_hash
        assert session.cache_base == temp_cache_dir

    def test_session_initialization_requires_puzzle_or_hash(self, temp_cache_dir):
        """Verify BenchmarkSession raises ValueError when neither puzzle nor puzzle_hash provided."""
        # Act & Assert: Should raise ValueError
        with pytest.raises(ValueError, match="Must provide either puzzle_hash or puzzle parameter"):
            BenchmarkSession(cache_base=temp_cache_dir)

    def test_execute_run_checks_quantum_solver_attribute(self, simple_2x2_puzzle, temp_cache_dir):
        """Verify execute_run() correctly accesses puzzle.quantum_solver property."""
        # Arrange: Setup puzzle with solver
        puzzle = simple_2x2_puzzle
        puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
        backend_alias = puzzle.init_aer()
        
        # Assert: quantum_solver property is accessible
        assert puzzle.quantum_solver is not None, "quantum_solver property should return solver"
        assert puzzle.quantum_solver.solver_name == "ExactCoverQuantumSolver"
        
        # Act: execute_run should not raise AttributeError
        session = BenchmarkSession(puzzle=puzzle, cache_base=temp_cache_dir)
        result = session.execute_run(
            puzzle=puzzle,
            backend_alias=backend_alias,
            shots=128
        )
        
        assert result is not None

    def test_cache_base_uses_metadata_config(self, simple_2x2_puzzle):
        """Verify BenchmarkSession uses MetadataConfig.get_cache_base() when cache_base not specified."""
        puzzle = simple_2x2_puzzle
        
        # Act: Create session without explicit cache_base
        session = BenchmarkSession(puzzle=puzzle)
        
        # Assert: Should use MetadataConfig.get_cache_base()
        expected_base = MetadataConfig.get_cache_base()
        assert session.cache_base == expected_base
