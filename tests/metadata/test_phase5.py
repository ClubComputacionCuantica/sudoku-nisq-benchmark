"""Phase 5 tests: InstanceMetadataManager, IRPolicyMetadataManager, and BenchmarkSession.

Tests cover:
- Instance registration and querying (Stage 1)
- IR policy recording and querying (Stage 2b)
- BenchmarkSession orchestration
- Cross-stage integration
"""

import pytest
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import Mock, patch

from sudoku_nisq.metadata import (
    InstanceMetadataManager,
    IRPolicyMetadataManager
)
from sudoku_nisq.metadata.benchmark_session import BenchmarkSession


# ============================================================================
# Stage 1: InstanceMetadataManager Tests
# ============================================================================

class TestInstanceMetadataManager:
    """Test Stage 1: Instance Selection metadata manager."""
    
    def test_record_basic_instance(self, tmp_path):
        """Test basic instance registration."""
        manager = InstanceMetadataManager(tmp_path)
        
        puzzle_hash = manager.record(
            puzzle_hash="abc123",
            size=4,
            subgrid_size=2,
            num_missing_cells=2,
            board=[(0, 0, 1), (0, 1, 2)],
            open_tuples=[(0, 2), (1, 3)],
            pre_tuples=[(0, 0), (0, 1)],
            solution_count=1
        )
        
        assert puzzle_hash == "abc123"
        
        # Verify file created
        registry_path = tmp_path / "instances" / "registry.json"
        assert registry_path.exists()
        
        # Verify content
        with registry_path.open() as f:
            data = json.load(f)
        
        assert "abc123" in data
        assert data["abc123"]["size"] == 4
        assert data["abc123"]["subgrid_size"] == 2
        assert data["abc123"]["num_missing_cells"] == 2
        assert data["abc123"]["solution_count"] == 1
    
    def test_record_duplicate_updates_timestamp(self, tmp_path):
        """Test re-registering same puzzle updates last_accessed."""
        manager = InstanceMetadataManager(tmp_path)
        
        # First registration
        manager.record(puzzle_hash="abc123", size=4)
        
        registry_path = tmp_path / "instances" / "registry.json"
        with registry_path.open() as f:
            data1 = json.load(f)
        first_accessed = data1["abc123"]["last_accessed"]
        
        import time
        time.sleep(0.1)  # Ensure timestamp difference
        
        # Second registration
        manager.record(puzzle_hash="abc123", size=4)
        
        with registry_path.open() as f:
            data2 = json.load(f)
        second_accessed = data2["abc123"]["last_accessed"]
        
        assert second_accessed > first_accessed
        assert data2["abc123"]["size"] == 4  # Original data preserved
    
    def test_query_by_puzzle_hash(self, tmp_path):
        """Test exact lookup by puzzle_hash."""
        manager = InstanceMetadataManager(tmp_path)
        
        manager.record(puzzle_hash="abc123", size=4, num_missing_cells=2)
        manager.record(puzzle_hash="def456", size=9, num_missing_cells=20)
        
        result = manager.query(puzzle_hash="abc123")
        
        assert result is not None
        assert result["puzzle_hash"] == "abc123"
        assert result["size"] == 4
    
    def test_query_by_size(self, tmp_path):
        """Test filtering by board size."""
        manager = InstanceMetadataManager(tmp_path)
        
        manager.record(puzzle_hash="puzzle_4x4_a", size=4, num_missing_cells=2)
        manager.record(puzzle_hash="puzzle_4x4_b", size=4, num_missing_cells=3)
        manager.record(puzzle_hash="puzzle_9x9", size=9, num_missing_cells=20)
        
        results = manager.query(size=4)
        
        assert len(results) == 2
        assert all(r["size"] == 4 for r in results)
    
    def test_query_by_difficulty_range(self, tmp_path):
        """Test filtering by min/max missing cells."""
        manager = InstanceMetadataManager(tmp_path)
        
        manager.record(puzzle_hash="easy", size=4, num_missing_cells=2)
        manager.record(puzzle_hash="medium", size=4, num_missing_cells=5)
        manager.record(puzzle_hash="hard", size=4, num_missing_cells=8)
        
        # Query with min/max
        results = manager.query(min_missing_cells=3, max_missing_cells=6)
        
        assert len(results) == 1
        assert results[0]["puzzle_hash"] == "medium"
        assert results[0]["num_missing_cells"] == 5
    
    def test_query_by_date_range(self, tmp_path):
        """Test filtering by generation timestamp."""
        manager = InstanceMetadataManager(tmp_path)
        
        # Record with explicit timestamps
        now = datetime.now(timezone.utc)
        yesterday = now - timedelta(days=1)
        tomorrow = now + timedelta(days=1)
        
        manager.record(
            puzzle_hash="old",
            size=4,
            generation_timestamp=yesterday.isoformat()
        )
        manager.record(
            puzzle_hash="new",
            size=4,
            generation_timestamp=tomorrow.isoformat()
        )
        
        # Query date range
        results = manager.query(
            date_range=(now - timedelta(hours=1), now + timedelta(hours=1))
        )
        
        assert len(results) == 0  # No puzzles in this narrow range
        
        # Wider range
        results = manager.query(
            date_range=(yesterday - timedelta(hours=1), now + timedelta(days=2))
        )
        
        assert len(results) == 2
    
    def test_query_empty_registry(self, tmp_path):
        """Test querying empty registry."""
        manager = InstanceMetadataManager(tmp_path)
        
        # Exact lookup returns None
        assert manager.query(puzzle_hash="nonexistent") is None
        
        # Filter query returns empty list
        assert manager.query(size=4) == []
    
    def test_query_nonexistent_puzzle_hash(self, tmp_path):
        """Test querying nonexistent puzzle_hash."""
        manager = InstanceMetadataManager(tmp_path)
        
        manager.record(puzzle_hash="exists", size=4)
        
        result = manager.query(puzzle_hash="nonexistent")
        assert result is None
    
    def test_record_missing_puzzle_hash_raises(self, tmp_path):
        """Test recording without puzzle_hash raises ValueError."""
        manager = InstanceMetadataManager(tmp_path)
        
        with pytest.raises(ValueError, match="puzzle_hash is required"):
            manager.record(size=4)


# ============================================================================
# Stage 2b: IRPolicyMetadataManager Tests
# ============================================================================

class TestIRPolicyMetadataManager:
    """Test Stage 2b: IR Policy metadata manager."""
    
    def test_record_basic_policy(self, tmp_path):
        """Test basic IR policy recording."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="pytket",
            sdk_version="1.31.1",
            solver_options={"decompose_cnz": True, "track_memory": False}
        )
        
        # Verify file created
        policy_path = tmp_path / "puzzle_abc123" / "stage_2b_ir_policy.json"
        assert policy_path.exists()
        
        # Verify content
        with policy_path.open() as f:
            data = json.load(f)
        
        assert "ExactCoverQuantumSolver" in data
        assert "simple" in data["ExactCoverQuantumSolver"]
        
        policy = data["ExactCoverQuantumSolver"]["simple"]
        assert policy["sdk"] == "pytket"
        assert policy["sdk_version"] == "1.31.1"
        assert policy["solver_options"]["decompose_cnz"] is True
    
    def test_record_multiple_encodings(self, tmp_path):
        """Test recording multiple encodings for same solver."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        # Record simple encoding
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="pytket",
            solver_options={"decompose_cnz": True}
        )
        
        # Record pattern encoding
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="pattern",
            sdk="pytket",
            solver_options={"decompose_cnz": False}
        )
        
        # Query both
        simple_policy = manager.query(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple"
        )
        pattern_policy = manager.query(
            solver_name="ExactCoverQuantumSolver",
            encoding="pattern"
        )
        
        assert simple_policy["solver_options"]["decompose_cnz"] is True
        assert pattern_policy["solver_options"]["decompose_cnz"] is False
    
    def test_query_specific_policy(self, tmp_path):
        """Test querying specific solver/encoding policy."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="qiskit",
            sdk_version="1.0.0"
        )
        
        policy = manager.query(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple"
        )
        
        assert policy is not None
        assert policy["sdk"] == "qiskit"
        assert policy["sdk_version"] == "1.0.0"
    
    def test_query_all_encodings(self, tmp_path):
        """Test querying all encodings for a solver."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="pytket"
        )
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="pattern",
            sdk="qiskit"
        )
        
        # Query without encoding returns all
        policies = manager.query(solver_name="ExactCoverQuantumSolver")
        
        assert policies is not None
        assert "simple" in policies
        assert "pattern" in policies
        assert policies["simple"]["sdk"] == "pytket"
        assert policies["pattern"]["sdk"] == "qiskit"
    
    def test_query_nonexistent_solver(self, tmp_path):
        """Test querying nonexistent solver."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="pytket"
        )
        
        result = manager.query(solver_name="NonexistentSolver")
        assert result is None
    
    def test_query_nonexistent_encoding(self, tmp_path):
        """Test querying nonexistent encoding."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="pytket"
        )
        
        result = manager.query(
            solver_name="ExactCoverQuantumSolver",
            encoding="nonexistent"
        )
        assert result is None
    
    def test_record_missing_solver_name_raises(self, tmp_path):
        """Test recording without solver_name raises ValueError."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        with pytest.raises(ValueError, match="solver_name is required"):
            manager.record(encoding="simple", sdk="pytket")
    
    def test_record_missing_encoding_raises(self, tmp_path):
        """Test recording without encoding raises ValueError."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        with pytest.raises(ValueError, match="encoding is required"):
            manager.record(solver_name="ExactCoverQuantumSolver", sdk="pytket")
    
    def test_query_without_solver_name_raises(self, tmp_path):
        """Test querying without solver_name raises ValueError."""
        manager = IRPolicyMetadataManager(tmp_path, "puzzle_abc123")
        
        with pytest.raises(ValueError, match="solver_name is required"):
            manager.query(encoding="simple")


# ============================================================================
# BenchmarkSession Tests
# ============================================================================

class TestBenchmarkSession:
    """Test BenchmarkSession orchestration."""
    
    def test_initialization(self, tmp_path):
        """Test BenchmarkSession initializes all stage managers."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        assert session.puzzle_hash == "abc123"
        assert session.cache_base == tmp_path
        
        # Verify all stage managers initialized
        assert isinstance(session.stage1, InstanceMetadataManager)
        assert isinstance(session.stage2b, IRPolicyMetadataManager)
        assert hasattr(session, 'stage2a')  # LogicalIRMetadataManager
        assert hasattr(session, 'stage3')   # CompilationMetadataManager
        assert hasattr(session, 'stage4')   # ExecutableMetadataManager
        assert hasattr(session, 'stage5')   # ExecutionMetadataManager
        assert hasattr(session, 'stage6_7') # MetricsMetadataManager
    
    def test_register_puzzle(self, tmp_path):
        """Test puzzle instance registration."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        # Mock puzzle
        mock_puzzle = Mock()
        mock_puzzle.size = 4
        mock_puzzle.subgrid_size = 2
        mock_puzzle.open_tuples = [(0, 2), (1, 3)]
        mock_puzzle.board = [(0, 0, 1), (0, 1, 2)]
        mock_puzzle.pre_tuples = [(0, 0), (0, 1)]
        
        puzzle_hash = session.register_puzzle(mock_puzzle, solution_count=1)
        
        assert puzzle_hash == "abc123"
        
        # Verify Stage 1 recorded
        instance = session.stage1.query(puzzle_hash="abc123")
        assert instance is not None
        assert instance["size"] == 4
        assert instance["num_missing_cells"] == 2
    
    def test_register_puzzle_with_qsudoku_wrapper(self, tmp_path):
        """Test registration with QSudoku wrapper."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        # Mock QSudoku wrapper
        mock_sudoku_puzzle = Mock()
        mock_sudoku_puzzle.size = 4
        mock_sudoku_puzzle.subgrid_size = 2
        mock_sudoku_puzzle.open_tuples = [(0, 2)]
        
        mock_qsudoku = Mock()
        mock_qsudoku.puzzle = mock_sudoku_puzzle
        
        puzzle_hash = session.register_puzzle(mock_qsudoku)
        
        assert puzzle_hash == "abc123"
        
        # Verify Stage 1 recorded
        instance = session.stage1.query(puzzle_hash="abc123")
        assert instance is not None
        assert instance["size"] == 4
    
    def test_execute_run_requires_solver(self, tmp_path):
        """Test execute_run raises if no solver attached."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        # Mock puzzle without solver
        mock_puzzle = Mock()
        mock_puzzle.quantum_solver = None
        
        with pytest.raises(ValueError, match="Puzzle must have solver attached"):
            session.execute_run(
                puzzle=mock_puzzle,
                backend_alias="aer_simulator",
                shots=1024
            )
    
    @patch.dict('os.environ', {'SUDOKU_NISQ_NEW_METADATA': '1'})
    def test_execute_run_with_validation_context(self, tmp_path):
        """Test execute_run sets validation context."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        # Mock puzzle with solver
        mock_puzzle = Mock()
        mock_puzzle.quantum_solver = Mock()
        mock_puzzle.run = Mock(return_value={"counts": {}})
        mock_puzzle.set_validation_context = Mock()
        
        validation_ctx = {"valid_solutions": [(1, 2, 3, 4)]}
        
        session.execute_run(
            puzzle=mock_puzzle,
            backend_alias="aer_simulator",
            shots=1024,
            validation_context=validation_ctx
        )
        
        # Verify validation context was set
        mock_puzzle.set_validation_context.assert_called_once_with(
            valid_solutions=[(1, 2, 3, 4)]
        )
        
        # Verify run was called
        mock_puzzle.run.assert_called_once()
    
    @patch.dict('os.environ', {'SUDOKU_NISQ_NEW_METADATA': '1'})
    def test_execute_batch(self, tmp_path):
        """Test batch execution with multiple runs."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        # Mock puzzle
        mock_puzzle = Mock()
        mock_puzzle.quantum_solver = Mock()
        mock_results = [{"counts": {}, "run_id": f"run_{i}"} for i in range(3)]
        mock_puzzle.run = Mock(side_effect=mock_results)
        mock_puzzle.set_validation_context = Mock()
        
        results = session.execute_batch(
            puzzle=mock_puzzle,
            backend_alias="aer_simulator",
            n_runs=3,
            shots=1024,
            validation_context={"valid_solutions": [(1, 2, 3, 4)]}
        )
        
        assert len(results) == 3
        assert mock_puzzle.run.call_count == 3
    
    def test_query_executions_empty(self, tmp_path):
        """Test query_executions with no recorded executions."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        results = session.query_executions(backend_alias="aer_simulator")
        
        assert results == []
    
    def test_get_metrics_summary_empty(self, tmp_path):
        """Test get_metrics_summary with no metrics."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        summary = session.get_metrics_summary()
        
        assert summary == {}


# ============================================================================
# Integration Tests
# ============================================================================

class TestPhase5Integration:
    """Integration tests for Phase 5 components."""
    
    @patch.dict('os.environ', {'SUDOKU_NISQ_NEW_METADATA': '1'})
    def test_full_workflow_simulation(self, tmp_path):
        """Test complete workflow: register → build → execute."""
        session = BenchmarkSession(puzzle_hash="abc123", cache_base=tmp_path)
        
        # Step 1: Register puzzle
        mock_puzzle = Mock()
        mock_puzzle.size = 4
        mock_puzzle.subgrid_size = 2
        mock_puzzle.open_tuples = [(0, 2), (1, 3)]
        
        puzzle_hash = session.register_puzzle(mock_puzzle, solution_count=1)
        
        assert puzzle_hash == "abc123"
        
        # Step 2: Record IR policy (simulated)
        session.stage2b.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple",
            sdk="pytket",
            sdk_version="1.31.1",
            solver_options={"decompose_cnz": True}
        )
        
        # Verify Stage 1 registration
        instance = session.stage1.query(puzzle_hash="abc123")
        assert instance["size"] == 4
        
        # Verify Stage 2b policy
        policy = session.stage2b.query(
            solver_name="ExactCoverQuantumSolver",
            encoding="simple"
        )
        assert policy["sdk"] == "pytket"
    
    def test_multi_puzzle_tracking(self, tmp_path):
        """Test tracking multiple puzzles in global registry."""
        session1 = BenchmarkSession(puzzle_hash="puzzle_a", cache_base=tmp_path)
        session2 = BenchmarkSession(puzzle_hash="puzzle_b", cache_base=tmp_path)
        
        # Register different puzzles
        mock_puzzle_a = Mock(size=4, subgrid_size=2, open_tuples=[(0, 2)])
        mock_puzzle_b = Mock(size=9, subgrid_size=3, open_tuples=[(0, 2), (1, 3)])
        
        session1.register_puzzle(mock_puzzle_a)
        session2.register_puzzle(mock_puzzle_b)
        
        # Query both from same Stage 1 manager
        puzzles_4x4 = session1.stage1.query(size=4)
        puzzles_9x9 = session1.stage1.query(size=9)
        
        assert len(puzzles_4x4) == 1
        assert len(puzzles_9x9) == 1
        assert puzzles_4x4[0]["puzzle_hash"] == "puzzle_a"
        assert puzzles_9x9[0]["puzzle_hash"] == "puzzle_b"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
