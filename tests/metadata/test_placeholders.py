"""Tests for placeholder stage managers."""

import pytest


@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory."""
    return tmp_path / "test_cache"


def test_instance_manager_placeholder(tmp_cache):
    """Test InstanceMetadataManager validates required fields."""
    from sudoku_nisq.metadata import InstanceMetadataManager
    
    manager = InstanceMetadataManager(tmp_cache)
    assert manager.stage_number == 1
    # Use Path.parts to check path components (Windows-safe)
    assert "instances" in manager.storage_path.parts
    assert manager.storage_path.name == "registry.json"
    
    # Should raise ValueError when required field puzzle_hash is missing
    with pytest.raises(ValueError, match="puzzle_hash is required"):
        manager.record()
    
    # Query should work without filters
    results = manager.query()
    assert isinstance(results, list)


def test_logical_ir_manager_placeholder(tmp_cache):
    """Test LogicalIRMetadataManager - NOW IMPLEMENTED in Phase 1."""
    from sudoku_nisq.metadata import LogicalIRMetadataManager
    from pytket import Circuit
    
    manager = LogicalIRMetadataManager(tmp_cache, "abc123")
    assert manager.stage_number == 2
    assert "abc123" in str(manager.storage_path)
    assert "stage_2a_logical_ir.json" in str(manager.storage_path)
    
    # Phase 1 implementation: record() and query() should work
    circuit = Circuit(2)
    circuit.H(0)
    
    circuit_hash = manager.record(
        solver_name="TestSolver",
        encoding="simple",
        circuit=circuit
    )
    assert len(circuit_hash) == 64  # SHA256 hash
    
    result = manager.query(solver_name="TestSolver", encoding="simple")
    assert result is not None
    assert result["circuit_hash"] == circuit_hash


def test_ir_policy_manager_placeholder(tmp_cache):
    """Test IRPolicyMetadataManager validates required fields."""
    from sudoku_nisq.metadata import IRPolicyMetadataManager
    
    manager = IRPolicyMetadataManager(tmp_cache, "abc123")
    assert manager.stage_number == 2
    assert "stage_2b_ir_policy.json" in str(manager.storage_path)
    
    # Should raise ValueError when required field solver_name is missing
    with pytest.raises(ValueError, match="solver_name is required"):
        manager.record()
    
    # Query with proper filters should work
    results = manager.query(solver_name="TestSolver")
    # Returns None when solver doesn't exist (no records yet)
    assert results is None or isinstance(results, dict)

def test_compilation_manager_placeholder(tmp_cache):
    """Test CompilationMetadataManager - now fully implemented in Phase 2."""
    from sudoku_nisq.metadata import CompilationMetadataManager
    
    manager = CompilationMetadataManager(tmp_cache, "abc123")
    assert manager.stage_number == 3
    assert "stage_3_compilation.jsonl" in str(manager.storage_path)
    
    # Phase 2 implementation complete - test basic functionality
    compilation_id = manager.record(
        circuit_hash="test_hash",
        backend_alias="aer",
        opt_level=0,
        resources={"n_qubits": 4, "n_gates": 20, "depth": 8}
    )
    assert compilation_id is not None
    
    results = manager.query()
    assert len(results) == 1
    assert results[0]["circuit_hash"] == "test_hash"

def test_executable_manager_placeholder(tmp_cache):
    """Test ExecutableMetadataManager placeholder."""
    from sudoku_nisq.metadata import ExecutableMetadataManager
    
    manager = ExecutableMetadataManager(tmp_cache, "abc123")
    assert manager.stage_number == 4
    assert "stage_4_executable.json" in str(manager.storage_path)
    
    with pytest.raises(NotImplementedError, match="Phase 3"):
        manager.record()
    
    with pytest.raises(NotImplementedError, match="Phase 3"):
        manager.query()


def test_execution_manager_placeholder(tmp_cache):
    """Test ExecutionMetadataManager implementation (Phase 3 complete)."""
    from sudoku_nisq.metadata import ExecutionMetadataManager
    
    manager = ExecutionMetadataManager(tmp_cache, "abc123")
    assert manager.stage_number == 5
    assert "stage_5_executions.jsonl" in str(manager.storage_path)
    
    # Phase 3 complete - test basic functionality
    run_id = manager.record(
        compilation_id="comp_test",
        backend_name="aer_simulator",
        counts={"00": 512, "11": 512},
        shots=1024,
        execution_time_ms=100.0
    )
    assert run_id is not None
    
    records = manager.query(run_id=run_id)
    assert len(records) == 1
    assert records[0]["run_id"] == run_id


def test_metrics_manager_placeholder(tmp_cache):
    """Test MetricsMetadataManager placeholder."""
    from sudoku_nisq.metadata import MetricsMetadataManager
    
    manager = MetricsMetadataManager(tmp_cache, "abc123")
    assert manager.stage_number == 6
    assert "stage_6_7_metrics.json" in str(manager.storage_path)
    
    # Note: MetricsMetadataManager is now fully implemented (no longer a placeholder)
    # Basic sanity check that it can be instantiated
    assert manager is not None
