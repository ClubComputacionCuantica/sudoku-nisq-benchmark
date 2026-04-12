"""Unit tests for ExecutionMetadataManager (Stage 5).

Tests execution metadata recording, querying, and hardware snapshot integration.
"""

import pytest
from datetime import datetime, timezone
from sudoku_nisq.metadata.execution import ExecutionMetadataManager

pytestmark = pytest.mark.unit


@pytest.fixture
def exec_manager(tmp_path):
    """Create ExecutionMetadataManager with temporary storage."""
    cache_base = tmp_path / ".metadata"
    puzzle_hash = "test_puzzle_abc123"
    manager = ExecutionMetadataManager(cache_base, puzzle_hash)
    return manager


@pytest.fixture
def sample_counts():
    """Sample measurement counts distribution."""
    return {
        "0000": 512,
        "1111": 256,
        "0101": 128,
        "1010": 128
    }


@pytest.fixture
def sample_hardware_snapshot():
    """Sample hardware calibration data (with string keys for JSON compatibility)."""
    return {
        "backend_name": "ibm_brisbane",
        "provider": "ibm",
        "calibration_timestamp": "2024-01-15T08:00:00",
        "single_qubit_gate_error": {"0": 0.001, "1": 0.0012},  # String keys for JSON
        "two_qubit_gate_error": {"0,1": 0.01, "1,2": 0.011},  # String keys for JSON
        "readout_error": {"0": 0.02, "1": 0.025},  # String keys for JSON
        "t1_times": {"0": 95.5, "1": 102.3},  # String keys for JSON
        "t2_times": {"0": 120.1, "1": 115.8},  # String keys for JSON
        "extra_properties": {"max_shots": 10000}
    }


class TestExecutionMetadataManagerBasics:
    """Test basic functionality of ExecutionMetadataManager."""
    
    def test_stage_number(self, exec_manager):
        """Test that stage number is 5."""
        assert exec_manager.stage_number == 5
    
    def test_storage_path_structure(self, exec_manager, tmp_path):
        """Test that storage path follows expected structure."""
        expected_path = tmp_path / ".metadata" / "test_puzzle_abc123" / "stage_5_executions.jsonl"
        assert exec_manager.storage_path == expected_path
    
    def test_record_requires_compilation_id(self, exec_manager, sample_counts):
        """Test that record() raises ValueError if compilation_id is missing."""
        with pytest.raises(ValueError, match="compilation_id is required"):
            exec_manager.record(
                compilation_id="",
                backend_name="aer_simulator",
                counts=sample_counts,
                shots=1024,
                execution_time_ms=125.5
            )
        
        with pytest.raises(ValueError, match="compilation_id is required"):
            exec_manager.record(
                compilation_id=None,
                backend_name="aer_simulator",
                counts=sample_counts,
                shots=1024,
                execution_time_ms=125.5
            )


class TestExecutionRecording:
    """Test execution metadata recording."""
    
    def test_record_minimal_execution(self, exec_manager, sample_counts):
        """Test recording with only required fields."""
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5
        )
        
        # Verify run_id was generated
        assert run_id is not None
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        
        # Verify record was saved
        records = exec_manager.query()
        assert len(records) == 1
        assert records[0]["run_id"] == run_id
        assert records[0]["compilation_id"] == "comp_abc123"
        assert records[0]["backend_name"] == "aer_simulator"
        assert records[0]["counts"] == sample_counts
        assert records[0]["shots"] == 1024
        assert records[0]["execution_time_ms"] == 125.5
    
    def test_record_with_hardware_snapshot(
        self, exec_manager, sample_counts, sample_hardware_snapshot
    ):
        """Test recording with hardware calibration data."""
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="ibm_brisbane",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=3450.2,
            hardware_snapshot=sample_hardware_snapshot
        )
        
        records = exec_manager.query(run_id=run_id)
        assert len(records) == 1
        assert records[0]["hardware_snapshot"] == sample_hardware_snapshot
    
    def test_record_with_job_id(self, exec_manager, sample_counts):
        """Test recording with provider job identifier."""
        run_id = exec_manager.record(
            compilation_id="comp_xyz789",
            backend_name="ibm_brisbane",
            counts=sample_counts,
            shots=2048,
            execution_time_ms=5123.4,
            job_id="cxyz123abc456"
        )
        
        records = exec_manager.query(run_id=run_id)
        assert records[0]["job_id"] == "cxyz123abc456"
    
    def test_record_with_circuit_metrics(self, exec_manager, sample_counts):
        """Test recording with circuit characteristics."""
        circuit_metrics = {
            "n_qubits": 4,
            "depth": 15,
            "gate_counts": {"h": 4, "cx": 6, "rz": 8, "measure": 4}
        }
        
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5,
            circuit_metrics=circuit_metrics
        )
        
        records = exec_manager.query(run_id=run_id)
        assert records[0]["circuit_metrics"] == circuit_metrics
    
    def test_record_with_custom_timestamp(self, exec_manager, sample_counts):
        """Test recording with explicit timestamp."""
        custom_time = datetime(2024, 1, 15, 14, 30, 45)
        
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5,
            timestamp=custom_time
        )
        
        records = exec_manager.query(run_id=run_id)
        assert records[0]["timestamp"] == custom_time.isoformat()
    
    def test_record_with_custom_run_id(self, exec_manager, sample_counts):
        """Test recording with explicit run_id (useful for testing)."""
        custom_run_id = "test_run_12345"
        
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5,
            run_id=custom_run_id
        )
        
        assert run_id == custom_run_id
        records = exec_manager.query(run_id=custom_run_id)
        assert len(records) == 1
    
    def test_record_auto_generates_timestamp(self, exec_manager, sample_counts):
        """Test that timestamp is auto-generated if not provided."""
        before_time = datetime.now(timezone.utc)
        
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5
        )
        
        after_time = datetime.now(timezone.utc)
        
        records = exec_manager.query(run_id=run_id)
        recorded_time = datetime.fromisoformat(records[0]["timestamp"])
        
        assert before_time <= recorded_time <= after_time
    
    def test_record_multiple_runs(self, exec_manager, sample_counts):
        """Test recording multiple execution runs."""
        run_ids = []
        for i in range(5):
            run_id = exec_manager.record(
                compilation_id=f"comp_{i}",
                backend_name="aer_simulator",
                counts=sample_counts,
                shots=1024,
                execution_time_ms=100.0 + i * 10
            )
            run_ids.append(run_id)
        
        # Verify all runs were recorded
        all_records = exec_manager.query()
        assert len(all_records) == 5
        
        # Verify run_ids are unique
        recorded_run_ids = [r["run_id"] for r in all_records]
        assert len(set(recorded_run_ids)) == 5
        assert set(recorded_run_ids) == set(run_ids)


class TestExecutionQuerying:
    """Test execution metadata querying with various filters."""
    
    @pytest.fixture
    def populated_manager(self, exec_manager, sample_counts):
        """Create manager with multiple test records."""
        # Record 3 runs with different parameters
        exec_manager.record(
            compilation_id="comp_1",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=100.0,
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            run_id="run_1"
        )
        exec_manager.record(
            compilation_id="comp_1",
            backend_name="ibm_brisbane",
            counts=sample_counts,
            shots=2048,
            execution_time_ms=3000.0,
            timestamp=datetime(2024, 1, 15, 11, 0, 0),
            run_id="run_2"
        )
        exec_manager.record(
            compilation_id="comp_2",
            backend_name="ibm_brisbane",
            counts=sample_counts,
            shots=2048,
            execution_time_ms=3200.0,
            timestamp=datetime(2024, 1, 15, 12, 0, 0),
            run_id="run_3"
        )
        return exec_manager
    
    def test_query_all_records(self, populated_manager):
        """Test querying without filters returns all records."""
        records = populated_manager.query()
        assert len(records) == 3
    
    def test_query_by_run_id(self, populated_manager):
        """Test filtering by specific run ID."""
        records = populated_manager.query(run_id="run_2")
        assert len(records) == 1
        assert records[0]["run_id"] == "run_2"
        assert records[0]["backend_name"] == "ibm_brisbane"
    
    def test_query_by_compilation_id(self, populated_manager):
        """Test filtering by compilation ID."""
        records = populated_manager.query(compilation_id="comp_1")
        assert len(records) == 2
        assert all(r["compilation_id"] == "comp_1" for r in records)
    
    def test_query_by_backend_name(self, populated_manager):
        """Test filtering by backend name."""
        records = populated_manager.query(backend_name="ibm_brisbane")
        assert len(records) == 2
        assert all(r["backend_name"] == "ibm_brisbane" for r in records)
        
        records = populated_manager.query(backend_name="aer_simulator")
        assert len(records) == 1
        assert records[0]["backend_name"] == "aer_simulator"
    
    def test_query_by_date_range(self, populated_manager):
        """Test filtering by timestamp range."""
        start = datetime(2024, 1, 15, 10, 30, 0)
        end = datetime(2024, 1, 15, 12, 30, 0)
        
        records = populated_manager.query(date_range=(start, end))
        assert len(records) == 2
        assert "run_1" not in [r["run_id"] for r in records]
        assert "run_2" in [r["run_id"] for r in records]
        assert "run_3" in [r["run_id"] for r in records]
    
    def test_query_with_limit(self, populated_manager):
        """Test limiting number of returned records."""
        records = populated_manager.query(limit=2)
        assert len(records) == 2
        
        records = populated_manager.query(limit=1)
        assert len(records) == 1
    
    def test_query_combined_filters(self, populated_manager):
        """Test using multiple filters simultaneously."""
        records = populated_manager.query(
            compilation_id="comp_1",
            backend_name="ibm_brisbane"
        )
        assert len(records) == 1
        assert records[0]["run_id"] == "run_2"
    
    def test_query_no_matches(self, populated_manager):
        """Test query with no matching records."""
        records = populated_manager.query(run_id="nonexistent")
        assert len(records) == 0
        
        records = populated_manager.query(backend_name="quantinuum_h1")
        assert len(records) == 0


class TestExecutionPersistence:
    """Test persistence and file operations."""
    
    def test_storage_file_created(self, exec_manager, sample_counts):
        """Test that JSONL file is created on first record."""
        assert not exec_manager.storage_path.exists()
        
        exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5
        )
        
        assert exec_manager.storage_path.exists()
        assert exec_manager.storage_path.suffix == ".jsonl"
    
    def test_reload_from_disk(self, exec_manager, sample_counts):
        """Test that records persist across manager instances."""
        # Record with first manager instance
        run_id_1 = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5
        )
        
        # Create new manager instance with same storage
        new_manager = ExecutionMetadataManager(
            exec_manager.cache_base,
            exec_manager.puzzle_hash
        )
        
        # Verify record is still accessible
        records = new_manager.query(run_id=run_id_1)
        assert len(records) == 1
        assert records[0]["run_id"] == run_id_1
    
    def test_append_only_behavior(self, exec_manager, sample_counts):
        """Test that records are appended, not overwritten."""
        run_ids = []
        for i in range(3):
            run_id = exec_manager.record(
                compilation_id=f"comp_{i}",
                backend_name="aer_simulator",
                counts=sample_counts,
                shots=1024,
                execution_time_ms=100.0
            )
            run_ids.append(run_id)
        
        # All records should exist
        all_records = exec_manager.query()
        assert len(all_records) == 3
        assert set(r["run_id"] for r in all_records) == set(run_ids)


class TestExecutionEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_counts_dict(self, exec_manager):
        """Test recording with empty counts (valid but unusual)."""
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts={},
            shots=0,
            execution_time_ms=0.0
        )
        
        records = exec_manager.query(run_id=run_id)
        assert records[0]["counts"] == {}
    
    def test_large_counts_distribution(self, exec_manager):
        """Test recording with large counts dictionary."""
        # Generate 1000 unique bitstrings
        large_counts = {format(i, '010b'): i % 100 for i in range(1000)}
        
        run_id = exec_manager.record(
            compilation_id="comp_abc123",
            backend_name="aer_simulator",
            counts=large_counts,
            shots=sum(large_counts.values()),
            execution_time_ms=500.0
        )
        
        records = exec_manager.query(run_id=run_id)
        assert len(records[0]["counts"]) == 1000
    
    def test_query_empty_manager(self, exec_manager):
        """Test querying when no records exist."""
        records = exec_manager.query()
        assert records == []
    
    def test_special_characters_in_ids(self, exec_manager, sample_counts):
        """Test IDs with special characters."""
        run_id = exec_manager.record(
            compilation_id="comp-abc_123.xyz",
            backend_name="ibm_brisbane-v2",
            counts=sample_counts,
            shots=1024,
            execution_time_ms=125.5,
            job_id="job:abc-123_xyz"
        )
        
        records = exec_manager.query(run_id=run_id)
        assert records[0]["compilation_id"] == "comp-abc_123.xyz"
        assert records[0]["job_id"] == "job:abc-123_xyz"
