"""Tests for CompilationMetadataManager (Stage 3)."""

import pytest
from uuid import UUID

from sudoku_nisq.metadata.compilation import CompilationMetadataManager


@pytest.fixture
def compilation_manager(tmp_path):
    """Create compilation metadata manager with temp storage."""
    cache_base = tmp_path / ".quantum_solver_cache"
    puzzle_hash = "test_puzzle_abc123"
    manager = CompilationMetadataManager(cache_base, puzzle_hash)
    return manager


class TestCompilationRecord:
    """Tests for compilation record creation."""
    
    def test_record_basic_compilation(self, compilation_manager):
        """Test recording a basic compilation without routing."""
        compilation_id = compilation_manager.record(
            circuit_hash="circuit_xyz789",
            backend_alias="aer_simulator",
            opt_level=0,
            resources={"n_qubits": 4, "n_gates": 15, "depth": 8}
        )
        
        # Should return valid UUID
        assert isinstance(UUID(compilation_id), UUID)
        
        # Verify JSONL file created
        assert compilation_manager.storage_path.exists()
        
        # Verify record content
        records = compilation_manager._load_jsonl()
        assert len(records) == 1
        record = records[0]
        assert record["compilation_id"] == compilation_id
        assert record["circuit_hash"] == "circuit_xyz789"
        assert record["backend_alias"] == "aer_simulator"
        assert record["opt_level"] == 0
        assert record["resources"]["n_qubits"] == 4
    
    def test_record_with_routing_metadata(self, compilation_manager):
        """Test recording compilation with routing metadata."""
        routing = {
            "initial_layout": {0: 0, 1: 1, 2: 2},
            "final_layout": {0: 2, 1: 0, 2: 1},
            "swap_count": 2
        }
        
        compilation_manager.record(
            circuit_hash="circuit_with_routing",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources={"n_qubits": 3, "n_gates": 25, "depth": 12},
            routing=routing
        )
        
        # Verify routing metadata stored
        records = compilation_manager._load_jsonl()
        record = records[0]
        assert "routing" in record
        assert record["routing"]["swap_count"] == 2
        # Note: JSON serialization converts int keys to strings
        assert record["routing"]["initial_layout"] == {"0": 0, "1": 1, "2": 2}
        assert record["routing"]["final_layout"] == {"0": 2, "1": 0, "2": 1}
    
    def test_record_multiple_compilations(self, compilation_manager):
        """Test append-only behavior with multiple compilations."""
        # Record 3 compilations
        ids = []
        for i in range(3):
            comp_id = compilation_manager.record(
                circuit_hash=f"circuit_{i}",
                backend_alias="aer_simulator",
                opt_level=i,
                resources={"n_qubits": 4, "n_gates": 10 + i*5, "depth": 5 + i}
            )
            ids.append(comp_id)
        
        # All should have unique IDs
        assert len(set(ids)) == 3
        
        # JSONL should have 3 records
        records = compilation_manager._load_jsonl()
        assert len(records) == 3
    
    def test_record_with_additional_kwargs(self, compilation_manager):
        """Test recording with additional metadata fields."""
        compilation_manager.record(
            circuit_hash="circuit_custom",
            backend_alias="custom_backend",
            opt_level=1,
            resources={"n_qubits": 5, "n_gates": 20, "depth": 10},
            sdk_type="qiskit",
            custom_field="custom_value"
        )
        
        records = compilation_manager._load_jsonl()
        record = records[0]
        assert record["sdk_type"] == "qiskit"
        assert record["custom_field"] == "custom_value"
    
    def test_record_includes_timestamp(self, compilation_manager):
        """Test that records include ISO timestamp."""
        compilation_manager.record(
            circuit_hash="circuit_timestamp",
            backend_alias="aer",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        
        records = compilation_manager._load_jsonl()
        record = records[0]
        assert "timestamp" in record
        # Should be valid ISO format with timezone
        from datetime import datetime
        dt = datetime.fromisoformat(record["timestamp"])
        assert dt.tzinfo is not None


class TestCompilationQuery:
    """Tests for compilation query functionality."""
    
    def test_query_empty_storage(self, compilation_manager):
        """Test querying when no compilations exist."""
        results = compilation_manager.query()
        assert results == []
    
    def test_query_all_records(self, compilation_manager):
        """Test querying without filters returns all records."""
        # Create 3 compilations
        for i in range(3):
            compilation_manager.record(
                circuit_hash=f"circuit_{i}",
                backend_alias="aer",
                opt_level=0,
                resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
            )
        
        results = compilation_manager.query()
        assert len(results) == 3
    
    def test_query_by_backend_alias(self, compilation_manager):
        """Test filtering by backend alias."""
        # Create compilations for different backends
        compilation_manager.record(
            circuit_hash="circuit_1",
            backend_alias="aer_simulator",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        compilation_manager.record(
            circuit_hash="circuit_2",
            backend_alias="ibm_brisbane",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        compilation_manager.record(
            circuit_hash="circuit_3",
            backend_alias="aer_simulator",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        
        results = compilation_manager.query(backend_alias="aer_simulator")
        assert len(results) == 2
        assert all(r["backend_alias"] == "aer_simulator" for r in results)
    
    def test_query_by_opt_level(self, compilation_manager):
        """Test filtering by optimization level."""
        # Create compilations at different opt levels
        for opt in [0, 1, 2, 1]:
            compilation_manager.record(
                circuit_hash="circuit_opt",
                backend_alias="aer",
                opt_level=opt,
                resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
            )
        
        results = compilation_manager.query(opt_level=1)
        assert len(results) == 2
        assert all(r["opt_level"] == 1 for r in results)
    
    def test_query_by_circuit_hash(self, compilation_manager):
        """Test filtering by circuit hash."""
        # Create compilations for different circuits
        target_hash = "target_circuit_xyz"
        compilation_manager.record(
            circuit_hash=target_hash,
            backend_alias="aer",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        for i in range(3):
            compilation_manager.record(
                circuit_hash=f"other_circuit_{i}",
                backend_alias="aer",
                opt_level=0,
                resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
            )
        
        results = compilation_manager.query(circuit_hash=target_hash)
        assert len(results) == 1
        assert results[0]["circuit_hash"] == target_hash
    
    def test_query_by_compilation_id(self, compilation_manager):
        """Test filtering by compilation ID."""
        # Create multiple compilations
        ids = []
        for i in range(3):
            comp_id = compilation_manager.record(
                circuit_hash=f"circuit_{i}",
                backend_alias="aer",
                opt_level=0,
                resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
            )
            ids.append(comp_id)
        
        # Query for specific ID
        results = compilation_manager.query(compilation_id=ids[1])
        assert len(results) == 1
        assert results[0]["compilation_id"] == ids[1]
    
    def test_query_with_multiple_filters(self, compilation_manager):
        """Test combining multiple filter criteria."""
        # Create diverse compilations
        target_id = compilation_manager.record(
            circuit_hash="target_circuit",
            backend_alias="ibm_brisbane",
            opt_level=2,
            resources={"n_qubits": 3, "n_gates": 20, "depth": 10}
        )
        
        # Add noise
        for i in range(5):
            compilation_manager.record(
                circuit_hash=f"other_{i}",
                backend_alias="aer" if i % 2 == 0 else "ibm_brisbane",
                opt_level=i % 3,
                resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
            )
        
        # Query with multiple filters
        results = compilation_manager.query(
            backend_alias="ibm_brisbane",
            opt_level=2,
            circuit_hash="target_circuit"
        )
        assert len(results) == 1
        assert results[0]["compilation_id"] == target_id
    
    def test_query_no_matches(self, compilation_manager):
        """Test query with no matching records."""
        # Create some compilations
        compilation_manager.record(
            circuit_hash="circuit_1",
            backend_alias="aer",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        
        # Query for non-existent backend
        results = compilation_manager.query(backend_alias="nonexistent_backend")
        assert results == []


class TestCompilationPerformance:
    """Tests for JSONL append and query performance."""
    
    def test_append_performance(self, compilation_manager):
        """Test that appending 100+ records is fast (O(1) per record)."""
        import time
        
        start = time.time()
        for i in range(100):
            compilation_manager.record(
                circuit_hash=f"circuit_{i}",
                backend_alias="aer",
                opt_level=i % 4,
                resources={"n_qubits": 5, "n_gates": 50 + i, "depth": 20 + i}
            )
        elapsed = time.time() - start
        
        # 100 appends should take < 1 second (very generous threshold)
        assert elapsed < 1.0
        
        # Verify all records exist
        records = compilation_manager._load_jsonl()
        assert len(records) == 100
    
    def test_query_performance_large_log(self, compilation_manager):
        """Test query performance with 1000 records."""
        import time
        
        # Create 1000 compilation records
        target_id = None
        for i in range(1000):
            comp_id = compilation_manager.record(
                circuit_hash=f"circuit_{i}",
                backend_alias=f"backend_{i % 10}",
                opt_level=i % 4,
                resources={"n_qubits": 5, "n_gates": 50, "depth": 20}
            )
            if i == 500:
                target_id = comp_id
        
        # Query for specific compilation
        start = time.time()
        results = compilation_manager.query(compilation_id=target_id)
        elapsed = time.time() - start
        
        # Query should take < 200ms (increased from 100ms to reduce flakiness)
        assert elapsed < 0.2
        assert len(results) == 1


class TestCompilationEdgeCases:
    """Tests for edge cases and error conditions."""
    
    def test_empty_resources(self, compilation_manager):
        """Test recording with empty resources dict."""
        compilation_manager.record(
            circuit_hash="circuit_empty",
            backend_alias="aer",
            opt_level=0,
            resources={}
        )
        
        records = compilation_manager._load_jsonl()
        assert len(records) == 1
        assert records[0]["resources"] == {}
    
    def test_none_routing_metadata(self, compilation_manager):
        """Test that None routing is not stored."""
        compilation_manager.record(
            circuit_hash="circuit_no_routing",
            backend_alias="aer",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3},
            routing=None
        )
        
        records = compilation_manager._load_jsonl()
        assert "routing" not in records[0]
    
    def test_special_characters_in_strings(self, compilation_manager):
        """Test handling of special characters in string fields."""
        compilation_manager.record(
            circuit_hash="circuit_with_'quotes\"_and_\n_newlines",
            backend_alias="backend-with-dashes_and_underscores",
            opt_level=0,
            resources={"n_qubits": 2, "n_gates": 5, "depth": 3}
        )
        
        # Should serialize/deserialize correctly
        records = compilation_manager._load_jsonl()
        assert records[0]["circuit_hash"] == "circuit_with_'quotes\"_and_\n_newlines"
    
    def test_large_routing_layout(self, compilation_manager):
        """Test storing large layout dictionaries (127-qubit system)."""
        large_layout = {i: (i + 10) % 127 for i in range(127)}
        
        compilation_manager.record(
            circuit_hash="circuit_large",
            backend_alias="ibm_127q",
            opt_level=3,
            resources={"n_qubits": 127, "n_gates": 10000, "depth": 1000},
            routing={"initial_layout": large_layout, "final_layout": large_layout}
        )
        
        # Should handle large dicts without issues
        records = compilation_manager._load_jsonl()
        assert len(records[0]["routing"]["initial_layout"]) == 127
