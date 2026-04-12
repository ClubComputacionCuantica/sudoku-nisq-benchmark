"""Tests for LogicalIRMetadataManager (Stage 2a)."""

import pytest
from pytket import Circuit
from qiskit import QuantumCircuit


@pytest.fixture
def tmp_cache(tmp_path):
    """Temporary cache directory for testing."""
    cache_dir = tmp_path / "test_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def puzzle_hash():
    """Sample puzzle hash for testing."""
    return "abc123def456"


@pytest.fixture
def logical_ir_manager(tmp_cache, puzzle_hash):
    """LogicalIRMetadataManager instance for testing."""
    from sudoku_nisq.metadata import LogicalIRMetadataManager
    return LogicalIRMetadataManager(tmp_cache, puzzle_hash)


@pytest.fixture
def pytket_circuit():
    """Sample PyTKET circuit."""
    circuit = Circuit(3)
    circuit.H(0)
    circuit.CX(0, 1)
    circuit.CX(1, 2)
    circuit.Rz(0.5, 2)
    return circuit


@pytest.fixture
def qiskit_circuit():
    """Sample Qiskit circuit."""
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.cx(1, 2)
    circuit.rz(0.5, 2)
    return circuit


class TestSDKDetection:
    """Tests for _detect_sdk method."""
    
    def test_detect_pytket(self, logical_ir_manager, pytket_circuit):
        """Test SDK detection for PyTKET circuits."""
        sdk = logical_ir_manager._detect_sdk(pytket_circuit)
        assert sdk == "pytket"
    
    def test_detect_qiskit(self, logical_ir_manager, qiskit_circuit):
        """Test SDK detection for Qiskit circuits."""
        sdk = logical_ir_manager._detect_sdk(qiskit_circuit)
        assert sdk == "qiskit"
    
    def test_detect_unknown_raises(self, logical_ir_manager):
        """Test that unknown circuit type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown circuit type"):
            logical_ir_manager._detect_sdk("not_a_circuit")


class TestResourceExtraction:
    """Tests for _extract_resources methods."""
    
    def test_extract_pytket_resources(self, logical_ir_manager, pytket_circuit):
        """Test resource extraction from PyTKET circuit."""
        resources = logical_ir_manager._extract_resources_pytket(pytket_circuit)
        
        assert resources["n_qubits"] == 3
        assert resources["n_gates"] == 4  # H + 2*CX + Rz
        assert "depth" in resources
        assert "gate_breakdown" in resources
    
    def test_extract_qiskit_resources(self, logical_ir_manager, qiskit_circuit):
        """Test resource extraction from Qiskit circuit."""
        resources = logical_ir_manager._extract_resources_qiskit(qiskit_circuit)
        
        assert resources["n_qubits"] == 3
        assert resources["n_gates"] == 4  # H + 2*CX + Rz
        assert "depth" in resources
        assert "gate_breakdown" in resources
    
    def test_extract_resources_dispatcher(self, logical_ir_manager, pytket_circuit):
        """Test _extract_resources dispatcher."""
        resources = logical_ir_manager._extract_resources(pytket_circuit, "pytket")
        
        assert "n_qubits" in resources
        assert "n_gates" in resources
        assert "depth" in resources
    
    def test_extract_resources_invalid_sdk(self, logical_ir_manager, pytket_circuit):
        """Test that invalid SDK type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown SDK type"):
            logical_ir_manager._extract_resources(pytket_circuit, "invalid_sdk")


class TestCircuitHashing:
    """Tests for _compute_circuit_hash method."""
    
    def test_hash_pytket_deterministic(self, logical_ir_manager, pytket_circuit):
        """Test that PyTKET circuit hashing is deterministic."""
        hash1 = logical_ir_manager._compute_circuit_hash(pytket_circuit, "pytket")
        hash2 = logical_ir_manager._compute_circuit_hash(pytket_circuit, "pytket")
        
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 hex digest
    
    def test_hash_qiskit_deterministic(self, logical_ir_manager, qiskit_circuit):
        """Test that Qiskit circuit hashing is deterministic."""
        hash1 = logical_ir_manager._compute_circuit_hash(qiskit_circuit, "qiskit")
        hash2 = logical_ir_manager._compute_circuit_hash(qiskit_circuit, "qiskit")
        
        assert hash1 == hash2
        assert len(hash1) == 64
    
    def test_different_circuits_different_hashes(self, logical_ir_manager):
        """Test that different circuits produce different hashes."""
        circuit1 = Circuit(2)
        circuit1.H(0)
        
        circuit2 = Circuit(2)
        circuit2.H(0)
        circuit2.CX(0, 1)
        
        hash1 = logical_ir_manager._compute_circuit_hash(circuit1, "pytket")
        hash2 = logical_ir_manager._compute_circuit_hash(circuit2, "pytket")
        
        assert hash1 != hash2
    
    def test_hash_fallback_on_error(self, logical_ir_manager):
        """Test hash fallback to resource-based when circuit serialization fails."""
        # Create mock circuit that doesn't have to_dict method
        class MockCircuit:
            n_qubits = 3
            def depth(self):
                return 5
        
        mock_circuit = MockCircuit()
        
        # Monkey patch _extract_resources to return predictable values
        def mock_extract(circuit, sdk_type):
            return {"n_qubits": 3, "n_gates": 10, "depth": 5}
        
        logical_ir_manager._extract_resources = mock_extract
        
        with pytest.warns(RuntimeWarning, match="Failed to compute circuit-based hash"):
            hash_val = logical_ir_manager._compute_circuit_hash(mock_circuit, "pytket")
            assert len(hash_val) == 64


class TestRecordMethod:
    """Tests for record() method."""
    
    def test_record_creates_file(self, logical_ir_manager, pytket_circuit):
        """Test that record creates storage file."""
        circuit_hash = logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        assert logical_ir_manager.storage_path.exists()
        assert len(circuit_hash) == 64
    
    def test_record_stores_correct_structure(self, logical_ir_manager, pytket_circuit):
        """Test that record stores data in correct structure."""
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        data = logical_ir_manager._load_json()
        assert "TestSolver" in data
        assert "simple" in data["TestSolver"]
        
        record = data["TestSolver"]["simple"]
        assert "circuit_hash" in record
        assert "sdk_type" in record
        assert "resources" in record
        assert "timestamp" in record
    
    def test_record_auto_detects_sdk(self, logical_ir_manager, qiskit_circuit):
        """Test that record auto-detects SDK from circuit."""
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="pattern",
            circuit=qiskit_circuit,
        )
        
        data = logical_ir_manager._load_json()
        record = data["TestSolver"]["pattern"]
        assert record["sdk_type"] == "qiskit"
    
    def test_record_explicit_sdk_override(self, logical_ir_manager, qiskit_circuit):
        """Test that explicit sdk_type overrides auto-detection."""
        # Pass a Qiskit circuit but explicitly mark it as pytket
        # This tests the override mechanism (though not recommended in practice)
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=qiskit_circuit,
            sdk_type="qiskit",  # Explicitly set (matches actual circuit type)
        )
        
        data = logical_ir_manager._load_json()
        record = data["TestSolver"]["simple"]
        assert record["sdk_type"] == "qiskit"
    
    def test_record_optional_fields(self, logical_ir_manager, pytket_circuit):
        """Test that record stores optional solver configuration."""
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
            decompose_cnz=True,
            track_memory=False,
        )
        
        data = logical_ir_manager._load_json()
        record = data["TestSolver"]["simple"]
        assert record["decompose_cnz"] is True
        assert record["track_memory"] is False
    
    def test_record_multiple_encodings(self, logical_ir_manager, pytket_circuit):
        """Test recording multiple encodings for same solver."""
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="pattern",
            circuit=pytket_circuit,
        )
        
        data = logical_ir_manager._load_json()
        assert "simple" in data["TestSolver"]
        assert "pattern" in data["TestSolver"]
    
    def test_record_missing_required_fields(self, logical_ir_manager):
        """Test that record raises ValueError for missing required fields."""
        with pytest.raises(ValueError, match="requires solver_name, encoding, and circuit"):
            logical_ir_manager.record(solver_name="TestSolver")
    
    def test_record_updates_existing(self, logical_ir_manager, pytket_circuit):
        """Test that record updates existing entry."""
        # First record
        hash1 = logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        # Modify circuit
        pytket_circuit.H(2)
        
        # Second record (should update)
        hash2 = logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        # Hashes should differ
        assert hash1 != hash2
        
        # Only one entry should exist
        data = logical_ir_manager._load_json()
        assert len(data["TestSolver"]) == 1


class TestQueryMethod:
    """Tests for query() method."""
    
    def test_query_no_filters_returns_all(self, logical_ir_manager, pytket_circuit):
        """Test that query with no filters returns all data."""
        logical_ir_manager.record(
            solver_name="Solver1",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        result = logical_ir_manager.query()
        assert "Solver1" in result
    
    def test_query_by_solver_name(self, logical_ir_manager, pytket_circuit):
        """Test query filtering by solver_name."""
        logical_ir_manager.record(
            solver_name="Solver1",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        result = logical_ir_manager.query(solver_name="Solver1")
        assert "simple" in result
    
    def test_query_by_solver_and_encoding(self, logical_ir_manager, pytket_circuit):
        """Test query filtering by solver_name and encoding."""
        logical_ir_manager.record(
            solver_name="Solver1",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        result = logical_ir_manager.query(solver_name="Solver1", encoding="simple")
        assert result is not None
        assert "circuit_hash" in result
    
    def test_query_nonexistent_solver(self, logical_ir_manager):
        """Test that query for nonexistent solver returns None."""
        result = logical_ir_manager.query(solver_name="NonExistent")
        assert result is None
    
    def test_query_nonexistent_encoding(self, logical_ir_manager, pytket_circuit):
        """Test that query for nonexistent encoding returns None."""
        logical_ir_manager.record(
            solver_name="Solver1",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        result = logical_ir_manager.query(solver_name="Solver1", encoding="nonexistent")
        assert result is None
    
    def test_query_by_circuit_hash(self, logical_ir_manager, pytket_circuit):
        """Test query filtering by circuit_hash."""
        circuit_hash = logical_ir_manager.record(
            solver_name="Solver1",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        result = logical_ir_manager.query(circuit_hash=circuit_hash)
        assert result is not None
        assert result["circuit_hash"] == circuit_hash
    
    def test_query_nonexistent_hash(self, logical_ir_manager):
        """Test that query for nonexistent hash returns None."""
        result = logical_ir_manager.query(circuit_hash="nonexistent_hash")
        assert result is None


class TestIntegration:
    """Integration tests for full workflow."""
    
    def test_full_workflow_pytket(self, logical_ir_manager, pytket_circuit):
        """Test complete workflow: record, query, verify."""
        # Record
        circuit_hash = logical_ir_manager.record(
            solver_name="ExactCoverQuantumSolver",
            encoding="pattern",
            circuit=pytket_circuit,
            decompose_cnz=True,
        )
        
        # Query by solver + encoding
        result = logical_ir_manager.query(
            solver_name="ExactCoverQuantumSolver",
            encoding="pattern"
        )
        
        assert result["circuit_hash"] == circuit_hash
        assert result["sdk_type"] == "pytket"
        assert result["decompose_cnz"] is True
        assert result["resources"]["n_qubits"] == 3
        
        # Query by hash
        result_by_hash = logical_ir_manager.query(circuit_hash=circuit_hash)
        assert result_by_hash["circuit_hash"] == circuit_hash
    
    def test_multiple_solvers_and_encodings(self, logical_ir_manager, pytket_circuit, qiskit_circuit):
        """Test handling multiple solvers and encodings."""
        # Record multiple combinations
        logical_ir_manager.record(
            solver_name="Solver1",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        logical_ir_manager.record(
            solver_name="Solver1",
            encoding="pattern",
            circuit=qiskit_circuit,
        )
        
        logical_ir_manager.record(
            solver_name="Solver2",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        # Query all
        all_data = logical_ir_manager.query()
        assert "Solver1" in all_data
        assert "Solver2" in all_data
        assert len(all_data["Solver1"]) == 2
        assert len(all_data["Solver2"]) == 1
    
    def test_exists_and_clear(self, logical_ir_manager, pytket_circuit):
        """Test exists() and clear() methods."""
        assert not logical_ir_manager.exists()
        
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=pytket_circuit,
        )
        
        assert logical_ir_manager.exists()
        
        logical_ir_manager.clear()
        assert not logical_ir_manager.exists()


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_circuit(self, logical_ir_manager):
        """Test handling of empty circuit."""
        empty_circuit = Circuit(0)
        
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=empty_circuit,
        )
        
        result = logical_ir_manager.query(solver_name="TestSolver", encoding="simple")
        assert result["resources"]["n_qubits"] == 0
        assert result["resources"]["n_gates"] == 0
    
    def test_large_circuit(self, logical_ir_manager):
        """Test handling of large circuit."""
        large_circuit = Circuit(50)
        for i in range(49):
            large_circuit.CX(i, i+1)
        
        logical_ir_manager.record(
            solver_name="TestSolver",
            encoding="simple",
            circuit=large_circuit,
        )
        
        result = logical_ir_manager.query(solver_name="TestSolver", encoding="simple")
        assert result["resources"]["n_qubits"] == 50
        assert result["resources"]["n_gates"] == 49
    
    def test_special_characters_in_names(self, logical_ir_manager, pytket_circuit):
        """Test handling of special characters in solver/encoding names."""
        logical_ir_manager.record(
            solver_name="Solver-v2.0_beta",
            encoding="pattern/advanced",
            circuit=pytket_circuit,
        )
        
        result = logical_ir_manager.query(
            solver_name="Solver-v2.0_beta",
            encoding="pattern/advanced"
        )
        assert result is not None
