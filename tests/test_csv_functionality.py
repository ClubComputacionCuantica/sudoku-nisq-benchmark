import tempfile
import shutil
from pathlib import Path
import pytest
import csv
from sudoku_nisq.metadata_manager import MetadataManager

@pytest.fixture
def temp_cache_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)

def test_csv_logging_disabled_by_default(temp_cache_dir):
    """Test that CSV logging is disabled when no log_csv_path is provided"""
    mm = MetadataManager(temp_cache_dir, "hash_csv1")
    mm.set_backend_resources("SolverA", "enc1", "backend1", 1, {"n_qubits": 5, "n_gates": 10, "depth": 3})
    mm.save()
    
    # No CSV should be created
    assert not any(temp_cache_dir.rglob("*.csv"))

def test_csv_logging_enabled(temp_cache_dir):
    """Test that CSV logging works when log_csv_path is provided"""
    csv_path = temp_cache_dir / "test_log.csv"
    mm = MetadataManager(temp_cache_dir, "hash_csv2", log_csv_path=csv_path)
    
    # Set puzzle fields
    mm.ensure_puzzle_fields(size=4, num_missing_cells=8, board=[[1,0,0,4],[0,3,4,0],[0,4,1,0],[4,0,0,1]])
    
    # Set main circuit resources
    mm.set_main_circuit_resources("TestSolver", "default", {
        "n_qubits": 16, "n_gates": 45, "n_mcx_gates": 8, "depth": 12
    })
    
    # Set backend resources (this should trigger CSV logging)
    mm.set_backend_resources("TestSolver", "default", "aer_simulator", 1, {
        "n_qubits": 16, "n_gates": 52, "depth": 15
    })
    
    mm.save()
    
    # Verify CSV was created
    assert csv_path.exists()
    
    # Verify CSV content
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        assert len(rows) == 1
        row = rows[0]

        # Check all expected columns exist
        expected_columns = [
            "puzzle_hash", "size", "num_missing_cells",
            "solver_name", "encoding", "backend_alias", "opt_level",
            "main_n_qubits", "main_n_gates", "main_n_mcx_gates", "main_depth",
            "backend_n_qubits", "backend_n_gates", "backend_depth"
        ]
        
        for col in expected_columns:
            assert col in row
        
        # Check values
        assert row["puzzle_hash"] == "hash_csv2"
        assert row["size"] == "4"
        assert row["num_missing_cells"] == "8"
        assert row["solver_name"] == "TestSolver"
        assert row["encoding"] == "default"
        assert row["backend_alias"] == "aer_simulator"
        assert row["opt_level"] == "1"
        assert row["main_n_qubits"] == "16"
        assert row["main_n_gates"] == "45"
        assert row["main_n_mcx_gates"] == "8"
        assert row["main_depth"] == "12"
        assert row["backend_n_qubits"] == "16"
        assert row["backend_n_gates"] == "52"
        assert row["backend_depth"] == "15"

def test_csv_multiple_backend_resources(temp_cache_dir):
    """Test CSV logging with multiple backend resources"""
    csv_path = temp_cache_dir / "multi_log.csv"
    mm = MetadataManager(temp_cache_dir, "hash_csv3", log_csv_path=csv_path)
    
    mm.ensure_puzzle_fields(size=2, num_missing_cells=2, board=[[1,0],[0,2]])
    mm.set_main_circuit_resources("Solver", "enc1", {"n_qubits": 4, "n_gates": 8, "n_mcx_gates": 2, "depth": 4})
    
    # Add multiple backend resources
    mm.set_backend_resources("Solver", "enc1", "backend1", 0, {"n_qubits": 4, "n_gates": 8, "depth": 4})
    mm.set_backend_resources("Solver", "enc1", "backend1", 1, {"n_qubits": 4, "n_gates": 10, "depth": 5})
    mm.set_backend_resources("Solver", "enc1", "backend2", 0, {"n_qubits": 4, "n_gates": 12, "depth": 6})
    
    mm.save()
    
    # Verify CSV has 3 rows (one for each backend resource)
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        assert len(rows) == 3
        
        # Check that all rows have the same puzzle and main circuit data
        for row in rows:
            assert row["puzzle_hash"] == "hash_csv3"
            assert row["solver_name"] == "Solver"
            assert row["encoding"] == "enc1"
            assert row["main_n_qubits"] == "4"
            assert row["main_n_gates"] == "8"

def test_export_full_metadata_csv(temp_cache_dir):
    """Test full metadata export to CSV"""
    mm = MetadataManager(temp_cache_dir, "hash_export")
    
    # Set up complex metadata
    mm.ensure_puzzle_fields(size=3, num_missing_cells=4, board=[[1,0,3],[0,2,0],[3,0,1]])
    
    # Multiple solvers and encodings
    mm.set_main_circuit_resources("SolverA", "enc1", {"n_qubits": 9, "n_gates": 20, "n_mcx_gates": 5, "depth": 8})
    mm.set_main_circuit_resources("SolverA", "enc2", {"n_qubits": 12, "n_gates": 25, "n_mcx_gates": 6, "depth": 10})
    mm.set_main_circuit_resources("SolverB", "enc1", {"n_qubits": 15, "n_gates": 30, "n_mcx_gates": 8, "depth": 12})
    
    # Backend resources
    mm.set_backend_resources("SolverA", "enc1", "backend1", 0, {"n_qubits": 9, "n_gates": 22, "depth": 9})
    mm.set_backend_resources("SolverA", "enc1", "backend1", 1, {"n_qubits": 9, "n_gates": 24, "depth": 10})
    mm.set_backend_resources("SolverA", "enc2", "backend2", 0, {"n_qubits": 12, "n_gates": 28, "depth": 11})
    
    mm.save()
    
    # Export to CSV
    export_path = temp_cache_dir / "full_export.csv"
    mm.export_full_metadata_csv(export_path)
    
    assert export_path.exists()
    
    with open(export_path, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        
        # Should have 3 rows with backend data + 1 for SolverB with no backend data
        assert len(rows) == 4
        
        # Find the row for SolverB (should have None for backend fields)
        solver_b_rows = [r for r in rows if r["solver_name"] == "SolverB"]
        assert len(solver_b_rows) == 1
        
        solver_b_row = solver_b_rows[0]
        assert solver_b_row["backend_alias"] == ""
        assert solver_b_row["opt_level"] == ""
        assert solver_b_row["main_n_qubits"] == "15"
        assert solver_b_row["backend_n_qubits"] == ""

def test_csv_headers_written_once(temp_cache_dir):
    """Test that CSV headers are only written for new files"""
    csv_path = temp_cache_dir / "headers_test.csv"
    
    # First MetadataManager instance
    mm1 = MetadataManager(temp_cache_dir / "cache1", "hash1", log_csv_path=csv_path)
    mm1.ensure_puzzle_fields(size=2, num_missing_cells=1, board=[[1,0],[0,2]])
    mm1.set_main_circuit_resources("Solver", "enc", {"n_qubits": 4, "n_gates": 8, "n_mcx_gates": 2, "depth": 4})
    mm1.set_backend_resources("Solver", "enc", "backend", 0, {"n_qubits": 4, "n_gates": 8, "depth": 4})
    mm1.save()
    
    # Second MetadataManager instance using same CSV file
    mm2 = MetadataManager(temp_cache_dir / "cache2", "hash2", log_csv_path=csv_path)
    mm2.ensure_puzzle_fields(size=2, num_missing_cells=1, board=[[2,0],[0,1]])
    mm2.set_main_circuit_resources("Solver", "enc", {"n_qubits": 4, "n_gates": 10, "n_mcx_gates": 3, "depth": 5})
    mm2.set_backend_resources("Solver", "enc", "backend", 1, {"n_qubits": 4, "n_gates": 12, "depth": 6})
    mm2.save()
    
    # Check that file has header only once but two data rows
    with open(csv_path, 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        
        # Should have 3 lines: 1 header + 2 data rows
        assert len(lines) == 3
        
        # First line should be header
        assert lines[0].startswith("puzzle_hash,")
        
        # Data rows should have different hashes
        assert "hash1" in lines[1]
        assert "hash2" in lines[2]
