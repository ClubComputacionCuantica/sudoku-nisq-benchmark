#!/usr/bin/env python3
"""
Test to verify the duplication bug fix
"""

import sys
from pathlib import Path
import tempfile
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sudoku_nisq.metadata_manager import MetadataManager

def test_no_duplicate_rows():
    """Test that set_backend_resources only writes one CSV row per call"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "test_duplication.csv"
        
        # Create MetadataManager with CSV logging
        mm = MetadataManager(
            cache_base=tmpdir_path / "cache",
            puzzle_hash="test_duplication_hash",
            log_csv_path=csv_path
        )
        
        # Set puzzle fields
        mm.ensure_puzzle_fields(
            size=4,
            num_missing_cells=8,
            board=[[1, 0, 0, 4], [0, 3, 4, 0], [0, 4, 1, 0], [4, 0, 0, 1]]
        )
        
        # Set main circuit resources
        mm.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="default",
            resources={
                "n_qubits": 16,
                "n_gates": 45,
                "n_mcx_gates": 8,
                "depth": 12
            }
        )
        
        # Set backend resources - this should write exactly ONE row
        mm.set_backend_resources(
            solver_name="TestSolver",
            encoding="default",
            backend_alias="aer_simulator",
            opt_level=1,
            resources={
                "n_qubits": 16,
                "n_gates": 52,
                "depth": 15
            }
        )
        
        mm.save()
        
        # Check CSV content
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                rows = list(reader)
                assert len(rows) == 2  # Header + 1 data row
        
        # Verify no duplicate rows
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == len(set(tuple(row.items()) for row in rows)), "CSV should not contain duplicate rows"

def test_consistent_headers():
    """Test that CSV headers are consistent"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        csv_path = tmpdir_path / "test_headers.csv"
        
        mm = MetadataManager(
            cache_base=tmpdir_path / "cache",
            puzzle_hash="test_headers_hash",
            log_csv_path=csv_path
        )
        
        # Expected header order from _CSV_FIELDNAMES
        expected_headers = [
            "puzzle_hash", "size", "num_missing_cells",
            "solver_name", "encoding", "backend_alias", "opt_level",
            "main_n_qubits", "main_n_gates", "main_n_mcx_gates", "main_depth",
            "backend_n_qubits", "backend_n_gates", "backend_depth", "error"
        ]
        
        # Set up minimal data and trigger CSV write
        mm.ensure_puzzle_fields(size=2, num_missing_cells=1, board=[[1, 0], [0, 2]])
        mm.set_main_circuit_resources("TestSolver", "default", {"n_qubits": 4, "n_gates": 5, "n_mcx_gates": 1, "depth": 2})
        mm.set_backend_resources("TestSolver", "default", "test_backend", 0, {"n_qubits": 4, "n_gates": 7, "depth": 3})
        mm.save()
        
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.reader(f)
                actual_headers = next(reader)
                assert actual_headers == expected_headers
        else:
            print("❌ FAIL: CSV file was not created")

if __name__ == "__main__":
    print("Testing CSV duplication bug fix...")
    print("=" * 50)
    
    test_no_duplicate_rows()
    
    print("\n" + "=" * 50)
    print("Testing consistent headers...")
    
    test_consistent_headers()
    
    print("\n" + "=" * 50)
    print("Test complete!")
