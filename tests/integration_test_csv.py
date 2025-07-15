#!/usr/bin/env python3
"""
Integration test demonstrating the complete CSV export functionality
"""
import pytest
import sys
from pathlib import Path
import tempfile
import csv

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from sudoku_nisq.metadata_manager import MetadataManager
from sudoku_nisq.quantum_solver import QuantumSolver
from sudoku_nisq.backends import BackendManager

class MockSudoku:
    """Mock QSudoku for demonstration"""
    def __init__(self, board, hash_value):
        self.board = board
        self.board_size = len(board)
        self.num_missing_cells = sum(row.count(0) for row in board)
        self._hash = hash_value
    
    def get_hash(self):
        return self._hash


def test_basic_csv_logging():
    """Test basic CSV logging functionality."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create test puzzle
        puzzle = MockSudoku([[1, 0, 3, 4], [0, 3, 4, 0], [0, 4, 1, 0], [4, 0, 0, 1]], "puzzle_hash_1")
        
        # Test basic CSV logging
        csv_path = tmpdir / "basic_test.csv"
        
        mm = MetadataManager(
            cache_base=tmpdir / "cache",
            puzzle_hash=puzzle.get_hash(),
            log_csv_path=csv_path
        )
        
        # Set puzzle fields
        mm.ensure_puzzle_fields(
            size=puzzle.board_size,
            num_missing_cells=puzzle.num_missing_cells,
            board=puzzle.board
        )
        
        # Set main circuit resources
        mm.set_main_circuit_resources(
            solver_name="TestSolver",
            encoding="basic_encoding",
            resources={
                "n_qubits": 16,
                "n_gates": 45,
                "n_mcx_gates": 8,
                "depth": 12
            }
        )
        
        # Set backend resources (this should trigger CSV logging)
        mm.set_backend_resources(
            solver_name="TestSolver",
            encoding="basic_encoding",
            backend_alias="simulator",
            opt_level=1,
            resources={
                "n_qubits": 16,
                "n_gates": 52,
                "depth": 15
            }
        )
        
        mm.save()
        
        # Verify CSV was created and has content
        assert csv_path.exists()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 1
        row = rows[0]
        assert row['puzzle_hash'] == "puzzle_hash_1"
        assert row['solver_name'] == "TestSolver"
        assert row['backend_alias'] == "simulator"
        assert row['opt_level'] == "1"


def test_multiple_solvers_csv():
    """Test CSV logging with multiple solvers."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        puzzle = MockSudoku([[1, 0], [0, 2]], "puzzle_hash_2")
        csv_path = tmpdir / "multi_test.csv"
        
        mm = MetadataManager(
            cache_base=tmpdir / "cache",
            puzzle_hash=puzzle.get_hash(),
            log_csv_path=csv_path
        )
        
        # Set puzzle fields
        mm.ensure_puzzle_fields(
            size=puzzle.board_size,
            num_missing_cells=puzzle.num_missing_cells,
            board=puzzle.board
        )
        
        # Add multiple solvers
        for solver_name in ["SolverA", "SolverB"]:
            mm.set_main_circuit_resources(
                solver_name=solver_name,
                encoding="default",
                resources={"n_qubits": 4, "n_gates": 10, "depth": 3}
            )
            
            mm.set_backend_resources(
                solver_name=solver_name,
                encoding="default",
                backend_alias="simulator",
                opt_level=1,
                resources={"n_qubits": 4, "n_gates": 12, "depth": 4}
            )
        
        mm.save()
        
        # Verify CSV has multiple rows
        assert csv_path.exists()
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        solver_names = {row['solver_name'] for row in rows}
        assert solver_names == {"SolverA", "SolverB"}


if __name__ == "__main__":
    test_basic_csv_logging()
    test_multiple_solvers_csv()
    print("All tests passed!")
