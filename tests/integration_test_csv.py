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
from sudoku_nisq.backend_manager import BackendManager

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
        print("  Building main circuit...")
        solver1.build_main_circuit()
        
        print("  Transpiling for simulator...")
        solver1.transpile_and_analyze("simulator", [0, 1, 2])
        
        print("  Transpiling for hardware...")
        solver1.transpile_and_analyze("hardware", [0, 1])
        
        # Check CSV
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                print(f"  ✅ CSV created with {len(rows)} rows")
                
                if rows:
                    print(f"  Sample row keys: {list(rows[0].keys())}")
                    print(f"  First row puzzle_hash: {rows[0]['puzzle_hash']}")
                    print(f"  First row solver_name: {rows[0]['solver_name']}")
        
        # Test 2: High-volume mode (no transpiled circuit storage)
        print("\n⚡ Test 2: High-volume mode (store_transpiled=False)")
        
        solver2 = DemoSolver(
            sudoku=puzzle2,
            encoding="volume_encoding", 
            cache_base_dir=tmpdir / "cache2",
            save_csv=True,
            csv_path=csv_path,  # Append to same CSV
            store_transpiled=False  # Don't store transpiled circuits
        )
        
        solver2.backends["simulator"] = MockBackend("simulator")
        
        print("  Building main circuit...")
        solver2.build_main_circuit()
        
        print("  Transpiling (no circuit storage)...")
        solver2.transpile_and_analyze("simulator", [0, 1, 2, 3])
        
        # Check that main circuit is stored but transpiled circuits are not
        main_circuit_path = solver2.main_circuit_path
        transpiled_path = solver2.transpiled_circuit_path("simulator", 0)
        
        print(f"  Main circuit stored: {main_circuit_path.exists()}")
        print(f"  Transpiled circuit stored: {transpiled_path.exists()}")
        
        # Test 3: Manual CSV export
        print("\n📊 Test 3: Manual metadata export")
        export_path = tmpdir / "manual_export.csv"
        solver1.export_metadata_csv(export_path)
        
        if export_path.exists():
            with open(export_path, 'r') as f:
                reader = csv.DictReader(f)
                export_rows = list(reader)
                print(f"  ✅ Manual export created with {len(export_rows)} rows")
        
        # Test 4: Check final CSV content
        print("\n📈 Test 4: Final CSV analysis")
        
        if csv_path.exists():
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                all_rows = list(reader)
                
                print(f"  Total rows in main CSV: {len(all_rows)}")
                
                # Analyze by puzzle
                puzzle_counts = {}
                solver_counts = {}
                backend_counts = {}
                
                for row in all_rows:
                    puzzle = row['puzzle_hash']
                    solver = row['solver_name']
                    backend = row['backend_alias']
                    
                    puzzle_counts[puzzle] = puzzle_counts.get(puzzle, 0) + 1
                    solver_counts[solver] = solver_counts.get(solver, 0) + 1
                    backend_counts[backend] = backend_counts.get(backend, 0) + 1
                
                print(f"  Rows per puzzle: {puzzle_counts}")
                print(f"  Rows per solver: {solver_counts}")
                print(f"  Rows per backend: {backend_counts}")
                
                # Show sample data
                if all_rows:
                    sample = all_rows[0]
                    print(f"\n  Sample row data:")
                    for key in ['puzzle_hash', 'solver_name', 'encoding', 'backend_alias', 'opt_level', 
                               'main_n_qubits', 'main_n_gates', 'backend_n_qubits', 'backend_n_gates']:
                        print(f"    {key}: {sample.get(key, 'N/A')}")
        
        print("\n✅ Integration test completed successfully!")
        print("\nKey features demonstrated:")
        print("  - Automatic CSV logging during transpilation")
        print("  - Row-by-row appending with proper headers")
        print("  - High-volume mode with selective caching")
        print("  - Manual metadata export")
        print("  - Multiple puzzles/solvers in same CSV")

        # Verify CSV content
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) > 0, "CSV should contain at least one row"
            assert "size" in rows[0], "CSV should include 'size' column"
            assert "num_missing_cells" in rows[0], "CSV should include 'num_missing_cells' column"

# Mock solver and backend
solver1 = QuantumSolver(sudoku=puzzle, encoding="default")
DemoSolver = QuantumSolver
MockBackend = lambda alias: BackendManager.get(alias)

if __name__ == "__main__":
    main()
