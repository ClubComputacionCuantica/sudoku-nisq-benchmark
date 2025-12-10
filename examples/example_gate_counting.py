"""Example script demonstrating the gate counting feature.

This script shows how to use the automatic gate counting feature that tracks
fundamental gate types during circuit construction.
"""

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

def main():
    print("=" * 70)
    print("Gate Counting Feature Demonstration")
    print("=" * 70)
    
    # Create a 2x2 Sudoku puzzle (4x4 grid)
    print("\n1. Generating a 2x2 Sudoku puzzle with 2 missing cells...")
    puzzle = QSudoku.generate(size=4, num_missing_cells=2)
    print(f"   Puzzle generated: {puzzle.board}")
    
    # Set up the exact cover solver
    print("\n2. Setting up ExactCoverQuantumSolver with 'simple' encoding...")
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    # Build circuit with PyTKET (gate counting happens automatically)
    print("\n3. Building quantum circuit with PyTKET...")
    circuit_pytket = puzzle.build_circuit(sdk="pytket")
    print(f"   Circuit: {circuit_pytket.n_qubits} qubits, {circuit_pytket.n_gates} gates")
    
    # Access gate counts
    gate_counts_pytket = puzzle._solver.gate_counts
    print("\n   Gate type breakdown (PyTKET):")
    for gate_name, count in sorted(gate_counts_pytket.items()):
        print(f"     {gate_name:8s}: {count}")
    
    # Build the same circuit with Qiskit for comparison
    print("\n4. Building quantum circuit with Qiskit...")
    puzzle2 = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle2.set_solver(ExactCoverQuantumSolver, encoding="simple")
    circuit_qiskit = puzzle2.build_circuit(sdk="qiskit")
    print(f"   Circuit: {circuit_qiskit.num_qubits} qubits")
    
    # Access gate counts
    gate_counts_qiskit = puzzle2._solver.gate_counts
    print("\n   Gate type breakdown (Qiskit):")
    for gate_name, count in sorted(gate_counts_qiskit.items()):
        print(f"     {gate_name:8s}: {count}")
    
    # Using the convenience method
    print("\n5. Accessing gate counts via get_gate_counts() method...")
    counts_via_method = puzzle._solver.get_gate_counts()
    print(f"   Total gate types tracked: {len(counts_via_method)}")
    print(f"   Total gates counted: {sum(counts_via_method.values())}")
    
    # Gate counts are included in metadata
    print("\n6. Gate counts are automatically included in circuit metadata!")
    print("   This enables tracking and analysis of gate usage across runs.")
    
    print("\n" + "=" * 70)
    print("Key Features:")
    print("=" * 70)
    print("✓ Automatic gate counting (enabled by default)")
    print("✓ Tracks fundamental gates: H, X, CX, CCX, C3X, ..., CnZ, Measure")
    print("✓ Works with both PyTKET and Qiskit implementations")
    print("✓ Counts gates directly during construction (most accurate)")
    print("✓ Accessible via solver.gate_counts or solver.get_gate_counts()")
    print("✓ Included in circuit metadata for persistence")
    print("=" * 70)

if __name__ == "__main__":
    main()