"""Example showing gate counting with optional CnZ decomposition."""

from sudoku_nisq.q_sudoku import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver

def main():
    print("=" * 70)
    print("Gate Counting: CnZ Decomposition Options")
    print("=" * 70)
    
    # Create a 2x2 Sudoku puzzle
    puzzle1 = QSudoku.generate(size=4, num_missing_cells=2)
    puzzle2 = QSudoku.generate(size=4, num_missing_cells=2)
    
    # Option 1: Default behavior (decompose_cnz=True) - counts at fundamental level
    print("\n1. Default: decompose_cnz=True (Consistent with Qiskit)")
    print("   Counts CnZ gates as their H+MCX+H decomposition")
    puzzle1.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=True)
    circuit1 = puzzle1.build_circuit(sdk="pytket")
    counts1 = puzzle1._solver.gate_counts
    
    print("\n   PyTKET gate counts (decomposed):")
    for gate_name, count in sorted(counts1.items()):
        print(f"     {gate_name:8s}: {count}")
    
    # Option 2: Native counting (decompose_cnz=False) - counts PyTKET's native gates
    print("\n2. Optional: decompose_cnz=False (Native PyTKET gates)")
    print("   Counts CnZ as a single gate operation")
    puzzle2.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=False)
    circuit2 = puzzle2.build_circuit(sdk="pytket")
    counts2 = puzzle2._solver.gate_counts
    
    print("\n   PyTKET gate counts (native):")
    for gate_name, count in sorted(counts2.items()):
        print(f"     {gate_name:8s}: {count}")
    
    # Compare
    print("\n" + "=" * 70)
    print("Comparison:")
    print("=" * 70)
    all_gates = set(counts1.keys()) | set(counts2.keys())
    for gate in sorted(all_gates):
        c1 = counts1.get(gate, 0)
        c2 = counts2.get(gate, 0)
        diff = "" if c1 == c2 else f"  ← Difference: {c1 - c2:+d}"
        print(f"  {gate:8s}: decomposed={c1:2d}, native={c2:2d}{diff}")
    
    print("\n" + "=" * 70)
    print("Key Points:")
    print("=" * 70)
    print("✓ Default (decompose_cnz=True): Consistent gate counts across SDKs")
    print("  - PyTKET and Qiskit will report identical fundamental gate counts")
    print("  - Best for cross-SDK comparison and algorithm analysis")
    print()
    print("✓ Optional (decompose_cnz=False): Native PyTKET gate representation")
    print("  - Shows CnZ as a single native gate operation")
    print("  - Useful for SDK-specific optimizations")
    print("=" * 70)

if __name__ == "__main__":
    main()
