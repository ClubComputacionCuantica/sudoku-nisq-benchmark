"""Example usage of the new Benchmark API for staged quantum hardware comparison."""

from sudoku_nisq import Benchmark


def example_full_pipeline():
    """Complete benchmark pipeline: Stage 0 → 1 using run_aer (no backend setup needed)."""
    print("\n" + "="*70)
    print("EXAMPLE: Full Benchmark Pipeline (using Aer)")
    print("="*70)
    
    # Create benchmark with small puzzle
    bench = Benchmark(size=4, missing_cells=6, encoding="simple")
    
    # Stage 0: Analyze logical circuit
    print("\n--- Running Stage 0 ---")
    logical = bench.analyze_logical_circuits()
    
    print(f"\nLogical Circuit Summary:")
    print(f"  Qubits: {logical.n_qubits}")
    print(f"  Gates: {logical.n_gates}")
    print(f"  Depth: {logical.depth}")
    
    print("\nNote: Stage 1-2 require registered backends.")
    print("For Aer simulation, use puzzle.run_aer() directly instead.")
    print("\n[OK] Benchmark Stage 0 complete!")


def example_feasibility_only():
    """Check feasibility without running hardware (Stages 0-1 only)."""
    print("\n" + "="*70)
    print("EXAMPLE: Feasibility Check (No Hardware Execution)")
    print("="*70)
    
    # Create benchmark with larger puzzle
    bench = Benchmark(size=9, missing_cells=20, encoding="pattern")
    
    # Stage 0
    logical = bench.analyze_logical_circuits()
    print(f"\nPuzzle requires {logical.n_qubits} qubits")
    print(f"\nNote: Stage 1 requires registered backends (IBM, Quantinuum, etc.)")
    print("This example demonstrates Stage 0 (logical analysis) only.")


def example_quick_compare():
    """Quick one-liner for logical analysis."""
    print("\n" + "="*70)
    print("EXAMPLE: Quick Logical Analysis")
    print("="*70)
    
    # Quick analysis
    bench = Benchmark(size=4, missing_cells=6)
    logical = bench.analyze_logical_circuits()
    
    print(f"\nQuick Analysis Complete:")
    print(f"  Puzzle: {logical.puzzle_size}x{logical.puzzle_size}")
    print(f"  Qubits needed: {logical.n_qubits}")
    print(f"  Total gates: {logical.n_gates}")


if __name__ == "__main__":
    # Run examples
    print("\n" + "#"*70)
    print("# Benchmark API Examples")
    print("#"*70)
    
    # Example 1: Full pipeline
    example_full_pipeline()
    
    # Example 2: Feasibility check only
    example_feasibility_only()
    
    # Example 3: Quick compare
    example_quick_compare()
    
    print("\n" + "#"*70)
    print("# All examples complete!")
    print("#"*70)
