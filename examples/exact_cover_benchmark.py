"""
Benchmark: Generic Exact Cover vs Sudoku 2×2

This example demonstrates why general exact cover instances can be useful for 
benchmarking quantum algorithms when even minimal Sudoku puzzles are too large.

Key points:
1. A 2×2 Sudoku (smallest possible) requires significant quantum resources
2. Smaller generic exact cover instances can succeed where Sudoku fails
3. This allows testing/validating algorithms on near-term quantum devices

Mathematical insight:
Every Sudoku puzzle is an exact cover instance, but the converse is not true.
The space of all exact cover problems is much larger, and includes many
instances smaller than the minimal Sudoku.
"""

from sudoku_nisq import QSudoku, QExactCover, ExactCoverProblem, ExactCoverQuantumSolver


def compare_resources():
    """Compare quantum resources for generic vs Sudoku exact cover instances."""
    
    print("=" * 70)
    print("QUANTUM RESOURCE COMPARISON: Generic Exact Cover vs Sudoku")
    print("=" * 70)
    print()
    
    # 1. Create a small generic exact cover instance
    print("1. Small Generic Exact Cover Instance")
    print("-" * 70)
    
    # Classic example: cover {0,1,2,3} with 3 subsets
    universe = [0, 1, 2, 3]
    subsets = {
        'S_0': [0, 3],      # Covers elements 0 and 3
        'S_1': [1, 2],      # Covers elements 1 and 2
        'S_2': [0, 1, 2],   # Alternative cover for 0,1,2
    }
    
    problem = ExactCoverProblem(
        universe=universe,
        subsets=subsets,
        num_solutions=1,
        metadata={'description': 'Minimal 4-element instance'}
    )
    
    print(f"Universe: {problem.universe}")
    print(f"Subsets: {problem.subsets}")
    print(f"Problem hash: {problem.get_hash()[:16]}...")
    print()
    
    # Canonicalize to show structure
    n, canonical_subsets = problem.canonicalize()
    print(f"Canonical form: U_{n} = {{0, ..., {n-1}}}")
    print(f"Canonical subsets: {canonical_subsets}")
    print()
    
    # Create quantum solver
    qec = QExactCover(problem)
    resources_generic = qec.report_resources()
    
    print("Resource Requirements:")
    print(f"  Universe size: {resources_generic['problem']['universe_size']}")
    print(f"  Number of subsets: {resources_generic['problem']['num_subsets']}")
    print(f"  Estimated qubits: {resources_generic['estimated']['n_qubits']}")
    print(f"  Estimated gates: {resources_generic['estimated']['n_gates']:,}")
    print()
    print()
    
    # 2. Compare with minimal Sudoku (2×2)
    print("2. Minimal Sudoku (2×2 grid)")
    print("-" * 70)
    
    sudoku = QSudoku.generate(size=4, num_missing_cells=2)
    sudoku.set_solver(ExactCoverQuantumSolver, encoding="simple")
    
    print(f"Board size: {sudoku.board_size}×{sudoku.board_size}")
    print(f"Missing cells: {sudoku.num_missing_cells}")
    print("Puzzle board:")
    for row in sudoku.board:
        print(f"  {row}")
    print()
    
    resources_sudoku = sudoku.report_resources()
    
    # Extract estimated resources from sudoku structure
    solver_name = "ExactCoverQuantumSolver"
    encoding_name = "simple"
    if solver_name in resources_sudoku.get("solvers", {}):
        main_circuit = resources_sudoku["solvers"][solver_name][encoding_name]["main_circuit"]
        sudoku_qubits = main_circuit["n_qubits"]
        sudoku_gates = main_circuit["n_gates"]
    else:
        # Fallback to direct estimation
        sudoku_qubits = sudoku._solver.resource_estimation()["n_qubits"]
        sudoku_gates = sudoku._solver.resource_estimation()["n_gates"]
    
    print("Resource Requirements:")
    print(f"  Estimated qubits: {sudoku_qubits}")
    print(f"  Estimated gates: {sudoku_gates:,}")
    print()
    print()
    
    # 3. Summary comparison
    print("3. Side-by-Side Comparison")
    print("=" * 70)
    print(f"{'Metric':<30} {'Generic EC':<20} {'Sudoku 2×2':<20}")
    print("-" * 70)
    
    generic_qubits = resources_generic['estimated']['n_qubits']
    generic_gates = resources_generic['estimated']['n_gates']
    
    print(f"{'Qubits':<30} {generic_qubits:<20} {sudoku_qubits:<20}")
    print(f"{'Gates (estimated)':<30} {generic_gates:<20,} {sudoku_gates:<20,}")
    print(f"{'Reduction factor':<30} {'1×':<20} {f'{sudoku_qubits/generic_qubits:.1f}×':<20}")
    print()
    
    print("Conclusion:")
    print(f"  The generic exact cover instance is {sudoku_qubits/generic_qubits:.1f}× smaller in qubits")
    print(f"  and {sudoku_gates/generic_gates:.1f}× smaller in gate count than minimal Sudoku.")
    print("  This makes it feasible to test/validate algorithms on near-term devices!")
    print()
    print()


def demonstrate_generic_execution():
    """Run a generic exact cover instance on Qiskit Aer."""
    
    print("=" * 70)
    print("RUNNING GENERIC EXACT COVER INSTANCE")
    print("=" * 70)
    print()
    
    # Use the built-in small example
    qec = QExactCover.create_small_example()
    
    print(f"Problem: {qec.problem.metadata.get('description', 'N/A')}")
    print(f"Universe: {qec.universe}")
    print(f"Subsets: {qec.subsets}")
    print(f"Expected solutions: {qec.num_solutions}")
    print()
    
    # Build circuit
    print("Building quantum circuit...")
    circuit = qec.build_circuit(sdk="qiskit")
    print(f"  Circuit qubits: {circuit.num_qubits}")
    print(f"  Circuit depth: {circuit.depth()}")
    print(f"  Circuit size: {circuit.size()} gates")
    print()
    
    # Run on Aer
    print("Running on Qiskit Aer simulator (shots=1024)...")
    result = qec.run_aer(shots=1024, opt_level=1)
    print(f"  Execution successful: {result['metadata']['success']}")
    print()
    
    # Display top results
    print("Top measurement outcomes:")
    counts = result['counts']
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    
    for i, (bitstring, count) in enumerate(sorted_counts[:5], 1):
        # Decode bitstring to selected subsets
        selected = [f"S_{j}" for j, bit in enumerate(bitstring) if bit == '1']
        probability = count / 1024
        
        print(f"  {i}. {bitstring} (count: {count:4d}, prob: {probability:.3f})")
        print(f"     Selected subsets: {selected}")
    
    print()
    print("[SUCCESS] Generic exact cover execution completed successfully!")
    print()
    print()


def enumerate_small_instances():
    """Demonstrate enumerating small exact cover instances."""
    
    print("=" * 70)
    print("ENUMERATING SMALL EXACT COVER INSTANCES")
    print("=" * 70)
    print()
    
    print("Generating all 2×2 exact cover instances (16 total)...")
    print()
    
    count = 0
    for problem in ExactCoverProblem.enumerate_instances(max_n=2, max_m=2, max_total=4):
        count += 1
        if count <= 5:  # Show first 5
            n, canonical = problem.canonicalize()
            print(f"Instance {count}:")
            print(f"  Universe: {problem.universe}")
            print(f"  Subsets: {problem.subsets}")
            print(f"  Canonical: U_{n}, {canonical}")
            print()
    
    print(f"... (showing 5 of {count} instances)")
    print()
    print(f"Total instances generated: {count}")
    print()
    print("Each of these could be used for quantum algorithm testing!")
    print()


def main():
    """Run all benchmark comparisons and demonstrations."""
    
    print("\n")
    print("=" * 70)
    print(" " * 10 + "EXACT COVER BENCHMARK: Generic vs Sudoku")
    print("=" * 70)
    print()
    
    try:
        # 1. Resource comparison
        compare_resources()
        
        # 2. Actual execution
        demonstrate_generic_execution()
        
        # 3. Enumeration demo
        enumerate_small_instances()
        
        print("=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print()
        print("This benchmark demonstrates three key insights:")
        print()
        print("1. RESOURCE ADVANTAGE: Generic exact cover instances can be")
        print("   significantly smaller than minimal Sudoku puzzles, making")
        print("   them suitable for near-term quantum devices.")
        print()
        print("2. ALGORITHM VALIDATION: Smaller instances allow testing/")
        print("   validating quantum exact cover algorithms before scaling")
        print("   to full Sudoku problems.")
        print()
        print("3. PROBLEM SPACE: The space of exact cover problems is vast,")
        print("   and Sudoku represents only a structured subfamily. Exploring")
        print("   generic instances provides broader benchmarking coverage.")
        print()
        print("Next steps:")
        print("  - Use QExactCover for algorithm development/testing")
        print("  - Scale to larger instances as quantum hardware improves")
        print("  - Compare performance across different problem structures")
        print()
        
    except Exception as e:
        print(f"\n[ERROR] Error during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
