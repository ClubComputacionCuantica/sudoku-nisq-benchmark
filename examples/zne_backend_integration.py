"""Example: ZNE Error Mitigation with Multi-Backend Support

This example demonstrates how Zero Noise Extrapolation (ZNE) integrates with
the backend system to provide provider-agnostic error mitigation.

Key Concepts:
- Mitiq never talks to backends directly
- Executors bridge Mitiq and your backend infrastructure
- Same mitigation code works with IBM, Quantinuum, AWS, simulators, etc.
- Backend switching is trivial - just change which backend you pass

Architecture:
    [Grover Circuit] → [Mitiq ZNE] → [Executor] → [Backend (IBM/Quantinuum/etc.)]
"""

from sudoku_nisq import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.backends import BackendManager
from sudoku_nisq.mitigation.expectation_wrapper import compute_success_expectation


def example_1_basic_zne():
    """Example 1: Basic ZNE usage with Aer simulator."""
    print("=" * 70)
    print("Example 1: Basic ZNE with Aer Simulator")
    print("=" * 70)
    
    # Generate a simple 2x2 Sudoku puzzle
    qs = QSudoku.generate(size=2, num_missing_cells=2, subgrid_size=1)
    print(f"Puzzle:\n{qs.puzzle}\n")
    
    # Set the exact cover solver
    qs.set_solver(ExactCoverQuantumSolver, encoding='simple')
    print(f"Solver: {qs._solver.__class__.__name__}")
    print(f"Universe size: {qs._solver.u_size}")
    print(f"Number of subsets: {qs._solver.s_size}\n")
    
    # Get Aer backend from BackendManager
    manager = BackendManager()
    # Note: You would normally register the backend first with:
    # manager.add_backend('ibm', 'aer', alias='aer')
    # For this example, we'll use the direct method:
    from pytket.extensions.qiskit import AerBackend
    aer = AerBackend()
    
    print("Running WITHOUT ZNE:")
    print("-" * 70)
    
    # Standard execution (no mitigation)
    result_raw = qs._solver.run(
        backend=aer,
        backend_alias='aer',
        shots=2048,
        use_zne=False  # Disabled
    )
    
    counts_raw = result_raw.get_counts()
    raw_prob = compute_success_expectation(counts_raw, qs._solver._is_valid_solution)
    
    print(f"Raw success probability: {raw_prob:.4f} ({raw_prob:.2%})")
    print(f"Top measurement: {max(counts_raw.items(), key=lambda x: x[1])}")
    
    print("\nRunning WITH ZNE:")
    print("-" * 70)
    
    # Mitigated execution with ZNE
    result_zne = qs._solver.run(
        backend=aer,
        backend_alias='aer',
        shots=2048,
        use_zne=True  # Enabled!
    )
    
    counts_zne = result_zne.get_counts()
    mitigated_prob = result_zne._mitigated_success_prob
    
    print(f"Mitigated success probability: {mitigated_prob:.4f} ({mitigated_prob:.2%})")
    print(f"Top measurement: {max(counts_zne.items(), key=lambda x: x[1])}")
    
    improvement = mitigated_prob - raw_prob
    print(f"\n✨ Improvement: {improvement:+.4f} ({improvement:+.2%})")
    print()


def example_2_multi_backend():
    """Example 2: Same mitigation code works with different backends."""
    print("=" * 70)
    print("Example 2: Multi-Backend Support (Provider-Agnostic)")
    print("=" * 70)
    
    # Generate puzzle
    qs = QSudoku.generate(size=2, num_missing_cells=2, subgrid_size=1)
    qs.set_solver(ExactCoverQuantumSolver, encoding='simple')
    
    # Initialize backend manager
    manager = BackendManager()
    
    # Demonstration: How to use different backends
    # (Commented out as they require authentication)
    
    print("Backend switching demonstration:")
    print("-" * 70)
    
    backends_demo = [
        ("Aer Simulator", "aer", "Local ideal simulator"),
        ("IBM Brisbane", "ibm_brisbane", "127-qubit IBM Quantum chip"),
        ("Quantinuum H1-1", "H1-1", "Quantinuum trapped-ion computer"),
        ("AWS Rigetti", "aspen_m3", "Rigetti superconducting processor"),
    ]
    
    for name, alias, description in backends_demo:
        print(f"\n{name} ({alias}):")
        print(f"  Description: {description}")
        print(f"  Usage:")
        print(f"    backend = manager.get('{alias}')")
        print(f"    result = qs._solver.run(backend, '{alias}', use_zne=True)")
        print(f"  → Mitiq handles the rest automatically!")
    
    print("\n" + "=" * 70)
    print("Key Insight: Same ZNE code, different backends!")
    print("The executor pattern abstracts away provider differences.")
    print("=" * 70)
    print()


def example_3_executor_internals():
    """Example 3: Understanding what executors do under the hood."""
    print("=" * 70)
    print("Example 3: Executor Internals (How Mitiq Talks to Backends)")
    print("=" * 70)
    
    print("\nThe executor is YOUR bridge between Mitiq and backends:")
    print("-" * 70)
    
    print("""
    def executor(circuit):
        # Step 1: Run circuit on backend (standard pytket pattern)
        handle = backend.process_circuit(circuit, n_shots=shots)
        result = backend.get_result(handle)
        counts = result.get_counts()
        
        # Step 2: Compute success probability (new for mitigation)
        success_prob = 0.0
        total = sum(counts.values())
        for bitstring, count in counts.items():
            if solver._is_valid_solution(bitstring):
                success_prob += count / total
        
        # Step 3: Return scalar expectation (what Mitiq needs)
        return success_prob  # Float in [0, 1]
    """)
    
    print("\nMitiq then:")
    print("  1. Scales noise (gate folding) → creates modified circuits")
    print("  2. Calls executor(modified_circuit) for each noise level")
    print("  3. Collects noisy success probabilities")
    print("  4. Extrapolates to zero-noise estimate")
    print("  5. Returns mitigated success probability")
    
    print("\nWhy this works across providers:")
    print("-" * 70)
    print("  ✅ All backends expose: process_circuit() and get_result()")
    print("  ✅ All results provide: get_counts()")
    print("  ✅ All use pytket as common interface")
    print("  ✅ Provider details hidden inside backend implementation")
    print()


def example_4_cost_analysis():
    """Example 4: Understanding mitigation costs."""
    print("=" * 70)
    print("Example 4: ZNE Cost Analysis")
    print("=" * 70)
    
    print("\nZNE Cost Structure:")
    print("-" * 70)
    
    shots = 4096
    scale_factors = [1, 3, 5]  # Default ZNE scales
    
    print(f"Shots per circuit: {shots}")
    print(f"Noise scale factors: {scale_factors}")
    print(f"Number of circuits: {len(scale_factors)}")
    print(f"\nTotal shots: {shots * len(scale_factors):,}")
    print(f"Cost multiplier: {len(scale_factors)}x vs raw execution")
    
    print("\nWhat each scale factor means:")
    print("  Scale 1: Original circuit (baseline noise)")
    print("  Scale 3: Gates folded to 3x noise")
    print("  Scale 5: Gates folded to 5x noise")
    print("\n  → Extrapolate back to scale 0 (zero noise)")
    
    print("\nComparison:")
    print("-" * 70)
    print(f"  Raw execution:     {shots:,} shots")
    print(f"  ZNE execution:     {shots * len(scale_factors):,} shots ({len(scale_factors)}x)")
    print(f"  PEC execution:     ~{shots * 100:,} shots (~100x, highly variable)")
    
    print("\n💡 Recommendation: Start with ZNE for cost-effectiveness!")
    print()


if __name__ == '__main__':
    # Run all examples
    example_1_basic_zne()
    example_2_multi_backend()
    example_3_executor_internals()
    example_4_cost_analysis()
    
    print("=" * 70)
    print("✅ All examples complete!")
    print("=" * 70)
    print("\nNext Steps:")
    print("  1. Try with real hardware by authenticating with IBM/Quantinuum")
    print("  2. Experiment with different shot counts")
    print("  3. Compare ZNE vs raw results on noisy simulators")
    print("  4. Explore PEC for unbiased estimates (when you have noise models)")
