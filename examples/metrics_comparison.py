"""Encoding comparison example: Simple vs Pattern encoding metrics.

This example demonstrates how to compare quantum performance across different
encoding strategies using the metrics API.

Key features:
- Side-by-side comparison of simple and pattern encodings
- Same puzzle, same shots, different encodings
- Resource usage vs performance trade-offs
- Visualization-ready data output

Prerequisites:
    - New metadata architecture enabled (SUDOKU_NISQ_NEW_METADATA=1)
    - matplotlib (for optional plotting)
    
Expected outcome:
    This script will:
    1. Generate a single 2×2 Sudoku puzzle
    2. Run with simple encoding
    3. Run with pattern encoding
    4. Compare metrics side-by-side
    5. Analyze resource efficiency trade-offs
"""

import os
from sudoku_nisq import QSudoku
from sudoku_nisq.solvers import ExactCoverQuantumSolver
from sudoku_nisq.metadata.config import MetadataConfig

# Enable new metadata architecture
MetadataConfig.ENABLE_NEW_ARCHITECTURE = True


def run_with_encoding(puzzle, encoding_type, shots=1024):
    """Run quantum execution with specified encoding and return metrics."""
    print(f"\n{'='*70}")
    print(f"TESTING: {encoding_type.upper()} ENCODING")
    print(f"{'='*70}")
    
    # Configure solver with encoding
    print(f"\nStep 1: Configuring {encoding_type} encoding...")
    puzzle.set_solver(ExactCoverQuantumSolver, encoding=encoding_type, decompose_cnz=True)
    
    # Build circuit
    print("Step 2: Building quantum circuit...")
    puzzle.build_circuit(sdk="qiskit")
    
    # Get resources
    resources = puzzle.report_resources()
    main_circuit = (
        resources.get("solvers", {})
        .get("ExactCoverQuantumSolver", {})
        .get(encoding_type, {})
        .get("main_circuit", {})
    )
    
    n_qubits = main_circuit.get('n_qubits', 'N/A')
    n_gates = main_circuit.get('n_gates', 'N/A')
    depth = main_circuit.get('depth', 'N/A')
    two_q_gates = main_circuit.get('two_qubit_gates', 'N/A')
    
    print(f"  Qubits: {n_qubits}")
    print(f"  Total gates: {n_gates}")
    print(f"  Circuit depth: {depth}")
    print(f"  2Q gates: {two_q_gates}")
    
    # Set up validation context for this encoding
    print("Step 3: Setting up validation...")
    context = puzzle.puzzle.create_validation_context(encoding_type=encoding_type)
    puzzle.set_validation_context(context.valid_solutions)
    print(f"  ✓ {len(context.valid_solutions)} valid solution(s) enumerated")
    
    # Run execution
    print(f"Step 4: Executing with {shots} shots...")
    result = puzzle.run_aer(shots=shots, method="statevector")
    print(f"  ✓ Measured {len(result.get_counts())} unique outcomes")
    
    # Get metrics
    print("Step 5: Retrieving metrics...")
    metrics = puzzle.calculate_metrics()
    
    if metrics:
        print("  ✓ Metrics computed successfully")
        return {
            'encoding': encoding_type,
            'resources': {
                'qubits': n_qubits,
                'gates': n_gates,
                'depth': depth,
                'two_q_gates': two_q_gates
            },
            'metrics': metrics
        }
    else:
        print("  ⚠ No metrics available")
        return None


def compare_results(simple_result, pattern_result):
    """Compare metrics between two encoding strategies."""
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    
    # Extract data
    simple_metrics = simple_result['metrics']
    pattern_metrics = pattern_result['metrics']
    simple_res = simple_result['resources']
    pattern_res = pattern_result['resources']
    
    # Resource comparison
    print("\n1. RESOURCE USAGE")
    print("-" * 70)
    print(f"{'Metric':<20} {'Simple':<15} {'Pattern':<15} {'Difference'}")
    print("-" * 70)
    
    def compare_metric(name, simple_val, pattern_val):
        diff = pattern_val - simple_val
        diff_pct = (diff / simple_val * 100) if simple_val > 0 else 0
        arrow = "↑" if diff > 0 else "↓" if diff < 0 else "="
        print(f"{name:<20} {simple_val:<15} {pattern_val:<15} {arrow} {abs(diff_pct):.1f}%")
    
    compare_metric("Qubits", simple_res['qubits'], pattern_res['qubits'])
    compare_metric("Total Gates", simple_res['gates'], pattern_res['gates'])
    compare_metric("Circuit Depth", simple_res['depth'], pattern_res['depth'])
    compare_metric("2Q Gates", simple_res['two_q_gates'], pattern_res['two_q_gates'])
    
    # Performance comparison
    print("\n2. PERFORMANCE METRICS")
    print("-" * 70)
    print(f"{'Metric':<20} {'Simple':<15} {'Pattern':<15} {'Better'}")
    print("-" * 70)
    
    # Success probability
    simple_p = simple_metrics.get('p_succ', 0)
    pattern_p = pattern_metrics.get('p_succ', 0)
    winner = "Simple" if simple_p > pattern_p else "Pattern" if pattern_p > simple_p else "Tie"
    print(f"{'Success Prob':<20} {simple_p:<15.4f} {pattern_p:<15.4f} {winner}")
    
    # Valid odds
    simple_odds = simple_metrics.get('valid_odds')
    pattern_odds = pattern_metrics.get('valid_odds')
    if simple_odds and pattern_odds:
        winner = "Simple" if simple_odds > pattern_odds else "Pattern" if pattern_odds > simple_odds else "Tie"
        print(f"{'Valid Odds':<20} {simple_odds:<15.2f} {pattern_odds:<15.2f} {winner}")
    
    # Retention per gate
    simple_ret = simple_metrics.get('retention_per_2q')
    pattern_ret = pattern_metrics.get('retention_per_2q')
    if simple_ret and pattern_ret:
        winner = "Simple" if simple_ret > pattern_ret else "Pattern" if pattern_ret > simple_ret else "Tie"
        print(f"{'Retention/Gate':<20} {simple_ret:<15.6f} {pattern_ret:<15.6f} {winner}")
    
    # Shot budget
    simple_shots = simple_metrics.get('shots_detect_point')
    pattern_shots = pattern_metrics.get('shots_detect_point')
    if simple_shots and pattern_shots:
        winner = "Simple" if simple_shots < pattern_shots else "Pattern" if pattern_shots < simple_shots else "Tie"
        print(f"{'Shots (95% det)':<20} {simple_shots:<15} {pattern_shots:<15} {winner}")
    
    # Overall analysis
    print("\n3. TRADE-OFF ANALYSIS")
    print("-" * 70)
    
    # Resource efficiency
    simple_eff = simple_p / simple_res['two_q_gates'] if simple_res['two_q_gates'] > 0 else 0
    pattern_eff = pattern_p / pattern_res['two_q_gates'] if pattern_res['two_q_gates'] > 0 else 0
    
    print(f"Resource Efficiency (p_succ / 2Q gates):")
    print(f"  Simple:  {simple_eff:.6f}")
    print(f"  Pattern: {pattern_eff:.6f}")
    
    if pattern_eff > simple_eff * 1.1:
        print("  → Pattern encoding is significantly more efficient")
    elif simple_eff > pattern_eff * 1.1:
        print("  → Simple encoding is significantly more efficient")
    else:
        print("  → Encodings have similar efficiency")
    
    # Depth vs performance
    depth_ratio = pattern_res['depth'] / simple_res['depth'] if simple_res['depth'] > 0 else 0
    perf_ratio = pattern_p / simple_p if simple_p > 0 else 0
    
    print(f"\nDepth vs Performance:")
    print(f"  Pattern depth is {depth_ratio:.2f}× simple depth")
    print(f"  Pattern p_succ is {perf_ratio:.2f}× simple p_succ")
    
    if perf_ratio > depth_ratio:
        print("  → Pattern encoding provides better performance than expected from depth increase")
    elif perf_ratio < 1/depth_ratio:
        print("  → Pattern encoding suffers worse performance than expected from depth increase")
    else:
        print("  → Performance scales roughly proportionally with depth")
    
    print("\n" + "=" * 70)


def main():
    print("=" * 70)
    print("Encoding Comparison Example: Simple vs Pattern")
    print("=" * 70)
    print()
    
    # Generate a single puzzle for fair comparison
    print("Generating 2×2 Sudoku puzzle for comparison...")
    puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=2, canonicalize=True)
    print(f"Puzzle hash: {puzzle.get_hash()[:12]}...")
    print(f"Board:\n{puzzle}")
    
    # Run with simple encoding
    simple_result = run_with_encoding(puzzle, encoding_type="simple", shots=1024)
    
    # Run with pattern encoding
    pattern_result = run_with_encoding(puzzle, encoding_type="pattern", shots=1024)
    
    # Compare results
    if simple_result and pattern_result:
        compare_results(simple_result, pattern_result)
    else:
        print("\n⚠ Could not complete comparison (missing metrics)")
    
    print()
    print("✅ Comparison example completed successfully!")
    print()
    print("Tip: Try different puzzle sizes to see how encoding trade-offs")
    print("     change with problem scale. Pattern encoding often shines")
    print("     on larger, more constrained puzzles.")


if __name__ == "__main__":
    main()
