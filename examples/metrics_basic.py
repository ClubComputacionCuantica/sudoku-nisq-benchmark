"""Basic metrics example: Single-run workflow with automatic validation.

This example demonstrates the simplest possible workflow for computing metrics
on a quantum execution using the new Phase 3 usability API.

Key features:
- Automatic solution enumeration
- One-line validation context creation
- Automatic metrics computation
- Clean programmatic access to results

Prerequisites:
    - New metadata architecture enabled (SUDOKU_NISQ_NEW_METADATA=1)
    
Expected outcome:
    This script will:
    1. Generate a 2×2 Sudoku puzzle (minimal size for demonstration)
    2. Set up quantum solver with simple encoding
    3. Automatically enumerate solutions and create validation context
    4. Run on Aer simulator with automatic metrics recording
    5. Display comprehensive metrics (success probability, odds, retention, etc.)
"""

import os
from sudoku_nisq import QSudoku
from sudoku_nisq.solvers import ExactCoverQuantumSolver
from sudoku_nisq.metadata.config import MetadataConfig

# Enable new metadata architecture for automatic metrics
MetadataConfig.ENABLE_NEW_ARCHITECTURE = True
# Or: os.environ['SUDOKU_NISQ_NEW_METADATA'] = '1'


def main():
    print("=" * 70)
    print("Basic Metrics Example: Single-Run Workflow")
    print("=" * 70)
    print()
    
    # Step 1: Generate a minimal 2×2 Sudoku puzzle
    print("Step 1: Generating 2×2 Sudoku puzzle...")
    puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=2, canonicalize=True)
    print(f"  Puzzle hash: {puzzle.get_hash()[:12]}...")
    print(f"  Missing cells: {puzzle.num_missing_cells}")
    print(f"  Board:\n{puzzle}")
    print()
    
    # Step 2: Set up exact cover quantum solver
    print("Step 2: Configuring exact cover quantum solver...")
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=True)
    print("  ✓ Solver configured: ExactCoverQuantumSolver")
    print("  ✓ Encoding: simple")
    print()
    
    # Step 3: Build quantum circuit
    print("Step 3: Building quantum circuit...")
    circuit = puzzle.build_circuit(sdk="qiskit")
    resources = puzzle.report_resources()
    solver_name = "ExactCoverQuantumSolver"
    encoding = "simple"
    main_circuit = (
        resources.get("solvers", {})
        .get(solver_name, {})
        .get(encoding, {})
        .get("main_circuit", {})
    )
    print(f"  Qubits: {main_circuit.get('n_qubits', 'N/A')}")
    print(f"  Gates: {main_circuit.get('n_gates', 'N/A')}")
    print(f"  Depth: {main_circuit.get('depth', 'N/A')}")
    print(f"  2Q gates: {main_circuit.get('two_qubit_gates', 'N/A')}")
    print()
    
    # Step 4: ONE-LINE VALIDATION SETUP (New Phase 3 API!)
    print("Step 4: Setting up automatic validation...")
    context = puzzle.puzzle.create_validation_context(encoding_type='simple', max_solutions=10)
    print(f"  ✓ Enumerated {len(context.valid_solutions)} solution(s)")
    print(f"  ✓ Converted to bitstrings (length {len(context.valid_solutions[0])})")
    print(f"  ✓ Validator function created")
    
    # Attach validation context to enable automatic metrics
    puzzle.set_validation_context(context.valid_solutions)
    print("  ✓ Automatic metrics recording: ENABLED")
    print()
    
    # Step 5: Run quantum execution (metrics computed automatically!)
    print("Step 5: Executing on Aer simulator...")
    result = puzzle.run_aer(shots=1024, method="statevector")
    print(f"  ✓ Execution completed!")
    print(f"  ✓ Measured {len(result.get_counts())} unique outcomes")
    print()
    
    # Step 6: Access metrics programmatically (New Phase 3 API!)
    print("Step 6: Retrieving computed metrics...")
    try:
        metrics = puzzle.calculate_metrics()
        
        if metrics:
            print("  ✓ Metrics retrieved successfully!")
            print()
            print("=" * 70)
            print("METRICS SUMMARY")
            print("=" * 70)
            
            # Success probability
            p_succ = metrics.get('p_succ', 0)
            p_succ_lower = metrics.get('p_succ_ci_lower', 0)
            p_succ_upper = metrics.get('p_succ_ci_upper', 0)
            print(f"Success Probability: {p_succ:.4f}")
            print(f"  95% CI: [{p_succ_lower:.4f}, {p_succ_upper:.4f}]")
            print()
            
            # Distinct solutions
            distinct = metrics.get('distinct_valid_solutions', 0)
            print(f"Distinct Valid Solutions: {distinct}")
            print()
            
            # Valid odds (replaces deprecated SNR)
            valid_odds = metrics.get('valid_odds')
            if valid_odds is not None:
                if metrics.get('valid_odds_is_infinite', False):
                    print("Valid Odds: ∞ (perfect discrimination)")
                else:
                    odds_lower = metrics.get('valid_odds_ci_lower', 0)
                    odds_upper = metrics.get('valid_odds_ci_upper', 0)
                    print(f"Valid Odds: {valid_odds:.2f}")
                    print(f"  95% CI: [{odds_lower:.2f}, {odds_upper:.2f}]")
                print()
            
            # Peak discrimination
            p_best_valid = metrics.get('p_best_valid')
            p_best_invalid = metrics.get('p_best_invalid')
            if p_best_valid is not None:
                peak_ratio = metrics.get('peak_ratio')
                peak_gap = metrics.get('peak_gap')
                print(f"Peak Valid Probability: {p_best_valid:.4f}")
                if p_best_invalid is not None:
                    print(f"Peak Invalid Probability: {p_best_invalid:.4f}")
                if peak_ratio is not None:
                    if metrics.get('peak_ratio_is_infinite', False):
                        print("Peak Ratio: ∞ (no invalid solutions)")
                    else:
                        print(f"Peak Ratio: {peak_ratio:.2f}")
                if peak_gap is not None:
                    print(f"Peak Gap: {peak_gap:.4f}")
                print()
            
            # Retention-based efficiency (replaces deprecated eta_*)
            retention = metrics.get('retention_per_2q')
            if retention is not None:
                loss_per_gate = (1 - retention) * 100
                retention_lower = metrics.get('retention_per_2q_ci_lower')
                retention_upper = metrics.get('retention_per_2q_ci_upper')
                print(f"Per-Gate Retention: {retention:.6f}")
                print(f"  Loss per gate: {loss_per_gate:.4f}%")
                if retention_lower and retention_upper:
                    print(f"  95% CI: [{retention_lower:.6f}, {retention_upper:.6f}]")
                print()
            
            # Shot budget metrics
            shots_detect = metrics.get('shots_detect_point')
            if shots_detect is not None:
                shots_pessimistic = metrics.get('shots_detect_pessimistic')
                shots_optimistic = metrics.get('shots_detect_optimistic')
                print(f"Shots for 95% Detection:")
                print(f"  Point estimate: {shots_detect}")
                if shots_pessimistic:
                    print(f"  Pessimistic (CI lower): {shots_pessimistic}")
                if shots_optimistic:
                    print(f"  Optimistic (CI upper): {shots_optimistic}")
                print()
            
            print("=" * 70)
        else:
            print("  ⚠ No metrics available (validation context may not have been set)")
    
    except ValueError as e:
        print(f"  ⚠ {e}")
        print()
        print("Note: Make sure SUDOKU_NISQ_NEW_METADATA=1 is set!")
    
    print()
    print("✅ Example completed successfully!")
    print()
    print("Tip: Try different puzzle sizes, encodings, or shot counts to see how")
    print("     metrics change. The entire workflow remains this simple!")


if __name__ == "__main__":
    main()
