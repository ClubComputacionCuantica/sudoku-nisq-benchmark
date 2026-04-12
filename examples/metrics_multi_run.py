"""Multi-run metrics example: Statistical analysis with aggregation.

This example demonstrates how to run multiple quantum executions and compute
aggregate statistics to characterize variability and reliability.

Key features:
- Multiple independent quantum runs
- Automatic metrics computation for each run
- Statistical aggregation (mean, std, IQR)
- Variability analysis across runs

Prerequisites:
    - New metadata architecture enabled (SUDOKU_NISQ_NEW_METADATA=1)
    
Expected outcome:
    This script will:
    1. Generate a 2×2 Sudoku puzzle
    2. Run quantum execution 5 times independently
    3. Compute metrics for each run
    4. Aggregate statistics across all runs
    5. Display mean, std, and confidence intervals
"""

import os
from sudoku_nisq import QSudoku
from sudoku_nisq.solvers import ExactCoverQuantumSolver
from sudoku_nisq.metadata.config import MetadataConfig
from sudoku_nisq.metadata.metrics import MetricsMetadataManager

# Enable new metadata architecture
MetadataConfig.ENABLE_NEW_ARCHITECTURE = True


def main():
    print("=" * 70)
    print("Multi-Run Metrics Example: Statistical Aggregation")
    print("=" * 70)
    print()
    
    # Step 1: Generate puzzle
    print("Step 1: Generating 2×2 Sudoku puzzle...")
    puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=2, canonicalize=True)
    puzzle_hash = puzzle.get_hash()
    print(f"  Puzzle hash: {puzzle_hash[:12]}...")
    print()
    
    # Step 2: Configure solver
    print("Step 2: Configuring quantum solver...")
    puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=True)
    puzzle.build_circuit(sdk="qiskit")
    print("  ✓ Circuit built")
    print()
    
    # Step 3: Set up validation context (once)
    print("Step 3: Setting up automatic validation...")
    context = puzzle.puzzle.create_validation_context(encoding_type='simple')
    puzzle.set_validation_context(context.valid_solutions)
    print(f"  ✓ {len(context.valid_solutions)} valid solution(s) enumerated")
    print()
    
    # Step 4: Run multiple independent executions
    print("Step 4: Running 5 independent quantum executions...")
    n_runs = 5
    shots = 512  # Fewer shots per run to see variability
    run_ids = []
    
    for i in range(n_runs):
        print(f"  Run {i+1}/{n_runs}... ", end="", flush=True)
        result = puzzle.run_aer(shots=shots, method="statevector")
        
        # Get the most recent run_id from execution metadata
        # (stored automatically when new metadata is enabled)
        metrics_manager = MetricsMetadataManager(
            cache_base=".quantum_solver_cache",
            puzzle_hash=puzzle_hash
        )
        all_metrics = metrics_manager.query()
        
        # Find most recent non-aggregated entry
        from datetime import datetime
        recent_run_id = None
        recent_timestamp = None
        for rid, data in all_metrics.items():
            if rid.startswith("aggregated_"):
                continue
            timestamp_str = data.get("timestamp")
            if timestamp_str:
                try:
                    ts = datetime.fromisoformat(timestamp_str)
                    if recent_timestamp is None or ts > recent_timestamp:
                        recent_timestamp = ts
                        recent_run_id = rid
                except:
                    pass
        
        if recent_run_id:
            run_ids.append(recent_run_id)
        
        print("✓ Done")
    
    print(f"  ✓ Completed {len(run_ids)} runs")
    print()
    
    # Step 5: Display individual run metrics
    print("Step 5: Individual run metrics:")
    print("-" * 70)
    
    metrics_manager = MetricsMetadataManager(
        cache_base=".quantum_solver_cache",
        puzzle_hash=puzzle_hash
    )
    
    individual_p_succ = []
    for i, run_id in enumerate(run_ids, 1):
        metrics = puzzle.calculate_metrics(run_id=run_id)
        if metrics:
            p_succ = metrics.get('p_succ', 0)
            individual_p_succ.append(p_succ)
            valid_odds = metrics.get('valid_odds')
            retention = metrics.get('retention_per_2q')
            
            print(f"Run {i}:")
            print(f"  Success probability: {p_succ:.4f}")
            if valid_odds is not None:
                print(f"  Valid odds: {valid_odds:.2f}")
            if retention is not None:
                print(f"  Per-gate retention: {retention:.6f}")
            print()
    
    # Step 6: Compute aggregate statistics
    print("Step 6: Computing aggregate statistics...")
    try:
        aggregated = metrics_manager.compute_aggregated(
            run_ids=run_ids,
            aggregation_key=f"multi_run_example_{n_runs}runs"
        )
        print("  ✓ Aggregation completed")
        print()
        
        print("=" * 70)
        print("AGGREGATED METRICS SUMMARY")
        print("=" * 70)
        
        # Success probability statistics
        p_succ_stats = aggregated.get('p_succ', {})
        if p_succ_stats:
            print("Success Probability:")
            print(f"  Mean: {p_succ_stats.get('mean', 0):.4f}")
            print(f"  Std Dev: {p_succ_stats.get('std', 0):.4f}")
            print(f"  Median: {p_succ_stats.get('median', 0):.4f}")
            print(f"  Q1-Q3: [{p_succ_stats.get('q1', 0):.4f}, {p_succ_stats.get('q3', 0):.4f}]")
            print(f"  IQR: {p_succ_stats.get('iqr', 0):.4f}")
            print()
        
        # Valid odds statistics
        odds_stats = aggregated.get('valid_odds', {})
        if odds_stats and odds_stats.get('mean') is not None:
            print("Valid Odds:")
            print(f"  Mean: {odds_stats.get('mean', 0):.2f}")
            print(f"  Std Dev: {odds_stats.get('std', 0):.2f}")
            print(f"  Median: {odds_stats.get('median', 0):.2f}")
            print()
        
        # Retention statistics
        retention_stats = aggregated.get('retention_per_2q', {})
        if retention_stats and retention_stats.get('mean') is not None:
            mean_retention = retention_stats.get('mean', 0)
            mean_loss = (1 - mean_retention) * 100
            print("Per-Gate Retention:")
            print(f"  Mean: {mean_retention:.6f} ({mean_loss:.4f}% loss/gate)")
            print(f"  Std Dev: {retention_stats.get('std', 0):.6f}")
            print(f"  Median: {retention_stats.get('median', 0):.6f}")
            print()
        
        # Shot budget statistics
        shots_stats = aggregated.get('shots_detect_point', {})
        if shots_stats and shots_stats.get('mean') is not None:
            print("Shots for 95% Detection:")
            print(f"  Mean: {shots_stats.get('mean', 0):.0f}")
            print(f"  Std Dev: {shots_stats.get('std', 0):.0f}")
            print(f"  Median: {shots_stats.get('median', 0):.0f}")
            print()
        
        print("=" * 70)
        print()
        
        # Analysis summary
        print("VARIABILITY ANALYSIS")
        print("-" * 70)
        
        if p_succ_stats:
            mean_p = p_succ_stats.get('mean', 0)
            std_p = p_succ_stats.get('std', 0)
            cv = (std_p / mean_p * 100) if mean_p > 0 else 0
            
            print(f"Coefficient of Variation: {cv:.2f}%")
            print()
            
            if cv < 10:
                print("✓ Low variability - Results are highly consistent")
            elif cv < 25:
                print("⚠ Moderate variability - Results show some spread")
            else:
                print("⚠ High variability - Consider more shots or runs")
        
    except Exception as e:
        print(f"  ⚠ Aggregation failed: {e}")
    
    print()
    print("✅ Multi-run example completed successfully!")
    print()
    print("Tip: Adjust n_runs and shots to explore the trade-off between")
    print("     sampling variance and computational cost.")


if __name__ == "__main__":
    main()
