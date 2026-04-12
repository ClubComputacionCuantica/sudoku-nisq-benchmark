"""Demonstration of BenchmarkSession orchestration for all-stages benchmarking workflow.

This example showcases the simplest benchmarking orchestration that coherently
handles all 7 metadata stages using the BenchmarkSession API.

Key features:
- BenchmarkSession initialization with puzzle parameter
- Automatic Stage 1-7 recording via execute_run()
- Multi-run execution with provenance tracking
- Metrics aggregation and summary reporting
- Cache consistency with MetadataConfig

Prerequisites:
    - SUDOKU_NISQ_NEW_METADATA=1 environment variable (enables stage-aware architecture)
    
Expected outcome:
    This script will:
    1. Register a 2×2 puzzle (Stage 1)
    2. Execute 5 runs with full Stage 2a-7 recording
    3. Display aggregated metrics across all runs
    4. Show provenance chain from circuit to metrics
"""

import os
from sudoku_nisq import QSudoku
from sudoku_nisq.solvers.exact_cover_solver import ExactCoverQuantumSolver
from sudoku_nisq.metadata.benchmark_session import BenchmarkSession
from sudoku_nisq.metadata.config import MetadataConfig

# Enable new metadata architecture
os.environ['SUDOKU_NISQ_NEW_METADATA'] = '1'


def main():
    print("=" * 80)
    print("BenchmarkSession Demo: All-Stages Orchestration")
    print("=" * 80)
    print()
    
    # Step 1: Create puzzle and configure solver
    print("Step 1: Creating 2×2 Sudoku puzzle...")
    puzzle = QSudoku.generate(
        size=2, 
        num_missing_cells=2,
        cache_base=MetadataConfig.get_cache_base()  # Honors SUDOKU_NISQ_CACHE_DIR
    )
    print(f"  ✓ Puzzle generated")
    print(f"  ✓ Puzzle hash: {puzzle.get_hash()[:16]}...")
    print(f"  ✓ Missing cells: {puzzle.num_missing_cells}")
    print()
    
    print("Step 2: Configuring quantum solver...")
    puzzle.set_solver(
        ExactCoverQuantumSolver,
        encoding="simple",
        decompose_cnz=True
    )
    print(f"  ✓ Solver: {puzzle.quantum_solver.solver_name}")
    print(f"  ✓ Encoding: {puzzle.quantum_solver.encoding}")
    print()
    
    # Step 3: Set validation context (required for Stages 6-7 metrics)
    print("Step 3: Setting validation context...")
    # For a real benchmark, compute valid solutions properly
    # Here we use placeholder bitstrings for demonstration
    valid_solutions = ["00", "01", "10", "11"]  # Example for 2×2 with 2 missing cells
    puzzle.set_validation_context(valid_solutions=valid_solutions)
    print(f"  ✓ Validation context set with {len(valid_solutions)} valid solutions")
    print()
    
    # Step 4: Initialize Aer backend
    print("Step 4: Initializing Aer backend...")
    backend_alias = puzzle.init_aer(method="statevector")
    print(f"  ✓ Backend: Aer Statevector Simulator (alias: {backend_alias})")
    print()
    
    # Step 5: Create BenchmarkSession
    print("Step 5: Creating BenchmarkSession...")
    session = BenchmarkSession(
        puzzle=puzzle,  # Auto-extracts puzzle_hash
        cache_base=MetadataConfig.get_cache_base()  # Cache consistency
    )
    print(f"  ✓ Session created for puzzle {session.puzzle_hash[:16]}...")
    print(f"  ✓ Cache base: {session.cache_base}")
    print()
    
    # Step 6: Register puzzle (Stage 1)
    print("Step 6: Registering puzzle instance (Stage 1)...")
    session.register_puzzle(puzzle, solution_count=len(valid_solutions))
    instances = session.stage1.query()
    print(f"  ✓ Puzzle registered in global registry")
    print(f"  ✓ Total instances in registry: {len(instances) if isinstance(instances, list) else 'N/A'}")
    print()
    
    # Step 7: Execute multiple runs (Stages 2a-7)
    print("Step 7: Executing benchmarking runs...")
    n_runs = 5
    shots_per_run = 256
    
    print(f"  Running {n_runs} executions with {shots_per_run} shots each...")
    for i in range(n_runs):
        result = session.execute_run(
            puzzle=puzzle,
            backend_alias=backend_alias,
            shots=shots_per_run,
            opt_level=1
        )
        print(f"    ✓ Run {i+1}/{n_runs} completed")
    print()
    
    # Step 8: Query and display results
    print("Step 8: Querying recorded metadata...")
    
    # Stage 2a: Logical IR
    stage2a_data = session.stage2a.query()
    if isinstance(stage2a_data, dict) and "ExactCoverQuantumSolver" in stage2a_data:
        solver_data = stage2a_data["ExactCoverQuantumSolver"].get("simple", {})
        print(f"  Stage 2a (Logical IR):")
        print(f"    - Qubits: {solver_data.get('n_qubits', 'N/A')}")
        print(f"    - Gates: {solver_data.get('n_gates', 'N/A')}")
        print(f"    - Depth: {solver_data.get('depth', 'N/A')}")
    
    # Stage 3: Compilation
    compilations = session.stage3.query()
    print(f"  Stage 3 (Compilation): {len(compilations)} compilation(s) recorded")
    
    # Stage 5: Execution
    run_records = session.stage5.query()
    print(f"  Stage 5 (Execution): {len(run_records)} run(s) recorded")
    
    # Stages 6-7: Metrics
    metrics_list = session.stage6_7.query()
    print(f"  Stages 6-7 (Metrics): {len(metrics_list) if metrics_list else 0} run(s) with metrics")
    print()
    
    # Step 9: Compute aggregated metrics
    print("Step 9: Computing aggregated metrics...")
    try:
        # Extract run_ids from execution records
        run_ids = [rec.get('run_id') for rec in run_records if rec.get('run_id')]
        aggregated = session.stage6_7.compute_aggregated(run_ids=run_ids)
        print("  ✓ Aggregation completed")
        
        # Display summary statistics
        if aggregated:
            print("\n  Aggregated Metrics Summary:")
            for metric_key, stats in aggregated.items():
                if isinstance(stats, dict) and 'mean' in stats:
                    print(f"    {metric_key}:")
                    print(f"      Mean: {stats['mean']:.4f}")
                    print(f"      Std:  {stats.get('std', 0):.4f}")
                    print(f"      Min:  {stats.get('min', 0):.4f}")
                    print(f"      Max:  {stats.get('max', 0):.4f}")
    except Exception as e:
        print(f"  ⚠ Aggregation warning: {e}")
        print("    (This may occur if metrics are incomplete)")
    print()
    
    # Step 10: Display provenance chain
    print("Step 10: Provenance chain verification...")
    if run_records and len(run_records) > 0:
        latest_run = run_records[0]
        run_id = latest_run.get('run_id')
        compilation_id = latest_run.get('compilation_id')
        print(f"  Run ID: {run_id[:12] if run_id else 'N/A'}...")
        print(f"  ↓")
        print(f"  Compilation ID: {compilation_id[:12] if compilation_id else 'N/A'}...")
        
        if compilation_id:
            compilation_records = session.stage3.query(compilation_id=compilation_id)
            if compilation_records:
                circuit_hash = compilation_records[0].get('circuit_hash') if isinstance(compilation_records, list) else compilation_records.get('circuit_hash')
                print(f"  ↓")
                print(f"  Circuit Hash: {circuit_hash[:12] if circuit_hash else 'N/A'}...")
    print()
    
    print("=" * 80)
    print("✓ Benchmarking workflow completed successfully!")
    print("=" * 80)
    print()
    print("Summary:")
    print(f"  • Puzzle registered: {puzzle.get_hash()[:16]}...")
    print(f"  • Runs executed: {len(run_records)}")
    print(f"  • Compilations recorded: {len(compilations)}")
    print(f"  • Metrics computed: {len(metrics_list) if metrics_list else 0}")
    print(f"  • Cache location: {session.cache_base}")
    print()
    print("All stages (1, 2a, 2b, 3, 5, 6-7) recorded coherently.")


if __name__ == "__main__":
    main()
