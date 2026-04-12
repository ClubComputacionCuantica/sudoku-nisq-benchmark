import os
# 1. ENABLE THE MAGIC FLAG **BEFORE** IMPORTS (required!)
os.environ['SUDOKU_NISQ_NEW_METADATA'] = '1'

from sudoku_nisq import QSudoku, ExactCoverQuantumSolver
from sudoku_nisq.metadata.benchmark_session import BenchmarkSession
from sudoku_nisq.metadata.config import MetadataConfig

# Also set the config flag directly (belt and suspenders approach)
MetadataConfig.ENABLE_NEW_ARCHITECTURE = True

# 2. CREATE YOUR PUZZLE
puzzle = QSudoku.generate(size=2, num_missing_cells=2)
print(f"Puzzle hash: {puzzle.get_hash()[:12]}...")

# 3. TELL IT HOW TO SOLVE (attach quantum algorithm)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")

# 4. TELL IT WHAT'S CORRECT (validation context - explained below!)
valid_solutions = ["00", "01", "10", "11"]  # Example
puzzle.set_validation_context(valid_solutions)

# 5. PICK YOUR HARDWARE/SIMULATOR
backend_alias = puzzle.init_aer(method="statevector")

# 5.5. BUILD CIRCUIT IN QISKIT FORMAT (required for Qiskit backends like Aer)
puzzle.build_circuit(sdk="qiskit")

# 6. CREATE THE BENCHMARK SESSION
session = BenchmarkSession(puzzle=puzzle)  # Auto-extracts puzzle_hash!
session.register_puzzle(puzzle)  # Logs to global registry (Stage 1)

# 7. RUN YOUR EXPERIMENTS
for opt_level in [0, 1, 2]:  # Try different optimizations
    result = session.execute_run(
        puzzle=puzzle,
        backend_alias=backend_alias,
        shots=512,
        opt_level=opt_level  # ← This transpiles the circuit!
    )
    print(f"✓ Completed opt_level={opt_level}")

# 8. ANALYZE YOUR RESULTS
runs = session.stage5.query()  # Get all execution records
metrics = session.stage6_7.query()  # Get metrics for all runs
print(f"Total runs: {len(runs)}")
print(f"Runs with metrics: {len(metrics) if metrics else 0}")

# 9. AGGREGATE STATISTICS
run_ids = [r['run_id'] for r in runs]
aggregated = session.stage6_7.compute_aggregated(run_ids=run_ids)
print("\nAggregated metrics:")
for key, value in aggregated.items():
    if isinstance(value, dict) and 'mean' in value:
        print(f"  {key}: mean={value.get('mean', 'N/A'):.4f}, std={value.get('std', 'N/A'):.4f}")
    else:
        print(f"  {key}: {value}")

# Let's also look at individual run metrics
print("\nIndividual run metrics:")
for run in runs[:1]:  # Just show first run
    run_id = run['run_id']
    metric = session.stage6_7.query(run_id=run_id)
    if metric:
        stage6 = metric.get('stage_6_evaluation', {})
        print(f"  Run {run_id[:8]}...")
        print(f"    Success prob: {stage6.get('p_succ', 'N/A')}")
        print(f"    Distinct valid: {stage6.get('distinct_valid_solutions', 'N/A')}")
        print(f"    Top-1 valid: {stage6.get('top_1_is_valid', 'N/A')}")