# Quick Start: Integrating Metrics into Your Workflow

This guide shows how to use the metrics system once implemented.

## Basic Usage (Single Run)

```python
from sudoku_nisq.solvers import ExactCoverQuantumSolver
from sudoku_nisq.metrics import ValidationContext
from sudoku_nisq.backends import BackendManager

# 1. Setup solver as usual
puzzle = SudokuPuzzle.generate(size=4, num_missing_cells=5)
solver = ExactCoverQuantumSolver(puzzle, encoding="pattern")

# 2. Create validation context
validation_ctx = ValidationContext(
    valid_solutions=puzzle.enumerate_all_solutions(),
    total_valid_count=puzzle.count_solutions(),
    solution_validator=lambda bs: solver._is_valid_solution(bs)
)

# 3. Run with metrics collection
backend = BackendManager.get_backend("aer_simulator")
result, metrics = solver.run(
    backend=backend,
    shots=2048,
    opt_level=2,
    collect_metrics=True,
    validation_context=validation_ctx
)

# 4. Access metrics
print(f"Success probability: {metrics.p_succ:.4f}")
print(f"95% CI: [{metrics.p_succ_ci_lower:.4f}, {metrics.p_succ_ci_upper:.4f}]")
print(f"Distinct valid solutions: {metrics.distinct_valid_solutions}")
print(f"SNR: {metrics.snr:.2f}")
print(f"Gate efficiency (η_gate): {metrics.eta_gate:.6f}")

# 5. Access ranking metrics
for k in [1, 3, 5]:
    print(f"Top-{k} valid mass: {metrics.top_k_valid_mass[k]:.4f}")
    print(f"Precision@{k}: {metrics.precision_at_k[k]:.4f}")
    print(f"Recall@{k}: {metrics.recall_at_k[k]:.4f}")
```

## Multi-Run Benchmarking

```python
from sudoku_nisq.metrics.benchmarking import BenchmarkSuite

# 1. Setup benchmark suite
benchmark = BenchmarkSuite(
    solver=solver,
    validation_context=validation_ctx,
    n_runs=5,  # Repeat 5 times for variability
    different_seeds=True,  # Use different transpiler seeds
    k_values=[1, 3, 5, 10]
)

# 2. Run benchmark
results = benchmark.run_benchmark(
    backend=backend,
    shots=2048,
    opt_level=2
)

# 3. Access aggregated metrics
agg = results['aggregated']
print(f"Mean p_succ: {agg.p_succ_mean:.4f} ± {agg.p_succ_std:.4f}")
print(f"IQR: [{agg.p_succ_iqr[0]:.4f}, {agg.p_succ_iqr[1]:.4f}]")

# 4. Access individual runs
for i, run in enumerate(results['individual_runs']):
    print(f"Run {i+1}: p_succ = {run.p_succ:.4f}")
```

## Classical Baseline Comparison

```python
from sudoku_nisq.metrics.benchmarking import ClassicalBaseline

# 1. Create classical baseline
classical = ClassicalBaseline(solver.problem)

# 2. Time first solution
first_sol = classical.time_to_first_solution(algorithm="dlx")
print(f"Classical time to first solution: {first_sol['time_seconds']:.4f}s")

# 3. Compare with quantum
quantum_time = sum(r.execution_time for r in agg.execution_results)
print(f"Quantum time (5 runs): {quantum_time:.4f}s")
```

## Exporting Results

```python
from sudoku_nisq.metrics.reporters import JSONReporter, TableReporter, PlotReporter
from pathlib import Path

output_dir = Path("benchmark_results")
output_dir.mkdir(exist_ok=True)

# 1. Export to JSON
JSONReporter.export(agg, output_dir / "metrics.json")

# 2. Generate tables
markdown_table = TableReporter.format(agg, format="markdown")
with open(output_dir / "metrics.md", "w") as f:
    f.write(markdown_table)

latex_table = TableReporter.format(agg, format="latex")
with open(output_dir / "metrics.tex", "w") as f:
    f.write(latex_table)

# 3. Generate plots
PlotReporter.plot_success_probability(
    results['individual_runs'],
    save_path=output_dir / "p_succ.png"
)

PlotReporter.plot_ranking_metrics(
    agg,
    save_path=output_dir / "ranking.png"
)
```

## Comparing Multiple Backends

```python
backends = {
    "aer": BackendManager.get_backend("aer_simulator"),
    "fake_jakarta": BackendManager.get_backend("fake_jakarta"),
    # Add more backends
}

all_results = {}

for name, backend in backends.items():
    print(f"Running on {name}...")
    benchmark = BenchmarkSuite(solver, validation_ctx, n_runs=3)
    all_results[name] = benchmark.run_benchmark(
        backend=backend,
        shots=2048,
        opt_level=2
    )

# Compare
for name, result in all_results.items():
    agg = result['aggregated']
    print(f"{name}: p_succ = {agg.p_succ_mean:.4f} ± {agg.p_succ_std:.4f}")
```

## Advanced: Custom Validation

```python
# For generic exact cover problems (not Sudoku)

from sudoku_nisq.exact_cover_problem import ExactCoverProblem

problem = ExactCoverProblem(
    universe=['A', 'B', 'C', 'D'],
    subsets={'S1': ['A', 'B'], 'S2': ['B', 'C'], 'S3': ['C', 'D']}
)

solver = ExactCoverQuantumSolver(exact_cover_problem=problem)

# Custom validator
def is_valid_cover(bitstring: str) -> bool:
    selected = [i for i, bit in enumerate(bitstring) if bit == '1']
    covered = set()
    for idx in selected:
        subset_name = list(problem.subsets.keys())[idx]
        covered.update(problem.subsets[subset_name])
    return covered == set(problem.universe)

validation_ctx = ValidationContext(
    valid_solutions=problem.enumerate_valid_bitstrings(),
    total_valid_count=len(problem.enumerate_valid_bitstrings()),
    solution_validator=is_valid_cover
)

# Use as before...
```

## Accessing Hardware Metadata

```python
# After running with collect_metrics=True

hw_meta = metrics.hardware_metadata

if hw_meta:
    print(f"Backend: {hw_meta.backend_name}")
    print(f"Provider: {hw_meta.provider}")
    print(f"Calibration: {hw_meta.calibration_timestamp}")
    
    if hw_meta.two_qubit_gate_error:
        avg_2q_error = sum(hw_meta.two_qubit_gate_error.values()) / len(hw_meta.two_qubit_gate_error)
        print(f"Average 2q gate error: {avg_2q_error:.4e}")
    
    if hw_meta.t1_times:
        avg_t1 = sum(hw_meta.t1_times.values()) / len(hw_meta.t1_times)
        print(f"Average T1: {avg_t1:.2f} μs")
```

## Accessing Compilation Metadata

```python
comp_meta = metrics.compilation_metadata

if comp_meta:
    print(f"Optimization level: {comp_meta.optimization_level}")
    print(f"Transpiler seed: {comp_meta.transpiler_seed}")
    
    if comp_meta.pre_transpile_gates and comp_meta.post_transpile_gates:
        print("Gate count changes:")
        for gate_type in comp_meta.post_transpile_gates:
            pre = comp_meta.pre_transpile_gates.get(gate_type, 0)
            post = comp_meta.post_transpile_gates[gate_type]
            print(f"  {gate_type}: {pre} → {post}")
```

## Troubleshooting
