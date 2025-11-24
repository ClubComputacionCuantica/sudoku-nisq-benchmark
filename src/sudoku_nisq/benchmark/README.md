# Benchmark Module

High-level interface for staged quantum hardware benchmarking with early-exit points and feasibility checks.

## Features

- **Stage 0: Logical Circuit Analysis** - Analyze quantum circuits before hardware
- **Stage 1: Transpilation & Feasibility** - Check if problems fit on target devices
- **Stage 2: Hardware Execution** - Run on real quantum hardware (opt-in with confirmation)
- **Early Exit Points** - Stop before wasting hardware credits on infeasible problems
- **Friendly API** - Works great in Jupyter notebooks and CLI scripts

## Quick Start

```python
from sudoku_nisq import Benchmark

# Create benchmark
bench = Benchmark(size=4, missing_cells=6, encoding="simple")

# Stage 0: Analyze logical circuit (no hardware needed)
logical = bench.analyze_logical_circuits()
print(f"Circuit needs {logical.n_qubits} qubits")
print(f"Total gates: {logical.n_gates}")

# Stage 1: Check if it fits on hardware (requires registered backends)
bench.add_backend("ibm_brisbane", opt_level=2)
transpiled = bench.transpile_all()
transpiled.print_summary()

# Only proceed if feasible
if transpiled.get_feasible_backends():
    # Stage 2: Run on hardware (with confirmation prompt)
    results = bench.run_hardware(shots=2048)
    results.print_summary()
    
    # Save results
    bench.save_results("my_benchmark.json")
```

## Three-Stage Pipeline

### Stage 0: Logical Analysis
**Goal**: Understand the problem size before touching hardware

- Build quantum circuit with chosen solver/encoding
- Extract metrics: qubits, gates, depth
- No hardware access required
- Fast and free

**Output**: `LogicalAnalysis` object with:
- `n_qubits`: Number of qubits needed
- `n_gates`: Total gate count
- `depth`: Circuit depth
- `gate_breakdown`: Gate type distribution
- `sdk_type`: SDK used (qiskit/pytket/braket)

### Stage 1: Transpilation & Feasibility
**Goal**: Determine if problem fits on target hardware WITHOUT running

- Transpile circuit for each backend
- Check qubit requirements vs device capacity
- Detect potential issues (high utilization, deep circuits)
- Estimate runtime and costs

**Output**: `TranspilationReport` with per-backend analysis:
- `status`: FEASIBLE, TOO_LARGE, TIGHT, etc.
- `required_qubits`: Qubits after routing
- `transpiled_depth`/`transpiled_gates`: After compilation
- `feasibility_score`: 0-1 rating
- `warnings`: List of potential issues
- `estimated_runtime`: Expected execution time

**Example Output**:
```
================================================================================
TRANSPILATION FEASIBILITY REPORT
================================================================================

Backend              Status       Qubits          Depth      Score
--------------------------------------------------------------------------------
ibm_brisbane         ✓ READY      16/127          892        0.85
quantinuum_h1        ⚠ TIGHT      16/20           450        0.45
fake_small_device    ✗ TOO_LARGE  16/10           N/A        0.00
```

### Stage 2: Hardware Execution
**Goal**: Run on selected backends and collect real data

- Optional confirmation prompt (prevents accidents)
- Executes on feasible backends only
- Collects measurement counts and success rates
- Applies error mitigation if requested
- Tracks execution time and costs

**Output**: `ExecutionResults` with per-backend data:
- `status`: COMPLETED or FAILED
- `execution_time`: Wall-clock time
- `success_rate`: Fraction of valid solutions
- `counts`: Raw measurement outcomes
- `mitigated_success_rate`: After ZNE (if enabled)

## API Reference

### Benchmark Class

```python
Benchmark(
    puzzle=None,           # Optional pre-built puzzle
    size=9,                # Grid size (4, 9, 16, ...)
    missing_cells=20,      # Number of empty cells
    solver_class=None,     # Solver class (default: ExactCoverQuantumSolver)
    encoding="simple",     # Encoding strategy
    cache_base=".benchmark_cache"  # Cache directory
)
```

### Methods

- `add_backend(alias, label=None, opt_level=1, **kwargs)` - Register backend for comparison
- `analyze_logical_circuits(force_rebuild=False)` - Stage 0: Analyze logical circuit
- `transpile_all(opt_level=None, force_rebuild=False)` - Stage 1: Transpile and check feasibility
- `run_hardware(backends=None, shots=1024, confirm=True, use_mitigation=False)` - Stage 2: Execute
- `save_results(filepath)` - Save all stages to JSON

### Class Methods

- `Benchmark.quick_compare(backends, size=4, shots=1024, **kwargs)` - One-liner for quick comparisons

## Usage Examples

### Example 1: Full Pipeline

```python
bench = Benchmark(size=9, missing_cells=20)
bench.add_backend("ibm_brisbane", opt_level=2)
bench.add_backend("quantinuum_h1")

# Stage 0
logical = bench.analyze_logical_circuits()

# Stage 1
transpiled = bench.transpile_all()
feasible = transpiled.get_feasible_backends()

# Stage 2 (only if feasible)
if feasible:
    results = bench.run_hardware(backends=feasible, shots=2048)
    results.print_summary()
```

### Example 2: Feasibility Check Only

```python
# Check if puzzle fits without running
bench = Benchmark(size=16, missing_cells=100)
bench.add_backend("ibm_brisbane")

logical = bench.analyze_logical_circuits()
print(f"Needs {logical.n_qubits} qubits")

transpiled = bench.transpile_all()
if transpiled["ibm_brisbane"].status == FeasibilityStatus.TOO_LARGE:
    print("Puzzle too large! Try smaller size.")
```

### Example 3: Compare Multiple Backends

```python
bench = Benchmark(size=4, missing_cells=6)
bench.add_backend("ibm_brisbane", opt_level=2)
bench.add_backend("ibm_kyoto", opt_level=2)
bench.add_backend("quantinuum_h1")

# Quick feasibility check
transpiled = bench.transpile_all()
transpiled.print_summary()

# Run on all feasible backends
results = bench.run_hardware(shots=4096, use_mitigation=True)
results.print_summary()
```

## Integration with Existing Code

The Benchmark API is designed to complement (not replace) the existing QSudoku interface:

```python
from sudoku_nisq import Benchmark, QSudoku

# For benchmarking: use Benchmark
bench = Benchmark(size=4, missing_cells=6)
logical = bench.analyze_logical_circuits()

# For detailed solver work: use QSudoku
puzzle = QSudoku.generate(subgrid_size=2, num_missing_cells=6)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="pattern")
result = puzzle.run_aer(shots=1024)
puzzle.counts_plot(result)
```

## Next Steps

- **Stage 3 Analysis** (TODO): Comparison plots, metric calculations, report generation
- **Scalability Estimates** (TODO): Predict max puzzle size from small tests
- **Cost Tracking** (TODO): Track actual hardware costs per execution
- **CLI Tool** (Optional): Command-line wrapper for batch benchmarks

## See Also

- `example_benchmark.py` - Runnable examples
- `sudoku_nisq.benchmark.results` - Result data classes
- `sudoku_nisq.benchmark.comparison` - Main Benchmark implementation
