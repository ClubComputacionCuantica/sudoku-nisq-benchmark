# Benchmark Data Capture Reference

**Generated:** January 12, 2026  
**Build Version:** Phase 5.3 (Stage-aware infrastructure exists; full integration pending Phase 5.5)  
**Status:** 🟡 **PARTIALLY IMPLEMENTED** - Individual stage managers operational, BenchmarkSession orchestration in progress

This document describes the **intended architecture** for the stage-aware metadata system. While infrastructure components exist, full integration and BenchmarkSession orchestration are under active development.

⚠️ **Production Warning:** For production benchmarks, continue using legacy `metadata.json` format via `QuantumSolver.run()`. Stage-aware orchestration is experimental.

---

## Overview: 7-Stage Metadata Pipeline

The benchmarking system uses a **stage-aware metadata architecture** that coherently tracks data through 7 distinct stages:

| Stage | Name | Purpose | Storage |
|-------|------|---------|---------|
| **1** | Instance Selection | Puzzle generation & characteristics | `instances/registry.json` (global) |
| **2a** | Logical IR | Circuit construction (pre-transpilation) | `{puzzle_hash}/stage_2a_logical_ir.json` |
| **2b** | IR Policy | Circuit construction parameters | `{puzzle_hash}/stage_2b_ir_policy.json` |
| **3** | Compilation | Transpilation provenance | `{puzzle_hash}/stage_3_compilation.json` |
| **4** | Executable | Backend-specific circuit artifacts | `{puzzle_hash}/stage_4_executables/` |
| **5** | Execution | Runtime data & hardware snapshots | `{puzzle_hash}/stage_5_executions.json` |
| **6-7** | Metrics | Evaluation & normalization metrics | `{puzzle_hash}/stage_6_7_metrics.json` |

**Implementation Status:**
- ✅ **Stage managers operational:** Individual managers for all 7 stages exist and function correctly
- 🟡 **BenchmarkSession integration:** Orchestration layer under construction (Phase 5.4-5.5)
- ✅ **Legacy compatibility:** Current `metadata.json` format remains primary production path

---

## Stage 1: Instance Selection (𝓘, μ)

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `InstanceMetadataManager`  
**Storage:** `.quantum_solver_cache/instances/registry.json` (global registry)

### Data Captured

```python
{
    "puzzle_hash": str,                    # Deterministic puzzle identifier
    "size": int,                           # Board size (2, 4, 9, 16)
    "subgrid_size": int,                   # Subgrid dimension (1, 2, 3)
    "num_missing_cells": int,              # Number of empty cells (difficulty)
    "board": List[Tuple[int, int, int]],  # Full board state [(row, col, digit), ...]
    "open_tuples": List[Tuple[int, int]], # Empty cell positions
    "pre_tuples": List[Tuple[int, int]],  # Given/clue cell positions
    "generation_timestamp": str,           # ISO 8601 timestamp (UTC)
    "last_accessed": str,                  # ISO 8601 timestamp (UTC)
    "solution_count": Optional[int],       # Number of valid solutions (if known)
    "prng_seed": Optional[Any]            # PRNG seed for reproducibility (future: Phase 5.5)
}
```

### Key Features
- **Global registry:** Single file tracks all generated puzzles across experiments
- **Deterministic hashing:** Puzzle hash includes board state and constraints
- **Deduplication:** Identical puzzles share same hash (no duplicates)
- **Access tracking:** `last_accessed` updated on each query

---

## Stage 2a: Logical IR Construction

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `LogicalIRMetadataManager`  
**Storage:** `{puzzle_hash}/stage_2a_logical_ir.json`

### Data Captured

```python
{
    "solver_name": {                       # e.g., "ExactCoverQuantumSolver"
        "encoding": {                      # e.g., "simple", "pattern"
            "n_qubits": int,               # Number of qubits
            "n_gates": int,                # Total gate count
            "depth": int,                  # Circuit depth
            "gate_breakdown": Dict[str, int],  # Per-gate-type counts {"cx": 50, "h": 10, ...}
            "circuit_hash": str,           # SHA256 hash of logical circuit
            "sdk_type": str,               # "pytket", "qiskit", or "braket"
            "decompose_cnz": bool,         # Whether CNZ decomposition was applied
            "timestamp": str               # ISO 8601 timestamp
        }
    }
}
```

### Key Features
- **Multi-SDK support:** Auto-detects PyTKET, Qiskit, or Braket circuits
- **Gate-level breakdown:** Detailed counts per gate type (H, CX, T, etc.)
- **Pre-transpilation metrics:** Captures logical resources before optimization
- **Circuit hashing:** Deterministic identifier for provenance chains

---

## Stage 2b: IR Policy

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `IRPolicyMetadataManager`  
**Storage:** `{puzzle_hash}/stage_2b_ir_policy.json`

### Data Captured

```python
{
    "solver_name": {
        "encoding": {
            "decompose_cnz": bool,         # CNZ gate decomposition flag
            "track_memory": bool,          # Memory tracking enabled
            "encoding_type": str,          # "simple" or "pattern"
            "solver_options": Dict[str, Any],  # Additional solver parameters
            "timestamp": str
        }
    }
}
```

### Key Features
- **Construction parameters:** Tracks circuit building options
- **Solver configuration:** Records all solver-specific settings
- **Reproducibility:** Enables exact circuit reconstruction

---

## Stage 3: Compilation (𝖢)

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `CompilationMetadataManager`  
**Storage:** `{puzzle_hash}/stage_3_compilation.jsonl` (append-only log)

### Data Captured

```python
{
    "compilation_id": str,                 # UUID for this compilation
    "circuit_hash": str,                   # Links to Stage 2a circuit
    "backend_alias": str,                  # Target backend identifier
    "opt_level": int,                      # Optimization level (0-3)
    "timestamp": str,                      # ISO 8601 timestamp (UTC)
    
    # Pre-transpilation resources
    "resources": {
        "n_qubits": int,
        "n_gates": int,
        "depth": int,
        "gate_counts": Dict[str, int],     # Pre-transpilation gate breakdown
        "two_qubit_gates": int,            # Pre-transpilation 2Q count
        
        # Post-transpilation resources
        "transpiled_n_qubits": Optional[int],
        "transpiled_n_gates": Optional[int],
        "transpiled_depth": Optional[int],
        "transpiled_gate_counts": Optional[Dict[str, int]],
        "transpiled_two_qubit_gates": Optional[int]
    },
    
    # Routing metadata
    "routing": Optional[{
        "initial_layout": Optional[List[int]],  # Logical -> physical qubit mapping
        "final_layout": Optional[List[int]],    # Final qubit positions
        "swap_count": Optional[int],            # Number of SWAP gates inserted
        "routing_method": Optional[str]         # Routing algorithm used
    }],
    
    "sdk_type": str                        # "pytket", "qiskit", or "braket"
}
```

### Key Features
- **Append-only log:** Every compilation creates a new record (no overwrites)
- **Pre/post comparison:** Tracks resource changes from transpilation
- **Routing provenance:** Captures layout decisions and SWAP insertions
- **UUID linking:** `compilation_id` links to Stage 5 executions

---

## Stage 4: Executable

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `ExecutableMetadataManager`  
**Storage:** `{puzzle_hash}/stage_4_executables/{executable_id}.json`

### Data Captured

```python
{
    "executable_id": str,                  # UUID for this executable
    "compilation_id": str,                 # Links to Stage 3 compilation
    "backend_alias": str,
    "circuit_format": str,                 # "qpy", "qasm", "json", etc.
    "circuit_path": str,                   # Path to serialized circuit file
    "timestamp": str,
    "sdk_type": str
}
```

### Key Features
- **Circuit serialization:** Preserves exact executable circuit state
- **Format tracking:** Records serialization format for reproducibility
- **Backend-specific:** Stores backend-optimized circuit artifacts

---

## Stage 5: Execution (Runtime)

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `ExecutionMetadataManager`  
**Storage:** `{puzzle_hash}/stage_5_executions.jsonl` (append-only log)

### Data Captured

```python
{
    "run_id": str,                         # UUID for this execution
    "compilation_id": str,                 # Links to Stage 3 compilation
    "backend_name": str,                   # Backend that executed the circuit
    "timestamp": str,                      # ISO 8601 timestamp (UTC)
    "shots": int,                          # Number of measurement shots
    "execution_time_ms": float,            # Execution wall-clock time (milliseconds)
    "counts": Dict[str, int],              # Measurement results {bitstring: count}
    
    # Hardware calibration snapshot (captured at execution time)
    "hardware_snapshot": Optional[{
        "backend_name": str,
        "provider": str,                   # "ibm", "quantinuum", "aer", "aws"
        "calibration_timestamp": Optional[str],  # When calibration data was collected
        
        # Error rates per qubit
        "single_qubit_gate_errors": Optional[Dict[int, float]],  # {qubit_idx: error_rate}
        "two_qubit_gate_errors": Optional[Dict[Tuple[int, int], float]],  # {(q1, q2): error_rate}
        "readout_errors": Optional[Dict[int, float]],  # {qubit_idx: readout_error}
        
        # Coherence times (microseconds)
        "t1_times": Optional[Dict[int, float]],  # {qubit_idx: T1_time_us}
        "t2_times": Optional[Dict[int, float]],  # {qubit_idx: T2_time_us}
        
        # Additional provider-specific properties
        "extra_properties": Dict[str, Any]  # Provider-specific metadata
    }],
    
    # Circuit characteristics at execution (for metrics computation)
    "circuit_metrics": Optional[{
        "n_qubits": int,
        "depth": int,
        "n_gates": int,
        "two_qubit_gates": int,
        "circuit_volume": Optional[int]
    }],
    
    "job_id": Optional[str]                # Provider-specific job identifier
}
```

### Key Features
- **Hardware snapshots:** Captures calibration data at execution time (not compilation time)
- **Provider-agnostic:** Works with IBM, Quantinuum, AWS Braket, Aer simulator
- **Timing data:** Execution and queue times for performance analysis
- **Raw counts:** Unprocessed measurement results for flexible post-analysis

---

## Stages 6-7: Metrics (Evaluation & Normalization)

**Status:** ✅ **Operational** (manager exists and functional)  
**Manager:** `MetricsMetadataManager`  
**Storage:** `{puzzle_hash}/stage_6_7_metrics.json`

### Data Captured

```python
{
    "run_id": str,                         # Links to Stage 5 execution
    "timestamp": str,
    
    # === STAGE 6: EVALUATION METRICS (α, σ) ===
    "stage_6_evaluation": {
        # Success Probability
        "p_succ": float,                   # Probability of measuring valid solution
        "p_succ_ci_lower": float,          # 95% Clopper-Pearson CI lower bound
        "p_succ_ci_upper": float,          # 95% Clopper-Pearson CI upper bound
        "distinct_valid_solutions": int,   # Number of unique valid solutions found
        
        # Valid Odds (replaces deprecated SNR)
        "valid_odds": Optional[float],     # Odds ratio p_succ / (1 - p_succ)
        "valid_odds_ci_lower": Optional[float],
        "valid_odds_ci_upper": Optional[float],
        "valid_odds_is_infinite": bool,    # True if p_succ = 1.0 (perfect)
        
        # Peak-based Discrimination
        "p_best_valid": Optional[float],   # Probability of most frequent valid solution
        "p_best_invalid": Optional[float], # Probability of most frequent invalid solution
        "peak_ratio": Optional[float],     # Ratio of best valid to best invalid
        "peak_gap": Optional[float],       # Difference: p_best_valid - p_best_invalid
        "peak_ratio_is_infinite": bool,    # True if no invalid solutions measured
        
        # Count-based Ranking Metrics
        "top_k_valid_mass": Dict[int, float],     # {k: mass} for k=[1,3,5,10]
        "precision_at_k": Dict[int, float],       # {k: precision}
        "recall_at_k": Dict[int, float],          # {k: recall}
        
        # Mass-weighted Ranking (improved metrics)
        "mass_precision_at_k": Dict[int, float],  # Mass-weighted precision
        "valid_mass_capture_at_k": Dict[int, float],  # Valid mass in top-k
        
        # Deprecated (backward compatibility)
        "snr": Optional[float]             # DEPRECATED: Use valid_odds instead
    },
    
    # === STAGE 7: NORMALIZATION METRICS (τ) ===
    "stage_7_normalization": {
        # Retention-based Normalization (replaces eta_gate/eta_volume)
        "log_loss_per_2q": Optional[float],        # -log(p_succ) / two_qubit_gates
        "log_loss_per_2q_ci_lower": Optional[float],
        "log_loss_per_2q_ci_upper": Optional[float],
        "retention_per_2q": Optional[float],       # p_succ^(1/two_qubit_gates)
        "retention_per_2q_ci_lower": Optional[float],
        "retention_per_2q_ci_upper": Optional[float],
        
        "log_loss_per_volume": Optional[float],    # -log(p_succ) / circuit_volume
        "log_loss_per_volume_ci_lower": Optional[float],
        "log_loss_per_volume_ci_upper": Optional[float],
        "retention_per_volume": Optional[float],   # p_succ^(1/circuit_volume)
        "retention_per_volume_ci_lower": Optional[float],
        "retention_per_volume_ci_upper": Optional[float],
        
        # Shot Budget Metrics (replaces eta_shot)
        "shots_detect_point": Optional[int],       # Shots for 95% detection (point estimate)
        "shots_detect_pessimistic": Optional[int], # Using CI lower bound
        "shots_detect_optimistic": Optional[int],  # Using CI upper bound
        "shot_budget_reliability": float,          # Target reliability (default: 0.95)
        
        # Cost-normalized Heuristic Metrics (alternatives)
        "eta_product": Optional[float],            # η_× = p_succ / (depth × 2Q_gates)
        "eta_weighted_sum": Optional[float],       # η_+ = p_succ / (α·depth + β·2Q)
        "decay_rate": Optional[float],             # k = -log(p) / (α·depth + β·2Q)
        "cost_alpha": Optional[float],             # Weight for depth (reproducibility)
        "cost_beta": Optional[float],              # Weight for 2Q gates (reproducibility)
        
        # Deprecated (backward compatibility)
        "eta_gate": Optional[float],               # DEPRECATED: Use retention_per_2q
        "eta_volume": Optional[float],             # DEPRECATED: Use retention_per_volume
        "eta_shot": Optional[float]                # DEPRECATED: Use shot_budget metrics
    }
}
```

### Key Features
- **Auto-computed:** Metrics calculated automatically from Stage 5 execution data
- **Confidence intervals:** Statistical rigor with Clopper-Pearson CIs
- **Improved metrics:** Replaces deprecated SNR/eta metrics with retention/odds
- **Multi-run support:** Designed for aggregation across multiple runs

---

## Multi-Run Aggregation

**Computed by:** `MetricsMetadataManager.compute_aggregated()`

### Aggregated Statistics

For each metric, computes:
```python
{
    "mean": float,                         # Arithmetic mean
    "std": float,                          # Sample standard deviation (Bessel's)
    "median": float,                       # Median (50th percentile)
    "q1": float,                           # First quartile (25th percentile)
    "q3": float,                           # Third quartile (75th percentile)
    "iqr": float,                          # Interquartile range (Q3 - Q1)
    "min": float,                          # Minimum value
    "max": float                           # Maximum value
}
```

**Metrics aggregated:**
- `p_succ` (success probability)
- `valid_odds`
- `peak_ratio`, `peak_gap`
- `retention_per_2q`, `retention_per_volume`
- `log_loss_per_2q`, `log_loss_per_volume`
- `shots_detect_point`, `shots_detect_pessimistic`, `shots_detect_optimistic`
- All ranking metrics (`precision_at_k`, `recall_at_k`, etc.)

---

## Hardware Metadata Collection

**Collectors:** `QiskitMetadataCollector`, `AerMetadataCollector` (in `metrics/collectors/`)

### IBM Quantum (Real Hardware)

```python
{
    "backend_name": str,
    "provider": "ibm",
    "calibration_timestamp": datetime,     # From backend.properties()
    
    # Per-qubit error rates
    "single_qubit_gate_error": Dict[int, float],  # Average of gate errors on qubit
    "readout_error": Dict[int, float],            # Readout assignment error
    
    # Per-qubit-pair error rates
    "two_qubit_gate_error": Dict[Tuple[int, int], float],  # CX gate errors
    
    # Coherence times
    "t1_times": Dict[int, float],          # T1 relaxation (microseconds)
    "t2_times": Dict[int, float],          # T2 dephasing (microseconds)
    
    # Additional metadata
    "extra_properties": {
        "max_shots": int,
        "coupling_map": List[List[int]],
        "backend_version": str,
        "job_id": Optional[str]
    }
}
```

### Aer Simulator

```python
{
    "backend_name": str,                   # e.g., "aer_simulator"
    "provider": "aer",
    "calibration_timestamp": None,         # No calibration for simulator
    
    "single_qubit_gate_error": None,       # Exact simulation (zero error)
    "two_qubit_gate_error": None,
    "readout_error": None,
    "t1_times": None,
    "t2_times": None,
    
    "extra_properties": {
        "simulation_method": str,          # "statevector", "density_matrix", etc.
        "noise_model": Optional[str],      # If noise model applied
        "max_memory_mb": Optional[int]
    }
}
```

### Quantinuum

```python
{
    "backend_name": str,
    "provider": "quantinuum",
    "calibration_timestamp": Optional[datetime],
    
    # Quantinuum-specific characterization
    "single_qubit_gate_error": Dict[int, float],
    "two_qubit_gate_error": Dict[Tuple[int, int], float],
    "readout_error": Dict[int, float],
    
    "extra_properties": {
        "system_type": str,                # "H1-1", "H1-2", "H2-1", etc.
        "emulator_mode": bool,
        "api_version": str
    }
}
```

### AWS Braket

```python
{
    "backend_name": str,
    "provider": "aws",
    "calibration_timestamp": Optional[datetime],
    
    # Device-dependent characterization
    "single_qubit_gate_error": Optional[Dict[int, float]],
    "two_qubit_gate_error": Optional[Dict[Tuple[int, int], float]],
    "readout_error": Optional[Dict[int, float]],
    "t1_times": Optional[Dict[int, float]],
    "t2_times": Optional[Dict[int, float]],
    
    "extra_properties": {
        "device_arn": str,                 # AWS resource identifier
        "device_type": str,                # "QPU", "SIMULATOR", etc.
        "region": str
    }
}
```

---

## ExecutionResult Data Model

**Dataclass:** `ExecutionResult` (in `metrics/data_models.py`)

### Complete Structure

```python
@dataclass
class ExecutionResult:
    # Core execution data
    counts: Dict[str, int]                 # Measurement counts {bitstring: count}
    shots: int                             # Total shots
    execution_time: float                  # Wall-clock time (seconds)
    timestamp: datetime                    # When execution occurred
    backend_name: str                      # Backend identifier
    
    # Circuit characteristics
    num_qubits: int
    circuit_depth: int
    gate_counts: Dict[str, int]            # Per-gate-type counts
    two_qubit_gates: int                   # Total 2Q gates
    circuit_volume: Optional[int]          # Sum of active gates per layer
    
    # Provenance linking
    job_id: Optional[str]                  # Provider job ID
    run_id: Optional[str]                  # Stage 5 execution UUID
    
    # Metadata references
    metadata: Optional[Dict[str, Any]]     # Hardware snapshot, compilation_id, etc.
    raw_result: Any                        # Original provider result object
```

---

## Data Storage Locations

### Active Directory Structure (Current Production)

```
.quantum_solver_cache/                     # Default cache base (override with SUDOKU_NISQ_CACHE_DIR)
└── {puzzle_hash}/
    └── {solver_name}/
        └── {encoding}/
            ├── main_circuit.json          # PyTKET circuit cache
            ├── metadata.json              # Legacy format (active)
            └── transpiled/
                └── {backend}_{opt_level}/
                    ├── circuit.json       # Transpiled circuit
                    └── metadata.json      # Transpilation metadata
```

### Planned Directory Structure (Phase 5.5)

```
.quantum_solver_cache/                     # Default cache base
├── instances/
│   └── registry.json                      # Stage 1: Global puzzle registry (manager exists)
└── {puzzle_hash}/                         # Per-puzzle directory
    ├── stage_2a_logical_ir.json           # Stage 2a: Circuit resources (manager exists)
    ├── stage_2b_ir_policy.json            # Stage 2b: Construction params (manager exists)
    ├── stage_3_compilation.jsonl          # Stage 3: Compilation log (manager exists)
    ├── stage_4_executables/               # Stage 4: Serialized circuits (manager exists)
    │   ├── {executable_id_1}.json
    │   └── {executable_id_2}.json
    ├── stage_5_executions.jsonl           # Stage 5: Execution log (manager exists)
    └── stage_6_7_metrics.json             # Stages 6-7: Computed metrics (manager exists)
```

**Migration Note:** While stage-aware managers exist and can be used directly, automatic migration from legacy to stage-aware format will occur in Phase 5.5 when BenchmarkSession integration is complete.

### File Formats

- **JSON:** Structured data with nested dictionaries (Stages 1, 2a, 2b, 6-7)
- **JSONL (JSON Lines):** Append-only logs with one JSON object per line (Stages 3, 5)
- **Atomic writes:** All writes use tempfile + rename for consistency

---

## BenchmarkSession Orchestration

**Status:** 🚧 **UNDER CONSTRUCTION** (Phase 5.4-5.5)  
**Class:** `BenchmarkSession` (high-level API for all-stages benchmarking)

⚠️ **Implementation Notice:** The `BenchmarkSession` class skeleton exists, but all methods (`register_puzzle`, `execute_run`, `execute_batch`) are currently placeholders returning `None` or mock data. Individual stage managers can be used directly for testing.

**Production Alternative:** Use `puzzle.run()` with `BackendManager` directly:
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
result = puzzle.run_aer(shots=1024)  # Uses legacy metadata.json
```

### Planned Workflow (Phase 5.5)

```python
from sudoku_nisq import QSudoku
from sudoku_nisq.metadata import BenchmarkSession

# Create puzzle and solver
puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
puzzle.set_validation_context(valid_solutions=[...])

# Initialize session
session = BenchmarkSession(puzzle=puzzle)

# Register puzzle (Stage 1)
session.register_puzzle(puzzle, solution_count=8)

# Execute run (Stages 2a-7 auto-recorded)
result = session.execute_run(
    puzzle=puzzle,
    backend_alias="aer_simulator",
    shots=1024,
    opt_level=1
)

# Execute batch with aggregation
results = session.execute_batch(
    puzzle=puzzle,
    backend_alias="aer_simulator",
    n_runs=5,
    shots=1024,
    opt_level=1
)

# Query and aggregate
metrics = session.stage6_7.query()
aggregated = session.stage6_7.compute_aggregated(run_ids=[...])
```

---

## Environment Variables

- **`SUDOKU_NISQ_CACHE_DIR`**: Override default cache base directory (`.quantum_solver_cache`)
- **`SUDOKU_NISQ_SUPPRESS_DEPRECATION=1`**: Suppress legacy MetadataManager warnings (not yet implemented)

**Note:** No environment variable toggle exists for stage-aware metadata. Managers are always available for direct use; BenchmarkSession integration will be enabled automatically in Phase 5.5.

---

## Deprecated Metrics (Still Captured)

These metrics are deprecated but still computed for backward compatibility:

| Deprecated Metric | Replacement | Reason |
|-------------------|-------------|--------|
| `snr` | `valid_odds` | Odds ratio more statistically sound |
| `eta_gate` | `retention_per_2q` | Geometric mean retention more interpretable |
| `eta_volume` | `retention_per_volume` | Volume-normalized retention |
| `eta_shot` | `shots_detect_*` | Shot budgets more actionable |

**Migration guide:** See `docs/guide/metrics_migration.md` (if exists)

---

## Summary: Complete Data Inventory

### Per-Puzzle Data
1. **Puzzle characteristics:** Size, difficulty, board state, hash
2. **Circuit resources:** Qubits, gates, depth, gate breakdown
3. **Solver configuration:** Encoding, decomposition, options

### Per-Compilation Data
1. **Routing metadata:** Layouts, SWAPs, routing algorithm
2. **Pre/post resources:** Gate counts, depth before/after transpilation
3. **Compilation parameters:** Optimization level, backend target

### Per-Execution Data
1. **Measurement results:** Full count distribution
2. **Hardware snapshot:** Calibration, error rates, coherence times at execution time
3. **Timing data:** Execution time, queue time (if available)
4. **Circuit metrics:** Qubits, depth, gates, volume

### Per-Run Metrics
1. **Success metrics:** p_succ, confidence intervals, distinct solutions
2. **Discrimination metrics:** Valid odds, peak ratio, peak gap
3. **Ranking metrics:** Precision@k, recall@k, mass precision, valid capture
4. **Normalization metrics:** Retention, log loss, shot budgets
5. **Cost heuristics:** eta_product, eta_weighted_sum, decay_rate

### Multi-Run Aggregations
1. **Central tendency:** Mean, median
2. **Variability:** Std, IQR, min, max
3. **Quartiles:** Q1, Q3

---

## See Also

- **Architecture docs:** `docs/internal/architecture/stage_aware_metadata.md`
- **Metrics reference:** `docs/guide/metrics_reference.md`
- **API documentation:** `docs/api/metadata.rst`
- **Example usage:** `examples/benchmark_session_demo.py`

---

**Total data points:** ~50+ per run, ~100+ per aggregated multi-run benchmark
