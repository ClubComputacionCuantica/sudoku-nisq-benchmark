# Benchmarking Metrics Reference Guide

> **⚠️ Module Status:**
> 
> **✅ IMPLEMENTED (Phase 1 - Foundation):**
> - Data models (`ExecutionResult`, `HardwareMetadata`, `CompilationMetadata`, `ValidationContext`, `MetricsResult`)
> - Calculator modules (pure functions): `success_metrics`, `ranking_metrics`, `statistical_metrics`, `efficiency_metrics`, `mass_ranking_metrics`, `odds_metrics`, `peak_metrics`, `retention_metrics`, `shot_budget_metrics`, `cost_metrics`
> - Unit tests with comprehensive edge case coverage
> - Dependencies: `scipy` >=1.11.0 for statistical calculations
> 
> **🚧 UNDER CONSTRUCTION (Phase 2-6):**
> - Provider-specific metadata collectors (Qiskit/PyTKET/Braket)
> - Multi-run aggregation system (`BenchmarkSuite`)
> - Solver integration (`collect_metrics` parameter in `run()` methods)
> - Classical baseline comparisons
> - Export/reporting utilities (JSON, Markdown, LaTeX, plots)
> 
> APIs for implemented calculators are stable; collector/aggregator interfaces may change before stable release.

## Introduction

Modern quantum benchmarking requires a **benchmarking specification** $\mathbf{B}^\star$ that explicitly defines the entire workflow from algorithm to final metrics. This section explores the theoretical foundation for how sudoku-nisq implements reproducible benchmarks. 

The quantum benchmarking process follows a sequence that translates a conceptual task through the quantum computing stack to final performance quantification:

$$
\begin{align*}
   &(\mathcal{I}, \mu) \\
   &\downarrow \text{select } I_k \\
   &\text{Instance } I \\
   &\downarrow \mathsf{C}_{IR} \\
   &\mathrm{IR}(I) \\
   &\downarrow \mathsf{C}(\theta) \\
   &\text{Circ}^{\text{native}} \\
   &\downarrow \text{compile} \\
   &\mathrm{Exec} \\
   &\downarrow (\mathcal{H}_t, N_{\text{shots}}) \\
   &\{\text{bitstring}_i\} \\
   &\downarrow \alpha \\
   &\{p(x)\} \\
   &\downarrow \sigma \\
   &\text{Metrics} \\
   &\downarrow \tau \\
   &\text{Normalized FOM}
\end{align*}
$$

where:
- $\mathcal{I} = \{I_1, \ldots, I_N\}$: set of $N$ test problem instances
- $\mu$: instance sampling distribution (probability measure over $\mathcal{I}$)
- $I_k \in \mathcal{I}$: selected instance from the test set (index $k \in \{1, \ldots, N\}$)
- $I$: the specific instance being benchmarked
- $\mathsf{C}_{IR}$: Intermediate Representation (IR) transformation policy (target-independent compilation rules)
- $\mathrm{IR}(I)$: logical intermediate representation for instance $I$ (platform-agnostic circuit)
- $\mathsf{C}(\theta)$: compilation policy with parameters $\theta \in \Theta$ (optimization level, seed, routing)
- $\text{Circ}^{\text{native}}$: hardware-native circuit (gates from device's native set)
- $\mathrm{Exec}$: executable low-level control sequence (pulses, timings, calibrations)
- $\mathcal{H}_t$: time-dependent hardware state at execution time $t$
- $N_{\text{shots}} \in \mathbb{N}$: measurement shot budget
- $\{\text{bitstring}_i\}_{i=1}^{N_{\text{shots}}}$: collection of measured computational basis states, each $\text{bitstring}_i \in \{0,1\}^n$
- $\alpha$: preprocessing map converting raw counts to probability distribution
- $\{p(x): x \in \{0,1\}^n\}$: empirical probability distribution over $n$-qubit bitstrings
- $\sigma$: scoring functional evaluating performance relative to valid solutions
- $\tau$: normalization rule mapping (score, resources) to figure of merit
- FOM: Figure of Merit (normalized performance metric)

Each stage has precise meaning and corresponding implementation in this system:

#### Stage 1: Algorithm & Test Instances ($\mathcal{I}$)


Selection of computational task and instantiation of test problem instances $\mathcal{I} = \{I_1, I_2, \ldots, I_N\}$, each characterized by problem size, constraint structure, and solution space cardinality.

**sudoku-nisq implementation**

- `SudokuPuzzle.generate(...)`, `QSudoku.generate(...)`, or `ExactCoverProblem(universe, subsets)`
- Instance parameters: `puzzle.size`, `puzzle.open_tuples` (search space), `puzzle.pre_tuples` (constraints)
- Solution space: `ValidationContext.total_valid_count`

**Metrics connection:**
- Drives total solution count for recall@k calculations
- Determines problem complexity for cross-instance comparisons

**Code example:**
```python
from sudoku_nisq import SudokuPuzzle

# Generate instance (non-deterministic, no seed parameter)
puzzle = SudokuPuzzle.generate(subgrid_size=2, num_missing_cells=8, canonicalize=True)
total_solutions = puzzle.num_solutions  # |Sol(I)|

# Validate bitstring from measurement
valid_solutions = ["..."]
valid_solutions_set = set(valid_solutions)
validator = lambda bs: bs in valid_solutions_set
# Note: in full workflows, you typically attach a solver via QSudoku.set_solver()
# and use a known list of valid solution bitstrings for small instances.
```

**Instance Sampling Distribution ($\mu$)**

Benchmark instances must be drawn from a fixed, documented distribution to ensure consistency across devices and over time. The sampling measure $\mu$ specifies how instances are selected and parameterized.

Examples within sudoku-nisq:
- Uniform sampling over puzzle templates with a fixed difficulty parameter (e.g., number of missing cells, canonicalization enabled/disabled)
- Deterministic instance sets (e.g., a fixed list of 20 puzzles shipped with the benchmark)
- Parameterized distributions (e.g., size $n$, constraint density $\rho$, or subset generation models for exact cover)

Record the following:
- PRNG type and seed handling policy
- Sampling hyperparameters and selection rules
- Guarantees of reproducibility (or reasons for non-determinism)

Concretely: record the RNG library/version and generator name (e.g., NumPy PCG64), the actual seed values (and how you derive them), all sampling parameters used (e.g., `size=9`, `num_missing_cells=40`, `canonicalize=True`), the instance selection rule (uniform/stratified/fixed set), and whether runs are deterministic (persist boards/templates and seeds) or stochastic (explain why and how you preserve replayability).

Note: Sudoku generation is currently non-deterministic (no public seeding); therefore, benchmarks using stochastic sampling must persist the effective seeds or source templates as part of metadata to enable reproduction.

#### Stage 2 Intermediate Representation

##### Stage 2a: IR Construction – Logical Circuit

The logical, platform-agnostic circuit artifact that describes the algorithm's quantum behavior, encompassing the circuit structure and its interpretation.

For Grover-based exact-cover formulations, a useful shorthand is:

$$
	\text{IR}(I) = \mathcal{G}^{(r)}(\mathcal{O}_I)
$$

where:
- $I$: problem instance from test set $\mathcal{I}$
- $\mathcal{G}^{(r)}: \mathcal{H}_n \to \mathcal{H}_n$: Grover operator with $r \in \mathbb{N}$ iterations on $n$-qubit Hilbert space $\mathcal{H}_n$
- $\mathcal{O}_I$: an oracle unitary operator implementing phase flip which depends on the specific exact cover problem
- $r$: number of Grover iterations dependent on number of solutions

However, IR is broader in scope:
- Includes initial state preparation (e.g., uniform superposition), ancilla allocation and usage, and measurement definitions/basis choices.
- Includes the classical decoding map that interprets measured bitstrings into candidate solutions (subset selection, Sudoku assignments) and post-processing checks.

**sudoku-nisq implementation:**
- `ExactCoverEncoding.simple_subsets()` or `.pattern_subsets()`: Constraint → subset mapping
- `ExactCoverQuantumSolver._build_circuit()`: Constructs PyTKET `Circuit` object (logical IR)
- SDK conversions: PyTKET ↔ Qiskit `QuantumCircuit` and OpenQASM export capability (provider-dependent)

**Resource accounting (IR-level):**
- `CompilationMetadata.pre_transpile_gates`: Gate counts at IR level $G^{\text{pre}}$
- `CompilationMetadata.pre_transpile_depth`: Logical circuit depth $d^{\text{pre}}$
- `CompilationMetadata.pre_transpile_num_qubits`: Total logical qubits
- `CompilationMetadata.pre_transpile_num_ancilla`: Number of ancilla qubits used in oracle/diffusion constructions

Recorded IR artifacts (deterministic and reproducible):
- Logical circuit representation (PyTKET/Qiskit/OpenQASM) plus a hash of that logical circuit
- Gate counts, depth, and ancilla count at IR level
- SDK versions used for IR conversion (e.g., `pytket`, `qiskit`) and provenance

**SDK and transpiler versions (conversion provenance):**
- Record SDK package versions involved in IR formatting and conversion (e.g., `pytket`, `qiskit`, `qiskit-ibm-runtime`) and transpiler component versions.
- Store these in `CompilationMetadata` alongside circuit hashes to ensure reproducibility when PyTKET → Qiskit → OpenQASM conversions are performed.

---

##### Stage 2b: IR Policy – Allowed Logical Transformation Family ($\mathsf{C}_{\text{IR}}$)

Defines what transformations are allowed during IR construction and before hardware mapping. This is the target-independent part of the compilation pipeline and governs legal implementation variants at the logical level.

The policy should be documented in SDK-agnostic terms: which families of gate decompositions are permissible for multi-controlled and oracle subroutines; whether and how ancilla may be used (clean-only vs reuse, measurement-based resets permitted or not); which transpilation techniques are in scope (e.g., algebraic/pattern rewrites, subgraph re-synthesis, gate unrolling) with abstract terminology; target-independent bounds on logical changes before hardware mapping (depth/size trade-offs, preserving oracle structure); and determinism requirements (fixed seeds, version-locked passes). Stage 2b policy operates at the logical IR level, whereas Stage 3 defines the target-aware mapping scope to native gates.


These policy elements should be documented in `CompilationMetadata` along with the IR resources and conversion provenance so that two runs of the same IR yield comparable native circuits under the same allowed transformation family. The IR policy ties directly to what we can record today: SDK choice and versions, IR-to-SDK conversion provenance, and a circuit hash at handoff; advanced pass families are out of scope, so we document only parameters and artifacts we can reliably capture.

**Code example:**
```python
from sudoku_nisq import QSudoku, ExactCoverQuantumSolver

qs = QSudoku.generate(size=4, num_missing_cells=8)
qs.set_solver(ExactCoverQuantumSolver, encoding="simple", decompose_cnz=True)
circuit = qs.build_circuit(sdk="qiskit")  # IR in Qiskit format
print(f"Pre-transpile: {circuit.num_qubits} qubits, depth {circuit.depth()}")

# Recommended: capture conversion provenance
import qiskit
import importlib
versions = {
   'qiskit': qiskit.__version__,
   'pytket': importlib.import_module('pytket').__version__,
}
print("SDK versions:", versions)
```

---

#### Stage 3: Mapped Circuit – Compilation Policy $\mathsf{C}$

Application of compilation policy $\mathsf{C}(\text{IR}, \mathcal{H}, \theta)$ that maps logical circuit to hardware-specific implementation:

$$
\mathsf{C}: (\text{IR}, \mathcal{H}, \theta) \mapsto \text{Circuit}^{\text{native}}
$$

where:
- $\text{IR}$: logical intermediate representation (abstract, platform-agnostic circuit)
- $\mathcal{H}$: hardware specification (topology graph, native gate set, qubit connectivity, coherence properties)
- $\theta \in \Theta$: compilation parameters from policy space $\Theta$ (optimization level, transpiler seed, routing strategy)
- $\text{Circuit}^{\text{native}}$: hardware-native circuit using gates from $\mathcal{H}$'s native set

**sudoku-nisq implementation:**
- `QuantumSolver._transpile_qiskit()`: Uses `generate_preset_pass_manager` or `qiskit.compiler.transpile`
- `QuantumSolver._transpile_pytket()`: Calls `backend.get_compiled_circuit()`
- Respects hardware topology, native gate sets, coupling maps

**Metrics connection:**
- `CompilationMetadata.optimization_level`: Explicit $\theta_{\text{opt}}$ parameter
- Transpiler seed: not currently supported/recorded (Qiskit transpilation is deterministic given backend + optimization level)
- `CompilationMetadata.post_transpile_gates`: Gate counts after compilation $G^{\text{post}}(\mathsf{C})$
- `CompilationMetadata.circuit_hash`: Fingerprint $h(\text{Circuit}^{\text{native}})$ for caching

Compilation metrics are constrained to recorded parameters: optimisation level, transpiler seed, backend target/alias, SDK path, and the post-transpile circuit fingerprint with gate/depth counts. When parameters aren’t set, note provider defaults to keep runs comparable.

Recorded compilation artifacts (deterministic and reproducible):
- `optimization_level`
- Backend/provider identity (e.g., IBM backend name), target details, coupling map (when available)
- Provider defaults used
- Post-transpile native circuit incluing hash
- Post-transpile gate counts and depth

Pending controls: pass-level configuration (explicit optimisation passes/families) are not recorded yet, but SDKs support these settings; we will surface and record them when exposed in the public API.

**Implementation Family ($\mathsf{C}$ scope)**

Different compilation policies can produce vastly different resource profiles for identical algorithms. The complete benchmark specification $\mathbf{B}^\star$ **requires explicit documentation** of $\mathsf{C}$ for reproducibility.

Abstractly, the allowed implementation family specifies the space of legal transformations between IR and native circuits: decomposition choices, routing/layout strategies, optimization classes, ancilla use, and controls on non-determinism. This complements Stage 2b's target-independent IR policy ($\mathsf{C}_{\text{IR}}$) by defining Stage 3's target-aware scope (what mappings are permitted on the way to native gates).

To ensure reproducibility, we treat $\mathsf{C}$ as a **constraint set** that limits compiler freedoms when lowering IR to native circuits. We document the parameters and provenance one can reliably capture:

- optimisation level: record in `CompilationMetadata.optimization_level`.
- transpiler seed: not currently supported/recorded.
- backend target/alias: record the device or simulator alias, provider, and SDK used.
- SDK choice and versions: record whether the build used `pytket`, `qiskit`, or `braket`, and capture package versions involved in IR conversion/transpilation.
- circuit fingerprint: record the post-transpile circuit hash/fingerprint and cache key.
- resource summary: record gate counts, depth, and qubit count before/after transpilation.
- build-time decomposition choice: record solver option `decompose_cnz` when constructing oracles and multi-controlled subroutines.

When parameters aren’t set, note the provider/compiler defaults (e.g., preset pass manager vs `qiskit.transpile`, PyTKET `get_compiled_circuit`, and no client-side Braket transpilation). Documenting these items constrains $\mathsf{C}$ to a reproducible subset of behaviors without relying on pass-specific or ancilla-policy terminology.

**Code example:**
```python
result = qs.run(
    backend_alias="ibm_brisbane",
    opt_level=2,  # θ_opt
    shots=2048,
)
# Post-transpilation metrics available in result.gate_counts
```

---


#### Stage 4: Executable – Low-Level Compilation

Conversion of native-gate circuit to hardware control sequences (pulse schedules, microwave pulses, laser sequences):

$$
\text{Circuit}^{\text{native}} \xrightarrow{\text{backend compiler}} \text{Executable}(\text{pulses}, \text{timings}, \text{calibrations})
$$

where:
- $\text{Circuit}^{\text{native}}$: gate-level circuit from Stage 3
- backend compiler: provider-controlled pulse synthesis and scheduling pipeline
- $\text{Executable}$: low-level control sequence (pulse schedules, microwave waveforms, timing constraints, calibration parameters)

**sudoku-nisq implementation:**
- Provider-specific job submission via `BackendManager` and `providers/` modules. This stage is entirely backend-controlled; users influence it only by choosing the backend and supplying the gate-level circuit from Stage 3.

**IBM (Qiskit):**
- IBM backends perform pulse-level schedule synthesis, alignment/timing, and apply current calibrations via the backend compiler stack. Primitives (Sampler/Estimator) are supported by Qiskit but are not used by sudoku-nisq at present; we submit gate-level circuits through the provider integration, and the backend handles low-level compilation.

**Quantinuum:**
- PyTKET’s Quantinuum backend submits the compiled TKET circuit; the H-series performs native decomposition validation and pulse-level translation for ion-trap hardware.

**AWS Braket:**
- Device-specific compilation occurs automatically per provider (Rigetti/QCI/IonQ) and submitted circuit representation; no client-side pulse control is performed.

**Metrics connection:**
- `ExecutionResult.job_id`: Unique identifier for the submitted job (execution instance)
- `HardwareMetadata.provider`: Execution platform (e.g., "ibm", "quantinuum", "aws")
- `HardwareMetadata.calibration_timestamp`: Backend calibration snapshot time used during execution

Note: Pulse schedules and hardware-level optimizations are not recorded in the current version; only execution-instance metadata is captured for reproducibility.

---

#### Stage 5: Execution – Physical QPU Run

Run the Stage 4 executable on hardware to sample measurement outcomes:

$$
   \text{Execution}(\text{Executable}, N_{\text{shots}}, \mathcal{H}_t) \longrightarrow \{\text{bitstring}_i\}_{i=1}^{N_{\text{shots}}}
$$

where:
- $N_{\text{shots}} \in \mathbb{N}$: sampling budget (number of measurement repetitions)
- $\mathcal{H}_t$: time-dependent hardware state at timestamp $t$ (calibration snapshot, coherence times, gate error rates)
- $\text{bitstring}_i \in \{0,1\}^n$: the $i$-th measured computational basis state
- $n \in \mathbb{N}$: number of qubits in the circuit

**sudoku-nisq implementation:**
- `QuantumSolver.run()` submits the compiled circuit with the requested shots
- `QuantumSolver.run_aer()` wraps simulator execution (Qiskit Aer)
- Returns `ExecutionResult` with raw counts and execution metadata

**Metrics connection (execution-time):**
- `ExecutionResult.execution_time` (wall-clock job runtime)
- **Shot budget:** `ExecutionResult.shots` (exact $N_{\text{shots}}$ used)
- **Temporal context:** `ExecutionResult.timestamp` (run occurrence)
- **Hardware snapshot:** `HardwareMetadata.t1_times`, `t2_times`, `single_qubit_gate_error`, `two_qubit_gate_error`

**Code example:**
```python
result = qs.run_aer(shots=2048, method="statevector")
print(f"Execution time: {result.execution_time:.3f}s")
print(f"Counts: {result.counts}")
print(f"Hardware: {result.backend_name} at {result.timestamp}")
```

---

#### Stage 6: Statistical Evaluation – Preprocessing $\alpha$ and Scoring $\sigma$

Application of analysis pipeline to convert raw measurement counts to performance scores:

$$
\alpha: \{(\text{bitstring}_i, n_i)\}_{i=1}^{m} \mapsto \{p(x): x \in \{0,1\}^n\}
$$

$$
\sigma: \{p(x): x \in \{0,1\}^n\}, \text{Sol}(I) \mapsto \mathbb{R}_{\geq 0}
$$

where:
- $m \in \mathbb{N}$: number of distinct observed bitstrings
- $\text{bitstring}_i \in \{0,1\}^n$: the $i$-th distinct measured bitstring
- $n_i \in \mathbb{N}$: observed count (frequency) of $\text{bitstring}_i$, with $\sum_{i=1}^{m} n_i = N_{\text{shots}}$
- $n \in \mathbb{N}$: number of qubits (bitstring length)
- $\alpha$: preprocessing map that normalizes counts to probability distribution over $\{0,1\}^n$
- $p(x) = \frac{n_x}{N_{\text{shots}}} \in [0,1]$: empirical probability of bitstring $x$, where $n_x$ is its observed count
- $\text{Sol}(I) \subseteq \{0,1\}^n$: set of valid solutions for instance $I$
- $\sigma$: scoring functional that evaluates quality relative to $\text{Sol}(I)$ (returns performance score $\in \mathbb{R}_{\geq 0}$)

**sudoku-nisq implementation:**
- `calculate_p_succ`, `calculate_distinct_solutions`: Implement $\alpha$ (counts → probabilities) and $\sigma$ (validation)
- `ValidationContext.solution_validator`: Defines $\text{Sol}(I)$ membership test

**Metrics connection:**
- `MetricsResult.p_succ`: Primary outcome $\sigma_1 = P(x \in \text{Sol}(I))$
- `MetricsResult.p_succ_ci_lower`, `p_succ_ci_upper`: Uncertainty quantification via Clopper-Pearson
- `MetricsResult.valid_odds`: Signal quality (odds ratio) — replaces deprecated "snr"
- `MetricsResult.distinct_valid_solutions`: Distribution breadth $|\{x: x \in \text{Sol}(I), p(x) > 0\}|$

**Analogy to standard benchmarks:**
- Heavy Output Probability (QV): $\sigma = P(x > \text{median}(\text{ideal distribution}))$
- Cross-Entropy Benchmarking: $\sigma = \sum_x p_{\text{ideal}}(x) \log p_{\text{measured}}(x)$
- Our framework: $\sigma = P(x \in \text{Sol}(I))$ with rich auxiliary metrics

**Code example:**
```python
from sudoku_nisq.metrics.calculators import calculate_p_succ, calculate_distinct_solutions

valid_solutions = ["..."]
valid_solutions_set = set(valid_solutions)
validator = lambda bitstring: bitstring in valid_solutions_set
p_succ = calculate_p_succ(result.counts, validator)
distinct = calculate_distinct_solutions(result.counts, validator)
```

---

#### Stage 7: Normalized Performance Metrics – Interpretation Rule $\tau$

Normalization by resource consumption to enable fair cross-platform comparison:

$$
\tau: (\mathbb{R}_{\geq 0} \times \mathcal{R}) \mapsto \mathbb{R}_{\geq 0}
$$

where:
- $\mathbb{R}_{\geq 0}$: domain of performance scores (output of $\sigma$)
- $\mathcal{R}$: resource parameter space (gate counts $G_{2q} \in \mathbb{N}$, circuit volume $V \in \mathbb{N}$, shot count $N_{\text{shots}} \in \mathbb{N}$)
- $\tau$: normalization rule mapping (score, resources) to normalized figure of merit

Used metrics (See following section):

- Gate-normalized $\eta_{\text{gate}} = \sigma / G_{2q}$
- Volume-normalized $\eta_{\text{volume}} = \sigma / (w \times d_c)$
- Shot-normalized $\eta_{\text{shot}} = \sigma / N_{\text{shots}}$

**sudoku-nisq implementation:**
- Deprecated (legacy): `calculate_eta_gate`, `calculate_eta_volume`, `calculate_eta_shot` (linear normalizations)
- Recommended: retention/log-loss per resource and shot-budget metrics (see below)

> **⚠️ DEPRECATED (Dec 2025):** The metrics `eta_gate`, `eta_volume`, `eta_shot` listed below have been replaced with statistically rigorous alternatives. See **Resource-Normalized Efficiency Metrics** section for `retention_per_2q`, `log_loss_per_2q`, and `shots_to_detect` metrics.

**Metrics connection (deprecated):**
- `MetricsResult.eta_gate = p_succ / two_qubit_gates`: ❌ Replaced by `retention_per_2q` (geometric mean)
- `MetricsResult.eta_volume = p_succ / (width × depth)`: ❌ Replaced by `retention_per_volume` (geometric mean)
- `MetricsResult.eta_shot = p_succ / shots`: ❌ Replaced by `shots_to_detect` (reliability-based)

**Why replaced:**
- Linear normalization p/n assumes additive error model (invalid for multiplicative gate errors)
- Geometric mean captures true per-resource efficiency in multiplicative settings
- See migration guide in Resource-Normalized section for detailed comparison

**Alignment with standard benchmarks:**
- Quantum Volume: Reports $V_Q = 2^{n}$ for largest passing $n$ (single-number FOM)
- Algorithmic Qubits: Reports $\#AQ(d)$ for depth $d$ achieving target fidelity
- Our framework: Reports multiple normalizations to avoid imposing artificial composite scores

**Code example (deprecated - for backward compatibility only):**
```python
# ⚠️ DEPRECATED: Use new retention/shot_budget metrics instead
# Manual calculation
eta_gate = p_succ / result.two_qubit_gates  # ❌ Use retention_per_2q
eta_volume = p_succ / (result.num_qubits * result.circuit_depth)  # ❌ Use retention_per_volume
eta_shot = p_succ / result.shots  # ❌ Use shots_to_detect

# Recommended replacement:
from sudoku_nisq.metrics.calculators import (
    calculate_retention_per_2q,
    shots_to_detect
)
retention_2q = calculate_retention_per_2q(p_succ, result.two_qubit_gates)
shots_needed = shots_to_detect(p_succ, reliability=0.95)
```

---

#### Benchmark Specification $\mathbf{B}^\star$

A **complete benchmark specification** requires explicit definition across all seven stages:

$$
\mathbf{B}^\star = (\mathcal{I} = \{I_k\}, \mu, \mathrm{IR}(I), \mathsf{C}_{IR}, \mathsf{C}(\theta), \text{Circ}^{\text{native}}, \mathrm{Exec}, \mathcal{H}_t, N_{\text{shots}}, \alpha, \sigma, \tau)
$$

| Stage | Component | Object | Description | sudoku-nisq Implementation |
|-------|-----------|--------|-------------|----------------------------|
| 1 | **Test instances** | $\mathcal{I} = \{I_k\}$ | Set of problem instances | `SudokuPuzzle`, `ExactCoverProblem` with documented size, constraints, solution count |
| 1 | **Instance sampling** | $\mu$ | Instance sampling distribution | Generation parameters, PRNG seed/version, determinism guarantees |
| 2a | **IR construction** | $\mathrm{IR}(I)$ | Logical intermediate representation for instance $I$ | Logical circuit object (PyTKET/Qiskit), $G^{\text{pre}}$, $d^{\text{pre}}$, ancilla, SDK versions, IR hash |
| 2b | **IR policy** | $\mathsf{C}_{IR}$ | IR-level transformation policy | `decompose_cnz`, allowed rewrites, SDK conversion provenance, canonicalization rules |
| 3 | **Compilation policy** | $\mathsf{C}(\theta)$ | Compilation policy with parameters $\theta$ | `CompilationMetadata`: `optimization_level`, backend target, implementation family |
| 3 | **Mapped circuit** | $\text{Circ}^{\text{native}}$ | Hardware-native circuit | Post-transpile gates $G^{\text{post}}$, depth, circuit hash, provider defaults |
| 4 | **Executable** | $\mathrm{Exec}$ | Executable pulses/schedules | Provider-controlled; `job_id`, backend compiler version (when available) |
| 5 | **Hardware state** | $\mathcal{H}_t$ | Hardware state at time $t$ | `HardwareMetadata`: calibration timestamp, $T_1$/$T_2$, gate errors at execution time |
| 5 | **Execution parameters** | $N_{\text{shots}}$ | Measurement shot budget | `ExecutionResult.shots`, timestamp, execution time |
| 6 | **Preprocessing** | $\alpha$ | Preprocessing map | `calculate_p_succ`: counts → probabilities, bitstring filtering |
| 6 | **Scoring functional** | $\sigma$ | Scoring functional | Success metrics: $p_{\text{succ}}$, confidence intervals, valid odds, distinct solutions |
| 7 | **Normalization** | $\tau$ | Normalization rule | Efficiency metrics: retention/log-loss per resource, shot budgets, ranking metrics |

**Reproducibility requirements:**

1. **Stage 1 (Instances)**: Document instance parameters, solution counts, and sampling method; persist puzzle definitions or generation seeds
2. **Stage 2a (IR Construction)**: Record logical circuit artifact with SDK versions, IR-level resources ($G^{\text{pre}}$, $d^{\text{pre}}$, ancilla count), and circuit hash
3. **Stage 2b (IR Policy)**: Document allowed transformation family ($\mathsf{C}_{\text{IR}}$): `decompose_cnz`, canonicalization rules, SDK conversion rules, deterministic IR normalization
4. **Stage 3 (Compilation)**: Explicit $\mathsf{C}$ documentation with transpiler seed, optimization level, backend target, and post-transpile circuit fingerprint
5. **Stage 4 (Executable)**: Capture `job_id` and provider-specific compilation metadata when available
6. **Stage 5 (Execution)**: Record hardware state snapshot ($\mathcal{H}_t$), calibration timestamp, shot count $N_{\text{shots}}$, and temporal context
7. **Stage 6 (Evaluation)**: Report statistical rigor with confidence intervals, validation logic, and multi-run variability
8. **Stage 7 (Interpretation)**: Provide multiple normalized metrics ($\eta_{\text{gate}}$, $\eta_{\text{volume}}$, $\eta_{\text{shot}}$) rather than single composite scores
9. **Classical baseline**: Time-to-solution comparison for quantum advantage claims

---

### Advantadges

**Transparency:**  
Reporting only $p_{\text{succ}}$ without compilation details, hardware context, or resource normalization is scientifically insufficient. Different choices of $\mathsf{C}$ can produce 10× differences in gate count for identical algorithms.

**Cross-platform fairness:**  
Volume-normalized metrics ($\eta_{\text{volume}}$) enable comparing superconducting, trapped-ion, and neutral-atom systems on equal footing, since volume captures width × depth universally.

**Reproducibility:**  
Circuit hashes ($h(\text{Circuit}^{\text{native}})$) and transpiler seeds enable exact reproduction of experiments, critical for validating benchmark claims.

**Quantum advantage evaluation:**  
Without classical baselines and statistical uncertainty quantification, no rigorous statement about quantum utility can be made.

---

## Resource-Normalized Efficiency Metrics

> **Core Framework Feature:** Resource normalization ($\tau$) is central to fair cross-platform benchmarking. Rather than reporting a single composite score, this framework provides multiple normalized figures of merit that reveal different aspects of quantum-classical performance trade-offs.

> **⚠️ Metrics Update (Dec 2025):** The original linear normalization metrics (`eta_gate`, `eta_volume`, `eta_shot`) have been replaced with statistically rigorous alternatives. Original metrics are deprecated and will be removed in v0.5.0. See migration guide below.

The interpretation rule $\tau$ maps raw success scores to resource-normalized metrics, enabling fair comparison across devices with different architectures, compilation strategies, and shot budgets. Each normalization targets a specific resource constraint relevant to practitioners:

### Retention-based Normalization (Recommended)

#### Log Loss per 2-Qubit Gate

Measures average information loss per entangling gate operation using proper information theory.

$$
\text{log\_loss\_per\_2q} = \frac{-\log(p_{\text{succ}})}{G_{2q}}
$$

where $G_{2q}$ is the total number of two-qubit gates.

**Why this replaces η_gate:**
- Properly models multiplicative error accumulation (gates compound exponentially)
- Monotone and well-behaved near $p_{\text{succ}} \to 0$ (unlike linear $p/G$ which is unstable)
- Information-theoretic interpretation: bits of information lost per gate
- Smaller values are better (less loss per gate)

**Interpretation:**
- `0.001`: ~0.1% information loss per gate (excellent)
- `0.01`: ~1% loss per gate (good for NISQ devices)
- `0.1`: ~10% loss per gate (challenging for longer circuits)

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_log_loss_with_ci

loss, loss_ci = calculate_log_loss_with_ci(p_succ, (ci_lower, ci_upper), two_qubit_gates)
```

---

#### Retention per 2-Qubit Gate

Geometric mean success retention per gate operation (intuitive percentage form).

$$
\text{retention\_per\_2q} = p_{\text{succ}}^{1/G_{2q}}
$$

**Why this replaces η_gate:**
- Equivalent to exponential form of log loss: $\exp(-\text{log\_loss\_per\_2q})$
- Provides intuitive "percent retained per gate" interpretation
- Monotone increasing (closer to 1.0 is better)
- Stable across full range $p \in [0, 1]$

**Interpretation:**
- `0.999`: 99.9% retention per gate, 0.1% loss (excellent)
- `0.99`: 99% retention per gate, 1% loss (good)
- `0.95`: 95% retention per gate, 5% loss (challenging)

**Example:**
```python
# Algorithm A: p_succ=0.75, gates=60
retention_A = 0.75 ** (1/60) = 0.9952  # 99.52% per gate

# Algorithm B: p_succ=0.50, gates=40  
retention_B = 0.50 ** (1/40) = 0.9827  # 98.27% per gate

# Algorithm A has better per-gate fidelity despite lower overall success
```

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_retention_with_ci

retention, retention_ci = calculate_retention_with_ci(p_succ, (ci_lower, ci_upper), two_qubit_gates)
```

**Confidence Interval Propagation:**
Both metrics include CI transforms since they're monotone functions of $p_{\text{succ}}$. Retention is increasing (preserves CI order), log loss is decreasing (reverses CI order).

---

#### Volume-based Retention Metrics

Same concepts applied to circuit volume $V = w \times d_c$:

$$
\text{log\_loss\_per\_volume} = \frac{-\log(p_{\text{succ}})}{V}, \quad \text{retention\_per\_volume} = p_{\text{succ}}^{1/V}
$$

**Use when:**
- Comparing architectures with different gate sets (volume abstracts gate details)
- Aligning with Volumetric Benchmarking standards
- Circuit depth and width are key constraints

---

### Shot Budget Metrics (Replaces η_shot)

#### Shots to Detect with 95% Reliability

Answers: "How many shots do I need for 95% confidence of seeing at least one valid solution?"

$$
N_{\text{detect}} = \left\lceil \frac{\log(1 - r)}{\log(1 - p_{\text{succ}})} \right\rceil
$$

where $r = 0.95$ is the target reliability.


For IID shots with per-shot success probability $p_{\text{succ}}$,
$$
\Pr(\text{no successes in }N) = (1-p_{\text{succ}})^N,\quad \Pr(\ge 1\text{ success}) = 1-(1-p_{\text{succ}})^N.
$$
Requiring $\Pr(\ge 1\text{ success}) \ge r$ implies $(1-p_{\text{succ}})^N \le 1-r$, which yields the expression above after taking logs.

**Useful approximation:** For small $p_{\text{succ}}$, $\log(1-p_{\text{succ}}) \approx -p_{\text{succ}}$, so
$$
N_{\text{detect}} \approx \frac{-\ln(1-r)}{p_{\text{succ}}}.
$$

**Why this replaces η_shot:**
- Original `eta_shot = p_succ / shots` had 1/N² scaling artifact (doubling shots halved the metric even with constant success)
- Shot budgets directly answer practitioner questions: "How many runs do I need?"
- Inversely proportional to $p_{\text{succ}}$ (intuitive: lower success = more shots needed)

**Interpretation:**
- `shots_detect_point = 34`: Expect to need 34 shots for 95% confidence
- `shots_detect_pessimistic = 42`: Conservative estimate (uses CI lower bound)
- `shots_detect_optimistic = 29`: Optimistic estimate (uses CI upper bound)
- Range `[29, 42]` is an uncertainty band *induced by the CI on* $p_{\text{succ}}$ (a practical heuristic, not a formal CI on $N_{\text{detect}}$)

**Implementation notes:**
- Handle edge cases explicitly: $p_{\text{succ}}=0 \Rightarrow N_{\text{detect}}=\infty$ (or a sentinel); $p_{\text{succ}}=1 \Rightarrow N_{\text{detect}}=1$
- Use numerically stable logs: `log1p(-r)` and `log1p(-p_succ)`

**Assumption:** Shots are independent and identically distributed (fixed $p_{\text{succ}}$ per shot). Correlations or drift can make $N_{\text{detect}}$ over-optimistic.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_shot_budgets

budgets = calculate_shot_budgets(p_succ, (ci_lower, ci_upper), reliability=0.95)
# Returns: {
#   "shots_detect_point": 34,
#   "shots_detect_pessimistic": 42,
#   "shots_detect_optimistic": 29,
#   "reliability": 0.95,
# }
```

---

### Cost-Normalized Efficiency Metrics (Heuristic Alternatives)

> **Note:** These metrics provide alternative normalization approaches to the geometric-mean retention metrics. They use different cost models (product, weighted sum, exponential decay) and may suit different analysis needs.

These metrics complement the retention-based approach by offering tunable cost functions that can be adapted to hardware characteristics or fitted from empirical data.

#### Product-Based Normalization (η_×)

**Formula:**

$$
\eta_{\times} = \frac{p_{\text{succ}}}{\text{depth} \times n_{2q}}
$$

**Interpretation:**
- "Success probability per unit of circuit volume (depth × 2Q count)"
- Higher is better
- Treats the product as a single "volume" measure
- Strong penalty for simultaneously large depth and gate count

**When to use:**
- Comparing circuits with same compilation settings
- When both dimensions matter equally and you want simplicity
- Exploratory analysis before fitting a weighted model
- When depth and 2Q count are uncorrelated in your dataset

**Gotchas:**
- **Double-counting risk:** Depth and 2Q count are often correlated (more gates → deeper circuit). Multiplying them can over-penalize compared to treating them independently.
- **Sensitive to scheduling:** Re-scheduling that reduces depth without changing 2Q count can dramatically change this metric.
- **Not directly actionable:** Doesn't answer "how many shots until success?" (see shot budgets) or "what's the per-gate failure rate?" (see retention metrics).

**Example:**
```python
from sudoku_nisq.metrics.calculators import calculate_eta_product

eta_prod = calculate_eta_product(p_succ=0.8, depth=10, two_qubit_gates=5)
# Returns: 0.016 (0.8 / 50)

# Same p_succ but 4× cost → 1/4 the efficiency
eta_prod2 = calculate_eta_product(p_succ=0.8, depth=20, two_qubit_gates=10)
# Returns: 0.004
```

---

#### Weighted-Sum Normalization (η_+)

**Formula:**

$$
C = \alpha \cdot \text{depth} + \beta \cdot n_{2q}
$$

$$
\eta_{+} = \frac{p_{\text{succ}}}{C}
$$

**Interpretation:**
- "Success per unit weighted cost"
- Higher is better
- Tunable to hardware/workload characteristics via α, β
- Avoids over-penalization from product formulation

**Weight selection strategies:**

1. **α=0, β=1:** "Per 2Q gate" (ignores depth entirely) — use when gate errors dominate
2. **α=1, β=0:** "Per depth" (ignores gates) — use when decoherence dominates
3. **α=1, β=1:** Simple balanced blend (default)
4. **Hardware-informed:** α ~ layer decoherence time, β ~ 2Q error rate
5. **Fitted weights:** Empirical from dataset using `fit_cost_weights()` (most accurate)

**When to use:**
- When one dimension dominates noise (adjust weights accordingly)
- Cross-hardware comparisons (refit weights per backend)
- When product penalty (η_×) seems excessive
- When you have insight into hardware characteristics

**Example:**
```python
from sudoku_nisq.metrics.calculators import calculate_eta_weighted_sum

# Default weights (α=1, β=1)
eta_wsum = calculate_eta_weighted_sum(p_succ=0.8, depth=10, two_qubit_gates=5)
# Returns: 0.0533 (0.8 / 15)

# Focus on 2Q gates only (α=0, β=1)
eta_wsum_gates = calculate_eta_weighted_sum(p_succ=0.8, depth=10, two_qubit_gates=5, 
                                             alpha=0, beta=1)
# Returns: 0.16 (0.8 / 5)

# Hardware-informed: depth costs 10× more than each 2Q gate
eta_wsum_hw = calculate_eta_weighted_sum(p_succ=0.8, depth=10, two_qubit_gates=5,
                                          alpha=10, beta=1)
# Returns: 0.00762 (0.8 / 105)
```

---

#### Decay Rate (k) — Exponential Model

**Formula:**

Assumes exponential decay model:

$$
p_{\text{succ}} \approx e^{-k \cdot C} \quad \text{where} \quad C = \alpha \cdot \text{depth} + \beta \cdot n_{2q}
$$

Solve for decay constant:

$$
k = \frac{-\ln(p_{\text{succ}})}{C}
$$

**Interpretation:**
- k is "decay constant" or "penalty per unit cost"
- **SMALLER is better** (less penalty per resource unit)
- If k is roughly constant across circuits, your cost model C captures the dominant scaling correctly
- Connects to physics: exponential fidelity decay with gates/time

**When to use:**
- Believe failures accumulate exponentially with cost (common in quantum computing)
- Want model-aligned metric (not just normalized ratio)
- Checking if cost model (α, β) fits your data
- Cross-validating against retention metrics (which also use geometric mean)

**Relationship to retention metrics:**
- `retention_per_2q = p_succ^(1/n_2q)` also captures exponential decay
- Decay rate generalizes to weighted cost C instead of just n_2q
- k = -log(p_succ) / C connects directly to exponential decay rate

**Example:**
```python
from sudoku_nisq.metrics.calculators import calculate_decay_rate

# High success, moderate cost → small k (good)
k = calculate_decay_rate(p_succ=0.8, depth=10, two_qubit_gates=5)
# Returns: 0.01489 (-log(0.8) / 15)

# Low success → larger k (worse)
k_low = calculate_decay_rate(p_succ=0.1, depth=10, two_qubit_gates=5)
# Returns: 0.1536 (-log(0.1) / 15)
```

---

#### Fitting Weights from Data

If you have a diverse dataset of circuits with varying depth and gate counts, you can empirically fit the weights α and β:

```python
from sudoku_nisq.metrics.calculators import fit_cost_weights

# Collect results: [(p_succ, depth, two_qubit_gates), ...]
results = [
    (0.8, 10, 5),
    (0.5, 20, 10),
    (0.3, 30, 15),
]

alpha, beta, r_squared = fit_cost_weights(results)
print(f"Fitted weights: α={alpha:.4f}, β={beta:.4f}, R²={r_squared:.3f}")

# Use fitted weights in subsequent calculations
eta_fitted = calculate_eta_weighted_sum(p_succ, depth, gates, alpha=alpha, beta=beta)
k_fitted = calculate_decay_rate(p_succ, depth, gates, alpha=alpha, beta=beta)
```

**Interpreting fit results:**
- **α (depth weight):** Penalty per layer (reflects decoherence exposure)
- **β (2Q weight):** Penalty per 2Q gate (reflects gate errors)
- **R² near 1:** Model fits well; cost function captures scaling
- **R² near 0:** Cost model doesn't explain your data; consider different factors

**Requirements for good fit:**
- 10+ diverse circuits (vary depth and gates independently)
- Exponential decay assumption holds
- Low outlier sensitivity (consider robust regression if needed)

---

#### Comparison Guide: Which Metric to Use?

| **Metric** | **Cost Model** | **Best For** | **Key Trade-off** |
|------------|----------------|--------------|-------------------|
| `retention_per_2q` | Geometric mean per gate | Multiplicative per-gate efficiency | Ignores depth; focuses only on gates |
| `retention_per_volume` | Geometric mean per volume | Multiplicative per-volume efficiency | Assumes uniform cost per volume unit |
| `eta_product` (η_×) | depth × n_2q | Simple volume penalty | Risk of double-counting correlated dims |
| `eta_weighted_sum` (η_+) | α·depth + β·n_2q | Tunable blend | Requires weight selection/fitting |
| `decay_rate` (k) | Log-transformed weighted cost | Exponential decay model | Same as η_+ but log-scaled (physics-aligned) |

**Decision tree:**
1. **Start with `retention_per_2q`** (geometric mean baseline) → most statistically principled
2. **Add `eta_weighted_sum`** with default α=β=1 → simple additive cost baseline
3. **Fit weights** if you have 10+ diverse circuits → hardware-specific optimization
4. **Compare `decay_rate` with retention** → if trends agree, model is robust
5. **Use `eta_product`** for quick exploratory analysis when depth/gates uncorrelated

**Important:** These are complementary views, not replacements. Use multiple metrics to triangulate on true efficiency.

---

### Deprecated Metrics (Removal in v0.5.0)

#### ~~η_gate = p_succ / G_2q~~ (DEPRECATED)

**Issues:**
- Linear normalization doesn't match multiplicative error model
- Unstable near $p \to 0$ (small changes in $p$ cause large swings)
- Not monotone for comparisons at different $p$ values

**Replacement:** Use `retention_per_2q` or `log_loss_per_2q` (see above)

---

#### ~~η_volume = p_succ / V~~ (DEPRECATED)

Same issues as η_gate, applied to circuit volume.

**Replacement:** Use `retention_per_volume` or `log_loss_per_volume`

---

#### ~~η_shot = p_succ / shots~~ (DEPRECATED)

**Fatal Issue:** Since $p_{\text{succ}}$ is computed from the same `shots`, the formula creates 1/N² scaling: doubling shots halves the metric even if true success rate is constant. This makes the metric meaningless for cost-efficiency analysis.

**Replacement:** Use `shots_detect_*` metrics (see above)

---

## Success & Coverage Metrics

### Success probability (p_succ)

Fraction of measured bitstrings that correspond to _valid exact covers_ (valid Sudoku solutions).

- Direct indicator of quantum algorithm correctness.
- Universally interpretable: regardless of device architecture, solution structure, or QPU mapping.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_p_succ

p_succ = calculate_p_succ(counts, validator)
```

---

### Distinct valid solutions observed ("coverage")

Number of _unique_ valid solutions observed across all shots or runs.

- Grover amplification in multi-solution spaces does not guarantee the algorithm outputs _all_ solutions.
- Devices or noise profiles may collapse the distribution, biasing towards one solution.
- Coverage reveals **whether the distribution is multimodal** or collapsed due to noise or interference.

Coverage allows distinguishing "one good solution found" from "algorithm broadly samples valid space." This distinction is important for _search_ complexity and for applications that benefit from multiple solutions (e.g., optimization variants).

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_distinct_solutions

coverage = calculate_distinct_solutions(counts, validator)
```

---

## Ranking & Coverage Metrics
### Count-Based Ranking Metrics (Still Useful)

> **Note:** These metrics count unique bitstrings rather than weighting by probability. See **Mass-Weighted Ranking Metrics** above for sampling-focused alternatives that weight outcomes by frequency.
### Top-k valid mass

Probability mass of the $k$ most-probable valid solutions.

$$
	\text{Top-}k\ \text{mass} \,=\, \sum_{i=1}^{k} p(x_i)\quad\text{where } x_1,\ldots,x_k \text{ are the top-}k \text{ valid solutions}
$$

- In noisy quantum devices, probability mass spreads. Top-k mass quantifies _how concentrated_ high-quality outcomes are.
- Helps quantify _solution sharpness_ vs _solution diffuseness_.
- Often used in ML ranking tasks; here, it measures "usefulness density" among valid solutions.

Provides a richer picture than p_succ alone; shows whether correct answers dominate the output distribution.

**Interpretation:**
- **High top-k mass**: Valid solutions concentrated at high probabilities (good amplification)
- **Low top-k mass**: Valid solutions spread across distribution (poor amplification or high noise)

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_top_k_valid_mass

k_values = [1, 3, 5, 10]
top_k_mass = calculate_top_k_valid_mass(counts, validation_context, k_values)
```

---

### Precision@k and Recall@k

- **Precision@k**: among the top-k returned bitstrings, what fraction are valid?
- **Recall@k**: among all valid solutions (total possible for the problem), what fraction appear in the top-k returned bitstrings?

$$
\mathrm{Precision@}k \,=\, \frac{\#\,\text{valid in top-}k}{k}
\qquad\qquad
\mathrm{Recall@}k \,=\, \frac{\#\,\text{valid in top-}k}{\#\,\text{total valid solutions}}
$$

These metrics are standard in classical ML/search evaluation. They make quantum device output _comparable_ to classical solvers and meaningful for practitioners.

- Quantum search tasks are _ranking tasks_: the device produces a distribution over bitstrings, and we assess how well it prioritizes the correct ones.
- Precision@k evaluates "quality of highest-probability samples."
- Recall@k evaluates "coverage among the top results."

**Interpretation:**
- **High precision, high recall**: Excellent - valid solutions dominate top-k
- **High precision, low recall**: Good ranking but limited coverage
- **Low precision, high recall**: Many solutions found but mixed with invalid results
- **Low precision, low recall**: Poor - algorithm not finding or ranking solutions well

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import (
    calculate_precision_at_k,
    calculate_recall_at_k
)

# total_valid_count from classical enumeration (e.g., 2 solutions for 2×2)
prec = calculate_precision_at_k(counts, validation_context, k_values=[1,3,5,10])
recall = calculate_recall_at_k(counts, validation_context, total_valid_count=2, k_values=[1,3,5,10])
```

**Note:**  
The `total_valid_count` parameter represents the total number of possible valid solutions for the problem, typically obtained from classical enumeration.

---

### Mass-Weighted Ranking Metrics (Improved)

> **⚠️ New Metrics:** Count-based precision/recall metrics treat all bitstrings equally. Mass-weighted variants account for probability, providing more intuitive sampling-based interpretations.

#### Mass Precision@k

Fraction of probability mass that is valid within the top-k outcomes.

$$
\text{mass\_precision@}k = \frac{\sum_{i=1}^k p(x_i) \cdot \mathbb{1}[x_i \in \text{Sol}(I)]}{\sum_{i=1}^k p(x_i)}
$$

where top-k is ranked by measurement frequency (descending).

**Interpretation:**
- Answers: "If I sample from the top-k most frequent outcomes, what fraction of my samples will be valid?"
- `mass_precision@k = 0.85`: 85% of probability mass in top-k is valid
- More intuitive than count-based precision for sampling scenarios
- Not monotone (can increase/decrease with k depending on distribution)

**Example:**
```python
# Counts: {"00": 500, "01": 300, "10": 150, "11": 50}, validator: ["00", "10"]
# k=1: "00" valid, mass=0.5/1.0 = 100%
# k=2: "00"+"01", valid_mass=0.5, total=0.8, = 62.5%
# k=3: adds "10" (valid), valid_mass=0.65, total=0.95, = 68.4% (increases!)
```

---

#### Valid Mass Capture@k

Fraction of all valid probability mass contained in the top-k outcomes.

$$
\text{capture@}k = \frac{\sum_{i=1}^k p(x_i) \cdot \mathbb{1}[x_i \in \text{Sol}(I)]}{p_{\text{succ}}}
$$

**Interpretation:**
- Answers: "How much of the valid probability is concentrated in the top-k list?"
- `capture@k = 0.92`: Top-k contains 92% of all valid probability
- Monotone increasing (more coverage as k grows)
- High capture at small k → valid solutions highly concentrated (good amplification)

**Example:**
```python
# Same data, p_succ = 0.65
# k=1: captures 0.50/0.65 = 76.9% of valid mass
# k=2: still 76.9% (k=2 adds invalid "01")
# k=3: captures 0.65/0.65 = 100% (all valid mass in top-3)
```

**Use Together:**
- High `mass_precision` + high `capture`: Excellent (clean top-k with complete coverage)
- Low `mass_precision` + high `capture`: Top-k polluted with invalid but finds all valid
- High `mass_precision` + low `capture`: Clean top-k but missing valid solutions
- Low both: Poor algorithm performance

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import (
    calculate_mass_precision_at_k,
    calculate_valid_mass_capture_at_k
)

k_values = [1, 3, 5, 10]
mass_prec = calculate_mass_precision_at_k(counts, validation_context, k_values)
capture = calculate_valid_mass_capture_at_k(counts, validation_context, k_values)
```

---

## Uncertainty & Statistical Robustness

### Exact Clopper–Pearson confidence intervals (CIs) for p_succ

An exact binomial confidence interval for success probability. Exact binomial confidence interval using the Clopper-Pearson method, which guarantees coverage for any sample size.

Benchmark comparisons must include uncertainty. Without confidence intervals, two devices' p_succ values cannot be meaningfully compared.

- Quantum sampling is _finite-shot,_ often small-shot.
- Normal approximations can be misleading for small sample sizes or extreme probabilities.
- Clopper–Pearson is exact, distribution-free, and guarantees coverage.

**Interpretation:**
- Narrow CI: High confidence in p_succ estimate (many shots or extreme probability)
- Wide CI: Low confidence (few shots or probability near 0.5)
- Non-overlapping CIs between devices: Statistically significant difference

**Note:**  
Clopper–Pearson CIs are implemented via `calculate_clopper_pearson_ci`.
```python
from sudoku_nisq.metrics.calculators.success_metrics import SuccessMetricsCalculator
from sudoku_nisq.metrics.calculators import calculate_clopper_pearson_ci

valid_shots = SuccessMetricsCalculator.count_valid_shots(counts, validator)
total_shots = sum(counts.values())
p_succ = valid_shots / total_shots

ci_lower, ci_upper = calculate_clopper_pearson_ci(valid_shots, total_shots, confidence_level=0.95)
```

---

### Valid Odds (Recommended) and Peak Discrimination

> **⚠️ Metrics Update:** "SNR" has been renamed to "Valid Odds" for statistical honesty. The formula is identical, but the name now correctly reflects what it measures: an odds ratio, not signal-to-noise.

#### Valid Odds

Ratio of valid to invalid probability (odds ratio).

$$
\text{valid\_odds} = \frac{p_{\text{succ}}}{1 - p_{\text{succ}}}
$$

**Why "odds" instead of "SNR":**
- This is mathematically an odds ratio, a standard statistical concept
- "Signal-to-noise ratio" implies amplitude or power ratios with specific statistical properties
- Honest naming prevents confusion with engineering SNR definitions

**Interpretation:**
- **odds < 1**: More invalid than valid outcomes (algorithm struggles)
- **odds = 1**: Equal valid and invalid probability (50/50)
- **odds > 10**: Excellent discrimination (>90% valid mass)
- **odds → ∞**: Perfect discrimination (no invalid measurements)
- **odds = 0**: No valid measurements

**Confidence Intervals:**
Odds CI is derived from $p_{\text{succ}}$ CI using monotone transform. Since odds is increasing in $p$, order is preserved:

$$
\text{odds\_ci} = \left(\frac{p\_\text{lower}}{1 - p\_\text{lower}}, \frac{p\_\text{upper}}{1 - p\_\text{upper}}\right)
$$

**JSON Representation:**
When $p_{\text{succ}} \geq 1$ (perfect success), `valid_odds` is stored as `null` with companion field `valid_odds_is_infinite: true` to maintain JSON compatibility.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_valid_odds_with_ci

odds, odds_ci, is_infinite = calculate_valid_odds_with_ci(p_succ, (ci_lower, ci_upper))
# Returns: (7.0, (5.77, 8.47), False) for p_succ=0.875
```

---

#### Peak Discrimination Metrics

While valid odds measures overall probability mass ratio, **peak metrics** reveal distribution shape by comparing the most frequent valid and invalid outcomes.

**Motivation:**
- Two algorithms can have same `valid_odds` but different concentration
- Peak metrics detect whether valid solutions truly dominate the frequency ranking
- Sensitive to amplitude amplification quality

**Metrics:**
- `p_best_valid`: Probability of most frequent valid solution
- `p_best_invalid`: Probability of most frequent invalid solution  
- `peak_ratio = p_best_valid / p_best_invalid`: How much best valid beats best invalid
- `peak_gap = p_best_valid - p_best_invalid`: Absolute separation

**Interpretation:**
- `peak_ratio > 1`: Best valid solution more frequent than best invalid (good)
- `peak_ratio >> 1`: Strong concentration on valid solutions (excellent amplification)
- `peak_gap > 0.1`: Large separation (clear winner in frequency ranking)
- `peak_ratio → ∞`: No invalid solutions observed (perfect)

**Example:**
```python
# Algorithm A: Concentrated on one valid solution
# p_best_valid=0.50, p_best_invalid=0.03
# peak_ratio=16.67, peak_gap=0.47 → Excellent

# Algorithm B: Spread across many valid solutions
# p_best_valid=0.15, p_best_invalid=0.12
# peak_ratio=1.25, peak_gap=0.03 → Weak discrimination despite high valid_odds
```

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import calculate_peak_metrics

peak = calculate_peak_metrics(counts, validation_context)
# Returns: {"p_best_valid": 0.50, "p_best_invalid": 0.03, 
#           "peak_ratio": 16.67, "peak_gap": 0.47, "peak_ratio_is_infinite": False}
```

**Use Together:**
- `valid_odds` tells you overall discrimination (how much total valid mass)
- `peak_ratio` tells you shape discrimination (is probability concentrated?)
- High odds + high peak ratio = excellent algorithm performance
- High odds + low peak ratio = valid solutions spread thin (may need more amplification rounds)

---

### ~~SNR (signal-to-noise ratio)~~ (DEPRECATED — Use Valid Odds)

**Name changed to Valid Odds (see above).** The formula is identical: $p_{\text{succ}} / (1 - p_{\text{succ}})$

**Why deprecated:** "SNR" is a misleading name for what is mathematically an odds ratio. The new name accurately reflects the statistical concept.

**Migration:** Replace `snr` field with `valid_odds` in code and analysis scripts. Deprecated calculator `calculate_snr()` emits a warning and will be removed in v0.5.0.

---

## Hardware & Compilation Attribution Metrics

To ensure reproducibility and interpretability, we must record:

- **Calibration timestamp**: When hardware was last calibrated
- **Hardware error rates**: Single-qubit, two-qubit, and readout errors
- **Coherence times**: T1, T2 when available
- **Transpiler configuration**: Seed and optimization level
- **Circuit fingerprint**: Transpiled circuit hash for reproducibility
- **Resource usage**: Gate counts by type, depth, volume

Application performance cannot be interpreted without knowing:

1. **Hardware conditions** (noise, coherence)
2. **Compilation decisions** (mapping, gate decomposition)

**Note:**  
Availability of hardware metadata is provider-dependent.

---

## Variability and Reproducibility Metrics

### Inter-run variability

Since single-run results can be misleading without variability metrics. Repeat each experiment ≥3 times across different calibrations or transpiler seeds; report mean ± std or IQR. 

Quantum devices exhibit:

- **Calibration drift**: Error rates change over time
- **Randomized compilation effects**: Different gate decompositions
- **Stochastic transpiler mapping**: Random qubit assignments
- **Load-dependent performance**: Other users affect device performance

**Interpretation:**
- Low std: Consistent performance across runs
- High std: High variability - consider more runs or investigate causes
- IQR complements std for non-normal distributions

**Implementation:**

Repeat each experiment ≥3 times across different calibrations or transpiler seeds; report mean ± std or IQR.

**Note:**  
`BenchmarkSuite` multi-run aggregation system is under development. Currently, users must run experiments multiple times manually and aggregate results using standard statistical tools.

---

## Classical Baseline Metrics

### Classical solver time-to-first-solution & time-to-full-enumeration

- **Time-to-first-solution**: Wall-clock time for classical solver to find one valid solution
- **Time-to-full-enumeration**: Wall-clock time to find all valid solutions

To evaluate whether quantum devices provide a speed or accuracy advantage for any instance size or difficulty.

- Without classical baselines, a quantum benchmark cannot speak to _utility_ or _advantage_.
- Even if quantum does not surpass classical today, it anchors results in a real computational landscape.
- Industry-standard practice: every quantum optimization/result paper includes classical baselines for fairness.

**Interpretation:**
- Speedup > 1: Quantum faster than classical
- Speedup < 1: Classical faster (common for small instances)
- Consider both time and success probability for fair comparison
- Note: Comparison should account for quantum execution time from `ExecutionResult.execution_time`

**Note:**  
The `ClassicalBaseline` system is under development. Classical solver integration (Algorithm X, DLX for exact cover problems) will be available in a future release. For now, users should benchmark classical solvers separately using standard tools.