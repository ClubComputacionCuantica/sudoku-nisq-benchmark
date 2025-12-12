# Benchmarking Metrics Reference Guide

> This module is currently under construction. 
> APIs and interfaces may change before the stable release.

## Introduction

Modern quantum benchmarking requires a **complete specification** $\mathbf{B}^\star$ that explicitly defines the entire workflow from algorithm to final metrics. This section establishes the theoretical foundation for how sudoku-nisq implements rigorous, reproducible benchmarks following established quantum computing evaluation standards, see e.g. [1]. 

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

Each stage has precise meaning and corresponding implementation in this system:

#### Stage 1: Algorithm & Test Instances ($\mathcal{I}$)


Selection of computational task and instantiation of test problem instances $\mathcal{I} = \{I_1, I_2, \ldots, I_N\}$ where $N \in \mathbb{N}$ is the number of test instances, each characterized by problem size, constraint structure, and solution space cardinality.

**sudoku-nisq implementation**

- `SudokuPuzzle(size=n)` or `ExactCoverProblem(universe, subsets)`
- Instance parameters: `puzzle.size`, `puzzle.open_tuples` (search space), `puzzle.pre_tuples` (constraints)
- Solution space: `ValidationContext.total_valid_count`

**Metrics connection:**
- Drives total solution count for recall@k calculations
- Determines problem complexity for cross-instance comparisons

**Code example:**
```python
from sudoku_nisq import SudokuPuzzle

# Generate instance (non-deterministic, no seed parameter)
puzzle = SudokuPuzzle.generate(size=4, num_missing_cells=8, canonicalize=True)
total_solutions = puzzle.count_solutions()  # |Sol(I)|

# Validate bitstring from measurement
validator = lambda bs: puzzle._solver._is_valid_solution(bs)
# Note: requires solver to be attached via QSudoku.set_solver()
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


#### Stage 2a: IR Construction – Logical Circuit

The logical, platform-agnostic circuit artifact that describes the algorithm's quantum behavior, encompassing the circuit structure and its interpretation.

For Grover-based exact-cover formulations, a useful shorthand is:

$$
	\text{IR}(I) = \mathcal{G}^{(r)}(\mathcal{O}_I, \mathcal{D})
$$

where:
- $I$: problem instance from test set $\mathcal{I}$
- $\mathcal{G}^{(r)}: \mathcal{H}_n \to \mathcal{H}_n$: Grover operator with $r \in \mathbb{N}$ iterations on $n$-qubit Hilbert space $\mathcal{H}_n$
- $\mathcal{O}_I: \{0,1\}^n \to \{0,1\}$: oracle encoding instance $I$ constraints (marks solutions)
- $\mathcal{D}$: diffusion operator (inversion-about-average)
- $r$: number of Grover iterations (typically $\sim \frac{\pi}{4}\sqrt{2^n/|\text{Sol}(I)|}$ for optimal amplification)

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

#### Stage 2b: IR Policy – Allowed Logical Transformation Family ($\mathsf{C}_{\text{IR}}$)

Defines what transformations are allowed during IR construction and before hardware mapping. This is the target-independent part of the compilation pipeline and governs legal implementation variants at the logical level.

The policy should be documented in SDK-agnostic terms: which families of gate decompositions are permissible for multi-controlled and oracle subroutines; whether and how ancilla may be used (clean-only vs reuse, measurement-based resets permitted or not); which transpilation techniques are in scope (e.g., algebraic/pattern rewrites, subgraph re-synthesis, gate unrolling) with abstract terminology; target-independent bounds on logical changes before hardware mapping (depth/size trade-offs, preserving oracle structure); and determinism requirements (fixed seeds, version-locked passes). Stage 2b policy operates at the logical IR level, whereas Stage 3 defines the target-aware mapping scope to native gates.


These policy elements should be documented in `CompilationMetadata` along with the IR resources and conversion provenance so that two runs of the same IR yield comparable native circuits under the same allowed transformation family. The IR policy ties directly to what we can record today: SDK choice and versions, IR-to-SDK conversion provenance, and a circuit hash at handoff; advanced pass families are out of scope, so we document only parameters and artifacts we can reliably capture.

**Code example:**
```python
from sudoku_nisq import QSudoku
qs = QSudoku(puzzle)
qs.set_solver("exact_cover", encoding="simple", decompose_cnz=True)
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
- $\text{IR}$: logical intermediate representation (platform-agnostic circuit)
- $\mathcal{H}$: hardware specification (topology graph, native gate set, qubit connectivity, coherence properties)
- $\theta \in \Theta$: compilation parameters from policy space $\Theta$ (optimization level, transpiler seed, routing strategy)
- $\text{Circuit}^{\text{native}}$: hardware-native circuit using gates from $\mathcal{H}$'s native set

**sudoku-nisq implementation:**
- `QuantumSolver._transpile_qiskit()`: Uses `generate_preset_pass_manager` or `qiskit.compiler.transpile`
- `QuantumSolver._transpile_pytket()`: Calls `backend.get_compiled_circuit()`
- Respects hardware topology, native gate sets, coupling maps

**Metrics connection:**
- `CompilationMetadata.optimization_level`: Explicit $\theta_{\text{opt}}$ parameter
- `CompilationMetadata.transpiler_seed`: Ensures reproducibility of $\mathsf{C}$
- `CompilationMetadata.post_transpile_gates`: Gate counts after compilation $G^{\text{post}}(\mathsf{C})$
- `CompilationMetadata.circuit_hash`: Fingerprint $h(\text{Circuit}^{\text{native}})$ for caching

Compilation metrics are constrained to recorded parameters: optimisation level, transpiler seed, backend target/alias, SDK path, and the post-transpile circuit fingerprint with gate/depth counts. When parameters aren’t set, note provider defaults to keep runs comparable.

Recorded compilation artifacts (deterministic and reproducible):
- `optimization_level`, `transpiler_seed`
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
- transpiler seed: record in `CompilationMetadata.transpiler_seed` for deterministic runs.
- backend target/alias: record the device or simulator alias, provider, and SDK used.
- SDK choice and versions: record whether the build used `pytket`, `qiskit`, or `braket`, and capture package versions involved in IR conversion/transpilation.
- circuit fingerprint: record the post-transpile circuit hash/fingerprint and cache key.
- resource summary: record gate counts, depth, and qubit count before/after transpilation.
- build-time decomposition choice: record solver option `decompose_cnz` when constructing oracles and multi-controlled subroutines.

When parameters aren’t set, note the provider/compiler defaults (e.g., preset pass manager vs `qiskit.transpile`, PyTKET `get_compiled_circuit`, and no client-side Braket transpilation). Documenting these items constrains $\mathsf{C}$ to a reproducible subset of behaviors without relying on pass-specific or ancilla-policy terminology.

**Code example:**
```python
result = qs.run(
    backend=backend,
    backend_alias="ibm_brisbane",
    shots=2048,
    optimisation_level=2,  # θ_opt
    transpiler_seed=42      # θ_seed for reproducible 𝖢
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
- `SuccessMetricsCalculator`: Implements $\alpha$ (counts → probabilities) and $\sigma$ (validation)
- `ValidationContext.solution_validator`: Defines $\text{Sol}(I)$ membership test

**Metrics connection:**
- `MetricsResult.p_succ`: Primary outcome $\sigma_1 = P(x \in \text{Sol}(I))$
- `MetricsResult.p_succ_ci_lower`, `p_succ_ci_upper`: Uncertainty quantification via Clopper-Pearson
- `MetricsResult.snr`: Signal quality $\sigma_2 = \frac{P(\text{valid})}{P(\text{invalid})}$
- `MetricsResult.distinct_valid_solutions`: Distribution breadth $|\{x: x \in \text{Sol}(I), p(x) > 0\}|$

**Analogy to standard benchmarks:**
- Heavy Output Probability (QV): $\sigma = P(x > \text{median}(\text{ideal distribution}))$
- Cross-Entropy Benchmarking: $\sigma = \sum_x p_{\text{ideal}}(x) \log p_{\text{measured}}(x)$
- Our framework: $\sigma = P(x \in \text{Sol}(I))$ with rich auxiliary metrics

**Code example:**
```python
from sudoku_nisq.metrics.calculators import SuccessMetricsCalculator

validator = lambda bitstring: puzzle.is_valid_solution(bitstring)
p_succ = SuccessMetricsCalculator.calculate_p_succ(result.counts, validator)
distinct = SuccessMetricsCalculator.calculate_distinct_solutions(result.counts, validator)
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
- `EfficiencyMetricsCalculator` : Computes resource-normalized scores
- See detailed implementation and interpretation in the **Resource-Normalized Efficiency Metrics** section below

**Metrics connection:**
- `MetricsResult.eta_gate = p_succ / two_qubit_gates`: Comparable to "algorithmic qubits" efficiency
- `MetricsResult.eta_volume = p_succ / (width × depth)`: Volumetric benchmarking analog
- `MetricsResult.eta_shot = p_succ / shots`: Cost-efficiency metric

**Alignment with standard benchmarks:**
- Quantum Volume: Reports $V_Q = 2^{n}$ for largest passing $n$ (single-number FOM)
- Algorithmic Qubits: Reports $\#AQ(d)$ for depth $d$ achieving target fidelity
- Our framework: Reports multiple normalizations to avoid imposing artificial composite scores

- $\eta_{\text{gate}}$ – aligns conceptually with Algorithmic Qubits: efficiency relative to two-qubit operations and circuit fidelity thresholds.
- $\eta_{\text{volume}}$ – aligns with Volumetric Benchmarking (width $w$ × depth $d_c$); note that provider-specific depth definitions and non-square circuits can affect comparability.
- $\eta_{\text{shot}}$ – reflects statistical cost-efficiency; uncertainty (CI width) scales with shot count.

**Code example:**
```python
# Manual calculation (automated in future release)
eta_gate = p_succ / result.two_qubit_gates
eta_volume = p_succ / (result.num_qubits * result.circuit_depth)
eta_shot = p_succ / result.shots

print(f"η_gate = {eta_gate:.6f}")
print(f"η_volume = {eta_volume:.6f}")
print(f"η_shot = {eta_shot:.2e}")
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
| 3 | **Compilation policy** | $\mathsf{C}(\theta)$ | Compilation policy with parameters $\theta$ | `CompilationMetadata`: `optimization_level`, `transpiler_seed`, backend target, implementation family |
| 3 | **Mapped circuit** | $\text{Circ}^{\text{native}}$ | Hardware-native circuit | Post-transpile gates $G^{\text{post}}$, depth, circuit hash, provider defaults |
| 4 | **Executable** | $\mathrm{Exec}$ | Executable pulses/schedules | Provider-controlled; `job_id`, backend compiler version (when available) |
| 5 | **Hardware state** | $\mathcal{H}_t$ | Hardware state at time $t$ | `HardwareMetadata`: calibration timestamp, $T_1$/$T_2$, gate errors at execution time |
| 5 | **Execution parameters** | $N_{\text{shots}}$ | Measurement shot budget | `ExecutionResult.shots`, timestamp, execution time |
| 6 | **Preprocessing** | $\alpha$ | Preprocessing map | `SuccessMetricsCalculator`: counts → probabilities, bitstring filtering |
| 6 | **Scoring functional** | $\sigma$ | Scoring functional | Success metrics: $p_{\text{succ}}$, confidence intervals, SNR, distinct solutions |
| 7 | **Normalization** | $\tau$ | Normalization rule | Efficiency metrics: $\eta_{\text{gate}}$, $\eta_{\text{volume}}$, $\eta_{\text{shot}}$, ranking metrics |

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

The interpretation rule $\tau$ maps raw success scores to resource-normalized metrics, enabling fair comparison across devices with different architectures, compilation strategies, and shot budgets. Each normalization targets a specific resource constraint relevant to practitioners:

### Gate-normalized success: η_gate

Success probability per two-qubit gate.

$$
\eta_{\mathrm{gate}} \,=\, \frac{p_{\mathrm{succ}}}{G_{\mathrm{2q}}}
$$

where G_2q is the total number of two-qubit gates in the circuit.

It ties performance directly to a physically meaningful resource (error-prone operations).

- Two-qubit gates are the main contributors to error in NISQ and early post-NISQ systems.
- Normalizing by G_2q allows comparing circuits with different topologies, mappings, and decompositions.

**Interpretation:**
- Higher η_gate: More efficient use of two-qubit gates
- Enables comparison across different circuit decompositions
- Accounts for compilation quality (different transpilations have different 2q gate counts)

**Alignment:** Conceptually consistent with Algorithmic Qubits metric — efficiency relative to two-qubit operations and circuit fidelity thresholds.

**Note:**  
`EfficiencyMetricsCalculator` under development. The `ExecutionResult.two_qubit_gates` field is available from `solver.run()` for manual calculation.

---

### Volume-normalized success: η_volume

Success probability divided by a circuit complexity proxy consistent with Volumetric Benchmarking (VB).

$$
\eta_{\mathrm{volume}} \,=\, \frac{p_{\mathrm{succ}}}{V}
$$

VB frames circuit complexity in terms of **width** and **depth**:

- Width $w$ (or $N_q$): number of qubits used
- Depth $d_c$: circuit layer count or number of sequential native-gate layers per qubit (provider-specific)

In this project, we define $V$ to align with VB conventions while remaining practical for application circuits:

- Preferred definition: $V = w \times d_c$ (width–depth product), using provider-reported or computed $d_c$.
- Optional refined definition (when available): $V = \sum \text{(active gates per layer)}$ to capture parallelism; this should be treated as a provider-specific enhancement and clearly documented when used.

This reconciles our earlier intuition (measuring how much quantum work is sustained) with standard VB methodology that emphasizes width and depth.

Note:

- Quantum Volume ($V_Q$): square circuits with $d_c = N_q$; reported as $V_Q = 2^{n_{\mathrm{pass}}}$ for the largest $N_q$ that passes.
- Algorithmic Qubits ($\#\mathrm{AQ}$): success regions on the $N_q$–$d_c$ plane; some definitions use total CNOT count as a proxy for depth.

**Interpretation:**
- Higher η_volume: More efficient success per unit of width–depth complexity
- Comparable across hardware via `w` and `d_c`; optionally more granular when using per-layer activity
- Accounts for both circuit depth and qubit count; refined per-layer definition additionally captures gate parallelism

**Alignment:** Directly consistent with Volumetric Benchmarking (width $w$ × depth $d_c$); note that provider-specific depth definitions and non-square circuits can affect comparability.

**Note:**
- `d_c` extraction is provider-dependent and not yet implemented for all SDKs. `ExecutionResult.circuit_depth`/`circuit_volume` may be `None`.
- When only gate counts are available, prefer reporting η_gate (Section 4.1) and include `w` and approximate `d_c` for context.

---

### Shot-normalized success: η_shot

Success probability per shot.

$$
\eta_{\mathrm{shot}} \,=\, \frac{p_{\mathrm{succ}}}{\text{shots}}
$$

Connects algorithmic performance with practical usage patterns (latency, throughput).

- Some devices allow more shots cheaply; others penalize them.
- Shot efficiency measures the marginal gain per sample.

**Interpretation:**
- Higher η_shot: Better return per measurement
- Useful for cost-benefit analysis (cloud pricing often per-shot)
- Helps determine optimal shot allocation

**Alignment:** Reflects statistical cost-efficiency; uncertainty (CI width) scales with shot count.

**Note:**  
`EfficiencyMetricsCalculator` under development. Calculate manually from `SuccessMetricsCalculator.calculate_p_succ()` and total shots.

---

## Success & Coverage Metrics

### Success probability (p_succ)

Fraction of measured bitstrings that correspond to _valid exact covers_ (valid Sudoku solutions).

- Direct indicator of quantum algorithm correctness.
- Universally interpretable: regardless of device architecture, solution structure, or QPU mapping.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import SuccessMetricsCalculator

p_succ = SuccessMetricsCalculator.calculate_p_succ(counts, validator)
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
from sudoku_nisq.metrics.calculators import SuccessMetricsCalculator

coverage = SuccessMetricsCalculator.calculate_distinct_solutions(counts, validator)
```

---

## Ranking & Coverage Metrics

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

**Note:**  
The `RankingMetricsCalculator` class is currently under development. Ranking metrics (top-k mass, precision@k, recall@k) will be available in a future release.

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

**Note:**  
Implementation pending. The `total_valid_count` parameter represents the total number of possible valid solutions for the problem, typically obtained from classical enumeration.

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
The `StatisticalMetricsCalculator` class is under development. For now, compute using:
```python
valid_shots = SuccessMetricsCalculator.count_valid_shots(counts, validator)
total_shots = sum(counts.values())
p_succ = valid_shots / total_shots

# Use scipy.stats.beta or statsmodels for Clopper-Pearson CI
# CI will be automated in future release
```

---

### SNR (signal-to-noise ratio), robust definition

Ratio of total valid probability mass to total invalid probability mass.

$$
\mathrm{SNR} \,=\, \frac{\text{total valid mass}}{\text{total invalid mass}} \,=\, \frac{p_{\mathrm{succ}}}{1 - p_{\mathrm{succ}}}
$$

SNR is complementary to p_succ:

- Two devices may have the same p_succ but vastly different noise floors.
- SNR helps diagnose quality of amplitude amplification and underlying coherence.

- SNR captures "how well the device separates solutions from noise."
- High SNR implies a clear signal; low SNR indicates noise dominates.

**Interpretation:**
- **SNR > 10**: Excellent signal clarity (>90% valid mass)
- **SNR 1–10**: Good signal with manageable noise (50–90% valid mass)
- **SNR < 1**: Noise dominates signal (<50% valid mass)
- **SNR = \(\infty\)**: Perfect (no invalid measurements)
- **SNR = 0**: No valid measurements

**Note:**  
`StatisticalMetricsCalculator` under development. Calculate manually as `p_succ / (1 - p_succ)` where `p_succ` is from `SuccessMetricsCalculator.calculate_p_succ()`.

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