# Benchmarking Metrics Reference Guide

> ⚠️ **DEVELOPMENT STATUS**: This module is currently in development. 
> APIs and interfaces may change before the stable release.

## Overview

This guide provides detailed definitions, rationale, and importance for each metric in the sudoku-nisq benchmarking system. These metrics follow modern quantum algorithm evaluation standards and enable fair comparison across quantum devices, providers, and problem instances.

---

# 1. Success & Coverage Metrics

## **1.1 Success probability (p_succ)**

**Definition:**  
Fraction of measured bitstrings that correspond to _valid exact covers_ (valid Sudoku solutions).

**Rationale:**

- **Most direct indicator** of quantum algorithm correctness.
- Universally interpretable: regardless of device architecture, solution structure, or QPU mapping.
- Used in virtually all modern application benchmarks (optimization, chemistry, search, QAOA, and Grover-based tasks).
- A single scalar allows rapid comparison between devices, but still contains semantic meaning (probability of obtaining a correct solution per shot).

**Why essential:**  
Without p_succ, no statement can be made about real-world usefulness. It is the **primary outcome variable**.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import SuccessMetricsCalculator

p_succ = SuccessMetricsCalculator.calculate_p_succ(counts, validator)
```

**Related metrics:**
- Clopper-Pearson confidence intervals (Section 3.1)
- Gate-normalized success η_gate (Section 4.1)

---

## **1.2 Distinct valid solutions observed ("coverage")**

**Definition:**  
Number of _unique_ valid solutions observed across all shots or runs.

**Rationale:**

- Grover amplification in multi-solution spaces does not guarantee the algorithm outputs _all_ solutions.
- Devices or noise profiles may collapse the distribution, biasing towards one solution.
- Coverage reveals **whether the distribution is multimodal** or collapsed due to noise or interference.

**Why essential:**  
Coverage allows distinguishing "one good solution found" from "algorithm broadly samples valid space." This distinction is important for _search_ complexity and for applications that benefit from multiple solutions (e.g., optimization variants).

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import SuccessMetricsCalculator

coverage = SuccessMetricsCalculator.calculate_distinct_solutions(counts, validator)
```

**Interpretation:**
- Low coverage (1-2 solutions): Distribution collapsed, possible noise bias
- Medium coverage: Partial sampling of solution space
- High coverage (close to total): Broad exploration of valid solutions

---

# 2. Ranking & Coverage Metrics

## **2.1 Top-k valid mass**

**Definition:**  
Probability mass of the k most-probable _valid_ solutions.

**Rationale:**

- In noisy quantum devices, probability mass spreads. Top-k mass quantifies _how concentrated_ high-quality outcomes are.
- Helps quantify _solution sharpness_ vs _solution diffuseness_.
- Used in ML ranking tasks; here, it measures "usefulness density" among valid solutions.

**Why essential:**  
Provides a richer picture than p_succ alone; shows whether correct answers dominate the output distribution.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import RankingMetricsCalculator

top_k_mass = RankingMetricsCalculator.calculate_top_k_valid_mass(
    counts, validator, k_values=[1, 3, 5, 10]
)
# Returns: {1: 0.15, 3: 0.35, 5: 0.45, 10: 0.60}
```

**Interpretation:**
- High top-k mass: Valid solutions concentrated at high probabilities (good amplification)
- Low top-k mass: Valid solutions spread across distribution (poor amplification or high noise)

---

## **2.2 Precision@k, Recall@k**

**Definition:**

- **Precision@k**: among the top-k returned bitstrings, what fraction are valid?
- **Recall@k**: among all valid solutions, what fraction appear in the top-k returned bitstrings?

**Rationale:**

- Quantum search tasks are _ranking tasks_: the device produces a distribution over bitstrings, and we assess how well it prioritizes the correct ones.
- Precision@k evaluates "quality of highest-probability samples."
- Recall@k evaluates "coverage among the top results."

**Why essential:**  
These metrics are standard in classical ML/search evaluation. They make quantum device output _comparable_ to classical solvers and meaningful for practitioners.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import RankingMetricsCalculator

precision = RankingMetricsCalculator.calculate_precision_at_k(
    counts, validator, k_values=[1, 3, 5, 10]
)

recall = RankingMetricsCalculator.calculate_recall_at_k(
    counts, validator, total_valid_count=num_solutions, k_values=[1, 3, 5, 10]
)
```

**Interpretation:**
- **High precision, high recall**: Excellent - valid solutions dominate top-k
- **High precision, low recall**: Good ranking but limited coverage
- **Low precision, high recall**: Many solutions found but mixed with invalid results
- **Low precision, low recall**: Poor - algorithm not finding or ranking solutions well

---

# 3. Uncertainty & Statistical Robustness

## **3.1 Exact Clopper–Pearson confidence intervals (CIs) for p_succ**

**Definition:**  
An exact binomial confidence interval for success probability.

**Rationale:**

- Quantum sampling is _finite-shot,_ often small-shot.
- Normal approximations can be misleading for small sample sizes or extreme probabilities.
- Clopper–Pearson is exact, distribution-free, and guarantees coverage.

**Why essential:**  
Benchmark comparisons must include uncertainty. Without confidence intervals, two devices' p_succ values cannot be meaningfully compared.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import StatisticalMetricsCalculator

valid_shots = SuccessMetricsCalculator.count_valid_shots(counts, validator)
total_shots = sum(counts.values())

ci_lower, ci_upper = StatisticalMetricsCalculator.clopper_pearson_ci(
    successes=valid_shots,
    trials=total_shots,
    confidence=0.95  # 95% confidence interval
)

print(f"p_succ: {p_succ:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**Interpretation:**
- Narrow CI: High confidence in p_succ estimate (many shots or extreme probability)
- Wide CI: Low confidence (few shots or probability near 0.5)
- Non-overlapping CIs between devices: Statistically significant difference

**Best practices:**
- Always report CIs alongside p_succ
- Use at least 1000 shots for reasonable CI width
- Consider increasing shots if CI is too wide for meaningful comparison

---

## **3.2 SNR (signal-to-noise ratio), robust definition**

**Definition:**  
Ratio of total valid probability mass to total invalid probability mass.

**Mathematical formula:**
```
SNR = (total valid mass) / (total invalid mass)
    = p_succ / (1 - p_succ)
```

**Rationale:**

- SNR captures "how well the device separates solutions from noise."
- High SNR implies a clear signal; low SNR indicates noise dominates.

**Why essential:**  
SNR is complementary to p_succ:

- Two devices may have the same p_succ but vastly different noise floors.
- SNR helps diagnose quality of amplitude amplification and underlying coherence.

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import StatisticalMetricsCalculator

snr = StatisticalMetricsCalculator.calculate_snr(counts, validator)
```

**Interpretation:**
- SNR > 10: Excellent signal clarity (>90% valid mass)
- SNR 1-10: Good signal with manageable noise (50-90% valid mass)
- SNR < 1: Noise dominates signal (<50% valid mass)
- SNR = ∞: Perfect (no invalid measurements)
- SNR = 0: No valid measurements

---

# 4. Resource-Normalized Efficiency Metrics

In modern benchmarking, reporting **only** p_succ is insufficient. We need normalized metrics that expose _efficiency_, not just correctness.

## **4.1 Gate-normalized success: η_gate = p_succ / G_2q**

**Definition:**  
Success probability per two-qubit gate.

**Mathematical formula:**
```
η_gate = p_succ / G_2q
```
where G_2q is the total number of two-qubit gates in the circuit.

**Rationale:**

- Two-qubit gates are the main contributors to error in NISQ and early post-NISQ systems.
- Normalizing by G_2q allows comparing circuits with different topologies, mappings, and decompositions.

**Why essential:**  
It ties performance directly to a physically meaningful resource (error-prone operations).

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import EfficiencyMetricsCalculator

eta_gate = EfficiencyMetricsCalculator.calculate_eta_gate(
    p_succ=0.45,
    two_qubit_gates=150
)
# Result: 0.003 (0.3% success per 2q gate)
```

**Interpretation:**
- Higher η_gate: More efficient use of two-qubit gates
- Enables comparison across different circuit decompositions
- Accounts for compilation quality (different transpilations have different 2q gate counts)

---

## **4.2 Volume-normalized success: η_volume = p_succ / V**

**Definition:**  
Success probability divided by total circuit _volume_ (sum of active gates per layer).

**Mathematical formula:**
```
η_volume = p_succ / V
```
where V = Σ(number of active gates in layer i) over all layers.

**Rationale:**

- Circuit volume reflects how much "quantum stuff" the system actually had to maintain in superposition.
- Volume captures depth × parallelism more faithfully than G×D.
- Volume is already used in cross-platform benchmarks and is architecture-agnostic.

**Why essential:**  
Volume is one of the few resource metrics that is **universally comparable across hardware types** (superconducting, trapped ion, neutral atoms).

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import EfficiencyMetricsCalculator

eta_volume = EfficiencyMetricsCalculator.calculate_eta_volume(
    p_succ=0.45,
    circuit_volume=500  # May be None if not calculable
)
```

**Note:**  
⚠️ Circuit volume calculation is provider-specific and may not be available for all SDKs in the initial release. Will return `None` if unavailable.

**Interpretation:**
- Higher η_volume: Better efficiency per unit of quantum computation
- Comparable across different hardware architectures
- Captures both circuit depth and gate parallelism

---

## **4.3 Shot-normalized success: η_shot = p_succ / shots**

**Definition:**  
Success probability per shot.

**Mathematical formula:**
```
η_shot = p_succ / shots
```

**Rationale:**

- Some devices allow more shots cheaply; others penalize them.
- Shot efficiency measures the marginal gain per sample.

**Why essential:**  
Connects algorithmic performance with practical usage patterns (latency, throughput).

**Implementation:**
```python
from sudoku_nisq.metrics.calculators import EfficiencyMetricsCalculator

eta_shot = EfficiencyMetricsCalculator.calculate_eta_shot(
    p_succ=0.45,
    shots=2048
)
# Result: 0.000220 (0.022% per shot)
```

**Interpretation:**
- Higher η_shot: Better return per measurement
- Useful for cost-benefit analysis (cloud pricing often per-shot)
- Helps determine optimal shot allocation

---

## Why multiple η-metrics are needed

Because **no single normalization reflects "resource usage" across all architectures**.

- **Gate-count** matters for superconducting devices where 2q gates are error-prone.
- **Depth** matters for decoherence-prone platforms with limited coherence times.
- **Volume** works universally across architectures.
- **Shot cost** matters for cloud economics and throughput constraints.

Therefore, a scientifically honest benchmark must report **multiple normalizations**, not force a single composite score.

**Best practice:**  
Report all available η-metrics in your benchmarking results to enable meaningful cross-platform comparison.

---

# 5. Hardware & Compilation Attribution Metrics

To ensure reproducibility and interpretability, we must record:

- **Calibration timestamp**: When hardware was last calibrated
- **Hardware error rates**: Single-qubit, two-qubit, and readout errors
- **Coherence times**: T1, T2 when available
- **Transpiler configuration**: Seed and optimization level
- **Circuit fingerprint**: Transpiled circuit hash for reproducibility
- **Resource usage**: Gate counts by type, depth, volume

**Rationale:**  
Application performance cannot be interpreted without knowing:

1. **Hardware conditions** (noise, coherence)
2. **Compilation decisions** (mapping, gate decomposition)

The 2025 benchmarking reviews explicitly emphasize _**hardware–compiler–application disentanglement**_.  
Reporting metadata is essential for reproducibility and scientific transparency.

**Implementation:**
```python
# Hardware metadata automatically collected when collect_metrics=True
result, metrics = solver.run(
    backend=backend,
    collect_metrics=True,
    validation_context=validation_ctx
)

# Access hardware metadata
hw_meta = metrics.hardware_metadata
if hw_meta:
    print(f"Backend: {hw_meta.backend_name}")
    print(f"Calibration: {hw_meta.calibration_timestamp}")
    print(f"Average T1: {sum(hw_meta.t1_times.values()) / len(hw_meta.t1_times)} μs")
    
# Access compilation metadata
comp_meta = metrics.compilation_metadata
if comp_meta:
    print(f"Optimization level: {comp_meta.optimization_level}")
    print(f"Transpiler seed: {comp_meta.transpiler_seed}")
```

**Note:**  
⚠️ Availability of hardware metadata depends on the provider. Simulators typically don't provide calibration data. The system gracefully degrades when metadata is unavailable.

---

# 6. Variability and Reproducibility Metrics

## **6.1 Inter-run variability**

**Definition:**  
Repeat each experiment ≥3 times across different calibrations or transpiler seeds; report mean ± std or IQR.

**Rationale:**  
Quantum devices exhibit:

- **Calibration drift**: Error rates change over time
- **Randomized compilation effects**: Different gate decompositions
- **Stochastic transpiler mapping**: Random qubit assignments
- **Load-dependent performance**: Other users affect device performance

Single-run results are misleading. Without variability metrics, benchmarking results are scientifically questionable.

**Implementation:**
```python
from sudoku_nisq.metrics.benchmarking import BenchmarkSuite

# Multi-run benchmark with variability tracking
benchmark = BenchmarkSuite(
    solver=solver,
    validation_context=validation_ctx,
    n_runs=5,  # Minimum 3, recommended 5+
    different_seeds=True  # Use different transpiler seeds
)

results = benchmark.run_benchmark(
    backend=backend,
    shots=2048,
    opt_level=2
)

# Access variability metrics
agg = results['aggregated']
print(f"p_succ: {agg.p_succ_mean:.4f} ± {agg.p_succ_std:.4f}")
print(f"IQR: [{agg.p_succ_iqr[0]:.4f}, {agg.p_succ_iqr[1]:.4f}]")

# Individual run results
for i, run in enumerate(results['individual_runs']):
    print(f"Run {i+1}: p_succ = {run.p_succ:.4f}")
```

**Best practices:**
- Use **n_runs ≥ 5** for publication-quality results
- Always report **mean ± std** or **median with IQR**
- Use different transpiler seeds to capture compilation variability
- If possible, span multiple calibration cycles

**Interpretation:**
- Low std: Consistent performance across runs
- High std: High variability - consider more runs or investigate causes
- IQR complements std for non-normal distributions

---

# 7. Classical Baseline Metrics

## **7.1 Classical solver time-to-first-solution & time-to-full-enumeration**

**Definition:**
- **Time-to-first-solution**: Wall-clock time for classical solver to find one valid solution
- **Time-to-full-enumeration**: Wall-clock time to find all valid solutions

**Rationale:**

- Without classical baselines, a quantum benchmark cannot speak to _utility_ or _advantage_.
- Even if quantum does not surpass classical today, it anchors results in a real computational landscape.
- Industry-standard practice: every quantum optimization/result paper includes classical baselines for fairness.

**Why essential:**  
To evaluate whether quantum devices provide a speed or accuracy advantage for any instance size or difficulty.

**Implementation:**
```python
from sudoku_nisq.metrics.benchmarking import ClassicalBaseline

# Create classical baseline
classical = ClassicalBaseline(solver.problem)

# Time first solution
first_sol = classical.time_to_first_solution(algorithm="dlx")
print(f"Classical time to first solution: {first_sol['time_seconds']:.4f}s")

# Time full enumeration
all_sols = classical.time_to_enumerate_all(algorithm="dlx")
print(f"Classical enumeration time: {all_sols['time_seconds']:.4f}s")
print(f"Solutions found: {all_sols['solution_count']}")

# Compare with quantum
quantum_time = sum(r.execution_time for r in metrics.execution_results)
speedup = first_sol['time_seconds'] / quantum_time
print(f"Quantum speedup factor: {speedup:.2f}x")
```

**Note:**  
⚠️ Classical solver selection is under development. Initial implementation will support standard exact cover algorithms (Algorithm X, DLX).

**Interpretation:**
- Speedup > 1: Quantum faster than classical
- Speedup < 1: Classical faster (common for small instances)
- Consider both time and success probability for fair comparison

---

# 8. Metric Composition

All metrics are designed to work together. A complete benchmark report should include:

1. **Core success metrics**:
   - p_succ with 95% CI
   - Distinct valid solutions
   - SNR

2. **Ranking metrics** (for k ∈ {1, 3, 5, 10}):
   - Top-k valid mass
   - Precision@k
   - Recall@k

3. **Efficiency metrics**:
   - η_gate
   - η_volume (if available)
   - η_shot

4. **Variability** (multi-run):
   - Mean ± std for p_succ
   - IQR for p_succ

5. **Hardware context**:
   - Calibration timestamp
   - Average error rates
   - T1/T2 coherence times

6. **Classical baseline**:
   - Time-to-first-solution
   - Comparison with quantum execution time

---

# 9. Reporting Guidelines

## Minimal Report (Single Run)

```
Success Probability: 0.4523 [0.4312, 0.4736]
Distinct Solutions: 3
SNR: 0.82
Gate Efficiency: 0.00301
Backend: ibm_brisbane
Shots: 2048
```

## Standard Report (Multi-Run)

```
Success Probability: 0.4523 ± 0.0312
  95% CI: [0.4312, 0.4736]
Distinct Solutions: 3.2 ± 0.4
SNR: 0.82 ± 0.09

Ranking Metrics:
  Top-3 Valid Mass: 0.3421
  Precision@3: 0.667
  Recall@3: 0.375

Efficiency:
  η_gate: 0.00301
  η_volume: 0.00090
  η_shot: 0.000221

Runs: 5
Backend: ibm_brisbane
Calibration: 2025-12-10 14:32:01
```

## Publication-Quality Report

Include all of the above plus:
- Hardware error rates (1q, 2q, readout)
- Compilation details (optimization level, seed)
- Classical baseline comparison
- Plots (success probability, ranking metrics)
- Full reproducibility information

---

# 10. Development Status & Roadmap

> ⚠️ **CURRENT STATUS**: Pre-alpha development

| Component | Status | Notes |
|-----------|--------|-------|
| Data models | ✅ Complete | All structures defined |
| Success metrics | ✅ Implemented | Basic functionality ready |
| Ranking metrics | Implementation Phase 1 |
| Statistical metrics | Implementation Phase 1 |
| Efficiency metrics | Implementation Phase 1 |
| Variability metrics | Implementation Phase 1 |
| Qiskit collector | Implementation Phase 4 |
| Benchmark suite | Implementation Phase 5 |
| Classical baseline | Implementation Phase 5 |
| Reporters | Implementation Phase 6 |