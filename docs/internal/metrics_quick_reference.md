# Metrics System - Quick Reference Card

> ⚠️ **PRE-ALPHA/BETA**: System under active development. APIs subject to change.

## 📊 All Available Metrics

### 1️⃣ Success & Coverage
| Metric | Symbol | Definition | Why It Matters |
|--------|--------|------------|----------------|
| Success Probability | `p_succ` | Fraction of valid measurements | Primary correctness indicator |
| Coverage | - | # of unique valid solutions | Reveals distribution collapse |
| Confidence Interval | CI | 95% Clopper-Pearson bounds | Statistical significance |

### 2️⃣ Ranking Quality
| Metric | Symbol | Definition | Purpose |
|--------|--------|------------|---------|
| Top-k Valid Mass | - | Probability of top-k valid solutions | Solution concentration |
| Precision@k | P@k | Valid fraction in top-k | Ranking quality |
| Recall@k | R@k | Coverage of valid in top-k | Solution diversity |

### 3️⃣ Signal Quality
| Metric | Symbol | Definition | Interpretation |
|--------|--------|------------|----------------|
| Signal-to-Noise | SNR | valid_mass / invalid_mass | Noise floor assessment |

### 4️⃣ Resource Efficiency
| Metric | Symbol | Formula | Normalizes By |
|--------|--------|---------|---------------|
| Gate Efficiency | η_gate | p_succ / G_2q | Two-qubit gates |
| Volume Efficiency | η_volume | p_succ / V | Circuit volume |
| Shot Efficiency | η_shot | p_succ / shots | Measurement shots |

### 5️⃣ Hardware Context
- Calibration timestamp
- Error rates (1q, 2q, readout)
- Coherence times (T1, T2)
- Transpilation metadata

### 6️⃣ Reproducibility
- Inter-run mean ± std
- Interquartile range (IQR)
- Seed variation effects

### 7️⃣ Classical Baseline
- Time-to-first-solution
- Time-to-enumerate-all
- Solution correctness

---

## 🚀 Basic Usage Pattern

```python
# 1. Setup
from sudoku_nisq.metrics import ValidationContext

validation_ctx = ValidationContext(
    valid_solutions=puzzle.enumerate_all_solutions(),
    total_valid_count=puzzle.count_solutions(),
    solution_validator=lambda bs: solver._is_valid_solution(bs)
)

# 2. Run with metrics
result, metrics = solver.run(
    backend=backend,
    shots=2048,
    collect_metrics=True,
    validation_context=validation_ctx
)

# 3. Access metrics
print(f"p_succ: {metrics.p_succ:.4f} [{metrics.p_succ_ci_lower:.4f}, {metrics.p_succ_ci_upper:.4f}]")
print(f"Coverage: {metrics.distinct_valid_solutions} solutions")
print(f"SNR: {metrics.snr:.2f}")
print(f"η_gate: {metrics.eta_gate:.6f}")
```

---

## 📈 Multi-Run Benchmarking

```python
from sudoku_nisq.metrics.benchmarking import BenchmarkSuite

benchmark = BenchmarkSuite(solver, validation_ctx, n_runs=5)
results = benchmark.run_benchmark(backend=backend, shots=2048)

agg = results['aggregated']
print(f"p_succ: {agg.p_succ_mean:.4f} ± {agg.p_succ_std:.4f}")
```

---

## 🎯 When to Use Which Metric

| Your Question | Use These Metrics |
|---------------|-------------------|
| "Does it work?" | p_succ, CI, SNR |
| "How well does it rank solutions?" | Precision@k, Recall@k, Top-k mass |
| "Is it efficient?" | η_gate, η_volume, η_shot |
| "Is it reproducible?" | Multi-run mean ± std, IQR |
| "Better than classical?" | Classical baseline time |
| "What were device conditions?" | Hardware metadata |

---

## 🔢 Interpretation Guidelines

### Success Probability (p_succ)
- **p_succ > 0.9**: Excellent
- **p_succ 0.5-0.9**: Good
- **p_succ 0.1-0.5**: Moderate
- **p_succ < 0.1**: Poor (noisy device or hard problem)

### Signal-to-Noise Ratio (SNR)
- **SNR > 10**: Excellent signal clarity
- **SNR 1-10**: Good, manageable noise
- **SNR < 1**: Noise dominates
- **SNR = ∞**: Perfect (no invalid measurements)

### Precision@k
- **P@k > 0.9**: Top results are mostly valid
- **P@k 0.5-0.9**: Decent ranking
- **P@k < 0.5**: Poor ranking quality

### Recall@k
- **R@k > 0.8**: Good coverage in top-k
- **R@k 0.3-0.8**: Moderate coverage
- **R@k < 0.3**: Limited coverage

### Efficiency Metrics (η)
- **Higher is better** for all η-metrics
- Compare relative values across backends/configurations
- No universal "good" threshold (problem-dependent)

---

## 📚 Documentation Navigation

| Document | Purpose | Audience |
|----------|---------|----------|
| [Metrics Reference](metrics_reference.md) | Complete metric definitions | All users |
| [Quick Start](metrics_quick_start.md) | Usage tutorial | New users |
| [Architecture Design](../architecture/metrics_system_design.md) | Technical spec | Developers |
| [Implementation Roadmap](../architecture/metrics_implementation_roadmap.md) | Development plan | Contributors |

---

## ⚠️ Development Status by Component

| Component | Status | Available? |
|-----------|--------|-----------|
| Data models | ✅ Complete | Yes |
| Success metrics | ✅ Implemented | Yes |
| Ranking metrics | 📋 Phase 1 | Soon |
| Statistical metrics | 📋 Phase 1 | Soon |
| Efficiency metrics | 📋 Phase 1 | Soon |
| Variability metrics | 📋 Phase 1 | Soon |
| Qiskit collector | 📋 Phase 4 | Later |
| Benchmark suite | 📋 Phase 5 | Later |
| Visualization | 📋 Phase 6 | Later |

**Legend**: ✅ Done | 📋 Designed, not implemented

---

## 🤝 Best Practices

1. **Always report confidence intervals** with p_succ
2. **Use multi-run benchmarks** (n ≥ 5) for publications
3. **Include hardware metadata** for reproducibility
4. **Compare against classical baselines** for context
5. **Report multiple η-metrics** for fair comparison
6. **Document transpilation settings** (seed, opt_level)

---

## 💡 Common Patterns

### Quick Single-Run Check
```python
result, metrics = solver.run(..., collect_metrics=True, validation_context=ctx)
print(metrics.summary_str())
```

### Production Multi-Run Benchmark
```python
benchmark = BenchmarkSuite(solver, ctx, n_runs=5, different_seeds=True)
results = benchmark.run_benchmark(backend, shots=2048)
```

### Cross-Backend Comparison
```python
for name, backend in backends.items():
    result, metrics = solver.run(backend, collect_metrics=True, validation_context=ctx)
    print(f"{name}: {metrics.p_succ:.4f}")
```

### Export Results
```python
from sudoku_nisq.metrics.reporters import JSONReporter
JSONReporter.export(metrics, "results/metrics.json")
```

---

## 🐛 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ValidationContext` required | Pass `validation_context=ctx` to `run()` |
| Hardware metadata is None | Expected for simulators; check provider documentation |
| Circuit volume is None | Not all SDKs support volume calculation yet |
| η_volume returns None | Use η_gate instead for now |

---

## 📞 Need Help?

- **Detailed definitions**: See [Metrics Reference](metrics_reference.md)
- **Usage examples**: See [Quick Start Guide](metrics_quick_start.md)
- **Implementation details**: See [System Design](../architecture/metrics_system_design.md)
- **Report issues**: Open a GitHub issue

---

**Last Updated**: December 10, 2025  
**Version**: 0.1.0-pre-alpha
