# Metrics System V2: Migration Guide

**Date**: December 29, 2025  
**Status**: Implemented, not yet published

## Summary

The metrics system has been completely redesigned to address statistical interpretability, JSON safety, and cost-realistic normalization issues. Since the original metrics were never published, we've replaced them directly rather than versioning.

## What Changed

### Stage 6 (Evaluation): Discrimination Metrics

| Old Metric | Issue | Replacement | Improvement |
|------------|-------|-------------|-------------|
| `snr` | Misleading name (actually odds ratio) | `valid_odds` + CI | Honest naming, JSON-safe (None instead of inf) |
| (none) | No shape-aware discrimination | `peak_metrics` | Measures best valid vs. best invalid directly |

**New metrics:**
- `valid_odds`: p_succ / (1 - p_succ) with confidence intervals
- `valid_odds_is_infinite`: Boolean flag for perfect discrimination (p_succ=1)
- `p_best_valid`, `p_best_invalid`, `peak_ratio`, `peak_gap`: Peak-based discrimination
- `peak_ratio_is_infinite`: Flag for no invalid solutions case

###Stage 6 (Evaluation): Ranking Metrics

| Old Metric | Issue | Addition | Improvement |
|------------|-------|----------|-------------|
| `precision_at_k` | Count-based, not sampling-aware | `mass_precision_at_k` | Probability-weighted precision |
| `recall_at_k` | (kept, still useful) | `valid_mass_capture_at_k` | Fraction of valid probability in top-k |

**Improvements:**
- All ranking metrics now use deterministic tie-breaking: `sorted(counts.items(), key=lambda x: (-x[1], x[0]))`
- `mass_precision@k = valid_mass_in_top_k / total_mass_in_top_k`
- `valid_mass_capture@k = valid_mass_in_top_k / total_valid_mass`

### Stage 7 (Normalization): Resource Metrics

| Old Metric | Issue | Replacement | Improvement |
|------------|-------|-------------|-------------|
| `eta_gate = p/n` | Linear scaling near p=0, unstable | `retention_per_2q`, `log_loss_per_2q` | Multiplicative error model, monotone |
| `eta_volume = p/V` | Same issue | `retention_per_volume`, `log_loss_per_volume` | Same improvement |
| `eta_shot = p/N` | 1/N² scaling artifact | `shots_detect_*` | Answers "shots for 95% confidence" |

**New metrics:**
- `log_loss_per_2q = -log(p_succ) / n_gates`: Per-gate information loss (smaller is better)
- `retention_per_2q = p_succ^(1/n_gates)`: Per-gate retention factor (closer to 1 is better)
- Both include CI propagation: `*_ci_lower`, `*_ci_upper`
- Volume variants: `log_loss_per_volume`, `retention_per_volume`
- Shot budgets: `shots_detect_point`, `shots_detect_pessimistic`, `shots_detect_optimistic`

### Edge Case Fixes

1. **CI for num_trials=0**: Now returns `(None, None)` instead of `(0.0, 0.0)` (undefined proportion)
2. **Infinite values**: All `float('inf')` replaced with `None` + companion boolean flags for JSON safety
3. **Tie-breaking**: Ranking metrics now deterministic across Python versions

## API Changes

### Calculator Imports

```python
# Old (deprecated but still available)
from sudoku_nisq.metrics.calculators import (
    calculate_snr,  # Deprecated
    calculate_eta_gate,  # Deprecated
    calculate_eta_volume,  # Deprecated
    calculate_eta_shot  # Deprecated
)

# New (recommended)
from sudoku_nisq.metrics.calculators import (
    calculate_valid_odds_with_ci,
    calculate_peak_metrics,
    calculate_mass_precision_at_k,
    calculate_valid_mass_capture_at_k,
    calculate_retention_with_ci,
    calculate_log_loss_with_ci,
    calculate_shot_budgets
)
```

### MetricsMetadataManager

No API changes - automatically computes new metrics when calling `record()`. JSON structure extended with new fields:

```json
{
  "run_id_123": {
    "timestamp": "2025-12-29T...",
    "stage_6_evaluation": {
      "p_succ": 0.875,
      "p_succ_ci_lower": 0.8523,
      "p_succ_ci_upper": 0.8944,
      "distinct_valid_solutions": 3,
      
      "valid_odds": 7.0,
      "valid_odds_ci_lower": 5.77,
      "valid_odds_ci_upper": 8.47,
      "valid_odds_is_infinite": false,
      
      "p_best_valid": 0.50,
      "p_best_invalid": 0.03,
      "peak_ratio": 16.67,
      "peak_gap": 0.47,
      "peak_ratio_is_infinite": false,
      
      "top_k_valid_mass": {"1": 0.65, "3": 0.82, "5": 0.87, "10": 0.88},
      "precision_at_k": {"1": 1.0, "3": 0.67, "5": 0.6, "10": 0.5},
      "recall_at_k": {"1": 0.33, "3": 0.67, "5": 0.83, "10": 1.0},
      "mass_precision_at_k": {"1": 0.65, "3": 0.85, "5": 0.92, "10": 0.88},
      "valid_mass_capture_at_k": {"1": 0.74, "3": 0.94, "5": 0.99, "10": 1.0}
    },
    "stage_7_normalization": {
      "log_loss_per_2q": 0.00223,
      "log_loss_per_2q_ci_lower": 0.00187,
      "log_loss_per_2q_ci_upper": 0.00267,
      "retention_per_2q": 0.9978,
      "retention_per_2q_ci_lower": 0.9974,
      "retention_per_2q_ci_upper": 0.9982,
      
      "log_loss_per_volume": 0.000096,
      "retention_per_volume": 0.999904,
      
      "shots_detect_point": 34,
      "shots_detect_pessimistic": 42,
      "shots_detect_optimistic": 29,
      "shot_budget_reliability": 0.95
    }
  }
}
```

## Deprecation Timeline

Since metrics were never published:

1. **Now**: New metrics active, old calculators deprecated with warnings
2. **v0.5.0** (target: Q2 2026): Remove `calculate_snr`, `calculate_eta_*` entirely
3. **Documentation**: Update all examples to use new metrics

## Testing Status

- ✅ Calculator functions implemented
- ✅ Data models updated
- ✅ MetricsMetadataManager integration complete
- ⏳ Comprehensive tests pending (see tests/test_metrics.py)
- ⏳ Documentation updates pending (see docs/internal/architecture/calculator_implementations.md)

## For Developers

### Running with new metrics

No code changes needed - just run experiments as before:

```python
from sudoku_nisq import QSudoku

puzzle = QSudoku.generate(size=4, num_missing_cells=2)
puzzle.set_solver(ExactCoverQuantumSolver, encoding="simple")
circuit = puzzle.build_circuit(sdk="qiskit")
result = puzzle.run_aer(shots=1024)

# Metrics auto-computed and stored in .quantum_solver_cache/{puzzle_hash}/stage_6_7_metrics.json
```

### Interpreting new metrics

**Valid odds** (replaces SNR):
- `odds < 1`: More invalid than valid
- `odds = 1`: Equal probability
- `odds > 1`: More valid than invalid
- `odds = None` + `is_infinite=True`: Perfect discrimination (100% valid)

**Peak ratio**:
- How much the best valid solution beats the best invalid
- `peak_ratio > 1`: Valid solutions dominate frequency ranking
- `peak_gap > 0`: Absolute separation between peaks

**Retention per gate**:
- Geometric mean success retention per gate operation
- `retention_per_2q = 0.9978` means 99.78% retained per gate, 0.22% loss
- Closer to 1.0 is better
- Enables intuitive comparison: "Algorithm A retains 99.5% per gate, B retains 98.2%"

**Shot budgets**:
- Directly answers: "How many shots for 95% confidence of finding solution?"
- `shots_detect_point=34`: Expected shots needed
- `pessimistic=42`: Conservative estimate (lower CI bound)
- `optimistic=29`: Optimistic estimate (upper CI bound)

## References

- Implementation: [src/sudoku_nisq/metrics/calculators/](src/sudoku_nisq/metrics/calculators/)
- Integration: [src/sudoku_nisq/metadata/metrics.py](src/sudoku_nisq/metadata/metrics.py)
- Data models: [src/sudoku_nisq/metrics/data_models.py](src/sudoku_nisq/metrics/data_models.py)
- Architecture docs: [docs/internal/architecture/calculator_implementations.md](docs/internal/architecture/calculator_implementations.md)
