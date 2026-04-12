# Deprecation Status Report

Generated: January 13, 2026  
**Status**: All legacy metadata features removed ✅

This document clarifies which features remain deprecated in the codebase.

---

## ✅ **Currently Deprecated Features (With Runtime Warnings)**

### 1. **`calculate_snr()` — Statistical Metrics**
- **Location**: [src/sudoku_nisq/metrics/calculators/statistical_metrics.py](src/sudoku_nisq/metrics/calculators/statistical_metrics.py#L85-L115)
- **Status**: ✅ **Emits DeprecationWarning**
- **Warning Message**: 
  ```
  calculate_snr() is deprecated. Use calculate_valid_odds(p_succ) from 
  odds_metrics.py instead. This function will be removed in v0.5.0.
  ```
- **Replacement**: `calculate_valid_odds()` from `odds_metrics.py`
- **Removal Timeline**: v0.5.0
- **Tests**: Verified with `pytest.warns(DeprecationWarning)` in tests/test_statistical_metrics.py

---

### 2. **`MetadataManager` Class and Methods**
- **Location**: [src/sudoku_nisq/metadata_manager.py](src/sudoku_nisq/metadata_manager.py)
- **Status**: ✅ **Entire class deprecated with decorator-based warnings**
- **Decorator**: `@_deprecated_method(alternative)` on selected methods
- **Environment Variable**: Set `SUDOKU_NISQ_SUPPRESS_DEPRECATION=1` to suppress warnings

#### Deprecated Methods with Runtime Warnings:
1. **`set_main_circuit_resources()`** (line 373)
   - Warning: "Use LogicalIRMetadataManager.record() or BenchmarkSession.execute_run()"
   - Removal: v2.0

2. **`set_backend_resources()`** (line 477)
   - Warning: "Use CompilationMetadataManager.record() or BenchmarkSession.execute_run()"
   - Removal: v2.0

3. **`get_main_circuit_resources()`** (line 624)
   - Warning: "Use LogicalIRMetadataManager.query() or BenchmarkSession"
   - Removal: v2.0

4. **`get_backend_resources()`** (line 663)
   - Warning: "Use BenchmarkSession.get_metrics_summary()"
   - Removal: v2.0

- **Class Docstring**: Contains Sphinx `.. deprecated::` directive for documentation
- **Replacement**: `BenchmarkSession` from `src/sudoku_nisq/metadata/benchmark_session.py`
- **Migration Guide**: [docs/guide/upgrading_from_metadata_manager.md](docs/guide/upgrading_from_metadata_manager.md)
- **Removal Timeline**: v2.0

---

## �️ **Removed Features (January 11, 2026)**

The following deprecated features have been **completely removed** from the codebase:

### 1. **`MetadataManager` Class** — REMOVED ✅
- **Previous Location**: `src/sudoku_nisq/metadata_manager.py` (deleted)
- **Status**: ✅ **Completely removed in January 2026**
- **Removed Files**:
  - `src/sudoku_nisq/metadata_manager.py`
  - `tests/test_metadata_manager.py`
  - `docs/api/metadata_manager.md`
  - `docs/internal/upgrading_guide.md`
  - `examples/migrate_to_benchmark_session.py`
  - `scripts/migrate_metadata.py`
- **Replacement**: Stage-aware metadata architecture (7-stage pipeline)
- **Migration**: Use `BenchmarkSession` for all metadata operations

### 2. **Linear Normalization Efficiency Metrics**
- ❌ **`calculate_eta_gate()`** - REMOVED
  - Replaced by: `calculate_retention_per_2q()` or `calculate_log_loss_per_2q()`
- ❌ **`calculate_eta_volume()`** - REMOVED
  - Replaced by: `calculate_retention_per_volume()` or `calculate_log_loss_per_volume()`
- ❌ **`calculate_eta_shot()`** - REMOVED
  - Replaced by: `shots_to_detect()` from shot_budget_metrics

**Files removed**:
- `src/sudoku_nisq/metrics/calculators/efficiency_metrics.py` ❌
- `tests/test_efficiency_metrics.py` ❌

**Rationale**: Linear normalization (p/n) doesn't properly model multiplicative error accumulation in quantum circuits. Retention-based metrics use geometric mean (nth root) which correctly captures per-resource efficiency.

---

## �📊 **MetricsResult Fields**

### In `src/sudoku_nisq/metrics/data_models.py`:

**Fields marked deprecated in docstrings:**
- `snr`: "Signal-to-noise ratio (DEPRECATED: use valid_odds)" — line 137
  - Still present in data model for backward compatibility
  - Calculators emit warnings when computing this field

**Legacy efficiency fields** (deprecated but still present):
- `eta_gate`, `eta_volume`, `eta_shot` — Present in data model
  - Fields themselves aren't deprecated (for backward compatibility)
  - **The calculator functions have been removed** (use retention metrics instead)

---

## ✅ **Migration Paths Available**

### From `calculate_snr()`:
```python
# Old (emits warning):
from sudoku_nisq.metrics.calculators import calculate_snr
snr = calculate_snr(counts, validation_context)

# New:
from sudoku_nisq.metrics.calculators import calculate_valid_odds, calculate_p_succ
p_succ = calculate_p_succ(counts, validator)
valid_odds = calculate_valid_odds(p_succ)
```

### From removed `calculate_eta_*()` functions:
```python
# Old (REMOVED):
from sudoku_nisq.metrics.calculators import calculate_eta_gate, calculate_eta_volume
# ImportError: cannot import name 'calculate_eta_gate'

# New (use retention-based metrics):
from sudoku_nisq.metrics.calculators import (
    calculate_retention_per_2q,
    calculate_retention_per_volume,
    calculate_log_loss_per_2q,
    shots_to_detect
)
retention_2q = calculate_retention_per_2q(p_succ, two_qubit_gates)
retention_vol = calculate_retention_per_volume(p_succ, circuit_volume)
log_loss = calculate_log_loss_per_2q(p_succ, two_qubit_gates)
shots_needed = shots_to_detect(p_succ, reliability=0.95)
```

---

## 📊 **Summary Table**

| Feature | Location | Status | Removal Date | Replacement |
|---------|----------|--------|--------------|-------------|
| `calculate_snr()` | statistical_metrics.py | ⚠️ Deprecated | v0.5.0 (planned) | `calculate_valid_odds()` |
| `calculate_eta_gate()` | efficiency_metrics.py | ❌ **REMOVED** | Jan 11, 2026 | `calculate_retention_per_2q()` |
| `calculate_eta_volume()` | efficiency_metrics.py | ❌ **REMOVED** | Jan 11, 2026 | `calculate_retention_per_volume()` |
| `calculate_eta_shot()` | efficiency_metrics.py | ❌ **REMOVED** | Jan 11, 2026 | `shots_to_detect()` |
| `MetadataManager` | metadata_manager.py | ❌ **REMOVED** | Jan 13, 2026 | `BenchmarkSession` |

---

## 🔧 **Action Items**

### For Users:
1. **Migrate** from `calculate_snr()` to `calculate_valid_odds()` before v0.5.0
2. Use `BenchmarkSession` for all metadata operations (legacy `MetadataManager` completely removed)

### For Maintainers:
1. ✅ **COMPLETED**: Removed `MetadataManager` and all legacy metadata code (Jan 13, 2026)
2. ✅ **COMPLETED**: Removed efficiency_metrics.py (Jan 11, 2026)
3. **Track removal timeline**:
   - v0.5.0: Remove `calculate_snr()` (planned)
   - v2.0: Remove `MetadataManager` class

---

## 📝 **Summary Table**

| Feature | Location | Runtime Warning? | Removal Version | Replacement |
|---------|----------|------------------|-----------------|-------------|
| `calculate_snr()` | statistical_metrics.py | ✅ Yes | v0.5.0 | `calculate_valid_odds()` |
| `calculate_eta_gate()` | efficiency_metrics.py | ✅ Yes | v0.5.0 | `calculate_retention_per_2q()` |
| `calculate_eta_volume()` | efficiency_metrics.py | ✅ Yes | v0.5.0 | `calculate_retention_per_volume()` |
| `calculate_eta_shot()` | efficiency_metrics.py | ✅ Yes | v0.5.0 | `shots_to_detect()` |
### For Users:
1. **Immediate**: Update any code using `calculate_eta_*()` functions (will cause ImportError)
2. **Migrate** from `calculate_snr()` to `calculate_valid_odds()` before v0.5.0
3. **Before v2.0**: Migrate from `MetadataManager` to `BenchmarkSession`
4. **Optional**: Set `SUDOKU_NISQ_SUPPRESS_DEPRECATION=1` during MetadataManager transition

### For Maintainers:
1. ✅ **COMPLETED**: Removed `MetadataManager` and all legacy metadata code (Jan 13, 2026)
2. ✅ **COMPLETED**: Removed efficiency_metrics.py (Jan 11, 2026)
3. **Track removal timeline**:
   - ✅ v0.4.0: Removed `calculate_eta_*()` functions (completed Jan 11, 2026)
   - v0.5.0: Remove `calculate_snr()` (planned)

---

## 🔍 **How to Check Your Code**

Run with warnings enabled:
```bash
# Show all deprecation warnings
python -W always::DeprecationWarning your_script.py

# Or in code:
import warnings
warnings.simplefilter('always', DeprecationWarning)
```

Expected warnings when using remaining deprecated features:
```
DeprecationWarning: calculate_snr() is deprecated. Use calculate_valid_odds...
```

**If you see ImportError for removed features:**
```python
ImportError: cannot import name 'calculate_eta_gate' from 'sudoku_nisq.metrics.calculators'
ImportError: cannot import name 'MetadataManager' from 'sudoku_nisq'
```
Update to retention-based metrics and `BenchmarkSession` (see migration examples above).

---

## ✅ **Verification**

- **Removed** (Jan 13, 2026): Legacy `MetadataManager` class and all associated code
- **Removed** (Jan 11, 2026): Linear normalization efficiency metrics (`calculate_eta_gate`, `calculate_eta_volume`, `calculate_eta_shot`)
- **Deprecated** (runtime warnings): `calculate_snr()`
- **Recommended**: Use `BenchmarkSession` for metadata operations and retention-based metrics for resource normalization
