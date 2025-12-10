# Benchmarking Metrics System - Architecture Summary

## What We've Designed

A **modular, extensible benchmarking metrics system** for the sudoku-nisq quantum solver framework that implements modern quantum algorithm evaluation standards.

## Key Features

### ✅ Comprehensive Metrics Coverage

Implements all 7 categories from your requirements:

1. **Success Metrics**: p_succ, distinct solutions (coverage)
2. **Ranking Metrics**: Top-k valid mass, Precision@k, Recall@k
3. **Statistical Metrics**: Clopper-Pearson CIs, SNR
4. **Efficiency Metrics**: η_gate, η_volume, η_shot
5. **Hardware Metadata**: Calibration data, error rates, T1/T2
6. **Variability Metrics**: Multi-run mean/std/IQR
7. **Classical Baselines**: Time-to-solution comparisons

### ✅ Provider Agnostic Design

- Abstract `MetadataCollector` interface
- Provider-specific implementations (Qiskit, PyTKET, Braket)
- Graceful degradation when metadata unavailable
- Works with simulators and real hardware

### ✅ Modular Architecture

```
metrics/
├── data_models.py          # Type-safe data structures
├── calculators/            # Pure metric computation functions
│   ├── success_metrics.py
│   ├── ranking_metrics.py
│   ├── statistical_metrics.py
│   ├── efficiency_metrics.py
│   └── variability_metrics.py
├── collectors/             # Provider-specific data collection
│   ├── base_collector.py
│   ├── qiskit_collector.py
│   ├── pytket_collector.py
│   └── braket_collector.py
├── aggregators/            # Multi-run aggregation
├── reporters/              # Export & visualization
└── benchmarking/           # High-level orchestration
```

### ✅ Backward Compatible Integration

- Extends existing `QuantumSolver.run()` with optional `collect_metrics` parameter
- Preserves existing `decode_counts()` method
- No breaking changes to current API

### ✅ Production-Ready Standards

- Type hints throughout
- Comprehensive error handling
- TODO markers for provider-dependent features
- Clear separation of concerns

## Implementation Status

### ✅ Complete (Architecture Phase)

- [x] Full architectural design document
- [x] Implementation roadmap (8 weeks)
- [x] Quick start guide
- [x] Data model definitions
- [x] Module structure skeleton
- [x] Sample calculator implementation

### 📋 Ready to Implement (Phase 1)

Can start immediately:
- Calculator modules (success, ranking, statistical, efficiency, variability)
- Unit tests for calculators
- No external dependencies except scipy

### 🔄 Requires Integration (Phase 3+)

Depends on calculator completion:
- Solver integration (`QuantumSolver.run()` modification)
- Provider-specific collectors
- Benchmark orchestration

## Key Design Decisions

### 1. Dataclasses for Type Safety

```python
@dataclass
class MetricsResult:
    p_succ: float
    p_succ_ci_lower: float
    p_succ_ci_upper: float
    # ... all metrics with types
```

**Why**: Type hints enable IDE autocomplete, catch bugs early, enable serialization.

### 2. Separate Calculators from Collectors

```python
# Calculator: Pure computation
p_succ = SuccessMetricsCalculator.calculate_p_succ(counts, validator)

# Collector: Provider-specific data extraction
hw_meta = QiskitMetadataCollector().collect_hardware_metadata(backend)
```

**Why**: Calculators are testable in isolation, collectors handle SDK differences.

### 3. ValidationContext for Flexibility

```python
validation_ctx = ValidationContext(
    valid_solutions=puzzle.enumerate_all_solutions(),
    total_valid_count=puzzle.count_solutions(),
    solution_validator=lambda bs: is_valid(bs)
)
```

**Why**: Decouples solution validation from metrics, supports both Sudoku and generic exact cover.

### 4. Optional Metadata Collection

```python
# Lightweight: just get result
result = solver.run(backend, shots=1024)

# Comprehensive: get result + metrics
result, metrics = solver.run(backend, shots=1024, collect_metrics=True, 
                              validation_context=ctx)
```

**Why**: Users can opt into metrics overhead only when needed.

### 5. Graceful Degradation

```python
if circuit_volume is None:
    # Volume calculation not available for this SDK
    eta_volume = None
else:
    eta_volume = p_succ / circuit_volume
```

**Why**: Not all providers expose all metadata. System still works with partial data.

## Integration Points

### With Existing Code

1. **`quantum_solver.py`**:
   - Add `collect_metrics` parameter to `run()`
   - Add `_calculate_metrics()` helper method
   - Maintains backward compatibility

2. **`exact_cover_solver.py`**:
   - Add `calculate_metrics()` method (new API)
   - Keep `decode_counts()` (existing API)
   - Both return structured results

3. **`metadata_manager.py`**:
   - No changes required
   - Metrics system is separate from caching

4. **Provider classes**:
   - No changes to provider APIs
   - Collectors wrap existing backends

### New User Workflow

```python
# 1. Setup (unchanged)
solver = ExactCoverQuantumSolver(puzzle)

# 2. Create validation context (new)
validation_ctx = ValidationContext(
    valid_solutions=puzzle.enumerate_all_solutions(),
    total_valid_count=puzzle.count_solutions(),
    solution_validator=lambda bs: solver._is_valid_solution(bs)
)

# 3. Run with metrics (enhanced)
result, metrics = solver.run(
    backend=backend,
    shots=2048,
    collect_metrics=True,  # New parameter
    validation_context=validation_ctx  # New parameter
)

# 4. Access structured metrics (new)
print(f"p_succ: {metrics.p_succ:.4f} [{metrics.p_succ_ci_lower:.4f}, "
      f"{metrics.p_succ_ci_upper:.4f}]")
print(f"η_gate: {metrics.eta_gate:.6f}")
```

## Critical TODOs by Provider

### IBM Qiskit
- ✅ Hardware metadata: `backend.properties()` API well-documented
- ✅ Compilation metadata: Transpiler provides layout info
- ⚠️ Circuit volume: Requires DAG layer analysis (doable)
- ✅ Execution time: Available in job result

### Quantinuum PyTKET
- ⚠️ Hardware metadata: API differs from Qiskit (need research)
- ⚠️ Compilation metadata: PyTKET pass manager metadata
- ⚠️ Circuit volume: Different circuit representation
- ⚠️ Execution time: Need to check pytket-quantinuum API

### AWS Braket
- ⚠️ Hardware metadata: Different per device type (IonQ, Rigetti, OQC)
- ⚠️ Compilation metadata: Provider-specific transpilation
- ⚠️ Circuit volume: Braket circuit structure
- ✅ Execution time: Available in task result

**Legend**: ✅ = Straightforward | ⚠️ = Requires investigation

## Dependencies

### Core (Phase 1)
- `scipy` (for Clopper-Pearson CI)
- `numpy` (already in project)

### Optional (Phase 6)
- `tabulate` (for table formatting)
- `matplotlib` (for plotting)

### Classical Solver (Phase 5)
- TBD: python-constraint, pycosat, or custom Algorithm X

## Success Criteria

The implementation will be successful when:

1. ✅ Can compute all 7 metric categories
2. ✅ Works with Qiskit and PyTKET backends
3. ✅ Multi-run benchmarks capture variability
4. ✅ Can generate publication-ready tables/figures
5. ✅ Backward compatible with existing code
6. ✅ Performance overhead < 5%
7. ✅ Comprehensive documentation

## Timeline

- **Phase 1-3** (3 weeks): Core functionality with simulators
- **Phase 4-5** (3 weeks): Provider integration + orchestration
- **Phase 6-7** (2 weeks): Visualization + polish
- **Total**: 8 weeks to production-ready system

**Fast-track option**: Phase 1-3 provides complete metrics for simulator-based development.

## Next Steps

### Immediate Actions

1. **Review architecture documents**:
   - [metrics_system_design.md](metrics_system_design.md) - Full technical spec
   - [metrics_implementation_roadmap.md](metrics_implementation_roadmap.md) - Detailed phases
   - [metrics_quick_start.md](../guide/metrics_quick_start.md) - User guide
   - This summary

2. **Begin Phase 1 implementation**:
   - Implement calculator modules
   - Write unit tests
   - No integration required yet

3. **Set up development environment**:
   - Add scipy to dependencies
   - Create test fixtures for calculators

### Questions to Resolve

1. **Classical solver preference**: Which library for exact cover baselines?
2. **Provider priority**: Which provider to implement first? (Suggest: IBM Qiskit)
3. **Testing resources**: Access to real quantum hardware for testing?
4. **Reporting preferences**: JSON + Markdown sufficient, or need LaTeX/HTML?

## Documentation Structure

```
docs/
├── architecture/
│   ├── metrics_system_design.md        # This is the technical bible
│   ├── metrics_implementation_roadmap.md  # Phase-by-phase plan
│   └── metrics_architecture_summary.md    # This document
└── guide/
    └── metrics_quick_start.md          # User-facing tutorial
```

## Files Created

### Architecture Documents (3)
- `docs/architecture/metrics_system_design.md` (400+ lines)
- `docs/architecture/metrics_implementation_roadmap.md` (300+ lines)
- `docs/guide/metrics_quick_start.md` (200+ lines)

### Code Structure (8 files)
- `src/sudoku_nisq/metrics/__init__.py`
- `src/sudoku_nisq/metrics/data_models.py` (200+ lines)
- `src/sudoku_nisq/metrics/calculators/__init__.py`
- `src/sudoku_nisq/metrics/calculators/success_metrics.py` (100+ lines, example)
- `src/sudoku_nisq/metrics/collectors/__init__.py`
- `src/sudoku_nisq/metrics/aggregators/__init__.py`
- `src/sudoku_nisq/metrics/reporters/__init__.py`
- `src/sudoku_nisq/metrics/benchmarking/__init__.py`

## References

All design decisions are grounded in:
- 2025 quantum benchmarking literature
- IBM Quantum benchmarking standards
- Information retrieval metrics (Precision/Recall@k)
- Statistical best practices (Clopper-Pearson)
- Modern software architecture patterns

## Contact Points

For implementation questions, refer to:
- **Data structures**: See `data_models.py` docstrings
- **Calculator logic**: See calculator module docstrings
- **Integration**: See `metrics_system_design.md` Section 4
- **Provider specifics**: See `metrics_system_design.md` Section 6
- **Usage examples**: See `metrics_quick_start.md`

---

**This architecture is ready for implementation. All major design decisions are documented and justified. The module structure supports gradual implementation without blocking development.**
