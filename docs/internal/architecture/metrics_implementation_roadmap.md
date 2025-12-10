# Metrics System Implementation Roadmap

## Overview

This document provides a phased implementation plan for the benchmarking metrics system, with clear milestones and dependencies.

---

## Phase 1: Foundation (Week 1)

**Goal**: Implement core data structures and calculator logic with no external dependencies.

### Tasks

- [ ] **1.1** Create `src/sudoku_nisq/metrics/` directory structure
- [ ] **1.2** Implement `data_models.py` with all dataclasses:
  - `ExecutionResult`
  - `HardwareMetadata`
  - `CompilationMetadata`
  - `ValidationContext`
  - `MetricsResult`
- [ ] **1.3** Implement calculator modules (pure functions):
  - `calculators/success_metrics.py`
  - `calculators/ranking_metrics.py`
  - `calculators/statistical_metrics.py`
  - `calculators/efficiency_metrics.py`
  - `calculators/variability_metrics.py`
- [ ] **1.4** Add `scipy` dependency to `pyproject.toml` (for Clopper-Pearson)
- [ ] **1.5** Write unit tests for all calculators
  - Test edge cases (zero shots, no valid solutions, etc.)
  - Test against known analytical results

**Deliverables**:
- Fully tested calculator modules
- Data model definitions
- No integration with existing solvers yet

**Success Criteria**:
- All calculator tests pass
- 100% code coverage on calculator modules
- Type hints validated

---

## Phase 2: Collector Interfaces (Week 2)

**Goal**: Define abstract interfaces and implement provider-agnostic aggregation.

### Tasks

- [ ] **2.1** Create `collectors/base_collector.py` with `MetadataCollector` ABC
- [ ] **2.2** Implement `aggregators/multi_run_aggregator.py`
- [ ] **2.3** Create stub implementations for collectors:
  - `collectors/qiskit_collector.py` (with TODOs)
  - `collectors/pytket_collector.py` (with TODOs)
  - `collectors/braket_collector.py` (with TODOs)
- [ ] **2.4** Write tests for aggregator logic
- [ ] **2.5** Document collector interface contracts

**Deliverables**:
- Abstract collector interface
- Working multi-run aggregator
- Stub collectors for each provider

**Success Criteria**:
- Aggregator correctly computes mean/std/IQR
- Collector interface is well-documented
- Stub collectors instantiate without errors

---

## Phase 3: Solver Integration (Week 3)

**Goal**: Integrate metrics calculation into existing solver infrastructure.

### Tasks

- [ ] **3.1** Extend `QuantumSolver` base class:
  - Add `collect_metrics` parameter to `run()` method
  - Add `_get_metadata_collector()` helper
  - Add `_calculate_metrics()` helper
- [ ] **3.2** Extend `ExactCoverQuantumSolver`:
  - Add `calculate_metrics()` method
  - Keep existing `decode_counts()` for backward compatibility
  - Add `_get_total_valid_count()` helper (uses problem enumeration)
- [ ] **3.3** Update `ValidationContext` creation:
  - Add helper method in `ExactCoverProblem` for creating context
  - Add helper in `SudokuPuzzle` for creating context
- [ ] **3.4** Write integration tests:
  - Test metrics calculation with simulator backend
  - Test backward compatibility (existing code still works)
- [ ] **3.5** Update examples:
  - Create `examples/example_metrics_basic.py`
  - Show single-run metrics collection

**Deliverables**:
- Working end-to-end metrics collection for simulators
- Backward compatible API
- Example demonstrating basic usage

**Success Criteria**:
- Can run solver with `collect_metrics=True` on Aer simulator
- All existing tests still pass
- New integration tests pass

---

## Phase 4: Provider-Specific Collectors (Week 4-5)

**Goal**: Implement real hardware metadata collection for each provider.

### Subtasks by Provider

#### 4.1 Qiskit/IBM Collector
- [ ] Implement `collect_hardware_metadata()`:
  - Extract T1/T2 from `backend.properties()`
  - Extract gate error rates
  - Extract readout error rates
  - Handle IBMBackend vs FakeBackend gracefully
- [ ] Implement `collect_compilation_metadata()`:
  - Extract initial/final layout from transpiled circuit
  - Compute gate count differences pre/post transpilation
- [ ] Implement `extract_execution_result()`:
  - Parse Qiskit `Result` object
  - Extract job timing information
  - Store job ID
- [ ] Implement `calculate_circuit_volume()`:
  - Use DAGCircuit layer analysis
  - Count active gates per layer
- [ ] Test on real IBM hardware (if available) and simulators
- [ ] Handle edge cases (properties unavailable, fake backends, etc.)

#### 4.2 PyTKET/Quantinuum Collector
- [ ] Research PyTKET backend characterization API
- [ ] Implement `collect_hardware_metadata()`:
  - Extract gate fidelities if available
  - Handle H-series vs simulator differences
- [ ] Implement `collect_compilation_metadata()`:
  - Extract PyTKET pass sequence information
  - Track gate count changes
- [ ] Implement `extract_execution_result()`:
  - Parse pytket `BackendResult` object
- [ ] Implement `calculate_circuit_volume()`:
  - Use pytket circuit commands and depth
- [ ] Test on Quantinuum emulator

#### 4.3 AWS Braket Collector
- [ ] Research Braket device properties API
- [ ] Implement `collect_hardware_metadata()`:
  - Extract provider-specific calibration data
  - Handle IonQ vs Rigetti vs OQC differences
- [ ] Implement compilation metadata extraction
- [ ] Implement result parsing
- [ ] Test on Braket simulators

**Deliverables**:
- Fully functional collectors for 1-2 primary providers
- Graceful degradation for missing metadata
- Tests with mock backend objects

**Success Criteria**:
- Can collect full metadata from IBM Qiskit backends
- Can collect partial metadata from simulators
- No crashes when properties are unavailable

---

## Phase 5: Benchmark Orchestration (Week 6)

**Goal**: Build high-level benchmark suite for multi-run experiments.

### Tasks

- [ ] **5.1** Implement `benchmarking/benchmark_suite.py`:
  - `BenchmarkSuite` class with multi-run logic
  - Seed variation support
  - Progress tracking (optional tqdm integration)
- [ ] **5.2** Implement `benchmarking/classical_baseline.py`:
  - Research classical exact cover libraries
  - Implement timing harness for first solution
  - Implement timing harness for full enumeration
  - Compare solutions for correctness
- [ ] **5.3** Create comprehensive benchmark example:
  - `examples/example_full_benchmark.py`
  - Demonstrates multi-run with variability
  - Includes classical comparison
  - Shows result export
- [ ] **5.4** Add benchmark configuration system:
  - YAML/JSON config for benchmark parameters
  - Support for batch experiments

**Deliverables**:
- Working `BenchmarkSuite` for multi-run experiments
- Classical baseline timing functionality
- Comprehensive example

**Success Criteria**:
- Can run 5-run benchmark with different seeds
- Variability metrics (mean/std/IQR) computed correctly
- Classical baseline provides meaningful comparison

---

## Phase 6: Reporting & Visualization (Week 7)

**Goal**: Export metrics in multiple formats for analysis and publication.

### Tasks

- [ ] **6.1** Implement `reporters/json_reporter.py`:
  - Dataclass serialization (handle datetime, numpy types)
  - Pretty-printed JSON output
  - Schema versioning for future compatibility
- [ ] **6.2** Implement `reporters/table_reporter.py`:
  - Markdown table generation
  - LaTeX table generation (for papers)
  - Console-friendly formatting with `tabulate`
- [ ] **6.3** Implement `reporters/plot_reporter.py`:
  - Bar charts for p_succ across backends
  - Error bars with confidence intervals
  - Heatmaps for precision@k / recall@k
  - Multi-run variability plots
- [ ] **6.4** Create reporting examples:
  - `examples/example_metrics_export.py`
  - `examples/example_metrics_visualization.py`
- [ ] **6.5** Generate benchmark report template:
  - Jupyter notebook for interactive analysis
  - Auto-generated HTML report

**Deliverables**:
- JSON, Markdown, and LaTeX exporters
- matplotlib-based plotting utilities
- Example reports

**Success Criteria**:
- Can export MetricsResult to JSON and reload
- Can generate publication-ready tables
- Can create comparative plots across backends

---

## Phase 7: Documentation & Hardening (Week 8)

**Goal**: Polish documentation, add examples, and prepare for production use.

### Tasks

- [ ] **7.1** Write comprehensive API documentation:
  - Sphinx docstrings for all public APIs
  - Usage guide in `docs/guide/metrics.md`
  - Architecture guide (already created)
- [ ] **7.2** Create tutorial notebooks:
  - `notebooks/metrics_tutorial.ipynb`
  - `notebooks/benchmark_comparison.ipynb`
- [ ] **7.3** Add metrics to existing examples:
  - Update `examples/error_mitigation_comparison.py` to use metrics
  - Update `examples/exact_cover_benchmark.py` to use new system
- [ ] **7.4** Performance optimization:
  - Profile metrics calculation overhead
  - Optimize hot paths (e.g., bitstring validation)
  - Add caching where appropriate
- [ ] **7.5** Error handling hardening:
  - Graceful degradation when metadata unavailable
  - Informative error messages
  - Validation of input data
- [ ] **7.6** Integration testing:
  - End-to-end tests on real backends (CI/CD permitting)
  - Smoke tests for all examples
  - Compatibility tests across provider versions

**Deliverables**:
- Complete documentation
- Tutorial notebooks
- Hardened error handling
- Performance benchmarks

**Success Criteria**:
- All examples run without errors
- Documentation covers 100% of public API
- Performance overhead < 5% of execution time

---

## Dependencies & Blockers

### External Dependencies

1. **scipy** (Phase 1): For Clopper-Pearson CI calculation
   - Action: Add to `pyproject.toml`
   
2. **numpy** (Phase 1): Already in project, used for IQR calculation
   
3. **tabulate** (Phase 6): For table formatting
   - Action: Add to `pyproject.toml` as optional dependency
   
4. **matplotlib** (Phase 6): For plotting
   - Action: Add to `pyproject.toml` as optional dependency

5. **Classical solver library** (Phase 5): TBD
   - Options: python-constraint, pycosat, custom Algorithm X
   - Action: Research and select in Phase 5

### Potential Blockers

1. **Provider API access**: Some metadata may require authenticated access to real hardware
   - Mitigation: Test with simulators and mock objects first
   
2. **Circuit volume calculation**: Complex for some SDKs
   - Mitigation: Start with approximations, refine later
   
3. **Classical solver performance**: May be slow for large instances
   - Mitigation: Add timeout logic, focus on small instances initially

---

## Testing Strategy

### Unit Tests (Each Phase)

- Test each calculator function independently
- Test edge cases (zero, negative, extreme values)
- Test type validation
- Mock external dependencies

### Integration Tests (Phase 3+)

- Test metrics collection with Aer simulator
- Test multi-run aggregation
- Test backward compatibility
- Test with multiple solver types

### System Tests (Phase 7)

- End-to-end benchmarks on simulators
- Smoke tests for all examples
- Performance regression tests

---

## Validation Criteria

### Correctness

- [ ] Clopper-Pearson CIs match scipy.stats reference implementation
- [ ] SNR calculation matches definition in signal processing literature
- [ ] Precision@k / Recall@k match information retrieval definitions
- [ ] Multi-run aggregation produces correct mean/std/IQR

### Performance

- [ ] Metrics calculation overhead < 5% of execution time
- [ ] No memory leaks in multi-run benchmarks
- [ ] Efficient handling of large count dictionaries

### Usability

- [ ] API is intuitive (minimal required parameters)
- [ ] Error messages are informative
- [ ] Documentation includes runnable examples
- [ ] Backward compatibility maintained

---

## Future Enhancements (Post-MVP)

1. **Database integration**: Store metrics in SQLite/PostgreSQL for longitudinal analysis
2. **Web dashboard**: Interactive visualization of benchmarking results
3. **Automated regression detection**: Alert when metrics degrade
4. **Cross-platform comparison**: Automated generation of comparison tables
5. **Uncertainty propagation**: Propagate measurement uncertainty through metrics
6. **Bayesian inference**: Bayesian confidence intervals as alternative to Clopper-Pearson
7. **Cost-aware metrics**: Incorporate cloud pricing into efficiency metrics
8. **Carbon footprint**: Track energy usage and emissions

---

## Risk Mitigation

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Provider API changes | High | Medium | Version pin dependencies, abstract interfaces |
| Missing hardware metadata | Medium | High | Graceful degradation, clear documentation |
| Classical solver too slow | Medium | Medium | Timeout logic, focus on small instances |
| Metrics calculation overhead | Low | Low | Profile early, optimize hot paths |
| User adoption | High | Medium | Excellent documentation, many examples |

---

## Success Metrics for This Implementation

We'll know this implementation is successful when:

1. ✅ Can compute all 7 metric categories from design document
2. ✅ Works with at least 2 quantum providers (IBM, Quantinuum/Aer)
3. ✅ Multi-run benchmarks capture variability correctly
4. ✅ Classical baseline comparison is automated
5. ✅ Can generate publication-ready figures and tables
6. ✅ Existing code continues to work (backward compatible)
7. ✅ Documentation enables new users to run benchmarks
8. ✅ Performance overhead is negligible

---

## Team Roles (if applicable)

- **Architecture**: Complete (this document)
- **Core Implementation** (Phase 1-3): Primary developer
- **Provider Integration** (Phase 4): Could be parallelized across team members
- **Visualization** (Phase 6): Could be separate contributor
- **Documentation** (Phase 7): Technical writer or primary developer

---

## Timeline Summary

| Phase | Duration | Dependencies | Deliverable |
|-------|----------|--------------|-------------|
| 1. Foundation | 1 week | None | Working calculators |
| 2. Interfaces | 1 week | Phase 1 | Abstract collectors |
| 3. Integration | 1 week | Phase 1-2 | Solver integration |
| 4. Collectors | 2 weeks | Phase 3 | Provider metadata |
| 5. Orchestration | 1 week | Phase 4 | Benchmark suite |
| 6. Reporting | 1 week | Phase 5 | Export & visualization |
| 7. Polish | 1 week | Phase 6 | Production-ready |

**Total**: 8 weeks for MVP

**Fast-track option**: Phases 1-3 (3 weeks) provide core functionality with simulators only.
