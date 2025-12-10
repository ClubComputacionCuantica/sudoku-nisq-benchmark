# Metrics System Implementation Checklist

Use this checklist to track implementation progress through all phases.

## Phase 1: Foundation (Week 1)

### Module Structure
- [x] Create `src/sudoku_nisq/metrics/` directory
- [x] Create `calculators/` subdirectory  
- [x] Create `collectors/` subdirectory
- [x] Create `aggregators/` subdirectory
- [x] Create `reporters/` subdirectory
- [x] Create `benchmarking/` subdirectory

### Data Models
- [x] Implement `ExecutionResult` dataclass
- [x] Implement `HardwareMetadata` dataclass
- [x] Implement `CompilationMetadata` dataclass
- [x] Implement `ValidationContext` dataclass
- [x] Implement `MetricsResult` dataclass
- [ ] Add serialization methods (to_dict, from_dict)
- [ ] Add validation for data model fields

### Calculator: Success Metrics
- [x] Create `calculators/success_metrics.py`
- [x] Implement `calculate_p_succ()`
- [x] Implement `calculate_distinct_solutions()`
- [x] Implement `count_valid_shots()` helper
- [ ] Write unit tests for `calculate_p_succ()`
- [ ] Write unit tests for `calculate_distinct_solutions()`
- [ ] Test edge case: zero shots
- [ ] Test edge case: no valid solutions
- [ ] Test edge case: all valid solutions

### Calculator: Ranking Metrics
- [ ] Create `calculators/ranking_metrics.py`
- [ ] Implement `calculate_top_k_valid_mass()`
- [ ] Implement `calculate_precision_at_k()`
- [ ] Implement `calculate_recall_at_k()`
- [ ] Write unit tests for top-k valid mass
- [ ] Write unit tests for precision@k
- [ ] Write unit tests for recall@k
- [ ] Test with k > number of results
- [ ] Test with known precision/recall values

### Calculator: Statistical Metrics
- [ ] Create `calculators/statistical_metrics.py`
- [ ] Implement `clopper_pearson_ci()`
- [ ] Implement `calculate_snr()`
- [ ] Write unit tests for Clopper-Pearson
- [ ] Verify against scipy.stats reference
- [ ] Write unit tests for SNR
- [ ] Test edge case: all valid (SNR = inf)
- [ ] Test edge case: all invalid (SNR = 0)

### Calculator: Efficiency Metrics
- [ ] Create `calculators/efficiency_metrics.py`
- [ ] Implement `calculate_eta_gate()`
- [ ] Implement `calculate_eta_volume()`
- [ ] Implement `calculate_eta_shot()`
- [ ] Write unit tests for all η-metrics
- [ ] Test with zero denominators
- [ ] Test with None circuit_volume

### Calculator: Variability Metrics
- [ ] Create `calculators/variability_metrics.py`
- [ ] Implement `calculate_statistics()`
- [ ] Write unit tests with known mean/std
- [ ] Test IQR calculation
- [ ] Test with single value (std = 0)
- [ ] Test with empty list

### Dependencies & Configuration
- [ ] Add scipy to pyproject.toml dependencies
- [ ] Update project requirements
- [ ] Document version requirements
- [ ] Run dependency vulnerability scan

### Documentation
- [ ] Document all calculator functions with docstrings
- [ ] Add usage examples in docstrings
- [ ] Create API reference for calculators
- [ ] Review and update type hints

---

## Phase 2: Interfaces (Week 2)

### Base Collector
- [ ] Create `collectors/base_collector.py`
- [ ] Define `MetadataCollector` abstract base class
- [ ] Define `collect_hardware_metadata()` abstract method
- [ ] Define `collect_compilation_metadata()` abstract method
- [ ] Define `extract_execution_result()` abstract method
- [ ] Define `calculate_circuit_volume()` abstract method
- [ ] Document collector interface contract

### Collector Stubs
- [ ] Create `collectors/qiskit_collector.py` stub
- [ ] Create `collectors/pytket_collector.py` stub
- [ ] Create `collectors/braket_collector.py` stub
- [ ] Implement minimal __init__ methods
- [ ] Add TODO markers for each method
- [ ] Verify stubs instantiate without errors

### Multi-Run Aggregator
- [ ] Create `aggregators/multi_run_aggregator.py`
- [ ] Implement `MultiRunAggregator` class
- [ ] Implement `aggregate()` method
- [ ] Aggregate p_succ statistics
- [ ] Aggregate top-k metrics (average per k)
- [ ] Aggregate efficiency metrics
- [ ] Write unit tests with mock MetricsResult objects
- [ ] Test with 1, 3, 5, 10 runs
- [ ] Test empty list handling

### Testing & Documentation
- [ ] Write unit tests for aggregator
- [ ] Document aggregation algorithms
- [ ] Create example of multi-run aggregation
- [ ] Review all Phase 2 type hints

---

## Phase 3: Solver Integration (Week 3)

### QuantumSolver Base Class
- [ ] Add `collect_metrics` parameter to `run()`
- [ ] Add `validation_context` parameter to `run()`
- [ ] Implement `_get_metadata_collector()` helper
- [ ] Implement `_calculate_metrics()` helper
- [ ] Implement `_get_circuit_metadata()` helper
- [ ] Handle SDK-specific collector selection
- [ ] Maintain backward compatibility (optional parameters)

### ExactCoverQuantumSolver
- [ ] Implement `calculate_metrics()` method
- [ ] Create ValidationContext from problem
- [ ] Integrate with existing `_is_valid_solution()`
- [ ] Add `_get_total_valid_count()` helper
- [ ] Keep existing `decode_counts()` for compatibility
- [ ] Test both APIs return consistent results

### ValidationContext Helpers
- [ ] Add `create_validation_context()` to ExactCoverProblem
- [ ] Add `create_validation_context()` to SudokuPuzzle
- [ ] Include solution enumeration
- [ ] Include validator callable
- [ ] Document context creation best practices

### Integration Testing
- [ ] Test metrics collection with Aer simulator
- [ ] Test with real IBM backend (if available)
- [ ] Test with different shot counts
- [ ] Test with different optimization levels
- [ ] Test backward compatibility (old API still works)
- [ ] Test error handling (missing validation_context)

### Examples
- [ ] Create `examples/example_metrics_basic.py`
- [ ] Show single-run metrics collection
- [ ] Show accessing different metric categories
- [ ] Show hardware metadata access
- [ ] Add to existing example scripts

### Documentation Updates
- [ ] Update QuantumSolver docstring
- [ ] Update ExactCoverQuantumSolver docstring
- [ ] Update user guide with integration examples
- [ ] Create migration guide from old API

---

## Phase 4: Provider Collectors (Week 4-5)

### Qiskit Collector
- [ ] Research IBM Qiskit backend.properties() API
- [ ] Implement `collect_hardware_metadata()`
  - [ ] Extract T1 times
  - [ ] Extract T2 times
  - [ ] Extract single-qubit gate errors
  - [ ] Extract two-qubit gate errors (CX/ECR)
  - [ ] Extract readout errors
  - [ ] Handle calibration timestamp
  - [ ] Gracefully handle FakeBackend vs real backend
- [ ] Implement `collect_compilation_metadata()`
  - [ ] Extract initial layout
  - [ ] Extract final layout
  - [ ] Compare pre/post gate counts
  - [ ] Extract transpiler seed
  - [ ] Extract optimization level
- [ ] Implement `extract_execution_result()`
  - [ ] Parse Qiskit Result object
  - [ ] Extract counts
  - [ ] Extract job ID
  - [ ] Extract execution time
- [ ] Implement `calculate_circuit_volume()`
  - [ ] Convert to DAGCircuit
  - [ ] Analyze layers
  - [ ] Count active gates per layer
- [ ] Test on Aer simulator
- [ ] Test on FakeBackend
- [ ] Test on real IBM hardware (if available)
- [ ] Handle missing properties gracefully

### PyTKET Collector
- [ ] Research pytket backend characterization API
- [ ] Research pytket-quantinuum backend API
- [ ] Implement `collect_hardware_metadata()`
  - [ ] Investigate available calibration data
  - [ ] Handle H-series vs simulator differences
- [ ] Implement `collect_compilation_metadata()`
  - [ ] Extract pass sequence information
  - [ ] Track gate transformations
- [ ] Implement `extract_execution_result()`
  - [ ] Parse pytket BackendResult
  - [ ] Extract counts and metadata
- [ ] Implement `calculate_circuit_volume()`
  - [ ] Analyze pytket Circuit structure
  - [ ] Compute volume metric
- [ ] Test on pytket Aer backend
- [ ] Test on Quantinuum emulator
- [ ] Document PyTKET-specific considerations

### Braket Collector
- [ ] Research AWS Braket device properties API
- [ ] Implement `collect_hardware_metadata()`
  - [ ] Handle IonQ devices
  - [ ] Handle Rigetti devices  
  - [ ] Handle OQC devices
  - [ ] Extract device-specific calibration
- [ ] Implement `collect_compilation_metadata()`
  - [ ] Extract Braket compilation info
- [ ] Implement `extract_execution_result()`
  - [ ] Parse Braket QuantumTask result
- [ ] Implement `calculate_circuit_volume()`
  - [ ] Analyze Braket circuit
- [ ] Test on Braket local simulator
- [ ] Test on Braket SV1
- [ ] Document multi-device support

### Testing
- [ ] Create mock backend objects for testing
- [ ] Test graceful degradation (missing metadata)
- [ ] Test with all supported providers
- [ ] Integration tests with real backends
- [ ] Performance tests (overhead < 5%)

---

## Phase 5: Benchmark Orchestration (Week 6)

### BenchmarkSuite
- [ ] Create `benchmarking/benchmark_suite.py`
- [ ] Implement `BenchmarkSuite` class
- [ ] Implement `__init__()` with configuration
- [ ] Implement `run_benchmark()` method
- [ ] Add seed variation support
- [ ] Add progress tracking (optional tqdm)
- [ ] Aggregate results across runs
- [ ] Generate benchmark report dict
- [ ] Add timeout handling
- [ ] Add error recovery (continue on failure)

### Classical Baseline
- [ ] Research classical exact cover libraries
- [ ] Select library (python-constraint vs pycosat vs custom)
- [ ] Create `benchmarking/classical_baseline.py`
- [ ] Implement `ClassicalBaseline` class
- [ ] Implement `time_to_first_solution()`
- [ ] Implement `time_to_enumerate_all()`
- [ ] Implement solution correctness verification
- [ ] Add timeout handling
- [ ] Test on small instances (2x2 Sudoku)
- [ ] Test on medium instances (4x4 Sudoku)
- [ ] Document classical solver choice

### Benchmark Configuration
- [ ] Design YAML/JSON config schema
- [ ] Implement config parser
- [ ] Support batch experiments
- [ ] Validate configuration
- [ ] Add config examples

### Examples
- [ ] Create `examples/example_full_benchmark.py`
- [ ] Demonstrate multi-run with variability
- [ ] Include classical comparison
- [ ] Show result aggregation
- [ ] Show report generation

### Testing
- [ ] Test BenchmarkSuite with 3-5 runs
- [ ] Test seed variation
- [ ] Test error handling
- [ ] Test classical baseline timing
- [ ] Integration test full pipeline

---

## Phase 6: Reporting (Week 7)

### JSON Reporter
- [ ] Create `reporters/json_reporter.py`
- [ ] Implement `JSONReporter` class
- [ ] Implement dataclass serialization
- [ ] Handle datetime serialization
- [ ] Handle numpy type serialization
- [ ] Add pretty-printing option
- [ ] Add schema versioning
- [ ] Implement deserialization (from_json)
- [ ] Test round-trip serialization

### Table Reporter
- [ ] Create `reporters/table_reporter.py`
- [ ] Implement `TableReporter` class
- [ ] Implement Markdown table generation
- [ ] Implement LaTeX table generation
- [ ] Implement CSV export
- [ ] Use tabulate library
- [ ] Add custom formatting options
- [ ] Add column selection
- [ ] Test with different table formats

### Plot Reporter
- [ ] Create `reporters/plot_reporter.py`
- [ ] Implement `PlotReporter` class
- [ ] Implement `plot_success_probability()` - bar chart with CI
- [ ] Implement `plot_ranking_metrics()` - heatmap
- [ ] Implement `plot_efficiency_metrics()` - comparison bars
- [ ] Implement `plot_multi_run_variability()` - box plots
- [ ] Add backend comparison plots
- [ ] Add customization options (colors, sizes)
- [ ] Save plots in multiple formats (PNG, PDF, SVG)
- [ ] Test all plot types

### Report Templates
- [ ] Create Jupyter notebook template
- [ ] Create HTML report template
- [ ] Add auto-generated summary section
- [ ] Add figures and tables
- [ ] Test template rendering

### Examples
- [ ] Create `examples/example_metrics_export.py`
- [ ] Create `examples/example_metrics_visualization.py`
- [ ] Show JSON export/import
- [ ] Show table generation
- [ ] Show plot creation

### Dependencies
- [ ] Add tabulate to pyproject.toml (optional)
- [ ] Add matplotlib to pyproject.toml (optional)
- [ ] Document visualization dependencies

---

## Phase 7: Polish (Week 8)

### Documentation
- [ ] Write complete API reference (Sphinx)
- [ ] Create `docs/guide/metrics.md` tutorial
- [ ] Update README with metrics examples
- [ ] Create architecture diagrams
- [ ] Document all design decisions
- [ ] Add FAQ section

### Tutorial Notebooks
- [ ] Create `notebooks/metrics_tutorial.ipynb`
- [ ] Create `notebooks/benchmark_comparison.ipynb`
- [ ] Add step-by-step explanations
- [ ] Include visualization examples
- [ ] Test all notebook cells

### Update Existing Examples
- [ ] Update `examples/error_mitigation_comparison.py`
- [ ] Update `examples/exact_cover_benchmark.py`
- [ ] Ensure all examples use new metrics system
- [ ] Add metrics collection to existing workflows

### Performance Optimization
- [ ] Profile metrics calculation overhead
- [ ] Optimize hot paths
- [ ] Add caching for expensive operations
- [ ] Benchmark performance (target < 5% overhead)
- [ ] Document performance characteristics

### Error Handling
- [ ] Review all error paths
- [ ] Add informative error messages
- [ ] Handle edge cases gracefully
- [ ] Add input validation
- [ ] Test error recovery

### Code Quality
- [ ] Run linter (ruff/pylint)
- [ ] Fix all linting issues
- [ ] Ensure 100% type hint coverage
- [ ] Run mypy type checking
- [ ] Fix all type errors

### Testing
- [ ] Achieve 100% test coverage for calculators
- [ ] Add integration tests for full pipeline
- [ ] Add smoke tests for all examples
- [ ] Test on multiple Python versions
- [ ] Test with different provider versions
- [ ] Set up CI/CD for metrics tests

### Release Preparation
- [ ] Update version numbers
- [ ] Write CHANGELOG
- [ ] Update installation instructions
- [ ] Create release checklist
- [ ] Tag release in git

---

## Future Enhancements (Post-MVP)

### Database Integration
- [ ] Design metrics database schema
- [ ] Implement SQLite storage
- [ ] Add PostgreSQL support
- [ ] Create query interface
- [ ] Add longitudinal analysis tools

### Web Dashboard
- [ ] Design dashboard UI
- [ ] Implement backend API
- [ ] Create interactive visualizations
- [ ] Add real-time updates
- [ ] Deploy dashboard

### Advanced Features
- [ ] Automated regression detection
- [ ] Cross-platform comparison automation
- [ ] Uncertainty propagation
- [ ] Bayesian confidence intervals
- [ ] Cost-aware metrics
- [ ] Carbon footprint tracking

---

## Sign-off Checklist

Before considering metrics system production-ready:

- [ ] All Phase 1-7 tasks complete
- [ ] All tests passing (unit + integration)
- [ ] Code coverage > 90%
- [ ] Documentation complete and reviewed
- [ ] All examples working
- [ ] Performance benchmarks meet targets
- [ ] Security review completed
- [ ] User acceptance testing
- [ ] Release notes written
- [ ] Project stakeholders approve

---

## Notes

Use this format to track progress:
- `[ ]` = Not started
- `[~]` = In progress  
- `[x]` = Complete
- `[!]` = Blocked (add note)

Add dates to track velocity:
- Started: YYYY-MM-DD
- Completed: YYYY-MM-DD
