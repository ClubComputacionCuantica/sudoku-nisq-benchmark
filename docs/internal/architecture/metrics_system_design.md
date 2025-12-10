# Benchmarking Metrics System Architecture

## Overview

This document outlines the modular architecture for implementing comprehensive benchmarking metrics for the sudoku-nisq quantum solver framework. The design acknowledges provider/backend/SDK dependencies and provides clear integration points with the existing codebase.

## Core Design Principles

1. **Modularity**: Each metric category is independently calculable
2. **Provider Agnosticism**: Abstract interfaces for provider-specific data
3. **Composability**: Metrics can be combined into custom benchmarking pipelines
4. **Extensibility**: Easy to add new metrics without modifying core infrastructure
5. **Reproducibility**: All metrics include provenance tracking

---

## Module Structure

```
src/sudoku_nisq/metrics/
├── __init__.py
├── base.py                      # Abstract base classes & interfaces
├── data_models.py               # Data structures for results & metadata
├── collectors/                  # Provider-specific data collectors
│   ├── __init__.py
│   ├── base_collector.py        # Abstract collector interface
│   ├── qiskit_collector.py      # IBM/Qiskit-specific data
│   ├── pytket_collector.py      # Quantinuum/PyTKET-specific data
│   └── braket_collector.py      # AWS Braket-specific data
├── calculators/                 # Metric calculation implementations
│   ├── __init__.py
│   ├── success_metrics.py       # p_succ, coverage, distinct solutions
│   ├── ranking_metrics.py       # Top-k, precision@k, recall@k
│   ├── statistical_metrics.py   # Clopper-Pearson CIs, SNR
│   ├── efficiency_metrics.py    # Resource-normalized metrics
│   └── variability_metrics.py   # Inter-run variance, IQR
├── aggregators/                 # Multi-run & multi-experiment aggregation
│   ├── __init__.py
│   └── multi_run_aggregator.py
├── reporters/                   # Output formatting & visualization
│   ├── __init__.py
│   ├── json_reporter.py
│   ├── table_reporter.py
│   └── plot_reporter.py
└── benchmarking/                # Full benchmark orchestration
    ├── __init__.py
    ├── benchmark_suite.py       # Main benchmark runner
    └── classical_baseline.py    # Classical solver comparisons
```

---

## 1. Data Models (`data_models.py`)

### Core Data Structures

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime

@dataclass
class ExecutionResult:
    """Results from a single quantum execution."""
    counts: Dict[str, int]                    # Measurement counts
    shots: int                                # Total shots
    execution_time: float                     # Wall-clock time (seconds)
    timestamp: datetime                       # Execution timestamp
    backend_name: str                         # Backend identifier
    
    # Circuit metadata
    num_qubits: int
    circuit_depth: int
    gate_counts: Dict[str, int]              # Gate type -> count
    two_qubit_gates: int                     # Total 2q gates
    circuit_volume: Optional[int] = None     # Circuit volume (sum of active gates per layer)
    
    # Provider-specific data (optional)
    raw_result: Any = None                   # Original provider result object
    job_id: Optional[str] = None
    
@dataclass
class HardwareMetadata:
    """Hardware calibration and error rates."""
    backend_name: str
    provider: str                            # 'ibm', 'quantinuum', 'aws', etc.
    calibration_timestamp: Optional[datetime] = None
    
    # Error rates (provider-dependent)
    single_qubit_gate_error: Optional[Dict[int, float]] = None  # qubit -> error rate
    two_qubit_gate_error: Optional[Dict[tuple, float]] = None   # (q1, q2) -> error rate
    readout_error: Optional[Dict[int, float]] = None            # qubit -> error rate
    
    # Coherence times (provider-dependent)
    t1_times: Optional[Dict[int, float]] = None  # qubit -> T1 (μs)
    t2_times: Optional[Dict[int, float]] = None  # qubit -> T2 (μs)
    
    # Additional metadata
    extra_properties: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class CompilationMetadata:
    """Circuit compilation/transpilation metadata."""
    transpiler_seed: Optional[int] = None
    optimization_level: int = 0
    initial_layout: Optional[List[int]] = None
    final_layout: Optional[List[int]] = None
    
    # Resource changes from transpilation
    pre_transpile_gates: Optional[Dict[str, int]] = None
    post_transpile_gates: Optional[Dict[str, int]] = None
    pre_transpile_depth: Optional[int] = None
    post_transpile_depth: Optional[int] = None
    
    # Circuit hash for reproducibility
    circuit_hash: Optional[str] = None
    
@dataclass
class ValidationContext:
    """Context for validating quantum results."""
    valid_solutions: List[str]               # Known valid bitstrings
    total_valid_count: int                   # Total number of valid solutions
    solution_validator: Any                  # Callable: bitstring -> bool
    
@dataclass
class MetricsResult:
    """Aggregated metrics from benchmark run(s)."""
    # Success metrics
    p_succ: float                            # Success probability
    p_succ_ci_lower: float                   # Lower 95% CI
    p_succ_ci_upper: float                   # Upper 95% CI
    distinct_valid_solutions: int            # Coverage
    
    # Ranking metrics
    top_k_valid_mass: Dict[int, float]       # k -> probability mass
    precision_at_k: Dict[int, float]         # k -> precision
    recall_at_k: Dict[int, float]            # k -> recall
    
    # Statistical metrics
    snr: float                               # Signal-to-noise ratio
    
    # Efficiency metrics
    eta_gate: float                          # p_succ / G_2q
    eta_volume: Optional[float] = None       # p_succ / V
    eta_shot: float = None                   # p_succ / shots
    
    # Variability (multi-run)
    p_succ_mean: Optional[float] = None
    p_succ_std: Optional[float] = None
    p_succ_iqr: Optional[tuple] = None       # (Q1, Q3)
    
    # Metadata references
    execution_results: List[ExecutionResult] = field(default_factory=list)
    hardware_metadata: Optional[HardwareMetadata] = None
    compilation_metadata: Optional[CompilationMetadata] = None
```

---

## 2. Abstract Collector Interface (`collectors/base_collector.py`)

```python
from abc import ABC, abstractmethod
from typing import Optional
from ..data_models import HardwareMetadata, CompilationMetadata, ExecutionResult

class MetadataCollector(ABC):
    """Abstract interface for collecting provider-specific metadata."""
    
    @abstractmethod
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect calibration data and error rates from backend.
        
        Args:
            backend: Provider-specific backend object
            
        Returns:
            HardwareMetadata with available information
            
        Note:
            Not all fields may be available for all providers.
            Implementations should populate what's accessible and 
            leave the rest as None.
        """
        pass
    
    @abstractmethod
    def collect_compilation_metadata(
        self, 
        original_circuit: Any,
        transpiled_circuit: Any,
        transpile_args: dict
    ) -> CompilationMetadata:
        """Collect transpilation metadata.
        
        Args:
            original_circuit: Pre-transpilation circuit
            transpiled_circuit: Post-transpilation circuit
            transpile_args: Arguments used for transpilation
            
        Returns:
            CompilationMetadata with compilation details
        """
        pass
    
    @abstractmethod
    def extract_execution_result(
        self,
        raw_result: Any,
        circuit_metadata: dict
    ) -> ExecutionResult:
        """Convert provider-specific result to standard ExecutionResult.
        
        Args:
            raw_result: Provider's native result object
            circuit_metadata: Pre-computed circuit properties
            
        Returns:
            ExecutionResult with standardized data
        """
        pass
    
    @abstractmethod
    def calculate_circuit_volume(self, circuit: Any) -> int:
        """Calculate circuit volume (sum of gates per layer).
        
        TODO: Volume calculation is architecture-specific and may
        require layer-by-layer analysis of circuit DAG.
        
        Args:
            circuit: Circuit object in provider's format
            
        Returns:
            Circuit volume
        """
        pass
```

---

## 3. Metric Calculators

### 3.1 Success Metrics (`calculators/success_metrics.py`)

```python
from typing import Dict, List, Tuple, Callable
from ..data_models import ExecutionResult, ValidationContext

class SuccessMetricsCalculator:
    """Calculate basic success probability and coverage metrics."""
    
    @staticmethod
    def calculate_p_succ(
        counts: Dict[str, int],
        validator: Callable[[str], bool]
    ) -> float:
        """Calculate success probability.
        
        Args:
            counts: Measurement counts dict
            validator: Function that returns True if bitstring is valid
            
        Returns:
            Fraction of shots that produced valid solutions
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        valid_shots = sum(
            count for bitstring, count in counts.items()
            if validator(bitstring)
        )
        
        return valid_shots / total_shots
    
    @staticmethod
    def calculate_distinct_solutions(
        counts: Dict[str, int],
        validator: Callable[[str], bool]
    ) -> int:
        """Count number of distinct valid solutions observed.
        
        Args:
            counts: Measurement counts dict
            validator: Function that returns True if bitstring is valid
            
        Returns:
            Number of unique valid bitstrings with count > 0
        """
        return sum(
            1 for bitstring, count in counts.items()
            if count > 0 and validator(bitstring)
        )
```

### 3.2 Ranking Metrics (`calculators/ranking_metrics.py`)

```python
from typing import Dict, List, Tuple, Callable

class RankingMetricsCalculator:
    """Calculate ranking-based metrics (Top-k, Precision@k, Recall@k)."""
    
    @staticmethod
    def calculate_top_k_valid_mass(
        counts: Dict[str, int],
        validator: Callable[[str], bool],
        k_values: List[int]
    ) -> Dict[int, float]:
        """Calculate probability mass of top-k valid solutions.
        
        Args:
            counts: Measurement counts dict
            validator: Function that returns True if bitstring is valid
            k_values: List of k values to compute (e.g., [1, 3, 5, 10])
            
        Returns:
            Dict mapping k -> cumulative probability of top-k valid solutions
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return {k: 0.0 for k in k_values}
        
        # Get valid solutions sorted by probability
        valid_probs = sorted(
            [count / total_shots for bitstring, count in counts.items()
             if validator(bitstring)],
            reverse=True
        )
        
        results = {}
        for k in k_values:
            results[k] = sum(valid_probs[:k])
        
        return results
    
    @staticmethod
    def calculate_precision_at_k(
        counts: Dict[str, int],
        validator: Callable[[str], bool],
        k_values: List[int]
    ) -> Dict[int, float]:
        """Calculate Precision@k.
        
        Among top-k most probable bitstrings, what fraction are valid?
        
        Args:
            counts: Measurement counts dict
            validator: Function that returns True if bitstring is valid
            k_values: List of k values to compute
            
        Returns:
            Dict mapping k -> precision@k
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return {k: 0.0 for k in k_values}
        
        # Sort all bitstrings by probability
        sorted_bitstrings = sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = {}
        for k in k_values:
            top_k = sorted_bitstrings[:k]
            valid_in_top_k = sum(
                1 for bitstring, _ in top_k
                if validator(bitstring)
            )
            results[k] = valid_in_top_k / k if k > 0 else 0.0
        
        return results
    
    @staticmethod
    def calculate_recall_at_k(
        counts: Dict[str, int],
        validator: Callable[[str], bool],
        total_valid_count: int,
        k_values: List[int]
    ) -> Dict[int, float]:
        """Calculate Recall@k.
        
        Among all valid solutions, what fraction appear in top-k?
        
        Args:
            counts: Measurement counts dict
            validator: Function that returns True if bitstring is valid
            total_valid_count: Total number of valid solutions (from enumeration)
            k_values: List of k values to compute
            
        Returns:
            Dict mapping k -> recall@k
        """
        if total_valid_count == 0:
            return {k: 0.0 for k in k_values}
        
        # Sort all bitstrings by probability
        sorted_bitstrings = sorted(
            counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        results = {}
        for k in k_values:
            top_k = sorted_bitstrings[:k]
            valid_in_top_k = sum(
                1 for bitstring, _ in top_k
                if validator(bitstring)
            )
            results[k] = valid_in_top_k / total_valid_count
        
        return results
```

### 3.3 Statistical Metrics (`calculators/statistical_metrics.py`)

```python
from typing import Tuple
from scipy import stats  # For Clopper-Pearson

class StatisticalMetricsCalculator:
    """Calculate confidence intervals and statistical robustness metrics."""
    
    @staticmethod
    def clopper_pearson_ci(
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate exact Clopper-Pearson confidence interval.
        
        Args:
            successes: Number of successful outcomes
            trials: Total number of trials
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if trials == 0:
            return (0.0, 0.0)
        
        alpha = 1 - confidence
        
        # Use scipy's beta distribution for exact calculation
        lower = stats.beta.ppf(alpha / 2, successes, trials - successes + 1)
        upper = stats.beta.ppf(1 - alpha / 2, successes + 1, trials - successes)
        
        # Handle edge cases
        lower = 0.0 if successes == 0 else lower
        upper = 1.0 if successes == trials else upper
        
        return (lower, upper)
    
    @staticmethod
    def calculate_snr(
        counts: Dict[str, int],
        validator: Callable[[str], bool]
    ) -> float:
        """Calculate signal-to-noise ratio.
        
        SNR = (total valid probability mass) / (total invalid probability mass)
        
        Args:
            counts: Measurement counts dict
            validator: Function that returns True if bitstring is valid
            
        Returns:
            SNR value (inf if no invalid counts, 0 if no valid counts)
        """
        total_shots = sum(counts.values())
        if total_shots == 0:
            return 0.0
        
        valid_mass = sum(
            count for bitstring, count in counts.items()
            if validator(bitstring)
        ) / total_shots
        
        invalid_mass = 1.0 - valid_mass
        
        if invalid_mass == 0:
            return float('inf') if valid_mass > 0 else 0.0
        
        return valid_mass / invalid_mass
```

### 3.4 Efficiency Metrics (`calculators/efficiency_metrics.py`)

```python
from typing import Optional
from ..data_models import ExecutionResult

class EfficiencyMetricsCalculator:
    """Calculate resource-normalized efficiency metrics."""
    
    @staticmethod
    def calculate_eta_gate(p_succ: float, two_qubit_gates: int) -> float:
        """Gate-normalized success: p_succ / G_2q.
        
        Args:
            p_succ: Success probability
            two_qubit_gates: Number of two-qubit gates
            
        Returns:
            Efficiency per two-qubit gate
        """
        if two_qubit_gates == 0:
            return 0.0
        return p_succ / two_qubit_gates
    
    @staticmethod
    def calculate_eta_volume(
        p_succ: float,
        circuit_volume: Optional[int]
    ) -> Optional[float]:
        """Volume-normalized success: p_succ / V.
        
        Args:
            p_succ: Success probability
            circuit_volume: Circuit volume (may be None if not calculable)
            
        Returns:
            Efficiency per unit volume, or None if volume unavailable
        """
        if circuit_volume is None or circuit_volume == 0:
            return None
        return p_succ / circuit_volume
    
    @staticmethod
    def calculate_eta_shot(p_succ: float, shots: int) -> float:
        """Shot-normalized success: p_succ / shots.
        
        Args:
            p_succ: Success probability
            shots: Number of measurement shots
            
        Returns:
            Efficiency per shot
        """
        if shots == 0:
            return 0.0
        return p_succ / shots
```

### 3.5 Variability Metrics (`calculators/variability_metrics.py`)

```python
import numpy as np
from typing import List, Tuple

class VariabilityMetricsCalculator:
    """Calculate inter-run variability metrics."""
    
    @staticmethod
    def calculate_statistics(values: List[float]) -> dict:
        """Calculate mean, std, and IQR for a list of values.
        
        Args:
            values: List of metric values from multiple runs
            
        Returns:
            Dict with 'mean', 'std', 'iqr_q1', 'iqr_q3', 'iqr_range'
        """
        if not values:
            return {
                'mean': 0.0,
                'std': 0.0,
                'iqr_q1': 0.0,
                'iqr_q3': 0.0,
                'iqr_range': 0.0
            }
        
        arr = np.array(values)
        q1, q3 = np.percentile(arr, [25, 75])
        
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            'iqr_q1': float(q1),
            'iqr_q3': float(q3),
            'iqr_range': float(q3 - q1)
        }
```

---

## 4. Integration with Existing Code

### 4.1 Extending `QuantumSolver.run()` Method

The existing `run()` method in `quantum_solver.py` should be enhanced to optionally collect metrics:

```python
# In quantum_solver.py

def run(
    self,
    backend: Any,
    shots: int = 1024,
    opt_level: int = 1,
    collect_metrics: bool = False,
    validation_context: Optional[ValidationContext] = None,
    **kwargs
) -> Union[Any, Tuple[Any, MetricsResult]]:
    """
    Execute circuit on backend with optional metrics collection.
    
    Args:
        backend: Quantum backend
        shots: Number of shots
        opt_level: Optimization level
        collect_metrics: If True, calculate and return metrics
        validation_context: Required if collect_metrics=True
        **kwargs: Additional backend-specific arguments
        
    Returns:
        If collect_metrics=False: raw result object
        If collect_metrics=True: (result, MetricsResult)
    """
    # ... existing execution logic ...
    
    if collect_metrics:
        if validation_context is None:
            raise ValueError("validation_context required when collect_metrics=True")
        
        # Collect provider-specific metadata
        collector = self._get_metadata_collector(backend)
        hw_metadata = collector.collect_hardware_metadata(backend)
        comp_metadata = collector.collect_compilation_metadata(
            self.main_circuit,
            transpiled_circuit,
            {'opt_level': opt_level}
        )
        exec_result = collector.extract_execution_result(
            result,
            self._get_circuit_metadata()
        )
        
        # Calculate metrics
        metrics = self._calculate_metrics(
            exec_result,
            validation_context,
            hw_metadata,
            comp_metadata
        )
        
        return result, metrics
    
    return result
```

### 4.2 Integration with `ExactCoverQuantumSolver`

The `decode_counts()` method already computes some metrics. We should:

1. Keep existing `decode_counts()` for backward compatibility
2. Add new method that returns structured `MetricsResult`:

```python
# In solvers/exact_cover_solver.py

def calculate_metrics(
    self,
    counts: Dict[str, int],
    execution_result: ExecutionResult,
    hardware_metadata: Optional[HardwareMetadata] = None,
    compilation_metadata: Optional[CompilationMetadata] = None,
    k_values: List[int] = [1, 3, 5, 10]
) -> MetricsResult:
    """Calculate comprehensive metrics using new metrics system.
    
    This wraps the calculators and produces a structured MetricsResult.
    """
    from sudoku_nisq.metrics.calculators import (
        SuccessMetricsCalculator,
        RankingMetricsCalculator,
        StatisticalMetricsCalculator,
        EfficiencyMetricsCalculator
    )
    
    # Create validator
    validator = lambda bs: self._is_valid_solution(bs)
    
    # Calculate base metrics
    p_succ = SuccessMetricsCalculator.calculate_p_succ(counts, validator)
    distinct_valid = SuccessMetricsCalculator.calculate_distinct_solutions(counts, validator)
    
    # Statistical metrics
    total_shots = sum(counts.values())
    valid_shots = int(p_succ * total_shots)
    ci_lower, ci_upper = StatisticalMetricsCalculator.clopper_pearson_ci(
        valid_shots, total_shots
    )
    snr = StatisticalMetricsCalculator.calculate_snr(counts, validator)
    
    # Ranking metrics
    top_k_mass = RankingMetricsCalculator.calculate_top_k_valid_mass(
        counts, validator, k_values
    )
    precision_at_k = RankingMetricsCalculator.calculate_precision_at_k(
        counts, validator, k_values
    )
    recall_at_k = RankingMetricsCalculator.calculate_recall_at_k(
        counts, validator, self._get_total_valid_count(), k_values
    )
    
    # Efficiency metrics
    eta_gate = EfficiencyMetricsCalculator.calculate_eta_gate(
        p_succ, execution_result.two_qubit_gates
    )
    eta_volume = EfficiencyMetricsCalculator.calculate_eta_volume(
        p_succ, execution_result.circuit_volume
    )
    eta_shot = EfficiencyMetricsCalculator.calculate_eta_shot(
        p_succ, execution_result.shots
    )
    
    return MetricsResult(
        p_succ=p_succ,
        p_succ_ci_lower=ci_lower,
        p_succ_ci_upper=ci_upper,
        distinct_valid_solutions=distinct_valid,
        top_k_valid_mass=top_k_mass,
        precision_at_k=precision_at_k,
        recall_at_k=recall_at_k,
        snr=snr,
        eta_gate=eta_gate,
        eta_volume=eta_volume,
        eta_shot=eta_shot,
        execution_results=[execution_result],
        hardware_metadata=hardware_metadata,
        compilation_metadata=compilation_metadata
    )
```

---

## 5. Benchmark Suite Orchestration

### 5.1 Main Benchmark Runner (`benchmarking/benchmark_suite.py`)

```python
from typing import List, Dict, Any, Optional
from ..data_models import MetricsResult, ValidationContext
from ..aggregators.multi_run_aggregator import MultiRunAggregator

class BenchmarkSuite:
    """Orchestrates multi-run benchmarking experiments with variability tracking."""
    
    def __init__(
        self,
        solver: Any,
        validation_context: ValidationContext,
        n_runs: int = 3,
        different_seeds: bool = True,
        k_values: List[int] = [1, 3, 5, 10]
    ):
        """
        Args:
            solver: QuantumSolver instance
            validation_context: Context for validating solutions
            n_runs: Number of independent runs (≥3 recommended)
            different_seeds: Use different transpiler seeds per run
            k_values: Values of k for ranking metrics
        """
        self.solver = solver
        self.validation_context = validation_context
        self.n_runs = n_runs
        self.different_seeds = different_seeds
        self.k_values = k_values
        self.aggregator = MultiRunAggregator()
    
    def run_benchmark(
        self,
        backend: Any,
        shots: int = 1024,
        opt_level: int = 1,
        **kwargs
    ) -> Dict[str, Any]:
        """Execute full benchmark with multiple runs and aggregation.
        
        Returns:
            Dict containing:
            - 'individual_runs': List[MetricsResult]
            - 'aggregated': MetricsResult with mean/std/IQR
            - 'metadata': Benchmark configuration
        """
        individual_results = []
        
        for run_idx in range(self.n_runs):
            # Vary transpiler seed if requested
            seed = run_idx if self.different_seeds else None
            
            # Execute with metrics collection
            result, metrics = self.solver.run(
                backend=backend,
                shots=shots,
                opt_level=opt_level,
                collect_metrics=True,
                validation_context=self.validation_context,
                seed=seed,
                **kwargs
            )
            
            individual_results.append(metrics)
        
        # Aggregate across runs
        aggregated = self.aggregator.aggregate(individual_results)
        
        return {
            'individual_runs': individual_results,
            'aggregated': aggregated,
            'metadata': {
                'n_runs': self.n_runs,
                'shots_per_run': shots,
                'opt_level': opt_level,
                'different_seeds': self.different_seeds,
                'k_values': self.k_values
            }
        }
```

### 5.2 Classical Baseline (`benchmarking/classical_baseline.py`)

```python
import time
from typing import Optional, List

class ClassicalBaseline:
    """Benchmark classical exact cover solvers for comparison."""
    
    def __init__(self, exact_cover_problem):
        self.problem = exact_cover_problem
    
    def time_to_first_solution(self, algorithm: str = "dlx") -> dict:
        """Time classical solver to find first solution.
        
        Args:
            algorithm: Classical algorithm ('dlx', 'backtrack', etc.)
            
        Returns:
            Dict with 'time_seconds', 'solution', 'algorithm'
        """
        # TODO: Implement based on chosen classical solver library
        # Options: python-constraint, pycosat, Algorithm X implementations
        start = time.perf_counter()
        # solution = classical_solve_first(self.problem, algorithm)
        elapsed = time.perf_counter() - start
        
        return {
            'time_seconds': elapsed,
            'solution': None,  # TODO
            'algorithm': algorithm
        }
    
    def time_to_enumerate_all(self, algorithm: str = "dlx") -> dict:
        """Time classical solver to enumerate all solutions.
        
        Args:
            algorithm: Classical algorithm
            
        Returns:
            Dict with 'time_seconds', 'solution_count', 'algorithm'
        """
        # TODO: Implement full enumeration
        start = time.perf_counter()
        # solutions = classical_solve_all(self.problem, algorithm)
        elapsed = time.perf_counter() - start
        
        return {
            'time_seconds': elapsed,
            'solution_count': 0,  # TODO
            'algorithm': algorithm
        }
```

---

## 6. Provider-Specific Collector Implementations

### 6.1 Qiskit Collector (`collectors/qiskit_collector.py`)

```python
from typing import Any, Dict, Optional
from datetime import datetime
from .base_collector import MetadataCollector
from ..data_models import HardwareMetadata, CompilationMetadata, ExecutionResult

class QiskitMetadataCollector(MetadataCollector):
    """Collector for IBM Qiskit backends."""
    
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect from Qiskit backend properties."""
        try:
            properties = backend.properties()
            timestamp = properties.last_update_date if properties else None
            
            # Extract error rates
            single_qubit_error = {}
            two_qubit_error = {}
            readout_error = {}
            t1_times = {}
            t2_times = {}
            
            if properties:
                for qubit_idx in range(backend.configuration().n_qubits):
                    # Single-qubit gate error (use sx gate as reference)
                    # TODO: Average over all 1q gates or select specific gate
                    t1_times[qubit_idx] = properties.t1(qubit_idx)
                    t2_times[qubit_idx] = properties.t2(qubit_idx)
                    readout_error[qubit_idx] = properties.readout_error(qubit_idx)
                
                # Two-qubit gates (typically CX/ECR)
                for gate in properties.gates:
                    if gate.gate in ['cx', 'ecr'] and len(gate.qubits) == 2:
                        two_qubit_error[tuple(gate.qubits)] = gate.parameters[0].value
            
            return HardwareMetadata(
                backend_name=backend.name(),
                provider='ibm',
                calibration_timestamp=timestamp,
                single_qubit_gate_error=single_qubit_error,
                two_qubit_gate_error=two_qubit_error,
                readout_error=readout_error,
                t1_times=t1_times,
                t2_times=t2_times
            )
        except Exception as e:
            # Gracefully degrade if properties unavailable (e.g., simulator)
            return HardwareMetadata(
                backend_name=str(backend),
                provider='ibm',
                extra_properties={'error': str(e)}
            )
    
    def collect_compilation_metadata(
        self,
        original_circuit: Any,
        transpiled_circuit: Any,
        transpile_args: dict
    ) -> CompilationMetadata:
        """Collect Qiskit transpilation metadata."""
        from qiskit.converters import circuit_to_dag
        
        pre_depth = original_circuit.depth() if original_circuit else None
        post_depth = transpiled_circuit.depth() if transpiled_circuit else None
        
        pre_gates = dict(original_circuit.count_ops()) if original_circuit else None
        post_gates = dict(transpiled_circuit.count_ops()) if transpiled_circuit else None
        
        # TODO: Extract initial/final layout from transpiled circuit metadata
        # This requires accessing transpiled_circuit._layout or similar
        
        return CompilationMetadata(
            transpiler_seed=transpile_args.get('seed_transpiler'),
            optimization_level=transpile_args.get('optimization_level', 0),
            pre_transpile_gates=pre_gates,
            post_transpile_gates=post_gates,
            pre_transpile_depth=pre_depth,
            post_transpile_depth=post_depth
        )
    
    def extract_execution_result(
        self,
        raw_result: Any,
        circuit_metadata: dict
    ) -> ExecutionResult:
        """Extract from Qiskit Result object."""
        counts = raw_result.get_counts()
        
        # TODO: Extract execution time (may require job metadata)
        execution_time = 0.0  # TODO: get from job.result().time_taken or similar
        
        return ExecutionResult(
            counts=counts,
            shots=sum(counts.values()),
            execution_time=execution_time,
            timestamp=datetime.now(),
            backend_name=circuit_metadata.get('backend_name', 'unknown'),
            num_qubits=circuit_metadata.get('num_qubits', 0),
            circuit_depth=circuit_metadata.get('depth', 0),
            gate_counts=circuit_metadata.get('gate_counts', {}),
            two_qubit_gates=circuit_metadata.get('two_qubit_gates', 0),
            circuit_volume=circuit_metadata.get('circuit_volume'),
            raw_result=raw_result,
            job_id=getattr(raw_result, 'job_id', None)
        )
    
    def calculate_circuit_volume(self, circuit: Any) -> int:
        """Calculate volume for Qiskit circuit.
        
        TODO: Requires DAG-based layer analysis.
        Circuit volume = sum over layers of (# active gates in layer).
        """
        # Placeholder: Return depth * num_qubits as approximation
        return circuit.depth() * circuit.num_qubits
```

### 6.2 PyTKET Collector Stub (`collectors/pytket_collector.py`)

```python
class PyTKETMetadataCollector(MetadataCollector):
    """Collector for Quantinuum/PyTKET backends."""
    
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect from PyTKET backend characterization."""
        # TODO: Access backend.backend_info or similar
        return HardwareMetadata(
            backend_name=str(backend),
            provider='quantinuum',
            extra_properties={'status': 'TODO'}
        )
    
    # ... other methods with TODO markers
```

### 6.3 AWS Braket Collector Stub (`collectors/braket_collector.py`)

```python
class BraketMetadataCollector(MetadataCollector):
    """Collector for AWS Braket backends."""
    
    def collect_hardware_metadata(self, backend: Any) -> HardwareMetadata:
        """Collect from Braket device properties."""
        # TODO: Access device.properties
        return HardwareMetadata(
            backend_name=str(backend),
            provider='aws',
            extra_properties={'status': 'TODO'}
        )
    
    # ... other methods with TODO markers
```

---

## 7. Multi-Run Aggregator (`aggregators/multi_run_aggregator.py`)

```python
from typing import List
from ..data_models import MetricsResult
from ..calculators.variability_metrics import VariabilityMetricsCalculator

class MultiRunAggregator:
    """Aggregate metrics across multiple independent runs."""
    
    def aggregate(self, results: List[MetricsResult]) -> MetricsResult:
        """Compute mean, std, IQR across runs.
        
        Args:
            results: List of MetricsResult from independent runs
            
        Returns:
            Aggregated MetricsResult with variability metrics
        """
        if not results:
            raise ValueError("Cannot aggregate empty results list")
        
        # Extract p_succ values
        p_succ_values = [r.p_succ for r in results]
        stats = VariabilityMetricsCalculator.calculate_statistics(p_succ_values)
        
        # Use first run's structure as template, add variability
        template = results[0]
        
        return MetricsResult(
            p_succ=stats['mean'],
            p_succ_ci_lower=min(r.p_succ_ci_lower for r in results),
            p_succ_ci_upper=max(r.p_succ_ci_upper for r in results),
            distinct_valid_solutions=int(sum(r.distinct_valid_solutions for r in results) / len(results)),
            top_k_valid_mass=template.top_k_valid_mass,  # TODO: Average per k
            precision_at_k=template.precision_at_k,      # TODO: Average per k
            recall_at_k=template.recall_at_k,            # TODO: Average per k
            snr=sum(r.snr for r in results) / len(results),
            eta_gate=stats['mean'] / template.execution_results[0].two_qubit_gates if template.execution_results else 0,
            eta_volume=None,  # TODO
            eta_shot=stats['mean'] / template.execution_results[0].shots if template.execution_results else 0,
            p_succ_mean=stats['mean'],
            p_succ_std=stats['std'],
            p_succ_iqr=(stats['iqr_q1'], stats['iqr_q3']),
            execution_results=[er for r in results for er in r.execution_results],
            hardware_metadata=results[0].hardware_metadata,
            compilation_metadata=results[0].compilation_metadata
        )
```

---

## 8. Reporter Interfaces (`reporters/`)

### 8.1 JSON Reporter

```python
import json
from pathlib import Path
from ..data_models import MetricsResult

class JSONReporter:
    """Export metrics to JSON format."""
    
    @staticmethod
    def export(metrics: MetricsResult, output_path: Path):
        """Save metrics to JSON file."""
        # TODO: Implement dataclass serialization
        pass
```

### 8.2 Table Reporter

```python
from tabulate import tabulate

class TableReporter:
    """Format metrics as tables for console/markdown."""
    
    @staticmethod
    def format(metrics: MetricsResult) -> str:
        """Generate formatted table."""
        # TODO: Use tabulate or similar
        pass
```

---

## 9. Integration Summary

### Adding Metrics to Existing Workflow

```python
# Example usage in user code:

from sudoku_nisq.solvers import ExactCoverQuantumSolver
from sudoku_nisq.metrics import ValidationContext, BenchmarkSuite

# Setup solver
solver = ExactCoverQuantumSolver(puzzle, ...)

# Define validation context
validation_ctx = ValidationContext(
    valid_solutions=puzzle.enumerate_all_solutions(),
    total_valid_count=puzzle.count_solutions(),
    solution_validator=lambda bs: solver._is_valid_solution(bs)
)

# Run benchmark suite
benchmark = BenchmarkSuite(
    solver=solver,
    validation_context=validation_ctx,
    n_runs=5  # Inter-run variability
)

results = benchmark.run_benchmark(
    backend=my_backend,
    shots=2048,
    opt_level=2
)

# Access metrics
print(f"p_succ: {results['aggregated'].p_succ:.4f} ± {results['aggregated'].p_succ_std:.4f}")
print(f"Coverage: {results['aggregated'].distinct_valid_solutions} solutions")
print(f"η_gate: {results['aggregated'].eta_gate:.6f}")
```

---

## 10. TODO Summary

### Immediate Implementation Priorities

1. **Core data models** (`data_models.py`) - ✅ No dependencies
2. **Calculator modules** (`calculators/`) - ✅ Mostly independent
3. **Base collector interface** (`collectors/base_collector.py`) - ✅ No dependencies

### Provider-Dependent TODOs

1. **Circuit volume calculation**: Requires DAG layer analysis (SDK-specific)
2. **Hardware metadata extraction**: Different APIs per provider
3. **Execution time tracking**: Provider-specific job metadata access
4. **Initial/final layout extraction**: Transpiler-specific metadata

### Classical Solver Integration

1. Research and select classical exact cover library (python-constraint, pycosat, etc.)
2. Implement timing harness
3. Add fair comparison methodology

### Future Enhancements

1. Plotting/visualization reporters
2. Automated report generation
3. Database storage for longitudinal benchmarking
4. Web dashboard for metrics exploration

---

## 11. File Dependencies & Implementation Order

```
Phase 1 (No dependencies):
  ✓ data_models.py
  ✓ calculators/success_metrics.py
  ✓ calculators/ranking_metrics.py
  ✓ calculators/statistical_metrics.py
  ✓ calculators/efficiency_metrics.py
  ✓ calculators/variability_metrics.py

Phase 2 (Depends on Phase 1):
  ✓ collectors/base_collector.py (interface only)
  ✓ aggregators/multi_run_aggregator.py

Phase 3 (Requires solver integration):
  → Modify quantum_solver.py (add collect_metrics parameter)
  → Modify exact_cover_solver.py (add calculate_metrics method)

Phase 4 (Provider-specific):
  → collectors/qiskit_collector.py
  → collectors/pytket_collector.py
  → collectors/braket_collector.py

Phase 5 (Orchestration):
  → benchmarking/benchmark_suite.py
  → benchmarking/classical_baseline.py

Phase 6 (Output):
  → reporters/json_reporter.py
  → reporters/table_reporter.py
  → reporters/plot_reporter.py
```

---

## Notes on Design Decisions

1. **Why separate collectors?**: Hardware APIs vary wildly. Abstraction lets us gracefully handle missing data.

2. **Why calculator classes vs functions?**: Allows stateful calculators in future (e.g., caching expensive computations).

3. **Why ValidationContext?**: Keeps solution validation logic separate from metrics, enables reuse.

4. **Why multiple η-metrics?**: No single metric captures all resource costs across platforms.

5. **Why dataclasses?**: Type safety, automatic equality, easy serialization.

6. **Why not integrate metrics into MetadataManager?**: Separation of concerns—metadata is about caching, metrics about evaluation.

---

## References

- Clopper-Pearson intervals: scipy.stats.beta.ppf
- Circuit volume: IBM Quantum benchmarking standards
- SNR definition: Signal processing literature
- Precision/Recall@k: Information retrieval literature
