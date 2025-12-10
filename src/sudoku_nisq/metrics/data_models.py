"""
Data models for benchmarking metrics system.

This module defines all data structures used for collecting, storing, and 
reporting quantum benchmarking metrics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime


@dataclass
class ExecutionResult:
    """Results from a single quantum execution.
    
    Attributes:
        counts: Measurement counts dictionary (bitstring -> count)
        shots: Total number of measurement shots
        execution_time: Wall-clock execution time in seconds
        timestamp: When the execution occurred
        backend_name: Name/identifier of the quantum backend
        num_qubits: Number of qubits in the circuit
        circuit_depth: Circuit depth
        gate_counts: Dictionary mapping gate type to count
        two_qubit_gates: Total number of two-qubit gates
        circuit_volume: Circuit volume (sum of active gates per layer), if available
        raw_result: Original provider-specific result object
        job_id: Job identifier from provider, if available
    """
    counts: Dict[str, int]
    shots: int
    execution_time: float
    timestamp: datetime
    backend_name: str
    num_qubits: int
    circuit_depth: int
    gate_counts: Dict[str, int]
    two_qubit_gates: int
    circuit_volume: Optional[int] = None
    raw_result: Any = None
    job_id: Optional[str] = None


@dataclass
class HardwareMetadata:
    """Hardware calibration and error rate information.
    
    Provider-specific fields may be None if not available. This is expected
    for simulators and some quantum providers.
    
    Attributes:
        backend_name: Backend identifier
        provider: Provider name ('ibm', 'quantinuum', 'aws', 'aer', etc.)
        calibration_timestamp: When calibration data was last updated
        single_qubit_gate_error: Mapping from qubit index to error rate
        two_qubit_gate_error: Mapping from qubit pair to error rate
        readout_error: Mapping from qubit index to readout error rate
        t1_times: T1 coherence times in microseconds per qubit
        t2_times: T2 coherence times in microseconds per qubit
        extra_properties: Additional provider-specific metadata
    """
    backend_name: str
    provider: str
    calibration_timestamp: Optional[datetime] = None
    single_qubit_gate_error: Optional[Dict[int, float]] = None
    two_qubit_gate_error: Optional[Dict[tuple, float]] = None
    readout_error: Optional[Dict[int, float]] = None
    t1_times: Optional[Dict[int, float]] = None
    t2_times: Optional[Dict[int, float]] = None
    extra_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationMetadata:
    """Circuit compilation/transpilation metadata.
    
    Tracks how the circuit was modified during compilation and the parameters
    used for transpilation.
    
    Attributes:
        transpiler_seed: Random seed used for transpilation
        optimization_level: Optimization level (typically 0-3)
        initial_layout: Initial qubit layout (logical -> physical)
        final_layout: Final qubit layout after transpilation
        pre_transpile_gates: Gate counts before transpilation
        post_transpile_gates: Gate counts after transpilation
        pre_transpile_depth: Circuit depth before transpilation
        post_transpile_depth: Circuit depth after transpilation
        circuit_hash: Hash of transpiled circuit for reproducibility
    """
    transpiler_seed: Optional[int] = None
    optimization_level: int = 0
    initial_layout: Optional[List[int]] = None
    final_layout: Optional[List[int]] = None
    pre_transpile_gates: Optional[Dict[str, int]] = None
    post_transpile_gates: Optional[Dict[str, int]] = None
    pre_transpile_depth: Optional[int] = None
    post_transpile_depth: Optional[int] = None
    circuit_hash: Optional[str] = None


@dataclass
class ValidationContext:
    """Context for validating quantum measurement results.
    
    Provides the information needed to determine whether a measured bitstring
    corresponds to a valid solution.
    
    Attributes:
        valid_solutions: List of all valid solution bitstrings (for small problems)
        total_valid_count: Total number of valid solutions
        solution_validator: Function that takes bitstring and returns True if valid
    """
    valid_solutions: List[str]
    total_valid_count: int
    solution_validator: Callable[[str], bool]


@dataclass
class MetricsResult:
    """Comprehensive benchmarking metrics from one or more runs.
    
    This dataclass contains all computed metrics following the benchmarking
    standards outlined in the architecture document.
    
    Attributes:
        # Success Metrics (Section 1)
        p_succ: Success probability (fraction of valid measurements)
        p_succ_ci_lower: Lower bound of 95% Clopper-Pearson confidence interval
        p_succ_ci_upper: Upper bound of 95% Clopper-Pearson confidence interval
        distinct_valid_solutions: Number of unique valid solutions observed
        
        # Ranking Metrics (Section 2)
        top_k_valid_mass: Probability mass of top-k valid solutions (k -> mass)
        precision_at_k: Precision@k (k -> precision)
        recall_at_k: Recall@k (k -> recall)
        
        # Statistical Metrics (Section 3)
        snr: Signal-to-noise ratio (valid mass / invalid mass)
        
        # Efficiency Metrics (Section 4)
        eta_gate: Gate-normalized success (p_succ / two_qubit_gates)
        eta_volume: Volume-normalized success (p_succ / circuit_volume)
        eta_shot: Shot-normalized success (p_succ / shots)
        
        # Variability Metrics (Section 6) - populated by multi-run aggregation
        p_succ_mean: Mean p_succ across runs
        p_succ_std: Standard deviation of p_succ across runs
        p_succ_iqr: Interquartile range (Q1, Q3) of p_succ across runs
        
        # Metadata References (Section 5)
        execution_results: List of ExecutionResult objects
        hardware_metadata: Hardware calibration information
        compilation_metadata: Compilation/transpilation information
    """
    # Success metrics
    p_succ: float
    p_succ_ci_lower: float
    p_succ_ci_upper: float
    distinct_valid_solutions: int
    
    # Ranking metrics
    top_k_valid_mass: Dict[int, float]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    
    # Statistical metrics
    snr: float
    
    # Efficiency metrics
    eta_gate: float
    eta_volume: Optional[float] = None
    eta_shot: Optional[float] = None
    
    # Variability metrics (multi-run)
    p_succ_mean: Optional[float] = None
    p_succ_std: Optional[float] = None
    p_succ_iqr: Optional[tuple] = None
    
    # Metadata
    execution_results: List[ExecutionResult] = field(default_factory=list)
    hardware_metadata: Optional[HardwareMetadata] = None
    compilation_metadata: Optional[CompilationMetadata] = None
    
    def summary_str(self) -> str:
        """Generate human-readable summary of metrics."""
        lines = [
            "=== Benchmarking Metrics Summary ===",
            f"Success Probability: {self.p_succ:.4f} [{self.p_succ_ci_lower:.4f}, {self.p_succ_ci_upper:.4f}]",
            f"Distinct Valid Solutions: {self.distinct_valid_solutions}",
            f"Signal-to-Noise Ratio: {self.snr:.2f}",
            f"Gate Efficiency (η_gate): {self.eta_gate:.6f}",
        ]
        
        if self.eta_volume is not None:
            lines.append(f"Volume Efficiency (η_volume): {self.eta_volume:.6f}")
        
        if self.p_succ_mean is not None:
            lines.append(f"Multi-run Mean: {self.p_succ_mean:.4f} ± {self.p_succ_std:.4f}")
        
        return "\n".join(lines)
