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
        run_id: Execution run ID from Stage 5 metadata (UUID), if available
        metadata: Additional execution metadata (hardware snapshot, compilation_id), if available
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
    run_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class HardwareMetadata:
    """Hardware calibration and error rate information.
    
    Provider-specific fields may be None if not available. This is expected
    for simulators and some quantum providers.
    
    Attributes:
        backend_name: Backend identifier
        provider: Provider name ('ibm', 'quantinuum', 'aws', 'aer', etc.)
        calibration_timestamp: When calibration data was last updated
        single_qubit_gate_error: Mapping from qubit index (string) to error rate
        two_qubit_gate_error: Mapping from qubit pair ("q1,q2") to error rate
        readout_error: Mapping from qubit index (string) to readout error rate
        t1_times: T1 coherence times in microseconds per qubit (string keys)
        t2_times: T2 coherence times in microseconds per qubit (string keys)
        extra_properties: Additional provider-specific metadata
    """
    backend_name: str
    provider: str
    calibration_timestamp: Optional[datetime] = None
    single_qubit_gate_error: Optional[Dict[str, float]] = None
    two_qubit_gate_error: Optional[Dict[str, float]] = None
    readout_error: Optional[Dict[str, float]] = None
    t1_times: Optional[Dict[str, float]] = None
    t2_times: Optional[Dict[str, float]] = None
    extra_properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompilationMetadata:
    """Circuit compilation/transpilation metadata.
    
    Tracks how the circuit was modified during compilation and the parameters
    used for transpilation.
    
    Note: Transpiler seed is not supported. Qiskit transpilation is deterministic
    given backend + optimization_level.
    
    Attributes:
        optimization_level: Optimization level (typically 0-3)
        initial_layout: Initial qubit layout (logical -> physical)
        final_layout: Final qubit layout after transpilation
        pre_transpile_gates: Gate counts before transpilation
        post_transpile_gates: Gate counts after transpilation
        pre_transpile_depth: Circuit depth before transpilation
        post_transpile_depth: Circuit depth after transpilation
        circuit_hash: Hash of transpiled circuit for reproducibility
    """
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
        mass_precision_at_k: Mass-weighted precision@k (improved)
        valid_mass_capture_at_k: Fraction of valid mass in top-k (improved)
        
        # Statistical Metrics (Section 3)
        snr: Signal-to-noise ratio (DEPRECATED: use valid_odds)
        valid_odds: Odds ratio p_succ / (1 - p_succ)
        valid_odds_ci_lower: Lower bound of odds CI
        valid_odds_ci_upper: Upper bound of odds CI
        valid_odds_is_infinite: Flag for perfect discrimination case
        
        # Peak-based Discrimination Metrics
        p_best_valid: Probability of most frequent valid solution
        p_best_invalid: Probability of most frequent invalid solution
        peak_ratio: Ratio of best valid to best invalid
        peak_gap: Difference between best valid and best invalid
        peak_ratio_is_infinite: Flag for perfect peak discrimination
        
        # Efficiency Metrics (Section 4) - DEPRECATED
        eta_gate: Gate-normalized success (DEPRECATED: use log_loss_per_2q/retention_per_2q)
        eta_volume: Volume-normalized success (DEPRECATED: use log_loss_per_volume)
        eta_shot: Shot-normalized success (DEPRECATED: use shot_budgets)
        
        # Retention-based Normalization Metrics (Improved efficiency)
        log_loss_per_2q: Per-gate log loss -log(p_succ)/gates
        log_loss_per_2q_ci_lower: Lower CI bound
        log_loss_per_2q_ci_upper: Upper CI bound
        retention_per_2q: Geometric mean retention p_succ^(1/gates)
        retention_per_2q_ci_lower: Lower CI bound
        retention_per_2q_ci_upper: Upper CI bound
        log_loss_per_volume: Per-volume log loss
        log_loss_per_volume_ci_lower: Lower CI bound
        log_loss_per_volume_ci_upper: Upper CI bound
        retention_per_volume: Geometric mean retention per volume
        retention_per_volume_ci_lower: Lower CI bound
        retention_per_volume_ci_upper: Upper CI bound
        
        # Shot Budget Metrics (Replaces eta_shot)
        shots_detect_point: Shots needed for 95% detection reliability (point estimate)
        shots_detect_pessimistic: Pessimistic estimate (using CI lower bound)
        shots_detect_optimistic: Optimistic estimate (using CI upper bound)
        shot_budget_reliability: Target reliability used (default 0.95)
        
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
    p_succ_ci_lower: Optional[float]
    p_succ_ci_upper: Optional[float]
    distinct_valid_solutions: int
    
    # Count-based ranking metrics
    top_k_valid_mass: Dict[int, float]
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    
    # Mass-weighted ranking metrics (improved)
    mass_precision_at_k: Optional[Dict[int, float]] = None
    valid_mass_capture_at_k: Optional[Dict[int, float]] = None
    
    # Statistical metrics
    snr: Optional[float] = None  # Deprecated
    valid_odds: Optional[float] = None
    valid_odds_ci_lower: Optional[float] = None
    valid_odds_ci_upper: Optional[float] = None
    valid_odds_is_infinite: bool = False
    
    # Peak-based discrimination
    p_best_valid: Optional[float] = None
    p_best_invalid: Optional[float] = None
    peak_ratio: Optional[float] = None
    peak_gap: Optional[float] = None
    peak_ratio_is_infinite: bool = False
    
    # Legacy efficiency metrics (deprecated)
    eta_gate: Optional[float] = None
    eta_volume: Optional[float] = None
    eta_shot: Optional[float] = None
    
    # Retention-based normalization (improved)
    log_loss_per_2q: Optional[float] = None
    log_loss_per_2q_ci_lower: Optional[float] = None
    log_loss_per_2q_ci_upper: Optional[float] = None
    retention_per_2q: Optional[float] = None
    retention_per_2q_ci_lower: Optional[float] = None
    retention_per_2q_ci_upper: Optional[float] = None
    log_loss_per_volume: Optional[float] = None
    log_loss_per_volume_ci_lower: Optional[float] = None
    log_loss_per_volume_ci_upper: Optional[float] = None
    retention_per_volume: Optional[float] = None
    retention_per_volume_ci_lower: Optional[float] = None
    retention_per_volume_ci_upper: Optional[float] = None
    
    # Shot budget metrics (improved)
    shots_detect_point: Optional[int] = None
    shots_detect_pessimistic: Optional[int] = None
    shots_detect_optimistic: Optional[int] = None
    shot_budget_reliability: float = 0.95
    
    # Cost-normalized efficiency metrics (heuristic alternatives)
    eta_product: Optional[float] = None  # η_× = p/(depth×2Q)
    eta_weighted_sum: Optional[float] = None  # η_+ = p/(α·depth+β·2Q)
    decay_rate: Optional[float] = None  # k = -log(p)/(α·depth+β·2Q), smaller=better
    cost_alpha: Optional[float] = None  # Weight used for depth (reproducibility)
    cost_beta: Optional[float] = None  # Weight used for 2Q gates (reproducibility)
    
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
            f"Success Probability: {self.p_succ:.4f}",
        ]
        
        if self.p_succ_ci_lower is not None and self.p_succ_ci_upper is not None:
            lines.append(f"  95% CI: [{self.p_succ_ci_lower:.4f}, {self.p_succ_ci_upper:.4f}]")
        
        lines.append(f"Distinct Valid Solutions: {self.distinct_valid_solutions}")
        
        if self.valid_odds is not None:
            if self.valid_odds_is_infinite:
                lines.append("Valid Odds: ∞ (perfect discrimination)")
            else:
                lines.append(f"Valid Odds: {self.valid_odds:.2f}")
        
        if self.peak_ratio is not None:
            if self.peak_ratio_is_infinite:
                lines.append("Peak Ratio: ∞ (no invalid solutions)")
            else:
                lines.append(f"Peak Ratio: {self.peak_ratio:.2f} (gap: {self.peak_gap:.4f})")
        
        if self.retention_per_2q is not None:
            lines.append(f"Per-Gate Retention: {self.retention_per_2q:.6f} ({(1-self.retention_per_2q)*100:.3f}% loss/gate)")
        
        if self.shots_detect_point is not None:
            lines.append(f"Shots for 95% Detection: {self.shots_detect_point}")
        
        if self.p_succ_mean is not None:
            lines.append(f"Multi-run Mean: {self.p_succ_mean:.4f} ± {self.p_succ_std:.4f}")
        
        return "\n".join(lines)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple runs.
    
    Contains statistical aggregations (mean, std, median, Q1, Q3, IQR) for
    each numeric metric from multiple MetricsResult objects. Used for multi-run
    benchmarking to capture variability and statistical significance.
    
    Each metric field contains a dictionary with keys:
        - mean: Arithmetic mean
        - std: Sample standard deviation (Bessel's correction)
        - median: Middle value
        - q1: First quartile (25th percentile)
        - q3: Third quartile (75th percentile)
        - iqr: Interquartile range (Q3 - Q1)
    
    Values may be None if insufficient data or metric not applicable.
    
    Attributes:
        n_runs: Number of runs aggregated
        timestamp: When aggregation was performed
        aggregation_notes: Optional notes about aggregation process
        
        # Success metrics
        p_succ: Probability of measuring a valid solution
        distinct_valid: Number of distinct valid solutions measured
        
        # Ranking metrics
        top_k_valid_mass: Per-k stats for valid mass in top-k
        precision_at_k: Per-k stats for precision at k
        recall_at_k: Per-k stats for recall at k
        mass_precision_at_k: Per-k stats for mass-weighted precision
        valid_mass_capture_at_k: Per-k stats for valid mass capture
        
        # Odds metrics
        valid_odds: Odds ratio of valid:invalid mass
        valid_odds_lower: Lower bound of 95% CI for valid odds
        valid_odds_upper: Upper bound of 95% CI for valid odds
        
        # Peak metrics
        peak_ratio: Ratio of highest valid to highest invalid solution
        peak_gap: Difference between highest valid and highest invalid
        p_best_valid: Probability of most frequent valid solution
        p_best_invalid: Probability of most frequent invalid solution
        
        # Retention metrics (circuit quality)
        retention_per_2q: Per-two-qubit-gate retention factor
        retention_lower: Lower bound of 95% CI for retention
        retention_upper: Upper bound of 95% CI for retention
        retention_per_volume: Per-volume retention factor
        retention_volume_lower: Lower bound of 95% CI for volume retention
        retention_volume_upper: Upper bound of 95% CI for volume retention
        
        # Log loss (information-theoretic quality)
        log_loss: Cross-entropy between measured and uniform valid dist
        log_loss_lower: Lower bound of 95% CI for log loss
        log_loss_upper: Upper bound of 95% CI for log loss
        log_loss_per_volume: Log loss normalized by circuit volume
        log_loss_volume_lower: Lower bound of 95% CI for volume log loss
        log_loss_volume_upper: Upper bound of 95% CI for volume log loss
        
        # Shot budget metrics
        shots_detect_point: Shots needed for 95% detection probability
        shots_detect_pessimistic: Shots based on lower CI bound
        shots_detect_optimistic: Shots based on upper CI bound
        
        # Deprecated metrics (included for backward compatibility)
        snr: Signal-to-noise ratio (deprecated, use valid_odds)
        eta_gate: Gate efficiency (deprecated, use retention)
        eta_volume: Volume efficiency (deprecated)
        eta_shot: Shot efficiency (deprecated)
    """
    n_runs: int
    timestamp: datetime
    aggregation_notes: Optional[str] = None
    
    # Success metrics
    p_succ: Optional[Dict[str, Optional[float]]] = None
    distinct_valid: Optional[Dict[str, Optional[float]]] = None
    
    # Ranking metrics
    top_k_valid_mass: Optional[Dict[int, Optional[Dict[str, Optional[float]]]]] = None
    precision_at_k: Optional[Dict[int, Optional[Dict[str, Optional[float]]]]] = None
    recall_at_k: Optional[Dict[int, Optional[Dict[str, Optional[float]]]]] = None
    mass_precision_at_k: Optional[Dict[int, Optional[Dict[str, Optional[float]]]]] = None
    valid_mass_capture_at_k: Optional[Dict[int, Optional[Dict[str, Optional[float]]]]] = None
    
    # Odds metrics
    valid_odds: Optional[Dict[str, Optional[float]]] = None
    valid_odds_lower: Optional[Dict[str, Optional[float]]] = None
    valid_odds_upper: Optional[Dict[str, Optional[float]]] = None
    
    # Peak metrics
    peak_ratio: Optional[Dict[str, Optional[float]]] = None
    peak_gap: Optional[Dict[str, Optional[float]]] = None
    p_best_valid: Optional[Dict[str, Optional[float]]] = None
    p_best_invalid: Optional[Dict[str, Optional[float]]] = None
    
    # Retention metrics
    retention_per_2q: Optional[Dict[str, Optional[float]]] = None
    retention_lower: Optional[Dict[str, Optional[float]]] = None
    retention_upper: Optional[Dict[str, Optional[float]]] = None
    retention_per_volume: Optional[Dict[str, Optional[float]]] = None
    retention_volume_lower: Optional[Dict[str, Optional[float]]] = None
    retention_volume_upper: Optional[Dict[str, Optional[float]]] = None
    
    # Log loss
    log_loss: Optional[Dict[str, Optional[float]]] = None
    log_loss_lower: Optional[Dict[str, Optional[float]]] = None
    log_loss_upper: Optional[Dict[str, Optional[float]]] = None
    log_loss_per_volume: Optional[Dict[str, Optional[float]]] = None
    log_loss_volume_lower: Optional[Dict[str, Optional[float]]] = None
    log_loss_volume_upper: Optional[Dict[str, Optional[float]]] = None
    
    # Shot budget
    shots_detect_point: Optional[Dict[str, Optional[float]]] = None
    shots_detect_pessimistic: Optional[Dict[str, Optional[float]]] = None
    shots_detect_optimistic: Optional[Dict[str, Optional[float]]] = None
    
    # Deprecated (backward compatibility)
    snr: Optional[Dict[str, Optional[float]]] = None
    eta_gate: Optional[Dict[str, Optional[float]]] = None
    eta_volume: Optional[Dict[str, Optional[float]]] = None
    eta_shot: Optional[Dict[str, Optional[float]]] = None
